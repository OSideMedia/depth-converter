"""How much memory a depth pass may use on this machine, worked out fresh before each pass.

A depth pass needs several GB of unified memory. Past what is free, macOS compresses and swaps
until the whole machine stalls, and nothing in the pass notices. So before each pass this module
reads what THIS machine has free right now, keeps a margin for everything else, and either sets a
limit the pass cannot cross or refuses with a plain message saying what to do.

The same limit serves an 8 GB laptop and a 128 GB desktop, because it comes from the machine and
not from a fixed number. What a pass needs is the part that is fixed, and it was measured.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

GB = 1024**3

# The smallest GPU limit a pass finished under, on a 16:9 clip, to within 0.2 GB:
#
#     --input-size   140   196   252   280   336   406   462   518
#     GB             1.32  1.29  2.39  2.39  2.74  3.50  4.68  5.55
#
# So part of the pool is fixed and the rest follows the area of the picture going in. These two
# figures draw a line above every one of those points, by 0.19 GB at the closest (252).
# ponytail: one clip shape on one machine. The pass still has a hard GPU limit behind this
# estimate, so a low estimate ends in a plain message mid-pass, not a stalled machine.
_POOL_FIXED = 1.5 * GB
_POOL_AT_DEFAULT_SIZE = 4.6 * GB  # on top of the fixed part
_DEFAULT_PIXELS = 518 * 924
_PROCESS = int(0.5 * GB)  # Python, torch and the model, outside the GPU pool (measured)
# Held outside the pool: 3 bytes a pixel for the RGB frames, and 4 for depth. Upstream keeps all 32
# frames of every window at full size, overlaps included, padded out on a short clip. At the end of
# the pass it aligns them into a second list and stacks the result, with the fixed part of the pool
# still held. That stage is the peak on a long clip or at a small --input-size: 5.1 GB for 8 s of
# 720 x 1280 at --input-size 140, with the windows at 2.9.
_RGB = 3
_DEPTH = 4
_WINDOW = 32  # engine.INFER_LEN
_WINDOW_STEP = 22  # engine.INFER_LEN - engine.OVERLAP
_MEASURED_PIXELS = 1280 * 720  # the pool was measured at this output size
_KEEP_FREE = int(1.5 * GB)  # room for everything else on the machine to keep working
_SMALLEST_INPUT_SIZE = 140
_PATCH = 14  # --input-size must be a multiple of this (DINOv2 patches)

OVERRIDE_ENV = "DEPTHC_MEMORY_GB"
_MAC = sys.platform == "darwin"
# ponytail: a read under this goes unchecked, even with less than that free. It is a moment of pressure
# and not a stall, a depth pass's own check still runs after it, and a listing should survive a tight
# machine. Check these reads too if one is ever seen to stall a machine.
_SMALL_READ = GB


@dataclass(frozen=True)
class Plan:
    """What this pass needs against what this machine can spare."""

    need: int
    allowed: int
    gpu_limit: int
    refusal: str | None


def total_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def parse_vm_stat(text: str) -> int | None:
    """Free memory from `vm_stat` output: pages free, plus those macOS hands over on demand."""
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    total = 0
    for name in ("free", "inactive", "speculative"):
        count = re.search(rf"^Pages {name}:\s+(\d+)\.", text, re.MULTILINE)
        if not count:
            return None
        total += int(count.group(1))
    return total * int(page.group(1))


def available_bytes() -> int | None:
    """What this machine has free right now, or None if it will not say."""
    try:
        done = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_vm_stat(done.stdout)


def own_bytes() -> int:
    """What this process holds right now, GPU memory included. 0 where the system will not say."""
    info = ctypes.create_string_buffer(256)  # macOS's rusage_info_v0, flavor 0
    try:
        failed = ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 0, info)
    except (OSError, AttributeError):  # not macOS
        return 0
    return 0 if failed else int.from_bytes(info.raw[72:80], sys.byteorder)  # ri_phys_footprint


def _pool_need(height: int, width: int, input_size: int) -> int:
    """GPU memory for one window, following upstream's sizing rule (`infer_video_depth`)."""
    ratio = max(height, width) / max(1, min(height, width))
    if ratio > 1.78:
        input_size = round(int(input_size * 1.777 / ratio) / _PATCH) * _PATCH
    pixels = input_size * input_size * ratio
    # Each window's depth is also upsampled to the clip's size on the GPU, which the 720p measurements
    # already hold. Source-resolution 4K adds about a GB.
    upsampled = max(0, height * width - _MEASURED_PIXELS) * _WINDOW * _DEPTH
    return int(_POOL_FIXED + _POOL_AT_DEFAULT_SIZE * pixels / _DEFAULT_PIXELS) + upsampled


def _window_frames(count: int) -> int:
    return _WINDOW * math.ceil(count / _WINDOW_STEP)


def _cpu_while_windows_run(shape: tuple[int, ...]) -> int:
    count, height, width = shape[:3]
    return height * width * (_RGB * count + _DEPTH * _window_frames(count))


def _cpu_at_the_end(shape: tuple[int, ...]) -> int:
    count, height, width = shape[:3]
    return height * width * (_RGB * count + _DEPTH * (2 * _window_frames(count) + count))


def need_bytes(shape: tuple[int, ...], input_size: int) -> int:
    """What a pass over frames of `shape` [N, H, W, 3] needs at its peak, whichever stage that is."""
    while_windows_run = _pool_need(shape[1], shape[2], input_size) + _cpu_while_windows_run(shape)
    at_the_end = int(_POOL_FIXED) + _cpu_at_the_end(shape)
    return _PROCESS + max(while_windows_run, at_the_end)


def _allowed_bytes(credit_own: bool = True) -> tuple[int, str]:
    """What this pass may use, and how that was decided, in words for the refusal.

    The estimate is for the whole process at its peak, and `vm_stat` has already taken out what
    this process holds: the decoded frames, and what an earlier pass in the same run left behind
    (`inbox` and the UI run many). So that goes back in, or it is charged twice.

    ponytail: treats all of it as room the pass will reuse. Something large kept alongside the pass
    would eat the margin instead. Nothing here does that today.
    """
    override = os.environ.get(OVERRIDE_ENV, "").strip()
    if override:
        try:
            allowed = int(float(override) * GB)
        except (ValueError, OverflowError):  # "lots", "nan", "inf"
            allowed = 0
        if allowed > 0:
            return allowed, f"{OVERRIDE_ENV} allows {_gb(allowed)}"

    total = total_bytes()
    available = available_bytes()
    if available is None:
        # A Mac that cannot answer may be too busy to answer, which is no time to start a pass.
        return 0 if _MAC else total // 2, f"This machine has {_gb(total)} and would not report what is free"
    if credit_own:
        available += own_bytes()
    return (
        available - _KEEP_FREE,
        f"This machine has about {_gb(available)} free out of {_gb(total)}, "
        f"and {_gb(_KEEP_FREE)} of that stays free for everything else",
    )


def _gb(value: int) -> str:
    return f"{value / GB:.1f} GB"


def largest_input_size_that_fits(shape: tuple[int, ...], input_size: int, allowed: int) -> int | None:
    """The biggest --input-size under this one that this machine can spare, if any."""
    size = (input_size // _PATCH) * _PATCH
    if size == input_size:
        size -= _PATCH
    while size >= _SMALLEST_INPUT_SIZE:
        if need_bytes(shape, size) <= allowed:
            return size
        size -= _PATCH
    return None


def decode_refusal(frame_bytes: int) -> str | None:
    """Reading a clip holds its frames twice for a moment, and the pass check can only run after it.

    All of that has to come out of what is free now. An array this process already holds may still be
    alive while the read runs, as `look-compare --manifest` holds the last clip's, so none of it counts.
    """
    if frame_bytes < _SMALL_READ:
        return None
    need = 2 * frame_bytes
    allowed, why = _allowed_bytes(credit_own=False)
    if need <= allowed:
        return None
    return (
        f"Not enough free memory to read this clip: it needs about {_gb(need)}. {why}.\n"
        f"  Nothing ran. Use a shorter clip or a lower resolution, or close other apps and run it again.\n"
        f"  To run it anyway, set {OVERRIDE_ENV} to the GB this read may use."
    )


def plan(shape: tuple[int, ...], input_size: int) -> Plan:
    """Decide this pass against this machine, right now.

    This reads the machine once, before the pass. Another app can still fill memory after the pass
    starts, so `Watch` reads it again while the pass runs.
    """
    need = need_bytes(shape, input_size)
    allowed, why = _allowed_bytes()
    gpu_limit = max(0, allowed - _PROCESS - _cpu_while_windows_run(shape))
    if need <= allowed:
        return Plan(need, allowed, gpu_limit, None)

    lines = [
        f"Not enough free memory for this depth pass: it needs about {_gb(need)}. {why}.",
        "Nothing ran. Close other apps and run it again.",
    ]
    smaller = largest_input_size_that_fits(shape, input_size, allowed)
    if smaller is not None:
        lines.append(
            f"Or use --input-size {smaller}, which needs about {_gb(need_bytes(shape, smaller))} "
            f"and gives a coarser depth picture."
        )
    else:  # the clip's own frames and depth are what is left, and only a smaller clip shrinks those
        lines.append("Or use a shorter clip or a lower resolution.")
    lines.append(f"To run it anyway, set {OVERRIDE_ENV} to the GB this pass may use.")
    return Plan(need, allowed, gpu_limit, "\n  ".join(lines))


# While a pass runs. What `vm_stat` calls free (`available_bytes`), sampled every half second through
# real default passes (`describe`, 8 s clips, one pass at a time, nothing else running a model), with
# the process's own footprint (`own_bytes`' reading) beside it. GB and seconds:
#
#     clip         free at start   lowest   at   of    drop   own peak
#     640 x 360        11.15        5.38    9.1  37.8  5.77    5.80
#     640 x 360        12.53        6.75   33.8  35.8  5.78    6.05
#     480 x 270        12.09        6.24    9.1  36.3  5.85    6.07
#     1280 x 720       10.77        5.97    9.1  27.8  4.81    5.71
#     1280 x 720       11.94        5.41   34.8  37.9  6.53    7.33
#     1280 x 720       12.14        6.12   34.8  37.3  6.01    6.75
#     720 x 1280       10.79        5.17   13.1  37.8  5.61    7.29
#     720 x 1280       12.02        5.94   30.8  37.3  6.08    6.82
#     720 x 1280       11.93        5.72   34.8  36.8  6.21    7.31
#
# "of" is the whole command, model load and describe's own reading included. Free fell by up to 2.76
# GB between two readings, and no pass stayed within 0.5 GB of its lowest for more than 2 readings in a
# row. Every drop was at least 0.94 GB under what `need_bytes` asks for that clip, so a pass let in at
# the very edge, with 1.5 GB to spare, would bottom out near 2.4 GB if nothing else grew. The floor is
# half of that 1.5 GB, held for three readings in a row: about 1.7 GB under where such a pass would
# bottom out, and a second or more of the machine below 0.75 GB before a pass is stopped.
# ponytail: one 24 GB machine, four clip shapes, 8 s clips, and never a machine in real trouble, which
# was not safe to make. Under real pressure macOS may free pages in bursts, and a reading back over the
# floor restarts the count. Nothing reads in upstream's end-of-pass alignment, the peak on a long clip.
# If a good pass is ever stopped or a stall gets through, measure that machine the same way, set a
# floor per machine size, and add a reading point at the alignment if long clips need it.
WATCH_FLOOR = int(0.75 * GB)  # half of _KEEP_FREE
WATCH_READINGS = 3  # in a row, under the floor, before a pass stops: a good pass's low held for 2 at most
WATCH_EVERY = 0.5  # seconds between readings: each one starts `vm_stat`, and a pass settles far more often


class Watch:
    """Reads what this machine has free while a pass runs, and says when to stop it.

    `check` is called each time the model has settled its GPU work, many times a window. It reads at
    most every WATCH_EVERY seconds, and a pass stops only when WATCH_READINGS readings in a row come in
    under WATCH_FLOOR, so a moment's dip does not end a good pass. On a Mac a reading that fails counts
    as under the floor: `vm_stat` not answering is what a stalling Mac looks like, as it is before a
    pass. Elsewhere there is no `vm_stat` to read at all, and the check before the pass already works
    from half of RAM, so the watch stands down rather than stop every pass.
    `DEPTHC_MEMORY_GB` sets what a pass may use; it does not let a pass run the machine out.
    """

    def __init__(self, read: Callable[[], int | None] | None = None, clock: Callable[[], float] = time.monotonic):
        self._read = read
        self._clock = clock
        self._last: float | None = None
        self._under: list[int | None] = []

    def check(self) -> str | None:
        """None while the pass may go on; once it must stop, the words that say so."""
        now = self._clock()
        if self._last is not None and now - self._last < WATCH_EVERY:
            return None
        self._last = now
        free = (self._read or available_bytes)()
        if free is None and not _MAC:
            return None
        if free is not None and free >= WATCH_FLOOR:
            self._under.clear()
            return None
        self._under.append(free)
        if len(self._under) < WATCH_READINGS:
            return None
        return _stopped_partway(self._under)


def _stopped_partway(readings: list[int | None]) -> str:
    last = readings[-1]
    if last is None:
        what = "this Mac stopped reporting how much memory is free"
    else:
        what = (
            f"this machine was down to about {_gb(last)} free out of {_gb(total_bytes())}, "
            f"under the {WATCH_FLOOR / GB:.2f} GB a running pass must leave free"  # 0.75, not "0.8"
        )
    return (
        f"The depth pass stopped partway: {what}, {len(readings)} readings in a row.\n"
        f"  Nothing was written. Close other apps and run it again."
    )
