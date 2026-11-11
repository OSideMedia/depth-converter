"""VDA-Small inference wrapper: device selection, windowed inference, normalization.

Upstream's `run.py` hardcodes `'cuda' if available else 'cpu'` — there is no MPS branch.
Rather than patch a vendored file, we import their model class and own device and precision
selection here.

PRECISION: fp32 IS THE DEFAULT ON MPS, and this contradicts PLAN.md §0 pitfall 2, which
predicted "torch>=2.5 and fp16 works; keep a --fp32 escape hatch". Measured on M4 Pro /
torch 2.13.0 / macOS 15.5, VDA-Small under `torch.autocast('mps')`:

    fp32   clean output   1.40 s/window   44 ms/frame
    fp16   ALL NaN        1.55 s/window   49 ms/frame
    bf16   ALL NaN        1.55 s/window   48 ms/frame

Reduced precision is not a speed/quality trade here — it is simply broken, and slower. So
the escape hatch is inverted: fp32 by default, `--precision fp16` opt-in for anyone whose
hardware behaves differently, guarded by a finiteness check that falls back automatically.

Worse, the NaN is *sticky*: an fp16 run poisons the next fp32 run on the same model object
until `torch.mps.empty_cache()` is called, which points at a cached MPS graph rather than
arithmetic overflow. Our per-window cache purge already clears it; the fallback path calls
it explicitly rather than relying on that coincidence.

MEMORY: upstream hands all 32 frames of a window to the DINOv2 backbone as one batch, and without
xFormers (CUDA only) each of its 12 blocks builds a full attention matrix for that batch: frames x
heads x tokens^2 floats, or 4.27 GiB at 518 x 924. That took a pass to 18 GB of unified memory and
all but froze a 24 GB Mac with other apps open. Attention never crosses frames in the backbone, so
`bound_memory` feeds it a few frames at a time, the way upstream's own decoder already does, and the
output bytes do not change. `memory.py` then sets a limit from what the machine has free.
"""

from __future__ import annotations

import contextvars
import gc
import sys
import threading
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from depthc import memory
from depthc.vendor_path import MODEL_CONFIG, ensure_checkpoint, ensure_vendor_on_path

ProgressFn = Callable[[int, int], None]
WarnFn = Callable[[str], None]

# bf16 was measured and is equally broken on MPS; it is deliberately not offered, since
# upstream's autocast call exposes only an on/off flag and wrapping it to force a dtype
# would be complexity in service of a known-bad path.
PRECISIONS = ("fp32", "fp16")

# Upstream's fixed windowing constants, mirrored here only so we can report progress
# in the same units the model actually works in. Do not "tune" these.
INFER_LEN = 32
OVERLAP = 10

# Frames the backbone sees at once. Groups of 8 and of 4 gave identical bytes at the same speed;
# 4 peaked lower (7.9 against 8.5 GB) and matches upstream's decoder micro-batch.
BACKBONE_FRAMES = 4

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def pick_device(prefer: str | None = None) -> str:
    """mps when available, else cpu (with the caller expected to warn loudly)."""
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


def _settle(device: str) -> None:
    """Finish queued GPU work and hand back what it held. Emptying alone frees nothing still queued."""
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _in_groups(whole: Callable, device: str) -> Callable:
    """Wrap the backbone's feature call so it sees BACKBONE_FRAMES frames at a time."""

    def grouped(x, n=1, reshape=False, return_class_token=False, norm=True):
        if not return_class_token:  # upstream never asks for this shape; leave it whole
            return whole(x, n, reshape=reshape, return_class_token=False, norm=norm)
        parts = []
        for start in range(0, x.shape[0], BACKBONE_FRAMES):
            parts.append(whole(x[start:start + BACKBONE_FRAMES], n, reshape=reshape,
                               return_class_token=True, norm=norm))
            _checkpoint(device)
        return tuple(
            (torch.cat([part[layer][0] for part in parts]), torch.cat([part[layer][1] for part in parts]))
            for layer in range(len(parts[0]))
        )

    return grouped


def bound_memory(model: torch.nn.Module, device: str) -> None:
    """Keep a window's memory down, with no vendor edit and no change to the output.

    Each place it settles the GPU is also where a pass under `watching` reads the machine.
    """
    model.pretrained.get_intermediate_layers = _in_groups(model.pretrained.get_intermediate_layers, device)
    for stage in model.head.motion_modules:
        stage.register_forward_hook(lambda *_: _checkpoint(device))


# The watch on the pass this thread is running, if any. The model is built once and kept, so its
# settle points cannot carry a pass's watch themselves.
_WATCH: contextvars.ContextVar[memory.Watch | None] = contextvars.ContextVar("depthc_watch", default=None)


@contextmanager
def watching():
    """Stop a depth pass run inside this block if the machine runs out of memory partway.

    `infer` runs every pass under it, from the first window to the end of `normalize`. Anything that
    runs the model outside `infer` needs it too, as `smoke.py` has it. Each block gets a fresh
    `memory.Watch`, reading from its own thread until the block ends, so one pass's dips never count
    against the next.
    """
    watch = memory.Watch()
    token = _WATCH.set(watch)
    try:  # the thread starts in here, so an interrupt while it starts still closes it
        watch.start()
        yield
    finally:
        watch.close()
        _WATCH.reset(token)
    # Reached only when the block ran to its end. A reading still running at the pass's last stop point
    # ends while `close` waits for it, so the stop it settles is checked here, once the thread is done.
    stop = watch.check()
    if stop:
        raise _Stopped(stop)


def _stop_if_told() -> None:
    """A stop point: if the watch on this pass has said stop, the pass stops here."""
    watch = _WATCH.get()
    if watch is not None:
        stop = watch.check()
        if stop:
            raise _Stopped(stop)


def _checkpoint(device: str) -> None:
    """Hand back what the GPU work just held, then stop here if the watch has said so.

    These points come a dozen times a window, so a pass can stop partway through one. The per-window
    hook comes once a window and never inside smoke's single window.
    """
    _settle(device)
    _stop_if_told()


def limit_gpu(device: str, limit: int) -> None:
    """A hard line behind the estimate: MPS raises at it instead of swapping the machine to a stall."""
    if device == "mps":
        fraction = min(2.0, limit / torch.mps.recommended_max_memory())  # 2 is torch's ceiling
        torch.mps.set_per_process_memory_fraction(fraction)


def load_model(device: str | None = None) -> tuple[torch.nn.Module, str]:
    """Load VDA-Small, cached per device.

    The Gradio process is long-running and reloading a 116 MB checkpoint per conversion
    would be pure waste, so the model is kept resident and only activations are freed.
    """
    device = device or pick_device()
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device], device

    ensure_vendor_on_path()
    checkpoint = ensure_checkpoint()
    with redirect_stdout(sys.stderr):  # upstream prints on import, and stdout may be carrying --json
        from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIG)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    bound_memory(model, device)
    _MODEL_CACHE[device] = model
    return model, device


def unload_model() -> None:
    """Drop the resident model (UI teardown / tests that want a clean slate)."""
    for device in list(_MODEL_CACHE):
        del _MODEL_CACHE[device]
        empty_cache(device)
    gc.collect()


@contextmanager
def _window_hook(device: str, on_window: ProgressFn | None):
    """Hook upstream's per-window loop without editing a vendored file.

    `infer_video_depth` drives its window loop with `for frame_id in tqdm(range(...))`.
    We temporarily rebind the `tqdm` name *inside their module's namespace* to a generator
    of our own. Control returns to us after each loop body completes, which is precisely
    the moment to report progress and release MPS's cached allocations — the two hooks
    PLAN.md §2 asks for, with zero vendor edits and no reimplementation of their
    scale/shift alignment maths. It is also a stop point for the memory watch, and it puts
    the watch's stop points into the work after the last window too (`_tail_hook`).
    """
    import video_depth_anything.video_depth as vda

    original = vda.tqdm

    def hooked(iterable: Iterable, *args, **kwargs):
        items = list(iterable)
        total = len(items)
        for index, item in enumerate(items):
            yield item
            # Reached only once the window's forward pass has finished.
            empty_cache(device)
            if on_window:
                on_window(index + 1, total)
            _stop_if_told()

    vda.tqdm = hooked
    try:
        with _tail_hook(vda):
            yield
    finally:
        vda.tqdm = original


class _StopBeforeStack:
    """Upstream's `np`, with a stop point before `stack`: one call that copies the whole clip's depth."""

    def __init__(self, numpy):
        self._numpy = numpy

    def __getattr__(self, name: str):
        return getattr(self._numpy, name)

    def stack(self, *args, **kwargs):
        _stop_if_told()
        return self._numpy.stack(*args, **kwargs)


@contextmanager
def _tail_hook(vda):
    """Stop points in what upstream does after its last window, without editing a vendored file.

    It aligns each window's depth to the one before into a second list, then stacks that list into one
    array. No GPU work settles there, and on a long clip it is the CPU-side peak of the pass. The two
    helpers the alignment calls once a window, and the `np` it stacks with, are rebound in its module
    for the pass, as `tqdm` is, so each comes with a stop point first.
    """
    names = ("compute_scale_and_shift", "get_interpolate_frames", "np")
    originals = {name: getattr(vda, name) for name in names}

    def stopping_first(call: Callable) -> Callable:
        def stopped_or_called(*args, **kwargs):
            _stop_if_told()
            return call(*args, **kwargs)

        return stopped_or_called

    vda.compute_scale_and_shift = stopping_first(originals["compute_scale_and_shift"])
    vda.get_interpolate_frames = stopping_first(originals["get_interpolate_frames"])
    vda.np = _StopBeforeStack(originals["np"])
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(vda, name, value)


@dataclass
class DepthResult:
    """Normalized depth frames plus the numbers worth putting in kit.json."""

    frames: np.ndarray  # [N, H, W] uint8
    fps: float
    device: str
    dtype: str
    elapsed: float
    raw_min: float
    raw_max: float
    inverted: bool

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def ms_per_frame(self) -> float:
        return self.elapsed / max(1, self.count) * 1000


class DepthFailure(RuntimeError):
    """A depth pass could not give a usable result. The message is written for the person running it."""


class NonFiniteDepth(DepthFailure):
    """Inference produced NaN/Inf — see the precision note at the top of this module."""


class NotEnoughMemory(DepthFailure):
    """This machine cannot spare what the pass needs — see `memory.py`."""


class _Stopped(NotEnoughMemory):
    """The watch's stop, raised at a stop point. `infer` raises a fresh NotEnoughMemory in its place."""


class PassAlreadyRunning(NotEnoughMemory):
    """A second pass in a process that is running one: the memory it was sized with is already taken."""


ALREADY_RUNNING = (
    "Another depth pass is already running in this app, and the memory check before a pass counts on "
    "there being only one.\n  Wait for it to finish, then run this one again."
)

# One depth pass at a time in a process, from the check before it to its end. The check sizes a pass
# against what is free with no other pass of ours running, `limit_gpu` sets one limit for the whole
# process, and the hooks rebind names in upstream's module for the length of a pass and put back what
# they found, so two passes at once would size each other wrong and could leave each other's hooks in
# place. A second pass is refused rather than made to wait: the UI already queues its conversions one
# at a time (Gradio runs each event one call at a time unless told otherwise), so only a caller
# running passes on threads of its own meets this, and waiting would hang a pass started from inside
# another's progress callback on the lock its own thread holds.
_ONE_PASS = threading.Lock()


@contextmanager
def _only_pass():
    if not _ONE_PASS.acquire(blocking=False):
        raise PassAlreadyRunning(ALREADY_RUNNING)
    try:
        yield
    finally:
        _ONE_PASS.release()


STOPPED_AT_LIMIT = (
    f"The depth pass stopped at its memory limit before it finished. Close other apps and "
    f"run it again, or use a smaller --input-size. To allow more, set "
    f"{memory.OVERRIDE_ENV} to the GB this pass may use."
)


def is_out_of_memory(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def normalize(
    depth: np.ndarray, invert: bool = False, between: Callable[[], None] | None = None
) -> tuple[np.ndarray, float, float]:
    """Global (whole-clip) min/max normalization to uint8.

    This is upstream's normalization *logic*, reused deliberately: normalizing per-video
    rather than per-frame is what keeps brightness from pumping between frames. Combined
    with VDA's cross-window scale/shift alignment it is why the output is flicker-free.

    VDA emits disparity, so larger values are nearer: the default is near = bright.
    `between` runs between its whole-clip steps: `infer` hands it the memory watch's stop point.
    """
    # Casting NaN to uint8 is silent and yields zeros, so a broken run would otherwise
    # export as a plausible-looking all-black depth video. Fail loudly instead.
    if not np.isfinite(depth).all():
        raise NonFiniteDepth(
            "Depth output contains NaN/Inf — refusing to encode a corrupt depth video."
        )
    if between:
        between()

    d_min = float(depth.min())
    d_max = float(depth.max())
    if between:
        between()
    span = d_max - d_min
    if span <= 0:
        # Degenerate (perfectly flat) depth — emit mid-gray rather than divide by zero.
        out = np.full(depth.shape, 128, dtype=np.uint8)
    else:
        out = ((depth - d_min) / span * 255.0).astype(np.uint8)
    if invert:
        out = 255 - out
    return out, d_min, d_max


def _run_windows(
    model,
    frames: np.ndarray,
    fps: float,
    input_size: int,
    device: str,
    use_fp32: bool,
    on_window: ProgressFn | None,
) -> tuple[np.ndarray, float]:
    try:
        with _window_hook(device, on_window):
            return model.infer_video_depth(
                frames, fps, input_size=input_size, device=device, fp32=use_fp32
            )
    except NotEnoughMemory:  # the watch's stop: `infer` lets go of the pass and says so
        raise
    except RuntimeError as e:
        if not is_out_of_memory(e):
            raise
        raise NotEnoughMemory(STOPPED_AT_LIMIT) from e
    finally:
        empty_cache(device)


def infer(
    frames: np.ndarray,
    fps: float,
    input_size: int = 518,
    precision: str = "fp32",
    invert: bool = False,
    device: str | None = None,
    on_window: ProgressFn | None = None,
    on_warn: WarnFn | None = None,
) -> DepthResult:
    """Run windowed depth inference over [N, H, W, 3] uint8 RGB frames."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [N, H, W, 3] RGB frames, got {frames.shape}")
    if input_size % 14:
        raise ValueError(f"--input-size must be a multiple of 14 (DINOv2 patches), got {input_size}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    with _only_pass():
        return _one_pass(frames, fps, input_size, precision, invert, device, on_window, on_warn)


def _one_pass(
    frames: np.ndarray,
    fps: float,
    input_size: int,
    precision: str,
    invert: bool,
    device: str | None,
    on_window: ProgressFn | None,
    on_warn: WarnFn | None,
) -> DepthResult:
    # Before the model loads, so a pass that cannot fit touches nothing.
    plan = memory.plan(frames.shape, input_size)
    if plan.refusal:
        raise NotEnoughMemory(plan.refusal)

    model, device = load_model(device)
    limit_gpu(device, plan.gpu_limit)
    # CPU autocast would be pointless; fp32 is also the default everywhere (see module docs).
    use_fp32 = precision == "fp32" or device == "cpu"

    start = time.perf_counter()
    depth = normalized = None
    # ponytail: a stop lands at a stop point, never inside one numpy call, so the longest single call
    # sets how late it can come. Upstream's stack, and each step of `normalize`, goes over the whole
    # clip's depth in one call: on 30 s of 1280 x 534 the stack took 0.18 s and all of `normalize`
    # 0.67 s, in three calls. The watch's thread reads on through them. If a stop ever comes too late,
    # run `normalize` a slice of frames at a time with a stop point between slices; the stack is
    # upstream's, so it stays one call.
    try:
        with watching():
            depth, out_fps = _run_windows(
                model, frames, fps, input_size, device, use_fp32, on_window
            )

            # Reduced precision NaNs out on MPS. Rather than hand the user a black video, detect it
            # and redo the clip at fp32 — which is the faster path anyway.
            if not use_fp32 and not np.isfinite(depth).all():
                message = (
                    f"{precision} produced non-finite depth on {device} (a known MPS defect) — "
                    f"retrying at fp32."
                )
                if on_warn:
                    on_warn(message)
                del depth
                gc.collect()
                # Mandatory: the poisoned graph survives into the next fp32 run without this.
                empty_cache(device)
                use_fp32 = True
                depth, out_fps = _run_windows(
                    model, frames, fps, input_size, device, True, on_window
                )

            elapsed = time.perf_counter() - start
            _stop_if_told()
            normalized, raw_min, raw_max = normalize(depth, invert=invert, between=_stop_if_told)
            # Back-to-back conversions in one long-lived Gradio session must not creep toward
            # OOM: drop the float32 intermediates before returning the uint8 result.
            depth = None
            _stop_if_told()  # what the watch read during the last step counts too
    except _Stopped as stop:
        message = str(stop)
    else:
        gc.collect()
        empty_cache(device)
        return DepthResult(
            frames=normalized,
            fps=out_fps,
            device=device,
            dtype="fp32" if use_fp32 else precision,
            elapsed=elapsed,
            raw_min=raw_min,
            raw_max=raw_max,
            inverted=invert,
        )
    # Raised out here, not from the handler: the watch's error holds what the stopped pass held, the
    # window's frames and tensors or the clip's depth, and a chained one would keep them alive in a UI
    # that goes on running. Once it and the depth here are gone, collect them, then hand the GPU back.
    depth = normalized = None
    gc.collect()
    _settle(device)
    raise NotEnoughMemory(message)


def expected_windows(frame_count: int) -> int:
    """How many inference windows a clip of this length will take (for progress bars)."""
    step = INFER_LEN - OVERLAP
    return max(1, -(-frame_count // step))
