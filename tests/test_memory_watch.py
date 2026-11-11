"""The watch on a running depth pass: a machine that runs out partway stops the pass in plain words.

No model and no real memory reading. A fake sampler says what is free, and a stand-in model runs
its windows through the same settle points `engine.bound_memory` installs on the real one, then
does what upstream does after its last window: the alignment and the stack.
"""

from __future__ import annotations

import sys
import threading
import time
import types
import weakref
from typing import Callable

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from depthc import cli, engine, memory

GB = memory.GB
LOW = int(0.1 * GB)  # far under any floor
PLENTY = 12 * GB
# Where a stand-in pass can stop in each window: once per backbone group, once per temporal stage,
# and once the window is done.
STOPS_PER_WINDOW = engine.INFER_LEN // engine.BACKBONE_FRAMES + 4 + 1
WATCH_THREAD = "depthc-memory-watch"
runner = CliRunner()


class Sampler:
    """What the machine says is free: `before` readings of plenty, then `after` from there on."""

    def __init__(self, before: int = 1, after: int | None = LOW):
        self.before, self.after, self.calls = before, after, 0

    def __call__(self) -> int | None:
        self.calls += 1
        return PLENTY if self.calls <= self.before else self.after


def _watch(readings: list[int | None]) -> list[str | None]:
    """A watch fed these readings one at a time, as its thread would, and asked after each."""
    feed = iter(readings)
    watch = memory.Watch(read=lambda: next(feed))
    said = []
    for _ in readings:
        watch.reading()
        said.append(watch.check())
    return said


def _watch_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == WATCH_THREAD]


# ------------------------------------------------------------------ the watch alone

def test_memory_that_stays_under_the_floor_stops_the_pass():
    under = memory.WATCH_FLOOR - 1
    said = _watch([PLENTY] + [under] * memory.WATCH_READINGS)
    assert said[:-1] == [None] * memory.WATCH_READINGS, "it stopped before the floor had held"
    assert said[-1] is not None and "stopped partway" in said[-1]


def test_a_dip_that_recovers_never_stops_a_good_pass():
    """The floor has to hold for more than one reading: a moment's dip is what a healthy Mac does."""
    assert memory.WATCH_READINGS >= 2
    under = memory.WATCH_FLOOR - 1
    dips = ([under] * (memory.WATCH_READINGS - 1) + [memory.WATCH_FLOOR]) * 5
    assert _watch(dips) == [None] * len(dips)


def test_once_the_floor_has_held_the_stop_stands_until_the_pass_gets_to_it():
    """The thread reads on between two stop points. A reading back over the floor in that gap must
    not take back a stop the pass has not reached yet."""
    under = memory.WATCH_FLOOR - 1
    said = _watch([under] * memory.WATCH_READINGS + [PLENTY, PLENTY])
    assert said[-1] is not None and said[-1] == said[memory.WATCH_READINGS - 1]


def test_it_reads_once_every_interval():
    """Each reading starts `vm_stat`, so it reads no more often than its interval."""
    reads, waits = [], []
    watch = memory.Watch(read=lambda: reads.append(1) or PLENTY)

    def wait(seconds: float) -> bool:  # stands in for the thread's wait; `close` ends the third
        waits.append(seconds)
        return len(waits) == 3

    watch._sample(wait)
    assert len(reads) == 3 and waits == [memory.WATCH_EVERY] * 3
    assert memory.WATCH_EVERY == 0.5


def test_a_mac_that_stops_answering_partway_is_stopped_not_waved_through(monkeypatch):
    """`vm_stat` failing is what a stalling Mac looks like, so it counts as under the floor."""
    monkeypatch.setattr(memory, "_MAC", True)
    said = _watch([PLENTY] + [None] * memory.WATCH_READINGS)
    assert said[:-1] == [None] * memory.WATCH_READINGS
    assert said[-1] is not None and "stopped reporting" in said[-1]
    # One failed reading between good ones is a hiccup, not a stall.
    assert _watch([None, PLENTY] * 5) == [None] * 10


def test_a_reading_that_blows_up_is_a_failed_reading_not_a_dead_watch(monkeypatch):
    """The thread must not die on an error and leave the rest of the pass unwatched."""
    monkeypatch.setattr(memory, "_MAC", True)

    def broken() -> int:
        raise ValueError("vm_stat said something new")

    watch = memory.Watch(read=broken)
    for _ in range(memory.WATCH_READINGS):
        watch.reading()
    assert watch.check() is not None and "stopped reporting" in watch.check()


def test_off_the_mac_there_is_nothing_to_read_so_the_watch_stands_down(monkeypatch):
    """No `vm_stat` at all, and the check before the pass already works from half of RAM there.
    Counting that as a stall would stop every pass on the first windows."""
    monkeypatch.setattr(memory, "_MAC", False)
    assert _watch([None] * 10) == [None] * 10
    assert _watch([LOW] * memory.WATCH_READINGS)[-1] is not None, "a number it is given still counts"


def test_the_refusal_says_what_happened_how_much_was_free_and_what_to_do(monkeypatch):
    monkeypatch.setattr(memory, "total_bytes", lambda: 24 * GB)
    monkeypatch.setattr(memory, "WATCH_FLOOR", int(0.75 * GB))
    words = _watch([int(0.4 * GB)] * memory.WATCH_READINGS)[-1]
    assert words is not None
    assert "stopped partway" in words
    assert "0.4 GB free out of 24.0 GB" in words
    assert "under the 0.75 GB" in words, "the floor was rounded into a number it is not"
    assert f"{memory.WATCH_READINGS} readings in a row" in words
    assert "Nothing was written" in words and "Close other apps" in words
    # The setting sizes a pass; it does not let one run the machine out, so it is not offered here.
    assert memory.OVERRIDE_ENV not in words


def test_the_floor_sits_under_what_the_check_before_a_pass_keeps_free():
    """A pass let in at the edge leaves that much free by design. A floor above it stops good passes."""
    assert 0 < memory.WATCH_FLOOR < memory._KEEP_FREE


def test_its_thread_reads_on_while_the_pass_sits_in_one_long_call(monkeypatch):
    """Inside one numpy call the pass never comes back to Python, so only a thread can read there."""
    monkeypatch.setattr(memory, "WATCH_EVERY", 0.01)
    sampler, settled = Sampler(before=0), threading.Event()

    def read() -> int | None:
        value = sampler()
        if sampler.calls > memory.WATCH_READINGS:  # the third has been counted by now
            settled.set()
        return value

    watch = memory.Watch(read=read)
    watch.start()
    try:
        # This thread waits without coming back to Python, as it does inside one numpy call.
        assert settled.wait(timeout=10), "nothing was read while the pass sat in one call"
        assert watch.check() is not None
    finally:
        watch.close()
    taken = sampler.calls
    time.sleep(0.1)
    assert sampler.calls == taken, "it went on reading after the pass had ended"
    assert not _watch_threads()


def test_closing_waits_for_a_slow_reading_to_end(monkeypatch):
    """A slow `vm_stat` is what a struggling Mac looks like. Closing must not hand back while that
    reading is still running, or the stop it settles comes after anything could check it."""
    monkeypatch.setattr(memory, "WATCH_EVERY", 0.01)
    slow = 1.5  # past a one-second wait, inside the time a real reading is given
    assert slow < memory.READ_TIMEOUT
    inside, calls = threading.Event(), []

    def read() -> int:
        calls.append(1)
        if len(calls) == 2:
            inside.set()
            time.sleep(slow)
        return PLENTY

    watch = memory.Watch(read=read)
    watch.start()
    assert inside.wait(timeout=5)
    watch.close()
    assert not _watch_threads(), "close came back with a reading still running"
    taken = len(calls)
    time.sleep(0.05)
    assert len(calls) == taken


# ------------------------------------------------------------------ inside a pass

class _Held:
    """Stands for what a window holds on the GPU."""


class _Backbone(torch.nn.Module):
    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        tiny = x.reshape(x.shape[0], -1)[:, :1]
        return ((tiny, tiny),)


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_modules = torch.nn.ModuleList(torch.nn.Identity() for _ in range(4))


class StandIn(torch.nn.Module):
    """Runs a pass the way upstream does: each window's backbone a group at a time, then its temporal
    stages. After the last window it aligns each window's depth to the last with upstream's two
    helpers and stacks the result with its module's `np`, the names the real one calls there."""

    def __init__(self, **_):
        super().__init__()
        self.pretrained, self.head = _Backbone(), _Head()
        self.started = self.windows = 0
        self.sizes: list[int] = []
        self.held: weakref.ref | None = None
        self.phase = "windows"
        self.aligned = 0
        self.stacked: weakref.ref | None = None
        self.long_call: Callable[[], object] = lambda: None  # a test's one long call, after the windows

    def forward(self, x):  # one window, which is all smoke runs
        self.started += 1
        self.pretrained.get_intermediate_layers(x.flatten(0, 1), [0], return_class_token=True)
        for stage in self.head.motion_modules:
            stage(x)
        self.windows += 1
        return x[:, :, 0]

    def infer_video_depth(self, frames, fps, input_size=518, device="cpu", fp32=True):
        import video_depth_anything.video_depth as vda

        self.sizes.append(input_size)
        self.phase, self.aligned, self.stacked = "windows", 0, None
        held = _Held()
        self.held = weakref.ref(held)
        depth = []
        for _ in vda.tqdm(range(0, frames.shape[0], engine.INFER_LEN - engine.OVERLAP)):
            self.forward(torch.zeros(1, engine.INFER_LEN, 3, 2, 2))
            depth.append(np.random.rand(engine.INFER_LEN, *frames.shape[1:3]).astype(np.float32))

        self.phase = "tail"
        self.long_call()
        aligned = list(depth[0])
        for window in depth[1:]:
            scale, shift = vda.compute_scale_and_shift(window, window, None)
            self.aligned += 1
            aligned[-1:] = vda.get_interpolate_frames(aligned[-1:], [window[engine.OVERLAP - 1] * scale + shift])
            self.aligned += 1
            aligned += list(window[engine.OVERLAP:])
        stacked = vda.np.stack(aligned[: frames.shape[0]], axis=0)
        self.stacked = weakref.ref(stacked)
        self.phase = "stacked"
        return stacked, fps


def _stand_in(monkeypatch) -> StandIn:
    """The model every pass gets, wrapped by the real `bound_memory`, behind a stand-in vendor module."""
    vda = types.ModuleType("video_depth_anything.video_depth")
    vda.tqdm = lambda iterable, *_a, **_k: iterable
    vda.compute_scale_and_shift = lambda *_a: (1.0, 0.0)
    vda.get_interpolate_frames = lambda pre, post: post
    vda.np = np
    vda.VideoDepthAnything = StandIn  # what smoke builds
    monkeypatch.setitem(sys.modules, "video_depth_anything", types.ModuleType("video_depth_anything"))
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", vda)

    model = StandIn()
    engine.bound_memory(model, "cpu")
    monkeypatch.setattr(engine, "load_model", lambda device=None: (model, "cpu"))
    monkeypatch.setenv(memory.OVERRIDE_ENV, "20")  # let in before the pass, without a reading
    return model


@pytest.fixture
def stand_in(monkeypatch):
    """The stand-in, with the watch's thread swapped for one reading at each stop point, as though half
    a second passed between any two, so a test can say exactly where a pass stops. The thread itself
    is tested above and by `threaded` below."""
    model = _stand_in(monkeypatch)
    check = memory.Watch.check

    def read_then_check(self):
        self.reading()
        return check(self)

    monkeypatch.setattr(memory.Watch, "start", lambda self: None)
    monkeypatch.setattr(memory.Watch, "check", read_then_check)
    return model


@pytest.fixture
def threaded(monkeypatch):
    """The stand-in, with the watch's own thread reading every hundredth of a second."""
    model = _stand_in(monkeypatch)
    monkeypatch.setattr(memory, "WATCH_EVERY", 0.01)
    return model


def test_memory_falling_partway_stops_the_pass_at_the_next_stop_point(stand_in, monkeypatch):
    sampler = Sampler(before=2 * STOPS_PER_WINDOW)
    monkeypatch.setattr(memory, "available_bytes", sampler)
    frames = np.zeros((110, 8, 8, 3), np.uint8)  # five windows

    with pytest.raises(engine.NotEnoughMemory, match="stopped partway") as caught:
        engine.infer(frames, 24.0)

    assert stand_in.windows == 2, "the pass ran on past the stop"
    assert sampler.calls == 2 * STOPS_PER_WINDOW + memory.WATCH_READINGS
    assert engine.STOPPED_AT_LIMIT not in str(caught.value)
    assert stand_in.sizes == [518], "a stopped pass must not be run again, at any size"


def test_the_end_of_each_window_is_a_stop_point_too(stand_in, monkeypatch):
    """Once a window's depth is back on the CPU, before the next window or the alignment begins."""
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=STOPS_PER_WINDOW - memory.WATCH_READINGS))
    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)
    assert stand_in.windows == 1 and stand_in.started == 1, "the next window began after the floor had held"


def test_memory_falling_after_the_last_window_stops_the_pass_in_the_alignment(stand_in, monkeypatch):
    """On a long clip the alignment and the stack are the CPU-side peak, and no GPU work settles there."""
    monkeypatch.setattr(memory, "available_bytes", lambda: LOW if stand_in.phase == "tail" else PLENTY)
    frames = np.zeros((110, 8, 8, 3), np.uint8)  # five windows

    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(frames, 24.0)

    assert stand_in.windows == 5, "it stopped before the windows were done"
    assert stand_in.aligned == memory.WATCH_READINGS - 1, "it did not stop at the alignment's stop points"
    assert stand_in.stacked is None, "the whole clip was stacked after the floor had held"
    assert sys.modules["video_depth_anything.video_depth"].np is np, "upstream's module was left rebound"


def test_memory_falling_after_the_stack_stops_the_pass_and_lets_go_of_the_depth(stand_in, monkeypatch):
    """The stack and `normalize` hold the clip's depth more than once. A stop there frees it all."""
    monkeypatch.setattr(memory, "available_bytes", lambda: LOW if stand_in.phase == "stacked" else PLENTY)
    still_held: list[bool] = []
    real_settle = engine._settle

    def settle(device: str) -> None:
        still_held.append(stand_in.stacked is not None and stand_in.stacked() is not None)
        real_settle(device)

    monkeypatch.setattr(engine, "_settle", settle)
    with pytest.raises(engine.NotEnoughMemory, match="stopped partway") as caught:
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)

    assert stand_in.stacked is not None, "it stopped before the stack, not after it"
    assert stand_in.stacked() is None, "the clip's depth is still held after the stop"
    assert still_held[-1] is False, "the GPU was handed back while the depth was still held"
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_a_long_call_after_the_last_window_is_read_through_and_the_pass_stops_right_after(
    threaded, monkeypatch
):
    """The real thread this time. While the pass sits in one call that never comes back to Python,
    as it does in a numpy call over the whole clip, the watch reads on; the next stop point stops it."""
    settled, sampler = threading.Event(), Sampler(before=0)

    def read() -> int | None:
        if threaded.phase != "tail":
            return PLENTY
        value = sampler()
        if sampler.calls > memory.WATCH_READINGS:  # the third has been counted by now
            settled.set()
        return value

    monkeypatch.setattr(memory, "available_bytes", read)
    threaded.long_call = lambda: settled.wait(timeout=5)

    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(np.zeros((110, 8, 8, 3), np.uint8), 24.0)

    assert settled.is_set(), "nothing was read while the pass sat in one long call"
    assert threaded.aligned == 0, "the pass went on past the first stop point after the call"
    assert not _watch_threads(), "the watch outlived its pass"


def test_a_reading_that_ends_while_the_watch_closes_still_stops_the_pass(threaded, monkeypatch):
    """The third reading under the floor can still be running at the pass's last stop point. It ends
    while the watch closes, and the stop it settles must stop the pass, not come after it returned."""
    under, inside, release = [], threading.Event(), threading.Event()

    def read() -> int | None:
        if threaded.phase != "tail":
            return PLENTY
        under.append(1)
        if len(under) == memory.WATCH_READINGS:  # held here until the pass has passed every stop point
            inside.set()
            release.wait(timeout=5)
        return LOW

    real_close = memory.Watch.close

    def close(self) -> None:
        release.set()
        real_close(self)

    monkeypatch.setattr(memory, "available_bytes", read)
    monkeypatch.setattr(memory.Watch, "close", close)
    threaded.long_call = lambda: inside.wait(timeout=5)

    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)

    assert inside.is_set() and threaded.phase == "stacked", "the pass did not get past its last stop point"
    assert not _watch_threads()


def _run_in_thread(name: str, run: Callable[[], object]) -> tuple[threading.Thread, dict]:
    outcome: dict = {}

    def body() -> None:
        try:
            outcome["result"] = run()
        except BaseException as e:  # noqa: BLE001 - handed back to the test to assert on
            outcome["error"] = e

    thread = threading.Thread(target=body, name=name)
    thread.start()
    return thread, outcome


def test_a_second_pass_in_the_same_process_is_refused_and_upstreams_names_come_back(stand_in):
    """Each pass rebinds names in upstream's module and puts back what it found. Two at once put back
    each other's, and the pre-pass check counts on one pass at a time, so a second one is refused."""
    vda = sys.modules["video_depth_anything.video_depth"]
    names = ("tqdm", "compute_scale_and_shift", "get_interpolate_frames", "np")
    originals = {name: getattr(vda, name) for name in names}
    first_in_tail, second_tried, first_done = threading.Event(), threading.Event(), threading.Event()

    def long_call() -> None:
        if threading.current_thread().name == "first":
            first_in_tail.set()
            second_tried.wait(timeout=5)
        else:  # a second pass that got in: hold it until the first has ended
            second_tried.set()
            first_done.wait(timeout=5)

    stand_in.long_call = long_call
    frames = np.zeros((48, 8, 8, 3), np.uint8)

    def first_pass():
        try:
            return engine.infer(frames, 24.0)
        finally:
            first_done.set()

    def second_pass():
        try:
            return engine.infer(frames, 24.0)
        finally:
            second_tried.set()

    first, first_out = _run_in_thread("first", first_pass)
    assert first_in_tail.wait(timeout=5)
    second, second_out = _run_in_thread("second", second_pass)
    first.join(timeout=10)
    second.join(timeout=10)

    assert {name: getattr(vda, name) for name in names} == originals, "upstream's module was left rebound"
    assert isinstance(first_out.get("result"), engine.DepthResult), first_out
    refused = second_out.get("error")
    assert isinstance(refused, engine.NotEnoughMemory), second_out
    assert "already running" in str(refused) and "Wait for it to finish" in str(refused)
    assert stand_in.windows == engine.expected_windows(frames.shape[0]), "the refused pass ran a window"
    assert isinstance(engine.infer(frames, 24.0), engine.DepthResult), "the first pass kept the lock"


def test_an_interrupt_while_the_watch_starts_leaves_no_watch_behind(threaded, monkeypatch):
    """Ctrl-C can land while the watch's thread is starting. The thread, the pass's watch and the
    one-pass lock must all be let go of, or a UI that goes on running keeps them."""
    import contextvars

    monkeypatch.setattr(memory, "available_bytes", lambda: PLENTY)
    real_start, started = memory.Watch.start, []

    def start(self) -> None:  # the thread is running by the time the interrupt lands, as it can be
        real_start(self)
        started.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(memory.Watch, "start", start)
    frames = np.zeros((48, 8, 8, 3), np.uint8)

    def interrupted_pass():
        with pytest.raises(KeyboardInterrupt):
            engine.infer(frames, 24.0)
        return engine._WATCH.get()

    try:
        left = contextvars.copy_context().run(interrupted_pass)
        alive = bool(_watch_threads())
    finally:
        for watch in started:  # so a failure here does not leave a thread reading for the rest of the run
            watch._done.set()
    assert left is None, "the context still holds the interrupted pass's watch"
    assert not alive, "the watch's thread outlived the interrupted pass"
    assert threaded.windows == 0
    monkeypatch.setattr(memory.Watch, "start", real_start)
    assert isinstance(engine.infer(frames, 24.0), engine.DepthResult), "the interrupted pass kept the lock"


def test_an_interrupt_while_the_watch_closes_still_takes_the_watch_off_the_pass(monkeypatch):
    """Closing waits for the thread, up to a reading's own limit, and Ctrl-C can land in that wait.
    The pass's watch must still come off the context, or whatever runs next there inherits it."""
    import contextvars

    monkeypatch.setattr(memory, "available_bytes", lambda: PLENTY)

    def close(self) -> None:
        self._done.set()
        raise KeyboardInterrupt

    monkeypatch.setattr(memory.Watch, "close", close)

    def interrupted_block():
        with pytest.raises(KeyboardInterrupt):
            with engine.watching():
                pass
        return engine._WATCH.get()

    assert contextvars.copy_context().run(interrupted_block) is None


def test_the_override_does_not_let_a_pass_run_the_machine_out(stand_in, monkeypatch):
    """The stand-in fixture sets DEPTHC_MEMORY_GB, so the tests above already run under it. Said
    here on its own so a change to that fixture cannot quietly drop the case."""
    import os

    assert os.environ[memory.OVERRIDE_ENV] == "20"
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=0))
    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)


def test_a_mac_that_stops_answering_partway_stops_the_pass(stand_in, monkeypatch):
    monkeypatch.setattr(memory, "_MAC", True)
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=STOPS_PER_WINDOW, after=None))
    with pytest.raises(engine.NotEnoughMemory, match="stopped reporting"):
        engine.infer(np.zeros((110, 8, 8, 3), np.uint8), 24.0)
    assert stand_in.windows == 1


def test_the_stop_lets_go_of_the_stopped_window_before_the_gpu_is_handed_back(stand_in, monkeypatch):
    """The UI keeps running after a stop. If the error still holds the stopped window's frames, the
    memory the stop was for stays taken."""
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=0))
    still_held: list[bool] = []
    real_settle = engine._settle

    def settle(device: str) -> None:
        still_held.append(stand_in.held is not None and stand_in.held() is not None)
        real_settle(device)

    monkeypatch.setattr(engine, "_settle", settle)
    with pytest.raises(engine.NotEnoughMemory) as caught:
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)

    assert stand_in.held() is None, "the stopped window is still held while the error is"
    assert still_held[-1] is False, "the GPU was handed back while the stopped window still held it"
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_each_pass_counts_from_zero(stand_in, monkeypatch):
    """A watch left over from the last pass would stop the next one on its first dip."""
    frames = np.zeros((10, 8, 8, 3), np.uint8)  # one window each
    counting = Sampler(before=10**6)
    monkeypatch.setattr(memory, "available_bytes", counting)
    engine.infer(frames, 24.0)
    stops = counting.calls  # the stop points in one such pass, each read once here

    under = memory.WATCH_FLOOR - 1
    almost = [under] * (memory.WATCH_READINGS - 1)  # one short of a stop
    first = [PLENTY] * (stops - len(almost)) + almost  # ends on the near miss
    second = [under] + [PLENTY] * (stops - 1)  # opens on one more dip
    readings = iter(first + second)
    monkeypatch.setattr(memory, "available_bytes", lambda: next(readings))
    engine.infer(frames, 24.0)
    engine.infer(frames, 24.0)
    assert stand_in.windows == 3


def test_nothing_is_read_outside_a_pass(stand_in, monkeypatch):
    sampler = Sampler(before=0)
    monkeypatch.setattr(memory, "available_bytes", sampler)
    stand_in(torch.zeros(1, engine.INFER_LEN, 3, 2, 2))
    assert sampler.calls == 0
    assert not _watch_threads()


def test_smoke_runs_its_window_under_the_same_watch(stand_in, monkeypatch, tmp_path, capsys):
    from depthc import smoke

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"x")
    monkeypatch.setattr(smoke, "ensure_vendor_on_path", lambda: None)
    monkeypatch.setattr(smoke, "ensure_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(smoke, "vendor_commit", lambda: smoke.VDA_COMMIT)
    monkeypatch.setattr(smoke.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(engine.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(engine.torch, "load", lambda *_a, **_k: {})
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=2))

    assert smoke.main() == 1
    out = capsys.readouterr().out
    assert "stopped partway" in out and "Close other apps" in out
    assert "Traceback" not in out and "GATE 0 OK" not in out


@pytest.mark.vendor
def test_the_stop_points_after_the_last_window_reach_upstreams_own_calls():
    """The stand-in calls these names because upstream does. This holds that against upstream."""
    import inspect

    engine.ensure_vendor_on_path()
    import video_depth_anything.video_depth as vda

    after_the_windows = inspect.getsource(vda.VideoDepthAnything.infer_video_depth).split("del frame_list")[1]
    for call in ("compute_scale_and_shift(", "get_interpolate_frames(", "np.stack("):
        assert call in after_the_windows, call
    names = ("tqdm", "compute_scale_and_shift", "get_interpolate_frames", "np")
    before = {name: getattr(vda, name) for name in names}
    with engine._window_hook("cpu", None):
        assert all(getattr(vda, name) is not before[name] for name in names)
        assert vda.np.concatenate is before["np"].concatenate, "the rest of numpy is not upstream's"
    assert {name: getattr(vda, name) for name in names} == before


# ------------------------------------------------------------------ every surface

class TestEverySurfaceSaysItPlainly:
    """Each command that runs a pass stops with the watch's words, exit 1, and nothing written."""

    @pytest.fixture(autouse=True)
    def _runs_out_after_one_window(self, stand_in, monkeypatch):
        self.model = stand_in
        monkeypatch.setattr(memory, "available_bytes", Sampler(before=STOPS_PER_WINDOW))

    def _stopped(self, result) -> None:
        text = result.output
        assert result.exit_code == 1, text
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "stopped partway" in text and "Close other apps" in text
        assert "Traceback" not in text
        assert self.model.windows == 1, "the pass went on past the stop"

    def test_kit_leaves_no_kit_behind(self, clip_3s, tmp_path):
        lib = tmp_path / "lib"
        self._stopped(runner.invoke(cli.app, ["kit", str(clip_3s), "--library", str(lib), "--tags", "x"]))
        assert not lib.exists() or not [p for p in lib.iterdir() if p.is_dir()]

    def test_convert_leaves_no_depth_file(self, clip_3s, tmp_path):
        out = tmp_path / "d.mp4"
        self._stopped(runner.invoke(cli.app, ["convert", str(clip_3s), "--out", str(out)]))
        assert list(tmp_path.iterdir()) == []

    def test_describe(self, clip_3s):
        self._stopped(runner.invoke(cli.app, ["describe", str(clip_3s), "--json"]))

    def test_compare(self, clip_3s, tmp_path):
        import shutil

        from test_cli import _fake_kit

        kit_dir = _fake_kit(tmp_path)
        shutil.copy(clip_3s, kit_dir / "depth.mp4")
        self._stopped(runner.invoke(cli.app, ["compare", str(kit_dir), str(clip_3s)]))

    def test_scout(self, clip_3s):
        self._stopped(runner.invoke(cli.app, ["scout", str(clip_3s)]))

    def test_inbox_says_so_for_that_clip_and_carries_on(self, clip_3s, tmp_path):
        import shutil

        shutil.copy(clip_3s, tmp_path / "one.mp4")
        result = runner.invoke(cli.app, ["inbox", "--inbox", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "scout failed" in result.output and "stopped partway" in result.output
        assert "Traceback" not in result.output

    def test_ui_leaves_no_kit_behind(self, clip_3s, tmp_path):
        from depthc import ui

        *_, status = ui._convert(
            str(clip_3s), "memtest", "", "seedance-2.5", "720p", "auto", "conform-24", 0, 3,
            False, True, "auto", 518, "fp32", False, False, str(tmp_path),
        )
        assert "stopped partway" in status and "Close other apps" in status
        assert "Traceback" not in status and "Unexpected error" not in status
        assert not [p for p in tmp_path.iterdir() if p.is_dir()], "a stopped pass left a kit folder behind"
        assert self.model.windows == 1
