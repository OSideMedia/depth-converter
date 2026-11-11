"""The watch on a running depth pass: a machine that runs out partway stops the pass in plain words.

No model and no real memory reading. A fake sampler says what is free, and a stand-in model runs
its windows through the same settle points `engine.bound_memory` installs on the real one.
"""

from __future__ import annotations

import sys
import types
import weakref

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from depthc import cli, engine, memory

GB = memory.GB
LOW = int(0.1 * GB)  # far under any floor
PLENTY = 12 * GB
# Where a stand-in window settles: once per backbone group, then once per temporal stage.
SETTLES_PER_WINDOW = engine.INFER_LEN // engine.BACKBONE_FRAMES + 4
runner = CliRunner()


class Sampler:
    """What the machine says is free: `before` readings of plenty, then `after` from there on."""

    def __init__(self, before: int = 1, after: int | None = LOW):
        self.before, self.after, self.calls = before, after, 0

    def __call__(self) -> int | None:
        self.calls += 1
        return PLENTY if self.calls <= self.before else self.after


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _watch(readings: list[int | None]) -> list[str | None]:
    """A watch fed these readings, one full interval apart."""
    clock, feed = Clock(), iter(readings)
    watch = memory.Watch(read=lambda: next(feed), clock=clock)
    said = []
    for _ in readings:
        said.append(watch.check())
        clock.now += memory.WATCH_EVERY
    return said


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


def test_it_reads_no_more_often_than_its_interval():
    """Each reading starts `vm_stat`, and a pass settles a dozen times a window."""
    clock, reads = Clock(), []
    watch = memory.Watch(read=lambda: reads.append(clock.now) or PLENTY, clock=clock)
    for now in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        clock.now = now
        watch.check()
    assert reads == [0.0, 0.6, 1.2]
    assert memory.WATCH_EVERY == 0.5


def test_a_mac_that_stops_answering_partway_is_stopped_not_waved_through(monkeypatch):
    """`vm_stat` failing is what a stalling Mac looks like, so it counts as under the floor."""
    monkeypatch.setattr(memory, "_MAC", True)
    said = _watch([PLENTY] + [None] * memory.WATCH_READINGS)
    assert said[:-1] == [None] * memory.WATCH_READINGS
    assert said[-1] is not None and "stopped reporting" in said[-1]
    # One failed reading between good ones is a hiccup, not a stall.
    assert _watch([None, PLENTY] * 5) == [None] * 10


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
    """Runs windows the way upstream does: the backbone a group at a time, then the temporal stages."""

    def __init__(self, **_):
        super().__init__()
        self.pretrained, self.head = _Backbone(), _Head()
        self.windows = 0
        self.sizes: list[int] = []
        self.held: weakref.ref | None = None

    def forward(self, x):  # one window, which is all smoke runs
        self.pretrained.get_intermediate_layers(x.flatten(0, 1), [0], return_class_token=True)
        for stage in self.head.motion_modules:
            stage(x)
        self.windows += 1
        return x[:, :, 0]

    def infer_video_depth(self, frames, fps, input_size=518, device="cpu", fp32=True):
        import video_depth_anything.video_depth as vda

        self.sizes.append(input_size)
        held = _Held()
        self.held = weakref.ref(held)
        for _ in vda.tqdm(range(0, frames.shape[0], engine.INFER_LEN - engine.OVERLAP)):
            self.forward(torch.zeros(1, engine.INFER_LEN, 3, 2, 2))
        return np.random.rand(*frames.shape[:3]).astype(np.float32), fps


@pytest.fixture
def stand_in(monkeypatch):
    """The model every pass gets, wrapped by the real `bound_memory`, behind a stand-in vendor module."""
    vda = types.ModuleType("video_depth_anything.video_depth")
    vda.tqdm = lambda iterable, *_a, **_k: iterable
    vda.VideoDepthAnything = StandIn  # what smoke builds
    monkeypatch.setitem(sys.modules, "video_depth_anything", types.ModuleType("video_depth_anything"))
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", vda)

    model = StandIn()
    engine.bound_memory(model, "cpu")
    monkeypatch.setattr(engine, "load_model", lambda device=None: (model, "cpu"))
    monkeypatch.setattr(memory, "WATCH_EVERY", 0.0)  # every settle reads; time is tested above
    monkeypatch.setenv(memory.OVERRIDE_ENV, "20")  # let in before the pass, without a reading
    return model


def test_memory_falling_partway_stops_the_pass_at_the_next_settle(stand_in, monkeypatch):
    sampler = Sampler(before=2 * SETTLES_PER_WINDOW)
    monkeypatch.setattr(memory, "available_bytes", sampler)
    frames = np.zeros((110, 8, 8, 3), np.uint8)  # five windows

    with pytest.raises(engine.NotEnoughMemory, match="stopped partway") as caught:
        engine.infer(frames, 24.0)

    assert stand_in.windows == 2, "the pass ran on past the stop"
    assert sampler.calls == 2 * SETTLES_PER_WINDOW + memory.WATCH_READINGS
    assert engine.STOPPED_AT_LIMIT not in str(caught.value)
    assert stand_in.sizes == [518], "a stopped pass must not be run again, at any size"


def test_the_override_does_not_let_a_pass_run_the_machine_out(stand_in, monkeypatch):
    """The stand-in fixture sets DEPTHC_MEMORY_GB, so the test above already runs under it. Said
    here on its own so a change to that fixture cannot quietly drop the case."""
    import os

    assert os.environ[memory.OVERRIDE_ENV] == "20"
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=0))
    with pytest.raises(engine.NotEnoughMemory, match="stopped partway"):
        engine.infer(np.zeros((48, 8, 8, 3), np.uint8), 24.0)


def test_a_mac_that_stops_answering_partway_stops_the_pass(stand_in, monkeypatch):
    monkeypatch.setattr(memory, "_MAC", True)
    monkeypatch.setattr(memory, "available_bytes", Sampler(before=SETTLES_PER_WINDOW, after=None))
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
    under = memory.WATCH_FLOOR - 1
    almost = [under] * (memory.WATCH_READINGS - 1)  # one short of a stop
    first = [PLENTY] * (SETTLES_PER_WINDOW - len(almost)) + almost  # ends on the near miss
    second = [under] + [PLENTY] * (SETTLES_PER_WINDOW - 1)  # opens on one more dip
    readings = iter(first + second)
    monkeypatch.setattr(memory, "available_bytes", lambda: next(readings))
    frames = np.zeros((10, 8, 8, 3), np.uint8)  # one window each
    engine.infer(frames, 24.0)
    engine.infer(frames, 24.0)
    assert stand_in.windows == 2


def test_nothing_is_read_outside_a_pass(stand_in, monkeypatch):
    sampler = Sampler(before=0)
    monkeypatch.setattr(memory, "available_bytes", sampler)
    stand_in(torch.zeros(1, engine.INFER_LEN, 3, 2, 2))
    assert sampler.calls == 0


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


# ------------------------------------------------------------------ every surface

class TestEverySurfaceSaysItPlainly:
    """Each command that runs a pass stops with the watch's words, exit 1, and nothing written."""

    @pytest.fixture(autouse=True)
    def _runs_out_after_one_window(self, stand_in, monkeypatch):
        self.model = stand_in
        monkeypatch.setattr(memory, "available_bytes", Sampler(before=SETTLES_PER_WINDOW))

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
