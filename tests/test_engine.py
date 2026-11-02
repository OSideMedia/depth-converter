from __future__ import annotations

import contextlib

import numpy as np
import torch
import pytest

from depthc import engine, memory
from depthc.video_io import probe, read_frames


# ------------------------------------------------------------------ unit (fast)

def test_normalize_spans_full_range():
    depth = np.array([[[0.0, 1.0], [2.0, 4.0]]], dtype=np.float32)
    out, lo, hi = engine.normalize(depth)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    assert (lo, hi) == (0.0, 4.0)


def test_normalize_near_is_bright_by_default():
    """VDA emits disparity, so the largest value is the nearest surface."""
    depth = np.array([[[1.0, 9.0]]], dtype=np.float32)
    out, _, _ = engine.normalize(depth)
    assert out[0, 0, 1] > out[0, 0, 0]


def test_normalize_invert_flips():
    depth = np.array([[[1.0, 9.0]]], dtype=np.float32)
    out, _, _ = engine.normalize(depth, invert=True)
    assert out[0, 0, 1] < out[0, 0, 0]


def test_normalize_rejects_nan():
    """NaN casts silently to 0, i.e. a plausible-looking all-black video. Must raise."""
    depth = np.array([[[1.0, np.nan]]], dtype=np.float32)
    with pytest.raises(engine.NonFiniteDepth):
        engine.normalize(depth)


def test_normalize_rejects_inf():
    with pytest.raises(engine.NonFiniteDepth):
        engine.normalize(np.array([[[1.0, np.inf]]], dtype=np.float32))


def test_normalize_handles_flat_depth():
    """A perfectly flat clip must not divide by zero."""
    out, lo, hi = engine.normalize(np.full((2, 4, 4), 3.0, dtype=np.float32))
    assert lo == hi == 3.0
    assert (out == 128).all()


def test_normalize_is_global_not_per_frame():
    """The anti-flicker property: a dimming clip must stay dim, not be re-stretched."""
    bright = np.full((1, 4, 4), 10.0, dtype=np.float32)
    dim = np.full((1, 4, 4), 5.0, dtype=np.float32)
    out, _, _ = engine.normalize(np.concatenate([bright, dim]), invert=False)
    assert out[0].mean() > out[1].mean()


def test_expected_windows():
    assert engine.expected_windows(1) == 1
    assert engine.expected_windows(22) == 1
    assert engine.expected_windows(23) == 2
    assert engine.expected_windows(220) == 10


def test_infer_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\[N, H, W, 3\]"):
        engine.infer(np.zeros((4, 64, 64), dtype=np.uint8), 24)


def test_infer_rejects_non_multiple_of_14():
    frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of 14"):
        engine.infer(frames, 24, input_size=100)


def test_infer_rejects_unknown_precision():
    frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="precision"):
        engine.infer(frames, 24, precision="int8")


def test_fp32_is_the_default():
    """Guards the inverted escape hatch: fp16 NaNs out on MPS, so fp32 must stay default."""
    import inspect

    assert inspect.signature(engine.infer).parameters["precision"].default == "fp32"


@pytest.mark.vendor
def test_window_hook_restores_tqdm():
    """The hook must never leave upstream's module mutated."""
    engine.ensure_vendor_on_path()
    import video_depth_anything.video_depth as vda

    original = vda.tqdm
    with engine._window_hook("cpu", None):
        assert vda.tqdm is not original
    assert vda.tqdm is original


@pytest.mark.vendor
def test_window_hook_restores_tqdm_on_exception():
    engine.ensure_vendor_on_path()
    import video_depth_anything.video_depth as vda

    original = vda.tqdm
    with pytest.raises(RuntimeError):
        with engine._window_hook("cpu", None):
            raise RuntimeError("boom")
    assert vda.tqdm is original


@pytest.mark.vendor
def test_window_hook_reports_progress():
    """Progress must fire *after* each body, so a window counts only once it is done."""
    engine.ensure_vendor_on_path()
    import video_depth_anything.video_depth as vda

    seen: list[tuple[int, int]] = []
    order: list[str] = []
    with engine._window_hook("cpu", lambda done, total: (seen.append((done, total)), order.append("hook"))):
        for _ in vda.tqdm(range(3)):
            order.append("body")

    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert order == ["body", "hook", "body", "hook", "body", "hook"]


# ------------------------------------------------------------------ memory (fast)

class _FakeBackbone:
    """Answers like DINOv2: per layer, (patch tokens, class token), one row per frame."""

    def __init__(self):
        self.batches: list[int] = []

    def __call__(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        self.batches.append(x.shape[0])
        flat = x.flatten(1)
        if not return_class_token:
            return tuple(flat * (layer + 1) for layer in n)
        return tuple((flat * (layer + 1), flat[:, :1] * (layer + 1)) for layer in n)


def test_backbone_never_sees_more_than_a_group_and_the_answer_does_not_change():
    frames = torch.arange(32 * 3 * 2 * 2, dtype=torch.float32).reshape(32, 3, 2, 2)
    layers = [2, 5, 8, 11]
    whole = _FakeBackbone()(frames, layers, return_class_token=True)

    backbone = _FakeBackbone()
    grouped = engine._in_groups(backbone, "cpu")(frames, layers, return_class_token=True)

    assert max(backbone.batches) == engine.BACKBONE_FRAMES
    assert sum(backbone.batches) == 32
    assert len(grouped) == len(whole)
    for (patches, token), (whole_patches, whole_token) in zip(grouped, whole):
        assert torch.equal(patches, whole_patches)  # same values, same frame order
        assert torch.equal(token, whole_token)


def test_group_size_is_small_enough_to_matter():
    """Attention memory is per batch, so this figure IS the fix. 8 was measured no better than 4."""
    assert engine.BACKBONE_FRAMES <= 4


def test_a_batch_that_does_not_fill_the_last_group_still_comes_back_whole():
    frames = torch.zeros(5, 3, 2, 2)
    backbone = _FakeBackbone()
    out = engine._in_groups(backbone, "cpu")(frames, [0], return_class_token=True)
    assert backbone.batches == [4, 1]
    assert out[0][0].shape[0] == 5 and out[0][1].shape[0] == 5


def test_a_call_upstream_never_makes_passes_through_untouched():
    frames = torch.zeros(32, 3, 2, 2)
    backbone = _FakeBackbone()
    out = engine._in_groups(backbone, "cpu")(frames, [0])
    assert backbone.batches == [32]
    assert out[0].shape[0] == 32


class _FakeModelBackbone(torch.nn.Module):
    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        return ((x, x),)


class _FakeModelHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_modules = torch.nn.ModuleList(torch.nn.Identity() for _ in range(4))


def _fake_vendor(monkeypatch) -> list:
    """Stand in for the vendored model, so the wiring is testable where the assets are absent."""
    import sys
    import types

    built: list = []

    class Fake(torch.nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.pretrained, self.head = _FakeModelBackbone(), _FakeModelHead()
            built.append(self)

        def forward(self, x):
            return x[:, :, 0] + torch.rand_like(x[:, :, 0])

    fake = types.ModuleType("video_depth_anything.video_depth")
    fake.VideoDepthAnything = Fake
    monkeypatch.setitem(sys.modules, "video_depth_anything", types.ModuleType("video_depth_anything"))
    monkeypatch.setitem(sys.modules, "video_depth_anything.video_depth", fake)
    monkeypatch.setattr(engine.torch, "load", lambda *_a, **_k: {})
    return built


def test_load_model_installs_both_memory_measures_once(monkeypatch):
    """The unit tests above prove grouping works; this proves the real door actually uses it."""
    _fake_vendor(monkeypatch)
    monkeypatch.setattr(engine, "ensure_vendor_on_path", lambda: None)
    monkeypatch.setattr(engine, "ensure_checkpoint", lambda: "unused")
    monkeypatch.setitem(engine._MODEL_CACHE, "cpu", None)
    del engine._MODEL_CACHE["cpu"]

    settled: list[str] = []
    monkeypatch.setattr(engine, "_settle", settled.append)

    model, device = engine.load_model("cpu")
    assert device == "cpu"
    assert engine.load_model("cpu")[0] is model  # the cache must not wrap it a second time

    model.pretrained.get_intermediate_layers(torch.zeros(10, 1), [0], return_class_token=True)
    assert len(settled) == 3, "10 frames in groups of 4 is 3 groups, each settled once"

    settled.clear()
    for stage in model.head.motion_modules:
        stage(torch.zeros(1))
    assert len(settled) == 4, "one settle per temporal stage"


def _smoke_on_a_cpu_machine(monkeypatch, tmp_path):
    from depthc import smoke

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"x")
    monkeypatch.setattr(smoke, "ensure_vendor_on_path", lambda: None)
    monkeypatch.setattr(smoke, "ensure_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(smoke, "vendor_commit", lambda: smoke.VDA_COMMIT)
    monkeypatch.setattr(smoke.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(engine.torch.backends.mps, "is_available", lambda: False)
    return smoke, _fake_vendor(monkeypatch)


def test_smoke_refuses_on_a_machine_that_cannot_fit_it_before_building_the_model(monkeypatch, tmp_path, capsys):
    """Every new user runs this straight after install, so it cannot be the door the stall comes through."""
    smoke, built = _smoke_on_a_cpu_machine(monkeypatch, tmp_path)
    monkeypatch.setenv(memory.OVERRIDE_ENV, "0.1")

    assert smoke.main() == 1
    assert "Not enough free memory" in capsys.readouterr().out
    assert built == [], "the model was built for a run that was refused"


def test_smoke_runs_under_the_same_two_measures_as_a_real_pass(monkeypatch, tmp_path):
    smoke, built = _smoke_on_a_cpu_machine(monkeypatch, tmp_path)
    monkeypatch.setenv(memory.OVERRIDE_ENV, "20")
    limits: list[tuple[str, int]] = []
    monkeypatch.setattr(engine, "limit_gpu", lambda device, limit: limits.append((device, limit)))

    assert smoke.main() == 0
    assert "get_intermediate_layers" in built[0].pretrained.__dict__, "the backbone was not grouped"
    assert limits == [("cpu", memory.plan((32, 518, 518, 3), 518).gpu_limit)]


def test_limit_gpu_turns_bytes_into_a_capped_fraction(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(engine.torch.mps, "recommended_max_memory", lambda: 16 * memory.GB, raising=False)
    monkeypatch.setattr(engine.torch.mps, "set_per_process_memory_fraction", calls.append, raising=False)

    engine.limit_gpu("mps", 8 * memory.GB)
    engine.limit_gpu("mps", 64 * memory.GB)  # more than the GPU pool can ever be
    assert calls == [0.5, 1.0]

    engine.limit_gpu("cpu", 8 * memory.GB)
    assert len(calls) == 2, "the limit is an MPS setting; a CPU pass must not touch it"


def test_a_pass_that_does_not_fit_stops_before_the_model_loads(monkeypatch):
    monkeypatch.setenv(memory.OVERRIDE_ENV, "0.1")

    def loaded(*_a, **_k):
        pytest.fail("the model loaded for a pass that was refused")

    monkeypatch.setattr(engine, "load_model", loaded)
    with pytest.raises(engine.NotEnoughMemory, match="Close other apps"):
        engine.infer(np.zeros((4, 720, 1280, 3), dtype=np.uint8), 24)


def test_a_pass_that_fits_gets_the_limit_from_the_plan(monkeypatch):
    monkeypatch.setenv(memory.OVERRIDE_ENV, "20")
    frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    limits: list[tuple[str, int]] = []

    class Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise Stop

    monkeypatch.setattr(engine, "load_model", lambda device=None: (object(), "mps"))
    monkeypatch.setattr(engine, "limit_gpu", lambda device, limit: limits.append((device, limit)))
    monkeypatch.setattr(engine, "_run_windows", stop)
    with pytest.raises(Stop):
        engine.infer(frames, 24)
    assert limits == [("mps", memory.plan(frames.shape, 518).gpu_limit)]


def test_infer_does_not_charge_twice_for_what_the_process_already_holds(monkeypatch):
    """The decoded frames, and what an earlier pass left. Exactly enough only with that credited."""
    monkeypatch.delenv(memory.OVERRIDE_ENV, raising=False)
    frames = np.zeros((4, 720, 1280, 3), dtype=np.uint8)
    own = 2 * memory.GB
    free = memory.need_bytes(frames.shape, 518) - own + memory._KEEP_FREE
    monkeypatch.setattr(memory, "available_bytes", lambda: free)

    class Reached(Exception):
        pass

    def reached(*_a, **_k):
        raise Reached

    monkeypatch.setattr(engine, "load_model", reached)
    monkeypatch.setattr(memory, "own_bytes", lambda: own)
    with pytest.raises(Reached):
        engine.infer(frames, 24)

    monkeypatch.setattr(memory, "own_bytes", lambda: own - 1)
    with pytest.raises(engine.NotEnoughMemory):
        engine.infer(frames, 24)


class _FailingModel:
    def __init__(self, message: str):
        self.message = message

    def infer_video_depth(self, *_a, **_k):
        raise RuntimeError(self.message)


def _run_windows_with(monkeypatch, model):
    @contextlib.contextmanager
    def no_hook(*_a):
        yield

    monkeypatch.setattr(engine, "_window_hook", no_hook)
    return engine._run_windows(model, np.zeros((1, 2, 2, 3), np.uint8), 24.0, 518, "cpu", True, None)


def test_hitting_the_limit_mid_pass_reads_as_plain_words(monkeypatch):
    with pytest.raises(engine.NotEnoughMemory, match=memory.OVERRIDE_ENV) as caught:
        _run_windows_with(monkeypatch, _FailingModel("MPS backend out of memory (MPS allocated: 4 GiB)"))
    assert "MPS backend" not in str(caught.value)


def test_any_other_runtime_error_is_not_dressed_up_as_memory(monkeypatch):
    with pytest.raises(RuntimeError, match="shape mismatch") as caught:
        _run_windows_with(monkeypatch, _FailingModel("shape mismatch"))
    assert not isinstance(caught.value, engine.NotEnoughMemory)


def test_both_failures_share_the_family_the_callers_catch():
    assert issubclass(engine.NonFiniteDepth, engine.DepthFailure)
    assert issubclass(engine.NotEnoughMemory, engine.DepthFailure)


# ------------------------------------------------------------------ real model (slow)

@pytest.mark.slow
def test_grouping_changes_no_byte_of_the_real_models_depth(clip_tiny):
    """The promise behind the fix: same picture out, less memory on the way."""
    info = probe(clip_tiny)
    frames = read_frames(info, fps=24, size=(140, 140))
    model, device = engine.load_model()

    grouped = engine._run_windows(model, frames, 24.0, 140, device, True, None)[0]

    wrapper = model.pretrained.__dict__.pop("get_intermediate_layers")
    try:
        whole = engine._run_windows(model, frames, 24.0, 140, device, True, None)[0]
    finally:
        model.pretrained.get_intermediate_layers = wrapper

    assert np.array_equal(grouped, whole)


@pytest.mark.slow
def test_infer_real_model(clip_tiny):
    info = probe(clip_tiny)
    frames = read_frames(info, fps=24, size=(140, 140))

    windows: list[tuple[int, int]] = []
    result = engine.infer(frames, 24.0, on_window=lambda d, t: windows.append((d, t)))

    assert result.frames.shape == (frames.shape[0], 140, 140)
    assert result.frames.dtype == np.uint8
    assert result.fps == 24.0
    assert result.count == frames.shape[0]
    assert windows and windows[-1][0] == windows[-1][1]  # progress reached 100%

    # A real depth map is not a constant plane.
    assert result.frames.std() > 5


def _flicker(result) -> tuple[float, float]:
    """(largest frame-to-frame jump, overall spread) of per-frame mean brightness."""
    means = result.frames.reshape(result.count, -1).mean(axis=1)
    return float(np.abs(np.diff(means)).max()), float(means.std())


@pytest.mark.slow
def test_no_flicker_on_a_static_scene(clip_static):
    """Flicker regression guard.

    Every input frame is byte-identical (lossless fixture), so the correct depth map is
    identical too and any high-frequency output variation is flicker. This is what per-frame
    depth models fail and what VDA's windowed alignment plus whole-clip normalization exist
    to prevent.

    We assert on frame-to-frame JUMP, not on overall spread, because they measure different
    things. Even on byte-identical input the spread is ~3.3: windowed inference re-derives
    scale/shift per window, so the clip drifts *slowly*. That drift is smooth and by design,
    and is harmless for motion transfer. Flicker is the high-frequency component, and that
    is what jump captures. Real footage confirms the two are independent — CLIP C shows
    higher spread (3.7) with lower jump (0.65) than any synthetic fixture here.

    Calibrated then frozen. Given byte-identical input and a fixed model, these numbers are
    reproducible run to run; the headroom is for future torch/MPS kernel changes.

        this fixture (lossless, structured)   jump 0.96   spread 3.35
        same frame at crf 18 (noisy input)    jump 1.64
        flat gradient (ill-conditioned)       jump 3.23
        real footage (CLIP C, 720p)           jump 0.65   spread 3.66
    """
    info = probe(clip_static)
    frames = read_frames(info, fps=24, size=(140, 140))
    jump, spread = _flicker(engine.infer(frames, 24.0))

    assert jump < 2.0, f"frame-to-frame flicker {jump:.3f} (calibrated 0.96)"
    assert spread < 8.0, f"implausible drift on a static clip: {spread:.3f} (calibrated 3.35)"


@pytest.mark.slow
def test_depth_is_temporally_smooth_under_motion(clip_tiny):
    """Companion guard: with a subject crossing frame, depth legitimately drifts.

    A gradual ramp as the box moves is real signal; a sudden step is not. So only the jump
    is bounded here, deliberately looser than the static case.
    """
    info = probe(clip_tiny)
    frames = read_frames(info, fps=24, size=(140, 140))
    jump, _ = _flicker(engine.infer(frames, 24.0))

    assert jump < 4.0, f"largest frame-to-frame jump {jump:.2f}"


@pytest.mark.slow
def test_fp16_falls_back_to_fp32_on_mps(clip_tiny):
    """The documented MPS defect: fp16 returns NaN, so infer() must recover, not export it.

    If a future torch fixes MPS fp16, this test still passes — it only asserts that the
    result is usable, and that any fallback that *did* happen was announced.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    info = probe(clip_tiny)
    frames = read_frames(info, fps=24, size=(140, 140))

    warnings: list[str] = []
    result = engine.infer(frames, 24.0, precision="fp16", on_warn=warnings.append)

    assert np.isfinite(result.frames).all()
    assert result.frames.std() > 5, "fell back but produced a flat image"
    if warnings:
        assert result.dtype == "fp32"
        assert "non-finite" in warnings[0]


@pytest.mark.slow
def test_engine_does_not_leak_between_conversions(clip_tiny):
    """Back-to-back conversions in a long-lived UI session must not creep toward OOM."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    info = probe(clip_tiny)
    frames = read_frames(info, fps=24, size=(140, 140))

    engine.infer(frames, 24.0)  # warm the model and allocator
    baseline = torch.mps.current_allocated_memory()
    for _ in range(3):
        engine.infer(frames, 24.0)
    growth = torch.mps.current_allocated_memory() - baseline

    assert growth < 200_000_000, f"MPS allocation grew {growth / 1e6:.0f} MB over 3 runs"
