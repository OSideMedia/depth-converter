from __future__ import annotations

import numpy as np
import torch
import pytest

from depthc import engine
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


# ------------------------------------------------------------------ real model (slow)

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
