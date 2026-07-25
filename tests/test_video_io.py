from __future__ import annotations

import numpy as np
import pytest

from depthc.video_io import (
    VideoIOError,
    encode_gray,
    probe,
    read_frames,
    resolve_fps,
    side_by_side,
    target_size,
)


def test_probe_reports_basics(clip_3s):
    info = probe(clip_3s)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(24, abs=0.1)
    assert info.duration == pytest.approx(3.0, abs=0.15)
    assert info.codec == "h264"
    assert info.size_bytes > 0


def test_probe_rejects_non_video(not_a_video):
    with pytest.raises(VideoIOError):
        probe(not_a_video)


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(VideoIOError, match="No such file"):
        probe(tmp_path / "ghost.mp4")


def test_target_size_preserves_aspect_and_evenness(clip_3s):
    info = probe(clip_3s)
    info.width, info.height = 1920, 1080
    assert target_size(info, "720p") == (1280, 720)
    assert target_size(info, "480p") == (854, 480)
    assert target_size(info, "source") == (1920, 1080)


def test_target_size_never_upscales(clip_3s):
    info = probe(clip_3s)  # 320x240, below every preset
    assert target_size(info, "720p") == (320, 240)


def test_target_size_forces_even_dimensions(clip_3s):
    info = probe(clip_3s)
    info.width, info.height = 1999, 1001  # odd, and odd after scaling
    width, height = target_size(info, "720p")
    assert width % 2 == 0 and height % 2 == 0


def test_target_size_rejects_unknown_preset(clip_3s):
    with pytest.raises(VideoIOError):
        target_size(probe(clip_3s), "4k")


def test_resolve_fps_modes(clip_60fps):
    info = probe(clip_60fps)
    assert resolve_fps(info, "conform-24") == 24.0
    assert resolve_fps(info, "keep") == pytest.approx(60, abs=0.1)
    assert resolve_fps(info, "30") == 30.0
    with pytest.raises(VideoIOError):
        resolve_fps(info, "banana")


def test_read_frames_shape_and_dtype(clip_3s):
    info = probe(clip_3s)
    frames = read_frames(info, fps=24, size=(320, 240))
    assert frames.dtype == np.uint8
    assert frames.shape[1:] == (240, 320, 3)
    assert frames.shape[0] == pytest.approx(72, abs=3)  # 3s @ 24fps


def test_read_frames_honors_trim(clip_3s):
    info = probe(clip_3s)
    frames = read_frames(info, trim=(1.0, 2.0), fps=24, size=(320, 240))
    assert frames.shape[0] == pytest.approx(24, abs=3)


def test_read_frames_conforms_60_to_24(clip_60fps):
    info = probe(clip_60fps)
    frames = read_frames(info, fps=24, size=(320, 240))
    assert frames.shape[0] == pytest.approx(48, abs=3)  # 2s @ 24fps, not 120


def test_read_frames_scales(clip_3s):
    info = probe(clip_3s)
    frames = read_frames(info, fps=24, size=(160, 120))
    assert frames.shape[1:3] == (120, 160)


def test_read_frames_rejects_empty_trim(clip_3s):
    info = probe(clip_3s)
    with pytest.raises(VideoIOError, match="Empty trim"):
        read_frames(info, trim=(2.0, 2.0))


def test_encode_gray_roundtrip(tmp_path):
    frames = np.linspace(0, 255, 24 * 64 * 64, dtype=np.float64)
    frames = frames.reshape(24, 64, 64).astype(np.uint8)
    out = encode_gray(frames, tmp_path / "depth.mp4", fps=24)

    info = probe(out)
    assert (info.width, info.height) == (64, 64)
    assert info.fps == pytest.approx(24, abs=0.1)
    assert out.stat().st_size > 0


def test_encode_gray_output_is_neutral_gray(tmp_path):
    """yuv420p from a gray source must stay colorless: R == G == B on decode."""
    frames = np.full((24, 64, 64), 200, dtype=np.uint8)
    out = encode_gray(frames, tmp_path / "flat.mp4", fps=24)
    decoded = read_frames(probe(out))
    r, g, b = decoded[..., 0], decoded[..., 1], decoded[..., 2]
    assert np.abs(r.astype(int) - g.astype(int)).max() <= 2
    assert np.abs(g.astype(int) - b.astype(int)).max() <= 2


def test_encode_gray_rejects_odd_dimensions(tmp_path):
    frames = np.zeros((10, 65, 65), dtype=np.uint8)
    with pytest.raises(VideoIOError, match="Odd dimensions"):
        encode_gray(frames, tmp_path / "odd.mp4", fps=24)


def test_encode_gray_rejects_wrong_shape(tmp_path):
    with pytest.raises(VideoIOError, match=r"\[N, H, W\]"):
        encode_gray(np.zeros((10, 64, 64, 3), dtype=np.uint8), tmp_path / "x.mp4", fps=24)


def test_encode_gray_rejects_wrong_dtype(tmp_path):
    with pytest.raises(VideoIOError, match="uint8"):
        encode_gray(np.zeros((10, 64, 64), dtype=np.float32), tmp_path / "x.mp4", fps=24)


def test_side_by_side_doubles_width(tmp_path):
    original = np.zeros((12, 64, 64, 3), dtype=np.uint8)
    depth = np.full((12, 64, 64), 128, dtype=np.uint8)
    out = side_by_side(original, depth, tmp_path / "sbs.mp4", fps=24)
    info = probe(out)
    assert (info.width, info.height) == (128, 64)
