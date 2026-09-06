from __future__ import annotations

import numpy as np
import pytest

from depthc.video_io import (
    Crop,
    VideoIOError,
    detect_crop,
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


def test_detect_crop_finds_letterbox(clip_letterboxed):
    crop = detect_crop(probe(clip_letterboxed))
    assert crop is not None
    assert crop.width == 320
    assert crop.height == pytest.approx(160, abs=4)
    assert crop.y == pytest.approx(40, abs=4)
    assert crop.x == 0


def test_detect_crop_returns_none_on_full_frame(clip_3s):
    assert detect_crop(probe(clip_3s)) is None


def test_detect_crop_offsets_and_extents_are_even(clip_letterboxed):
    """Odd offsets shift chroma in yuv420p; odd extents fail libx264 outright."""
    crop = detect_crop(probe(clip_letterboxed))
    assert crop.width % 2 == 0 and crop.height % 2 == 0
    assert crop.x % 2 == 0 and crop.y % 2 == 0


def test_detect_crop_stays_inside_the_frame(clip_letterboxed):
    info = probe(clip_letterboxed)
    crop = detect_crop(info)
    assert crop.x + crop.width <= info.width
    assert crop.y + crop.height <= info.height


def test_read_frames_applies_crop(clip_letterboxed):
    info = probe(clip_letterboxed)
    crop = detect_crop(info)
    frames = read_frames(info, crop=crop)
    assert frames.shape[1:3] == (crop.height, crop.width)
    # The matte is gone: no all-black row bands remain at the edges.
    assert frames[0, 0].mean() > 5 or frames[0, -1].mean() > 5


def test_target_size_uses_cropped_dimensions(clip_letterboxed):
    info = probe(clip_letterboxed)
    info.width, info.height = 1920, 1080
    crop = Crop(1920, 800, 0, 140)
    # 2.4:1 content scaled to a 1280 long edge is 1280x534, not 1280x720.
    width, height = target_size(info, "720p", crop=crop)
    assert width == 1280
    assert height == pytest.approx(534, abs=2)
    assert height % 2 == 0


def test_crop_parse_roundtrip():
    crop = Crop.parse("1920:800:0:140")
    assert (crop.width, crop.height, crop.x, crop.y) == (1920, 800, 0, 140)
    assert str(crop) == "1920:800:0:140"


@pytest.mark.parametrize("bad", ["1920:800:0", "a:b:c:d", "1920:800:0:-4", ""])
def test_crop_parse_rejects_junk(bad):
    with pytest.raises(VideoIOError):
        Crop.parse(bad)


def test_crop_removed_fraction(clip_3s):
    info = probe(clip_3s)
    info.width, info.height = 1920, 1080
    assert Crop(1920, 800, 0, 140).removed_fraction(info) == pytest.approx(0.259, abs=0.01)
    assert Crop(1920, 1080, 0, 0).removed_fraction(info) == pytest.approx(0.0)


def test_side_by_side_doubles_width(tmp_path):
    original = np.zeros((12, 64, 64, 3), dtype=np.uint8)
    depth = np.full((12, 64, 64), 128, dtype=np.uint8)
    out = side_by_side(original, depth, tmp_path / "sbs.mp4", fps=24)
    info = probe(out)
    assert (info.width, info.height) == (128, 64)


def test_encode_gray_is_tagged_full_range(tmp_path):
    """H.264's default for an UNTAGGED stream is limited range: a spec-conformant decoder
    expands 16-235 and clips everything outside — flattening the nearest and farthest depth
    bands. ffmpeg-world tools pass the levels through, which is why this never showed
    locally; the file itself must carry the truth for whatever decodes it provider-side."""
    import shutil
    import subprocess

    frames = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (10, 64, 1))
    out = encode_gray(frames, tmp_path / "tagged.mp4", fps=24)
    reported = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_range", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert reported == "pc"


def test_encode_gray_preserves_the_full_luma_range(tmp_path):
    """0 and 255 must survive the round trip — they are the nearest and farthest planes.

    Read the STORED Y plane with the range pinned on both ends (`scale=in_range=pc:
    out_range=pc` and `-color_range pc` on the rawvideo output), so no swscale range
    conversion can sit between the file and the assertion. On ubuntu's ffmpeg 6.1.1 (CI,
    2026-09-06) a plain `-pix_fmt gray` decode returned 17-234, and a plain `-pix_fmt
    yuv420p` decode returned 31-218 — the file's luma had been compressed once on encode
    and once more on decode, under a `pc` tag. A guard that reads through a transform
    measures the transform; this measures the file, on every ffmpeg.
    """
    import shutil
    import subprocess

    frames = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (10, 64, 1))
    out = encode_gray(frames, tmp_path / "levels.mp4", fps=24)
    raw = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(out),
         "-vf", "scale=in_range=pc:out_range=pc",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-color_range", "pc", "-"],
        capture_output=True, check=True,
    ).stdout
    n, h, w = frames.shape
    luma_bytes, chroma_bytes = h * w, 2 * (h // 2) * (w // 2)
    per_frame = np.frombuffer(raw, dtype=np.uint8).reshape(n, luma_bytes + chroma_bytes)
    luma = per_frame[:, :luma_bytes]
    assert luma.min() <= 4 and luma.max() >= 251, (luma.min(), luma.max())


def test_mux_audio_never_truncates_the_video(tmp_path):
    """`-shortest` is gone: a source whose audio track ends before the video does must not
    cut the depth video below its snapped, exactly-requestable length."""
    import shutil
    import subprocess

    from depthc.video_io import mux_audio

    silent = encode_gray(np.full((96, 64, 64), 128, dtype=np.uint8), tmp_path / "d.mp4", fps=24)

    # A 4 s video whose audio track lasts only 1 s.
    source = tmp_path / "short_audio.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-nostdin",
         "-f", "lavfi", "-i", "testsrc2=s=64x64:d=4:r=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         str(source)],
        capture_output=True, check=True,
    )

    out = mux_audio(silent, source, tmp_path / "muxed.mp4")
    info = probe(out)
    assert info.duration == pytest.approx(4.0, abs=0.1)
    assert info.has_audio


def test_probe_duration_is_the_video_streams_not_the_containers(tmp_path):
    """Audio outrunning video must not inflate the duration snap_trim plans against."""
    import shutil
    import subprocess

    path = tmp_path / "long_audio.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-nostdin",
         "-f", "lavfi", "-i", "testsrc2=s=64x64:d=2:r=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         str(path)],
        capture_output=True, check=True,
    )
    assert probe(path).duration == pytest.approx(2.0, abs=0.15)


def test_keep_fps_rejects_a_vfr_artifact_rate(clip_3s):
    """avg_frame_rate 0/0 falls back to r_frame_rate, which can be a 1000/1 timebase
    artifact — `--fps keep` must refuse it before duration*1000 frames hit RAM."""
    info = probe(clip_3s)
    info.fps = 1000.0
    with pytest.raises(VideoIOError):
        resolve_fps(info, "keep")
