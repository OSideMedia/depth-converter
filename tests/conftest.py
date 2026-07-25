"""Synthetic fixtures.

Tests use ffmpeg-generated clips rather than the real footage in `TEST CLIPS/` so they are
deterministic and runnable on any machine. The real clips are the manual gate checks.

DETERMINISM: every `gradients` source below pins `seed`. Its default is -1, i.e. random,
which silently made each generated clip a *different* image — and therefore made the
engine's calibrated flicker thresholds swing run to run for reasons that had nothing to do
with the engine. With the seed pinned, repeated runs are bit-identical.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        pytest.skip("ffmpeg not on PATH")
    return path


def _make_clip(path: Path, seconds: float, fps: int, width: int, height: int) -> Path:
    """A moving white box over a static gradient: a subject in motion, fixed background.

    The box is sized as a FRACTION of the frame, not in absolute pixels. A fixed 120 px box
    is a quarter of a 320-wide frame but swallows a 140-wide one, and a subject covering
    most of the frame makes mean-brightness swing wildly for entirely legitimate reasons —
    which would drown out the flicker signal the engine tests are trying to measure.
    """
    box_w = max(8, round(width * 0.22))
    box_h = max(8, round(height * 0.22))
    box = (
        f"drawbox=x='(iw-{box_w})*t/{seconds}':y='ih/2-{box_h // 2}'"
        f":w={box_w}:h={box_h}:color=white@1.0:t=fill"
    )
    subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-y", "-nostdin",
            "-f", "lavfi",
            "-i", f"gradients=s={width}x{height}:c0=black:c1=gray:d={seconds}:r={fps}:seed=42",
            "-vf", box,
            "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-t", str(seconds),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def clip_3s(tmp_path_factory) -> Path:
    """3 s / 24 fps / 320×240 — the general-purpose fixture."""
    path = tmp_path_factory.mktemp("clips") / "moving_box.mp4"
    return _make_clip(path, seconds=3.0, fps=24, width=320, height=240)


@pytest.fixture(scope="session")
def clip_60fps(tmp_path_factory) -> Path:
    """2 s / 60 fps — exercises the fps conform path."""
    path = tmp_path_factory.mktemp("clips") / "fast.mp4"
    return _make_clip(path, seconds=2.0, fps=60, width=320, height=240)


@pytest.fixture(scope="session")
def clip_tiny(tmp_path_factory) -> Path:
    """1.5 s / 24 fps / 140×140 — small enough for a real-model test to stay quick."""
    path = tmp_path_factory.mktemp("clips") / "tiny.mp4"
    return _make_clip(path, seconds=1.5, fps=24, width=140, height=140)


@pytest.fixture(scope="session")
def clip_static(tmp_path_factory) -> Path:
    """1.5 s of one unchanging, *structured* frame — the pure flicker probe.

    Nothing moves, so the true depth map is identical in every frame and any variation in
    the output is flicker by definition.

    Built by looping a single `testsrc2` frame and encoding LOSSLESSLY (`-qp 0`), for two
    reasons. First, lossless is what actually makes the decoded frames byte-identical —
    verified; at crf 18 the same looped PNG still decodes differently frame to frame, and
    that compression noise alone inflated the measured flicker from 0.96 to 1.64. Second,
    `testsrc2` has edges, shapes and texture, so the depth estimate is *well-conditioned* —
    a flat gradient gives a depth model nothing to lock onto and its output wanders for
    reasons that are not the engine's fault (measured 3.2 on a gradient vs 0.96 here).
    """
    tmp = tmp_path_factory.mktemp("clips")
    still = tmp / "still.png"
    subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-y", "-nostdin",
            "-f", "lavfi", "-i", "testsrc2=s=140x140:d=1:r=1",
            "-frames:v", "1", str(still),
        ],
        check=True,
        capture_output=True,
    )

    path = tmp / "static.mp4"
    subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-y", "-nostdin",
            "-loop", "1", "-i", str(still),
            "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-r", "24", "-t", "1.5", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def clip_long(tmp_path_factory) -> Path:
    """20 s — over Seedance's 15 s reference ceiling.

    Tiny and low-rate on purpose: it exists to be *rejected* before any inference happens,
    so encode cost is all it ever costs.
    """
    path = tmp_path_factory.mktemp("clips") / "long.mp4"
    return _make_clip(path, seconds=20.0, fps=6, width=96, height=96)


@pytest.fixture(scope="session")
def clip_letterboxed(tmp_path_factory) -> Path:
    """320×240 with a 40 px black matte top and bottom — content is 320×160.

    Mirrors the real defect found on CLIP A: a depth model reads a flat matte as the
    nearest surface in the scene, so the bars have to come off before inference.
    """
    path = tmp_path_factory.mktemp("clips") / "matted.mp4"
    inner = _make_clip(path.with_name("inner.mp4"), seconds=2.0, fps=24, width=320, height=160)
    subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-y", "-nostdin", "-i", str(inner),
            "-vf", "pad=320:240:0:40:black",
            "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def not_a_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "notes.txt"
    path.write_text("this is emphatically not a video")
    return path
