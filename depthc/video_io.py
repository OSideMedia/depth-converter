"""Video probing, decoding, and encoding. All I/O is ours; none is upstream's.

DECODER CHOICE (deviates from PLAN.md §2, deliberately). The plan specified an OpenCV
`cv2.VideoCapture` frame reader. We pipe raw frames from ffmpeg instead, because the three
transforms we need on the way in — trim, fps conform, and scale — are exactly what a filter
graph does deterministically in one pass, and because **ffmpeg auto-applies rotation
metadata while OpenCV silently ignores it**. Phone footage coming out upright was an
explicit Phase 1 requirement; getting it for free from the decoder beats hand-rolling a
rotation branch. OpenCV is still used for image analysis (still selection).

`decord` is deliberately absent (no reliable arm64 macOS wheels).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from depthc import memory

# Seedance reference-input pixel window (PLAN.md §0 constraint table).
MIN_PIXELS = 409_600
MAX_PIXELS = 2_086_876

RESOLUTION_PRESETS = {"480p": 854, "720p": 1280, "source": None}


class VideoIOError(RuntimeError):
    """Anything wrong with the input file or an ffmpeg invocation."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VideoIOError(f"{name} not found on PATH. Install ffmpeg.")
    return path


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, **kw)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise VideoIOError(f"{Path(cmd[0]).name} failed:\n  " + "\n  ".join(tail))
    return proc


@dataclass
class VideoInfo:
    """Display-oriented facts about a source file (rotation already applied)."""

    path: Path
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    size_bytes: int
    rotation: int = 0
    has_audio: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000

    def summary(self) -> str:
        rot = f", rotated {self.rotation}°" if self.rotation else ""
        return (
            f"{self.width}×{self.height} · {self.fps:g} fps · {self.duration:.2f}s · "
            f"{self.size_mb:.1f} MB · {self.codec}{rot}"
        )


def _parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def _stream_seconds(stream: dict) -> float:
    """The video stream's own duration, or 0.0 when the container doesn't record one.

    The format-level duration is the max over ALL streams, so a source whose audio outruns
    its video (screen recordings, some phone exports) would otherwise report seconds that
    contain no picture — and a snap that trusts it places the out-point in an audio-only
    tail, delivering a depth video shorter than the length it just promised.
    """
    if stream.get("duration"):
        try:
            return float(stream["duration"])
        except (TypeError, ValueError):
            pass
    tag = (stream.get("tags") or {}).get("DURATION")  # Matroska: HH:MM:SS.nnnnnnnnn
    if tag:
        try:
            hours, minutes, seconds = tag.split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except ValueError:
            pass
    return 0.0


def _rotation_of(stream: dict) -> int:
    """Rotation in degrees from either a display matrix or the legacy rotate tag."""
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return int(round(float(side["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag:
        try:
            return int(round(float(tag))) % 360
        except ValueError:
            pass
    return 0


def probe(path: str | Path) -> VideoInfo:
    """ffprobe a file. Raises VideoIOError if it has no decodable video stream."""
    # Absolute, so ffmpeg and ffprobe cannot read `-clip.mp4` as an option or `file:clip.mp4` as a
    # protocol. Every later decode reuses this path through `info.path`.
    path = Path(path).absolute()
    if not path.is_file():
        raise VideoIOError(f"No such file: {path}")

    try:
        proc = _run(
            [
                _tool("ffprobe"), "-v", "error",
                "-show_streams", "-show_format", "-of", "json", str(path),
            ]
        )
    except VideoIOError as e:
        # Don't leak raw ffprobe output at the user for the commonest mistake of all —
        # handing the tool something that simply isn't a video.
        if "Invalid data found" in str(e):
            raise VideoIOError(
                f"{path.name} is not a video file (ffprobe could not read it)."
            ) from e
        raise

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoIOError(f"{path.name} contains no video stream — not a usable clip.")

    rotation = _rotation_of(video)
    width, height = int(video["width"]), int(video["height"])
    # A 90/270° rotation means the *displayed* frame is the transpose of the coded one,
    # and ffmpeg's decoder hands us display orientation.
    if rotation in (90, 270):
        width, height = height, width

    duration = _stream_seconds(video) or float(data.get("format", {}).get("duration") or 0)
    fps = _parse_fps(video.get("avg_frame_rate") or "") or _parse_fps(
        video.get("r_frame_rate") or ""
    )
    if duration <= 0 or fps <= 0:
        raise VideoIOError(f"{path.name}: could not determine duration/fps — corrupt file?")

    return VideoInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        codec=video.get("codec_name", "?"),
        size_bytes=path.stat().st_size,
        rotation=rotation,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        raw=data,
    )


@dataclass(frozen=True)
class Crop:
    """A crop rectangle in source pixels, in ffmpeg's w:h:x:y order."""

    width: int
    height: int
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.width}:{self.height}:{self.x}:{self.y}"

    @classmethod
    def parse(cls, text: str) -> "Crop":
        try:
            w, h, x, y = (int(part) for part in text.split(":"))
        except ValueError as e:
            raise VideoIOError(f"--crop expects W:H:X:Y (got {text!r})") from e
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise VideoIOError(f"--crop values must be positive (got {text!r})")
        return cls(w, h, x, y)

    def removed_fraction(self, info: "VideoInfo") -> float:
        return 1 - (self.width * self.height) / (info.width * info.height)


MATTE_LUMA_LIMIT = 18  # 0-255; compressed mattes are near-black, rarely exactly 0
MATTE_SAMPLE_FRAMES = 9
MIN_CROP_FRACTION = 0.02


def detect_crop(info: VideoInfo, trim: tuple[float, float] | None = None) -> Crop | None:
    """Find matte (letterbox/pillarbox) bars, or None if the frame is already full.

    Why this matters more than it sounds: a depth model reads a flat black bar as a
    *surface*, and specifically as the nearest one in the scene. Measured on a 2.39:1 clip
    matted into 16:9, the bars occupied 26% of the frame and came back at 160/255 while the
    actual content averaged 55/255 — a bright slab across the top and bottom of the motion
    channel, brighter than the subject it is meant to be following.

    Deliberately NOT ffmpeg's `cropdetect`. On a test clip with a legitimately dark right
    edge, cropdetect removed 20 px of real picture along with the matte — it thresholds on
    an average and cannot tell a dark *image* from a bar. Cropping away real content is the
    one failure mode that must not happen, since it silently changes the motion the ref is
    supposed to carry.

    Instead: sample frames across the clip and take the per-row and per-column MAXIMUM over
    every sampled pixel. A row is matte only if it is near-black at every pixel in every
    sampled frame. A single bright pixel anywhere, at any point in the clip, disqualifies
    the row — so real content is kept unless it is genuinely black throughout, in which case
    removing it costs nothing.
    """
    duration = (trim[1] - trim[0]) if trim else info.duration
    origin = trim[0] if trim else 0.0
    end = origin + duration
    # The sample COUNT is fixed, so the rate falls with duration. An earlier 0.5 fps floor
    # made the count grow with duration instead — a 5-minute 4K inbox clip decoded ~150
    # full-resolution frames into RAM (multiple GB) before any trim existed.
    sample_fps = MATTE_SAMPLE_FRAMES / duration if duration > 0 else 1.0

    frames = read_frames(info, trim=(origin, end), fps=sample_fps)
    if frames.shape[0] == 0:
        return None

    luma = frames.max(axis=3)  # brightest channel per pixel; a cheap, strict luma proxy
    row_max = luma.max(axis=(0, 2))  # over frames and columns -> one value per row
    col_max = luma.max(axis=(0, 1))  # over frames and rows    -> one value per column

    def _leading(values: np.ndarray) -> int:
        dark = values <= MATTE_LUMA_LIMIT
        count = 0
        for is_dark in dark:
            if not is_dark:
                break
            count += 1
        return count

    top = _leading(row_max)
    bottom = _leading(row_max[::-1])
    left = _leading(col_max)
    right = _leading(col_max[::-1])

    height_px, width_px = luma.shape[1], luma.shape[2]
    if top + bottom >= height_px or left + right >= width_px:
        return None  # entirely black clip; nothing to crop toward

    # Even offsets and extents: odd offsets shift chroma in yuv420p, odd extents fail x264.
    top += top % 2
    left += left % 2
    width = width_px - left - right
    height = height_px - top - bottom
    width -= width % 2
    height -= height % 2

    if width <= 0 or height <= 0:
        return None

    crop = Crop(width, height, left, top)
    # Ignore trivial detections: a couple of stray dark rows are not a matte, and cropping
    # for them would change framing for no benefit.
    if crop.removed_fraction(info) < MIN_CROP_FRACTION:
        return None
    return crop


def target_size(
    info: VideoInfo, resolution: str, crop: Crop | None = None
) -> tuple[int, int]:
    """Scale to a preset by long edge, preserving aspect ratio, forcing even dims.

    Sized against the *cropped* frame when a crop applies, so a matted clip yields a depth
    video at the aspect ratio of its real image rather than of its container.
    """
    if resolution not in RESOLUTION_PRESETS:
        raise VideoIOError(
            f"Unknown resolution {resolution!r}; choose from {list(RESOLUTION_PRESETS)}"
        )
    long_edge = RESOLUTION_PRESETS[resolution]
    w, h = (crop.width, crop.height) if crop else (info.width, info.height)
    if long_edge is not None and max(w, h) > long_edge:
        scale = long_edge / max(w, h)
        w, h = round(w * scale), round(h * scale)
    # libx264 + yuv420p requires even dimensions.
    return max(2, w - w % 2), max(2, h - h % 2)


# VFR sources can report a timebase artifact instead of a real rate (avg_frame_rate 0/0
# falls back to r_frame_rate, which can be e.g. 1000/1). `--fps keep` would then decode
# duration×1000 duplicated raw frames into RAM before any range check runs.
MAX_KEEP_FPS = 240.0


def resolve_fps(info: VideoInfo, mode: str) -> float:
    """`conform-24` matches Seedance's own output cadence; `keep` preserves the source."""
    if mode == "conform-24":
        return 24.0
    if mode == "keep":
        if info.fps > MAX_KEEP_FPS:
            raise VideoIOError(
                f"{info.path.name} reports {info.fps:g} fps — likely a variable-frame-rate "
                f"artifact, not a real rate. Use --fps conform-24 (or an explicit number)."
            )
        return info.fps
    try:
        value = float(mode)
    except ValueError as e:
        raise VideoIOError(f"--fps must be 'conform-24', 'keep', or a number (got {mode!r})") from e
    if value <= 0:
        raise VideoIOError("--fps must be positive")
    return value


def read_frames(
    info: VideoInfo,
    trim: tuple[float, float] | None = None,
    fps: float | None = None,
    size: tuple[int, int] | None = None,
    crop: Crop | None = None,
) -> np.ndarray:
    """Decode to an [N, H, W, 3] uint8 RGB array, trimmed/cropped/conformed/scaled in one pass."""
    source_size = (crop.width, crop.height) if crop else (info.width, info.height)
    width, height = size or source_size
    cmd = [_tool("ffmpeg"), "-v", "error", "-nostdin"]

    duration = info.duration
    if trim:
        start, end = trim
        start = max(0.0, start)
        end = min(info.duration, end)
        if end - start <= 0:
            raise VideoIOError(f"Empty trim range: {start:.2f}s → {end:.2f}s")
        duration = end - start
        # -ss before -i is both fast and frame-accurate in modern ffmpeg.
        cmd += ["-ss", f"{start:.6f}"]

    refusal = memory.decode_refusal(int(duration * (fps or info.fps)) * width * height * 3)
    if refusal:
        raise VideoIOError(refusal)

    cmd += ["-i", str(info.path)]
    if trim:
        cmd += ["-t", f"{duration:.6f}"]

    filters = []
    if fps:
        filters.append(f"fps={fps:g}")
    if crop:
        # Crop before scale, so the scale target describes the delivered picture.
        filters.append(f"crop={crop}")
    if (width, height) != source_size:
        filters.append(f"scale={width}:{height}:flags=lanczos")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if not fps:
        # No conform asked for: every decoded frame, once. rawvideo's default output sync is
        # CFR at the nominal rate, which drops bunched frames of a VFR source — the last
        # frame of a take among them (Codex review 2026-10-26).
        cmd += ["-fps_mode", "passthrough"]

    cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    frame_bytes = width * height * 3
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise VideoIOError("ffmpeg decode failed:\n  " + "\n  ".join(tail))
    if not proc.stdout or len(proc.stdout) < frame_bytes:
        raise VideoIOError(f"Decoded no complete frames from {info.path.name}")

    count = len(proc.stdout) // frame_bytes
    return np.frombuffer(proc.stdout[: count * frame_bytes], dtype=np.uint8).reshape(
        count, height, width, 3
    )


def _yuv420p_planes(frames: np.ndarray) -> bytes:
    """[N, H, W] uint8 luma → planar yuv420p bytes with neutral (128) chroma, per frame.

    Byte-exact: what libx264 receives as Y is the array, untouched by any scaler.
    """
    n, height, width = frames.shape
    chroma = np.full((n, (height // 2) * (width // 2)), 128, dtype=np.uint8)
    luma = np.ascontiguousarray(frames).reshape(n, height * width)
    return np.concatenate([luma, chroma, chroma], axis=1).tobytes()


def encode_gray(
    frames: np.ndarray,
    out_path: str | Path,
    fps: float,
    crf: int = 17,
    preset: str = "medium",
) -> Path:
    """Encode [N, H, W] uint8 grayscale to the fixed depth.mp4 spec.

    libx264, yuv420p, +faststart. The yuv420p planes are built HERE, in numpy — Y is the
    depth, U and V are flat 128 (neutral chroma) — and handed to ffmpeg as rawvideo
    `yuv420p`, so no swscale conversion sits between the array and libx264.

    [2026-09-06] It used to feed `gray` and let ffmpeg convert to yuv420p. On ffmpeg 6.1.1
    (ubuntu's, the CI runner) that conversion COMPRESSED the luma to 16-235 while the
    container still said full range — measured on the stored Y plane, 0-255 in → 16-235
    stored — and ffmpeg 9.0.1 (this Mac) did not. A version-dependent scaler default is not
    a place to keep the one invariant this file exists for, so the conversion is gone.

    THE STREAM IS TAGGED FULL-RANGE, EXPLICITLY. The luma here is 0-255 (depth), but H.264's
    spec default for an untagged stream is LIMITED range: a spec-conformant decoder on the
    consumer's side expands 16-235 → 0-255 and clips everything outside — flattening exactly
    the nearest and farthest depth bands, ~13% of the scale. ffmpeg-world tools pass the
    levels through untouched, which is why this never showed locally; what Seedance's
    decoder does is not observable from here, so the file must carry the truth itself:
    `-color_range pc` plus the h264_metadata VUI flag (verified: ffprobe reports
    color_range=pc and a 0-255 gradient survives a round trip on 9.0.1 and 6.1.1).
    """
    if frames.ndim != 3:
        raise VideoIOError(f"Expected [N, H, W] grayscale, got shape {frames.shape}")
    if frames.dtype != np.uint8:
        raise VideoIOError(f"Expected uint8 frames, got {frames.dtype}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, height, width = frames.shape
    if width % 2 or height % 2:
        raise VideoIOError(f"Odd dimensions {width}×{height}; libx264/yuv420p needs even.")

    cmd = [
        _tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-color_range", "pc",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-color_range", "pc",
        "-bsf:v", "h264_metadata=video_full_range_flag=1",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=_yuv420p_planes(frames), capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise VideoIOError("ffmpeg encode failed:\n  " + "\n  ".join(tail))
    return out_path


def mux_audio(video_path: Path, source: Path, out_path: Path, trim=None) -> Path:
    """Copy the source's audio onto a silent depth video.

    Seedance ignores reference-video audio entirely — this exists so the filmmaker can hear
    the beat while cutting. Rhythm sync is the @Audio slot's job, not this one.

    THE VIDEO STREAM IS AUTHORITATIVE. The audio is padded (`apad`) or cut (`-t`) to the
    video's exact length. `-shortest` is deliberately absent: a source whose audio track
    ends before the trim window does would truncate the depth VIDEO with it — silently
    breaking the snapped, exactly-requestable duration the whole kit is built around.
    """
    video_seconds = probe(video_path).duration
    cmd = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin", "-i", str(video_path)]
    if trim:
        cmd += ["-ss", f"{trim[0]:.6f}", "-t", f"{trim[1] - trim[0]:.6f}"]
    cmd += [
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-af", "apad", "-t", f"{video_seconds:.6f}",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def trim_copy(
    info: VideoInfo, out_path: Path, trim: tuple[float, float] | None = None
) -> Path:
    """Write the trimmed source segment beside the kit, at full source resolution.

    Deliberately NOT cropped and NOT conformed. The crop is a derived setting recorded in
    kit.json, so keeping the original framing here is what makes a kit re-convertible later
    at different settings — a pre-cropped copy would bake in one decision permanently.

    Re-encoded rather than stream-copied: `-c copy` can only cut on keyframes, which would
    silently shift the in-point away from the motion the kit is built around.
    """
    cmd = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin"]
    if trim:
        cmd += ["-ss", f"{trim[0]:.6f}"]
    cmd += ["-i", str(info.path)]
    if trim:
        cmd += ["-t", f"{trim[1] - trim[0]:.6f}"]
    cmd += [
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if info.has_audio else ["-an"]
    cmd += [str(out_path)]
    _run(cmd)
    return out_path


def side_by_side(
    original: np.ndarray, depth: np.ndarray, out_path: Path, fps: float
) -> Path:
    """Render an original | depth comparison MP4 for eyeballing (and socials)."""
    if original.shape[0] != depth.shape[0]:
        n = min(original.shape[0], depth.shape[0])
        original, depth = original[:n], depth[:n]
    rgb_depth = np.repeat(depth[..., None], 3, axis=-1)
    stacked = np.concatenate([original, rgb_depth], axis=2)

    n, height, width = stacked.shape[:3]
    width -= width % 2
    height -= height % 2
    stacked = stacked[:, :height, :width]

    cmd = [
        _tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=stacked.tobytes(), capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise VideoIOError("ffmpeg preview encode failed:\n  " + "\n  ".join(tail))
    return out_path
