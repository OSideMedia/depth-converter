"""Motion measurement — the part of a depth ref that actually decides whether it works.

Everything else the checker grades (duration, file size, resolution, fps) is mechanical and
cannot fail if the code is right. What decides whether Seedance follows a reference is how
much motion survives into the depth channel and where in the clip it sits. This module
measures that, and it exists because of a specific failure.

WHY DEPTH SPACE AND NOT RGB. Measured on the four real reference clips, 2026-07-25, with both
sides decoded at the same 24 fps so the ratio means something: CLIP A's trim window carried
RGB energy of 5.98 against CLIP B's 4.93 — MORE motion than the clip that worked — yet in
depth space A collapsed to 0.95 while B held 4.27. That is 16% retention versus 87%. A prone
crawl along the ground moves parallel to the camera plane, so the bodies barely change
distance and the depth channel flattens the motion almost entirely. **RGB motion is not depth
motion, and only the second kind reaches the model.** A's generation started late; B's did
not. Nothing in the old checker had an opinion about either clip.

(An earlier pass put those figures at 11% and 52% by comparing depth sampled at 24 fps against
source sampled at 12. Per-frame deltas scale with the frame interval; the ranking survived the
error, the numbers did not. Both sides are now measured on the same frames.)

UNITS. Every energy figure is the mean absolute difference between consecutive frames of the
8-bit depth video, measured on a fixed 160x90 analysis grid. So 1.0 means "the average pixel
changes by one grey level per frame". The grid is fixed rather than aspect-preserving so that
figures are comparable across clips of different shapes; the aspect distortion does not
affect a mean. A frozen frame pushed back through the same encoder measures 0.002, so the
thresholds below are signal, not codec noise.

Energy is per FRAME, so it scales with the frame interval — figures are only comparable at
the same fps. Everything a kit ships is conformed to 24 fps; the scout pass runs at 12 and
rescales, and says so.

THRESHOLDS are calibrated against those four clips rather than derived from theory:

    A prone crawl   0.99   weak    near-static after the first half second — the clip whose
                                   generation "started very late"
    B sword duel    4.42   strong  sustained and rising; the one that behaved
    C walk-in       1.76   fine    steady, unspectacular, works
    D slow dolly    0.54   weak    a camera move with a 1.5 s dead head

Two ideas here are cherry-picked from video-dataset curation pipelines (see
DOCS/PRIOR-ART.md): motion scoring is a BAND rather than a floor — too much frame-to-frame
change is jitter, cuts and whip-pans, not usable choreography — and a hard cut inside a
window is the worst possible competing signal, so it is detected and refused rather than
averaged into the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- the analysis grid --------------------------------------------------------------------
GRID_WIDTH = 160
GRID_HEIGHT = 90

# --- thresholds (calibrated on the four real clips; see module docstring) ------------------
STATIC_FLOOR = 0.30       # at or below this a stretch is as good as frozen
WEAK_ENERGY = 1.50        # below this the model has little to lock onto
STRONG_ENERGY = 3.00      # at or above this the signal is unambiguous
JITTER_ENERGY = 12.0      # above this it is shake/flicker, not choreography
DEAD_STRETCH_SECONDS = 0.75   # a dead run this long is a beat the model has to invent

# CUT DETECTION, measured rather than assumed. The first attempt flagged any frame gap 4x the
# clip's median, and it fired on all three clips that contain no cut at all: a sword impact or
# a stumble legitimately spikes several times above its own median. What separates a cut is
# not magnitude but EXTENT — a cut changes the whole frame at once, an action changes the part
# of it a body occupies. Measured on the four reference clips: the worst legitimate frame gap
# changed 24% of the pixels by more than 25 grey levels, while splicing two of the clips
# together changed 46%. The threshold sits between them.
#
# HONEST LIMIT: this runs in depth space, where two different rooms can share a structure, so
# a cut between similar geometry can slide under 0.35. It is a WARN aid, not a guarantee.
CUT_LEVEL = 25.0          # grey levels; "this pixel really changed"
CUT_FRACTION = 0.35       # share of the frame that has to change at once to call it a cut
SPIKE_MULTIPLE = 6.0      # fallback for a bare delta series with no pixels to look at
SPIKE_FLOOR = 8.0
LOW_RETENTION = 0.25      # depth energy / source energy below this = in-plane motion
CAMERA_DOMINANT = 0.60    # median-flow share above this = a camera-path ref, not a body one

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SPARK_DEAD = "·"


def analysis_gray(frames: np.ndarray) -> np.ndarray:
    """Downsample [N,H,W] grey or [N,H,W,3] RGB to the fixed [N,90,160] uint8 analysis grid.

    INTER_AREA is the correct filter for shrinking: it averages the pixels it discards, so a
    small fast-moving subject still contributes to the delta instead of falling between
    sample points the way nearest-neighbour would let it.
    """
    import cv2

    if frames.ndim == 4 and frames.shape[-1] == 3:
        source = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    elif frames.ndim == 3:
        source = list(frames)
    else:
        raise ValueError(f"Expected [N,H,W] or [N,H,W,3] frames, got {frames.shape}")

    return np.stack(
        [cv2.resize(f, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA) for f in source]
    )


def frame_deltas(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames, one value per frame gap."""
    return frame_stats(frames)[0]


def frame_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame gap: (mean absolute change, fraction of pixels that really changed).

    Both come from one pass over the same difference. The mean is the motion signal; the
    fraction is what tells a cut apart from a lunge — see CUT_FRACTION above.
    """
    grid = analysis_gray(frames).astype(np.float32)
    if grid.shape[0] < 2:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    difference = np.abs(np.diff(grid, axis=0))
    return (
        difference.mean(axis=(1, 2)),
        (difference > CUT_LEVEL).mean(axis=(1, 2)).astype(np.float32),
    )


def camera_share(frames: np.ndarray) -> float:
    """How much of the movement is the camera rather than the subject, 0.0-1.0.

    Farneback dense flow, then split each frame pair's field into the median vector (every
    pixel moving together — a pan, a dolly, a tracking move) and the residual around it (a
    subject moving inside the frame). The share is |median| / (|median| + mean residual).

    ONLY MOVING FRAMES COUNT. Frame gaps below the static floor are measured but excluded from
    the average, because flow on a still frame is estimator noise with no direction, and
    including it drags every share toward zero. CLIP D — a slow dolly with a dead first half —
    measured 3% "whole-frame" across the whole clip and reads correctly once the frozen half
    stops voting.

    HONEST LIMIT: a median *translation* is the only camera motion this separates cleanly. A
    zoom, orbit or roll moves pixels in opposing directions, so its median is near zero and it
    books as subject motion. Treat a high number as "definitely camera", a low one as "not a
    simple camera translation" rather than "definitely subject".
    """
    import cv2

    grid = analysis_gray(frames)
    if grid.shape[0] < 2:
        return 0.0

    deltas = np.abs(np.diff(grid.astype(np.float32), axis=0)).mean(axis=(1, 2))
    camera, subject = [], []
    for index, (previous, current) in enumerate(zip(grid[:-1], grid[1:])):
        if deltas[index] <= STATIC_FLOOR:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        median = np.median(flow.reshape(-1, 2), axis=0)
        camera.append(float(np.hypot(*median)))
        subject.append(float(np.hypot(*(flow - median).transpose(2, 0, 1)).mean()))

    if not camera:
        return 0.0
    total = float(np.mean(camera) + np.mean(subject))
    return round(float(np.mean(camera)) / total, 3) if total > 0 else 0.0


def sparkline(values, floor: float = STATIC_FLOOR, ceiling: float | None = None) -> str:
    """Render a series as blocks, with dead buckets shown as `·` so they are visible.

    A dead stretch is the finding most worth seeing at a glance, and it is exactly the thing
    a normalised bar chart hides: scaled against a weak clip's own maximum, near-nothing
    still draws a bar.
    """
    values = [float(v) for v in values]
    if not values:
        return ""
    top = ceiling if ceiling is not None else max(values)
    if top <= 0:
        return SPARK_DEAD * len(values)

    out = []
    for value in values:
        if value <= floor:
            out.append(SPARK_DEAD)
            continue
        level = int(value / top * len(SPARK_BLOCKS))
        out.append(SPARK_BLOCKS[min(level, len(SPARK_BLOCKS) - 1)])
    return "".join(out)


@dataclass
class DeadStretch:
    """The longest run of frames with no usable motion, and where in the clip it sits."""

    start: float
    seconds: float
    where: str  # "head" | "tail" | "middle"

    def as_dict(self) -> dict:
        return {
            "startSeconds": round(self.start, 2),
            "seconds": round(self.seconds, 2),
            "where": self.where,
        }


@dataclass
class MotionProfile:
    """Everything measurable about one clip's motion, in depth space."""

    fps: float
    deltas: np.ndarray = field(repr=False)
    bucket_seconds: float = 0.5
    camera: float = 0.0
    changed: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ scalars
    @property
    def energy(self) -> float:
        return round(float(self.deltas.mean()), 3) if self.deltas.size else 0.0

    @property
    def median(self) -> float:
        return round(float(np.median(self.deltas)), 3) if self.deltas.size else 0.0

    @property
    def peak(self) -> float:
        return round(float(self.deltas.max()), 3) if self.deltas.size else 0.0

    @property
    def seconds(self) -> float:
        return round((self.deltas.size + 1) / self.fps, 3) if self.fps else 0.0

    def window_energy(self, start: float, end: float) -> float:
        """Mean energy over a time slice, for head/tail questions."""
        if not self.deltas.size:
            return 0.0
        lo = max(0, int(round(start * self.fps)))
        hi = min(self.deltas.size, int(round(end * self.fps)))
        if hi <= lo:
            return 0.0
        return round(float(self.deltas[lo:hi].mean()), 3)

    @property
    def head(self) -> float:
        return self.window_energy(0.0, DEAD_STRETCH_SECONDS)

    @property
    def tail(self) -> float:
        return self.window_energy(max(0.0, self.seconds - DEAD_STRETCH_SECONDS), self.seconds)

    # ------------------------------------------------------------------ shapes
    @property
    def buckets(self) -> list[float]:
        step = max(1, int(round(self.bucket_seconds * self.fps)))
        return [
            round(float(self.deltas[i:i + step].mean()), 2)
            for i in range(0, self.deltas.size, step)
        ]

    @property
    def spark(self) -> str:
        return sparkline(self.buckets)

    @property
    def dead_stretch(self) -> DeadStretch | None:
        """Longest run below the static floor, if it lasts long enough to matter."""
        if not self.deltas.size:
            return None
        dead = self.deltas <= STATIC_FLOOR
        best_start, best_length, run_start = 0, 0, None
        for index, is_dead in enumerate([*dead, False]):
            if is_dead and run_start is None:
                run_start = index
            elif not is_dead and run_start is not None:
                if index - run_start > best_length:
                    best_start, best_length = run_start, index - run_start
                run_start = None
        if best_length / self.fps < DEAD_STRETCH_SECONDS:
            return None

        end = best_start + best_length
        where = "middle"
        if best_start <= max(1, int(0.1 * self.fps)):
            where = "head"
        elif end >= dead.size - max(1, int(0.1 * self.fps)):
            where = "tail"
        return DeadStretch(best_start / self.fps, best_length / self.fps, where)

    @property
    def spikes(self) -> list[float]:
        """Timestamps of frame gaps that look like a cut, not choreography.

        Uses the changed-pixel fraction when the pixels were available to measure it, which is
        every profile that came from `measure()`. A profile assembled from a bare delta series
        — a synthetic one, or a manifest read back from disk — falls back to a deliberately
        blunt magnitude rule, because a series alone cannot distinguish extent from intensity.
        """
        if self.deltas.size < 3:
            return []
        if self.changed is not None and self.changed.size == self.deltas.size:
            return [
                round(i / self.fps, 2)
                for i, fraction in enumerate(self.changed)
                if fraction > CUT_FRACTION
            ]
        limit = max(SPIKE_FLOOR, self.median * SPIKE_MULTIPLE)
        return [round(i / self.fps, 2) for i, d in enumerate(self.deltas) if d > limit]

    def as_dict(self) -> dict:
        dead = self.dead_stretch
        return {
            "energy": self.energy,
            "median": self.median,
            "peak": self.peak,
            "headEnergy": self.head,
            "tailEnergy": self.tail,
            "cameraShare": self.camera,
            "bucketSeconds": self.bucket_seconds,
            "buckets": self.buckets,
            "sparkline": self.spark,
            "deadStretch": dead.as_dict() if dead else None,
            "spikeSeconds": self.spikes,
            "grid": f"{GRID_WIDTH}x{GRID_HEIGHT}",
            "fps": round(self.fps, 3),
        }


def measure(frames: np.ndarray, fps: float, with_flow: bool = True) -> MotionProfile:
    """Profile one clip. `with_flow=False` skips the optical-flow pass on very long clips."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    deltas, changed = frame_stats(frames)
    return MotionProfile(
        fps=float(fps),
        deltas=deltas,
        changed=changed,
        camera=camera_share(frames) if with_flow else 0.0,
    )


def retention(depth: MotionProfile, source: MotionProfile) -> float | None:
    """Fraction of the source's frame-to-frame motion that survived into depth.

    The number that catches an A-shaped clip: motion parallel to the camera plane looks fine
    in RGB and nearly vanishes in depth. Only meaningful when both profiles were measured on
    the same frames at the same fps — which is exactly what a kit build has to hand.
    """
    if not source.energy:
        return None
    return round(depth.energy / source.energy, 3)


def manifest_block(depth: MotionProfile, source: MotionProfile | None = None) -> dict:
    """The `motion` object recorded in kit.json."""
    block = depth.as_dict()
    if source is not None:
        block["sourceEnergy"] = source.energy
        block["retention"] = retention(depth, source)
    return block


# ---------------------------------------------------------------------- window selection

@dataclass
class Window:
    """One candidate trim, scored."""

    start: float
    end: float
    energy: float
    head: float
    score: float
    dead: bool
    cut: bool
    sharpness: float = 1.0
    balance: float = 1.0

    @property
    def seconds(self) -> float:
        return round(self.end - self.start, 2)

    def note(self) -> str:
        notes = []
        if self.cut:
            notes.append("contains a cut")
        if self.dead:
            notes.append("has a dead stretch")
        if self.balance < 0.5:
            notes.append(
                "back-loaded" if self.head < self.energy else "front-loaded"
            )
        if self.sharpness < 0.7:
            notes.append(f"soft ({self.sharpness:.2f} of this clip's best sharpness)")
        return "; ".join(notes)


def score_windows(
    profile: MotionProfile,
    target_seconds: float,
    *,
    stride_seconds: float = 0.25,
    sharpness: np.ndarray | None = None,
) -> list[Window]:
    """Rank every target-length window in a clip. Best first.

    The score is the window's own energy, multiplied by three penalties that encode what went
    wrong on the four real clips:

      head factor    a window whose first 0.75 s is much weaker than its own average produces
                     a generation that sits still and then starts — the CLIP A complaint.
      balance        a lopsided window is a mean hiding a shape. Verified on CLIP A: the
                     top-scoring 5 s window turned out to be 2.5 s of empty forest followed by
                     the fall, which the head factor missed because swaying trees kept the
                     first 0.75 s off the floor. Comparing the two halves catches both a slow
                     build and a beat that dies out.
      cut penalty    a hard cut inside the ref is a second motion signal; heavily penalised.
      dead penalty   a dead run anywhere inside is a beat the model will have to invent.

    Sharpness (variance of the Laplacian per frame, relative to this clip's best) is reported
    but deliberately NOT scored. Blur is what talked us out of CLIP A's real motion in the
    first place: the fall was the signal, and it was rejected for being soft. A human can see
    "soft" in the note and decide; the scorer should not quietly prefer a still frame.

    THE LIMITATION THAT MATTERS, and it is not fixable here. This ranks HOW MUCH motion a
    window carries, and it has no idea how many separate things are moving — so it actively
    works against the tool's own "one dominant motion signal" law, because a second figure
    walking into frame is more energy, not less. Verified on CLIP C: the top-ranked window is
    exactly the stretch the original hand trim cut away because extra figures enter there.

    Counting subjects was tried and cut. Connected components on the frame difference splits a
    single body into limbs and edges: CLIP C's clean one-walker window measured 3-6 "movers"
    against 3-5 for the crowded one, i.e. no signal at all. Doing it properly needs person
    segmentation, which is a second model and a second licence.

    So: LOOK AT sheet.png BEFORE FIRING. That is not a nicety, it is the step that catches the
    one failure this scorer is structurally blind to.
    """
    if not profile.deltas.size or target_seconds <= 0:
        return []

    fps = profile.fps
    length = int(round(target_seconds * fps))
    if length >= profile.deltas.size:
        length = profile.deltas.size
    stride = max(1, int(round(stride_seconds * fps)))
    head_length = max(1, int(round(DEAD_STRETCH_SECONDS * fps)))
    # One cut mask for the whole clip, by the same rule `spikes` uses, so a window's verdict
    # and the profile's finding can never disagree.
    cut_frames = np.zeros(profile.deltas.size, dtype=bool)
    for second in profile.spikes:
        index = int(round(second * fps))
        if 0 <= index < cut_frames.size:
            cut_frames[index] = True

    windows: list[Window] = []
    for start in range(0, max(1, profile.deltas.size - length + 1), stride):
        slice_ = profile.deltas[start:start + length]
        if slice_.size < length:
            break
        energy = float(slice_.mean())
        head = float(slice_[:head_length].mean())
        cut = bool(cut_frames[start:start + length].any())

        dead_run, longest = 0, 0
        for delta in slice_:
            dead_run = dead_run + 1 if delta <= STATIC_FLOOR else 0
            longest = max(longest, dead_run)
        dead = longest / fps >= DEAD_STRETCH_SECONDS

        first, second = np.array_split(slice_, 2)
        halves = (float(first.mean()), float(second.mean()))
        balance = min(halves) / max(halves) if max(halves) > 0 else 1.0

        head_factor = 1.0 if energy <= 0 else min(1.0, max(0.35, head / (0.5 * energy)))
        score = (
            energy
            * head_factor
            * (0.5 + 0.5 * balance)
            * (0.25 if cut else 1.0)
            * (0.6 if dead else 1.0)
        )

        sharp = 1.0
        if sharpness is not None and sharpness.size:
            piece = sharpness[start:start + length + 1]
            top = float(sharpness.max()) or 1.0
            sharp = round(float(piece.mean()) / top, 2) if piece.size else 1.0

        windows.append(Window(
            start=round(start / fps, 2),
            end=round((start + length) / fps, 2),
            energy=round(energy, 3),
            head=round(head, 3),
            score=round(score, 3),
            dead=dead,
            cut=cut,
            sharpness=sharp,
            balance=round(balance, 2),
        ))

    windows.sort(key=lambda w: w.score, reverse=True)
    return windows


def spread(windows: list[Window], count: int = 3, overlap: float = 0.6) -> list[Window]:
    """Take the best `count` windows that are not mostly the same footage.

    Without this the top ten candidates are the winner shifted by a quarter second each time,
    which tells a human nothing they can choose between.
    """
    chosen: list[Window] = []
    for window in windows:
        span = window.end - window.start
        if any(
            max(0.0, min(window.end, other.end) - max(window.start, other.start)) / span > overlap
            for other in chosen
        ):
            continue
        chosen.append(window)
        if len(chosen) >= count:
            break
    return chosen


def sharpness_series(frames: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian per frame — high is crisp, low is motion-blurred."""
    import cv2

    grid = analysis_gray(frames)
    return np.array([cv2.Laplacian(f, cv2.CV_64F).var() for f in grid], dtype=np.float32)


# ---------------------------------------------------------------------- the scout pass

@dataclass
class ScoutResult:
    """A cheap whole-clip depth pass: the profile, the ranked windows, and what it cost."""

    profile: MotionProfile
    windows: list[Window]
    per_second: list[float]
    scout_fps: float
    elapsed: float
    target_seconds: float

    @property
    def best(self) -> Window | None:
        return self.windows[0] if self.windows else None


SCOUT_FPS = 12.0
SCOUT_LONG_EDGE = 384
SCOUT_INPUT_SIZE = 196   # multiple of 14, as DINOv2 patching requires


def scout_clip(
    path: str | Path,
    target_seconds: float,
    *,
    scout_fps: float = SCOUT_FPS,
    long_edge: int = SCOUT_LONG_EDGE,
    input_size: int = SCOUT_INPUT_SIZE,
    crop: bool = True,
    on_window=None,
) -> ScoutResult:
    """Run a low-resolution depth pass over a WHOLE clip to rank its candidate windows.

    Cheap on purpose: 12 fps at a 384 px long edge through VDA-Small at input_size 196, which
    is seconds rather than the 20-35 s a real 720p conversion takes. The point is only to
    rank windows *within one clip*.

    THE ENERGY FIGURES ARE ESTIMATES, and there are two reasons they will not match the kit's:
    depth is normalised over the whole clip here versus over the chosen window there, which
    compresses within-window contrast; and per-frame deltas scale with the frame interval, so
    the 12 fps deltas are halved to express them at the 24 fps a kit ships at. The kit's own
    gate is the authority. This picks where to point it.
    """
    import time

    from depthc import engine
    from depthc.video_io import detect_crop, probe, read_frames, target_size

    info = probe(path)
    rect = detect_crop(info) if crop else None
    size = target_size(info, "source", crop=rect)
    scale = long_edge / max(size)
    if scale < 1:
        size = (max(2, int(size[0] * scale) & ~1), max(2, int(size[1] * scale) & ~1))

    started = time.perf_counter()
    frames = read_frames(info, fps=scout_fps, size=size, crop=rect)
    result = engine.infer(
        frames, scout_fps, input_size=input_size, precision="fp32", on_window=on_window
    )
    elapsed = time.perf_counter() - started

    deltas, changed = frame_stats(result.frames)
    profile = MotionProfile(
        fps=scout_fps,
        deltas=deltas * (scout_fps / 24.0),
        changed=changed,
        bucket_seconds=1.0,
        camera=camera_share(result.frames),
    )
    windows = score_windows(
        profile, target_seconds, sharpness=sharpness_series(frames)
    )
    return ScoutResult(
        profile=profile,
        windows=spread(windows),
        per_second=profile.buckets,
        scout_fps=scout_fps,
        elapsed=elapsed,
        target_seconds=target_seconds,
    )


# ---------------------------------------------------------------------- ref variants

def contrast_stretch(frames: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Rescale the depth range between two percentiles, clipping the tails.

    Rationale: normalisation spends the 0-255 range on the single nearest and single furthest
    pixels in the clip, so one bright foreground object or one hole to infinity can leave the
    subject occupying a narrow band of greys. Clipping the outer 2% at each end gives the
    moving subject more of the range.

    EXPERIMENTAL, and honestly so: no published depth-conditioning pipeline does this (VACE's
    annotator uses plain per-frame min-max, no percentile clipping), so whether Seedance
    follows a stretched ref better is an open question. That is what --variants is for.
    Percentiles are taken over the WHOLE clip, never per frame — a per-frame stretch would
    reintroduce the brightness pumping global normalisation exists to prevent.
    """
    data = frames.astype(np.float32)
    lo, hi = np.percentile(data, [low, high])
    if hi - lo < 1e-6:
        return frames.copy()
    return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def invert(frames: np.ndarray) -> np.ndarray:
    """Far bright instead of near bright."""
    return (255 - frames).astype(np.uint8)


VARIANTS = {
    "contrast": (
        contrast_stretch,
        "percentile-stretched (2-98%) — more of the grey range spent on the subject",
    ),
    "invert": (
        invert,
        "polarity flipped — far bright, near dark",
    ),
}


# ---------------------------------------------------------------------- contact sheet

def contact_sheet(
    frames: np.ndarray,
    fps: float,
    out_path: str | Path,
    columns: int = 5,
    rows: int = 2,
    tile_width: int = 320,
) -> Path:
    """Tile evenly-spaced depth frames with timestamps into one PNG.

    The letterbox bug was caught by looking at a picture, not by a test, and the CLIP A
    diagnosis needed to see *when* the motion happened. A player cannot show either at a
    glance; a contact sheet can, and it costs nothing to write.
    """
    import cv2

    out_path = Path(out_path)
    count = min(columns * rows, frames.shape[0])
    if count == 0:
        raise ValueError("No frames to tile")
    indices = np.linspace(0, frames.shape[0] - 1, count).round().astype(int)

    height = max(2, round(tile_width * frames.shape[1] / frames.shape[2]))
    tiles = []
    for index in indices:
        tile = cv2.resize(frames[index], (tile_width, height), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR) if tile.ndim == 2 else tile
        label = f"{index / fps:.2f}s"
        cv2.putText(tile, label, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))

    grid = np.concatenate(
        [np.concatenate(tiles[r:r + columns], axis=1) for r in range(0, len(tiles), columns)],
        axis=0,
    )
    cv2.imwrite(str(out_path), grid)
    return out_path
