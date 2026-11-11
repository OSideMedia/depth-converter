"""Describe a lone clip's motion in the words a prompt uses.

`depthc compare` answers "did the output follow the REFERENCE clip's motion?" — it needs a
kit. A shot fired from text alone has no kit, and until now the only way to check whether
its motion matched the prompt was to look at frames and say so — a read, not a measurement.

This module turns one MotionProfile (v0.7.0: energy, regions, camera path) into a short
description in prompt vocabulary — "camera pans left then pushes in; subject enters
frame-left and travels to frame-right, arriving late; strong, sustained" — so the check
becomes: read the prompt, read the description, do they say the same thing?

EVERYTHING HERE IS UNCALIBRATED against a fire. The labels are the same measurements compare
reports, read as words; the camera words alone have a clay score
(DOCS/CAMERA-PATH-CALIBRATION.md). Nothing votes; there is no verdict. It measures WHAT MOVED and WHICH WAY,
in depth space (the channel Seedance actually reads a ref through), on the same analysis
grid as a kit — so a description of a generation and the profile of the kit that drove it
speak the same units. The camera words are the one exception: since 2026-11-11 they are read
from the picture the depth was made from (`motion.camera_path`), because the depth picture
invented moves on real footage. It cannot read intent, identity, mood, or whether the "subject" is
the one the prompt named — that stays a by-eye call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from depthc.motion import (
    REGION_COLS, REGION_ROWS, STATIC_FLOOR, STRONG_ENERGY, WEAK_ENERGY, MotionProfile,
    path_label, path_speed, path_tempo,
)

BUCKET_SECONDS = 0.5
COLUMN_NAMES = ("frame-left", "centre", "frame-right")
ROW_NAMES = ("top", "middle", "bottom")
COLUMN_SHORT = ("L", "C", "R")
ROW_SHORT = ("T", "M", "B")
# A bucket carries "no placement" below this energy — the placement track skips it rather
# than naming a random cell of noise (STATIC_FLOOR is the same floor the sparkline uses).
PLACEMENT_FLOOR = STATIC_FLOOR
# A travel is named only when the placement track actually moves this many columns.
TRAVEL_MIN_COLUMNS = 1

# --- framing: how much of the frame the nearest figure occupies (depth space ONLY —
# in RGB, bright is not near, so the CLI never passes frames in --rgb mode) -------------
# A pixel within this share of the frame's value range, measured from the bright (near)
# end, belongs to the near mask. VDA outputs disparity: near is bright.
FRAMING_NEAR_BAND = 0.35
# A frame whose full depth range is narrower than this many grey levels is flat — there
# is no near figure to size (an empty room, a wall, a matte card).
FRAMING_MIN_RELIEF = 24.0
# A largest-component smaller than this share of the frame is speckle, not a figure.
FRAMING_MIN_AREA = 0.01
# A standing figure is iso-depth over its height; a floor, table or water surface is a
# depth RAMP (near at the bottom, far at the top). Components whose in-component
# bottom-minus-top depth difference exceeds this share of the frame's range are surfaces
# and are not sized. Measured on the six 2026-08-23 validation clips: open water ramped
# +0.25 while every true figure sat between −0.06 and +0.08; the sign+grass foreground
# ramped +0.08. The cut sits in the middle of that gap.
FRAMING_MAX_RAMP = 0.15
# What depth alone CANNOT tell: an insert of a near surface and a vast landscape have the
# same relative-depth structure (depth has no absolute scale), so a frame-filling close
# surface that survives the gates reads as a wide. The render footer's by-eye caveat is
# the honest boundary; do not try to fix this with a cleverer mask.
# Height-share thresholds → the standard size ladder. The idea and the cut points are
# CozyClay's SIZE_TABLE (subject height / frame height, derived there from real 3D
# geometry). An in-frame silhouette saturates at 1.0 — a cropped figure's true size is
# unknowable from its silhouette — so everything tighter than "medium" collapses into
# one honest "tight" rung here instead of pretending to tell an MCU from an ECU.
# ponytail: heuristic ladder, validated by eye on reference clips, never calibrated
# against a fire; the near-mask can merge a foreground floor into the figure.
FRAMING_LADDER = [
    (0.97, "tight (medium close-up or closer)"),
    (0.80, "medium shot"),
    (0.52, "medium-wide shot"),
    (0.30, "wide shot"),
    (0.0, "extreme wide shot"),
]


@dataclass
class Segment:
    """One stretch of the clip where the label held."""

    start: float
    end: float
    label: str
    tempo: str = ""                              # "slowly" | "quickly" | "" (unremarkable)

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 2), "end": round(self.end, 2), "label": self.label,
            "tempo": self.tempo,
        }


@dataclass
class Framing:
    """The nearest figure's share of frame height, read as a shot-size word."""

    fraction: float          # median height share over frames that had a figure, 0..1
    label: str               # one rung of FRAMING_LADDER
    cropped: bool            # the figure touches top AND bottom in most sized frames
    trend: str | None        # "wide shot → medium shot" when the clip crosses a rung
    tightening: bool | None  # direction of the trend; None when there is no trend

    def as_dict(self) -> dict:
        return {
            "fraction": round(self.fraction, 3), "label": self.label,
            "cropped": self.cropped, "trend": self.trend, "tightening": self.tightening,
        }


def framing_label(fraction: float) -> str:
    for threshold, label in FRAMING_LADDER:
        if fraction >= threshold:
            return label
    return FRAMING_LADDER[-1][1]


def _near_figure(gray: np.ndarray) -> tuple[float, bool, tuple[int, int, int, int]] | None:
    """One analysis-grid depth frame → (height share, touches both edges, bbox), or None.

    Near mask (bright band of the frame's own range) → largest connected component →
    its bounding-box height. The frame's OWN range, not a fixed level: depth videos are
    normalised per clip, and a fixed threshold would read a far scene as empty. A
    component that is a depth ramp (see FRAMING_MAX_RAMP) is a surface, not a figure.
    """
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < FRAMING_MIN_RELIEF:
        return None
    mask = (gray >= hi - FRAMING_NEAR_BAND * (hi - lo)).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask)
    if count <= 1:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    if float(component.mean()) < FRAMING_MIN_AREA:
        return None
    rows = np.where(component.any(axis=1))[0]
    cols = np.where(component.any(axis=0))[0]
    quarter = max(1, len(rows) // 4)
    row_means = [float(gray[r][component[r]].mean()) for r in rows]
    ramp = (float(np.mean(row_means[-quarter:])) - float(np.mean(row_means[:quarter]))) / (hi - lo)
    if ramp > FRAMING_MAX_RAMP:
        return None
    share = float(rows[-1] - rows[0] + 1) / gray.shape[0]
    cropped = bool(rows[0] <= 1 and rows[-1] >= gray.shape[0] - 2)
    return share, cropped, (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))


def read_framing(
    gray_frames: np.ndarray,
    fps: float = 24.0,
    region_energy: list[list[float]] | None = None,
) -> Framing | None:
    """Size the nearest figure across a clip of analysis-grid DEPTH frames.

    Median over the frames that had a figure; the trend compares the first and last
    second's medians and is named only when they land on different ladder rungs.

    When `region_energy` (the profile's 3×3 grid) is given, the sized figure must
    overlap the motion hotspot cell in most frames — describe's subject IS the moving
    thing, and a static near sliver (a bystander's shoulder, a doorframe) sized while
    the actual subject dances in the midground is a confident lie.
    """
    sized = [_near_figure(frame) for frame in gray_frames]
    live = [s for s in sized if s is not None]
    if not live:
        return None

    if region_energy is not None:
        grid = np.array(region_energy, dtype=np.float64)
        if grid.size and float(grid.sum()) > 0:
            r, c = np.unravel_index(int(grid.argmax()), grid.shape)
            height, width = gray_frames.shape[1], gray_frames.shape[2]
            top, bottom = r * height / 3, (r + 1) * height / 3
            left, right = c * width / 3, (c + 1) * width / 3
            hits = sum(
                1 for _, _, (r0, r1, c0, c1) in live
                if r0 < bottom and r1 >= top and c0 < right and c1 >= left
            )
            if hits <= len(live) / 2:
                return None

    shares = np.array([s for s, _, _ in live], dtype=np.float64)
    median = float(np.median(shares))
    cropped = sum(1 for _, c, _ in live if c) > len(live) / 2
    label = FRAMING_LADDER[0][1] if cropped else framing_label(median)

    trend = None
    tightening = None
    second = max(1, int(round(fps)))
    head = [s for s in sized[:second] if s is not None]
    tail = [s for s in sized[-second:] if s is not None]
    if head and tail:
        first = framing_label(float(np.median([s for s, _, _ in head])))
        last = framing_label(float(np.median([s for s, _, _ in tail])))
        if first != last:
            trend = f"{first} → {last}"
            tightening = float(np.median([s for s, _, _ in tail])) > float(
                np.median([s for s, _, _ in head])
            )
    return Framing(median, label, cropped, trend, tightening)


@dataclass
class Description:
    seconds: float
    energy: float
    band: str                                    # weak | fine | strong | static
    spark: str
    dead: dict | None
    camera_share: float | None
    camera_overall: dict | None                  # {dx, dy, radial, label}
    camera_segments: list[Segment] = field(default_factory=list)
    placement: list[str] = field(default_factory=list)      # per bucket: "L"/"C"/"R"/"·"
    placement_rows: list[str] = field(default_factory=list) # per bucket: "T"/"M"/"B"/"·"
    travel: str | None = None                    # "frame-left → frame-right" or None
    hotspot: str | None = None                   # "centre-middle (38%)"
    region_share: list[list[float]] | None = None
    framing: Framing | None = None               # depth space only; None in --rgb
    sentences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "seconds": self.seconds,
            "energy": self.energy,
            "band": self.band,
            "sparkline": self.spark,
            "deadStretch": self.dead,
            "cameraShare": self.camera_share,
            "cameraOverall": self.camera_overall,
            "cameraSegments": [s.as_dict() for s in self.camera_segments],
            "placement": "".join(self.placement),
            "placementRows": "".join(self.placement_rows),
            "travel": self.travel,
            "hotspot": self.hotspot,
            "regionShare": self.region_share,
            "framing": self.framing.as_dict() if self.framing else None,
            "sentences": self.sentences,
            "note": "uncalibrated — a description of what moved and which way, not a verdict",
        }


def _band(energy: float) -> str:
    if energy <= STATIC_FLOOR:
        return "static"
    if energy < WEAK_ENERGY:
        return "weak"
    if energy >= STRONG_ENERGY:
        return "strong"
    return "fine"


def _segments(labels: list[str], bucket_seconds: float) -> list[Segment]:
    """Collapse a per-bucket label series into runs."""
    out: list[Segment] = []
    for index, label in enumerate(labels):
        start = index * bucket_seconds
        if out and out[-1].label == label:
            out[-1].end = start + bucket_seconds
        else:
            out.append(Segment(start, start + bucket_seconds, label))
    return out


def path_buckets(
    path: np.ndarray | None, fps: float, bucket_seconds: float = BUCKET_SECONDS,
) -> list[tuple[str, float, tuple[float, float, float]]]:
    """Per bucket: (label, speed, mean (dx, dy, radial) over the gaps that moved).

    The one place a path series becomes words. `camera_segments` collapses these into runs;
    the camera-path calibration (`depthc.calibrate`) scores them bucket by bucket against a
    Blender ground truth — so what the calibration measures is exactly what describe says.
    """
    if path is None or not path.size:
        return []
    step = max(1, int(round(bucket_seconds * fps)))
    n = path.shape[0] - step + 1
    if n <= 0:
        return []
    out: list[tuple[str, float, tuple[float, float, float]]] = []
    for i in range(0, n, step):
        chunk = path[i:i + step]
        moving = np.any(chunk != 0, axis=1)
        if not moving.any():
            out.append(("static", 0.0, (0.0, 0.0, 0.0)))
            continue
        dx, dy, radial = (float(v) for v in chunk[moving].mean(axis=0))
        out.append((path_label(dx, dy, radial), path_speed(dx, dy, radial), (dx, dy, radial)))
    return out


def camera_segments(profile: MotionProfile, bucket_seconds: float = BUCKET_SECONDS) -> list[Segment]:
    """The camera path named per bucket, then collapsed into runs ("pans left 0-1.5s")."""
    buckets = path_buckets(profile.path, profile.fps, bucket_seconds)
    if not buckets:
        return []
    labels = [label for label, _, _ in buckets]
    speeds = [speed for _, speed, _ in buckets]
    segments = _segments(labels, bucket_seconds)
    # Tempo is graded over the whole run, not per bucket: a per-bucket adverb would split
    # one "pans left" into fragments at every flicker across a band edge.
    for segment in segments:
        if segment.label == "static":
            continue
        lo = int(round(segment.start / bucket_seconds))
        hi = int(round(segment.end / bucket_seconds))
        segment.tempo = path_tempo(float(np.mean(speeds[lo:hi])))
    return segments


def placement_track(
    profile: MotionProfile, bucket_seconds: float = BUCKET_SECONDS
) -> tuple[list[str], list[str]]:
    """Per bucket: which COLUMN and which ROW carries the most energy ("·" = too still).

    Column of the peak cell, not the energy centroid: a centroid of two figures at either
    edge reads "centre", which is a lie about both. The peak names where the strongest
    thing is; the sentence built from it says so.
    """
    buckets = profile.region_buckets(bucket_seconds)
    if buckets is None or not buckets.shape[0]:
        return [], []
    cols, rows = [], []
    for cell in buckets:
        if float(cell.mean()) <= PLACEMENT_FLOOR:
            cols.append("·")
            rows.append("·")
            continue
        r, c = np.unravel_index(int(cell.argmax()), cell.shape)
        cols.append(COLUMN_SHORT[c])
        rows.append(ROW_SHORT[r])
    return cols, rows


def travel(placement: list[str]) -> str | None:
    """Name the net horizontal travel of the placement track, if there is one."""
    live = [p for p in placement if p != "·"]
    if len(live) < 2:
        return None
    first, last = COLUMN_SHORT.index(live[0]), COLUMN_SHORT.index(live[-1])
    if abs(last - first) < TRAVEL_MIN_COLUMNS:
        return None
    return f"{COLUMN_NAMES[first]} → {COLUMN_NAMES[last]}"


def hotspot(profile: MotionProfile) -> tuple[str | None, list[list[float]] | None]:
    """The cell that carried the most energy over the clip, with its share."""
    energy = profile.region_energy
    if energy is None:
        return None, None
    grid = np.array(energy, dtype=np.float64)
    total = float(grid.sum())
    if total <= 0:
        return None, [[0.0] * REGION_COLS for _ in range(REGION_ROWS)]
    share = grid / total
    r, c = np.unravel_index(int(share.argmax()), share.shape)
    name = f"{COLUMN_NAMES[c]}, {ROW_NAMES[r]} ({share[r, c] * 100:.0f}% of the motion)"
    return name, [[round(float(v), 3) for v in row] for row in share]


def describe(
    profile: MotionProfile,
    bucket_seconds: float = BUCKET_SECONDS,
    framing: Framing | None = None,
) -> Description:
    dead = profile.dead_stretch
    cols, rows = placement_track(profile, bucket_seconds)
    hot, share = hotspot(profile)
    d = Description(
        seconds=profile.seconds,
        energy=profile.energy,
        band=_band(profile.energy),
        spark=profile.spark,
        dead=dead.as_dict() if dead else None,
        camera_share=profile.camera,
        camera_overall=profile.camera_path,
        camera_segments=camera_segments(profile, bucket_seconds),
        placement=cols,
        placement_rows=rows,
        travel=travel(cols),
        hotspot=hot,
        region_share=share,
        framing=framing,
    )
    d.sentences = _sentences(d)
    return d


def _sentences(d: Description) -> list[str]:
    out: list[str] = []

    # 1. Camera — in the words a prompt uses.
    if d.camera_overall is not None:
        moving = [s for s in d.camera_segments if s.label != "static"]
        if not moving:
            out.append("Camera: locked off — no whole-frame move measured.")
        else:
            runs = ", then ".join(
                " ".join(filter(None, [s.tempo, s.label])) + f" {s.start:g}–{s.end:g}s"
                for s in moving
            )
            kind = ""
            if d.camera_share is not None:
                kind = (
                    " (a camera-path shot)" if d.camera_share >= 0.60
                    else f" ({d.camera_share * 100:.0f}% of the movement is whole-frame — "
                         f"the rest is the subject)"
                )
            out.append(f"Camera: {runs}{kind}.")
    else:
        out.append("Camera: not measured (flow pass skipped).")

    # 2. Framing — the shot-size word a prompt uses, when depth frames were available.
    if d.framing is not None:
        f = d.framing
        if f.cropped:
            line = f"Framing: {f.label} — the nearest figure is cropped by the frame"
        else:
            article = "an" if f.label[0] in "aeiou" else "a"
            line = (
                f"Framing: reads as {article} {f.label} — nearest figure ≈ "
                f"{f.fraction * 100:.0f}% of frame height"
            )
        if f.trend:
            line += f", {'tightening' if f.tightening else 'widening'} {f.trend}"
        out.append(line + ".")

    # 3. Subject placement and travel.
    if d.placement:
        live = [p for p in d.placement if p != "·"]
        if not live:
            out.append("Subject: nothing moves above the still floor.")
        else:
            where = d.hotspot or "unknown"
            if d.travel:
                first_live = next(i for i, p in enumerate(d.placement) if p != "·")
                enters = "late — " if first_live * BUCKET_SECONDS >= 0.75 else ""
                out.append(
                    f"Subject: the strongest motion travels {d.travel} "
                    f"({enters}starts {first_live * BUCKET_SECONDS:g}s in); "
                    f"hottest cell {where}."
                )
            else:
                out.append(f"Subject: the strongest motion stays put — hottest cell {where}.")

    # 4. Energy shape.
    shape = {"static": "still", "weak": "weak", "fine": "clear", "strong": "strong"}[d.band]
    dead = ""
    if d.dead:
        dead = f", with a {d.dead['seconds']:g}s dead stretch at the {d.dead['where']}"
    out.append(f"Energy: {d.energy:.2f} — {shape}{dead}  {d.spark}")
    return out


def render(d: Description, name: str, colour: bool = True, space: str = "depth") -> str:
    bold, dim, off = ("\033[1m", "\033[2m", "\033[0m") if colour else ("", "", "")
    lines = [f"{bold}{name}{off}  {dim}{d.seconds:g}s · {space} space · 24 fps{off}", ""]
    for sentence in d.sentences:
        lines.append(f"  {sentence}")
    lines.append("")
    if d.placement:
        lines.append(f"  placement  {''.join(d.placement)}   {dim}(strongest column per "
                     f"{BUCKET_SECONDS:g}s: L C R · = still){off}")
        lines.append(f"  rows       {''.join(d.placement_rows)}   {dim}(T M B){off}")
    if d.region_share is not None:
        lines.append(f"  {dim}where the motion sat (share):{off}")
        for row in d.region_share:
            lines.append("    " + " ".join(f"{v * 100:3.0f}%" for v in row))
    lines.append("")
    lines.append(
        f"  {dim}Uncalibrated. This says what moved and which way, in {space} space; read it "
        f"against the prompt. Whether the moving thing is the one the prompt named is a "
        f"by-eye call.{off}"
    )
    return "\n".join(lines)
