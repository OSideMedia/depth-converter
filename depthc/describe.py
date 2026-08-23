"""Describe a lone clip's motion in the words a prompt uses.

`depthc compare` answers "did the output follow the REFERENCE clip's motion?" — it needs a
kit. A shot fired from text alone has no kit, and until now the only way to check whether
its motion matched the prompt was to look at frames and say so — a read, not a measurement.

This module turns one MotionProfile (v0.7.0: energy, regions, camera path) into a short
description in prompt vocabulary — "camera pans left then pushes in; subject enters
frame-left and travels to frame-right, arriving late; strong, sustained" — so the check
becomes: read the prompt, read the description, do they say the same thing?

EVERYTHING HERE IS UNCALIBRATED. The labels are the same measurements compare reports,
read as words. Nothing votes; there is no verdict. It measures WHAT MOVED and WHICH WAY,
in depth space (the channel Seedance actually reads a ref through), on the same analysis
grid as a kit — so a description of a generation and the profile of the kit that drove it
speak the same units. It cannot read intent, identity, mood, or whether the "subject" is
the one the prompt named — that stays a by-eye call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def camera_segments(profile: MotionProfile, bucket_seconds: float = BUCKET_SECONDS) -> list[Segment]:
    """The camera path named per bucket, then collapsed into runs ("pans left 0-1.5s")."""
    if profile.path is None or not profile.path.size:
        return []
    step = max(1, int(round(bucket_seconds * profile.fps)))
    n = profile.path.shape[0] - step + 1
    if n <= 0:
        return []
    labels: list[str] = []
    speeds: list[float] = []
    for i in range(0, n, step):
        chunk = profile.path[i:i + step]
        moving = np.any(chunk != 0, axis=1)
        if not moving.any():
            labels.append("static")
            speeds.append(0.0)
            continue
        dx, dy, radial = (float(v) for v in chunk[moving].mean(axis=0))
        labels.append(path_label(dx, dy, radial))
        speeds.append(path_speed(dx, dy, radial))
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


def describe(profile: MotionProfile, bucket_seconds: float = BUCKET_SECONDS) -> Description:
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

    # 2. Subject placement and travel.
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

    # 3. Energy shape.
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
