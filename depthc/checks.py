"""Seedance constraint checker.

Every export is graded against the reference-input limits in three tiers. The numbers come
from triangulating fal's schema, OSIDE's Ark adapter, and several hands-on guides — the
official BytePlus doc pages are JS-rendered and could not be read directly, so a human
console check is on the Phase 4 checklist before any of these become hard failures.

Where sources disagree, the checker follows the STRICTER value:

    max file size    50 MB   (OSIDE's arkSeedance.ts header says 200 MB — contested)
    max mixed inputs 12      (lint.ts says 12, the adapter says 15)

The tiers mean different things and should not be collapsed:

    PASS  inside the 3-8 s single-signal sweet spot, everything else in window
    WARN  it will generate, but adherence is known to degrade — or a human needs to look
    FAIL  Seedance is expected to reject it, or the ref is unusable as motion

MECHANICAL CHECKS ARE THE EASY HALF. Duration, file size, resolution and fps cannot fail if
the code is right — they are shape assertions. Whether a ref *works* depends on how much
motion survives into the depth channel and where it sits, which is why `motion.py` exists and
why a profile passed in here produces the findings that actually predict a bad generation.
Those findings are omitted entirely when no profile is supplied, so a shape-only check stays
a shape-only check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from depthc import motion as motionlib

# --- the constraint table (PLAN.md §0) --------------------------------------------------
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = 15.0
SWEET_SPOT = (3.0, 8.0)
MAX_FILE_MB = 50.0
CONTESTED_FILE_MB = 200.0
MIN_PIXELS = 409_600
MAX_PIXELS = 2_086_876
MIN_FPS = 24
MAX_FPS = 60
OUTPUT_SECONDS = (4.0, 15.0)
MAX_VIDEO_REFS = 3
MAX_MIXED_INPUTS = 12

# Output length is requested in WHOLE SECONDS. OSIDE's Ark adapter sends
# `duration: Math.round(req.durationSeconds ?? 5)` against `durationRange: [4, 15]`
# (lib/fal/adapters/arkSeedance.ts:159,238), so a 3.92 s ref cannot be matched by any legal
# request: the generation is at least 4 s and whatever it does past the end of the reference
# is undefined. That is the grid trims are snapped to — integer seconds, minimum 4.
OUTPUT_GRID_SECONDS = 1.0
SNAP_TOLERANCE = 0.05


class Tier(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


_ORDER = {Tier.PASS: 0, Tier.WARN: 1, Tier.FAIL: 2}


@dataclass
class Finding:
    tier: Tier
    check: str
    detail: str
    remedy: str = ""

    def as_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "check": self.check,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def tier(self) -> Tier:
        """The worst tier present — an export is only as good as its weakest check."""
        return max((f.tier for f in self.findings), key=lambda t: _ORDER[t], default=Tier.PASS)

    @property
    def ok(self) -> bool:
        """WARN still ships; FAIL does not."""
        return self.tier is not Tier.FAIL

    def of(self, tier: Tier) -> list[Finding]:
        return [f for f in self.findings if f.tier is tier]

    def as_dict(self) -> dict:
        return {"tier": self.tier.value, "findings": [f.as_dict() for f in self.findings]}


def suggested_output_seconds(ref_seconds: float) -> float:
    """Output length to request for this ref, clamped into Seedance's 4-15 s window.

    Seedance's output minimum is 4 s, so a 3 s ref necessarily drives a longer generation
    and the uncovered tail is undefined behavior. Matching output to ref length is the rule;
    where that is impossible the kit at least records the closest legal value.
    """
    low, high = OUTPUT_SECONDS
    return round(min(max(ref_seconds, low), high), 2)


def on_output_grid(seconds: float) -> bool:
    """Is this ref length exactly requestable as an output length?"""
    low, _ = OUTPUT_SECONDS
    return seconds >= low - SNAP_TOLERANCE and abs(seconds - round(seconds)) <= SNAP_TOLERANCE


def snap_trim(
    start: float, end: float, source_duration: float
) -> tuple[tuple[float, float], str | None]:
    """Widen or nudge a trim so its span is a whole number of seconds, at least 4.

    Returns the adjusted (start, end) and a one-line note, or (unchanged, None).

    THE RULE: never ship a ref shorter than the generation it drives. A ref that stops before
    the output does leaves the model to invent the tail, and a ref whose length is not
    requestable at all guarantees that gap. CLIP A shipped at 3.92 s against a 4 s floor.

    The in-point is held wherever possible, because that is the deliberate choice — the frame
    the beat starts on. The out-point moves first; only when the source runs out does the
    in-point slide back, and if the source is too short for a legal 4 s window nothing is
    changed and the checker says so instead of quietly fabricating footage.
    """
    span = end - start
    low, high = OUTPUT_SECONDS
    target = min(max(round(span), low), high)

    if source_duration < low:
        return (start, end), None  # nothing legal exists; the duration check owns this
    if target > source_duration:
        target = math.floor(source_duration)
        if target < low:
            return (start, end), None

    new_start, new_end = start, start + target
    if new_end > source_duration:
        new_end = source_duration
        new_start = max(0.0, round(source_duration - target, 3))

    new_start, new_end = round(new_start, 3), round(new_end, 3)
    if abs(new_start - start) < 1e-3 and abs(new_end - end) < 1e-3:
        return (start, end), None

    moved = "out-point" if abs(new_start - start) < 1e-3 else "window"
    return (new_start, new_end), (
        f"snapped {start:g}-{end:g}s ({span:.2f}s) to {new_start:g}-{new_end:g}s "
        f"({target:g}s) — {moved} moved so the ref length is requestable as an output length"
    )


def _motion_findings(
    add,
    motion: "motionlib.MotionProfile",
    retention: float | None,
) -> None:
    """The findings that predict a bad generation rather than a rejected upload."""
    energy = motion.energy
    spark = motion.spark

    # ------------------------------------------------------------------ signal strength
    if energy < motionlib.STATIC_FLOOR:
        add(Finding(
            Tier.FAIL, "motion signal",
            f"{energy:.2f} mean depth-space energy — this ref carries no usable motion "
            f"({spark}).",
            "Pick a beat where something changes distance to camera. Run `depthc scout` on "
            "the source to see where the motion actually is.",
        ))
    elif energy < motionlib.WEAK_ENERGY:
        add(Finding(
            Tier.WARN, "motion signal",
            f"{energy:.2f} mean depth-space energy is weak — below {motionlib.WEAK_ENERGY:g} "
            f"the model has little to lock onto and tends to free-run ({spark}).",
            "Reference points: 4.42 = a sword duel that behaved, 0.99 = the prone crawl whose "
            "generation started late. Run `depthc scout` for a stronger window.",
        ))
    elif energy > motionlib.JITTER_ENERGY:
        add(Finding(
            Tier.WARN, "motion signal",
            f"{energy:.2f} mean depth-space energy is very high — at this rate the ref is "
            f"shake or flicker rather than choreography ({spark}).",
            "Check the depth video and sheet.png. A whip-pan or handheld jitter transfers as "
            "noise; prefer a beat the eye can follow.",
        ))
    else:
        add(Finding(
            Tier.PASS, "motion signal",
            f"{energy:.2f} mean depth-space energy — a clear signal ({spark}).",
        ))

    # ------------------------------------------------------------------ dead stretches
    dead = motion.dead_stretch
    if dead and dead.where == "head":
        add(Finding(
            Tier.WARN, "late-start risk",
            f"the first {dead.seconds:.2f}s of the ref is effectively frozen — a generation "
            f"driven by it is expected to sit still and then start.",
            "Move the in-point to where the motion begins: `depthc scout` ranks windows with "
            "a live head.",
        ))
    elif dead:
        add(Finding(
            Tier.WARN, "dead stretch",
            f"{dead.seconds:.2f}s from {dead.start:.2f}s carries no motion ({dead.where}) — "
            f"the model has to invent that beat.",
            "Trim to the live part, or accept that this stretch is the model's to fill.",
        ))

    # ------------------------------------------------------------------ cuts / whip-pans
    if motion.spikes:
        where = ", ".join(f"{t:g}s" for t in motion.spikes[:4])
        add(Finding(
            Tier.WARN, "motion spikes",
            f"{len(motion.spikes)} frame gap(s) far above this clip's median, at {where} — "
            f"usually a hard cut or a whip-pan inside the ref.",
            "A cut is a second motion signal and the worst kind. Re-trim to one side of it.",
        ))

    # ------------------------------------------------------------------ depth retention
    if retention is not None:
        percent = retention * 100
        if retention < motionlib.LOW_RETENTION and energy < motionlib.STRONG_ENERGY:
            add(Finding(
                Tier.WARN, "depth retention",
                f"only {percent:.0f}% of the source's frame-to-frame motion survived into "
                f"depth — the movement is mostly parallel to the camera plane.",
                "Depth carries change in DISTANCE. Prefer a beat that moves toward or away "
                "from the lens; a crawl or a lateral slide across a flat floor nearly "
                "vanishes here (measured: 16% on the prone crawl, 87% on the sword duel).",
            ))
        else:
            add(Finding(
                Tier.PASS, "depth retention",
                f"{percent:.0f}% of the source's motion survived into depth.",
            ))

    # ------------------------------------------------------------------ what kind of ref
    if motion.camera:
        kind = (
            "camera-path" if motion.camera >= motionlib.CAMERA_DOMINANT
            else "subject-motion"
        )
        add(Finding(
            Tier.PASS, "motion type",
            f"{motion.camera * 100:.0f}% of the movement is whole-frame — this is a {kind} "
            f"reference.",
        ))


def check(
    *,
    duration: float,
    width: int,
    height: int,
    fps: float,
    size_bytes: int,
    container: str = "mp4",
    still_has_real_faces: bool = True,
    motion: "motionlib.MotionProfile | None" = None,
    retention: float | None = None,
) -> Report:
    """Grade one depth ref against the Seedance reference-input limits.

    `motion` is optional: without it this is a shape check, and the findings that predict
    adherence are simply absent rather than guessed at.
    """
    report = Report()
    add = report.findings.append
    pixels = width * height
    size_mb = size_bytes / 1_000_000

    # ------------------------------------------------------------------ duration
    low, high = SWEET_SPOT
    if duration < MIN_REF_SECONDS:
        add(Finding(
            Tier.FAIL, "duration",
            f"{duration:.2f}s is under the {MIN_REF_SECONDS:g}s minimum for a video reference.",
            "Extend the trim, or pick a longer beat from the source.",
        ))
    elif duration > MAX_REF_SECONDS:
        add(Finding(
            Tier.FAIL, "duration",
            f"{duration:.2f}s exceeds the {MAX_REF_SECONDS:g}s ceiling for video references.",
            "Trim to a single motion signal — 3-8s follows best.",
        ))
    elif duration < low:
        add(Finding(
            Tier.WARN, "duration",
            f"{duration:.2f}s is shorter than the {low:g}-{high:g}s sweet spot.",
            "Fine for a single beat; expect less for the model to lock onto.",
        ))
    elif duration > high:
        add(Finding(
            Tier.WARN, "duration",
            f"{duration:.2f}s is past the {low:g}-{high:g}s sweet spot — adherence degrades.",
            "Trim to the one clear motion signal you actually want copied.",
        ))
    else:
        add(Finding(
            Tier.PASS, "duration",
            f"{duration:.2f}s is inside the {low:g}-{high:g}s sweet spot.",
        ))

    # ------------------------------------------------------------------ output coverage
    # Independent of the sweet-spot check above: a ref can be a perfectly good length and
    # still be un-matchable by any legal output request.
    if duration >= MIN_REF_SECONDS and not on_output_grid(duration):
        low, _ = OUTPUT_SECONDS
        if duration < low:
            add(Finding(
                Tier.WARN, "output coverage",
                f"{duration:.2f}s is under Seedance's {low:g}s output floor, so the "
                f"generation is necessarily longer than the ref and its tail is uncovered.",
                f"Re-cut to a whole number of seconds, {low:g}s or more — `--snap` does it.",
            ))
        else:
            add(Finding(
                Tier.WARN, "output coverage",
                f"{duration:.2f}s is not a whole number of seconds; output length is "
                f"requested as an integer and rounded, so part of the generation runs past "
                f"the end of the ref.",
                "Re-cut to a whole number of seconds — `--snap` does it.",
            ))
    elif duration >= MIN_REF_SECONDS:
        add(Finding(
            Tier.PASS, "output coverage",
            f"{duration:.2f}s is exactly requestable as an output length.",
        ))

    # ------------------------------------------------------------------ file size
    if size_mb > MAX_FILE_MB:
        add(Finding(
            Tier.FAIL, "file size",
            f"{size_mb:.1f} MB exceeds the {MAX_FILE_MB:g} MB limit "
            f"(some sources say {CONTESTED_FILE_MB:g} MB — the stricter value is enforced).",
            "Re-encode at a higher CRF, or drop to --resolution 480p.",
        ))
    else:
        add(Finding(Tier.PASS, "file size", f"{size_mb:.1f} MB is within {MAX_FILE_MB:g} MB."))

    # ------------------------------------------------------------------ pixels
    if pixels < MIN_PIXELS:
        add(Finding(
            Tier.FAIL, "resolution",
            f"{width}×{height} = {pixels:,} px is below the {MIN_PIXELS:,} px minimum.",
            "Raise --resolution (720p), or crop less aggressively.",
        ))
    elif pixels > MAX_PIXELS:
        add(Finding(
            Tier.FAIL, "resolution",
            f"{width}×{height} = {pixels:,} px exceeds the {MAX_PIXELS:,} px maximum.",
            "Drop to --resolution 720p.",
        ))
    else:
        add(Finding(
            Tier.PASS, "resolution", f"{width}×{height} = {pixels:,} px is in window."
        ))

    # ------------------------------------------------------------------ frame rate
    if not (MIN_FPS <= round(fps) <= MAX_FPS):
        add(Finding(
            Tier.FAIL, "frame rate",
            f"{fps:g} fps is outside the accepted {MIN_FPS}-{MAX_FPS} fps range.",
            "Use --fps conform-24.",
        ))
    elif round(fps) != 24:
        add(Finding(
            Tier.WARN, "frame rate",
            f"{fps:g} fps does not match Seedance's 24 fps output cadence — "
            f"a reported cause of jitter.",
            "Use --fps conform-24.",
        ))
    else:
        add(Finding(Tier.PASS, "frame rate", "24 fps matches Seedance's output cadence."))

    # ------------------------------------------------------------------ container
    if container.lower() not in ("mp4", "mov"):
        add(Finding(
            Tier.FAIL, "container",
            f".{container} is not an accepted container (MP4 or MOV only).",
            "Re-export as MP4.",
        ))

    # ------------------------------------------------------------------ faces in the still
    if still_has_real_faces:
        add(Finding(
            Tier.WARN, "still.png faces",
            "The still is flagged as containing real photographic faces, which Seedance "
            "rejects on image inputs.",
            "Keep still.png local — never attach it as an @Image ref. "
            "Use --no-faces if the source has no people in it.",
        ))
    else:
        add(Finding(
            Tier.PASS, "still.png faces", "Flagged as free of real photographic faces."
        ))

    # ------------------------------------------------------------------ does it carry motion
    if motion is not None:
        _motion_findings(add, motion, retention)

    return report


def format_report(report: Report, colour: bool = True) -> str:
    """Render the report as the block printed after every export."""
    marks = {Tier.PASS: "✓", Tier.WARN: "!", Tier.FAIL: "✗"}
    colours = {Tier.PASS: "\033[32m", Tier.WARN: "\033[33m", Tier.FAIL: "\033[31m"}
    dim, off = ("\033[2m", "\033[0m") if colour else ("", "")

    lines = []
    for finding in report.findings:
        tint = colours[finding.tier] if colour else ""
        end = off if colour else ""
        lines.append(f"  {tint}{marks[finding.tier]}{end} {finding.check}: {finding.detail}")
        if finding.remedy and finding.tier is not Tier.PASS:
            lines.append(f"      {dim}→ {finding.remedy}{end}")

    tint = colours[report.tier] if colour else ""
    end = off if colour else ""
    lines.append(f"  {tint}{report.tier.value}{end}")
    return "\n".join(lines)
