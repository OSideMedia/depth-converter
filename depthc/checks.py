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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


def check(
    *,
    duration: float,
    width: int,
    height: int,
    fps: float,
    size_bytes: int,
    container: str = "mp4",
    still_has_real_faces: bool = True,
) -> Report:
    """Grade one depth ref against the Seedance reference-input limits."""
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
