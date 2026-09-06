"""Seedance constraint checker.

Every export is graded against the reference-input limits in three tiers, and the limits are
PER MODEL: `DIALECTS` below holds one record per envelope and `--dialect` selects it. A number
that differs between Seedance 2.0 and 2.5 is not a disagreement to be resolved, it is two
facts, and collapsing them elects one model's envelope as universal law.

    Where sources disagree ABOUT THE SAME MODEL:  prefer the primary vendor doc, record the
                                                  disagreement, and say which page won.
    Where sources describe DIFFERENT MODELS:      hold both, select by the kit's target model,
                                                  never collapse them.

That replaces the old "follow the STRICTER value" rule, which was a holding pattern pending a
human console check. Its two subjects both resolved against this file — the vendor states
200 MB, and the mixed-input 12 was a foreign platform's cap (see the note on DIALECTS) — so
the rule outlived its subject. Stricter is also not automatically safe: a too-strict PROVIDER
limit merely wastes headroom, but a too-strict MODEL-VERSION limit refuses capability that has
already been paid for, which is how this table came to fail a 2.5-legal 20-second reference.

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

import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from depthc import motion as motionlib

# --- the constraint table, PER MODEL ----------------------------------------------------
# One record per (model, provider) envelope, keyed by the same string the prompt template is
# named for, so `--dialect` selects the prose AND the numbers. Adding a model is a new record
# plus a new template file — no logic change, which is what PLAN.md §3 promised and what the
# 2.0-only table quietly broke.
#
# SOURCE OF TRUTH: BytePlus ModelArk "Create video task", § Video input requirements, vendored
# at OSIDE AI FILMMAKER/DOCS/vendor/seedance-2.5/ARK-API-create-video-task-2026-08-15.md:700-712.
# It states the dimension/size/fps block ONCE for both model lines and splits only duration and
# reference counts. Values below are quoted from it, not inferred.
#
# [CORRECTED 2026-08-24] The previous table was triangulated on 2026-07-24, before the vendor
# docs were on disk, and every drift ran in the WORK-REFUSED direction: a 2.5-legal 20s ref, a
# 1440p ref, a 4K ref and a 120MB ref all graded FAIL. Three specifics worth not re-deriving:
#   * MAX_PIXELS was 2_086_876 — a number that appears in NO vendor source. It traced to a
#     PLAN.md estimate ("≈480p-1080p total pixels"). The vendor ceiling is 4x higher, and the
#     refusal landed AFTER conversion, so the GPU time was already spent.
#   * MAX_FILE_MB was 50 as "the stricter value, sources contested". Never contested: the
#     vendor states 200 MB on three pages for both lines. The 50 also drove a lossy crf-20
#     re-encode of the one asset where re-encode artifacts ARE the signal.
#   * max_mixed_inputs was 12, cited to OSIDE's lint.ts. lint.ts now says 15, under a note
#     recording that 12 "traced to the Higgsfield PLATFORM surface ... a platform cap that
#     leaked into a direct-API validator and rejected three legal files." This file was the
#     second host of the same leak.
#
# [CLOSED 2026-09-06, audit E-3] The leak's SHAPE — one vendor table hand-copied into two
# languages with no exporter and no drift gate — is closed. OSIDE now exports its live
# envelope (`npm run export:seedance-envelope`) to `seedance-envelope.json` beside this
# file, and `DIALECTS` is built from that payload: the fields OSIDE enforces (durations,
# counts, file size, output ladder, start-frame rule) are READ from it, and the table below
# is the hand FALLBACK, held equal to the payload by `test_hand_table_matches_the_envelope`.
# The pixel window, aspect, edge and sweet-spot stay hand-owned because OSIDE enforces no
# input pixel window at all (audit DC-7). Absent file → hand table with a stderr notice;
# present-but-broken → refuse at import, naming the key. A number can now drift in exactly
# one direction — OSIDE changes it, the export changes, the parity test goes red here.


@dataclass(frozen=True)
class Dialect:
    """One model's reference-input and output envelope.

    `max_video_refs`, `max_mixed_inputs`, `aspect` and `edge_px` are RECORDED, not enforced —
    this tool grades one depth ref at a time and never assembles the request, so it cannot see
    how many refs ride together. They are here so the number has one correct home when
    something does wire them, and so nobody re-imports a foreign platform's cap.
    """

    name: str
    ref_seconds: tuple[float, float]     # per clip, vendor § Duration
    ref_total_seconds: float             # across all video refs in one request
    max_video_refs: int                  # recorded, not enforced
    max_mixed_inputs: int                # recorded, not enforced
    output_seconds: tuple[float, float]
    pixels: tuple[int, int]              # total px of the uploaded ref
    max_file_mb: float
    output_resolutions: tuple[str, ...]  # the OUTPUT ladder — see the dated note per record
    refs_with_start_frame: bool          # can reference arrays ride alongside a start frame?
    aspect: tuple[float, float] = (0.4, 2.5)      # recorded, not enforced
    edge_px: tuple[int, int] = (300, 6000)        # recorded, not enforced
    sweet_spot: tuple[float, float] = (3.0, 8.0)  # [community], not vendor — same on both


HAND_DIALECTS: dict[str, Dialect] = {
    "seedance-2.0": Dialect(
        name="seedance-2.0",
        ref_seconds=(2.0, 15.0),
        ref_total_seconds=15.0,
        max_video_refs=3,
        max_mixed_inputs=15,          # vendor: 15 = 9 image + 3 video + 3 audio
        output_seconds=(4.0, 15.0),
        pixels=(407_696, 8_295_044),
        max_file_mb=200.0,
        # [VERIFIED 2026-07-17 vs docs + console] 4k is the console's "Exclusive" lane.
        output_resolutions=("480p", "720p", "1080p", "4k"),
        refs_with_start_frame=False,  # a start frame takes the whole request; refs are dropped
    ),
    "seedance-2.5": Dialect(
        name="seedance-2.5",
        ref_seconds=(2.0, 30.0),
        ref_total_seconds=30.0,
        max_video_refs=10,
        max_mixed_inputs=50,          # vendor: 50 = 30 image + 10 video + 10 audio
        output_seconds=(4.0, 30.0),   # 30s LIVE-CONFIRMED 2026-08-07 by OSIDE
        pixels=(407_696, 8_295_044),
        max_file_mb=200.0,
        # THE LADDER AS OF 2026-08-24, not a forecast. 1080p went live on the API 2026-08-17
        # (probed green: a 4s render came back and the FILE ffprobed 1920x1080 — measured, not
        # an echoed value). 4k is NOT offered on this model TODAY; whether ByteDance adds it
        # later is unknown and unknowable from here, so this is a dated observation and a
        # future 4k rung is a one-word edit to this tuple, not a rewrite.
        output_resolutions=("480p", "720p", "1080p"),
        refs_with_start_frame=True,   # the frame demotes to reference_image; refs still ride
    ),
}

# --- the exported envelope --------------------------------------------------------------
# OSIDE is the owning source for everything it enforces on a request. It exports that
# envelope off its live adapter cards (`lib/fal/adapters/arkSeedance.ts`) and lint input
# pools; the file is GENERATED and never hand-edited here, and OSIDE's own
# `check:seedance-envelope-drift` holds the file equal to the cards on every OSIDE commit.

ENVELOPE_PATH = Path(__file__).with_name("seedance-envelope.json")
ENVELOPE_FORMAT = "oside-seedance-envelope/1"
ENVELOPE_REFRESH = "npm run export:seedance-envelope -- {path}"  # run inside OSIDE

# Dialect field → (payload key, coercion). Only the fields OSIDE ENFORCES are listed —
# pixels / aspect / edge_px / sweet_spot are deliberately absent (see the record note).
_ENVELOPE_FIELDS: dict[str, tuple[str, Callable[[object], object]]] = {}


class EnvelopeError(ValueError):
    """The exported envelope is present but not the shape this reader was written for.

    Present-but-broken is NOT absent: a silent fall back to the hand table would grade
    against numbers OSIDE may have moved, which is the exact leak the export closes.
    """


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {value!r}")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected true/false, got {value!r}")
    return value


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"expected [low, high], got {value!r}")
    return (_number(value[0]), _number(value[1]))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"expected a list of strings, got {value!r}")
    return tuple(value)


_ENVELOPE_FIELDS.update({
    "ref_seconds": ("refSeconds", _pair),
    "ref_total_seconds": ("refTotalSeconds", _number),
    "max_video_refs": ("maxVideoRefs", _integer),
    "max_mixed_inputs": ("maxMixedInputs", _integer),
    "output_seconds": ("outputSeconds", _pair),
    "max_file_mb": ("maxFileMb", _number),
    "output_resolutions": ("outputResolutions", _strings),
    "refs_with_start_frame": ("refsWithStartFrame", _boolean),
})


def _stderr(message: str) -> None:
    print(f"depthc: {message}", file=sys.stderr)


def _envelope_fields(name: str, record: object) -> dict[str, object]:
    """Coerce one dialect's payload into Dialect field values, naming any broken key."""
    if not isinstance(record, dict):
        raise EnvelopeError(f"dialects.{name} is not an object")
    out: dict[str, object] = {}
    for attr, (key, coerce) in _ENVELOPE_FIELDS.items():
        if key not in record:
            raise EnvelopeError(f"dialects.{name}.{key} is missing")
        try:
            out[attr] = coerce(record[key])
        except TypeError as e:
            raise EnvelopeError(f"dialects.{name}.{key}: {e}") from None
    return out


def envelope_age_days(payload: dict, now: datetime | None = None) -> float:
    """Days since the payload was exported. `exportedAt` is ISO-8601 with a Z suffix."""
    stamp = payload.get("exportedAt")
    if not isinstance(stamp, str):
        raise EnvelopeError("exportedAt is missing")
    try:
        exported = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        raise EnvelopeError(f"exportedAt {stamp!r} is not an ISO-8601 timestamp") from None
    if exported.tzinfo is None:
        exported = exported.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - exported).total_seconds() / 86_400


def envelope_drift(hand: dict[str, Dialect], payload: dict) -> list[str]:
    """Every overlapping field where the hand table and the payload disagree.

    Empty means the fallback is faithful. This is the parity gate's predicate, kept apart
    from the loader so a test can hand it a deliberately wrong table and require a hit.
    """
    drift: list[str] = []
    for name, record in payload.get("dialects", {}).items():
        if name not in hand:
            drift.append(f"{name}: in the envelope, not in the hand table")
            continue
        fields = _envelope_fields(name, record)
        for attr, value in fields.items():
            ours = getattr(hand[name], attr)
            if ours != value:
                drift.append(f"{name}.{attr}: hand {ours!r} != envelope {value!r}")
    return drift


def load_envelope(
    path: Path | str = ENVELOPE_PATH,
    *,
    hand: dict[str, Dialect] | None = None,
    now: datetime | None = None,
    warn: Callable[[str], None] = _stderr,
) -> dict[str, Dialect]:
    """The constraint table: OSIDE's exported envelope laid over the hand table.

    Absent file → the hand table, with one notice. Present file with a broken shape →
    EnvelopeError naming the key. Older than its own `staleAfterDays` → a warning naming
    the age and the refresh command; the numbers are still used, because a stale export is
    still the last thing OSIDE actually enforced.
    """
    hand = dict(HAND_DIALECTS if hand is None else hand)
    path = Path(path)
    if not path.is_file():
        warn(f"no exported envelope at {path} — grading from the hand table in checks.py")
        return hand

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EnvelopeError(f"{path.name} is not readable JSON: {e}") from None
    if not isinstance(payload, dict):
        raise EnvelopeError(f"{path.name}: top level is not an object")
    if payload.get("format") != ENVELOPE_FORMAT:
        raise EnvelopeError(
            f"{path.name}: format {payload.get('format')!r} is not {ENVELOPE_FORMAT!r}"
        )
    dialects = payload.get("dialects")
    if not isinstance(dialects, dict) or not dialects:
        raise EnvelopeError(f"{path.name}: dialects is missing or empty")

    merged = dict(hand)
    for name, record in dialects.items():
        fields = _envelope_fields(name, record)
        if name not in hand:
            # A dialect OSIDE ships that this tool has no template or pixel window for.
            # Not broken, just not gradable here yet: say so and leave it out.
            warn(f"{path.name} names {name!r}, which has no hand record or template here — ignored")
            continue
        merged[name] = dataclasses.replace(hand[name], **fields)

    limit = payload.get("staleAfterDays")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool):
        raise EnvelopeError(f"{path.name}: staleAfterDays is missing or not a number")
    age = envelope_age_days(payload, now)
    if age > limit:
        warn(
            f"{path.name} is {age:.0f} days old (staleAfterDays {limit:g}); refresh it "
            f"in OSIDE with `{ENVELOPE_REFRESH.format(path=path)}`"
        )
    return merged


DIALECTS: dict[str, Dialect] = load_envelope()

# [CHANGED 2026-09-06, audit DC-5] 2.5 is OSIDE's live line and the only kit on disk is a
# 2.5 kit; defaulting to 2.0 meant the inbox hint — which named no dialect — built a kit
# graded by 2.0 law (15s ceiling, refs-only, 4K rung) for a model that has none of those.
# `--dialect seedance-2.0` still selects the 2.0 template, prose and envelope in full.
DEFAULT_DIALECT = "seedance-2.5"


def dialect_for(name: str | Dialect | None = None) -> Dialect:
    """The constraint record for a dialect name. Unknown names name their alternatives."""
    if isinstance(name, Dialect):
        return name
    key = name or DEFAULT_DIALECT
    try:
        return DIALECTS[key]
    except KeyError:
        raise ValueError(
            f"Unknown dialect {key!r}. Available: {sorted(DIALECTS)}"
        ) from None


# Module-level names kept for the call sites that grade a lone ref before a dialect is known.
# They are DERIVED from the default record so there is exactly one source for each number.
_DEFAULT = DIALECTS[DEFAULT_DIALECT]
MIN_REF_SECONDS = _DEFAULT.ref_seconds[0]
MAX_REF_SECONDS = _DEFAULT.ref_seconds[1]
SWEET_SPOT = _DEFAULT.sweet_spot
MAX_FILE_MB = _DEFAULT.max_file_mb
MIN_PIXELS, MAX_PIXELS = _DEFAULT.pixels
OUTPUT_SECONDS = _DEFAULT.output_seconds
MIN_FPS = 24
MAX_FPS = 60

# Output length is requested in WHOLE SECONDS. OSIDE's Ark adapter sends
# `duration: Math.round(req.durationSeconds ?? 5)` against `durationRange: [4, 15]`
# (lib/fal/adapters/arkSeedance.ts:159,238), so a 3.92 s ref cannot be matched by any legal
# request: the generation is at least 4 s and whatever it does past the end of the reference
# is undefined. That is the grid trims are snapped to — integer seconds, minimum 4.
OUTPUT_GRID_SECONDS = 1.0
SNAP_TOLERANCE = 0.05

# fps grading: the accepted range gets ±0.5 of slack (a container reporting 59.94 for a
# 60 fps export is not a rejection case), but the exact-24 PASS is deliberately tight —
# 23.976 NTSC is NOT 24, it is precisely the cadence mismatch the WARN exists to catch,
# and a round() comparison used to wave it through as an exact match.
FPS_RANGE_SLACK = 0.5
FPS_EXACT_TOLERANCE = 0.005


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


def suggested_output_seconds(ref_seconds: float, dialect: str | Dialect | None = None) -> float:
    """Output length to request for this ref: a WHOLE number of seconds in the model's window.

    Seedance's output minimum is 4 s, so a 3 s ref necessarily drives a longer generation
    and the uncovered tail is undefined behavior. Matching output to ref length is the rule;
    where that is impossible the kit at least records the closest legal value. The result
    always lands on the integer grid — the adapter sends `Math.round(duration)`, so a
    fractional suggestion (5.04) would be silently rounded and the manifest would claim a
    match no legal request can make.
    """
    low, high = dialect_for(dialect).output_seconds
    return float(min(max(round(ref_seconds), low), high))


def on_output_grid(seconds: float, dialect: str | Dialect | None = None) -> bool:
    """Is this ref length exactly requestable as an output length?"""
    low, high = dialect_for(dialect).output_seconds
    return (
        low - SNAP_TOLERANCE <= seconds <= high + SNAP_TOLERANCE
        and abs(seconds - round(seconds)) <= SNAP_TOLERANCE
    )


def snap_trim(
    start: float, end: float, source_duration: float, dialect: str | Dialect | None = None
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
    low, high = dialect_for(dialect).output_seconds
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
            "Reference points: 4.40 = a sword duel that behaved, 0.94 = the prone crawl whose "
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
        if retention > motionlib.NOISE_RETENTION:
            add(Finding(
                Tier.WARN, "depth retention",
                f"{percent:.0f}% — the depth channel carries MORE motion than the source. "
                f"The surplus is usually depth flicker on a low-texture source, not "
                f"choreography.",
                "Watch depth.mp4 / sheet.png for shimmer in flat areas before firing; a "
                "ratio far above 100% is a noise flag, not good news.",
            ))
        elif retention < motionlib.LOW_RETENTION and energy < motionlib.STRONG_ENERGY:
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
    # `is not None`: a measured 0.0 (pure subject motion) is a real finding; None means the
    # flow pass was skipped, which is a different fact and stays silent.
    if motion.camera is not None:
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
    dialect: str | Dialect | None = None,
) -> Report:
    """Grade one depth ref against the Seedance reference-input limits.

    `motion` is optional: without it this is a shape check, and the findings that predict
    adherence are simply absent rather than guessed at.

    `dialect` selects WHICH model's envelope to grade against; it defaults to
    DEFAULT_DIALECT. Every limit below reads from that record — a message naming a number
    must also name the model, because "exceeds the 15s ceiling" is true of 2.0 and false of
    2.5 and the operator cannot tell which one refused them otherwise.
    """
    d = dialect_for(dialect)
    report = Report()
    add = report.findings.append
    pixels = width * height
    size_mb = size_bytes / 1_000_000
    min_ref, max_ref = d.ref_seconds
    min_px, max_px = d.pixels

    # ------------------------------------------------------------------ duration
    low, high = d.sweet_spot
    if duration < min_ref:
        add(Finding(
            Tier.FAIL, "duration",
            f"{duration:.2f}s is under the {min_ref:g}s minimum for a video reference.",
            "Extend the trim, or pick a longer beat from the source.",
        ))
    elif duration > max_ref:
        add(Finding(
            Tier.FAIL, "duration",
            f"{duration:.2f}s exceeds the {max_ref:g}s ceiling for video references "
            f"on {d.name}.",
            f"Trim to a single motion signal — {low:g}-{high:g}s follows best."
            + ("" if d.name != "seedance-2.0" else
               "  (seedance-2.5 accepts up to 30s: build the kit with --dialect seedance-2.5.)"),
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
    # still be un-matchable by any legal output request. Scoped to the legal ref window —
    # outside it the duration FAIL owns the finding, and a 16 s ref must not also collect a
    # misleading "not a whole number of seconds" note.
    if min_ref <= duration <= max_ref and not on_output_grid(duration, d):
        low, _ = d.output_seconds
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
    elif min_ref <= duration <= max_ref:
        add(Finding(
            Tier.PASS, "output coverage",
            f"{duration:.2f}s is exactly requestable as an output length.",
        ))

    # ------------------------------------------------------------------ file size
    if size_mb > d.max_file_mb:
        add(Finding(
            Tier.FAIL, "file size",
            f"{size_mb:.1f} MB exceeds the {d.max_file_mb:g} MB limit for {d.name}.",
            "Re-encode at a higher CRF, or drop to --resolution 480p.",
        ))
    else:
        add(Finding(Tier.PASS, "file size", f"{size_mb:.1f} MB is within {d.max_file_mb:g} MB."))

    # ------------------------------------------------------------------ pixels
    if pixels < min_px:
        add(Finding(
            Tier.FAIL, "resolution",
            f"{width}×{height} = {pixels:,} px is below the {min_px:,} px minimum.",
            "Raise --resolution (720p), or crop less aggressively.",
        ))
    elif pixels > max_px:
        add(Finding(
            Tier.FAIL, "resolution",
            f"{width}×{height} = {pixels:,} px exceeds the {max_px:,} px maximum.",
            "Drop to --resolution 720p.",
        ))
    else:
        add(Finding(
            Tier.PASS, "resolution", f"{width}×{height} = {pixels:,} px is in window."
        ))

    # ------------------------------------------------------------------ frame rate
    if not (MIN_FPS - FPS_RANGE_SLACK <= fps <= MAX_FPS + FPS_RANGE_SLACK):
        add(Finding(
            Tier.FAIL, "frame rate",
            f"{fps:g} fps is outside the accepted {MIN_FPS}-{MAX_FPS} fps range.",
            "Use --fps conform-24.",
        ))
    elif abs(fps - 24.0) > FPS_EXACT_TOLERANCE:
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
