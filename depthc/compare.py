"""Adherence measurement — was a generation actually driven by its reference?

This institutionalises the Gate-4 method (DOCS/LIVE-FIRE.md): the generated video is run
back through the same depth pipeline and its depth-space motion envelope is compared with
the reference's, bucket for bucket. Until 2026-07-25 this math lived only in session notes
and was re-derived by hand for every fire — and the one recorded historical measurement bug
(a 24 fps depth profile compared against a 12 fps source) lived in exactly that kind of
ad-hoc math. A command computes it the same way every time.

TWO NUMBERS, IN THIS ORDER:

1.  ENERGY RATIO (generated / reference) — the artifact predictor, judged FIRST.
    Over-following is what smears bodies and dissolves props: a blade survives a lunge, not
    a lunge rendered a third too fast. Every adherent run sat at +6-8%; all three 1080p
    failures ran +22-32%. Correlation cannot see this — the first 1080p failure's beat
    SHAPE still tracked at r=0.954 while its amplitude ran a third hot.

2.  ZERO-LAG CORRELATION of the 0.5 s-bucket envelopes, with lag controls. A real alignment
    peaks at zero lag and collapses on either side (Gate 4: r=+0.916 at 0, ~0 at ±0.5 s —
    the collapse is what makes it an alignment rather than two clips that happen to be
    busy). A peak at a positive lag instead means the generation runs BEHIND the ref — the
    "late" verdict the motion gate was built around.

The ratio is computed on the OVERLAPPING duration only: output length is integer seconds
with a 4 s floor, so a generation legally outruns a short ref, and whatever it does past
the end of the reference is the model's own invention — counting it would poison the ratio.

THE SUGGESTION IS A HEURISTIC, NOT A VERDICT. A human watches the clip and records what
actually happened (`depthc result`); this module only guarantees the numbers are computed
consistently. The thresholds are calibrated on the seven measured fires of 2026-07-25 and
say so by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from depthc.motion import MotionProfile, sparkline

BUCKET_SECONDS = 0.5
# Lags measured in buckets: ±1.0 s in 0.5 s steps. Wider lags on a 5 s clip leave too few
# buckets to correlate honestly.
LAG_BUCKETS = (-2, -1, 0, 1, 2)
MIN_LAG_BUCKETS = 4  # fewer points than this and a correlation is numerology

# Calibration, from the measured fires (LIVE-FIRE.md results table):
#   adhered: ratio +6-8%,  r0 = 0.92-0.95      failed 1080p: ratio +22-32%, r0 ≤ 0.72
RATIO_LOW = 0.85     # under this the ref is barely driving the output
RATIO_HIGH = 1.20    # over this is the over-following artifact signature
R_ADHERED = 0.75     # zero-lag r at or above this tracks timing beat for beat
R_IGNORED = 0.30     # below this the ref did essentially nothing
R_LATE = 0.60        # an off-zero peak this strong, with a dead zero, means "late"


@dataclass
class LagPoint:
    """Correlation of the two envelopes at one time offset."""

    seconds: float
    r: float | None  # None: not computable (too short, or a flat envelope)


@dataclass
class Comparison:
    """Everything `depthc compare` measures, in the order it should be judged."""

    ref_energy: float
    gen_energy: float
    ratio: float | None            # gen/ref on the overlap; None if the ref is flat
    overlap_seconds: float
    ref_tail_seconds: float        # ref beyond the overlap (gen shorter than ref)
    gen_tail_seconds: float        # generation beyond the overlap (uncovered invention)
    ref_spark: str
    gen_spark: str
    lags: list[LagPoint] = field(default_factory=list)
    ref_camera: float | None = None
    gen_camera: float | None = None
    suggestion: str = "partial"
    reasons: list[str] = field(default_factory=list)
    # Set when the two clips were not measured on comparable framing (aspect mismatch).
    # The frame-share energy metric is biased by framing alone (~34% between 2.39:1 and
    # 16:9 — motion.py's UNITS note), so the ratio must not drive the verdict; the lag
    # correlation compares each envelope's SHAPE against itself over time and survives.
    geometry_note: str | None = None

    @property
    def r0(self) -> float | None:
        return next((p.r for p in self.lags if p.seconds == 0.0), None)

    @property
    def best_lag(self) -> LagPoint | None:
        scored = [p for p in self.lags if p.r is not None]
        return max(scored, key=lambda p: p.r) if scored else None

    def as_dict(self) -> dict:
        return {
            "refEnergy": self.ref_energy,
            "genEnergy": self.gen_energy,
            "energyRatio": self.ratio,
            "overlapSeconds": self.overlap_seconds,
            "genTailSeconds": self.gen_tail_seconds,
            "refTailSeconds": self.ref_tail_seconds,
            "lags": {f"{p.seconds:+.1f}s": p.r for p in self.lags},
            "refCameraShare": self.ref_camera,
            "genCameraShare": self.gen_camera,
            "suggestion": self.suggestion,
            "reasons": self.reasons,
            "geometryNote": self.geometry_note,
        }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r with the degenerate cases named instead of NaN'd.

    A constant envelope has zero variance and no defined correlation — np.corrcoef would
    return NaN and a report would carry it silently. None forces the caller to say "flat".
    """
    if a.size < MIN_LAG_BUCKETS or a.size != b.size:
        return None
    if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _overlap_deltas(profile: MotionProfile, seconds: float) -> np.ndarray:
    """The raw per-gap deltas over the first `seconds` of a profile."""
    limit = int(round(seconds * profile.fps))
    return profile.deltas[:limit]


def _buckets(profile: MotionProfile, seconds: float) -> np.ndarray:
    """The 0.5 s energy envelope over the first `seconds` of a profile.

    FULL buckets only. A clip is almost never an exact multiple of the bucket, and the
    trailing partial bucket — sometimes a single frame gap — used to vote in the lag
    correlation with the same weight as twelve. The envelope exists for TIMING; the energy
    ratio is computed from the raw deltas, where every frame gap already weighs equally.
    """
    step = max(1, int(round(BUCKET_SECONDS * profile.fps)))
    deltas = _overlap_deltas(profile, seconds)
    return np.array(
        [float(deltas[i:i + step].mean()) for i in range(0, deltas.size - step + 1, step)],
        dtype=np.float64,
    )


def compare_profiles(
    ref: MotionProfile,
    gen: MotionProfile,
    *,
    ref_full_seconds: float | None = None,
    gen_full_seconds: float | None = None,
    geometry_note: str | None = None,
) -> Comparison:
    """Compare a generation's depth-motion profile against its reference's.

    Both profiles must be measured at the same fps (the kit pipeline guarantees 24); this
    is asserted rather than resampled, because resampling is exactly where the historical
    ratio bug came from.

    The energy ratio comes from the RAW deltas over the overlap — every frame gap weighs
    equally, so a clip that is not a whole multiple of the bucket cannot skew it. Buckets
    exist only for the lag correlation, and only full ones vote there.

    `ref_full_seconds` / `gen_full_seconds`: the clips' true durations when the caller
    decoded less than the whole file (the CLI trims the generation's decode to the ref's
    length so normalisation happens over the overlap only — a depth extreme in an excluded
    tail must not compress the greys of the part being judged). The tails in the report
    stay honest either way.
    """
    if abs(ref.fps - gen.fps) > 0.01:
        raise ValueError(
            f"Profiles measured at different rates ({ref.fps:g} vs {gen.fps:g} fps) — "
            f"per-frame energies are not comparable across rates."
        )

    overlap = min(ref.seconds, gen.seconds)
    rd = _overlap_deltas(ref, overlap)
    gd = _overlap_deltas(gen, overlap)
    m = min(rd.size, gd.size)
    rd, gd = rd[:m], gd[:m]
    rb = _buckets(ref, overlap)
    gb = _buckets(gen, overlap)
    n = min(rb.size, gb.size)
    rb, gb = rb[:n], gb[:n]

    ref_energy = round(float(rd.mean()), 3) if m else 0.0
    gen_energy = round(float(gd.mean()), 3) if m else 0.0
    ratio = round(gen_energy / ref_energy, 3) if ref_energy > 0 else None

    lags = []
    for k in LAG_BUCKETS:
        # Positive k: the generation's envelope is compared k buckets EARLIER in the ref —
        # a peak there means the generation runs behind the reference.
        a = rb[: n - k] if k >= 0 else rb[-k:]
        b = gb[k:] if k >= 0 else gb[: n + k]
        lags.append(LagPoint(seconds=k * BUCKET_SECONDS, r=_pearson(a, b)))

    # One ceiling for both sparklines: each scaled to its own peak, a +30% over-follow
    # draws bars identical to the ref's — the one picture in the report would hide the
    # very amplitude difference the report says to judge first.
    spark_top = float(max(rb.max() if n else 0.0, gb.max() if n else 0.0)) or None
    ref_seconds = ref.seconds if ref_full_seconds is None else ref_full_seconds
    gen_seconds = gen.seconds if gen_full_seconds is None else gen_full_seconds

    comparison = Comparison(
        ref_energy=ref_energy,
        gen_energy=gen_energy,
        ratio=ratio,
        overlap_seconds=round(overlap, 2),
        ref_tail_seconds=round(max(0.0, ref_seconds - overlap), 2),
        gen_tail_seconds=round(max(0.0, gen_seconds - overlap), 2),
        ref_spark=sparkline(list(rb), ceiling=spark_top),
        gen_spark=sparkline(list(gb), ceiling=spark_top),
        lags=lags,
        ref_camera=ref.camera,
        gen_camera=gen.camera,
        geometry_note=geometry_note,
    )
    comparison.suggestion, comparison.reasons = _suggest(comparison)
    return comparison


def _suggest(c: Comparison) -> tuple[str, list[str]]:
    """The heuristic verdict — ratio first, correlation second, human always last.

    Unless the geometries were not comparable: then the ratio is framing-biased and only
    the correlation votes, and the reasons say so up front.
    """
    reasons: list[str] = []
    r0, best = c.r0, c.best_lag

    over = c.ratio is not None and c.ratio > RATIO_HIGH
    under = c.ratio is not None and c.ratio < RATIO_LOW
    if c.geometry_note:
        over = under = False
        reasons.append(
            f"energy ratio NOT judged: {c.geometry_note} — the frame-share metric is "
            f"biased by framing alone; timing correlation carries the verdict"
        )
    if over:
        reasons.append(
            f"{(c.ratio - 1) * 100:+.0f}% energy — over-following, the artifact signature "
            f"(every failed 1080p run carried +22-32%); expect smearing and prop loss"
        )
    if under:
        reasons.append(
            f"{(c.ratio - 1) * 100:+.0f}% energy — the ref is barely driving the output"
        )

    if c.ratio is None:
        return "partial", ["the reference envelope is flat — nothing to compare against"]
    if r0 is None:
        return "partial", reasons + [
            "envelopes too flat or too short to correlate — judge this one by eye"
        ]

    if r0 >= R_ADHERED and not over and not under:
        off_zero = [abs(p.r) for p in c.lags if p.seconds != 0.0 and p.r is not None]
        if off_zero and max(off_zero) < r0 - 0.3:
            reasons.append(
                f"r={r0:+.3f} at zero lag, collapsing off-zero — a real alignment"
            )
        else:
            reasons.append(f"r={r0:+.3f} at zero lag")
        return "adhered", reasons

    # Late = the peak sits at a positive lag and clearly DOMINATES zero. Not "zero is
    # dead": a delayed beat keeps moderate zero-lag correlation (the overlap still
    # rhymes), so demanding a dead zero misses exactly the case this verdict names.
    if (
        best is not None
        and best.seconds > 0
        and best.r >= R_LATE
        and best.r - r0 >= 0.25
    ):
        reasons.append(
            f"envelope peaks at +{best.seconds:g}s (r={best.r:+.3f}), well above zero lag "
            f"(r={r0:+.3f}) — the generation runs behind the reference"
        )
        return "late", reasons

    if r0 < R_IGNORED:
        reasons.append(f"r={r0:+.3f} at zero lag, no usable peak — the ref did not drive this")
        return "ignored", reasons

    reasons.append(f"r={r0:+.3f} at zero lag — some of it transferred, not all")
    return "partial", reasons


def render(c: Comparison, kit_name: str, generated_name: str, colour: bool = True) -> str:
    """The report block, ratio first — the order the eye should read it in."""
    bold, dim, off = ("\033[1m", "\033[2m", "\033[0m") if colour else ("", "", "")
    green, yellow, red = ("\033[32m", "\033[33m", "\033[31m") if colour else ("", "", "")

    if c.ratio is None:
        band, tint = "reference envelope is flat", red
    elif c.geometry_note:
        band, tint = "RATIO UNRELIABLE — framing mismatch", red
    elif c.ratio > RATIO_HIGH:
        band, tint = "OVER-FOLLOWING — the artifact signature", red
    elif c.ratio < RATIO_LOW:
        band, tint = "under-following", yellow
    else:
        band, tint = "disciplined — the band every adherent run sat in", green

    lines = [
        f"{bold}{kit_name}{off}  ←  {generated_name}",
        "",
        f"  ENERGY   ref {c.ref_energy:.2f} · gen {c.gen_energy:.2f}"
        + (
            f" · ratio {c.ratio:.2f} ({(c.ratio - 1) * 100:+.0f}%)   "
            f"{tint}{band}{off}"
            if c.ratio is not None else f"   {tint}{band}{off}"
        ),
        f"           {dim}judge this first: over-following smears bodies and dissolves "
        f"props, and correlation cannot see it{off}",
        *(
            [f"  GEOMETRY {red}{c.geometry_note}{off}"]
            if c.geometry_note else []
        ),
        "  TIMING   "
        + " · ".join(
            (f"{bold}" if p.seconds == 0 else "")
            + f"{p.seconds:+.1f}s r={p.r:+.3f}".replace("+0.0s", " 0.0s")
            + (f"{off}" if p.seconds == 0 else "")
            if p.r is not None
            else f"{p.seconds:+.1f}s r=—"
            for p in c.lags
        ),
        f"  CAMERA   ref "
        + (f"{c.ref_camera * 100:.0f}%" if c.ref_camera is not None else "—")
        + " whole-frame · gen "
        + (f"{c.gen_camera * 100:.0f}%" if c.gen_camera is not None else "—"),
        "",
        f"  ref {c.ref_spark}",
        f"  gen {c.gen_spark}   {dim}({c.overlap_seconds:g}s overlap"
        + (f"; gen runs {c.gen_tail_seconds:g}s past the ref — excluded"
           if c.gen_tail_seconds > 0.5 else "")
        + (f"; ref runs {c.ref_tail_seconds:g}s past the gen"
           if c.ref_tail_seconds > 0.5 else "")
        + f"){off}",
        "",
        f"  suggested verdict: {bold}{c.suggestion}{off}",
    ]
    for reason in c.reasons:
        lines.append(f"    {dim}— {reason}{off}")
    lines.append(
        f"  {dim}The suggestion is a heuristic. Watch the clip, then record what actually "
        f"happened.{off}"
    )
    return "\n".join(lines)
