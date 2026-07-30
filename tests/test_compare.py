"""Adherence comparison tests — pure math, no model, no codec.

The calibration these thresholds encode comes from the seven measured fires of
2026-07-25 (LIVE-FIRE.md): adhered = +6-8% energy with r0 ≥ 0.92; the three 1080p
failures = +22-32% with r0 ≤ 0.72. Synthetic envelopes here reproduce those shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthc import compare as c
from depthc.motion import MotionProfile

FPS = 24
STEP = int(c.BUCKET_SECONDS * FPS)  # frames per bucket

# A busy, non-repeating 5s envelope (10 half-second buckets) — varied enough that a
# correlation has something to hold onto and a shift genuinely decorrelates zero lag.
ENVELOPE = [4.0, 2.0, 5.5, 3.0, 6.0, 2.5, 5.0, 3.5, 4.5, 2.0]


def profile(buckets: list[float], fps: float = 24.0) -> MotionProfile:
    """A profile whose 0.5s bucket envelope is exactly `buckets`."""
    deltas = np.repeat(np.array(buckets, dtype=np.float32), STEP)
    return MotionProfile(fps=fps, deltas=deltas)


def test_identical_envelopes_are_an_adhered_alignment():
    comp = c.compare_profiles(profile(ENVELOPE), profile(ENVELOPE))
    assert comp.ratio == pytest.approx(1.0)
    assert comp.r0 == pytest.approx(1.0)
    assert comp.suggestion == "adhered"


def test_the_gate4_shape_is_adhered():
    """+8% energy, same beat shape — the measured 720p signature."""
    hotter = [v * 1.08 for v in ENVELOPE]
    comp = c.compare_profiles(profile(ENVELOPE), profile(hotter))
    assert comp.suggestion == "adhered"
    assert comp.ratio == pytest.approx(1.08, abs=0.01)


def test_over_following_is_never_adhered_even_with_perfect_timing():
    """The first 1080p failure: beat shape tracked at r=0.954 while amplitude ran a third
    hot — ratio outranks correlation, which is the whole metric lesson."""
    hot = [v * 1.32 for v in ENVELOPE]
    comp = c.compare_profiles(profile(ENVELOPE), profile(hot))
    assert comp.r0 == pytest.approx(1.0)
    assert comp.suggestion != "adhered"
    assert any("over-following" in r for r in comp.reasons)


def test_an_unrelated_envelope_is_ignored():
    """The second 1080p failure: r=-0.056, the ref did nothing."""
    comp = c.compare_profiles(profile(ENVELOPE), profile(ENVELOPE[::-1]))
    assert comp.r0 is not None and comp.r0 < c.R_IGNORED
    assert comp.suggestion == "ignored"


def test_a_delayed_generation_reads_as_late():
    """The generation runs 1s behind the ref: peak at +1.0s, dead zero lag."""
    delayed = [0.3, 0.3] + ENVELOPE[:-2]  # everything arrives two buckets later
    comp = c.compare_profiles(profile(ENVELOPE), profile(delayed))
    best = comp.best_lag
    assert best is not None and best.seconds == pytest.approx(1.0)
    assert comp.suggestion == "late"


def test_flat_envelopes_say_flat_instead_of_nan():
    comp = c.compare_profiles(profile([2.0] * 10), profile([2.0] * 10))
    assert comp.r0 is None
    assert comp.suggestion == "partial"
    assert any("flat" in r for r in comp.reasons)
    assert not any("nan" in str(r).lower() for r in comp.reasons)


def test_a_flat_reference_never_divides_by_zero():
    comp = c.compare_profiles(profile([0.0] * 10), profile(ENVELOPE))
    assert comp.ratio is None
    assert comp.suggestion == "partial"


def test_the_ratio_is_computed_on_the_overlap_only():
    """A generation legally outruns a short ref; its invented tail must not poison the
    ratio. Here the tail is wild (x10) but the overlapping 5s is identical."""
    long_gen = ENVELOPE + [40.0] * 6  # 3s of invented tail
    comp = c.compare_profiles(profile(ENVELOPE), profile(long_gen))
    assert comp.overlap_seconds == pytest.approx(5.0, abs=0.1)
    assert comp.gen_tail_seconds == pytest.approx(3.0, abs=0.1)
    assert comp.ratio == pytest.approx(1.0, abs=0.02)
    assert comp.suggestion == "adhered"


def test_mismatched_fps_is_refused_not_resampled():
    """The historical ratio bug was a 24fps profile against a 12fps one. Refuse loudly."""
    with pytest.raises(ValueError):
        c.compare_profiles(profile(ENVELOPE), profile(ENVELOPE, fps=12.0))


def test_render_is_plain_text_without_colour_and_leads_with_energy():
    comp = c.compare_profiles(profile(ENVELOPE), profile(ENVELOPE))
    text = c.render(comp, "Kit_001", "gen.mp4", colour=False)
    assert "\033[" not in text
    lines = [line for line in text.splitlines() if line.strip()]
    assert "ENERGY" in lines[1]  # ratio before timing, always
    assert "TIMING" in text
    assert "suggested verdict" in text


# ------------------------------------------------------- 2026-07-30 council audit pins

def test_partial_tail_bucket_cannot_skew_the_ratio():
    """A 119-delta clip (every real 5s/24fps decode) must not weight its 11-delta tail
    like a full bucket. Energy comes from the raw deltas; buckets are timing-only."""
    deltas = np.repeat(np.array(ENVELOPE, dtype=np.float32), STEP)[:-1]  # 119 deltas
    ref = MotionProfile(fps=24.0, deltas=deltas)
    gen = MotionProfile(fps=24.0, deltas=deltas * 1.05)
    comp = c.compare_profiles(ref, gen)
    assert comp.ratio == pytest.approx(1.05, abs=0.005)  # the true per-gap ratio
    # and the envelope kept only FULL buckets: 119 // 12 = 9
    assert len(c._buckets(ref, ref.seconds)) == 9


def test_geometry_note_takes_the_ratio_out_of_the_verdict():
    """An aspect mismatch biases frame-share energy ~34% (motion.py UNITS): a perfectly
    adherent generation must not be downgraded for a ratio the framing manufactured."""
    biased = [v * 0.74 for v in ENVELOPE]  # the measured 2.39:1-vs-16:9 shape
    plain = c.compare_profiles(profile(ENVELOPE), profile(biased))
    assert plain.suggestion != "adhered"  # without the note, the bias reads as under-following

    noted = c.compare_profiles(
        profile(ENVELOPE), profile(biased),
        geometry_note="ref 2.39:1 vs generation 16:9",
    )
    assert noted.suggestion == "adhered"
    assert any("NOT judged" in reason for reason in noted.reasons)
    assert "RATIO UNRELIABLE" in c.render(noted, "Kit_001", "gen.mp4", colour=False)


def test_sparklines_share_one_ceiling():
    """A +30% over-follow used to draw glyph-identical bars (each spark scaled to its
    own peak) directly under the line saying amplitude is what to judge first."""
    hot = [v * 1.3 for v in ENVELOPE]
    comp = c.compare_profiles(profile(ENVELOPE), profile(hot))
    assert comp.ref_spark != comp.gen_spark


def test_gen_full_seconds_keeps_the_reported_tail_honest():
    """The CLI trims the generation's decode to the ref's length; the report must still
    say how far the generation actually ran past the reference."""
    comp = c.compare_profiles(
        profile(ENVELOPE), profile(ENVELOPE), gen_full_seconds=8.0
    )
    assert comp.gen_tail_seconds == pytest.approx(8.0 - comp.overlap_seconds, abs=0.05)
