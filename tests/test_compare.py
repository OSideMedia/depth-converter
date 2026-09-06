"""Adherence comparison tests — pure math, no model, no codec.

The calibration these thresholds encode comes from the seven measured fires of
2026-07-25 (LIVE-FIRE.md): adhered = +6-8% energy with r0 ≥ 0.92; the three 1080p
failures = +22-32% with r0 ≤ 0.72. Synthetic envelopes here reproduce those shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthc import compare as c
from depthc import compare as comparelib
from depthc import motion as m
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


# ------------------------------------------------- structure layers: regions + camera path

from depthc import motion as m  # noqa: E402
from tests.test_motion import moving_clip, panning_clip, still_clip  # noqa: E402


def _mirror(clip: np.ndarray) -> np.ndarray:
    return clip[:, :, ::-1].copy()


def test_the_envelope_cannot_tell_a_crossing_from_its_mirror_but_regions_can():
    """The blindness the layer was built for: same beat, opposite direction."""
    clip = moving_clip(72, speed=5)
    ref = m.measure(clip, 24.0, with_flow=False)
    gen = m.measure(_mirror(clip), 24.0, with_flow=False)
    comp = c.compare_profiles(ref, gen)
    # Full-frame: indistinguishable — ratio 1.0, r0 1.0, "adhered".
    assert comp.ratio == pytest.approx(1.0, abs=0.01)
    assert comp.r0 == pytest.approx(1.0, abs=0.01)
    assert comp.suggestion == "adhered"
    # Regions: the mirrored control wins and the report names it.
    assert comp.structure_r is not None and comp.mirror_r is not None
    assert comp.mirror_r > comp.structure_r
    assert comp.mirrored
    assert any("MIRRORED" in r for r in comp.reasons)
    # ...without moving the calibrated verdict.
    assert comp.suggestion == "adhered"


def test_the_same_crossing_agrees_with_itself_in_every_region():
    clip = moving_clip(72, speed=5)
    ref = m.measure(clip, 24.0, with_flow=False)
    comp = c.compare_profiles(ref, m.measure(clip.copy(), 24.0, with_flow=False))
    assert comp.structure_r == pytest.approx(1.0, abs=0.01)
    assert not comp.mirrored
    # Weights sum to one over the cells the ref actually used.
    assert sum(sum(row) for row in comp.region_weights) == pytest.approx(1.0, abs=0.01)


def test_regions_are_skipped_on_bare_profiles_and_when_switched_off():
    bare = profile(ENVELOPE)
    assert c.compare_profiles(bare, bare).region_r is None
    clip = moving_clip(48, speed=5)
    ref = m.measure(clip, 24.0, with_flow=False)
    assert c.compare_profiles(ref, ref, regions=False).region_r is None
    assert c.compare_profiles(ref, ref, camera_path=False).ref_path is None


def test_opposite_pans_have_opposite_labels_and_a_negative_direction_cosine():
    ref = m.measure(panning_clip(72), 24.0)
    gen = m.measure(panning_clip(72, speed=-4), 24.0)
    comp = c.compare_profiles(ref, gen)
    assert comp.path_judged is True  # a pan is a camera-path ref
    assert comp.ref_path["label"] == "pans left"
    assert comp.gen_path["label"] == "pans right"
    assert comp.path_cos is not None and comp.path_cos < -0.9
    same = c.compare_profiles(ref, m.measure(panning_clip(72), 24.0))
    assert same.path_cos == pytest.approx(1.0, abs=0.01)


def test_the_path_is_not_judged_on_a_body_ref():
    """On a subject-motion ref the median flow is noise: name it, do not score it."""
    ref = m.measure(moving_clip(48, speed=6), 24.0)
    gen = m.measure(moving_clip(48, speed=6), 24.0)
    comp = c.compare_profiles(ref, gen)
    assert comp.path_judged is False
    assert comp.path_cos is None
    assert comp.ref_path is not None  # still named


def test_render_carries_the_structure_lines_and_the_dict_round_trips():
    import json

    clip = moving_clip(72, speed=5)
    ref = m.measure(clip, 24.0)
    comp = c.compare_profiles(ref, m.measure(_mirror(clip), 24.0))
    text = c.render(comp, "kit", "gen.mp4", colour=False)
    assert "REGIONS" in text and "PATH" in text and "MIRRORED" in text
    payload = json.loads(json.dumps(comp.as_dict()))
    assert payload["mirrored"] is True
    assert len(payload["regionR"]) == 3 and len(payload["regionR"][0]) == 3


# --------------------------------------------------------------- the variant ratio bias
# Measured 2026-08-24, before the fix: an identical generation compared against depth.mp4
# scored ratio 1.000 ("in band") and against variants/midgrey.mp4 scored 1.363 ("+36%
# OVER-FOLLOWING — the artifact signature"), which is louder than every 1080p failure in
# DOCS/LIVE-FIRE.md (+22% to +32%). The variant caps luma at 190 so its measured energy is
# ~0.74x the base, while the generation returns through engine.normalize() at full range.


def _moving_blob(n=120, h=90, w=160, seed=7):
    """A bright blob crossing a dark field — full 0-255 range, as normalize() guarantees."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        x = int(10 + (w - 40) * (i / (n - 1)))
        frames[i, 25:70, x:x + 30] = 255
    noisy = frames.astype(np.int16) + rng.integers(-4, 5, frames.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_a_luma_scaled_variant_does_not_flip_a_perfect_fire_to_over_following():
    """THE regression test for the bias. Same generation, two references, one verdict."""
    base_frames = _moving_blob()
    base = m.measure(base_frames, 24.0)
    midgrey = m.measure(m.midgrey(base_frames), 24.0)
    scale = midgrey.energy / base.energy

    # A generation that followed its reference perfectly: identical motion, full range.
    against_base = comparelib.compare_profiles(base, base, regions=False, camera_path=False)
    against_variant = comparelib.compare_profiles(
        midgrey, base, ref_scale=scale, regions=False, camera_path=False
    )

    assert against_base.suggestion == against_variant.suggestion
    assert abs(against_variant.ratio - against_base.ratio) < 0.05
    assert against_variant.ratio <= comparelib.RATIO_HIGH


def test_without_the_correction_the_same_pair_reads_as_over_following():
    """The counterexample, so the test above cannot pass for the wrong reason.

    This is the pre-fix behaviour, asserted deliberately: uncorrected, the identical
    generation crosses RATIO_HIGH. If this ever stops being true the fixture has drifted
    and the test above is no longer measuring anything.
    """
    base_frames = _moving_blob()
    base = m.measure(base_frames, 24.0)
    midgrey = m.measure(m.midgrey(base_frames), 24.0)

    uncorrected = comparelib.compare_profiles(
        midgrey, base, regions=False, camera_path=False
    )
    assert uncorrected.ratio > comparelib.RATIO_HIGH
    assert uncorrected.suggestion != "adhered"


def test_the_correction_is_named_in_the_reasons_not_applied_silently():
    base_frames = _moving_blob()
    base = m.measure(base_frames, 24.0)
    midgrey = m.measure(m.midgrey(base_frames), 24.0)
    c = comparelib.compare_profiles(
        midgrey, base, ref_scale=midgrey.energy / base.energy,
        regions=False, camera_path=False,
    )
    assert any("ratio corrected for the fired variant" in r for r in c.reasons)


def test_an_uncorrectable_variant_takes_the_ratio_out_of_the_verdict():
    """`edges` is a binary Canny series — no linear factor puts it on the base scale."""
    base_frames = _moving_blob()
    base = m.measure(base_frames, 24.0)
    c = comparelib.compare_profiles(
        base, base, ratio_note="binary Canny series", regions=False, camera_path=False
    )
    assert c.ratio is None
    assert any("energy ratio NOT judged" in r for r in c.reasons)


def test_an_unscaled_variant_is_left_alone():
    """invert flips near/far but |diff| is unchanged, so its scale is 1 and nothing moves."""
    base_frames = _moving_blob()
    base = m.measure(base_frames, 24.0)
    inverted = m.measure(m.invert(base_frames), 24.0)
    assert abs(inverted.energy / base.energy - 1.0) < 0.02
    c = comparelib.compare_profiles(
        inverted, base, ref_scale=inverted.energy / base.energy,
        regions=False, camera_path=False,
    )
    assert not any("ratio corrected" in r for r in c.reasons)
