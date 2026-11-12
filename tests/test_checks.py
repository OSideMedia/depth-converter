from __future__ import annotations

import numpy as np
import pytest

from depthc import checks, motion as m
from depthc.checks import (
    Tier, check, format_report, on_output_grid, snap_trim, suggested_output_seconds,
)

GOOD = dict(
    duration=5.0, width=1280, height=720, fps=24.0,
    size_bytes=3_000_000, container="mp4", still_has_real_faces=False,
)


def tier_of(name: str, report) -> Tier:
    return next(f.tier for f in report.findings if f.check == name)


def test_sweet_spot_is_a_clean_pass():
    report = check(**GOOD)
    assert report.tier is Tier.PASS
    assert report.ok


@pytest.mark.parametrize("seconds", [3.0, 5.0, 8.0])
def test_duration_sweet_spot_boundaries_pass(seconds):
    assert tier_of("duration", check(**{**GOOD, "duration": seconds})) is Tier.PASS


@pytest.mark.parametrize("seconds", [2.0, 2.9])
def test_short_but_legal_duration_warns(seconds):
    assert tier_of("duration", check(**{**GOOD, "duration": seconds})) is Tier.WARN


@pytest.mark.parametrize("seconds", [8.1, 14.9])
def test_long_duration_warns(seconds):
    assert tier_of("duration", check(**{**GOOD, "duration": seconds})) is Tier.WARN


@pytest.mark.parametrize("seconds", [0.5, 1.99])
def test_under_two_seconds_fails(seconds):
    """The 2s floor was an explicit amendment — under it, Seedance rejects the ref."""
    report = check(**{**GOOD, "duration": seconds})
    assert tier_of("duration", report) is Tier.FAIL
    assert not report.ok


@pytest.mark.parametrize("seconds", [15.1, 30.0])
def test_over_fifteen_seconds_fails_on_2_0(seconds):
    """2.0 law, so 2.0 is named: the default moved to 2.5 on 2026-09-06 (audit DC-5)."""
    assert tier_of("duration", check(**{**GOOD, "duration": seconds},
                                     dialect="seedance-2.0")) is Tier.FAIL


@pytest.mark.parametrize("seconds", [30.1, 45.0])
def test_over_thirty_seconds_fails_on_the_default(seconds):
    assert tier_of("duration", check(**{**GOOD, "duration": seconds})) is Tier.FAIL


def test_oversize_file_fails_at_the_vendor_limit():
    """200 MB, quoted from the Ark API page for BOTH model lines.

    [RE-SCOPED 2026-08-24] This asserted FAIL at 60 MB under a "sources are contested, take
    the stricter" rule. The vendor states 200 MB on three pages, so 60 MB was legal input the
    checker refused — and this assertion is what would have made that correction look like a
    regression. The counterexample below is the point: a vendor-legal 120 MB ref must PASS.
    """
    report = check(**{**GOOD, "size_bytes": 210_000_000})
    assert tier_of("file size", report) is Tier.FAIL
    assert "200" in next(f.detail for f in report.findings if f.check == "file size")


def test_a_vendor_legal_large_file_passes():
    """The refusal this file used to certify. 120 MB is inside 200 and must ship."""
    assert tier_of("file size", check(**{**GOOD, "size_bytes": 120_000_000})) is Tier.PASS


def test_file_just_under_the_limit_passes():
    assert tier_of("file size", check(**{**GOOD, "size_bytes": 199_000_000})) is Tier.PASS


def test_too_few_pixels_fails():
    report = check(**{**GOOD, "width": 320, "height": 240})
    assert tier_of("resolution", report) is Tier.FAIL


def test_too_many_pixels_fails():
    """DCI 4K (4096x2160 = 8,847,360) is genuinely over the vendor ceiling of 8,295,044."""
    report = check(**{**GOOD, "width": 4096, "height": 2160})
    assert tier_of("resolution", report) is Tier.FAIL


@pytest.mark.parametrize("w,h", [(1920, 1080), (2560, 1440), (3840, 2160)])
def test_vendor_legal_resolutions_pass(w, h):
    """[RE-SCOPED 2026-08-24] UHD 3840x2160 = 8,294,400 clears the vendor ceiling by 644 px.

    This file previously asserted that exact resolution FAILS, against a ceiling
    (2,086,876) that appears in no vendor source. 1440p and UHD refs were being refused
    AFTER conversion — the GPU time already spent — and this assertion certified it.
    """
    assert tier_of("resolution", check(**{**GOOD, "width": w, "height": h})) is Tier.PASS


def test_letterboxed_720p_still_passes_the_pixel_window():
    """A cropped 2.39:1 kit is 1280x536 — comfortably inside the window."""
    assert tier_of("resolution", check(**{**GOOD, "width": 1280, "height": 536})) is Tier.PASS


def test_non_24_fps_warns_but_ships():
    report = check(**{**GOOD, "fps": 30.0})
    assert tier_of("frame rate", report) is Tier.WARN
    assert report.ok


def test_ntsc_23976_is_a_cadence_warn_not_an_exact_match():
    """23.976 is NOT 24 — it is the jitter case the WARN exists for, and a round()
    comparison used to book it as '24 fps matches'."""
    report = check(**{**GOOD, "fps": 23.976})
    assert tier_of("frame rate", report) is Tier.WARN
    assert report.ok  # near-range, so it ships with a warning rather than failing


@pytest.mark.parametrize("fps", [12.0, 120.0])
def test_out_of_range_fps_fails(fps):
    assert tier_of("frame rate", check(**{**GOOD, "fps": fps})) is Tier.FAIL


def test_bad_container_fails():
    assert tier_of("container", check(**{**GOOD, "container": "webm"})) is Tier.FAIL


def test_mov_is_accepted():
    report = check(**{**GOOD, "container": "mov"})
    assert all(f.check != "container" for f in report.findings)


def test_real_faces_warn_and_say_local_only():
    report = check(**{**GOOD, "still_has_real_faces": True})
    finding = next(f for f in report.findings if f.check == "still.png faces")
    assert finding.tier is Tier.WARN
    assert "never attach it as an @Image ref" in finding.remedy
    assert report.ok  # a WARN still ships


def test_report_tier_is_the_worst_finding():
    report = check(**{**GOOD, "duration": 9.0, "size_bytes": 210_000_000})
    assert report.tier is Tier.FAIL
    assert len(report.of(Tier.WARN)) >= 1


def test_every_non_pass_finding_carries_a_remedy():
    report = check(duration=1.0, width=100, height=100, fps=9.0,
                   size_bytes=99_000_000, container="webm", still_has_real_faces=True)
    for finding in report.findings:
        if finding.tier is not Tier.PASS:
            assert finding.remedy, f"{finding.check} has no remedy"


@pytest.mark.parametrize(
    "ref,expected",
    [(2.0, 4.0), (3.0, 4.0), (5.0, 5.0), (8.0, 8.0), (15.0, 15.0), (20.0, 15.0),
     (5.02, 5.0), (6.4, 6.0), (3.92, 4.0)],
)
def test_suggested_output_clamps_to_the_output_window(ref, expected):
    """Seedance's output floor is 4s, so short refs necessarily drive a longer generation.

    And the suggestion always lands on the integer grid: duration is sent as an integer, so
    a fractional suggestion (5.04) is one no legal request can honour — the adapter would
    round it silently while the manifest claimed a match.
    """
    assert suggested_output_seconds(ref, "seedance-2.0") == expected


def test_suggested_output_clamps_to_the_default_models_window():
    """The default is 2.5 (audit DC-5): a 35s ref clamps to its 30s ceiling, not 15."""
    assert suggested_output_seconds(20.0) == 20.0
    assert suggested_output_seconds(35.0) == 30.0


# --------------------------------------------------------------------- the motion gate

def profile(energy: float, *, seconds: float = 5.0, head: float | None = None,
            spike: bool = False, camera: float = 0.0) -> m.MotionProfile:
    """A profile with a chosen mean energy, optionally with a weak head or a cut in it."""
    count = int(seconds * 24)
    deltas = np.full(count, energy, dtype=np.float32)
    if head is not None:
        deltas[: int(m.DEAD_STRETCH_SECONDS * 24) + 2] = head
    if spike:
        deltas[count // 2] = max(m.SPIKE_FLOOR * 2, energy * 10)
    return m.MotionProfile(fps=24.0, deltas=deltas, camera=camera)


def test_no_motion_findings_without_a_profile():
    """A shape-only check must stay a shape-only check rather than guess."""
    names = {f.check for f in check(**GOOD).findings}
    assert not names & {"motion signal", "depth retention", "late-start risk", "motion type"}


def test_a_frozen_ref_fails_the_gate():
    report = check(**GOOD, motion=profile(0.05))
    assert tier_of("motion signal", report) is Tier.FAIL
    assert not report.ok


def test_a_weak_ref_warns_and_still_ships():
    """0.99 is the prone crawl. It generates; it just generates badly."""
    report = check(**GOOD, motion=profile(0.99))
    assert tier_of("motion signal", report) is Tier.WARN
    assert report.ok


def test_a_strong_ref_passes_the_gate():
    """4.42 is the sword duel — the one clip that behaved."""
    assert tier_of("motion signal", check(**GOOD, motion=profile(4.42))) is Tier.PASS


def test_a_jittery_ref_warns_from_the_other_side():
    """The band has a ceiling: past it the ref is shake, not choreography."""
    assert tier_of("motion signal", check(**GOOD, motion=profile(40.0))) is Tier.WARN


def test_the_motion_finding_carries_the_sparkline():
    detail = next(
        f.detail for f in check(**GOOD, motion=profile(4.0)).findings
        if f.check == "motion signal"
    )
    assert any(block in detail for block in m.SPARK_BLOCKS)


def test_a_dead_first_beat_is_reported_as_late_start_risk():
    report = check(**GOOD, motion=profile(4.0, head=0.05))
    assert tier_of("late-start risk", report) is Tier.WARN


def test_a_dead_tail_is_a_dead_stretch_not_a_late_start():
    deltas = np.full(120, 4.0, dtype=np.float32)
    deltas[-30:] = 0.05
    report = check(**GOOD, motion=m.MotionProfile(fps=24.0, deltas=deltas))
    assert tier_of("dead stretch", report) is Tier.WARN
    assert all(f.check != "late-start risk" for f in report.findings)


def test_a_cut_inside_the_ref_warns():
    report = check(**GOOD, motion=profile(2.0, spike=True))
    assert tier_of("motion spikes", report) is Tier.WARN


def test_low_depth_retention_warns_on_a_weak_ref():
    """The CLIP A shape: real motion in RGB, almost none of it in depth."""
    report = check(**GOOD, motion=profile(0.99), retention=0.11)
    assert tier_of("depth retention", report) is Tier.WARN
    assert "distance" in next(
        f.remedy.lower() for f in report.findings if f.check == "depth retention"
    )


def test_low_retention_is_tolerated_when_the_signal_is_strong_anyway():
    assert tier_of("depth retention", check(**GOOD, motion=profile(6.0), retention=0.15)) \
        is Tier.PASS


def test_retention_is_omitted_when_it_could_not_be_measured():
    report = check(**GOOD, motion=profile(4.0))
    assert all(f.check != "depth retention" for f in report.findings)


def test_retention_far_above_100_percent_is_a_noise_flag_not_a_pass():
    """Depth flicker on a low-texture source can exceed the source's own motion; '150%
    survived' used to read as a cheerful PASS when it is evidence of shimmer."""
    report = check(**GOOD, motion=profile(4.0), retention=1.5)
    finding = next(f for f in report.findings if f.check == "depth retention")
    assert finding.tier is Tier.WARN
    assert "flicker" in finding.detail


def test_a_measured_zero_camera_share_still_names_the_motion_type():
    """0.0 is a measurement (pure subject motion); only None means 'flow pass skipped'."""
    report = check(**GOOD, motion=profile(4.0, camera=0.0))
    assert any(f.check == "motion type" for f in report.findings)
    silent = check(**GOOD, motion=m.MotionProfile(fps=24.0, deltas=np.full(120, 4.0, dtype=np.float32)))
    assert all(f.check != "motion type" for f in silent.findings)


def test_motion_type_names_a_camera_reference():
    report = check(**GOOD, motion=profile(4.0, camera=0.8))
    assert "camera-path" in next(
        f.detail for f in report.findings if f.check == "motion type"
    )


def test_motion_type_says_what_was_measured_and_where_the_camera_words_are():
    """The share is measured on the depth, where people crossing a locked-off shot look like a
    camera move (scored, DOCS/CAMERA-PATH-CALIBRATION.md). The line must not state a camera move
    as a fact: it says what it measured and points at the per-shot read."""
    for share in (0.8, 0.2):
        detail = next(
            f.detail for f in check(**GOOD, motion=profile(4.0, camera=share)).findings
            if f.check == "motion type"
        )
        assert "in the depth" in detail and "depthc describe" in detail
        assert "locked-off" in detail  # the caveat itself, not just the pointer
        assert "this is a" not in detail


def test_no_line_states_the_whole_frame_share_as_a_camera_fact():
    """One rule (motion.camera_share): the share is "whole-frame" and a class drawn from it
    "reads as" one. Each surface is rendered and held to the same predicate, and the old
    wording of each is run through it to show it can fail."""
    import numpy as np

    from depthc import compare as comparelib, describe as describelib, motion as motionlib

    def states_a_fact(text: str) -> bool:
        return any(bad in text for bad in ("this is a", "(a camera-path shot)", "is a body ref",
                                           "(camera)", "CAMERA   ref", "the rest is the subject"))

    for old in ("80% of the movement is whole-frame — this is a camera-path reference.",
                "Camera: pans left 0–3s (a camera-path shot).",
                "not judged: ref is a body ref (3% whole-frame)",
                "80% whole-frame (camera)", "  CAMERA   ref 80% whole-frame"):
        assert states_a_fact(old)

    base = np.random.default_rng(7).random((90, 160)) * 255
    pan = np.stack([np.roll(base, 2 * i, axis=1) for i in range(48)]).astype(np.uint8)
    measured = motionlib.measure(pan, 24.0)
    rendered = [
        f.detail for share in (0.8, 0.2)
        for f in check(**GOOD, motion=profile(4.0, camera=share)).findings
        if f.check == "motion type"
    ] + list(describelib.describe(measured).sentences)
    # A square crossing a still wall: a low share, so compare takes its "not judged" branch.
    body = np.repeat(base[None], 48, axis=0).astype(np.uint8)
    for i in range(48):
        body[i, 30:60, 10 + 2 * i:40 + 2 * i] = 255
    walked = motionlib.measure(body, 24.0)
    for prof in (measured, walked):
        comp = comparelib.compare_profiles(prof, prof)
        rendered += comparelib.render(comp, "kit", "take", colour=False).splitlines()
    # The kit README's share line, and describe on the low-share clip (its other branch).
    from pathlib import Path

    from depthc import kit as kitlib
    report = check(**GOOD, motion=profile(4.0, camera=0.8))
    manifest = kitlib.build_manifest(
        kit_name="Share_001", source_path=Path("/tmp/x.mp4"), source_sha256="a" * 64,
        source_duration=8.0, source_fps=24.0, source_size=(1920, 1080), trim=(1.0, 6.0),
        crop=None, depth_duration=5.0, depth_fps=24.0, depth_size=(1280, 536),
        depth_bytes=800_000, depth_sha256="b" * 64, binding_sha256="c" * 64,
        frame_count=120, still_index=0, still_has_real_faces=False,
        tags=[], description="", settings={}, report=report, device="cpu", elapsed=1.0,
    )
    manifest["motion"] = {"sparkline": "▁▂▃", "energy": 4.0, "cameraShare": 0.8}
    readme = kitlib.render_kit_readme(manifest, report)
    assert "80% whole-frame in the depth" in readme
    rendered += readme.splitlines() + list(describelib.describe(walked).sentences)
    assert any("SHARE    ref" in line for line in rendered)
    assert any("not judged: ref reads as a body ref" in line for line in rendered)
    assert any("reads as a camera-path shot" in line for line in rendered)
    assert not [line for line in rendered if states_a_fact(line)]


def test_motion_type_names_a_body_reference():
    report = check(**GOOD, motion=profile(4.0, camera=0.2))
    assert "subject-motion" in next(
        f.detail for f in report.findings if f.check == "motion type"
    )


def test_motion_findings_never_lack_a_remedy():
    report = check(**GOOD, motion=profile(0.05, head=0.0, spike=True), retention=0.05)
    for finding in report.findings:
        if finding.tier is not Tier.PASS:
            assert finding.remedy, f"{finding.check} has no remedy"


# ------------------------------------------------------------------- the output grid

@pytest.mark.parametrize("seconds,expected", [
    (4.0, True), (5.0, True), (10.0, True), (5.02, True), (15.0, True),
    (3.917, False), (5.5, False), (3.0, False), (6.4, False),
    (16.0, False),  # over the output ceiling: whole seconds alone don't make it requestable
])
def test_on_output_grid(seconds, expected):
    assert on_output_grid(seconds, "seedance-2.0") is expected


def test_on_output_grid_follows_the_default_models_ceiling():
    assert on_output_grid(16.0) is True     # legal on 2.5, the default
    assert on_output_grid(31.0) is False    # over 2.5's 30s ceiling


def test_a_ref_shorter_than_the_output_floor_warns():
    """CLIP A shipped at 3.92s against a 4s output floor."""
    report = check(**{**GOOD, "duration": 3.917})
    assert tier_of("output coverage", report) is Tier.WARN
    assert "uncovered" in next(
        f.detail for f in report.findings if f.check == "output coverage"
    )


def test_a_fractional_ref_length_warns():
    assert tier_of("output coverage", check(**{**GOOD, "duration": 6.4})) is Tier.WARN


def test_a_whole_second_ref_passes_coverage():
    assert tier_of("output coverage", check(**{**GOOD, "duration": 7.0})) is Tier.PASS


def test_snap_holds_the_in_point_and_moves_the_out_point():
    (start, end), note = snap_trim(1.0, 4.4, 20.0)
    assert (start, end) == (1.0, 5.0)
    assert "out-point" in note


def test_snap_rounds_down_as_readily_as_up():
    """6.4s rounds to 6, not up to 7: the nearest requestable length wins, not the longest."""
    (start, end), _ = snap_trim(2.0, 8.4, 20.0)
    assert (start, end) == (2.0, 8.0)


def test_snap_pulls_the_in_point_back_when_the_source_runs_out():
    """The real CLIP A case: 4.0-7.9s of a 7.944s source cannot grow forwards."""
    (start, end), note = snap_trim(4.0, 7.9, 7.944)
    assert end == pytest.approx(7.944)
    assert end - start == pytest.approx(4.0)
    assert "window" in note


def test_snap_leaves_an_already_snapped_trim_alone():
    assert snap_trim(1.0, 6.0, 20.0) == ((1.0, 6.0), None)


def test_snap_gives_up_rather_than_fabricate_footage():
    """A 3s source has no legal 4s window; the duration check owns that failure."""
    assert snap_trim(0.0, 3.0, 3.0) == ((0.0, 3.0), None)


def test_snap_never_exceeds_the_output_ceiling():
    (start, end), _ = snap_trim(0.0, 19.4, 30.0, "seedance-2.0")
    assert end - start == pytest.approx(15.0)
    (start, end), _ = snap_trim(0.0, 34.4, 40.0)   # default: 2.5's 30s ceiling
    assert end - start == pytest.approx(30.0)


def test_snapped_lengths_always_land_on_the_grid():
    for span in (2.3, 3.9, 4.4, 5.6, 7.1, 9.9):
        (start, end), _ = snap_trim(1.0, 1.0 + span, 30.0)
        assert on_output_grid(round(end - start, 3)), span


def test_format_report_is_plain_text_without_colour():
    text = format_report(check(**GOOD), colour=False)
    assert "\033[" not in text
    assert text.strip().endswith("PASS")


def test_format_report_shows_remedies_for_problems():
    text = format_report(check(**{**GOOD, "duration": 12.0}), colour=False)
    assert "→" in text
    assert "WARN" in text


# ---------------------------------------------------------------- per-model envelopes
# The drift these exist to catch: until 2026-08-24 the table held ONE envelope (2.0's) and
# graded every kit against it, so a 2.5-legal 20s reference was refused before it reached the
# GPU. Each test below pairs the 2.0 refusal with the 2.5 acceptance, so neither can silently
# become the other.


@pytest.mark.parametrize("seconds", [20.0, 30.0])
def test_long_refs_fail_on_2_0_and_pass_on_2_5(seconds):
    """The headline drift: 2.0 caps a video ref at 15s, 2.5 at 30s."""
    assert tier_of("duration", check(**{**GOOD, "duration": seconds},
                                     dialect="seedance-2.0")) is Tier.FAIL
    assert tier_of("duration", check(**{**GOOD, "duration": seconds},
                                     dialect="seedance-2.5")) is not Tier.FAIL


def test_the_2_0_refusal_names_the_model_and_the_way_out():
    """A number in a refusal is meaningless without the model it belongs to."""
    report = check(**{**GOOD, "duration": 20.0}, dialect="seedance-2.0")
    detail = next(f.detail for f in report.findings if f.check == "duration")
    remedy = next(f.remedy for f in report.findings if f.check == "duration")
    assert "seedance-2.0" in detail
    assert "seedance-2.5" in remedy and "30s" in remedy


def test_output_window_is_per_model():
    assert checks.suggested_output_seconds(20.0, "seedance-2.0") == 15.0
    assert checks.suggested_output_seconds(20.0, "seedance-2.5") == 20.0
    assert checks.on_output_grid(20.0, "seedance-2.5") is True
    assert checks.on_output_grid(20.0, "seedance-2.0") is False


def test_snap_trim_is_per_model():
    """A 19.4s window snaps to 15s on 2.0 and to 19s on 2.5 — the ceiling differs."""
    (s20, e20), _ = checks.snap_trim(0.0, 19.4, 30.0, "seedance-2.0")
    (s25, e25), _ = checks.snap_trim(0.0, 19.4, 30.0, "seedance-2.5")
    assert round(e20 - s20, 3) == 15.0
    assert round(e25 - s25, 3) == 19.0


def test_the_default_dialect_is_2_5_osides_live_line():
    """[CHANGED 2026-09-06, audit DC-5] The default WAS 2.0 "so nothing silently changed
    model" on 2026-08-24. But 2.5 is OSIDE's live line and the only kit on disk is 2.5, so
    the inbox hint — which named no dialect — built a kit graded by 2.0 law. A kit built
    without --dialect now grades as a 2.5 kit; 2.0 is one flag away and unchanged.
    """
    assert checks.DEFAULT_DIALECT == "seedance-2.5"
    explicit = check(**{**GOOD, "duration": 20.0}, dialect="seedance-2.5").as_dict()
    implicit = check(**{**GOOD, "duration": 20.0}).as_dict()
    assert explicit == implicit
    assert implicit["tier"] != "FAIL"
    assert check(**{**GOOD, "duration": 20.0}, dialect="seedance-2.0").tier is Tier.FAIL
    # the module-level names derive from the default record, so they moved with it
    assert checks.MAX_REF_SECONDS == 30.0
    assert checks.OUTPUT_SECONDS == (4.0, 30.0)


def test_an_unknown_dialect_names_the_alternatives_instead_of_defaulting():
    """Silently falling back to 2.0 would grade a 2.6 kit against the wrong envelope."""
    with pytest.raises(ValueError, match="seedance-2.5"):
        checks.dialect_for("seedance-9.9")


def test_2_5_has_no_4k_rung_today_and_2_0_does():
    """Dated observation, not a forecast — see the note on the record.

    Guards the fact behind the kit prose: 'master at 4K' is followable on 2.0 and names a
    tier 2.5 does not currently offer.
    """
    assert "4k" in checks.DIALECTS["seedance-2.0"].output_resolutions
    assert "4k" not in checks.DIALECTS["seedance-2.5"].output_resolutions
    assert "1080p" in checks.DIALECTS["seedance-2.5"].output_resolutions


def test_no_dialect_carries_the_higgsfield_platform_cap():
    """12 was a Higgsfield PLATFORM cap that leaked into this direct-API validator once.

    Ark's own table: 2.0 = 15 (9+3+3), 2.5 = 50 (30+10+10). If either reads 12 again, the
    same foreign number has been re-imported.
    """
    assert checks.DIALECTS["seedance-2.0"].max_mixed_inputs == 15
    assert checks.DIALECTS["seedance-2.5"].max_mixed_inputs == 50
    assert all(d.max_mixed_inputs != 12 for d in checks.DIALECTS.values())
