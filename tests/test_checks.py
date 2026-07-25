from __future__ import annotations

import pytest

from depthc.checks import Tier, check, format_report, suggested_output_seconds

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
def test_over_fifteen_seconds_fails(seconds):
    assert tier_of("duration", check(**{**GOOD, "duration": seconds})) is Tier.FAIL


def test_oversize_file_fails_at_the_stricter_limit():
    """Sources disagree (50 vs 200 MB); the checker enforces the stricter one."""
    report = check(**{**GOOD, "size_bytes": 60_000_000})
    assert tier_of("file size", report) is Tier.FAIL
    assert "200" in next(f.detail for f in report.findings if f.check == "file size")


def test_file_just_under_the_limit_passes():
    assert tier_of("file size", check(**{**GOOD, "size_bytes": 49_000_000})) is Tier.PASS


def test_too_few_pixels_fails():
    report = check(**{**GOOD, "width": 320, "height": 240})
    assert tier_of("resolution", report) is Tier.FAIL


def test_too_many_pixels_fails():
    report = check(**{**GOOD, "width": 3840, "height": 2160})
    assert tier_of("resolution", report) is Tier.FAIL


def test_letterboxed_720p_still_passes_the_pixel_window():
    """A cropped 2.39:1 kit is 1280x536 — comfortably inside the window."""
    assert tier_of("resolution", check(**{**GOOD, "width": 1280, "height": 536})) is Tier.PASS


def test_non_24_fps_warns_but_ships():
    report = check(**{**GOOD, "fps": 30.0})
    assert tier_of("frame rate", report) is Tier.WARN
    assert report.ok


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
    report = check(**{**GOOD, "duration": 9.0, "size_bytes": 99_000_000})
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
    [(2.0, 4.0), (3.0, 4.0), (5.0, 5.0), (8.0, 8.0), (15.0, 15.0), (20.0, 15.0)],
)
def test_suggested_output_clamps_to_the_output_window(ref, expected):
    """Seedance's output floor is 4s, so short refs necessarily drive a longer generation."""
    assert suggested_output_seconds(ref) == expected


def test_format_report_is_plain_text_without_colour():
    text = format_report(check(**GOOD), colour=False)
    assert "\033[" not in text
    assert text.strip().endswith("PASS")


def test_format_report_shows_remedies_for_problems():
    text = format_report(check(**{**GOOD, "duration": 12.0}), colour=False)
    assert "→" in text
    assert "WARN" in text
