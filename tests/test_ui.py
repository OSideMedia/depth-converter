"""UI logic tests.

Only the pure helpers are covered here — the browser run-through is Gate 3's manual check.
What matters most is that no input path returns an exception instead of a message: a UI
that dies on a bad file is worse than one that says "that isn't a video".
"""

from __future__ import annotations

import pytest

from depthc import ui
from depthc.checks import Tier, check


def test_build_constructs_without_error():
    assert len(ui.build().blocks) > 0


def test_default_port_avoids_the_oside_dev_server():
    assert ui.DEFAULT_PORT != 3000
    assert ui.DEFAULT_PORT == 7861


# ------------------------------------------------------------------- trim feedback

@pytest.mark.parametrize(
    "start,end,expected",
    [
        (0, 5, "sweet spot"),
        (0, 3, "sweet spot"),
        (0, 8, "sweet spot"),
        (0, 2.5, "single beat"),
        (0, 1.5, "under the 2s minimum"),
        (0, 12, "adherence degrades"),
        (0, 20, "over the 15s ceiling"),
        (5, 5, "empty range"),
        (9, 4, "empty range"),
    ],
)
def test_trim_feedback_matches_the_checker_tiers(start, end, expected):
    assert expected in ui._trim_feedback(start, end)


def test_trim_feedback_reports_the_span():
    assert "5.00s" in ui._trim_feedback(1.0, 6.0)


# ------------------------------------------------------------------- probe readout

def test_probe_readout_on_empty_input_is_blank_not_an_error():
    html, start, end = ui._probe_readout(None)
    assert html == ""


def test_probe_readout_reports_facts(clip_3s):
    html, start, end = ui._probe_readout(str(clip_3s))
    assert "moving_box.mp4" in html
    assert "320×240" in html
    assert start["value"] == 0.0
    assert end["value"] == pytest.approx(3.0, abs=0.2)


def test_probe_readout_on_a_non_video_explains_itself(not_a_video):
    """A wrong file must produce a message, never a traceback."""
    html, _, _ = ui._probe_readout(str(not_a_video))
    assert "is not a video file" in html
    assert "Invalid data found" not in html  # no raw ffprobe output at the user


def test_probe_readout_flags_a_matte(clip_letterboxed):
    html, _, _ = ui._probe_readout(str(clip_letterboxed))
    assert "matte detected" in html


def test_probe_readout_sets_slider_bounds_to_the_clip(clip_3s):
    _, start, end = ui._probe_readout(str(clip_3s))
    assert end["maximum"] == pytest.approx(3.0, abs=0.2)


# ------------------------------------------------------------------- report rendering

def test_report_html_tags_each_tier():
    report = check(duration=1.0, width=1280, height=720, fps=24.0,
                   size_bytes=1_000_000, still_has_real_faces=True)
    html = ui._report_html(report)
    assert 'class="depthc-tier FAIL"' in html
    assert 'class="fail"' in html
    assert 'class="warn"' in html
    assert 'class="pass"' in html


def test_report_html_includes_remedies():
    report = check(duration=12.0, width=1280, height=720, fps=24.0,
                   size_bytes=1_000_000, still_has_real_faces=False)
    assert "remedy" in ui._report_html(report)
    assert "→" in ui._report_html(report)


def test_report_html_clean_pass_has_no_remedy_rows():
    report = check(duration=5.0, width=1280, height=720, fps=24.0,
                   size_bytes=1_000_000, still_has_real_faces=False)
    assert report.tier is Tier.PASS
    assert "remedy" not in ui._report_html(report)


# ------------------------------------------------------------------- guard rails

def test_convert_without_a_video_returns_a_message():
    *_, status = ui._convert(
        None, "", "", "720p", "auto", "conform-24", 0, 5,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "Drop a clip" in status


def test_convert_rejects_an_over_long_clip(clip_long):
    """>15s must be refused with an explanation, before any inference is attempted."""
    *_, status = ui._convert(
        str(clip_long), "", "", "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "15s ceiling" in status


@pytest.mark.slow
def test_convert_clamps_an_out_point_past_the_end(clip_3s, tmp_path):
    """Asking for 0-20s of a 3s clip is a clamp, not an error.

    Marked slow: this is the one UI test that runs a real conversion. It also passes an
    explicit library path — an empty one resolves to the user's real Motion Library, which
    is how this suite silently wrote kits into it.
    """
    *_, status = ui._convert(
        str(clip_3s), "", "", "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, str(tmp_path / "lib"),
    )
    assert "ceiling" not in status


def test_convert_on_a_non_video_returns_a_message(not_a_video):
    *_, status = ui._convert(
        str(not_a_video), "", "", "720p", "auto", "conform-24", 0, 5,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "is not a video file" in status
    assert "Traceback" not in status


def test_convert_on_an_empty_range_returns_a_message(clip_3s):
    *_, status = ui._convert(
        str(clip_3s), "", "", "720p", "auto", "conform-24", 2.0, 2.0,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "Empty trim range" in status


def test_reveal_on_a_missing_path_does_not_raise(tmp_path):
    assert "Not found" in ui._reveal(tmp_path / "ghost")


# ------------------------------------------------------------------- library tab

def test_library_rows_on_an_empty_library(tmp_path):
    assert ui._library_rows("", str(tmp_path)) == []


def test_library_rows_lists_and_filters(tmp_path):
    import json

    for name, tag in (("A_001", "chase"), ("B_001", "fight")):
        kit = tmp_path / name
        kit.mkdir()
        (kit / "kit.json").write_text(json.dumps({
            "kitName": name, "tags": [tag], "createdAt": "2026-07-24T00:00:00+00:00",
            "depth": {"durationSeconds": 5.0}, "checks": {"tier": "PASS"},
        }))

    assert len(ui._library_rows("", str(tmp_path))) == 2
    rows = ui._library_rows("chase", str(tmp_path))
    assert len(rows) == 1 and rows[0][0] == "A_001"


def test_library_row_shape_matches_the_table_headers(tmp_path):
    import json

    kit = tmp_path / "A_001"
    kit.mkdir()
    (kit / "kit.json").write_text(json.dumps({
        "kitName": "A_001", "tags": ["x"], "createdAt": "2026-07-24T00:00:00+00:00",
        "depth": {"durationSeconds": 4.0}, "checks": {"tier": "WARN"},
    }))
    row = ui._library_rows("", str(tmp_path))[0]
    assert len(row) == 6  # kit, duration, tier, tags, created, path
    assert row[2] == "WARN"
    assert row[5].endswith("A_001")
