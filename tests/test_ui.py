"""UI logic tests.

Only the pure helpers are covered here — the browser run-through is Gate 3's manual check.
What matters most is that no input path returns an exception instead of a message: a UI
that dies on a bad file is worse than one that says "that isn't a video".
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from depthc import checks, ui
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
        (0, 20, "adherence degrades"),   # legal on 2.5, the default since 2026-09-06
        (0, 35, "ceiling"),
        (5, 5, "empty range"),
        (9, 4, "empty range"),
    ],
)
def test_trim_feedback_matches_the_checker_tiers(start, end, expected):
    assert expected in ui._trim_feedback(start, end)


def test_the_ui_reads_its_bounds_from_the_dialect_record_not_a_literal():
    """The UI had seven bare `15`s, so correcting checks.py would have left it refusing
    references the CLI accepts, quoting the stale number as fact."""
    from depthc import checks

    d = checks.dialect_for(checks.DEFAULT_DIALECT)
    assert ui._UI_MAX_REF == d.ref_seconds[1]
    assert f"{d.ref_seconds[1]:g}s ceiling" in ui._trim_feedback(0, d.ref_seconds[1] + 5)
    assert d.name in ui._trim_feedback(0, d.ref_seconds[1] + 5)


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
        None, "", "", checks.DEFAULT_DIALECT, "720p", "auto", "conform-24", 0, 5,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "Drop a clip" in status


def test_the_ui_default_is_2_5_and_its_bounds_follow(tmp_path):
    """[2026-09-06, audit DC-5] The UI's probe-time bounds and the model radio both read
    the default record; the default is 2.5, so a 20s upload is no longer refused up front."""
    assert checks.DEFAULT_DIALECT == "seedance-2.5"
    assert ui._UI_DIALECT.name == "seedance-2.5"
    assert ui._UI_MAX_REF == 30.0
    assert "seedance-2.5" in ui._trim_feedback(0, 35)


def test_convert_rejects_an_over_long_clip(clip_long):
    """>15s on 2.0 must be refused with an explanation, before any inference is attempted.

    2.0 is passed explicitly — on the default (2.5) a 20s clip is legal.
    """
    *_, status = ui._convert(
        str(clip_long), "", "", "seedance-2.0", "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "ceiling" in status and "seedance-2.0" in status


@pytest.mark.slow
def test_convert_clamps_an_out_point_past_the_end(clip_3s, tmp_path):
    """Asking for 0-20s of a 3s clip is a clamp, not an error.

    Marked slow: this is the one UI test that runs a real conversion. It also passes an
    explicit library path — an empty one resolves to the user's real Motion Library, which
    is how this suite silently wrote kits into it.
    """
    depth, *_, status = ui._convert(
        str(clip_3s), "", "", checks.DEFAULT_DIALECT, "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, str(tmp_path / "lib"),
    )
    assert depth and Path(depth).is_file(), status  # the run itself, not just the absence of a word
    assert "ceiling" not in status


def test_convert_on_a_non_video_returns_a_message(not_a_video):
    *_, status = ui._convert(
        str(not_a_video), "", "", checks.DEFAULT_DIALECT, "720p", "auto", "conform-24", 0, 5,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "is not a video file" in status
    assert "Traceback" not in status


def test_convert_on_an_empty_range_returns_a_message(clip_3s):
    *_, status = ui._convert(
        str(clip_3s), "", "", checks.DEFAULT_DIALECT, "720p", "auto", "conform-24", 2.0, 2.0,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "Empty trim range" in status


def test_the_model_picker_changes_which_clips_are_refused(clip_long, monkeypatch):
    """The picker must move the LIMITS, not just the prompt template.

    A 20s reference is over 2.0's 15s ceiling and inside 2.5's 30s one. Before the picker
    existed the UI graded every clip against 2.0 and had seven hardcoded 15s literals, so
    this clip was unbuildable here regardless of the target model.
    """
    *_, on_20 = ui._convert(
        str(clip_long), "", "", "seedance-2.0", "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "ceiling" in on_20 and "seedance-2.0" in on_20
    # ...and it points at the model that would accept it rather than just refusing.
    assert "seedance-2.5 accepts up to 30s" in on_20

    # Stops at the model rather than running it. Without the marker, any failure on the way
    # there, a missing checkout included, would also have read as "accepted".
    class ReachedTheModel(Exception):
        pass

    def stop_at_the_model(*_args, **_kwargs):
        raise ReachedTheModel

    monkeypatch.setattr(ui.engine, "infer", stop_at_the_model)
    *_, on_25 = ui._convert(
        str(clip_long), "", "", "seedance-2.5", "720p", "auto", "conform-24", 0, 20,
        False, True, "auto", 518, "fp32", False, False, "",
    )
    assert "ceiling" not in on_25
    assert "ReachedTheModel" in on_25


def test_a_pass_that_does_not_fit_reads_as_plain_words_in_the_ui(clip_long, monkeypatch, tmp_path):
    """The UI's catch-all would print a traceback here if this failure were not its own case."""
    from depthc import memory

    monkeypatch.setenv(memory.OVERRIDE_ENV, "0.01")
    *_, status = ui._convert(
        str(clip_long), "memtest", "", "seedance-2.5", "720p", "auto", "conform-24", 0, 5,
        False, True, "auto", 518, "fp32", False, False, str(tmp_path),
    )
    assert "Not enough free memory" in status and "Close other apps" in status
    assert "Traceback" not in status and "Unexpected error" not in status
    assert not [p for p in tmp_path.iterdir() if p.is_dir()], "a refused pass left a kit folder behind"


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


# ------------------------------------------------------------------ a library kept elsewhere

def _outside_every_folder_gradio_allows(tmp_path, monkeypatch) -> Path:
    """A file outside both the working folder and the temp folder, with both moved aside."""
    (tmp_path / "cwd").mkdir()
    (tmp_path / "temp").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))
    clip = tmp_path / "elsewhere" / "depth.mp4"
    clip.write_bytes(b"depth clip")
    return clip


def test_a_clip_outside_the_allowed_folders_is_copied_where_gradio_can_serve_it(tmp_path, monkeypatch):
    clip = _outside_every_folder_gradio_allows(tmp_path, monkeypatch)
    served = Path(ui._servable(clip))
    assert served.is_relative_to(tmp_path / "temp")
    assert served.read_bytes() == b"depth clip"
    assert clip.is_file()


def test_a_clip_gradio_can_already_serve_is_not_copied(tmp_path, monkeypatch):
    _outside_every_folder_gradio_allows(tmp_path, monkeypatch)
    for folder in ("cwd", "temp"):
        clip = tmp_path / folder / "depth.mp4"
        clip.write_bytes(b"x")
        assert ui._servable(clip) == str(clip)


def test_picking_a_kit_from_a_library_kept_elsewhere_serves_a_copy(tmp_path, monkeypatch):
    clip = _outside_every_folder_gradio_allows(tmp_path, monkeypatch)
    rows = [["name", "1s", "pass", "", "", str(clip.parent)]]
    depth, _prompt, path = ui._library_pick(rows, SimpleNamespace(index=[0, 0]))
    assert Path(depth).is_relative_to(tmp_path / "temp")
    assert path == str(clip.parent)  # the kit is still where the library says it is


def test_picking_a_kit_whose_prompt_is_not_text_still_loads_it(tmp_path, monkeypatch):
    clip = _outside_every_folder_gradio_allows(tmp_path, monkeypatch)
    (clip.parent / "prompt.txt").write_bytes(b"\xff\xfe not text")
    rows = [["name", "1s", "pass", "", "", str(clip.parent)]]
    _depth, prompt, path = ui._library_pick(rows, SimpleNamespace(index=[0, 0]))
    assert "not text" in prompt
    assert path == str(clip.parent)


def test_picking_a_kit_whose_files_are_folders_still_loads_it(tmp_path, monkeypatch):
    clip = _outside_every_folder_gradio_allows(tmp_path, monkeypatch)
    clip.unlink()
    clip.mkdir()
    (clip.parent / "prompt.txt").mkdir()
    rows = [["name", "1s", "pass", "", "", str(clip.parent)]]
    assert ui._library_pick(rows, SimpleNamespace(index=[0, 0])) == (None, "", str(clip.parent))


def test_the_finished_conversion_goes_through_the_same_door():
    source = Path(ui.__file__).read_text(encoding="utf-8")
    assert "_servable(paths.depth)" in source and "_servable(paths.preview)" in source
