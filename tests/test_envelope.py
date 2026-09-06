"""The exported Seedance envelope: OSIDE writes it, this file reads it (audit E-3).

Two hand-maintained copies of one vendor table, in two languages, with no exporter and no
drift gate, agreed on every number on 2026-09-06 — and had failed twice before that. Now
OSIDE exports its live envelope and `checks.DIALECTS` is built from the payload; the hand
table in checks.py is the fallback and is held equal to the payload here. The test that
matters is the parity one: it goes RED when either side moves.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from depthc import checks


def _payload() -> dict:
    return json.loads(checks.ENVELOPE_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "seedance-envelope.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ (a) parity

def test_the_committed_envelope_is_the_one_oside_exports():
    payload = _payload()
    assert payload["format"] == checks.ENVELOPE_FORMAT
    assert set(payload["dialects"]) == {"seedance-2.0", "seedance-2.5"}


def test_hand_table_matches_the_envelope_on_every_overlapping_field():
    """The fallback must be faithful, or an absent file silently changes the grading.

    The assertion carries its own counterexample: the same predicate must REPORT a hand
    table with one number moved, so an empty drift list is a finding and not a no-op.
    """
    payload = _payload()
    assert checks.envelope_drift(checks.HAND_DIALECTS, payload) == []

    wrong = dict(checks.HAND_DIALECTS)
    wrong["seedance-2.5"] = dataclasses.replace(wrong["seedance-2.5"], max_video_refs=3)
    drift = checks.envelope_drift(wrong, payload)
    assert drift == ["seedance-2.5.max_video_refs: hand 3 != envelope 10"]


def test_dialects_in_use_carry_the_envelope_numbers():
    """What the checker grades with came through the loader, not the literal."""
    payload = _payload()
    for name, record in payload["dialects"].items():
        d = checks.DIALECTS[name]
        assert d.ref_seconds == tuple(record["refSeconds"])
        assert d.ref_total_seconds == record["refTotalSeconds"]
        assert d.max_video_refs == record["maxVideoRefs"]
        assert d.max_mixed_inputs == record["maxMixedInputs"]
        assert d.output_seconds == tuple(record["outputSeconds"])
        assert d.max_file_mb == record["maxFileMb"]
        assert d.output_resolutions == tuple(record["outputResolutions"])
        assert d.refs_with_start_frame is record["refsWithStartFrame"]
        # the fields OSIDE does not enforce stay ours
        assert d.pixels == checks.HAND_DIALECTS[name].pixels
        assert d.sweet_spot == checks.HAND_DIALECTS[name].sweet_spot


def test_an_envelope_number_overrides_the_hand_table(tmp_path):
    """The payload wins on an overlapping field; the hand pixel window survives."""
    payload = _payload()
    payload["dialects"]["seedance-2.5"]["maxVideoRefs"] = 12
    table = checks.load_envelope(_write(tmp_path, payload), warn=lambda _m: None)
    assert table["seedance-2.5"].max_video_refs == 12
    assert table["seedance-2.5"].pixels == checks.HAND_DIALECTS["seedance-2.5"].pixels
    assert table["seedance-2.0"] == checks.HAND_DIALECTS["seedance-2.0"]


# ------------------------------------------------------------------ (b) broken shape

def test_a_missing_key_refuses_loudly_naming_it(tmp_path):
    payload = _payload()
    del payload["dialects"]["seedance-2.5"]["maxVideoRefs"]
    with pytest.raises(checks.EnvelopeError, match=r"dialects\.seedance-2\.5\.maxVideoRefs"):
        checks.load_envelope(_write(tmp_path, payload), warn=lambda _m: None)


def test_a_non_numeric_value_refuses_loudly_naming_the_key(tmp_path):
    payload = _payload()
    payload["dialects"]["seedance-2.0"]["refTotalSeconds"] = "15"
    with pytest.raises(checks.EnvelopeError, match=r"seedance-2\.0\.refTotalSeconds"):
        checks.load_envelope(_write(tmp_path, payload), warn=lambda _m: None)


def test_a_boolean_where_a_number_belongs_is_broken_not_one(tmp_path):
    """True == 1 in Python; a typed reader must not let it through as a count."""
    payload = _payload()
    payload["dialects"]["seedance-2.0"]["maxVideoRefs"] = True
    with pytest.raises(checks.EnvelopeError, match="maxVideoRefs"):
        checks.load_envelope(_write(tmp_path, payload), warn=lambda _m: None)


def test_a_foreign_format_string_refuses(tmp_path):
    payload = _payload()
    payload["format"] = "oside-seedance-envelope/2"
    with pytest.raises(checks.EnvelopeError, match="oside-seedance-envelope/2"):
        checks.load_envelope(_write(tmp_path, payload), warn=lambda _m: None)


def test_unreadable_json_is_broken_not_absent(tmp_path):
    path = tmp_path / "seedance-envelope.json"
    path.write_text("{not json", encoding="utf-8")
    notices: list[str] = []
    with pytest.raises(checks.EnvelopeError, match="JSON"):
        checks.load_envelope(path, warn=notices.append)
    assert notices == []  # no silent fall back


# ------------------------------------------------------------------ (c) absent

def test_an_absent_envelope_falls_back_to_the_hand_table_with_a_notice(tmp_path):
    notices: list[str] = []
    table = checks.load_envelope(tmp_path / "nope.json", warn=notices.append)
    assert table == checks.HAND_DIALECTS
    assert len(notices) == 1 and "hand table" in notices[0]


# ------------------------------------------------------------------ (d) staleness

def test_a_stale_export_warns_with_the_age_and_the_refresh_command(tmp_path):
    payload = _payload()
    old = datetime.now(timezone.utc) - timedelta(days=payload["staleAfterDays"] + 10)
    payload["exportedAt"] = old.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    path = _write(tmp_path, payload)
    notices: list[str] = []
    table = checks.load_envelope(path, warn=notices.append)
    assert table["seedance-2.5"].max_video_refs == 10  # still used
    assert len(notices) == 1
    assert f"{payload['staleAfterDays'] + 10} days old" in notices[0]
    assert "npm run export:seedance-envelope" in notices[0]
    assert str(path) in notices[0]


def test_a_fresh_export_does_not_warn(tmp_path):
    payload = _payload()
    payload["exportedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    notices: list[str] = []
    checks.load_envelope(_write(tmp_path, payload), warn=notices.append)
    assert notices == []


def test_the_committed_export_is_within_its_own_window():
    """If this goes red the fix is in OSIDE: re-run the exporter and commit the file."""
    payload = _payload()
    assert checks.envelope_age_days(payload) <= payload["staleAfterDays"]


# ------------------------------------------------------------------ unknown dialects

def test_a_dialect_this_tool_has_no_template_for_is_left_out_with_a_notice(tmp_path):
    payload = _payload()
    payload["dialects"]["seedance-3.0"] = dict(payload["dialects"]["seedance-2.5"])
    notices: list[str] = []
    table = checks.load_envelope(_write(tmp_path, payload), warn=notices.append)
    assert "seedance-3.0" not in table
    assert any("seedance-3.0" in n for n in notices)
