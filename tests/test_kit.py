from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from depthc import kit as kitlib
from depthc.checks import check

SCHEMA_PATH = Path(__file__).parent.parent / "depthc" / "kit_schema.json"


# ------------------------------------------------------------------- naming & library

@pytest.mark.parametrize(
    "text,expected",
    [
        ("sword fight", "Sword-fight"),
        ("Tracking orbit", "Tracking-orbit"),
        ("  spaced  out  ", "Spaced-out"),
        ("punct!u@ation#", "Punctuation"),
        ("under_score", "Under-score"),
        ("dash--dash", "Dash-dash"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_slugify(text, expected):
    assert kitlib.slugify(text) == expected


def test_kit_folder_name_joins_tags():
    assert kitlib.kit_folder_name(["sword fight", "tracking orbit"], 1) == \
        "Sword-fight_Tracking-orbit_001"


def test_kit_folder_name_falls_back_when_untagged():
    assert kitlib.kit_folder_name([], 7, fallback="CLIP A") == "CLIP-A_007"


def test_kit_folder_name_survives_unusable_tags():
    assert kitlib.kit_folder_name(["!!!", "  "], 2, fallback="") == "Motion_002"


def test_allocate_kit_dir_increments(tmp_path):
    first = kitlib.allocate_kit_dir(tmp_path, ["sword fight"])
    second = kitlib.allocate_kit_dir(tmp_path, ["sword fight"])
    assert first.name == "Sword-fight_001"
    assert second.name == "Sword-fight_002"
    assert first.is_dir() and second.is_dir()


def test_allocate_kit_dir_creates_immediately(tmp_path):
    """Allocating without creating would let two runs in the same second collide."""
    allocated = kitlib.allocate_kit_dir(tmp_path, ["x"])
    assert allocated.exists()


def test_library_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPTHC_LIBRARY", str(tmp_path / "from_env"))
    assert kitlib.library_root() == tmp_path / "from_env"
    assert kitlib.library_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("DEPTHC_LIBRARY")
    assert kitlib.library_root() == kitlib.DEFAULT_LIBRARY


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(b"depth")
    assert kitlib.sha256_file(path) == hashlib.sha256(b"depth").hexdigest()


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert kitlib.sha256_file(a) == kitlib.sha256_file(b)
    b.write_bytes(b"different")
    assert kitlib.sha256_file(a) != kitlib.sha256_file(b)
    assert len(kitlib.sha256_file(a)) == 64


# ----------------------------------------------------------------------- still picking

def test_sharpest_frame_skips_a_blurred_first_frame():
    """Frame 0 is a bad default — a trim in-point often lands mid-whip-pan."""
    rng = np.random.default_rng(0)
    sharp = (rng.integers(0, 255, (64, 64, 3))).astype(np.uint8)
    blurred = np.full((64, 64, 3), 128, dtype=np.uint8)  # flat = no edges = low variance

    frames = np.stack([blurred, blurred, sharp, blurred])
    assert kitlib.sharpest_frame_index(frames) == 2


def test_sharpest_frame_only_searches_the_window():
    rng = np.random.default_rng(1)
    flat = np.full((32, 32, 3), 100, dtype=np.uint8)
    sharp = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
    frames = np.stack([flat] * 5 + [sharp])
    assert kitlib.sharpest_frame_index(frames, search=3) == 0  # sharp frame is out of range


def test_write_still_writes_a_png(tmp_path):
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 255, (4, 32, 32, 3)).astype(np.uint8)
    out = tmp_path / "still.png"
    index = kitlib.write_still(frames, out)
    assert out.is_file() and out.stat().st_size > 0
    assert 0 <= index < 4


def test_write_still_honours_explicit_index(tmp_path):
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    assert kitlib.write_still(frames, tmp_path / "s.png", index=3) == 3


def test_write_still_rejects_out_of_range(tmp_path):
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        kitlib.write_still(frames, tmp_path / "s.png", index=9)


# --------------------------------------------------------------------------- prompt.txt

def _render(**overrides):
    kwargs = dict(
        kit_name="Sword-fight_001", duration=5.0, width=1280, height=536,
        fps=24.0, still_has_real_faces=False,
    )
    kwargs.update(overrides)
    return kitlib.render_prompt(**kwargs)


def test_prompt_renders_both_variants():
    text = _render()
    assert "VARIANT A" in text and "VARIANT B" in text


def test_prompt_variant_a_is_a_bare_binding_phrase():
    """Variant A must NOT carry its own ACTIVE REFERENCES block — the compiler writes one.

    Scoped to the section between the two variant headers: the file's preamble mentions
    ACTIVE REFERENCES legitimately, when explaining why the two must not be combined.
    """
    text = _render()
    variant_a = text.split("VARIANT A")[1].split("VARIANT B")[0]
    assert "the depth motion blueprint" in variant_a
    assert "ACTIVE REFERENCES" not in variant_a


def test_prompt_variant_b_is_self_contained():
    variant_b = _render().split("VARIANT B")[1].split("GUIDANCE")[0]
    assert "ACTIVE REFERENCES" in variant_b
    assert "@Video 1" in variant_b


def test_prompt_warns_against_two_competing_blocks():
    # Whitespace-normalised: the warning is line-wrapped in the template.
    assert "two competing reference blocks" in " ".join(_render().split())


def test_prompt_substitutes_every_placeholder():
    text = _render(kit_name="Chase_003", duration=6.5, width=1280, height=720)
    assert "{" not in text and "}" not in text
    assert "Chase_003" in text
    assert "6.50s" in text
    assert "1280×720" in text


def test_prompt_still_guidance_flips_on_faces():
    with_faces = _render(still_has_real_faces=True)
    without = _render(still_has_real_faces=False)
    assert "LOCAL-ONLY" in with_faces
    assert "never as a ref" in with_faces
    assert "may be attached" in without
    assert "LOCAL-ONLY" not in without


def test_prompt_states_the_refs_only_rule():
    """A frame-first shot silently drops the depth ref — the kit must say so."""
    text = _render()
    assert "skip the create-frame" in text
    assert "refs-only" in text


def test_prompt_carries_the_suggested_output_length():
    assert "5s" in _render(duration=5.0)
    assert "4s" in _render(duration=2.5)  # clamped up to the 4s output floor


def test_unknown_dialect_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="Unknown dialect"):
        _render(dialect="seedance-9.9")


# ------------------------------------------------------------------------- the manifest

def _manifest(**overrides):
    kwargs = dict(
        kit_name="Sword-fight_001",
        source_path=Path("/tmp/CLIP B.mp4"),
        source_sha256="a" * 64,
        source_duration=8.12,
        source_fps=29.97,
        source_size=(1920, 1080),
        trim=(1.0, 6.0),
        crop="1920:804:0:140",
        depth_duration=5.0,
        depth_fps=24.0,
        depth_size=(1280, 536),
        depth_bytes=1_500_000,
        frame_count=120,
        still_index=7,
        still_has_real_faces=True,
        tags=["sword fight"],
        description="two fighters, locked camera",
        settings={
            "resolution": "720p", "fps": "conform-24", "invert": False,
            "inputSize": 518, "precision": "fp32", "crop": "auto", "keepAudio": False,
        },
        report=check(duration=5.0, width=1280, height=536, fps=24.0,
                     size_bytes=1_500_000, still_has_real_faces=True),
        device="mps",
        elapsed=22.7,
    )
    kwargs.update(overrides)
    return kitlib.build_manifest(**kwargs)


def test_manifest_validates_against_the_checked_in_schema():
    import jsonschema

    jsonschema.validate(_manifest(), json.loads(SCHEMA_PATH.read_text()))


def test_manifest_records_provenance():
    manifest = _manifest()
    assert manifest["source"]["sha256"] == "a" * 64
    assert manifest["tool"]["vdaCommit"] == kitlib.VDA_COMMIT
    assert manifest["tool"]["name"] == "depthc"
    assert manifest["promptDialect"] == "seedance-2.0"


def test_manifest_records_the_crop_and_trim():
    manifest = _manifest()
    assert manifest["source"]["crop"] == "1920:804:0:140"
    assert manifest["source"]["trim"] == [1.0, 6.0]


def test_manifest_records_the_real_face_flag():
    assert _manifest()["still"]["containsRealFaces"] is True
    assert "do not attach" in _manifest()["still"]["usage"]
    assert _manifest(still_has_real_faces=False)["still"]["containsRealFaces"] is False
    assert "may be attached" in _manifest(still_has_real_faces=False)["still"]["usage"]


def test_manifest_carries_suggested_output_seconds():
    assert _manifest()["suggestedOutputSeconds"] == 5.0
    assert _manifest(depth_duration=2.5)["suggestedOutputSeconds"] == 4.0


def test_manifest_embeds_the_check_report():
    manifest = _manifest()
    assert manifest["checks"]["tier"] in ("PASS", "WARN", "FAIL")
    assert manifest["checks"]["findings"]


def test_manifest_untrimmed_and_uncropped_is_null_not_missing():
    manifest = _manifest(trim=None, crop=None)
    assert manifest["source"]["trim"] is None
    assert manifest["source"]["crop"] is None


# ------------------------------------------------------------------------- library scan

def _write_kit(root: Path, name: str, tags: list[str], tier: str = "PASS") -> Path:
    kit_dir = root / name
    kit_dir.mkdir(parents=True)
    (kit_dir / "kit.json").write_text(json.dumps({
        "kitName": name, "tags": tags, "createdAt": "2026-07-24T12:00:00+00:00",
        "depth": {"durationSeconds": 5.0}, "checks": {"tier": tier},
    }))
    return kit_dir


def test_library_list_finds_kits(tmp_path):
    _write_kit(tmp_path, "Sword-fight_001", ["sword fight"])
    _write_kit(tmp_path, "Chase_001", ["chase"])
    entries = kitlib.library_list(tmp_path)
    assert {e.name for e in entries} == {"Sword-fight_001", "Chase_001"}


def test_library_list_filters_by_tag_case_insensitively(tmp_path):
    _write_kit(tmp_path, "Sword-fight_001", ["Sword Fight"])
    _write_kit(tmp_path, "Chase_001", ["chase"])
    assert [e.name for e in kitlib.library_list(tmp_path, tag="sword fight")] == \
        ["Sword-fight_001"]


def test_library_list_on_missing_root_is_empty_not_an_error(tmp_path):
    assert kitlib.library_list(tmp_path / "nope") == []


def test_library_list_skips_malformed_manifests(tmp_path):
    _write_kit(tmp_path, "Good_001", ["ok"])
    bad = tmp_path / "Bad_001"
    bad.mkdir()
    (bad / "kit.json").write_text("{ not json")
    entries = kitlib.library_list(tmp_path)
    assert [e.name for e in entries] == ["Good_001"]


def test_library_list_ignores_folders_without_a_manifest(tmp_path):
    _write_kit(tmp_path, "Good_001", ["ok"])
    (tmp_path / "random_folder").mkdir()
    assert len(kitlib.library_list(tmp_path)) == 1


def test_library_entry_exposes_summary_fields(tmp_path):
    _write_kit(tmp_path, "Sword-fight_001", ["sword"], tier="WARN")
    entry = kitlib.library_list(tmp_path)[0]
    assert entry.duration == 5.0
    assert entry.tier == "WARN"
    assert entry.tags == ["sword"]
    assert entry.created.startswith("2026-07-24")
