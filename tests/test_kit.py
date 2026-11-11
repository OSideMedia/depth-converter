from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from depthc import checks
from depthc import kit as kitlib
from depthc.checks import check

SCHEMA_PATH = Path(__file__).parent.parent / "depthc" / "kit_schema.json"
RESULT_SCHEMA_PATH = Path(__file__).parent.parent / "depthc" / "result_schema.json"


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
    monkeypatch.delenv("DEPTHC_LIBRARY", raising=False)
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


# --------------------------------------------------------------------- inbox workflow

def test_inbox_and_processed_default_into_the_project(monkeypatch):
    # The autouse isolation fixture sets these; unset to see the real defaults.
    monkeypatch.delenv("DEPTHC_INBOX", raising=False)
    monkeypatch.delenv("DEPTHC_PROCESSED", raising=False)
    assert kitlib.inbox_dir().name == "Inbox"
    assert kitlib.processed_dir().name == "Processed"
    assert kitlib.inbox_dir().parent == kitlib.PROJECT_ROOT


def test_inbox_and_processed_honour_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPTHC_INBOX", str(tmp_path / "in"))
    monkeypatch.setenv("DEPTHC_PROCESSED", str(tmp_path / "done"))
    assert kitlib.inbox_dir() == tmp_path / "in"
    assert kitlib.processed_dir() == tmp_path / "done"


def test_inbox_clips_finds_videos_only(tmp_path):
    for name in ("a.mp4", "b.MOV", "c.mkv", "notes.txt", "sheet.csv"):
        (tmp_path / name).write_bytes(b"x")
    found = {p.name for p in kitlib.inbox_clips(tmp_path)}
    assert found == {"a.mp4", "b.MOV", "c.mkv"}


def test_inbox_clips_ignores_dotfiles(tmp_path):
    """.DS_Store and macOS resource forks live in every real folder."""
    (tmp_path / "real.mp4").write_bytes(b"x")
    (tmp_path / "._real.mp4").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    assert [p.name for p in kitlib.inbox_clips(tmp_path)] == ["real.mp4"]


def test_inbox_clips_ignores_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.mp4").write_bytes(b"x")
    (tmp_path / "top.mp4").write_bytes(b"x")
    assert [p.name for p in kitlib.inbox_clips(tmp_path)] == ["top.mp4"]


def test_inbox_clips_on_missing_folder_is_empty(tmp_path):
    assert kitlib.inbox_clips(tmp_path / "nope") == []


def test_inbox_clips_are_oldest_first(tmp_path):
    import os
    import time

    for index, name in enumerate(("third.mp4", "first.mp4", "second.mp4")):
        path = tmp_path / name
        path.write_bytes(b"x")
        os.utime(path, (time.time() + index, time.time() + index))
    assert [p.name for p in kitlib.inbox_clips(tmp_path)] == \
        ["third.mp4", "first.mp4", "second.mp4"]


def test_archive_source_moves_the_file(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"data")
    destination = tmp_path / "done"

    moved = kitlib.archive_source(source, destination)
    assert moved == destination / "clip.mp4"
    assert moved.read_bytes() == b"data"
    assert not source.exists()


def test_archive_source_never_overwrites(tmp_path):
    """Two clips of the same name from different shoots must both survive."""
    destination = tmp_path / "done"
    destination.mkdir()
    (destination / "clip.mp4").write_bytes(b"original")

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"newer")
    moved = kitlib.archive_source(source, destination)

    assert moved.name == "clip_2.mp4"
    assert (destination / "clip.mp4").read_bytes() == b"original"
    assert moved.read_bytes() == b"newer"


def test_archive_source_creates_the_destination(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    moved = kitlib.archive_source(source, tmp_path / "brand" / "new")
    assert moved.is_file()


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
    """On 2.0 a frame-first shot silently drops the depth ref — the kit must say so."""
    text = _render(dialect="seedance-2.0")
    assert "skip the create-frame" in text
    assert "refs-only" in text


def test_the_start_frame_rule_inverts_on_2_5():
    """The two models disagree, and printing 2.0's rule into a 2.5 kit costs a staging lock.

    2.0: a start frame takes the whole request and the depth ref is dropped.
    2.5: the frame demotes to a reference image and both ride.
    Until 2026-08-24 there was one template, so every kit carried the 2.0 rule.
    """
    text = _render(dialect="seedance-2.5")
    assert "You MAY use create-frame" in text
    assert "skip the create-frame" not in text


def test_no_kit_instructs_a_4k_master_on_2_5():
    """'Master at 4K' is followable on 2.0 and names a tier 2.5 does not offer today.

    Asserted on the INSTRUCTION, not the substring: the 2.5 file mentions the phrase in the
    sentence that negates it, and a bare `not in` would fail on correct prose.
    """
    assert "Iterate at 720p, master at 4K" in _render(dialect="seedance-2.0")
    text25 = _render(dialect="seedance-2.5")
    assert "THERE IS NO 4K RUNG" in text25
    assert "Iterate at 720p, master at 4K" not in text25


def test_the_2_5_template_does_not_present_the_0_of_3_as_its_own_result():
    """The 1080p finding is a 2.0 measurement. Quoting it as a 2.5 result would be false.

    It may still be MENTIONED — it must be labelled as not transferring.
    """
    text = _render(dialect="seedance-2.5")
    assert "DOES NOT TRANSFER" in text
    assert "a hard reference is unfired" in text
    assert "UNMEASURED" not in text  # the 2026-08-24 fire measured it, once


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
        depth_sha256="b" * 64,
        binding_sha256="c" * 64,
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
    assert manifest["promptDialect"] == checks.DEFAULT_DIALECT == "seedance-2.5"


def test_a_2_0_kit_still_gets_2_0_prose_and_numbers_end_to_end():
    """Every place that switches on the dialect name must still pick 2.0 when asked.

    The default moved to 2.5 (audit DC-5); this is the guard that the flag, not the
    default, decides the template, the binding, the manifest envelope and the README.
    """
    assert kitlib.DEFAULT_DIALECT == checks.DEFAULT_DIALECT
    manifest = _manifest(dialect="seedance-2.0")
    assert manifest["promptDialect"] == "seedance-2.0"
    assert manifest["targetModel"]["dialect"] == "seedance-2.0"
    assert manifest["targetModel"]["refSeconds"] == [2.0, 15.0]
    assert manifest["targetModel"]["refsWithStartFrame"] is False
    assert "4k" in manifest["targetModel"]["outputResolutions"]
    assert "dialect seedance-2.0" in _render(dialect="seedance-2.0")
    assert kitlib.render_binding("seedance-2.0") == kitlib.render_binding("seedance-2.0")
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000,
                   dialect="seedance-2.0")
    readme = kitlib.render_kit_readme(manifest, report)
    assert "Skip create-frame" in readme
    assert "never 1080p" in readme


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

def _write_kit(root: Path, name: str, tags: list[str], tier: str = "PASS",
               with_media: bool = True) -> Path:
    """A real kit folder. `with_media=False` builds a HUSK — kit.json restored from git with
    the media it can never bring back (Outbox media is untracked by the rights law)."""
    kit_dir = root / name
    kit_dir.mkdir(parents=True)
    (kit_dir / "kit.json").write_text(json.dumps({
        "kitName": name, "tags": tags, "createdAt": "2026-07-24T12:00:00+00:00",
        "depth": {"durationSeconds": 5.0, "filename": "depth.mp4"}, "checks": {"tier": tier},
    }))
    if with_media:
        (kit_dir / "depth.mp4").write_bytes(b"\x00not a real clip, but it is present")
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


def test_a_manifest_that_is_not_text_is_skipped_like_any_broken_one(tmp_path):
    _write_kit(tmp_path, "Good_001", ["ok"])
    bad = tmp_path / "Bad_001"
    bad.mkdir()
    (bad / "kit.json").write_bytes(b"\xff\xfe not text")
    assert [e.name for e in kitlib.library_list(tmp_path)] == ["Good_001"]
    assert kitlib.read_manifest(bad) is None


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


# ------------------------------------------------------------- binding.txt & README.txt

def test_render_binding_is_one_line():
    """It is pasted into a single-line binding field; hard wraps are a nuisance there."""
    binding = kitlib.render_binding()
    assert "\n" not in binding
    assert binding.startswith("the depth motion blueprint")
    assert "appearance comes entirely from the image references" in binding


def test_render_binding_has_no_active_references_block():
    """Variant A must never carry its own block — the Composer compiles one."""
    assert "ACTIVE REFERENCES" not in kitlib.render_binding()


def test_binding_matches_the_phrase_inside_prompt_txt():
    """One home for the phrase: prompt.txt and binding.txt cannot drift apart."""
    prompt = _render()
    normalised = " ".join(prompt.split())
    assert kitlib.render_binding() in normalised


def test_prompt_has_no_leftover_markers():
    prompt = _render()
    assert kitlib.BINDING_OPEN not in prompt
    assert kitlib.BINDING_CLOSE not in prompt
    assert "[[" not in prompt


def test_render_binding_rejects_unknown_dialect():
    with pytest.raises(FileNotFoundError):
        kitlib.render_binding("seedance-9.9")


def test_kit_readme_names_the_one_file_to_upload():
    """Seven files in the folder, exactly one goes to Seedance.

    It was two until the 2026-07-25 live run: an uploaded video reference has no binding
    field in OSIDE, and the compiler writes its own @Video line, so pasting binding.txt on
    top was two instructions for one reference. The README must not send the operator looking for
    a field that isn't there.
    """
    report = check(duration=5.0, width=1280, height=536, fps=24.0,
                   size_bytes=800_000, still_has_real_faces=True)
    text = kitlib.render_kit_readme(_manifest(), report, has_preview=True)

    assert "TAKE THIS ONE FILE TO SEEDANCE" in text
    assert "depth.mp4    upload as the VIDEO reference" in text
    assert "WRITE NOTHING ABOUT THE MOTION" in text
    assert text.index("TAKE THIS ONE FILE") < text.index("EVERYTHING ELSE")


def test_kit_readme_states_the_refs_only_rule_on_2_0():
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000,
                   dialect="seedance-2.0")
    text = kitlib.render_kit_readme(_manifest(dialect="seedance-2.0"), report)
    assert "MAKE VIDEO" in text
    assert "Skip create-frame" in text


def test_kit_readme_on_the_default_dialect_allows_a_start_frame():
    """The default is 2.5 (audit DC-5): the frame demotes to a reference, refs still ride."""
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000)
    text = kitlib.render_kit_readme(_manifest(), report)
    assert "You MAY use create-frame" in text
    assert "Skip create-frame" not in text


def test_kit_readme_flags_a_real_face_still_as_local_only():
    report = check(duration=5.0, width=1280, height=536, fps=24.0,
                   size_bytes=800_000, still_has_real_faces=True)
    text = kitlib.render_kit_readme(_manifest(still_has_real_faces=True), report)
    assert "LOCAL ONLY" in text


def test_kit_readme_allows_a_faceless_still_as_a_ref():
    report = check(duration=5.0, width=1280, height=536, fps=24.0,
                   size_bytes=800_000, still_has_real_faces=False)
    text = kitlib.render_kit_readme(_manifest(still_has_real_faces=False), report)
    assert "may be attached" in text
    assert "LOCAL ONLY" not in text


def test_kit_readme_lists_problems_but_not_passes():
    report = check(duration=12.0, width=1280, height=536, fps=24.0,
                   size_bytes=800_000, still_has_real_faces=False)
    text = kitlib.render_kit_readme(_manifest(), report)
    assert "NOTES" in text
    assert "[WARN] duration" in text
    assert "[PASS]" not in text


def test_kit_readme_omits_preview_when_absent():
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000)
    assert "preview.mp4" not in kitlib.render_kit_readme(_manifest(), report, has_preview=False)
    assert "preview.mp4" in kitlib.render_kit_readme(_manifest(), report, has_preview=True)


# --------------------------------------------------------------------------- the Outbox

def test_default_output_is_the_project_outbox(monkeypatch):
    """Finals belong beside Inbox/ and Processed/, not buried in the home folder."""
    monkeypatch.delenv("DEPTHC_LIBRARY", raising=False)
    assert kitlib.library_root() == kitlib.PROJECT_ROOT / "Outbox"
    assert kitlib.DEFAULT_LIBRARY.name == "Outbox"


def test_index_names_the_two_files_to_upload(tmp_path):
    _write_kit(tmp_path, "Sword-duel_001", ["sword"])
    text = kitlib.write_outbox_index(tmp_path).read_text()
    assert "depth.mp4" in text and "binding.txt" in text
    assert "do not upload" in text


def test_index_lists_every_kit(tmp_path):
    _write_kit(tmp_path, "A_001", ["x"])
    _write_kit(tmp_path, "B_001", ["y"], tier="FAIL")
    text = kitlib.write_outbox_index(tmp_path).read_text()
    assert "A_001" in text and "B_001" in text
    assert "2 kits" in text
    assert "FAIL" in text


def test_index_on_an_empty_outbox_says_so(tmp_path):
    text = kitlib.write_outbox_index(tmp_path).read_text()
    assert "nothing here yet" in text


def test_index_is_rewritten_not_appended(tmp_path):
    _write_kit(tmp_path, "A_001", ["x"])
    kitlib.write_outbox_index(tmp_path)
    first = (tmp_path / "INDEX.txt").read_text()
    kitlib.write_outbox_index(tmp_path)
    assert (tmp_path / "INDEX.txt").read_text() == first


def test_index_creates_the_outbox_if_missing(tmp_path):
    target = tmp_path / "brand_new"
    assert kitlib.write_outbox_index(target).is_file()


def test_index_marks_a_kit_whose_media_is_gone(tmp_path):
    """A kit folder can outlive its media: kit.json and result.json are tracked, depth.mp4
    never is, so a restore brings back the record alone (Sha256Proof_001,
    2026-09-15). Every other line of INDEX.txt says "open its README.txt" and "upload
    depth.mp4" — for a husk both files are gone, so the entry has to say so."""
    _write_kit(tmp_path, "Alive_001", ["x"])
    _write_kit(tmp_path, "Husk_001", ["y"], with_media=False)
    text = kitlib.write_outbox_index(tmp_path).read_text()

    husk_entry = text.split("Husk_001")[1].split("\n\n")[0]
    assert "MEDIA GONE" in husk_entry, text
    assert "Nothing here can be fired" in husk_entry

    # ...and the kit that still HAS its clip is not marked. Without this the mark could be
    # unconditional and the test would still be green.
    alive_entry = text.split("Alive_001")[1].split("\n\n")[0]
    assert "MEDIA GONE" not in alive_entry, alive_entry
    assert "Nothing here can be fired" not in alive_entry, alive_entry


def test_index_file_is_not_mistaken_for_a_kit(tmp_path):
    """library_list globs */kit.json, so a loose INDEX.txt must not appear as a kit."""
    _write_kit(tmp_path, "A_001", ["x"])
    kitlib.write_outbox_index(tmp_path)
    assert [e.name for e in kitlib.library_list(tmp_path)] == ["A_001"]


# ------------------------------------------------------------------- motion in the manifest

def _motion_block(energy: float = 4.42) -> dict:
    import numpy as np

    from depthc import motion as motionlib

    depth = motionlib.MotionProfile(
        fps=24.0, deltas=np.full(120, energy, dtype="float32"), camera=0.3
    )
    source = motionlib.MotionProfile(fps=24.0, deltas=np.full(120, 8.5, dtype="float32"))
    return motionlib.manifest_block(depth, source)


def test_manifest_with_motion_validates_against_the_schema():
    import jsonschema

    manifest = _manifest(
        motion=_motion_block(),
        variants=[{"name": "contrast", "filename": "contrast.mp4",
                   "description": "stretched", "energy": 6.1}],
        has_sheet=True,
    )
    jsonschema.validate(manifest, json.loads(SCHEMA_PATH.read_text()))


def test_motion_block_is_absent_rather_than_null_when_unmeasured():
    """A kit built before the gate existed and one that measured nothing are different."""
    assert "motion" not in _manifest()
    assert "variants" not in _manifest()
    assert "sheet" not in _manifest()


def test_readme_shows_the_sparkline_and_the_scale():
    manifest = _manifest(motion=_motion_block())
    text = kitlib.render_kit_readme(manifest, check(duration=5.0, width=1280, height=536,
                                                    fps=24.0, size_bytes=1_500_000,
                                                    still_has_real_faces=True))
    assert manifest["motion"]["sparkline"] in text
    assert "4.42" in text  # the calibration anchor, so the number means something
    assert "52%" in text   # retention, rendered as a percentage


def test_readme_omits_the_motion_block_when_there_is_none():
    """Matches the block HEADER, not the bare word — the Composer guidance says
    'WRITE NOTHING ABOUT THE MOTION' in every README."""
    text = kitlib.render_kit_readme(_manifest(), check(
        duration=5.0, width=1280, height=536, fps=24.0, size_bytes=1_500_000,
        still_has_real_faces=True))
    assert "MOTION  (mean grey-level change" not in text


def test_readme_tells_you_to_upload_one_variant_not_three():
    manifest = _manifest(
        motion=_motion_block(),
        variants=[{"name": "invert", "filename": "invert.mp4",
                   "description": "flipped", "energy": 4.4}],
    )
    text = kitlib.render_kit_readme(manifest, check(
        duration=5.0, width=1280, height=536, fps=24.0, size_bytes=1_500_000,
        still_has_real_faces=True))
    assert "ONE PER GENERATION" in text
    assert "variants/invert.mp4" in text


def test_readme_always_explains_how_to_record_a_fire():
    text = kitlib.render_kit_readme(_manifest(), check(
        duration=5.0, width=1280, height=536, fps=24.0, size_bytes=1_500_000,
        still_has_real_faces=True))
    assert "depthc result" in text
    # ...and how to name the OSIDE generation it became (step d of the shared-id plan).
    assert "--generation" in text


# ------------------------------------------------------------------------ the fire log

def _kit_on_disk(tmp_path, name: str = "Sword-duel_001") -> Path:
    """A kit folder whose binding.txt really is the file kit.json records, so the binding
    check passes on it — the state every freshly built kit is in."""
    kit_dir = tmp_path / name
    kit_dir.mkdir()
    binding = kit_dir / kitlib.BINDING_FILE
    binding.write_text(kitlib.render_binding() + "\n")
    manifest = _manifest(binding_sha256=kitlib.sha256_file(binding))
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(manifest))
    return kit_dir


def test_an_unfired_kit_has_an_empty_log(tmp_path):
    log = kitlib.read_results(_kit_on_disk(tmp_path))
    assert log["fires"] == []


def test_recording_a_fire_writes_result_json(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="adhered", surface="oside", cost_usd=0.31,
                         notes="followed the lunge exactly")
    saved = json.loads((kit_dir / "result.json").read_text())
    assert saved["fires"][0]["verdict"] == "adhered"
    assert saved["fires"][0]["costUsd"] == 0.31
    assert saved["fires"][0]["meaning"] == kitlib.VERDICTS["adhered"]
    assert saved["kitName"] == kit_dir.name


def test_fires_accumulate_rather_than_overwrite(tmp_path):
    """Two fires of one ref is a data point about prompts; keeping only the last hides it."""
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="ignored")
    log = kitlib.record_result(kit_dir, verdict="adhered", notes="second prompt")
    assert [f["verdict"] for f in log["fires"]] == ["ignored", "adhered"]


def test_an_unknown_verdict_is_refused(tmp_path):
    with pytest.raises(ValueError):
        kitlib.record_result(_kit_on_disk(tmp_path), verdict="great")


def _one_fire_log(kit_dir) -> dict:
    return {"schemaVersion": kitlib.RESULT_SCHEMA_VERSION, "kitName": kit_dir.name,
            "fires": [{"firedAt": "2026-11-01T00:00:00+00:00", "verdict": "adhered", "notes": "paid"}]}


@pytest.mark.parametrize("make", [
    lambda log: json.dumps(log).encode("utf-16"),                  # not UTF-8 text
    lambda log: json.dumps(log, indent=2).encode()[:-9],           # truncated JSON
    lambda log: b"{not json",
    lambda log: json.dumps(log["fires"]).encode(),                 # valid JSON, a list
    lambda log: json.dumps({"kitName": "x", "fires": "none"}).encode(),  # no fires list
    lambda log: json.dumps({k: v for k, v in log.items() if k != "fires"}).encode(),
], ids=["utf16", "truncated", "not-json", "list", "fires-not-a-list", "no-fires"])
def test_an_unreadable_fire_log_is_never_written_over(tmp_path, make):
    """A fire is a paid generation. A log that cannot be read is not an empty log: recording
    onto it must refuse, name the file, and leave its bytes exactly as they were."""
    kit_dir = _kit_on_disk(tmp_path)
    path = kit_dir / "result.json"
    path.write_bytes(make(_one_fire_log(kit_dir)))
    before = path.read_bytes()
    with pytest.raises(ValueError) as err:
        kitlib.record_result(kit_dir, verdict="partial")
    message = str(err.value)
    assert "result.json" in message and "untouched" in message and "move" in message
    assert path.read_bytes() == before
    # Read-only callers still work: a broken log shows as an unfired kit.
    assert kitlib.read_results(kit_dir)["fires"] == []
    assert kitlib.library_list(tmp_path)[0].verdict == "untested"


def test_an_unreadable_log_also_stops_a_keyed_fire(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    (kit_dir / "result.json").write_text("{not json")
    with pytest.raises(ValueError, match="untouched"):
        kitlib.record_result(kit_dir, verdict="partial", generation_id=GEN)
    assert (kit_dir / "result.json").read_text() == "{not json"


def test_a_missing_result_file_is_an_empty_log_and_writing_works(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    assert not (kit_dir / "result.json").exists()
    kitlib.record_result(kit_dir, verdict="partial")
    assert len(kitlib.read_results(kit_dir)["fires"]) == 1


def test_library_entry_reports_the_latest_verdict(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    entry = kitlib.library_list(tmp_path)[0]
    assert entry.verdict == "untested"
    kitlib.record_result(kit_dir, verdict="late")
    assert kitlib.library_list(tmp_path)[0].verdict == "late"


def test_library_entry_exposes_motion_energy(tmp_path):
    kit_dir = tmp_path / "Kit_001"
    kit_dir.mkdir()
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(_manifest(motion=_motion_block(2.5))))
    assert kitlib.library_list(tmp_path)[0].energy == pytest.approx(2.5)


def test_index_shows_motion_and_verdicts(tmp_path):
    kit_dir = tmp_path / "Kit_001"
    kit_dir.mkdir()
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(_manifest(motion=_motion_block())))
    kitlib.record_result(kit_dir, verdict="adhered", notes="clean")
    text = kitlib.write_outbox_index(tmp_path).read_text()
    assert "motion 4.42" in text
    assert "FIRED 1x · last: adhered — clean" in text


def test_index_says_nothing_about_fires_for_an_untested_kit(tmp_path):
    kit_dir = tmp_path / "Kit_001"
    kit_dir.mkdir()
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(_manifest()))
    # "FIRED" also appears in the header's explanation, so check the entry itself.
    entry = kitlib.write_outbox_index(tmp_path).read_text().split("1 kit:")[1]
    assert "FIRED" not in entry


# --------------------------------------------- depth.sha256 — the content half of the join
#
# OSIDE's upload door records the uploaded clip's SHA-256 on the reference
# (request_payload.request.refs[].sha256). kit.json carries the same hash of depth.mp4, so
# kit.json.depth.sha256 == refs[].sha256 names the kit a generation rode
# (PLAN-SHARED-GENERATION-ID step d, 2026-09-06).


def test_manifest_depth_sha256_is_the_hash_of_the_written_file(tmp_path):
    depth = tmp_path / "depth.mp4"
    depth.write_bytes(b"\x00\x00\x00\x1cftypisom" + bytes(range(256)) * 8)
    manifest = _manifest(depth_sha256=kitlib.sha256_file(depth))
    assert manifest["depth"]["sha256"] == hashlib.sha256(depth.read_bytes()).hexdigest()
    assert len(manifest["depth"]["sha256"]) == 64


@pytest.mark.slow
def test_a_built_kit_records_the_hash_of_its_own_depth_mp4(clip_3s, tmp_path):
    """End to end: the hash in kit.json is of the depth.mp4 that sits beside it — the
    bytes that get uploaded, after the size gate and any audio mux, not an intermediate."""
    from typer.testing import CliRunner

    from depthc import cli

    lib = tmp_path / "lib"
    result = CliRunner().invoke(cli.app, ["kit", str(clip_3s), "--library", str(lib), "--tags", "hash"])
    kit_dirs = [p for p in lib.iterdir() if p.is_dir()] if lib.is_dir() else []
    assert kit_dirs, result.output
    kit_dir = kit_dirs[0]
    manifest = json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())
    on_disk = hashlib.sha256((kit_dir / "depth.mp4").read_bytes()).hexdigest()
    assert manifest["depth"]["sha256"] == on_disk
    assert kitlib.depth_sha256(kit_dir) == on_disk


def test_an_old_manifest_without_depth_sha256_still_loads(tmp_path):
    """A kit built before 2026-09-06 has no depth.sha256 and real fires behind it; the
    reader says None, it never refuses the kit."""
    kit_dir = tmp_path / "Old-kit_001"
    kit_dir.mkdir()
    manifest = _manifest()
    del manifest["depth"]["sha256"]
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(manifest))
    assert [e.path for e in kitlib.library_list(tmp_path)] == [kit_dir]
    assert kitlib.depth_sha256(kit_dir) is None
    kitlib.record_result(kit_dir, verdict="adhered")
    assert kitlib.library_list(tmp_path)[0].verdict == "adhered"


def test_depth_sha256_reads_a_new_manifest(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    assert kitlib.depth_sha256(kit_dir) == "b" * 64
    assert kitlib.depth_sha256(tmp_path / "no-such-kit") is None


def test_schema_requires_a_well_formed_depth_sha256():
    """Accepted: 64 lowercase hex. Refused: anything else, and absence — on a NEW kit."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(_manifest(depth_sha256="0" * 64), schema)
    for bad in ("", "xyz", "0" * 63, "A" * 64, "0" * 65):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_manifest(depth_sha256=bad), schema)
    absent = _manifest()
    del absent["depth"]["sha256"]
    with pytest.raises(jsonschema.ValidationError, match="sha256"):
        jsonschema.validate(absent, schema)


def test_schema_version_stays_1_with_the_additive_key():
    """Audit DC-6 ruling: additive keys do not bump schemaVersion; the schema says so."""
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["properties"]["schemaVersion"] == {"const": 1}
    assert "sha256" in schema["properties"]["depth"]["required"]
    assert "additive" in schema.get("$comment", "").lower()


# ------------------------------------------- binding.sha256 — the kit's words, bound to it
#
# `binding.txt` sat beside `depth.mp4` with nothing tying the two together: a kit recorded
# the clip's hash from 2026-09-06 and the words' hash not at all. OSIDE's Import kit door
# fills a reference's binding field from this file, and the blockout kit — which
# binds BOTH halves by SHA-256 — caught one kit's legend riding on another kit's clip.


def test_manifest_binding_sha256_is_the_hash_of_the_written_file(tmp_path):
    binding = tmp_path / "binding.txt"
    binding.write_text(kitlib.render_binding() + "\n")
    manifest = _manifest(binding_sha256=kitlib.sha256_file(binding))
    assert manifest["binding"]["filename"] == "binding.txt"
    assert manifest["binding"]["sha256"] == hashlib.sha256(binding.read_bytes()).hexdigest()
    assert len(manifest["binding"]["sha256"]) == 64


@pytest.mark.slow
def test_a_built_kit_records_the_hash_of_its_own_binding_txt(clip_3s, tmp_path):
    """End to end: the hash in kit.json is of the binding.txt that sits beside it, and the
    kit it built verifies against itself."""
    from typer.testing import CliRunner

    from depthc import cli

    lib = tmp_path / "lib"
    result = CliRunner().invoke(
        cli.app, ["kit", str(clip_3s), "--library", str(lib), "--tags", "binding"]
    )
    kit_dirs = [p for p in lib.iterdir() if p.is_dir()] if lib.is_dir() else []
    assert kit_dirs, result.output
    kit_dir = kit_dirs[0]
    manifest = json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())
    on_disk = hashlib.sha256((kit_dir / "binding.txt").read_bytes()).hexdigest()
    assert manifest["binding"]["sha256"] == on_disk
    assert kitlib.binding_sha256(kit_dir) == on_disk
    assert kitlib.verify_binding(kit_dir).tier is checks.Tier.PASS


def test_an_old_manifest_without_binding_sha256_still_loads(tmp_path):
    """A kit built before 2026-09-15 has no binding.sha256 and real fires behind it. The
    reader says None and the verifier says None — a hash is never invented at read time,
    because an invented one would certify whatever happens to be in the folder."""
    kit_dir = _kit_on_disk(tmp_path)
    manifest = json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())
    del manifest["binding"]
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(manifest))
    assert [e.path for e in kitlib.library_list(tmp_path)] == [kit_dir]
    assert kitlib.binding_sha256(kit_dir) is None
    assert kitlib.verify_binding(kit_dir) is None
    kitlib.record_result(kit_dir, verdict="adhered")
    assert kitlib.library_list(tmp_path)[0].verdict == "adhered"


def test_binding_sha256_reads_a_new_manifest(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    assert kitlib.binding_sha256(kit_dir) == kitlib.sha256_file(kit_dir / "binding.txt")
    assert kitlib.binding_sha256(tmp_path / "no-such-kit") is None


def test_verify_binding_passes_on_the_file_the_kit_was_built_with(tmp_path):
    finding = kitlib.verify_binding(_kit_on_disk(tmp_path))
    assert finding.tier is checks.Tier.PASS
    assert finding.check == "binding"


def test_verify_binding_fails_on_a_single_changed_byte(tmp_path):
    """The counterexample the check exists for: the naive reader — trust binding.txt because
    it is in the folder — passes this kit, and the hashed one must not."""
    kit_dir = _kit_on_disk(tmp_path)
    binding = kit_dir / kitlib.BINDING_FILE
    naive_verdict = binding.is_file() and bool(binding.read_text().strip())
    binding.write_text(binding.read_text().replace("depth", "Depth", 1))

    assert naive_verdict, "the naive check must PASS here, or this proves nothing"
    finding = kitlib.verify_binding(kit_dir)
    assert finding.tier is checks.Tier.FAIL
    assert finding.check == "binding"
    assert "does not match" in finding.detail
    assert "templates/" in finding.remedy and "rebuild" in finding.remedy


def test_verify_binding_fails_when_the_words_are_another_kits(tmp_path):
    """The blockout door's real catch: two intact kits, the wrong words in one of them."""
    a = _kit_on_disk(tmp_path, name="Duel_001")
    b = _kit_on_disk(tmp_path, name="Walk-in_002")
    (b / kitlib.BINDING_FILE).write_text(
        (a / kitlib.BINDING_FILE).read_text() + "…and hold the final frame\n"
    )
    assert kitlib.verify_binding(a).tier is checks.Tier.PASS
    assert kitlib.verify_binding(b).tier is checks.Tier.FAIL


def test_verify_binding_fails_when_the_file_is_gone(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    (kit_dir / kitlib.BINDING_FILE).unlink()
    finding = kitlib.verify_binding(kit_dir)
    assert finding.tier is checks.Tier.FAIL
    assert "not here" in finding.detail


def test_the_remedy_names_the_dialect_that_would_rewrite_the_file(tmp_path):
    """A remedy that says "restore it" without saying from where is not a remedy."""
    kit_dir = _kit_on_disk(tmp_path)
    manifest = json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())
    manifest["promptDialect"] = "seedance-2.0"
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(manifest))
    (kit_dir / kitlib.BINDING_FILE).write_text("wrong words\n")
    assert "templates/seedance-2.0.txt" in kitlib.verify_binding(kit_dir).remedy


def test_schema_requires_a_well_formed_binding_sha256():
    """Accepted: 64 lowercase hex. Refused: anything else, and absence — on a NEW kit."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(_manifest(binding_sha256="0" * 64), schema)
    for bad in ("", "xyz", "0" * 63, "A" * 64, "0" * 65):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_manifest(binding_sha256=bad), schema)
    absent = _manifest()
    del absent["binding"]
    with pytest.raises(jsonschema.ValidationError, match="binding"):
        jsonschema.validate(absent, schema)


def test_schema_version_stays_1_with_the_binding_key():
    """Additive, like depth.sha256 before it (audit DC-6 ruling)."""
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["properties"]["schemaVersion"] == {"const": 1}
    assert "binding" in schema["required"]
    assert schema["properties"]["binding"]["required"] == ["filename", "sha256"]
    assert "2026-09-15" in schema.get("$comment", "")


# ------------------------------ fires[].generationId — the generation half of the join
#
# OSIDE's `generations.id` is the hub every store joins to. A fire that names it is a
# row in the ledger; one that does not is a number in a human's memory.

GEN = "3f2504e0-4f89-4d3c-9a6b-0305e82c3301"  # v4: version nibble 4, variant nibble a


def test_a_fire_records_the_generation_it_became(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="adhered", generation_id=GEN.upper())
    saved = json.loads((kit_dir / "result.json").read_text())
    assert saved["fires"][0]["generationId"] == GEN  # canonical: lowercase, hyphenated


def test_a_fire_without_a_generation_is_unkeyed_as_before(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="adhered")
    assert "generationId" not in json.loads((kit_dir / "result.json").read_text())["fires"][0]


@pytest.mark.parametrize("bad", [
    "gen-42",
    "",
    "3f2504e0-4f89-4d3c-9a6b-0305e82c330",     # 11 in the last group
    "3f2504e04f894d3c9a6b0305e82c3301",         # no hyphens
    "3f2504e0-4f89-1d3c-9a6b-0305e82c3301",     # version nibble 1
    "3f2504e0-4f89-4d3c-1a6b-0305e82c3301",     # variant nibble 1
    "{3f2504e0-4f89-4d3c-9a6b-0305e82c3301}",   # braced form
])
def test_a_non_uuid_generation_is_refused_naming_the_grammar(tmp_path, bad):
    kit_dir = _kit_on_disk(tmp_path)
    with pytest.raises(ValueError, match="UUID v4") as e:
        kitlib.record_result(kit_dir, verdict="adhered", generation_id=bad)
    assert "8-4-4-4-12" in str(e.value)
    assert not (kit_dir / "result.json").exists()  # refused before anything is written


def test_an_old_result_json_without_generation_ids_still_loads(tmp_path):
    """Every fire recorded before 2026-09-06 lacks the key; the log reads, and a new fire
    appended beside it carries the id while the old one stays unkeyed."""
    kit_dir = _kit_on_disk(tmp_path)
    old = {"schemaVersion": 1, "kitName": kit_dir.name, "fires": [{
        "firedAt": "2026-08-25T04:54:01+00:00", "verdict": "adhered",
        "meaning": kitlib.VERDICTS["adhered"], "surface": "ark", "model": "ark-seedance-2.5",
        "variant": "base", "outputSeconds": 5.0, "costUsd": 2.45,
        "outputRef": "DOCS/livefire/x/1080p.mp4", "notes": "ratio 0.93, r0=+0.820",
    }]}
    (kit_dir / "result.json").write_text(json.dumps(old))
    assert kitlib.read_results(kit_dir)["fires"][0]["verdict"] == "adhered"
    kitlib.record_result(kit_dir, verdict="partial", generation_id=GEN)
    fires = kitlib.read_results(kit_dir)["fires"]
    assert "generationId" not in fires[0]
    assert fires[1]["generationId"] == GEN


def test_result_schema_governs_old_and_new_records(tmp_path):
    import jsonschema

    schema = json.loads(RESULT_SCHEMA_PATH.read_text())
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="adhered", cost_usd=0.31)        # unkeyed, as before
    kitlib.record_result(kit_dir, verdict="partial", generation_id=GEN)   # keyed
    jsonschema.validate(kitlib.read_results(kit_dir), schema)
    bad = kitlib.read_results(kit_dir)
    bad["fires"][1]["generationId"] = "gen-42"
    with pytest.raises(jsonschema.ValidationError, match="generationId|pattern"):
        jsonschema.validate(bad, schema)
    # The schema's verdict list IS kit.VERDICTS — the two cannot drift apart.
    assert sorted(schema["properties"]["fires"]["items"]["properties"]["verdict"]["enum"]) \
        == sorted(kitlib.VERDICTS)


def test_every_tracked_result_json_validates():
    """The per-kit records tracked by law (audit DC-4) are the schema's real subjects; an
    empty subject set is a failed enumeration, not a pass."""
    import subprocess

    import jsonschema

    root = Path(__file__).parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "Outbox/*/result.json"], cwd=root, capture_output=True, text=True,
    ).stdout.split()
    assert tracked, "no tracked Outbox/*/result.json — the DC-4 evidence set is empty"
    schema = json.loads(RESULT_SCHEMA_PATH.read_text())
    for rel in tracked:
        jsonschema.validate(json.loads((root / rel).read_text()), schema)


# ------------------------------------------------------------------ a kit from someone else

def test_a_symlinked_result_json_is_replaced_and_its_target_left_alone(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    precious = tmp_path / "my-document.txt"
    precious.write_text("mine")
    (kit_dir / "result.json").symlink_to(precious)
    kitlib.record_result(kit_dir, verdict="adhered")
    assert precious.read_text() == "mine"
    assert not (kit_dir / "result.json").is_symlink()
    assert json.loads((kit_dir / "result.json").read_text())["fires"][0]["verdict"] == "adhered"


@pytest.fixture
def umask_022():
    before = os.umask(0o022)
    yield
    os.umask(before)


def test_a_replaced_file_is_readable_and_leaves_no_temp_file(tmp_path, umask_022):
    target = tmp_path / "result.json"
    kitlib.write_replacing(target, "{}\n")
    assert target.read_text() == "{}\n"
    assert target.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_a_replaced_file_keeps_the_mode_it_had(tmp_path, umask_022):
    target = tmp_path / "result.json"
    target.write_text("old")
    target.chmod(0o600)
    kitlib.write_replacing(target, "new")
    assert target.read_text() == "new"
    assert target.stat().st_mode & 0o777 == 0o600


def test_a_private_file_is_never_readable_while_its_copy_is_written(tmp_path, umask_022, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text("old")
    target.chmod(0o600)
    real_open, modes = os.open, []

    def recording_open(path, flags, mode=0o777, **kwargs):
        modes.append(mode)
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    kitlib.write_replacing(target, "new")
    assert modes == [0o600]


def test_a_new_file_follows_the_umask(tmp_path):
    before = os.umask(0o077)
    try:
        kitlib.write_replacing(tmp_path / "result.json", "{}\n")
    finally:
        os.umask(before)
    assert (tmp_path / "result.json").stat().st_mode & 0o777 == 0o600


def test_a_file_in_place_of_a_link_does_not_take_the_mode_of_what_it_pointed_at(tmp_path, umask_022):
    open_to_all = tmp_path / "open.txt"
    open_to_all.write_text("mine")
    open_to_all.chmod(0o666)
    target = tmp_path / "result.json"
    target.symlink_to(open_to_all)
    kitlib.write_replacing(target, "{}\n")
    assert not target.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o644
    assert open_to_all.read_text() == "mine"


def test_a_manifest_that_is_not_an_object_is_skipped_everywhere(tmp_path):
    good = _kit_on_disk(tmp_path, "Good_001")
    bad = tmp_path / "Bad_001"
    bad.mkdir()
    (bad / kitlib.KIT_MANIFEST).write_text("[]")
    assert [e.path for e in kitlib.library_list(tmp_path)] == [good]
    assert kitlib.write_outbox_index(tmp_path).is_file()
# ------------------------------------------------------------------ one render, one fire (audit DC-2)

def test_one_render_cannot_be_recorded_twice_on_one_kit(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    kitlib.record_result(kit_dir, verdict="adhered", generation_id=GEN)
    with pytest.raises(ValueError, match=kit_dir.name):
        kitlib.record_result(kit_dir, verdict="partial", generation_id=GEN.upper())
    fires = json.loads((kit_dir / "result.json").read_text())["fires"]
    assert [f["verdict"] for f in fires] == ["adhered"]  # the first record stands, untouched


def test_one_render_cannot_be_recorded_on_two_kits(tmp_path):
    first = _kit_on_disk(tmp_path, "Sword-duel_001")
    second = _kit_on_disk(tmp_path, "Sword-duel_002")
    kitlib.record_result(first, verdict="adhered", generation_id=GEN)
    with pytest.raises(ValueError, match="Sword-duel_001"):
        kitlib.record_result(second, verdict="adhered", generation_id=GEN)
    assert not (second / "result.json").exists()


def test_different_renders_and_unkeyed_fires_still_accumulate(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    other = "9c858901-8a57-4791-81fe-4c455b099bc9"
    kitlib.record_result(kit_dir, verdict="adhered", generation_id=GEN)
    kitlib.record_result(kit_dir, verdict="partial", generation_id=other)
    kitlib.record_result(kit_dir, verdict="ignored")
    log = kitlib.record_result(kit_dir, verdict="late")
    assert len(log["fires"]) == 4


def test_a_broken_log_on_a_sibling_kit_never_stops_a_fire_being_recorded(tmp_path):
    sibling = _kit_on_disk(tmp_path, "Sword-duel_002")
    (sibling / "result.json").write_text("{not json")
    odd = _kit_on_disk(tmp_path, "Sword-duel_003")
    (odd / "result.json").write_text(json.dumps({"fires": ["not a fire", {"generationId": 7}]}))
    binary = _kit_on_disk(tmp_path, "Sword-duel_004")
    (binary / "result.json").write_bytes(b"\xff\xfe not text")
    log = kitlib.record_result(_kit_on_disk(tmp_path), verdict="adhered", generation_id=GEN)
    assert log["fires"][0]["generationId"] == GEN


def test_the_refusal_reaches_the_command_line_and_writes_nothing(tmp_path):
    from typer.testing import CliRunner

    from depthc.cli import app

    kit_dir = _kit_on_disk(tmp_path)
    args = ["result", str(kit_dir), "--verdict", "adhered", "--generation", GEN, "--model", "m"]
    assert CliRunner().invoke(app, args).exit_code == 0
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1
    assert "already recorded" in again.output
    assert len(json.loads((kit_dir / "result.json").read_text())["fires"]) == 1


# ------------------------------------------------------------------ a keyed fire names its model (audit DC-5)

def test_a_keyed_fire_with_no_model_is_recorded_and_says_what_is_missing(tmp_path):
    from typer.testing import CliRunner

    from depthc.cli import app

    kit_dir = _kit_on_disk(tmp_path)
    bare = CliRunner().invoke(app, ["result", str(kit_dir), "--verdict", "partial", "--generation", GEN])
    assert bare.exit_code == 0, bare.output
    assert "--model" in bare.output
    assert json.loads((kit_dir / "result.json").read_text())["fires"][0]["generationId"] == GEN

    named = _kit_on_disk(tmp_path, "Sword-duel_002")
    other = "9c858901-8a57-4791-81fe-4c455b099bc9"
    done = CliRunner().invoke(
        app, ["result", str(named), "--verdict", "partial", "--generation", other, "--model", "m"])
    assert done.exit_code == 0, done.output
    assert "--model" not in done.output


# ------------------------------------------------------------------ a broken envelope (audit DC-6)
#
# The envelope's path is fixed beside the module, so these run a COPY of the package. Each one
# goes through a real process: the defect was at import, which an in-process test cannot reach.

def _copy_with_a_broken_envelope(tmp_path) -> Path:
    root = tmp_path / "copy"
    shutil.copytree(Path(kitlib.__file__).parent, root / "depthc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "depthc" / checks.ENVELOPE_PATH.name).write_text("{not json", encoding="utf-8")
    return root


def _python(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": str(root)},
    )


def _depthc(root: Path, *args: str) -> subprocess.CompletedProcess:
    return _python(root, "-m", "depthc.cli", *args)


def test_the_first_read_of_a_broken_table_still_refuses(tmp_path):
    """The control: this is the copy that is broken, and importing it is all that got easier."""
    root = _copy_with_a_broken_envelope(tmp_path)
    imported = _python(root, "-c", "from depthc import checks, cli, kit; print(checks.__file__)")
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip().startswith(str(root))
    for read in ("checks.DIALECTS", "checks.dialect_for()", "checks.MAX_FILE_MB"):
        refused = _python(root, "-c", f"from depthc import checks; {read}")
        assert refused.returncode != 0, read
        assert "EnvelopeError" in refused.stderr, read


def test_a_broken_envelope_does_not_stop_a_paid_fire_being_recorded(tmp_path):
    root = _copy_with_a_broken_envelope(tmp_path)
    library = tmp_path / "lib"
    library.mkdir()
    kit_dir = _kit_on_disk(library)
    done = _depthc(root, "result", str(kit_dir), "--verdict", "partial", "--notes", "paid evidence")
    assert done.returncode == 0, done.stderr
    assert json.loads((kit_dir / "result.json").read_text())["fires"][0]["notes"] == "paid evidence"
    name = json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())["kitName"]
    assert name in (library / "INDEX.txt").read_text()


def test_a_broken_envelope_still_lists_the_library(tmp_path):
    root = _copy_with_a_broken_envelope(tmp_path)
    library = tmp_path / "lib"
    library.mkdir()
    kit_dir = _kit_on_disk(library)
    listed = _depthc(root, "library", "list", "--library", str(library))
    assert listed.returncode == 0, listed.stderr
    assert json.loads((kit_dir / kitlib.KIT_MANIFEST).read_text())["kitName"] in listed.stdout


def test_a_grading_command_refuses_a_broken_envelope_before_doing_any_work(tmp_path):
    root = _copy_with_a_broken_envelope(tmp_path)
    refused = _depthc(root, "kit", str(tmp_path / "no-such-clip.mp4"))
    assert refused.returncode == 1
    assert "not readable JSON" in refused.stderr  # the envelope's fault, not the missing clip's
    assert "depthc result" in refused.stderr
    assert "Traceback" not in refused.stderr


def test_help_still_names_the_dialects_with_a_broken_envelope(tmp_path):
    root = _copy_with_a_broken_envelope(tmp_path)
    assert _depthc(root, "--help").returncode == 0


@pytest.mark.parametrize("inverted, near", [(False, "bright"), (True, "dark")])
def test_a_kit_build_reads_its_camera_from_the_source_picture(clip_3s, tmp_path, monkeypatch, inverted, near):
    """kit.json's cameraPath comes from the picture the depth was made from, frame for frame;
    the depth picture is the read that invented moves on real footage. The depth pass is stood
    in for, so this runs on CI too: the wiring is what is under test, not the model. The reader is
    told which end of the depth is near, so its no-Vision fallback leaves the subject out and not
    the background."""
    from typer.testing import CliRunner

    from depthc import cli, engine, motion as motionlib

    def fake_pass(frames, fps, **_):
        return engine.DepthResult(
            frames=frames[..., 0], fps=fps, device="cpu", dtype="fp32", elapsed=0.0,
            raw_min=0.0, raw_max=1.0, inverted=inverted,
        )

    calls = []
    real_measure = motionlib.measure

    def measure(frames, fps, with_flow=True, picture=None, near="bright"):
        calls.append((frames.shape, with_flow, None if picture is None else picture.shape, near))
        return real_measure(frames, fps, with_flow=with_flow, picture=picture, near=near)

    monkeypatch.setattr(engine, "pick_device", lambda: "cpu")
    monkeypatch.setattr(engine, "infer", fake_pass)
    monkeypatch.setattr(motionlib, "measure", measure)
    lib = tmp_path / "lib"
    result = CliRunner().invoke(cli.app, ["kit", str(clip_3s), "--library", str(lib), "--tags", "cam"])
    with_flow = [c for c in calls if c[1]]
    assert len(with_flow) == 1, result.output
    depth_shape, _, picture_shape, told = with_flow[0]
    assert picture_shape is not None and picture_shape[-1] == 3
    assert picture_shape[0] == depth_shape[0]
    assert told == near
