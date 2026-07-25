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


def test_kit_readme_names_the_two_files_to_upload():
    """The whole point: six files in the folder, only two go to Seedance."""
    report = check(duration=5.0, width=1280, height=536, fps=24.0,
                   size_bytes=800_000, still_has_real_faces=True)
    text = kitlib.render_kit_readme(_manifest(), report, has_preview=True)

    assert "TAKE THESE TWO TO OSIDE" in text
    assert "1. depth.mp4" in text
    assert "2. binding.txt" in text
    assert text.index("TAKE THESE TWO") < text.index("EVERYTHING ELSE")


def test_kit_readme_states_the_refs_only_rule():
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000)
    text = kitlib.render_kit_readme(_manifest(), report)
    assert "MAKE VIDEO" in text
    assert "Skip create-frame" in text


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
    text = kitlib.render_kit_readme(_manifest(), check(
        duration=5.0, width=1280, height=536, fps=24.0, size_bytes=1_500_000,
        still_has_real_faces=True))
    assert "MOTION" not in text


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


# ------------------------------------------------------------------------ the fire log

def _kit_on_disk(tmp_path, name: str = "Sword-duel_001") -> Path:
    kit_dir = tmp_path / name
    kit_dir.mkdir()
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(_manifest()))
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


def test_a_corrupt_result_file_is_replaced_not_fatal(tmp_path):
    kit_dir = _kit_on_disk(tmp_path)
    (kit_dir / "result.json").write_text("{not json")
    assert kitlib.read_results(kit_dir)["fires"] == []
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
