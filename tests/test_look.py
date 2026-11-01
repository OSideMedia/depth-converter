"""depthc.look — the still-image sibling of the motion kit. Synthetic frames only."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from depthc import look as L


def solid(rgb, h=64, w=96):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[...] = rgb
    return f


def split(left, right, h=64, w=96):
    f = solid(left, h, w)
    f[:, w // 2 :] = right
    return f


WARM_IVORY = (214, 187, 149)
SLATE = (75, 96, 117)
NEAR_BLACK = (30, 32, 34)
SKY = (140, 175, 220)
WHITE = (245, 245, 245)


def test_lab_roundtrip_hex():
    lab = L.rgb_to_lab(np.array([[WARM_IVORY]], dtype=np.uint8))[0, 0]
    assert L.lab_to_hex(lab) == "#d6bb95"


def test_key_labels():
    assert L.measure_frame(solid(WHITE)).key == "high-key"
    assert L.measure_frame(solid(NEAR_BLACK)).key == "low-key"
    assert L.measure_frame(split(WARM_IVORY, SLATE)).key == "mid-key"


def test_palette_shares_and_warmth():
    fl = L.measure_frame(split(WARM_IVORY, SLATE))
    tops = sorted(s.hex for s in fl.palette[:2])
    assert tops == sorted(["#d6bb95", "#4b6075"])
    assert abs(fl.palette[0].share - 0.5) < 0.05
    assert L.measure_frame(solid(WARM_IVORY)).warmth > 0 > L.measure_frame(solid(SKY)).warmth


def test_palette_distance_orders_candidates():
    ref = L.build_look([split(WARM_IVORY, SLATE)], [0.0], "ref", 0.0)
    close = L.compare_frame(ref, split(WARM_IVORY, NEAR_BLACK))
    far = L.compare_frame(ref, split(SKY, WHITE))
    assert close.palette < far.palette
    assert far.warmth < 0  # cool candidate reads cooler than the warm reference


def test_look_json_shape_carries_visual_dna_and_per_shot_key():
    look = L.build_look([solid(WHITE), split(WARM_IVORY, SLATE)], [0.0, 1.0], "clip", 2.0)
    d = json.loads(json.dumps(look.as_dict()))
    assert d["schema"] == L.SCHEMA
    assert set(d["visualDna"]) == {"palette", "lighting", "materials", "era", "juxtaposition", "forbiddenList"}
    assert all(h.startswith("#") for h in d["visualDna"]["palette"])
    assert d["visualDna"]["lighting"] == ""  # words are authored, never derived
    assert d["perShotKey"] == {"0": "high-key", "1": "mid-key"}
    assert any("per shot" in n or "perShotKey" in n for n in d["notes"])
    # and it round-trips into something compare can use
    back = L.look_from_dict(d)
    assert len(back.frames) == 2 and back.keys["high-key"] == 1


def test_pick_keyframes_splits_at_a_colour_cut():
    a = [split(WARM_IVORY, SLATE)] * 6
    b = [split(SKY, WHITE)] * 6
    idx = L.pick_keyframes(np.array(a + b), 2.0)
    assert len(idx) == 2 and idx[0] < 6 <= idx[1]


def test_pick_keyframes_no_cut_is_even_spacing_and_capped():
    frames = np.array([split(WARM_IVORY, SLATE)] * 40)
    idx = L.pick_keyframes(frames, 2.0, max_frames=8)
    assert 1 <= len(idx) <= 8 and idx == sorted(idx)


def test_drift_reads_first_vs_last():
    ref = L.build_look([split(WARM_IVORY, SLATE)], [0.0], "ref", 0.0)
    frames = [split(WARM_IVORY, SLATE), split(WARM_IVORY, SKY), split(SKY, WHITE)]
    deltas = L.compare_frames(ref, frames, [0.0, 1.0, 2.0])
    dr = L.drift(deltas)
    assert dr["palette_drift"] > 0 and dr["warmth_drift"] < 0
    assert len(dr["keys"]) == 3


def test_render_never_says_verdict():
    look = L.build_look([split(WARM_IVORY, SLATE)], [0.0], "ref", 0.0)
    text = L.render_look(look) + L.render_compare(look, L.compare_frames(look, [solid(SKY)], [0.0]), "x", None)
    for word in ("adhered", "matched", "PASS", "verdict:"):
        assert word not in text


def test_style_constant_block_is_paste_ready_and_unauthored():
    look = L.build_look([solid(WHITE), split(WARM_IVORY, SLATE), solid(NEAR_BLACK)], [0.0, 1.0, 2.0], "clip", 3.0)
    d = json.loads(json.dumps(look.as_dict()))
    sc = d["styleConstant"]
    # OSIDE StyleConstant field names, exactly
    for k in ("name", "presetId", "mediumLock", "anchor", "lightingLaw", "color", "continuity"):
        assert k in sc
    assert set(sc["color"]) == {"dominant", "secondary", "accent", "hex"}
    # measured part filled: top-3 hexes in share order
    assert sc["color"]["hex"] == [s["hex"] for s in d["global"]["palette"][:3]]
    # authored part EMPTY on purpose
    for k in ("mediumLock", "anchor", "lightingLaw", "continuity"):
        assert sc[k] == ""
    assert sc["color"]["dominant"] == "" and sc["color"]["accent"] == ""
    # hints guide the author and carry the mixed-key advisory
    assert len(sc["hints"]["color"]) == 3 and "dominant:" in sc["hints"]["color"][0]
    assert "high-key" in sc["hints"]["key"] and "lighting pick" in sc["hints"]["key"]
    assert "EMPTY on purpose" in sc["authoring"]


def test_swatch_hint_words():
    assert L.swatch_hint([90.0, 0.0, 0.0], 0.5) == "50% · pale, neutral, greyed"
    assert L.swatch_hint([40.0, 5.0, 30.0], 0.2) == "20% · dark, warm, saturated"
    assert L.swatch_hint([55.0, -2.0, -12.0], 0.1) == "10% · mid, cool, muted"


def test_look_from_manifest_block_and_package_outliers():
    block = {"name": "Film look", "hex": ["#d6bb95", "#4b6075", "#1e2022"], "targets": {"luma_mean": 50.0, "warmth": 6.0}}
    ref = L.look_from_manifest_block(block)
    assert ref.source == "Film look" and len(ref.palette) == 3
    assert abs(sum(s.share for s in ref.palette) - 1.0) < 1e-6 and ref.palette[0].share > ref.palette[-1].share
    # four "clips": three on-look, one wildly cool/bright → flagged relative to its siblings
    on = [split(WARM_IVORY, SLATE)] * 3
    off = [split(SKY, WHITE)] * 3
    scores = [L.score_clip(ref, on, [0, 1, 2], f"v{i}.mp4", f"1/{i}") for i in range(3)]
    scores.append(L.score_clip(ref, off, [0, 1, 2], "v9.mp4", "2/9"))
    scores = L.flag_outliers(scores)
    assert scores[3].outlier and not scores[0].outlier
    text = L.render_package(ref, scores)
    assert "outliers 1/4" in text and "PASS" not in text and "adhered" not in text


def test_flag_outliers_needs_three_clips_and_ignores_tiny_spread():
    ref = L.build_look([split(WARM_IVORY, SLATE)], [0.0], "ref", 0.0)
    two = [L.score_clip(ref, [solid(SKY)], [0], "a", "a"), L.score_clip(ref, [solid(WARM_IVORY)], [0], "b", "b")]
    assert not any(s.outlier for s in L.flag_outliers(two))
    same = [L.score_clip(ref, [split(WARM_IVORY, SLATE)], [0], f"{i}", f"{i}") for i in range(4)]
    assert not any(s.outlier for s in L.flag_outliers(same))


def test_cdl_from_deltas_direction_and_clamps():
    # too bright & too warm & too saturated → darker, cooler (r down, b up), desaturated
    c = L.cdl_from_deltas(12.0, 4.0, 4.0, 11.0)
    assert c["offset"][1] < 0 and c["offset"][0] < c["offset"][2] and c["saturation"] < 1
    assert c["slope"] == [1.0, 1.0, 1.0] and c["power"] == [1.0, 1.0, 1.0]
    # on-look clip → identity
    assert L.cdl_is_identity(L.cdl_from_deltas(0.0, 0.0, 0.0, 11.0))
    # absurd deltas clamp
    x = L.cdl_from_deltas(200.0, -200.0, 500.0, 11.0)
    assert abs(x["offset"][1]) <= L.CDL_OFFSET_MAX and L.CDL_SAT_MIN <= x["saturation"] <= L.CDL_SAT_MAX


def test_cdl_manifest_shape():
    ref = L.build_look([split(WARM_IVORY, SLATE)], [0.0], "ref", 0.0)
    scores = [L.score_clip(ref, [split(SKY, WHITE)], [0], "videos/a.mp4", "1/1"),
              L.ClipScore("videos/b.mp4", "1/2", 0, 0, 0, 0, 0, 0, "", error="unreadable")]
    m = L.cdl_manifest(ref, scores, "/pkg/manifest.json")
    assert m["schema"] == L.CDL_SCHEMA and "HEURISTIC" in m["derivation"]
    assert m["clips"][0]["cdl"]["saturation"] and m["clips"][0]["identity"] is False
    assert m["clips"][1] == {"file": "videos/b.mp4", "error": "unreadable"}


# ------------------------------------------------------------------ a manifest from someone else

def _package(tmp_path, clip_tiny, videos: list[dict], format_="oside-davinci/v1") -> tuple:
    """A package holding one real clip, and a measured look to score against."""
    package = tmp_path / "package"
    (package / "videos").mkdir(parents=True)
    shutil.copy(clip_tiny, package / "videos" / "shot.mp4")
    look_json = tmp_path / "look.json"
    look_json.write_text(json.dumps(L.build_look(solid((128, 128, 128))[None], [0.0], "still", 0.0).as_dict()))
    manifest = package / "manifest.json"
    manifest.write_text(json.dumps({"format": format_, "videos": videos}))
    return look_json, manifest


def _look_compare(look_json, manifest, *extra):
    from typer.testing import CliRunner

    from depthc.cli import app

    return CliRunner().invoke(app, ["look-compare", str(look_json), "--manifest", str(manifest), "--json", *extra])


def test_a_manifest_reads_the_clips_in_its_own_package(tmp_path, clip_tiny):
    look_json, manifest = _package(tmp_path, clip_tiny, [{"file": "videos/shot.mp4"}])
    done = _look_compare(look_json, manifest)
    assert done.exit_code == 0, done.output
    clip = json.loads(done.output)["clips"][0]
    assert not clip.get("error")


@pytest.mark.parametrize("escape", ["../private.mp4", "ABSOLUTE", "videos/link.mp4"])
def test_a_manifest_cannot_read_a_file_outside_its_package(tmp_path, clip_tiny, escape):
    private = tmp_path / "private.mp4"
    shutil.copy(clip_tiny, private)
    target = str(private) if escape == "ABSOLUTE" else escape
    look_json, manifest = _package(tmp_path, clip_tiny, [{"file": target}])
    if escape == "videos/link.mp4":
        (manifest.parent / "videos" / "link.mp4").symlink_to(private)
    done = _look_compare(look_json, manifest)
    assert done.exit_code == 0, done.output
    assert json.loads(done.output)["clips"][0]["error"] == "outside the package"


def test_a_file_that_is_not_a_package_is_refused(tmp_path, clip_tiny):
    look_json, manifest = _package(tmp_path, clip_tiny, [{"file": "videos/shot.mp4"}], format_="something/else")
    done = _look_compare(look_json, manifest)
    assert done.exit_code == 1
    assert "not an oside-davinci/v1 package" in done.output


def test_the_cdl_goes_in_place_of_a_symlink_and_not_through_it(tmp_path, clip_tiny):
    look_json, manifest = _package(tmp_path, clip_tiny, [{"file": "videos/shot.mp4"}])
    precious = tmp_path / "my-document.txt"
    precious.write_text("mine")
    (manifest.parent / "look-cdl.json").symlink_to(precious)
    done = _look_compare(look_json, manifest, "--emit-cdl")
    assert done.exit_code == 0, done.output
    assert precious.read_text() == "mine"
    assert not (manifest.parent / "look-cdl.json").is_symlink()
