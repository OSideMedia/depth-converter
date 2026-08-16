"""depthc.look — the still-image sibling of the motion kit. Synthetic frames only."""

from __future__ import annotations

import json

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
