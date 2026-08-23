"""`depthc describe` — a lone clip's motion in prompt words. Pure math + one fast CLI run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from depthc import describe as d, motion as m
from tests.test_motion import moving_clip, panning_clip, still_clip


def test_a_crossing_travels_left_to_right_and_says_so():
    desc = d.describe(m.measure(moving_clip(72, speed=5), 24.0))
    assert desc.travel == "frame-left → frame-right"
    live = [p for p in desc.placement if p != "·"]
    assert live[0] == "L" and live[-1] == "R"
    assert any("travels frame-left → frame-right" in s for s in desc.sentences)
    assert desc.hotspot is not None and "middle" in desc.hotspot  # the box sits at h//3


def test_the_mirror_crossing_travels_the_other_way():
    clip = moving_clip(72, speed=5)[:, :, ::-1].copy()
    assert d.describe(m.measure(clip, 24.0)).travel == "frame-right → frame-left"


def test_a_pan_is_named_as_a_camera_move_over_time():
    desc = d.describe(m.measure(panning_clip(72), 24.0))
    assert desc.camera_overall is not None and desc.camera_overall["label"] == "pans left"
    assert desc.camera_segments and desc.camera_segments[0].label == "pans left"
    # 4 source px/frame is 2 grid px/frame — over the PATH_FAST crossing-time band.
    assert desc.camera_segments[0].tempo == "quickly"
    assert any(
        s.startswith("Camera: quickly pans left") and "camera-path" in s for s in desc.sentences
    )


def test_two_camera_runs_chain_with_then():
    desc = d.Description(
        seconds=3.0, energy=2.0, band="fine", spark="▅▅▅", dead=None, camera_share=0.8,
        camera_overall={"label": "pans left"},
        camera_segments=[
            d.Segment(0.0, 1.5, "pans left", tempo="slowly"),
            d.Segment(1.5, 3.0, "pushes in"),
        ],
    )
    line = d._sentences(desc)[0]
    assert "slowly pans left 0–1.5s, then pushes in 1.5–3s" in line


def test_a_still_clip_is_static_everywhere():
    desc = d.describe(m.measure(still_clip(48), 24.0))
    assert desc.band == "static"
    assert desc.travel is None
    assert set(desc.placement) <= {"·"}
    assert any("locked off" in s for s in desc.sentences)
    assert any("nothing moves" in s for s in desc.sentences)


def test_a_late_start_is_called_late():
    """A frozen opening then a crossing: the sentence says it starts late."""
    clip = moving_clip(96, speed=5)
    frozen = np.concatenate([np.repeat(clip[:1], 36, axis=0), clip[:60]])
    desc = d.describe(m.measure(frozen, 24.0, with_flow=False))
    assert desc.travel is not None
    assert any("late —" in s for s in desc.sentences)
    assert desc.dead is not None and desc.dead["where"] == "head"


def test_bare_profile_degrades_to_energy_only():
    desc = d.describe(m.MotionProfile(fps=24.0, deltas=np.full(48, 2.0, dtype="float32")))
    assert desc.placement == [] and desc.travel is None and desc.hotspot is None
    assert any(s.startswith("Camera: not measured") for s in desc.sentences)
    assert any(s.startswith("Energy: 2.00") for s in desc.sentences)


def depth_figure(top: int, bottom: int, frames: int = 12) -> np.ndarray:
    """A synthetic depth clip: near-bright figure spanning rows [top, bottom) on a far bg."""
    clip = np.full((frames, 90, 160), 40, dtype=np.uint8)
    clip[:, top:bottom, 70:90] = 220
    return clip


def test_framing_ladder_words():
    assert d.framing_label(0.2) == "extreme wide shot"
    assert d.framing_label(0.4) == "wide shot"
    assert d.framing_label(0.6) == "medium-wide shot"
    assert d.framing_label(0.85) == "medium shot"
    assert d.framing_label(1.0).startswith("tight")


def test_a_small_figure_reads_wide_and_a_cropped_one_reads_tight():
    wide = d.read_framing(depth_figure(30, 60))  # 30/90 ≈ 0.33 of frame height
    assert wide is not None and wide.label == "wide shot" and not wide.cropped
    tight = d.read_framing(depth_figure(0, 90))
    assert tight is not None and tight.cropped and tight.label.startswith("tight")


def test_flat_depth_has_no_figure_to_size():
    assert d.read_framing(np.full((6, 90, 160), 128, dtype=np.uint8)) is None


def test_a_push_in_names_its_framing_trend():
    clip = np.concatenate([depth_figure(40, 65, frames=24), depth_figure(15, 80, frames=24)])
    f = d.read_framing(clip)
    assert f is not None
    assert f.trend == "extreme wide shot → medium-wide shot"
    assert f.tightening is True


def test_a_depth_ramp_is_a_surface_not_a_figure():
    """Open water / a floor: near at the bottom, far at the top — never sized."""
    ramp = np.tile(np.linspace(40, 220, 90).astype(np.uint8)[:, None], (1, 160))
    assert d.read_framing(np.repeat(ramp[None], 6, axis=0)) is None


def test_framing_requires_overlap_with_the_motion_hotspot():
    clip = depth_figure(30, 60)  # figure in the centre-column band
    away = [[0.0, 0.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # motion top-right
    assert d.read_framing(clip, region_energy=away) is None
    near = [[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]]
    assert d.read_framing(clip, region_energy=near) is not None


def test_framing_sentence_appears_only_when_frames_were_given():
    profile = m.measure(moving_clip(48, speed=5), 24.0)
    without = d.describe(profile)
    assert not any(s.startswith("Framing:") for s in without.sentences)
    assert without.as_dict()["framing"] is None
    framed = d.describe(profile, framing=d.Framing(0.4, "wide shot", False, None, None))
    assert any("reads as a wide shot" in s for s in framed.sentences)
    assert framed.as_dict()["framing"]["label"] == "wide shot"


def test_render_and_dict_are_plain():
    desc = d.describe(m.measure(moving_clip(72, speed=5), 24.0))
    text = d.render(desc, "gen.mp4", colour=False)
    assert "placement" in text and "Uncalibrated" in text
    payload = json.loads(json.dumps(desc.as_dict()))
    assert payload["travel"] == "frame-left → frame-right"
    assert "uncalibrated" in payload["note"]


def test_cli_describe_rgb_runs_without_the_model(clip_3s):
    from typer.testing import CliRunner

    from depthc.cli import app

    result = CliRunner().invoke(app, ["describe", str(clip_3s), "--rgb", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["space"] == "rgb"
    assert "sentences" in payload and len(payload["sentences"]) >= 2
