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
