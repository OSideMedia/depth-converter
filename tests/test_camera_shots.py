"""The camera round (2026-11-11): shots, people, "can't tell", slow moves, and the scout's count.

The second real-footage score (DOCS/CAMERA-PATH-CALIBRATION.md) failed four ways: a fit read
across cuts, dancers filling the frame outvoted the wall, a crowd was read as a move instead of
nothing, and slow pushes stayed under the floor. Each fixture here is a synthetic picture of one
of those, and each test was run against the reader before this round and failed there. Vision is
off in every test not marked `vision` (tests/conftest.py), so a person mask is handed in.

Camera words never vote in any verdict; nothing here tests a pass or a fail, only what is said.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from depthc import motion as m
from test_camera_picture import FRAMES, HEIGHT, WIDTH, texture, words, zooming


def darker(frames: np.ndarray) -> np.ndarray:
    """The same kind of texture, sixty grey levels down: a different shot to cut to."""
    return cv2.convertScaleAbs(frames, alpha=1.0, beta=-60)


def brighter(frames: np.ndarray) -> np.ndarray:
    """Sixty grey levels up: a third shot, unlike either of the other two."""
    return cv2.convertScaleAbs(frames, alpha=1.0, beta=60)


def panned(base: np.ndarray, frames: int, step: int) -> np.ndarray:
    """The wall sliding `step` px a frame: right (step > 0) is the camera panning left."""
    return np.stack([np.roll(base, i * step, axis=1) for i in range(frames)])


def two_shots(frames: int = 24) -> np.ndarray:
    """A second of a pan left, a cut, a second of a pan right on another wall."""
    return np.concatenate([panned(texture(), frames, 2), darker(panned(texture(seed=23), frames, -2))])


def halves(frames: int = 24) -> np.ndarray:
    """Two halves of the frame sliding apart: no single motion carries half the points."""
    base = texture()
    half = WIDTH // 2
    return np.stack([
        np.concatenate([np.roll(base[:, :half], -3 * i, axis=1), np.roll(base[:, half:], 3 * i, axis=1)], axis=1)
        for i in range(frames)
    ])


def big_figure(frames: int = FRAMES, cover: float = 0.6, speed: int = 2):
    """A locked-off wall and one figure filling `cover` of it, walking right: (picture, masks, depth).

    The seen failure: a dancer close to the lens has more points than the wall behind, and the
    fit follows the dancer. The depth is graded top to bottom, as a real one is, with the figure
    nearer (brighter) than any of the wall.
    """
    wall = texture()
    width = int(WIDTH * cover)
    skin = texture(seed=11, height=HEIGHT, width=width)
    rows = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    picture = np.repeat(wall[None], frames, axis=0).copy()
    masks = np.zeros((frames, HEIGHT, WIDTH), dtype=bool)
    depth = np.repeat(np.repeat(30 + 60 * rows, WIDTH, axis=1)[None], frames, axis=0)
    for index in range(frames):
        x = 10 + index * speed
        picture[index, :, x:x + width] = skin
        masks[index, :, x:x + width] = True
        depth[index, :, x:x + width] = 180 + 40 * rows
    return picture, masks, depth.astype(np.uint8)


# ------------------------------------------------------------------------------------- cuts

def test_a_cut_splits_the_clip_and_no_reading_crosses_it():
    clip = two_shots()
    path = m.camera_path(clip, 24.0)
    assert not np.isfinite(path[23]).any()  # gap 23 joins the last frame of one shot to the next
    assert words(path[:23]) == "pans left"
    assert words(path[24:]) == "pans right"
    assert m.measure(clip, 24.0, picture=clip).as_dict().get("cameraShots") == [
        {"startSeconds": 0.0, "endSeconds": 1.0, "label": "pans left"},
        {"startSeconds": 1.0, "endSeconds": 2.0, "label": "pans right"},
    ]


def test_a_shot_too_short_to_read_gets_no_word():
    clip = np.concatenate([
        panned(texture(), 24, 2), darker(panned(texture(seed=31), 4, 2)), panned(texture(seed=41), 24, 2),
    ])
    shots = m.measure(clip, 24.0, picture=clip).camera_shots
    assert [shot["label"] for shot in shots or []] == ["pans left", None, "pans left"]


def test_a_histogram_jump_with_the_points_still_tracked_is_not_a_cut():
    """A flat shape sweeping over part of the frame (a train, a door) moves the histogram as far as
    a cut does, while the points on the rest of the frame stay tracked: one shot."""
    clip = np.repeat(texture()[None], 24, axis=0).copy()
    clip[12:, :, :WIDTH * 2 // 5] = 255
    grey = m._track_gray(clip)
    hist = [cv2.calcHist([f], [0], None, [m.CUT_HISTOGRAM_BINS], [0, 256]).ravel() for f in grey[11:13]]
    jump = 0.5 * float(np.abs(hist[1] / hist[1].sum() - hist[0] / hist[0].sum()).sum())
    assert jump >= m.CUT_HISTOGRAM_ALONE  # the counterexample: the histogram alone would cut here
    assert [(s.start, s.end) for s in m.camera_reading(clip, 24.0).shots] == [(0, 24)]


# ----------------------------------------------------------------------------------- people

def test_a_person_filling_most_of_the_frame_is_left_out_of_the_vote():
    picture, masks, _ = big_figure()
    assert words(m.camera_path(picture, 24.0)) == "pans left"  # every point votes: the figure wins
    reading = m.camera_reading(picture, 24.0, people=masks)
    assert words(reading.path) == "static"
    assert reading.mask == "people"


def test_a_mask_that_leaves_too_little_to_track_reads_cant_tell():
    held = np.repeat(texture()[None], 24, axis=0)
    masks = np.ones((24, HEIGHT, WIDTH), dtype=bool)
    masks[:, :, :16] = False  # a strip of wall at the edge, too few points to stand on
    assert words(m.camera_path(held, 24.0)) == "static"
    assert words(m.camera_path(held, 24.0, people=masks)) == "can't tell"


def test_without_vision_the_near_part_of_the_depth_is_left_out():
    picture, _, depth = big_figure()
    profile = m.measure(depth, 24.0, picture=picture)
    assert profile.camera_path is not None and profile.camera_path["label"] == "static"
    assert profile.camera_mask == "depth"


def test_an_inverted_depth_is_read_from_its_dark_end():
    picture, _, depth = big_figure()
    inverted = m.measure(255 - depth, 24.0, picture=picture, near="dark")
    assert inverted.camera_path is not None and inverted.camera_path["label"] == "static"
    # Told the wrong end, it leaves the wall out and reads the figure: `near` is what decides.
    wrong = m.measure(255 - depth, 24.0, picture=picture, near="bright")
    assert wrong.camera_path is not None and wrong.camera_path["label"] == "pans left"


def test_a_ref_that_is_not_a_depth_leaves_nobody_out():
    picture, _, depth = big_figure()
    profile = m.measure(depth, 24.0, picture=picture, near=None)
    assert profile.camera_mask == "none"
    assert profile.camera_path is not None and profile.camera_path["label"] == "pans left"


def test_a_picture_read_as_its_own_depth_is_not_masked_by_itself():
    """`describe --rgb` hands the picture as both: its brightness says nothing about distance."""
    picture, _, _ = big_figure()
    assert m.measure(picture, 24.0, picture=picture).camera_mask == "none"


def test_vision_is_off_in_a_test_that_does_not_ask_for_it():
    from depthc import people

    assert people.available() is False
    assert people.find(np.zeros((1, 8, 8, 3), dtype=np.uint8)) is None


# ------------------------------------------------------------------------------ can't tell

def test_no_motion_carrying_half_the_points_is_cant_tell_not_still():
    clip = halves()
    assert words(m.camera_path(clip, 24.0)) == "can't tell"
    profile = m.measure(clip, 24.0, picture=clip)
    assert profile.camera_path is None
    assert profile.as_dict()["cameraShots"] == [{"startSeconds": 0.0, "endSeconds": 1.0, "label": None}]


# ------------------------------------------------------------------------------ slow moves

@pytest.mark.parametrize("rate, word", [(0.0005, "pushes in"), (-0.0005, "pulls out")])
def test_a_zoom_under_the_frame_to_frame_floor_is_still_named(rate, word):
    """About 0.024 grid px a frame: half the floor a frame-to-frame zoom is named from."""
    clip = zooming(rate, frames=72)
    profile = m.measure(clip, 24.0, picture=clip)
    assert profile.camera_path is not None
    assert m.PATH_RADIAL_SLOW < abs(profile.camera_path["radial"]) < m.PATH_RADIAL_STILL
    assert profile.camera_path["label"] == word


def test_beside_a_pan_a_small_zoom_is_not_named_from_the_slow_floor():
    """Real pans show a zoom of 0.01 to 0.05 of their own (DOCS/CAMERA-PATH-CALIBRATION.md)."""
    pan = m.path_label(0.3, 0.0, 0.0)
    assert m.path_label(0.3, 0.0, 0.03, radial_floor=m.PATH_RADIAL_SLOW) == pan
    assert m.path_label(0.0, 0.0, 0.03, radial_floor=m.PATH_RADIAL_SLOW) == "pushes in"
    assert m.path_label(0.0, 0.0, 0.03) == "static"


# ------------------------------------------------------------------- describe and compare

def test_describe_words_each_shot_on_its_own_and_lists_what_it_could_not_read():
    from depthc.describe import describe

    clip = np.concatenate([panned(texture(), 24, 2), darker(halves(24))])
    d = describe(m.measure(clip, 24.0, picture=clip))
    record = d.as_dict()
    assert [shot["label"] for shot in record["shots"]] == ["pans left", None]
    assert [(s["label"], s["start"], s["end"]) for s in record["cameraSegments"]] == [("pans left", 0.0, 1.0)]
    assert record["cameraUnread"] == [{"start": 1.0, "end": 2.0}]
    text = " ".join(d.sentences)
    assert "Shots: 2, cut at 1s" in text
    assert "Can't tell 1–2s" in text


def test_compare_gives_no_camera_word_where_it_could_not_read():
    from depthc import compare

    assert compare._path_summary(np.full((4, 3), np.nan))["label"] == "can't tell"
    ref = np.array([[0.3, 0.0, 0.0], [np.nan] * 3, [0.3, 0.0, 0.0]])
    gen = np.array([[0.3, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    assert compare._cosine(ref, gen) == pytest.approx(1.0)  # the unread bucket is left out of both


def test_a_kit_records_its_shots_and_the_schema_takes_them():
    import jsonschema

    from test_kit import SCHEMA_PATH, _manifest

    clip = np.concatenate([two_shots(), brighter(halves(24))])
    block = m.manifest_block(m.measure(clip, 24.0, picture=clip))
    assert [shot["label"] for shot in block["cameraShots"]] == ["pans left", "pans right", None]
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(_manifest(motion=block), schema)
    block["cameraShots"][0]["label"] = 3
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_manifest(motion=block), schema)


def test_a_profile_read_the_old_way_records_no_shots():
    """Kits built before this round, and paths read from the depth, carry no cameraShots at all."""
    _, _, depth = big_figure()
    assert "cameraShots" not in m.measure(depth, 24.0).as_dict()


def test_compare_knows_which_end_of_a_kits_ref_is_near(tmp_path):
    from depthc import cli, kit as kitlib

    def kit(invert: bool):
        folder = tmp_path / f"invert-{invert}"
        folder.mkdir(exist_ok=True)
        (folder / kitlib.KIT_MANIFEST).write_text(json.dumps({"settings": {"invert": invert}}))
        return folder

    assert cli._ref_near(kit(False), "") == "bright"
    assert cli._ref_near(kit(True), "") == "dark"
    assert cli._ref_near(kit(False), "invert") == "dark"
    assert cli._ref_near(kit(True), "invert") == "bright"
    assert cli._ref_near(kit(False), "edges") is None
    assert cli._ref_near(tmp_path / "no-kit", "") is None


# --------------------------------------------------------------------------- the scout

def _scout(counts: list[int] | None):
    profile = m.MotionProfile(fps=24.0, deltas=np.full(24 * 8, 3.0, dtype=np.float32))
    result = m.ScoutResult(
        profile=profile, windows=m.score_windows(profile, 5.0), per_second=list(profile.buckets),
        scout_fps=12.0, elapsed=1.0, target_seconds=5.0,
        people=None if counts is None else np.array(counts), people_fps=2.0,
    )
    for window in result.windows:
        window.figures = result.figures_between(window.start, window.end)
    return result


def test_the_scout_warns_when_a_second_figure_is_in_frame(capsys):
    from pathlib import Path

    from depthc import cli

    result = _scout([1] * 10 + [2] * 6)  # eight seconds counted twice a second; two people from 5 s
    assert result.second_figure == pytest.approx(6 / 16)
    by_start = {window.start: window for window in result.windows}
    assert "second figure" not in by_start[0.0].note()
    assert any("a second figure in" in window.note() for window in result.windows if window.start >= 1.0)
    cli._print_scout(Path("x.mp4"), result)
    out = capsys.readouterr().out
    assert "a second figure is in frame for 38% of the clip" in out
    assert "check sheet.png for a second figure walking into the window" not in out


def test_the_scout_says_what_it_counted_when_nobody_else_was_there(capsys):
    from pathlib import Path

    from depthc import cli

    cli._print_scout(Path("x.mp4"), _scout([1] * 16))
    out = capsys.readouterr().out
    assert "people counted: two or more in frame for 0% of the clip" in out
    assert "a second figure is in frame" not in out


def test_the_scout_keeps_its_old_warning_where_it_could_not_count(capsys):
    from pathlib import Path

    from depthc import cli

    result = _scout(None)
    assert result.second_figure is None and all(window.figures is None for window in result.windows)
    cli._print_scout(Path("x.mp4"), result)
    assert "check sheet.png for a second figure walking into the window" in capsys.readouterr().out


def test_scout_clip_counts_people_twice_a_second(clip_3s, monkeypatch):
    """The wiring, with the depth pass and Vision both stood in for, so it runs on CI too."""
    from depthc import engine, people

    def fake_infer(frames, fps, **_):
        return engine.DepthResult(
            frames=frames[..., 0], fps=fps, device="cpu", dtype="fp32", elapsed=0.0,
            raw_min=0.0, raw_max=1.0, inverted=False,
        )

    seen = {}

    def find(picture, size=None):
        seen["frames"] = picture.shape[0]
        return people.People(masks=np.zeros((picture.shape[0], 1, 1), dtype=bool),
                             counts=np.full(picture.shape[0], 2))

    monkeypatch.setattr(engine, "infer", fake_infer)
    monkeypatch.setattr(people, "find", find)
    result = m.scout_clip(clip_3s, 1.0)
    assert seen["frames"] == 6  # three seconds at 12 fps, one frame in six
    assert result.second_figure == 1.0
    assert result.windows and all(window.figures == 1.0 for window in result.windows)


# ------------------------------------------------------------------------ Vision, for real

@pytest.mark.vision
def test_vision_finds_nobody_on_a_textured_wall():
    """The instance mask alone outlines SOMETHING on a picture with nobody in it (it covered a
    flower on seen footage); the body detector's boxes are what keep that out."""
    from depthc import people

    clip = np.repeat(np.repeat(texture()[None], 3, axis=0)[..., None], 3, axis=-1)
    found = people.find(clip, size=(64, 36))
    assert found is not None
    assert found.masks.shape == (3, 36, 64)
    assert not found.counts.any() and not found.masks.any()


@pytest.mark.vision
def test_with_vision_the_reader_says_so_and_a_bare_pan_reads_the_same():
    clip = panned(texture(), 24, 2)
    reading = m.camera_reading(clip, 24.0)
    assert reading.mask == "people"
    assert np.array_equal(reading.path, m.camera_reading(clip, 24.0, people=None).path)
