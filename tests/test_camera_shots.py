"""The camera round (2026-11-11): shots, people, "can't tell", slow moves, and the scout's count.

The second real-footage score (DOCS/CAMERA-PATH-CALIBRATION.md) failed four ways: a fit read
across cuts, dancers filling the frame outvoted the wall, a crowd was read as a move instead of
nothing, and slow pushes stayed under the floor. Each fixture here is a synthetic picture of one
of those, and each test was run against the reader before this round and failed there. Vision is
off in every test not marked `vision` (tests/conftest.py), so a person mask is handed in.

Camera words never vote in any verdict; nothing here tests a pass or a fail, only what is said.
"""

from __future__ import annotations

import functools
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

    blind = m.MotionProfile(fps=24.0, deltas=np.zeros(48, np.float32), path=np.full((48, 3), np.nan))
    assert compare._path_word(blind, 2.0)["label"] == "can't tell"
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


# ------------------------------------------------------------- the review of this round
# Findings from two reviews of the round's diff; each test below failed on the code it reviewed.
# The rule they hold the reader to is ONE CAMERA, READ PER SHOT (depthc/motion.py): every camera
# word is derived from the per-shot read in cameraShots, so none can contradict it.

def hold(base: np.ndarray, frames: int) -> np.ndarray:
    return np.repeat(base[None], frames, axis=0)


def hold_then_pan(base: np.ndarray, held: int, moving: int, step: int = 2) -> np.ndarray:
    """One shot: the wall held, then sliding `step` px a frame."""
    return np.concatenate([hold(base, held), np.stack([np.roll(base, (i + 1) * step, axis=1) for i in range(moving)])])


def pan_then_hold(base: np.ndarray, moving: int, held: int, step: int = 2) -> np.ndarray:
    return hold_then_pan(base, held, moving, step)[::-1].copy()


def pan_and_back(base: np.ndarray, moving: int, step: int = 2) -> np.ndarray:
    """One shot: the wall sliding `moving` frames one way and `moving` frames back to where it began.
    Moving throughout, and read as still: the shot is one word, the mean of its moves, and they
    cancel (the limit written beside ONE CAMERA)."""
    out = [np.roll(base, i * step, axis=1) for i in range(moving + 1)]
    out += [np.roll(base, (moving - i) * step, axis=1) for i in range(1, moving + 1)]
    return np.stack(out)


def cut_together(*shots: np.ndarray) -> np.ndarray:
    """Shots on their own walls and in turn at three brightnesses, so every join is a cut."""
    looks = (lambda frames: frames, brighter, darker)
    return np.concatenate([looks[index % 3](shot) for index, shot in enumerate(shots)])


def generated_clips(fps: float):
    """Hold, pan and cut combinations at `fps`, Codex's cases among them."""
    s = int(round(fps))
    wall = [texture(seed=seed) for seed in (7, 23, 31, 41)]
    for frames in range(4, 13):  # a held second, then a short pan (the first review)
        yield f"hold+pan{frames}", cut_together(hold(wall[0], s), panned(wall[1], frames, 2))
    for moving in range(3, 10):  # a held second, then a shot that holds and then pans briefly
        yield f"hold+hold-pan{moving}", cut_together(hold(wall[0], s), hold_then_pan(wall[1], s // 2 + 1, moving))
    held, moving = (13, 5) if s == 24 else (16, 7)  # Codex: the remainder describe dropped
    yield "codex-remainder", cut_together(hold(wall[0], s), hold_then_pan(wall[1], held, moving))
    outer, middle = (8, 9) if s == 24 else (10, 11)  # Codex: the longest shot owns no bucket
    for step in (2, -2):
        yield f"codex-three-pans{step}", cut_together(
            panned(wall[0], outer, step), panned(wall[1], middle, -step), panned(wall[2], outer, step))
    for frames in (6, 7, 9, 13, 20):
        yield f"pan{frames}+hold", cut_together(panned(wall[0], frames, -2), hold(wall[1], s))
        yield f"pan-hold{frames}", cut_together(pan_then_hold(wall[2], frames, s // 2))
    yield "hold", hold(wall[0], s * 2)
    yield "pan", panned(wall[0], s, 2)
    yield "pan+pan", cut_together(panned(wall[0], s, 2), panned(wall[1], s + 1, -2))
    for moving in (s // 2, s // 2 + 3, s):  # moving inside a shot that reads still
        yield f"pan-back{moving}", pan_and_back(wall[3], moving)
        yield f"hold+pan-back{moving}", cut_together(hold(wall[0], s), pan_and_back(wall[1], moving))


@functools.lru_cache(maxsize=None)
def _measured(fps: float) -> tuple:
    """Every generated clip measured once, with takes cut from its start to compare it against:
    (name, profile, ((frames, take profile), ...)). A take too short to read, one second, and half
    the clip, wherever each is shorter than the clip."""
    out = []
    for name, clip in generated_clips(fps):
        lengths = sorted({m.shot_min_gaps(fps), int(round(fps)), clip.shape[0] // 2})
        takes = tuple(
            (frames, m.measure(clip[:frames], fps, picture=clip[:frames]))
            for frames in lengths if 2 <= frames < clip.shape[0]
        )
        out.append((name, m.measure(clip, fps, picture=clip), takes))
    return tuple(out)


def _main_word(profile) -> str:
    from depthc.describe import UNREAD

    main = m.main_shot(profile.shots)
    return UNREAD if main is None else (profile.camera_shots[main]["label"] or UNREAD)


def _broken(fps: float) -> list[tuple[str, str]]:
    """Every place a camera word contradicts cameraShots on the generated clips, as (clip, what).

    describe's line and segments, the kit's cameraPath and compare's PATH words: a shot named a
    move has that word inside it in describe and no "locked off" or "can't tell" for the clip; a
    shot read as still has no move inside it; an unread shot is "can't tell"; the one word for a
    clip is its main shot's. Compared with a shorter take, both sides are worded over the overlap,
    the stretch compared: what cameraShots says of the clip cut to the take's length.
    """
    from depthc import compare
    from depthc.describe import describe

    broken = []
    for name, profile, takes in _measured(fps):
        for label, one in [(name, profile)] + [(f"{name}[:{frames}]", take) for frames, take in takes]:
            shots = one.camera_shots
            d = describe(one)
            record, text = d.as_dict(), " ".join(d.sentences)
            for shot in shots:
                said = [s["label"] for s in _segments_inside(record, shot, "cameraSegments")]
                if shot["label"] is None and not (_segments_inside(record, shot, "cameraUnread") and not said):
                    broken.append((label, f"unread shot {shot} worded {said}"))
                elif shot["label"] == "static" and not set(said) <= {"static"}:
                    broken.append((label, f"still shot {shot} worded {said}"))
                elif shot["label"] not in (None, "static") and shot["label"] not in said:
                    broken.append((label, f"moving shot {shot} worded {said}"))
            if any(shot["label"] not in (None, "static") for shot in shots):
                if "locked off" in text or "Camera: can't tell" in text:
                    broken.append((label, f"a shot moved and describe said: {text}"))
            word = _main_word(one)
            kit = (m.manifest_block(one).get("cameraPath") or {}).get("label", "can't tell")
            if kit != word:
                broken.append((label, f"kit cameraPath {kit}, main shot {word}"))
            itself = compare.compare_profiles(one, one, regions=False)
            if (itself.ref_path or {}).get("label") != word:
                broken.append((label, f"compare with itself {itself.ref_path}, main shot {word}"))
        for frames, take in takes:
            word = _main_word(take)
            for ref, gen in ((profile, take), (take, profile)):
                result = compare.compare_profiles(ref, gen, regions=False)
                said = {(result.ref_path or {}).get("label"), (result.gen_path or {}).get("label")}
                if said != {word}:
                    broken.append((f"{name}[:{frames}]", f"compare over the overlap said {said}, cut there {word}"))
    return broken


@pytest.mark.parametrize("fps", [24.0, 30.0])
def test_no_camera_word_ever_contradicts_the_shot_read(fps):
    """For every generated clip and every take cut from it: no camera word contradicts cameraShots
    (`_broken`, above)."""
    assert _broken(fps) == []


def test_the_shot_read_check_goes_red_when_bucket_words_stop_following_the_shot(monkeypatch):
    """The check above is only as good as what it can see: with `follow_shot` made to change
    nothing, it must fail, on the shots that move and read still above all."""
    import importlib

    monkeypatch.setattr(importlib.import_module("depthc.describe"), "follow_shot", lambda buckets, word: buckets)
    broken = {name for name, _ in _broken(24.0)}
    assert {"pan-back12", "pan-back15", "pan-back24", "hold+pan-back12"} <= broken, sorted(broken)


def test_bucket_words_follow_the_shot_read():
    """The shot read decides whether a shot moved; its buckets only say when (`follow_shot`)."""
    from depthc.describe import UNREAD, follow_shot

    zero = (0.0, 0.0, 0.0)
    still, unread = ("static", 0.0, zero), (UNREAD, 0.0, zero)
    left = ("pans left", 0.5, (0.5, 0.0, 0.0))
    stirred = ("static", 0.04, (0.04, 0.0, 0.0))  # some motion, under the floor on its own
    word = {"dx": 0.3, "dy": 0.0, "radial": 0.0, "label": "pans left"}
    named = ("pans left", 0.3, (0.3, 0.0, 0.0))
    held = {"dx": 0.0, "dy": 0.0, "radial": 0.0, "label": "static"}
    assert follow_shot([still, left, unread], held) == [still, still, unread]
    assert follow_shot([still, left, unread], word) == [still, left, unread]
    assert follow_shot([still, stirred, unread], word) == [still, named, unread]
    assert follow_shot([still, still, unread], word) == [named, named, unread]
    assert follow_shot([still, left], None) == [unread, unread]


def _segments_inside(record: dict, shot: dict, key: str) -> list[dict]:
    return [s for s in record[key] if shot["startSeconds"] - 0.01 <= s["start"] and s["end"] <= shot["endSeconds"] + 0.01]


def test_compare_never_averages_a_bucket_across_a_cut():
    """Seven frames panning left, a cut, a pan right: the first half-second holds six gaps of the
    one and five of the other, and averaged together they named no move (dx about 0.09)."""
    from depthc import compare
    from depthc.describe import path_buckets

    clip = np.concatenate([panned(texture(), 7, 2), brighter(panned(texture(seed=23), 41, -2))])
    profile = m.measure(clip, 24.0, picture=clip)
    assert [(s.start, s.end) for s in profile.shots] == [(0, 7), (7, 48)]
    series = compare._path_series(profile, 2.0)
    assert m.path_label(*series[0]) == "pans left"
    assert {m.path_label(*row) for row in series[1:]} == {"pans right"}
    assert path_buckets(profile.path, 24.0, shots=profile.shots)[0][0] == "pans left"


def test_a_clip_with_cuts_is_summed_up_by_one_shot_never_an_average_of_two():
    """A pan left of 25 frames cut against a pan right of 26: across the cut they cancel. The kit's
    cameraPath, describe's overall line and compare's PATH line all name the longer shot."""
    from depthc import compare

    clip = np.concatenate([panned(texture(), 25, 2), brighter(panned(texture(seed=23), 26, -2))])
    profile = m.measure(clip, 24.0, picture=clip)
    assert [shot["label"] for shot in profile.camera_shots] == ["pans left", "pans right"]
    assert profile.camera_path is not None and profile.camera_path["label"] == "pans right"
    result = compare.compare_profiles(profile, profile)
    assert result.ref_path["label"] == "pans right"
    assert result.path_judged and result.path_cos == pytest.approx(1.0)


def test_compare_words_a_shot_too_short_for_a_bucket():
    """Codex, third read: a seven-frame pan at 24 fps is six gaps, long enough to read, and has no
    half-second bucket; compared with itself, it lost its PATH words along with the cosine. The
    words never wait on a bucket; only the cosine does."""
    from depthc import compare

    clip = panned(texture(), 7, 2)
    profile = m.measure(clip, 24.0, picture=clip)
    assert [shot["label"] for shot in profile.camera_shots] == ["pans left"]
    result = compare.compare_profiles(profile, profile)
    assert result.ref_path["label"] == result.gen_path["label"] == "pans left"
    assert result.path_cos is None
    line = [line for line in compare._render_structure(result, "", "", "", "") if "PATH" in line]
    assert line and line[0].startswith("  PATH     ref pans left · gen pans left"), line


def test_compare_words_the_overlap_as_the_clip_cut_there():
    """Codex, third read: against a take shorter than the reference, compare words the reference
    over the overlap only. That is the rule, not an accident (ONE CAMERA, depthc/motion.py): the
    overlap is the stretch compared, read as if the clip ended there, its shots cut short and read
    by the same gates as whole shots. So it can differ from the kit's cameraPath for the whole
    clip, and a stretch too short to read a shot in is "can't tell", as the take itself is."""
    from depthc import compare
    from depthc.describe import UNREAD

    clip = np.concatenate([panned(texture(), 25, 2), brighter(panned(texture(seed=23), 72, -2))])
    ref = m.measure(clip, 24.0, picture=clip)
    assert [shot["label"] for shot in ref.camera_shots] == ["pans left", "pans right"]
    assert ref.camera_path["label"] == "pans right"
    for frames, word in ((24, "pans left"), (27, "pans left"), (6, UNREAD)):
        take = m.measure(clip[:frames], 24.0, picture=clip[:frames])
        assert (take.camera_path or {"label": UNREAD})["label"] == word, frames
        result = compare.compare_profiles(ref, take)
        assert result.ref_path["label"] == result.gen_path["label"] == word, (frames, result.ref_path)


def test_a_stretch_mostly_unread_inside_a_shot_read_whole_is_cant_tell():
    """The overlap's shots pass the same gates as whole shots: SHOT_MIN_READ of the kept gaps."""
    from depthc import compare
    from depthc.describe import UNREAD

    path = np.tile(np.array([[0.3, 0.0, 0.0]]), (24, 1))
    path[:12] = np.nan  # half of the shot unread: the whole shot is still read
    profile = m.MotionProfile(fps=24.0, deltas=np.zeros(24, np.float32), path=path, shots=[m.Shot(0, 25)])
    assert profile.camera_path["label"] == m.path_label(0.3, 0.0, 0.0)
    assert compare._path_word(profile, 17 / 24.0)["label"] == UNREAD  # 4 of its 16 gaps read
    assert compare._path_word(profile, 25 / 24.0)["label"] == m.path_label(0.3, 0.0, 0.0)  # all 24


def _fake_vision(monkeypatch, boxes: list[tuple[float, float, float, float]], outline: np.ndarray | None):
    """Vision, Quartz, objc and Foundation faked: the body detector returns `boxes` (x, y, w, h,
    normalised, origin at the bottom left) and the instance mask `outline` (a label image), or
    no outline at all. So the matching of outlines to bodies runs everywhere, CI included."""
    import contextlib
    import sys
    import types

    from depthc import people

    def body(x: float, y: float, w: float, h: float):
        shape = types.SimpleNamespace(origin=types.SimpleNamespace(x=x, y=y),
                                      size=types.SimpleNamespace(width=w, height=h))
        return types.SimpleNamespace(boundingBox=lambda: shape)

    class Address:
        def __init__(self, raw: bytes):
            self.raw = raw

        def as_buffer(self, length: int) -> bytes:
            return self.raw[:length]

    observation = types.SimpleNamespace(instanceMask=lambda: outline)

    class Request:
        found: list = []

        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def results(self):
            return self.found

        def setUpperBodyOnly_(self, flag):
            pass

    class Outline(Request):
        found = [] if outline is None else [observation]

    class Bodies(Request):
        found = [body(*b) for b in boxes]

    class Handler:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithData_options_(self, data, options):
            return self

        def performRequests_error_(self, requests, error):
            return True, None

    vision = types.ModuleType("Vision")
    vision.VNGeneratePersonInstanceMaskRequest = Outline
    vision.VNDetectHumanRectanglesRequest = Bodies
    vision.VNImageRequestHandler = Handler
    quartz = types.ModuleType("Quartz")
    quartz.kCVPixelBufferLock_ReadOnly = 1
    quartz.CVPixelBufferLockBaseAddress = lambda buffer, flags: 0
    quartz.CVPixelBufferUnlockBaseAddress = lambda buffer, flags: 0
    quartz.CVPixelBufferGetHeight = lambda buffer: buffer.shape[0]
    quartz.CVPixelBufferGetWidth = lambda buffer: buffer.shape[1]
    quartz.CVPixelBufferGetBytesPerRow = lambda buffer: buffer.shape[1]
    quartz.CVPixelBufferGetBaseAddress = lambda buffer: Address(np.ascontiguousarray(buffer).tobytes())
    objc = types.ModuleType("objc")
    objc.autorelease_pool = contextlib.nullcontext
    foundation = types.ModuleType("Foundation")
    foundation.NSData = types.SimpleNamespace(dataWithBytes_length_=lambda data, length: data)
    for name, module in (("Vision", vision), ("objc", objc), ("Foundation", foundation), ("Quartz", quartz)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(people, "available", lambda: True)
    return people


BOTH = [(0.05, 0.1, 0.3, 0.8), (0.6, 0.1, 0.3, 0.8)]  # two bodies, left and right, 24% of the frame each


def test_two_bodies_with_no_outline_are_counted_as_two(monkeypatch):
    """Vision found two bodies and drew no outline: each box is its own person. As one union they
    counted one, and the scout's second-figure warning never fired."""
    people = _fake_vision(monkeypatch, BOTH, None)
    found = people.find(np.zeros((1, 90, 160, 3), dtype=np.uint8))
    assert found is not None
    assert found.counts.tolist() == [2]
    assert found.masks[0].mean() == pytest.approx(0.48, abs=0.03)


def test_a_body_the_outline_missed_is_masked_by_its_box(monkeypatch):
    """Two bodies, an outline for the left one only: the right body is masked by its box and counted,
    so its motion does not vote and the second-figure warning still fires. The outline drawn at
    half size, as Vision draws it at its own."""
    outline = np.zeros((45, 80), dtype=np.uint8)
    outline[8:38, 6:26] = 1  # inside the left box
    people = _fake_vision(monkeypatch, BOTH, outline)
    found = people.find(np.zeros((1, 90, 160, 3), dtype=np.uint8))
    assert found is not None
    assert found.counts.tolist() == [2]
    right = found.masks[0][:, 100:]
    assert right.mean() > 0.4  # the right body is masked
    assert found.masks[0][16:76, 12:52].all() and not found.masks[0][:, 56:96].any()


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
