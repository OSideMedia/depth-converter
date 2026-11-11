"""The camera is read from the picture, not from the depth (`motion.camera_path`).

The failure these pin was scored on real footage (DOCS/CAMERA-PATH-CALIBRATION.md, "Real
footage"): a locked-off shot with people walking through was named a pan, because in a depth
picture the walkers are nearly the only thing with an edge. Each fixture here is a pair, the
picture and the depth of the same frames, and the first test runs the old read through the same
assertion to show the assertion can fail.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from depthc import motion as m

HEIGHT, WIDTH, FRAMES = 180, 320, 36


def texture(seed: int = 7, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """Seeded, smoothed noise: localised features that do not repeat (see test_motion.textured)."""
    noise = np.random.default_rng(seed).random((height, width)) * 255
    return cv2.GaussianBlur(noise.astype(np.uint8), (0, 0), 3)


def walkers(frames: int = FRAMES, speed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A locked-off shot with four figures crossing it: (picture, depth), frame for frame.

    The picture is a textured wall that never moves, with textured figures sliding across it. The
    depth is what a depth model makes of that: one flat far wall, and each figure a near, bright
    slab. So in the depth nothing has an edge except the figures.
    """
    wall = texture()
    skin = texture(seed=11, height=70, width=30)
    picture = np.repeat(wall[None], frames, axis=0).copy()
    depth = np.full((frames, HEIGHT, WIDTH), 40, dtype=np.uint8)
    for index in range(frames):
        for row, start in ((10, 0), (55, 60), (100, 120), (30, 180)):
            x = start + index * speed
            picture[index, row:row + 70, x:x + 30] = skin
            depth[index, row:row + 70, x:x + 30] = 200
    return picture, depth


def words(path: np.ndarray) -> str:
    """The move a path names over its moving gaps, the way MotionProfile.camera_path words it."""
    moving = np.any(path != 0, axis=1)
    if not moving.any():
        return "static"
    return m.path_label(*(float(v) for v in path[moving].mean(axis=0)))


def moving_share(path: np.ndarray) -> float:
    return float(np.any(path != 0, axis=1).mean())


def zooming(rate: float, frames: int = FRAMES) -> np.ndarray:
    base = texture()
    out = []
    for index in range(frames):
        matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 0.0, 1.0 + rate * index)
        out.append(cv2.warpAffine(base, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


# ------------------------------------------------------------------ the failure, and its fix

def test_walkers_through_a_locked_off_shot_name_no_camera_move():
    picture, depth = walkers()
    # The counterexample, through the same predicate: read from the depth, the walkers ARE the
    # camera. If this stops failing the fixture has stopped being the real-footage failure.
    assert words(m.flow_series(depth, 24.0).path) != "static"
    assert words(m.camera_path(picture, 24.0)) == "static"


def test_measure_reads_the_path_from_the_picture_and_the_rest_from_the_depth():
    picture, depth = walkers()
    alone = m.measure(depth, 24.0)
    paired = m.measure(depth, 24.0, picture=picture)
    assert alone.camera_path is not None and alone.camera_path["label"] != "static"
    assert paired.camera_path is not None and paired.camera_path["label"] == "static"
    # Energy, regions and the camera share are still the depth's: only the path moved.
    assert paired.energy == alone.energy
    assert paired.camera == alone.camera
    assert np.array_equal(paired.deltas, alone.deltas)


def test_measure_refuses_a_picture_that_is_not_frame_for_frame():
    picture, depth = walkers()
    with pytest.raises(ValueError, match="same frames"):
        m.measure(depth, 24.0, picture=picture[:-3])


def test_no_picture_means_the_path_is_read_as_before():
    _, depth = walkers()
    assert np.array_equal(m.measure(depth, 24.0).path, m.flow_series(depth, 24.0).path)


# ------------------------------------------------------------------------- real moves survive

def test_a_pan_keeps_its_direction_with_walkers_in_frame():
    """The wall slides right under the walkers: the camera pans left, whatever they do."""
    picture, _ = walkers()
    wall = texture()
    panned = np.stack([
        np.where(picture[i] == wall, np.roll(wall, i * 2, axis=1), picture[i]) for i in range(FRAMES)
    ])
    path = m.camera_path(panned, 24.0)
    assert words(path) == "pans left"
    assert moving_share(path) > 0.8


def test_a_push_reads_as_a_push():
    assert "pushes in" in words(m.camera_path(zooming(0.012), 24.0))
    assert "pans" not in words(m.camera_path(zooming(0.012), 24.0))


def test_a_slow_push_the_region_reader_loses_is_still_a_push():
    """A push too slow for dense flow on the analysis grid, on every gap or nearly."""
    clip = zooming(0.0012, frames=72)
    assert moving_share(m.flow_series(clip, 24.0).path) < 0.5  # the old read, same predicate
    path = m.camera_path(clip, 24.0)
    assert moving_share(path) > 0.9
    assert words(path) == "pushes in"  # it clears the naming floor, not merely zero


def test_a_short_pan_is_not_averaged_away_by_the_still_shot_after_it():
    """Half a second of pan, then seven seconds held, with a little sensor noise on every frame.
    Tracking noise on the held gaps must read as still, or it dilutes the pan to nothing."""
    base = texture()
    rng = np.random.default_rng(3)
    clip = np.stack([
        np.clip(np.roll(base, min(i, 12) * 2, axis=1).astype(int)
                + rng.choice([-1, 0, 1], base.shape, p=[0.005, 0.99, 0.005]), 0, 255).astype(np.uint8)
        for i in range(180)
    ])
    path = m.camera_path(clip, 24.0)
    assert words(path) == "pans left"
    assert moving_share(path) < 0.15
    profile = m.measure(clip, 24.0, picture=clip)
    assert profile.camera_path is not None and profile.camera_path["label"] == "pans left"


def test_a_held_shot_reads_as_exact_zeros():
    rng = np.random.default_rng(5)
    base = texture()
    clip = np.stack([
        np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8) for _ in range(48)
    ])
    assert not np.any(m.camera_path(clip, 24.0))


# ------------------------------------------------------------------------------ the two exits

def test_a_frame_with_nothing_to_track_is_read_by_the_region_reader():
    """A soft ramp has no corners: every gap falls back, so the two readers agree to the digit."""
    ramp = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    clip = np.stack([np.roll(ramp, i, axis=1) for i in range(12)])
    grey = m._track_gray(clip)
    points = cv2.goodFeaturesToTrack(grey[0], m.TRACK_MAX_POINTS, m.TRACK_QUALITY, m.TRACK_SPACING)
    assert points is None or len(points) < m.TRACK_MIN_POINTS
    assert np.array_equal(m.camera_path(clip, 24.0), m.flow_series(clip, 24.0).path)


def test_two_halves_moving_apart_name_no_camera():
    """No single motion carries half the points, so nothing in the frame is the world."""
    base = texture()
    half = WIDTH // 2
    clip = np.stack([
        np.concatenate([np.roll(base[:, :half], -3 * i, axis=1), np.roll(base[:, half:], 3 * i, axis=1)], axis=1)
        for i in range(24)
    ])
    assert moving_share(m.camera_path(clip, 24.0)) < 0.2


def test_the_same_clip_reads_the_same_twice():
    picture, _ = walkers()
    wall = texture()
    panned = np.stack([
        np.where(picture[i] == wall, np.roll(wall, i * 2, axis=1), picture[i]) for i in range(FRAMES)
    ])
    cv2.setRNGSeed(12345)  # whatever ran before must not change the answer
    first = m.camera_path(panned, 24.0)
    cv2.setRNGSeed(999)
    assert np.array_equal(first, m.camera_path(panned, 24.0))


def test_colour_and_grey_pictures_read_alike():
    clip = zooming(0.012)
    rgb = np.repeat(clip[..., None], 3, axis=-1)
    assert np.allclose(m.camera_path(rgb, 24.0), m.camera_path(clip, 24.0), atol=1e-3)


def test_a_single_frame_has_no_path():
    assert m.camera_path(texture()[None], 24.0).shape == (0, 3)


def test_gaps_the_region_reader_filled_count_as_motion_around_a_tracked_gap():
    """A pan read half by tracking and half by the region reader is one pan. Judged against holes
    where the fallback belongs, every tracked gap would average under the floor and be dropped."""
    gaps = 48
    path = np.zeros((gaps, 3), dtype=np.float32)
    path[:, 0] = 0.15
    keep = np.zeros(gaps, dtype=bool)
    keep[1::2] = True
    assert np.array_equal(m._settle(path, 24.0, keep), path)
    holes = path.copy()
    holes[1::2] = 0.0  # the same predicate on the order that loses it
    assert not np.any(m._settle(holes, 24.0)[0::2])


def test_settling_never_zeroes_a_gap_it_was_told_to_keep():
    path = np.zeros((24, 3), dtype=np.float32)
    path[5] = (0.02, 0.0, 0.0)
    keep = np.zeros(24, dtype=bool)
    keep[5] = True
    assert m._settle(path, 24.0, keep)[5, 0] == pytest.approx(0.02)
    assert not np.any(m._settle(path, 24.0))


def test_settling_a_clip_shorter_than_its_window_or_off_24fps():
    slow = np.tile(np.array([[0.0, 0.0, 0.06]], dtype=np.float32), (5, 1))
    assert np.array_equal(m._settle(slow, 24.0), slow)
    assert np.array_equal(m._settle(np.zeros((0, 3), dtype=np.float32), 24.0), np.zeros((0, 3), dtype=np.float32))
    pan = np.tile(np.array([[0.2, 0.0, 0.0]], dtype=np.float32), (120, 1))
    assert np.array_equal(m._settle(pan, 60.0), pan)


# ---------------------------------------------------------------------------------- fit_picture

def test_fit_picture_cuts_or_holds_within_the_slack():
    clip = np.arange(10, dtype=np.uint8)[:, None, None] * np.ones((1, 4, 4), dtype=np.uint8)
    assert m.fit_picture(clip, 10) is clip or np.array_equal(m.fit_picture(clip, 10), clip)
    assert m.fit_picture(clip, 9).shape[0] == 9
    held = m.fit_picture(clip, 12)
    assert held.shape[0] == 12 and np.array_equal(held[-1], clip[-1]) and np.array_equal(held[-2], clip[-1])


def test_fit_picture_refuses_a_different_clip():
    clip = np.zeros((10, 4, 4), dtype=np.uint8)
    assert m.fit_picture(clip, 10 + m.PICTURE_SLACK + 1) is None
    assert m.fit_picture(clip, 10 - m.PICTURE_SLACK - 1) is None
    assert m.fit_picture(clip[:0], 1) is None
