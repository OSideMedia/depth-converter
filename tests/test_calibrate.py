"""The camera-path ground truth must name each pure camera move the way path_label would.

Synthetic tracks, no Blender: a camera 1.6 m up looking along +Y, tilted 10° down
(Blender euler X = 80°), 35 mm on a 36 mm sensor, 1280x720 → the 160x90 analysis grid.
Each test moves ONE thing per frame and asserts the truth's words — the sign conventions
(image rows run down, Blender ndc runs up; pixels moving right = the camera panned left)
are exactly where a truth like this goes silently wrong.
"""

import json

import numpy as np
import pytest

from depthc import calibrate
from depthc.describe import path_buckets

GRID_W, GRID_H = 160, 90
GROUND_HALF = 200.0


def _track(n=24, loc=lambda i: (0.0, 0.0, 1.6), rot=lambda i: (80.0, 0.0, 0.0), lens=35.0):
    poses = [
        calibrate.Pose(calibrate.euler_xyz(rot(i)), np.asarray(loc(i), dtype=np.float64), lens)
        for i in range(n)
    ]
    return calibrate.Track(poses=poses, sensor_w=36.0, aspect=16 / 9, resolution=(1280, 720))


def _words(track):
    path, share = calibrate.truth_path(track, GRID_W, GRID_H, GROUND_HALF)
    buckets = path_buckets(path, 24.0)
    labels = {label for label, _, _ in buckets}
    assert len(labels) == 1, f"a constant move must read as one label, got {labels}"
    return labels.pop(), path, share


def test_static_camera_reads_static():
    label, path, share = _words(_track())
    assert label == "static"
    assert np.allclose(path, 0.0)
    assert 0.5 < share < 1.0  # a 10° down-tilt sees mostly ground and a band of sky


@pytest.mark.parametrize("step, expected", [(+0.5, "pans left"), (-0.5, "pans right")])
def test_yaw_is_a_pan(step, expected):
    label, path, _ = _words(_track(rot=lambda i: (80.0, 0.0, step * i)))
    assert label == expected
    assert np.allclose(path[:, 2], 0.0), "a pure rotation must carry no radial term"


@pytest.mark.parametrize("step, expected", [(-0.05, "pans left"), (+0.05, "pans right")])
def test_truck_reads_as_the_pan_it_looks_like(step, expected):
    # truck LEFT (x decreasing) slides the picture right — the same pixels as a pan left
    label, _, _ = _words(_track(loc=lambda i: (step * i, 0.0, 1.6)))
    assert label == expected


@pytest.mark.parametrize("step, expected", [(+0.05, "pushes in"), (-0.05, "pulls out")])
def test_forward_travel_is_radial(step, expected):
    forward = calibrate.euler_xyz((80.0, 0.0, 0.0)) @ np.array([0.0, 0.0, -1.0])
    label, path, _ = _words(_track(loc=lambda i: tuple(np.array([0.0, 0.0, 1.6]) + forward * step * i)))
    assert expected in label
    assert np.sign(path[:, 2]).tolist() == [np.sign(step)] * len(path)


def test_the_truth_reads_its_radial_around_the_median():
    # A regression pin, not a derivation: the ground below centre gives a push's flow a
    # downward median, and the truth takes the radial on the flow AROUND its median.
    # Read on the raw flow, this push books 0.357 (+3%). The pixel-index centre (-0.5) is
    # NOT pinned: on this symmetric grid it moves the radial by under 0.01%.
    forward = calibrate.euler_xyz((80.0, 0.0, 0.0)) @ np.array([0.0, 0.0, -1.0])
    track = _track(n=6, loc=lambda i: tuple(np.array([0.0, 0.0, 1.6]) + forward * 0.05 * i))
    path, _ = calibrate.truth_path(track, GRID_W, GRID_H, GROUND_HALF)
    assert float(path[:, 2].mean()) == pytest.approx(0.3465, rel=0.01)


def test_a_lens_zoom_is_radial_like_the_push_it_looks_like():
    # The camera holds still and the lens lengthens: the picture expands about its centre,
    # which is what flow_series reads and path_label calls "pushes in". A truth built from
    # camera travel alone books it static.
    path, _ = calibrate.truth_path(_zoom_track(), GRID_W, GRID_H, GROUND_HALF)
    labels = [label for label, _, _ in path_buckets(path, 24.0)]
    assert labels and all("pushes in" in label for label in labels), labels


def _zoom_track(n=24, start=35.0, step=0.4):
    poses = [calibrate.Pose(calibrate.euler_xyz((80.0, 0.0, 0.0)), np.array([0.0, 0.0, 1.6]), start + step * i)
             for i in range(n)]
    return calibrate.Track(poses=poses, sensor_w=36.0, aspect=16 / 9, resolution=(1280, 720))


@pytest.mark.parametrize("label_for, loc, rot", [
    ("tilt up", lambda i: (0.0, 0.0, 1.6), lambda i: (80.0 + 0.3 * i, 0.0, 0.0)),
    ("crane up", lambda i: (0.0, 0.0, 1.6 + 0.03 * i), lambda i: (80.0, 0.0, 0.0)),
])
def test_up_moves_slide_the_picture_down(label_for, loc, rot):
    label, path, _ = _words(_track(loc=loc, rot=rot))
    assert label == "tilts up", label_for
    assert (path[:, 1] > 0).all(), "pixels moving DOWN is what path_label calls a tilt up"


def _ground_depth(track, i):
    """Frame i's full-resolution planar depth of the ground plane — what a Z pass would hold."""
    width, height = track.resolution
    pose = track.poses[i]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    local = calibrate._rays(track, pose, xs + 0.5, ys + 0.5, width, height)
    dz = (local @ pose.rotation.T)[..., 2]
    with np.errstate(divide="ignore"):
        z = np.where(dz < -1e-9, -pose.location[2] / dz, np.nan)  # planar: local z is -1 per unit
    points_x = pose.location[0] + (local @ pose.rotation.T)[..., 0] * z
    points_y = pose.location[1] + (local @ pose.rotation.T)[..., 1] * z
    z[(np.abs(points_x) > GROUND_HALF) | (np.abs(points_y) > GROUND_HALF)] = np.nan
    return z


@pytest.mark.parametrize("loc, rot", [
    (lambda i: (0.05 * i, 0.0, 1.6), lambda i: (80.0, 0.0, 0.0)),                 # truck
    (lambda i: (0.0, 0.0, 1.6), lambda i: (80.0, 0.0, 0.5 * i)),                  # pan
    (lambda i: tuple(np.array([0.0, 0.0, 1.6])                                   # push
                     + calibrate.euler_xyz((80.0, 0.0, 0.0)) @ np.array([0.0, 0.0, -0.05 * i])),
     lambda i: (80.0, 0.0, 0.0)),
])
def test_depth_truth_agrees_with_the_ground_plane_it_rendered(loc, rot):
    # The rendered-clip path (a Z pass) and the analytic path must tell the same story on a
    # world that is only ground — this is the check that the unprojection is planar-correct.
    track = _track(n=6, loc=loc, rot=rot)
    by_depth, share_d = calibrate.truth_path(track, GRID_W, GRID_H, depth=lambda i: _ground_depth(track, i))
    by_plane, share_p = calibrate.truth_path(track, GRID_W, GRID_H, GROUND_HALF)
    assert share_d == pytest.approx(share_p, abs=0.02)
    assert np.allclose(by_depth, by_plane, rtol=0.03, atol=0.02)


def test_the_shift_is_the_median_of_the_world_not_its_mean():
    # flow_series names the camera by the MEDIAN flow, so the truth must too: a truck past a
    # world that is 70% far wall and 30% near wall moves the picture by the far wall's shift.
    # A mean would read the near 30% in at ten times the speed (about 3.7x the far shift).
    track = _track(n=4, loc=lambda i: (0.05 * i, 0.0, 1.6), rot=lambda i: (90.0, 0.0, 0.0))
    width, height = track.resolution
    depth = np.full((height, width), 50.0)
    depth[:, : int(0.3 * width)] = 5.0
    path, _ = calibrate.truth_path(track, GRID_W, GRID_H, depth=lambda i: depth)
    focal_grid = 35.0 / 36.0 * GRID_W
    far, near = 0.05 * focal_grid / 50.0, 0.05 * focal_grid / 5.0
    assert np.allclose(np.abs(path[:, 0]), far, rtol=0.05), (path[:, 0], far, near)
    assert np.allclose(path[:, 1:], 0.0, atol=1e-3)


def test_projection_puts_the_optical_axis_at_the_grid_centre():
    track = _track()
    pose = track.poses[0]
    ahead = pose.location + pose.rotation @ np.array([0.0, 0.0, -10.0])
    uv, front = calibrate.project(track, pose, ahead[None, :], GRID_W, GRID_H)
    assert front.all()
    assert np.allclose(uv[0], [GRID_W / 2, GRID_H / 2], atol=1e-6)


def test_a_cut_gap_carries_no_flow():
    track = _track(rot=lambda i: (80.0, 0.0, 0.5 * i))
    track.camera_ids = ["a"] * 12 + ["b"] * 12
    path, _ = calibrate.truth_path(track, GRID_W, GRID_H, GROUND_HALF)
    assert np.allclose(path[11], 0.0)
    assert not np.allclose(path[10], 0.0)


def test_load_track_reads_the_blender_mcp_shape(tmp_path):
    frames = [
        {"frame": i + 1, "t": i / 24, "camera": "cam", "location": [0, 0, 1.6],
         "rotation_euler_deg": [80, 0, 0.5 * i], "forward": [0, 1, 0], "lens_mm": 35.0,
         "aim": None, "aim_target": None}
        for i in range(24)
    ]
    data = {"fps": 24, "resolution": [1280, 720], "frame_start": 1, "frame_end": 24, "step": 1,
            "cameras": {"cam": {"lens_mm": 35.0, "sensor_width_mm": 36.0, "sensor_height_mm": 24.0,
                                "sensor_fit": "AUTO", "fov_x_deg": 54.43, "shift": [0.0, 0.0],
                                "clip": [0.1, 1000], "handheld": 0.0}},
            "frames": frames}
    p = tmp_path / "camera_track.json"
    p.write_text(json.dumps(data))
    track = calibrate.load_track(p)
    assert len(track.poses) == 24 and track.aspect == pytest.approx(16 / 9)
    label, _, _ = _words(track)
    assert label == "pans left"

    # a cut to a camera with its own sensor would be projected through the first one's
    data["cameras"]["wide"] = dict(data["cameras"]["cam"], sensor_width_mm=72.0)
    for f in frames[12:]:
        f["camera"] = "wide"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="one per track"):
        calibrate.load_track(p)


def test_label_parts_and_scoring():
    assert calibrate.label_parts("pans left, pushes in") == ("pans left", "pushes in")
    assert calibrate.label_parts("static") == ("", "")
    truth = [("pans left", 1.0, (1.0, 0.0, 0.0)), ("static", 0.0, (0.0, 0.0, 0.0))]
    est = [("pans left, pushes in", 1.2, (1.0, 0.0, 0.2)), ("tilts up", 0.3, (0.0, 0.3, 0.0))]
    s = calibrate.score_buckets(truth, est)
    assert (s["exact"], s["translation"], s["radial"]) == (0, 1, 1)
    assert s["falseMotion"] == 1 and s["missedMotion"] == 0
    assert s["speedRatioMedian"] == pytest.approx(1.2)
