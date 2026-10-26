"""Camera-path calibration — score the flow-based camera path against a Blender ground truth.

`depthc compare` and `depthc describe` name the camera's path from optical flow ("pans
left, pushes in"), and both say the PATH layer is UNCALIBRATED: no fire has ever been
scored on it. A paid fire is the wrong first instrument anyway — it tests the model and
the estimator at once. This is the CorridorKey trick: they trained their keyer on CG
renders because a render hands you the right answer for free. oside-blender-mcp renders
clay clips with a known camera and writes the exact per-frame pose (`camera_track.json`),
so the TRUE screen motion of the static world can be computed, frame by frame, and the
estimator scored against it at $0.

THE TRUTH, and why it is in the estimator's own units. For every analysis-grid pixel of
frame i, the surface it sees is put back into the world — from the render's own metric Z
pass, so boxes and ground alike — and projected through the camera of frame i+1; the
displacement is the camera-induced flow at that pixel. Per gap: the median displacement is the path's (dx, dy), and the radial
term is the mean projection of the flow around the median onto the outward unit vector —
the same reduction `motion.flow_series` applies to the estimated field. The truth series is
then named by the SAME `describe.path_buckets` the describe command uses, so a mismatch is
the estimator's, never a vocabulary difference.

What the truth deliberately is NOT: a world verb. A truck left and a pan left both slide the
picture right; a crane up and a tilt up both slide it down. `path_label` is a
translation-and-zoom vocabulary by construction (motion.py), and so is this truth. Pan vs
truck stays unnameable from pixels (closed won't-do 2026-08-23).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Below this much forward travel per frame (metres along the frame-i view axis) the truth
# carries NO radial term. Perspective makes a pure pan's centred flow slightly non-uniform,
# and without this gate that residue would read as a zoom the camera never made. The
# slowest push in the case set travels ~5 mm/frame, so 1 mm separates them cleanly.
FORWARD_STILL_M = 0.001
# A world point must project at least this far inside the frame at i+1 to count — a point
# leaving the frame is not visible to the estimator in the second image.
EDGE_MARGIN_PX = 0.5


@dataclass
class Pose:
    rotation: np.ndarray   # 3x3, camera-local → world
    location: np.ndarray   # (3,)
    lens_mm: float


@dataclass
class Track:
    poses: list[Pose]
    sensor_w: float
    aspect: float
    shift: tuple[float, float] = (0.0, 0.0)
    resolution: tuple[int, int] = (1280, 720)
    camera_ids: list[str] = field(default_factory=list)


def euler_xyz(deg: list[float] | tuple[float, float, float]) -> np.ndarray:
    """Blender's default XYZ Euler as a matrix: X applied first, so R = Rz @ Ry @ Rx."""
    x, y, z = (math.radians(v) for v in deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def load_track(path: Path) -> Track:
    """Read oside-blender-mcp's `camera_track.json` (blender/measure.py `camera_track`)."""
    data = json.loads(Path(path).read_text())
    res_x, res_y = data["resolution"]
    frames = data["frames"]
    if not frames:
        raise ValueError(f"{path}: the track has no frames")
    cams = data["cameras"]
    first = cams[frames[0]["camera"]]
    # One sensor for the whole track: a cut to a camera with another sensor, shift or fit would
    # be projected through the first camera's and scored wrong (Codex review 2026-10-26). Refuse it.
    for cam_id in {f["camera"] for f in frames}:
        cam = cams[cam_id]
        if (cam["sensor_width_mm"], list(cam["shift"]), cam.get("sensor_fit", "AUTO")) != (
            first["sensor_width_mm"], list(first["shift"]), first.get("sensor_fit", "AUTO")
        ):
            raise ValueError(f"{path}: camera {cam_id!r} has its own sensor, shift or fit — one per track is modelled")
    if first.get("sensor_fit", "AUTO") not in ("AUTO", "HORIZONTAL"):
        raise ValueError(f"{path}: sensor_fit {first['sensor_fit']!r} — only horizontal fit is modelled")
    if res_x < res_y:
        raise ValueError(f"{path}: portrait frames — AUTO fit would go vertical; not modelled")
    poses = [
        Pose(euler_xyz(f["rotation_euler_deg"]), np.asarray(f["location"], dtype=np.float64), float(f["lens_mm"]))
        for f in frames
    ]
    return Track(
        poses=poses,
        sensor_w=float(first["sensor_width_mm"]),
        aspect=res_x / res_y,
        shift=(float(first["shift"][0]), float(first["shift"][1])),
        resolution=(res_x, res_y),
        camera_ids=[f["camera"] for f in frames],
    )


def _rays(track: Track, pose: Pose, u: np.ndarray, v: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """Camera-LOCAL ray directions (a, b, -1) through grid positions (u, v): planar depth 1."""
    sensor_h = track.sensor_w / track.aspect
    x_ndc = u / grid_w
    y_ndc = 1.0 - v / grid_h  # image rows run down; Blender's ndc y runs up
    a = (x_ndc - 0.5 - track.shift[0]) * track.sensor_w / pose.lens_mm
    b = (y_ndc - 0.5 - track.shift[1] * track.aspect) * sensor_h / pose.lens_mm
    return np.stack([a, b, -np.ones_like(a)], axis=-1)       # camera looks down local -Z


def _ground_points(
    track: Track, pose: Pose, grid_w: int, grid_h: int, ground_half_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Static world points seen at every grid pixel centre, on the ground plane z = 0 only.
    Returns (points [M, 3], their grid (u, v) [M, 2], share of the grid that sees ground)."""
    ys, xs = np.mgrid[0:grid_h, 0:grid_w].astype(np.float64)
    u, v = xs + 0.5, ys + 0.5
    rays = _rays(track, pose, u, v, grid_w, grid_h) @ pose.rotation.T
    dz = rays[..., 2]
    t = np.where(dz < -1e-9, -pose.location[2] / np.where(dz < -1e-9, dz, -1.0), -1.0)
    points = pose.location + rays * t[..., None]
    hit = (t > 0) & (np.abs(points[..., 0]) <= ground_half_m) & (np.abs(points[..., 1]) <= ground_half_m)
    return points[hit], np.stack([u, v], axis=-1)[hit], float(hit.mean())


def _depth_points(
    track: Track, pose: Pose, depth_m: np.ndarray, grid_w: int, grid_h: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Static world points from a full-resolution PLANAR metric depth map (NaN = sky): every
    surface the render saw — boxes and ground alike — so the truth covers what an estimator
    looks at. One full-res pixel per grid cell, at the cell centre, and the ray goes through
    THAT pixel's centre (no averaging: a mean across a box edge invents a point in mid-air).
    """
    height, width = depth_m.shape
    cols = np.minimum(((np.arange(grid_w) + 0.5) * width / grid_w).astype(int), width - 1)
    rows = np.minimum(((np.arange(grid_h) + 0.5) * height / grid_h).astype(int), height - 1)
    z = depth_m[np.ix_(rows, cols)]
    u = (cols[None, :] + 0.5) * grid_w / width + np.zeros((grid_h, 1))
    v = (rows[:, None] + 0.5) * grid_h / height + np.zeros((1, grid_w))
    seen = np.isfinite(z) & (z > 0)
    local = _rays(track, pose, u, v, grid_w, grid_h) * np.where(seen, z, 0.0)[..., None]
    points = pose.location + local @ pose.rotation.T
    return points[seen], np.stack([u, v], axis=-1)[seen], float(seen.mean())


def project(track: Track, pose: Pose, points: np.ndarray, grid_w: int, grid_h: int) -> tuple[np.ndarray, np.ndarray]:
    """World points → grid (u, v) and an in-front mask. Same maths as measure._project_local."""
    sensor_h = track.sensor_w / track.aspect
    local = (points - pose.location) @ pose.rotation          # R^T (p - L), row-vector form
    depth = -local[..., 2]
    front = depth > 1e-6
    safe = np.where(front, depth, 1.0)
    x_ndc = (local[..., 0] / safe) * pose.lens_mm / track.sensor_w + 0.5 + track.shift[0]
    y_ndc = (local[..., 1] / safe) * pose.lens_mm / sensor_h + 0.5 + track.shift[1] * track.aspect
    return np.stack([x_ndc * grid_w, (1.0 - y_ndc) * grid_h], axis=-1), front


def truth_path(
    track: Track, grid_w: int, grid_h: int, ground_half_m: float | None = None,
    depth: Callable[[int], np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Per frame gap: (dx, dy, radial) of the STATIC world, in grid px/frame. Plus the mean
    share of the grid that sees geometry (the rest is sky, where an estimator has nothing).

    The static world is either `depth(i)` — frame i's planar metric depth at full resolution,
    NaN for sky (a Blender Z pass: every surface in shot) — or, without it, the ground plane
    z = 0 out to ±`ground_half_m` (synthetic tracks in the tests). On a rendered clip use the
    depth: the ground alone books a truck or push at the ground's distance, while the
    estimator also sees boxes standing on it, nearer and farther.
    """
    if depth is None and ground_half_m is None:
        raise ValueError("truth_path needs a depth source or a ground plane")
    gaps = len(track.poses) - 1
    path = np.zeros((max(gaps, 0), 3), dtype=np.float32)
    shares: list[float] = []
    cx, cy = (grid_w - 1) / 2.0, (grid_h - 1) / 2.0
    for i in range(gaps):
        here, there = track.poses[i], track.poses[i + 1]
        if track.camera_ids and track.camera_ids[i] != track.camera_ids[i + 1]:
            continue  # a cut: no flow across it, the estimator sees a new picture
        if depth is not None:
            points, start, share = _depth_points(track, here, depth(i), grid_w, grid_h)
        else:
            points, start, share = _ground_points(track, here, grid_w, grid_h, ground_half_m)
        shares.append(share)
        if not len(points):
            continue
        moved, front = project(track, there, points, grid_w, grid_h)
        inside = front & _inside(moved, grid_w, grid_h)
        if not inside.any():
            continue
        median = np.median(moved[inside] - start[inside], axis=0)
        path[i, 0], path[i, 1] = float(median[0]), float(median[1])
        path[i, 2] = _forward_radial(track, here, there, points, start, grid_w, grid_h, cx, cy)
    return path, (float(np.mean(shares)) if shares else 0.0)


def _inside(uv: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    return (
        (uv[:, 0] >= EDGE_MARGIN_PX) & (uv[:, 0] <= grid_w - EDGE_MARGIN_PX)
        & (uv[:, 1] >= EDGE_MARGIN_PX) & (uv[:, 1] <= grid_h - EDGE_MARGIN_PX)
    )


def _forward_radial(track, here, there, points, start, grid_w, grid_h, cx, cy) -> float:
    """The zoom term of the truth: radial flow from the camera's travel ALONG ITS OWN VIEW
    AXIS and from a lens change, rotation held. (A zoom and a push look alike to the flow, and
    path_label names both "pushes in".)

    Not the radial of the full field: a camera rising while it looks down sees the near
    ground slide down faster than the far ground, and that vertical stretch reduces to a
    coherent "pushes in" although the camera never moved toward anything (red on the first
    version of this module, crane test). The camera's motion is the truth; whether the
    ESTIMATOR also reads the stretch as a push is a finding about the estimator.
    """
    forward = here.rotation @ np.array([0.0, 0.0, -1.0])
    along = float((there.location - here.location) @ forward)
    zoomed = there.lens_mm != here.lens_mm
    if abs(along) < FORWARD_STILL_M and not zoomed:
        return 0.0
    dolly = Pose(here.rotation, here.location + forward * along, there.lens_mm)
    moved, front = project(track, dolly, points, grid_w, grid_h)
    inside = front & _inside(moved, grid_w, grid_h)
    if not inside.any():
        return 0.0
    d = moved[inside] - start[inside]
    centred = d - np.median(d, axis=0)
    # pixel-index coordinates around the grid centre, as flow_series uses
    rx = (start[inside, 0] - 0.5) - cx
    ry = (start[inside, 1] - 0.5) - cy
    norm = np.hypot(rx, ry)
    norm[norm == 0] = 1.0
    return float(((centred[:, 0] * rx + centred[:, 1] * ry) / norm).mean())


def label_parts(label: str) -> tuple[str, str]:
    """'pans left, pushes in' → ('pans left', 'pushes in'); 'static' → ('', '')."""
    translation, radial = "", ""
    for part in (p.strip() for p in label.split(",")):
        if part in ("pushes in", "pulls out"):
            radial = part
        elif part and part != "static":
            translation = part
    return translation, radial


def score_buckets(
    truth: list[tuple[str, float, tuple[float, float, float]]],
    estimate: list[tuple[str, float, tuple[float, float, float]]],
) -> dict:
    """Bucket-by-bucket agreement between two `path_buckets` series of the same clip."""
    n = min(len(truth), len(estimate))
    rows = []
    for k in range(n):
        t_label, t_speed, t_vec = truth[k]
        e_label, e_speed, e_vec = estimate[k]
        t_tr, t_rad = label_parts(t_label)
        e_tr, e_rad = label_parts(e_label)
        rows.append({
            "bucket": k, "truth": t_label, "estimate": e_label,
            "exact": t_label == e_label,
            "translation": t_tr == e_tr, "radial": t_rad == e_rad,
            "truthSpeed": round(t_speed, 3), "estimateSpeed": round(e_speed, 3),
            "truthVec": [round(v, 3) for v in t_vec], "estimateVec": [round(v, 3) for v in e_vec],
        })
    moving = [r for r in rows if r["truth"] != "static" and r["truthSpeed"] > 0]
    ratios = [r["estimateSpeed"] / r["truthSpeed"] for r in moving if r["estimate"] != "static"]
    return {
        "buckets": n,
        "exact": sum(r["exact"] for r in rows),
        "translation": sum(r["translation"] for r in rows),
        "radial": sum(r["radial"] for r in rows),
        "missedMotion": sum(1 for r in moving if r["estimate"] == "static"),
        "falseMotion": sum(1 for r in rows if r["truth"] == "static" and r["estimate"] != "static"),
        "speedRatioMedian": round(float(np.median(ratios)), 3) if ratios else None,
        "rows": rows,
    }
