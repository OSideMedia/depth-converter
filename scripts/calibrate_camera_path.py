"""Calibrate the camera-path words against Blender clips whose camera move is known. $0.

    ./.tools/uv run python scripts/calibrate_camera_path.py [--cases a,b] [--holdout] [--estimators rgb,vda,blender]

For each case: write an oside-blockout/v1 spec, render it with oside-blender-mcp (clay +
exact Z pass + camera_track.json), compute the TRUE screen motion of the static world from
the track and the Z pass (`depthc.calibrate`), then run the estimator three ways and score
each against the truth bucket by bucket, in the words `depthc describe` would print:

    rgb      the clay picture itself (describe --rgb)
    vda      VDA-Small depth of the clay picture (describe / compare's generation side)
    blender  Blender's exact depth.mp4 (a kit's reference side, no estimation error)
    pic      the clay picture through `motion.camera_path`, the reader depthc ships since
             2026-11-11 for every camera word (tracked points, movers dropped)

Each takes a `-tex` suffix (`rgb-tex`, `blender-tex`, `vda-tex`): the same Farneback pass
with the static floor and the median read on TEXTURED pixels only (Shi-Tomasi corner
strength). `-sim` reads the same textured pixels through a RANSAC similarity fit, shift and
zoom together. Both are diagnostics, not the reader depthc ships: they separate "the clay has no
texture to see" from "the method misreads what it can see". `--tag` names the outputs so a
diagnostic run never overwrites the main report.

Everything lands in Outbox/calibration/ (gitignored media). Nothing here costs money: no
network, no paid API — Blender and the depth model both run on this Mac.

CLAY IS THE HARD CASE. Flat-colour clay has no texture on the ground or the sky; an AI take
has texture everywhere. A miss here is a lower bound on the estimator, not a forecast of
what it does on a real take — the report says so.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthc import calibrate, motion  # noqa: E402
from depthc.describe import path_buckets  # noqa: E402

BLENDER_MCP = ROOT.parent / "oside-blender-mcp"
OUT = ROOT / "Outbox" / "calibration"
DURATION_S = 4
FPS = 24
CAM_Z = 1.6
DOWN_DEG = 10.0          # the camera looks 10° below the horizon: ~20% sky, the rest ground and boxes
FAR_AIM_Y = 1000.0       # a fixed aim this far away barely rotates a camera that travels a few metres
RAIL_Y = -150.0          # box rows hang off a rail far behind the camera, never in shot
ROWS = (                 # (y of the row, colour, count over the 300 m rail, dims L×W×H)
    (8.0, "red", 200, (0.5, 0.5, 1.2)),
    (14.0, "green", 160, (0.8, 0.8, 2.0)),
    (22.0, "blue", 120, (1.2, 1.2, 3.0)),
    (35.0, "yellow", 100, (1.6, 1.6, 4.5)),
    (55.0, "orange", 75, (2.4, 2.4, 6.0)),
    (90.0, "cyan", 60, (3.0, 3.0, 9.0)),
)
MARKER_R = 100.0         # pans and tilts aim a hidden marker this far out round the camera


def _aim_far() -> list[float]:
    return [0.0, FAR_AIM_Y, round(CAM_Z - FAR_AIM_Y * math.tan(math.radians(DOWN_DEG)), 4)]


def _marker_path(yaw0: float, yaw1: float, pitch0: float, pitch1: float) -> list[dict]:
    """Keys every 0.25 s on a sphere round the camera, so the aim sweeps at a near-constant
    angular rate (yaw 0 = straight down +Y, + = left; pitch + = up). The marker sits below
    the ground plane while the camera looks down, so the render never shows it."""
    keys = []
    steps = DURATION_S * 4
    for k in range(steps + 1):
        f = k / steps
        yaw = math.radians(yaw0 + (yaw1 - yaw0) * f)
        pitch = math.radians(pitch0 + (pitch1 - pitch0) * f)
        flat = MARKER_R * math.cos(pitch)
        keys.append({"t": k / 4, "pos": [round(-flat * math.sin(yaw), 4), round(flat * math.cos(yaw), 4),
                                         round(CAM_Z + MARKER_R * math.sin(pitch), 4)]})
    return keys


def _rotate(yaw0, yaw1, pitch0=-DOWN_DEG, pitch1=-DOWN_DEG, **cam) -> dict:
    """A camera that stands still and turns: aim at the marker as it sweeps."""
    return {"camera": {"rig": "static", "params": {"start": [0, 0, CAM_Z]},
                       "aim": {"target": "marker", "height_frac": 0.5}, **cam},
            "marker": _marker_path(yaw0, yaw1, pitch0, pitch1)}


def _travel(rig: str, params: dict, **cam) -> dict:
    return {"camera": {"rig": rig, "params": params, "aim": {"point": _aim_far()}, **cam}}


def _per_frame(deg_per_frame: float) -> float:
    return deg_per_frame * DURATION_S * FPS


def _pan(deg_per_frame: float, **kw) -> dict:
    """A pan CENTRED on straight ahead (+ = left), so even a fast one stays on the box rows —
    a sweep that starts ahead ends looking back at bare ground, and that tests the scene."""
    half = _per_frame(deg_per_frame) / 2
    return _rotate(-half, half, **kw)


# name → (what the camera does, in words a person would use; the case's camera)
CASES: dict[str, tuple[str, dict]] = {
    "static": ("locked off", _travel("static", {"start": [0, 0, CAM_Z]})),
    "static-handheld": ("locked off, light handheld", _travel("static", {"start": [0, 0, CAM_Z]}, handheld=0.3)),
    "pan-left-slow": ("pan left 0.15°/frame", _pan(0.15)),
    "pan-left": ("pan left 0.5°/frame", _pan(0.5)),
    "pan-right": ("pan right 0.5°/frame", _pan(-0.5)),
    "pan-right-fast": ("pan right 1.2°/frame", _pan(-1.2)),
    "pan-left-24mm": ("pan left 0.5°/frame on a 24 mm", _pan(0.5, lens_mm=24)),
    "pan-left-85mm": ("pan left 0.5°/frame on an 85 mm", _pan(0.5, lens_mm=85)),
    "pan-left-sky": ("pan left 0.5°/frame, horizon mid-frame (half sky)",
                     _pan(0.5, pitch0=-1.0, pitch1=-1.0)),
    "tilt-up": ("tilt up 0.15°/frame", _rotate(0, 0, -DOWN_DEG - 4, -DOWN_DEG - 4 + _per_frame(0.15))),
    "tilt-down": ("tilt down 0.15°/frame", _rotate(0, 0, -4, -4 - _per_frame(0.15))),
    "truck-left": ("truck left 5 cm/frame", _travel("dolly", {"start": [2.4, 0, CAM_Z], "end": [-2.4, 0, CAM_Z]})),
    "truck-right": ("truck right 5 cm/frame", _travel("dolly", {"start": [-2.4, 0, CAM_Z], "end": [2.4, 0, CAM_Z]})),
    "truck-right-fast": ("truck right 15 cm/frame",
                         _travel("dolly", {"start": [-7.2, 0, CAM_Z], "end": [7.2, 0, CAM_Z]})),
    "push-in": ("push in 5 cm/frame", _travel("push_in", {"start": [0, 0, CAM_Z], "distance_m": 4.8})),
    "push-in-slow": ("push in 2 cm/frame", _travel("push_in", {"start": [0, 0, CAM_Z], "distance_m": 1.92})),
    "pull-out": ("pull out 5 cm/frame", _travel("pull_out", {"start": [0, 0, CAM_Z], "distance_m": 4.8})),
    "crane-up": ("crane up 2.5 cm/frame", _travel("crane", {"start": [0, 0, CAM_Z], "rise_m": 2.4})),
    "crane-down": ("crane down 1.25 cm/frame", _travel("crane", {"start": [0, 0, CAM_Z], "rise_m": -1.2})),
    "static-subject-crosses": ("locked off; a figure walks across the frame", {
        **_travel("static", {"start": [0, 0, CAM_Z]}),
        "walker": [{"t": 0, "pos": [-5.0, 6.0, 0]}, {"t": DURATION_S, "pos": [5.0, 6.0, 0]}],
    }),
    # Two moves at once, and what a zoom reader must not fall for: a figure walking at a camera
    # that never moves. Alone on the bare field, the figure is the only thing with any texture.
    "pan-left-push-in": ("pan left 0.5°/frame while pushing in 5 cm/frame", {
        "camera": {"rig": "push_in", "params": {"start": [0, 0, CAM_Z], "distance_m": 4.8},
                   "aim": {"target": "marker", "height_frac": 0.5}},
        "marker": _marker_path(-_per_frame(0.5) / 2, _per_frame(0.5) / 2, -DOWN_DEG, -DOWN_DEG),
    }),
    "truck-left-push-in": ("truck left and push in, 5 cm/frame each", _travel(
        "dolly", {"start": [2.4, 0, CAM_Z], "end": [-2.4, 4.8, CAM_Z]})),
    "static-subject-approaches": ("locked off; a figure walks at the camera past the boxes", {
        **_travel("static", {"start": [0, 0, CAM_Z]}),
        "walker": [{"t": 0, "pos": [0.5, 14.0, 0]}, {"t": DURATION_S, "pos": [0.5, 5.0, 0]}],
    }),
    "static-subject-approaches-alone": ("locked off; a figure walks at the camera on a bare field", {
        **_travel("static", {"start": [0, 0, CAM_Z]}),
        "bare": True,
        "walker": [{"t": 0, "pos": [0.5, 14.0, 0]}, {"t": DURATION_S, "pos": [0.5, 5.0, 0]}],
    }),
}

# Rendered and scored once, AFTER flow_series's thresholds were frozen, so these numbers say how
# the reader does on moves it was not tuned on. Tune against them and they stop saying that.
HOLDOUT: dict[str, tuple[str, dict]] = {
    "holdout-push-in-85mm": ("push in 5 cm/frame on an 85 mm", _travel(
        "push_in", {"start": [0, 0, CAM_Z], "distance_m": 4.8}, lens_mm=85)),
    "holdout-push-in-horizon": ("push in 5 cm/frame, horizon mid-frame", {
        "camera": {"rig": "push_in", "params": {"start": [0, 0, CAM_Z], "distance_m": 4.8},
                   "aim": {"target": "marker", "height_frac": 0.5}},
        "marker": _marker_path(0.0, 0.0, -1.0, -1.0),
    }),
    "holdout-pull-out-slow": ("pull out 2 cm/frame", _travel("pull_out", {"start": [0, 0, CAM_Z], "distance_m": 1.92})),
    "holdout-pan-right-pull-out": ("pan right 0.3°/frame while pulling out 5 cm/frame", {
        "camera": {"rig": "pull_out", "params": {"start": [0, 0, CAM_Z], "distance_m": 4.8},
                   "aim": {"target": "marker", "height_frac": 0.5}},
        "marker": _marker_path(_per_frame(0.3) / 2, -_per_frame(0.3) / 2, -DOWN_DEG, -DOWN_DEG),
    }),
    "holdout-tilt-up-24mm": ("tilt up 0.15°/frame on a 24 mm",
                             _rotate(0, 0, -DOWN_DEG - 4, -DOWN_DEG - 4 + _per_frame(0.15), lens_mm=24)),
    "holdout-truck-left-85mm": ("truck left 5 cm/frame on an 85 mm", _travel(
        "dolly", {"start": [2.4, 0, CAM_Z], "end": [-2.4, 0, CAM_Z]}, lens_mm=85)),
    "holdout-crane-up-24mm": ("crane up 2.5 cm/frame on a 24 mm", _travel(
        "crane", {"start": [0, 0, CAM_Z], "rise_m": 2.4}, lens_mm=24)),
    "holdout-truck-right-subject-crosses": ("truck right 5 cm/frame; a figure crosses", {
        **_travel("dolly", {"start": [-2.4, 0, CAM_Z], "end": [2.4, 0, CAM_Z]}),
        "walker": [{"t": 0, "pos": [-4.0, 5.0, 0]}, {"t": DURATION_S, "pos": [4.0, 5.0, 0]}],
    }),
    "holdout-static-subject-near": ("locked off; a figure walks close past the lens", {
        **_travel("static", {"start": [0, 0, CAM_Z]}),
        "walker": [{"t": 0, "pos": [-1.5, 10.0, 0]}, {"t": DURATION_S, "pos": [1.0, 2.5, 0]}],
    }),
}


def spec_for(name: str, words: str, case: dict) -> dict:
    subjects: list[dict] = [{
        "id": "rail", "kind": "object", "shape": "box", "dims_m": [0.05, 0.05, 0.05], "role_color": "grey",
        "describe": "a hidden rail the box rows are laid along", "interp": "linear",
        "path": [{"t": 0, "pos": [-150.0, RAIL_Y, 0]}, {"t": DURATION_S, "pos": [150.0, RAIL_Y, 0]}],
    }]
    for k, (y, colour, count, dims) in enumerate(() if case.get("bare") else ROWS):
        subjects.append({
            "id": f"row{k + 1}", "kind": "structure", "shape": "box_array", "count": count, "along": "rail",
            # right of travel is -Y for a rail running +X, so a NEGATIVE offset puts the row ahead
            "offset_m": round(RAIL_Y - y, 3), "dims_m": list(dims), "role_color": colour,
            "describe": f"a row of {colour} blocks {y:g} m out",
        })
    if "marker" in case:
        subjects.append({"id": "marker", "kind": "object", "shape": "sphere", "dims_m": [0.05, 0.05, 0.05],
                         "role_color": "magenta", "describe": "the hidden aim marker", "interp": "linear",
                         "path": case["marker"]})
    characters = []
    if "walker" in case:
        characters.append({"id": "walker", "height_m": 1.75, "role_color": "magenta", "interp": "linear",
                           "describe": "a figure crossing the frame", "path": case["walker"]})
    return {
        "format": "oside-blockout/v1",
        "slug": f"cal-{name}",
        "target": {"model": "seedance-2.5", "output_seconds": DURATION_S, "aspect": "16:9", "resolution": "720p"},
        "timing": {"fps": FPS, "duration_s": DURATION_S},
        "world": {"ground": {"name": "field", "size_m": 400}, "sky": "flat"},
        "subjects": subjects,
        "characters": characters,
        "camera": {"id": "cam", "lens_mm": 35, **case["camera"]},
        "notes": {"event": f"camera-path calibration: {words}", "look": "clay calibration field"},
    }


def build(spec_path: Path, kits: Path) -> Path:
    # this project's pinned uv, not whatever `uv` is on PATH (CLAUDE.md: always ./.tools/uv)
    cmd = [str(ROOT / ".tools" / "uv"), "run", "--project", str(BLENDER_MCP), "oside-blender", "build", str(spec_path),
           "--out", str(kits), "--passes", "model,depth", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"build failed for {spec_path.name} ({proc.returncode}):\n{proc.stderr[-2000:]}")
    # a red CHECK verdict still writes the kit; the checker judges Seedance-readiness, not this
    return Path(result["kit_dir"])


def depth_reader(kit: Path):
    """Frame i (0-based) of the kit's depth.mp4 as PLANAR metres, NaN for sky.

    depth.mp4 is inverse depth, normalised over the clip: luma 0–255 spans the recorded
    disparity range 1/farM … 1/nearM (oside-blender-mcp encode.depth_pngs_to_mp4), and luma 0
    is sky. Inverse depth is the quantity camera translation scales with, so 8 bits of it cost
    the truth ~0.03 px/frame on a 5 cm/frame truck. Planar (not distance along the ray) was
    measured on the static case: median error 0.15% against a planar ground, 4.7% against the
    ray-length reading. The nearest 0.5% clip to nearM and the farthest 0.5% read as sky.
    """
    from depthc.video_io import probe, read_frames

    meta = json.loads((kit / "kit.json").read_text())["depth"]
    low, high = 1.0 / float(meta["farM"]), 1.0 / float(meta["nearM"])
    luma = read_frames(probe(kit / "depth.mp4"))[..., 0]

    def read(i: int) -> np.ndarray:
        level = luma[i].astype(np.float64)
        disparity = low + level / 255.0 * (high - low)
        return np.where(level > 0, 1.0 / disparity, np.nan)

    return read, luma.shape[0]


TEX_QUALITY = 0.01   # Shi-Tomasi's own default: a pixel counts at >= 1% of the frame's strongest corner
TEX_MIN_PIXELS = 40  # a quadrant or a frame with fewer textured pixels than this has nothing to read
TEX_RADIAL_QUADRANT = 0.5  # the -tex diagnostic's own quadrant test, as flow_series once read a zoom


def textured_path(frames: np.ndarray, fps: float = 24.0) -> np.ndarray:
    """motion.flow_series's path, with the static floor and every median read on textured pixels.

    Flow is only observable where the picture has texture in two directions (the aperture
    problem): a flat wall or an untextured clay floor moves without showing it, and a smooth
    depth ramp shows only the motion across its gradient. Those pixels vote zero in an
    all-pixel median. Here the mask is the previous frame's minimum structure-tensor
    eigenvalue, so an edge or a ramp drops out with the flat colour and corners stay.
    """
    import cv2

    grid = motion.analysis_gray(frames)
    gaps = max(0, grid.shape[0] - 1)
    path = np.zeros((gaps, 3), dtype=np.float32)
    height, width = grid.shape[1:]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    rx, ry = xs - (width - 1) / 2.0, ys - (height - 1) / 2.0
    norm = np.hypot(rx, ry)
    norm[norm == 0] = 1.0
    rx, ry = rx / norm, ry / norm
    half_h, half_w = height // 2, width // 2
    quadrants = (
        (slice(0, half_h), slice(0, half_w)), (slice(0, half_h), slice(half_w, None)),
        (slice(half_h, None), slice(0, half_w)), (slice(half_h, None), slice(half_w, None)),
    )
    floor = motion.STATIC_FLOOR * (24.0 / fps)
    quadrant_floor = motion.PATH_RADIAL_STILL * TEX_RADIAL_QUADRANT
    for index, (previous, current) in enumerate(zip(grid[:-1], grid[1:])):
        strength = cv2.cornerMinEigenVal(previous, blockSize=3, ksize=3)
        mask = strength >= TEX_QUALITY * float(strength.max()) if strength.max() > 0 else np.zeros_like(strength, bool)
        if mask.sum() < TEX_MIN_PIXELS:
            continue
        delta = np.abs(current.astype(np.float32) - previous.astype(np.float32))[mask].mean()
        if delta <= floor:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        median = np.median(flow[mask], axis=0)
        centred = flow - median
        path[index, 0], path[index, 1] = float(median[0]), float(median[1])
        radial_field = centred[..., 0] * rx + centred[..., 1] * ry
        radial = float(radial_field[mask].mean())
        per_quadrant = [radial_field[q][mask[q]] for q in quadrants]
        coherent = all(
            q.size >= TEX_MIN_PIXELS // 4 and np.sign(q.mean()) == np.sign(radial) and abs(q.mean()) >= quadrant_floor
            for q in per_quadrant
        )
        path[index, 2] = radial if coherent else 0.0
    return path


def similarity_path(frames: np.ndarray, fps: float = 24.0) -> np.ndarray:
    """textured_path, with translation and zoom FITTED TOGETHER: a RANSAC similarity (shift, uniform
    scale, rotation) from each textured pixel to where the flow sends it.

    The median-then-radial split books a push as a tilt whenever the texture sits off-centre (the
    outward flow of boxes below centre drags the median down). A joint fit has no such bias, and
    RANSAC drops a lone subject the way the quadrant test did. The radial figure is the fitted
    scale times the mean radius of the WHOLE grid, so it stays in flow_series's units.
    """
    import cv2

    grid = motion.analysis_gray(frames)
    gaps = max(0, grid.shape[0] - 1)
    path = np.zeros((gaps, 3), dtype=np.float32)
    height, width = grid.shape[1:]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    mean_radius = float(np.hypot(xs - cx, ys - cy).mean())
    floor = motion.STATIC_FLOOR * (24.0 / fps)
    for index, (previous, current) in enumerate(zip(grid[:-1], grid[1:])):
        strength = cv2.cornerMinEigenVal(previous, blockSize=3, ksize=3)
        mask = strength >= TEX_QUALITY * float(strength.max()) if strength.max() > 0 else np.zeros_like(strength, bool)
        if mask.sum() < TEX_MIN_PIXELS:
            continue
        if np.abs(current.astype(np.float32) - previous.astype(np.float32))[mask].mean() <= floor:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        src = np.stack([xs[mask] - cx, ys[mask] - cy], axis=1).astype(np.float32)
        dst = src + flow[mask]
        fit, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.5,
                                             maxIters=2000, confidence=0.995)
        if fit is None:
            continue
        scale = float(np.sqrt(abs(np.linalg.det(fit[:, :2]))))
        path[index] = (float(fit[0, 2]), float(fit[1, 2]), (scale - 1.0) * mean_radius)
    return path


def estimate(kit: Path, which: str) -> "np.ndarray | motion.CameraReading":
    from depthc.video_io import probe, read_frames, target_size

    reader = textured_path if which.endswith("-tex") else similarity_path if which.endswith("-sim") else None
    which = which.removesuffix("-tex").removesuffix("-sim")
    if which == "pic":
        info = probe(kit / "blockout.mp4")
        return motion.camera_reading(read_frames(info, fps=24.0, size=target_size(info, "480p")), 24.0)
    if which == "rgb":
        info = probe(kit / "blockout.mp4")
        frames = read_frames(info, fps=24.0, size=target_size(info, "480p"))
    elif which == "blender":
        info = probe(kit / "depth.mp4")
        frames = read_frames(info, fps=24.0)
    elif which == "vda":
        from depthc import engine

        info = probe(kit / "blockout.mp4")
        frames = read_frames(info, fps=24.0, size=target_size(info, "720p"))
        frames = engine.infer(frames, 24.0, device=engine.pick_device()).frames
    else:
        raise ValueError(which)
    return reader(frames, 24.0) if reader else motion.flow_series(frames, 24.0).path


def run_case(name: str, estimators: list[str], rebuild: bool) -> dict:
    words, case = {**CASES, **HOLDOUT}[name]
    case_dir = OUT / name
    kit_link = case_dir / "kit.txt"
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if rebuild or not kit_link.exists():
        # a rebuild renders a NEW kit beside the old one (oside-blender numbers them);
        # estimates are cached per kit, so nothing is ever deleted
        spec_path = case_dir / "spec.json"
        spec_path.write_text(json.dumps(spec_for(name, words, case), indent=2))
        kit_link.write_text(str(build(spec_path, case_dir / "kits")) + "\n")
    kit = Path(kit_link.read_text().strip())
    track = calibrate.load_track(kit / "camera_track.json")
    read, count = depth_reader(kit)
    if count != len(track.poses):
        raise SystemExit(f"{name}: {count} depth frames for {len(track.poses)} poses")
    grid_w, grid_h = motion.analysis_size(*track.resolution)  # the grid flow_series measures on
    truth, share = calibrate.truth_path(track, grid_w, grid_h, depth=read)
    truth_buckets = path_buckets(truth, 24.0)
    row = {"case": name, "move": words, "geometryShare": round(share, 3),
           "truth": _runs(truth_buckets), "estimators": {}}
    for which in estimators:
        # Keyed by the reader's version: a change to flow_series must not be scored on the numbers
        # the old one left here. Earlier estimates stay on disk under their own names.
        version = motion.PICTURE_READER if which == "pic" else motion.PATH_READER
        cached = case_dir / f"est-{which}-{kit.name}-{version}.npy"
        # The picture reader's shots ride beside its path, so its buckets are cut the way
        # `describe` and `compare` cut them: never across a cut (`motion.bucket_rows`).
        shots_file = cached.with_suffix(".shots.json")
        shots = None
        if cached.exists():
            path = np.load(cached)
            if shots_file.exists():
                shots = [motion.Shot(**s) for s in json.loads(shots_file.read_text())]
        else:
            got = estimate(kit, which)
            if isinstance(got, motion.CameraReading):
                path, shots = got.path, got.shots
                shots_file.write_text(json.dumps([vars(s) for s in shots]))
            else:
                path = got
            np.save(cached, path)
        # The picture reader names a zoom from its own floor (it reads the zoom across half a
        # second); every other reader from the region reader's.
        floor = motion.PATH_RADIAL_SLOW if which == "pic" else motion.PATH_RADIAL_STILL
        est_buckets = path_buckets(path, 24.0, radial_floor=floor, shots=shots)
        score = calibrate.score_buckets(truth_buckets, est_buckets)
        score["said"] = _runs(est_buckets)
        row["estimators"][which] = score
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _runs(buckets) -> str:
    """'pans left ×8' / 'static ×2, pans left ×6' — the bucket labels, run-length coded."""
    out: list[list] = []
    for label, _, _ in buckets:
        if out and out[-1][0] == label:
            out[-1][1] += 1
        else:
            out.append([label, 1])
    return ", ".join(f"{label} ×{n}" for label, n in out)


def report(rows: list[dict], estimators: list[str]) -> str:
    lines = [
        "# Camera-path calibration — Blender clay, known camera",
        "",
        f"{len(rows)} cases × {DURATION_S} s at {FPS} fps, cut into half-second buckets; a bucket is EXACT"
        " when the estimator's words equal the truth's.",
        "Clay is the hard case: flat colour, no texture on the ground or the sky.",
        "",
        "| case | truth | " + " | ".join(f"{e} exact" for e in estimators) + " | " +
        " | ".join(f"{e} said" for e in estimators) + " |",
        "|---|---|" + "---|" * (2 * len(estimators)),
    ]
    for r in rows:
        exact = [f"{r['estimators'][e]['exact']}/{r['estimators'][e]['buckets']}" for e in estimators]
        said = [r["estimators"][e]["said"] for e in estimators]
        lines.append(f"| {r['case']} | {r['truth']} | " + " | ".join(exact) + " | " + " | ".join(said) + " |")
    lines += ["", "| estimator | exact | translation right | radial right | missed motion | false motion | can't tell | speed ratio (median of medians) |",
              "|---|---|---|---|---|---|---|---|"]
    for e in estimators:
        tot = {k: sum(r["estimators"][e][k] for r in rows)
               for k in ("buckets", "exact", "translation", "radial", "missedMotion", "falseMotion", "unread")}
        ratios = [r["estimators"][e]["speedRatioMedian"] for r in rows if r["estimators"][e]["speedRatioMedian"] is not None]
        lines.append(f"| {e} | {tot['exact']}/{tot['buckets']} | {tot['translation']}/{tot['buckets']} | "
                     f"{tot['radial']}/{tot['buckets']} | {tot['missedMotion']} | {tot['falseMotion']} | {tot['unread']} | "
                     f"{round(float(np.median(ratios)), 3) if ratios else '—'} |")
    lines += ["", "## Every bucket that missed", "",
              "Vectors are (dx, dy, radial) in grid px/frame, mean over the bucket's moving gaps.", "",
              "| case | estimator | t (s) | truth | said | truth vec | said vec |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for e in estimators:
            for b in r["estimators"][e]["rows"]:
                if not b["exact"]:
                    lines.append(f"| {r['case']} | {e} | {b['bucket'] * 0.5:g} | {b['truth']} | {b['estimate']} | "
                                 f"{tuple(b['truthVec'])} | {tuple(b['estimateVec'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cases", help="comma list (default every non-holdout case): " + ", ".join(CASES))
    ap.add_argument("--holdout", action="store_true", help="score the HOLDOUT cases instead")
    ap.add_argument("--estimators", default="rgb,blender,vda")
    ap.add_argument("--rebuild", action="store_true", help="re-render even if a case has a saved truth")
    ap.add_argument("--tag", help="name the outputs results-<tag>.json / report-<tag>.md (the main report is left alone)")
    args = ap.parse_args()
    names = list(HOLDOUT) if args.holdout else args.cases.split(",") if args.cases else list(CASES)
    unknown = [n for n in names if n not in CASES and n not in HOLDOUT]
    if unknown:
        raise SystemExit(f"unknown cases {unknown}")
    estimators = args.estimators.split(",")
    if not BLENDER_MCP.is_dir():
        raise SystemExit(f"oside-blender-mcp not found beside this repo ({BLENDER_MCP})")
    rows = []
    for name in names:
        print(f"… {name}", file=sys.stderr, flush=True)
        rows.append(run_case(name, estimators, args.rebuild))
        r = rows[-1]
        print(f"  truth: {r['truth']}  ({r['seconds']} s)", file=sys.stderr)
        for e in estimators:
            print(f"  {e:8} {r['estimators'][e]['said']}", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""
    (OUT / f"results{suffix}.json").write_text(json.dumps(rows, indent=2))
    (OUT / f"report{suffix}.md").write_text(report(rows, estimators))
    print(report(rows, estimators))


if __name__ == "__main__":
    main()
