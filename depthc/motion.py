"""Motion measurement — the part of a depth ref that actually decides whether it works.

Everything else the checker grades (duration, file size, resolution, fps) is mechanical and
cannot fail if the code is right. What decides whether Seedance follows a reference is how
much motion survives into the depth channel and where in the clip it sits. This module
measures that, and it exists because of a specific failure.

WHY DEPTH SPACE AND NOT RGB. Measured on the four real reference clips, 2026-07-25, with both
sides decoded at the same 24 fps so the ratio means something: CLIP A's trim window carried
RGB energy of 5.98 against CLIP B's 4.93 — MORE motion than the clip that worked — yet in
depth space A collapsed to 0.95 while B held 4.27. That is 16% retention versus 87%. A prone
crawl along the ground moves parallel to the camera plane, so the bodies barely change
distance and the depth channel flattens the motion almost entirely. **RGB motion is not depth
motion, and only the second kind reaches the model.** A's generation started late; B's did
not. Nothing in the old checker had an opinion about either clip.

(An earlier pass put those figures at 11% and 52% by comparing depth sampled at 24 fps against
source sampled at 12. Per-frame deltas scale with the frame interval; the ranking survived the
error, the numbers did not. Both sides are now measured on the same frames.)

UNITS. Every energy figure is the mean absolute difference between consecutive frames of the
8-bit depth video, measured on an ASPECT-PRESERVING analysis grid with a 160 px long edge.
So 1.0 means "the average pixel changes by one grey level per frame". The mean is per PIXEL,
which means motion is measured as a SHARE OF THE FRAME: the same subject moving the same
distance reads ~34% hotter in a 2.39:1 frame than in 16:9, because it occupies more of the
shorter frame (verified empirically, and true under any grid — it is a property of the
metric, not of the resampling). That is deliberate: the model receives the ref as its whole
frame, so frame-share is the scale adherence actually happens on.

The grid preserved aspect from v2 (2026-07-25). For the full-frame mean the old fixed
160x90 grid and this one agree — all four calibration clips re-measured within ±0.02 — but
the squashed geometry fed distorted shapes to the optical-flow camera/subject split and to
sharpness, and would have mis-measured portrait sources; v2 makes the geometry faithful. A
frozen frame pushed back through the same encoder measures 0.002, so the thresholds below
are signal, not codec noise.

Energy is per FRAME, so it scales with the frame interval — figures are only comparable at
the same fps. Everything a kit ships is conformed to 24 fps; the scout pass runs at 12 and
rescales, and says so.

THRESHOLDS are calibrated against those four clips rather than derived from theory
(re-measured 2026-07-25 on the v2 grid from each kit's encoded depth.mp4; the two grids
agree on the mean, so the tiny shifts against the v1 figures are encode round-trip, not
metric change):

    A prone crawl   0.94   weak    near-static after the first half second — the clip whose
                                   generation "started very late"
    B sword duel    4.40   strong  sustained and rising; the one that behaved
    C walk-in       1.69   fine    steady, unspectacular, works
    D slow dolly    0.72   weak    a camera move with a dead first second

Two ideas here are cherry-picked from video-dataset curation pipelines (see
DOCS/PRIOR-ART.md): motion scoring is a BAND rather than a floor — too much frame-to-frame
change is jitter, cuts and whip-pans, not usable choreography — and a hard cut inside a
window is the worst possible competing signal, so it is detected and refused rather than
averaged into the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- the analysis grid --------------------------------------------------------------------
GRID_LONG_EDGE = 160
# The 16:9 case (and the test fixtures): a 160 px long edge lands on the original 160x90.
GRID_WIDTH = 160
GRID_HEIGHT = 90


def analysis_size(width: int, height: int) -> tuple[int, int]:
    """Grid dimensions for one frame shape: long edge 160, aspect preserved."""
    scale = GRID_LONG_EDGE / max(width, height)
    return max(2, round(width * scale)), max(2, round(height * scale))

# --- thresholds (calibrated on the four real clips; see module docstring) ------------------
STATIC_FLOOR = 0.30       # at or below this a stretch is as good as frozen
WEAK_ENERGY = 1.50        # below this the model has little to lock onto
STRONG_ENERGY = 3.00      # at or above this the signal is unambiguous
JITTER_ENERGY = 12.0      # above this it is shake/flicker, not choreography
DEAD_STRETCH_SECONDS = 0.75   # a dead run this long is a beat the model has to invent

# CUT DETECTION, measured rather than assumed. The first attempt flagged any frame gap 4x the
# clip's median, and it fired on all three clips that contain no cut at all: a sword impact or
# a stumble legitimately spikes several times above its own median. What separates a cut is
# not magnitude but EXTENT — a cut changes the whole frame at once, an action changes the part
# of it a body occupies. Measured on the four reference clips: the worst legitimate frame gap
# changed 24% of the pixels by more than 25 grey levels, while splicing two of the clips
# together changed 46%. The threshold sits between them.
#
# HONEST LIMIT: this runs in depth space, where two different rooms can share a structure, so
# a cut between similar geometry can slide under 0.35. It is a WARN aid, not a guarantee.
CUT_LEVEL = 25.0          # grey levels; "this pixel really changed"
CUT_FRACTION = 0.35       # share of the frame that has to change at once to call it a cut
SPIKE_MULTIPLE = 6.0      # fallback for a bare delta series with no pixels to look at
SPIKE_FLOOR = 8.0
LOW_RETENTION = 0.25      # depth energy / source energy below this = in-plane motion
NOISE_RETENTION = 1.20    # depth energy ABOVE source energy = depth flicker, not choreography
CAMERA_DOMINANT = 0.60    # camera share above this = a camera-path ref, not a body one

# --- structure: WHERE the motion sits and WHICH WAY the frame moves (v0.7.0) ---------------
# The full-frame envelope is blind to placement and direction: a subject crossing left→right
# and the same subject crossing right→left draw the same sparkline. These two layers keep
# the geometry the mean throws away, on the same analysis grid, from the same passes.
REGION_ROWS = 3           # the 3x3 region grid; cells are equal splits of the analysis grid
REGION_COLS = 3
PATH_STILL = 0.10         # grid px per frame; a median flow below this has no direction to name
PATH_RADIAL_STILL = 0.05  # px per frame of expansion/contraction below this is not a zoom
PATH_READER = "regions-2026-11"  # names the reader in the calibration's saved estimates
# --- the camera, read from the PICTURE (2026-11-11) -----------------------------------------
# flow_series reads whatever frames it is handed, and for a kit, a compare and a describe those
# were DEPTH frames. On real footage that failed (DOCS/CAMERA-PATH-CALIBRATION.md, "Real footage"):
# a depth picture of a real scene is mostly flat, so the only texture is the outline of whatever
# stands at a different distance, and in a locked-off shot those outlines are the people walking
# through. They cast most of the votes and the reader named a pan. So the camera is now read from
# the picture, by tracking points instead of dense flow, and the points that do not move with the
# rest are dropped before anything votes.
# "tracks-2026-11" was the first picture reader; "shots-2026-11" adds the camera round below.
PICTURE_READER = "shots-2026-11"  # names this reader in the calibration's saved estimates
# Tracked at this long edge: twice the analysis grid, so a slow push that moves a point a
# twentieth of a grid pixel per frame is a tenth of a pixel here, which Lucas-Kanade resolves.
TRACK_LONG_EDGE = 320
# Shi-Tomasi corners: at most this many, at the detector's own default quality, this far apart.
TRACK_MAX_POINTS = 600
TRACK_QUALITY = 0.01
TRACK_SPACING = 6
# Fewer points than this that survive the there-and-back check and the gap is not tracked at all:
# the region reader takes it, which is what it was calibrated for (clay through a long lens holds
# about 18 corners).
TRACK_MIN_POINTS = 40
# A point is kept when tracking it forward and then back lands within this many pixels of where it
# started. A point on a blur, a reflection or an edge that slides along itself does not.
TRACK_RETURN_PX = 0.5
# WHO moves with the camera. One similarity (shift, uniform scale, rotation) is fitted to the points
# by RANSAC; a point further than this from where the fit sends it is something moving on its own.
TRACK_FIT_PX = 0.75
# And the fit has to carry at least this share of the points, or nothing in the frame is the world:
# the gap names no camera. A subject that fills most of the frame still wins this vote, and the
# reader cannot tell it from a pan (the train passing the windows in the real-footage set).
# ponytail: one dominant motion, no depth cue to say which layer is the background. A second cue
# (the depth pass, when there is one) is the upgrade if a frame-filling subject ever matters.
TRACK_MIN_SHARE = 0.5
# A region of the 8x8 split votes with this many surviving points; its vote is their median travel.
TRACK_REGION_POINTS = 2
# A tracked gap counts as moving when the path averaged over this long around it clears the
# naming floors (`_settle`). Half a second: the bucket every camera word is read in.
TRACK_SETTLE_SECONDS = 0.5
# --- the camera round (2026-11-11): shots, people, "can't tell", slow moves -----------------
# The second real-footage score (DOCS/CAMERA-PATH-CALIBRATION.md) failed four ways and each block
# below answers one. Every number in it was chosen on SEEN clips, which is development signal and
# never a score: only clips this reader has not been tuned on can say whether any of it works.
#
# CUTS. A clip is split into shots before anything is read, and no fit or average crosses a cut.
# A cut is a gap where almost none of the tracked points survive the there-and-back check AND the
# brightness histogram jumps. Either cue alone fires on real footage: a passing train keeps the
# histogram churning (0.20 between frames) and a dark, smeared shot loses most of its points
# (survival 0.10), but no seen gap did both except at a cut. On the seen cuts survival was 0.03 or
# less and the histogram moved 0.12 or more.
# ponytail: two fixed bars set on nine seen cuts in three clips; a dissolve or a fade has no such
# gap and reads as one shot. An adaptive detector (PySceneDetect's AdaptiveDetector, BSD-3) is
# the upgrade if soft transitions ever matter.
CUT_SURVIVAL = 0.05
CUT_HISTOGRAM = 0.10        # half the L1 distance between the two frames' grey histograms
CUT_HISTOGRAM_ALONE = 0.30  # a frame with too few corners to judge survival (black, flat): this
CUT_HISTOGRAM_BINS = 32
CUT_MIN_CORNERS = 10        # fewer corners than this and survival is not judged (a dark frame has 20)
SHOT_MIN_SECONDS = 0.25     # a shot shorter than this is not read at all
# PEOPLE. A person is the commonest thing in frame that moves on its own, and in the seen failures
# one filled enough of it to outvote the background: dancers against a plain wall, a walker close
# to the lens. Their points are dropped before the fit (`depthc.people`, Apple Vision, macOS only).
# Without Vision, given a depth of the same frames, the points on the nearer part of the depth are
# dropped instead, since a subject usually stands nearer than what is behind it. With neither,
# every point votes, as before. A point this many px (at TRACK_LONG_EDGE) from a person is dropped
# too: its tracking window holds part of the person.
TRACK_MASK_GROW = 5
# ponytail: "near" is everything nearer than this quantile of each frame's depth, a fixed share;
# a subject no nearer than its background (a figure on a floor that runs to the lens) keeps its
# points. Splitting the depth into layers is the upgrade if the fallback ever carries weight.
DEPTH_FAR_QUANTILE = 0.5
# CAN'T TELL. A gap is written NaN, "can't tell", when the fit has nothing to stand on: a person
# mask left fewer than TRACK_MIN_POINTS points to track, or no single motion carries
# TRACK_MIN_SHARE of them (it used to read as still). A shot names no camera at all when fewer than
# SHOT_MIN_READ of its gaps could be read: a word built from the few readable gaps of a shot full
# of dancers is the guess that failed.
# ponytail: half, by judgement, not tuned; a shot read on 51% of its gaps can still be worded from
# the wrong 51%. A per-shot confidence from the spread of the fit is the upgrade.
SHOT_MIN_READ = 0.5
# SLOW MOVES. A slow push moves a point a few hundredths of a pixel per frame, under what tracking
# from one frame to the next resolves. So the zoom is also read across SLOW_BASELINE_SECONDS, one
# reading every SLOW_STRIDE_SECONDS: half a second later the same point has moved twelve times as
# far, and the tracking error has not grown twelvefold. That reading replaces the frame-to-frame
# zoom wherever it exists, and is named from PATH_RADIAL_SLOW up, not from PATH_RADIAL_STILL. The
# fit's outlier distance grows with the square root of the baseline, as tracking error does.
SLOW_BASELINE_SECONDS = 0.5
SLOW_STRIDE_SECONDS = 0.125
PATH_RADIAL_SLOW = 0.012
# WHERE the path is read. Flow only shows on pixels with texture in two directions (the aperture
# problem): a flat wall moves without showing it, and in an all-pixel median it votes zero. So the
# path is read on pixels whose structure-tensor strength clears this share of the frame's strongest
# corner (Shi-Tomasi's own default). The ceiling stops one hard silhouette from raising that bar
# over a soft background: a bright box on blurred noise silenced the noise, and the box's own
# crossing read as a pan on a locked camera.
PATH_TEXTURE_QUALITY = 0.01
PATH_TEXTURE_CEILING = 2e-4
# WHO votes. Each region of an 8x8 split of the grid with this many textured pixels casts one
# vote, its median flow. By pixel, a hard silhouette outvotes the world behind it; by region, a
# figure has only the few regions it covers. Fewer regions than this and the gap names nothing:
# the picture cannot tell a lone figure on a bare field from a camera moving past it. Measured on
# clay: 3 regions or fewer for the lone figure, 13 or more for every scene with a world in it.
PATH_REGIONS = 8
PATH_REGION_PIXELS = 3
PATH_MIN_REGIONS = 12
# The zoom. A median cannot be read first and the zoom around it: when texture sits off-centre
# (distant boxes above a bare ground), the outward flow of a push drags the median and reads as a
# tilt. So the stretch is read from opposite sides of the frame, where a shift cancels: the median
# flow on each side of centre, over how far apart those sides sit. A side needs this many votes,
# reaching this share of the way to the frame's edge, or that axis is not measured.
PATH_ZOOM_SIDE_VOTES = 2
PATH_ZOOM_REACH = 0.33
# And a stretch is a zoom only if the flow says so. A truck past boxes at different depths has one
# side faster than the other, which looks like stretch. This share of the votes must move outward
# (or inward) once the shift is out. Medians on clay: 0.78 or more on every real push and pull,
# two moves at once included, and 0.73 or less on pans, trucks and cranes, the one exception being
# a figure crossing in front of boxes, which the check in _read_camera catches.
PATH_ZOOM_AGREEMENT = 0.75
# A pace word beside the path label ("slowly pans left"). The idea is CozyClay's — its
# camera-move classifier grades tempo from physical lens travel in m/s — but those units
# do not exist here, so the bands are re-derived as TIME TO CROSS THE FRAME: a move at
# v grid-px/frame crosses the GRID_LONG_EDGE in 160/(24·v) seconds at the 24 fps analysis
# rate. Slower than a ~15 s crossing reads "slowly"; faster than ~4 s reads "quickly".
# ponytail: heuristic crossing-time bands, never calibrated against a fire — re-derive
# from measured generations before tempo is ever allowed to vote in a verdict.
PATH_SLOW = GRID_LONG_EDGE / (24.0 * 15.0)  # ≈ 0.44 px/frame
PATH_FAST = GRID_LONG_EDGE / (24.0 * 4.0)   # ≈ 1.67 px/frame

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SPARK_DEAD = "·"


def analysis_gray(frames: np.ndarray) -> np.ndarray:
    """Downsample [N,H,W] grey or [N,H,W,3] RGB to the uint8 analysis grid.

    The grid preserves the frame's aspect at a 160 px long edge (see the module docstring:
    the full-frame mean is grid-agnostic, but flow and sharpness need faithful geometry).

    INTER_AREA is the correct filter for shrinking: it averages the pixels it discards, so a
    small fast-moving subject still contributes to the delta instead of falling between
    sample points the way nearest-neighbour would let it.
    """
    import cv2

    if frames.ndim == 4 and frames.shape[-1] == 3:
        source = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    elif frames.ndim == 3:
        source = list(frames)
    else:
        raise ValueError(f"Expected [N,H,W] or [N,H,W,3] frames, got {frames.shape}")

    if not source:
        return np.zeros((0, GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

    height, width = source[0].shape[:2]
    grid_w, grid_h = analysis_size(width, height)
    return np.stack(
        [cv2.resize(f, (grid_w, grid_h), interpolation=cv2.INTER_AREA) for f in source]
    )


def frame_deltas(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames, one value per frame gap."""
    return frame_stats(frames)[0]


def frame_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame gap: (mean absolute change, fraction of pixels that really changed).

    Both come from one pass over the same difference. The mean is the motion signal; the
    fraction is what tells a cut apart from a lunge — see CUT_FRACTION above.
    """
    return frame_stats_regions(frames)[:2]


def _region_edges(size: int, cells: int) -> list[int]:
    """Integer cell boundaries that cover `size` exactly, largest remainder spread evenly."""
    return [round(size * i / cells) for i in range(cells + 1)]


def region_means(difference: np.ndarray) -> np.ndarray:
    """Per-cell mean of a [gaps,H,W] difference stack -> [gaps, REGION_ROWS, REGION_COLS].

    The cells are equal splits of the analysis grid, so a cell's mean is on the SAME scale
    as the full-frame energy (grey levels per frame per pixel) and the full-frame mean is
    the pixel-weighted mean of the cells. Which cell carries the motion is what tells a
    left→right crossing from a right→left one; the full-frame mean cannot.
    """
    if difference.ndim != 3:
        raise ValueError(f"Expected [gaps,H,W], got {difference.shape}")
    gaps, height, width = difference.shape
    rows = _region_edges(height, REGION_ROWS)
    cols = _region_edges(width, REGION_COLS)
    out = np.zeros((gaps, REGION_ROWS, REGION_COLS), dtype=np.float32)
    for r in range(REGION_ROWS):
        for c in range(REGION_COLS):
            cell = difference[:, rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
            if cell.shape[1] and cell.shape[2]:
                out[:, r, c] = cell.mean(axis=(1, 2))
    return out


def frame_stats_regions(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`frame_stats` plus the per-region means, all from the one difference pass."""
    grid = analysis_gray(frames).astype(np.float32)
    if grid.shape[0] < 2:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, np.zeros((0, REGION_ROWS, REGION_COLS), dtype=np.float32)
    difference = np.abs(np.diff(grid, axis=0))
    return (
        difference.mean(axis=(1, 2)),
        (difference > CUT_LEVEL).mean(axis=(1, 2)).astype(np.float32),
        region_means(difference),
    )


@dataclass
class FlowSeries:
    """Dense-flow summary per frame gap, on the analysis grid, in grid px per frame.

    `path[:, 0:2]` is the shift (dx, dy) the textured regions vote for, i.e. the camera
    translation; `path[:, 2]` is the RADIAL component: the zoom those votes earn, as what it
    moves a pixel at the mean distance from centre. Positive = the picture expands (push-in /
    zoom-in), negative = it contracts (pull-out). `residual` is the mean flow magnitude of
    the textured pixels around the shift — the subject moving inside the frame, and a zoom. `moving`
    marks the gaps above the static floor; on the others every figure is written as 0.0
    because flow on a still frame is estimator noise with no direction (see camera_share).
    """

    path: np.ndarray = field(repr=False)      # [gaps, 3]  dx, dy, radial
    residual: np.ndarray = field(repr=False)  # [gaps]
    moving: np.ndarray = field(repr=False)    # [gaps] bool

    @property
    def gaps(self) -> int:
        return int(self.moving.size)


def _stretch(flow: np.ndarray, offset: np.ndarray, half: float) -> float | None:
    """Flow per pixel of distance along one axis, read from opposite sides of centre.

    A shift moves both sides alike and cancels here, whichever way the texture leans.
    None when one side has too few votes, or they sit too near the centre, to measure.
    """
    plus, minus = offset > 0, offset < 0
    if plus.sum() < PATH_ZOOM_SIDE_VOTES or minus.sum() < PATH_ZOOM_SIDE_VOTES:
        return None
    far, near = float(np.median(offset[plus])), float(np.median(offset[minus]))
    if far < PATH_ZOOM_REACH * half or -near < PATH_ZOOM_REACH * half:
        return None
    return (float(np.median(flow[plus])) - float(np.median(flow[minus]))) / (far - near)


def _read_camera(
    fx: np.ndarray, fy: np.ndarray, x: np.ndarray, y: np.ndarray, half_w: float, half_h: float,
) -> tuple[float, float, float]:
    """(dx, dy, stretch) from the regions' votes: each one's flow and offset from centre."""
    shift_x, shift_y = float(np.median(fx)), float(np.median(fy))
    across, down = _stretch(fx, x, half_w), _stretch(fy, y, half_h)
    if across is None or down is None:
        stretch = across if down is None else down
    else:
        # A zoom stretches both axes. A crane stretches only the vertical, as the near ground
        # slides faster than the far.
        # ponytail: with texture on one side of centre only, one axis has to speak alone, so a truck
        # whose left half slides uniformly faster than its right reads as a zoom. A second cue
        # (depth, or a real rotation fit) is the upgrade if a clip ever needs it.
        stretch = 0.0 if across * down <= 0 else min(across, down, key=abs)
    if not stretch:
        return shift_x, shift_y, 0.0
    fit_x, fit_y = float(np.median(fx - stretch * x)), float(np.median(fy - stretch * y))
    outward = ((fx - fit_x) * x + (fy - fit_y) * y) * np.sign(stretch)
    if float((outward > 0).mean()) < PATH_ZOOM_AGREEMENT:
        return shift_x, shift_y, 0.0
    # The fit must also leave less unexplained than the plain shift. A figure filling one side of
    # a still frame has opposite sides that disagree, and votes that agree only by accident.
    plain = float(np.median(np.hypot(fx - shift_x, fy - shift_y)))
    fitted = float(np.median(np.hypot(fx - fit_x - stretch * x, fy - fit_y - stretch * y)))
    if fitted > plain:
        return shift_x, shift_y, 0.0
    return fit_x, fit_y, stretch


def flow_series(frames: np.ndarray, fps: float = 24.0) -> FlowSeries:
    """One Farneback pass per moving frame gap; everything flow-derived reads from this.

    `fps` scales the static floor: STATIC_FLOOR is calibrated at 24 fps and raw per-gap
    deltas grow with the frame interval (see camera_share for why the floor exists at all).
    A gap moves when its TEXTURED pixels change, and regions of them vote on the path (see
    PATH_TEXTURE_QUALITY). Scored against a known camera in DOCS/CAMERA-PATH-CALIBRATION.md.
    """
    import cv2

    grid = analysis_gray(frames)
    gaps = max(0, grid.shape[0] - 1)
    path = np.zeros((gaps, 3), dtype=np.float32)
    residual = np.zeros(gaps, dtype=np.float32)
    moving = np.zeros(gaps, dtype=bool)
    if gaps == 0:
        return FlowSeries(path=path, residual=residual, moving=moving)

    floor = STATIC_FLOOR * (24.0 / fps) if fps > 0 else STATIC_FLOOR
    height, width = grid.shape[1:]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    rx, ry = xs - (width - 1) / 2.0, ys - (height - 1) / 2.0
    mean_radius = float(np.hypot(rx, ry).mean())
    regions = [
        (slice(height * row // PATH_REGIONS, height * (row + 1) // PATH_REGIONS),
         slice(width * col // PATH_REGIONS, width * (col + 1) // PATH_REGIONS))
        for row in range(PATH_REGIONS) for col in range(PATH_REGIONS)
    ]

    for index, (previous, current) in enumerate(zip(grid[:-1], grid[1:])):
        strength = cv2.cornerMinEigenVal(previous, blockSize=3, ksize=3)
        top = float(strength.max())
        if top <= 0:
            continue
        mask = strength >= min(PATH_TEXTURE_QUALITY * top, PATH_TEXTURE_CEILING)
        voting = [region for region in regions if int(mask[region].sum()) >= PATH_REGION_PIXELS]
        if len(voting) < PATH_MIN_REGIONS:
            continue
        if float(np.abs(current.astype(np.float32) - previous.astype(np.float32))[mask].mean()) <= floor:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        votes = [
            (float(np.median(flow[region][..., 0][mask[region]])), float(np.median(flow[region][..., 1][mask[region]])),
             float(rx[region][mask[region]].mean()), float(ry[region][mask[region]].mean()))
            for region in voting
        ]
        dx, dy, stretch = _read_camera(*np.array(votes).T, (width - 1) / 2.0, (height - 1) / 2.0)
        moving[index] = True
        # A stretch per pixel, in the px/frame path_label's floors are written in: what it moves
        # a pixel at the grid's mean distance from centre.
        path[index] = (dx, dy, stretch * mean_radius)
        residual[index] = float(np.hypot(flow[..., 0][mask] - dx, flow[..., 1][mask] - dy).mean())
    return FlowSeries(path=path, residual=residual, moving=moving)


def _track_gray(picture: np.ndarray) -> np.ndarray:
    """The picture as uint8 grey at TRACK_LONG_EDGE, aspect kept. [N,H,W] or [N,H,W,3] in."""
    import cv2

    if picture.ndim == 4 and picture.shape[-1] == 3:
        source = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in picture]
    elif picture.ndim == 3:
        source = list(picture)
    else:
        raise ValueError(f"Expected [N,H,W] or [N,H,W,3] frames, got {picture.shape}")
    if not source:
        return np.zeros((0, 1, 1), dtype=np.uint8)
    height, width = source[0].shape[:2]
    scale = TRACK_LONG_EDGE / max(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in source])


@dataclass
class Shot:
    """One run of frames between cuts, frames [start, end), and whether its camera was read."""

    start: int
    end: int
    read: bool = True

    @property
    def gaps(self) -> slice:
        """This shot's rows of a path: gap i joins frames i and i+1, so a cut gap is in no shot."""
        return slice(self.start, max(self.start, self.end - 1))


@dataclass
class CameraReading:
    """The camera over time, read from the picture: what `camera_reading` returns.

    `path` is [gaps, 3] (dx, dy, radial) in grid px per frame, as `FlowSeries.path`, with one
    addition: a NaN row is a gap that could not be read ("can't tell"), never a still one. `shots`
    are the runs between cuts. `mask` says what was kept out of the fit: "people" (Vision),
    "depth" (the nearer part of a depth of the same frames) or "none". `radial_floor` is the zoom
    a label may name from (PATH_RADIAL_SLOW: the zoom is read across half a second here).
    """

    path: np.ndarray = field(repr=False)
    shots: list[Shot] = field(default_factory=list)
    mask: str = "none"
    radial_floor: float = PATH_RADIAL_SLOW


def _track(
    previous: np.ndarray, current: np.ndarray, allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Corners of `previous` (where `allowed`), tracked to `current` and back.

    (start, end, survival): the points that came back within TRACK_RETURN_PX, where they went,
    and the share of the corners that did. None when there were fewer than CUT_MIN_CORNERS
    corners, too few to say even whether the frame cut; a reading needs TRACK_MIN_POINTS back.
    """
    import cv2

    points = cv2.goodFeaturesToTrack(previous, TRACK_MAX_POINTS, TRACK_QUALITY, TRACK_SPACING, mask=allowed)
    if points is None or len(points) < CUT_MIN_CORNERS:
        return None
    there, found, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3)
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, there, None, winSize=(21, 21), maxLevel=3)
    kept = (
        (found.ravel() == 1) & (found_back.ravel() == 1)
        & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) < TRACK_RETURN_PX)
    )
    return points[kept].reshape(-1, 2), there[kept].reshape(-1, 2), float(kept.mean())


def _fit(
    start: np.ndarray, end: np.ndarray, size: tuple[int, int], grid_w: int, grid_h: int,
    mean_radius: float, frames: int = 1,
) -> tuple[float, float, float] | None | bool:
    """Tracked points to (dx, dy, radial) in grid px per frame, over a span of `frames` frames.

    None when no single motion carries TRACK_MIN_SHARE of the points (the gap names no camera).
    False when the points that do bunch in too few regions to vote (the region reader decides).
    """
    import cv2

    fit, inliers = cv2.estimateAffinePartial2D(
        start, end, method=cv2.RANSAC, ransacReprojThreshold=TRACK_FIT_PX * float(np.sqrt(frames)),
        maxIters=2000, confidence=0.995,
    )
    if fit is None or float(inliers.mean()) < TRACK_MIN_SHARE:
        return None
    world = inliers.ravel() == 1
    start, end = start[world], end[world]

    height, width = size
    to_grid = grid_w / width
    travel = (end - start) * to_grid / frames
    offset = (start - np.array([(width - 1) / 2.0, (height - 1) / 2.0])) * to_grid
    column = np.clip((start[:, 0] * PATH_REGIONS / width).astype(int), 0, PATH_REGIONS - 1)
    row = np.clip((start[:, 1] * PATH_REGIONS / height).astype(int), 0, PATH_REGIONS - 1)
    cell = row * PATH_REGIONS + column
    votes = [
        (float(np.median(travel[cell == k, 0])), float(np.median(travel[cell == k, 1])),
         float(offset[cell == k, 0].mean()), float(offset[cell == k, 1].mean()))
        for k in np.unique(cell) if int((cell == k).sum()) >= TRACK_REGION_POINTS
    ]
    if len(votes) < PATH_MIN_REGIONS:
        # Points enough, but bunched in a few regions: the same picture PATH_MIN_REGIONS refuses,
        # a figure on a bare field. The region reader decides, by its own rule.
        return False
    dx, dy, stretch = _read_camera(*np.array(votes, dtype=np.float64).T, (grid_w - 1) / 2.0, (grid_h - 1) / 2.0)
    return dx, dy, stretch * mean_radius


def _cuts(tracked: np.ndarray, survival: np.ndarray) -> np.ndarray:
    """The gaps that are cuts: [gaps] bool, from point survival and the histogram jump (see CUT_SURVIVAL)."""
    import cv2

    hist = np.stack([
        cv2.calcHist([frame], [0], None, [CUT_HISTOGRAM_BINS], [0, 256]).ravel() for frame in tracked
    ])
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    jump = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)
    unknown = np.isnan(survival)
    return np.where(
        unknown, jump >= CUT_HISTOGRAM_ALONE,
        (np.nan_to_num(survival, nan=1.0) <= CUT_SURVIVAL) & (jump >= CUT_HISTOGRAM),
    )


def _shots(frames: int, cuts: np.ndarray) -> list[Shot]:
    """The runs of frames between cut gaps; gap g is a cut between frame g and frame g+1."""
    edges = [0, *(int(g) + 1 for g in np.flatnonzero(cuts)), frames]
    return [Shot(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _slow_radial(
    tracked: np.ndarray, allowed: list[np.ndarray | None], shots: list[Shot], unread: np.ndarray,
    fps: float, grid_w: int, grid_h: int, mean_radius: float,
) -> np.ndarray:
    """The zoom read across SLOW_BASELINE_SECONDS, per gap: [gaps], NaN where nothing covered it."""
    gaps = tracked.shape[0] - 1
    span = max(2, int(round(SLOW_BASELINE_SECONDS * fps)))
    stride = max(1, int(round(SLOW_STRIDE_SECONDS * fps)))
    total = np.zeros(gaps, dtype=np.float64)
    count = np.zeros(gaps, dtype=np.int32)
    for shot in shots:
        last = shot.end - 1 - span  # the last first frame whose partner is still in this shot
        if last < shot.start:
            continue
        starts = list(range(shot.start, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        for first in starts:
            if unread[first:first + span].mean() > 0.5:
                continue
            tracks = _track(tracked[first], tracked[first + span], allowed[first])
            if tracks is None or len(tracks[0]) < TRACK_MIN_POINTS:
                continue
            reading = _fit(tracks[0], tracks[1], tracked.shape[1:], grid_w, grid_h, mean_radius, frames=span)
            if not isinstance(reading, tuple):
                continue
            total[first:first + span] += reading[2]
            count[first:first + span] += 1
    out = np.full(gaps, np.nan, dtype=np.float64)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def _camera_reading_from(
    grid: np.ndarray, tracked: np.ndarray, fps: float = 24.0, people: np.ndarray | None = None,
) -> CameraReading:
    """camera_reading on frames already reduced: the analysis grid, the TRACK_LONG_EDGE greys, and
    `people`, [N, H, W] bool at the greys' size, True where the fit must not look (or None)."""
    import cv2

    frames = tracked.shape[0]
    gaps = max(0, frames - 1)
    path = np.zeros((gaps, 3), dtype=np.float32)
    if gaps == 0:
        return CameraReading(path=path, shots=[Shot(0, frames, read=False)] if frames else [])
    grid_h, grid_w = grid.shape[1:]
    ys, xs = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
    mean_radius = float(np.hypot(xs - (grid_w - 1) / 2.0, ys - (grid_h - 1) / 2.0).mean())
    allowed: list[np.ndarray | None] = [None] * frames
    if people is not None:
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TRACK_MASK_GROW + 1,) * 2)
        for index, mask in enumerate(people):
            if mask.any():
                allowed[index] = np.where(cv2.dilate(mask.astype(np.uint8), grow) > 0, 0, 255).astype(np.uint8)
    cv2.setRNGSeed(0)  # RANSAC draws at random; the same clip must read the same twice

    unread = np.zeros(gaps, dtype=bool)      # can't tell
    untracked = np.zeros(gaps, dtype=bool)   # the region reader's to read
    survival = np.full(gaps, np.nan)
    for index in range(gaps):
        tracks = _track(tracked[index], tracked[index + 1], allowed[index])
        # A cut is judged on every corner, people included: on what the mask leaves, a train
        # rushing past a masked bystander loses its points on every gap and reads as cut after cut.
        whole = tracks if allowed[index] is None else _track(tracked[index], tracked[index + 1])
        if whole is not None:
            survival[index] = whole[2]
        reading = False
        if tracks is not None and len(tracks[0]) >= TRACK_MIN_POINTS:
            reading = _fit(tracks[0], tracks[1], tracked.shape[1:], grid_w, grid_h, mean_radius)
        if reading is None:
            unread[index] = True
        elif reading is False:
            # With a mask in this frame, too little was left to read, and the region reader
            # cannot leave the people out: can't tell. Without one, the region reader decides.
            if allowed[index] is not None:
                unread[index] = True
            else:
                untracked[index] = True
        else:
            path[index] = reading

    cuts = _cuts(tracked, survival)
    shots = _shots(frames, cuts)
    unread |= cuts
    for shot in shots:
        if (shot.end - shot.start) / fps < SHOT_MIN_SECONDS:
            shot.read = False
            unread[shot.gaps] = True
    untracked &= ~unread
    if untracked.any():
        # Filled in BEFORE settling: a tracked gap is judged against what really happened around
        # it, and a hole where the region reader's answer belongs would read as stillness there.
        fallback = flow_series(grid, fps).path
        path[untracked] = fallback[untracked]

    slow = _slow_radial(tracked, allowed, shots, unread, fps, grid_w, grid_h, mean_radius)
    path[unread] = np.nan
    for shot in shots:
        rows = shot.gaps
        if not shot.read or rows.stop <= rows.start:
            shot.read = False
            path[rows] = np.nan
            continue
        part = path[rows]
        # Where the frame-to-frame zoom clears PATH_RADIAL_STILL it stands: across half a second a
        # dolly past things at different distances fits the far layer and reads a fraction of
        # its push (seen: 0.01 against 0.14). Under that floor the half-second reading replaces
        # it, and may be named from PATH_RADIAL_SLOW: there it is the quieter of the two (a flower
        # swaying on a locked-off shot read 0.016 frame to frame and 0.003 across half a second).
        quick = _around(part, fps)[:, 2]
        swap = np.isfinite(slow[rows]) & ~untracked[rows] & np.isfinite(part).all(axis=1)
        swap &= np.abs(quick) < PATH_RADIAL_STILL
        part[swap, 2] = slow[rows][swap]
        path[rows] = _settle(part, fps, untracked[rows], np.where(swap, PATH_RADIAL_SLOW, PATH_RADIAL_STILL))
        if float(np.isfinite(path[rows]).all(axis=1).mean()) < SHOT_MIN_READ:
            shot.read = False
            path[rows] = np.nan
    return CameraReading(path=path, shots=shots)


def _camera_path_from(grid: np.ndarray, tracked: np.ndarray, fps: float = 24.0) -> np.ndarray:
    """The path of `_camera_reading_from` with no mask: what calibration scripts call."""
    return _camera_reading_from(grid, tracked, fps).path


def _settle(
    path: np.ndarray, fps: float, keep: np.ndarray | None = None,
    radial_floor: float | np.ndarray = PATH_RADIAL_STILL,
) -> np.ndarray:
    """Zero the gaps where the camera is not moving, as `flow_series` does with its static floor.

    Everything that reads a path takes a zero row to mean "still" and averages the rest
    (`MotionProfile.camera_path`, the half-second buckets). Tracking never returns an exact zero:
    a held shot comes back as hundredths of a pixel of noise on every gap, and 169 such gaps
    averaged with 10 gaps of a real pan would read as static. So a gap is kept only where the
    path around it, averaged over TRACK_SETTLE_SECONDS, would earn a word from `path_label`.
    Averaging first is what keeps a slow push: each gap of it is under the floor's noise, the
    half-second around it is not. Gaps marked in `keep` count toward the averages and are never
    zeroed here. NaN rows (can't tell) are left as they are and count toward no average.

    The zoom is judged against `radial_floor` (one per gap, or one for all), and a kept gap whose
    zoom does not clear it on its own loses the zoom: a pan's frame-to-frame zoom is noise, and
    named from the slow floor it would turn every pan into "pans left, pushes in".

    ponytail: a gate per gap plus "average the survivors" downstream means a bucket where the
    camera reverses can be worded from the half that survived. `flow_series` has the same shape
    with its static floor. The upgrade is for buckets to average every gap, which changes what
    the calibrated floors mean and so needs the clay re-scored.
    """
    gaps = path.shape[0]
    if gaps == 0:
        return path
    finite = np.isfinite(path).all(axis=1)
    around = _around(path, fps)
    zoom = np.abs(around[:, 2]) >= radial_floor
    moving = (np.hypot(around[:, 0], around[:, 1]) >= PATH_STILL) | zoom
    told = np.zeros(gaps, dtype=bool) if keep is None else keep.astype(bool)
    settled = path.copy()
    settled[finite & ~(moving | told)] = 0.0
    settled[finite & ~zoom & ~told, 2] = 0.0
    return settled


def _around(path: np.ndarray, fps: float) -> np.ndarray:
    """Each gap's path averaged over the TRACK_SETTLE_SECONDS around it, NaN rows left out."""
    gaps = path.shape[0]
    span = max(1, min(gaps, int(round(TRACK_SETTLE_SECONDS * fps))))
    kernel = np.ones(span, dtype=np.float64) / span
    finite = np.isfinite(path).all(axis=1)
    values = np.where(finite[:, None], path, 0.0)
    # 'same' with the tail counted honestly: divide by how many readable gaps each window held
    held = np.maximum(np.convolve(finite.astype(np.float64), kernel, mode="same"), 1e-12)
    return np.stack([np.convolve(values[:, k], kernel, mode="same") / held for k in range(path.shape[1])], axis=1)


def camera_reading(
    picture: np.ndarray, fps: float = 24.0, *, depth: np.ndarray | None = None,
    near: str | None = "bright", people: str | np.ndarray | None = "auto",
) -> CameraReading:
    """The camera over time, read from the PICTURE, shot by shot, with people left out of the fit.

    Points are tracked from each frame to the next, the ones that do not move with the rest are
    dropped by a RANSAC similarity fit, and the survivors vote by region through `_read_camera`,
    the same shift-and-stretch reading the region reader uses. Before that, the clip is cut into
    shots (CUT_SURVIVAL) and the points on people are left out (`depthc.people`; with no Vision,
    the nearer part of `depth`, where `near` says which end of the depth is near: "bright", the
    model's own, or "dark" for an inverted one). The zoom is also read across half a second
    (SLOW_BASELINE_SECONDS). A gap with too little to track and no mask is read by `flow_series`
    on the same picture. A gap or shot that cannot be read is NaN. Never hand this depth frames as
    the picture: see PICTURE_READER for why.

    `people`: "auto" asks Vision, None leaves everyone in, an [N, H, W] bool array is used as given
    (at any size; it is resized to the tracking size).
    """
    import cv2

    tracked = _track_gray(picture)
    grid = analysis_gray(picture)
    size = (tracked.shape[2], tracked.shape[1])
    masks, how = None, "none"
    if isinstance(people, str):
        from depthc import people as peoplelib

        found = peoplelib.find(picture, size=size) if people == "auto" else None
        if found is not None:
            masks, how = found.masks, "people"
    elif people is not None:
        masks = np.stack([
            cv2.resize(np.asarray(m, dtype=np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in people
        ])
        how = "people"
    if masks is None and depth is not None and near in ("bright", "dark") and depth.shape[0] == picture.shape[0]:
        masks, how = _near_masks(depth, size, near), "depth"
    reading = _camera_reading_from(grid, tracked, fps, masks)
    reading.mask = how
    return reading


def _near_masks(depth: np.ndarray, size: tuple[int, int], near: str) -> np.ndarray:
    """[N, H, W] bool at `size` (width, height): the part of each depth frame nearer than DEPTH_FAR_QUANTILE."""
    import cv2

    out = []
    for frame in depth:
        grey = frame if frame.ndim == 2 else frame[..., 0]
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if near == "dark":
            grey = 255.0 - grey
        out.append(grey > float(np.quantile(grey, DEPTH_FAR_QUANTILE)))
    return np.stack(out)


def camera_path(picture: np.ndarray, fps: float = 24.0, **kwargs) -> np.ndarray:
    """`camera_reading(...).path`: [gaps, 3] (dx, dy, radial) in grid px per frame, NaN = can't tell.

    The same figures and units as `FlowSeries.path`, so every word, bucket and tempo band built on
    that reads this unchanged, once its NaN rows are left out. Never hand this depth frames.
    """
    return camera_reading(picture, fps, **kwargs).path


PICTURE_SLACK = 2  # frames: two decodes of one clip at the same fps can differ by a frame or two


def fit_picture(picture: np.ndarray, count: int) -> np.ndarray | None:
    """`picture` cut or held to exactly `count` frames, or None when it is not the same clip.

    For a picture decoded from a different file than the depth it rides beside (a kit's
    `source.mp4` next to its `depth.mp4`). Within PICTURE_SLACK frames the tail is dropped, or
    the last frame held, which reads as a still gap. Further apart than that the two are not
    frame for frame and the caller reads the path from what it has.
    """
    have = picture.shape[0]
    if have == 0 or abs(have - count) > PICTURE_SLACK:
        return None
    if have >= count:
        return picture[:count]
    return np.concatenate([picture, np.repeat(picture[-1:], count - have, axis=0)])


def camera_share(frames: np.ndarray, fps: float = 24.0) -> float:
    """How much of the movement is the camera rather than the subject, 0.0-1.0.

    Farneback dense flow, then split each frame pair's field into the shift the textured
    regions vote for (a pan, a dolly, a tracking move) and the residual around it (a subject
    moving inside the frame). The share is |shift| / (|shift| + mean residual).

    ONLY MOVING FRAMES COUNT. Frame gaps below the static floor are measured but excluded from
    the average, because flow on a still frame is estimator noise with no direction, and
    including it drags every share toward zero. CLIP D — a slow dolly with a dead first half —
    measured 3% "whole-frame" across the whole clip and reads correctly once the frozen half
    stops voting.

    HONEST LIMIT: a *translation* is the only camera motion this separates cleanly. A zoom,
    orbit or roll moves pixels in opposing directions, so its shift is near zero and it books
    as subject motion. Treat a high number as "definitely camera", a low one as "not a
    simple camera translation" rather than "definitely subject". (The radial term in
    `flow_series` names a zoom/push separately, but does not fold it into this share — the
    share's thresholds were calibrated without it.)
    """
    return camera_share_from(flow_series(frames, fps))


def camera_share_from(flow: FlowSeries) -> float:
    """The share, computed from an existing flow pass (identical numbers to camera_share)."""
    if not flow.moving.any():
        return 0.0
    camera = np.hypot(flow.path[flow.moving, 0], flow.path[flow.moving, 1])
    subject = flow.residual[flow.moving]
    total = float(camera.mean() + subject.mean())
    return round(float(camera.mean()) / total, 3) if total > 0 else 0.0


def path_label(dx: float, dy: float, radial: float, radial_floor: float = PATH_RADIAL_STILL) -> str:
    """Name a mean camera path in camera terms, from the pixel motion it produced.

    Pixels moving RIGHT means the camera panned LEFT; pixels moving DOWN, a tilt UP;
    expansion, a push-in. Translation and radial are named independently ("pans left,
    pushes in"). Below the floors it is "static". This is a translation-and-zoom
    vocabulary by construction — an orbit or a roll has no honest name here. `radial_floor`
    is the zoom the path's reader can resolve: PATH_RADIAL_SLOW for a picture path, whose zoom
    is read across half a second (`MotionProfile.radial_floor` carries it). That lower floor
    names a zoom only when nothing else moves: a pan or tilt past things at different distances
    shows a small zoom of its own (seen on real pans: 0.01 to 0.05), so beside one a zoom still
    has to clear PATH_RADIAL_STILL.
    """
    parts = []
    moved = bool(np.hypot(dx, dy) >= PATH_STILL)
    if moved:
        if abs(dx) >= abs(dy):
            parts.append("pans left" if dx > 0 else "pans right")
        else:
            parts.append("tilts up" if dy > 0 else "tilts down")
    if abs(radial) >= (max(radial_floor, PATH_RADIAL_STILL) if moved else radial_floor):
        parts.append("pushes in" if radial > 0 else "pulls out")
    return ", ".join(parts) if parts else "static"


def path_speed(dx: float, dy: float, radial: float) -> float:
    """One pace number for a path sample: translation plus expansion, grid px/frame."""
    return float(np.hypot(dx, dy) + abs(radial))


def path_tempo(speed: float) -> str:
    """"slowly" / "quickly" / "" for a NAMED path; meaningless beside "static"."""
    if speed >= PATH_FAST:
        return "quickly"
    if speed < PATH_SLOW:
        return "slowly"
    return ""


def sparkline(values, floor: float = STATIC_FLOOR, ceiling: float | None = None) -> str:
    """Render a series as blocks, with dead buckets shown as `·` so they are visible.

    A dead stretch is the finding most worth seeing at a glance, and it is exactly the thing
    a normalised bar chart hides: scaled against a weak clip's own maximum, near-nothing
    still draws a bar.
    """
    values = [float(v) for v in values]
    if not values:
        return ""
    top = ceiling if ceiling is not None else max(values)
    if top <= 0:
        return SPARK_DEAD * len(values)

    out = []
    for value in values:
        if value <= floor:
            out.append(SPARK_DEAD)
            continue
        level = int(value / top * len(SPARK_BLOCKS))
        out.append(SPARK_BLOCKS[min(level, len(SPARK_BLOCKS) - 1)])
    return "".join(out)


@dataclass
class DeadStretch:
    """The longest run of frames with no usable motion, and where in the clip it sits."""

    start: float
    seconds: float
    where: str  # "head" | "tail" | "middle"

    def as_dict(self) -> dict:
        return {
            "startSeconds": round(self.start, 2),
            "seconds": round(self.seconds, 2),
            "where": self.where,
        }


@dataclass
class MotionProfile:
    """Everything measurable about one clip's motion, in depth space."""

    fps: float
    deltas: np.ndarray = field(repr=False)
    # The frame rate the DELTAS are expressed in, which is not always the rate they were
    # SAMPLED at. `fps` is the time axis (i / fps is a timestamp); `delta_fps` is the scale
    # of the magnitudes. They agree for everything measure() builds, whose deltas are raw.
    # They diverge for scout_clip(), which samples at 12 and rescales its deltas to
    # 24fps-equivalent so its numbers are comparable with a kit's — it reports fps=12 for
    # the timestamps and delta_fps=24 for the magnitudes.
    #
    # Splitting them fixed a live wrong answer: `spikes` read `fps` as the scale and
    # rescaled its absolute floor a SECOND time, landing the scout's threshold at 16.0
    # instead of 8.0. Measured on real footage 2026-08-24 — the same 31s clip ranked
    # 13-18s at --scout-fps 12 and 25-30s at 24, with a 7.6 spike neither pass called a
    # cut. None = "same as fps", the raw case.
    delta_fps: float | None = None
    bucket_seconds: float = 0.5
    # None = the flow pass was skipped; 0.0 = measured, and it is pure subject motion.
    # Collapsing the two made a genuine zero drop the motion-type finding entirely.
    camera: float | None = None
    changed: np.ndarray | None = field(default=None, repr=False)
    # Structure layers (v0.7.0). None = not measured (a bare-delta profile, or a manifest
    # read back). regions: [gaps, REGION_ROWS, REGION_COLS] per-cell energy; path: the
    # FlowSeries.path [gaps, 3] (dx, dy, radial) with zeros on static gaps.
    regions: np.ndarray | None = field(default=None, repr=False)
    path: np.ndarray | None = field(default=None, repr=False)
    # The camera round (2026-11-11). A picture-read path has NaN rows where the camera could not
    # be read, comes in shots split at the cuts, and names a zoom from a lower floor. None / the
    # defaults = a path read the old way, which has none of the three.
    shots: list[Shot] | None = field(default=None, repr=False)
    radial_floor: float = PATH_RADIAL_STILL
    camera_mask: str | None = None  # "people" | "depth" | "none": what was kept out of the fit

    # ------------------------------------------------------------------ scalars
    @property
    def delta_scale(self) -> float:
        """The fps the magnitudes are expressed in — `delta_fps`, or `fps` when unset."""
        return float(self.delta_fps if self.delta_fps is not None else self.fps)

    @property
    def energy(self) -> float:
        return round(float(self.deltas.mean()), 3) if self.deltas.size else 0.0

    @property
    def median(self) -> float:
        return round(float(np.median(self.deltas)), 3) if self.deltas.size else 0.0

    @property
    def peak(self) -> float:
        return round(float(self.deltas.max()), 3) if self.deltas.size else 0.0

    @property
    def seconds(self) -> float:
        return round((self.deltas.size + 1) / self.fps, 3) if self.fps else 0.0

    def window_energy(self, start: float, end: float) -> float:
        """Mean energy over a time slice, for head/tail questions."""
        if not self.deltas.size:
            return 0.0
        lo = max(0, int(round(start * self.fps)))
        hi = min(self.deltas.size, int(round(end * self.fps)))
        if hi <= lo:
            return 0.0
        return round(float(self.deltas[lo:hi].mean()), 3)

    @property
    def head(self) -> float:
        return self.window_energy(0.0, DEAD_STRETCH_SECONDS)

    @property
    def tail(self) -> float:
        return self.window_energy(max(0.0, self.seconds - DEAD_STRETCH_SECONDS), self.seconds)

    # ------------------------------------------------------------------ shapes
    @property
    def buckets(self) -> list[float]:
        step = max(1, int(round(self.bucket_seconds * self.fps)))
        return [
            round(float(self.deltas[i:i + step].mean()), 2)
            for i in range(0, self.deltas.size, step)
        ]

    @property
    def spark(self) -> str:
        return sparkline(self.buckets)

    @property
    def dead_stretch(self) -> DeadStretch | None:
        """Longest run below the static floor, if it lasts long enough to matter."""
        if not self.deltas.size:
            return None
        dead = self.deltas <= STATIC_FLOOR
        best_start, best_length, run_start = 0, 0, None
        for index, is_dead in enumerate([*dead, False]):
            if is_dead and run_start is None:
                run_start = index
            elif not is_dead and run_start is not None:
                if index - run_start > best_length:
                    best_start, best_length = run_start, index - run_start
                run_start = None
        if best_length / self.fps < DEAD_STRETCH_SECONDS:
            return None

        end = best_start + best_length
        # 0.25 s of slop: a 2-3 frame blip before a frozen opening is still a late start in
        # practice, and the old 0.1 s window classed it "middle" — burying the CLIP A
        # finding under the generic dead-stretch label.
        slop = max(1, int(round(0.25 * self.fps)))
        where = "middle"
        if best_start <= slop:
            where = "head"
        elif end >= dead.size - slop:
            where = "tail"
        return DeadStretch(best_start / self.fps, best_length / self.fps, where)

    @property
    def spikes(self) -> list[float]:
        """Timestamps of frame gaps that look like a cut, not choreography.

        Uses the changed-pixel fraction when the pixels were available to measure it, which is
        every profile that came from `measure()`. A profile assembled from a bare delta series
        — a synthetic one, or a manifest read back from disk — falls back to a deliberately
        blunt magnitude rule, because a series alone cannot distinguish extent from intensity.
        """
        if not self.deltas.size:
            return []
        # The changed-pixel CUT_FRACTION was calibrated at 24 fps and is only honest there:
        # at the scout's 12 fps a legitimate fast action sweeps roughly twice the pixels per
        # frame gap (verified empirically: 2.0x) and blows through 0.35, flagging a sword
        # impact as a cut and quartering the best window's score. Any non-24 fps profile
        # falls back to the magnitude rule, which self-normalises — deltas and median scale
        # with the frame interval together.
        if (
            self.changed is not None
            and self.changed.size == self.deltas.size
            and abs(self.fps - 24.0) <= 0.5
        ):
            return [
                round(i / self.fps, 2)
                for i, fraction in enumerate(self.changed)
                if fraction > CUT_FRACTION
            ]
        if self.deltas.size < 3:
            return []
        # The multiple self-normalises (deltas and median scale with the frame interval
        # together) but the absolute floor does not: calibrated at 24 fps, it sits ~2.5×
        # too high at 60 fps, where per-gap deltas shrink by the same factor — a real cut
        # would slide under an unscaled 8.0 and ship inside the ref undetected.
        #
        # It scales by the DELTA scale, never the time axis. A profile whose deltas are
        # already 24fps-equivalent needs no correction at all, and applying one anyway is
        # what hid cuts from the scout.
        scale = self.delta_scale
        floor = SPIKE_FLOOR * (24.0 / scale) if scale > 0 else SPIKE_FLOOR
        limit = max(floor, self.median * SPIKE_MULTIPLE)
        return [round(i / self.fps, 2) for i, d in enumerate(self.deltas) if d > limit]

    # ---------------------------------------------------------------- structure
    @property
    def region_energy(self) -> list[list[float]] | None:
        """Mean energy per cell over the whole clip, rows top→bottom, cols left→right."""
        if self.regions is None or not self.regions.size:
            return None
        return [[round(float(v), 3) for v in row] for row in self.regions.mean(axis=0)]

    def region_buckets(self, bucket_seconds: float | None = None) -> np.ndarray | None:
        """[buckets, rows, cols] mean per cell per FULL bucket (partial tail dropped)."""
        if self.regions is None or not self.regions.size:
            return None
        step = max(1, int(round((bucket_seconds or self.bucket_seconds) * self.fps)))
        n = self.regions.shape[0] - step + 1
        if n <= 0:
            return np.zeros((0, REGION_ROWS, REGION_COLS), dtype=np.float32)
        return np.stack(
            [self.regions[i:i + step].mean(axis=0) for i in range(0, n, step)]
        )

    def _path_mean(self, rows: np.ndarray) -> dict | None:
        """Mean over the MOVING readable rows: dx, dy, radial (grid px/frame) + label; None
        when no row could be read."""
        finite = np.isfinite(rows).all(axis=1)
        if not finite.any():
            return None
        moving = finite & np.any(np.nan_to_num(rows) != 0, axis=1)
        if not moving.any():
            return {"dx": 0.0, "dy": 0.0, "radial": 0.0, "label": "static"}
        dx, dy, radial = (float(v) for v in rows[moving].mean(axis=0))
        return {
            "dx": round(dx, 3), "dy": round(dy, 3), "radial": round(radial, 3),
            "label": path_label(dx, dy, radial, self.radial_floor),
        }

    @property
    def camera_path(self) -> dict | None:
        """Mean camera path over the MOVING gaps: dx, dy, radial (grid px/frame) + label.

        None when there is no path, and when no gap of it could be read (can't tell): either
        way there is no camera word to give. `camera_shots` says which.
        """
        if self.path is None or not self.path.size:
            return None
        return self._path_mean(self.path)

    @property
    def camera_shots(self) -> list[dict] | None:
        """Per shot: where it sits and the camera word read in it (None = can't tell)."""
        if self.shots is None or self.path is None:
            return None
        out = []
        for shot in self.shots:
            mean = self._path_mean(self.path[shot.gaps]) if shot.read else None
            out.append({
                "startSeconds": round(shot.start / self.fps, 2),
                "endSeconds": round(shot.end / self.fps, 2),
                "label": mean["label"] if mean else None,
            })
        return out

    def as_dict(self) -> dict:
        dead = self.dead_stretch
        return {
            "energy": self.energy,
            "median": self.median,
            "peak": self.peak,
            "headEnergy": self.head,
            "tailEnergy": self.tail,
            "cameraShare": self.camera,
            "bucketSeconds": self.bucket_seconds,
            "buckets": self.buckets,
            "sparkline": self.spark,
            "deadStretch": dead.as_dict() if dead else None,
            "spikeSeconds": self.spikes,
            "regionEnergy": self.region_energy,
            "cameraPath": self.camera_path,
            # Only on a profile whose camera was read from the picture: the shots, each with its
            # own camera word or null for "can't tell". A reader that predates it ignores it.
            **({"cameraShots": self.camera_shots} if self.shots is not None else {}),
            # v2: aspect-preserving grid. Energy means agree with v1's fixed "160x90";
            # the flow-based cameraShare is the figure that may differ on non-16:9 kits.
            "grid": f"long-edge {GRID_LONG_EDGE} (aspect-preserving, v2)",
            "fps": round(self.fps, 3),
        }


def measure(
    frames: np.ndarray, fps: float, with_flow: bool = True, picture: np.ndarray | None = None,
    near: str | None = "bright",
) -> MotionProfile:
    """Profile one clip. `with_flow=False` skips the optical-flow pass on very long clips.

    `picture` is the same clip as the eye sees it, frame for frame, when `frames` are its depth.
    Given, the camera PATH is read from it (`camera_reading`) and everything else, the camera
    share included, still comes from `frames`. Without it the path is read from `frames` as
    before, which is right only when `frames` are themselves a picture.

    When Vision cannot find the people, `frames` stand in as the depth that leaves the near part
    out of the camera fit, unless `picture` IS `frames` (`describe --rgb`, no depth) or `near` is
    None (a ref that is not a depth, the edges variant). `near` says which end is near: "bright"
    as the model writes it, "dark" for an inverted depth.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    deltas, changed, regions = frame_stats_regions(frames)
    flow = flow_series(frames, fps) if with_flow else None
    path = flow.path if flow is not None else None
    reading = None
    if with_flow and picture is not None:
        if picture.shape[0] != frames.shape[0]:
            raise ValueError(
                f"picture has {picture.shape[0]} frames and the clip {frames.shape[0]}: "
                "they must be the same frames"
            )
        reading = camera_reading(picture, fps, depth=None if picture is frames else frames, near=near)
        path = reading.path
    return MotionProfile(
        fps=float(fps),
        deltas=deltas,
        changed=changed,
        camera=camera_share_from(flow) if flow is not None else None,
        regions=regions,
        path=path,
        shots=reading.shots if reading is not None else None,
        radial_floor=reading.radial_floor if reading is not None else PATH_RADIAL_STILL,
        camera_mask=reading.mask if reading is not None else None,
    )


def retention(depth: MotionProfile, source: MotionProfile) -> float | None:
    """Fraction of the source's frame-to-frame motion that survived into depth.

    The number that catches an A-shaped clip: motion parallel to the camera plane looks fine
    in RGB and nearly vanishes in depth. Only meaningful when both profiles were measured on
    the same frames at the same fps — which is exactly what a kit build has to hand.
    """
    if not source.energy:
        return None
    return round(depth.energy / source.energy, 3)


def manifest_block(depth: MotionProfile, source: MotionProfile | None = None) -> dict:
    """The `motion` object recorded in kit.json."""
    block = depth.as_dict()
    if source is not None:
        block["sourceEnergy"] = source.energy
        block["retention"] = retention(depth, source)
    return block


# ---------------------------------------------------------------------- window selection

@dataclass
class Window:
    """One candidate trim, scored."""

    start: float
    end: float
    energy: float
    head: float
    score: float
    dead: bool
    cut: bool
    sharpness: float = 1.0
    balance: float = 1.0
    # Share of this window's people samples with two or more people in frame (`scout_clip`, Apple
    # Vision). None = not counted (no Vision here). Reported in the note; never in the score.
    figures: float | None = None

    @property
    def seconds(self) -> float:
        return round(self.end - self.start, 2)

    def note(self) -> str:
        notes = []
        if self.figures is not None and self.figures >= SECOND_FIGURE_SHARE:
            notes.append(f"a second figure in {self.figures * 100:.0f}% of it")
        if self.cut:
            notes.append("contains a cut")
        if self.dead:
            notes.append("has a dead stretch")
        if self.balance < 0.5:
            notes.append(
                "back-loaded" if self.head < self.energy else "front-loaded"
            )
        if self.sharpness < 0.7:
            notes.append(f"soft ({self.sharpness:.2f} of this clip's best sharpness)")
        return "; ".join(notes)


def score_windows(
    profile: MotionProfile,
    target_seconds: float,
    *,
    stride_seconds: float = 0.25,
    sharpness: np.ndarray | None = None,
) -> list[Window]:
    """Rank every target-length window in a clip. Best first.

    The score is the window's own energy, multiplied by three penalties that encode what went
    wrong on the four real clips:

      head factor    a window whose first 0.75 s is much weaker than its own average produces
                     a generation that sits still and then starts — the CLIP A complaint.
      balance        a lopsided window is a mean hiding a shape. Verified on CLIP A: the
                     top-scoring 5 s window turned out to be 2.5 s of empty forest followed by
                     the fall, which the head factor missed because swaying trees kept the
                     first 0.75 s off the floor. Comparing the two halves catches both a slow
                     build and a beat that dies out.
      cut penalty    a hard cut inside the ref is a second motion signal; heavily penalised.
      dead penalty   a dead run anywhere inside is a beat the model will have to invent.

    Sharpness (variance of the Laplacian per frame, relative to this clip's best) is reported
    but deliberately NOT scored. Blur is what talked us out of CLIP A's real motion in the
    first place: the fall was the signal, and it was rejected for being soft. A human can see
    "soft" in the note and decide; the scorer should not quietly prefer a still frame.

    THE LIMITATION THAT MATTERS, and it is not fixable here. This ranks HOW MUCH motion a
    window carries, and it has no idea how many separate things are moving — so it actively
    works against the tool's own "one dominant motion signal" law, because a second figure
    walking into frame is more energy, not less. Verified on CLIP C: the top-ranked window is
    exactly the stretch the original hand trim cut away because extra figures enter there.

    Counting subjects was tried and cut. Connected components on the frame difference splits a
    single body into limbs and edges: CLIP C's clean one-walker window measured 3-6 "movers"
    against 3-5 for the crowded one, i.e. no signal at all. Since the camera round (2026-11-11)
    `scout_clip` counts people with Apple Vision (`depthc.people`, an OS framework on macOS, no
    weights) and each window's note says when a second figure is in it. The count WARNS; it does
    not enter this score, and off macOS there is no count at all.

    So: LOOK AT sheet.png BEFORE FIRING. That is not a nicety, it is the step that catches the
    one failure this scorer is structurally blind to, and the people count is only a person
    count: a dog, a car or a flag is a second motion it cannot see.
    """
    if not profile.deltas.size or target_seconds <= 0:
        return []

    fps = profile.fps
    length = int(round(target_seconds * fps))
    if length >= profile.deltas.size:
        length = profile.deltas.size
    stride = max(1, int(round(stride_seconds * fps)))
    head_length = max(1, int(round(DEAD_STRETCH_SECONDS * fps)))
    # One cut mask for the whole clip, by the same rule `spikes` uses, so a window's verdict
    # and the profile's finding can never disagree.
    cut_frames = np.zeros(profile.deltas.size, dtype=bool)
    for second in profile.spikes:
        index = int(round(second * fps))
        if 0 <= index < cut_frames.size:
            cut_frames[index] = True

    windows: list[Window] = []
    for start in range(0, max(1, profile.deltas.size - length + 1), stride):
        slice_ = profile.deltas[start:start + length]
        if slice_.size < length:
            break
        energy = float(slice_.mean())
        head = float(slice_[:head_length].mean())
        cut = bool(cut_frames[start:start + length].any())

        dead_run, longest = 0, 0
        for delta in slice_:
            dead_run = dead_run + 1 if delta <= STATIC_FLOOR else 0
            longest = max(longest, dead_run)
        dead = longest / fps >= DEAD_STRETCH_SECONDS

        first, second = np.array_split(slice_, 2)
        halves = (float(first.mean()), float(second.mean()))
        balance = min(halves) / max(halves) if max(halves) > 0 else 1.0

        head_factor = 1.0 if energy <= 0 else min(1.0, max(0.35, head / (0.5 * energy)))
        score = (
            energy
            * head_factor
            * (0.5 + 0.5 * balance)
            * (0.25 if cut else 1.0)
            * (0.6 if dead else 1.0)
        )

        sharp = 1.0
        if sharpness is not None and sharpness.size:
            piece = sharpness[start:start + length + 1]
            top = float(sharpness.max()) or 1.0
            sharp = round(float(piece.mean()) / top, 2) if piece.size else 1.0

        windows.append(Window(
            start=round(start / fps, 2),
            end=round((start + length) / fps, 2),
            energy=round(energy, 3),
            head=round(head, 3),
            score=round(score, 3),
            dead=dead,
            cut=cut,
            sharpness=sharp,
            balance=round(balance, 2),
        ))

    windows.sort(key=lambda w: w.score, reverse=True)
    return windows


def spread(windows: list[Window], count: int = 3, overlap: float = 0.6) -> list[Window]:
    """Take the best `count` windows that are not mostly the same footage.

    Without this the top ten candidates are the winner shifted by a quarter second each time,
    which tells a human nothing they can choose between.
    """
    chosen: list[Window] = []
    for window in windows:
        span = window.end - window.start
        if any(
            max(0.0, min(window.end, other.end) - max(window.start, other.start)) / span > overlap
            for other in chosen
        ):
            continue
        chosen.append(window)
        if len(chosen) >= count:
            break
    return chosen


def sharpness_series(frames: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian per frame — high is crisp, low is motion-blurred."""
    import cv2

    grid = analysis_gray(frames)
    return np.array([cv2.Laplacian(f, cv2.CV_64F).var() for f in grid], dtype=np.float32)


# ---------------------------------------------------------------------- the scout pass

@dataclass
class ScoutResult:
    """A cheap whole-clip depth pass: the profile, the ranked windows, and what it cost."""

    profile: MotionProfile
    windows: list[Window]
    per_second: list[float]
    scout_fps: float
    elapsed: float
    target_seconds: float
    # People in frame, one count per SCOUT_PEOPLE_FPS sample (Apple Vision); None = not counted.
    people: np.ndarray | None = field(default=None, repr=False)
    people_fps: float = 0.0

    @property
    def best(self) -> Window | None:
        return self.windows[0] if self.windows else None

    @property
    def second_figure(self) -> float | None:
        """Share of the people samples with two or more people in frame; None = not counted."""
        if self.people is None or not self.people.size:
            return None
        return float((self.people >= 2).mean())

    def figures_between(self, start: float, end: float) -> float | None:
        """`second_figure` over one stretch of the clip, in seconds."""
        if self.people is None or not self.people.size or self.people_fps <= 0:
            return None
        lo = max(0, int(np.floor(start * self.people_fps)))
        hi = min(self.people.size, max(lo + 1, int(np.ceil(end * self.people_fps))))
        piece = self.people[lo:hi]
        return float((piece >= 2).mean()) if piece.size else None


SCOUT_FPS = 12.0
# The scout counts people twice a second: a figure walking in stays a second or more, and Vision
# costs tens of milliseconds a frame, which at the scout's 12 fps would outweigh the depth pass.
SCOUT_PEOPLE_FPS = 2.0
# The scout warns when two or more people are in frame for at least this share of the samples,
# over the clip or over one window.
# ponytail: a tenth, by judgement: half a second of a five-second window is already a second motion
# the generation may copy. A share weighted by the second figure's size or motion is the upgrade.
SECOND_FIGURE_SHARE = 0.10
SCOUT_LONG_EDGE = 384
SCOUT_INPUT_SIZE = 196   # multiple of 14, as DINOv2 patching requires


def scout_clip(
    path: str | Path,
    target_seconds: float,
    *,
    scout_fps: float = SCOUT_FPS,
    long_edge: int = SCOUT_LONG_EDGE,
    input_size: int = SCOUT_INPUT_SIZE,
    crop: bool = True,
    on_window=None,
) -> ScoutResult:
    """Run a low-resolution depth pass over a WHOLE clip to rank its candidate windows.

    Cheap on purpose: 12 fps at a 384 px long edge through VDA-Small at input_size 196, which
    is seconds rather than the 20-35 s a real 720p conversion takes. The point is only to
    rank windows *within one clip*.

    THE ENERGY FIGURES ARE ESTIMATES, and there are two reasons they will not match the kit's:
    depth is normalised over the whole clip here versus over the chosen window there, which
    compresses within-window contrast; and per-frame deltas scale with the frame interval, so
    the 12 fps deltas are halved to express them at the 24 fps a kit ships at. The kit's own
    gate is the authority. This picks where to point it.
    """
    import time

    from depthc import engine
    from depthc.video_io import detect_crop, probe, read_frames, target_size

    info = probe(path)
    rect = detect_crop(info) if crop else None
    size = target_size(info, "source", crop=rect)
    scale = long_edge / max(size)
    if scale < 1:
        size = (max(2, int(size[0] * scale) & ~1), max(2, int(size[1] * scale) & ~1))

    started = time.perf_counter()
    frames = read_frames(info, fps=scout_fps, size=size, crop=rect)
    result = engine.infer(
        frames, scout_fps, input_size=input_size, precision="fp32", on_window=on_window
    )

    deltas, changed = frame_stats(result.frames)
    profile = MotionProfile(
        fps=scout_fps,
        # Rescaled so the scout's energies are directly comparable with a kit's, which are
        # always measured at 24. `delta_fps` records that, so magnitude rules downstream
        # judge against the right scale instead of re-deriving it from the sample rate.
        deltas=deltas * (scout_fps / 24.0),
        delta_fps=24.0,
        changed=changed,
        bucket_seconds=1.0,
        camera=camera_share(result.frames, scout_fps),
    )
    windows = score_windows(
        profile, target_seconds, sharpness=sharpness_series(frames)
    )
    from depthc import people as peoplelib

    stride = max(1, int(round(scout_fps / SCOUT_PEOPLE_FPS)))
    found = peoplelib.find(frames[::stride])
    result = ScoutResult(
        profile=profile,
        windows=spread(windows),
        per_second=profile.buckets,
        scout_fps=scout_fps,
        elapsed=time.perf_counter() - started,
        target_seconds=target_seconds,
        people=found.counts if found is not None else None,
        people_fps=scout_fps / stride,
    )
    for window in result.windows:
        window.figures = result.figures_between(window.start, window.end)
    return result


# ---------------------------------------------------------------------- ref variants

def contrast_stretch(frames: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Rescale the depth range between two percentiles, clipping the tails.

    Rationale: normalisation spends the 0-255 range on the single nearest and single furthest
    pixels in the clip, so one bright foreground object or one hole to infinity can leave the
    subject occupying a narrow band of greys. Clipping the outer 2% at each end gives the
    moving subject more of the range.

    EXPERIMENTAL, and honestly so: no published depth-conditioning pipeline does this (VACE's
    annotator uses plain per-frame min-max, no percentile clipping), so whether Seedance
    follows a stretched ref better is an open question. That is what --variants is for.
    Percentiles are taken over the WHOLE clip, never per frame — a per-frame stretch would
    reintroduce the brightness pumping global normalisation exists to prevent.
    """
    data = frames.astype(np.float32)
    lo, hi = np.percentile(data, [low, high])
    if hi - lo < 1e-6:
        return frames.copy()
    return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def invert(frames: np.ndarray) -> np.ndarray:
    """Far bright instead of near bright."""
    return (255 - frames).astype(np.uint8)


MIDGREY_CEILING = 190  # brightest grey the midgrey variant will ship


def midgrey(frames: np.ndarray, ceiling: int = MIDGREY_CEILING) -> np.ndarray:
    """Scale the whole range down so no figure blows toward white.

    Made after the one kit on record refused at input (`InputVideoSensitiveContentDetected`):
    two depth silhouettes, prone and overlapping, blown to near-white filling the frame.
    Upright, separated figures from the same pipeline passed. An experiment that has never
    been fired, and not a way round a provider's rules: a refused reference stays refused.

    The map is linear, so relative depth ordering and timing are untouched; motion energy
    scales down by the same ceiling/255 factor and the per-variant energy in the manifest
    records that honestly. A rejection costs nothing, so probing with this is cheap.
    """
    return (frames.astype(np.float32) * (ceiling / 255.0)).round().astype(np.uint8)


def depth_edges(frames: np.ndarray) -> np.ndarray:
    """Canny edges of the DEPTH frames — outlines only, dilated to survive encoding.

    Computed from depth, never from the source: an edge map of the original pixels would
    carry face and texture contours back into the upload, exactly what the depth channel
    exists to strip. Depth edges keep silhouettes and prop outlines (the sword survives)
    while discarding the surface shading depth still carries.

    EXPERIMENTAL, and honestly so: no published pipeline feeds Seedance edge refs, Canny
    flickers frame to frame by nature, and a line-drawing ref may read as a STYLE cue as
    much as a motion one. That is what --variants exists to find out.
    """
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    out = np.empty_like(frames)
    for index, frame in enumerate(frames):
        out[index] = cv2.dilate(cv2.Canny(frame, 40, 120), kernel)
    return out


VARIANTS = {
    "contrast": (
        contrast_stretch,
        "percentile-stretched (2-98%) — more of the grey range spent on the subject",
    ),
    "invert": (
        invert,
        "polarity flipped — far bright, near dark",
    ),
    "midgrey": (
        midgrey,
        f"luma capped at {MIDGREY_CEILING} — figures stay grey instead of blowing to "
        f"near-white; experimental, never fired",
    ),
    "edges": (
        depth_edges,
        "Canny edges of the depth video — outlines only; may read as style as much as "
        "motion (experiment)",
    ),
}


# ---------------------------------------------------------------------- contact sheet

def contact_sheet(
    frames: np.ndarray,
    fps: float,
    out_path: str | Path,
    columns: int = 5,
    rows: int = 2,
    tile_width: int = 320,
) -> Path:
    """Tile evenly-spaced depth frames with timestamps into one PNG.

    The letterbox bug was caught by looking at a picture, not by a test, and the CLIP A
    diagnosis needed to see *when* the motion happened. A player cannot show either at a
    glance; a contact sheet can, and it costs nothing to write.
    """
    import cv2

    out_path = Path(out_path)
    count = min(columns * rows, frames.shape[0])
    if count == 0:
        raise ValueError("No frames to tile")
    indices = np.linspace(0, frames.shape[0] - 1, count).round().astype(int)

    height = max(2, round(tile_width * frames.shape[1] / frames.shape[2]))
    tiles = []
    for index in indices:
        tile = cv2.resize(frames[index], (tile_width, height), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR) if tile.ndim == 2 else tile
        label = f"{index / fps:.2f}s"
        cv2.putText(tile, label, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))

    grid = np.concatenate(
        [np.concatenate(tiles[r:r + columns], axis=1) for r in range(0, len(tiles), columns)],
        axis=0,
    )
    cv2.imwrite(str(out_path), grid)
    return out_path
