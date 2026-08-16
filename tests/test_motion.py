"""Motion measurement tests.

The synthetic clips here are arrays, not encoded video: the measurement runs on the frames a
conversion already holds in memory, so there is nothing to gain from a codec round trip and
plenty to lose in determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthc import motion as m

GRID = (m.GRID_HEIGHT, m.GRID_WIDTH)


def textured(height: int = 180, width: int = 320) -> np.ndarray:
    """One well-conditioned frame: seeded, smoothed noise.

    Deliberately NOT a periodic ramp. A diagonal `(row*a + col*b) % n` pattern was tried first
    and made `camera_share` measure 0.0 on a synthetic pan that the implementation handles
    correctly (0.98 on this texture) — a repeating gradient is ambiguous to optical flow in
    both the aperture and the wraparound sense, so the fixture was testing the fixture.
    Blurred noise has localised, non-repeating features, which is what flow needs.
    """
    import cv2

    noise = np.random.default_rng(7).random((height, width)) * 255
    return cv2.GaussianBlur(noise.astype(np.uint8), (0, 0), 3)


def still_clip(frames: int = 48, height: int = 180, width: int = 320) -> np.ndarray:
    return np.repeat(textured(height, width)[None], frames, axis=0)


def moving_clip(
    frames: int = 48, speed: int = 4, freeze_after: int | None = None, height: int = 180,
    width: int = 320,
) -> np.ndarray:
    """A bright box crossing a static background, optionally freezing partway through."""
    clip = still_clip(frames, height, width).copy()
    for index in range(frames):
        step = index if freeze_after is None else min(index, freeze_after)
        x = min(width - 40, step * speed)
        clip[index, height // 3: height // 3 + 40, x: x + 40] = 255
    return clip


def panning_clip(frames: int = 48, speed: int = 4) -> np.ndarray:
    """Every pixel moving together — a camera translation and nothing else."""
    base = textured()
    return np.stack([np.roll(base, index * speed, axis=1) for index in range(frames)])


# ----------------------------------------------------------------------------- the grid

def test_analysis_grid_is_fixed_for_grey_and_rgb():
    grey = m.analysis_gray(still_clip(3))
    rgb = m.analysis_gray(np.repeat(still_clip(3)[..., None], 3, axis=3))
    assert grey.shape == (3, *GRID)
    assert rgb.shape == (3, *GRID)


def test_analysis_grid_rejects_other_shapes():
    with pytest.raises(ValueError):
        m.analysis_gray(np.zeros((4, 4), dtype=np.uint8))


def test_a_still_clip_measures_essentially_zero():
    """The floor has to be signal, not noise: identical frames must not invent motion."""
    assert m.frame_deltas(still_clip()).mean() == pytest.approx(0.0, abs=1e-6)


def test_motion_reads_higher_than_stillness():
    moving = m.measure(moving_clip(), 24.0, with_flow=False).energy
    assert moving > m.STATIC_FLOOR
    assert m.measure(still_clip(), 24.0, with_flow=False).energy < 0.01


def test_single_frame_clip_has_no_deltas():
    assert m.frame_deltas(still_clip(1)).size == 0


def test_measure_rejects_a_nonsense_fps():
    with pytest.raises(ValueError):
        m.measure(still_clip(4), 0.0)


# ------------------------------------------------------------------------ profile shapes

def test_head_and_tail_read_the_right_ends():
    profile = m.measure(moving_clip(freeze_after=24), 24.0, with_flow=False)
    assert profile.head > profile.tail


def test_buckets_cover_the_clip():
    profile = m.measure(moving_clip(48), 24.0, with_flow=False)
    assert len(profile.buckets) == pytest.approx(4, abs=1)  # 47 deltas / 12 per bucket


def test_dead_tail_is_found_and_placed():
    dead = m.measure(moving_clip(96, freeze_after=24), 24.0, with_flow=False).dead_stretch
    assert dead is not None
    assert dead.where == "tail"
    assert dead.seconds > m.DEAD_STRETCH_SECONDS


def test_dead_head_is_found_and_placed():
    """The late-start case: nothing moves, then it does."""
    clip = np.concatenate([still_clip(36), moving_clip(60, speed=4)])
    dead = m.measure(clip, 24.0, with_flow=False).dead_stretch
    assert dead is not None
    assert dead.where == "head"


def test_a_clip_that_never_stops_has_no_dead_stretch():
    """speed=2 so the box never reaches the right edge and parks there."""
    assert m.measure(moving_clip(96, speed=2), 24.0, with_flow=False).dead_stretch is None


def test_a_brief_pause_is_not_a_dead_stretch():
    """Under 0.75s of stillness is a beat, not a hole."""
    clip = np.concatenate([moving_clip(24), still_clip(12), moving_clip(24, speed=4)])
    assert m.measure(clip, 24.0, with_flow=False).dead_stretch is None


def test_a_hard_cut_registers_as_a_spike():
    """Two unrelated shots joined: the join is one enormous frame delta."""
    other = np.full((24, 180, 320), 5, dtype=np.uint8)
    clip = np.concatenate([moving_clip(24), other])
    spikes = m.measure(clip, 24.0, with_flow=False).spikes
    assert spikes, "a cut must be visible"
    assert spikes[0] == pytest.approx(23 / 24, abs=0.1)


def test_a_smooth_clip_has_no_spikes():
    assert m.measure(moving_clip(48, speed=2), 24.0, with_flow=False).spikes == []


def test_violent_but_local_motion_is_not_a_cut():
    """The rule that replaced a magnitude threshold: extent, not intensity.

    A magnitude rule flagged a cut in all three real clips that contain none — a sword impact
    spikes several times above its own median. This subject jumps 60 px a frame and must still
    read as choreography, because it only ever changes the part of the frame it occupies.
    """
    assert m.measure(moving_clip(24, speed=60), 24.0, with_flow=False).spikes == []


def test_a_whole_frame_change_is_a_cut():
    """Inverting the texture would NOT do: it is centred near 128, so 255-x barely moves."""
    other = np.full((12, 180, 320), 250, dtype=np.uint8)
    clip = np.concatenate([still_clip(12), other])
    assert m.measure(clip, 24.0, with_flow=False).spikes == [pytest.approx(11 / 24, abs=0.05)]


# ------------------------------------------------------------------- camera vs subject

def test_a_pan_reads_as_camera_motion():
    assert m.camera_share(panning_clip()) >= m.CAMERA_DOMINANT


def test_a_moving_subject_does_not_read_as_camera_motion():
    assert m.camera_share(moving_clip(24, speed=6)) < m.CAMERA_DOMINANT


# ----------------------------------------------------------------------------- retention

def test_retention_is_depth_over_source():
    depth = m.MotionProfile(fps=24.0, deltas=np.full(10, 1.0, dtype=np.float32))
    source = m.MotionProfile(fps=24.0, deltas=np.full(10, 4.0, dtype=np.float32))
    assert m.retention(depth, source) == pytest.approx(0.25)


def test_retention_of_a_still_source_is_undefined_not_a_crash():
    depth = m.MotionProfile(fps=24.0, deltas=np.full(10, 1.0, dtype=np.float32))
    source = m.MotionProfile(fps=24.0, deltas=np.zeros(10, dtype=np.float32))
    assert m.retention(depth, source) is None


def test_manifest_block_carries_the_source_side_when_given():
    depth = m.MotionProfile(fps=24.0, deltas=np.full(10, 1.0, dtype=np.float32))
    source = m.MotionProfile(fps=24.0, deltas=np.full(10, 2.0, dtype=np.float32))
    block = m.manifest_block(depth, source)
    assert block["retention"] == pytest.approx(0.5)
    assert block["sourceEnergy"] == pytest.approx(2.0)
    assert "retention" not in m.manifest_block(depth)


# ----------------------------------------------------------------------------- sparkline

def test_sparkline_marks_dead_buckets_visibly():
    line = m.sparkline([0.0, 0.1, 5.0, 5.0])
    assert line.startswith(m.SPARK_DEAD * 2)
    assert line.endswith(m.SPARK_BLOCKS[-1] * 2)


def test_sparkline_length_matches_the_series():
    assert len(m.sparkline([1.0] * 17)) == 17


def test_sparkline_of_nothing_is_empty():
    assert m.sparkline([]) == ""


def test_sparkline_of_an_all_dead_series_is_all_dead():
    assert set(m.sparkline([0.0, 0.0, 0.0])) == {m.SPARK_DEAD}


# ------------------------------------------------------------------------ window scoring

def _profile(values: list[float], fps: float = 24.0) -> m.MotionProfile:
    return m.MotionProfile(fps=fps, deltas=np.array(values, dtype=np.float32))


def test_windows_prefer_the_livelier_half():
    profile = _profile([0.1] * 48 + [4.0] * 48)
    best = m.score_windows(profile, 2.0)[0]
    assert best.start >= 1.5


def test_a_dead_headed_window_scores_below_its_own_energy():
    """The CLIP A shape: strong window, dead first beat. It must lose to a live-headed one."""
    dead_head = m.score_windows(_profile([0.05] * 24 + [6.0] * 24), 2.0)
    live = m.score_windows(_profile([3.0] * 48), 2.0)
    assert dead_head[0].score < live[0].score
    assert dead_head[0].energy > live[0].energy  # ...despite carrying more motion


def test_a_window_containing_a_cut_is_heavily_penalised():
    with_cut = _profile([2.0] * 20 + [40.0] + [2.0] * 27)
    windows = m.score_windows(with_cut, 1.0)
    cut_windows = [w for w in windows if w.cut]
    clean = [w for w in windows if not w.cut]
    assert cut_windows and clean
    assert max(w.score for w in cut_windows) < max(w.score for w in clean)


def test_windows_report_a_dead_stretch_inside_them():
    windows = m.score_windows(_profile([3.0] * 12 + [0.05] * 24 + [3.0] * 12), 2.0)
    assert any(w.dead for w in windows)


def test_spread_drops_near_duplicate_windows():
    profile = _profile([2.0] * 240)
    all_windows = m.score_windows(profile, 2.0)
    picked = m.spread(all_windows, count=3)
    assert len(picked) == 3
    starts = sorted(w.start for w in picked)
    assert all(b - a >= 0.8 for a, b in zip(starts, starts[1:]))


def test_no_windows_from_an_empty_profile():
    assert m.score_windows(_profile([]), 5.0) == []


def test_window_note_mentions_what_is_wrong():
    window = m.Window(0, 5, 1.0, 0.1, 0.4, dead=True, cut=True, sharpness=0.4)
    note = window.note()
    assert "cut" in note and "dead" in note and "soft" in note


def test_sharpness_series_ranks_a_blurred_frame_lower():
    import cv2

    sharp = textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    series = m.sharpness_series(np.stack([sharp, blurred]))
    assert series[0] > series[1]


# ----------------------------------------------------------------------------- variants

def test_contrast_stretch_widens_the_used_range():
    narrow = (still_clip(4).astype(np.float32) * 0.3 + 90).astype(np.uint8)
    stretched = m.contrast_stretch(narrow)
    assert np.ptp(stretched) > np.ptp(narrow)
    assert stretched.dtype == np.uint8
    assert stretched.shape == narrow.shape


def test_contrast_stretch_of_a_flat_clip_is_a_no_op():
    flat = np.full((4, 90, 160), 128, dtype=np.uint8)
    assert np.array_equal(m.contrast_stretch(flat), flat)


def test_invert_flips_polarity():
    clip = still_clip(2)
    assert np.array_equal(m.invert(clip), 255 - clip)


def test_every_variant_is_a_named_transform_with_a_description():
    for name, (transform, description) in m.VARIANTS.items():
        out = transform(still_clip(2))
        assert out.shape == (2, 180, 320) and out.dtype == np.uint8
        assert description and name


def test_midgrey_caps_the_brightest_level():
    """Nothing may approach white: near-white body-shapes are the moderation trigger."""
    hot = np.full((3, 90, 160), 255, dtype=np.uint8)
    capped = m.midgrey(hot)
    assert capped.max() == m.MODERATION_CEILING
    assert capped.dtype == np.uint8


def test_midgrey_is_linear_so_depth_ordering_survives():
    ramp = np.tile(np.arange(0, 256, 8, dtype=np.uint8), (2, 4, 1))
    row = m.midgrey(ramp)[0, 0].astype(int)
    assert (np.diff(row) >= 0).all()
    assert row[0] == 0  # black stays black; only the top is pulled down


def test_depth_edges_is_binary_outlines():
    """Canny + dilate: only pure black and pure white may appear."""
    edges = m.depth_edges(moving_clip(3))
    assert set(np.unique(edges)) <= {0, 255}
    assert edges.shape == moving_clip(3).shape
    assert edges.any(), "a moving box must produce some outline"


# --------------------------------------------------------------- fps-honesty regressions

def test_analysis_grid_preserves_aspect():
    """2.39:1 frames keep their shape (160x67) instead of being squashed into 90 rows."""
    wide = np.zeros((2, 536, 1280), dtype=np.uint8)
    assert m.analysis_gray(wide).shape == (2, 67, 160)
    assert m.analysis_size(1280, 720) == (160, 90)  # 16:9 lands on the v1 grid exactly


def test_analysis_gray_of_an_empty_clip_is_empty_not_a_crash():
    assert m.analysis_gray(np.zeros((0, 90, 160), dtype=np.uint8)).shape[0] == 0


def test_cut_detection_falls_back_off_24fps():
    """CUT_FRACTION is calibrated at 24 fps; a 12 fps profile must not use it.

    At 12 fps a legitimate fast action sweeps ~2x the pixels per frame gap, so the extent
    rule flags choreography as a cut and the scorer buries the best window. The magnitude
    fallback self-normalises and stays honest at any rate.
    """
    deltas = np.full(48, 2.0, dtype=np.float32)
    changed = np.full(48, 0.48, dtype=np.float32)  # legit action at 12fps; a "cut" by 0.35
    scout = m.MotionProfile(fps=12.0, deltas=deltas, changed=changed)
    assert scout.spikes == []
    kit = m.MotionProfile(fps=24.0, deltas=deltas, changed=changed)
    assert kit.spikes, "at 24 fps the same extent IS a cut"


def test_unmeasured_camera_is_none_not_zero():
    """None = flow skipped; 0.0 = measured pure subject motion. Different facts."""
    assert m.measure(moving_clip(6), 24.0, with_flow=False).camera is None
    assert m.measure(moving_clip(24, speed=6), 24.0, with_flow=True).camera is not None


# -------------------------------------------------------------------------- contact sheet

def test_contact_sheet_writes_a_tiled_png(tmp_path):
    import cv2

    out = m.contact_sheet(moving_clip(48), 24.0, tmp_path / "sheet.png", columns=5, rows=2)
    image = cv2.imread(str(out))
    assert image is not None
    assert image.shape[1] == 320 * 5


def test_contact_sheet_handles_fewer_frames_than_tiles(tmp_path):
    """A 3-frame clip must still produce a full grid rather than crash on the remainder."""
    import cv2

    out = m.contact_sheet(moving_clip(3), 24.0, tmp_path / "few.png", columns=5, rows=2)
    image = cv2.imread(str(out))
    assert image.shape[1] == 320 * 5


def test_contact_sheet_refuses_an_empty_clip(tmp_path):
    with pytest.raises(ValueError):
        m.contact_sheet(np.zeros((0, 90, 160), dtype=np.uint8), 24.0, tmp_path / "x.png")


def test_spike_floor_scales_with_fps():
    """2026-07-30 council audit pin: the absolute SPIKE_FLOOR was calibrated at 24 fps.
    At 60 fps per-gap deltas shrink by 24/60, so an unscaled 8.0 floor sat ~2.5× too
    high and a real hard cut shipped inside the ref undetected."""
    # A quiet 60 fps clip with one cut-sized jump: 4.0 is a clear cut at 60 fps
    # (equivalent to 10.0 at 24 fps) yet sits under the unscaled floor of 8.0.
    deltas = np.full(120, 0.4, dtype=np.float32)
    deltas[60] = 4.0
    profile = m.MotionProfile(fps=60.0, deltas=deltas)
    assert profile.spikes == [pytest.approx(1.0)]

    # The same shape expressed at 24 fps agrees — the rule is rate-invariant now.
    deltas24 = np.full(48, 1.0, dtype=np.float32)
    deltas24[24] = 10.0
    assert m.MotionProfile(fps=24.0, deltas=deltas24).spikes == [pytest.approx(1.0)]


# ------------------------------------------------------------- structure: regions + path

def test_region_means_are_on_the_energy_scale_and_cover_the_grid():
    """The pixel-weighted mean of the cells IS the full-frame energy — same units, no gap."""
    clip = moving_clip(24, speed=6)
    grid = m.analysis_gray(clip).astype("float32")
    difference = np.abs(np.diff(grid, axis=0))
    regions = m.region_means(difference)
    assert regions.shape == (23, m.REGION_ROWS, m.REGION_COLS)
    h, w = grid.shape[1:]
    rows, cols = m._region_edges(h, m.REGION_ROWS), m._region_edges(w, m.REGION_COLS)
    weights = np.array([[(rows[r + 1] - rows[r]) * (cols[c + 1] - cols[c]) for c in range(3)]
                        for r in range(3)], dtype="float64") / (h * w)
    assert (regions * weights).sum(axis=(1, 2)) == pytest.approx(difference.mean(axis=(1, 2)), abs=1e-3)


def test_a_left_to_right_crossing_lights_the_columns_in_order():
    """The whole point of the layer: placement over time, which the mean throws away."""
    profile = m.measure(moving_clip(72, speed=5), 24.0, with_flow=False)
    buckets = profile.region_buckets(0.5)
    assert buckets is not None and buckets.shape[1:] == (3, 3)
    columns = buckets.sum(axis=1)  # [buckets, cols]
    left_peak, right_peak = int(columns[:, 0].argmax()), int(columns[:, 2].argmax())
    assert left_peak < right_peak


def test_region_energy_is_none_on_a_bare_profile():
    assert m.MotionProfile(fps=24.0, deltas=np.ones(24, dtype="float32")).region_energy is None
    assert m.MotionProfile(fps=24.0, deltas=np.ones(24, dtype="float32")).camera_path is None


def test_a_pan_names_its_direction():
    """np.roll(+speed) moves pixels RIGHT, which is the camera panning LEFT."""
    profile = m.measure(panning_clip(), 24.0, with_flow=True)
    path = profile.camera_path
    assert path is not None and path["label"] == "pans left"
    assert path["dx"] > m.PATH_STILL and abs(path["dy"]) < m.PATH_STILL
    reverse = m.measure(panning_clip(speed=-4), 24.0, with_flow=True).camera_path
    assert reverse is not None and reverse["label"] == "pans right"


def test_a_zoom_reads_as_a_push_and_not_as_a_pan():
    """A pure scale-up expands the picture: radial > 0, no translation to name."""
    import cv2

    base = textured()
    h, w = base.shape
    frames = []
    for i in range(24):
        scale = 1.0 + 0.012 * i
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, scale)
        frames.append(cv2.warpAffine(base, matrix, (w, h), borderMode=cv2.BORDER_REFLECT))
    path = m.measure(np.stack(frames), 24.0, with_flow=True).camera_path
    assert path is not None
    assert path["radial"] > m.PATH_RADIAL_STILL
    assert "pushes in" in path["label"] and "pans" not in path["label"]


def test_camera_share_is_unchanged_by_the_flow_refactor():
    """camera_share and camera_share_from(flow_series) must agree to the digit."""
    clip = panning_clip()
    assert m.camera_share(clip) == m.camera_share_from(m.flow_series(clip))
    still = still_clip(8)
    assert m.camera_share(still) == 0.0
    assert m.measure(still, 24.0).camera_path == {"dx": 0.0, "dy": 0.0, "radial": 0.0, "label": "static"}


def test_path_label_vocabulary():
    assert m.path_label(0.0, 0.0, 0.0) == "static"
    assert m.path_label(0.5, 0.1, 0.0) == "pans left"
    assert m.path_label(-0.5, 0.1, 0.0) == "pans right"
    assert m.path_label(0.1, 0.5, 0.0) == "tilts up"
    assert m.path_label(0.1, -0.5, 0.0) == "tilts down"
    assert m.path_label(0.0, 0.0, -0.2) == "pulls out"
    assert m.path_label(0.5, 0.0, 0.2) == "pans left, pushes in"
