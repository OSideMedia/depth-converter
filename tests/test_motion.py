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
    """Nothing may approach white: that is the whole of what this variant does."""
    hot = np.full((3, 90, 160), 255, dtype=np.uint8)
    capped = m.midgrey(hot)
    assert capped.max() == m.MIDGREY_CEILING
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


def test_path_tempo_bands():
    assert m.path_tempo(m.path_speed(0.2, 0.1, 0.0)) == "slowly"    # >15 s to cross
    assert m.path_tempo(m.path_speed(0.8, 0.0, 0.1)) == ""          # unremarkable
    assert m.path_tempo(m.path_speed(2.0, 0.0, 0.0)) == "quickly"   # <4 s to cross


def test_path_label_vocabulary():
    assert m.path_label(0.0, 0.0, 0.0) == "static"
    assert m.path_label(0.5, 0.1, 0.0) == "pans left"
    assert m.path_label(-0.5, 0.1, 0.0) == "pans right"
    assert m.path_label(0.1, 0.5, 0.0) == "tilts up"
    assert m.path_label(0.1, -0.5, 0.0) == "tilts down"
    assert m.path_label(0.0, 0.0, -0.2) == "pulls out"
    assert m.path_label(0.5, 0.0, 0.2) == "pans left, pushes in"


def test_a_subject_walking_toward_centre_is_not_a_camera_pull():
    """One box crossing frame is radial by accident in one quadrant; a push/pull is every
    quadrant at once. The radial term must stay 0 on the crossing (first describe smoke)."""
    path = m.measure(moving_clip(72, speed=5), 24.0, with_flow=True).camera_path
    assert path is not None
    assert path["radial"] == pytest.approx(0.0, abs=1e-6)
    assert path["label"] == "static"


# ------------------------------------------------------- the scout's cut detector, at 12 fps
# `MotionProfile.fps` was doing two jobs: the TIME axis (i / fps gives a timestamp) and the
# SCALE the deltas are expressed in. They are the same thing for measure(), whose deltas are
# raw at fps — and different for scout_clip(), which rescales its deltas to 24fps-equivalent
# and still reports fps=12. `spikes` read fps as the scale and rescaled the floor a second
# time, so at the scout's default the threshold landed 2x too high.
#
# This matters because the changed-pixel CUT_FRACTION rule is deliberately disabled off
# 24 fps, which makes the magnitude rule the scout's ONLY cut detector — and a missed cut
# puts a spliced window at the top of `depthc inbox` with a ready-to-paste --trim, against
# the module's own law that a hard cut is the worst possible competing signal.


def _scout_shaped_profile(spike_24fps_equivalent: float):
    """A profile built the way scout_clip builds one: deltas already on the 24 fps scale."""
    raw = np.full(60, 1.0, dtype=np.float64)      # quiet clip: median 1.0, so median*6 = 6
    raw[30] = spike_24fps_equivalent
    return m.MotionProfile(fps=12.0, deltas=raw, delta_fps=24.0, bucket_seconds=1.0)


@pytest.mark.parametrize("magnitude", [9.0, 12.0, 15.0])
def test_a_cut_between_the_floor_and_twice_it_is_seen_at_the_scout_default(magnitude):
    """The band the double-compensation hid: above SPIKE_FLOOR, below 2x it.

    On a clip quiet enough that median*SPIKE_MULTIPLE stays under the doubled floor, a cut
    of this size was invisible at --scout-fps 12 and visible at 24 — two different answers
    from the same footage, decided by a flag nobody passes.
    """
    assert m.SPIKE_FLOOR < magnitude < 2 * m.SPIKE_FLOOR   # the test has a real subject
    profile = _scout_shaped_profile(magnitude)
    assert profile.spikes, f"a {magnitude} spike on the 24fps scale must clear SPIKE_FLOOR"


def test_the_delta_scale_is_independent_of_the_time_axis():
    """Same deltas, same scale, two different frame rates → the same verdict.

    The floor follows the scale the deltas are IN, never the rate they were sampled at.
    """
    raw = np.full(60, 1.0); raw[30] = 10.0
    at12 = m.MotionProfile(fps=12.0, deltas=raw.copy(), delta_fps=24.0)
    at24 = m.MotionProfile(fps=24.0, deltas=raw.copy(), delta_fps=24.0)
    assert bool(at12.spikes) == bool(at24.spikes) is True
    # ...and the timestamps still come from the time axis, which is what fps is for.
    assert at12.spikes[0] == round(30 / 12.0, 2)
    assert at24.spikes[0] == round(30 / 24.0, 2)


def test_raw_deltas_still_get_their_floor_rescaled():
    """The 60 fps case the original scaling existed for must not regress.

    A profile from measure() carries RAW deltas at its own fps, so delta_fps defaults to fps
    and a 60 fps clip still gets the floor scaled down — per-gap deltas shrink by the same
    factor, and an unscaled 8.0 would let a real cut slide under.
    """
    raw = np.full(60, 0.4); raw[30] = 4.0     # 4.0 raw at 60fps = 10.0 at 24fps
    at60 = m.MotionProfile(fps=60.0, deltas=raw.copy())
    assert at60.delta_fps is None and at60.delta_scale == 60.0  # None means "same as fps"
    assert at60.spikes, "a 60fps cut must still be caught by the scaled floor"
    # The same series read as raw 24 fps is genuinely below the floor and must NOT fire.
    at24 = m.MotionProfile(fps=24.0, deltas=raw.copy())
    assert not at24.spikes


# ------------------------------------------------------------ the path, read where it shows
# Every clip below has a known camera. Most are built so the all-pixel reader this replaced gets
# them wrong, or so a reader without the guard the test names does. The rest pin what the old reader
# already got right.

def _dots(
    height: int = 180, width: int = 320, top: int = 0, bottom: int | None = None, seed: int = 3, density: float = 0.012,
) -> np.ndarray:
    """Hard-edged blocks on a flat grey field, only between `top` and `bottom`: clay, in effect."""
    bottom = height if bottom is None else bottom
    field = np.full((height, width), 120, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(int(density * width * (bottom - top))):
        y, x = int(rng.integers(top, max(top + 1, bottom - 7))), int(rng.integers(0, width - 7))
        field[y:y + 6, x:x + 6] = int(rng.integers(20, 240))
    return field


def _warped(base: np.ndarray, frames: int, dx: float = 0.0, dy: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """`base` moved each frame by (dx, dy) pixels and scaled by `zoom` about the frame's centre."""
    import cv2

    height, width = base.shape
    out = []
    for i in range(frames):
        s = zoom ** i
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        matrix = np.array([[s, 0.0, cx * (1 - s) + dx * i], [0.0, s, cy * (1 - s) + dy * i]], dtype=np.float64)
        out.append(cv2.warpAffine(base, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def _words(clip: np.ndarray) -> set[str]:
    from depthc.describe import path_buckets

    return {label for label, _, _ in path_buckets(m.flow_series(clip).path, 24.0)}


def test_a_push_with_all_its_texture_above_centre_is_a_push_and_not_a_tilt():
    # The clay finding: boxes on the horizon above a bare ground. The all-pixel reader saw a flat
    # field voting zero; a texture-masked median saw the boxes' outward flow as a tilt down.
    clip = _warped(_dots(top=10, bottom=80), 48, zoom=1.006)
    assert _words(clip) == {"pushes in"}


def test_a_pull_with_all_its_texture_above_centre_is_a_pull():
    clip = _warped(_dots(top=10, bottom=80), 48, zoom=1 / 1.006)
    assert _words(clip) == {"pulls out"}


def test_a_slow_pan_over_a_mostly_flat_field_is_seen():
    # 0.3 grid px/frame on a field of few blocks: most pixels are flat and vote zero.
    clip = _warped(_dots(top=70, bottom=110), 48, dx=0.6)
    assert _words(clip) == {"pans left"}


def test_a_pan_and_a_push_at_once_are_both_named():
    clip = _warped(_dots(), 48, dx=2.0, zoom=1.006)
    assert _words(clip) == {"pans left, pushes in"}


def test_a_zoom_on_full_texture_reads_at_about_its_true_rate():
    # A real take has texture everywhere. Stretch 0.8% per frame, read at the grid's mean radius.
    clip = _warped(textured(), 48, zoom=1.008)
    series = m.flow_series(clip)
    height, width = GRID
    ys, xs = np.mgrid[0:height, 0:width]
    mean_radius = float(np.hypot(xs - (width - 1) / 2, ys - (height - 1) / 2).mean())
    assert _words(clip) == {"pushes in"}
    assert float(np.median(series.path[:, 2])) == pytest.approx(0.008 * mean_radius, rel=0.2)


def test_a_truck_past_layers_at_different_depths_is_not_a_zoom():
    # One side of the frame slides faster than the other, which reads as a stretch across. Nothing
    # stretches up and down, and a zoom has to stretch both.
    base = _dots(seed=5)
    width = base.shape[1]
    clip = np.stack([
        np.concatenate([np.roll(base, 3 * i, axis=1)[:, :width // 2], np.roll(base, i, axis=1)[:, width // 2:]], axis=1)
        for i in range(48)
    ])
    assert _words(clip) == {"pans left"}


def test_a_stretch_most_votes_contradict_is_not_a_zoom(monkeypatch):
    # Layers interleaved along one band of texture. Each side's median slides at a different speed,
    # so a stretch reads out of them and leaves most votes explained, but four votes in ten go the
    # other way. A real zoom has nearly all of them agreeing.
    x = np.array([-70.0, -60, -50, -40, -30] * 2 + [30.0, 40, 50, 60, 70] * 2)
    y = np.full(x.size, -30.0)
    fx = np.array([3.0, 1, 3, 1, 3, 1, 3, 3, 1, 3] + [1.0, 3, 1, 3, 1, 3, 1, 1, 3, 1])
    fy = np.zeros(x.size)
    assert m._read_camera(fx, fy, x, y, 80.0, 45.0)[2] == 0.0
    monkeypatch.setattr(m, "PATH_ZOOM_AGREEMENT", 0.0)
    assert m._read_camera(fx, fy, x, y, 80.0, 45.0)[2] != 0.0


def test_a_figure_filling_one_side_of_a_still_frame_is_not_a_zoom():
    # A band of world, its left third carried along by a figure. The two sides of the frame
    # disagree and every vote agrees with the stretch they imply, but most of the picture has not
    # moved: the fit leaves more unexplained than the plain shift of nothing.
    base = _dots(top=20, bottom=70, seed=11)
    third = base.shape[1] // 3
    frames = []
    for i in range(48):
        frame = base.copy()
        frame[:, :third] = np.roll(base[:, :third], 2 * i, axis=1)
        frames.append(frame)
    assert _words(np.stack(frames)) == {"static"}


def test_a_crane_stretches_one_axis_and_is_not_a_zoom():
    # Flow down the frame that grows toward the bottom, and none across: the near ground sliding
    # faster than the far as the camera rises. Two checks stop this, both axes and the better fit,
    # so it goes red only with both off.
    import cv2

    base = _dots(seed=9)
    height, width = base.shape
    frames = []
    for i in range(48):
        s = 1.012 ** i
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, s, (height - 1) / 2.0 * (1 - s) + 1.5 * i]], dtype=np.float64)
        frames.append(cv2.warpAffine(base, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    assert all("pushes" not in w and "pulls" not in w for w in _words(np.stack(frames)))


def _figure_on(field: np.ndarray, frames: int, x0: int, x1: int, grow: float = 1.0) -> np.ndarray:
    out = []
    height, width = field.shape
    for i in range(frames):
        frame = field.copy()
        f = i / max(frames - 1, 1)
        size = int(30 * grow ** i)
        x, y = int(x0 + (x1 - x0) * f), height // 2
        frame[max(0, y - size):y + size, max(0, x - size // 2):x + size // 2] = 250
        out.append(frame)
    return np.stack(out)


def test_a_lone_figure_on_a_bare_field_names_no_camera():
    # The only texture in the picture is the figure, so its movement cannot be told from the
    # camera's. Without PATH_MIN_REGIONS this reads as the figure's own pan.
    clip = _figure_on(np.full((180, 320), 120, dtype=np.uint8), 48, 60, 240)
    series = m.flow_series(clip)
    assert not series.moving.any()
    assert _words(clip) == {"static"}


def test_a_lone_figure_walking_at_the_lens_on_a_bare_field_is_no_push():
    clip = _figure_on(np.full((180, 320), 120, dtype=np.uint8), 48, 160, 160, grow=1.02)
    assert _words(clip) == {"static"}


def test_a_hard_silhouette_does_not_silence_the_soft_world_behind_it():
    # A bright box's corner is far stronger than blurred texture. Measured against the frame's
    # strongest corner alone, the texture drops out and the box's crossing reads as a pan.
    assert _words(moving_clip(72, speed=5)) == {"static"}


def test_a_figure_crossing_in_front_of_a_world_names_no_camera():
    assert _words(_figure_on(_dots(top=20, bottom=160), 48, 40, 280)) == {"static"}


def test_a_flat_or_tiny_picture_is_read_as_nothing_and_does_not_fail():
    flat = np.full((12, 180, 320), 90, dtype=np.uint8)
    assert not m.flow_series(flat).moving.any()
    tiny = np.random.default_rng(1).integers(0, 255, (6, 2, 2), dtype=np.uint8)
    assert m.flow_series(tiny).path.shape == (5, 3)
