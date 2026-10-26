"""Boil and flicker must separate from each other, from motion, and from film grain.

Synthetic 720p clips of a smooth random texture panning diagonally at a sub-pixel rate, so the
flow has real work to do. Each test injects ONE fault and checks it lands on its own reading
and not on the others — the separations the module docstring claims.
"""

import cv2
import numpy as np
import pytest

from depthc import stability

W, H = 1280, 720
FRAMES = 7


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.normal(0, 1, (H + 200, W + 200)).astype(np.float32), (0, 0), 4.0)
    ramp = np.linspace(-30, 30, W + 200, dtype=np.float32)[None, :]
    return np.clip(field / field.std() * 40 + 128 + ramp, 0, 255)


BASE = _texture(1)


def _clip(*, step=1.3, reroll=False, grain=0.0, flicker=0.0, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(FRAMES):
        picture = BASE.copy()
        if reroll:  # a patch whose texture is drawn fresh every frame: boil
            picture[300:600, 400:900] = _texture(1000 + t)[300:600, 400:900]
        shift = np.array([[1, 0, -100 + step * t], [0, 1, -100 + 0.4 * step * t]], dtype=np.float32)
        frame = cv2.warpAffine(picture, shift, (W, H), flags=cv2.INTER_LINEAR)
        frame = frame * (1 + flicker * (-1) ** t)
        if grain:  # per-pixel noise at full resolution, new every frame: film grain
            frame = frame + rng.normal(0, grain, frame.shape)
        frames.append(np.repeat(np.clip(frame, 0, 255).astype(np.uint8)[..., None], 3, axis=-1))
    return np.stack(frames)


@pytest.fixture(scope="module")
def clean():
    return stability.measure(_clip(), 24.0)


@pytest.fixture(scope="module")
def boiling():
    return stability.measure(_clip(reroll=True), 24.0)


def test_a_frozen_frame_reads_zero():
    s = stability.measure(_clip(step=0.0), 24.0).summary()
    assert s["boilMedian"] < 0.05 and s["flickerMaxAbs"] < 0.05 and s["validMedian"] > 0.95


def test_smooth_motion_warps_away(clean):
    # the floor that flow error leaves on a clean pan — what every other reading is against
    s = clean.summary()
    assert s["scored"] == FRAMES - 1
    assert s["boilMedian"] < 1.0 and s["flickerStd"] < 0.05


def test_re_rolled_texture_reads_as_boil(clean, boiling):
    assert boiling.summary()["boilMedian"] > 3 * clean.summary()["boilMedian"]


@pytest.mark.parametrize("sigma, share", [(6.0, 0.15), (12.0, 0.35)])
def test_film_grain_is_not_boil(clean, boiling, sigma, share):
    # What grain adds must be a small fraction of what a re-rolling patch adds — on the BAND
    # reading. The plain full-band warp error is held to the same predicate and must FAIL it:
    # the band, not the metric's luck, is what keeps grain out.
    grainy = stability.measure(_clip(grain=sigma), 24.0)

    def added(reading):
        base = float(np.median(getattr(clean, reading)))
        return (float(np.median(getattr(grainy, reading))) - base,
                float(np.median(getattr(boiling, reading))) - base)

    grain_added, boil_added = added("boil")
    assert grain_added < share * boil_added
    if sigma == 12.0:
        naive_grain, naive_boil = added("warp")
        assert not naive_grain < share * naive_boil, "the full-band reading separates grain too — the band is unproven"


def test_flicker_is_its_own_reading_not_boil(clean):
    flickering = stability.measure(_clip(flicker=0.05), 24.0)
    s, c = flickering.summary(), clean.summary()
    assert s["flickerStd"] > 5.0
    assert np.sign(flickering.flicker[0]) != np.sign(flickering.flicker[1]), "a pump alternates; the sign must show it"
    assert s["boilMedian"] < 1.1 * c["boilMedian"], "exposure must be matched out before structure is read"


def test_a_cut_is_the_worst_boil_and_does_not_move_the_median(clean):
    other = np.clip(_texture(99)[100:100 + H, 100:100 + W], 0, 255).astype(np.uint8)
    cut = np.concatenate([_clip()[:4], np.repeat(other[None, ..., None], 3, axis=-1).repeat(3, axis=0)])
    s = stability.measure(cut, 24.0).summary()
    assert s["worstBoil"][0]["t"] == pytest.approx(3 / 24.0, abs=0.01)
    assert s["worstBoil"][0]["boil"] > 10 * clean.summary()["boilMedian"]
    assert s["boilMedian"] < 1.5 * clean.summary()["boilMedian"], "one cut must not move the median"


def test_an_occluding_card_is_dropped_not_booked_as_boil(clean):
    # A textured card sliding fast over the textured ground uncovers a strip every frame that
    # has no match in the frame before. The forward-backward test must drop it; scored, it is
    # boil that is not there (1.9x clean with the mask removed, against 1.4x with it).
    card = np.repeat(np.clip(_texture(42)[:220, :220], 0, 255).astype(np.uint8)[..., None], 3, axis=-1)
    frames = _clip()
    for t, frame in enumerate(frames):
        x = 200 + 30 * t
        frame[250:470, x:x + 220] = card
    s, c = stability.measure(frames, 24.0).summary(), clean.summary()
    assert s["validMedian"] < c["validMedian"] - 0.01, "the uncovered strip must leave the valid set"
    assert s["boilMedian"] < 1.6 * c["boilMedian"]


def test_grain_reads_the_noise_it_was_given_and_not_boil():
    # Immerkær's σ on the flat half: the clean pan reads near zero, planted grain reads back as
    # its own σ, and a re-rolling patch with no noise in it is boil, not grain.
    clean, boiling = _clip(), _clip(reroll=True)
    assert stability.grain(clean[0]) < 1.0
    assert stability.grain(boiling[0]) < stability.grain(clean[0]) + 0.2
    for sigma in (6.0, 12.0):
        assert stability.grain(_clip(grain=sigma)[0]) == pytest.approx(sigma, rel=0.08)


def test_sharp_edges_are_not_grain():
    # Anti-aliased lines and discs: the one structure the 3×3 high-pass answers to. Read on the
    # flat half they add nothing; the plain whole-frame estimate is held to the same predicate
    # and must FAIL it — the edge exclusion, not the picture's luck, is what keeps edges out.
    rng = np.random.default_rng(5)
    picture = np.full((H, W), 120, np.uint8)
    for _ in range(80):
        a, b = (int(rng.integers(0, W)), int(rng.integers(0, H))), (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.line(picture, a, b, int(rng.integers(0, 255)), int(rng.integers(1, 6)), cv2.LINE_AA)
    for _ in range(40):
        centre = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.circle(picture, centre, int(rng.integers(10, 120)), int(rng.integers(0, 255)), -1, cv2.LINE_AA)
    noisy = np.clip(picture + rng.normal(0, 3.0, picture.shape), 0, 255).astype(np.uint8)
    assert stability.grain(picture) < 0.2
    assert stability.grain(noisy) == pytest.approx(3.0, rel=0.08)
    assert not stability.grain(picture, flat_share=1.0) < 0.2, "the whole-frame estimate passes too — the mask is unproven"
    # dense thin lines, no noise: a blur before the edge test softens them into 'flat' and books
    # them as grain (3.3–4.0 over three seeds, against 0.5–0.7 on the raw gradient)
    lines = np.random.default_rng(5)
    thin = np.full((H, W), 120, np.float32)
    for _ in range(240):
        a, b = (int(lines.integers(0, W)), int(lines.integers(0, H))), (int(lines.integers(0, W)), int(lines.integers(0, H)))
        cv2.line(thin, a, b, float(120 + 80 * lines.choice([-1, 1])), int(lines.integers(1, 4)), cv2.LINE_AA)
    assert stability.grain(thin.astype(np.uint8)) < 1.5


def test_grain_is_absent_unless_full_resolution_frames_are_given():
    frames = _clip()
    assert "grainMedian" not in stability.measure(frames, 24.0).summary()
    s = stability.measure(frames, 24.0, grain_frames=_clip(grain=6.0)[:3]).summary()
    assert s["grainMedian"] == pytest.approx(6.0, rel=0.08)


def test_size_is_aspect_preserving_and_never_upscales():
    assert stability.stability_size(1280, 720) == (480, 270)
    assert stability.stability_size(720, 1280) == (270, 480)
    assert stability.stability_size(320, 180) == (320, 180)


# ------------------------------------------------------------------ the CLI's joins

def _write(path, frames, fps):
    # near-lossless H.264 from RGB frames, so a join reads the pictures written and not the codec
    import subprocess

    from depthc.video_io import _tool

    h, w = frames.shape[1:3]
    subprocess.run(
        [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv444p", str(path)],
        input=frames.tobytes(), check=True,
    )
    return path


def _picture(seed, w=480, h=270):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3.0)
    return np.repeat(np.clip(field / field.std() * 40 + 128, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def _stability_json(*args):
    import json

    from typer.testing import CliRunner

    from depthc.cli import app

    result = CliRunner().invoke(app, ["stability", *map(str, args), "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{"):])


def test_the_seam_reads_the_prior_take_s_real_last_frame(tmp_path):
    # A 60 fps prior whose LAST frame alone is the next take's opening picture. A 24 fps
    # conform drops that frame and reads the seam against the prior's other picture.
    joined, other = _picture(1), _picture(2)
    prior = np.stack([other] * 59 + [joined])
    nxt = np.stack([joined] * 24)
    seam = _stability_json(_write(tmp_path / "next.mp4", nxt, 24), "--after", _write(tmp_path / "prior.mp4", prior, 60))["seam"]
    assert seam["scored"] and seam["boil"] < 0.5, seam


def test_the_seam_finds_the_last_frame_of_a_slow_prior(tmp_path):
    # A 2 fps prior has no frame in its final quarter second (frames at 0, 0.5, 1.0, 1.5 s
    # of 2.0). A fixed end window decodes nothing and the whole command dies.
    joined, other = _picture(1), _picture(2)
    nxt = np.stack([joined] * 24)
    prior = np.stack([other] * 3 + [joined])
    seam = _stability_json(_write(tmp_path / "next.mp4", nxt, 24), "--after", _write(tmp_path / "prior.mp4", prior, 2))["seam"]
    assert seam["scored"] and seam["boil"] < 0.5, seam


def test_the_seam_keeps_a_vfr_prior_s_bunched_last_frame(tmp_path):
    # A VFR prior whose last frames are bunched onto (nearly) one timestamp — setpts asks for
    # a millisecond apart and the 1/25 timebase collapses them — the very last one alone
    # being the next take's opening picture. A CFR decode at the nominal rate drops it.
    import subprocess

    from depthc.video_io import _tool

    joined, other = _picture(1), _picture(2)
    prior = tmp_path / "prior.mp4"
    subprocess.run(
        [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "480x270",
         "-r", "25", "-i", "-", "-vf", "setpts='if(lt(N,25),N/25,0.96+(N-24)*0.001)/TB'",
         "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv444p", str(prior)],
        input=np.stack([other] * 27 + [joined]).tobytes(), check=True,
    )
    seam = _stability_json(_write(tmp_path / "next.mp4", np.stack([joined] * 24), 24), "--after", prior)["seam"]
    assert seam["scored"] and seam["boil"] < 0.5, seam


def test_a_join_that_cannot_be_tracked_is_not_printed_as_a_zero(tmp_path, monkeypatch):
    # Below the tracking floor gap() returns sentinel zeros. The repaint's gap alone is forced
    # there (the clip's own gaps still track, or the text report stops before its joins), so
    # it must come back unscored, with no boil or exposure to read as a clean join, in the
    # JSON and in the printed report.
    from typer.testing import CliRunner

    from depthc.cli import app

    clip = _write(tmp_path / "take.mp4", np.stack([_picture(1)] * 24), 24)
    still, picture = tmp_path / "still.png", _picture(3)
    cv2.imwrite(str(still), picture)
    real = stability.gap

    def gap(previous, current):
        if np.array_equal(previous, picture):
            return stability.GapStability(flicker=0.0, boil=0.0, warp=0.0, valid=0.1)
        return real(previous, current)

    monkeypatch.setattr(stability, "gap", gap)
    repaint = _stability_json(clip, "--still", still)["repaint"]
    assert repaint == {"from": "still.png", "tracked": 0.1, "scored": False}
    printed = CliRunner().invoke(app, ["stability", str(clip), "--still", str(still)]).output
    assert "not scored, only 10% of pixels tracked" in printed and "boil 0.00" not in printed, printed
