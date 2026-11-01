"""Temporal stability — how much a take BOILS and FLICKERS once its motion is accounted for.

`motion.py` asks how much moves. This asks the opposite question of a GENERATED take: after
the real motion is warped away, what is still changing? An AI take's characteristic faults
live there — texture that re-rolls frame to frame (hair, foliage, fabric, brick: "boil") and
exposure that breathes ("flicker") — and neither is motion, so the motion figures cannot see
them. CorridorKey 2.0 trains against exactly this (a flow-compared temporal loss); this is the
same comparison used as a meter instead of a loss.

THE MEASURE (the video-consistency literature's warp error, Lai et al. 2018). For each frame
gap t → t+1: dense flow both ways (Farneback — classical, no weights, the estimator the
camera-path calibration scored), frame t+1 warped back onto t, and every pixel the two flows
disagree on dropped as occluded or untrackable (forward-backward consistency, Sundaram et al.
2010). On what survives:

    flicker  the signed mean luma change, grey levels — the whole picture brightening or
             darkening at once. Sign kept so a pump (+, -, +, -) is visible.
    boil     mean |change| in a DETAIL BAND (difference of Gaussians, σ 1 → 3 px at a 480 px
             long edge), grey levels — structure that changed although the flow says it only
             moved.
    warp     mean |change| of the full luma, exposure matched — the plain warp error, for
             comparison with published figures.

WHY 480 px AND A BAND, NOT FULL RESOLUTION. Film grain is new per-pixel noise every frame —
by this measure's own definition it would read as boil. At 480 px a 1080p frame's pixels are
averaged ~5:1 and the σ 1 blur averages again, so per-pixel grain mostly cancels before the
band sees it, while texture re-rolling at 2–6 px survives. That separation is a test, not a
claim (tests/test_stability.py: added grain barely moves `boil`; re-rolled texture does).

GRAIN, READ SEPARATELY. `grain` answers the opposite question — how much per-pixel noise a
frame carries — so an A/B on grain words can tell "the words changed the boil" from "the words
did nothing visible". Immerkær's fast noise estimate (1996: one fixed 3×3 high-pass whose
response to a smooth surface is zero, mean |response| scaled to a Gaussian σ), read at the
take's FULL resolution — at 480 px grain is already averaged away, which is the point of the
boil band — and only on the flatter half of the picture by Sobel gradient. The Sobel pair is
orthogonal to the high-pass, so choosing pixels by gradient does not bias the noise reading
(σ 4 planted reads 4.15–4.35 under 20–80 grey-level lines); a blur before the edge test was
tried and let thin lines through (0.91 against 0.35, no noise at all). Two leaks remain:
fine texture (hair, distant foliage) carries gradient and noise-like response alike, and a
saddle (a checker corner) has little gradient for its response. So it too ranks takes of one
shot and grades nothing.

WHAT THE NUMBERS ARE FOR. Ranking takes of the SAME shot (an A/B arm against its pair, a
re-fire against the take it replaces) and finding the bad second inside one take. Not an
absolute grade: flow error leaves a floor that depends on the content (the clay calibration
clips measure it), and nothing here has been calibrated against a human's "that boils".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STABILITY_LONG_EDGE = 480
DETAIL_SIGMAS = (1.0, 3.0)
# Sundaram et al. 2010: a pixel is consistent when |f + b|² < 0.01 (|f|² + |b|²) + 0.5
FB_RELATIVE = 0.01
FB_ABSOLUTE = 0.5
# A gap whose consistent share falls below this (a whip, a flash, a smear the flow cannot
# follow) is reported, not scored. A hard CUT usually does not fall below it — flow finds
# some agreement between two unrelated pictures — and reads instead as the clip's largest
# boil spike (~17 against ~0.7 on the synthetic test); `worstBoil` names it.
MIN_VALID_SHARE = 0.25
# grain is read on this share of the picture, the flattest by Sobel gradient (Tai & Yang 2008's
# edge exclusion, as a quantile instead of a tuned threshold)
GRAIN_FLAT_SHARE = 0.5
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


@dataclass
class GapStability:
    flicker: float
    boil: float
    warp: float
    valid: float


@dataclass
class Stability:
    fps: float
    flicker: np.ndarray = field(repr=False)
    boil: np.ndarray = field(repr=False)
    warp: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)
    grain: np.ndarray | None = field(default=None, repr=False)  # per sampled frame, full resolution

    @property
    def scored(self) -> np.ndarray:
        return self.valid >= MIN_VALID_SHARE

    def summary(self) -> dict:
        ok = self.scored
        if not ok.any():
            return {"gaps": int(self.valid.size), "scored": 0}
        boil = self.boil[ok]
        flicker = self.flicker[ok]
        worst = np.argsort(np.where(ok, self.boil, -1.0))[::-1][:3]
        return {
            "gaps": int(self.valid.size),
            "scored": int(ok.sum()),
            "boilMedian": round(float(np.median(boil)), 3),
            "boilP95": round(float(np.percentile(boil, 95)), 3),
            "flickerStd": round(float(np.std(flicker)), 3),
            "flickerMaxAbs": round(float(np.max(np.abs(flicker))), 3),
            "warpMedian": round(float(np.median(self.warp[ok])), 3),
            "validMedian": round(float(np.median(self.valid[ok])), 3),
            **({"grainMedian": round(float(np.median(self.grain)), 3)} if self.grain is not None and self.grain.size else {}),
            "worstBoil": [{"t": round(float(i) / self.fps, 2), "boil": round(float(self.boil[i]), 3)}
                          for i in worst if ok[i]],
        }


def stability_size(width: int, height: int) -> tuple[int, int]:
    """Aspect-preserving, long edge STABILITY_LONG_EDGE, even sides; never upscales."""
    scale = min(1.0, STABILITY_LONG_EDGE / max(width, height))
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def _luma(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = frame.shape
    size = stability_size(width, height)
    if size != (width, height):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def grain(frame: np.ndarray, *, flat_share: float = GRAIN_FLAT_SHARE) -> float:
    """Per-pixel noise σ in grey levels (Immerkær 1996) on the flattest `flat_share` of one frame.

    Pass the frame at its own resolution: this is the reading the 480 px grid throws away.
    `flat_share=1.0` is the plain whole-frame estimate, which books sharp edges as grain.
    NaN when a side is under 3 px: the reading is taken inside a one-pixel border, and with no
    interior there is nothing to read, which is not the same as no grain.
    """
    import cv2

    luma = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame).astype(np.float32)
    if min(luma.shape) < 3:
        return float("nan")
    response = np.abs(cv2.filter2D(luma, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    edge = np.hypot(cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3))[1:-1, 1:-1]
    flat = edge <= np.quantile(edge, flat_share)
    return float(np.sqrt(np.pi / 2.0) * response[flat].mean() / 6.0)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=4, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )


def _detail(luma: np.ndarray) -> np.ndarray:
    import cv2

    low, high = DETAIL_SIGMAS
    return cv2.GaussianBlur(luma, (0, 0), low) - cv2.GaussianBlur(luma, (0, 0), high)


def gap(previous: np.ndarray, current: np.ndarray) -> GapStability:
    """One gap: warp `current` back onto `previous` along the flow and measure what remains.

    Takes RGB or grey frames at any size; both are brought to the stability grid. Also the
    tool for a start-still repaint (still → first frame) and a chain seam (last frame of A →
    first frame of B): two pictures that should be one picture, moved.
    """
    import cv2

    a, b = _luma(previous), _luma(current)
    if a.shape != b.shape:
        raise ValueError(f"frames differ in size after resizing: {a.shape} vs {b.shape}")
    forward = _flow(a, b)
    backward = _flow(b, a)
    height, width = a.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x, map_y = xs + forward[..., 0], ys + forward[..., 1]
    inside = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    back_at = cv2.remap(backward, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mismatch = ((forward + back_at) ** 2).sum(axis=-1)
    bound = FB_RELATIVE * ((forward ** 2).sum(axis=-1) + (back_at ** 2).sum(axis=-1)) + FB_ABSOLUTE
    valid = inside & (mismatch < bound)

    af, bf = a.astype(np.float32), b.astype(np.float32)
    warped = cv2.remap(bf, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    share = float(valid.mean())
    if not valid.any():
        return GapStability(flicker=0.0, boil=0.0, warp=0.0, valid=0.0)
    flicker = float((warped - af)[valid].mean())
    # Exposure out before structure is read: a 5% gain step scales every edge by 5%, and the
    # first version of this booked a pure flicker as boil (1.0 → 2.3 on the synthetic test).
    # Match the warped frame's mean and contrast to t's over the consistent pixels.
    spread = float(warped[valid].std())
    gain = float(af[valid].std()) / spread if spread > 1e-6 else 1.0
    matched = (warped - float(warped[valid].mean())) * gain + float(af[valid].mean())
    # detail of the WARPED frame, not a warp of the detail: the band is a property of the
    # picture the viewer sees at t+1, sampled where the flow says each pixel came from
    boil = float(np.abs(_detail(matched) - _detail(af))[valid].mean())
    warp = float(np.abs(matched - af)[valid].mean())
    return GapStability(flicker=flicker, boil=boil, warp=warp, valid=share)


def measure(frames: np.ndarray, fps: float, *, grain_frames: np.ndarray | None = None) -> Stability:
    """Every gap of [N, H, W, 3] RGB or [N, H, W] grey frames.

    `grain_frames` are read for `grain` as given: a few full-resolution frames of the same
    take (the CLI samples 2 per second). Without them the grain reading is absent, never
    guessed from the 480 px frames.
    """
    gaps = [gap(frames[i], frames[i + 1]) for i in range(max(0, len(frames) - 1))]
    readings = None if grain_frames is None else [grain(f) for f in grain_frames]
    return Stability(
        fps=float(fps),
        flicker=np.array([g.flicker for g in gaps], dtype=np.float32),
        boil=np.array([g.boil for g in gaps], dtype=np.float32),
        warp=np.array([g.warp for g in gaps], dtype=np.float32),
        valid=np.array([g.valid for g in gaps], dtype=np.float32),
        grain=None if readings is None else np.array([r for r in readings if np.isfinite(r)], dtype=np.float32),
    )
