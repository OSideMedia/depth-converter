"""Distil a reference clip's LOOK — palette, key, warmth, detail — into numbers a prompt
can carry, and measure a generation against it.

Depth Converter copies a reference's MOTION through the depth channel. This module is
its still-image sibling: it copies the reference's LOOK through words. Born from
LIVEFIRE-2026-08-16-LOOK-DNA-AB (OSIDE): a Visual DNA distilled from four reference
keyframes pulled generated stills measurably toward the reference on TWO image models
(3/4 subjects each) — and the one subject that lost, lost on both models for the same
reason: a single GLOBAL palette ("dark timber, slate shadow, charcoal") darkened a
high-key shot. So a look is a global palette PLUS a per-shot KEY, never the palette alone.

What this measures, per keyframe and globally:
  palette   6 dominant colours, k-means in CIE Lab, with area shares (hex + Lab)
  key       luma mean, dark share (L* < 35), bright share (L* > 80) → high / mid / low key
  warmth    mean b* (positive = warm/yellow, negative = cool/blue)
  chroma    mean C*ab (how saturated the frame is overall)
  detail    mean edge magnitude (ornament density, "detail frequency")

What it emits: `look.json` with a `visualDna` block in OSIDE's VisualDnaSelection shape
(lib/prompter/visualDna.ts) — hexes filled, WORDS left empty on purpose. The live fire
showed the words are the lever and that a mechanical hex→name step is too coarse
(OSIDE's rendered the measured palette as "light orange, light orange, muted orange…");
naming the colours and the materials is authored work, by a person or an agent reading
the sheet, and the JSON says so.

EVERYTHING HERE IS UNCALIBRATED. `compare()` reports distances; it never says "matched".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

SCHEMA = "depthc-look/1"
PALETTE_K = 6
ANALYSIS_LONG_EDGE = 256          # colour statistics need no more than this
DARK_L, BRIGHT_L = 35.0, 80.0     # L* cut-offs for the shadow / highlight shares
# key thresholds — coarse on purpose; the sheet is the authority, these are the words
HIGH_KEY_L, LOW_KEY_L = 62.0, 38.0
HIGH_KEY_DARK, LOW_KEY_DARK = 0.12, 0.45
# shot-cut detection: Lab-histogram distance between consecutive samples above this
# opens a new shot; frames are sampled at SAMPLE_FPS for that pass only.
CUT_THRESHOLD = 0.35
SAMPLE_FPS = 2.0
MAX_KEYFRAMES = 8


# ---------------------------------------------------------------------------- colour maths

def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """[..., 3] uint8 sRGB → [..., 3] float Lab (L* 0–100, a*/b* ≈ −128..127). D65."""
    rgb = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_hex(lab: np.ndarray) -> str:
    L, a, b = (float(v) for v in lab)
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t: float) -> float:
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xyz = np.array([finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    lin = np.clip(m @ xyz, 0.0, None)
    srgb = np.where(lin > 0.0031308, 1.055 * np.power(lin, 1 / 2.4) - 0.055, 12.92 * lin)
    r, g, bb = (int(round(float(v) * 255)) for v in np.clip(srgb, 0.0, 1.0))
    return f"#{r:02x}{g:02x}{bb:02x}"


def _kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means (fixed seed, k-means++-lite init). Returns centroids sorted by share, shares."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        raise ValueError("no pixels")
    k = min(k, n)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        p = d / d.sum() if d.sum() > 0 else None
        centres.append(x[rng.choice(n, p=p)])
    c = np.array(centres, dtype=np.float64)
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = x[lab == j]
            if len(m):
                c[j] = m.mean(0)
    lab = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
    share = np.bincount(lab, minlength=k) / n
    order = np.argsort(-share)
    return c[order], share[order]


# ---------------------------------------------------------------------------- measurement

@dataclass
class Swatch:
    hex: str
    share: float
    lab: list[float]

    def as_dict(self) -> dict:
        return {"hex": self.hex, "share": round(self.share, 3), "lab": [round(v, 1) for v in self.lab]}


@dataclass
class FrameLook:
    index: int
    t: float
    luma_mean: float
    luma_std: float
    dark_share: float
    bright_share: float
    chroma: float
    warmth: float
    detail: float
    palette: list[Swatch]
    key: str = ""

    def as_dict(self) -> dict:
        d = {
            "index": self.index, "t": round(self.t, 2), "key": self.key,
            "luma_mean": round(self.luma_mean, 1), "luma_std": round(self.luma_std, 1),
            "dark_share": round(self.dark_share, 3), "bright_share": round(self.bright_share, 3),
            "chroma": round(self.chroma, 1), "warmth": round(self.warmth, 1), "detail": round(self.detail, 2),
            "palette": [s.as_dict() for s in self.palette],
        }
        return d


def key_label(luma_mean: float, dark_share: float) -> str:
    """Coarse exposure word. High key = bright and nearly shadowless; low key = dark and shadow-heavy."""
    if luma_mean >= HIGH_KEY_L and dark_share <= HIGH_KEY_DARK:
        return "high-key"
    if luma_mean <= LOW_KEY_L or dark_share >= LOW_KEY_DARK:
        return "low-key"
    return "mid-key"


def _shrink(frame: np.ndarray, long_edge: int = ANALYSIS_LONG_EDGE) -> np.ndarray:
    h, w = frame.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / long_edge)))
    return frame[::step, ::step]


def _edge_mean(gray: np.ndarray) -> float:
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    return float(gx + gy)


def measure_frame(frame: np.ndarray, index: int = 0, t: float = 0.0, k: int = PALETTE_K) -> FrameLook:
    """One RGB uint8 frame → its look numbers."""
    small = _shrink(np.asarray(frame))
    lab = rgb_to_lab(small).reshape(-1, 3)
    L = lab[:, 0]
    centres, shares = _kmeans(lab, k)
    palette = [Swatch(lab_to_hex(c), float(s), [float(v) for v in c]) for c, s in zip(centres, shares)]
    gray = (0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]).astype(np.float64)
    fl = FrameLook(
        index=index, t=t,
        luma_mean=float(L.mean()), luma_std=float(L.std()),
        dark_share=float((L < DARK_L).mean()), bright_share=float((L > BRIGHT_L).mean()),
        chroma=float(np.hypot(lab[:, 1], lab[:, 2]).mean()), warmth=float(lab[:, 2].mean()),
        detail=_edge_mean(gray), palette=palette,
    )
    fl.key = key_label(fl.luma_mean, fl.dark_share)
    return fl


# ---------------------------------------------------------------------------- keyframes

def _lab_hist(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    lab = rgb_to_lab(_shrink(frame, 96)).reshape(-1, 3)
    edges = [np.linspace(0, 100, bins + 1), np.linspace(-90, 90, bins + 1), np.linspace(-90, 90, bins + 1)]
    h, _ = np.histogramdd(lab, bins=edges)
    h = h.ravel()
    return h / max(h.sum(), 1.0)


def pick_keyframes(frames: np.ndarray, fps: float, max_frames: int = MAX_KEYFRAMES,
                   threshold: float = CUT_THRESHOLD) -> list[int]:
    """Split a sampled clip into shots at colour cuts and return the middle frame index of each.

    A cut is a Lab-histogram L1 distance above `threshold` between consecutive samples.
    No cuts → evenly spaced frames. Long shot lists are thinned evenly to `max_frames`
    (reported by the caller, never silent).
    """
    n = len(frames)
    if n == 0:
        return []
    hists = [_lab_hist(f) for f in frames]
    cuts = [0]
    for i in range(1, n):
        if 0.5 * np.abs(hists[i] - hists[i - 1]).sum() > threshold:
            cuts.append(i)
    cuts.append(n)
    shots = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    if len(shots) <= 1:
        want = min(max_frames, max(1, n)) if n < 4 * max_frames else 4
        idx = np.linspace(0, n - 1, num=min(want, n)).round().astype(int)
        return sorted(set(int(i) for i in idx))
    mids = [(a + b) // 2 for a, b in shots]
    if len(mids) > max_frames:
        keep = np.linspace(0, len(mids) - 1, num=max_frames).round().astype(int)
        mids = [mids[i] for i in keep]
    return mids



# ---------------------------------------------------------------------------- swatch words

def swatch_hint(lab: list[float], share: float) -> str:
    """A COARSE, honest descriptor for authoring — lightness / temperature / saturation.
    A hint beside the hex, never the prose itself (a mechanical name proved too coarse)."""
    L, a, b = lab
    light = ("deep" if L < 30 else "dark" if L < 45 else "mid" if L < 65 else "light" if L < 80 else "pale")
    temp = "cool" if b < -5 else "warm" if b > 5 else "neutral"
    c = float(np.hypot(a, b))
    sat = "greyed" if c < 8 else "muted" if c < 25 else "saturated"
    return f"{share:.0%} · {light}, {temp}, {sat}"


def style_constant_block(palette: list["Swatch"], frames: list["FrameLook"], keys: dict[str, int],
                         warmth: float, chroma: float, detail: float, dark_share: float, source: str) -> dict:
    """A paste-ready block in OSIDE's StyleConstant shape (lib/prompter/styleConstant.ts).

    Only the measurable part is filled — `color.hex` (top-3 shares, the 60:30:10 slots) —
    every prose field is EMPTY on purpose and the `hints` say what the sheet shows, so the
    author (person or agent) writes positive, observable prose from evidence instead of
    inventing structure. Nothing here compiles until the words are written.
    """
    top = palette[:3]
    Ls = [f.luma_mean for f in frames]
    key_parts = ", ".join(f"{k} ×{v}" for k, v in keys.items() if v)
    override = ""
    if keys.get("high-key", 0) and (keys.get("low-key", 0) or dark_share > 0.25):
        override = " — mixed keys: give the high-key scenes their own lighting pick, do not let the global palette darken them"
    elif keys.get("high-key", 0) == 0:
        override = " — no high-key shot in the reference: a bright scene in the film needs its own lighting pick (LOOK-DNA-AB gate lesson)"
    slot = ("dominant", "secondary", "accent")
    return {
        "name": f"{source} look",
        "presetId": "",
        "mediumLock": "",
        "anchor": "",
        "lightingLaw": "",
        "color": {"dominant": "", "secondary": "", "accent": "", "hex": [s.hex for s in top]},
        "continuity": "",
        "hints": {
            "color": [f"{n}: {s.hex} — {swatch_hint(s.lab, s.share)}" for n, s in zip(slot, top)],
            "palette_rest": [f"{s.hex} — {swatch_hint(s.lab, s.share)}" for s in palette[3:]],
            "key": f"{key_parts} (L* {min(Ls):.0f}–{max(Ls):.0f}), dark share {dark_share:.0%}{override}",
            "warmth": f"b* {warmth:+.1f} — {'warm bias' if warmth > 2 else 'cool bias' if warmth < -2 else 'neutral'}",
            "chroma": f"{chroma:.1f} — {'restrained' if chroma < 12 else 'moderate' if chroma < 25 else 'saturated'}",
            "detail": f"{detail:.1f} edge mean — {'sparse' if detail < 15 else 'moderate' if detail < 30 else 'dense ornament'}",
        },
        "authoring": (
            "Prose fields are EMPTY on purpose. Read sheet.jpg and write POSITIVE, observable prose: "
            "mediumLock (what the image IS), anchor (light/lens/era, no names), lightingLaw, "
            "color.dominant/secondary/accent as material + light words, continuity (palette pins, design locks). "
            "Then paste into OSIDE › Look tab › Style Constant. Keep the key per scene."
        ),
    }

# ---------------------------------------------------------------------------- the look

@dataclass
class Look:
    source: str
    seconds: float
    frames: list[FrameLook]
    palette: list[Swatch]
    luma_mean: float
    dark_share: float
    bright_share: float
    chroma: float
    warmth: float
    detail: float
    keys: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def key_range(self) -> str:
        Ls = [f.luma_mean for f in self.frames]
        parts = [f"{k} ×{v}" for k, v in self.keys.items() if v]
        return f"{', '.join(parts)} (L* {min(Ls):.0f}–{max(Ls):.0f})"

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "seconds": round(self.seconds, 2),
            "global": {
                "palette": [s.as_dict() for s in self.palette],
                "luma_mean": round(self.luma_mean, 1), "dark_share": round(self.dark_share, 3),
                "bright_share": round(self.bright_share, 3), "chroma": round(self.chroma, 1),
                "warmth": round(self.warmth, 1), "detail": round(self.detail, 2),
                "key_range": self.key_range,
            },
            "frames": [f.as_dict() for f in self.frames],
            # OSIDE lib/prompter/visualDna.ts VisualDnaSelection — hexes filled, words AUTHORED.
            "visualDna": {
                "palette": [s.hex for s in self.palette],
                "lighting": "", "materials": "", "era": "", "juxtaposition": "",
                "forbiddenList": [],
            },
            # OSIDE lib/prompter/styleConstant.ts — paste-ready; hexes measured, prose authored
            "styleConstant": style_constant_block(
                self.palette, self.frames, self.keys, self.warmth, self.chroma, self.detail, self.dark_share, self.source
            ),
            # the Stage-1 lesson: a look is the palette PLUS a per-shot key, never the palette alone
            "perShotKey": {str(f.index): f.key for f in self.frames},
            "notes": self.notes,
        }


def build_look(frames: np.ndarray, times: list[float], source: str, seconds: float,
               notes: list[str] | None = None) -> Look:
    if len(frames) == 0:
        raise ValueError("no keyframes")
    fls = [measure_frame(f, i, t) for i, (f, t) in enumerate(zip(frames, times))]
    # global palette: k-means over the union of every keyframe's pixels, equal weight per frame
    union = np.concatenate([rgb_to_lab(_shrink(f, 128)).reshape(-1, 3) for f in frames])
    centres, shares = _kmeans(union, PALETTE_K)
    palette = [Swatch(lab_to_hex(c), float(s), [float(v) for v in c]) for c, s in zip(centres, shares)]
    keys: dict[str, int] = {"high-key": 0, "mid-key": 0, "low-key": 0}
    for f in fls:
        keys[f.key] += 1
    base_notes = [
        "Uncalibrated. Numbers describe the reference; nothing here is a verdict.",
        "visualDna.palette holds measured hexes; lighting/materials/era/juxtaposition/forbiddenList are AUTHORED from the sheet — a mechanical hex→name step proved too coarse (LOOK-DNA-AB 2026-08-16).",
        "Apply perShotKey: a global palette pushed a high-key shot dark on two models — carry the key per shot.",
    ]
    return Look(
        source=source, seconds=seconds, frames=fls, palette=palette,
        luma_mean=float(np.mean([f.luma_mean for f in fls])),
        dark_share=float(np.mean([f.dark_share for f in fls])),
        bright_share=float(np.mean([f.bright_share for f in fls])),
        chroma=float(np.mean([f.chroma for f in fls])),
        warmth=float(np.mean([f.warmth for f in fls])),
        detail=float(np.mean([f.detail for f in fls])),
        keys=keys, notes=base_notes + list(notes or []),
    )


def look_from_dict(d: dict) -> "Look":
    """Rebuild enough of a Look from look.json to compare against."""
    g = d["global"]
    frames = [
        FrameLook(
            index=f["index"], t=f["t"], luma_mean=f["luma_mean"], luma_std=f.get("luma_std", 0.0),
            dark_share=f["dark_share"], bright_share=f["bright_share"], chroma=f["chroma"],
            warmth=f["warmth"], detail=f["detail"],
            palette=[Swatch(s["hex"], s["share"], s["lab"]) for s in f["palette"]], key=f.get("key", ""),
        )
        for f in d["frames"]
    ]
    keys: dict[str, int] = {"high-key": 0, "mid-key": 0, "low-key": 0}
    for f in frames:
        keys[f.key or key_label(f.luma_mean, f.dark_share)] += 1
    return Look(
        source=d.get("source", "?"), seconds=d.get("seconds", 0.0), frames=frames,
        palette=[Swatch(s["hex"], s["share"], s["lab"]) for s in g["palette"]],
        luma_mean=g["luma_mean"], dark_share=g["dark_share"], bright_share=g["bright_share"],
        chroma=g["chroma"], warmth=g["warmth"], detail=g["detail"], keys=keys, notes=list(d.get("notes", [])),
    )


# ---------------------------------------------------------------------------- compare

def palette_distance(a: list[Swatch], b: list[Swatch]) -> float:
    """Symmetric share-weighted nearest-centroid distance in Lab (0 = same palette)."""
    def one(p: list[Swatch], q: list[Swatch]) -> float:
        return sum(s.share * min(np.linalg.norm(np.array(s.lab) - np.array(o.lab)) for o in q) for s in p)
    return float((one(a, b) + one(b, a)) / 2.0)


@dataclass
class Delta:
    t: float
    palette: float      # Lab units
    luma: float         # candidate − reference L*
    dark: float         # candidate − reference dark share
    warmth: float       # candidate − reference b*
    chroma: float
    detail: float
    key: str
    nearest_frame: int  # which reference keyframe this candidate frame is closest to (palette)

    def as_dict(self) -> dict:
        return {
            "t": round(self.t, 2), "palette": round(self.palette, 1), "luma": round(self.luma, 1),
            "dark": round(self.dark, 3), "warmth": round(self.warmth, 1), "chroma": round(self.chroma, 1),
            "detail": round(self.detail, 2), "key": self.key, "nearest_frame": self.nearest_frame,
        }


def compare_frame(look: Look, frame: np.ndarray, t: float = 0.0, against: int | None = None) -> Delta:
    """Distance of one candidate frame from the look — against one named keyframe, or the global."""
    x = measure_frame(frame, 0, t)
    dists = [palette_distance(x.palette, f.palette) for f in look.frames]
    nearest = int(np.argmin(dists)) if dists else -1
    if against is not None:
        ref = look.frames[against]
        return Delta(t, dists[against], x.luma_mean - ref.luma_mean, x.dark_share - ref.dark_share,
                     x.warmth - ref.warmth, x.chroma - ref.chroma, x.detail - ref.detail, x.key, nearest)
    return Delta(t, palette_distance(x.palette, look.palette), x.luma_mean - look.luma_mean,
                 x.dark_share - look.dark_share, x.warmth - look.warmth, x.chroma - look.chroma,
                 x.detail - look.detail, x.key, nearest)


def compare_frames(look: Look, frames: np.ndarray, times: list[float], against: int | None = None) -> list[Delta]:
    return [compare_frame(look, f, t, against) for f, t in zip(frames, times)]


def drift(deltas: list[Delta]) -> dict:
    """First-vs-last summary for a clip: did the look hold over time?"""
    if not deltas:
        return {}
    a, z = deltas[0], deltas[-1]
    return {
        "palette_first": round(a.palette, 1), "palette_last": round(z.palette, 1),
        "palette_drift": round(z.palette - a.palette, 1),
        "luma_drift": round(z.luma - a.luma, 1), "warmth_drift": round(z.warmth - a.warmth, 1),
        "keys": [d.key for d in deltas],
    }


# ---------------------------------------------------------------------------- render

def render_look(look: Look) -> str:
    lines = [f"LOOK  {look.source}  {look.seconds:.1f}s  {len(look.frames)} keyframes"]
    lines.append(f"  key      {look.key_range}")
    lines.append(f"  luma     mean L* {look.luma_mean:.0f} · dark {look.dark_share:.0%} · bright {look.bright_share:.0%}")
    lines.append(f"  warmth   b* {look.warmth:+.1f}   chroma {look.chroma:.1f}   detail {look.detail:.1f}")
    lines.append("  palette  " + "  ".join(f"{s.hex} {s.share:.0%}" for s in look.palette))
    for f in look.frames:
        lines.append(
            f"  [{f.index}] {f.t:5.1f}s  {f.key:8}  L* {f.luma_mean:4.0f}  dark {f.dark_share:.0%}  "
            f"b* {f.warmth:+.1f}  detail {f.detail:.1f}  " + " ".join(s.hex for s in f.palette[:4])
        )
    sc = style_constant_block(look.palette, look.frames, look.keys, look.warmth, look.chroma, look.detail,
                              look.dark_share, look.source)
    lines.append("  STYLE CONSTANT (paste-ready, prose empty on purpose)")
    for h in sc["hints"]["color"]:
        lines.append(f"    {h}")
    lines.append(f"    key      {sc['hints']['key']}")
    lines.append(f"    warmth   {sc['hints']['warmth']} · chroma {sc['hints']['chroma']} · detail {sc['hints']['detail']}")
    lines.append("  words: authored, not derived — read the sheet, name the colours and materials, keep the key per shot.")
    return "\n".join(lines)


def render_compare(look: Look, deltas: list[Delta], label: str, against: int | None) -> str:
    target = f"keyframe [{against}]" if against is not None else "the global look"
    lines = [f"COMPARE  {label}  vs {look.source} ({target}) — distances, not verdicts"]
    lines.append("      t   palette   ΔL*   Δdark   Δb*   Δchroma  Δdetail  key       nearest")
    for d in deltas:
        lines.append(
            f"  {d.t:5.1f}   {d.palette:6.1f}  {d.luma:+5.1f}  {d.dark:+.2f}  {d.warmth:+5.1f}   "
            f"{d.chroma:+6.1f}  {d.detail:+7.2f}  {d.key:8}  [{d.nearest_frame}]"
        )
    if len(deltas) > 1:
        dr = drift(deltas)
        lines.append(
            f"  drift  palette {dr['palette_first']} → {dr['palette_last']} ({dr['palette_drift']:+.1f})   "
            f"L* {dr['luma_drift']:+.1f}   b* {dr['warmth_drift']:+.1f}   keys {'→'.join(dr['keys'])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- package (manifest) mode

def hex_to_lab(h: str) -> list[float]:
    h = h.strip().lstrip("#")
    rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]], dtype=np.uint8)
    return [float(v) for v in rgb_to_lab(rgb)[0]]


def look_from_manifest_block(block: dict, source: str = "manifest") -> "Look":
    """Build a comparable Look from an OSIDE manifest `look` block (piece 1 of look-travels):
    `hex` (colour-law swatches, dominant first) + `targets` {luma_mean, dark_share, warmth, chroma}.
    Shares fall back to the 60:30:10 law when the block carries none. Frames are synthetic
    (one frame = the global), so `nearest_frame` is meaningless here and reported as 0."""
    hexes = list(block.get("hex") or [])
    if not hexes:
        raise ValueError("manifest look block carries no hex swatches")
    shares = list(block.get("shares") or [])
    if len(shares) != len(hexes):
        law = [0.6, 0.3, 0.1]
        shares = (law + [0.0] * len(hexes))[: len(hexes)]
        tot = sum(shares) or 1.0
        shares = [x / tot for x in shares]
    palette = [Swatch(h, float(sh), hex_to_lab(h)) for h, sh in zip(hexes, shares)]
    t = block.get("targets") or {}
    lum = float(t.get("luma_mean", np.mean([p.lab[0] for p in palette])))
    dark = float(t.get("dark_share", 0.0))
    warm = float(t.get("warmth", np.mean([p.lab[2] for p in palette])))
    chroma = float(t.get("chroma", np.mean([np.hypot(p.lab[1], p.lab[2]) for p in palette])))
    fl = FrameLook(0, 0.0, lum, 0.0, dark, 0.0, chroma, warm, 0.0, palette, key_label(lum, dark))
    keys = {"high-key": 0, "mid-key": 0, "low-key": 0}
    keys[fl.key] += 1
    return Look(source=block.get("name") or source, seconds=0.0, frames=[fl], palette=palette,
                luma_mean=lum, dark_share=dark, bright_share=0.0, chroma=chroma, warmth=warm, detail=0.0,
                keys=keys, notes=["built from a manifest look block — targets, not measured keyframes"])


@dataclass
class ClipScore:
    file: str
    label: str
    seconds: int
    palette: float
    luma: float
    warmth: float
    chroma: float
    dark: float
    key_track: str
    outlier: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "file": self.file, "label": self.label, "seconds": self.seconds,
            "palette": round(self.palette, 1), "luma": round(self.luma, 1), "warmth": round(self.warmth, 1),
            "chroma": round(self.chroma, 1), "dark": round(self.dark, 3), "key_track": self.key_track,
            "outlier": self.outlier, "error": self.error,
        }


def score_clip(look: Look, frames: np.ndarray, times: list[float], file: str, label: str) -> ClipScore:
    deltas = compare_frames(look, frames, times, None)
    if not deltas:
        return ClipScore(file, label, 0, 0, 0, 0, 0, 0, "", error="no frames")
    keys = [d.key for d in deltas]
    track = keys[0] if len(set(keys)) == 1 else "→".join(k[0].upper() for k in keys)
    return ClipScore(
        file, label, len(deltas),
        float(np.mean([d.palette for d in deltas])), float(np.mean([d.luma for d in deltas])),
        float(np.mean([d.warmth for d in deltas])), float(np.mean([d.chroma for d in deltas])),
        float(np.mean([d.dark for d in deltas])), track,
    )


# outlier rule — RELATIVE to the package, report-only: a clip is flagged on an axis when it
# sits more than OUTLIER_Z standard deviations from the package mean on that axis, and the
# axis spread is not trivially small. Nothing here is calibrated to "wrong"; it says
# "furthest from its siblings — look here first".
OUTLIER_Z = 1.5
OUTLIER_MIN_SPREAD = {"palette": 1.0, "luma": 3.0, "warmth": 2.0, "chroma": 2.0}


def flag_outliers(scores: list[ClipScore]) -> list[ClipScore]:
    ok = [s for s in scores if s.error is None]
    if len(ok) < 3:
        return scores
    for axis in ("palette", "luma", "warmth", "chroma"):
        vals = np.array([getattr(s, axis) for s in ok], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std())
        if sd < OUTLIER_MIN_SPREAD[axis]:
            continue
        for s, v in zip(ok, vals):
            if abs(v - mu) > OUTLIER_Z * sd:
                s.outlier.append(f"{axis} {v - mu:+.1f} vs package mean")
    return scores


def render_package(look: Look, scores: list[ClipScore]) -> str:
    lines = [f"PACKAGE DRIFT  {len(scores)} clips vs {look.source} — distances, not verdicts; outliers are relative to the package"]
    lines.append("  clip                            s   palette   ΔL*    Δb*  Δchroma  key      flag")
    for s in scores:
        if s.error:
            lines.append(f"  {s.label[:30]:30}  --   ERROR {s.error}")
            continue
        flag = "; ".join(s.outlier) if s.outlier else ""
        lines.append(
            f"  {s.label[:30]:30} {s.seconds:2d}   {s.palette:6.1f}  {s.luma:+5.1f}  {s.warmth:+5.1f}   {s.chroma:+6.1f}  {s.key_track:8} {flag}"
        )
    ok = [s for s in scores if s.error is None]
    if ok:
        lines.append(
            f"  package mean   palette {np.mean([s.palette for s in ok]):.1f}   ΔL* {np.mean([s.luma for s in ok]):+.1f}   "
            f"Δb* {np.mean([s.warmth for s in ok]):+.1f}   Δchroma {np.mean([s.chroma for s in ok]):+.1f}   "
            f"outliers {sum(1 for s in ok if s.outlier)}/{len(ok)}"
        )
    return "\n".join(lines)
