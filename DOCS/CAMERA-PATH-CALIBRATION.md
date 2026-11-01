# Camera-path calibration — Blender clay, known camera

`depthc describe` prints the camera over time ("pans left 0–1.5s, then pushes in"), and
`depthc compare` prints a PATH line beside its verdict. Until 2026-10-26 neither had been
checked against a camera whose move was known. This is that check. It cost $0: Blender
and the depth model both run on this Mac, and nothing calls a paid API.

```bash
./.tools/uv run python scripts/calibrate_camera_path.py                       # the main report
./.tools/uv run python scripts/calibrate_camera_path.py \
    --estimators rgb-tex,blender-tex,vda-tex --tag texture                     # diagnostic 1
./.tools/uv run python scripts/calibrate_camera_path.py \
    --estimators rgb-sim,blender-sim --tag similarity                          # diagnostic 2
```

Outputs land in `Outbox/calibration/` (`report*.md`, `results*.json`, the renders), which
is gitignored media. The numbers below are copied from those reports.

## How it works

1. **The scenes.** There are 20 cases × 4 s at 24 fps. Each one is an `oside-blockout/v1`
   spec rendered by oside-blender-mcp: a clay ground plane, rows of coloured boxes at 8–90 m,
   and sky. Blender writes the exact Z pass and `camera_track.json` alongside the clay.
2. **The truth.** `depthc.calibrate` projects the static world through the recorded camera
   with the Z pass to get the TRUE screen motion per frame, on the same analysis grid, in
   the same (dx, dy, radial) units that `motion.flow_series` produces.
3. **The scoring.** Both series go through `describe.path_buckets`, the one function that
   turns a path into words. Each half-second bucket is scored EXACT when the words match.
   What is scored is therefore exactly what `describe` prints.
4. **The readers.**
   - `rgb` is the clay picture itself (`describe --rgb`).
   - `vda` is VDA-Small depth of the clay picture (the generation side of `compare`).
   - `blender` is Blender's exact depth (the reference side of a kit, with no estimation
     error).

**Clay is the hard case.** Flat colour has no texture on the ground or the sky, and an AI
take has texture everywhere. A miss here is a lower bound on the reader, not a forecast of
what it does on a real take.

## Main result — the shipped reader

| reader | exact | translation right | radial right | missed motion | false motion | speed ratio |
|---|---|---|---|---|---|---|
| rgb | 111/140 | 124/140 | 126/140 | 24 | 0 | 0.698 |
| blender | 99/140 | 117/140 | 122/140 | 34 | 0 | 0.76 |
| vda | 97/140 | 116/140 | 121/140 | 37 | 0 | 0.496 |

- **It never invents motion.** All 140 buckets of every reader reported zero false motion.
  That includes `static-subject-crosses`, where a box crosses a locked frame.
- **It misses slow and lens-compressed motion.**
  - `pan-left-85mm`: 2, 0 and 1 of 7.
  - `pan-left-24mm` on depth: 2 and 0 of 7.
  - `push-in-slow` and `crane-down`: 1 of 7 on every reader.
- **Tilts pick up a radial word.** `tilt-down` reads "tilts down, pushes in" on 2–3
  buckets.
- **It under-reads speed.** The median speed ratio is 0.50–0.76, so a tempo word can sit
  one band slow.

## Diagnostic 1 — the same flow, read on textured pixels only (`-tex`)

This uses the same Farneback flow. Two things are read on textured pixels only
(Shi-Tomasi corner strength ≥ 1% of the frame's maximum, minimum 40 pixels): the median,
and the static floor that decides whether a gap moved at all. The diagnostic changes both,
so its gain cannot be pinned on either one alone.

| reader | exact | translation right | radial right | missed motion | false motion | speed ratio |
|---|---|---|---|---|---|---|
| rgb-tex | 122/140 | 134/140 | 126/140 | 14 | 0 | 0.987 |
| blender-tex | 131/140 | 131/140 | 140/140 | 1 | 0 | 0.99 |
| vda-tex | 130/140 | 133/140 | 137/140 | 2 | 0 | 0.979 |

**Every pan, tilt, truck and lens case is 7/7 on all three readers.** That includes 24 mm,
85 mm, both tilts and `pan-right-fast`. The speed ratio comes up to 0.98–0.99. **So the
main-report misses were mostly the scene, not the flow.** The reader was voting with flat
clay that carries no motion signal, and on a clay frame that is most of the pixels.

The misses left are in the radial cases:

- **`push-in`** is exact on only 1, 1 and 2 of 7, and reads "tilts down, pushes in" on 2, 6
  and 5 buckets. The main report read it 6, 5 and 5 of 7, so the mask caused this: the
  texture sits below centre (the boxes on the ground), so the outward flow of a push drags
  the median down, and the median-then-radial split books that as a tilt.
- **`pull-out`** reads 2, 5 and 4 of 7 (5, 4 and 2 on the main report). rgb-tex books its
  misses as static, so the textured static floor is part of it there.
- **`push-in-slow`** reads 2 of 7 on rgb-tex, and 7 and 6 on the depth readers (1 of 7 on
  every reader in the main report).
- **The cranes** each lose a bucket here and there (6 or 7 of 7).

## Diagnostic 2 — translation and zoom fitted together (`-sim`)

This keeps the same textured pixels but uses a RANSAC similarity fit (shift, uniform
scale, rotation) instead of median-then-radial. The fit has no off-centre bias.

| reader | exact | translation right | radial right | missed motion | false motion | speed ratio |
|---|---|---|---|---|---|---|
| rgb-sim | 128/140 | 137/140 | 131/140 | 3 | 0 | 0.981 |
| blender-sim | 132/140 | 139/140 | 133/140 | 0 | 1 | 0.979 |

- **It fixes the push.** `push-in` reads 6/7 and 7/7, and `pull-out` 7/7 on both.
- **It breaks fast lateral motion.** `pan-right-fast` drops to 2/7 and 3/7, with a spurious
  "pulls out" or "pushes in". `truck-right` drops to 4/7 on blender-sim. blender-sim also
  books one false motion.

**Neither diagnostic dominates.** The texture mask is right on lateral motion and wrong
on an off-centre push. The joint fit is the reverse.

## What this changes, and what it does not

- **Nothing in `depthc` changed.** Both diagnostics live in the calibration script only.
  The PATH line in `compare` and the camera line in `describe` still do not vote.
- **A texture-masked median is a candidate change to `motion.flow_series`, not an applied
  one.** It would move `compare`'s PATH numbers and `describe`'s camera words for every
  kit. It needs its own red/green tests, a fix for the off-centre push, and the owner's view
  before it ships.
- **The real-take reading is an inference, not a measurement.** AI takes carry texture
  everywhere, so the shipped all-pixel reader should land nearer diagnostic 1 than the
  clay report. Nothing here measured that. The next check is a generation with a camera
  the prompt named, read by `describe` and checked by eye.

## Limitations

- **One scene.** Clay ground, rows of boxes and sky. The truth counts the static world
  only, so a crossing subject is excluded from it by construction.
- **Cranes read as tilts by design.** Translation and rotation are not told apart: on
  screen, a crane and a tilt both slide the picture vertically. The truth uses the same
  words, so `crane-up` is scored against "tilts up".
- **A bucket is half a second.** A move shorter than that is outside what this
  calibration can score.
