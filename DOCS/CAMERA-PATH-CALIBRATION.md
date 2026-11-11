# Camera-path calibration — Blender clay, known camera

`depthc describe` prints the camera over time ("pans left 0–1.5s, then pushes in"), and
`depthc compare` prints a PATH line beside its verdict. Until 2026-10-26 neither had been
checked against a camera whose move was known. This is that check. It cost $0: Blender
and the depth model both run on this Mac, and nothing calls a paid API.

```bash
./.tools/uv run python scripts/calibrate_camera_path.py                       # the tuned clips
./.tools/uv run python scripts/calibrate_camera_path.py --holdout --tag holdout  # the unseen clips
./.tools/uv run python scripts/calibrate_camera_path.py \
    --estimators rgb-tex,blender-tex,vda-tex --tag texture                     # diagnostic 1
./.tools/uv run python scripts/calibrate_camera_path.py \
    --estimators rgb-sim,blender-sim --tag similarity                          # diagnostic 2
./.tools/uv run python scripts/calibrate_camera_path.py --estimators rgb,pic --tag pic  # the picture reader
```

Outputs land in `Outbox/calibration/` (`report*.md`, `results*.json`, the renders), which
is gitignored media. The numbers below are copied from those reports.

## How it works

1. **The scenes.** There are 24 cases the reader was tuned on and 9 held back from it, each
   4 s at 24 fps. Each one is an `oside-blockout/v1` spec rendered by oside-blender-mcp: a clay
   ground plane, usually rows of coloured boxes at 8–90 m, and sky. Blender writes the exact Z
   pass and `camera_track.json` alongside the clay.
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

## The reader now: tracked points on the picture (2026-11-11)

Since 2026-11-11 the camera words (`describe`, `compare`'s PATH line, a kit's `cameraPath`) are
read from the picture instead of the depth. The one exception is `compare` on a kit that has no
usable `source.mp4`, described further down. `motion.camera_path` does it in five steps:

1. **Track points.** Up to 600 corners per frame, followed to the next frame at a 320 px long
   edge. A point is kept only when following it there and back lands where it started.
2. **Drop what moves on its own.** One shift, scale and rotation is fitted to the points. A point
   that does not go where the fit sends it is a person, a car or a wave, and it loses its vote.
3. **Refuse a frame with no world in it.** If the fit carries fewer than half the points, the gap
   names no camera (since the camera round below, it reads "can't tell" rather than still).
4. **Vote by region, as before.** The points that are left vote through the same 8×8 split and
   the same stretch-then-shift reading as the region reader below, with the same floors and the
   same words. A gap with fewer than 40 usable points, or with votes in fewer than 12 regions, is
   handed to the region reader on the same picture.
5. **Call a held shot still.** Tracking never returns an exact zero, so a gap counts as moving
   only where the path averaged over the half-second around it would earn a word.

The reason is in "Real footage" below: a depth picture of a real scene is mostly flat, so what
has an edge in it is whatever stands nearer than the rest, and in a locked-off shot that is the
people walking through. They cast the votes. Tracking also resolves a slower move than dense
flow on the 160 px grid, which is how the missed slow push-ins come back without lowering a floor.

`compare` reads the reference's camera from the kit's own `source.mp4`, cropped to the
rectangle the build recorded. A kit without one, or
one whose picture does not match its depth frame for frame, is read from the depth as before,
and `compare` says so.

**On clay it changes little, as it should.** Clay has almost nothing to track, so most gaps fall
back to the region reader. `pic` is this reader, `rgb` is the region reader on the same picture:

| | `rgb` | `pic` |
|---|---|---|
| Tuned clips, exact buckets | 162/168 | 162/168 |
| Tuned, missed motion | 1 | 0 |
| Held-out clips, exact buckets | 50/63 | 52/63 |
| Held-out, missed motion | 4 | 3 |
| False motion, both sets | 0 | 0 |

On the tuned clips it gains two buckets and loses two: a crane and a truck with a push each read
one bucket better, and two buckets of a fast truck pick up "pulls out".

**On unseen real footage it failed** ("Second real-footage score" below): 7 of 10 moves, and a
move invented on 2 of 7 locked-off shots. The camera round that followed is described after that
score, and has not been scored. On the 17 spent clips it names 12 of 12 moves and
invents a move on 1 of 5 locked-off shots, but it was built while looking at those clips, so that
was never a score.

**What it cannot do.**

- **A subject that fills the frame is the world.** On the spent clips the one locked-off shot it still gets
  wrong is a train passing close to the lens, and on the fresh ones a person close to the lens and
  a flower opening: most of the points are on the subject, so the subject wins the
  vote and the shot reads as a pan. Nothing in the picture says which layer is the background.
- **The camera share still comes from the depth.** `compare` judges the PATH cosine only when the
  reference's camera share is 0.30 or more, and that share is still measured on the depth
  picture, with the weakness described below. It was left alone here.
- Everything under Limitations at the end still applies.

## The region reader: regions vote (2026-11-01)

The two diagnostics further down left a decision open, and this is the answer to it.
`motion.flow_series` now reads the camera in four steps:

1. **Find what can show motion.** A pixel counts when its corner strength clears 1% of the
   frame's strongest, a bar capped at an absolute value so one hard silhouette cannot silence
   a softer world.
2. **Vote by region.** Each cell of an 8×8 grid holding texture casts one vote: the median
   flow of its textured pixels. Fewer than 12 voting cells names no camera at all, which is a
   lone figure on a bare field.
3. **Read the stretch first.** It comes from opposite sides of the frame, where a shift
   cancels, and the shift is read after it.
4. **Refuse a stretch that does not earn the word.** A zoom has to stretch both axes the same
   way when both can be measured, most votes have to agree with it, and it has to leave less
   unexplained than a plain shift.

Exact buckets, rgb / blender / vda, the old all-pixel reader against this one on the same
frames:

| clips | all-pixel | regions |
|---|---|---|
| the original 20 (of 140) | 111 / 99 / 97 | 138 / 140 / 139 |
| 4 hard cases added while tuning (of 28) | 14 / 16 / 14 | 24 / 26 / 26 |
| **9 unseen clips, thresholds frozen first (of 63)** | **32 / 25 / 25** | **50 / 58 / 55** |

**The last row is the number to quote.** The first two were tuned on. Missed motion on the
unseen clips falls from 20 / 27 / 28 buckets to 4 / 2 / 2, and false motion stays 0 on every
row for both readers. On those clips translation is right on 59 / 61 / 60 of 63 and the radial
word on 54 / 60 / 58.

What the unseen clips still get wrong:

- **Long-lens pushes** read 3–5 of 7 and under-read speed. The speed ratio is 0.67–0.80 on the
  unseen clips against 0.92–0.98 on the tuned ones, so a tempo word can sit one band slow.
- **A few false zoom words beside a pan.** `holdout-truck-left-85mm` picks up "pulls out" on rgb,
  and `holdout-truck-right-subject-crosses` picks up "pushes in" on rgb and vda.
- **The truth is not spotless either.** It keeps the median-then-radial reduction, so on
  `holdout-push-in-horizon` it says "tilts up" on two buckets where the camera only pushes.

Each refusal in step 4, turned off, on the unseen clips (exact buckets, then false zoom buckets):

| | rgb | blender | vda |
|---|---|---|---|
| as built | 50, 4 | 58, 0 | 55, 2 |
| no agreement check | 46, 8 | 56, 2 | 52, 5 |
| no better-fit check | 50, 4 | 55, 4 | 51, 6 |
| neither | 43, 11 | 49, 10 | 47, 10 |

**A known limit.** With texture on one side of centre only, one axis has to speak alone, so a
truck whose left half slides uniformly faster than its right reads as a zoom. It is marked in
`motion.py`.

**Camera share moves too,** because it is read from the same shift. Real camera moves rise
(`pan-left-85mm` 0.09 → 0.85) and locked-off frames stay low: static 0.00–0.12, a box crossing
≤ 0.01, a figure approaching ≤ 0.05, the unseen near figure ≤ 0.26. Pushes and pulls stay ≤ 0.14,
since a zoom is left in the residual on purpose. `static-handheld` reads 0.48 / 0.22 / 0.34: that
camera truly shakes. The one other kit measured moves from under 0.01 to about 0.2, still counts as
subject motion, and reads static in words both ways.

## First result — the all-pixel reader (2026-10-26)

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
  and 5 buckets. The main report read it 6, 5 and 5 of 7, so the mask caused this: 94–97% of
  the textured pixels sit above centre (the boxes at the horizon), so the outward flow of a
  push drags the median up, and the median-then-radial split books a rising picture as a tilt
  down. An earlier version of this page had the texture below centre and the median going
  down, which the words themselves contradict; this was measured on 2026-11-01.
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

- **Both diagnostics stay in the calibration script.** Of the estimators compared here only the region reader went into
  `depthc`; the picture reader above replaced it as the first read on 2026-11-11.
- **It moves words and informational lines, not verdicts.** `compare`'s PATH numbers,
  `describe`'s camera words and every camera share line change for every kit, including a share
  that now crosses the line where PATH is judged or a clip counts as camera-led. The PATH line
  and the camera line still do not vote.
- **The real-take reading is no longer an inference.** When the region reader landed on
  2026-11-01 no real take had been scored. One has been since, and the reader failed: see the next
  section.

## Real footage: the first score (2026-11-10)

The clay score says the reader works on a scene built for it. This check asked whether it names
clean moves on real pictures. It does not do so reliably. It cost $0.

**The set.** 17 stock clips, each one 8 s window with no sound: 12 with one clear move (pan or
truck, tilt, push-in, pull-out, three of each) and 5 locked-off shots with people or a train moving
through. Plus 8 AI takes of one setup, 5 s each, where the camera pushes in on a man from a medium
shot to a close-up. The clips and takes are held outside the repo.

**The truth.** The clips were chosen by a person looking at frames, so that reading was not blind.
A video model then labelled every clip without seeing the key. The two agreed on 14 of 17 clips and
on all 8 takes; the owner watched the other three and ruled. All of that was written down, and the
pass mark with it, before any reader answer was read.

**The rules.** `depthc describe <clip> --json` on the default depth path, at the v0.16.0 reader
(`describe.py` and `motion.py` are unchanged since). A move clip is right when the reader's longest
moving segment holds the truth word. A locked-off clip is clean only when the reader names no move
at all, however short. Pass: at most 2 moves missed, and no invented move on a locked-off clip.

**The score: fail.**

| | Depth path (scored) | `--rgb` (information only) |
|---|---|---|
| Moves right | 9 of 12 | 10 of 12 |
| Locked-off clips with an invented move | 4 of 5 | 2 of 5 |

- **Invented moves.** On four of the five locked-off clips the depth path named pans, tilts or
  pulls while the background held still. On two the false move lasted 1.5 s in all; on the other
  two it ran for most of the clip. Passing people and a passing train are what those clips have in common.
- **Missed moves.** Two push-ins read as still for the whole clip: one very slow, one through a
  dark passage. A pull-out, the one clip played backward, was named for half a second and then read
  as still.
- **The two paths disagree.** `--rgb` was clean on two locked-off clips the depth path got wrong
  and found one of the missed push-ins and the missed pull-out, and it missed a push-in the depth
  path found. Neither is
  the better reader on this set.
- **The 8 takes.** Every take's main segment held "pushes in". Every one also said "tilts up"
  beside it, which neither watcher named. It is real (checked 2026-11-11): the takes push toward
  a face above the centre of the frame, the picture goes from chest to face, and an independent
  point tracker puts the centre of the push near the top edge. A push aimed above centre slides
  the picture down as it grows, and that is the word for it.

**What this means.** The camera words in `describe` and the PATH line in `compare` are a hint to
check by eye, most of all a move named on a shot that looks still while something moves through
the frame. They still do not vote in any verdict, and should not until a reader
passes on clips it has never seen.

**What it does not say.** The set was the easy case: a clip that could not be read by eye was left
out. It has no orbit, roll, handheld shake, whip pan or long lens. Messier footage was unmeasured
at this score and was expected to do no better; the second score below reports some.

**These clips are spent.** A reader changed after this score cannot be scored on the same 17 clips.
It needs fresh ones. The reader was changed on 2026-11-11 (see "The reader now" above) and scored on a
second, fresh set the same day.

## Second real-footage score: the picture reader (2026-11-11) — FAIL

**The set.** 40 clips the reader had never seen, each 8 seconds: stock clips picked by eye from
thumbnails, never with the reader, plus fixed windows cut from five videos the owner supplied
(three dance videos shot on a phone stand, two film scenes). The plan, the window rule and the
pass mark were written and hashed before the reader ran. Every clip was labelled by eye, then
by a second model that never saw those labels; the two agreed on all 20 locked-off clips and on 3
of 9 moves, and the owner ruled on the rest and on four clips promoted into the scored set.

**Scored set: 10 moves and 7 locked-off shots. Pass mark: at most 1 move missed, no move named
on a locked-off shot.**

| | result |
|---|---|
| moves named | 7 of 10 |
| locked-off shots with a move invented | 2 of 7 |
| verdict | **FAIL** |

The `--rgb` path gave 6 of 10 and the same 2 of 7.

- **Missed:** a slow zoom along a wall and a faint pull-out (played backwards from a push-in) got
  no move at all; a vertical move through trees was named a pan.
- **Invented:** a person walking away from close to the lens, and a flower opening in time-lapse.
  Both fill the frame, which is the limit named above.
- **The other five locked-off shots** (walkers, cars, horses, falling leaves) were all read as
  still. Small figures crossing a locked-off shot are the failure the first score found, and on
  these clips it did not recur.

**The messy clips, reported and not scored (23).** Of 15 more locked-off-style clips, a move was
named on 8: 7 of the 11 dance windows and 1 of 4 stock clips. The dancers take up much of a
vertical frame against a plain wall, so they outvote the background. Three film windows with
cuts came back as a string of different moves, one per shot or more. Handheld crowd shots were
named as tilts and pushes.

**What it does not say.** There was no clean tilt in the fresh stock clips, so the only vertical
move in the score is one promoted clip. On two clips the labellers disagreed on direction; the
count is 7 of 10 either way. These 40 clips are now spent as well.

## The camera round: shots, people, "can't tell", slow zooms (2026-11-11) — UNSCORED

Four changes to the picture reader, one for each way the second score failed. Every number in them
was chosen while looking at the 57 spent clips (the two sets above), so nothing in this section is
a score: that needs clips the reader has never seen. The camera words still do not vote in any
verdict. The constants and their ceilings are in `depthc/motion.py`, in the block headed "the
camera round".

1. **Cuts.** Three film windows came back as a string of moves, one per shot. Now the clip is
   split into shots first and each is read on its own; no fit, average or word crosses a cut. A
   gap is a cut when at most 5% of the tracked points survive the there-and-back check AND the
   grey histogram moves by 0.10 or more (0.30 alone when a frame has too few corners to judge).
   Either cue alone fires on real footage: a train keeps the histogram churning at 0.20 a frame,
   and a dark, smeared shot loses 90% of its points. A shot under a quarter of a second is not
   read. Our own detector, not PySceneDetect (BSD-3, allowed): the tracker already measures point
   survival, and survival is the cue that told a cut from the train; no dependency is added.
   PySceneDetect's adaptive detector is the named upgrade if dissolves and fades ever matter,
   because this one does not find them.
2. **People.** A dancer filling a vertical frame, or a walker close to the lens, has more points
   than the wall behind, and the fit followed them. Now, on macOS, Apple Vision's person instance
   mask is taken for each frame and the points on (or within 5 px of) a person get no vote. The
   instance mask outlines something even when nobody is there (it covered most of a flower and a
   forest on the seen clips), so an outline counts only where Vision's body detector also drew a
   box around it. Vision ships with macOS: no weights, reached through PyObjC (MIT, installed on
   macOS only). Without Vision, given a depth of the same frames (`compare`'s reference, a kit
   build), the nearer half of each depth frame is left out instead, reading "near" from the end
   the kit says (an inverted kit's dark end). With neither, every point votes, as before.
3. **"Can't tell".** A crowd was read as a move rather than as nothing. Now a gap is "can't tell"
   when the person mask leaves fewer than 40 trackable points, or when no single motion carries
   half of them, and a shot names no camera at all when fewer than half its gaps could be read.
   `describe` says `can't tell` and lists those stretches (`cameraUnread`); a kit's `cameraShots`
   gives such a shot a `null` word; `compare`'s PATH line says `can't tell` and leaves those
   buckets out of its cosine. Half is a judgement, not a tuned figure.
4. **Slow zooms.** A slow zoom along a wall and a faint pull-out got no word. Now the zoom is also
   read across half a second, one reading every eighth of a second, and where the frame-to-frame
   zoom is under the usual floor (0.05) that reading replaces it and is named from 0.012. Across
   half a second a dolly past things at different distances reads a fraction of its push (0.011
   against 0.137 on one seen clip), so where the frame-to-frame zoom clears 0.05 it stands. Beside
   a pan or tilt the zoom still needs 0.05: seen pans showed a zoom of their own of 0.01 to 0.05.

**The scout** counts people the same way, twice a second, and warns when two or more are in frame
for 10% of the clip or more, noting it on each window that has one. It counts people only and
never changes the ranking.

**On the spent clips (development signal, not a score).** Same clips, old reader against new,
with Vision on:

| | before | after |
|---|---|---|
| First set: moves named | 12/12 | 12/12 |
| First set: locked-off shots with a move invented | 1/5 | 0/5 |
| Second set, scored clips: moves named | 7/10 | 8/10 |
| Second set, scored clips: locked-off shots with a move invented | 2/7 | 2/7 |
| Second set, messy locked-off-style clips with a move named | 8/15 | 6/15 |
| Cuts found in the three film windows (nine, marked by eye) | not looked for | 9 of 9, none invented on any other clip |

Still wrong on the scored clips: the vertical move through trees is still named a pan; the slow
truck past a near bookshelf now reads "pulls out" for its first 2.5 seconds (its sideways travel
is under the pan floor, so only the zoom is left); wind on the grass keeps a move on one
locked-off shot once its walker is masked; and the flower is not a person, so nothing masks it.
Of the 6 messy clips still named, on one the background points the mask leaves move together
(95% of 230), so either the phone stand moved or the mask missed something; that is not settled
by eye. Without Vision (the CI path, no depth) the numbers are 10 of 12 and 1 of 5 on the first
set, 8 of 10 and 2 of 7 on the second, and 8 of 15 messy. Two further film scenes the owner
supplied came back with 7 and 8 cuts, all real by eye, and about three cuts missed in the last
stretch of fast action in one, which reads as "can't tell". With Vision on, a 25-second clip took
about 12 seconds to read on one Apple Silicon Mac.

**On clay** (the cached clips, $0):

| | `pic` before | `pic` after |
|---|---|---|
| Tuned clips, exact buckets | 162/168 | 160/168 |
| Tuned, missed motion / false motion | 0 / 0 | 0 / 1 |
| Held-out clips, exact buckets | 52/63 | 52/63 |
| Held-out, missed motion / false motion | 3 / 0 | 3 / 1 |

Both false-motion buckets are the first half-second of a slow zoom (one in, one out) whose true
zoom is 0.047, just under the 0.05 floor the truth is worded with; the new floor names it. The
other lost bucket is a push beside a truck that the beside-a-pan rule no longer names.

**What it cannot do.** Vision finds people only: a train, a car or a flower that fills the frame
still wins the vote. A macOS update can change what Vision finds. The depth fallback leaves out
half of every frame whatever is in it, and a subject no nearer than its background keeps its
points. A dissolve or a fade is one shot. A shot read on just over half its gaps can still be
worded from the wrong half.

## Limitations

- **Both readers failed on real footage.** The depth read failed the first set and the picture
  read that replaced it failed the second (7 of 10 moves, 2 of 7 locked-off shots). The camera
  round after that has not been scored on unseen clips. The clay
  figures in this file do not carry over to shots with people or vehicles moving through them.

- **One scene.** Clay ground, rows of boxes and sky. The truth counts the static world
  only, so a crossing subject is excluded from it by construction.
- **Cranes read as tilts by design.** Translation and rotation are not told apart: on
  screen, a crane and a tilt both slide the picture vertically. The truth uses the same
  words, so `crane-up` is scored against "tilts up".
- **A bucket is half a second.** A move shorter than that is outside what this
  calibration can score.
