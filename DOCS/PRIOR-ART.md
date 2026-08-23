# Prior art — what other people building this already know

Surveyed 2026-07-25, while adding the motion gate. The question was narrow: **has anyone
solved the parts we are guessing at**, and is there anything cheap worth taking?

Nobody is doing our exact thing — feeding depth to a *closed* video model through a generic
video-reference slot. The adjacent fields are (a) open depth-conditioned video models, where
depth goes into a trained ControlNet-style branch, and (b) video-dataset curation, where
millions of clips get filtered by motion before training. The second turned out to be the
useful one: curation pipelines have spent real effort on "does this clip carry usable
motion", which is precisely our gate.

---

## Taken

### 1. Motion scoring is a BAND, not a floor — Open-Sora / Open-Sora-Plan

Their curation filters clips from *both* ends: below a floor the clip is nearly static, above
a ceiling it is jitter and flicker rather than content. Open-Sora Plan reports the same shape
of finding — near-static below one bound, "significant jitter and flicker" above another.

**Applied:** `checks.py` grades motion in three bands rather than one. `JITTER_ENERGY = 12.0`
WARNs from the top end, which nothing in our checker previously did, even though our own
README already said "fast motion can flicker in the output".

Sources: [Open-Sora-Plan report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md),
[Open-Sora](https://github.com/hpcaitech/Open-Sora/releases)

### 2. Cut detection before anything else — PySceneDetect, and Open-Sora's jump-cut pass

Every curation pipeline splits on scene cuts *first*, because a clip spanning a cut is two
clips. For us a cut inside a 5 s reference is the worst possible version of the "one dominant
motion signal" problem.

**Applied, but not their way.** PySceneDetect's content detector works on RGB HSV deltas; we
only have depth at that point in the pipeline, and we did not want the dependency. What
transferred is the *principle they get right and our first attempt got wrong*: a cut is
distinguished by EXTENT, not magnitude. Our first rule flagged any frame gap 4× the clip
median and fired on all three real clips that contain no cut at all — a sword impact
legitimately spikes. Measuring the fraction of the frame that changes at once separates them
cleanly: worst legitimate frame 0.24, spliced cut 0.46.

Source: [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)

### 3. Optical-flow decomposition for camera vs subject

Camera-motion labelling is standard in video-dataset annotation. Splitting dense flow into a
global (median) component and a residual is the cheap version of it.

**Applied:** `motion.camera_share()`, reported in every kit and in the checker as "this is a
camera-path reference" or "a subject-motion reference". It immediately paid for itself:
CLIP D was described in its 001 kit as a "slow lateral dolly … a CAMERA-PATH reference" and
measures 3% whole-frame movement even at a one-second flow baseline. Its contact sheet
confirms a locked-off camera. The description was wrong and no one would have caught it.

---

## Deliberately not taken

### Per-frame depth normalisation — VACE

[VACE](https://github.com/ali-vilab/VACE) is the most widely used depth-control video
pipeline, and its depth annotator normalises **per frame**: `depth -= depth.min(); depth /=
depth.max()`, with no percentile clipping and no temporal consistency handling at all
(`vace/annotators/depth.py`).

We normalise once over the whole clip, which is upstream VDA's logic and the reason our output
does not pump. Adopting the common approach would inject brightness flicker straight into the
motion channel — frame-to-frame change that is not motion, which is exactly what our gate
measures. **Keeping ours, and the fact that the popular pipeline does the opposite is worth
knowing when comparing outputs with anyone else.**

One honest consequence: because no shipped pipeline does percentile clipping, our
`--variants contrast` ref is genuinely experimental rather than best practice. That is why it
is a variant to be tested rather than the default.

### LPIPS-based motion scoring — Open-Sora

They compute inter-frame *semantic* similarity with LPIPS and take the mean as a motion score.
It is more robust than pixel differencing to lighting change. It is also a neural network per
frame pair, on a machine where we deliberately ship one small Apache-2.0 model. Our inputs are
already depth maps, which have no lighting to be robust to. **Not worth the second model.**

### Subject counting

Tried and cut, in this repo, on this data. Connected components on the frame difference to
count how many things are moving: CLIP C's clean single-walker window measured 3–6 "movers"
against 3–5 for the window with three figures and a dog. It counts limbs, not people. Doing it
properly needs person segmentation — a second model and a second licence question.

The gap is real and now documented where it bites: the window scorer maximises *how much*
motion and is structurally blind to *how many things* are moving, so looking at `sheet.png`
before firing is a required step, not a nicety.

---

## Still unanswered by anyone

- **Whether a depth ref beats a text-only prompt** on a closed model with a generic video-ref
  slot. Nobody publishes this because nobody else is doing it this way. Only a live-fire
  answers it — see `LIVE-FIRE.md`.
- **What Seedance actually does with a reference shorter than the requested output.** The
  adapter rounds duration to whole seconds; what fills the gap is undocumented. Snapping trims
  to the grid sidesteps the question rather than answering it.
- **Whether polarity matters** (near-bright vs far-bright). No pipeline we found treats it as
  a choice, because a trained ControlNet has a fixed convention. A generic video-ref slot has
  none, so it is an open question and a `--variants` arm.

---

## CozyClay (NomaDamas) — surveyed 2026-08-23

A browser 3D staging studio (Three.js/R3F + an MCP server): block a scene with clay
mannequins, fly a camera, and it derives film vocabulary from the *known geometry* —
the geometry-forward twin of our pixels-backward pipeline. AGPL-3.0, so ideas were
re-derived in our own units; no code was copied.

### Taken

- **A tempo word beside the path label.** Their camera-move classifier grades slow/fast
  from physical lens travel in m/s. Re-derived here as time-to-cross-the-frame bands
  (`PATH_SLOW`/`PATH_FAST` in `motion.py`); `describe` now says "quickly pans left
  0–2s, then pushes in". Heuristic, uncalibrated — marked as such at the constants.
- **"then"-chained camera runs** in the describe sentence, their `moveSequencePhrase`
  shape, replacing the semicolon join.
- **Provenance tags on prompt rules.** Their ARDY prompt guide labels every line
  `[Upstream fact]` vs `[CozyClay policy]` vs `[pipeline observation]`. Applied to the
  kit template's GUIDANCE section as `[measured]` / `[community]` / `[reasoned]`, and
  the measured 720p-or-4K resolution law now appears in the template and kit README
  (it previously lived only in CLAUDE.md, invisible at fire time).

- **Framing vocabulary from the depth silhouette** (taken 2026-08-23, same day, after
  Peter's call to close the loose end). `read_framing` sizes the nearest iso-depth
  figure and names it on a five-rung ladder; everything tighter than "medium" collapses
  into one honest "tight" rung because an in-frame silhouette saturates at 1.0.
  Validated on six CC clips from Wikimedia Commons — the first pass failed 3/6 and the
  fixes were MEASURED, not guessed: open water ramped +0.25 while true figures sat
  within ±0.08 (→ the FRAMING_MAX_RAMP surface gate), and a static near sliver got
  sized while the dancers moved in the midground (→ the motion-hotspot overlap gate).
  Final score: 5/6 correct-or-honestly-silent; the sixth sizes a foreground SIGN at 68%
  (depth cannot know person from object — documented, by-eye caveat stands). What depth
  alone can never fix: an insert and a landscape share the same relative-depth
  structure, so a frame-filling near surface that slips the gates reads as a wide.

### Deferred (owed ledger)

- **Depth-parallax move disambiguation.** From RGB flow alone, pan-vs-truck and orbit
  are unnameable (`path_label` says so by construction). We hold the one signal that
  could split them from pixels: the parallax gradient across depth bands. Uncalibrated
  research with no current consumer; recorded so the seam is not lost. (Closed
  2026-08-23 as won't-do by Peter's call — no consumer; this note is the preserved
  idea.)

### Deliberately not taken

- **Nearest-prime lens snapping for OSIDE.** Their insight (models saw more "35mm"
  captions than "47mm") is already superseded there: OSIDE quantises to a real focal
  set (8/14/35/50/75mm) and the 2.5 dialect deliberately ships FOV anchors instead of
  mm — a measured dialect decision this would regress.
- **The geometric move classifier's thresholds** (DOLLY_M, ORBIT_DEG…). They need 3D
  positions we do not have; only the tempo idea transfers.
- **Their Seedance 2.0 provider facts** — corroboration only, 2.0-era, third-party:
  no-seed matches our seed-does-not-pin law; "reference-video and first/last-frame
  cannot be mixed" is *their validator hard-erroring* where we measured a *silent
  drop* in OSIDE's adapter — possibly an Ark-direct vs Composer surface difference;
  worth one line in any future parity pass, not a fire.
