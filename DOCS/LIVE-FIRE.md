# Phase 4 — Live-fire verification protocol

**STATUS: RUN 2026-07-25. GATE 4 PASSED — the technique works.** Two generations adhered
with a measured motion correlation of **r = +0.950** and **r = +0.916** at zero lag. Results
and corrections are at the bottom; the protocol below stands for future runs, with the
binding-field step corrected.

**Not to be executed autonomously.**

**LAW (mirrors OSIDE): nothing fires against a paid API automatically.** This document is a
protocol a human runs or supervises. Low-res first, always.

## What this phase does and does not test

The `reference_video` plumbing is **already proven**. OSIDE CHANGELOG v0.39.1 (2026-07-18,
Plan 9 Phase 7 sweep) fired three `@video`-chained shots successfully: chain handoff was
frame-near-exact, the anime cel lock held while chaining, and with-video billing was
confirmed.

So this phase tests **the depth-as-motion technique only**. A failure here points at the
technique or the ref weighting — never at the adapter. Do not go debugging `arkSeedance.ts`
if a generation comes back wrong.

## Before you start

- [ ] Budget approved for roughly **$0.50–$2.00** across the runs below.
- [ ] A kit built and checked: `depthc kit "<clip>" --trim <in> <out> --tags "<what it is>"`.
      Its checker must report PASS or WARN. **Do not live-fire a FAIL.**
- [ ] Read the kit's `prompt.txt`. Note its `suggestedOutputSeconds`.

## Identity references — use WILLIAM or ANNA

Both are already registered and approved in the Ark asset-library likeness system. Use them
via their `asset://` refs or crowned stills in OSIDE, **not** ad-hoc AI images.

Two reasons: they are pre-cleared against the real-face input filter, and they give a
known-good identity baseline — so any drift observed is attributable to the depth ref's
weighting rather than to an unfamiliar identity reference.

## The runs

### 1. Baseline at 480p

1. Pick a 5–8 s single-action reference clip. CLIP A, B, or D all qualify.
2. Upload `depth.mp4` through OSIDE's upload route (→ `references` bucket) or attach it
   directly in Composer.
3. Add WILLIAM and/or ANNA as identity `@Image` refs.
4. **Do not paste any binding phrase.** [CORRECTED 2026-07-25, from the live run.] An
   *uploaded* video reference has **no binding field** in the Composer — bindings are
   auto-generated only for entity refs (characters, elements), and an upload is serialized
   as `{url, role:'video'}` with nothing else. The compiler already writes its own line
   from `refManifest.ts:117`: *"@Video 1 is a motion and camera reference — take movement,
   action and pacing from it."* Pasting Variant A into the prompt body on top of that is
   two instructions for one reference — the same collision the two-variant design exists to
   prevent, arriving by a different road. Variant B remains for the Ark console only.

   If you want the appearance-stripping clause the compiler's generic line lacks, paste
   **only** its second half: *"it carries structure only: take no color, texture, clothing,
   face, lighting, or environment from it."*
5. **Go straight to "make video". Skip the create-frame step.** A depth-motion shot is a
   refs-only shot: the adapter sets `refsWithStartFrame:false`, so a frame-first shot
   silently drops the depth ref and you will have paid to test nothing.
6. Set output duration ≈ the kit's `suggestedOutputSeconds`.
7. Generate at **480p**. With-video billing is the cheaper token rate; expect roughly
   $0.10–$0.40 at 480p/5 s based on the app's existing ledger scale.

**Evaluate:** does the body motion track the source? Does the camera path track? Does the
timing track, beat for beat? Did any appearance leak from the depth ref (it should be
impossible — there are no source pixels in it)?

### 2. Confirmation at 720p

One run at 720p with the same inputs. Confirms the technique holds at delivery resolution.

### 3. Short-ref / long-output probe

Deliberately request an output noticeably longer than the ref (e.g. a 4 s ref driving a
12 s generation). **Record what the model does with the uncovered tail** — this is currently
undefined behavior and the answer determines how hard `suggestedOutputSeconds` should be
enforced rather than suggested.

### 4. Optional, budget-approved: 4K

One 4K generation to test whether adherence holds at 4K with a video ref. Supported by the
standard model on paper, untested, and several× the cost. Upscaling a 720p/1080p master
(already in Peter's stack) is usually the cheaper path to 4K.

## Also on this checklist — the console doc check

The Seedance constraint numbers in `checks.py` were triangulated from fal's schema, OSIDE's
Ark adapter, and several guides. The official BytePlus pages are JS-rendered and could not
be read directly.

- [ ] Open <https://docs.byteplus.com/en/docs/ModelArk/1520757> in a browser and confirm:
      video refs per generation, combined and per-clip duration bounds, max file size
      (**50 MB vs 200 MB is contested** — `checks.py` enforces the stricter 50), the
      resolution window, and accepted frame rates.
- [ ] Reconcile `lint.ts maxMixedInputs: 12` against the adapter's `total: 15`.
- [ ] Anything the docs contradict: loosen `checks.py` from FAIL to WARN rather than
      leaving a hard failure standing on inference.

## Recording results

Append below. What adhered, what drifted, what prompt wording changed, and the cost.

### Results — 2026-07-25

| Kit | Cast | Res | Cost | Verdict | Energy vs ref | Adherence |
|---|---|---|---|---|---|---|
| Prone-crawl_002 | ANNA | 720p | $0.00 | **rejected** | — | blocked at input moderation |
| Sword-duel_002 | ANNA | 720p | $0.93 | **adhered** | +8% | r = **+0.950** |
| Sword-duel_002 | ANNA + WILLIAM | 720p | $0.93 | **adhered** | +8% | r = **+0.916** |
| Sword-duel_002 | + custom location | 720p | $0.93 | **adhered** | +8% | r = **+0.926** |
| Sword-duel_002 | ANNA + WILLIAM | **1080p** | $2.29 | **partial** | **+32%** | r = +0.721 |
| Sword-duel_002 | ANNA + WILLIAM | **1080p** | $2.29 | **ignored** | **+31%** | r = **−0.056** |
| Sword-duel_002 | ANNA + WILLIAM | **4K** | — | **adhered** | +6% | r = +0.747 |

Total spend ~$8.70. Per-kit detail lives in each kit's `result.json`.

## THE RESOLUTION LAW — shoot 720p or 4K, never 1080p

**1080p is 0 for 2. Every other tier is 4 for 4.** This is the single most operationally
important thing the run found, and it cost $4.58 to learn.

Both 1080p generations, on independent seeds (`2133440621`, `197983437`), came back carrying
**+31% and +32% more motion than the reference asked for**, with the props failing — swords
dissolving from ~3 s in the first, absent entirely in the second, replaced by bare fists. The
second ignored the reference outright at r = −0.056 and invented its own shot: a push-in from
a wide two-shot to tight close-ups that a near-static reference never suggests.

Meanwhile 720p (three times) and 4K held their amplitude to **+6–8%** and kept the swords.

4K matters here because it kills the obvious explanation. If the mechanism were the reference
being coarser than the output grid, 4K — a 720p reference driving nine times the pixels —
would be the worst case. It was the best. Whatever goes wrong is inside Ark's 1080p lane, it
is not a resolution gradient, and it is not visible from the client side.

**A single 1080p sample was misread as seed variance before the second one landed.** The tell
was already in the data: both 1080p runs shared the +31/32% energy signature, which is a
fingerprint, not a draw. Two samples of a failure that looks identical twice is worth more
than one sample of anything.

## The metric lesson — amplitude beats correlation for predicting artifacts

Correlation and the eye disagreed, and **the eye was right.**

| | energy vs ref | correlation | looked |
|---|---|---|---|
| 720p | +8% | +0.926 | good |
| 4K | +6% | +0.747 | good, arguably best |
| 1080p | +32% | +0.721 | ghosting, props gone |

Ranked by correlation, 4K (0.747) sits barely above the broken 1080p (0.721). Ranked by
energy ratio, 4K and 720p are together and 1080p is the obvious outlier — which is what a
human watching them reported without seeing either number.

The reason: **correlation measures timing, and over-following destroys pictures.** A blade is
a thin high-frequency structure; it survives a lunge and does not survive a lunge rendered
50% faster than it should be. Small phase wobbles that nobody perceives punish correlation
hard, while excess amplitude — which nothing in the correlation notices — is what smears
bodies and dissolves props.

So any adherence report must lead with **energy ratio vs the reference**, and treat
correlation as the timing companion. That ordering would have flagged the first 1080p as bad
before it was watched.

**How adherence was measured.** The generated video was run back through this same converter
and its depth-space motion profile correlated against the reference's, bucket for bucket. The
figure is the Pearson r of the two energy envelopes at zero lag, with ±0.5 s and ±1.0 s lags
computed as controls. On the two-character run the sparklines came out identical bar for bar —
`▆▆▆▆█▆██▆▆` on both sides — and camera share matched at 0.16 against 0.17.

    lag −1.0s  r = −0.306
    lag −0.5s  r = −0.143
    lag  0.0s  r = +0.916   ← in phase, no delay
    lag +0.5s  r = −0.018
    lag +1.0s  r = +0.135

The collapse to ~0 on either side of zero is what makes it a real alignment rather than two
clips that happen to be busy. **Timing transfers beat for beat, with no delay** — the precise
opposite of the "started very late" complaint that motivated the motion gate.

### What else the run established

- **Props survive.** The swords came through as swords in both generations. That is the thing
  depth keeps and a pose skeleton would discard, and it is the strongest argument for this
  channel over a skeleton one.
- **Appearance comes from the image refs, as designed.** Source was a stone hall; the
  generations were a bright room and a Japanese dojo, both from the text. Zero source pixels
  leaked, which is structural — there are none in a depth map.
- **A location reference rides alongside without a fight.** A later run carried two identity
  refs + a custom location + the depth video, and all four channels held.
- **The video ref OUTRANKS text locks, consistently — three for three.** Given one identity
  against a two-body ref, the model cloned that identity rather than inventing a stranger, in
  direct defiance of an explicit `Do not generate duplicate avatars or a twin effect` lock.
  With two identities attached, the fighters crossed sides despite a `characters keep their
  sides of the frame` anchor. **Treat the depth ref as the blocking; text locks about count
  and position are advisory at best.**
- **Attach one identity per body in the ref.** The count comes from the ref and cannot be
  argued down in the prompt.
- **Cost is roughly 3× the pre-run estimate**: $0.93 for 5 s at 720p with a video reference,
  against the $0.10–0.40 guessed above for 480p. Budget accordingly.

### The moderation boundary — real, and narrower than feared

`Prone-crawl_002` was refused before generation: `[content-policy] InputVideoSensitiveContent
Detected`. The sword duel, from the same tool and the same pipeline, sailed through twice.

So **human depth maps are not categorically blocked.** The likely trigger is specific: depth
strips clothing, so every figure is a smooth featureless body-shape, and that clip's last
2.5 s is two of them prone, overlapping and blown to near-white filling the frame. Upright,
separated, clothed-reading figures pass.

Working constraint until more data: **avoid prone, intertwined, or close-to-camera bodies.**
A rejection costs nothing, so probing the boundary is cheap.

### One failure that was not ours

Two runs were blocked with `[real-person]` on an image reference, which looked like the
technique failing and was not. OSIDE's supporting-cast path rode the unregistered combined
character sheet instead of the registered `asset://` crops, so **every** two-character
Seedance shot was rejected — including plain text prompts with no video reference at all.
Fixed in `oside-studio` v0.66.3. Recorded here because it cost an hour of this session and
would otherwise read as a depth failure in the log.

## Gate 4

**PASSED 2026-07-25.** Two verified motion-transfer generations, with the reference and the
result measured side by side rather than eyeballed, and the source archived in the kit.
