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
Since 2026-09-06 a fire entry in a kit's `result.json` also carries `generationId` — the
OSIDE generation it became, given as `--generation <id>` to `depthc compare` / `depthc
result` (PLAN-SHARED-GENERATION-ID step d); every entry before that date is unkeyed.

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
| Sword-duel_HD_001 | ANNA + WILLIAM | **1080p** (1080p ref) | $2.29 | **ignored** | **+22%** | r = +0.107 |

Total spend ~$11.00. Per-kit detail lives in each kit's `result.json`.

**[2026-08-24] Those per-kit `result.json` files no longer exist**, nor do the kits, nor the
four CLIP A-D calibration clips `PLAN.md` §1a says live in `Processed/`. `Outbox/`,
`Processed/` and `Inbox/` hold only their README. No Time Machine destination is configured
and `~/.Trash` is empty, so there is no restore. The findings survive — they are in this file
— but the media and the per-fire metadata (which VARIANT was fired, model id, output URL,
timestamps) are gone. Do not spend an hour looking for them. The cause is a scope collision,
not carelessness: "never commit media" is a rights law that also swallowed the text RECORDS,
because `.gitignore` excludes `Outbox/*` wholesale.

### Results — 2026-08-24 · Seedance 2.5, native 1080p vs 720p, same depth reference

Kit `Probe1080p_001` — Pexels 2084066 (free commercial licence), 13-18s, 5.00s ref,
4.53 mean depth-space energy, 42% depth retention, 3% whole-frame. Both arms identical
except the output tier. Fired through `scripts/livefire-25-1080p-depthref.ts`.

| Arm | Res | Cost | Verdict | Energy vs ref | Adherence | Regions direct/mirror |
|---|---|---|---|---|---|---|
| Probe1080p_001 | **1080p** | $2.45 | **adhered** | **−7%** | r = **+0.820** | +0.877 / +0.189 |
| Probe1080p_001 | 720p (control) | $1.38 | **adhered** | +11% | r = +0.716 | +0.851 / +0.229 |

Total settled $3.83. Lag controls on the 1080p arm: −1.0s −0.157 · −0.5s −0.033 ·
**0.0s +0.820** · +0.5s −0.131 · +1.0s +0.305 — peaks at zero and collapses either side,
which is what makes it an alignment rather than two busy clips.

**THE 0/3 DOES NOT TRANSFER TO 2.5.** 2.0's native 1080p ran +22-32% excess energy with
r ≤ +0.107; 2.5's separately-wired 1080p lane (switched on 2026-08-17) ran −7% at r = +0.820
on the same kind of reference. The control confirms the clip itself behaves on the known-good
tier, so the result is about the LANE and not the subject.

**LIMIT ON THIS RESULT — read it before quoting it.** ONE GENTLE REFERENCE SURVIVED. The ref
is a lateral walk whose energy is substantially background parallax; the 0/3 was a sword duel
— fast, multi-limb, with props that dissolved. A demanding reference on 2.5's 1080p is still
unfired, so this is suggestive, not an all-clear. A FAIL here would have been conclusive; a
pass is not.

**BILLING, MEASURED.** Both arms billed exactly 2.008× the estimate — 241 frames where
`24 x seconds` predicts 120. Ark's "input WITH video" rate column bills **input seconds PLUS
output seconds**, and this ref was the same length as the render. OSIDE's `arkTokens`
(`lib/fal/adapters/arkSeedance.ts:442`) doubles only for `arkTask === 'edit'`, so any 2.5
render carrying a video reference is under-quoted — by 2x when ref length matches output
length. Approved ceiling was $1.91; settled $3.83. Recorded here because the estimate was
wrong for a documented reason, not because the fire was.

### Results — 2026-09-07 · the only depth kit ever fired through OSIDE's own door

| Kit | Surface | Res | Cost | Verdict | What followed | What did not |
|---|---|---|---|---|---|---|
| Sha256Proof_001 | **oside** | 480p, 4 s | $0.39 | **partial** | walk, direction, profile framing | the camera tracking |

Generation `441fe671-0bd9-4e62-80ce-a0de60d86d8e`, fired 2026-09-07 01:33 UTC, Seedance 2.0
480p 4 s, Anime Style Constant. Reference: a man walking L→R in profile past a storefront,
camera tracking so he stays mid-frame. Generated: a figure walking L→R in profile across a
**static** frame under a lamp.

**The metric failed here, not the motion.** `compare` read the energy flat — ratio 0.096,
r0 = −0.423 — because flat cel shading gives the depth pass no gradient to measure. Read that
as a limit of the instrument on stylised output, never as a verdict on adherence; the watch
says three of four things were followed.

**Why this row exists at all (2026-09-15).** It lived only in `Outbox/Sha256Proof_001/result.json`
and in no tracked document — the same shape as the 2026-08-24 loss recorded above, where
`Outbox/*` swallowed every 2026-07-25 kit's `result.json` and only the findings in THIS file
survived. The kit was nearly deleted as a superseded `_001` draft before anyone noticed it was
the sole record of a paid fire. The lesson the repo already paid for once: a fire is not
recorded until it is in this file. `result.json` is the instrument's output, not the archive.

**AND THE MEDIA DID NOT SURVIVE (ow-4ccd14).** "Nearly deleted" is true of the RECORD only.
The delete took the folder; the restore returned `kit.json` and `result.json` — the two files
`.gitignore` was amended to track on 2026-09-06 — and nothing else, because Outbox media is
never tracked. **`Sha256Proof_001`'s `depth.mp4`, `source.mp4`, `still.png`, `sheet.png`,
`prompt.txt`, `binding.txt` and `README.txt` are gone and cannot be recovered**: no Time
Machine destination is configured, `~/.Trash` is empty, Spotlight finds only the folder, and
a byte-exact sweep of `$HOME` for the recorded 922,046-byte clip found nothing. `Inbox/` and
`Processed/` hold only their READMEs, so the SOURCE clip went with it. The kit that carried
the only depth fire ever put through OSIDE's own door is now a husk, and `Outbox/INDEX.txt`
says so on its face (`MEDIA GONE — record only`).

What that costs, said plainly. The FINDINGS survive — they are the three paragraphs above,
and the measured manifest (4.54 energy, 42% retention, 3% whole-frame, the tier and every
check) is still in `kit.json`. What is gone is the ABILITY TO RE-MEASURE: the fired reference
cannot be re-compared, re-cut, or fired again on another tier, and generation `441fe671`'s
content key `2a134ac8…` now names bytes nobody holds. **A rebuild would be a NEW kit, not this
one** — `kit.json` names Pexels 2084066 (free commercial licence) and source SHA
`c91bf578…`, but a re-download that Pexels re-encoded at any point produces different bytes,
a different depth clip and a different `depth.sha256`, so it could not carry this fire's join
and must never be given this kit's name.

The husk STAYS, and the folder is not tidied away. Two gates read it as their subject:
`tests/test_kit.py::test_every_tracked_result_json_validates` fails on an empty DC-4 evidence
set ("no tracked `Outbox/*/result.json`"), and OSIDE's `check-depth-kit-import` D4 fails its
own "the subject set is real" assertion at zero kits. Deleting the husk would turn both red
and buy nothing back.

## THE RESOLUTION LAW — shoot 720p or 4K, never 1080p — **ON SEEDANCE 2.0**

**1080p is 0 for 3. Every other tier is 4 for 4.** (Fisher exact p ≈ 0.029.) This is the
single most operationally important thing these runs found, and it cost $6.87 to learn.

**[SCOPED 2026-08-24] This law is about SEEDANCE 2.0, and about 2.0's NATIVE 1080p lane.**
It was measured 2026-07-25, before 2.5 was on this account at all — the results table above
it has no model column because there was only one model. It does NOT carry to 2.5, and the
2026-08-24 fire above is the measurement that says so (−7% at r = +0.820). Two further
consequences on 2.5: there is no 4K rung at all, so "master at 4K" names a tier that does
not exist there; and a start frame no longer discards reference arrays. Both are handled by
`--dialect seedance-2.5`, which selects the constraint record and the kit prose together.

**The HD-ref experiment (2026-07-25 evening) killed the two remaining alternative
explanations.** A 1920×1080 depth ref of the identical window (energy 4.39 vs the 720p
ref's 4.40) was fired at 1080p output, with the prompt explicitly saying "fighting with
swords": r = +0.107 at zero lag with no peak at any lag, energy +22%, invented camera
movement (share 0.27 vs the ref's 0.17). The swords appeared in the opening — the prompt's
doing — then shrank to a dagger by 2 s and were gone by 3.5 s, the identical dissolution
signature. So the failure is NOT ref-vs-output resolution mismatch and NOT prompt
vagueness. Three seeds, two ref resolutions, two prompt phrasings: **the native 1080p lane
itself mistreats video references.**

**Why "4K works" does not contradict this** (research, 2026-07-25): the Seedance 1.0
technical report documents the family as a cascade — base generation plus a diffusion
refiner for higher tiers — and no 2.0 report exists. On Ark's price card 1080p carries a
per-token SURCHARGE over 720p (+10%) while 4K is ~43% CHEAPER per token; that pricing shape
fits 4K being super-resolution over a lower-res base pass. If so, "4K adhered" means "the
base lane adhered again, then got upscaled" — 4K never disproved a resolution effect, it
routed around the broken lane. Treat 4K as the premium master path and 720p as the
iteration path; there is no reason left to buy 1080p for depth-motion work.

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

**How adherence was measured.** (Since v0.5.0 this is a command — `depthc compare <kit>
<generated.mp4>` — with the ratio-first ordering built in.) The generated video was run back
through this same converter and its depth-space motion profile correlated against the
reference's, bucket for bucket. The
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
