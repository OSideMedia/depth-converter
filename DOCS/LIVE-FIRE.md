# Phase 4 — Live-fire verification protocol

**STATUS: awaiting Peter's budget-approved run. Not to be executed autonomously.**

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
4. Paste the kit's **Variant A binding phrase** into the video ref's binding field.
   **Never** paste the Variant B standalone block into OSIDE — the Composer compiles its
   own ACTIVE REFERENCES block, and two competing blocks is the exact failure mode
   `refManifest.ts` exists to prevent.
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

### Results

_(empty — awaiting the run)_

| Date | Clip | Res | Output len | Cost | Motion | Camera | Timing | Notes |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

## Gate 4

One verified motion-transfer generation, with the source and the result archived
side by side.
