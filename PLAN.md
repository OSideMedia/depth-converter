# Depth Converter — Build Plan

**For:** the building Claude model (Opus 5 as of 2026-07-24; originally targeted Opus 4.8 —
nothing in this plan is model-specific), building autonomously phase-by-phase in this
folder at default/high effort. Self-contained: no prior session context is required
beyond this file; all external facts were tool-verified on the dates marked.
**Date:** 2026-07-23. Researched and planned by Fable 5; all constraints below were verified this day.
**Amended 2026-07-24** after a pre-build verification pass against the real machine and the
OSIDE codebase (marked **[AMENDED 2026-07-24]** throughout). Load-bearing changes: the
x86_64-toolchain blocker fix in Phase 0, prompt.txt's two-variant law, the still.png
real-face flag, checker tier corrections, and the test-assets section (§1a).
**What this is:** a local macOS tool that converts a short reference video (MP4/MOV) into a
temporally-consistent grayscale **depth video** plus a ready-to-fire **Seedance Kit**, so a
filmmaker can copy real-world choreography and camera motion onto AI characters/scenes via
Seedance 2.0's video-reference channel.

---

## 0. Context you must not re-litigate (verified research findings)

### Why this tool exists
Seedance 2.0 accepts up to **9 images + 3 videos + 3 audio refs (12 files max)**, addressed in
the prompt as `@Image N` / `@Video N` / `@Audio N`. A video ref acts as a **motion + camera
channel**: the model reproduces choreography, walk cadence, dolly/orbit/whip-pan camera paths,
and timing from it. Feeding the **depth version** of a real clip instead of the raw clip:

- strips appearance/identity/style so nothing bleeds from the source into the output;
- preserves what pose skeletons throw away: **camera path, props, scene geometry** (a sword,
  a market stall, a crane move all survive in depth);
- avoids Seedance's strict real-human-face input filter that frequently rejects raw footage
  (this face-filter benefit is community-inferred, not documented — treat as likely, not law).

The reliable envelope (from hands-on community testing, not marketing): **3–8 s single-signal
clips** follow best; refs over ~10 s lose adherence; mixed-signal refs (camera move + action +
lighting all at once) get followed randomly; fast motion can flicker/distort in output. The
tool should surface this guidance, not pretend the technique is magic.

### The model choice (decided — do not substitute)
Use **Video Depth Anything — Small (VDA-Small)**, not per-frame Depth Anything V2.

- Repo: https://github.com/DepthAnything/Video-Depth-Anything (CVPR 2025, arXiv:2501.12375)
- Checkpoint: `video_depth_anything_vits.pth` (116 MB) from HF repo
  `depth-anything/Video-Depth-Anything-Small`
- Why: purpose-trained temporal attention head over the same DINOv2 encoder; windowed
  inference (32-frame windows, 10-frame overlap, cross-window scale/shift alignment) and
  **per-video global normalization** — flicker-free by construction. Per-frame DA-V2 +
  smoothing fixes brightness flicker but not depth-scale drift; VDA fixes both.
- **LICENSE LAW:** across DA-V2, VDA, and DA3, **only the Small models are Apache-2.0.
  Every Base/Large/Giant checkpoint is CC-BY-NC-4.0 (non-commercial).** Peter makes
  commercial films. Ship VDA-Small only. Do not add a model-size dropdown that downloads
  NC-licensed weights. Include the Apache license/attribution notice in the repo.

### Target machine (verified locally)
Apple M4 Pro, 24 GB unified memory, macOS (Darwin 25.5), Python 3.11.9, ffmpeg 8.1.2 at
`/usr/local/bin/ffmpeg`, `uv` installed. PyTorch MPS is the acceleration path. Expect a
10 s / 24 fps clip (~240 frames) to convert in roughly 1–5 minutes — fine for the use case.

**[AMENDED 2026-07-24] TOOLCHAIN BLOCKER — the installed toolchain is x86_64.** Verified:
Homebrew lives at `/usr/local` (Intel prefix); `uv` 0.11.28 is an x86_64 Mach-O; every
Python that uv can see is tagged `macos-x86_64` (including framework 3.11.9); ffmpeg 8.1.2
is x86_64 (runs under Rosetta). Meanwhile **PyTorch ships arm64-only macOS wheels**
(checked PyPI 2026-07-24: torch 2.13.0, sole macOS tag `macosx_14_0_arm64`). An x86_64 uv
resolves x86_64 wheels, finds none, and `uv add torch` fails outright — MPS never enters
the picture. Gate 0 cannot pass without fixing this.

> **[CORRECTED 2026-07-24, build session] Root cause is uv, not the interpreter.** The
> framework Python 3.11.9 is in fact a **universal binary** (`x86_64` + `arm64`) and
> reports `platform.machine() == 'arm64'` when run natively. It only *appears* as
> `macos-x86_64` because an x86_64 uv tags every interpreter it discovers with its own
> arch. Confirmed by installing the arm64 uv: the same framework Python is immediately
> re-listed as `cpython-3.11.9-macos-aarch64-none`. Consequence for Gate 0: asserting
> `platform.machine() == 'arm64'` is **not sufficient** — a universal Python passes that
> check while still having no torch installed. `smoke.py` therefore asserts on
> `import torch` succeeding (arm64-only wheels make that the real proof) plus
> `torch>=2.5` and MPS availability.

**Phase 0 therefore MUST:**
1. **[AMENDED 2026-07-24, Peter's call] Install the arm64 `uv` PROJECT-LOCALLY into
   `Depth Converter/.tools/`, not into `~/.local/bin`.** The plan's original global-shadow
   approach would have switched uv's architecture for the whole machine, and two unrelated
   projects already depend on the x86_64 one (`last30days-skill`, which has a committed
   `uv.lock` and an x86_64 `.venv`, and `oside-resolve-mcp`). Project-local install has
   zero blast radius: `setup.sh`, tests, and the launcher apps all invoke `./.tools/uv` by
   absolute path, and `rm -rf .tools/` fully undoes it. The installer prints a harmless
   "commands are shadowed by other commands in your PATH" warning — expected, since we
   never rely on PATH resolution for uv.
2. Have uv fetch/use an **arm64** CPython (`uv python install` a `macos-aarch64` build) —
   never the x86_64 framework 3.11.9. Gate 0 asserts `platform.machine() == 'arm64'`.
3. RECOMMENDED (not gate-blocking): drop a static arm64 ffmpeg binary into `~/.local/bin`
   (same shadowing, same reversibility). The Rosetta ffmpeg works, just ~2× slower encodes;
   `video_io.py` resolves ffmpeg via PATH so the swap is transparent.
4. The Phase 3 launcher apps need `arch -arm64` in their exec lines — the same trap the
   OSIDE Studio.app pair hit. Do not rediscover this in Phase 3.

### Known MPS/port pitfalls (all verified by reading VDA source — handle in Phase 0/1)
1. VDA's `run.py` hardcodes `DEVICE = 'cuda' ... else 'cpu'` — **no MPS branch**. Do not
   patch their files; import their model classes and manage device selection in our wrapper.
2. `infer_video_depth` uses `torch.autocast(device_type=device)`. MPS autocast needs
   **PyTorch ≥ 2.5** — pin `torch>=2.5` and fp16 works; keep a `--fp32` escape hatch.
   > **[CORRECTED 2026-07-24, Phase 1] fp16 DOES NOT WORK. The escape hatch is now the
   > default.** Measured on the target machine (M4 Pro, torch 2.13.0), VDA-Small under
   > `torch.autocast('mps')`: fp32 → clean, 44 ms/frame; **fp16 → ALL NaN**, 49 ms/frame;
   > **bf16 → ALL NaN**, 48 ms/frame. Reduced precision is not a speed/quality trade here,
   > it is broken *and slower*. So `--precision fp32` is the default and `--precision fp16`
   > is the opt-in, guarded by a finiteness check that auto-retries at fp32.
   > Two further traps this exposed: (a) the NaN is **sticky** — an fp16 run poisons the
   > next fp32 run on the same model object until `torch.mps.empty_cache()` is called,
   > implicating a cached MPS graph rather than arithmetic overflow; (b) NaN casts silently
   > to 0 in uint8, so an unguarded pipeline exports a plausible-looking **all-black depth
   > video** rather than failing. `normalize()` now raises `NonFiniteDepth`, and Gate 0's
   > smoke test asserts finiteness — it originally checked only tensor shape and therefore
   > **passed green on all-NaN output**.
3. **Ignore their pinned `requirements.txt`** (CUDA-era: torch==2.1.1, xformers==0.0.23,
   numpy==1.24.0). `xformers` is optional — `motion_module.py` try/excepts the import and
   falls back to plain PyTorch attention. Never install xformers on macOS.
4. **Do not depend on `decord`** (their video reader) — no reliable arm64 macOS wheels.
   Write our own frame reader/writer on OpenCV (`cv2.VideoCapture`) or imageio-ffmpeg,
   and shell out to the system ffmpeg for encode/mux.
5. Their `save_video` grayscale path already does per-video min/max normalization — reuse
   the normalization *logic* but do our own encoding (theirs may not produce
   yuv420p+faststart MP4s that Seedance/browsers like).

### Seedance-side constraints the tool must enforce/target (triangulated from fal schema,
Ark wrapper projects, and multiple guides; official BytePlus doc pages are JS-rendered and
were not directly extractable — one human console check before hard-failing on any of these
is noted in Phase 4)

| Constraint | Value |
|---|---|
| Video refs per generation | up to 3, **combined duration 2–15 s**; **[AMENDED 2026-07-24] each clip also ≥ 2 s — checker adds a FAIL tier for < 2 s** (OSIDE's adapter header: "2–15s each") |
| File | MP4 or MOV, **≤ 50 MB** each — **[AMENDED 2026-07-24] contested: OSIDE's `arkSeedance.ts` header says ≤ 200 MB. Enforce the stricter 50 MB in `checks.py`; the Phase 4 human console check settles it** |
| Input resolution window | ≈ 480p–1080p total pixels (~409,600–2,086,876 px); fal quotes 640×640–834×1112 for its surface |
| Frame rate accepted | 24–60 fps; **output is 24 fps** — mismatched ref fps is a reported jitter cause |
| Output duration | 4–15 s (15 s hard ceiling on 2.0) |
| Ref addressing | `@Video 1` ordinal mention in prompt, semantic role stated in text |
| First-frame + refs | API supports coexistence, but **OSIDE's adapter deliberately never sends refs alongside a start frame** — the kit therefore treats the extracted still as an `@Image` look ref, not a `first_frame` param |
| Faces | photographic human faces in any reference image are rejected — identity refs must be AI-generated/stylized (or go through Peter's Ark asset-library likeness path) |

Community adherence heuristic worth echoing in the kit's prompt guidance: ~**70% identity
refs / 30% motion ref** weighting; over-weighting the motion ref causes identity drift.

**[AMENDED 2026-07-24] Two more checker facts:**
- **Mixed-input total:** OSIDE disagrees with itself — `lib/prompter/lint.ts`
  `maxMixedInputs: 12` vs the adapter's `total: 15`. `checks.py` follows the stricter 12.
- **Ref length vs output length:** Seedance's output minimum is 4 s and a 3 s depth ref
  driving a longer generation has undefined behavior for the uncovered tail. `kit.json`
  carries a `suggestedOutputSeconds` (≈ the trimmed ref duration, clamped to the 4–15 s
  output window), the README states the rule ("match output length to the ref; never ask
  for much more output than you have motion"), and "short ref / long output" is an
  explicit Phase 4 observation item.

### How this aligns with OSIDE AI FILMMAKER (scouted 2026-07-23, app v0.66.2)
Peter's main app already has **all the plumbing** to consume this tool's output:
- Ark adapter `lib/fal/adapters/arkSeedance.ts` sends `video_url` refs with
  `role:'reference_video'` (URL or `asset://`, no base64); ref caps `{image:9, video:3,
  audio:3, total:15}`; video-ref presence flips billing to the cheaper with-video rate.
  **[AMENDED 2026-07-24] The header comment claiming video refs were "never live-fired"
  is STALE.** OSIDE CHANGELOG v0.39.1 (2026-07-18, Plan 9 Phase 7 sweep): three
  `@video`-chained shots fired successfully — chain handoff frame-near-exact, the anime
  cel lock held while chaining, with-video billing confirmed. So the `reference_video`
  plumbing is **proven**; what Phase 4 tests is the *depth-as-motion technique*, not the
  wiring — a Phase 4 failure points at the technique, never the adapter. The one-line
  comment correction in `arkSeedance.ts` goes into the **Phase 5 ticket** (zero OSIDE
  files touched until the technique is proven — Peter's call 2026-07-24).
- Composer dialect (`lib/prompter/compile.ts`) numbers refs `@Video N` and uses
  Define-binding sentences (`lib/prompter/refManifest.ts`, e.g. `characterBinding` line 47,
  `chainTailBinding` line 256). A depth video needs only a new binding sentence.
- Uploads: `app/api/upload/route.ts` (≤200 MB, video mimes OK) → Supabase `references`
  bucket → usable as a plain URL ref.
So: **v1 is a standalone tool whose output drops into OSIDE with zero app changes.**
Deeper integration is a deferred ticket (Phase 5), not part of this build.

### Prior art (both Apache-2.0, both target Seedance by name — steal ideas, not scope)
- `wassermanproductions/motion-previs-studio` — Electron app extracting pose+depth+edges+
  camera-solve as a multi-layer "Production Pack". Ideas stolen here: 720p short-edge
  Seedance preset; deterministic ffmpeg frame-by-frame encode; framing the export as a kit.
- `wassermanproductions/blockout` — synthetic 3D previs → perfect depth. Sibling approach,
  not a competitor. Its per-generator "prompt.txt profile" export pattern inspired our kit.
We deliberately build **narrower and deeper**: one input (real footage), one output (depth +
kit), best-in-class temporal quality, zero Electron weight.

---

## 1. Product definition

**Name:** Depth Converter (working name; binary/package `depthc`).
**One-liner:** drop in a reference clip → get a Seedance-ready depth motion kit.

### v1 deliverables
1. **CLI** — `depthc convert <input> [options]` — the core, fully scriptable/testable.
2. **Gradio web UI** — local, dark-themed, drag-and-drop, for Peter's daily use.
3. **Seedance Kit** per conversion (a folder + zip):
   - `depth.mp4` — grayscale depth video, near=bright (invertible), encode spec below
   - `still.png` — a clean frame from the *original* clip. Default `--still auto`: pick
     the **sharpest of the first 24 frames of the trimmed range** by Laplacian variance
     (`cv2.Laplacian(gray, cv2.CV_64F).var()` — standard blur detection; a whip-pan or
     obstructed frame 0 must never win). Overrides: `--still first` or `--still <N>`.
     For optional use as an `@Image` composition/look reference — NOT a first_frame param.
     **[AMENDED 2026-07-24] REAL-FACE FLAG:** the still comes from the *original* clip —
     if the source footage contains a person it carries a real photographic face, exactly
     what Seedance rejects on image inputs. Keep the still (it's the Library thumbnail and
     a framing reference for the filmmaker's eye) but: `kit.py` asks/flags whether real
     faces are present (`--faces` flag / UI checkbox, default assume-yes for footage with
     people), `kit.json` records `stillContainsRealFaces`, the checker WARNs, and
     prompt.txt's guidance says "still.png is LOCAL-ONLY when real faces are present —
     never attach it as an @Image ref." No face-detection dependency in v1 — a human flag
     is enough.
   - `prompt.txt` — a paste-ready prompt block in OSIDE's seedance-v2 dialect (below)
   - `kit.json` — manifest: source filename+SHA256, duration, fps, resolution, settings
     used, constraint-check results, tool version
4. **Constraint checker** — every export prints a three-tier report against the Seedance
   limits: **PASS** (3–8 s single-signal sweet spot, size/pixel/fps in window), **WARN**
   (8–15 s: "adherence degrades — trim to the one clear motion signal"; 2–3 s: "shorter
   than the sweet spot — fine for a single beat"; real-face still flag), **FAIL**
   (**< 2 s [AMENDED 2026-07-24]**, > 15 s, > 50 MB, pixel/fps out of window).

### The end-to-end workflow (v1 — put this in README, it is the product story)
1. **Curate** — find the clip with the camera move / choreography you want.
2. **Trim** — cut to the 3–8 s sweet spot, one clear motion signal only (in the tool's
   trim UI or beforehand in any editor — both are fine).
3. **Convert** — drop it into Depth Converter, tag it, hit Convert; it renders locally
   like an editor export and the kit lands in the Motion Library.
4. **Attach** — in OSIDE: Shotlist → pick the shot → **go straight to "make video" —
   SKIP the create-frame step.** A depth-motion shot is a refs-only shot: the adapter
   deliberately never sends refs alongside a start frame (`refsWithStartFrame:false`),
   so a frame-first shot would silently drop the depth ref. In Composer: add
   character/location/prop image refs (the look), upload `depth.mp4` as the video ref
   (the motion), paste the binding block from `prompt.txt`.
5. **Fire** — generate. (The kit's still is an `@Image` look ref, never first_frame.)
**Resolution guidance (also for README):** the depth ref itself is always ≤720p-class —
it is a motion guide, not picture, and Seedance's ref-input window caps it anyway.
Output resolution is a separate choice: test at 480p, confirm at 720p, master at
720p/1080p. 4K output is supported by the standard model per OSIDE's adapter but is
untested with video refs and costs several× more — treat it as a Phase 4 experiment;
upscaling a 720p/1080p master (already in Peter's stack) is usually the cheaper 4K path.
Note the cloud boundary: the **source clip never leaves the Mac**. Only the
appearance-stripped `depth.mp4` (zero original pixels, ~2–6 MB) is uploaded — and only
at fire time, into the existing `references` bucket, because the Ark API takes video
refs by URL only (no base64). Local library = permanent home; cloud = transient copies.

### The Motion Library (output behavior — reuse across films)
All kits land in a structured local library, **not** loose output folders and not any
database:
- Root: `~/Depth_Motion_Library/` (configurable via `--library` / env `DEPTHC_LIBRARY`).
- Each conversion = one folder named from tags + auto-increment:
  `Sword-fight_Tracking-orbit_001/` containing the full kit **plus a copy of the trimmed
  source clip** (`source.mp4`) so the motion can be re-converted or re-trimmed later.
- `kit.json` carries the metadata: `tags[]`, `description`, source filename + SHA256,
  duration/fps/resolution, settings used, tool version, created-at.
- CLI: `depthc library list [--tag X]` — table of kits from scanning `kit.json` files.
- UI: "Tags" and "Description" fields on the convert screen (folder name derives from
  tags, slugified); a simple **Library tab** listing kits with tag filter, depth preview,
  copy-path and reveal-in-Finder buttons. No database, ever — the folder tree + kit.json
  files ARE the library.

### The binding sentence (`prompt.txt` core — matches OSIDE Define-binding style)

**[AMENDED 2026-07-24] TWO-VARIANT LAW.** OSIDE's compiler (`lib/prompter/compile.ts`
~427–459) *auto-generates* the ACTIVE REFERENCES block from each ref's binding field —
pasting a second self-contained "ACTIVE REFERENCES" block into the Composer produces two
competing reference blocks, the exact failure mode `refManifest.ts` exists to prevent.
So `prompt.txt` carries **two clearly labeled sections**:

**Variant A — FOR OSIDE COMPOSER (binding phrase only).** A bare noun phrase to paste
into the video ref's binding field; the compiler wraps it into its own block and does the
`@Video N` numbering (matches the `motionOnlyBinding()` house style):
```
the depth motion blueprint (grayscale depth map of a real performance) — follow its body
motion, choreography, camera path, and timing exactly, beat for beat; it carries structure
only: take no color, texture, clothing, face, lighting, or environment from it; appearance
comes entirely from the image references
```

**Variant B — STANDALONE (Ark console or any non-OSIDE surface).** The full block:
```
ACTIVE REFERENCES
@Video 1 — depth motion blueprint (grayscale depth map of a real performance)

Define @Video 1 as the motion blueprint. Follow its body motion, choreography, camera
path, and timing exactly, beat for beat. @Video 1 carries structure only: take no color,
texture, clothing, face, lighting, or environment from it. Appearance comes entirely from
the image references.
```

Both variants live in the same template file; `kit.py` renders both. Followed by a short
GUIDANCE section (not part of either paste block) restating: keep clips 3–8 s for tightest
adherence; one dominant motion signal per clip; ~70/30 identity/motion ref weighting;
identity images must not contain real photographic faces; match output length to the ref
(`suggestedOutputSeconds`); still.png is local-only when it contains real faces.

**TEMPLATE LAW:** the binding block and guidance text live in
`depthc/templates/seedance-2.0.txt` (with `{placeholders}` filled by `kit.py`), NOT
hardcoded in Python. Seedance 2.5 is rolling out and reference syntax may shift — a
future dialect is a new template file (`seedance-2.5.txt`) plus a `--dialect` flag, with
zero logic changes. `kit.json` records which template version produced the prompt.

### Conversion options (CLI flags = UI controls; defaults chosen for Seedance)
| Option | Default | Notes |
|---|---|---|
| `--trim START END` | full clip | UI gets in/out handles with the **3–8 s sweet-spot zone visually marked**; WARN 8–15 s, hard-cap at 15 s unless `--allow-long` |
| `--tags` / `--desc` | none | Motion Library metadata; tags also name the kit folder |
| `--faces` / `--no-faces` | `--faces` (assume real faces present) | **[AMENDED 2026-07-24]** records `stillContainsRealFaces` in kit.json; drives the checker WARN + "still is local-only" guidance (§1 still.png). UI: checkbox |
| `--resolution` | `720p` (long edge 1280, even dims, AR preserved) | also `480p`, `source` (validated against pixel window) |
| `--crop` | `auto` **[ADDED 2026-07-24, Phase 1]** | strips letterbox/pillarbox mattes, which a depth model otherwise reads as the nearest surface in the scene. `none` keeps them; `W:H:X:Y` sets one by hand. Sizing and aspect ratio derive from the cropped frame |
| `--fps` | `conform-24` | conform to 24 fps to match Seedance output cadence; `keep` preserves source (validated 24–60) |
| `--invert` | off | white=near by default (matches VDA/community convention) |
| `--keep-audio` | off | mux source audio back onto depth.mp4 (for beat reference while editing; Seedance ignores ref-video audio — rhythm sync is the @Audio slot's job. Say this in help text.) |
| `--input-size` | 518 | VDA inference short-side, multiple of 14 |
| `--fp32` | off | escape hatch if MPS autocast misbehaves |
| `--preview` | off | also render side-by-side (original | depth) MP4 for eyeballing/socials |

### Encode spec for `depth.mp4` (do not deviate)
libx264, `-crf 17 -preset medium -pix_fmt yuv420p -movflags +faststart`, grayscale
replicated to 3 channels, even width/height, source-true duration. Post-encode: assert
file ≤ 50 MB (re-encode at crf 20 with WARN if exceeded).

### §1a Test assets on hand [AMENDED 2026-07-24]
- **`TEST CLIPS/` in this folder** — four real 1080p clips, probed 2026-07-24, ready for
  Gates 1 and 3 (no synthetic-only fallback needed):
  | Clip | Duration | fps | Size | Gate role |
  |---|---|---|---|---|
  | CLIP A.mp4 | 7.9 s | 60 | 6.5 MB | sweet-spot + 60→24 conform |
  | CLIP B.mp4 | 8.1 s | 29.97 | 2.9 MB | sweet-spot edge + NTSC conform |
  | CLIP C.mp4 | 9.9 s | 24 | 4.9 MB | exercises the 8–15 s WARN tier |
  | CLIP D.mp4 | 8.9 s | 23.976 | 1.3 MB | near-24 conform |
  Synthetic ffmpeg clips remain the *pytest* fixtures (deterministic); these real clips
  are the manual gate checks.
- **Approved identity characters: WILLIAM and ANNA** — both already registered and
  approved in the Ark asset-library likeness system. Phase 4's identity `@Image` refs use
  these two (via their `asset://` refs / crowned stills in OSIDE), NOT ad-hoc AI images —
  they're pre-cleared for the face filter and give the live-fire a known-good identity
  baseline, so any drift observed is attributable to the depth ref weighting.

### Repo law [AMENDED 2026-07-24]
**Local-only until the technique is proven.** `git init` locally at Phase 0 (local
commits per gate are fine and wanted — they're the rollback points), but NO remote until
Gate 4 passes. Once the live-fire verifies motion transfer, push as a private repo under
OSideMedia (remember the per-repo noreply email:
`261988918+OSideMedia@users.noreply.github.com`). If the technique fails, the repo never
leaves the Mac.

### What NOT to build (v1 scope law)
No pose/skeleton extraction, no camera-solve JSON, no edges/normals, no Electron, no
yt-dlp URL intake, no model-size selector, no CoreML path, no batch queue, no cloud
anything. All parked in the backlog (§4). The tool does one thing at reference quality.

---

## 2. Architecture & repo layout

```
Depth Converter/
├── PLAN.md                  # this file
├── README.md                # user-facing: install, usage, the technique, rights note
├── pyproject.toml           # uv-managed; requires-python >=3.11; torch>=2.5
├── LICENSES/                # Apache-2.0 texts + attribution for VDA + our license
├── scripts/
│   └── setup.sh             # clones vendor repo at PINNED commit; downloads checkpoint
├── vendor/
│   └── Video-Depth-Anything/  # pinned git clone (gitignored; recreated by setup.sh)
├── checkpoints/             # video_depth_anything_vits.pth (gitignored)
├── depthc/
│   ├── __init__.py          # __version__
│   ├── engine.py            # VDA wrapper: device pick (mps→cpu), model load, windowed
│   │                        #   inference, global normalization → ndarray frames.
│   │                        #   MEMORY HYGIENE: torch.mps.empty_cache() after EVERY
│   │                        #   32-frame window, and full cleanup (del tensors,
│   │                        #   gc.collect(), empty_cache()) after each conversion —
│   │                        #   the Gradio process is long-running and MPS holds
│   │                        #   cached allocations; back-to-back conversions must not
│   │                        #   creep toward OOM on 24 GB unified memory
│   ├── templates/
│   │   └── seedance-2.0.txt # prompt template (binding + guidance, {placeholders})
│   ├── video_io.py          # ffprobe probe, OpenCV frame reader, ffmpeg encoder/muxer,
│   │                        #   trim, fps conform, resolution scaling
│   ├── checks.py            # Seedance constraint table → PASS/WARN report
│   ├── kit.py               # still extraction, prompt.txt writer, kit.json, zip
│   ├── cli.py               # argparse/typer entrypoint `depthc`
│   ├── smoke.py             # `python -m depthc.smoke` — MPS + model sanity
│   └── ui.py                # Gradio app `depthc ui`
├── tests/                   # pytest; golden tests use synthetic ffmpeg clips
│   ├── conftest.py          # fixture: generate 3s test clip (moving box over gradient)
│   ├── test_video_io.py
│   ├── test_checks.py
│   ├── test_kit.py
│   └── test_engine.py       # marked `slow`; real model, tiny clip, asserts shape/
│                            #   grayscale/duration/fps + temporal-stability metric
└── DOCS/
    └── LIVE-FIRE.md         # Phase 4 human protocol (written in Phase 2)
```

Key engineering decisions:
- **Vendor, don't fork:** `setup.sh` clones VDA at a pinned commit into `vendor/` and the
  engine does `sys.path.insert` (or a `-e` path install) to import
  `video_depth_anything.video_depth.VideoDepthAnything` directly. We never edit vendor
  files; device/precision handling lives in `engine.py`. Record the pinned SHA in both
  `setup.sh` and `kit.json`.
- **Checkpoint download** via `huggingface_hub.hf_hub_download` (repo
  `depth-anything/Video-Depth-Anything-Small`, file `video_depth_anything_vits.pth`) into
  `checkpoints/`; `setup.sh` and first-run both check/fetch.
- **Own the I/O:** all decode/encode/trim/scale/mux through `video_io.py` using OpenCV +
  system ffmpeg subprocess. Reuse VDA's inference + its global min/max normalization
  approach; nothing else from their runtime scripts.
- **Windowing memory note:** VDA-Small processes 32-frame windows at 518 px — comfortable
  in 24 GB unified memory at fp16. If MPS OOMs at `--resolution source` on 1080p inputs,
  inference resolution is `--input-size` anyway (output upsampled to target res), so this
  should not occur; don't build tiling.

---

## 3. Build phases (each has a GATE — do not proceed past a failing gate)

### Phase 0 — Bootstrap & smoke (~small)
- **[AMENDED 2026-07-24] FIRST: fix the x86_64 toolchain (§0 blocker).** Install arm64
  standalone `uv` into `~/.local/bin`; `uv python install` an arm64 CPython 3.11/3.12;
  optionally drop a static arm64 ffmpeg into `~/.local/bin`. Verify with
  `file ~/.local/bin/uv` and `uv run python -c "import platform; print(platform.machine())"`
  → must print `arm64` before anything else happens.
- `git init` (local-only — see Repo law §1a); commit at every green gate.
- `uv init` the project; pyproject with deps: `torch>=2.5`, `torchvision`, `opencv-python`,
  `numpy`, `einops`, `easydict`, `tqdm`, `huggingface_hub`, `gradio`, `typer`, `pytest`
  (dev). **No xformers. No decord.**
- Write `scripts/setup.sh` (pin VDA commit — verified HEAD 2026-07-24:
  `4f5ae23172ba60fd7bc11ef671cca678842c7072`; re-check at build time and hard-code the
  SHA) + checkpoint fetch (HF URL returned 200 on 2026-07-24).
- Write `depthc/smoke.py`: assert `platform.machine() == 'arm64'` AND
  `torch.backends.mps.is_available()`, load VDA-Small on MPS, run a single synthetic
  32-frame window (random tensor), print device, dtype, output shape, and wall time.
- **GATE 0:** `bash scripts/setup.sh && ./.tools/uv run python -m depthc.smoke` prints OK on
  MPS **on an arm64 interpreter** (CPU fallback allowed only with a loud warning; x86_64
  interpreter is an automatic gate FAIL).
- **✅ GATE 0 PASSED 2026-07-24.** arm64 CPython 3.12.13, torch 2.13.0, MPS available,
  vendor pinned at `4f5ae23172`, checkpoint 116.4 MB, VDA-Small loaded on MPS in 0.4 s.
  One real 32-frame fp16 window: **2.57 s → 80 ms/frame**, i.e. **~19 s for a 10 s/24 fps
  clip** — roughly 5× faster than this plan's 1–5 min estimate, so the UI's progress
  affordances matter less than assumed. Harmless noise: VDA prints "xFormers not
  available" (its documented try/except fallback to plain PyTorch attention).

### Phase 1 — Engine + CLI (~the core)
- `video_io.py`: ffprobe JSON probe (reject non-video, report duration/fps/resolution/
  rotation — **apply rotation metadata** so phone footage comes out upright); frame
  reader honoring trim; fps conform via ffmpeg `fps=` filter pre-decode; scaler.
- `engine.py`: full-clip windowed inference (mirror VDA's INFER_LEN=32/OVERLAP=10 logic
  via their class), fp16 autocast on MPS (fp32 flag), global normalization to uint8.
- Encoder + optional audio mux + optional side-by-side preview (ffmpeg hstack of
  originals and depth at matched height).
- `cli.py`: `depthc convert`, `depthc probe`, wired options table from §1.
- Tests: synthetic-clip golden tests. Temporal-stability assertion for the engine test:
  on a clip whose scene geometry is static (moving box, static camera distance layout),
  the per-frame mean brightness std across frames must be small (calibrate threshold
  once, then freeze) — this is the flicker regression guard.
- **GATE 1:** `uv run pytest` green including one `slow` real-model test; converting
  **CLIP A and CLIP B from `TEST CLIPS/` [AMENDED 2026-07-24]** by hand produces upright,
  smooth, obviously-depth MP4s (conformed to 24 fps from 60/29.97) that QuickTime and
  Chrome both play.
- **✅ GATE 1 PASSED 2026-07-24.** 39 tests green (5 `slow`, real model). CLIP A 60→24 fps
  (191 frames, 33.2 s) and CLIP B 29.97→24 (195 frames, 34.2 s), both 1280×720 yuv420p
  with moov-before-mdat verified by atom parse, both decoded by AVFoundation (QuickTime's
  stack). Depth quality on real footage is excellent — in CLIP B's sword fight both
  fighters separate cleanly from the archway and **the sword itself resolves**, which is
  exactly the prop-geometry survival the technique depends on. The 8 s WARN fired for
  CLIP B (8.1 s) and correctly stayed silent for CLIP A (7.94 s).

  Decisions taken during Phase 1, both documented at their call sites:
  - **ffmpeg pipe replaces `cv2.VideoCapture` for decode.** Trim + fps conform + scale are
    one deterministic filter graph, and ffmpeg applies rotation metadata while OpenCV
    ignores it — so "phone footage comes out upright" is free rather than hand-rolled.
    OpenCV is retained for image analysis (Phase 2's still selection).
  - **Upstream's per-window loop is hooked, not reimplemented,** by rebinding `tqdm` inside
    their module namespace for the duration of a call. This yields both the per-window
    `empty_cache()` and the progress callback with zero vendor edits and without
    duplicating their cross-window scale/shift alignment maths.

  **⚠️ NEW ISSUE FOUND BY THE REAL-FOOTAGE CHECK — letterbox bars become a fake near-plane.**
  CLIP A is matted 2.39:1 inside 16:9. The model reads the flat black bars as *the nearest
  surface in the scene*: measured over 26% of the frame at mean brightness **160/255**,
  against **55/255** for the actual content. A depth ref handed to Seedance would therefore
  carry a large bright slab across the top and bottom of frame — pure noise in the motion
  channel, and brighter than the subject it is supposed to be following. Not anticipated
  anywhere in this plan; found only because Gate 1 insists on real footage.

  **RESOLVED — `--crop auto` is now the default** (Peter's call 2026-07-24), with
  `--crop none` and explicit `--crop W:H:X:Y` as overrides. Three of the four test clips
  (A, C, D) turn out to be matted; only B is full-frame.

  Detection is deliberately **not** ffmpeg's `cropdetect`. On a synthetic clip with a
  legitimately dark right edge, cropdetect removed 20 px of *real picture* along with the
  matte — it thresholds on an average and cannot distinguish a dark image from a bar.
  Silently cropping real content would change the very motion the ref exists to carry, so
  `detect_crop()` instead samples ~9 frames across the clip and takes the per-row and
  per-column MAXIMUM across every sampled pixel: a row is matte only if it is near-black at
  every pixel in every sampled frame. One bright pixel anywhere disqualifies the row. It
  also keeps 4 px more picture than cropdetect did on CLIP A (804 vs 800 rows).

  Verified on CLIP A — top rows 134.2 → **30.6**, bottom 186.5 → **88.2**, centre 53.5 →
  59.2: the brightness ordering is now correct (figure nearest, canopy far) instead of
  inverted by the matte. Conversion also ran 31% faster (22.7 s vs 33.2 s) on 26% fewer
  pixels.

### Phase 2 — Seedance Kit + checks (~small)
- `checks.py` constraint table (§0 values) with PASS/WARN/FAIL tiers and one-line remedies.
- `kit.py`: sharpest-still extraction (Laplacian variance per §1), prompt.txt rendered
  from `templates/seedance-2.0.txt`, kit.json, `source.mp4` copy, zip. Output goes into
  the Motion Library structure (§1): tag-slug folder naming + auto-increment.
  CLI: `depthc kit` (probe → convert → check → package in one shot; the headline command
  in README) and `depthc library list [--tag X]`.
- Write `DOCS/LIVE-FIRE.md` (protocol below, Phase 4).
- **GATE 2:** kit folder from the real test clip lands in the library, passes its own
  checker; prompt.txt matches the template; kit.json (incl. tags/description/template
  version) validates against a checked-in JSON schema; `library list --tag` filters it.
- **✅ GATE 2 PASSED 2026-07-24.** 127 tests green. Five real kits built into a library
  from the test clips; all five manifests validate against `depthc/kit_schema.json`.
  `Sword-fight_Tracking-orbit_001` (CLIP B trimmed 1–6 s) contains depth.mp4 / still.png /
  prompt.txt / kit.json / source.mp4 / preview.mp4 and reports WARN — the only finding
  being the real-face still flag, exactly as designed. `library list` and
  `library list --tag "sword fight"` both filter correctly. Tier drives the exit code: a
  deliberate 1.5 s ref reports FAIL on the < 2 s rule and exits 1, while `--no-faces` on a
  4 s ref gives a clean PASS and exits 0.

  Notes on what was built:
  - **`source.mp4` is stored trimmed but UNCROPPED**, at source resolution. The crop is a
    derived setting recorded in `kit.json`; baking it into the archived source would
    permanently fix one framing decision and defeat the point of keeping the source for
    re-conversion. It is re-encoded rather than stream-copied, because `-c copy` can only
    cut on keyframes and would shift the in-point off the motion the kit is built around.
  - **The still is cut from the cropped, scaled frames** — the same pixels fed to the
    model — so its framing matches depth.mp4 rather than the source container.
  - **The library is the filesystem.** `library_list()` is a directory scan of `kit.json`
    files; a malformed manifest is skipped, never fatal. No database, per §1.

### Phase 3 — Gradio UI (~medium)
`depthc ui` → localhost app, dark theme (near-black surfaces, minimal chrome — Peter's
taste runs to the OSIDE studio look; no rainbow Gradio defaults). Layout: left column =
source (upload/drop, probe readout, trim slider with duration+constraint feedback,
settings accordion matching CLI flags, **Tags + Description fields** for the Motion
Library); right column = result (depth preview player, side-by-side toggle, constraint
PASS/WARN/FAIL table, prompt.txt copy box, "Download Kit (.zip)" + reveal-in-Finder).
Second tab: **Library** — kit list with tag filter, depth preview, copy-path / reveal
buttons (folder scan of kit.json files, no database). Progress via Gradio's progress API
fed from the engine's per-window callback. Long-clip UX: >15 s uploads get an inline
warning and auto-trim suggestion, never a silent failure. Session UX: back-to-back
conversions in one session must not grow memory (engine memory hygiene, §2).

**macOS launcher pair (in scope — Peter asked for it):** build `Depth Converter.app` and
`Depth Stop.app` in `/Applications`, mirroring the existing OSIDE Studio.app /
OSIDE Stop.app pattern (AppleScript/Automator wrappers; note the `arch -arm64` fix that
pair needed). Start app: launch the Gradio server (uv env), wait for the port, open the
browser tab. Stop app: kill the server cleanly by port/pidfile. Pick a fixed port that
does not collide with OSIDE dev (3000) — e.g. 7861 — and record it in README.
- **GATE 3:** manual run-through in a browser with a real clip; screenshot saved to
  `DOCS/`; no unhandled exceptions in console for: happy path, >15 s clip, non-video
  file, cancel-and-retry. Launcher apps verified: double-click start → UI opens;
  double-click stop → server gone.
- **✅ GATE 3 PASSED 2026-07-25.** 156 tests green. Real browser run-through: CLIP B
  uploaded through the page produced `CLIP-B_001` — a complete, schema-valid kit
  (1280×720, 8.125 s @ 24 fps, WARN, 32.8 s on MPS) with the constraint report rendered
  in-page. Library tab lists all six kits with correct PASS/WARN/FAIL tiers. Screenshot at
  `DOCS/ui-convert-tab.jpg`.
  - **Launcher pair verified end to end from `/Applications`:** double-click start → UI up
    in ~3 s → double-click stop → listener gone, pidfile removed, port unreachable. Built
    by `scripts/build_launchers.sh` (re-runnable). Port **7861**, pidfile
    `~/.depthc-ui.pid`, log `~/Library/Logs/depthc-ui.log`. The `arch -arm64` exec line is
    in place per §0.4, and the venv Python and torch are both arm64-only binaries with
    `mps available: True` confirmed through the launcher's exact exec path.
  - **The pidfile alone is not enough — validated by accident.** The launcher's recorded
    child PID (56468) differed from the process actually holding the port (56471), and an
    earlier manual `pkill` killed a `uv` parent while its Python child kept listening,
    leaving a stale socket that answered TCP but not HTTP and blocked the next start. The
    stop app therefore sweeps the port with `lsof` after honouring the pidfile, escalating
    to `-9`. Keep that sweep.

  Three UI defects found and fixed during the run-through:
  - Gradio's Markdown/HTML blocks and secondary buttons ship a light fill, which rendered
    as **white slabs on the near-black page**. Fixed in the theme (`button_secondary_*`,
    table and label fills) rather than by piling on `!important`.
  - An **empty** `gr.HTML` renders as a bare slab. The trim readout is now seeded with
    "Load a clip to set the range." — no empty block, and a more useful default.
  - **An in-point equal to the out-point silently converted the whole clip.** The guard
    tested `end > start` before clamping, so an empty range fell through as "no trim" and
    the user got a full conversion they never asked for. Now clamped first, then rejected
    with an explicit message.
  - Also: a non-video upload leaked raw ffprobe output ("Invalid data found when
    processing input") at the user; `probe()` now says "<name> is not a video file".

  **[CORRECTED 2026-07-25] The "CDP cannot screenshot video pages" limitation was a
  MISDIAGNOSIS.** Re-tested deliberately: in a fresh browser tab, captures succeed with the
  source clip loaded, on the Library tab, and — the state originally claimed impossible —
  after a full conversion with all three video elements live. Neither video nor the
  Dataframe was ever the trigger.

  The real fault is an **intermittent renderer crash** that kills the tab outright (observed
  twice; each time the tab group had vanished on the next call, so "renderer unresponsive"
  was literal). It is **not memory exhaustion** — the page holds ~14 MB of a 4.4 GB heap
  with 362 elements — and no console errors precede it. Root cause not established; it may
  belong to the automation extension rather than the page. **Workaround: open a fresh tab,
  after which capture works normally.** Worth watching in ordinary use; if a tab ever dies
  mid-session the server is unaffected and a reload restores the UI, since all state lives
  in the Motion Library on disk rather than in the page.

  Chasing that also caught a fourth styling defect: **video players rendered as solid white
  slabs** on the dark page — before the first frame paints, and permanently in the pillarbox
  around a 2.39:1 depth clip. Now forced to black, which is also the honest backdrop for
  judging a grayscale depth map.

### Phase 4 — Live-fire verification (HUMAN-GATED — do not execute autonomously)
**LAW (mirrors OSIDE): nothing fires against a paid API automatically.** This phase is a
written protocol Peter runs (or supervises), budget-approved, low-res first.
**[AMENDED 2026-07-24]** the `reference_video` wiring is already proven (v0.39.1 chaining
sweep) — this phase tests the *depth-as-motion technique only*; any failure points at the
technique/weighting, never the adapter:
1. Pick a 5–8 s single-action reference clip (CLIP A/B/D qualify); `depthc kit` it.
2. Upload `depth.mp4` through OSIDE's upload route (references bucket) or attach in
   Composer; identity `@Image` refs = **WILLIAM and/or ANNA** (Ark-approved likeness
   characters, §1a) — pre-cleared for the face filter, known-good identity baseline;
   paste the kit's **Variant A binding phrase** into the ref's binding field (never the
   standalone block — the Composer compiles its own ACTIVE REFERENCES).
3. Generate at 480p first (with-video billing is the cheaper token rate; expect roughly
   $0.10–$0.40 at 480p/5 s based on the app's existing ledger scale). Evaluate: does
   motion/camera/timing track the source? **Set output duration ≈ the ref duration
   (kit.json `suggestedOutputSeconds`); also run one deliberate short-ref/long-output
   probe and record what the model does with the uncovered tail [AMENDED 2026-07-24].**
   Then one 720p confirmation. Optional budget-approved experiment: one 4K generation to
   test whether adherence holds at 4K with a video ref (supported by the standard model
   on paper, untested, several× cost).
4. Record results in `DOCS/LIVE-FIRE.md` (what adhered, what drifted, prompt tweaks).
   This is OSIDE's first *depth-motion* video-ref fire (plain video-ref chaining already
   verified v0.39.1).
- Also on this checklist: one human browser check of the official BytePlus Ark video-ref
  parameter docs (JS-rendered, machine-unreadable this session):
  https://docs.byteplus.com/en/docs/ModelArk/1520757 — confirm the §0 constraint numbers
  before hard-failing (vs warning) on any of them in `checks.py`.
- **GATE 4:** one verified motion-transfer generation, side-by-side archived.

### Phase 5 — OSIDE integration ticket (DEFERRED — write the ticket, don't build it)
Produce `DOCS/OSIDE-INTEGRATION.md` describing the later in-app path, with these scouted
pointers so the future session lands instantly: add a `motionBlueprintBinding()` next to
`characterBinding` in `lib/prompter/refManifest.ts` (~line 47; note `motionOnlyBinding()`
~line 218 already exists as the house-style starting point); surface it when a video
ref is tagged as a depth/motion ref; lint hint in `lib/prompter/lint.ts` (video caps at
~line 436/469); note the adapter's `refsWithStartFrame:false` rule (arkSeedance.ts:237)
means depth refs pair with @Image identity refs, never a first_frame. Optional far end:
"Motion" as a first-class reference role in the Composer's SourcesDialog.
**[AMENDED 2026-07-24] Also in the ticket:** fix the stale `arkSeedance.ts` header
comment ("roles are doc-grounded, not yet live-fired" — video refs fired v0.39.1,
2026-07-18) and reconcile `lint.ts maxMixedInputs: 12` vs the adapter's `total: 15`
against whatever the Phase 4 console check establishes.

---

## 4. Backlog (parked, in rough priority order)
1. CoreML fast-preview mode — Apple's official `coreml-depth-anything-v2-small` (Apache,
   ~25–33 ms/frame on the Neural Engine) for instant per-frame previews while VDA renders
   the final. Per-frame flicker acceptable for preview only.
2. Batch/queue mode for converting a shot-list of refs.
3. Optional pose-skeleton export layer (motion-previs-studio-style) for Kling Motion
   Control cross-compatibility.
4. yt-dlp intake (rights-sensitive — Peter decides).
5. In-app OSIDE integration (Phase 5 ticket).
6. Blockout interop (accept its synthetic depth.mp4 into the same kit/checker pipeline).

## 5. Risks & honest limits (surface these in README)
- **Adherence is probabilistic.** Seedance follows video refs well but not deterministically;
  >10 s and multi-signal clips degrade; fast motion can flicker. The kit's guidance text
  is the mitigation.
- **Moderation drift.** The technique is partly popular *because* Seedance blocks real-face
  inputs; ByteDance filter changes could affect the workflow in either direction.
- **Seedance 2.5** (out July 2026) officially supports green-screen/white-model motion
  references and 50 refs/30 s — it may eventually consume this tool's output even better,
  but nothing native replaces "I have real footage, give me a clean motion carrier" yet.
  Depth-from-real-footage remains the zero-3D-software path.
- **Rights.** Depth strips pixels, not authorship: choreography and music in the source
  clip carry their own rights. One-line reminder in README + UI footer.
- **License law** (§0): VDA-Small/Apache only. Any future "better quality" temptation
  toward Base/Large weights is a commercial-license violation.

## 6. Definition of done
`bash scripts/setup.sh` → `uv run depthc kit myclip.mov` → a kit folder that passes its
own checker, plus `uv run depthc ui` for daily use; all gates 0–3 green; Phase 4 protocol
written and awaiting Peter's budget-approved live-fire; README explains install, the
technique, the envelope (3–8 s, one signal), and the rights note.
