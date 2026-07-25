# Depth Converter

**Drop in a reference clip → get a Seedance-ready depth motion kit.**

A local macOS tool that converts a short reference video into a temporally-consistent
grayscale **depth video** plus a ready-to-fire **Seedance Kit**, so you can copy real-world
choreography and camera motion onto AI characters and scenes through Seedance 2.0's
video-reference channel.

Nothing is uploaded by this tool. The source clip never leaves the Mac.

---

## Why depth instead of the raw clip

Seedance treats a video reference as a **motion + camera channel** — it reproduces
choreography, walk cadence, dolly/orbit/whip-pan camera paths, and timing. Feeding the
*depth* version of a real clip rather than the clip itself:

- **strips appearance** — no identity, color, texture, or style bleeds from the source;
- **keeps what pose skeletons throw away** — camera path, props, and scene geometry survive
  (a sword, a market stall, a crane move all live in depth);
- **sidesteps the real-face input filter** that frequently rejects raw footage.
  (Community-inferred, not documented — treat as likely, not law.)

## Install

```bash
bash scripts/setup.sh
```

Installs a **project-local arm64 `uv`** into `.tools/`, fetches an arm64 CPython, vendors
[Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) at a pinned
commit, and downloads the VDA-Small checkpoint (116 MB).

> **Why project-local uv?** PyTorch ships **arm64-only** macOS wheels. A Homebrew `uv` under
> `/usr/local` is an x86_64 binary and resolves x86_64 wheels, so `torch` will never install
> through it. Keeping the arm64 uv inside `.tools/` fixes this build without changing `uv`
> for any other project on the machine. Fully undone by `rm -rf .tools/`.

Verify the toolchain at any time:

```bash
./.tools/uv run python -m depthc.smoke
```

## Usage — the inbox loop

The primary path. Drop clips in `Inbox/`, work through them, kits land in the Motion
Library and the sources file themselves into `Processed/`.

```bash
# 1. triage what's waiting — what each clip is, whether it needs trimming
./.tools/uv run depthc inbox

# 2. build a kit from one clip and archive the source
./.tools/uv run depthc kit "Inbox/CLIP C.mp4" \
    --trim 1.0 6.0 \
    --tags "walk toward camera,tracking pull-back" \
    --desc "single clean motion signal before secondary figures enter" \
    --preview --archive

# 3. browse what you've built
./.tools/uv run depthc library list --tag "tracking pull-back"
```

`--archive` moves the source to `Processed/` after a PASS or WARN. A **FAIL deliberately
stays in `Inbox/`** so a rejected clip stays visible.

**Between steps 1 and 2 sits the only real judgement call: which 3–8 s beat to keep.**
Watch the clip first — a contact sheet is enough:

```bash
ffmpeg -i "Inbox/CLIP C.mp4" -vf "fps=1,scale=320:-1,tile=5x2" -frames:v 1 sheet.png
```

On the clip above that showed a second set of figures walking in at 6 s — a competing
motion signal — which is what set the out-point at 6.0.

```bash
# other commands
./.tools/uv run depthc probe "myclip.mov"     # inspect one file
./.tools/uv run depthc ui                      # secondary: local web UI
```

### The web UI and desktop apps — secondary

The Gradio UI is a convenience surface, not the main path. It converts one clip at a time
and adds a server, a port, and two app bundles; the inbox loop above does everything it
does, plus batches, and puts the trim decision somewhere a human (or an agent that can
actually watch the clip) can reason about it.

```bash
bash scripts/build_launchers.sh          # installs into /Applications
```

Creates **Depth Converter.app** (start) and **Depth Stop.app** (shut down), mirroring the
OSIDE Studio pair. Double-click to start: the server comes up in about 3 seconds and a
browser tab opens. Double-click Stop to shut it down.

Runs on port **7861** — chosen to stay clear of OSIDE's dev server on 3000. Pidfile at
`~/.depthc-ui.pid`, log at `~/Library/Logs/depthc-ui.log`. Starting while already running
just focuses the existing tab rather than launching a second server.

Each conversion produces a kit folder containing `depth.mp4`, `still.png`, `prompt.txt`,
`kit.json`, and a copy of the trimmed `source.mp4`.

### The Motion Library

Kits land in `~/Depth_Motion_Library/` — override with `--library` or `$DEPTHC_LIBRARY`.
Each kit is one folder named from its tags plus an auto-incrementing index
(`Sword-fight_Tracking-orbit_001/`). There is no database and never will be: the folder tree
and the `kit.json` files *are* the index, so kits stay portable, greppable, and repairable
with a text editor.

`source.mp4` is stored trimmed but **uncropped**, at source resolution — the crop is
recorded in `kit.json` as a setting, so a kit can always be re-converted at different
settings later.

### The constraint checker

Every kit is graded in three tiers, and the worst finding sets the kit's tier:

| tier | meaning |
|---|---|
| **PASS** | inside the 3–8 s single-signal sweet spot, everything in window |
| **WARN** | it will generate, but adherence is known to degrade — or a human needs to look |
| **FAIL** | Seedance is expected to reject it, or the ref is unusable as motion |

A FAIL exits non-zero, so `depthc kit` is safe to use in a script. Where sources disagree
about a limit the stricter value is enforced (50 MB rather than 200 MB), and the Phase 4
console check is what settles those before any of them become hard failures.

### `prompt.txt` ships two variants — use exactly one

**Variant A** is a bare binding phrase for OSIDE's Composer, which generates its own
`ACTIVE REFERENCES` block from each reference's binding field. **Variant B** is the full
standalone block for the Ark console or any non-OSIDE surface. Pasting Variant B into OSIDE
produces two competing reference blocks — the precise failure the binding field exists to
prevent.

The binding text lives in `depthc/templates/seedance-2.0.txt`, never in Python. A future
dialect is a new template file plus `--dialect`, with no code change.

## The workflow

1. **Curate** — find the clip with the camera move or choreography you want.
2. **Trim** — cut to the **3–8 s sweet spot**, one clear motion signal only.
3. **Convert** — run `depthc kit`; it renders locally and lands in the Motion Library.
4. **Attach** — in OSIDE: Shotlist → pick the shot → go straight to **"make video"**,
   **skipping the create-frame step**. A depth-motion shot is a refs-only shot: the adapter
   never sends references alongside a start frame, so a frame-first shot would silently drop
   the depth ref. In Composer, add your character/location/prop image refs (the look), upload
   `depth.mp4` as the video ref (the motion), and paste the binding phrase from `prompt.txt`.
5. **Fire** — generate.

**Resolution guidance.** The depth ref is always ≤720p-class; it's a motion guide, not
picture, and Seedance's reference-input window caps it anyway. Output resolution is a
separate choice: test at 480p, confirm at 720p, master at 720p/1080p.

**Letterboxed sources are cropped automatically.** A depth model reads a flat black matte
as a *surface* — and specifically as the nearest one in the scene. On a 2.39:1 clip matted
into 16:9, the bars measured 160/255 against 55/255 for the actual picture, putting a bright
slab across the motion channel brighter than the subject it was meant to follow. `--crop
auto` (the default) removes them, so the depth video comes out at the aspect ratio of the
real image. Use `--crop none` to keep the frame as-is, or `--crop W:H:X:Y` to set it by hand.

**Cloud boundary.** The source clip never leaves the Mac. Only the appearance-stripped
`depth.mp4` (zero original pixels, ~2–6 MB) is uploaded, and only at fire time, because the
Ark API takes video refs by URL. The local library is the permanent home; cloud copies are
transient.

## Honest limits

- **Adherence is probabilistic.** Seedance follows video refs well but not deterministically.
  Refs beyond ~10 s and multi-signal clips degrade; fast motion can flicker in the output.
- **Moderation drift.** This technique works partly *because* Seedance blocks real-face
  inputs; filter changes could shift the workflow in either direction.
- **Rights.** Depth strips pixels, not authorship — choreography and music in the source clip
  carry their own rights.
- **Faces in `still.png`.** The still comes from the *original* clip, so it may carry a real
  photographic face. Keep it local as a framing reference; never attach it as an `@Image` ref
  when it does.

## Licensing

This tool ships **VDA-Small only**, which is Apache-2.0. Every Base/Large/Giant Depth
Anything checkpoint is **CC-BY-NC-4.0 (non-commercial)** and is deliberately unavailable
here — there is no model-size selector by design. See `LICENSES/`.
