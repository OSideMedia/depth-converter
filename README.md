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

The primary path. Three folders in the project, and nothing to hunt for:

```
Inbox/      ← drop reference clips here
Outbox/     ← finished kits land here      (start at INDEX.txt)
Processed/  ← sources file themselves here once converted
```

```bash
# 1. triage what's waiting — what each clip is, where its motion sits, which window to cut
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

```bash
# other commands
./.tools/uv run depthc scout "myclip.mp4" --target 5   # where is the motion? ~4s
./.tools/uv run depthc probe "myclip.mov"              # inspect one file
./.tools/uv run depthc result <kit> --verdict adhered  # record what a fire actually did
./.tools/uv run depthc ui                              # secondary: local web UI
```

### Choosing the window — measured, then vetoed

Between steps 1 and 2 sits the only real judgement call: which beat to keep. `depthc inbox`
and `depthc scout` now answer most of it, by running a deliberately cheap depth pass over the
whole clip and ranking every window of the target length:

```
▄▄▇█▄▃▄▃   overall 2.44
0s:1.9 1s:2.1 2s:3.6 3s:4.7 4s:2.1 5s:1.7 6s:1.8 7s:1.6
best 5s windows:
→ 0.75-5.75s   energy 2.87  head 2.44  balance 0.92  score 2.76   soft (0.50 of best sharpness)
  2.75-7.75s   energy 2.54  head 4.74  balance 0.50  score 1.91
```

The score is the window's motion energy penalised for a **dead first beat** (a generation that
sits still then starts), **imbalance** (a mean hiding a slow build or a beat that dies), a
**cut** inside the window, and any **dead stretch**. Softness is reported and deliberately not
scored — motion blur is what talked us out of the right window in the first place.

**Then look at `sheet.png` and override it when it is wrong.** The scorer ranks how much
motion a window carries and cannot see how many separate things are moving, so a second figure
walking into frame reads as more signal rather than as the competing signal it is. On the
walk-in clip the top-ranked window is exactly the stretch a human had cut away for that
reason. Kit `_003` of that clip is the override; `_002` is what the scorer wanted.

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

### The Outbox

Finished kits land in `Outbox/` — override with `--library` or `$DEPTHC_LIBRARY`. Each kit
is one folder named from its tags plus an auto-incrementing index
(`Sword-duel_Two-fighters_001/`), and `Outbox/INDEX.txt` lists them all, regenerated on
every build so it cannot drift from what is on disk.

There is no database and never will be: the folder tree and the `kit.json` files *are* the
index, so kits stay portable, greppable, and repairable with a text editor.

**From each kit you take exactly two files to OSIDE:**

| file | what to do with it |
|---|---|
| `depth.mp4` | upload as the **video** reference |
| `binding.txt` | paste its one line into that reference's **binding field** |

Every kit also carries a `README.txt` saying precisely that, plus the refs-only rule, the
suggested output length, and the kit's motion profile as a sparkline. Everything else in the
folder — `prompt.txt`, `kit.json`, `source.mp4`, `preview.mp4`, `still.png`, `sheet.png` — is
reference or archive, not for upload.

`sheet.png` is a contact sheet of the depth video with timestamps. **Look at it before
firing**: it is the one thing that catches what the window scorer is blind to, and the
letterbox bug was found by looking at a picture rather than by any test.

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

Duration, file size, resolution and fps are shape checks that cannot fail if the code is
right. The findings that actually predict a bad generation are about **motion**:

| finding | what it catches |
|---|---|
| **motion signal** | mean grey-level change per frame in depth space. FAIL under 0.30 (nothing to follow), WARN under 1.50 (the model free-runs), WARN over 12 (shake, not choreography) |
| **late-start risk** | the first 0.75 s is frozen, so the generation will sit still and then start |
| **dead stretch** | ≥0.75 s of nothing anywhere else — a beat the model has to invent |
| **motion spikes** | a hard cut inside the ref: a second motion signal, and the worst kind |
| **depth retention** | how much of the source's motion survived into depth. Low means the movement runs parallel to the camera plane |
| **motion type** | whether this is a camera-path or a subject-motion reference |
| **output coverage** | the ref length is not exactly requestable as an output length |

For scale, measured on the four reference clips: a sword duel reads **4.42**, a walk-in
**1.71**, a prone crawl whose generation started late **0.95**, a near-static interior
**0.71**.

**RGB motion is not depth motion.** The prone crawl's window carries *more* frame-to-frame
motion than the sword duel in colour (5.98 vs 4.93) and keeps 16% of it in depth against the
duel's 87% — bodies crawling along the ground barely change distance to the lens, so the depth
channel flattens the movement almost entirely. That is the failure behind a generation which
"started very late", and nothing in the old checker had an opinion about it.

### Output length is a whole number of seconds

Output duration is sent as an integer, minimum 4 s. A 3.92 s reference therefore cannot be
matched by any legal request: the generation is at least 4 s, and whatever it does past the
end of the reference is undefined. `depthc kit` snaps trims to that grid by default
(`--no-snap` to opt out), holding the in-point and moving the out-point where the source
allows.

### Recording what a fire actually did

```bash
./.tools/uv run depthc result Sword-duel_Two-fighters_002 \
    --verdict adhered --surface oside --cost 0.31 --notes "followed the lunge exactly"
```

Verdicts are `adhered`, `partial`, `ignored`, `late`, `rejected`. Each fire appends to the
kit's `result.json` and shows up in `depthc library list` and `Outbox/INDEX.txt`. A generation
is the only evidence that says whether any of this works, and it costs money — the library has
no other way to learn.

### Variants — three refs of one beat, one live-fire

`depthc kit --variants` also writes `variants/contrast.mp4` (percentile-stretched, more of the
grey range spent on the subject) and `variants/invert.mp4` (far bright instead of near).
**Upload exactly one per generation** and record which won with `--variant`. Both are
experiments, honestly labelled: no published depth pipeline does percentile clipping, and a
generic video-reference slot has no polarity convention to match.

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
3. **Convert** — run `depthc kit`; it renders locally and lands in `Outbox/`.
4. **Attach** — in OSIDE: Shotlist → pick the shot → go straight to **"make video"**,
   **skipping the create-frame step**. A depth-motion shot is a refs-only shot: the adapter
   never sends references alongside a start frame, so a frame-first shot would silently drop
   the depth ref. In Composer, add your character/location/prop image refs (the look), upload
   `depth.mp4` as the video ref (the motion), and paste the line from `binding.txt`.
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
