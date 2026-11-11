[![version](https://img.shields.io/badge/version-0.16.1-blue)](CHANGELOG.md)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![platform](https://img.shields.io/badge/platform-Apple%20Silicon%20Mac-lightgrey)](#requirements)
[![Seedance](https://img.shields.io/badge/Seedance-2.0%20%7C%202.5-e8552d)](#does-it-work-yes-measured)

# Depth Converter

Turns a short reference clip into a Seedance-ready depth video, so you can copy real
choreography and camera moves onto AI characters and scenes.

Point it at a clip and you get a **kit**: a temporally consistent grayscale `depth.mp4` that
keeps the motion and drops the appearance, a contact sheet to check by eye, paste-ready prompt
text, and a PASS, WARN or FAIL grade. It runs on your Mac and calls no paid API. You take
`depth.mp4` to Seedance yourself.

## What it does

- **Converts a clip to depth** with Video Depth Anything Small, locally on your Mac's GPU.
  An 8 s clip at 720p takes about 20–35 s.
- **Finds the window worth cutting** by ranking every window of a clip for motion, dead
  first beats and cuts, then leaves the final call to you.
- **Grades every kit** against Seedance's limits for the model you pick, and on what actually
  predicts a bad generation: how much motion survives into depth.
- **Packages the handover**: `depth.mp4`, prompt text for your surface, a contact sheet and a
  record of how the kit was made.
- **Measures what a generation did** by running it back through the same depth pass and
  comparing its motion with the reference's.
- **Reads clips with no kit**: the camera move and motion of a generated clip, its boil and
  flicker, or the palette and key of a reference.
- **Keeps the evidence.** Each fire's verdict, cost and notes land beside the kit, so the next
  choice rests on what happened.

## Do I need OSIDE to use this?

**No.** OSIDE Studio is the filmmaking app this tool grew up beside, but nothing here imports
it, calls it or needs an account with it. The input is any video file ffmpeg can read, the
output is a plain folder, and what you hand to Seedance is an ordinary MP4.

The smallest run that does something useful, after [Install](#install):

```bash
./.tools/uv run depthc kit my-clip.mp4 --trim 1 6 --tags "walk to camera"
```

That writes `Outbox/Walk-to-camera_001/`. Upload its `depth.mp4` as the video reference
wherever you use Seedance, and paste Variant B from its `prompt.txt`.

A few extras speak OSIDE, and you can ignore them all:

| extra | without OSIDE |
|---|---|
| `--generation <id>` on `compare` and `result` | leave it off; `--output` and `--notes` record the same fire |
| `visualDna` and `styleConstant` blocks in `look.json` | the measured palette and key sit beside them |
| `look-compare --manifest` | it reads an [`oside-davinci/v1`](https://github.com/OSideMedia/oside-resolve-mcp) package, a plain folder and JSON file |
| `depthc/seedance-envelope.json` | a JSON snapshot of Seedance's limits, exported from OSIDE and committed here. When it is over 180 days old `depthc` warns and keeps using it, so check the limits against the vendor's docs |
| `scripts/calibrate_camera_path.py` | a developer check that needs `oside-blender-mcp` and Blender; `depthc` never calls it |

**One limit on the evidence.** Every generation recorded in
[`DOCS/LIVE-FIRE.md`](DOCS/LIVE-FIRE.md) went through OSIDE to Seedance on Ark. The file you
upload is the same anywhere, but no fire recorded in this repo used Variant B directly.

## Requirements

- **An Apple Silicon Mac on macOS 14 or newer.** `setup.sh` stops on anything else. The model runs
  on the GPU through PyTorch's MPS backend.
- **About 8 GB of free memory for a default depth pass** (8 seconds of 720p), and by calculation
  about 14 GB for 30 seconds, because the clip's own frames add up. Before each pass the
  tool reads what your Mac has free right now and keeps 1.5 GB back for everything else. If the
  pass won't fit, it stops with a plain message before loading the model, and names a smaller
  `--input-size` when one would fit. A smaller size gives a coarser depth picture, so the tool never
  picks one for you. `DEPTHC_MEMORY_GB` sets the limit by hand. The 7.1 to 7.3 GB peak of a default
  pass was measured on one 24 GB Mac, and the estimate was checked there at eight picture sizes. For
  other machines and clip shapes it is an estimate, with a hard limit on GPU memory behind it. The
  check runs once per pass, so another app filling memory during a pass can still slow your Mac.
- **ffmpeg and ffprobe on your `PATH`.** Setup does not install them: `brew install ffmpeg`.
- **git, curl and a network connection for setup,** which downloads about 1.1 GB. After that,
  everything runs locally.
- **Your own access to Seedance** if you want to generate anything. Building and measuring kits
  needs none, and this tool never fires a generation.
- **The right to use your source clip.** See [Honest limits](#honest-limits).

## Install

```bash
git clone https://github.com/OSideMedia/depth-converter.git
cd depth-converter
bash scripts/setup.sh
./.tools/uv run python -m depthc.smoke
```

Everything lands inside the project folder: a pinned `uv` in `.tools/`, Python and the
locked dependencies in `.venv/`, Video Depth Anything at a pinned commit in `vendor/`, and the
116 MB checkpoint in `checkpoints/`, verified against its SHA-256. Deleting the folder undoes it,
apart from the Python build and download cache `uv` keeps in your home folder (`~/.local/share/uv`
and `~/.cache/uv`). Always run through `./.tools/uv`.

## Quick start

```bash
./.tools/uv run depthc scout my-clip.mp4 --target 5     # where is the motion? which window?
./.tools/uv run depthc kit my-clip.mp4 --trim 1 6 --tags "walk to camera"
open Outbox/Walk-to-camera_001/sheet.png                # look at the depth before you fire it
cat Outbox/Walk-to-camera_001/README.txt                # the steps for this kit
```

After you fire it, measure and record what came back:

```bash
./.tools/uv run depthc compare Walk-to-camera_001 ~/Downloads/take.mp4
./.tools/uv run depthc result Walk-to-camera_001 --verdict adhered --surface ark --cost 0.93
```

`depthc --help` lists every command.

A fresh clone already lists two kits in `Outbox/INDEX.txt`. They are text records from this
repo's own runs, with no media, and show as `MEDIA GONE — record only`. Each build rewrites
`INDEX.txt`, so it will show as modified in `git status`.

## Why depth instead of the raw clip

Seedance treats a video reference as a **motion + camera channel**: it reproduces choreography,
walk cadence, dolly, orbit and whip-pan camera paths, and timing. Feeding it the *depth* version
of a clip rather than the clip itself:

- **strips appearance**, so no identity, colour, texture or style bleeds from the source;
- **keeps what pose skeletons throw away**: camera path, props and scene geometry survive
  (a sword, a market stall and a crane move all live in depth);
- **carries no face.** A depth video holds shape and motion with no face or identity in it, so
  nobody's likeness rides into the generation.

## Usage: the inbox loop

The primary path. Three folders in the project, and nothing to hunt for:

```
Inbox/      ← drop reference clips here
Outbox/     ← finished kits land here      (start at INDEX.txt)
Processed/  ← sources file themselves here once converted
```

```bash
# 1. triage what's waiting: what each clip is, where its motion sits, which window to cut
./.tools/uv run depthc inbox

# 2. build a kit from one clip and archive the source
./.tools/uv run depthc kit "Inbox/CLIP C.mp4" \
    --trim 1.0 6.0 \
    --tags "walk toward camera,tracking pull-back" \
    --desc "single clean motion signal before secondary figures enter" \
    --preview --archive

# 3. browse what you've built
./.tools/uv run depthc library list --tag "tracking pull-back"
./.tools/uv run depthc library index                   # rewrite Outbox/INDEX.txt from disk
```

`--archive` moves the source to `Processed/` after a PASS or WARN. A **FAIL deliberately
stays in `Inbox/`** so a rejected clip stays visible.

```bash
# other commands
./.tools/uv run depthc scout "myclip.mp4" --target 5   # where is the motion? ~4s
./.tools/uv run depthc probe "myclip.mov"              # inspect one file
./.tools/uv run depthc result <kit> --verdict adhered  # record what a fire actually did
./.tools/uv run depthc stability "take.mp4"            # boil and flicker of a generated take
./.tools/uv run depthc ui                              # secondary: local web UI
```

### Choosing the window: measured, then vetoed

Between steps 1 and 2 sits the only real judgement call: which beat to keep. `depthc inbox`
and `depthc scout` answer most of it, by running a deliberately cheap depth pass over the
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
scored: motion blur is what talked us out of the right window in the first place.

**Then look at `sheet.png` and override it when it is wrong.** The scorer ranks how much
motion a window carries and cannot see how many separate things are moving, so a second figure
walking into frame reads as more signal rather than as the competing signal it is. On one
walk-in clip the top-ranked window was exactly the stretch a human had cut away for that reason.

### The web UI and desktop apps: secondary

The Gradio UI is a convenience surface, not the main path. It converts one clip at a time
and adds a server, a port and two app bundles; the inbox loop above does everything it
does, plus batches, and puts the trim decision somewhere a human can reason about it.

```bash
bash scripts/build_launchers.sh          # installs into /Applications
```

Creates **Depth Converter.app** (start) and **Depth Stop.app** (shut down). Double-click to
start: the server comes up in about 3 seconds and a browser tab opens. Double-click Stop to
shut it down.

It binds to `127.0.0.1:7861` only, with Gradio's analytics off. Pidfile at
`~/.depthc-ui.pid`, log at `~/Library/Logs/depthc-ui.log`. Starting while already running
just focuses the existing tab rather than launching a second server.

## Using a kit

From each kit you take **one file**: `depth.mp4`, uploaded as the **video** reference. That is
the whole handover. Everything else in the folder (`prompt.txt`, `kit.json`, `source.mp4`,
`preview.mp4`, `still.png`, `sheet.png`) is reference or archive, not for upload.

On any surface:

- **Describe the look in your prompt, not the motion.** The reference carries the motion.
- **Add one identity per body in the ref.** The ref sets the head-count and beats any text
  lock: given one identity against a two-body ref, the model cloned that identity.
- **Trim to the 3–8 s sweet spot,** one clear motion signal only.
- **Ask for the output length the kit suggests.** Its `README.txt` prints it.

### On the Ark console or another surface

Paste **Variant B** from `prompt.txt`: the full standalone block, and nothing else. Variant A
(the same text as `binding.txt`) is the bare binding phrase. `binding.txt` is for surfaces
that HAVE a per-reference binding field: if yours has one, paste it there instead of
Variant B.

### In OSIDE

Shotlist → pick the shot → **make video**. In the Composer, add your character, location and
prop image refs (the look) and upload `depth.mp4` as the video ref (the motion).

Do **not** type a binding phrase there, on either path, for two different reasons. On a
**hand upload**, the video reference has no binding field in the Composer at all, and the
compiler already writes its own line for it; adding one on top is two instructions for the
same reference. Through OSIDE's **Import kit** door, the tile beside the upload card in video
mode, which has taken depth kits since 2026-09-15, the field *is* there and the door fills it
from `binding.txt` itself, verifying it against the `binding.sha256` in `kit.json` first. On
that path you point the door at the whole kit folder (it reads `kit.json`, `binding.txt` and
`depth.mp4` locally) and `depth.mp4` is still the only file that uploads.

The create-frame step is per model, and the kit's `README.txt` prints the right rule: on
**seedance-2.0** skip it (a depth shot is refs-only, and a start frame silently drops the depth
ref); on **seedance-2.5** you may use it (the frame demotes to a reference image and the refs
ride).

Pasting Variant B into OSIDE produces two competing reference blocks, which is the precise
failure the binding field exists to prevent.

### What is in a kit

Finished kits land in `Outbox/` (override with `--library` or `$DEPTHC_LIBRARY`). Each is one
folder named from its tags plus an auto-incrementing index (`Sword-duel_Two-fighters_001/`),
and `Outbox/INDEX.txt` lists them all.

There is no database and never will be: the folder tree and the `kit.json` files *are* the
index, so kits stay portable, greppable and repairable with a text editor.

A **build** is the only event that refreshes `INDEX.txt`, so deleting a kit folder by hand
leaves it listing a kit that is no longer there. `depthc library index` is that rewrite with no
conversion attached. A kit folder whose `depth.mp4` is gone is listed as **`MEDIA GONE —
record only`**: `kit.json` and `result.json` are tracked in git and the media never is, so a
restore brings back the record and never the clip.

Every kit carries a `README.txt` with these steps, the suggested output length and the kit's
motion profile as a sparkline.

`sheet.png` is a contact sheet of the depth video with timestamps. **Look at it before
firing**: it is the one thing that catches what the window scorer is blind to, and the
letterbox bug was found by looking at a picture rather than by any test.

`source.mp4` is stored trimmed but **uncropped**, at source resolution. The crop is recorded in
`kit.json` as a setting, so a kit can always be re-converted at different settings later.

### `prompt.txt` ships two variants: use exactly one

**Variant A** is a bare binding phrase for a surface with a per-reference binding field.
**Variant B** is the full standalone block for the Ark console or any other surface.

The text lives in `depthc/templates/seedance-2.0.txt` and `seedance-2.5.txt`, never in Python.
A future dialect is a new template file plus a record in `checks.py`.

## The constraint checker

Every kit is graded in three tiers, and the worst finding sets the kit's tier:

| tier | meaning |
|---|---|
| **PASS** | inside the 3–8 s single-signal sweet spot, everything in window |
| **WARN** | it will generate, but adherence is known to degrade, or a human needs to look |
| **FAIL** | Seedance is expected to reject it, or the ref is unusable as motion |

A FAIL exits non-zero, so `depthc kit` is safe to use in a script. Limits are per model:
`--dialect seedance-2.0` or `seedance-2.5` (the default) selects the record and the kit's prose
together. On 2.0 a reference can run to 15 s, on 2.5 to 30 s, and both take up to 200 MB
(limits as exported 2026-10-26).

Duration, file size, resolution and fps are shape checks, and a kit rarely trips them (a very wide
source at `--resolution 480p` can fall under the pixel floor). The findings that actually predict a
bad generation are about **motion**:

| finding | what it catches |
|---|---|
| **motion signal** | mean grey-level change per frame in depth space. FAIL under 0.30 (nothing to follow), WARN under 1.50 (the model free-runs), WARN over 12 (shake, not choreography) |
| **late-start risk** | the first 0.75 s is frozen, so the generation will sit still and then start |
| **dead stretch** | ≥0.75 s of nothing anywhere else: a beat the model has to invent |
| **motion spikes** | a hard cut inside the ref: a second motion signal, and the worst kind |
| **depth retention** | how much of the source's motion survived into depth. Low means the movement runs parallel to the camera plane |
| **motion type** | whether this is a camera-path or a subject-motion reference |
| **output coverage** | the ref length is not exactly requestable as an output length |

For scale, measured on four reference clips: a sword duel reads **4.40**, a walk-in **1.69**, a
prone crawl whose generation started late **0.94**, a near-static interior **0.72**.

**RGB motion is not depth motion.** The prone crawl's window carries *more* frame-to-frame
motion than the sword duel in colour (5.98 vs 4.93) and keeps 16% of it in depth against the
duel's 87%: bodies crawling along the ground barely change distance to the lens, so the depth
channel flattens the movement almost entirely. That is the failure behind a generation which
"started very late".

### Output length is a whole number of seconds

Output duration is sent as an integer, minimum 4 s. A 3.92 s reference therefore cannot be
matched by any legal request: the generation is at least 4 s, and whatever it does past the
end of the reference is undefined. `depthc kit` snaps trims to that grid by default
(`--no-snap` to opt out), holding the in-point and moving the out-point where the source
allows.

### Letterboxed sources are cropped automatically

A depth model reads a flat black matte as a *surface*, and specifically as the nearest one in
the scene. On a 2.39:1 clip matted into 16:9, the bars measured 160/255 against 55/255 for the
actual picture, putting a bright slab across the motion channel brighter than the subject it
was meant to follow. `--crop auto` (the default) removes them, so the depth video comes out at
the aspect ratio of the real image. Use `--crop none` to keep the frame as-is, or
`--crop W:H:X:Y` to set it by hand.

### Variants: five refs of one beat, one fire each

`depthc kit --variants` also writes four processed versions of the same beat under
`variants/`:

- **contrast.mp4**: percentile-stretched (2–98%), more of the grey range spent on the subject
- **invert.mp4**: polarity flipped, far bright instead of near
- **midgrey.mp4**: luma capped at 190, so figures stay mid-grey instead of blowing to white.
  Made after the one kit on record that was refused at input was two depth silhouettes blown to
  near-white (see [below](#does-it-work-yes-measured)). It has never been fired, and it is not
  a way round a provider's rules: a refused reference is a refused reference.
- **edges.mp4**: Canny edges of the *depth* video (never the source: an edge map of the
  original pixels would carry face contours back into the upload). Outlines only; may read
  as a style cue as much as motion.

**Upload exactly one per generation** and record which won with `--variant`. All four are
experiments, honestly labelled: no published depth pipeline does percentile clipping or edge
refs, and a generic video-reference slot has no polarity convention to match.

## Measuring what a fire actually did

```bash
./.tools/uv run depthc compare Sword-duel_Two-fighters_002 ~/Desktop/take.mp4
```

Runs the generated video back through the same depth pipeline and compares its motion envelope
with the reference's, computed identically every time. The report leads with the **energy
ratio** (over-following is what smears bodies and dissolves props, and correlation cannot see
it: adherent runs sit at +6–8%, the three failures at 1080p on 2.0 ran +22–32%), then the
**zero-lag correlation** with ±0.5/±1.0 s controls (a real alignment peaks at zero and collapses
off-zero), then the camera share on both sides. The ratio is computed on the overlapping
duration only: the generation's decode is trimmed to the ref's length before inference, so
neither an excluded tail's motion nor its depth extremes can poison the numbers.

The generation gets the same geometry courtesies the kit build gave the reference: matte bars
are detected and stripped, and if the framing still differs from the ref's (a 2.39:1 kit fired
into a 16:9 output), the report flags the ratio as framing-biased and lets the timing correlation
carry the verdict. The frame-share energy metric moves ~34% on that aspect change alone, more
than the entire adhered-vs-failed band. If you fired a variant rather than the base ref, say so:
`--variant edges` compares against the file that was actually uploaded.

Below the envelope come two **structure** lines for what a full-frame mean cannot see: a
left→right crossing and its mirror image draw the same sparkline and correlate at r = 1.0.
**REGIONS** repeats the zero-lag correlation cell by cell on a 3×3 grid, weighted by where the
ref's energy actually sits, and runs the same check against the horizontally *mirrored*
generation: when the mirror wins clearly, the report says `MIRRORED — moves the other way`.
**PATH** names the camera move on both sides in camera terms (`pans left, pushes
in`) and, when the ref is a camera-path ref, scores direction agreement as a cosine; on a body
ref it is named but not judged. Neither moves the suggested verdict: no measured fire has been
scored on them yet, so they inform your read. `--no-regions` and `--no-camera-path` switch them
off. The camera reader was scored against a known Blender camera on clay, and then on real
footage, where it failed: reading the depth picture, it named 9 of 12 clean moves and invented a
move on 4 of 5 locked-off shots with people or a train passing through. Since then the camera is
read from the picture instead (the kit's `source.mp4` on the reference side), with whatever moves
on its own dropped first. That reader was scored on 40 clips it had never seen and failed too: 7 of 10
clear moves named, and a move invented on 2 of 7 locked-off shots, both with a subject filling the
frame. On phone-on-a-stand dance videos it named a move on 7 of 11 windows. Treat the camera words as a hint to check by eye,
most of all when they name a move on a shot that looks still. The scores and the method are in
[`DOCS/CAMERA-PATH-CALIBRATION.md`](DOCS/CAMERA-PATH-CALIBRATION.md).

It ends with a *suggested* verdict (`adhered`, `partial`, `ignored`, `late`) and a ready-to-paste
`depthc result` command. The suggestion is a heuristic calibrated on the measured fires: watch
the clip, then record what actually happened. In OSIDE, `--generation <id>` names the generation
(a UUID v4 from its row or URL; a non-UUID is refused before the depth pass). `compare` never
writes the fire itself, so the id rides into the pasted `depthc result` line.

### Describing a clip that has no kit

```bash
./.tools/uv run depthc describe ~/Desktop/take.mp4          # depth pass, same grid as a kit
./.tools/uv run depthc describe ~/Desktop/take.mp4 --rgb    # fast, no model, picture itself
```

`compare` needs the kit that drove a generation. A shot fired from text alone has none, so
`describe` runs the clip through the same depth pass and says what moved and which way, in prompt
words: the camera over time (`pans left 0–1.5s; pushes in 2–3.5s`, or `locked off`), where the
strongest motion sits and travels (`frame-left → frame-right`, `starts 1s in`, hottest cell), and
the energy shape. So "did it follow the prompt's direction?" becomes: read the prompt, read this,
do they agree. `--json` for a machine-readable block. Description only: it cannot tell whether
the moving thing is the one the prompt named, which stays a by-eye call. The radial push/pull term
is recorded only when all four quadrants of the frame agree, so a subject walking toward the lens
is not booked as a camera move.

The camera words are the least reliable part. They are read from the picture, on the depth path
and with `--rgb` alike. Read from the depth, as they were until 2026-11-11, they invented moves
on locked-off shots with people passing and missed slow push-ins; the picture read has not been
scored on unseen footage yet, so check them against the picture
(see [`DOCS/CAMERA-PATH-CALIBRATION.md`](DOCS/CAMERA-PATH-CALIBRATION.md)).

### Reading how much a take boils and flickers

```bash
./.tools/uv run depthc stability ~/Desktop/take.mp4
./.tools/uv run depthc stability ~/Desktop/take.mp4 --still start-frame.png --after previous-take.mp4
```

Warps each frame back along the optical flow and reports what still changed: **boil** (structure
re-rolling: hair, foliage, fabric), **flicker** (exposure breathing) and **grain**. `--still`
reads the repaint from a start frame to the first frame, and `--after` the seam from a previous
take's last frame. It reads the picture, not depth. Use it to rank takes of the same shot and find
the bad second: it is not a grade, and is not calibrated against a human eye.

### Distilling a reference's look: palette and key, not motion

```bash
./.tools/uv run depthc look ~/Desktop/reference.mp4                     # clip → shots → keyframes
./.tools/uv run depthc look ref/palace.jpg ref/gate.jpg ref/stair.jpg   # or stills you already have
./.tools/uv run depthc look-compare Outbox/looks/<name>/look.json ~/Desktop/still.png
./.tools/uv run depthc look-compare Outbox/looks/<name>/look.json ~/Desktop/take.mp4   # + drift over time
```

A kit copies a reference's MOTION through the depth channel; `look` copies its LOOK through words.
It splits a clip into shots at colour cuts, measures the middle frame of each (6-colour palette by
k-means in Lab with area shares, luma mean and dark/bright shares, warmth, chroma, detail), labels
every shot's **key** (high, mid, low), and writes `look.json`, a contact sheet and the keyframes to
`Outbox/looks/<name>/`.

The hexes are measured and the WORDS (lighting, materials, era, forbidden list) are left empty on
purpose: a mechanical hex-to-name step is too coarse, having rendered a measured ivory and slate
palette as "light orange, light orange, muted orange". Read the sheet, name the colours and
materials, and keep the key per shot. `look.json` also carries `hints` beside each swatch, so you
fill blanks from evidence instead of inventing structure.

Why the key rides per shot: in a 2026-08-16 test, a look distilled from four reference keyframes
ran against a generic prompt on two image models. The look won 3 of 4 subjects on both, and the
one it lost, it lost on both for the same reason: one global palette pushed a high-key shot dark.
`look-compare` shows exactly that. A still gives one row of distances (palette, ΔL*, Δdark, Δwarmth,
Δchroma, Δdetail, the candidate's key, the nearest keyframe); a clip gives one row per sampled
second and a DRIFT line, first frame to last. Distances, not verdicts.

```bash
./.tools/uv run depthc look-compare --manifest <package>/manifest.json            # the package's own look block
./.tools/uv run depthc look-compare look.json --manifest <package>/manifest.json  # or a measured look
```

`--manifest` scores every `videos/*.mp4` of an `oside-davinci/v1` package against the look, one row
per clip, then flags OUTLIERS **relative to the package** (beyond 1.5 σ from the package mean on
palette, ΔL*, Δb* or Δchroma, and only when the spread is not trivial). That is a triage list, not
a verdict.

`--emit-cdl` additionally writes `look-cdl.json` beside the manifest: one ASC CDL per clip that
pulls that clip's measured mean ΔL*, Δb* and Δchroma back toward the look. It is a heuristic, with
linear gains fitted once on one real Resolve render of 8 clips, and labelled so in the file: a
starting balance for the colourist, never a creative grade and never a palette fix.
[OSIDE Resolve MCP](https://github.com/OSideMedia/oside-resolve-mcp)'s `apply_look` reads this file
and puts each CDL on node 1 of its clip.

### Recording what a fire actually did

```bash
./.tools/uv run depthc result Sword-duel_Two-fighters_002 \
    --verdict adhered --surface ark --cost 0.93 --notes "followed the lunge exactly"
```

Verdicts are `adhered`, `partial`, `ignored`, `late`, `rejected`. Each fire appends to the kit's
`result.json` (shape: `depthc/result_schema.json`) and shows up in `depthc library list` and
`Outbox/INDEX.txt`. A generation is the only evidence that says whether any of this works, and it
costs money, so the library has no other way to learn.

In OSIDE, a fire also joins the generation it became, on two keys:

- `result.json` `fires[].generationId` equals OSIDE's generation id, written by `--generation <id>`.
  Without the flag the fire is unkeyed; the key is absent, never null.
- `kit.json` `depth.sha256` equals the hash OSIDE's upload door records on the reference. Every kit
  built since 2026-09-06 records the SHA-256 of `depth.mp4` exactly as written, and a browser upload
  cannot read a sibling `kit.json`, which is why the join is the file's content and not a kit id.

## Resolution

The depth ref defaults to 720p-class or below: it is a motion guide, not picture.
`--resolution source` keeps a larger source as it is, up to the pixel ceiling the checker enforces.
Output resolution is a separate choice, and the rule is per model. The kit's `README.txt` prints
the right one.

**On Seedance 2.0: test at 480p, confirm at 720p, master at 720p or 4K. Never 1080p.** That is the
one hard operational rule the 2026-07-25 runs produced, and it cost $6.87 to find:

| tier | fires | motion vs reference | props |
|---|---|---|---|
| 720p | 3 | +8% | ✅ |
| **1080p** | **3** | **+31%, +32%, +22%** | ❌ dissolve, shrink, vanish |
| 4K | 1 | +6% | ✅ |

**1080p was 0 for 3, across three seeds, two reference resolutions and two prompt phrasings.** The
third run was the controlled experiment: a 1920×1080 depth ref of the identical window with "fighting
with swords" stated outright in the prompt. It still ignored the reference (r = +0.107 at zero lag,
no peak at any lag), ran +22% hot, invented camera movement and dissolved the swords by 3.5 s.

4K adhering does not contradict this: the Seedance family is documented (1.0 technical report) as a
cascade of base generation plus a diffusion refiner, and Ark prices 4K ~43% cheaper per token than
1080p, which fits 4K being super-resolution over a lower-res base pass. That is an inference, not a
measurement.

**On Seedance 2.5 the 0 for 3 does not transfer.** Its separately wired 1080p lane adhered on
2026-08-24: −7% energy at r = +0.820 against a 720p control at +11%, r = +0.716, same ref and same
prompt ($3.83). That is one gentle reference, a lateral walk, so it is suggestive and not an
all-clear: the sword duel is unfired on 2.5's 1080p. 2.5 has no 4K rung as of that date. Iterate at
720p, and if you fire 1080p, run `depthc compare` and record it.

## Where your files go

The source clip never leaves the Mac. The only thing that goes anywhere is the appearance-stripped
`depth.mp4` (zero original pixels, ~2–6 MB), when you upload it to fire. This tool has no upload
code and no telemetry.

## Does it work? Yes, measured

Fired by hand on 2026-07-25 against Seedance 2.0 on Ark, through OSIDE. Full protocol and results in
[`DOCS/LIVE-FIRE.md`](DOCS/LIVE-FIRE.md).

Each generated video was run **back through this converter** and its depth-space motion profile
correlated against the reference's, bucket for bucket:

| run | cast | adherence |
|---|---|---|
| sword duel | one identity | **r = +0.950** at zero lag |
| sword duel | two identities | **r = +0.916** at zero lag |
| sword duel | two identities + a custom location | **r = +0.926** at zero lag |

Each collapses to ~0 at ±0.5 s, which is what makes it an alignment rather than two clips that
happen to be busy. **Timing transferred beat for beat, with no delay.** The swords survived as props.
Appearance came entirely from the image references, and a custom location rode alongside: two
identities, a location and the depth video, four channels holding at once.

**Judge adherence by motion ratio first, correlation second.** They disagreed on the 4K run and the
ratio was right: ranked by correlation, 4K sits barely above the broken 1080p; ranked by how much
motion each added, 4K and 720p are together and 1080p is the obvious outlier, which is what a human
watching them said without seeing either number.

Three things the runs taught:

- **The ref outranks the prompt.** Given one identity against a two-body reference, the model cloned
  that identity rather than inventing a stranger, in direct defiance of an explicit "do not generate
  duplicate avatars" lock.
- **One kit was refused at input.** It held two depth silhouettes, prone and overlapping, filling the
  frame, while upright, separated figures from the same pipeline passed. Depth strips clothing, so
  every figure reads as a body shape: keep that in mind when you choose a clip.
- **Budget 3× the obvious guess**: $0.93 for 5 s at 720p with a video reference.

## Honest limits

- **Adherence is probabilistic.** Seedance follows video refs well but not deterministically. Refs
  beyond ~10 s and multi-signal clips degrade; fast motion can flicker in the output.
- **Provider rules can change.** Seedance's handling of faces and references may shift, in either
  direction, and this tool has no say in it. Follow your provider's terms; a refusal is the provider's
  call, and nothing here tries to get around one.
- **Rights.** Depth strips pixels, not authorship: choreography and music in the source clip carry
  their own rights, and a person in it has their own. Whether you may use a clip is yours to establish.
- **Faces in `still.png`.** The still comes from the *original* clip, so it may carry a real
  photographic face. Keep it local as a framing reference, and never attach it as an `@Image` ref
  when it does.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Two things worth knowing before you open a PR: a model-size
selector will be declined, because it is a licensing violation rather than an upgrade, and media never
goes in git. To report a security problem, see [SECURITY.md](SECURITY.md).

## Licence

MIT, see [LICENSE](LICENSE). Third-party notices are in [`LICENSES/`](LICENSES/).

This tool ships **VDA-Small only**, which is Apache-2.0. Every Base, Large and Giant Depth Anything
checkpoint is **CC-BY-NC-4.0 (non-commercial)** and is deliberately unavailable here.

Seedance is a ByteDance product. This project is an independent tool and is **not affiliated with,
endorsed by, or supported by ByteDance**.
