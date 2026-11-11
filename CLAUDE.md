# Depth Converter — working notes

Local macOS tool: reference clip → temporally-consistent grayscale depth video + a
Seedance Kit, so real choreography and camera motion can be copied onto AI characters
through Seedance's video-reference channel.

**Two models, two envelopes.** `--dialect seedance-2.0 | seedance-2.5` selects the
constraint record AND the kit prose together (`checks.py DIALECTS`, read from OSIDE's
exported envelope since 2026-09-06; per-model records since a4772f6, 2026-08-24). The
default is **seedance-2.5** (OSIDE's live line) since 2026-09-06. Every law below that was
measured on one model says which — a 2.0 number applied to a 2.5 kit is a wrong number.

`PLAN.md` is the full build plan and the record of what each gate proved. Read it before
changing anything structural.

## Always use the project-local uv

```bash
./.tools/uv run depthc ...
```

**Always `./.tools/uv`.** It is arm64 and version-pinned to this project, so a machine-wide
uv upgrade can never silently re-resolve this build.

**[AMENDED 2026-08-24] The original reason is gone; the rule stands.** This used to read
"never plain `uv` — the machine-wide Homebrew uv is x86_64 and cannot install torch at all."
Homebrew migrated to `/opt/homebrew` (native arm64) on 2026-08-24, so plain `uv` is now
arm64 0.12.5 and *would* resolve torch. Keep using `.tools/uv` on the pinning argument alone.
The two sibling projects named here previously — `last30days-skill` and `oside-resolve-mcp` —
were migrated to arm64 the same day, so there is no longer an x86 venv anywhere to protect.
Both were rebuilt with **zero version drift** (pinned to their exact prior versions) and
re-verified: last30days 4098 passed / 1 pre-existing failure, identical to its x86 baseline
and 1.36× faster; oside-resolve-mcp 41/41, identical.

## The primary workflow: inbox → kit

The folder loop is the main path. The Gradio UI (`depthc ui`) still works, but it is the
secondary surface — it converts one clip at a time and adds a server, a port, and app
bundles for no real gain over doing it here.

1. The owner drops clips into `Inbox/`.
2. `./.tools/uv run depthc inbox` — triage plus a scout pass: what each clip is, whether it
   carries a matte, where its motion actually sits, and which window to cut. ~4s per clip.
3. **Look at `sheet.png` / the source before accepting the proposed window.** The scorer ranks
   HOW MUCH motion a window carries and is blind to HOW MANY things are moving — a second
   figure walking in reads as *more* signal, not as the competing signal it is. Verified on
   CLIP C: the top-ranked window is exactly the stretch the original trim cut away for that
   reason. Overriding the scout is a normal outcome; say why in `--desc`.
4. `./.tools/uv run depthc kit "Inbox/<clip>" --trim <in> <out> --tags "..." --desc "..." --archive`
5. **Look at the depth output**, don't just read the checker. The letterbox bug — matte bars
   becoming the brightest object in the frame at 160/255 against 55 for the real subject —
   was caught by looking at a picture, not by any test. Every kit now ships `sheet.png` for
   exactly this.
6. Report the tier, the motion energy, and the kit path.
7. After the owner fires it: `depthc compare <kit> <downloaded.mp4>` measures adherence the
   Gate-4 way (energy ratio FIRST, zero-lag correlation second, suggested verdict at the
   end) — never re-derive that math by hand. The REGIONS (3×3 placement + mirrored
   control) and PATH (camera direction) lines under it are structure signals no fire has
   scored (PATH has a clay score and FAILED on real footage 2026-11-10, inventing moves on
   locked-off shots, `DOCS/CAMERA-PATH-CALIBRATION.md`): read them, quote them in
   the notes, but they do not vote — when a fire is MIRRORED or the path cosine is negative, that is a by-eye finding to record, not a verdict the tool
   made. Then `depthc result <kit> --verdict … --notes
   "…"` with the measured numbers. A generation is the only evidence that says whether any
   of this works and it costs money; unrecorded, the finding has to be re-derived. Ask for
   both.
   **Name the generation (since 2026-09-06):** both commands take `--generation <id>` —
   the id on the generation's row/URL in OSIDE (a UUID v4; anything else is refused
   naming the grammar). `compare` checks it before the depth pass and carries it into the
   `depthc result` line it prints; `result` writes it as `fires[].generationId` in
   `result.json`. Without the flag the fire is unkeyed, as every fire before that day is.
   That is the fire → generation join (PLAN-SHARED-GENERATION-ID step d):
   `fires[].generationId == generations.id`, plus the content half `kit.json.depth.sha256
   == request_payload.request.refs[].sha256` — every kit built from 2026-09-06 records the SHA-256
   of its depth.mp4 as written, and OSIDE's upload door hashes the uploaded bytes onto the
   reference. The OSIDE half is MERGED on OSIDE's main (`f094e558`, the upload door records
   the clip's sha256 on the reference; `lib/fal/types.ts` carries the field) — it lives
   there, not here. Ask the owner for the id with the download. **The content half is weaker
   than the generation half:** the upload door hashes the bytes and hands the digest back, and
   the generate door keeps whatever the browser sends (it checks the shape only), so a match
   rests on the client, and a reused take has no digest at all. No match proves nothing.

8. A generation with **no kit** (fired from text alone) gets `depthc describe <clip>` —
   the same depth pass, described in prompt words (camera over time with a tempo word,
   framing of the nearest figure, where the motion sits and travels, energy). Read it
   against the prompt. Uncalibrated, description only; it never says "adhered". The
   framing line sizes the nearest iso-depth OBJECT — usually the figure, but a
   foreground sign reads the same, and a depth-only measure cannot tell an insert from
   a landscape (no absolute scale); silence means the gates found no distinct figure. `--rgb` skips the model for a fast look, but those figures are
   not comparable with kit/compare numbers.

9. A reference whose LOOK matters (palette, exposure, ornament density) gets
   `depthc look <clip|stills>` → `Outbox/looks/<name>/look.json` + sheet; the `visualDna`
   block is OSIDE-shaped with hexes filled and words EMPTY — author the words from the
   sheet, never derive them, and carry the per-shot key (a global palette darkened a
   high-key shot on two models, LOOK-DNA-AB 2026-08-16). After a fire,
   `depthc look-compare <look.json> <still|clip>` gives distances (+ drift on a clip);
   quote them, they do not vote. `look.json` also carries a paste-ready `styleConstant`
   block (OSIDE Look tab › Style Constant): hexes filled, prose empty, hints beside —
   author, then paste; there is deliberately no OSIDE import door.
   `depthc look-compare --manifest <pkg>/manifest.json` scores a whole DaVinci package
   (look-travels drift gauge): per-clip distances + RELATIVE outliers = a triage list;
   `--emit-cdl` writes look-cdl.json (one HEURISTIC per-clip ASC CDL) for resolve-mcp
   apply_look — starting balance only, never palette content.

## Motion is the measurement that matters

Duration, file size, resolution and fps are shape assertions that cannot fail if the code is
right. What predicts a bad generation is how much motion survives into the **depth** channel,
measured as mean grey-level change per frame on a fixed 160×90 grid at 24 fps (`motion.py`):

| | energy | what happened |
|---|---|---|
| B sword duel | 4.40 | sustained; the one that behaved |
| C walk-in | 1.69 | steady, unspectacular, fine |
| A prone crawl (001 trim) | 0.94 | generation started late |
| D "slow dolly" | 0.72 | weak, with a dead first second |

(Figures re-measured 2026-07-25 on the v2 aspect-preserving analysis grid — the means agree
with the old fixed 160×90 grid, so shifts against older notes are encode round-trip only.)

**RGB motion is not depth motion.** A's window carried *more* RGB motion than B's (5.98 vs
4.93) and kept 16% of it in depth against B's 87%: a crawl along the ground never changes
distance to the lens, so depth flattens it. Measure both sides at the same fps or the ratio is
meaningless — an earlier pass compared 24 fps depth against 12 fps source and reported 11% vs
52%, wrong numbers with the right ranking.

Output length is requested as a **whole number of seconds, minimum 4** (OSIDE's Ark adapter:
`duration: Math.round(...)`). The window is per model — seedance-2.0: `[4, 15]`;
seedance-2.5: `[4, 30]` (30 s live-confirmed by OSIDE 2026-08-07) — and the ref ceiling
follows it (15 s on 2.0, 30 s on 2.5). A ref that is not a whole number of seconds
guarantees the generation runs past its end. `--snap` is on by default and snaps to the
selected model's window.

Kits land in **`Outbox/<Tags>_NNN/`** inside the project — not in the home folder.
`Outbox/INDEX.txt` is regenerated on every build. `--archive` moves the source to
`Processed/` on PASS/WARN; a FAIL deliberately stays in `Inbox/`.

Each kit carries `README.txt` (what to do), `binding.txt` (the Variant A phrase alone, one
line — for surfaces that HAVE a per-reference binding field: a hand upload in OSIDE does
not, but the **Import kit** door does and fills it from this file, hash-checked
against `kit.json.binding.sha256`, which every kit records since 2026-09-15),
and `sheet.png` (depth contact sheet with timestamps). **Exactly one file goes to OSIDE:
`depth.mp4`, uploaded as the video reference. That is the whole handover.** The
surfaces that say so — this file, `README.md`, `PLAN.md`, `Outbox/README.txt`, `INDEX.txt` and
every kit's `README.txt` — are held to the same words by `tests/test_handover.py`; until 2026-09-06
two of them said "take two files and paste binding.txt" (audit DC-2). `--variants` adds four refs of the same
beat under `variants/` — contrast-stretched, inverted, **midgrey** (luma capped at 190 so
figures never blow to near-white, the shape of the one refused kit on record; never fired), and **edges** (Canny of
the depth video, experimental) — **exactly one is uploaded per generation**.

## Firing a kit — the rules the live run produced

- **On seedance-2.0: 720p or 4K. NEVER 1080p.** 1080p is 0/3 (2026-07-25; three seeds,
  two ref resolutions, two prompt phrasings; +31/32/22% excess motion, props dissolving
  every time; the HD-ref control run ignored its ref at r = +0.107). Ref-resolution
  matching and explicit prompts do NOT fix it — 2.0's native 1080p lane itself mistreats
  video refs. 4K adhering is consistent with 4K being SR over a lower-res base pass
  (cascade architecture + its cheaper per-token price), so on 2.0 iterate at 720p, master
  at 4K, never buy 1080p for depth-motion work.
- **On seedance-2.5 the 0/3 does NOT transfer.** 2.5's separately-wired 1080p lane
  (live 2026-08-17) ADHERED on 2026-08-24: −7 % energy at r = +0.820 against a 720p
  control at +11 % / r = +0.716, same ref, same prompt ($3.83 settled;
  `DOCS/LIVE-FIRE.md` § 2026-08-24). **Suggestive, not an all-clear:** one GENTLE
  reference survived (a lateral walk with background parallax); a demanding ref — the
  sword duel the 0/3 was measured on — is still unfired on 2.5's 1080p. There is **no 4K
  rung on 2.5** as of 2026-08-24, so "master at 4K" names a tier that does not exist
  there; iterate at 720p, and if you fire 1080p run `depthc compare` and record it.
- **Judge by motion ratio first, correlation second.** They disagreed and the ratio was right:
  over-following is what smears bodies and dissolves props, and correlation cannot see it.
- **One identity per body in the ref.** The ref sets the head-count and beats any text lock.
- **Costs:** ~$0.93 for 5 s at 720p with a video ref; $2.29 at 1080p. Rejections cost nothing.

## Laws — do not violate

- **VDA-Small only.** Across DA-V2, VDA and DA3 only the Small checkpoints are Apache-2.0;
  every Base/Large/Giant is CC-BY-NC-4.0. The owner makes commercial films. There is no
  model-size selector by design, and adding one is a licensing violation, not an upgrade.
- **Written for a public repo.** No personal names, machine paths, credentials or provider
  ids in any tracked file, commit message or PR text. Say "the owner". A vendor's rule is
  described, never worked around: a refusal is the provider's call.
- **Nothing fires against a paid API automatically.** Live-fire is human-run and
  budget-approved. See `DOCS/LIVE-FIRE.md`.
- **Never commit media.** `Inbox/`, `Processed/` and `Outbox/` media are gitignored and
  stay that way: source clips carry other people's rights, and a kit's `source.mp4` is the
  original footage. Evidence about a fire belongs in `DOCS/LIVE-FIRE.md`, which is tracked,
  AND in the per-kit text records, which are tracked too since 2026-09-06 (audit DC-4):
  `Outbox/*/kit.json`, `Outbox/*/result.json` and `Outbox/INDEX.txt`. Nothing else in a
  kit folder is — `depth.mp4`, `source.mp4`, `preview.mp4`, `still.png`, `sheet.png`,
  `prompt.txt`, `binding.txt`, `README.txt`, `variants/` never are
  (`tests/test_gitignore.py` proves each with `git check-ignore`). The wholesale
  `Outbox/*` rule had already swallowed every 2026-07-25 kit's `result.json` — paid
  evidence, gone (`DOCS/LIVE-FIRE.md`). (Commits use the per-repo noreply email
  `261988918+OSideMedia@users.noreply.github.com`.)
- **Never edit `vendor/`.** Upstream is pinned and imported; device, precision and memory
  handling live in `engine.py`. The per-window hook works by rebinding `tqdm` in their module
  namespace at runtime, and the backbone's grouping by rebinding one method on the loaded model.
- **fp32 is the default.** fp16/bf16 autocast returns all-NaN on MPS *and* is slower. The
  NaN is sticky until `mps.empty_cache()` and casts silently to an all-black video.

## Attaching a kit in OSIDE — corrected by the live run

Shotlist → pick the shot → **make video**. Upload `depth.mp4` as the video ref and add
character/location/prop `@Image` refs for the look. The create-frame step is the one
instruction that is OPPOSITE on the two models, and the kit README prints the right one:

- **seedance-2.0: go straight to "make video", skipping create-frame.** A depth-motion
  shot is refs-only there: the adapter sets `refsWithStartFrame:false`, so a frame-first
  shot silently drops the depth ref.
- **seedance-2.5: you MAY use create-frame.** `refsWithStartFrame:true` — a start frame
  demotes to a reference image and the reference arrays still ride, so the depth ref is
  kept. Seed does not pin output, so the start frame is the only staging lock available.

**Type no binding phrase — on either path, for different reasons.** On a HAND UPLOAD the
video ref has no binding field at all (bindings are auto-generated for entity refs only)
and the compiler already writes its own line from `refManifest.ts:117`; adding Variant A on
top is two instructions for one reference. Through OSIDE's **Import kit** door — MERGED
2026-09-15 (OSIDE PR #278, `lib/import/depthKit.ts`); the tile beside the upload card in
video mode, one door for both blockout kits and depth kits, which is why it dropped the
word "blockout" from its label — the field exists and the door sets it from `binding.txt`, verifying
`kit.json.binding.sha256` against the file's bytes first. You point that door at the kit
FOLDER (it reads `kit.json`, `binding.txt` and `depth.mp4`); only `depth.mp4` uploads, so
the one-file handover is unchanged. It refuses a FAIL kit, a kit whose manifest lacks
either hash, bytes that do not match what `kit.json` recorded, and a dialect or shape its
model card does not take. Either way, describe the LOOK only; the ref carries the motion.

**One identity per body in the ref.** The ref sets the head-count and wins: given one
identity against a two-body ref the model cloned it, in defiance of an explicit anti-twin
text lock. Three for three, the video ref outranks text locks on count and position — treat
it as the blocking.

Phase 5 (automating the attach) is still deferred, but Phase 4 has now passed.

## Gotchas

- Real footage is often matted. `--crop auto` handles it; detection deliberately avoids
  ffmpeg's `cropdetect`, which ate 20px of real picture on a dark edge.
- Conversions run ~20–35s for an 8s clip at 720p on MPS.
- A default depth pass (8 s of 720p) peaks near 7.3 GB, and `ps` will not show it: GPU memory is not in RSS. Read
  `top -l 1 -pid <pid> -stats mem`. The backbone's grouping (`engine.bound_memory`) is what holds
  it there, and `memory.py` refuses a pass the machine cannot spare. Anything that builds the
  model outside `load_model` needs the same two calls, as `smoke.py` has them. On a long clip or at
  a small `--input-size` the peak is at the end of the pass and on the CPU side, where upstream holds
  several copies of the depth, so a smaller `--input-size` stops helping. A memory test runs one clip
  at a time, under a kill switch below what is free.
- `pytest` is scoped to `tests/`; upstream ships a CUDA-only test file that errors on import.
