# Depth Converter — working notes

Local macOS tool: reference clip → temporally-consistent grayscale depth video + a
Seedance Kit, so real choreography and camera motion can be copied onto AI characters
through Seedance 2.0's video-reference channel.

`PLAN.md` is the full build plan and the record of what each gate proved. Read it before
changing anything structural.

## Always use the project-local uv

```bash
./.tools/uv run depthc ...
```

**Never plain `uv`.** The machine-wide Homebrew uv is x86_64 and PyTorch publishes
arm64-only macOS wheels, so it cannot install torch at all. `.tools/uv` is the arm64 one.
It is project-local on purpose — two other projects (`last30days-skill`,
`oside-resolve-mcp`) still run on the x86_64 uv and must not be disturbed.

## The primary workflow: inbox → kit

The folder loop is the main path. The Gradio UI (`depthc ui`) still works, but it is the
secondary surface — it converts one clip at a time and adds a server, a port, and app
bundles for no real gain over doing it here.

1. Peter drops clips into `Inbox/`.
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
7. After Peter fires it: `depthc compare <kit> <downloaded.mp4>` measures adherence the
   Gate-4 way (energy ratio FIRST, zero-lag correlation second, suggested verdict at the
   end) — never re-derive that math by hand. The REGIONS (3×3 placement + mirrored
   control) and PATH (camera direction) lines under it are uncalibrated structure signals:
   read them, quote them in the notes, but they do not vote — when a fire is MIRRORED or
   the path cosine is negative, that is a by-eye finding to record, not a verdict the tool
   made. Then `depthc result <kit> --verdict … --notes
   "…"` with the measured numbers. A generation is the only evidence that says whether any
   of this works and it costs money; unrecorded, the finding has to be re-derived. Ask for
   both.

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
`duration: Math.round(...)`, `durationRange: [4, 15]`). A ref that is not a whole number of
seconds guarantees the generation runs past its end. `--snap` is on by default.

Kits land in **`Outbox/<Tags>_NNN/`** inside the project — not in the home folder.
`Outbox/INDEX.txt` is regenerated on every build. `--archive` moves the source to
`Processed/` on PASS/WARN; a FAIL deliberately stays in `Inbox/`.

Each kit carries `README.txt` (what to do), `binding.txt` (the Variant A phrase alone, one
line, ready to paste), and `sheet.png` (depth contact sheet with timestamps). Those plus
`depth.mp4` are the only files Peter needs to touch. `--variants` adds four refs of the same
beat under `variants/` — contrast-stretched, inverted, **midgrey** (luma capped at 190 so
figures never blow to near-white — the moderation-rejection shape), and **edges** (Canny of
the depth video, experimental) — **exactly one is uploaded per generation**.

## Firing a kit — the rules the live run produced

- **720p or 4K. NEVER 1080p.** 1080p is 0/3 (three seeds, two ref resolutions, two prompt
  phrasings; +31/32/22% excess motion, props dissolving every time; the HD-ref control run
  ignored its ref at r = +0.107). Ref-resolution matching and explicit prompts do NOT fix
  it — the native 1080p lane itself mistreats video refs. 4K adhering is consistent with 4K
  being SR over a lower-res base pass (cascade architecture + its cheaper per-token price),
  so iterate at 720p, master at 4K, never buy 1080p for depth-motion work.
- **Judge by motion ratio first, correlation second.** They disagreed and the ratio was right:
  over-following is what smears bodies and dissolves props, and correlation cannot see it.
- **One identity per body in the ref.** The ref sets the head-count and beats any text lock.
- **Costs:** ~$0.93 for 5 s at 720p with a video ref; $2.29 at 1080p. Rejections cost nothing.

## Laws — do not violate

- **VDA-Small only.** Across DA-V2, VDA and DA3 only the Small checkpoints are Apache-2.0;
  every Base/Large/Giant is CC-BY-NC-4.0. Peter makes commercial films. There is no
  model-size selector by design, and adding one is a licensing violation, not an upgrade.
- **Nothing fires against a paid API automatically.** Live-fire is human-run and
  budget-approved. See `DOCS/LIVE-FIRE.md`.
- **Never commit media.** `Inbox/`, `Processed/` and `Outbox/` contents are gitignored and
  stay that way: source clips carry other people's rights, and a kit's `source.mp4` is the
  original footage. Evidence about a fire belongs in `DOCS/LIVE-FIRE.md`, which is tracked;
  the per-kit `result.json` is local. (Repo went private under OSideMedia on 2026-07-25
  once Gate 4 passed, with the per-repo noreply email
  `261988918+OSideMedia@users.noreply.github.com`.)
- **Never edit `vendor/`.** Upstream is pinned and imported; device and precision handling
  lives in `engine.py`. The per-window hook works by rebinding `tqdm` in their module
  namespace at runtime.
- **fp32 is the default.** fp16/bf16 autocast returns all-NaN on MPS *and* is slower. The
  NaN is sticky until `mps.empty_cache()` and casts silently to an all-black video.

## Attaching a kit in OSIDE — corrected by the live run

Shotlist → pick the shot → go straight to **"make video", skipping create-frame**. A
depth-motion shot is refs-only: the adapter sets `refsWithStartFrame:false`, so a
frame-first shot silently drops the depth ref. Upload `depth.mp4` as the video ref and add
character/location/prop `@Image` refs for the look.

**Paste no binding phrase.** An uploaded video ref has no binding field — bindings are
auto-generated for entity refs only — and the compiler already writes its own line from
`refManifest.ts:117`. Adding Variant A on top is two instructions for one reference.
Describe the LOOK only; the ref carries the motion.

**One identity per body in the ref.** The ref sets the head-count and wins: given one
identity against a two-body ref the model cloned it, in defiance of an explicit anti-twin
text lock. Three for three, the video ref outranks text locks on count and position — treat
it as the blocking.

Phase 5 (automating the attach) is still deferred, but Phase 4 has now passed.

## Gotchas

- Real footage is often matted. `--crop auto` handles it; detection deliberately avoids
  ffmpeg's `cropdetect`, which ate 20px of real picture on a dark edge.
- Conversions run ~20–35s for an 8s clip at 720p on MPS.
- `pytest` is scoped to `tests/`; upstream ships a CUDA-only test file that errors on import.
