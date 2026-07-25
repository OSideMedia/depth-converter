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
2. `./.tools/uv run depthc inbox` — triage: what each clip is, whether it carries a matte,
   whether it needs trimming.
3. **Watch each clip before choosing a trim.** Pull frames with ffmpeg, or use the
   `claude-video-mac:watch` skill. This is the step that justifies doing this in Claude
   Code rather than a slider: pick the single clearest motion beat and say why.
4. `./.tools/uv run depthc kit "Inbox/<clip>" --trim <in> <out> --tags "..." --desc "..." --archive`
5. **Look at the depth output**, don't just read the checker. The letterbox bug — matte bars
   becoming the brightest object in the frame at 160/255 against 55 for the real subject —
   was caught by looking at a picture, not by any test.
6. Report the tier and hand over the kit path.

Kits land in **`Outbox/<Tags>_NNN/`** inside the project — not in the home folder.
`Outbox/INDEX.txt` is regenerated on every build. `--archive` moves the source to
`Processed/` on PASS/WARN; a FAIL deliberately stays in `Inbox/`.

Each kit carries `README.txt` (what to do) and `binding.txt` (the Variant A phrase alone,
one line, ready to paste). Those plus `depth.mp4` are the only files Peter needs to touch.

## Laws — do not violate

- **VDA-Small only.** Across DA-V2, VDA and DA3 only the Small checkpoints are Apache-2.0;
  every Base/Large/Giant is CC-BY-NC-4.0. Peter makes commercial films. There is no
  model-size selector by design, and adding one is a licensing violation, not an upgrade.
- **Nothing fires against a paid API automatically.** Live-fire is human-run and
  budget-approved. See `DOCS/LIVE-FIRE.md`.
- **Local-only repo until Gate 4 passes.** Local commits per gate are wanted; no remote.
  After the technique is proven, push private under OSideMedia with the per-repo noreply
  email `261988918+OSideMedia@users.noreply.github.com`.
- **Never edit `vendor/`.** Upstream is pinned and imported; device and precision handling
  lives in `engine.py`. The per-window hook works by rebinding `tqdm` in their module
  namespace at runtime.
- **fp32 is the default.** fp16/bf16 autocast returns all-NaN on MPS *and* is slower. The
  NaN is sticky until `mps.empty_cache()` and casts silently to an all-black video.

## Attaching a kit in OSIDE (manual, by design)

Shotlist → pick the shot → go straight to **"make video", skipping create-frame**. A
depth-motion shot is refs-only: the adapter sets `refsWithStartFrame:false`, so a
frame-first shot silently drops the depth ref. Then add character/location/prop `@Image`
refs for the look, upload `depth.mp4` as the video ref, and paste **Variant A** from
`prompt.txt` into that ref's binding field — never Variant B, which is for the Ark console
and would collide with the Composer's own ACTIVE REFERENCES block.

Automating this is the Phase 5 ticket, deferred until Phase 4 proves the technique.

## Gotchas

- Real footage is often matted. `--crop auto` handles it; detection deliberately avoids
  ffmpeg's `cropdetect`, which ate 20px of real picture on a dark edge.
- Conversions run ~20–35s for an 8s clip at 720p on MPS.
- `pytest` is scoped to `tests/`; upstream ships a CUDA-only test file that errors on import.
