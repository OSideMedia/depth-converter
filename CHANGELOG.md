# Changelog

Per-kit evidence lives in `Outbox/*/result.json` and `DOCS/LIVE-FIRE.md`; this file is
the code's history. Started 2026-09-06.

## Unreleased

- **Ready for a public repo.** MIT `LICENSE`, `CONTRIBUTING.md` and `SECURITY.md`; a README that
  says what the tool does and whether you need OSIDE (no) before it says anything else; name and
  provider wording cleaned from the tracked docs. `tests/test_public_repo.py` holds the badges, the
  order of the README and the hardening below.
- **`scripts/setup.sh` worked only by accident on a clean machine.** `UV_UNMANAGED_INSTALL` sat on
  `curl`, not on the installer, so a machine with no `.tools/uv` got uv written to the home folder and
  then a FATAL. It now reaches the installer, the uv version is pinned, the checkpoint is checked
  against its SHA-256 before it lands, and the script stops early on anything but Apple Silicon.
- **`torch.load` runs with `weights_only=True`.** Nothing in `pyproject.toml` guaranteed the safe default.
- **Dependencies with known advisories bumped**: brotli, certifi, filelock, setuptools and Pillow.
  Pillow needs an override past gradio's `pillow<13` cap; depthc never imports it directly.
- **The kit README is written for any Seedance surface**, with OSIDE's steps in their own block. A
  `compare` run with no `--generation` no longer prints `--surface oside` in its paste line.
- **The stale-envelope warning no longer assumes OSIDE**, and the age test only holds where someone
  can act on it, so a clone does not go red by the calendar.
- **`MODERATION_CEILING` is `MIDGREY_CEILING`.** Same value. The name described a hope, not the code.

- **`depthc stability <take>`** — boil, flicker and grain on an RGB take, for ranking takes
  of the same shot and finding the bad second. Boil is the band-passed, exposure-matched
  error after a forward-backward flow warp at 480 px, so film grain mostly cancels;
  flicker is signed, so a pump shows as alternation; grain is Immerkær's σ on the flattest
  half of full-resolution frames sampled at 2 fps. `--still` reads the repaint from a start still to
  the first frame, `--after` the seam from a prior take's real last frame (read at its own
  rate, not a 24 fps conform). A join or a clip below the tracking floor says "not scored"
  and never prints a zero. Uncalibrated against a human eye.
- **`read_frames` without an `fps` conform now returns every decoded frame once**
  (`-fps_mode passthrough`). rawvideo's default output sync was CFR at the nominal rate,
  which dropped bunched frames of a VFR source, a take's real last frame among them. Every
  caller that conforms (`fps=`) is unchanged.
- **The camera-path reader is calibrated against a known camera** (`depthc/calibrate.py`,
  `scripts/calibrate_camera_path.py`, `DOCS/CAMERA-PATH-CALIBRATION.md`). Twenty Blender
  clay moves with the exact camera track and Z pass, scored through `describe`'s own
  wording (`describe.path_buckets`, extracted, output unchanged). The shipped reader never
  invented motion (0 false of 140 buckets on every reader) and under-reads slow and
  long-lens moves. Two diagnostics (a textured-pixel median, a similarity fit) are
  measured, neither dominates, and nothing in `depthc` changed because of them. $0: Blender
  and the depth model run locally.

## v0.14.0 — 2026-09-16

32 commits since v0.13.0 (2026-08-23). This file only started on 2026-09-06, so the last
group below recovers the changes that landed between the tag and the file — including a
default that changed and two measurement defects — which no CHANGELOG reader would
otherwise see.

### Since 2026-09-06

- **`kit.json` `binding.sha256`** — the SHA-256 of `binding.txt` as written, required on
  new kits, beside the `depth.sha256` that has bound the clip since 2026-09-06. The kit's
  WORDS are now bound to the kit's clip: an importer that fills a reference's binding field
  from `binding.txt` can prove the two came out of one build. The blockout kit binds both
  halves the same way and that pair caught one kit's legend riding on another kit's clip.
  `kit.verify_binding()` is the reader — FAIL when the bytes differ or the file is gone,
  PASS when they agree, `None` on a kit built before today, which is reported the way a
  missing `depth.sha256` already is and never invented at read time. `depthc compare` and
  `depthc result` both check it, and neither loses evidence to it: the measurement prints
  and the fire is written, and only then does the command exit 1. Additive, so
  `schemaVersion` stays 1 (audit DC-6 ruling).
- **The import door SHIPPED, and it is called "Import kit".** OSIDE PR #278 merged
  2026-09-15 (`lib/import/depthKit.ts`), routing depthc manifests through the Composer's
  existing kit dialog; the tile and the dialog title were renamed from "Import blockout
  kit" to **Import kit**, because one door now takes both blockout kits and depth kits.
  Every surface here that said the door was "being built" / "not merged yet" now says it
  is merged, and none of them use the old name. The substance is unchanged: `binding.txt`
  is still for surfaces that HAVE a per-reference binding field, a hand upload still has
  none, the door still hash-checks `binding.txt` and `depth.mp4` against `kit.json`, and
  you still type no binding phrase on either path. `tests/test_handover.py` holds the new
  wording the way it held the old, and its patterns carry the superseded sentence as their
  counterexample.
- **"Paste no binding phrase in OSIDE" became conditional.** It is true of a HAND UPLOAD —
  an uploaded video reference has no binding field there. OSIDE's **Import kit**
  door does have one and fills it from `binding.txt`, hash-checked. All six surfaces that
  state the rule now name which path the reader is on; two new greps in
  `tests/test_handover.py` hold it.
- **`depthc library index`** — rewrite `Outbox/INDEX.txt` from the kit.json files on disk,
  with no conversion attached. A build already regenerates it, but nothing did when a kit
  folder was deleted by hand, and `write_outbox_index`'s docstring claimed it "can never
  drift" — which three deleted drafts disproved on 2026-09-15. The claim is corrected and
  the delete-by-hand case now has a one-command repair.
- **A fire records the OSIDE generation it became** (PLAN-SHARED-GENERATION-ID step d).
  `depthc compare` and `depthc result` take `--generation <id>` (UUID v4, the id on the
  generation's row/URL in OSIDE; anything else is refused naming the grammar). `result`
  writes it as `fires[].generationId` in the kit's `result.json`; `compare` validates it
  before the depth pass and carries it into the `depthc result` line it prints, because
  `compare` measures and never writes a fire (the verdict is human-only). Without the
  flag a fire is unkeyed, as every fire before this day is.
- **`kit.json` `depth.sha256`** — the SHA-256 of `depth.mp4` as written, required on new
  kits. OSIDE's upload door records the uploaded clip's SHA-256 on the reference
  (`request_payload.refs[].sha256`, lands in OSIDE), so `kit.json.depth.sha256 ==
  refs[].sha256` names the kit a generation rode. The plan's `kitId` was replaced by
  content: a browser upload cannot read a sibling file and `kit.json` has no `id`. Older
  kits load without it; `compare` says so once.
- **`depthc/result_schema.json`** (new) documents `result.json`, which had no schema; held
  by tests, never a runtime gate. Both schemas stay at `schemaVersion` 1 — additive
  changes do not bump (audit DC-6 ruling), noted in each schema's `$comment`.

### A kit lost its media, and this release ships the husk (ow-4ccd14)

- **`Sha256Proof_001`'s media is gone and unrecoverable.** It is the only depth kit ever
  fired through OSIDE's own door ($0.39, generation `441fe671`, verdict **partial**). A
  delete on 2026-09-15 took the folder; the restore returned `kit.json` and `result.json`
  and nothing else, because the rights law keeps Outbox media out of git. `depth.mp4`,
  `source.mp4`, `still.png`, `sheet.png`, `prompt.txt`, `binding.txt` and `README.txt` are
  gone: no Time Machine destination, empty Trash, and a byte-exact sweep of `$HOME` for the
  recorded 922,046-byte clip found nothing. `Inbox/` and `Processed/` hold only their
  READMEs, so the source clip went too.
- **The husk stays, deliberately.** `kit.json` and `result.json` are the last per-fire
  record (cost, generation id, the measured manifest), and two gates take that folder as
  their subject: this repo's `test_every_tracked_result_json_validates` fails on an empty
  DC-4 evidence set, and OSIDE's `check-depth-kit-import` D4 fails "the subject set is
  real" at zero kits. Removing it would redden both and recover nothing.
- **`Outbox/INDEX.txt` now says so on its face.** A kit folder with no `depth.mp4` is
  listed as `MEDIA GONE — record only`, with the two lines that matter: git cannot bring
  the clip back, and nothing in that folder can be fired. Every other line of that file
  tells the reader to open a `README.txt` and upload a `depth.mp4`; for a husk both are
  gone, so an unmarked entry was an instruction to use something that is not there.
- **The consequence in OSIDE, named.** With the husk present, D4 skips it BY NAME and
  reports `D4 ran ZERO assertions against real kits` — loud, green, and certifying nothing.
  OSIDE's real-kit coverage of the import door is therefore zero until a kit with its clip
  exists on this machine. **A rebuild from Pexels 2084066 would restore that coverage but
  would be a NEW kit**, not this one: a re-encode gives different bytes, a different
  `depth.sha256`, and no join to generation `441fe671`. It must never carry this kit's name.

### Surfaces corrected in the release walk

- **`Outbox/INDEX.txt` "cannot drift" was still in `README.md`.** `write_outbox_index`'s
  docstring retracted that claim on 2026-09-15 ("nothing regenerates it when a kit folder is
  removed by hand"), but README.md still read "regenerated on every build so it cannot drift
  from what is on disk". Corrected, with the delete-by-hand case and its one-command repair.
- **`depthc library index` is now in the README's command list**, which named `library list`
  only — a command this release adds and the README did not mention.
- **`PLAN.md`'s attach step still gave the contradiction the DC-2 sweep killed.** Step 4 of
  "The end-to-end workflow" reads "paste the binding block from `prompt.txt`" and "SKIP the
  create-frame step" unscoped — the two things every other surface was corrected on, in the
  file `CLAUDE.md` orders read before any structural change. `test_handover.py` never saw
  it: `PLAN.md` is not in its subject set, and the wording ("binding block from prompt.txt")
  would not trip its greps even if it were. Amended in place, in the file's own dated
  convention. **Growing that gate's subject set is a separate job** and is NOT done here.
- **`CLAUDE.md` pointed at an in-flight OSIDE branch for work that shipped.** "The OSIDE half
  lands in OSIDE (branch `feat/ref-sha256`)" — verified merged on OSIDE's main (`f094e558`);
  it now says so. Same stale-claim class the import-door sweep above exists to kill.

### Before 2026-09-06 — in this release, never in this file

- **The dialect default changed to `seedance-2.5`** (audit DC-5). It was 2.0, and `inbox`
  printed no `--dialect` in its copy-paste hint, so following the tool's own hint built a
  kit graded by 2.0 law — a 15 s ceiling, refs-only, and a 4K rung 2.5 does not have. The
  hint now prints the default whatever it is, and `kit.py`'s second `DEFAULT_DIALECT`
  literal derives from `checks.DEFAULT_DIALECT` instead of repeating it.
- **Per-model constraint records** — a kit records the model it is for, and `--dialect
  seedance-2.0 | seedance-2.5` selects the constraint record and the kit prose together.
  The single hand table had been triangulated for 2.0 on 2026-07-24 and graded every kit
  against it; every drift ran in the WORK-REFUSED direction (a 2.5-legal 30 s ref FAILed as
  "exceeds the 15s ceiling"), and `MAX_PIXELS = 2_086_876` appeared in no vendor source.
- **`checks.DIALECTS` is read from OSIDE's exported envelope** (audit E-3, format
  `oside-seedance-envelope/1`); the hand table is the fallback and is held equal to it.
  Absent file → hand table plus one stderr notice; present but broken → `EnvelopeError`
  naming the key, never a silent fallback.
- **Two measurement defects fixed — both reported a confident wrong number rather than
  failing.** `compare` measured a `--variants` midgrey reference (luma capped at 190) against
  a generation normalised at full range, inflating the ratio by 255/190 before adherence was
  measured; and the `inbox`/`scout` scorer compensated for fps twice.
- **The depth luma reaches libx264 untouched.** `encode_gray` fed rawvideo `gray` and let
  ffmpeg convert; on ubuntu's ffmpeg 6.1.1 that conversion COMPRESSED the luma to 16-235
  while still declaring full range — 0-255 in, 31-218 back — the exact flattened-bands defect
  the function exists to prevent. The yuv420p planes are now built in numpy, so no scaler
  sits in the encode path on any ffmpeg version. Found by CI on its first run, by no test on
  this Mac (ffmpeg 9.0.1).
- **The per-kit text records are tracked** (audit DC-4, the owner's call, its own commit so it
  can be reverted alone): `kit.json`, `result.json` and `Outbox/INDEX.txt` are in git while
  every byte of media stays out. `Outbox/*` wholesale had already swallowed every 2026-07-25
  kit's `result.json` — paid evidence, gone.
- **One truth about the handover across four surfaces** (audit DC-2): `depth.mp4`, uploaded
  as the video reference, is the whole handover. Two surfaces had said "take two files and
  paste binding.txt".
- **The 2.0-only laws are scoped by dialect** (audit DC-3), so an agent obeying CLAUDE.md
  can no longer grade a 2.5 kit by 2.0 law. No measured number was softened.
- **CI: pytest on every push and PR** (audit E-11), with no path filter, plus an
  invisible-code-point scan of every tracked text file. The twelve real-model/vendor tests
  self-skip on the runner with a reason and a declared expected count.
