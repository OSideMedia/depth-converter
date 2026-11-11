# Changelog

Per-kit evidence lives in `Outbox/*/result.json` and `DOCS/LIVE-FIRE.md`; this file is
the code's history. Started 2026-09-06.

## Unreleased

- **`depthc result` no longer writes over a fire log it cannot read.** If a kit's `result.json` was
  there but unreadable (cut off, saved as UTF-16, or JSON of the wrong shape), the next fire was
  written over it and every earlier fire was lost. Now the command refuses, says which file it left
  untouched and to move it aside or repair it, and exits 1. A missing file still starts a new log,
  and the index, `compare` and the library list still treat a broken log as an unfired kit.
- **A depth pass now stops partway if the machine runs out of memory.** The check before a pass
  read the machine once, so another app filling memory during the pass could still stall it. The
  pass now keeps reading what is free, and if that stays under 0.75 GB for three readings in a row
  it stops with a plain message, exits 1 and writes nothing. `DEPTHC_MEMORY_GB` does not turn it
  off, and it never retries at a smaller size. The floor and the nine real passes it was set from
  are beside it in `memory.py`.
- **The camera is now read from the picture, not from the depth.** The real-footage failure in
  v0.16.1 came from the input: a depth picture of a real scene is mostly flat, so the people
  walking through a locked-off shot were nearly all the reader could see, and it named them a pan.
  `describe`, `compare`'s PATH line and a kit's `cameraPath` now track points on the picture, drop
  the ones that move on their own, and let the rest vote the way the region reader did. Slow
  push-ins the old read lost come back with the floors unchanged. `compare` reads the reference
  from the kit's `source.mp4` and says so when a kit has none. On clay it is level with the region
  reader on the tuned clips (162 of 168) and two buckets ahead on the held-out ones (52 of 63
  against 50), with no false motion. **Scored on 40 unseen real clips, it failed**: 7 of 10 moves named and
  a move invented on 2 of 7 locked-off shots (ordinary passers-by no longer fool it; a subject that
  fills the frame, or a dancer in a vertical frame, still does), and the camera share is still measured on the depth. The camera words
  still do not vote in any verdict.
- The "tilts up" the reader put beside every push-in on the 8 takes is settled: it is real. Those
  takes push toward a face above the centre of the frame.

## v0.16.1 — 2026-11-10

- **The camera reader was scored on real footage, and failed.** On 17 stock clips it named 9 of 12
  clean moves and invented a move on 4 of 5 locked-off shots with people or a train passing through;
  the pass mark was at most 2 misses and none invented. No code changed. The README and
  `DOCS/CAMERA-PATH-CALIBRATION.md` now say so: the camera words in `describe` and `compare`'s PATH
  line are a hint to check by eye, and they still do not vote in any verdict.
- `depthc describe --json` now prints only the JSON on stdout. The matte line and the progress bar
  landed in front of it, so a script's `json.loads` failed on a command that exited 0. Warnings from
  every command now go to stderr too.
- The Seedance limits snapshot now counts as old after 180 days instead of 30. The
  re-exports the 30-day window forced changed only the date. `depthc` still warns past
  the window and keeps using the numbers.
- **A depth pass no longer swells until the machine stalls.** A default pass reached 18 GB and
  all but froze a 24 GB Mac with other apps open. Two changes:
  - It now peaks at 7.1 to 7.3 GB, at the same speed. The model's backbone sees 4 frames at a time
    instead of 32, and its output is byte-identical on every clip compared. No vendor code changed.
  - Before each pass the tool reads what this machine has free right now, keeps 1.5 GB back, and
    sets a GPU limit from what is left. If the pass won't fit, it stops before loading the model
    with the figures and what to do, including a smaller `--input-size` when one would fit. It never
    lowers the size on its own, because that changes the output. `DEPTHC_MEMORY_GB` sets the limit
    by hand. Every command that runs a depth pass, the UI and the smoke test get this. One Mac was
    measured, at eight picture sizes; for other machines and clip shapes the figures are an estimate.
    Long clips need more, about 14 GB by calculation for 30 seconds of 720p, because the clip's own
    frames and depth add up. A read of over 1 GB that won't fit is refused before it starts; a smaller
    one goes unchecked. The check runs once per pass, so another app filling memory during a pass can
    still slow the machine.

## v0.16.0 — 2026-11-01

- **The camera path is read where the picture can show it.** It missed slow and lens-compressed moves
  because flat ground voted zero, and a texture mask alone turned pushes into tilts. Textured regions
  now cast one vote each and the zoom is read first, from opposite sides of the frame. On 9 clips held
  back until the thresholds were frozen, exact buckets go from 32 / 25 / 25 of 63 to 50 / 58 / 55 (rgb,
  blender, vda) with no false motion, all on clay. This moves `compare`'s PATH numbers, `describe`'s
  camera words and every camera share line for every kit, not any verdict. See
  `DOCS/CAMERA-PATH-CALIBRATION.md`, which also corrects where the texture sits.

## v0.15.0 — 2026-11-01

The first version meant for people outside this project: ready to go public, with the audit's
Depth Converter set closed. The envelope is re-exported, with no change in what it says.

- **Things in a kit or a folder name are data.** The launchers quote the project path, and the
  `depthc` paste lines quote what they echo. A `result.json` or `look-cdl.json` that is a symlink is
  replaced, not written through. `look-compare --manifest` reads only `oside-davinci/v1` packages and
  only files inside them, and a file named `-clip.mp4` or `file:clip.mp4` is read as a file.
- **Depth Stop stops only this app.** It checks a process is ours before killing it and sweeps
  listeners, not every connection, and Start says so if another program holds the port.
- **Setup is honest about macOS and the home folder.** It stops below macOS 14, which the locked torch
  needs, and no longer links a Python onto the `PATH`. The README names what `uv` keeps at home.
- **A library outside the project shows its videos in the UI.** Gradio refused the path after the kit
  was already built.
- **A `kit.json` that is valid JSON but not an object is skipped** rather than crashing the listing.
- **The kit's 2.5 advice catches up with the recorded fire:** 1080p has one good result on a gentle
  reference, and a hard one is unfired.
- **CI asserts its skips.** The expected count was only a step name, so one more skip could not fail
  the job. `scripts/check_ci_skips.py` reads pytest's report and fails on any other count or reason.
- **One render is one fire.** `depthc result` refuses a generation id already recorded on this kit or
  a sibling, instead of counting one paid render twice.
- **A broken envelope no longer stops `depthc result`.** The table is read on first use, so `result`
  and `library` run without it and every other command refuses by name before doing any work.
- **`depthc stability` handles very small inputs.** A clip too short for the 2 fps grain sample still
  measures, and a frame under 3 px on a side has no grain reading rather than crashing the command.
- **The join path is `request_payload.request.refs[].sha256`,** one level deeper than these docs said.
  `CLAUDE.md` now says what the content half can prove, and `DOCS/LIVE-FIRE.md` what the one keyed
  fire cannot.
- **The handover gate now reads `PLAN.md`,** in the phrasings that file actually uses. It keeps a
  superseded sentence with a dated `[AMENDED]` note after it, so a hit struck that way is history and
  any other hit is live. That found two more stale instructions, now amended.
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
  (`request_payload.request.refs[].sha256`, lands in OSIDE), so `kit.json.depth.sha256 ==
  refs[].sha256` names the kit a generation rode. The plan's `kitId` was replaced by
  content: a browser upload cannot read a sibling file and `kit.json` has no `id`. Older
  kits load without it; `compare` says so once.
- **`depthc/result_schema.json`** (new) documents `result.json`, which had no schema; held
  by tests, never a runtime gate. Both schemas stay at `schemaVersion` 1 — additive
  changes do not bump (audit DC-6 ruling), noted in each schema's `$comment`.

### A kit lost its media, and this release ships the husk

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
