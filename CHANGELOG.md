# Changelog

Per-kit evidence lives in `Outbox/*/result.json` and `DOCS/LIVE-FIRE.md`; this file is
the code's history. Started 2026-09-06.

## Unreleased

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
