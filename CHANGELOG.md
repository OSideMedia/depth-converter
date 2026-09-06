# Changelog

Per-kit evidence lives in `Outbox/*/result.json` and `DOCS/LIVE-FIRE.md`; this file is
the code's history. Started 2026-09-06.

## Unreleased

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
