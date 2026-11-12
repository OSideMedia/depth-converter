# Contributing

Thanks for looking. A few things about this repo are unusual, and knowing them first will save you
a rejected patch.

## Three rules that are not negotiable

- **VDA-Small only.** Across Depth Anything V2 and Video Depth Anything, only the Small
  checkpoints are Apache-2.0. Every Base, Large and Giant there is CC-BY-NC-4.0. Depth Anything 3
  is licensed model by model and none of it ships here. There is no
  model-size selector by design, so **a PR that adds one will be declined**: it is a licensing
  violation, not an upgrade.
- **Media never goes in git.** `Inbox/`, `Processed/` and `Outbox/` media stay ignored, because source
  clips carry other people's rights and a kit's `source.mp4` is the original footage. Only each kit's
  `kit.json` and `result.json`, plus `Outbox/INDEX.txt`, are tracked. `tests/test_gitignore.py` proves
  it. A build rewrites `INDEX.txt`, so leave that change out of your PR unless it is the point.
- **Nothing fires against a paid API.** No test, script or code path here may call one. Generations
  are fired by a person, and the results land in `DOCS/LIVE-FIRE.md` and the kit's `result.json`.

Also: never edit `vendor/`, which is pinned upstream code and rebuilt by `scripts/setup.sh`, and always
run through `./.tools/uv`.

## A vendor's rule is described, never worked around

This tool reports what Seedance accepts and refuses. A change that aims to get something past a
provider's content rules will be declined.

## Every gate must be able to fail

This suite is mostly gates, and a gate that has never gone red is a claim, not a check. If you add
one, show it failing on the code it is meant to catch:

```bash
git show main:depthc/checks.py > depthc/checks.py   # the pre-fix version
./.tools/uv run pytest -q tests/test_checks.py      # expect RED
git checkout -- depthc/checks.py                    # restore: this discards unstaged edits, so commit first
```

Two related rules the existing tests follow:

- **Unknown is never a pass.** A check that cannot reach its subject says so and fails, or skips with
  a reason naming what is missing. It never silently succeeds.
- **Name the subject set, and make it non-empty.** An assertion over an empty set is vacuously true.
  See `test_the_old_name_subject_set_is_not_empty` in `tests/test_handover.py`.

## Running the suite

```bash
./.tools/uv run pytest -q tests
```

That takes about three minutes on an Apple Silicon Mac, and synthesises its clips with ffmpeg. CI runs
the same suite on ubuntu with CPU torch and never downloads the model or the vendored checkout, so the
tests marked `slow` and `vendor` skip there with a reason that says so
(`.github/workflows/tests.yml` holds the expected count). Mark a test that needs the model, or it will
error in CI rather than skip.

## Facts carry a date

A claim about Seedance names the model and the day it was measured: a number measured on 2.0 is a wrong
number on 2.5. `PLAN.md` and `CLAUDE.md` are working notes, and corrections there are added as dated
`[AMENDED]` notes rather than overwriting the old claim, because the history of what was believed is
part of what they record.

## Style

Match the surrounding code. Comments here explain *why*, usually citing the run that produced the rule.
Please keep that when you touch those lines. A deliberate shortcut with a known limit gets a
`# ponytail:` comment naming the limit and the way past it.
