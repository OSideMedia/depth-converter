"""The law file and the scripts say what the code does (audits DC-3, DC-6).

CLAUDE.md is read by every agent before touching this repo. A 2.0-only law stated without
scope grades a 2.5 kit by 2.0 numbers, which is exactly what the 2026-09-06 audit found.
These greps are deliberately narrow: they hold the SCOPING, not the prose.
"""

from __future__ import annotations

import re
from pathlib import Path

from depthc import checks

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
SCRIPTS = [ROOT / "scripts" / "setup.sh", ROOT / "scripts" / "build_launchers.sh"]


def _lines_with(text: str, needle: str) -> list[str]:
    return [line for line in text.splitlines() if needle in line]


def test_claude_md_states_the_output_window_per_dialect_on_one_line():
    """`[4, 15]` alone was the law; each model's window now sits beside its name."""
    for name, d in checks.DIALECTS.items():
        window = f"[{d.output_seconds[0]:g}, {d.output_seconds[1]:g}]"
        hits = [line for line in _lines_with(CLAUDE, window) if name in line]
        assert hits, f"CLAUDE.md has no line naming {name} beside {window}"


def test_claude_md_scopes_the_1080p_law_to_the_model_it_was_measured_on():
    hits = _lines_with(CLAUDE, "NEVER 1080p")
    assert hits, "the 2.0 resolution law is gone from CLAUDE.md"
    assert all("2.0" in line for line in hits), hits
    # and the 2.5 measurement that says the 0/3 does not transfer is stated with its numbers
    assert "does NOT transfer" in CLAUDE
    assert "+0.820" in CLAUDE and "2026-08-24" in CLAUDE


def test_claude_md_states_the_create_frame_rule_per_dialect():
    """The one instruction that is OPPOSITE on the two models must carry both."""
    assert "seedance-2.0: go straight to" in CLAUDE
    assert "seedance-2.5: you MAY use create-frame" in CLAUDE


def test_scripts_do_not_claim_the_toolchain_is_x86():
    """Homebrew went native arm64 on 2026-08-24; a present-tense x86 claim is stale."""
    for path in SCRIPTS:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not re.search(r"\b(is|are) x86_64\b", line), f"{path.name}:{n}: {line}"


SHALLOW_JOIN = re.compile(r"request_payload\.refs")


def test_the_join_path_names_the_level_the_row_stores_refs_at():
    """Audit DC-3. OSIDE stores `requestPayload: { request: req, ... }`, so refs sit under `request`.

    Measured 2026-11-01 on OSIDE's database: the one generation row with refs has them under
    `request`, and none has them at the top. Whoever writes the SQL half of this join reads these
    lines, and the shorter path returns nothing without raising anything.
    """
    assert SHALLOW_JOIN.search("request_payload.refs[].sha256")  # the old wording trips it
    assert not SHALLOW_JOIN.search("request_payload.request.refs[].sha256")

    this_file = Path(__file__).resolve()
    readers = [ROOT / name for name in ("CHANGELOG.md", "CLAUDE.md", "PLAN.md", "README.md")]
    for folder, pattern in (("DOCS", "*.md"), ("depthc", "*.py"), ("depthc", "*.json"), ("tests", "*.py")):
        readers += sorted((ROOT / folder).glob(pattern))
    readers = [path for path in readers if path != this_file]
    assert len(readers) > 30  # a glob that stopped matching must not read as clean

    hits = [path.name for path in readers if SHALLOW_JOIN.search(path.read_text(encoding="utf-8"))]
    assert hits == []
    assert "request_payload.request.refs[].sha256" in CLAUDE
