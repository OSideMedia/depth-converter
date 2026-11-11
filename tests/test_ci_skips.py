"""CI's skip count is an assertion, not a step name (audit DC-1).

`scripts/check_ci_skips.py` reads the report pytest writes in CI. These tests run the real
script over reports written by real pytest runs, so the parser meets what pytest writes and
not what someone assumed it writes.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_ci_skips.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
EXPECTED: dict[str, int] = runpy.run_path(str(SCRIPT))["EXPECTED"]


def _report(tmp_path: Path, skips: dict[str, int], passes: int = 3) -> Path:
    """The JUnit report of a real pytest run holding exactly these skips."""
    lines = ["import pytest", ""]
    number = 0
    for reason, count in skips.items():
        for _ in range(count):
            number += 1
            # both shapes CI meets: conftest adds a marker, test_envelope calls pytest.skip
            if number % 2:
                lines += [f"@pytest.mark.skip(reason={reason!r})", f"def test_{number}(): pass", ""]
            else:
                lines += [f"def test_{number}(): pytest.skip({reason!r})", ""]
    lines += [line for i in range(passes) for line in (f"def test_ok_{i}(): pass", "")]
    module = tmp_path / "test_planted.py"
    module.write_text("\n".join(lines), encoding="utf-8")
    report = tmp_path / "report.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
         f"--junitxml={report}", str(module)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert report.is_file()
    return report


def _check(report: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(report)], capture_output=True, text=True)


def test_the_expected_skips_pass(tmp_path):
    done = _check(_report(tmp_path, EXPECTED))
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_new_skip_goes_red_and_names_itself(tmp_path):
    done = _check(_report(tmp_path, {**EXPECTED, "a new test lost its reach": 1}))
    assert done.returncode == 1
    assert "a new test lost its reach" in done.stdout


@pytest.mark.parametrize("reason", sorted(EXPECTED))
def test_one_more_of_an_expected_skip_goes_red(tmp_path, reason):
    assert _check(_report(tmp_path, {**EXPECTED, reason: EXPECTED[reason] + 1})).returncode == 1


@pytest.mark.parametrize("reason", sorted(EXPECTED))
def test_one_fewer_of_an_expected_skip_goes_red(tmp_path, reason):
    """Fewer is a change too: the number written down would no longer be the truth."""
    assert _check(_report(tmp_path, {**EXPECTED, reason: EXPECTED[reason] - 1})).returncode == 1


def test_an_empty_run_is_not_a_clean_one(tmp_path):
    assert _check(_report(tmp_path, EXPECTED, passes=0)).returncode == 0  # skips alone still ran
    assert _check(_report(tmp_path, {}, passes=0)).returncode == 1


def test_a_missing_or_broken_report_goes_red(tmp_path):
    assert _check(tmp_path / "no-such-report.xml").returncode == 1
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuites><not closed", encoding="utf-8")
    assert _check(broken).returncode == 1


def test_the_numbers_match_the_markers_on_any_machine():
    """A Mac with the assets skips none of these, so this is where a new `slow` test shows up first."""
    def collected(expression: str) -> int:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", expression, "tests"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        return sum("::" in line for line in done.stdout.splitlines())

    assert collected("slow") == EXPECTED["real-model test, no VDA assets"]
    assert collected("vendor and not slow") == EXPECTED["needs the vendored VDA checkout"]
    assert collected("vision and not slow and not vendor") == EXPECTED["needs Apple Vision"]


def test_every_expected_reason_is_one_the_suite_really_gives():
    source = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8") \
        + (ROOT / "tests" / "test_envelope.py").read_text(encoding="utf-8")
    for reason in EXPECTED:
        assert reason in source, reason


def test_ci_runs_the_check_on_the_report_and_gates_on_its_exit_code():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*run: uv run pytest -q -rs --junitxml=pytest-report\.xml tests$", workflow, re.M)
    assert re.search(r"^\s*run: uv run python scripts/check_ci_skips\.py pytest-report\.xml$", workflow, re.M)
    pytest_step = workflow.index("--junitxml=pytest-report.xml")
    assert pytest_step < workflow.index("check_ci_skips.py pytest-report.xml")
