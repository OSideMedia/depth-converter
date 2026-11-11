#!/usr/bin/env python3
"""Fail CI when the run skipped anything other than what it is meant to skip (audit DC-1).

Reads the JUnit report pytest writes. The expected skips are named by reason as well as counted,
because a real skip leaving and an unexpected one arriving add up to the same total. Any change
in either direction is a failure: more means a test lost its reach, and fewer means the number
written down here is no longer true.

On RED: read the reason it prints. A new test that needs the model wants the `slow` marker, and
then the number below goes up by one in the same commit; a new test that runs Apple Vision wants
the `vision` marker, likewise (CI runs ubuntu, which has no Vision). A reason not listed here is the run
losing something it used to do, so fix that rather than add it to the list.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# The start of the reason, and how many. tests/test_ci_skips.py holds the first two against the
# markers, on machines that skip none of them.
EXPECTED = {
    "real-model test, no VDA assets": 11,
    "needs the vendored VDA checkout": 4,
    "no OSIDE checkout beside this repo": 1,
    "needs Apple Vision": 2,
}


def _label(reason: str) -> str:
    return next((known for known in EXPECTED if reason.startswith(known)), reason)


def read_report(path: Path) -> tuple[int, Counter]:
    """How many tests the report holds, and its skips counted by reason."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skips = Counter(
        _label(skipped.get("message", "(no reason given)"))
        for case in cases
        for skipped in case.findall("skipped")
    )
    return len(cases), skips


def problems(total: int, skips: Counter) -> list[str]:
    found = []
    if total == 0:
        found.append("the report holds no tests, so nothing here was checked")
    for reason in sorted(set(EXPECTED) | set(skips)):
        want, got = EXPECTED.get(reason, 0), skips.get(reason, 0)
        if want != got:
            found.append(f"expected {want}, got {got}: {reason}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_ci_skips.py <pytest junit report>")
        return 2
    try:
        total, skips = read_report(Path(argv[1]))
    except (OSError, ET.ParseError) as e:
        print(f"cannot read the report {argv[1]}: {e}")
        return 1
    found = problems(total, skips)
    for line in found:
        print(line)
    print(f"{total} tests, {sum(skips.values())} skipped: {'NOT as expected' if found else 'as expected'}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
