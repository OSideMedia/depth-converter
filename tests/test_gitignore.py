"""The per-kit TEXT records are tracked; the media never is (audit DC-4).

`.gitignore Outbox/*` wholesale swallowed every 2026-07-25 kit's result.json — the paid
evidence — and left the one surviving $3.83 record in an unbacked local folder. The rule
was a rights law ("never commit media") that also ate the text. This asks git itself,
path by path, so the assertion cannot drift from what the ignore file actually does.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KIT = "Outbox/Some-kit_001"

TRACKED = [f"{KIT}/kit.json", f"{KIT}/result.json", "Outbox/INDEX.txt", "Outbox/README.txt"]
IGNORED = [
    f"{KIT}/depth.mp4", f"{KIT}/source.mp4", f"{KIT}/preview.mp4", f"{KIT}/still.png",
    f"{KIT}/sheet.png", f"{KIT}/prompt.txt", f"{KIT}/binding.txt", f"{KIT}/README.txt",
    f"{KIT}/variants/midgrey.mp4", f"{KIT}/anything.zip",
    "Inbox/clip.mp4", "Processed/clip.mp4",
]


def _ignored(path: str) -> bool:
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    # exit 0 = ignored, 1 = not ignored, anything else = an error worth seeing
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", path],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", TRACKED)
def test_the_text_records_are_not_ignored(path):
    assert not _ignored(path), f"{path} is ignored — a fire record would be lost again"


@pytest.mark.parametrize("path", IGNORED)
def test_the_media_and_paste_files_stay_ignored(path):
    assert _ignored(path), f"{path} is NOT ignored — media must never be committed"
