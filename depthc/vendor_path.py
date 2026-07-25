"""Locate the vendored Video-Depth-Anything checkout and make it importable.

We vendor rather than fork: `scripts/setup.sh` clones upstream at a pinned commit and
we never edit their files. Their package does `from utils.util import ...`, so the repo
ROOT (not the inner package dir) has to be on sys.path — which means a top-level `utils`
module becomes importable process-wide. It is appended, never prepended, so anything
already imported wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendor" / "Video-Depth-Anything"
CHECKPOINT = REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth"

# VDA-Small (vits) only. Base/Large are CC-BY-NC-4.0 and must never ship here.
MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


class SetupIncomplete(RuntimeError):
    """Raised when setup.sh has not been run (or was interrupted)."""


def ensure_vendor_on_path() -> Path:
    """Append the vendored VDA repo root to sys.path. Returns its path."""
    if not (VENDOR_DIR / "video_depth_anything").is_dir():
        raise SetupIncomplete(
            f"Vendored Video-Depth-Anything not found at {VENDOR_DIR}.\n"
            f"Run: bash scripts/setup.sh"
        )
    p = str(VENDOR_DIR)
    if p not in sys.path:
        sys.path.append(p)
    return VENDOR_DIR


def ensure_checkpoint() -> Path:
    """Return the VDA-Small checkpoint path, or explain how to get it."""
    if not CHECKPOINT.is_file():
        raise SetupIncomplete(
            f"Checkpoint not found at {CHECKPOINT}.\nRun: bash scripts/setup.sh"
        )
    return CHECKPOINT


def vendor_commit() -> str | None:
    """The pinned SHA actually checked out in vendor/ (for kit.json provenance)."""
    head = VENDOR_DIR / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref: "):
        target = VENDOR_DIR / ".git" / ref[5:]
        return target.read_text().strip() if target.is_file() else None
    return ref
