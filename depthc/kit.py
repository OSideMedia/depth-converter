"""Seedance Kit assembly and the Motion Library.

A kit is a self-contained folder: the depth video, a reference still, a paste-ready prompt,
a manifest, and a copy of the trimmed source so the motion can be re-converted later.

THE LIBRARY IS THE FILESYSTEM. There is no database and there must never be one — the
folder tree plus the kit.json files *are* the index. `library_list()` is a directory scan.
That keeps kits portable, greppable, survivable across tool rewrites, and repairable with
a text editor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthc import VDA_COMMIT, __version__
from depthc.checks import Report, suggested_output_seconds

DEFAULT_LIBRARY = Path.home() / "Depth_Motion_Library"
TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_DIALECT = "seedance-2.0"
KIT_MANIFEST = "kit.json"

STILL_SEARCH_FRAMES = 24


def library_root(override: str | Path | None = None) -> Path:
    """--library beats DEPTHC_LIBRARY beats ~/Depth_Motion_Library."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("DEPTHC_LIBRARY")
    return Path(env).expanduser() if env else DEFAULT_LIBRARY


def slugify(text: str) -> str:
    """Title-case-ish hyphen slug: 'sword fight!' -> 'Sword-fight'."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def kit_folder_name(tags: list[str], index: int, fallback: str = "Motion") -> str:
    """`Sword-fight_Tracking-orbit_001` — tags name it, an index keeps it unique."""
    parts = [slug for slug in (slugify(t) for t in tags) if slug]
    stem = "_".join(parts) if parts else slugify(fallback) or "Motion"
    return f"{stem}_{index:03d}"


def allocate_kit_dir(root: Path, tags: list[str], fallback: str = "Motion") -> Path:
    """Reserve the next free numbered folder for these tags.

    Creates the directory as it goes: allocating without creating would let two runs in the
    same second pick the same name.
    """
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = root / kit_folder_name(tags, index, fallback)
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"More than 999 kits already named for tags {tags!r} in {root}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sharpest_frame_index(frames: np.ndarray, search: int = STILL_SEARCH_FRAMES) -> int:
    """Pick the least-blurred of the first N frames by variance of the Laplacian.

    Frame 0 is a bad default: the in-point of a trim frequently lands mid-whip-pan or on an
    obstructed frame, and that would become the kit's thumbnail and framing reference.
    Laplacian variance is the standard blur measure — sharp edges produce a wide second
    -derivative distribution, motion blur collapses it.
    """
    import cv2

    window = min(search, frames.shape[0])
    scores = [
        cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        for i in range(window)
    ]
    return int(np.argmax(scores))


def write_still(frames: np.ndarray, out_path: Path, index: int | None = None) -> int:
    """Write one RGB frame as PNG. Returns the frame index used."""
    import cv2

    chosen = sharpest_frame_index(frames) if index is None else index
    if not 0 <= chosen < frames.shape[0]:
        raise ValueError(f"Still index {chosen} out of range (0-{frames.shape[0] - 1})")
    cv2.imwrite(str(out_path), cv2.cvtColor(frames[chosen], cv2.COLOR_RGB2BGR))
    return chosen


def render_prompt(
    *,
    kit_name: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
    still_has_real_faces: bool,
    dialect: str = DEFAULT_DIALECT,
) -> str:
    """Fill the dialect template.

    The binding text lives in `templates/<dialect>.txt`, never in Python. Seedance 2.5 is
    rolling out and its reference syntax may shift; a new dialect must be a new template
    file plus a --dialect flag, with no code change.
    """
    template_path = TEMPLATE_DIR / f"{dialect}.txt"
    if not template_path.is_file():
        available = sorted(p.stem for p in TEMPLATE_DIR.glob("*.txt"))
        raise FileNotFoundError(f"Unknown dialect {dialect!r}. Available: {available}")

    still_guidance = (
        "• still.png is LOCAL-ONLY: it is flagged as containing real photographic faces,\n"
        "  which Seedance rejects on image inputs. Use it to judge framing, never as a ref."
        if still_has_real_faces
        else "• still.png is flagged free of real faces, so it may be attached as an\n"
        "  @Image composition/look reference. It is never a first_frame parameter."
    )

    return template_path.read_text().format(
        kit_name=kit_name,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        version=__version__,
        suggested_output=suggested_output_seconds(duration),
        still_guidance=still_guidance,
    )


@dataclass
class KitPaths:
    root: Path

    @property
    def depth(self) -> Path:
        return self.root / "depth.mp4"

    @property
    def still(self) -> Path:
        return self.root / "still.png"

    @property
    def prompt(self) -> Path:
        return self.root / "prompt.txt"

    @property
    def manifest(self) -> Path:
        return self.root / KIT_MANIFEST

    @property
    def source(self) -> Path:
        return self.root / "source.mp4"

    @property
    def preview(self) -> Path:
        return self.root / "preview.mp4"


def build_manifest(
    *,
    kit_name: str,
    source_path: Path,
    source_sha256: str,
    source_duration: float,
    source_fps: float,
    source_size: tuple[int, int],
    trim: tuple[float, float] | None,
    crop: str | None,
    depth_duration: float,
    depth_fps: float,
    depth_size: tuple[int, int],
    depth_bytes: int,
    frame_count: int,
    still_index: int,
    still_has_real_faces: bool,
    tags: list[str],
    description: str,
    settings: dict,
    report: Report,
    dialect: str = DEFAULT_DIALECT,
    device: str = "",
    elapsed: float = 0.0,
) -> dict:
    width, height = depth_size
    return {
        "schemaVersion": 1,
        "kitName": kit_name,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "name": "depthc",
            "version": __version__,
            "model": "Video-Depth-Anything Small (vits, Apache-2.0)",
            "vdaCommit": VDA_COMMIT,
            "device": device,
            "elapsedSeconds": round(elapsed, 2),
        },
        "promptDialect": dialect,
        "tags": tags,
        "description": description,
        "source": {
            "filename": source_path.name,
            "sha256": source_sha256,
            "durationSeconds": round(source_duration, 3),
            "fps": round(source_fps, 3),
            "width": source_size[0],
            "height": source_size[1],
            "trim": list(trim) if trim else None,
            "crop": crop,
        },
        "depth": {
            "filename": "depth.mp4",
            "durationSeconds": round(depth_duration, 3),
            "fps": round(depth_fps, 3),
            "width": width,
            "height": height,
            "frameCount": frame_count,
            "sizeBytes": depth_bytes,
            "pixels": width * height,
        },
        "still": {
            "filename": "still.png",
            "frameIndex": still_index,
            # The still comes from the ORIGINAL pixels, so if the source had people in it,
            # it carries a real photographic face — precisely what Seedance rejects.
            "containsRealFaces": still_has_real_faces,
            "usage": (
                "local-only reference; do not attach as @Image"
                if still_has_real_faces
                else "may be attached as an @Image look reference (never first_frame)"
            ),
        },
        "settings": settings,
        "suggestedOutputSeconds": suggested_output_seconds(depth_duration),
        "checks": report.as_dict(),
    }


def write_manifest(paths: KitPaths, manifest: dict) -> Path:
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return paths.manifest


def zip_kit(kit_dir: Path, out_path: Path | None = None) -> Path:
    """Zip the kit folder for handing to someone else."""
    target = out_path or kit_dir.with_suffix(".zip")
    archive = shutil.make_archive(
        str(target.with_suffix("")), "zip", root_dir=kit_dir.parent, base_dir=kit_dir.name
    )
    return Path(archive)


@dataclass
class LibraryEntry:
    path: Path
    manifest: dict

    @property
    def name(self) -> str:
        return self.manifest.get("kitName", self.path.name)

    @property
    def tags(self) -> list[str]:
        return self.manifest.get("tags", []) or []

    @property
    def duration(self) -> float:
        return float(self.manifest.get("depth", {}).get("durationSeconds", 0.0))

    @property
    def tier(self) -> str:
        return self.manifest.get("checks", {}).get("tier", "?")

    @property
    def created(self) -> str:
        return self.manifest.get("createdAt", "")


def library_list(root: Path | None = None, tag: str | None = None) -> list[LibraryEntry]:
    """Scan the library for kits. A malformed kit.json is skipped, never fatal."""
    root = Path(root) if root else library_root()
    if not root.is_dir():
        return []

    entries: list[LibraryEntry] = []
    for manifest_path in sorted(root.glob(f"*/{KIT_MANIFEST}")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entry = LibraryEntry(path=manifest_path.parent, manifest=manifest)
        if tag and tag.lower() not in [t.lower() for t in entry.tags]:
            continue
        entries.append(entry)
    return entries
