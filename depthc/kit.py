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
from depthc.checks import Report, Tier, suggested_output_seconds

DEFAULT_LIBRARY = Path.home() / "Depth_Motion_Library"
TEMPLATE_DIR = Path(__file__).parent / "templates"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIALECT = "seedance-2.0"
KIT_MANIFEST = "kit.json"

STILL_SEARCH_FRAMES = 24
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm"}


def library_root(override: str | Path | None = None) -> Path:
    """--library beats DEPTHC_LIBRARY beats ~/Depth_Motion_Library."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("DEPTHC_LIBRARY")
    return Path(env).expanduser() if env else DEFAULT_LIBRARY


def inbox_dir(override: str | Path | None = None) -> Path:
    """Where clips are dropped for conversion. `Inbox/` in the project by default."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("DEPTHC_INBOX")
    return Path(env).expanduser() if env else PROJECT_ROOT / "Inbox"


def processed_dir(override: str | Path | None = None) -> Path:
    """Where sources are filed once a kit has been built from them."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("DEPTHC_PROCESSED")
    return Path(env).expanduser() if env else PROJECT_ROOT / "Processed"


def inbox_clips(root: Path | None = None) -> list[Path]:
    """Video files waiting in the inbox, oldest first. Dotfiles ignored."""
    root = Path(root) if root else inbox_dir()
    if not root.is_dir():
        return []
    clips = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith(".")
    ]
    return sorted(clips, key=lambda p: p.stat().st_mtime)


def archive_source(source: Path, destination: Path | None = None) -> Path:
    """Move a converted source into Processed/, never overwriting an existing file."""
    target_dir = Path(destination) if destination else processed_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / source.name
    stem, suffix = source.stem, source.suffix
    counter = 2
    while target.exists():
        target = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    return Path(shutil.move(str(source), str(target)))


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


BINDING_OPEN = "[[BINDING_A]]"
BINDING_CLOSE = "[[/BINDING_A]]"


def _load_template(dialect: str) -> str:
    path = TEMPLATE_DIR / f"{dialect}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in TEMPLATE_DIR.glob("*.txt"))
        raise FileNotFoundError(f"Unknown dialect {dialect!r}. Available: {available}")
    return path.read_text()


def render_binding(dialect: str = DEFAULT_DIALECT) -> str:
    """Just the Variant A binding phrase, as one paste-ready line.

    Extracted from the template between markers rather than duplicated in Python, so the
    phrase has exactly one home. The markers are stripped from the rendered prompt.txt.

    Newlines are collapsed: this text goes into a single-line binding field in the
    Composer, and pasted hard-wrapping is a nuisance there.
    """
    body = _load_template(dialect)
    try:
        phrase = body.split(BINDING_OPEN)[1].split(BINDING_CLOSE)[0]
    except IndexError as e:
        raise ValueError(f"Template {dialect!r} has no {BINDING_OPEN} block") from e
    return " ".join(phrase.split())


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
    body = _load_template(dialect)
    still_guidance = (
        "• still.png is LOCAL-ONLY: it is flagged as containing real photographic faces,\n"
        "  which Seedance rejects on image inputs. Use it to judge framing, never as a ref."
        if still_has_real_faces
        else "• still.png is flagged free of real faces, so it may be attached as an\n"
        "  @Image composition/look reference. It is never a first_frame parameter."
    )

    body = body.replace(BINDING_OPEN + "\n", "").replace(BINDING_CLOSE + "\n", "")
    return body.format(
        kit_name=kit_name,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        version=__version__,
        suggested_output=suggested_output_seconds(duration),
        still_guidance=still_guidance,
    )


def render_kit_readme(manifest: dict, report: Report, has_preview: bool = False) -> str:
    """The one file to open if you have no idea what this folder is for.

    Every kit ends up with six files and only two of them are things you hand to Seedance.
    This says which two, in order, at the top.
    """
    depth = manifest["depth"]
    faces = manifest["still"]["containsRealFaces"]
    tier = report.tier.value

    lines = [
        manifest["kitName"],
        "=" * len(manifest["kitName"]),
        "",
        manifest.get("description") or "(no description)",
        "",
        f"  {depth['width']}x{depth['height']} · {depth['durationSeconds']:g}s · "
        f"{depth['fps']:g} fps · check: {tier}",
        f"  from {manifest['source']['filename']}"
        + (f"  trimmed {manifest['source']['trim'][0]:g}-{manifest['source']['trim'][1]:g}s"
           if manifest["source"].get("trim") else "  (untrimmed)"),
        "",
        "TAKE THESE TWO TO OSIDE",
        "-----------------------",
        "  1. depth.mp4    upload as the VIDEO reference",
        "  2. binding.txt  paste its one line into that reference's binding field",
        "",
        "IN THE COMPOSER",
        "---------------",
        "  Shotlist -> your shot -> go straight to MAKE VIDEO. Skip create-frame:",
        "  a depth shot is refs-only, and a start frame silently drops the depth ref.",
        "  Add your character / location / prop images as the look references.",
        f"  Ask for about {manifest['suggestedOutputSeconds']:g}s of output.",
        "",
        "EVERYTHING ELSE IS REFERENCE, NOT FOR UPLOAD",
        "--------------------------------------------",
        "  prompt.txt   the full guidance, plus a standalone block for the Ark console",
        "  kit.json     manifest: source hash, settings, check results",
        "  source.mp4   the trimmed source, kept so this can be rebuilt later",
    ]
    if has_preview:
        lines.append("  preview.mp4  original | depth side by side, for eyeballing")
    lines += [
        "  still.png    "
        + ("LOCAL ONLY - has a real face, which Seedance rejects on image inputs"
           if faces else "may be attached as an @Image look reference"),
    ]

    problems = [f for f in report.findings if f.tier is not Tier.PASS]
    if problems:
        lines += ["", "NOTES", "-----"]
        for finding in problems:
            lines.append(f"  [{finding.tier.value}] {finding.check}: {finding.detail}")
            if finding.remedy:
                lines.append(f"         {finding.remedy}")

    return "\n".join(lines) + "\n"


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

    @property
    def binding(self) -> Path:
        return self.root / "binding.txt"

    @property
    def readme(self) -> Path:
        return self.root / "README.txt"


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
