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
import secrets
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthc import VDA_COMMIT, __version__
from depthc.checks import Finding, Report, Tier, suggested_output_seconds
from depthc.checks import DEFAULT_DIALECT as checks_default_dialect
from depthc.checks import dialect_for as checks_dialect_for

TEMPLATE_DIR = Path(__file__).parent / "templates"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Finished kits live in the project, beside Inbox/ and Processed/ — one visible place for
# work in, work out, and originals. An earlier default of ~/Depth_Motion_Library buried the
# deliverables in the home folder, away from the workflow they belong to.
DEFAULT_LIBRARY = PROJECT_ROOT / "Outbox"
DEFAULT_DIALECT = checks_default_dialect  # one source: checks.DEFAULT_DIALECT
KIT_MANIFEST = "kit.json"
BINDING_FILE = "binding.txt"

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
    try:
        return body.format(
            kit_name=kit_name,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            version=__version__,
            suggested_output=suggested_output_seconds(duration, dialect),
            still_guidance=still_guidance,
        )
    except (KeyError, IndexError, ValueError) as e:
        # str.format treats every brace as a placeholder, so a template carrying literal
        # JSON (a plausible shape for a future Seedance dialect) dies here — after the
        # paid inference, if nothing checked it earlier. Name the file and the fix.
        raise ValueError(
            f"templates/{dialect}.txt is not renderable ({e!r}): literal {{ or }} must be "
            f"doubled ({{{{ }}}}), and placeholders must be from the documented set."
        ) from e


def render_kit_readme(manifest: dict, report: Report, has_preview: bool = False) -> str:
    """The one file to open if you have no idea what this folder is for.

    Every kit ends up with six files and only two of them are things you hand to Seedance.
    This says which two, in order, at the top.
    """
    depth = manifest["depth"]
    faces = manifest["still"]["containsRealFaces"]
    tier = report.tier.value
    motion = manifest.get("motion") or {}

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
    ]

    if motion:
        retention = motion.get("retention")
        lines += [
            "",
            "MOTION  (mean grey-level change per frame, depth space; higher is more signal)",
            f"  {motion['sparkline']}   {motion['energy']:.2f} overall"
            + (f", {retention * 100:.0f}% of the source's motion survived into depth"
               if retention is not None else ""),
            f"  {'':{len(motion['sparkline'])}}   each block is "
            f"{motion.get('bucketSeconds', 0.5):g}s · · = no usable motion"
            + (f" · {motion['cameraShare'] * 100:.0f}% whole-frame (camera)"
               if motion.get("cameraShare") else ""),
            "  for scale: 4.40 = a sword duel that behaved · 0.94 = a crawl whose generation "
            "started late",
        ]

    lines += [
        "",
        "TAKE THIS ONE FILE TO SEEDANCE",
        "------------------------------",
        "  depth.mp4    upload as the VIDEO reference. That is the whole handover.",
        "",
        "ON ANY SURFACE",
        "--------------",
        "  Add one identity per body in the ref - the ref sets the head-count and the",
        "  prompt cannot argue it down. Add your location / prop images for the look.",
        f"  Ask for about {manifest['suggestedOutputSeconds']:g}s of output.",
        *_resolution_lines(manifest),
        "",
        "ON THE ARK CONSOLE OR ANOTHER SURFACE",
        "-------------------------------------",
        "  Paste Variant B from prompt.txt: the whole block, and nothing else.",
        "",
        "IN OSIDE",
        "--------",
        *_start_frame_lines(manifest),
        "  WRITE NOTHING ABOUT THE MOTION. The Composer already binds an uploaded video",
        "  ref ('take movement, action and pacing from it'); describing the action again",
        "  makes two instructions for one reference. Describe the LOOK only.",
        "  The binding field: nothing for you to type, on either path. Uploading",
        "  depth.mp4 by hand, there is no such field and the Composer writes its own",
        "  line. Coming in through OSIDE's 'Import kit' door - merged 2026-09-15, the",
        "  tile beside the upload card in video mode - you point it at this whole",
        "  folder and that door fills the field from binding.txt itself, hash-checked",
        "  against kit.json; depth.mp4 is still the only file that uploads.",
        "  binding.txt is for surfaces that HAVE a per-reference binding field - that",
        "  door, and the Ark console. See prompt.txt.",
        "",
        "EVERYTHING ELSE IS REFERENCE, NOT FOR UPLOAD",
        "--------------------------------------------",
        "  prompt.txt   the full guidance, plus a standalone block for the Ark console",
        "  kit.json     manifest: source hash, settings, check results",
        "  source.mp4   the trimmed source, kept so this can be rebuilt later",
    ]
    if has_preview:
        lines.append("  preview.mp4  original | depth side by side, for eyeballing")
    if manifest.get("sheet"):
        lines.append("  sheet.png    depth contact sheet with timestamps - look at this first")
    lines += [
        "  still.png    "
        + ("LOCAL ONLY - has a real face, which Seedance rejects on image inputs"
           if faces else "may be attached as an @Image look reference"),
    ]
    lines += [
        "",
        "AFTER YOU FIRE IT",
        "-----------------",
        "  Record what happened, so the next kit is chosen on evidence:",
        f"    depthc result {manifest['kitName']} --verdict adhered|partial|ignored|late|rejected \\",
        '        --surface oside|ark|other --cost 0.30 --notes "what actually came back"',
        "  Fired in OSIDE? Add --generation <id>, the id on the generation's row/URL.",
        "  That writes result.json here and shows up in Outbox/INDEX.txt.",
    ]

    variants = manifest.get("variants") or []
    if variants:
        lines += [
            "",
            "VARIANTS - ONE PER GENERATION, NEVER TOGETHER",
            "---------------------------------------------",
            "  Same beat, processed differently. Upload ONE as the video ref and compare;",
            "  stacking them would be three motion signals fighting each other.",
            "",
            f"  {'depth.mp4':<24} {'base':<10} energy {motion.get('energy', 0):.2f}",
        ]
        for variant in variants:
            lines.append(
                f"  {'variants/' + variant['filename']:<24} {variant['name']:<10} "
                f"energy {variant.get('energy', 0):.2f}"
            )
            lines.append(f"      {variant['description']}")
        lines.append(
            "  Record which one won with:  depthc result <kit> --verdict <v> --variant <name>"
        )

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

    @property
    def sheet(self) -> Path:
        return self.root / "sheet.png"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def variants(self) -> Path:
        return self.root / "variants"


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
    depth_sha256: str,
    binding_sha256: str,
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
    motion: dict | None = None,
    variants: list[dict] | None = None,
    has_sheet: bool = False,
) -> dict:
    width, height = depth_size
    manifest = {
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
        # The ENVELOPE this kit was graded against, recorded beside the name that selected it.
        # `promptDialect` alone is a template filename; a reader six months from now cannot
        # tell from it whether "15s ceiling" meant 2.0's real limit or a stale table applied
        # to a 2.5 kit. Writing the numbers down makes the grading reproducible after the
        # record itself has moved on.
        "targetModel": _target_model_block(dialect),
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
            # The content half of the fire -> generation join. OSIDE's upload door records
            # the uploaded clip's SHA-256 on the reference (request_payload.request.refs[].sha256);
            # a browser upload cannot read this file beside the clip, so the hash of the
            # BYTES is the only key both sides can hold. Hashed after the size gate and any
            # audio mux — the file as it is uploaded, never an intermediate.
            "sha256": depth_sha256,
            "durationSeconds": round(depth_duration, 3),
            "fps": round(depth_fps, 3),
            "width": width,
            "height": height,
            "frameCount": frame_count,
            "sizeBytes": depth_bytes,
            "pixels": width * height,
        },
        "binding": {
            "filename": BINDING_FILE,
            # The kit's WORDS, bound to the kit's clip. An importer that sets a reference's
            # binding field from this file — OSIDE's Import kit door, merged 2026-09-15 — would
            # otherwise have no way to tell that binding.txt belongs to the depth.mp4 beside
            # it: the blockout kit binds BOTH its clip and its legend by SHA-256 for exactly
            # that reason, and the pair caught one kit's words riding on another kit's clip.
            # Hashed after the file is written, like depth.sha256 — the bytes as they ship.
            "sha256": binding_sha256,
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
        "suggestedOutputSeconds": suggested_output_seconds(depth_duration, dialect),
        "checks": report.as_dict(),
    }
    # Optional blocks are omitted rather than written null: a kit built before the motion gate
    # existed and a kit whose motion was not measured are different things, and `in` should be
    # able to tell them apart.
    if motion is not None:
        manifest["motion"] = motion
    if variants:
        manifest["variants"] = variants
    if has_sheet:
        manifest["sheet"] = {"filename": "sheet.png", "usage": "local-only contact sheet"}
    return manifest


def _start_frame_lines(manifest: dict) -> list[str]:
    """The create-frame instruction, which is OPPOSITE on the two models.

    On 2.0 a start frame takes the whole request and the reference arrays never ride, so a
    frame-first shot silently drops the depth ref. On 2.5 the frame is demoted to a
    reference image and both ride. Printing 2.0's rule into a 2.5 kit costs the operator a
    staging lock for no reason — it was doing exactly that until 2026-08-24.
    """
    target = manifest.get("targetModel") or {}
    if target.get("refsWithStartFrame"):
        model = target.get("dialect", "this model")
        return [
            "  Shotlist -> your shot -> MAKE VIDEO. You MAY use create-frame here:",
            f"  {model} keeps reference arrays alongside a start frame (the frame",
            "  demotes to a reference image), so the depth ref still rides. Seed does",
            "  not pin output, so the start frame is your only staging lock.",
        ]
    return [
        "  Shotlist -> your shot -> go straight to MAKE VIDEO. Skip create-frame:",
        "  a depth shot is refs-only, and a start frame silently drops the depth ref.",
    ]


def _resolution_lines(manifest: dict) -> list[str]:
    """The resolution advice, scoped to the model the kit targets.

    The 0/3 result is a 2.0 measurement about 2.0's native 1080p lane. 2.5 has no 4K rung
    and wires 1080p separately, so the finding is not transferable and must not be printed
    as though it were — but 720p being the safe default is true on both.
    """
    target = manifest.get("targetModel") or {}
    tiers = target.get("outputResolutions") or []
    if "4k" in tiers:
        return [
            "  Fire at 720p or 4K, never 1080p — 1080p ignored or overshot the ref 3/3",
            "  while every other tier adhered 4/4 (DOCS/LIVE-FIRE.md).",
        ]
    return [
        f"  Tiers here: {' / '.join(tiers) or 'unknown'}. No 4K on this model as of 2026-08-24.",
        "  Iterate at 720p — that is the safe default and it is measured on 2.0.",
        "  1080p has one good result here: a gentle walk adhered on 2026-08-24, and a hard",
        "  reference is unfired. The 0/3 in DOCS/LIVE-FIRE.md is 2.0's native 1080p lane, not",
        "  this one. If you fire 1080p here, run `depthc compare` and record it.",
    ]


def _target_model_block(dialect: str) -> dict:
    """The model envelope a kit was built and graded against, frozen into the manifest."""
    d = checks_dialect_for(dialect)
    return {
        "dialect": d.name,
        "refSeconds": list(d.ref_seconds),
        "outputSeconds": list(d.output_seconds),
        "pixels": list(d.pixels),
        "maxFileMb": d.max_file_mb,
        "outputResolutions": list(d.output_resolutions),
        "refsWithStartFrame": d.refs_with_start_frame,
    }


def write_manifest(paths: KitPaths, manifest: dict) -> Path:
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return paths.manifest


def read_manifest(kit_dir: Path) -> dict | None:
    """The kit's manifest, or None when there is no readable kit.json there."""
    try:
        data = json.loads((Path(kit_dir) / KIT_MANIFEST).read_text())
    except (OSError, ValueError):  # ValueError holds bad JSON and bytes that are not text
        return None
    return data if isinstance(data, dict) else None


def depth_sha256(kit_dir: Path) -> str | None:
    """SHA-256 of the kit's depth.mp4 as recorded at build time, or None on a kit built
    before 2026-09-06 — those carry no such key and have real fires behind them, so a
    reader tolerates the absence (and `compare` says so once) rather than refusing the kit.
    The schema requires the key on NEW kits only.
    """
    manifest = read_manifest(kit_dir)
    if not manifest:
        return None
    value = (manifest.get("depth") or {}).get("sha256")
    return value if isinstance(value, str) and value else None


def binding_sha256(kit_dir: Path) -> str | None:
    """SHA-256 of the kit's binding.txt as recorded at build time, or None on a kit built
    before 2026-09-15 — those carry no such key and are real kits with real fires behind
    them, so a reader tolerates the absence (and says so once) rather than refusing the
    kit. The schema requires the key on NEW kits only, exactly as `depth.sha256` is.
    """
    manifest = read_manifest(kit_dir)
    if not manifest:
        return None
    value = (manifest.get("binding") or {}).get("sha256")
    return value if isinstance(value, str) and value else None


def verify_binding(kit_dir: Path) -> Finding | None:
    """Do the kit's WORDS still belong to the kit's clip?

    `binding.txt` sits beside `depth.mp4` and an importer that sets a reference's binding
    field from it — OSIDE's Import kit door, merged 2026-09-15 — is trusting that the two came out of
    the same build. The blockout kit binds both halves by SHA-256 and that pair caught one
    kit's legend riding on another kit's clip; a depth kit recorded only the clip's hash
    until 2026-09-15, so the same swap here was undetectable.

    Returns a FAIL Finding when the bytes on disk are not the bytes kit.json recorded (or
    the file is gone), a PASS Finding when they agree, and None when the manifest carries no
    `binding.sha256` at all — that is the pre-2026-09-15 case, which is NOT a failure and is
    never invented at read time; the caller names it the way `compare` names a missing
    `depth.sha256`.
    """
    recorded = binding_sha256(kit_dir)
    if recorded is None:
        return None

    path = Path(kit_dir) / BINDING_FILE
    dialect = (read_manifest(kit_dir) or {}).get("promptDialect", DEFAULT_DIALECT)
    remedy = (
        f"binding.txt is the Variant A phrase from templates/{dialect}.txt (it is also "
        f"inside this kit's prompt.txt) — restore it, or rebuild the kit, and do not import "
        f"this folder until its words and its depth.mp4 come from the same build. "
        f"`shasum -a 256 binding.txt` shows what is on disk now."
    )
    if not path.is_file():
        return Finding(
            Tier.FAIL, "binding", "kit.json records a binding.txt that is not here.", remedy,
        )

    actual = sha256_file(path)
    if actual != recorded:
        return Finding(
            Tier.FAIL, "binding",
            f"binding.txt does not match the hash kit.json recorded "
            f"(recorded {recorded[:12]}…, on disk {actual[:12]}…) — these words were not "
            f"built with this kit's depth.mp4.",
            remedy,
        )
    return Finding(Tier.PASS, "binding", "binding.txt is the file this kit was built with.")


# ------------------------------------------------------------------- the feedback loop

# What a fire is allowed to have proved. Deliberately about ADHERENCE, not about whether the
# shot was pretty: the only question a kit can answer is whether Seedance followed it.
VERDICTS = {
    "adhered": "followed the motion and timing",
    "partial": "followed some of it — drifted, or lost the timing",
    "ignored": "generated its own motion; the ref did nothing",
    "late": "sat still and then started — the ref's timing did not transfer",
    "rejected": "the API or moderation refused the input",
}
RESULT_SCHEMA_VERSION = 1
RESULT_SCHEMA = Path(__file__).parent / "result_schema.json"  # documents result.json; tests hold it

# The generation half of the fire -> generation join: OSIDE's `generations.id`, a Postgres
# gen_random_uuid() — RFC 4122 version 4. Stored canonical (lowercase, hyphenated) so
# `fires[].generationId == generations.id` is a plain string equality on both sides.
GENERATION_ID_GRAMMAR = (
    "UUID v4 — 8-4-4-4-12 hex with version nibble 4 and variant nibble 8/9/a/b, "
    "e.g. 3f2504e0-4f89-4d3c-9a6b-0305e82c3301; it is the id on the generation's row/URL in OSIDE"
)
GENERATION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def validate_generation_id(value: str) -> str:
    """The canonical form of an OSIDE generation id, or ValueError naming the grammar."""
    candidate = value.strip().lower()
    if not GENERATION_ID_RE.match(candidate):
        raise ValueError(f"--generation {value!r} is not a {GENERATION_ID_GRAMMAR}")
    return candidate


def write_replacing(path: Path, text: str) -> None:
    """Write `path` by swapping a finished temp file into place.

    A kit can come from someone else, and a `result.json` in it can be a symlink to one of your
    own files. A plain write would follow it and overwrite that file; the swap replaces the link
    itself. It also means a crash mid-write cannot leave paid evidence half written.
    """
    path = Path(path)
    # A file that was private stays private, from the moment its copy exists. Not the mode of a
    # link's target: that is someone else's file. A new file gets what the umask allows, which
    # os.open applies to 0o666.
    try:
        before = os.lstat(path)
        keep = stat.S_IMODE(before.st_mode) if stat.S_ISREG(before.st_mode) else None
    except FileNotFoundError:
        keep = None
    temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if keep is None else keep)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            if keep is not None:
                os.fchmod(out.fileno(), keep)
            out.write(text)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def read_results(kit_dir: Path, *, strict: bool = False) -> dict:
    """The kit's fire log, or an empty one. A malformed file is never fatal to a READER.

    A missing file is an empty log. A file that exists but cannot be read as a fire log (bad
    JSON, bytes that are not UTF-8 text, valid JSON of the wrong shape) reads as empty too,
    so the index, `compare` and the sibling scan keep working. A WRITER must pass `strict=True`:
    an unreadable log is not an empty one, and writing onto it would destroy paid evidence.
    """
    path = Path(kit_dir) / "result.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and isinstance(data.get("fires"), list):
                return data
        except (ValueError, OSError):  # ValueError holds bad JSON and bytes that are not text
            pass
        # A symlink is replaced, never followed (write_replacing): the link holds no fires, and
        # its target is someone else's file that the swap leaves alone, so there is nothing to lose.
        if strict and not path.is_symlink():
            raise ValueError(
                f"{path} exists but cannot be read as a fire log, so nothing was written and "
                f"the file was left untouched. It may hold fires you paid for: move it aside "
                f"or repair it (valid JSON, an object with a \"fires\" list), then run this again."
            )
    return {"schemaVersion": RESULT_SCHEMA_VERSION, "kitName": Path(kit_dir).name, "fires": []}


def _refuse_a_recorded_generation(kit_dir: Path, generation: str) -> None:
    """Raise if a kit in this library already holds this generation. A broken log holds nothing."""
    # ponytail: scan then write, no lock. Two `depthc result` runs at the same instant on sibling
    # kits could both pass. A person types these; a lock file if this ever runs unattended.
    kit_dir = Path(kit_dir)
    siblings = sorted(p.parent for p in kit_dir.parent.glob("*/result.json") if p.parent != kit_dir)
    for holder in (kit_dir, *siblings):
        for fire in read_results(holder)["fires"]:
            if isinstance(fire, dict) and str(fire.get("generationId", "")).lower() == generation:
                raise ValueError(
                    f"Generation {generation} is already recorded on {holder.name} "
                    f"({fire.get('firedAt', 'no date')}, {fire.get('verdict', 'no verdict')}); nothing "
                    f"was written. To change what that fire says, edit {holder / 'result.json'}."
                )


def record_result(
    kit_dir: Path,
    *,
    verdict: str,
    surface: str = "",
    model: str = "",
    output_seconds: float | None = None,
    cost_usd: float | None = None,
    output_ref: str = "",
    variant: str = "",
    notes: str = "",
    generation_id: str | None = None,
) -> dict:
    """Append one fire to the kit's result.json and return the updated log.

    WHY THIS EXISTS. A generation costs real money and produces the only evidence that says
    whether any of this works, and until now that evidence lived in a human's memory. The
    prone-crawl diagnosis had to be re-derived from scratch a day later. Fires accumulate
    rather than overwrite: firing the same ref twice with different prompts is a data point
    about prompts, and throwing the first one away would hide it.

    One render is still one fire. A generation id already on this kit or a sibling is refused
    before anything is written (audit DC-2): that was one paid render counted twice, and
    INDEX.txt then read FIRED 2x. Unkeyed fires cannot be told apart, so they still accumulate.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VERDICTS)} (got {verdict!r})")
    # Refused before anything is read or written: a fire keyed to a typo joins nothing.
    generation = validate_generation_id(generation_id) if generation_id is not None else None
    if generation is not None:
        _refuse_a_recorded_generation(kit_dir, generation)

    log = read_results(kit_dir, strict=True)  # a write never lands on a log it cannot read
    log["kitName"] = Path(kit_dir).name
    log["schemaVersion"] = RESULT_SCHEMA_VERSION
    entry = {
        "firedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "meaning": VERDICTS[verdict],
        "surface": surface,
        "model": model,
        "variant": variant,
        "outputSeconds": output_seconds,
        "costUsd": cost_usd,
        "outputRef": output_ref,
        "notes": notes,
    }
    # The OSIDE generation this fire became (PLAN-SHARED-GENERATION-ID step d). Absent,
    # not null, when unknown: every fire recorded before 2026-09-06 is unkeyed, and a
    # reader must be able to tell "never named" from "named nothing".
    if generation is not None:
        entry["generationId"] = generation
    log["fires"].append(entry)
    write_replacing(Path(kit_dir) / "result.json", json.dumps(log, indent=2) + "\n")
    return log


def zip_kit(kit_dir: Path, out_path: Path | None = None) -> Path:
    """Zip the kit folder for handing to someone else."""
    target = out_path or kit_dir.with_suffix(".zip")
    archive = shutil.make_archive(
        str(target.with_suffix("")), "zip", root_dir=kit_dir.parent, base_dir=kit_dir.name
    )
    return Path(archive)


def write_outbox_index(root: Path | None = None) -> Path:
    """Rewrite `Outbox/INDEX.txt` — the answer to "what's finished and what do I use?".

    Regenerated from the kit.json files on every BUILD. It used to say here that it "can
    never drift from what is actually on disk", and that was wrong: nothing regenerates it
    when a kit folder is removed by hand, and on 2026-09-15 three superseded drafts were
    deleted and INDEX.txt went on listing them until it was rewritten. A build is the only
    event that refreshes it — plus `depthc library index`, which is that rewrite with no
    conversion attached, for exactly the delete-by-hand case.

    A kit folder can OUTLIVE ITS MEDIA, and the index says so. `kit.json` and `result.json`
    are tracked (audit DC-4); `depth.mp4` and the rest never are, so restoring a deleted kit
    from git returns the record and not the clip — which is what happened to Sha256Proof_001
    on 2026-09-15. Such a folder is listed as `MEDIA GONE — record only`, because
    every other line of this file tells the reader to open a README.txt and upload a
    depth.mp4, and for a husk both of those files are gone.

    Delete INDEX.txt itself and the next build (or that command) brings it back.
    """
    root = Path(root) if root else library_root()
    root.mkdir(parents=True, exist_ok=True)
    entries = library_list(root)

    lines = [
        "FINISHED DEPTH MOTION KITS",
        "==========================",
        "",
        "Each folder below is one finished kit. Open its README.txt for instructions.",
        "",
        "For every kit, the only file you hand to Seedance is:",
        "    depth.mp4     upload as the VIDEO reference. That is the whole handover.",
        "",
        "In OSIDE you type NO binding phrase, on either path. Uploading depth.mp4 by hand,",
        "an uploaded video ref has no binding field and the Composer writes its own line;",
        "coming in through OSIDE's 'Import kit' door — merged 2026-09-15 — you point it at",
        "the kit folder and that door fills the field from binding.txt itself, hash-checked",
        "against kit.json. binding.txt is for surfaces that HAVE a per-reference binding",
        "field (see each kit's README.txt).",
        "",
        "Everything else in a kit folder is reference or archive — do not upload it.",
        "",
        "`motion` is the mean grey-level change per frame in depth space: how much signal the",
        "ref actually carries. 4.40 = a sword duel that behaved, 0.94 = a crawl whose",
        "generation started late. A kit with no FIRED line has never been tested.",
        "",
    ]

    if not entries:
        lines.append("(nothing here yet — drop clips in Inbox/ and run: depthc inbox)")
        text = "\n".join(lines) + "\n"
        write_replacing(root / "INDEX.txt", text)
        return root / "INDEX.txt"

    lines.append(f"{len(entries)} kit{'s' if len(entries) != 1 else ''}:")
    lines.append("")
    for entry in entries:
        depth = entry.manifest.get("depth", {})
        source = entry.manifest.get("source", {})
        motion = entry.manifest.get("motion") or {}
        trim = source.get("trim")
        media_gone = not (entry.path / depth.get("filename", "depth.mp4")).exists()
        lines += [
            f"  {entry.name}" + ("   ** MEDIA GONE — record only **" if media_gone else ""),
            f"      {depth.get('width', '?')}x{depth.get('height', '?')} · "
            f"{entry.duration:g}s · check: {entry.tier}"
            + (f" · motion {motion['energy']:.2f} {motion.get('sparkline', '')}"
               if motion.get("energy") is not None else ""),
            f"      from {source.get('filename', '?')}"
            + (f", trimmed {trim[0]:g}-{trim[1]:g}s" if trim else ""),
        ]
        if media_gone:
            lines.append("      Its depth.mp4 is not here and git cannot return it — Outbox "
                         "media is never tracked.")
            lines.append("      Nothing here can be fired. kit.json + result.json are kept "
                         "as the record of what was built and what it did.")
        if entry.manifest.get("description"):
            lines.append(f"      {entry.manifest['description']}")
        fires = entry.fires
        if fires:
            last = fires[-1]
            lines.append(
                f"      FIRED {len(fires)}x · last: {last.get('verdict', '?')}"
                + (f" — {last['notes']}" if last.get("notes") else "")
            )
        lines.append("")

    text = "\n".join(lines)
    write_replacing(root / "INDEX.txt", text)
    return root / "INDEX.txt"


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

    @property
    def energy(self) -> float | None:
        value = (self.manifest.get("motion") or {}).get("energy")
        return float(value) if value is not None else None

    @property
    def fires(self) -> list[dict]:
        return read_results(self.path).get("fires", [])

    @property
    def verdict(self) -> str:
        """The most recent fire's verdict, or "untested" if it has never been fired."""
        fires = self.fires
        return fires[-1].get("verdict", "?") if fires else "untested"


def library_list(root: Path | None = None, tag: str | None = None) -> list[LibraryEntry]:
    """Scan the library for kits. A malformed kit.json is skipped, never fatal."""
    root = Path(root) if root else library_root()
    if not root.is_dir():
        return []

    entries: list[LibraryEntry] = []
    for manifest_path in sorted(root.glob(f"*/{KIT_MANIFEST}")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError):  # ValueError holds bad JSON and bytes that are not text
            continue
        if not isinstance(manifest, dict):  # valid JSON and still not a manifest: skipped the same way
            continue
        entry = LibraryEntry(path=manifest_path.parent, manifest=manifest)
        if tag and tag.lower() not in [t.lower() for t in entry.tags]:
            continue
        entries.append(entry)
    return entries
