"""`depthc` command line. Every flag here has a matching control in the Gradio UI."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer

from depthc import __version__
# Deliberately imported eagerly (it is torch-free): SetupIncomplete is the most likely
# FIRST-run failure, and every command that loads the model must turn it into its
# hand-written "Run: bash scripts/setup.sh" message rather than a traceback.
from depthc.vendor_path import SetupIncomplete
# Also eager, and also torch-free (numpy only): the `--dialect` option's default and its
# help text are evaluated at import time, and they must name the dialects that actually
# exist rather than a literal that can drift from them. Importing it reads no envelope.
from depthc import checks

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Convert a reference clip into a Seedance-ready depth motion kit.",
)
library_app = typer.Typer(no_args_is_help=True, help="Browse finished kits in Outbox/.")
app.add_typer(library_app, name="library")

BOLD, DIM, GREEN, YELLOW, RED, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m",
)


def _die(message: str) -> None:
    typer.echo(f"{RED}error:{OFF} {message}", err=True)
    raise typer.Exit(1)


def _warn(message: str) -> None:
    typer.echo(f"{YELLOW}warning:{OFF} {message}")


# These two grade nothing: `result` records paid evidence and `library` reads what is already
# built, so neither waits on a table it never looks at (audit DC-6). Every other command is
# refused here rather than partway through a depth pass.
NEEDS_NO_ENVELOPE = {"result", "library"}


@app.callback()
def _refuse_a_broken_envelope(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand in NEEDS_NO_ENVELOPE:
        return
    try:
        checks.dialect_for()
    except checks.EnvelopeError as e:
        _die(
            f"{e}\n  Grading needs that file. Replace it from a good copy (git restore "
            "depthc/seedance-envelope.json); depthc result and depthc library still work."
        )


def _variant_ratio_basis(kit_dir: Path, variant: str) -> tuple[float | None, str | None]:
    """How to put a fired VARIANT's energy back on the base depth.mp4's scale.

    Returns (ref_scale, ratio_note). `ref_scale` is the measured variant/base energy ratio,
    read from kit.json where both were measured on the actual files. `ratio_note` is set
    instead when no scale correction is meaningful, and it takes the ratio out of the
    verdict the way `geometry_note` already does.

    Why this exists: `compare --variant midgrey` compared a luma-capped reference against a
    full-range generation and reported a PERFECTLY ADHERENT fire as +36% over-following —
    louder than every 1080p failure on record, on a workflow the kit README tells you to
    run. The variant was measured, the correction was not applied.
    """
    import json

    from depthc.kit import KIT_MANIFEST

    # Canny edges are not a scaled copy of the envelope, they are a different signal.
    # No linear factor puts them on the base scale, so the amplitude comparison is void.
    if variant == "edges":
        return None, (
            "the 'edges' variant is a binary Canny series, not a scaled depth envelope — "
            "its amplitude is not comparable with the generation's"
        )

    path = kit_dir / KIT_MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, (
            f"cannot read {KIT_MANIFEST} to find the '{variant}' variant's measured energy, "
            f"so the amplitude cannot be put on the base scale"
        )

    base = (manifest.get("motion") or {}).get("energy")
    entry = next(
        (v for v in (manifest.get("variants") or []) if v.get("name") == variant), None
    )
    measured = (entry or {}).get("energy")
    if not base or not measured:
        return None, (
            f"kit.json records no measured energy for '{variant}' (or for the base), so the "
            f"amplitude cannot be put on the base scale"
        )
    return float(measured) / float(base), None


def _check_kit_binding(kit_dir: Path) -> bool:
    """Say whether this kit's WORDS still belong to its clip. True when the check FAILED.

    Printed by every command that reads a finished kit, beside the note about a missing
    `depth.sha256`, and for the same reason: OSIDE's Import kit door (merged 2026-09-15) fills a
    reference's binding field from `binding.txt`, so the two halves of a kit have to be
    provably from one build. The blockout kit binds both by SHA-256 and that pair caught one
    kit's words riding on another kit's clip.

    A kit built before 2026-09-15 has no recorded hash. That is named once, like the depth
    one, and is NEVER treated as a failure — a hash is not invented at read time.
    """
    from depthc import checks, kit as kitlib

    if kitlib.binding_sha256(kit_dir) is None:
        _warn(
            "kit.json carries no binding.sha256 (kit built before 2026-09-15) — nothing "
            "binds this kit's binding.txt to its depth.mp4, so an import door cannot tell "
            "the two came from one build; rebuild the kit to record it."
        )
        return False

    finding = kitlib.verify_binding(kit_dir)
    if finding is None or finding.tier is not checks.Tier.FAIL:
        return False
    typer.echo(f"{RED}[FAIL]{OFF} {finding.check}: {finding.detail}")
    typer.echo(f"{DIM}      → {finding.remedy}{OFF}")
    return True


def _abandon_kit(kit_dir) -> None:
    """Remove a kit folder whose manifest never got written.

    A kit without kit.json is invisible to `library list` and INDEX.txt but still burns its
    numbered slot and looks like a real kit in Finder. If the build died partway, the debris
    goes; the source clip is untouched and the next run re-allocates the same slot.
    """
    import shutil

    from depthc.kit import KIT_MANIFEST

    if kit_dir is not None and kit_dir.is_dir() and not (kit_dir / KIT_MANIFEST).exists():
        shutil.rmtree(kit_dir, ignore_errors=True)


@dataclass
class Conversion:
    """Everything downstream needs from one convert run."""

    info: object
    frames: object          # [N, H, W, 3] uint8, cropped + scaled (the ORIGINAL pixels)
    result: object          # engine.DepthResult
    crop: object            # Crop | None
    size: Tuple[int, int]
    trim: Optional[Tuple[float, float]]


def _resolve_crop(info, crop: str, trim, announce: bool = True):
    from depthc.video_io import Crop, detect_crop

    if crop == "none":
        return None
    if crop != "auto":
        return Crop.parse(crop)

    rect = detect_crop(info, trim=trim)
    if rect and announce:
        typer.echo(
            f"{DIM}matte detected — cropping to {rect.width}×{rect.height} "
            f"({rect.removed_fraction(info) * 100:.0f}% of frame removed; "
            f"--crop none to keep it){OFF}"
        )
    return rect


def _convert(
    input: Path,
    *,
    trim,
    resolution: str,
    fps: str,
    crop: str,
    invert: bool,
    input_size: int,
    precision: str,
    allow_long: bool,
    snap: bool = False,
    quiet: bool = False,
    dialect: str | None = None,
) -> Conversion:
    """Probe → validate → decode → infer. Shared by `convert` and `kit`.

    `dialect` selects the model envelope every limit below is read from — the ref ceiling
    and the output grid differ between Seedance 2.0 (15s) and 2.5 (30s), so a 20s beat is
    a hard refusal on one and ordinary on the other.
    """
    from depthc import checks, engine
    from depthc.video_io import probe, read_frames, resolve_fps, target_size

    info = probe(input)
    if not quiet:
        typer.echo(f"{BOLD}{info.path.name}{OFF}  {DIM}{info.summary()}{OFF}")

    # An impossible trim is a typo, not a request. Rejected BEFORE snapping: snap_trim
    # exists to land a legal span on the output grid, and given a negative span or a
    # window past the end of the file it would instead manufacture a valid-looking clip
    # the user never asked for (--trim 10 5 became a 10-14s kit; --trim 20 25 on a 15s
    # source relocated to 10-15s).
    if trim:
        start, end = trim
        if start < 0 or end <= start:
            _die(f"--trim {start:g} {end:g} is not a window (need 0 ≤ start < end).")
        if end > info.duration + 0.05:
            _die(
                f"--trim {start:g} {end:g} runs past the end of the clip "
                f"({info.duration:.2f}s)."
            )

    # The 15s law is enforced on the trim AS GIVEN, before snapping. Snapping exists to land
    # a legal span on the output grid, never to shorten an illegal one: the old order let
    # snap_trim clamp a 20s trim to 15s silently, so this gate could not fire and
    # --allow-long was dead unless --no-snap was also passed.
    span = (trim[1] - trim[0]) if trim else info.duration
    d = checks.dialect_for(dialect)
    max_ref = d.ref_seconds[1]
    if span > max_ref and not allow_long:
        wider = next(
            (o for o in checks.DIALECTS.values() if o.ref_seconds[1] > max_ref), None
        )
        hint = (
            f"\n  {wider.name} accepts up to {wider.ref_seconds[1]:g}s — "
            f"rebuild with --dialect {wider.name} if that is the target model."
            if wider and span <= wider.ref_seconds[1] else ""
        )
        _die(
            f"Trimmed length is {span:.1f}s. {d.name} caps video references at "
            f"{max_ref:g}s and adherence degrades past ~{d.sweet_spot[1]:g}s.\n"
            f"  Trim to the one clear motion signal, or pass --allow-long to override."
            f"{hint}"
        )

    # Snap before anything is decoded: a ref shorter than the generation it drives leaves the
    # model to invent the tail, and the fix is free at this point. An --allow-long overlength
    # span is left alone — no output length can match it anyway.
    if snap and span <= max_ref:
        window = trim if trim else (0.0, info.duration)
        snapped, note = checks.snap_trim(window[0], window[1], info.duration, d)
        if note:
            trim = snapped
            if not quiet:
                typer.echo(f"{DIM}{note}{OFF}")
        span = (trim[1] - trim[0]) if trim else info.duration

    if span > 8:
        _warn(
            f"{span:.1f}s exceeds the 3-8s sweet spot — adherence degrades. "
            f"Consider trimming to a single motion signal."
        )

    crop_rect = _resolve_crop(info, crop, trim, announce=not quiet)
    size = target_size(info, resolution, crop=crop_rect)
    out_fps = resolve_fps(info, fps)

    if not quiet:
        typer.echo(f"{DIM}decoding at {size[0]}×{size[1]} @ {out_fps:g} fps…{OFF}")
    frames = read_frames(info, trim=trim, fps=out_fps, size=size, crop=crop_rect)

    device = engine.pick_device()
    if device == "cpu":
        _warn("MPS unavailable — running on CPU. This will be slow.")

    total = engine.expected_windows(frames.shape[0])
    with typer.progressbar(length=total, label="depth") as bar:
        seen = {"n": 0}

        def on_window(done: int, _total: int) -> None:
            bar.update(done - seen["n"])
            seen["n"] = done

        result = engine.infer(
            frames,
            out_fps,
            input_size=input_size,
            precision=precision,
            invert=invert,
            device=device,
            on_window=on_window,
            on_warn=lambda m: typer.echo(f"\n{YELLOW}warning:{OFF} {m}"),
        )

    return Conversion(
        info=info, frames=frames, result=result, crop=crop_rect, size=size, trim=trim
    )


# ---------------------------------------------------------------------------- commands

@app.command()
def version() -> None:
    """Print the tool version."""
    typer.echo(f"depthc {__version__}")


@app.command()
def probe(input: Path = typer.Argument(..., help="Video file to inspect")) -> None:
    """Report duration, fps, resolution, rotation, and any letterbox matte."""
    from depthc.video_io import VideoIOError, detect_crop, probe as probe_file

    try:
        info = probe_file(input)
        matte = detect_crop(info)
    except (VideoIOError, SetupIncomplete) as e:
        _die(str(e))

    typer.echo(f"{BOLD}{info.path.name}{OFF}")
    typer.echo(f"  {info.summary()}")
    typer.echo(f"  audio: {'yes' if info.has_audio else 'none'}")
    typer.echo(f"  pixels: {info.pixels:,}")
    if matte:
        typer.echo(
            f"  matte: {matte} → {matte.width}×{matte.height} "
            f"({matte.removed_fraction(info) * 100:.0f}% removed by --crop auto)"
        )
    else:
        typer.echo("  matte: none (full frame)")


@app.command()
def convert(
    input: Path = typer.Argument(..., help="Source video (MP4/MOV)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output depth.mp4 path"),
    trim: Optional[Tuple[float, float]] = typer.Option(
        None, "--trim", help="START END in seconds; 3-8s is the adherence sweet spot"
    ),
    resolution: str = typer.Option("720p", "--resolution", help="720p | 480p | source"),
    crop: str = typer.Option(
        "auto",
        "--crop",
        help="auto (strip letterbox/pillarbox bars, which a depth model otherwise reads "
        "as a near surface) | none | W:H:X:Y",
    ),
    fps: str = typer.Option(
        "conform-24", "--fps", help="conform-24 (matches Seedance cadence) | keep | <number>"
    ),
    invert: bool = typer.Option(False, "--invert", help="Make far bright instead of near"),
    keep_audio: bool = typer.Option(
        False,
        "--keep-audio",
        help="Mux the source audio onto depth.mp4. For your own beat reference while "
        "editing only — Seedance ignores reference-video audio; rhythm sync is the "
        "@Audio slot's job.",
    ),
    input_size: int = typer.Option(518, "--input-size", help="Inference short side, multiple of 14"),
    precision: str = typer.Option(
        "fp32",
        "--precision",
        help="fp32 (default; fp16 autocast returns NaN on MPS and is slower) | fp16",
    ),
    preview: bool = typer.Option(False, "--preview", help="Also render an original|depth MP4"),
    allow_long: bool = typer.Option(False, "--allow-long", help="Permit clips over 15s"),
) -> None:
    """Convert a clip to a grayscale depth video."""
    from depthc import engine
    from depthc.video_io import VideoIOError, encode_gray, mux_audio, side_by_side

    # Every encode runs ffmpeg -y, so a collision with the source is silent and fatal:
    # `--out clip.mp4` on clip.mp4 would replace the original footage with its depth pass.
    # Checked against every path this command writes, before any inference is paid for.
    out_path = Path(out) if out else input.with_name(f"{input.stem}_depth.mp4")
    source = input.resolve()
    writes = [out_path]
    if keep_audio:
        writes.append(out_path.with_name(f"{out_path.stem}_audio{out_path.suffix}"))
    if preview:
        writes.append(out_path.with_name(f"{out_path.stem}_preview.mp4"))
    for path in writes:
        if path.resolve() == source:
            _die(
                f"--out would overwrite the source clip itself ({input.name}). "
                f"Pick a different output name."
            )

    try:
        job = _convert(
            input, trim=trim, resolution=resolution, fps=fps, crop=crop, invert=invert,
            input_size=input_size, precision=precision, allow_long=allow_long,
        )

        encode_gray(job.result.frames, out_path, job.result.fps)

        if keep_audio:
            if not job.info.has_audio:
                _warn("--keep-audio: source has no audio track.")
            else:
                tmp = out_path.with_name(f"{out_path.stem}_audio{out_path.suffix}")
                mux_audio(out_path, job.info.path, tmp, trim=trim)
                tmp.replace(out_path)

        if preview:
            preview_path = out_path.with_name(f"{out_path.stem}_preview.mp4")
            side_by_side(job.frames, job.result.frames, preview_path, job.result.fps)
            typer.echo(f"  preview  {preview_path}")

    except (VideoIOError, SetupIncomplete) as e:
        _die(str(e))
    except engine.NotEnoughMemory as e:
        _die(str(e))
    except engine.NonFiniteDepth as e:
        _die(f"{e}\n  --precision fp32 is the default; if that also fails, the MPS backend "
             f"itself is suspect. Report the clip and `depthc probe` output.")
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    mb = out_path.stat().st_size / 1_000_000
    typer.echo(
        f"{GREEN}✓{OFF} {out_path}  {DIM}{mb:.1f} MB · {job.result.count} frames · "
        f"{job.result.device}/{job.result.dtype} · {job.result.elapsed:.1f}s "
        f"({job.result.ms_per_frame:.0f} ms/frame){OFF}"
    )


@app.command()
def kit(
    input: Path = typer.Argument(..., help="Source video (MP4/MOV)"),
    tags: str = typer.Option("", "--tags", help="Comma-separated; also names the kit folder"),
    desc: str = typer.Option("", "--desc", help="Free-text description for the manifest"),
    trim: Optional[Tuple[float, float]] = typer.Option(
        None, "--trim", help="START END in seconds; 3-8s is the adherence sweet spot"
    ),
    resolution: str = typer.Option("720p", "--resolution", help="720p | 480p | source"),
    crop: str = typer.Option("auto", "--crop", help="auto | none | W:H:X:Y"),
    fps: str = typer.Option("conform-24", "--fps", help="conform-24 | keep | <number>"),
    invert: bool = typer.Option(False, "--invert", help="Make far bright instead of near"),
    faces: bool = typer.Option(
        True,
        "--faces/--no-faces",
        help="Whether the source shows real people. Defaults to yes: still.png is cut from "
        "the ORIGINAL pixels, so it carries a real photographic face that Seedance rejects "
        "on image inputs. Pass --no-faces for landscapes, objects, animation.",
    ),
    still: str = typer.Option(
        "auto", "--still", help="auto (sharpest of the first 24 frames) | first | <N>"
    ),
    input_size: int = typer.Option(518, "--input-size", help="Inference short side, multiple of 14"),
    precision: str = typer.Option("fp32", "--precision", help="fp32 | fp16"),
    dialect: str = typer.Option(
        checks.DEFAULT_DIALECT, "--dialect",
        help="Target model: selects BOTH the prompt template and the constraint envelope "
             f"({', '.join(checks.DIALECT_NAMES)})",
    ),
    library: Optional[Path] = typer.Option(
        None, "--library", help="Kit output root (default Outbox/ in the project)"
    ),
    archive: bool = typer.Option(
        False,
        "--archive",
        help="Move the source into Processed/ after a successful kit. Skipped on FAIL, so a "
        "rejected clip stays in the inbox where you can see it.",
    ),
    preview: bool = typer.Option(False, "--preview", help="Also write preview.mp4 into the kit"),
    zip_output: bool = typer.Option(False, "--zip", help="Also write <kit>.zip beside the folder"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Mux source audio onto depth.mp4"),
    allow_long: bool = typer.Option(False, "--allow-long", help="Permit clips over 15s"),
    snap: bool = typer.Option(
        True,
        "--snap/--no-snap",
        help="Widen the trim to a whole number of seconds (min 4) so the ref length is "
        "exactly requestable as an output length. Output duration is sent as an integer, so "
        "an unsnapped ref leaves part of the generation running past its end.",
    ),
    variants: bool = typer.Option(
        False,
        "--variants",
        help="Also write contrast-stretched, inverted, luma-capped (midgrey) and "
        "edge-map versions of the same beat into variants/, to compare across live-fires. "
        "Upload exactly one per generation.",
    ),
) -> None:
    """Probe, convert, check, and package a full Seedance Kit into the Motion Library."""
    from depthc import checks, engine, kit as kitlib, motion as motionlib
    from depthc.video_io import (
        VideoIOError, encode_gray, mux_audio, probe as probe_file, side_by_side, trim_copy,
    )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Validate the cheap arguments BEFORE the multi-minute inference. A typo'd --still or
    # --dialect used to surface only after the conversion had already been paid for, as a
    # raw traceback beside a half-built kit folder.
    if still not in ("auto", "first"):
        try:
            int(still)
        except ValueError:
            _die(f"--still expects auto, first, or a frame number (got {still!r})")
    try:
        kitlib.render_binding(dialect)
        # Render the WHOLE template with stand-in values, not just the binding block: a
        # stray brace anywhere in it kills render_prompt after the inference otherwise.
        kitlib.render_prompt(
            kit_name="preflight", duration=5.0, width=1280, height=720, fps=24.0,
            still_has_real_faces=False, dialect=dialect,
        )
    except (FileNotFoundError, ValueError) as e:
        _die(str(e))

    kit_dir = None
    try:
        job = _convert(
            input, trim=trim, resolution=resolution, fps=fps, crop=crop, invert=invert,
            input_size=input_size, precision=precision, allow_long=allow_long, snap=snap,
            dialect=dialect,
        )
        trim = job.trim

        root = kitlib.library_root(library)
        kit_dir = kitlib.allocate_kit_dir(root, tag_list, fallback=input.stem)
        paths = kitlib.KitPaths(kit_dir)

        encode_gray(job.result.frames, paths.depth, job.result.fps)

        # Post-encode size guard (PLAN.md §1 encode spec) — on the SILENT encode, before any
        # audio mux. The old order re-encoded AFTER muxing, which overwrote the just-added
        # audio track without saying so; and the audio bytes themselves could be what pushed
        # the file over the limit and triggered their own removal.
        #
        # The ceiling is the TARGET MODEL's, and crossing it costs picture: this re-encode is
        # lossy, on the one asset where re-encode artifacts are the signal rather than noise.
        # It ran against 50 MB until 2026-08-24 — a quarter of the vendor's actual 200 MB —
        # so depth maps were being degraded to satisfy a limit that did not exist.
        _max_mb = checks.dialect_for(dialect).max_file_mb
        if paths.depth.stat().st_size > _max_mb * 1_000_000:
            _warn(f"depth.mp4 exceeded {_max_mb:g} MB — re-encoding at crf 20.")
            encode_gray(job.result.frames, paths.depth, job.result.fps, crf=20)

        audio_kept = False
        if keep_audio and job.info.has_audio:
            tmp = paths.depth.with_name("depth_audio.mp4")
            mux_audio(paths.depth, job.info.path, tmp, trim=trim)
            if tmp.stat().st_size > _max_mb * 1_000_000:
                tmp.unlink()
                _warn(
                    f"--keep-audio: the audio track pushed depth.mp4 over "
                    f"{_max_mb:g} MB — kept the silent version instead."
                )
            else:
                tmp.replace(paths.depth)
                audio_kept = True
        elif keep_audio:
            _warn("--keep-audio: source has no audio track.")

        still_index = None if still == "auto" else (0 if still == "first" else int(still))
        chosen = kitlib.write_still(job.frames, paths.still, index=still_index)
        trim_copy(job.info, paths.source, trim=trim)
        if preview:
            side_by_side(job.frames, job.result.frames, paths.preview, job.result.fps)

        # Measured on the frames themselves rather than the encoded file: same arrays, same
        # count, same fps for both sides, which is what makes the retention ratio meaningful.
        depth_motion = motionlib.measure(job.result.frames, job.result.fps)
        source_motion = motionlib.measure(job.frames, job.result.fps, with_flow=False)
        motion_block = motionlib.manifest_block(depth_motion, source_motion)

        motionlib.contact_sheet(job.result.frames, job.result.fps, paths.sheet)

        variant_blocks: list[dict] = []
        if variants:
            paths.variants.mkdir(exist_ok=True)
            for name, (transform, description) in motionlib.VARIANTS.items():
                frames = transform(job.result.frames)
                filename = f"{name}.mp4"
                variant_path = paths.variants / filename
                encode_gray(frames, variant_path, job.result.fps)
                # Same size gate as the base ref: every variant is presented as
                # uploadable, and Seedance's cap does not care which file it was.
                # (Edges especially — Canny flicker compresses badly.)
                if variant_path.stat().st_size > _max_mb * 1_000_000:
                    _warn(f"variants/{filename} exceeded {_max_mb:g} MB — re-encoding at crf 20.")
                    encode_gray(frames, variant_path, job.result.fps, crf=20)
                variant_blocks.append({
                    "name": name,
                    "filename": filename,
                    "description": description,
                    "energy": motionlib.measure(frames, job.result.fps, with_flow=False).energy,
                })

        depth_bytes = paths.depth.stat().st_size
        # Hashed here, after the size-gate re-encode and the audio mux above, so it names
        # the bytes that get uploaded — OSIDE hashes the upload and joins on this value.
        depth_sha = kitlib.sha256_file(paths.depth)
        depth_info = probe_file(paths.depth)
        report = checks.check(
            duration=depth_info.duration,
            width=depth_info.width,
            height=depth_info.height,
            fps=depth_info.fps,
            size_bytes=depth_bytes,
            container="mp4",
            still_has_real_faces=faces,
            motion=depth_motion,
            retention=motionlib.retention(depth_motion, source_motion),
            dialect=dialect,
        )

        paths.prompt.write_text(
            kitlib.render_prompt(
                kit_name=kit_dir.name,
                duration=depth_info.duration,
                width=depth_info.width,
                height=depth_info.height,
                fps=depth_info.fps,
                still_has_real_faces=faces,
                dialect=dialect,
            )
        )

        paths.binding.write_text(kitlib.render_binding(dialect) + "\n")
        # Hashed straight after the write, for the same reason depth.mp4 is: an import door
        # that fills a reference's binding field from this file can only trust it if the
        # words and the clip are bound to the same build.
        binding_sha = kitlib.sha256_file(paths.binding)

        manifest = kitlib.build_manifest(
            kit_name=kit_dir.name,
            source_path=job.info.path,
            source_sha256=kitlib.sha256_file(job.info.path),
            source_duration=job.info.duration,
            source_fps=job.info.fps,
            source_size=(job.info.width, job.info.height),
            trim=trim,
            crop=str(job.crop) if job.crop else None,
            depth_duration=depth_info.duration,
            depth_fps=depth_info.fps,
            depth_size=(depth_info.width, depth_info.height),
            depth_bytes=depth_bytes,
            depth_sha256=depth_sha,
            binding_sha256=binding_sha,
            frame_count=job.result.count,
            still_index=chosen,
            still_has_real_faces=faces,
            tags=tag_list,
            description=desc,
            settings={
                "resolution": resolution,
                "fps": fps,
                "invert": invert,
                "inputSize": input_size,
                "precision": job.result.dtype,
                "crop": crop,
                # What actually happened, not what was asked: --keep-audio on a silent
                # source keeps nothing — and neither does a mux discarded for pushing the
                # file over the size cap. A rebuild from kit.json must not claim either.
                "keepAudio": audio_kept,
            },
            report=report,
            dialect=dialect,
            device=job.result.device,
            elapsed=job.result.elapsed,
            motion=motion_block,
            variants=variant_blocks,
            has_sheet=paths.sheet.exists(),
        )
        kitlib.write_manifest(paths, manifest)
        paths.readme.write_text(
            kitlib.render_kit_readme(manifest, report, has_preview=preview)
        )

        kitlib.write_outbox_index(root)

        if zip_output:
            kitlib.zip_kit(kit_dir)

    except (VideoIOError, FileNotFoundError, ValueError, SetupIncomplete) as e:
        _abandon_kit(kit_dir)
        _die(str(e))
    except engine.DepthFailure as e:
        _abandon_kit(kit_dir)
        _die(str(e))
    except KeyboardInterrupt:
        _abandon_kit(kit_dir)
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    typer.echo(f"\n{BOLD}{kit_dir.name}{OFF}  {DIM}{kit_dir}{OFF}")
    typer.echo(checks.format_report(report))

    if not report.ok:
        if archive:
            typer.echo(f"{DIM}not archived — a FAIL stays in the inbox.{OFF}")
        raise typer.Exit(1)

    if archive:
        moved = kitlib.archive_source(input)
        typer.echo(f"{DIM}source → {moved}{OFF}")


def _print_scout(clip: Path, result, indent: str = "    ") -> None:
    """Render a scout pass: the per-second profile, then the ranked windows."""
    from depthc import motion as motionlib

    profile = result.profile
    typer.echo(
        f"{indent}{DIM}motion profile (depth space, per second, 24fps-equivalent; "
        f"{result.elapsed:.0f}s scout){OFF}"
    )
    typer.echo(f"{indent}{profile.spark}   {DIM}overall {profile.energy:.2f}{OFF}")
    typer.echo(
        f"{indent}{DIM}"
        + " ".join(f"{second}s:{value:.1f}" for second, value in enumerate(result.per_second))
        + OFF
    )
    if profile.camera is not None:
        kind = "camera-path" if profile.camera >= motionlib.CAMERA_DOMINANT else "subject-motion"
        typer.echo(f"{indent}{DIM}{profile.camera * 100:.0f}% whole-frame movement → {kind}{OFF}")

    if not result.windows:
        typer.echo(f"{indent}{YELLOW}too short to propose a {result.target_seconds:g}s window{OFF}")
        return

    typer.echo(f"{indent}best {result.target_seconds:g}s windows:")
    for rank, window in enumerate(result.windows, 1):
        mark = GREEN + "→" + OFF if rank == 1 else " "
        note = window.note()
        typer.echo(
            f"{indent}{mark} {window.start:g}-{window.end:g}s   "
            f"energy {window.energy:.2f}  head {window.head:.2f}  "
            f"balance {window.balance:.2f}  score {window.score:.2f}"
            + (f"   {YELLOW}{note}{OFF}" if note else "")
        )

    best = result.best
    typer.echo(
        f"{indent}{YELLOW}!{OFF} ranked by HOW MUCH motion, not how many things are moving — "
        f"check sheet.png for a second figure walking into the window."
    )
    typer.echo(
        f"{indent}{DIM}./.tools/uv run depthc kit {shlex.quote(str(clip))} "
        f"--trim {best.start:g} {best.end:g} --dialect {checks.DEFAULT_DIALECT} "
        f"--tags \"...\" --desc \"...\" --preview --archive{OFF}"
    )


@app.command()
def scout(
    input: Path = typer.Argument(..., help="Source video to profile"),
    target: float = typer.Option(5.0, "--target", help="Window length to look for, in seconds"),
    scout_fps: float = typer.Option(12.0, "--scout-fps", help="Sampling rate for the pass"),
    long_edge: int = typer.Option(384, "--long-edge", help="Inference long edge for the pass"),
) -> None:
    """Find where the motion is in a clip, before committing to a trim.

    Runs a deliberately cheap depth pass over the WHOLE clip, then ranks every window of the
    target length by how much motion survives into depth — penalising a dead first beat, a
    cut, and a dead stretch anywhere inside.

    This is the step that was previously a judgement made by eye, and the prone-crawl kit is
    what it got wrong: the fall at 3s was the motion, it looked blurry, so the trim kept the
    still aftermath instead. The scorer reports softness and refuses to prefer a frozen frame
    for it.
    """
    from depthc import engine, motion as motionlib
    from depthc.video_io import VideoIOError

    try:
        typer.echo(f"{BOLD}{input.name}{OFF}")
        with typer.progressbar(length=100, label="scout") as bar:
            state = {"n": 0}

            def on_window(done: int, total: int) -> None:
                target_pct = int(done / max(1, total) * 100)
                bar.update(target_pct - state["n"])
                state["n"] = target_pct

            result = motionlib.scout_clip(
                input, target, scout_fps=scout_fps, long_edge=long_edge, on_window=on_window
            )
        _print_scout(input, result, indent="  ")
    except (VideoIOError, SetupIncomplete, engine.DepthFailure) as e:
        _die(str(e))
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)


@app.command()
def inbox(
    path: Optional[Path] = typer.Option(None, "--inbox", help="Inbox folder to scan"),
    target: float = typer.Option(5.0, "--target", help="Window length to propose, in seconds"),
    no_scout: bool = typer.Option(
        False, "--no-scout", help="List and probe only — skip the depth pass"
    ),
) -> None:
    """Triage the clips waiting in the inbox.

    Reports what each one is, where its motion actually sits, and which window to cut — so the
    trim call is made from measurements rather than from a guess. Converts nothing and writes
    nothing: the scout pass is read-only and the proposed window is a proposal.
    """
    from depthc import engine, kit as kitlib, motion as motionlib
    from depthc.video_io import VideoIOError, detect_crop, probe as probe_file

    root = kitlib.inbox_dir(path)
    clips = kitlib.inbox_clips(root)
    if not clips:
        typer.echo(f"Inbox is empty: {root}")
        return

    typer.echo(f"{BOLD}{len(clips)} clip{'s' if len(clips) != 1 else ''} in {root}{OFF}\n")
    for clip in clips:
        try:
            info = probe_file(clip)
        except VideoIOError as e:
            typer.echo(f"{RED}✗{OFF} {clip.name}\n    {e}")
            continue

        typer.echo(f"{BOLD}{clip.name}{OFF}")
        typer.echo(f"    {info.summary()}")

        try:
            matte = detect_crop(info)
        except VideoIOError as e:
            typer.echo(f"    {RED}{e}{OFF}")
            continue
        if matte:
            typer.echo(
                f"    {DIM}matte {matte} → {matte.width}×{matte.height} "
                f"({matte.removed_fraction(info) * 100:.0f}% removed automatically){OFF}"
            )

        if info.duration < 2:
            typer.echo(f"    {RED}under the 2s minimum — unusable as a video reference{OFF}")
            typer.echo()
            continue
        if info.duration > 8:
            typer.echo(
                f"    {YELLOW}needs trimming{OFF} — {info.duration:.1f}s is past the 3-8s "
                f"sweet spot. Pick the single clearest motion beat."
            )
        else:
            typer.echo(f"    {GREEN}usable whole{OFF} — inside the 3-8s sweet spot.")

        if not no_scout:
            try:
                result = motionlib.scout_clip(clip, target)
                _print_scout(clip, result)
            except (VideoIOError, ValueError, SetupIncomplete, engine.DepthFailure) as e:
                typer.echo(f"    {YELLOW}scout failed:{OFF} {e}")
        typer.echo()


@app.command()
def result(
    kit_name: str = typer.Argument(..., help="Kit folder name, or a path to one"),
    verdict: str = typer.Option(
        ...,
        "--verdict",
        help="adhered | partial | ignored | late | rejected — what the ref actually did",
    ),
    surface: str = typer.Option("", "--surface", help="Where it was fired: oside, ark, other"),
    model: str = typer.Option("", "--model", help="Model id, e.g. ark-seedance-2.0"),
    seconds: Optional[float] = typer.Option(None, "--seconds", help="Output length requested"),
    cost: Optional[float] = typer.Option(None, "--cost", help="USD spent on this generation"),
    output: str = typer.Option("", "--output", help="URL or path of what came back"),
    variant: str = typer.Option(
        "", "--variant", help="Which ref was used: base, contrast, invert, midgrey, edges"
    ),
    notes: str = typer.Option("", "--notes", help="What actually happened, in your words"),
    generation: Optional[str] = typer.Option(
        None, "--generation",
        help="OSIDE only: the generation this fire became — its id (UUID v4) from its row/URL",
    ),
    library: Optional[Path] = typer.Option(None, "--library", help="Kit root (default Outbox/)"),
) -> None:
    """Record what happened when a kit was fired.

    The library has no other way to learn. A generation is the only evidence that says whether
    any of this works, it costs money, and before this command it lived in a human's memory —
    the prone-crawl diagnosis had to be re-derived from measurements a day later.
    """
    from depthc import kit as kitlib

    root = kitlib.library_root(library)
    kit_dir = Path(kit_name) if Path(kit_name).is_dir() else root / kit_name
    if not (kit_dir / kitlib.KIT_MANIFEST).is_file():
        _die(f"No kit at {kit_dir} (expected a {kitlib.KIT_MANIFEST} inside)")

    # Checked before the write and acted on after it: a generation is the only evidence this
    # library learns from and it costs money, so a broken binding.txt never blocks recording
    # the fire — it just makes the command exit non-zero once the evidence is safely on disk.
    binding_failed = _check_kit_binding(kit_dir)

    try:
        log = kitlib.record_result(
            kit_dir, verdict=verdict, surface=surface, model=model,
            output_seconds=seconds, cost_usd=cost, output_ref=output,
            variant=variant, notes=notes, generation_id=generation,
        )
    except ValueError as e:
        _die(str(e))

    # The kit's own library gets the refreshed INDEX — for a kit given by path, that is its
    # parent, which is not necessarily the default/--library root.
    kitlib.write_outbox_index(kit_dir.parent)
    fires = log["fires"]
    typer.echo(
        f"{GREEN}✓{OFF} {kit_dir.name}: {BOLD}{verdict}{OFF} "
        f"{DIM}({kitlib.VERDICTS[verdict]}) — fire {len(fires)} of this kit{OFF}"
    )
    typer.echo(f"{DIM}{kit_dir / 'result.json'}{OFF}")
    if generation and not model:
        # Recorded either way. The only keyed fire in the library went in like this, and what
        # it ran on now rests on its notes (audit DC-5).
        _warn("no --model named, so this fire does not say which model made it; add it to result.json.")

    if binding_failed:
        raise typer.Exit(1)


def _depth_profile(info, resolution: str, input_size: int, precision: str,
                   trim_seconds: float | None = None):
    """Decode a clip (matte stripped, optionally trimmed), run the depth pass, profile it.

    Shared by `compare` (the generation side) and `describe`, so a lone clip is measured on
    exactly the frames and grid a compare would use — the same-frames law that the ratio
    bug taught. Returns (crop_rect, MotionProfile at 24 fps).
    """
    from depthc import engine, motion as motionlib
    from depthc.video_io import detect_crop, read_frames, target_size

    rect = detect_crop(info)
    if rect:
        typer.echo(f"{DIM}matte detected: cropping to {rect.width}×{rect.height}{OFF}")
    trim = (0.0, trim_seconds) if trim_seconds is not None else None
    size = target_size(info, resolution, crop=rect)
    frames = read_frames(info, trim=trim, fps=24.0, size=size, crop=rect)

    device = engine.pick_device()
    if device == "cpu":
        _warn("MPS unavailable — running on CPU. This will be slow.")
    total = engine.expected_windows(frames.shape[0])
    with typer.progressbar(length=total, label="depth") as bar:
        seen = {"n": 0}

        def on_window(done: int, _total: int) -> None:
            bar.update(done - seen["n"])
            seen["n"] = done

        result = engine.infer(
            frames, 24.0, input_size=input_size, precision=precision,
            device=device, on_window=on_window,
            on_warn=lambda m: typer.echo(f"\n{YELLOW}warning:{OFF} {m}"),
        )
    # The analysis-grid depth frames ride along for framing (describe); compare ignores
    # them — its questions are all answered by the profile.
    return rect, motionlib.measure(result.frames, 24.0), motionlib.analysis_gray(result.frames)


@app.command()
def describe(
    clip: Path = typer.Argument(..., help="Any video — typically a generation fired from text alone"),
    resolution: str = typer.Option("720p", "--resolution", help="Depth pass size"),
    input_size: int = typer.Option(518, "--input-size", help="Inference short side, multiple of 14"),
    precision: str = typer.Option("fp32", "--precision", help="fp32 | fp16"),
    rgb: bool = typer.Option(
        False, "--rgb",
        help="Skip the depth pass and describe the picture itself (fast, no model; NOT "
             "comparable with kit/compare figures, which are depth-space)",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Describe a lone clip's motion in prompt words — no kit needed.

    `compare` needs the kit that drove a generation. A shot fired from text alone has none;
    this runs it through the same depth pass and says what moved and which way — camera
    (pans left, pushes in, over time), where the strongest motion sits and travels, and the
    energy shape — so "did it follow the prompt's direction?" becomes: read the prompt,
    read this, do they agree. Uncalibrated, description only; whether the moving thing is
    the one the prompt named is still a by-eye call.
    """
    import json

    from depthc import describe as describelib, engine, motion as motionlib
    from depthc.video_io import VideoIOError, probe as probe_file, read_frames, target_size

    try:
        info = probe_file(clip)
        if not as_json:
            typer.echo(f"{BOLD}{clip.name}{OFF}  {DIM}{info.summary()}{OFF}")
        if rgb:
            frames = read_frames(info, fps=24.0, size=target_size(info, "480p"))
            profile = motionlib.measure(frames, 24.0)
            framing = None  # in RGB, bright is not near — there is no figure to size
        else:
            _, profile, gray = _depth_profile(info, resolution, input_size, precision)
            framing = describelib.read_framing(gray, region_energy=profile.region_energy)
    except (VideoIOError, SetupIncomplete, engine.DepthFailure) as e:
        _die(str(e))
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    description = describelib.describe(profile, framing=framing)
    if as_json:
        payload = description.as_dict()
        payload["space"] = "rgb" if rgb else "depth"
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo("")
    typer.echo(describelib.render(description, clip.name, space="rgb" if rgb else "depth"))


@app.command()
def compare(
    kit_name: str = typer.Argument(..., help="Kit folder name, or a path to one"),
    generated: Path = typer.Argument(..., help="The video that came back from firing that kit"),
    variant: Optional[str] = typer.Option(
        None, "--variant",
        help="Which ref was actually fired (contrast|invert|midgrey|edges); default the base",
    ),
    resolution: str = typer.Option("720p", "--resolution", help="Depth pass size for the generated clip"),
    input_size: int = typer.Option(518, "--input-size", help="Inference short side, multiple of 14"),
    precision: str = typer.Option("fp32", "--precision", help="fp32 | fp16"),
    library: Optional[Path] = typer.Option(None, "--library", help="Kit root (default Outbox/)"),
    regions: bool = typer.Option(
        True, "--regions/--no-regions",
        help="Per-cell (3x3) zero-lag correlation + mirrored control — WHERE the motion sits",
    ),
    camera_path: bool = typer.Option(
        True, "--camera-path/--no-camera-path",
        help="Median-flow camera path (pan/tilt/push) on both sides — WHICH WAY the frame moves",
    ),
    generation: Optional[str] = typer.Option(
        None, "--generation",
        help="OSIDE only: the generation this download IS — its id (UUID v4) from the generation's "
             "row/URL; carried into the `depthc result` line that records the fire",
    ),
) -> None:
    """Measure whether a generation followed its kit — energy ratio first, correlation second.

    Runs the generated video back through the same depth pipeline and compares its motion
    envelope with the reference's, the Gate-4 way, computed identically every time. The
    ratio is judged FIRST (over-following is what destroys pictures and correlation cannot
    see it), the zero-lag correlation second, and the human verdict last: watch the clip,
    then record it with `depthc result`. The REGIONS and PATH lines (placement and
    direction — what the full-frame envelope cannot see) are reported and never vote.
    """
    from depthc import compare as comparelib, engine, kit as kitlib, motion as motionlib
    from depthc.video_io import VideoIOError, probe as probe_file, read_frames, target_size

    # Refused before the depth pass: a 30 s model run keyed to a typo is a wasted run.
    generation_id = None
    if generation is not None:
        try:
            generation_id = kitlib.validate_generation_id(generation)
        except ValueError as e:
            _die(str(e))

    root = kitlib.library_root(library)
    kit_dir = Path(kit_name) if Path(kit_name).is_dir() else root / kit_name
    if variant:
        depth_path = kit_dir / "variants" / f"{variant}.mp4"
        if not depth_path.is_file():
            _die(
                f"No {variant!r} variant in {kit_dir} — compare against the ref that was "
                f"actually fired (this kit has: "
                f"{', '.join(sorted(p.stem for p in (kit_dir / 'variants').glob('*.mp4'))) or 'no variants'})"
            )
    else:
        depth_path = kit_dir / "depth.mp4"
        if not depth_path.is_file():
            _die(f"No kit at {kit_dir} (expected a depth.mp4 inside)")

    # The content half of the fire -> generation join lives in kit.json; a kit built before
    # 2026-09-06 has none. Said once, before the depth pass, and never fatal.
    # ponytail: only the base depth.mp4 is hashed — a fired VARIANT has its own bytes and no
    # recorded hash yet; hash variants/ at build time if a variant fire ever needs the join.
    if kitlib.depth_sha256(kit_dir) is None:
        _warn(
            "kit.json carries no depth.sha256 (kit built before 2026-09-06) — the content "
            "join to OSIDE's refs[].sha256 cannot be made for this kit; "
            "`shasum -a 256 depth.mp4` gives the value by hand."
        )

    # The kit's other half. Reported here, before the depth pass, so the operator sees it
    # with the missing-hash note; the FAIL is carried to the END rather than raised, because
    # a paid fire is already spent and its measurement must never be lost to a text file.
    binding_failed = _check_kit_binding(kit_dir)

    # A processed variant is not measured in the same units as the depth.mp4 it came from:
    # midgrey caps luma at 190, so its motion energy is ~0.74x the base while the generation
    # returns at full range. The manifest already records BOTH energies, measured on the
    # actual files, so the correction is data rather than theory. `edges` is not a scaled
    # version of the envelope at all (binary Canny), so its ratio is withheld instead.
    ref_scale, ratio_note = None, None
    if variant:
        ref_scale, ratio_note = _variant_ratio_basis(kit_dir, variant)

    try:
        # The ref decode is capped: the motion metric lives on a 160 px analysis grid, so
        # decoding a 4K master at native size buys nothing and costs gigabytes.
        # measure() takes RGB directly — no full-array float64 channel-mean intermediate.
        ref_info = probe_file(depth_path)
        ref_size = target_size(ref_info, "480p")
        ref_frames = read_frames(ref_info, fps=24.0, size=ref_size)
        ref_profile = motionlib.measure(ref_frames, 24.0)
        del ref_frames  # the depth pass below needs the room, and the memory check counts on it

        gen_info = probe_file(generated)
        typer.echo(f"{BOLD}{generated.name}{OFF}  {DIM}{gen_info.summary()}{OFF}")
        # The generation gets the same courtesies the kit build gave the reference:
        # matte bars stripped (a letterboxed download otherwise measures its bars as the
        # brightest moving surface), and the decode trimmed to the ref's length so the
        # depth normalisation covers only the stretch actually being judged — an extreme
        # in an excluded tail must not compress the greys of the overlap.
        gen_full_seconds = gen_info.duration
        gen_rect, gen_profile, _ = _depth_profile(
            gen_info, resolution, input_size, precision,
            trim_seconds=ref_info.duration if gen_info.duration > ref_info.duration else None,
        )

        geometry_note = None
        ref_aspect = ref_info.width / ref_info.height
        gen_w, gen_h = (
            (gen_rect.width, gen_rect.height) if gen_rect else (gen_info.width, gen_info.height)
        )
        gen_aspect = gen_w / gen_h
        if abs(gen_aspect - ref_aspect) / ref_aspect > 0.02:
            geometry_note = (
                f"ref framed {ref_info.width}×{ref_info.height} "
                f"({ref_aspect:.2f}:1) vs generation {gen_w}×{gen_h} ({gen_aspect:.2f}:1)"
            )

    except (VideoIOError, SetupIncomplete, engine.DepthFailure) as e:
        _die(str(e))
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    comparison = comparelib.compare_profiles(
        ref_profile, gen_profile,
        gen_full_seconds=gen_full_seconds,
        geometry_note=geometry_note,
        ref_scale=ref_scale,
        ratio_note=ratio_note,
        regions=regions,
        camera_path=camera_path,
    )
    typer.echo("")
    typer.echo(comparelib.render(
        comparison,
        kit_dir.name + (f" [{variant}]" if variant else ""),
        generated.name,
    ))
    typer.echo("")
    typer.echo(f"{DIM}record it:{OFF}")
    typer.echo(_record_hint(kit_name, comparison, generation_id))

    # Measured first, failed after: the numbers are on screen and the exit code still says
    # the kit is broken, so nothing downstream reads this kit as sound.
    if binding_failed:
        raise typer.Exit(1)


def _record_hint(kit_name: str, comparison, generation_id: str | None) -> str:
    """The ready-to-paste `depthc result` line under a compare report.

    `compare` measures and never writes the fire — the verdict is human-only (the plan's
    "no automatic verdicts"; the suggestion is a heuristic) — so the generation id it was
    given rides into result.json through this line, and `result --generation` is what
    writes fires[].generationId.
    """
    ratio = comparison.ratio if comparison.ratio is not None else "n/a"
    r0 = comparison.r0 if comparison.r0 is not None else "n/a"
    # An OSIDE id is the only evidence of the surface here; without one, nothing is assumed.
    generation = f" --surface oside --generation {generation_id}" if generation_id else ""
    return (
        f'{DIM}  ./.tools/uv run depthc result {shlex.quote(str(kit_name))} --verdict {comparison.suggestion} '
        f'--seconds {comparison.overlap_seconds:g}{generation} \\\n'
        f'      --notes "ratio {ratio} r0={r0} — <what you saw>"{OFF}'
    )


@app.command()
def look(
    sources: list[Path] = typer.Argument(..., help="A reference clip, or one or more still images"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output folder (default Outbox/looks/<name>/)"),
    max_frames: int = typer.Option(8, "--max-frames", help="Keyframe cap for a clip"),
    as_json: bool = typer.Option(False, "--json", help="Print look.json to stdout"),
) -> None:
    """Distil a reference's LOOK — palette, per-shot key, warmth, detail — into look.json.

    The still-image sibling of a motion kit: `kit` copies a reference's motion through the
    depth channel, `look` copies its palette and exposure through words. Splits a clip into
    shots at colour cuts and measures the middle frame of each (or measures the stills you
    hand it), writes a contact sheet, the keyframes, and `look.json` carrying a `visualDna`
    block in OSIDE's VisualDnaSelection shape — hexes measured, WORDS left for authoring,
    plus a per-shot key. Uncalibrated, description only. Proven to move two image models
    (LIVEFIRE-2026-08-16-LOOK-DNA-AB) — and to darken a high-key shot when the key is
    ignored, which is why the key rides per shot.
    """
    import json

    import cv2

    from depthc import look as looklib
    from depthc.video_io import VideoIOError, probe as probe_file, read_frames, target_size

    if not sources:
        _die("give a clip or at least one image")
    stills = {".png", ".jpg", ".jpeg", ".webp"}
    dropped = ""
    try:
        if len(sources) == 1 and sources[0].suffix.lower() not in stills:
            info = probe_file(sources[0])
            sampled = read_frames(info, fps=looklib.SAMPLE_FPS, size=target_size(info, "480p"))
            idx = looklib.pick_keyframes(sampled, looklib.SAMPLE_FPS, max_frames=max_frames)
            frames = sampled[idx]
            times = [i / looklib.SAMPLE_FPS for i in idx]
            name, seconds = sources[0].stem, info.duration
            if len(idx) == max_frames:
                dropped = f"keyframes capped at {max_frames} (raise --max-frames to keep more shots)"
        else:
            imgs = []
            for p in sources:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    _die(f"cannot read image {p}")
                imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            frames = imgs
            times = [float(i) for i in range(len(imgs))]
            name, seconds = (sources[0].parent.name if len(sources) > 1 else sources[0].stem), 0.0
    except (VideoIOError, SetupIncomplete) as e:
        _die(str(e))

    result = looklib.build_look(np.array(frames) if not isinstance(frames, list) else frames, times, name, seconds,
                                notes=[dropped] if dropped else None)
    out_dir = out or (Path("Outbox") / "looks" / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "keyframes"
    frames_dir.mkdir(exist_ok=True)
    thumbs = []
    for i, f in enumerate(frames):
        f = np.asarray(f)
        cv2.imwrite(str(frames_dir / f"keyframe_{i:02d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = f.shape[:2]
        tw = 360
        thumbs.append(cv2.resize(f, (tw, max(1, int(h * tw / w)))))
    th = max(t.shape[0] for t in thumbs)
    sheet = np.zeros((th, sum(t.shape[1] for t in thumbs), 3), dtype=np.uint8)
    x = 0
    for t in thumbs:
        sheet[: t.shape[0], x : x + t.shape[1]] = t
        x += t.shape[1]
    cv2.imwrite(str(out_dir / "sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    payload = result.as_dict()
    (out_dir / "look.json").write_text(json.dumps(payload, indent=2) + "\n")
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(looklib.render_look(result))
    if dropped:
        typer.echo(f"  {YELLOW}{dropped}{OFF}")
    typer.echo(f"  → {out_dir}/look.json · sheet.jpg · keyframes/")


@app.command("look-compare")
def look_compare(
    look_json: Optional[Path] = typer.Argument(None, help="A look.json written by `depthc look` (optional with --manifest when the manifest carries a look block)"),
    candidate: Optional[Path] = typer.Argument(None, help="A generated still or clip"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="An OSIDE DaVinci package manifest.json — score every videos/*.mp4 in it (drift gauge)"),
    emit_cdl: bool = typer.Option(False, "--emit-cdl", help="With --manifest: write look-cdl.json beside the manifest — one measured ASC CDL per clip for resolve-mcp apply_look"),
    against: Optional[int] = typer.Option(None, "--against", help="Compare to this keyframe index instead of the global look"),
    fps: float = typer.Option(1.0, "--fps", help="Sample rate for a clip"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Measure a generation against a look — palette distance, ΔL*, Δdark, Δwarmth, per sample.

    A still gives one row. A clip gives one row per sampled second plus a DRIFT line —
    did the look hold from first frame to last? `--manifest` scores a whole OSIDE→DaVinci
    package (every videos/*.mp4) against the look — the look.json, or the manifest's own
    `look` block when no look.json is given — and flags outliers RELATIVE to the package
    (look here first). Distances only; nothing here votes.
    """
    import json

    import cv2

    from depthc import kit as kitlib, look as looklib
    from depthc.video_io import VideoIOError, probe as probe_file, read_frames, target_size

    if manifest is not None:
        # look.json is optional here: `depthc look-compare --manifest m.json` uses m.json's look block;
        # `depthc look-compare look.json --manifest m.json` uses the measured look. A lone
        # positional then IS the look.json (candidate must be absent).
        if candidate is not None:
            _die("--manifest takes at most one positional (the look.json)")
        try:
            mani = json.loads(manifest.read_text())
        except (OSError, ValueError) as e:
            _die(f"cannot read manifest: {e}")
        if not isinstance(mani, dict) or mani.get("format") != "oside-davinci/v1":
            _die(f"{manifest.name} is not an oside-davinci/v1 package")
        base = manifest.parent
        try:
            if look_json is not None:
                ref = looklib.look_from_dict(json.loads(look_json.read_text()))
            elif mani.get("look"):
                ref = looklib.look_from_manifest_block(mani["look"])
            else:
                _die("manifest carries no `look` block and no look.json was given — nothing to compare against")
        except (OSError, KeyError, ValueError) as e:
            _die(f"cannot build the look: {e}")
        videos = mani.get("videos") or []
        if not videos:
            _die("manifest lists no videos")
        scores = []
        for v in videos if isinstance(videos, list) else [None]:
            # One bad entry costs one clip, whatever shape it comes in.
            rel = v.get("file") if isinstance(v, dict) else None
            if not isinstance(rel, str) or not rel:
                scores.append(looklib.ClipScore(str(rel), "?", 0, 0, 0, 0, 0, 0, "", error="no file named"))
                continue
            label = f"{v.get('sceneIndex', '?')}/{v.get('shotNumber', '?')} {Path(rel).name}"
            path = base / rel
            # The manifest may have come from someone else, so it names files in its own package
            # and nowhere else: not an absolute path, a ../ or a link pointing out of it.
            try:
                inside = path.resolve().is_relative_to(base.resolve())
            except (OSError, RuntimeError, ValueError) as e:  # a looping link, a NUL in the name
                scores.append(looklib.ClipScore(rel, label, 0, 0, 0, 0, 0, 0, "", error=str(e).splitlines()[0][:80]))
                continue
            if not inside:
                scores.append(looklib.ClipScore(rel, label, 0, 0, 0, 0, 0, 0, "", error="outside the package"))
                continue
            try:
                info = probe_file(path)
                frames = read_frames(info, fps=fps, size=target_size(info, "480p"))
                times = [i / fps for i in range(len(frames))]
                scores.append(looklib.score_clip(ref, frames, times, rel, label))
            except (VideoIOError, SetupIncomplete, OSError) as e:
                scores.append(looklib.ClipScore(rel, label, 0, 0, 0, 0, 0, 0, "", error=str(e).splitlines()[0][:80]))
        scores = looklib.flag_outliers(scores)
        cdl_path = None
        if emit_cdl:
            cdl_path = base / "look-cdl.json"
            kitlib.write_replacing(cdl_path, json.dumps(looklib.cdl_manifest(ref, scores, str(manifest)), indent=2) + "\n")
        if as_json:
            typer.echo(json.dumps({"look": ref.source, "manifest": str(manifest), "clips": [s.as_dict() for s in scores],
                                   **({"cdl": str(cdl_path)} if cdl_path else {})}, indent=2))
            return
        typer.echo(looklib.render_package(ref, scores))
        if cdl_path:
            n_id = sum(1 for c in looklib.cdl_manifest(ref, scores, str(manifest))["clips"] if c.get("identity"))
            typer.echo(f"  → {cdl_path}  (one heuristic CDL per clip; {n_id} identity)")
        return

    if look_json is None or candidate is None:
        _die("give <look.json> <candidate>, or --manifest <manifest.json>")
    try:
        ref = looklib.look_from_dict(json.loads(look_json.read_text()))
    except (OSError, KeyError, ValueError) as e:
        _die(f"cannot read look: {e}")
    if against is not None and not (0 <= against < len(ref.frames)):
        _die(f"--against out of range (0–{len(ref.frames) - 1})")
    stills = {".png", ".jpg", ".jpeg", ".webp"}
    try:
        if candidate.suffix.lower() in stills:
            im = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if im is None:
                _die(f"cannot read image {candidate}")
            frames, times = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)], [0.0]
        else:
            info = probe_file(candidate)
            frames = read_frames(info, fps=fps, size=target_size(info, "480p"))
            times = [i / fps for i in range(len(frames))]
    except (VideoIOError, SetupIncomplete) as e:
        _die(str(e))
    deltas = looklib.compare_frames(ref, frames, times, against)
    if as_json:
        typer.echo(json.dumps({"look": ref.source, "candidate": candidate.name, "against": against,
                               "rows": [d.as_dict() for d in deltas], "drift": looklib.drift(deltas)}, indent=2))
        return
    typer.echo(looklib.render_compare(ref, deltas, candidate.name, against))


@app.command()
def stability(
    clip: Path = typer.Argument(..., help="A generated take"),
    still: Optional[Path] = typer.Option(
        None, "--still",
        help="The start frame the take was fired from — measures the REPAINT, still → first frame",
    ),
    after: Optional[Path] = typer.Option(
        None, "--after",
        help="The take this one continues — measures the SEAM, its last frame → this first frame",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """How much a take BOILS and FLICKERS once its motion is warped away.

    Flow both ways per frame gap, the next frame warped back, untrackable pixels dropped;
    what still changed is boil (structure re-rolling: hair, foliage, fabric) and flicker
    (exposure breathing). RGB, not depth — this is a reading of the picture. Measured at
    480 px through a band-pass so film grain mostly cancels (add grain in the grade, after).
    For ranking takes of the SAME shot and finding the bad second; not a grade, and not
    calibrated against a human eye.
    """
    import json

    import cv2

    from depthc import stability as stab
    from depthc.video_io import VideoIOError, probe as probe_file, read_frames

    def frames_of(info):
        return read_frames(info, fps=24.0, size=stab.stability_size(info.width, info.height))

    def end_frame(info, first: bool) -> np.ndarray | None:
        # the real first or last frame, at the clip's own rate: a 24 fps conform can drop it
        # (Codex review 2026-10-26), and the join is about exactly that frame. The window
        # widens until it holds one: a 2 fps clip, or a VFR clip whose last frame is long,
        # has no frame in its final quarter second.
        # ponytail: the last widening decodes the whole clip, fine at 480 px for a take; a
        # seek-to-last-keyframe read if long VFR sources ever come through here. A final frame
        # far shorter than one frame interval can round onto the -t cutoff and be cut (Codex
        # 2026-10-26; not reproduced here, H.264, FFV1 and PNG encodes all re-time it): read
        # to EOF instead of -t if a source like that ever turns up
        size = stab.stability_size(info.width, info.height)
        reach = 0.25
        while True:
            span = (0.0, min(info.duration, reach)) if first else (max(0.0, info.duration - reach), info.duration)
            try:
                got = read_frames(info, trim=span, size=size)
            except VideoIOError:
                if reach >= info.duration:
                    raise
                got = ()
            if len(got):
                return got[0] if first else got[-1]
            if reach >= info.duration:
                return None
            reach *= 4

    def grain_sample(info) -> np.ndarray | None:
        """2 frames a second, or 2 frames from a clip too short for that rate to land on any."""
        try:
            return read_frames(info, fps=max(2.0, 2.0 / max(info.duration, 1 / 30)))
        except VideoIOError:
            return None  # the clip was measured: its grain reading is absent, never guessed

    def join(previous: np.ndarray, current: np.ndarray, name: str) -> dict:
        g = stab.gap(to_size(previous, current), current)
        # below the clip summary's tracking floor the gap's numbers are sentinels, not readings
        if g.valid < stab.MIN_VALID_SHARE:
            return {"from": name, "tracked": round(g.valid, 3), "scored": False}
        return {"from": name, "boil": round(g.boil, 3), "exposure": round(g.flicker, 3),
                "tracked": round(g.valid, 3), "scored": True}

    def to_size(picture: np.ndarray, like: np.ndarray) -> np.ndarray:
        h, w = like.shape[:2]
        return picture if picture.shape[:2] == (h, w) else cv2.resize(picture, (w, h), interpolation=cv2.INTER_AREA)

    try:
        info = probe_file(clip)
        frames = frames_of(info)
        if frames.shape[0] < 2:
            _die(f"{clip.name}: fewer than two frames — nothing to compare")
        # grain is read on a few frames at the take's own resolution: the 480 px grid averages it away
        result = stab.measure(frames, 24.0, grain_frames=grain_sample(info))
        joins: dict[str, dict] = {}
        opening = end_frame(info, first=True) if still is not None or after is not None else None
        if (still is not None or after is not None) and opening is None:
            _die(f"{clip.name}: could not read its first frame")
        if still is not None:
            bgr = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if bgr is None:
                _die(f"cannot read the still {still}")
            joins["repaint"] = join(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), opening, still.name)
        if after is not None:
            closing = end_frame(probe_file(after), first=False)
            if closing is None:
                _die(f"{after.name}: could not read its last frame")
            joins["seam"] = join(closing, opening, after.name)
    except VideoIOError as e:
        _die(str(e))
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    summary = result.summary()
    if as_json:
        typer.echo(json.dumps({"clip": clip.name, **summary, **joins}, indent=2))
        return
    typer.echo(f"{BOLD}{clip.name}{OFF}  {DIM}{info.summary()}{OFF}")
    if not summary.get("scored"):
        _die("no gap could be tracked — a slideshow of cuts, or a flash throughout")
    worst = ", ".join(f"{w['t']:g}s ({w['boil']:.2f})" for w in summary["worstBoil"])
    typer.echo(f"  boil     median {summary['boilMedian']:.2f}  p95 {summary['boilP95']:.2f}   worst at {worst}")
    typer.echo(f"  flicker  std {summary['flickerStd']:.2f}  max {summary['flickerMaxAbs']:.2f} grey levels")
    if "grainMedian" in summary:
        typer.echo(f"  grain    σ {summary['grainMedian']:.2f} grey levels at full resolution (flat half of the frame)")
    untracked = summary["gaps"] - summary["scored"]
    typer.echo(f"  tracked  {summary['validMedian']:.0%} of pixels (median gap)"
               + (f"  {YELLOW}{untracked} gap(s) lost the flow, not scored{OFF}" if untracked else ""))
    for name, label in (("repaint", "repaint  still → first frame"), ("seam", "seam     prior last → first")):
        if name in joins:
            j = joins[name]
            if not j["scored"]:
                typer.echo(f"  {label}: {YELLOW}not scored, only {j['tracked']:.0%} of pixels tracked{OFF}")
                continue
            typer.echo(f"  {label}: boil {j['boil']:.2f}, exposure {j['exposure']:+.2f}, tracked {j['tracked']:.0%}")
    typer.echo(f"{DIM}Grey levels at 480 px. Rank takes of the same shot with it; a cut reads as the worst spike."
               f" Uncalibrated against a human eye.{OFF}")


@app.command()
def ui(
    port: int = typer.Option(7861, "--port", help="Port on 127.0.0.1"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a tab"),
) -> None:
    """Launch the local web UI."""
    from depthc.ui import launch

    launch(port=port, open_browser=not no_browser)


@library_app.command("list")
def library_list_cmd(
    tag: Optional[str] = typer.Option(None, "--tag", help="Only kits carrying this tag"),
    library: Optional[Path] = typer.Option(None, "--library", help="Kit output root (default Outbox/)"),
) -> None:
    """List kits in the Motion Library."""
    from depthc import kit as kitlib

    root = kitlib.library_root(library)
    entries = kitlib.library_list(root, tag=tag)
    if not entries:
        where = f" tagged {tag!r}" if tag else ""
        typer.echo(f"No kits{where} in {root}")
        return

    typer.echo(
        f"{BOLD}{'KIT':<34} {'DUR':>6} {'MOTION':>7}  {'TIER':<5} {'FIRED':<9} TAGS{OFF}"
        f"  {DIM}({root}){OFF}"
    )
    for entry in entries:
        tint = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}.get(entry.tier, "")
        energy = f"{entry.energy:.2f}" if entry.energy is not None else "—"
        verdict = entry.verdict
        vtint = {"adhered": GREEN, "partial": YELLOW}.get(
            verdict, DIM if verdict == "untested" else RED
        )
        typer.echo(
            f"{entry.name:<34} {entry.duration:>5.2f}s {energy:>7}  "
            f"{tint}{entry.tier:<5}{OFF} {vtint}{verdict:<9}{OFF} "
            f"{', '.join(entry.tags) or DIM + '—' + OFF}"
        )
    typer.echo(f"{DIM}{len(entries)} kit{'s' if len(entries) != 1 else ''}{OFF}")


@library_app.command("index")
def library_index_cmd(
    library: Optional[Path] = typer.Option(None, "--library", help="Kit output root (default Outbox/)"),
) -> None:
    """Rewrite Outbox/INDEX.txt from the kit.json files on disk.

    A build already does this. This is the same rewrite with no conversion attached, for
    the one case a build does not cover: a kit folder deleted by hand leaves INDEX.txt
    listing a kit that is gone, and until 2026-09-15 the only repair was to build another.
    """
    from depthc import kit as kitlib

    root = kitlib.library_root(library)
    path = kitlib.write_outbox_index(root)
    entries = kitlib.library_list(root)
    typer.echo(f"{path} — {len(entries)} kit{'s' if len(entries) != 1 else ''}")


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    app()
