"""`depthc` command line. Every flag here has a matching control in the Gradio UI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import typer

from depthc import __version__
# Deliberately imported eagerly (it is torch-free): SetupIncomplete is the most likely
# FIRST-run failure, and every command that loads the model must turn it into its
# hand-written "Run: bash scripts/setup.sh" message rather than a traceback.
from depthc.vendor_path import SetupIncomplete

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
) -> Conversion:
    """Probe → validate → decode → infer. Shared by `convert` and `kit`."""
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
    if span > checks.MAX_REF_SECONDS and not allow_long:
        _die(
            f"Trimmed length is {span:.1f}s. Seedance caps video references at 15s and "
            f"adherence degrades past ~8s.\n"
            f"  Trim to the one clear motion signal, or pass --allow-long to override."
        )

    # Snap before anything is decoded: a ref shorter than the generation it drives leaves the
    # model to invent the tail, and the fix is free at this point. An --allow-long overlength
    # span is left alone — no output length can match it anyway.
    if snap and span <= checks.MAX_REF_SECONDS:
        window = trim if trim else (0.0, info.duration)
        snapped, note = checks.snap_trim(window[0], window[1], info.duration)
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
    dialect: str = typer.Option("seedance-2.0", "--dialect", help="Prompt template to render"),
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
        help="Also write contrast-stretched, inverted, moderation-capped (midgrey) and "
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
        if paths.depth.stat().st_size > checks.MAX_FILE_MB * 1_000_000:
            _warn(f"depth.mp4 exceeded {checks.MAX_FILE_MB:g} MB — re-encoding at crf 20.")
            encode_gray(job.result.frames, paths.depth, job.result.fps, crf=20)

        audio_kept = False
        if keep_audio and job.info.has_audio:
            tmp = paths.depth.with_name("depth_audio.mp4")
            mux_audio(paths.depth, job.info.path, tmp, trim=trim)
            if tmp.stat().st_size > checks.MAX_FILE_MB * 1_000_000:
                tmp.unlink()
                _warn(
                    f"--keep-audio: the audio track pushed depth.mp4 over "
                    f"{checks.MAX_FILE_MB:g} MB — kept the silent version instead."
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
                if variant_path.stat().st_size > checks.MAX_FILE_MB * 1_000_000:
                    _warn(f"variants/{filename} exceeded {checks.MAX_FILE_MB:g} MB — re-encoding at crf 20.")
                    encode_gray(frames, variant_path, job.result.fps, crf=20)
                variant_blocks.append({
                    "name": name,
                    "filename": filename,
                    "description": description,
                    "energy": motionlib.measure(frames, job.result.fps, with_flow=False).energy,
                })

        depth_bytes = paths.depth.stat().st_size
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
    except engine.NonFiniteDepth as e:
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
        f"{indent}{DIM}./.tools/uv run depthc kit \"{clip}\" "
        f"--trim {best.start:g} {best.end:g} --tags \"...\" --desc \"...\" "
        f"--preview --archive{OFF}"
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
    from depthc import motion as motionlib
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
    except (VideoIOError, SetupIncomplete) as e:
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
    from depthc import kit as kitlib, motion as motionlib
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

        matte = detect_crop(info)
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
            except (VideoIOError, ValueError, SetupIncomplete) as e:
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

    try:
        log = kitlib.record_result(
            kit_dir, verdict=verdict, surface=surface, model=model,
            output_seconds=seconds, cost_usd=cost, output_ref=output,
            variant=variant, notes=notes,
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
    from depthc.video_io import (
        VideoIOError, detect_crop, probe as probe_file, read_frames, target_size,
    )

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

    try:
        # The ref decode is capped: the motion metric lives on a 160 px analysis grid, so
        # decoding a 4K master at native size buys nothing and costs gigabytes.
        # measure() takes RGB directly — no full-array float64 channel-mean intermediate.
        ref_info = probe_file(depth_path)
        ref_size = target_size(ref_info, "480p")
        ref_frames = read_frames(ref_info, fps=24.0, size=ref_size)
        ref_profile = motionlib.measure(ref_frames, 24.0)

        gen_info = probe_file(generated)
        typer.echo(f"{BOLD}{generated.name}{OFF}  {DIM}{gen_info.summary()}{OFF}")
        # The generation gets the same courtesies the kit build gave the reference:
        # matte bars stripped (a letterboxed download otherwise measures its bars as the
        # brightest moving surface), and the decode trimmed to the ref's length so the
        # depth normalisation covers only the stretch actually being judged — an extreme
        # in an excluded tail must not compress the greys of the overlap.
        gen_rect = detect_crop(gen_info)
        if gen_rect:
            typer.echo(f"{DIM}matte on the generation: cropping to {gen_rect.width}×{gen_rect.height}{OFF}")
        gen_full_seconds = gen_info.duration
        gen_trim = (0.0, ref_info.duration) if gen_info.duration > ref_info.duration else None
        size = target_size(gen_info, resolution, crop=gen_rect)
        frames = read_frames(gen_info, trim=gen_trim, fps=24.0, size=size, crop=gen_rect)

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
        gen_profile = motionlib.measure(result.frames, 24.0)
    except (VideoIOError, SetupIncomplete) as e:
        _die(str(e))
    except engine.NonFiniteDepth as e:
        _die(str(e))
    except KeyboardInterrupt:
        typer.echo("\ncancelled", err=True)
        raise typer.Exit(130)

    comparison = comparelib.compare_profiles(
        ref_profile, gen_profile,
        gen_full_seconds=gen_full_seconds,
        geometry_note=geometry_note,
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
    typer.echo(
        f'{DIM}  ./.tools/uv run depthc result "{kit_name}" --verdict {comparison.suggestion} '
        f'--surface oside --seconds {comparison.overlap_seconds:g} \\\n'
        f'      --notes "ratio {comparison.ratio if comparison.ratio is not None else "n/a"}'
        f' r0={comparison.r0 if comparison.r0 is not None else "n/a"} — <what you saw>"{OFF}'
    )


@app.command()
def ui(
    port: int = typer.Option(7861, "--port", help="Chosen to avoid OSIDE's dev server on 3000"),
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


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    app()
