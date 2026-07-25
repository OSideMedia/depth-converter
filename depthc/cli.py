"""`depthc` command line. Every flag here has a matching control in the Gradio UI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import typer

from depthc import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Convert a reference clip into a Seedance-ready depth motion kit.",
)
library_app = typer.Typer(no_args_is_help=True, help="Browse the Motion Library.")
app.add_typer(library_app, name="library")

BOLD, DIM, GREEN, YELLOW, RED, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m",
)


def _die(message: str) -> None:
    typer.echo(f"{RED}error:{OFF} {message}", err=True)
    raise typer.Exit(1)


def _warn(message: str) -> None:
    typer.echo(f"{YELLOW}warning:{OFF} {message}")


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
    quiet: bool = False,
) -> Conversion:
    """Probe → validate → decode → infer. Shared by `convert` and `kit`."""
    from depthc import engine
    from depthc.video_io import probe, read_frames, resolve_fps, target_size

    info = probe(input)
    if not quiet:
        typer.echo(f"{BOLD}{info.path.name}{OFF}  {DIM}{info.summary()}{OFF}")

    span = (trim[1] - trim[0]) if trim else info.duration
    if span > 15 and not allow_long:
        _die(
            f"Trimmed length is {span:.1f}s. Seedance caps video references at 15s and "
            f"adherence degrades past ~8s.\n"
            f"  Trim to the one clear motion signal, or pass --allow-long to override."
        )
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
    except VideoIOError as e:
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

    try:
        job = _convert(
            input, trim=trim, resolution=resolution, fps=fps, crop=crop, invert=invert,
            input_size=input_size, precision=precision, allow_long=allow_long,
        )

        out_path = Path(out) if out else input.with_name(f"{input.stem}_depth.mp4")
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

    except VideoIOError as e:
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
        None, "--library", help="Motion Library root (default $DEPTHC_LIBRARY or ~/Depth_Motion_Library)"
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
) -> None:
    """Probe, convert, check, and package a full Seedance Kit into the Motion Library."""
    from depthc import checks, engine, kit as kitlib
    from depthc.video_io import (
        VideoIOError, encode_gray, mux_audio, probe as probe_file, side_by_side, trim_copy,
    )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        job = _convert(
            input, trim=trim, resolution=resolution, fps=fps, crop=crop, invert=invert,
            input_size=input_size, precision=precision, allow_long=allow_long,
        )

        root = kitlib.library_root(library)
        kit_dir = kitlib.allocate_kit_dir(root, tag_list, fallback=input.stem)
        paths = kitlib.KitPaths(kit_dir)

        encode_gray(job.result.frames, paths.depth, job.result.fps)
        if keep_audio and job.info.has_audio:
            tmp = paths.depth.with_name("depth_audio.mp4")
            mux_audio(paths.depth, job.info.path, tmp, trim=trim)
            tmp.replace(paths.depth)
        elif keep_audio:
            _warn("--keep-audio: source has no audio track.")

        # Post-encode size guard (PLAN.md §1 encode spec).
        if paths.depth.stat().st_size > checks.MAX_FILE_MB * 1_000_000:
            _warn(f"depth.mp4 exceeded {checks.MAX_FILE_MB:g} MB — re-encoding at crf 20.")
            encode_gray(job.result.frames, paths.depth, job.result.fps, crf=20)

        still_index = None if still == "auto" else (0 if still == "first" else int(still))
        chosen = kitlib.write_still(job.frames, paths.still, index=still_index)
        trim_copy(job.info, paths.source, trim=trim)
        if preview:
            side_by_side(job.frames, job.result.frames, paths.preview, job.result.fps)

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

        kitlib.write_manifest(paths, kitlib.build_manifest(
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
                "keepAudio": keep_audio,
            },
            report=report,
            dialect=dialect,
            device=job.result.device,
            elapsed=job.result.elapsed,
        ))

        if zip_output:
            kitlib.zip_kit(kit_dir)

    except (VideoIOError, FileNotFoundError) as e:
        _die(str(e))
    except engine.NonFiniteDepth as e:
        _die(str(e))
    except KeyboardInterrupt:
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


@app.command()
def inbox(
    path: Optional[Path] = typer.Option(None, "--inbox", help="Inbox folder to scan"),
) -> None:
    """Triage the clips waiting in the inbox.

    Reports what each one is and what it needs, so the trim call can be made before any
    conversion is run. Converts nothing — deciding which 3-8s beat carries the motion is a
    judgement, not a default.
    """
    from depthc import kit as kitlib
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
        elif info.duration > 8:
            typer.echo(
                f"    {YELLOW}needs trimming{OFF} — {info.duration:.1f}s is past the 3-8s "
                f"sweet spot. Pick the single clearest motion beat."
            )
        else:
            typer.echo(f"    {GREEN}usable whole{OFF} — inside the 3-8s sweet spot.")
        typer.echo()


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
    library: Optional[Path] = typer.Option(None, "--library", help="Motion Library root"),
) -> None:
    """List kits in the Motion Library."""
    from depthc import kit as kitlib

    root = kitlib.library_root(library)
    entries = kitlib.library_list(root, tag=tag)
    if not entries:
        where = f" tagged {tag!r}" if tag else ""
        typer.echo(f"No kits{where} in {root}")
        return

    typer.echo(f"{BOLD}{'KIT':<34} {'DUR':>6}  {'TIER':<5} TAGS{OFF}  {DIM}({root}){OFF}")
    for entry in entries:
        tint = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}.get(entry.tier, "")
        typer.echo(
            f"{entry.name:<34} {entry.duration:>5.2f}s  "
            f"{tint}{entry.tier:<5}{OFF} {', '.join(entry.tags) or DIM + '—' + OFF}"
        )
    typer.echo(f"{DIM}{len(entries)} kit{'s' if len(entries) != 1 else ''}{OFF}")


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    app()
