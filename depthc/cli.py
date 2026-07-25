"""`depthc` command line. Every flag here has a matching control in the Gradio UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import typer

from depthc import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Convert a reference clip into a Seedance-ready depth motion kit.",
)

BOLD, DIM, GREEN, YELLOW, RED, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m",
)


def _die(message: str) -> None:
    typer.echo(f"{RED}error:{OFF} {message}", err=True)
    raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the tool version."""
    typer.echo(f"depthc {__version__}")


@app.command()
def probe(input: Path = typer.Argument(..., help="Video file to inspect")) -> None:
    """Report duration, fps, resolution, and rotation for a clip."""
    from depthc.video_io import VideoIOError, probe as probe_file

    try:
        info = probe_file(input)
    except VideoIOError as e:
        _die(str(e))

    typer.echo(f"{BOLD}{info.path.name}{OFF}")
    typer.echo(f"  {info.summary()}")
    typer.echo(f"  audio: {'yes' if info.has_audio else 'none'}")
    typer.echo(f"  pixels: {info.pixels:,}")


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
    from depthc.video_io import (
        Crop,
        VideoIOError,
        detect_crop,
        encode_gray,
        mux_audio,
        probe as probe_file,
        read_frames,
        resolve_fps,
        side_by_side,
        target_size,
    )

    try:
        info = probe_file(input)
        typer.echo(f"{BOLD}{info.path.name}{OFF}  {DIM}{info.summary()}{OFF}")

        span = (trim[1] - trim[0]) if trim else info.duration
        if span > 15 and not allow_long:
            _die(
                f"Trimmed length is {span:.1f}s. Seedance caps video references at 15s and "
                f"adherence degrades past ~8s.\n"
                f"  Trim to the one clear motion signal, or pass --allow-long to override."
            )
        if span > 8:
            typer.echo(
                f"{YELLOW}warning:{OFF} {span:.1f}s exceeds the 3-8s sweet spot — "
                f"adherence degrades. Consider trimming to a single motion signal."
            )

        if crop == "auto":
            crop_rect = detect_crop(info, trim=trim)
            if crop_rect:
                typer.echo(
                    f"{DIM}matte detected — cropping to {crop_rect.width}×{crop_rect.height} "
                    f"({crop_rect.removed_fraction(info) * 100:.0f}% of frame removed; "
                    f"--crop none to keep it){OFF}"
                )
        elif crop == "none":
            crop_rect = None
        else:
            crop_rect = Crop.parse(crop)

        size = target_size(info, resolution, crop=crop_rect)
        out_fps = resolve_fps(info, fps)
        typer.echo(f"{DIM}decoding at {size[0]}×{size[1]} @ {out_fps:g} fps…{OFF}")
        frames = read_frames(info, trim=trim, fps=out_fps, size=size, crop=crop_rect)
        typer.echo(f"{DIM}{frames.shape[0]} frames{OFF}")

        device = engine.pick_device()
        if device == "cpu":
            typer.echo(
                f"{YELLOW}warning:{OFF} MPS unavailable — running on CPU. This will be slow."
            )

        total_windows = engine.expected_windows(frames.shape[0])
        with typer.progressbar(length=total_windows, label="depth") as bar:
            last = {"n": 0}

            def on_window(done: int, total: int) -> None:
                bar.update(done - last["n"])
                last["n"] = done

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

        out_path = Path(out) if out else input.with_name(f"{input.stem}_depth.mp4")
        encode_gray(result.frames, out_path, result.fps)

        if keep_audio:
            if not info.has_audio:
                typer.echo(f"{YELLOW}warning:{OFF} --keep-audio: source has no audio track.")
            else:
                tmp = out_path.with_name(f"{out_path.stem}_audio{out_path.suffix}")
                mux_audio(out_path, info.path, tmp, trim=trim)
                tmp.replace(out_path)

        if preview:
            preview_path = out_path.with_name(f"{out_path.stem}_preview.mp4")
            side_by_side(frames, result.frames, preview_path, result.fps)
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
        f"{GREEN}✓{OFF} {out_path}  {DIM}{mb:.1f} MB · {result.count} frames · "
        f"{result.device}/{result.dtype} · {result.elapsed:.1f}s "
        f"({result.ms_per_frame:.0f} ms/frame){OFF}"
    )


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    app()
