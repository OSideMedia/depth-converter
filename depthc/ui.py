"""Local Gradio UI — `depthc ui`.

Dark, near-black, minimal chrome, deliberately not the default Gradio palette.

The conversion path here calls the same `video_io` + `engine` + `kit` functions the CLI
does; nothing about the pipeline is reimplemented for the browser. The UI's only real jobs
are collecting settings, surfacing the constraint report, and never failing silently.
"""

from __future__ import annotations

import subprocess
import traceback
from pathlib import Path

import gradio as gr

from depthc import __version__, checks, engine
from depthc import kit as kitlib
from depthc.checks import Tier
from depthc.video_io import (
    VideoIOError,
    Crop,
    detect_crop,
    encode_gray,
    probe,
    read_frames,
    resolve_fps,
    side_by_side,
    target_size,
    trim_copy,
)

DEFAULT_PORT = 7861  # deliberately not 3000 — OSIDE's dev server lives there

CSS = """
:root, .gradio-container {
  --depthc-bg: #0a0a0b;
  --depthc-panel: #131316;
  --depthc-line: #26262b;
  --depthc-text: #e8e8ea;
  --depthc-dim: #8b8b94;
}
.gradio-container { background: var(--depthc-bg) !important; max-width: 1500px !important; }
.depthc-head { display:flex; align-items:baseline; gap:.75rem; padding:.25rem 0 1rem; }
.depthc-head h1 { font-size:1.05rem; font-weight:600; letter-spacing:.02em;
  color:var(--depthc-text); margin:0; }
.depthc-head span { font-size:.75rem; color:var(--depthc-dim); }
.depthc-card { background:var(--depthc-panel) !important; border:1px solid var(--depthc-line)
  !important; border-radius:10px !important; padding:1rem !important; }
.depthc-note { font-size:.8rem; color:var(--depthc-dim); line-height:1.5; }
.depthc-report { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8rem;
  line-height:1.65; }
.depthc-report .pass { color:#4ade80; } .depthc-report .warn { color:#fbbf24; }
.depthc-report .fail { color:#f87171; }
.depthc-report .remedy { color:var(--depthc-dim); padding-left:1.4rem; }
.depthc-tier { display:inline-block; padding:.15rem .6rem; border-radius:5px;
  font-weight:600; font-size:.75rem; letter-spacing:.06em; }
.depthc-tier.PASS { background:#052e16; color:#4ade80; }
.depthc-tier.WARN { background:#3a2c05; color:#fbbf24; }
.depthc-tier.FAIL { background:#3f0d0d; color:#f87171; }

/* Markdown and HTML blocks ship a light fill of their own; on a near-black page that
   renders as a white slab. An EMPTY html block (the trim readout before a clip is
   loaded) is the worst case — a bare white bar with nothing in it. */
.gradio-container .prose, .gradio-container .md,
.gradio-container .html-container, .gradio-container .markdown {
  background: transparent !important;
  color: var(--depthc-text) !important;
  border: none !important;
}
.gradio-container .prose strong { color: var(--depthc-text) !important; }
.gradio-container .html-container:empty { display: none !important; padding: 0 !important; }
.gradio-container .block:has(> .html-container:empty) { display: none !important; }

/* Video players default to a white backdrop, which is a bright slab on this page both
   before the first frame paints and in the letterboxed area around a 2.39:1 depth clip.
   Black is also the honest backdrop for judging a grayscale depth map. */
.gradio-container video,
.gradio-container .video-container,
.gradio-container [data-testid="video"],
.gradio-container .wrap:has(> video) {
  background: #000 !important;
}

footer { display:none !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc", c400="#818cf8",
        c500="#6366f1", c600="#4f46e5", c700="#4338ca", c800="#3730a3", c900="#312e81",
        c950="#1e1b4b",
    ),
    neutral_hue="zinc",
    font=("ui-sans-serif", "-apple-system", "system-ui", "sans-serif"),
).set(
    body_background_fill="#0a0a0b",
    block_background_fill="#131316",
    block_border_color="#26262b",
    block_label_background_fill="#131316",
    block_label_text_color="#8b8b94",
    block_title_text_color="#8b8b94",
    input_background_fill="#0f0f11",
    input_border_color="#26262b",
    body_text_color="#e8e8ea",
    body_text_color_subdued="#8b8b94",
    border_color_primary="#26262b",
    panel_background_fill="#131316",
    # Gradio's secondary button and markdown blocks default to a light fill, which reads
    # as a white slab on a near-black page. Both have to be set explicitly.
    button_secondary_background_fill="#1e1e22",
    button_secondary_background_fill_hover="#2a2a30",
    button_secondary_text_color="#e8e8ea",
    button_secondary_border_color="#33333a",
    table_even_background_fill="#131316",
    table_odd_background_fill="#0f0f11",
    table_border_color="#26262b",
    table_text_color="#e8e8ea",
)


# ------------------------------------------------------------------------------ helpers

def _reveal(path: str | Path) -> str:
    """Reveal in Finder. Local-only tool, so shelling out to `open -R` is fair game."""
    target = Path(path)
    if not target.exists():
        return f"Not found: {target}"
    subprocess.run(["open", "-R", str(target)], check=False)
    return f"Revealed {target.name}"


def _report_html(report: checks.Report) -> str:
    marks = {Tier.PASS: "✓", Tier.WARN: "!", Tier.FAIL: "✗"}
    rows = [f'<div class="depthc-tier {report.tier.value}">{report.tier.value}</div>']
    for finding in report.findings:
        css = finding.tier.value.lower()
        rows.append(
            f'<div class="{css}">{marks[finding.tier]} <b>{finding.check}</b>: '
            f"{finding.detail}</div>"
        )
        if finding.remedy and finding.tier is not Tier.PASS:
            rows.append(f'<div class="remedy">→ {finding.remedy}</div>')
    return f'<div class="depthc-report">{"".join(rows)}</div>'


def _probe_readout(path: str | None) -> tuple[str, dict, dict]:
    """Inspect an upload: facts, matte, and the trim bounds the sliders should take."""
    if not path:
        return "", gr.update(maximum=15, value=0), gr.update(maximum=15, value=15)
    try:
        info = probe(path)
    except VideoIOError as e:
        return f'<div class="depthc-note" style="color:#f87171">{e}</div>', \
            gr.update(), gr.update()

    matte = detect_crop(info)
    lines = [f"<b>{info.path.name}</b>", info.summary()]
    if matte:
        lines.append(
            f"matte detected → cropping to {matte.width}×{matte.height} "
            f"({matte.removed_fraction(info) * 100:.0f}% removed)"
        )

    # Long clips get an inline warning and a concrete suggestion, never a silent failure.
    if info.duration > 15:
        lines.append(
            f'<span style="color:#f87171">{info.duration:.1f}s exceeds Seedance\'s 15s '
            f"ceiling. Trim below — 3–8s follows best. Suggested: 0 → 6.</span>"
        )
    elif info.duration > 8:
        lines.append(
            f'<span style="color:#fbbf24">{info.duration:.1f}s is past the 3–8s sweet '
            f"spot; adherence degrades. Consider trimming to one motion signal.</span>"
        )

    end_default = min(info.duration, 6.0) if info.duration > 15 else info.duration
    return (
        '<div class="depthc-note">' + "<br>".join(lines) + "</div>",
        gr.update(maximum=round(info.duration, 2), value=0.0),
        gr.update(maximum=round(info.duration, 2), value=round(end_default, 2)),
    )


def _trim_feedback(start: float, end: float) -> str:
    span = max(0.0, end - start)
    if span <= 0:
        tone, note = "#f87171", "empty range"
    elif span < 2:
        tone, note = "#f87171", "under the 2s minimum — Seedance will reject this"
    elif span < 3:
        tone, note = "#fbbf24", "shorter than the 3–8s sweet spot; fine for a single beat"
    elif span <= 8:
        tone, note = "#4ade80", "inside the 3–8s sweet spot"
    elif span <= 15:
        tone, note = "#fbbf24", "past the sweet spot — adherence degrades"
    else:
        tone, note = "#f87171", "over the 15s ceiling"
    return (
        f'<div class="depthc-note"><b style="color:{tone}">{span:.2f}s</b> — {note}</div>'
    )


# --------------------------------------------------------------------------- conversion

def _convert(
    video, tags, description, resolution, crop_mode, fps_mode, start, end,
    invert, faces, still_mode, input_size, precision, keep_audio, make_preview,
    library_path, progress=gr.Progress(),
):
    """Build a kit. Returns (depth, preview, report_html, prompt, kit_dir, status)."""
    blank = (None, None, "", "", "", "")
    if not video:
        return (*blank[:5], "Drop a clip in first.")

    try:
        info = probe(video)

        # Clamp to the clip, then decide. Testing `end > start` before clamping let an
        # in-point equal to the out-point fall through as "no trim" and quietly convert the
        # WHOLE clip — the user asks for nothing and is charged a full conversion.
        start = max(0.0, float(start))
        end = min(float(end), info.duration)
        span = end - start

        if span < 0.1:
            return (
                *blank[:5],
                f"Empty trim range: in {start:.2f}s is not before out {end:.2f}s.",
            )

        trim = None if (start <= 0 and end >= info.duration - 0.01) else (start, end)
        if span > 15:
            return (
                *blank[:5],
                f"{span:.1f}s exceeds Seedance's 15s ceiling for video references. "
                f"Trim to a single motion signal — 3–8s follows best.",
            )

        progress(0.02, desc="probing")
        crop_rect = (
            None if crop_mode == "none"
            else detect_crop(info, trim=trim) if crop_mode == "auto"
            else Crop.parse(crop_mode)
        )
        size = target_size(info, resolution, crop=crop_rect)
        out_fps = resolve_fps(info, fps_mode)

        progress(0.05, desc="decoding")
        frames = read_frames(info, trim=trim, fps=out_fps, size=size, crop=crop_rect)

        total = engine.expected_windows(frames.shape[0])

        def on_window(done: int, _t: int) -> None:
            progress(0.1 + 0.75 * (done / max(1, total)), desc=f"depth {done}/{total}")

        warnings: list[str] = []
        result = engine.infer(
            frames, out_fps, input_size=int(input_size), precision=precision,
            invert=invert, on_window=on_window, on_warn=warnings.append,
        )

        progress(0.9, desc="packaging")
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        root = kitlib.library_root(library_path or None)
        kit_dir = kitlib.allocate_kit_dir(root, tag_list, fallback=Path(video).stem)
        paths = kitlib.KitPaths(kit_dir)

        encode_gray(result.frames, paths.depth, result.fps)
        if keep_audio and info.has_audio:
            tmp = paths.depth.with_name("depth_audio.mp4")
            from depthc.video_io import mux_audio

            mux_audio(paths.depth, info.path, tmp, trim=trim)
            tmp.replace(paths.depth)
        if paths.depth.stat().st_size > checks.MAX_FILE_MB * 1_000_000:
            warnings.append(f"depth.mp4 over {checks.MAX_FILE_MB:g} MB — re-encoded at crf 20.")
            encode_gray(result.frames, paths.depth, result.fps, crf=20)

        still_index = None if still_mode == "auto" else 0
        chosen = kitlib.write_still(frames, paths.still, index=still_index)
        trim_copy(info, paths.source, trim=trim)
        if make_preview:
            side_by_side(frames, result.frames, paths.preview, result.fps)

        depth_info = probe(paths.depth)
        report = checks.check(
            duration=depth_info.duration, width=depth_info.width, height=depth_info.height,
            fps=depth_info.fps, size_bytes=paths.depth.stat().st_size,
            still_has_real_faces=faces,
        )
        prompt_text = kitlib.render_prompt(
            kit_name=kit_dir.name, duration=depth_info.duration, width=depth_info.width,
            height=depth_info.height, fps=depth_info.fps, still_has_real_faces=faces,
        )
        paths.prompt.write_text(prompt_text)

        kitlib.write_manifest(paths, kitlib.build_manifest(
            kit_name=kit_dir.name, source_path=info.path,
            source_sha256=kitlib.sha256_file(info.path), source_duration=info.duration,
            source_fps=info.fps, source_size=(info.width, info.height), trim=trim,
            crop=str(crop_rect) if crop_rect else None,
            depth_duration=depth_info.duration, depth_fps=depth_info.fps,
            depth_size=(depth_info.width, depth_info.height),
            depth_bytes=paths.depth.stat().st_size, frame_count=result.count,
            still_index=chosen, still_has_real_faces=faces, tags=tag_list,
            description=description or "",
            settings={
                "resolution": resolution, "fps": fps_mode, "invert": invert,
                "inputSize": int(input_size), "precision": result.dtype,
                "crop": crop_mode, "keepAudio": keep_audio,
            },
            report=report, device=result.device, elapsed=result.elapsed,
        ))
        kitlib.write_outbox_index(root)

        progress(1.0, desc="done")
        status = (
            f"{kit_dir.name} — {result.count} frames, {result.elapsed:.1f}s "
            f"({result.device}/{result.dtype})"
        )
        if warnings:
            status += " · " + " · ".join(warnings)

        return (
            str(paths.depth),
            str(paths.preview) if make_preview and paths.preview.exists() else None,
            _report_html(report),
            prompt_text,
            str(kit_dir),
            status,
        )

    except (VideoIOError, FileNotFoundError, ValueError) as e:
        return (*blank[:5], f"{e}")
    except engine.NonFiniteDepth as e:
        return (*blank[:5], f"{e}")
    except Exception:  # noqa: BLE001 - a UI must never die silently on an unknown fault
        return (*blank[:5], f"Unexpected error:\n{traceback.format_exc(limit=3)}")


# ------------------------------------------------------------------------------ library

def _library_rows(tag: str, library_path: str) -> list[list]:
    entries = kitlib.library_list(kitlib.library_root(library_path or None), tag=tag or None)
    return [
        [e.name, f"{e.duration:.2f}s", e.tier, ", ".join(e.tags), e.created[:10], str(e.path)]
        for e in entries
    ]


def _library_pick(rows, evt: gr.SelectData) -> tuple:
    """Load the clicked kit. The path lives in the last column of the table."""
    try:
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        row = rows.iloc[row_index] if hasattr(rows, "iloc") else rows[row_index]
        path = Path(str(row.iloc[-1] if hasattr(row, "iloc") else row[-1]))
    except Exception:  # noqa: BLE001 - a stray click must not break the tab
        return None, "", ""

    depth, prompt = path / "depth.mp4", path / "prompt.txt"
    return (
        str(depth) if depth.exists() else None,
        prompt.read_text() if prompt.exists() else "",
        str(path),
    )


# --------------------------------------------------------------------------------- app

def build() -> gr.Blocks:
    with gr.Blocks(title="Depth Converter", analytics_enabled=False) as demo:
        gr.HTML(
            f'<div class="depthc-head"><h1>DEPTH CONVERTER</h1>'
            f"<span>reference clip → Seedance motion kit · v{__version__} · "
            f"nothing leaves this Mac</span></div>"
        )

        with gr.Tabs():
            # ------------------------------------------------------------- convert tab
            with gr.Tab("Convert"):
                with gr.Row():
                    with gr.Column(scale=5):
                        video = gr.Video(label="Reference clip", height=280)
                        readout = gr.HTML()

                        with gr.Group(elem_classes="depthc-card"):
                            gr.Markdown("**Trim** — 3–8s follows best, one motion signal")
                            start = gr.Slider(0, 15, 0, step=0.1, label="In (s)")
                            end = gr.Slider(0, 15, 15, step=0.1, label="Out (s)")
                            # Seeded rather than empty: an empty HTML block renders as a
                            # bare slab, and a hint is more use than a blank bar anyway.
                            trim_note = gr.HTML(
                                '<div class="depthc-note">Load a clip to set the range.</div>'
                            )

                        tags = gr.Textbox(
                            label="Tags", placeholder="sword fight, tracking orbit",
                            info="Comma-separated. Also names the kit folder.",
                        )
                        description = gr.Textbox(label="Description", lines=2)

                        with gr.Accordion("Settings", open=False):
                            resolution = gr.Radio(["720p", "480p", "source"], value="720p",
                                                  label="Resolution")
                            crop_mode = gr.Radio(
                                ["auto", "none"], value="auto", label="Letterbox crop",
                                info="A depth model reads matte bars as the nearest surface.",
                            )
                            fps_mode = gr.Radio(["conform-24", "keep"], value="conform-24",
                                                label="Frame rate")
                            faces = gr.Checkbox(
                                True, label="Source shows real people",
                                info="still.png is cut from the original pixels, so it "
                                     "carries a real face Seedance rejects on image inputs.",
                            )
                            invert = gr.Checkbox(False, label="Invert (far = bright)")
                            keep_audio = gr.Checkbox(
                                False, label="Keep source audio",
                                info="For your own beat reference. Seedance ignores "
                                     "reference-video audio.",
                            )
                            make_preview = gr.Checkbox(True, label="Render side-by-side preview")
                            still_mode = gr.Radio(["auto", "first"], value="auto",
                                                  label="Still frame")
                            input_size = gr.Slider(266, 700, 518, step=14,
                                                   label="Inference size (×14)")
                            precision = gr.Radio(
                                ["fp32", "fp16"], value="fp32", label="Precision",
                                info="fp16 returns NaN on MPS and is slower; it "
                                     "auto-falls back to fp32.",
                            )
                            library_path = gr.Textbox(
                                label="Library root",
                                placeholder=str(kitlib.library_root()),
                            )

                        convert_btn = gr.Button("Convert", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        status = gr.Textbox(label="Status", interactive=False, lines=2)
                        depth_out = gr.Video(label="depth.mp4", height=300)
                        preview_out = gr.Video(label="original | depth", height=240, visible=True)
                        report_out = gr.HTML()
                        prompt_out = gr.Textbox(
                            label="prompt.txt", lines=12, interactive=False)
                        kit_dir_box = gr.Textbox(label="Kit folder", interactive=False)
                        with gr.Row():
                            reveal_btn = gr.Button("Reveal in Finder")
                            zip_note = gr.Markdown(
                                '<span class="depthc-note">Kits land in <b>Outbox/</b>. '
                                "Take <b>depth.mp4</b> and <b>binding.txt</b> to OSIDE; "
                                "the kit's README.txt has the steps.</span>"
                            )

                video.change(_probe_readout, video, [readout, start, end])
                for control in (start, end):
                    control.change(_trim_feedback, [start, end], trim_note)

                convert_btn.click(
                    _convert,
                    [video, tags, description, resolution, crop_mode, fps_mode, start, end,
                     invert, faces, still_mode, input_size, precision, keep_audio,
                     make_preview, library_path],
                    [depth_out, preview_out, report_out, prompt_out, kit_dir_box, status],
                )
                reveal_btn.click(_reveal, kit_dir_box, status)

            # ------------------------------------------------------------- library tab
            with gr.Tab("Library"):
                with gr.Row():
                    tag_filter = gr.Textbox(label="Filter by tag", scale=3)
                    lib_root = gr.Textbox(label="Library root",
                                          placeholder=str(kitlib.library_root()), scale=3)
                    refresh_btn = gr.Button("Refresh", scale=1)

                table = gr.Dataframe(
                    headers=["kit", "duration", "tier", "tags", "created", "path"],
                    datatype=["str"] * 6,
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    lib_depth = gr.Video(label="depth.mp4", height=280)
                    lib_prompt = gr.Textbox(label="prompt.txt", lines=12, interactive=False)
                lib_path = gr.Textbox(label="Kit folder", interactive=False)
                lib_reveal = gr.Button("Reveal in Finder")
                lib_status = gr.Textbox(label="", interactive=False, visible=False)

                refresh_btn.click(_library_rows, [tag_filter, lib_root], table)
                tag_filter.submit(_library_rows, [tag_filter, lib_root], table)
                table.select(_library_pick, table, [lib_depth, lib_prompt, lib_path])
                lib_reveal.click(_reveal, lib_path, lib_status)
                demo.load(_library_rows, [tag_filter, lib_root], table)

    return demo


def launch(port: int = DEFAULT_PORT, share: bool = False, open_browser: bool = True) -> None:
    build().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=open_browser,
        theme=THEME,
        css=CSS,
        quiet=False,
    )
