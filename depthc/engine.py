"""VDA-Small inference wrapper: device selection, windowed inference, normalization.

Upstream's `run.py` hardcodes `'cuda' if available else 'cpu'` — there is no MPS branch.
Rather than patch a vendored file, we import their model class and own device and precision
selection here.

PRECISION: fp32 IS THE DEFAULT ON MPS, and this contradicts PLAN.md §0 pitfall 2, which
predicted "torch>=2.5 and fp16 works; keep a --fp32 escape hatch". Measured on M4 Pro /
torch 2.13.0 / macOS 15.5, VDA-Small under `torch.autocast('mps')`:

    fp32   clean output   1.40 s/window   44 ms/frame
    fp16   ALL NaN        1.55 s/window   49 ms/frame
    bf16   ALL NaN        1.55 s/window   48 ms/frame

Reduced precision is not a speed/quality trade here — it is simply broken, and slower. So
the escape hatch is inverted: fp32 by default, `--precision fp16` opt-in for anyone whose
hardware behaves differently, guarded by a finiteness check that falls back automatically.

Worse, the NaN is *sticky*: an fp16 run poisons the next fp32 run on the same model object
until `torch.mps.empty_cache()` is called, which points at a cached MPS graph rather than
arithmetic overflow. Our per-window cache purge already clears it; the fallback path calls
it explicitly rather than relying on that coincidence.

MEMORY: upstream hands all 32 frames of a window to the DINOv2 backbone as one batch, and without
xFormers (CUDA only) each of its 12 blocks builds a full attention matrix for that batch: frames x
heads x tokens^2 floats, or 4.27 GiB at 518 x 924. That took a pass to 18 GB of unified memory and
all but froze a 24 GB Mac with other apps open. Attention never crosses frames in the backbone, so
`bound_memory` feeds it a few frames at a time, the way upstream's own decoder already does, and the
output bytes do not change. `memory.py` then sets a limit from what the machine has free.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from depthc import memory
from depthc.vendor_path import MODEL_CONFIG, ensure_checkpoint, ensure_vendor_on_path

ProgressFn = Callable[[int, int], None]
WarnFn = Callable[[str], None]

# bf16 was measured and is equally broken on MPS; it is deliberately not offered, since
# upstream's autocast call exposes only an on/off flag and wrapping it to force a dtype
# would be complexity in service of a known-bad path.
PRECISIONS = ("fp32", "fp16")

# Upstream's fixed windowing constants, mirrored here only so we can report progress
# in the same units the model actually works in. Do not "tune" these.
INFER_LEN = 32
OVERLAP = 10

# Frames the backbone sees at once. Groups of 8 and of 4 gave identical bytes at the same speed;
# 4 peaked lower (7.9 against 8.5 GB) and matches upstream's decoder micro-batch.
BACKBONE_FRAMES = 4

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def pick_device(prefer: str | None = None) -> str:
    """mps when available, else cpu (with the caller expected to warn loudly)."""
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


def _settle(device: str) -> None:
    """Finish queued GPU work and hand back what it held. Emptying alone frees nothing still queued."""
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _in_groups(whole: Callable, device: str) -> Callable:
    """Wrap the backbone's feature call so it sees BACKBONE_FRAMES frames at a time."""

    def grouped(x, n=1, reshape=False, return_class_token=False, norm=True):
        if not return_class_token:  # upstream never asks for this shape; leave it whole
            return whole(x, n, reshape=reshape, return_class_token=False, norm=norm)
        parts = []
        for start in range(0, x.shape[0], BACKBONE_FRAMES):
            parts.append(whole(x[start:start + BACKBONE_FRAMES], n, reshape=reshape,
                               return_class_token=True, norm=norm))
            _settle(device)
        return tuple(
            (torch.cat([part[layer][0] for part in parts]), torch.cat([part[layer][1] for part in parts]))
            for layer in range(len(parts[0]))
        )

    return grouped


def bound_memory(model: torch.nn.Module, device: str) -> None:
    """Keep a window's memory down, with no vendor edit and no change to the output."""
    model.pretrained.get_intermediate_layers = _in_groups(model.pretrained.get_intermediate_layers, device)
    for stage in model.head.motion_modules:
        stage.register_forward_hook(lambda *_: _settle(device))


def limit_gpu(device: str, limit: int) -> None:
    """A hard line behind the estimate: MPS raises at it instead of swapping the machine to a stall."""
    if device == "mps":
        fraction = min(2.0, limit / torch.mps.recommended_max_memory())  # 2 is torch's ceiling
        torch.mps.set_per_process_memory_fraction(fraction)


def load_model(device: str | None = None) -> tuple[torch.nn.Module, str]:
    """Load VDA-Small, cached per device.

    The Gradio process is long-running and reloading a 116 MB checkpoint per conversion
    would be pure waste, so the model is kept resident and only activations are freed.
    """
    device = device or pick_device()
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device], device

    ensure_vendor_on_path()
    checkpoint = ensure_checkpoint()
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIG)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    bound_memory(model, device)
    _MODEL_CACHE[device] = model
    return model, device


def unload_model() -> None:
    """Drop the resident model (UI teardown / tests that want a clean slate)."""
    for device in list(_MODEL_CACHE):
        del _MODEL_CACHE[device]
        empty_cache(device)
    gc.collect()


@contextmanager
def _window_hook(device: str, on_window: ProgressFn | None):
    """Hook upstream's per-window loop without editing a vendored file.

    `infer_video_depth` drives its window loop with `for frame_id in tqdm(range(...))`.
    We temporarily rebind the `tqdm` name *inside their module's namespace* to a generator
    of our own. Control returns to us after each loop body completes, which is precisely
    the moment to report progress and release MPS's cached allocations — the two hooks
    PLAN.md §2 asks for, with zero vendor edits and no reimplementation of their
    scale/shift alignment maths.
    """
    import video_depth_anything.video_depth as vda

    original = vda.tqdm

    def hooked(iterable: Iterable, *args, **kwargs):
        items = list(iterable)
        total = len(items)
        for index, item in enumerate(items):
            yield item
            # Reached only once the window's forward pass has finished.
            empty_cache(device)
            if on_window:
                on_window(index + 1, total)

    vda.tqdm = hooked
    try:
        yield
    finally:
        vda.tqdm = original


@dataclass
class DepthResult:
    """Normalized depth frames plus the numbers worth putting in kit.json."""

    frames: np.ndarray  # [N, H, W] uint8
    fps: float
    device: str
    dtype: str
    elapsed: float
    raw_min: float
    raw_max: float
    inverted: bool

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def ms_per_frame(self) -> float:
        return self.elapsed / max(1, self.count) * 1000


class DepthFailure(RuntimeError):
    """A depth pass could not give a usable result. The message is written for the person running it."""


class NonFiniteDepth(DepthFailure):
    """Inference produced NaN/Inf — see the precision note at the top of this module."""


class NotEnoughMemory(DepthFailure):
    """This machine cannot spare what the pass needs — see `memory.py`."""


STOPPED_AT_LIMIT = (
    f"The depth pass stopped at its memory limit before it finished. Close other apps and "
    f"run it again, or use a smaller --input-size. To allow more, set "
    f"{memory.OVERRIDE_ENV} to the GB this pass may use."
)


def is_out_of_memory(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def normalize(depth: np.ndarray, invert: bool = False) -> tuple[np.ndarray, float, float]:
    """Global (whole-clip) min/max normalization to uint8.

    This is upstream's normalization *logic*, reused deliberately: normalizing per-video
    rather than per-frame is what keeps brightness from pumping between frames. Combined
    with VDA's cross-window scale/shift alignment it is why the output is flicker-free.

    VDA emits disparity, so larger values are nearer: the default is near = bright.
    """
    # Casting NaN to uint8 is silent and yields zeros, so a broken run would otherwise
    # export as a plausible-looking all-black depth video. Fail loudly instead.
    if not np.isfinite(depth).all():
        raise NonFiniteDepth(
            "Depth output contains NaN/Inf — refusing to encode a corrupt depth video."
        )

    d_min = float(depth.min())
    d_max = float(depth.max())
    span = d_max - d_min
    if span <= 0:
        # Degenerate (perfectly flat) depth — emit mid-gray rather than divide by zero.
        out = np.full(depth.shape, 128, dtype=np.uint8)
    else:
        out = ((depth - d_min) / span * 255.0).astype(np.uint8)
    if invert:
        out = 255 - out
    return out, d_min, d_max


def _run_windows(
    model,
    frames: np.ndarray,
    fps: float,
    input_size: int,
    device: str,
    use_fp32: bool,
    on_window: ProgressFn | None,
) -> tuple[np.ndarray, float]:
    try:
        with _window_hook(device, on_window):
            return model.infer_video_depth(
                frames, fps, input_size=input_size, device=device, fp32=use_fp32
            )
    except RuntimeError as e:
        if not is_out_of_memory(e):
            raise
        raise NotEnoughMemory(STOPPED_AT_LIMIT) from e
    finally:
        empty_cache(device)


def infer(
    frames: np.ndarray,
    fps: float,
    input_size: int = 518,
    precision: str = "fp32",
    invert: bool = False,
    device: str | None = None,
    on_window: ProgressFn | None = None,
    on_warn: WarnFn | None = None,
) -> DepthResult:
    """Run windowed depth inference over [N, H, W, 3] uint8 RGB frames."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [N, H, W, 3] RGB frames, got {frames.shape}")
    if input_size % 14:
        raise ValueError(f"--input-size must be a multiple of 14 (DINOv2 patches), got {input_size}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    # Before the model loads, so a pass that cannot fit touches nothing.
    plan = memory.plan(frames.shape, input_size)
    if plan.refusal:
        raise NotEnoughMemory(plan.refusal)

    model, device = load_model(device)
    limit_gpu(device, plan.gpu_limit)
    # CPU autocast would be pointless; fp32 is also the default everywhere (see module docs).
    use_fp32 = precision == "fp32" or device == "cpu"

    start = time.perf_counter()
    depth, out_fps = _run_windows(
        model, frames, fps, input_size, device, use_fp32, on_window
    )

    # Reduced precision NaNs out on MPS. Rather than hand the user a black video, detect it
    # and redo the clip at fp32 — which is the faster path anyway.
    if not use_fp32 and not np.isfinite(depth).all():
        message = (
            f"{precision} produced non-finite depth on {device} (a known MPS defect) — "
            f"retrying at fp32."
        )
        if on_warn:
            on_warn(message)
        del depth
        gc.collect()
        # Mandatory: the poisoned graph survives into the next fp32 run without this.
        empty_cache(device)
        use_fp32 = True
        depth, out_fps = _run_windows(
            model, frames, fps, input_size, device, True, on_window
        )

    elapsed = time.perf_counter() - start
    normalized, raw_min, raw_max = normalize(depth, invert=invert)

    # Back-to-back conversions in one long-lived Gradio session must not creep toward
    # OOM: drop the float32 intermediates before returning the uint8 result.
    del depth
    gc.collect()
    empty_cache(device)

    return DepthResult(
        frames=normalized,
        fps=out_fps,
        device=device,
        dtype="fp32" if use_fp32 else precision,
        elapsed=elapsed,
        raw_min=raw_min,
        raw_max=raw_max,
        inverted=invert,
    )


def expected_windows(frame_count: int) -> int:
    """How many inference windows a clip of this length will take (for progress bars)."""
    step = INFER_LEN - OVERLAP
    return max(1, -(-frame_count // step))
