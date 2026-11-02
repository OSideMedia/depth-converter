"""Gate 0 — toolchain and model sanity.

    ./.tools/uv run python -m depthc.smoke

Asserts the arm64/MPS story end to end, then loads VDA-Small and runs one real
32-frame inference window. Exit 0 = green gate.

Checking `platform.machine()` alone is NOT sufficient: the system framework Python is a
universal binary that reports arm64 while an x86_64 `uv` still resolves x86_64 wheels and
never installs torch at all. So we assert on torch's own build, which is the thing that
actually has to be arm64.
"""

from __future__ import annotations

import platform
import sys
import time

from depthc import VDA_COMMIT, __version__
from depthc.vendor_path import (
    MODEL_CONFIG,
    SetupIncomplete,
    ensure_checkpoint,
    ensure_vendor_on_path,
    vendor_commit,
)

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{OFF} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{OFF} {YELLOW}{msg}{OFF}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{OFF}")


def main() -> int:
    print(f"{BOLD}Depth Converter smoke test{OFF} {DIM}v{__version__}{OFF}\n")
    problems: list[str] = []

    # ---------------------------------------------------------- interpreter
    print(f"{BOLD}Toolchain{OFF}")
    machine = platform.machine()
    if machine == "arm64":
        ok(f"interpreter is arm64 (Python {platform.python_version()})")
    else:
        fail(f"interpreter is {machine}, expected arm64")
        problems.append("x86_64 interpreter — see PLAN.md §0 toolchain blocker")

    # ------------------------------------------------------------- torch
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment failure
        fail(f"torch did not import: {e}")
        print("\nRun: bash scripts/setup.sh")
        return 1

    # The real arm64 proof: an x86_64 uv cannot install torch on macOS at all,
    # because PyTorch publishes arm64-only macOS wheels.
    ok(f"torch {torch.__version__} imported")
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 5):
        fail(f"torch {torch.__version__} < 2.5 — MPS autocast is not correct below 2.5")
        problems.append("torch too old for MPS autocast")

    # ------------------------------------------------------------- device
    if torch.backends.mps.is_available():
        device = "mps"
        ok("MPS backend available")
    else:
        device = "cpu"
        warn("MPS UNAVAILABLE — falling back to CPU. Conversions will be very slow.")

    # -------------------------------------------------------------- vendor
    print(f"\n{BOLD}Vendored model{OFF}")
    try:
        ensure_vendor_on_path()
        ckpt = ensure_checkpoint()
    except SetupIncomplete as e:
        fail(str(e).splitlines()[0])
        print(f"\n{e}")
        return 1

    sha = vendor_commit()
    if sha == VDA_COMMIT:
        ok(f"vendor pinned at {sha[:10]}")
    else:
        warn(f"vendor at {(sha or '?')[:10]}, expected pin {VDA_COMMIT[:10]}")
    ok(f"checkpoint {ckpt.stat().st_size / 1e6:.1f} MB")

    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

    from depthc import engine, memory  # after the torch check above: engine imports torch

    # The same check as a real pass, so the first run on a new machine cannot stall it either.
    plan = memory.plan((32, 518, 518, 3), 518)
    if plan.refusal:
        fail(plan.refusal)
        return 1

    t0 = time.perf_counter()
    model = VideoDepthAnything(**MODEL_CONFIG)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    engine.bound_memory(model, device)
    engine.limit_gpu(device, plan.gpu_limit)
    ok(f"VDA-Small loaded on {device} in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------- one real 32-frame window
    print(f"\n{BOLD}Inference{OFF}")
    # 518 is the default short side; must be a multiple of 14 (DINOv2 patch size).
    # fp32, matching the runtime default — fp16 autocast NaNs out on MPS (engine.py docs).
    x = torch.rand(1, 32, 3, 518, 518, device=device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            depth = model(x)
        if device == "mps":
            torch.mps.synchronize()
    except RuntimeError as e:
        if not engine.is_out_of_memory(e):
            raise
        fail(engine.STOPPED_AT_LIMIT)
        return 1
    elapsed = time.perf_counter() - t0

    expected = (1, 32, 518, 518)
    if tuple(depth.shape) == expected:
        ok(f"output shape {tuple(depth.shape)} dtype {depth.dtype}")
    else:
        fail(f"output shape {tuple(depth.shape)}, expected {expected}")
        problems.append("unexpected model output shape")

    # Shape alone is a worthless gate: the fp16 path returns a perfectly-shaped tensor of
    # NaN, which casts to an all-black depth video without complaint. Check the values.
    if torch.isfinite(depth).all():
        ok(f"output finite, range [{depth.min():.3f}, {depth.max():.3f}]")
    else:
        fail("output contains NaN/Inf — the model is producing garbage on this device")
        problems.append("non-finite model output")

    if float(depth.std()) > 1e-4:
        ok(f"output varies (std {depth.std():.3f}) — not a constant plane")
    else:
        fail("output is a constant plane — not real depth")
        problems.append("degenerate model output")

    ok(f"32-frame window in {elapsed:.2f}s ({elapsed / 32 * 1000:.0f} ms/frame)")
    print(f"  {DIM}→ ~{elapsed / 32 * 240:.0f}s for a 10 s / 24 fps clip{OFF}")

    if device == "mps":
        torch.mps.empty_cache()

    # -------------------------------------------------------------- verdict
    print()
    if problems:
        print(f"{RED}{BOLD}GATE 0 FAIL{OFF}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{GREEN}{BOLD}GATE 0 OK{OFF} — {device.upper()}, arm64, VDA-Small ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
