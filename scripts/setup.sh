#!/usr/bin/env bash
# Depth Converter — one-shot bootstrap. Idempotent; safe to re-run.
#
# Installs a PROJECT-LOCAL arm64 uv (.tools/uv), version-pinned to this project,
# rather than relying on the machine-wide uv. Until 2026-08-24 the Homebrew uv was
# x86_64 and could not resolve PyTorch's arm64-only macOS wheels; Homebrew is native
# arm64 now (/opt/homebrew) and the local pin stands on reproducibility alone — see
# CLAUDE.md "Always use the project-local uv". Nothing outside this folder is modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="$ROOT/.tools/uv"
VDA_COMMIT="4f5ae23172ba60fd7bc11ef671cca678842c7072"
VDA_REPO="https://github.com/DepthAnything/Video-Depth-Anything"
CKPT="checkpoints/video_depth_anything_vits.pth"
CKPT_URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- 1. arm64 uv
say "Toolchain: project-local arm64 uv"
if [ ! -x "$UV" ]; then
  mkdir -p "$ROOT/.tools"
  UV_UNMANAGED_INSTALL="$ROOT/.tools" curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! file "$UV" | grep -q arm64; then
  echo "FATAL: $UV is not an arm64 binary. Delete .tools/ and re-run." >&2
  exit 1
fi
echo "uv: $("$UV" --version) ($(file -b "$UV" | sed 's/,.*//'))"

# --------------------------------------------------------- 2. arm64 CPython + deps
say "Python environment (arm64 CPython 3.12)"
"$UV" python install cpython-3.12.13-macos-aarch64-none
"$UV" sync

# ------------------------------------------------------------- 3. vendor VDA
say "Vendoring Video-Depth-Anything @ ${VDA_COMMIT:0:10}"
if [ ! -d vendor/Video-Depth-Anything/.git ]; then
  rm -rf vendor/Video-Depth-Anything
  git clone --quiet "$VDA_REPO" vendor/Video-Depth-Anything
fi
git -C vendor/Video-Depth-Anything fetch --quiet origin "$VDA_COMMIT" 2>/dev/null || true
git -C vendor/Video-Depth-Anything checkout --quiet "$VDA_COMMIT"
echo "vendor pinned at $(git -C vendor/Video-Depth-Anything rev-parse HEAD)"
# NOTE: their requirements.txt is CUDA-era (torch==2.1.1, xformers, numpy==1.24.0).
# We deliberately never install it. See PLAN.md §0 pitfall 3.

# ---------------------------------------------------------- 4. checkpoint (Apache-2.0)
say "Checkpoint: VDA-Small (video_depth_anything_vits.pth, 116 MB)"
mkdir -p checkpoints
if [ ! -f "$CKPT" ]; then
  curl -L --fail --progress-bar -o "$CKPT" "$CKPT_URL"
else
  echo "already present"
fi
ls -lh "$CKPT" | awk '{print $5, $9}'

say "Setup complete. Next: ./.tools/uv run python -m depthc.smoke"
