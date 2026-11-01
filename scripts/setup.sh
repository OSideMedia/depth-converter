#!/usr/bin/env bash
# Depth Converter — one-shot bootstrap. Idempotent; safe to re-run.
#
# Installs a PROJECT-LOCAL arm64 uv (.tools/uv), version-pinned to this project,
# rather than relying on the machine-wide uv. Until 2026-08-24 the Homebrew uv was
# x86_64 and could not resolve PyTorch's arm64-only macOS wheels; Homebrew is native
# arm64 now (/opt/homebrew) and the local pin stands on reproducibility alone — see
# CLAUDE.md "Always use the project-local uv". Everything lands in this folder except what uv
# keeps for itself in the home folder: the Python build and its download cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="$ROOT/.tools/uv"
UV_VERSION="0.11.32"
VDA_COMMIT="4f5ae23172ba60fd7bc11ef671cca678842c7072"
VDA_REPO="https://github.com/DepthAnything/Video-Depth-Anything"
CKPT="checkpoints/video_depth_anything_vits.pth"
CKPT_URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
# Hugging Face's own LFS object id for that file, read 2026-11-01. torch.load runs with
# weights_only=True as well, so this is the first of two locks, not the only one.
CKPT_SHA256="13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# 0 when the file's SHA-256 is the pinned one. tests/test_public_repo.py runs this function.
verify_ckpt() {
  [ "$(shasum -a 256 "$1" | awk '{print $1}')" = "$CKPT_SHA256" ]
}

# ---------------------------------------------------------------- 0. can this run at all
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "FATAL: this needs an Apple Silicon Mac in a native shell (found $(uname -s) $(uname -m))." >&2
  exit 1
fi
# The locked torch has wheels for macOS 14 and newer only, and no source build to fall back on.
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]; then
  echo "FATAL: this needs macOS 14 or newer (found $(sw_vers -productVersion))." >&2
  exit 1
fi
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "NOTE: $tool is not on PATH. depthc needs it to run: brew install ffmpeg" >&2
  fi
done

# ---------------------------------------------------------------- 1. arm64 uv
say "Toolchain: project-local arm64 uv"
if [ ! -x "$UV" ]; then
  mkdir -p "$ROOT/.tools"
  # The variable belongs on sh, the installer. On curl it would never reach it, and the
  # installer would write to the home folder and edit the shell profile instead.
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | UV_UNMANAGED_INSTALL="$ROOT/.tools" sh
fi
if ! file "$UV" | grep -q arm64; then
  echo "FATAL: $UV is not an arm64 binary. Delete .tools/ and re-run." >&2
  exit 1
fi
echo "uv: $("$UV" --version) ($(file -b "$UV" | sed 's/,.*//'))"

# --------------------------------------------------------- 2. arm64 CPython + deps
say "Python environment (arm64 CPython 3.12)"
# --no-bin: the project only reaches this Python through .venv, so nothing goes on the PATH.
"$UV" python install --no-bin cpython-3.12.13-macos-aarch64-none
"$UV" sync --locked

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
  curl -L --fail --progress-bar -o "$CKPT.part" "$CKPT_URL"
  if ! verify_ckpt "$CKPT.part"; then
    rm -f "$CKPT.part"
    echo "FATAL: the downloaded checkpoint does not match its pinned SHA-256. Nothing was installed." >&2
    exit 1
  fi
  mv "$CKPT.part" "$CKPT"
elif verify_ckpt "$CKPT"; then
  echo "already present"
else
  echo "FATAL: $CKPT does not match its pinned SHA-256. Delete it and re-run." >&2
  exit 1
fi
ls -lh "$CKPT" | awk '{print $5, $9}'

say "Setup complete. Next: ./.tools/uv run python -m depthc.smoke"
