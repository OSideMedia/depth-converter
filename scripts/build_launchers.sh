#!/usr/bin/env bash
# Build the /Applications launcher pair, mirroring the OSIDE Studio.app / OSIDE Stop.app
# pattern. Re-runnable: each build replaces the previous bundle.
#
# THE arch -arm64 TRAP: an app bundle launched from Finder does not inherit the shell's
# architecture. Until 2026-08-24 this Mac's Homebrew tooling was x86_64, and without
# `arch -arm64` in the exec line the server came up under Rosetta with no MPS at all.
# Homebrew is native arm64 now (/opt/homebrew, CLAUDE.md), but the prefix stays: a Finder
# launch still inherits nothing, and the cost is nil. The OSIDE pair hit exactly this;
# PLAN.md §0 says not to rediscover it here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS="${1:-/Applications}"
PORT="${DEPTHC_PORT:-7861}"
PIDFILE="$HOME/.depthc-ui.pid"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

make_app() {
  local name="$1" script="$2"
  local bundle="$APPS/$name.app"
  local macos="$bundle/Contents/MacOS"

  rm -rf "$bundle"
  mkdir -p "$macos"

  cat > "$bundle/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundleIdentifier</key><string>media.oside.depthc.$(echo "$name" | tr -dc '[:alnum:]')</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>run</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

  printf '%s\n' "$script" > "$macos/run"
  chmod +x "$macos/run"
  echo "built $bundle"
}

say "Building launchers in $APPS (port $PORT)"

# ------------------------------------------------------------------------ start
make_app "Depth Converter" "#!/bin/bash
# Launch the Depth Converter UI, wait for the port, then open a browser tab.
ROOT=\"$ROOT\"
PORT=$PORT
PIDFILE=\"$PIDFILE\"
LOG=\"\$HOME/Library/Logs/depthc-ui.log\"
mkdir -p \"\$(dirname \"\$LOG\")\"

# Already running? Just focus the tab rather than starting a second server.
if lsof -ti :\$PORT >/dev/null 2>&1; then
  open \"http://127.0.0.1:\$PORT\"
  exit 0
fi

cd \"\$ROOT\" || {
  osascript -e 'display alert \"Depth Converter\" message \"Project folder not found.\"'
  exit 1
}

# arch -arm64 is load-bearing: a Finder launch does not inherit the shell's architecture,
# and under Rosetta there is no MPS.
arch -arm64 ./.tools/uv run depthc ui --port \$PORT --no-browser >>\"\$LOG\" 2>&1 &
echo \$! > \"\$PIDFILE\"

for _ in \$(seq 1 60); do
  if curl -s -o /dev/null \"http://127.0.0.1:\$PORT/\"; then
    open \"http://127.0.0.1:\$PORT\"
    exit 0
  fi
  sleep 1
done

osascript -e \"display alert \\\"Depth Converter\\\" message \\\"Server did not start within 60s. See \$LOG\\\"\"
exit 1
"

# ------------------------------------------------------------------------- stop
make_app "Depth Stop" "#!/bin/bash
# Stop the Depth Converter UI: pidfile first, then anything still holding the port.
PORT=$PORT
PIDFILE=\"$PIDFILE\"

if [ -f \"\$PIDFILE\" ]; then
  PID=\$(cat \"\$PIDFILE\")
  kill \"\$PID\" 2>/dev/null || true
  rm -f \"\$PIDFILE\"
fi

# The pidfile tracks the launcher's child; uv may have spawned its own. Sweep the port so
# a stale listener can never block the next start.
sleep 1
LEFT=\$(lsof -ti :\$PORT 2>/dev/null || true)
if [ -n \"\$LEFT\" ]; then
  echo \"\$LEFT\" | xargs kill 2>/dev/null || true
  sleep 1
  LEFT=\$(lsof -ti :\$PORT 2>/dev/null || true)
  [ -n \"\$LEFT\" ] && echo \"\$LEFT\" | xargs kill -9 2>/dev/null || true
fi
exit 0
"

say "Done. Double-click 'Depth Converter' to start, 'Depth Stop' to shut down."
echo "Port $PORT · pidfile $PIDFILE · log ~/Library/Logs/depthc-ui.log"
