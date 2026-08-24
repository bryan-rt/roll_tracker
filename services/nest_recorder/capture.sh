#!/usr/bin/env bash
set -euo pipefail

# CP-R9 + RECORDER-BACKLOG-1: Sleep-protected capture wrapper.
#
# Wraps diag_v7_2.sh with caffeinate so long captures are protected from
# macOS display/idle/disk sleep regardless of invocation path.
#
# Usage:
#   ./capture.sh                          # default: 65-min wall-clock window
#   ./capture.sh --window 1800            # 30-min wall-clock window
#   ./capture.sh --target 1800            # capture 30 min of content (auto wall cap: 5x)
#   ./capture.sh --target 1800 --window 5400  # 30 min content, 90 min wall cap
#   ./capture.sh --window 3900 --seg 120
#   ./capture.sh --cams "FP7oJQ:enterprises/.../devices/AAA"  # pin to one camera
#
# --target N  Capture N seconds of content. The run continues until that much
#             footage arrives, regardless of delivery speed. A wall-clock safety
#             cap prevents unbounded runs (default: 5x target; override with --window).
# --window N  Wall-clock cap in seconds. Without --target, this is the primary
#             bound (legacy mode). With --target, this overrides the auto-computed
#             safety cap.
#
# For detached operation:
#   nohup ./capture.sh --target 3600 &
#   # or use screen/tmux

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# --- Parse arguments ---
WINDOW_SECONDS=3900
SEG_SECONDS=120
TARGET_CONTENT_SECONDS=0
EXPLICIT_WINDOW=false
CAMS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window) WINDOW_SECONDS="$2"; EXPLICIT_WINDOW=true; shift 2 ;;
    --target) TARGET_CONTENT_SECONDS="$2"; shift 2 ;;
    --seg)    SEG_SECONDS="$2"; shift 2 ;;
    --cams)   CAMS="$2"; shift 2 ;;
    *)        echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# --- Compute wall-clock cap and caffeinate duration ---
MARGIN=300  # 5 minutes for startup, jitter, sidecar extraction, cleanup

if [[ "$TARGET_CONTENT_SECONDS" -gt 0 ]]; then
  if [[ "$EXPLICIT_WINDOW" == "true" ]]; then
    MAX_WALLCLOCK_SECONDS="$WINDOW_SECONDS"
  else
    # Auto: 5x target (covers two reconnects with ramp cost)
    MAX_WALLCLOCK_SECONDS=$(( TARGET_CONTENT_SECONDS * 5 ))
  fi
  CAFFEINATE_SECS=$(( MAX_WALLCLOCK_SECONDS + MARGIN ))
else
  MAX_WALLCLOCK_SECONDS=0  # legacy: v6 uses WINDOW_SECONDS directly
  CAFFEINATE_SECS=$(( WINDOW_SECONDS + MARGIN ))
fi

CAFF_PID=""

# --- Caffeinate (macOS only) ---
# Re-arm periodically: caffeinate -t cannot be extended, and a content-target
# run can last longer than the initial estimate if delivery is slower than the
# 5x safety margin assumed. Launch caffeinate without -t and kill it on exit.
if [[ "${OSTYPE:-}" == darwin* ]] && command -v caffeinate >/dev/null 2>&1; then
  # -d: prevent display sleep, -i: prevent idle sleep, -m: prevent disk sleep
  # No -t: runs until killed (cleanup trap handles it)
  caffeinate -dim &
  CAFF_PID=$!
  echo "[capture] caffeinate PID=$CAFF_PID (runs until capture exits)"
fi

# Belt and suspenders: kill caffeinate on early exit
cleanup() {
  if [[ -n "$CAFF_PID" ]] && kill -0 "$CAFF_PID" 2>/dev/null; then
    kill "$CAFF_PID" 2>/dev/null || true
    echo "[capture] caffeinate stopped"
  fi
}
trap cleanup EXIT INT TERM

# --- Ensure container is running ---
echo "[capture] Ensuring container is running..."
$COMPOSE up -d recorder 2>&1 | grep -v "^time=" || true

# --- Run the capture (blocking) ---
if [[ "$TARGET_CONTENT_SECONDS" -gt 0 ]]; then
  echo "[capture] Content-target mode: target=${TARGET_CONTENT_SECONDS}s, wall cap=${MAX_WALLCLOCK_SECONDS}s (${SEG_SECONDS}s segments)"
else
  echo "[capture] Legacy wall-clock mode: ${WINDOW_SECONDS}s window (${SEG_SECONDS}s segments)"
fi
echo "[capture] SOURCE_PTS and FPS_PASSTHROUGH use defaults (both 1 since CP-R3)"
echo ""

EXEC_ENV="WINDOW_SECONDS=$WINDOW_SECONDS SEG_SECONDS=$SEG_SECONDS TARGET_CONTENT_SECONDS=$TARGET_CONTENT_SECONDS MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
if [[ -n "$CAMS" ]]; then
  EXEC_ENV="$EXEC_ENV CAMS='$CAMS'"
  echo "[capture] CAMS=$CAMS"
fi

$COMPOSE exec recorder bash -lc "$EXEC_ENV /app/diag_v7_2.sh"

echo ""
echo "[capture] Done."
