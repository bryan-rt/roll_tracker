#!/usr/bin/env bash
set -euo pipefail

# CP-R9: Sleep-protected capture wrapper.
#
# Wraps diag_v7_2.sh with caffeinate so long captures are protected from
# macOS display/idle/disk sleep regardless of invocation path.
#
# Usage:
#   ./capture.sh                     # default: 65-min window, 120s segments
#   ./capture.sh --window 1800       # 30-min validation run
#   ./capture.sh --window 3900 --seg 120
#
# For detached operation:
#   nohup ./capture.sh --window 3900 &
#   # or use screen/tmux

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# --- Parse arguments ---
WINDOW_SECONDS=3900
SEG_SECONDS=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window) WINDOW_SECONDS="$2"; shift 2 ;;
    --seg)    SEG_SECONDS="$2"; shift 2 ;;
    *)        echo "Unknown argument: $1"; exit 1 ;;
  esac
done

MARGIN=300  # 5 minutes for startup, jitter, sidecar extraction, cleanup
CAFFEINATE_SECS=$(( WINDOW_SECONDS + MARGIN ))
CAFF_PID=""

# --- Caffeinate (macOS only) ---
if [[ "${OSTYPE:-}" == darwin* ]] && command -v caffeinate >/dev/null 2>&1; then
  # -d: prevent display sleep, -i: prevent idle sleep, -m: prevent disk sleep
  # -t: self-terminate after CAFFEINATE_SECS (no leak, no orphan)
  # No -s: holds on battery as well as AC power
  caffeinate -dim -t "$CAFFEINATE_SECS" &
  CAFF_PID=$!
  echo "[capture] caffeinate PID=$CAFF_PID (${CAFFEINATE_SECS}s = window + ${MARGIN}s margin)"
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
echo "[capture] Starting ${WINDOW_SECONDS}s capture (${SEG_SECONDS}s segments)..."
echo "[capture] SOURCE_PTS and FPS_PASSTHROUGH use defaults (both 1 since CP-R3)"
echo ""

$COMPOSE exec recorder bash -lc \
  "WINDOW_SECONDS=$WINDOW_SECONDS SEG_SECONDS=$SEG_SECONDS /app/diag_v7_2.sh"

echo ""
echo "[capture] Done."
