#!/usr/bin/env bash
# Run processor natively on Mac for better ARM performance.
# For Linux deployment, use docker compose instead.
#
# Usage: bash services/processor/run_local.sh
#        (run from repo root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT" || exit 1

# Kill any stale processor/pipeline worker processes from prior runs.
# ProcessPoolExecutor spawn-mode workers are orphaned on unclean parent exit.
# This ensures each run starts from a clean slate.
echo "[run_local] Cleaning up stale workers..."
pkill -f "bjj_pipeline.stages" 2>/dev/null || true
pkill -f "services/processor/processor.py" 2>/dev/null || true
sleep 1
echo "[run_local] Stale worker cleanup complete."

# Trap: kill spawned workers on any exit (clean, error, or signal).
_cleanup() {
    echo "[run_local] Cleaning up workers on exit..."
    pkill -f "bjj_pipeline.stages" 2>/dev/null || true
    pkill -f "services/processor/processor.py" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# Export all vars from .env
set -a
source "$REPO_ROOT/.env"
set +a

source "$REPO_ROOT/.venv/bin/activate"

PYTHONPATH="$REPO_ROOT/services/processor:$PYTHONPATH" \
  PYTHONUNBUFFERED=1 \
  exec python "$REPO_ROOT/services/processor/processor.py"
