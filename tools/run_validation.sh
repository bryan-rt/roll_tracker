#!/usr/bin/env bash
# Run the processor for validation, outputting to a separate directory
# for comparison against the baseline.
#
# All key env vars default to the CP17 cross-camera run but can be
# overridden from the caller:
#
#   OUTPUT_ROOT=outputs_color_hist_pose \
#   CONFIG_OVERLAY=configs/validation_color_hist_pose.yaml \
#   caffeinate -is bash tools/run_validation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT" || exit 1

# Clean stale workers (same as run_local.sh)
echo "[run_validation] Cleaning up stale workers..."
pkill -f "bjj_pipeline.stages" 2>/dev/null || true
pkill -f "services/processor/processor.py" 2>/dev/null || true
sleep 1

_cleanup() {
    echo "[run_validation] Cleaning up workers on exit..."
    pkill -f "bjj_pipeline.stages" 2>/dev/null || true
    pkill -f "services/processor/processor.py" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# Source .env first (baseline values)
set -a
source "$REPO_ROOT/.env"
set +a

# Override AFTER .env so our values win; respect pre-set env vars
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_cross_camera}"
export GYM_ID="${GYM_ID:-c8a592a4-2bca-400a-80e1-fec0e5cbea77}"
export CONFIG_OVERLAY="${CONFIG_OVERLAY:-$REPO_ROOT/configs/validation_cross_camera.yaml}"
export MAX_CLIP_AGE_HOURS="${MAX_CLIP_AGE_HOURS:-0}"

source "$REPO_ROOT/.venv/bin/activate"

echo "[run_validation] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[run_validation] GYM_ID=$GYM_ID"
echo "[run_validation] CONFIG_OVERLAY=$CONFIG_OVERLAY"
echo "[run_validation] MAX_CLIP_AGE_HOURS=$MAX_CLIP_AGE_HOURS"

PYTHONPATH="$REPO_ROOT/services/processor:$PYTHONPATH" \
  PYTHONUNBUFFERED=1 \
  exec python "$REPO_ROOT/services/processor/processor.py"
