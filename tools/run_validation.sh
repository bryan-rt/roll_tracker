#!/usr/bin/env bash
# Run the processor with CP17 Tier 2 coordinate evidence enabled,
# outputting to outputs_cross_camera for comparison against baseline.
#
# Usage: bash tools/run_validation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OUTPUT_ROOT=outputs_cross_camera
export GYM_ID=c8a592a4-2bca-400a-80e1-fec0e5cbea77
export CONFIG_OVERLAY="$REPO_ROOT/configs/validation_cross_camera.yaml"
export MAX_CLIP_AGE_HOURS=0

exec bash "$REPO_ROOT/services/processor/run_local.sh"
