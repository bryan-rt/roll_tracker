#!/usr/bin/env bash
# Run processor natively on Mac for better ARM performance.
# For Linux deployment, use docker compose instead.
#
# Usage: bash services/processor/run_local.sh
#        (run from repo root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT" || exit 1

# Export all vars from .env
set -a
source "$REPO_ROOT/.env"
set +a

source "$REPO_ROOT/.venv/bin/activate"

PYTHONPATH="$REPO_ROOT/services/processor:$PYTHONPATH" \
  PYTHONUNBUFFERED=1 \
  exec python "$REPO_ROOT/services/processor/processor.py"
