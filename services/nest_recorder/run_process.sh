#!/usr/bin/env bash
set -euo pipefail

# --- Keep macOS awake while this script runs (no-op elsewhere) ---
if [[ "${OSTYPE:-}" == darwin* ]] && command -v caffeinate >/dev/null 2>&1; then
  # -d: prevent display sleep, -i: prevent idle sleep, -m: prevent disk sleep
  # -s: only while on AC power (safe default), -w $$: hold until THIS script exits
  caffeinate -dims -s -w $$ &
fi

# Default diag script (override with first argument)
SCRIPT="${1:-/app/diag_v8.sh}"

# Compose files
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"

echo "[run_diag] Stopping and removing old containers/images..."
docker compose $COMPOSE_FILES down --volumes --remove-orphans

echo "[run_diag] Rebuilding image..."
docker compose $COMPOSE_FILES build --no-cache recorder

echo "[run_diag] Starting container in background..."
docker compose $COMPOSE_FILES up -d recorder

echo "[run_diag] Executing $SCRIPT inside container..."
docker compose $COMPOSE_FILES exec recorder bash -lc "$SCRIPT"
