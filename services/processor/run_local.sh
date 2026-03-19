#!/bin/bash
# Run processor natively on Mac for better ARM performance.
# For Linux deployment, use docker compose instead.
set -a
source "$(dirname "$0")/../../.env"
set +a
cd "$(dirname "$0")/../.."
source .venv/bin/activate
PYTHONPATH=services/processor:$PYTHONPATH \
  PYTHONUNBUFFERED=1 \
  python services/processor/processor.py
