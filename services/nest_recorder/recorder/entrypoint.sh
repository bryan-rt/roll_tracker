#!/usr/bin/env bash
set -euo pipefail

: "${TZ:=UTC}"
ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime && dpkg-reconfigure -f noninteractive tzdata >/dev/null 2>&1 || true

echo "[recorder] TZ=$TZ — handing off to diag_v8.sh scheduler"
exec /app/diag_v8.sh
