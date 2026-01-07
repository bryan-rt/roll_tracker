#!/usr/bin/env bash
set -euo pipefail

: "${TZ:=UTC}"
ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime && dpkg-reconfigure -f noninteractive tzdata >/dev/null 2>&1 || true

echo "[recorder] TZ=$TZ  window=${WINDOW_MINUTES:-30}m  seg=${SEG_SECONDS:-120}s"

# Simple loop scheduler (daily). For dev, this is enough; later you can swap in APScheduler.
while true; do
  NOW=$(date +%s)
  # Next weekday 20:00 local time
  TARGET=$(date -d "today 20:00" +%s)
  [ $NOW -ge $TARGET ] && TARGET=$(date -d "tomorrow 20:00" +%s)

  # If weekend, skip to Monday
  DOW=$(date +%u)  # 1..7 (Mon..Sun)
  if [ $DOW -ge 6 ]; then
    # jump to next Monday 20:00
    TARGET=$(date -d "next Monday 20:00" +%s)
  fi

  SLEEP=$(( TARGET - NOW ))
  echo "[recorder] Sleeping until $(date -d @$TARGET) ($SLEEP s)"
  sleep $SLEEP

  echo "[recorder] Starting recording window at $(date)"
  /app/record_window.sh || echo "[recorder] window failed (continuing)"
done
