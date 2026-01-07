#!/usr/bin/env bash
set -euo pipefail
SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
if [[ -z "${SDM_PROJECT_ID:-}" && -f "$SECRETS_DIR/project_id.txt" ]]; then
  SDM_PROJECT_ID="$(cat "$SECRETS_DIR/project_id.txt")"
fi
# ==============================
# diag_v8.sh — scheduler wrapper for diag_v7_2.sh
# ==============================
# Modes:
#  - SIMPLE (default): run once daily at HH:MM (local TZ)
#  - ADVANCED (CONFIG_SOURCE=env|file|supabase): use schedule windows from JSON config
#
# Requires: jq, curl, diag_v7_2.sh in /app
#
# Output: scheduler log at /recordings/diag/scheduler.log

# ---------- SIMPLE schedule defaults ----------
SCHED_TZ="${SCHED_TZ:-America/New_York}"
SCHED_DAILY_HHMM="${SCHED_DAILY_HHMM:-11:39}"  # e.g., "20:00"

# ---------- ADVANCED schedule config ----------
# CONFIG_SOURCE: env | file | supabase
CONFIG_SOURCE="${CONFIG_SOURCE:-}"   # empty => fallback SIMPLE
GYM_ID="${GYM_ID:-}"                 # for supabase
SUPABASE_URL="${SUPABASE_URL:-}"     # for supabase
SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-}"  # for supabase
CONFIG_PATH="${CONFIG_PATH:-}"       # for file
SCHEDULE_JSON="${SCHEDULE_JSON:-}"   # for env

# ---------- diag_v7_2 passthrough knobs (per-run) ----------
WINDOW_SECONDS_DEFAULT="${WINDOW_SECONDS:-2100}"  # used only if advanced config lacks duration
SEG_SECONDS="${SEG_SECONDS:-120}"
REENCODE="${REENCODE:-1}"
DISCOVER_DEFAULT="${DISCOVER:-1}"        # default discovery behavior if cameras not specified
JITTER_MAX="${JITTER_MAX:-7}"
FIRST_EXT_DELAY_SEC="${FIRST_EXT_DELAY_SEC:-120}"
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"

# ---------- Ensure worker exists ----------
sed -i "s/\r$//" /app/diag_v7_2.sh || true
chmod +x /app/diag_v7_2.sh

# ---------- Scheduler log ----------
SCHED_LOG="/recordings/diag/scheduler.log"
mkdir -p /recordings/diag
log() { echo "[$(TZ="$SCHED_TZ" date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$SCHED_LOG"; }

stop_now=0
trap 'stop_now=1; log "[v8] received stop signal; exiting after current cycle"' INT TERM

# ---------- Helpers ----------
next_run_epoch_simple() {
  # compute next run at SCHED_DAILY_HHMM in SCHED_TZ
  local now_ts today_ts next_ts
  now_ts="$(date -u +%s)"
  today_ts="$(TZ="$SCHED_TZ" date -d "$(TZ=$SCHED_TZ date +%Y-%m-%d) ${SCHED_DAILY_HHMM}" +%s)"
  if [ "$today_ts" -gt "$now_ts" ]; then
    echo "$today_ts"
  else
    next_ts="$(TZ="$SCHED_TZ" date -d "$(TZ=$SCHED_TZ date -d 'tomorrow' +%Y-%m-%d) ${SCHED_DAILY_HHMM}" +%s)"
    echo "$next_ts"
  fi
}

load_config() {
  CONFIG=""
  case "${CONFIG_SOURCE:-}" in
    supabase)
      : "${GYM_ID:?set GYM_ID for supabase}"
      : "${SUPABASE_URL:?set SUPABASE_URL for supabase}"
      : "${SUPABASE_ANON_KEY:?set SUPABASE_ANON_KEY for supabase}"
      # Expect a view or table that returns a single row with { config: <json> }
      # Adjust the path/query to your schema as needed.
      local resp
      resp="$(curl -s \
        -H "apikey: $SUPABASE_ANON_KEY" \
        -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
        "${SUPABASE_URL}/rest/v1/gym_config_view?gym_id=eq.${GYM_ID}&select=config&limit=1")"
      CONFIG="$(echo "$resp" | jq -r '.[0].config // empty' 2>/dev/null || true)"
      ;;
    file)
      : "${CONFIG_PATH:?set CONFIG_PATH}"
      CONFIG="$(cat "$CONFIG_PATH")"
      ;;
    env)
      CONFIG="$SCHEDULE_JSON"
      ;;
    ""|*)
      CONFIG=""
      ;;
  esac

  if [ -z "${CONFIG:-}" ]; then
    USE_SIMPLE_SCHEDULE=1
    log "[v8] no advanced config available; using simple daily schedule at ${SCHED_DAILY_HHMM} ${SCHED_TZ}"
  else
    USE_SIMPLE_SCHEDULE=0
    log "[v8] loaded advanced config from ${CONFIG_SOURCE}"
  fi
}

# Returns 0 if found window; sets NEXT_START_EPOCH, NEXT_END_EPOCH, WIN_CAM_ARG, WIN_DISCOVER
compute_next_window_advanced() {
  local tz now today_iso best_start=0 best_end=0
  tz="$(echo "$CONFIG" | jq -r '.timezone // "America/New_York"')"
  now="$(date -u +%s)"
  today_iso="$(TZ="$tz" date +%Y-%m-%d)"

  # Consider today + next 6 days
  for offset in 0 1 2 3 4 5 6; do
    local date_iso="$(TZ="$tz" date -d "$today_iso +$offset day" +%Y-%m-%d)"
    local day_name="$(TZ="$tz" date -d "$date_iso" +%a)"

    # Iterate all schedules that include this weekday
    local count
    count="$(echo "$CONFIG" | jq -r --arg d "$day_name" '.schedules | map(select(.days[]? == $d)) | length')"
    [ "$count" -eq 0 ] && continue

    for idx in $(echo "$CONFIG" | jq -r --arg d "$day_name" '.schedules | to_entries[] | select(.value.days[]? == $d) | .key'); do
      local start end s_epoch e_epoch
      start="$(echo "$CONFIG" | jq -r ".schedules[$idx].start // empty")"
      end="$(echo "$CONFIG" | jq -r ".schedules[$idx].end // empty")"
      [ -z "$start" ] || [ -z "$end" ] && continue

      s_epoch="$(TZ="$tz" date -d "$date_iso $start" +%s)"
      e_epoch="$(TZ="$tz" date -d "$date_iso $end" +%s)"
      [ "$e_epoch" -le "$s_epoch" ] && continue
      [ "$s_epoch" -le "$now" ] && continue  # skip past windows today

      if [ "$best_start" -eq 0 ] || [ "$s_epoch" -lt "$best_start" ]; then
        best_start="$s_epoch"; best_end="$e_epoch"
      fi
    done
  done

  if [ "$best_start" -eq 0 ]; then
    return 1
  fi

  NEXT_START_EPOCH="$best_start"
  NEXT_END_EPOCH="$best_end"

  # Choose cameras for this window:
  # If top-level .cameras exists, pass CAMS explicitly; else let v7_2 DISCOVER.
  if echo "$CONFIG" | jq -e '.cameras|length>0' >/dev/null 2>&1; then
    WIN_CAM_ARG="$(echo "$CONFIG" | jq -r '
      .cameras | map((.label // "cam") + ":" + .devicePath) | join(",")
    ')"
    WIN_DISCOVER=0
  else
    WIN_CAM_ARG=""
    WIN_DISCOVER="$DISCOVER_DEFAULT"
  fi
  return 0
}

run_window_advanced() {
  local tz dsec
  tz="$(echo "$CONFIG" | jq -r '.timezone // "America/New_York"')"
  dsec="$(( NEXT_END_EPOCH - NEXT_START_EPOCH ))"
  [ "$dsec" -lt 60 ] && dsec=60
  log "[v8] window: $(TZ="$tz" date -d "@$NEXT_START_EPOCH" '+%Y-%m-%d %H:%M:%S %Z') → $(TZ="$tz" date -d "@$NEXT_END_EPOCH" '+%Y-%m-%d %H:%M:%S %Z') (${dsec}s)"

  TZ="$tz" \
  WINDOW_SECONDS="$dsec" \
  SEG_SECONDS="$SEG_SECONDS" \
  REENCODE="$REENCODE" \
  JITTER_MAX="$JITTER_MAX" \
  FIRST_EXT_DELAY_SEC="$FIRST_EXT_DELAY_SEC" \
  EXT_EARLY_SEC="$EXT_EARLY_SEC" \
  DISCOVER="$WIN_DISCOVER" \
  CAMS="$WIN_CAM_ARG" \
  bash -lc '/app/diag_v7_2.sh' || true

  log "[v8] window completed"
}

run_once_simple_now() {
  # Immediate run for testing simple mode; duration = WINDOW_SECONDS_DEFAULT
  log "[v8] SIMPLE immediate run for ${WINDOW_SECONDS_DEFAULT}s"
  TZ="$SCHED_TZ" \
  WINDOW_SECONDS="$WINDOW_SECONDS_DEFAULT" \
  SEG_SECONDS="$SEG_SECONDS" \
  REENCODE="$REENCODE" \
  DISCOVER="$DISCOVER_DEFAULT" \
  JITTER_MAX="$JITTER_MAX" \
  FIRST_EXT_DELAY_SEC="$FIRST_EXT_DELAY_SEC" \
  EXT_EARLY_SEC="$EXT_EARLY_SEC" \
  bash -lc '/app/diag_v7_2.sh' || true
}

# ---------- Main ----------
load_config

if [ "${USE_SIMPLE_SCHEDULE:-0}" -eq 1 ]; then
  # Simple daily-at-HH:MM loop
  log "[v8] SIMPLE mode: daily at ${SCHED_DAILY_HHMM} ${SCHED_TZ}"
  if [ "${RUN_NOW:-0}" = "1" ]; then
    run_once_simple_now
  fi
  while [ "$stop_now" -eq 0 ]; do
    nr="$(next_run_epoch_simple)"
    now="$(date -u +%s)"
    sleep_s=$(( nr - now ))
    [ "$sleep_s" -lt 0 ] && sleep_s=0
    log "[v8] next simple start at $(TZ="$SCHED_TZ" date -d "@$nr" '+%Y-%m-%d %H:%M:%S %Z') (in ${sleep_s}s)"
    while [ "$sleep_s" -gt 0 ] && [ "$stop_now" -eq 0 ]; do
      chunk=$(( sleep_s > 30 ? 30 : sleep_s ))
      sleep "$chunk" || true
      sleep_s=$(( sleep_s - chunk ))
    done
    [ "$stop_now" -ne 0 ] && break
    run_once_simple_now
  done
  log "[v8] SIMPLE scheduler stopped"
  exit 0
fi

# Advanced config loop
log "[v8] ADVANCED mode enabled"
if [ "${RUN_NOW:-0}" = "1" ]; then
  # If RUN_NOW=1 with advanced mode: run a single immediate window using default duration and DISCOVER
  log "[v8] RUN_NOW=1 → ad-hoc run (${WINDOW_SECONDS_DEFAULT}s) before entering scheduler"
  TZ="$(echo "$CONFIG" | jq -r '.timezone // "America/New_York"')" \
  WINDOW_SECONDS="$WINDOW_SECONDS_DEFAULT" \
  SEG_SECONDS="$SEG_SECONDS" \
  REENCODE="$REENCODE" \
  JITTER_MAX="$JITTER_MAX" \
  FIRST_EXT_DELAY_SEC="$FIRST_EXT_DELAY_SEC" \
  EXT_EARLY_SEC="$EXT_EARLY_SEC" \
  DISCOVER="$DISCOVER_DEFAULT" \
  bash -lc '/app/diag_v7_2.sh' || true
fi

while [ "$stop_now" -eq 0 ]; do
  if compute_next_window_advanced; then
    now="$(date -u +%s)"
    sleep_s=$(( NEXT_START_EPOCH - now ))
    [ "$sleep_s" -lt 0 ] && sleep_s=0
    log "[v8] next window starts in ${sleep_s}s"
    while [ "$sleep_s" -gt 0 ] && [ "$stop_now" -eq 0 ]; do
      chunk=$(( sleep_s > 30 ? 30 : sleep_s ))
      sleep "$chunk" || true
      sleep_s=$(( sleep_s - chunk ))
    done
    [ "$stop_now" -ne 0 ] && break
    run_window_advanced
  else
    log "[v8] no upcoming windows in next 7 days; sleeping 10m then reloading config"
    sleep 600 || true
    load_config
  fi
done

log "[v8] ADVANCED scheduler stopped"
