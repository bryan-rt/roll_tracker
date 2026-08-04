#!/usr/bin/env bash
set -euo pipefail

# ========== config (env overridable) ==========
SEG_SECONDS="${SEG_SECONDS:-120}"           # segment length
WINDOW_SECONDS="${WINDOW_SECONDS:-900}"     # total wall-clock window (default 15 min)
FIRST_EXT_DELAY="${FIRST_EXT_DELAY_SEC:-120}"  # first extend ~2 min after start
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"         # schedule next extend at (expiresAt - this)
CAM_ID="${CAM_ID_1:-cam1}"
DEVICE="${DEVICE_1:?missing DEVICE_1 env}"
REENCODE="${REENCODE:-1}"                   # 1 = libx264 (robust), 0 = copy
SOURCE_PTS="${SOURCE_PTS:-1}"               # 1 = preserve camera capture timestamps (no wallclock override)
FPS_PASSTHROUGH="${FPS_PASSTHROUGH:-1}"    # 1 = VFR passthrough in re-encode mode (no CFR resampling)

# RTSP read timeout: if no data arrives for this many seconds, ffmpeg exits.
# Prevents ffmpeg from blocking on a dead stream for minutes (OS TCP timeout).
# 10s is safe (150x the normal inter-frame gap of ~67ms at 15fps).
RTSP_TIMEOUT_SEC="${RTSP_TIMEOUT_SEC:-10}"
RTSP_TIMEOUT_US=$(( RTSP_TIMEOUT_SEC * 1000000 ))

# --- SDM API quota-aware backoff ---
# ExecuteDeviceCommand: 10 QPM per project per user (shared across ALL cameras).
# Per-device (CAMERA): 30 QPM / 100 QPH. Per-command per-device: 5 QPM.
# Binding constraint: 10 QPM user-project.
# Source: developers.google.com/nest/device-access/project/limits
SDM_USER_QPM=10
N_CAMERAS="${N_CAMERAS:-3}"

# Compute minimum retry interval dynamically from quota + camera count.
# Target ~60-70% of quota to leave headroom for extends/stops/retried-429s.
CALLS_PER_MIN_BUDGET=$(( (SDM_USER_QPM * 7 / 10) / N_CAMERAS ))  # 70% of quota / N
[ "$CALLS_PER_MIN_BUDGET" -lt 1 ] && CALLS_PER_MIN_BUDGET=1
MIN_RETRY_INTERVAL=$(( 60 / CALLS_PER_MIN_BUDGET ))  # seconds between API calls

# Backoff tuning (sized to respect per-camera share of 10 QPM)
BACKOFF_INITIAL=8              # seconds; first retry after a quick failure
BACKOFF_QUICK_MAX=25           # cap for transient/unknown failures (~2.4 calls/min)
BACKOFF_404=15                 # RTSP relay lockout
BACKOFF_404_MAX=30             # cap for persistent RTSP 404
BACKOFF_429=60                 # SDM rate limit — start high, escalate
BACKOFF_429_MAX=300            # 5 min cap
HEALTHY_RUN_THRESHOLD=60       # seconds — longer = "healthy run", reconnect immediately
REUSE_FAIL_THRESHOLD=5         # seconds — if reuse attempt dies faster, fall through
DEVICE_404_MAX_RETRIES=3       # give up on persistent device-not-found
CONSECUTIVE_FAIL_ESCALATE=5    # after this many consecutive failures, escalate to slow poll
CONSECUTIVE_FAIL_BACKOFF=120   # slow-poll backoff (2 min)
CONSECUTIVE_FAIL_BACKOFF_MAX=300  # slow-poll cap (5 min)
JITTER_MAX_SEC=5               # random jitter added to every backoff

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
if [ -z "${DIAG_DIR:-}" ]; then
  DIAG_DIR="/recordings/diag/$TS"
fi
mkdir -p "$DIAG_DIR"
LOG="$DIAG_DIR/run.log"
echo "[v6] writing to $DIAG_DIR" | tee -a "$LOG"

# ========== globals ==========
ACCESS_TOKEN="" URL="" EXT_TOKEN="" STOP_TOKEN=""
FFMPEG_PID="" EXT_PID=""
START_EPOCH="$(date -u +%s)"
DEADLINE="$(( START_EPOCH + WINDOW_SECONDS ))"
ATTEMPT=0
BACKOFF=$BACKOFF_INITIAL
SIDECAR_PIDS=()   # background sidecar extraction PIDs

# Session state
NEED_NEW_SESSION=true   # first iteration must generate
SESSION_DEAD=true       # conservative start — forces generate on first attempt
WAS_REUSE=false         # tracks whether current attempt reused an existing URL
CONSECUTIVE_FAILURES=0  # consecutive failed attempts (reset on success)
CONSECUTIVE_DEVICE_404=0  # consecutive device-not-found from generate
GENERATE_FAIL_TYPE=""   # set by generate_stream on failure: "429", "device_404", "other"

# ========== helpers ==========
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

jittered_sleep() {
  local base="$1"
  local jitter=$(( RANDOM % (JITTER_MAX_SEC + 1) ))
  local total=$(( base + jitter ))
  # Enforce minimum retry interval (quota-aware)
  [ "$total" -lt "$MIN_RETRY_INTERVAL" ] && total="$MIN_RETRY_INTERVAL"
  sleep "$total"
}

stop_stream() {
  # Best-effort stop of the current RTSP stream session at the relay.
  # Only called when we believe the session is still ALIVE and we're
  # deliberately abandoning it. Never called for dead sessions.
  if [ -z "${STOP_TOKEN:-}" ]; then return 0; fi
  local http
  http=$(curl -s -w '%{http_code}' --max-time 5 \
    -o "$DIAG_DIR/stop_attempt_${ATTEMPT}.json" -X POST \
    "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$STOP_TOKEN\"}}")
  log "[v6] stop_stream HTTP=$http (attempt=$ATTEMPT)"
  STOP_TOKEN=""
  EXT_TOKEN=""
}

cleanup() {
  set +e
  [ -n "$EXT_PID" ]    && kill "$EXT_PID" 2>/dev/null || true
  [ -n "$FFMPEG_PID" ] && kill "$FFMPEG_PID" 2>/dev/null || true
  # Wait for any background sidecar extractions
  for pid in "${SIDECAR_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  # Final stop — session may still be alive at window end
  if [ -n "${STOP_TOKEN:-}" ]; then
    local http
    http=$(curl -s -w '%{http_code}' --max-time 5 \
      -o "$DIAG_DIR/stop.json" -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$STOP_TOKEN\"}}")
    echo "[v6] stop HTTP=$http (ignored if not 200)" | tee -a "$DIAG_DIR/stop_http.txt"
  fi
}
trap cleanup EXIT INT TERM

get_access_token() {
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  printf "%s" "$ACCESS_TOKEN" | wc -c | tr -d ' ' > "$DIAG_DIR/token_len.txt"
}

generate_stream() {
  GENERATE_FAIL_TYPE=""
  local out="$DIAG_DIR/generate_${ATTEMPT}.json"
  local headers="$DIAG_DIR/generate_${ATTEMPT}_headers.txt"
  local http
  http=$(curl -s -w '%{http_code}' -D "$headers" -o "$out" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
  echo "$http" > "$DIAG_DIR/generate_${ATTEMPT}_http.txt"

  # On 401: refresh token and retry once
  if [ "$http" = "401" ]; then
    log "[v6] Generate got 401; refreshing access token"
    get_access_token
    http=$(curl -s -w '%{http_code}' -D "$headers" -o "$out" \
      -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
    echo "$http (after refresh)" >> "$DIAG_DIR/generate_${ATTEMPT}_http.txt"
  fi

  if [ "$http" = "429" ]; then
    GENERATE_FAIL_TYPE="429"
    # Check for Retry-After header
    local retry_after
    retry_after=$(grep -i 'Retry-After' "$headers" 2>/dev/null | head -1 | tr -d '\r' | awk '{print $2}')
    if [ -n "$retry_after" ] && [ "$retry_after" -gt 0 ] 2>/dev/null; then
      log "[v6] Generate 429 rate-limited (Retry-After: ${retry_after}s)"
      BACKOFF_429_OVERRIDE="$retry_after"
    else
      log "[v6] Generate 429 rate-limited (no Retry-After header)"
      BACKOFF_429_OVERRIDE=""
    fi
    return 1
  fi

  if [ "$http" = "404" ]; then
    GENERATE_FAIL_TYPE="device_404"
    log "[v6] Generate 404 — device not found"
    return 1
  fi

  if [ "$http" != "200" ]; then
    GENERATE_FAIL_TYPE="other"
    log "[v6] Generate failed (HTTP=$http)"
    return 1
  fi

  URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "$out")"
  EXT_TOKEN="$(jq -r '.results.streamExtensionToken // empty' "$out")"
  STOP_TOKEN="$(jq -r '.results.streamToken // empty' "$out")"
  if [ -z "$URL" ] || [ -z "$EXT_TOKEN" ] || [ -z "$STOP_TOKEN" ]; then
    GENERATE_FAIL_TYPE="other"
    log "[v6] Generate missing fields (url/ext/stop)"
    return 1
  fi

  printf "%s\n" "$URL"       > "$DIAG_DIR/rtsp_url.txt"
  printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"
  printf "%s\n" "$STOP_TOKEN"> "$DIAG_DIR/stop_token.txt"
  date -u +%s > "$DIAG_DIR/generated_at_epoch.txt"
  log "[v6] Generated RTSP and tokens (attempt=$ATTEMPT)"
}

extend_loop() {
  local next_sleep="$FIRST_EXT_DELAY"
  while kill -0 "$FFMPEG_PID" 2>/dev/null; do
    sleep "$next_sleep"
    [ ! -e "/proc/$FFMPEG_PID" ] && break

    local stamp http jf new_ext new_stop exp_iso exp_epoch now
    stamp="$(date -u +%s)"

    # Refresh access token before extending (cheap — cache hit unless expired)
    get_access_token

    http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}.json" \
      -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
    echo "$stamp $http" >> "$DIAG_DIR/extend_http.txt"

    if [ "$http" = "200" ]; then
      jf="$DIAG_DIR/extend_${stamp}.json"
      new_ext="$(jq -r '.results.streamExtensionToken // empty' "$jf" 2>/dev/null || true)"
      [ -n "$new_ext" ] && EXT_TOKEN="$new_ext" && printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"

      new_stop="$(jq -r '.results.streamToken // empty' "$jf" 2>/dev/null || true)"
      [ -n "$new_stop" ] && STOP_TOKEN="$new_stop" && printf "%s\n" "$STOP_TOKEN" > "$DIAG_DIR/stop_token.txt"

      exp_iso="$(jq -r '.results.expiresAt // empty' "$jf" 2>/dev/null || true)"
      if [ -n "$exp_iso" ]; then
        exp_epoch=$(date -u -d "$exp_iso" +%s 2>/dev/null || echo "")
        if [ -n "$exp_epoch" ]; then
          # Publish expiry for main loop's reuse decision
          echo "$exp_epoch" > "$DIAG_DIR/_extend_expiry.txt"
          now="$(date -u +%s)"
          next_sleep=$(( exp_epoch - EXT_EARLY_SEC - now ))
          [ "$next_sleep" -lt 60 ] && next_sleep=60
        else
          next_sleep=240
        fi
      else
        next_sleep=240
      fi
    elif [ "$http" = "429" ]; then
      # Rate limited — back off longer before retrying extend
      log "[v6] extend got 429; backing off 60s"
      next_sleep=60
    else
      # On failure (including 401): refresh token, brief pause, retry once
      get_access_token
      sleep 3
      stamp="$(date -u +%s)"
      http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}_retry.json" \
        -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
      echo "$stamp $http (retry)" >> "$DIAG_DIR/extend_http.txt"
      [ "$http" = "200" ] || { log "[v6] extend failing; stopping extend loop"; break; }
      next_sleep=240
    fi
  done
}

build_ffmpeg_opts() {
  # Segment muxer options (array-safe)
  SEG_OPTS=(-f segment -segment_time "$SEG_SECONDS" -strftime 1 -movflags +faststart)
  if ffmpeg -hide_banner -h muxer=segment 2>&1 | grep -qi 'reset_timestamps'; then
    echo "[v6] segment muxer supports -reset_timestamps 1" | tee -a "$LOG"
    SEG_OPTS+=(-reset_timestamps 1)
  else
    echo "[v6] segment muxer lacks -reset_timestamps; proceeding" | tee -a "$LOG"
  fi

  # Defaults — overridden per mode below
  VF_OPTS=()
  FPS_MODE_OPTS=()
  INPUT_FFLAGS="+genpts+igndts"
  INPUT_WALLCLOCK=(-use_wallclock_as_timestamps 1)
  COPYTS_OPTS=()

  # RTSP socket timeout — exits ffmpeg when data stops arriving.
  # Without this, ffmpeg blocks on dead streams for minutes (OS TCP timeout).
  # The -timeout option sets the RTSP socket I/O timeout in microseconds.
  RTSP_TIMEOUT_OPTS=(-timeout "$RTSP_TIMEOUT_US")

  # SOURCE_PTS: preserve camera's own RTP capture timestamps instead of
  # substituting bursty network-arrival times. Proven in CAPTURE-TIME-1:
  # source PTS = uniform 33ms/67ms true capture cadence.
  if [ "$SOURCE_PTS" = "1" ]; then
    echo "[v6] SOURCE_PTS=1 → preserving camera capture timestamps (-copyts)" | tee -a "$LOG"
    INPUT_WALLCLOCK=()              # remove -use_wallclock_as_timestamps 1
    INPUT_FFLAGS="+igndts"          # drop +genpts — keep source PTS
    COPYTS_OPTS=(-copyts)           # preserve original timestamps
  fi

  # Guard: FPS_PASSTHROUGH=1 without SOURCE_PTS=1 would pass bursty arrival timestamps
  # through unresampled — worse than either default alone.
  # SOURCE_PTS defaults to 1 everywhere (CP-R3), so a non-1 value here is always an
  # explicit operator choice. Honour it: disable passthrough rather than overriding.
  if [ "$FPS_PASSTHROUGH" = "1" ] && [ "$SOURCE_PTS" != "1" ]; then
    echo "[v6] ⚠ SOURCE_PTS=$SOURCE_PTS explicit → disabling FPS_PASSTHROUGH (CFR path)" | tee -a "$LOG"
    FPS_PASSTHROUGH="0"
  fi

  # Video path: REENCODE=1|2 (CFR + timing sidecar), 0 (VFR passthrough)
  if [ "$REENCODE" = "1" ] || [ "$REENCODE" = "2" ]; then
    V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
    VF_OPTS=(-vf showinfo)
    if [ "$FPS_PASSTHROUGH" = "1" ]; then
      echo "[v6] REENCODE=$REENCODE + FPS_PASSTHROUGH=1 → libx264 veryfast + VFR passthrough + timing sidecar" | tee -a "$LOG"
      FPS_MODE_OPTS=(-fps_mode passthrough)
    else
      echo "[v6] REENCODE=$REENCODE → libx264 veryfast + timing sidecar (CFR)" | tee -a "$LOG"
    fi
  elif [ "$REENCODE" = "0" ]; then
    echo "[v6] REENCODE=0 → stream copy (VFR passthrough)" | tee -a "$LOG"
    V_OPTS=(-c:v copy)
    INPUT_FFLAGS="+igndts"
    FPS_MODE_OPTS=(-fps_mode passthrough)
  else
    echo "[v6] REENCODE=$REENCODE unknown, falling back to CFR + timing sidecar" | tee -a "$LOG"
    V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
    VF_OPTS=(-vf showinfo)
  fi

  echo "[v6] RTSP socket timeout: ${RTSP_TIMEOUT_SEC}s" | tee -a "$LOG"
  echo "[v6] API budget: ${SDM_USER_QPM} QPM / ${N_CAMERAS} cameras → ${CALLS_PER_MIN_BUDGET} calls/min, min interval ${MIN_RETRY_INTERVAL}s" | tee -a "$LOG"
  echo "[v6] timing config: SOURCE_PTS=$SOURCE_PTS REENCODE=$REENCODE FPS_PASSTHROUGH=$FPS_PASSTHROUGH" | tee -a "$LOG"
}

extract_timing_sidecars() {
  # Post-process ffmpeg stderr to produce per-segment .timing.jsonl sidecars.
  # Each sidecar has one row per OUTPUT frame (keyed on frame_index matching
  # FrameIterator's cap.read() counter).
  # When SOURCE_PTS=1: includes host_arrival_s, lower-envelope offset, drift.
  # When SOURCE_PTS=0: nearest-neighbor two-pointer mapping (arrival-PTS).
  local stderr="${1:-$DIAG_DIR/ffmpeg.stderr}"
  [ ! -f "$stderr" ] && return 0
  grep -q 'Parsed_showinfo.*pts_time:' "$stderr" || return 0

  # Step 1: Find segment mp4s and their opening-line positions in stderr
  local -a seg_paths seg_lines seg_epochs
  while IFS= read -r gline; do
    local lineno path base ymd_hms epoch=0
    lineno="${gline%%:*}"
    path=$(echo "$gline" | sed "s/.*Opening '//;s/' for writing.*//")
    base=$(basename "$path" .mp4)
    ymd_hms=$(echo "$base" | grep -oE '[0-9]{8}-[0-9]{6}')
    if [ -n "$ymd_hms" ]; then
      epoch=$(date -d "${ymd_hms:0:4}-${ymd_hms:4:2}-${ymd_hms:6:2} ${ymd_hms:9:2}:${ymd_hms:11:2}:${ymd_hms:13:2}" +%s 2>/dev/null || echo 0)
    fi
    seg_paths+=("$path")
    seg_lines+=("$lineno")
    seg_epochs+=("$epoch")
  done < <(grep -n "Opening.*\.mp4.*for writing" "$stderr")

  [ "${#seg_paths[@]}" -eq 0 ] && return 0
  local total_stderr_lines
  total_stderr_lines=$(wc -l < "$stderr")

  # Extract timebase from showinfo config line (once per attempt, before segments)
  local timebase=0
  local tb_line
  tb_line=$(grep 'config in time_base:' "$stderr" | head -1)
  if [ -n "$tb_line" ]; then
    timebase=$(echo "$tb_line" | sed -n 's/.*time_base: *1\/\([0-9][0-9]*\).*/\1/p')
  fi
  if [ "$timebase" -gt 0 ] 2>/dev/null; then
    log "[v6] sidecar: timebase=1/$timebase (from showinfo config)"
  else
    timebase=90000
    log "[v6] ⚠ sidecar: timebase not found in showinfo config; fallback to 1/$timebase"
  fi

  # Step 2: For each segment, extract showinfo PTS ticks, get output info, build sidecar
  for (( si=0; si<${#seg_paths[@]}; si++ )); do
    local seg_path="${seg_paths[$si]}"
    local from_line="${seg_lines[$si]}"
    local epoch="${seg_epochs[$si]}"
    local to_line="$total_stderr_lines"
    if (( si + 1 < ${#seg_lines[@]} )); then
      to_line="${seg_lines[$((si+1))]}"
    fi

    local sidecar="${seg_path%.mp4}.timing.jsonl"
    local pairs_tmp="$DIAG_DIR/_pairs_${ATTEMPT}_${si}.tmp"

    # Extract integer PTS ticks (not pts_time which truncates to 3 decimals).
    # Z1: regex requires at least one digit after optional sign; no trailing space required.
    if [ "$SOURCE_PTS" = "1" ]; then
      sed -n "${from_line},${to_line}p" "$stderr" \
        | grep 'Parsed_showinfo.*pts:' \
        | sed -n 's/^\([0-9.]*\) .*pts: *\(-\?[0-9][0-9]*\).*/\1 \2/p' \
        > "$pairs_tmp"
    else
      sed -n "${from_line},${to_line}p" "$stderr" \
        | grep 'Parsed_showinfo.*pts:' \
        | sed -n 's/.*pts: *\(-\?[0-9][0-9]*\).*/0 \1/p' \
        > "$pairs_tmp"
    fi

    local input_count
    input_count=$(wc -l < "$pairs_tmp" | tr -d ' ')

    # Z2: verify regex matched every showinfo line — detect silent frame drops
    local showinfo_count
    showinfo_count=$(sed -n "${from_line},${to_line}p" "$stderr" \
      | grep -c 'Parsed_showinfo.*pts:' || true)
    if [ "$input_count" -ne "$showinfo_count" ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — regex matched $input_count of $showinfo_count showinfo lines"
    fi

    if [ "$input_count" -eq 0 ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — no showinfo data, skipping"
      rm -f "$pairs_tmp"
      continue
    fi

    local output_count=0 output_fps=0
    if [ -f "$seg_path" ]; then
      output_count=$(ffprobe -hide_banner -select_streams v:0 \
        -show_entries stream=nb_frames -of csv=p=0 "$seg_path" 2>/dev/null | tr -d ' ')
      output_fps=$(ffprobe -hide_banner -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "$seg_path" 2>/dev/null | tr -d ' ')
    fi
    [ -z "$output_count" ] || [ "$output_count" = "N/A" ] && output_count=0
    [ -z "$output_fps" ] && output_fps="0/1"

    if [ "$output_count" -eq 0 ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — cannot read output frame count, skipping"
      rm -f "$pairs_tmp"
      continue
    fi

    local mismatch="false"
    [ "$input_count" -ne "$output_count" ] && mismatch="true"

    local use_source_pts="$SOURCE_PTS"
    local fps_passthrough_mode="${FPS_PASSTHROUGH:-1}"

    if [ "$fps_passthrough_mode" = "1" ]; then
    # Passthrough: 1:1 mapping, no CFR grid construction.
    # Separate awk to keep the CFR path structurally untouched.
    # Input: pairs_file has (host_arrival, pts_ticks) per line.
    # All PTS arithmetic in integer ticks; convert to seconds after base subtraction.
    awk -v output_count="$output_count" \
        -v epoch="$epoch" \
        -v mismatch="$mismatch" \
        -v pairs_file="$pairs_tmp" \
        -v source_pts_mode="$use_source_pts" \
        -v attempt="$ATTEMPT" \
        -v timebase="$timebase" \
    '
    BEGIN {
      # Read (host_arrival, pts_ticks) pairs
      ni = 0
      while ((getline line < pairs_file) > 0) {
        split(line, parts, " ")
        host_arr[ni] = parts[1] + 0.0
        raw_ticks[ni] = parts[2] + 0    # integer ticks
        ni++
      }
      close(pairs_file)

      # Sort by ticks
      for (a = 1; a < ni; a++) {
        kt = raw_ticks[a]; kh = host_arr[a]
        b = a - 1
        while (b >= 0 && raw_ticks[b] > kt) {
          raw_ticks[b+1] = raw_ticks[b]; host_arr[b+1] = host_arr[b]
          b--
        }
        raw_ticks[b+1] = kt; host_arr[b+1] = kh
      }

      # Integer base subtraction FIRST (exact), then convert to seconds
      base_ticks = (ni > 0) ? raw_ticks[0] : 0
      for (k = 0; k < ni; k++) {
        raw_pts[k] = (raw_ticks[k] - base_ticks) / timebase
      }

      # Tick deltas (integer exact — awk doubles hold integers exactly up to 2^53;
      # at 90000 ticks/sec a 65-min session is ~351M ticks, far below 2^53)
      nd = 0
      for (k = 1; k < ni; k++) {
        tick_deltas[nd] = raw_ticks[k] - raw_ticks[k-1]
        nd++
      }

      # Sort tick deltas (insertion sort)
      for (a = 1; a < nd; a++) {
        kv = tick_deltas[a]
        b = a - 1
        while (b >= 0 && tick_deltas[b] > kv) {
          tick_deltas[b+1] = tick_deltas[b]
          b--
        }
        tick_deltas[b+1] = kv
      }

      # Median tick delta
      if (nd > 0) {
        if (nd % 2 == 1) median_tick = tick_deltas[int(nd/2)]
        else median_tick = (tick_deltas[nd/2 - 1] + tick_deltas[nd/2]) / 2.0
      } else median_tick = 0

      # Trimmed mean: discard deltas outside [0.5×, 1.5×] median (Z3)
      trim_sum = 0; trim_n = 0
      lo_cutoff = median_tick * 0.5
      hi_cutoff = median_tick * 1.5
      for (k = 0; k < nd; k++) {
        if (tick_deltas[k] >= lo_cutoff && tick_deltas[k] <= hi_cutoff) {
          trim_sum += tick_deltas[k]
          trim_n++
        }
      }
      trimmed_mean_tick = (trim_n > 0) ? trim_sum / trim_n : median_tick

      # Mean tick delta
      tick_sum = 0
      for (k = 0; k < nd; k++) tick_sum += tick_deltas[k]
      mean_tick = (nd > 0) ? tick_sum / nd : 0

      # FPS from each method
      measured_fps = (trimmed_mean_tick > 0) ? timebase / trimmed_mean_tick : 0
      measured_fps_median = (median_tick > 0) ? timebase / median_tick : 0
      measured_fps_mean = 0
      if (ni > 1 && raw_pts[ni-1] > 0) {
        measured_fps_mean = (ni - 1) / raw_pts[ni-1]
      }

      # PTS delta stats (from ticks, converted to ms)
      sum_d = 0; sum_d2 = 0
      for (k = 0; k < nd; k++) {
        d_ms = tick_deltas[k] * 1000.0 / timebase
        sum_d += d_ms; sum_d2 += d_ms * d_ms
      }
      mean_d = (nd > 0) ? sum_d / nd : 0
      stdev_d = (nd > 0) ? sqrt(sum_d2/nd - mean_d*mean_d) : 0

      # Drift (operates on raw_pts seconds + host_arr, same as before)
      global_min_offset = 0; drift_rate = 0; drift_flat = "true"; drift_ppm = 0
      n_windows = 0

      if (source_pts_mode == "1" && ni > 0) {
        global_min_offset = 1e18
        for (k = 0; k < ni; k++) {
          off = host_arr[k] - raw_pts[k]
          if (off < global_min_offset) global_min_offset = off
        }

        win_size = 10.0
        for (w = 0; w < 60 && ni > 0; w++) {
          ws = w * win_size; we = (w + 1) * win_size
          if (ws >= raw_pts[ni-1]) break
          wmin = 1e18; wmid = (ws + we) / 2.0; found = 0
          for (k = 0; k < ni; k++) {
            if (raw_pts[k] >= ws && raw_pts[k] < we) {
              off = host_arr[k] - raw_pts[k]
              if (off < wmin) wmin = off
              found = 1
            }
          }
          if (found) {
            win_off[n_windows] = wmin; win_mid[n_windows] = wmid
            n_windows++
          }
        }

        if (n_windows >= 2) {
          sx=0;sy=0;sxx=0;sxy=0
          for (w=0;w<n_windows;w++) {
            sx+=win_mid[w]; sy+=win_off[w]
            sxx+=win_mid[w]*win_mid[w]; sxy+=win_mid[w]*win_off[w]
          }
          den = n_windows*sxx - sx*sx
          if (den != 0) drift_rate = (n_windows*sxy - sx*sy) / den
        }
        drift_ppm = drift_rate * 1e6
        drift_flat = (n_windows<2 || (drift_rate>-0.0001 && drift_rate<0.0001)) ? "true" : "false"
      }

      # _meta — passthrough mode
      if (source_pts_mode == "1") {
        printf "{\"_meta\":true,\"sidecar_schema\":2,\"timing_mode\":\"passthrough\",\"pts_origin\":\"segment_relative\",\"fps_method\":\"trimmed_mean\",\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"measured_fps\":%.4f,\"measured_fps_median\":%.4f,\"measured_fps_mean\":%.4f,\"pts_timebase\":%d,\"pts_tick_delta_median\":%.1f,\"pts_tick_delta_mean\":%.1f,\"pts_delta_trim_kept\":%d,\"pts_delta_trim_total\":%d,\"mismatch\":%s,\"pts_wallclock_offset_s\":%.6f,\"offset_method\":\"lower_envelope\",\"drift_rate_s_per_s\":%.9f,\"drift_flat\":%s,\"drift_ppm\":%.3f,\"n_drift_windows\":%d,\"pts_mean_delta_ms\":%.4f,\"pts_stdev_delta_ms\":%.4f}\n", \
          epoch, attempt, ni, output_count, measured_fps, measured_fps_median, measured_fps_mean, \
          timebase, median_tick, mean_tick, trim_n, nd, mismatch, \
          global_min_offset, drift_rate, drift_flat, drift_ppm, n_windows, mean_d, stdev_d
      } else {
        printf "{\"_meta\":true,\"sidecar_schema\":2,\"timing_mode\":\"passthrough\",\"pts_origin\":\"segment_relative\",\"fps_method\":\"trimmed_mean\",\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"measured_fps\":%.4f,\"measured_fps_median\":%.4f,\"measured_fps_mean\":%.4f,\"pts_timebase\":%d,\"pts_tick_delta_median\":%.1f,\"pts_tick_delta_mean\":%.1f,\"pts_delta_trim_kept\":%d,\"pts_delta_trim_total\":%d,\"mismatch\":%s}\n", \
          epoch, attempt, ni, output_count, measured_fps, measured_fps_median, measured_fps_mean, \
          timebase, median_tick, mean_tick, trim_n, nd, mismatch
      }

      # 1:1 mapping — each output frame IS the input frame
      for (i = 0; i < ni; i++) {
        if (source_pts_mode == "1") {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"host_arrival_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[i], host_arr[i], i
        } else {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[i], i
        }
      }
    }
    ' > "$sidecar"

    else
    # CFR grid: nearest-neighbour mapping of uniform output grid to input PTS.
    # Input: pairs_file has (host_arrival, pts_ticks) per line.
    # Same tick-based precision as passthrough; CFR grid structure unchanged.
    # measured_fps here describes INPUT capture cadence, not the output grid rate
    # (which is output_fps from ffprobe). Same semantics as before, better precision.
    awk -v output_count="$output_count" \
        -v output_fps="$output_fps" \
        -v epoch="$epoch" \
        -v input_count="$input_count" \
        -v mismatch="$mismatch" \
        -v pairs_file="$pairs_tmp" \
        -v source_pts_mode="$use_source_pts" \
        -v attempt="$ATTEMPT" \
        -v timebase="$timebase" \
    '
    BEGIN {
      # Read (host_arrival, pts_ticks) pairs
      ni = 0
      while ((getline line < pairs_file) > 0) {
        split(line, parts, " ")
        host_arr[ni] = parts[1] + 0.0
        raw_ticks[ni] = parts[2] + 0    # integer ticks
        ni++
      }
      close(pairs_file)

      # Sort by ticks
      for (a = 1; a < ni; a++) {
        kt = raw_ticks[a]; kh = host_arr[a]
        b = a - 1
        while (b >= 0 && raw_ticks[b] > kt) {
          raw_ticks[b+1] = raw_ticks[b]; host_arr[b+1] = host_arr[b]
          b--
        }
        raw_ticks[b+1] = kt; host_arr[b+1] = kh
      }

      # Integer base subtraction FIRST (exact), then convert to seconds
      base_ticks = (ni > 0) ? raw_ticks[0] : 0
      for (k = 0; k < ni; k++) {
        raw_pts[k] = (raw_ticks[k] - base_ticks) / timebase
      }

      # Output grid interval from ffprobe r_frame_rate
      if (index(output_fps, "/") > 0) {
        split(output_fps, fp, "/")
        fps_val = fp[1] / fp[2]
      } else {
        fps_val = output_fps + 0.0
      }
      if (fps_val <= 0) fps_val = 30.0
      interval = 1.0 / fps_val

      # Tick deltas (integer exact)
      nd = 0
      for (k = 1; k < ni; k++) {
        tick_deltas[nd] = raw_ticks[k] - raw_ticks[k-1]
        nd++
      }

      # Sort tick deltas (insertion sort)
      for (a = 1; a < nd; a++) {
        kv = tick_deltas[a]
        b = a - 1
        while (b >= 0 && tick_deltas[b] > kv) {
          tick_deltas[b+1] = tick_deltas[b]
          b--
        }
        tick_deltas[b+1] = kv
      }

      # Median tick delta
      if (nd > 0) {
        if (nd % 2 == 1) median_tick = tick_deltas[int(nd/2)]
        else median_tick = (tick_deltas[nd/2 - 1] + tick_deltas[nd/2]) / 2.0
      } else median_tick = 0

      # Trimmed mean: discard deltas outside [0.5×, 1.5×] median (Z3)
      trim_sum = 0; trim_n = 0
      lo_cutoff = median_tick * 0.5
      hi_cutoff = median_tick * 1.5
      for (k = 0; k < nd; k++) {
        if (tick_deltas[k] >= lo_cutoff && tick_deltas[k] <= hi_cutoff) {
          trim_sum += tick_deltas[k]
          trim_n++
        }
      }
      trimmed_mean_tick = (trim_n > 0) ? trim_sum / trim_n : median_tick

      # Mean tick delta
      tick_sum = 0
      for (k = 0; k < nd; k++) tick_sum += tick_deltas[k]
      mean_tick = (nd > 0) ? tick_sum / nd : 0

      # FPS: measured_fps = trimmed mean of INPUT capture cadence (not output grid)
      measured_fps = (trimmed_mean_tick > 0) ? timebase / trimmed_mean_tick : 0
      measured_fps_median = (median_tick > 0) ? timebase / median_tick : 0
      measured_fps_mean = 0
      if (ni > 1 && raw_pts[ni-1] > 0) {
        measured_fps_mean = (ni - 1) / raw_pts[ni-1]
      }

      # PTS delta stats (from ticks, converted to ms)
      sum_d = 0; sum_d2 = 0
      for (k = 0; k < nd; k++) {
        d_ms = tick_deltas[k] * 1000.0 / timebase
        sum_d += d_ms; sum_d2 += d_ms * d_ms
      }
      mean_d = (nd > 0) ? sum_d / nd : 0
      stdev_d = (nd > 0) ? sqrt(sum_d2/nd - mean_d*mean_d) : 0

      # Drift (operates on raw_pts seconds + host_arr)
      global_min_offset = 0; drift_rate = 0; drift_flat = "true"; drift_ppm = 0
      n_windows = 0

      if (source_pts_mode == "1" && ni > 0) {
        global_min_offset = 1e18
        for (k = 0; k < ni; k++) {
          off = host_arr[k] - raw_pts[k]
          if (off < global_min_offset) global_min_offset = off
        }

        win_size = 10.0
        for (w = 0; w < 60 && ni > 0; w++) {
          ws = w * win_size; we = (w + 1) * win_size
          if (ws >= raw_pts[ni-1]) break
          wmin = 1e18; wmid = (ws + we) / 2.0; found = 0
          for (k = 0; k < ni; k++) {
            if (raw_pts[k] >= ws && raw_pts[k] < we) {
              off = host_arr[k] - raw_pts[k]
              if (off < wmin) wmin = off
              found = 1
            }
          }
          if (found) {
            win_off[n_windows] = wmin; win_mid[n_windows] = wmid
            n_windows++
          }
        }

        if (n_windows >= 2) {
          sx=0;sy=0;sxx=0;sxy=0
          for (w=0;w<n_windows;w++) {
            sx+=win_mid[w]; sy+=win_off[w]
            sxx+=win_mid[w]*win_mid[w]; sxy+=win_mid[w]*win_off[w]
          }
          den = n_windows*sxx - sx*sx
          if (den != 0) drift_rate = (n_windows*sxy - sx*sy) / den
        }
        drift_ppm = drift_rate * 1e6
        drift_flat = (n_windows<2 || (drift_rate>-0.0001 && drift_rate<0.0001)) ? "true" : "false"
      }

      if (source_pts_mode == "1") {
        printf "{\"_meta\":true,\"sidecar_schema\":2,\"timing_mode\":\"cfr_grid\",\"pts_origin\":\"segment_relative\",\"fps_method\":\"trimmed_mean\",\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"measured_fps\":%.4f,\"measured_fps_median\":%.4f,\"measured_fps_mean\":%.4f,\"pts_timebase\":%d,\"pts_tick_delta_median\":%.1f,\"pts_tick_delta_mean\":%.1f,\"pts_delta_trim_kept\":%d,\"pts_delta_trim_total\":%d,\"mismatch\":%s,\"pts_wallclock_offset_s\":%.6f,\"offset_method\":\"lower_envelope\",\"drift_rate_s_per_s\":%.9f,\"drift_flat\":%s,\"drift_ppm\":%.3f,\"n_drift_windows\":%d,\"pts_mean_delta_ms\":%.4f,\"pts_stdev_delta_ms\":%.4f}\n", \
          epoch, attempt, ni, output_count, fps_val, measured_fps, measured_fps_median, measured_fps_mean, \
          timebase, median_tick, mean_tick, trim_n, nd, mismatch, \
          global_min_offset, drift_rate, drift_flat, drift_ppm, n_windows, mean_d, stdev_d
      } else {
        printf "{\"_meta\":true,\"sidecar_schema\":2,\"timing_mode\":\"cfr_grid\",\"pts_origin\":\"segment_relative\",\"fps_method\":\"trimmed_mean\",\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"measured_fps\":%.4f,\"measured_fps_median\":%.4f,\"measured_fps_mean\":%.4f,\"pts_timebase\":%d,\"pts_tick_delta_median\":%.1f,\"pts_tick_delta_mean\":%.1f,\"pts_delta_trim_kept\":%d,\"pts_delta_trim_total\":%d,\"mismatch\":%s}\n", \
          epoch, attempt, ni, output_count, fps_val, measured_fps, measured_fps_median, measured_fps_mean, \
          timebase, median_tick, mean_tick, trim_n, nd, mismatch
      }

      j = 0
      for (i = 0; i < output_count; i++) {
        t_out = i * interval
        while (j + 1 < ni) {
          d_cur = raw_pts[j] - t_out; if (d_cur < 0) d_cur = -d_cur
          d_nxt = raw_pts[j+1] - t_out; if (d_nxt < 0) d_nxt = -d_nxt
          if (d_nxt <= d_cur) j++
          else break
        }
        if (source_pts_mode == "1") {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"host_arrival_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[j], host_arr[j], j
        } else {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[j], j
        }
      }
    }
    ' > "$sidecar"

    fi

    rm -f "$pairs_tmp"

    local sidecar_lines
    sidecar_lines=$(( $(wc -l < "$sidecar") - 1 ))
    local fps_info=""
    fps_info=$(head -1 "$sidecar" | grep -oE '"measured_fps":[0-9.]+' | cut -d: -f2)

    if [ "$mismatch" = "true" ]; then
      log "[v6] ⚠ MISMATCH sidecar: $(basename "$sidecar") input=$input_count output=$output_count fps=${fps_info:-?}"
    else
      log "[v6] sidecar: $(basename "$sidecar") $sidecar_lines/$output_count ✓ fps=${fps_info:-?} (epoch=$epoch)"
    fi
  done
}

classify_failure() {
  # Classify ffmpeg exit for backoff + session-state decisions.
  # Sets FAILURE_TYPE and SESSION_DEAD.
  # Uses Parsed_showinfo as the reliable indicator of whether frame data was
  # received (validated: never-connected attempts have 0 showinfo lines;
  # Opening lines are unreliable — segment muxer may open a file early).
  local stderr_file="$1"
  local run_duration="$2"

  if [ "$run_duration" -ge "$HEALTHY_RUN_THRESHOLD" ]; then
    FAILURE_TYPE="healthy"
    SESSION_DEAD=false
    return
  fi

  local got_data=false
  if [ -f "$stderr_file" ] && grep -q 'Parsed_showinfo' "$stderr_file" 2>/dev/null; then
    got_data=true
  fi

  if [ -f "$stderr_file" ]; then
    if grep -q '404.*Not Found\|404Not Found\|session.*invalidated' "$stderr_file" 2>/dev/null; then
      FAILURE_TYPE="session_dead"
      SESSION_DEAD=true
      return
    fi
    if grep -q 'Connection timed out\|Connection refused\|Network is unreachable' "$stderr_file" 2>/dev/null; then
      if [ "$got_data" = "true" ]; then
        # Data was flowing, then connection died — mid-stream failure
        FAILURE_TYPE="connect_fail"
      else
        # Never got any data — session never established
        FAILURE_TYPE="session_dead"
      fi
      SESSION_DEAD=true
      return
    fi
  fi

  # Unknown quick failure
  FAILURE_TYPE="quick"
  SESSION_DEAD=true
}

start_ffmpeg() {
  mkdir -p "$DIAG_DIR"
  local out_tmpl="$DIAG_DIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4"
  log "[v6] recording until $(date -u -d "@$DEADLINE" +%H:%M:%S) in ${SEG_SECONDS}s segments → $out_tmpl"

  # Per-attempt stderr file (supports retry loop — each attempt gets its own file)
  local stderr_file="$DIAG_DIR/ffmpeg_attempt_${ATTEMPT}.stderr"
  TS_PID=""

  if [ "$SOURCE_PTS" = "1" ]; then
    # Host-arrival timestamping: pipe stderr through fifo, prepend $EPOCHREALTIME
    local stderr_fifo="$DIAG_DIR/_stderr_fifo_${ATTEMPT}"
    mkfifo "$stderr_fifo"
    while IFS= read -r line; do
      printf "%s %s\n" "$EPOCHREALTIME" "$line"
    done < "$stderr_fifo" > "$stderr_file" &
    TS_PID=$!
    local stderr_target="$stderr_fifo"
    log "[v6] attempt $ATTEMPT: source-PTS mode, host-arrival timestamping → $(basename "$stderr_file")"
  else
    local stderr_target="$stderr_file"
  fi

  # Record attempt metadata
  local ffmpeg_start_epoch="$EPOCHREALTIME"
  printf '{"attempt":%d,"generate_epoch":"%s","ffmpeg_start_epoch":"%s","reuse":%s}\n' \
    "$ATTEMPT" "$(cat "$DIAG_DIR/generated_at_epoch.txt" 2>/dev/null || echo 0)" \
    "$ffmpeg_start_epoch" \
    "$WAS_REUSE" \
    >> "$DIAG_DIR/attempt_log.jsonl"

  ffmpeg -hide_banner -loglevel info -nostdin -y \
    -rtsp_transport tcp \
    "${RTSP_TIMEOUT_OPTS[@]}" \
    "${COPYTS_OPTS[@]}" \
    "${INPUT_WALLCLOCK[@]}" -fflags "$INPUT_FFLAGS" -avoid_negative_ts make_zero \
    -analyzeduration 10M -probesize 10M \
    -i "$URL" \
    -map 0:v:0 -map 0:a:0 \
    "${VF_OPTS[@]}" \
    "${FPS_MODE_OPTS[@]}" \
    "${V_OPTS[@]}" \
    -c:a aac -ar 48000 -ac 1 -b:a 64k \
    -max_muxing_queue_size 1024 \
    -t "$(( DEADLINE - $(date -u +%s) ))" \
    "${SEG_OPTS[@]}" \
    "$out_tmpl" \
    1> "$DIAG_DIR/ffmpeg.stdout" 2> "$stderr_target" &
  FFMPEG_PID=$!
  FFMPEG_START_EPOCH_INT="${ffmpeg_start_epoch%%.*}"

  extend_loop & EXT_PID=$!
}

# ========== main ==========
get_access_token

build_ffmpeg_opts

# Initialize extend expiry file
echo "0" > "$DIAG_DIR/_extend_expiry.txt"

while :; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed"; break; }

  ATTEMPT=$((ATTEMPT+1))

  if [ "$NEED_NEW_SESSION" = "true" ]; then
    log "[v6] attempt #$ATTEMPT"

    # Refresh access token (cheap — cache hit unless expired)
    get_access_token

    # Stop the previous session ONLY if we believe it's still alive.
    # Dead sessions return 400 ("stream_token invalid") — confirmed in
    # RECORDER-RELIABILITY-1 test. Skipping saves an API call.
    if [ "$SESSION_DEAD" = "false" ]; then
      stop_stream
    fi

    if ! generate_stream; then
      # Handle generate failure by type
      case "$GENERATE_FAIL_TYPE" in
        429)
          backoff_val="${BACKOFF_429_OVERRIDE:-$BACKOFF_429}"
          [ "$backoff_val" -lt "$BACKOFF_429" ] && backoff_val="$BACKOFF_429"
          log "[v6] 429 rate-limited; backoff ${backoff_val}s + jitter"
          jittered_sleep "$backoff_val"
          [ "$BACKOFF_429" -lt "$BACKOFF_429_MAX" ] && BACKOFF_429=$(( BACKOFF_429 * 2 ))
          [ "$BACKOFF_429" -gt "$BACKOFF_429_MAX" ] && BACKOFF_429=$BACKOFF_429_MAX
          ;;
        device_404)
          CONSECUTIVE_DEVICE_404=$(( CONSECUTIVE_DEVICE_404 + 1 ))
          if [ "$CONSECUTIVE_DEVICE_404" -ge "$DEVICE_404_MAX_RETRIES" ]; then
            log "[v6] device not found after $CONSECUTIVE_DEVICE_404 attempts; giving up"
            break
          fi
          log "[v6] device 404 ($CONSECUTIVE_DEVICE_404/$DEVICE_404_MAX_RETRIES); backoff ${BACKOFF}s + jitter"
          jittered_sleep "$BACKOFF"
          ;;
        *)
          log "[v6] Generate failed; backoff ${BACKOFF}s + jitter"
          jittered_sleep "$BACKOFF"
          [ "$BACKOFF" -lt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$(( BACKOFF * 2 ))
          [ "$BACKOFF" -gt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$BACKOFF_QUICK_MAX
          ;;
      esac
      CONSECUTIVE_FAILURES=$(( CONSECUTIVE_FAILURES + 1 ))
      continue
    fi

    # Generate succeeded — reset counters
    CONSECUTIVE_DEVICE_404=0
    NEED_NEW_SESSION=false
    WAS_REUSE=false
  else
    log "[v6] attempt #$ATTEMPT (reusing URL, 0 API calls)"
    WAS_REUSE=true
  fi

  start_ffmpeg

  # Wait for ffmpeg to end or window to elapse
  wait "$FFMPEG_PID" || true
  rc=$?
  local_now="$(date -u +%s)"
  run_duration=$(( local_now - FFMPEG_START_EPOCH_INT ))
  log "[v6] ffmpeg exited rc=$rc after ${run_duration}s"

  # stop extend loop for this attempt
  [ -n "$EXT_PID" ] && kill "$EXT_PID" 2>/dev/null || true
  EXT_PID=""

  # Clean up stderr fifo + timestamper for this attempt
  if [ -n "${TS_PID:-}" ] && [ "$TS_PID" != "" ]; then
    wait "$TS_PID" 2>/dev/null || true
    TS_PID=""
  fi
  rm -f "$DIAG_DIR/_stderr_fifo_${ATTEMPT}"

  # Extract per-segment timing sidecars in BACKGROUND (off the critical path)
  extract_timing_sidecars "$DIAG_DIR/ffmpeg_attempt_${ATTEMPT}.stderr" &
  SIDECAR_PIDS+=($!)

  # Done if time is up
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed after attempt #$ATTEMPT"; break; }

  # Classify the failure and update session state
  classify_failure "$DIAG_DIR/ffmpeg_attempt_${ATTEMPT}.stderr" "$run_duration"

  # Track consecutive failures for escalation
  if [ "$FAILURE_TYPE" = "healthy" ]; then
    CONSECUTIVE_FAILURES=0
    BACKOFF=$BACKOFF_INITIAL
  else
    CONSECUTIVE_FAILURES=$(( CONSECUTIVE_FAILURES + 1 ))
  fi

  # Decide: reuse existing URL or get a new session?
  case "$FAILURE_TYPE" in
    healthy)
      # Check if the session's extend expiry is still in the future
      extend_expiry=$(cat "$DIAG_DIR/_extend_expiry.txt" 2>/dev/null || echo 0)
      local_now="$(date -u +%s)"
      if [ "$extend_expiry" -gt "$local_now" ] && [ "$SESSION_DEAD" = "false" ]; then
        NEED_NEW_SESSION=false
        log "[v6] healthy run (${run_duration}s); session still valid (expires $(date -u -d "@$extend_expiry" +%H:%M:%S)); reusing URL"
      else
        NEED_NEW_SESSION=true
        log "[v6] healthy run (${run_duration}s); session expired; reconnecting immediately"
      fi
      ;;
    session_dead)
      NEED_NEW_SESSION=true
      # Check for escalation on consecutive failures
      if [ "$CONSECUTIVE_FAILURES" -ge "$CONSECUTIVE_FAIL_ESCALATE" ]; then
        esc_backoff=$CONSECUTIVE_FAIL_BACKOFF
        [ "$CONSECUTIVE_FAILURES" -gt $(( CONSECUTIVE_FAIL_ESCALATE + 3 )) ] && esc_backoff=$CONSECUTIVE_FAIL_BACKOFF_MAX
        log "[v6] session dead (${CONSECUTIVE_FAILURES} consecutive failures); slow-polling ${esc_backoff}s + jitter"
        jittered_sleep "$esc_backoff"
      else
        log "[v6] session dead; backoff ${BACKOFF}s + jitter"
        jittered_sleep "$BACKOFF"
        [ "$BACKOFF" -lt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$BACKOFF_QUICK_MAX
      fi
      ;;
    rtsp_404)
      NEED_NEW_SESSION=true
      if [ "$CONSECUTIVE_FAILURES" -ge "$CONSECUTIVE_FAIL_ESCALATE" ]; then
        esc_backoff=$CONSECUTIVE_FAIL_BACKOFF
        [ "$CONSECUTIVE_FAILURES" -gt $(( CONSECUTIVE_FAIL_ESCALATE + 3 )) ] && esc_backoff=$CONSECUTIVE_FAIL_BACKOFF_MAX
        log "[v6] RTSP 404 (${CONSECUTIVE_FAILURES} consecutive failures); slow-polling ${esc_backoff}s + jitter"
        jittered_sleep "$esc_backoff"
      else
        log "[v6] RTSP 404 (relay lockout); backoff ${BACKOFF}s + jitter"
        jittered_sleep "$BACKOFF"
        [ "$BACKOFF" -lt "$BACKOFF_404_MAX" ] && BACKOFF=$(( BACKOFF < BACKOFF_404 ? BACKOFF_404 : BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$BACKOFF_404_MAX" ] && BACKOFF=$BACKOFF_404_MAX
      fi
      ;;
    connect_fail|*)
      NEED_NEW_SESSION=true
      # If this was a reuse attempt that failed quickly, skip backoff — fall through to generate
      if [ "$WAS_REUSE" = "true" ] && [ "$run_duration" -lt "$REUSE_FAIL_THRESHOLD" ]; then
        log "[v6] reuse failed (${run_duration}s); falling through to regenerate"
      elif [ "$CONSECUTIVE_FAILURES" -ge "$CONSECUTIVE_FAIL_ESCALATE" ]; then
        esc_backoff=$CONSECUTIVE_FAIL_BACKOFF
        [ "$CONSECUTIVE_FAILURES" -gt $(( CONSECUTIVE_FAIL_ESCALATE + 3 )) ] && esc_backoff=$CONSECUTIVE_FAIL_BACKOFF_MAX
        log "[v6] failure #${CONSECUTIVE_FAILURES} (${FAILURE_TYPE}, ${run_duration}s); slow-polling ${esc_backoff}s + jitter"
        jittered_sleep "$esc_backoff"
      else
        log "[v6] ${FAILURE_TYPE} (${run_duration}s); backoff ${BACKOFF}s + jitter"
        jittered_sleep "$BACKOFF"
        [ "$BACKOFF" -lt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt "$BACKOFF_QUICK_MAX" ] && BACKOFF=$BACKOFF_QUICK_MAX
      fi
      ;;
  esac
done

# Wait for any background sidecar extractions to finish
for pid in "${SIDECAR_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

log "[v6] done. Artifacts in $DIAG_DIR"
