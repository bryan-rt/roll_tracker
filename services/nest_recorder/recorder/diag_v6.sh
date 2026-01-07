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
BACKOFF=3    # seconds; exponential up to 60s

# ========== helpers ==========
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

cleanup() {
  set +e
  [ -n "$EXT_PID" ]    && kill "$EXT_PID" 2>/dev/null || true
  [ -n "$FFMPEG_PID" ] && kill "$FFMPEG_PID" 2>/dev/null || true
  if [ -n "${STOP_TOKEN:-}" ]; then
    http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/stop.json" -X POST \
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
  local out="$DIAG_DIR/generate_${ATTEMPT}.json"
  local http
  http=$(curl -s -w '%{http_code}' -o "$out" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
  echo "$http" > "$DIAG_DIR/generate_${ATTEMPT}_http.txt"
  if [ "$http" != "200" ]; then
    log "[v6] Generate failed (HTTP=$http)"
    return 1
  fi

  URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "$out")"
  EXT_TOKEN="$(jq -r '.results.streamExtensionToken // empty' "$out")"
  STOP_TOKEN="$(jq -r '.results.streamToken // empty' "$out")"
  if [ -z "$URL" ] || [ -z "$EXT_TOKEN" ] || [ -z "$STOP_TOKEN" ]; then
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
          now="$(date -u +%s)"
          next_sleep=$(( exp_epoch - EXT_EARLY_SEC - now ))
          [ "$next_sleep" -lt 60 ] && next_sleep=60
        else
          next_sleep=240
        fi
      else
        next_sleep=240
      fi
    else
      # brief retry
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

  # Video path
  if [ "$REENCODE" = "1" ]; then
    echo "[v6] REENCODE=1 → libx264 veryfast" | tee -a "$LOG"
    V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
  else
    V_OPTS=(-c:v copy)
  fi
}

start_ffmpeg() {
  local out_tmpl="$DIAG_DIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4"
  log "[v6] recording until $(date -u -d "@$DEADLINE" +%H:%M:%S) in ${SEG_SECONDS}s segments → $out_tmpl"

  # Freshness guard: if URL is old (>60s), re-generate once
  local now age
  now="$(date -u +%s)"
  age=$(( now - $(cat "$DIAG_DIR/generated_at_epoch.txt" 2>/dev/null || echo "$now") ))
  if [ "$age" -gt 60 ]; then
    log "[v6] URL is ${age}s old; regenerating before start"
    generate_stream || log "[v6] pre-start regenerate failed; proceeding with current URL"
  fi

  ffmpeg -hide_banner -loglevel info -nostdin -y \
    -rtsp_transport tcp \
    -use_wallclock_as_timestamps 1 -fflags +genpts+igndts -avoid_negative_ts make_zero \
    -analyzeduration 10M -probesize 10M \
    -i "$URL" \
    -map 0:v:0 -map 0:a:0 \
    "${V_OPTS[@]}" \
    -c:a aac -ar 48000 -ac 1 -b:a 64k \
    -max_muxing_queue_size 1024 \
    -t "$(( DEADLINE - $(date -u +%s) ))" \
    "${SEG_OPTS[@]}" \
    "$out_tmpl" \
    1> "$DIAG_DIR/ffmpeg.stdout" 2> "$DIAG_DIR/ffmpeg.stderr" &
  FFMPEG_PID=$!

  extend_loop & EXT_PID=$!
}

# ========== main ==========
get_access_token

build_ffmpeg_opts

while :; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed"; break; }

  ATTEMPT=$((ATTEMPT+1))
  log "[v6] attempt #$ATTEMPT"

  if ! generate_stream; then
    log "[v6] Generate failed; backoff ${BACKOFF}s"
    sleep "$BACKOFF"
    [ "$BACKOFF" -lt 60 ] && BACKOFF=$(( BACKOFF * 2 ))
    continue
  fi

  start_ffmpeg

  # Wait for ffmpeg to end or window to elapse
  wait "$FFMPEG_PID" || true
  rc=$?
  log "[v6] ffmpeg exited rc=$rc"

  # stop extend loop for this attempt
  [ -n "$EXT_PID" ] && kill "$EXT_PID" 2>/dev/null || true
  EXT_PID=""

  # Done if time is up
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed after attempt #$ATTEMPT"; break; }

  # If ffmpeg ended early, back off and try to recover with a fresh stream
  log "[v6] preparing next attempt (backoff ${BACKOFF}s)"
  sleep "$BACKOFF"
  [ "$BACKOFF" -lt 60 ] && BACKOFF=$(( BACKOFF * 2 ))
done

log "[v6] done. Artifacts in $DIAG_DIR"
