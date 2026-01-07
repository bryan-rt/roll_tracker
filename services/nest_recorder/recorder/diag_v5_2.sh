#!/usr/bin/env bash
set -euo pipefail

# --- config (env overridable) ---
SEG_SECONDS="${SEG_SECONDS:-120}"         # segment length
WINDOW_SECONDS="${WINDOW_SECONDS:-600}"   # total recording window (~10 min)
# EXT_INTERVAL="${EXT_INTERVAL_SEC:-230}"   # extend cadence (< expiry)
CAM_ID="${CAM_ID_1:-cam1}"
DEVICE="${DEVICE_1:?missing DEVICE_1 env}"

TS="$(date +%Y%m%d-%H%M%S)"
DIAG_DIR="/recordings/diag/$TS"
mkdir -p "$DIAG_DIR"
echo "[v5] writing to $DIAG_DIR"

# --- access token ---
ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
echo "$(printf "%s" "$ACCESS_TOKEN" | wc -c | tr -d ' ')" > "$DIAG_DIR/token_len.txt"

# --- generate RTSP ---
HTTP=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/generate.json" \
  -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
echo "$HTTP" > "$DIAG_DIR/generate_http.txt"
if [ "$HTTP" != "200" ]; then echo "[v5] generate HTTP=$HTTP"; exit 2; fi

URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "$DIAG_DIR/generate.json")"
EXT_TOKEN="$(jq -r '.results.streamExtensionToken // empty' "$DIAG_DIR/generate.json")"
STOP_TOKEN="$(jq -r '.results.streamToken // empty' "$DIAG_DIR/generate.json")"
printf "%s\n" "$URL"       > "$DIAG_DIR/rtsp_url.txt"
printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"
printf "%s\n" "$STOP_TOKEN"> "$DIAG_DIR/stop_token.txt"
date -u +%s > "$DIAG_DIR/generated_at_epoch.txt"

if [ -z "$URL" ] || [ -z "$EXT_TOKEN" ] || [ -z "$STOP_TOKEN" ]; then
  echo "[v5] missing URL/EXT/STOP"; exit 3
fi

# --- housekeeping on exit: stop extend loop, stop stream, kill ffmpeg if alive ---
EXT_PID=""
FFMPEG_PID=""
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
    echo "[v5] stop HTTP=$http (ignored if not 200)" | tee -a "$DIAG_DIR/stop_http.txt"
  fi
}
trap cleanup EXIT INT TERM

# First extend target: ~2 minutes after start (so we're safely < 5-min expiry)
FIRST_EXT_DELAY="${FIRST_EXT_DELAY_SEC:-120}"
# How early before expiry we should extend (seconds)
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"

extend_loop() {
  local n=0
  local next_sleep="$FIRST_EXT_DELAY"
  local last_expire_epoch=""

  while kill -0 "$FFMPEG_PID" 2>/dev/null; do
    sleep "$next_sleep"
    [ ! -e /proc/$FFMPEG_PID ] && break

    local stamp http
    stamp="$(date -u +%s)"

    http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}.json" \
      -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")

    echo "$stamp $http" >> "$DIAG_DIR/extend_http.txt"

    # If successful, adopt refreshed tokens for *next* extend & stop
    if [ "$http" = "200" ]; then
      NEW_EXT=$(jq -r '.results.streamExtensionToken // empty' "$DIAG_DIR/extend_${stamp}.json" 2>/dev/null || true)
      [ -n "$NEW_EXT" ] && EXT_TOKEN="$NEW_EXT" && printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"

      NEW_STOP=$(jq -r '.results.streamToken // empty' "$DIAG_DIR/extend_${stamp}.json" 2>/dev/null || true)
      [ -n "$NEW_STOP" ] && STOP_TOKEN="$NEW_STOP" && printf "%s\n" "$STOP_TOKEN" > "$DIAG_DIR/stop_token.txt"

      # Schedule next extend relative to new expiry (expiresAt - EXT_EARLY_SEC), else fall back to ~4 min
      exp_iso=$(jq -r '.results.expiresAt // empty' "$DIAG_DIR/extend_${stamp}.json" 2>/dev/null || true)
      if [ -n "$exp_iso" ]; then
        # Convert ISO to epoch (UTC)
        exp_epoch=$(date -u -d "$exp_iso" +%s 2>/dev/null || echo "")
        if [ -n "$exp_epoch" ]; then
          last_expire_epoch="$exp_epoch"
          now=$(date -u +%s)
          # Sleep until (expiry - EXT_EARLY_SEC), but not less than 60s
          next_sleep=$(( exp_epoch - EXT_EARLY_SEC - now ))
          [ "$next_sleep" -lt 60 ] && next_sleep=60
        else
          next_sleep=240
        fi
      else
        next_sleep=240
      fi
    else
      # Brief retry once after 3s; if still failing, we’ll stop extending
      sleep 3
      stamp="$(date -u +%s)"
      http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}_retry.json" \
        -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
      echo "$stamp $http (retry)" >> "$DIAG_DIR/extend_http.txt"

      if [ "$http" != "200" ]; then
        echo "[v5] extend failing; no further extends" | tee -a "$DIAG_DIR/extend.stderr"
        break
      else
        # Successful retry: set a safe default until we get an expiresAt
        next_sleep=240
      fi
    fi

    n=$((n+1))
  done
}

# --- record segmented for the whole window ---
OUT_TMPL="$DIAG_DIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4"
echo "[v5] recording ~${WINDOW_SECONDS}s in ${SEG_SECONDS}s segments to $OUT_TMPL"

# --- choose segment options based on ffmpeg support ---
# (Some builds lack -reset_timestamps on the segment muxer.)
# --- choose segment options based on ffmpeg support ---
SEG_OPTS=(-f segment -segment_time "$SEG_SECONDS" -strftime 1 -movflags +faststart)
if ffmpeg -hide_banner -h muxer=segment 2>&1 | grep -qi 'reset_timestamps'; then
  echo "[v5] segment muxer supports -reset_timestamps 1"
  SEG_OPTS+=(-reset_timestamps 1)
else
  echo "[v5] segment muxer DOES NOT support -reset_timestamps; proceeding without it"
fi

# Optional: set REENCODE=1 to normalize GOP/timestamps across all builds
if [ "${REENCODE:-1}" = "1" ]; then
  echo "[v5] REENCODE=1 → using libx264 for clean keyframes"
  V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
else
  V_OPTS=(-c:v copy)
fi

# OUT_TMPL="$DIAG_DIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4"
# echo "[v5] recording ~${WINDOW_SECONDS}s in ${SEG_SECONDS}s segments to $OUT_TMPL"

# ---- FFmpeg call (segmenting) ----
ffmpeg -hide_banner -loglevel info -nostdin -y \
  -rtsp_transport tcp \
  -use_wallclock_as_timestamps 1 -fflags +genpts+igndts -avoid_negative_ts make_zero \
  -analyzeduration 10M -probesize 10M \
  -i "$URL" \
  -map 0:v:0 -map 0:a:0 \
  "${V_OPTS[@]}" \
  -c:a aac -ar 48000 -ac 1 -b:a 64k \
  -max_muxing_queue_size 1024 \
  -t "$WINDOW_SECONDS" \
  "${SEG_OPTS[@]}" \
  "$OUT_TMPL" \
  1> "$DIAG_DIR/ffmpeg.stdout" 2> "$DIAG_DIR/ffmpeg.stderr" &
FFMPEG_PID=$!

# >>> START THE EXTEND LOOP (this was missing) <<<
extend_loop & EXT_PID=$!

wait "$FFMPEG_PID" || true
echo "[v5] ffmpeg exited with code $?"

# If nothing was written, try one quick re-generate + 30s capture (debug safety net)
if ! ls "$DIAG_DIR/${CAM_ID}-"*.mp4 >/dev/null 2>&1; then
  echo "[v5] no segments found; trying one quick regenerate + short capture"
  HTTP2=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/generate_retry.json" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
  echo "$HTTP2" > "$DIAG_DIR/generate_retry_http.txt"
  if [ "$HTTP2" = "200" ]; then
    URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "$DIAG_DIR/generate_retry.json")"
    printf "%s\n" "$URL" > "$DIAG_DIR/rtsp_url_retry.txt"
    ffmpeg -hide_banner -loglevel info -nostdin -y \
      -rtsp_transport tcp \
      -use_wallclock_as_timestamps 1 -fflags +genpts+igndts -avoid_negative_ts make_zero \
      -analyzeduration 10M -probesize 10M \
      -i "$URL" \
      -map 0:v:0 -map 0:a:0 \
      -c:v copy \
      -c:a aac -ar 48000 -ac 1 -b:a 64k \
      -t 30 \
      -movflags +faststart \
      "$DIAG_DIR/${CAM_ID}-retry.mp4" \
      1>> "$DIAG_DIR/ffmpeg.stdout" 2>> "$DIAG_DIR/ffmpeg.stderr" || true
  fi
fi

echo "[v5] done. Artifacts: $DIAG_DIR"
