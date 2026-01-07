#!/usr/bin/env bash
set -euo pipefail
SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
# ---- Config (single definitions)
WINDOW_MINUTES=${WINDOW_MINUTES:-30}
PRE_ROLL_SECONDS=${PRE_ROLL_SECONDS:-30}
SEG_SECONDS=${SEG_SECONDS:-120}
EXTEND_SLEEP=${EXTEND_SLEEP:-230}      # ~3m50s between extends
GEN_RETRIES=${GEN_RETRIES:-6}
GEN_BACKOFF_SEC=${GEN_BACKOFF_SEC:-3}

log() { echo "[$(date '+%F %T')] $*"; }

generate_rtsp() {
  local DEVICE="$1" OUT="$2" attempt=1
  while (( attempt <= GEN_RETRIES )); do
    local ACCESS_TOKEN; ACCESS_TOKEN=$(/app/get_access_token.sh)
    local RES; RES=$(mktemp)
    local HTTP
    HTTP=$(curl -s -w "%{http_code}" -o "$RES" -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/$DEVICE:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')

    cp "$RES" "$OUT/generate_attempt_${attempt}.json"

    local URL EXT STOP
    URL=$(jq -r '.results.streamUrls.rtspUrl // empty' "$RES")
    EXT=$(jq -r '.results.streamExtensionToken // empty' "$RES")
    STOP=$(jq -r '.results.streamStopToken   // empty' "$RES")

    if [[ "$HTTP" == "200" && -n "$URL" && -n "$EXT" && -n "$STOP" ]]; then
      log "[gen] success on attempt $attempt"
      echo "$URL|$EXT|$STOP"
      rm -f "$RES"
      return 0
    fi
    log "[gen] empty/failed (HTTP $HTTP) attempt $attempt → sleep ${GEN_BACKOFF_SEC}s"
    sleep "$GEN_BACKOFF_SEC"
    ((attempt++))
    rm -f "$RES"
  done
  log "[gen] ERROR after $GEN_RETRIES attempts (see $OUT)"
  return 1
}

extend_rtsp() {
  local DEVICE="$1" EXT="$2" OUT="$3"
  [[ -z "$EXT" ]] && { log "[extend] skipped: empty token"; echo ""; return 0; }
  local ACCESS_TOKEN; ACCESS_TOKEN=$(/app/get_access_token.sh)
  local RES; RES=$(curl -s -X POST \
    "https://smartdevicemanagement.googleapis.com/v1/$DEVICE:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT\"}}") || true
  echo "$RES" > "$OUT/extend.json"
  jq -r '.results.streamExtensionToken // empty' <<<"$RES"
}

stop_rtsp() {
  local DEVICE="$1" STOP="$2"
  [[ -z "$STOP" ]] && { log "[stop] skipped: empty token"; return 0; }
  local ACCESS_TOKEN; ACCESS_TOKEN=$(/app/get_access_token.sh)
  curl -s -X POST \
    "https://smartdevicemanagement.googleapis.com/v1/$DEVICE:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamStopToken\":\"$STOP\"}}" \
    >/dev/null 2>&1 || true
}

record_one() {
  local DEVICE="$1" CAM_ID="$2"
  local OUTDIR="/recordings/${CAM_ID}/$(date +%F)/$(date +%H)"
  local LOGDIR="/recordings/logs/${CAM_ID}/$(date +%F)/$(date +%H)"
  mkdir -p "$OUTDIR" "$LOGDIR"

  # tiny jitter per cam to avoid API bursts
  sleep $((RANDOM % 3))

  # pre-roll BEFORE Generate
  [[ "${PRE_ROLL_SECONDS:-0}" -gt 0 ]] && { log "[$CAM_ID] pre-roll ${PRE_ROLL_SECONDS}s"; sleep "$PRE_ROLL_SECONDS"; }

  local END=$(( $(date +%s) + WINDOW_MINUTES*60 ))
  local FFPID=0 STOP_TOKEN="" EXT_TOKEN="" RTSP_URL=""

  while [ "$(date +%s)" -lt "$END" ]; do
    if ! IFS='|' read -r RTSP_URL EXT_TOKEN STOP_TOKEN < <(generate_rtsp "$DEVICE" "$LOGDIR"); then
      log "[$CAM_ID] Generate failed; sleeping ${GEN_BACKOFF_SEC}s then retrying"
      sleep "$GEN_BACKOFF_SEC"
      continue
    fi

    # Guard: never start ffmpeg on empty URL
    local URL_LEN; URL_LEN=$(printf "%s" "$RTSP_URL" | wc -c)
    log "[$CAM_ID] RTSP URL len=$URL_LEN; starting ffmpeg"
    if [[ "$URL_LEN" -lt 10 ]]; then
      log "[$CAM_ID] ERROR: empty/short URL, regenerate"
      sleep "$GEN_BACKOFF_SEC"
      continue
    fi

    # unique log per ffmpeg start
    local FLOG="$LOGDIR/ffmpeg-$(date +%H%M%S).log"
    ffmpeg -hide_banner -loglevel info -rtsp_transport tcp -i "$RTSP_URL" \
      -c copy -f segment -segment_time "$SEG_SECONDS" -reset_timestamps 1 \
      -strftime 1 "$OUTDIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4" \
      2> "$FLOG" &
    FFPID=$!

    while kill -0 "$FFPID" 2>/dev/null && [ "$(date +%s)" -lt "$END" ]; do
      sleep "$EXTEND_SLEEP"
      local NEW_EXT; NEW_EXT=$(extend_rtsp "$DEVICE" "$EXT_TOKEN" "$LOGDIR")
      [[ -n "$NEW_EXT" ]] && EXT_TOKEN="$NEW_EXT"
      kill -0 "$FFPID" 2>/dev/null || { log "[$CAM_ID] ffmpeg exited; regenerate"; break; }
    done

    [ "$(date +%s)" -ge "$END" ] && break
  done

  if kill -0 "$FFPID" 2>/dev/null; then
    kill -INT "$FFPID" 2>/dev/null || true
    wait  "$FFPID" 2>/dev/null || true
  fi
  stop_rtsp "$DEVICE" "$STOP_TOKEN"
  log "[$CAM_ID] done; logs at $LOGDIR, segments in $OUTDIR"
}

# ---- Multi-cam launcher (parallel)
pids=()
for i in 1 2 3 4 5 6; do
  dev_var="DEVICE_$i"; cam_var="CAM_ID_$i"
  dev="${!dev_var-}"; cam="${!cam_var-}"
  [[ -n "$dev" && -n "$cam" ]] || continue
  log "launch $cam"
  record_one "$dev" "$cam" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
