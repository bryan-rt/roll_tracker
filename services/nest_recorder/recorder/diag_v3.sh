#!/usr/bin/env bash
set -euo pipefail

# One-camera, one-shot diagnostic. No scheduler, no parallelism.
# Writes logs to /recordings/diag/<timestamp> (host sees ./data/recordings/diag/<timestamp>)

TS="$(date +%Y%m%d-%H%M%S)"
DIAG_DIR="/recordings/diag/$TS"
mkdir -p "$DIAG_DIR"
echo "[diag] writing to $DIAG_DIR"

# 1) Show env inside the container
echo "[diag] ENV (filtered) -> $DIAG_DIR/env.txt"
env | sort | egrep -E '^(TZ|SDM_|DEVICE_1|CAM_ID_1|WINDOW_MINUTES|PRE_ROLL_SECONDS|SEG_SECONDS|PATH)=' \
  > "$DIAG_DIR/env.txt" || true
cat "$DIAG_DIR/env.txt"

# 2) Fetch access token
echo "[diag] get access token"
ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
AT_LEN=$(printf "%s" "$ACCESS_TOKEN" | wc -c | tr -d ' ')
echo "[diag] access_token length: $AT_LEN" | tee "$DIAG_DIR/token_len.txt"

# 3) Generate RTSP (SDM executeCommand)
echo "[diag] GenerateRtspStream"
HTTP=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/generate.json" \
  -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE_1}:executeCommand" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
echo "$HTTP" > "$DIAG_DIR/generate_http.txt"

# Extract URL/tokens
jq -r '.results.streamUrls.rtspUrl // empty' "$DIAG_DIR/generate.json" > "$DIAG_DIR/rtsp_url.txt"
jq -r '.results.streamExtensionToken // empty' "$DIAG_DIR/generate.json" > "$DIAG_DIR/ext_token.txt"
jq -r '.results.streamToken       // empty' "$DIAG_DIR/generate.json" > "$DIAG_DIR/stop_token.txt"

URL=$(cat "$DIAG_DIR/rtsp_url.txt" || true)
EXT=$(cat "$DIAG_DIR/ext_token.txt" || true)
STOP=$(cat "$DIAG_DIR/stop_token.txt" || true)

# helper for re-generating token if probe fails
regen_and_update_url() {
  echo "[diag] regenerating RTSP URL…"
  HTTP2=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/generate2.json" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE_1}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
  echo "$HTTP2" > "$DIAG_DIR/generate2_http.txt"
  if [[ "$HTTP2" != "200" ]]; then
    echo "[diag] ERROR: second generate failed: HTTP=$HTTP2"
    return 1
  fi
  jq -r '.results.streamUrls.rtspUrl // empty' "$DIAG_DIR/generate2.json" > "$DIAG_DIR/rtsp_url.txt"
  jq -r '.results.streamExtensionToken // empty' "$DIAG_DIR/generate2.json" > "$DIAG_DIR/ext_token2.txt"
  jq -r '.results.streamToken       // empty' "$DIAG_DIR/generate2.json" > "$DIAG_DIR/stop_token2.txt"
  URL=$(head -n1 "$DIAG_DIR/rtsp_url.txt")
  echo "[diag] new URL_len=$(printf "%s" "$URL" | wc -c | tr -d ' ')"
  return 0
}

ULEN=$(printf "%s" "$URL" | wc -c | tr -d ' ')
echo "[diag] HTTP=$HTTP  URL_len=$ULEN  EXT_len=$(printf "%s" "$EXT" | wc -c)  STOP_len=$(printf "%s" "$STOP" | wc -c)"

# Bail early if generate failed
if [[ "$HTTP" != "200" || -z "$URL" ]]; then
  echo "[diag] ERROR: generate failed or rtspUrl empty; see $DIAG_DIR/generate.json"
  exit 2
fi

# 4) Record a short, stable MP4 (12s)
DUR="${DURATION_SEC:-12}"
OUT_MP4="$DIAG_DIR/smoke_v1.mp4"
echo "[diag] recording ${DUR}s to $OUT_MP4 …"

ffmpeg -hide_banner -loglevel info -nostdin -y \
  -rtsp_transport tcp \
  -use_wallclock_as_timestamps 1 -fflags +genpts+igndts -avoid_negative_ts make_zero \
  -analyzeduration 10M -probesize 10M \
  -i "$URL" \
  -map 0:v:0 -map 0:a:0 \
  -c:v copy \
  -c:a aac -ar 48000 -ac 1 -b:a 64k \
  -max_muxing_queue_size 1024 \
  -t "${DURATION_SEC:-12}" \
  -movflags +faststart+frag_keyframe+empty_moov \
  "$DIAG_DIR/smoke_v1.mp4" \
  1> "$DIAG_DIR/ffmpeg.stdout" 2> "$DIAG_DIR/ffmpeg.stderr" || true

# Summarize result
SIZE="$(stat -c%s "$OUT_MP4" 2>/dev/null || stat -f%z "$OUT_MP4" 2>/dev/null || echo 0)"
VID_KIB="$(grep -Eo 'video:[0-9]+kB' "$DIAG_DIR/ffmpeg.stderr" | head -1 || true)"
AUD_KIB="$(grep -Eo 'audio:[0-9]+kB' "$DIAG_DIR/ffmpeg.stderr" | head -1 || true)"
echo "[diag] Output size: ${SIZE} bytes | ${VID_KIB:-video:NA} | ${AUD_KIB:-audio:NA}"
echo "[diag] done. Artifacts in: $DIAG_DIR"
