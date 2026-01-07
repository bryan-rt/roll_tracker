#!/usr/bin/env bash
set -euo pipefail

# One-camera, one-shot diagnostic. No scheduler, no parallelism.
# Writes logs to ./data/diagnostics/<timestamp>/

TS="$(date +%Y%m%d-%H%M%S)"
DIAG_DIR="./data/diagnostics/$TS"
mkdir -p "$DIAG_DIR"
echo "[diag] writing to $DIAG_DIR"

# 1) Show env as seen *inside* the container
echo "[diag] ENV (filtered) -> $DIAG_DIR/env.txt"
env | sort | egrep -E '^(TZ|SDM_|DEVICE_1|CAM_ID_1|WINDOW_MINUTES|PRE_ROLL_SECONDS|SEG_SECONDS|PATH)=' \
  > "$DIAG_DIR/env.txt" || true
cat "$DIAG_DIR/env.txt"

# 2) Fetch access token and show length (sanity check)
echo "[diag] get access token"
ACCESS_TOKEN="$(
  docker compose run --rm --no-TTY \
    --entrypoint bash recorder -lc '/app/get_access_token.sh' | tr -d '\r'
)"
AT_LEN=$(printf "%s" "$ACCESS_TOKEN" | wc -c | tr -d ' ')
echo "[diag] access_token length: $AT_LEN" | tee "$DIAG_DIR/token_len.txt"

# 3) Call GenerateRtspStream (capture HTTP + body)
echo "[diag] GenerateRtspStream"
docker compose run --rm --no-TTY \
  --entrypoint bash recorder -lc "
    set -euo pipefail
    ACCESS_TOKEN=\"$ACCESS_TOKEN\"
    RES=/tmp/gen.json
    HTTP=\$(curl -s -w '%{http_code}' -o \"\$RES\" -X POST \
      \"https://smartdevicemanagement.googleapis.com/v1/\$DEVICE_1:executeCommand\" \
      -H \"Authorization: Bearer \$ACCESS_TOKEN\" \
      -H \"Content-Type: application/json\" \
      -d '{\"command\":\"sdm.devices.commands.CameraLiveStream.GenerateRtspStream\",\"params\":{}}')
    mkdir -p /recordings/diag/$TS
    cp \"\$RES\" /recordings/diag/$TS/generate.json
    echo \"\$HTTP\" > /recordings/diag/$TS/generate_http.txt
    jq -r '.results.streamUrls.rtspUrl // empty' \"\$RES\" > /recordings/diag/$TS/rtsp_url.txt
    jq -r '.results.streamExtensionToken // empty' \"\$RES\" > /recordings/diag/$TS/ext_token.txt
    jq -r '.results.streamStopToken   // empty' \"\$RES\" > /recordings/diag/$TS/stop_token.txt
  " >/dev/null

# 4) Pull results from the container
cp -a ./data/recordings/diag/"$TS"/* "$DIAG_DIR"/ 2>/dev/null || true

HTTP=$(cat "$DIAG_DIR/generate_http.txt" 2>/dev/null || echo "")
URL=$(cat "$DIAG_DIR/rtsp_url.txt" 2>/dev/null || echo "")
EXT=$(cat "$DIAG_DIR/ext_token.txt" 2>/dev/null || echo "")
STOP=$(cat "$DIAG_DIR/stop_token.txt" 2>/dev/null || echo "")
ULEN=$(printf "%s" "$URL" | wc -c | tr -d ' ')

echo "[diag] Generate HTTP: $HTTP"
echo "[diag] RTSP URL length: $ULEN"
echo "[diag] EXT token present? $([[ -n "$EXT" ]] && echo yes || echo no)"
echo "[diag] STOP token present? $([[ -n "$STOP" ]] && echo yes || echo no)"

if [[ "$HTTP" != "200" || "$ULEN" -lt 10 ]]; then
  echo "[diag] STOP: Generate failed or URL empty. See $DIAG_DIR/generate.json"
  exit 1
fi

# 5) Immediate ffprobe (print first error line if any)
echo "[diag] ffprobe the URL now"
docker compose run --rm --no-TTY \
  --entrypoint bash recorder -lc "
    set -euo pipefail
    URL=\"$(printf "%s" "$URL")\"
    mkdir -p /recordings/diag/$TS
    ffprobe -hide_banner -loglevel info -rtsp_transport tcp -i \"\$URL\" \
      1> /recordings/diag/$TS/ffprobe.stdout \
      2> /recordings/diag/$TS/ffprobe.stderr || true
  " >/dev/null
cp -a ./data/recordings/diag/"$TS"/ffprobe.* "$DIAG_DIR"/ 2>/dev/null || true
echo "[diag] ffprobe stderr (head):"
sed -n '1,40p' "$DIAG_DIR/ffprobe.stderr" || true

# 6) If ffprobe saw streams, try a 10s ffmpeg copy
echo "[diag] attempting 10s recording to $DIAG_DIR/smoke.mp4"
docker compose run --rm --no-TTY \
  --entrypoint bash recorder -lc "
    set -euo pipefail
    URL=\"$(printf "%s" "$URL")\"
    ffmpeg -hide_banner -loglevel info -rtsp_transport tcp -i \"\$URL\" -t 10 -c copy \
      /recordings/diag/$TS/smoke.mp4 \
      1> /recordings/diag/$TS/ffmpeg.stdout \
      2> /recordings/diag/$TS/ffmpeg.stderr || true
  " >/dev/null
cp -a ./data/recordings/diag/"$TS"/ffmpeg.* "$DIAG_DIR"/ 2>/dev/null || true
cp -a ./data/recordings/diag/"$TS"/smoke.mp4 "$DIAG_DIR"/ 2>/dev/null || true

ls -lh "$DIAG_DIR" | sed 's/^/[diag] /'
echo "[diag] done."