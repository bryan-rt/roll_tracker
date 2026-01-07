#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
: "${SDM_PROJECT_ID:?Set SDM_PROJECT_ID in .env}"
: "${DEVICE_1:?Set DEVICE_1 in .env}"       # e.g. enterprises/xxx/devices/yyy
DURATION_SEC="${DURATION_SEC:-12}"            # short smoke clip
OUTROOT="${OUTROOT:-/recordings/diag}"        # or ./diagnostics if you prefer local
STAMP="$(date +%Y%m%d-%H%M%S)"
WORKDIR="${OUTROOT}/${STAMP}"
mkdir -p "${WORKDIR}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- Auth ----------
# Prefer your existing helper; fall back to env ACCESS_TOKEN if already present
ACCESS_TOKEN="${ACCESS_TOKEN:-}"
if [[ -z "${ACCESS_TOKEN}" ]]; then
  if [[ -x "./get_access_token.sh" ]]; then
    log "Fetching access token via get_access_token.sh"
    ACCESS_TOKEN="$(./get_access_token.sh)"
  else
    log "ERROR: No ACCESS_TOKEN and get_access_token.sh not found/executable."
    exit 1
  fi
fi

# Cache token length for quick sanity check
echo -n "[diag] access_token length: " > "${WORKDIR}/token_len.txt"
echo -n "${#ACCESS_TOKEN}" >> "${WORKDIR}/token_len.txt"
echo >> "${WORKDIR}/token_len.txt"
printf "%s" "${ACCESS_TOKEN}" > "${WORKDIR}/ext_token.txt"

# --- Generate RTSP (fixed) ---
GEN_URL="https://smartdevicemanagement.googleapis.com/v1/${DEVICE_1}:executeCommand"
log "Calling GenerateRtspStream via executeCommand…"
HTTP_CODE="$(curl -sS -w "%{http_code}" -o "${WORKDIR}/generate.json" \
  -X POST "${GEN_URL}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}' || true)"

echo "${HTTP_CODE}" > "${WORKDIR}/generate_http.txt"
if [[ "${HTTP_CODE}" != "200" ]]; then
  log "ERROR: generate returned HTTP ${HTTP_CODE} (see generate.json)"
  exit 2
fi

# Extract from .results.*
RTSP_URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "${WORKDIR}/generate.json")"
EXT_TOKEN="$(jq -r '.results.streamExtensionToken // empty' "${WORKDIR}/generate.json")"
STOP_TOKEN="$(jq -r '.results.streamStopToken   // empty' "${WORKDIR}/generate.json")"
EXPIRES_AT="$(jq -r '.results.expiresAt        // empty' "${WORKDIR}/generate.json")"

[[ -n "${RTSP_URL}" ]] || { log "ERROR: rtspUrl empty in generate.json"; exit 3; }
printf "%s\n" "${RTSP_URL}"  > "${WORKDIR}/rtsp_url.txt"
printf "%s\n" "${STOP_TOKEN}" > "${WORKDIR}/stop_token.txt"
printf "%s\n" "${EXPIRES_AT}" > "${WORKDIR}/expires_at.txt"

# ---------- Probe ----------
log "Running ffprobe (json + stderr)…"
# JSON (machine-readable)
ffprobe -v error -print_format json -show_streams -show_format "${RTSP_URL}" \
  > "${WORKDIR}/ffprobe.json" || true
# Human log (stderr)
ffprobe -hide_banner -v info "${RTSP_URL}" \
  1> "${WORKDIR}/ffprobe.stdout" 2> "${WORKDIR}/ffprobe.stderr" || true

# ---------- Record (stable MP4) ----------
# Notes:
# - force TCP
# - generate sane PTS/DTS for live
# - copy video (H.264) but re-encode audio to AAC (tiny bitrate)
# - short DURATION_SEC clip
# - faststart/fragment flags for robust moov placement
OUT_MP4="${WORKDIR}/smoke_v2.mp4"
log "Recording ${DURATION_SEC}s to ${OUT_MP4}…"

ffmpeg -hide_banner -loglevel info \
  -rtsp_transport tcp \
  -use_wallclock_as_timestamps 1 -fflags +genpts -avoid_negative_ts make_zero -reset_timestamps 1 \
  -stimeout 15000000 -analyzeduration 10M -probesize 10M \
  -i "${RTSP_URL}" \
  -map 0:v:0 -map 0:a:0 -c:v copy -c:a aac -b:a 48k \
  -t "${DURATION_SEC}" \
  -movflags +faststart+frag_keyframe+empty_moov \
  "${OUT_MP4}" \
  1> "${WORKDIR}/ffmpeg.stdout" 2> "${WORKDIR}/ffmpeg.stderr" || true

# ---------- Simple pass/fail summary ----------
SIZE="$(stat -f%z "${OUT_MP4}" 2>/dev/null || stat -c%s "${OUT_MP4}" 2>/dev/null || echo 0)"
VID_KIB="$(grep -Eo 'video:[0-9]+kB' "${WORKDIR}/ffmpeg.stderr" | head -1 || true)"
AUD_KIB="$(grep -Eo 'audio:[0-9]+kB' "${WORKDIR}/ffmpeg.stderr" | head -1 || true)"
log "Output size: ${SIZE} bytes | ${VID_KIB:-video:NA} | ${AUD_KIB:-audio:NA}"
log "Done. Artifacts in: ${WORKDIR}"
