#!/usr/bin/env bash
set -euo pipefail

# ===== Config (env-overridable) =====
# Provide cameras explicitly OR auto-discover from SDM.
# Explicit format (CSV): CAMS="Front Door:enterprises/.../devices/AAA,Garage:enterprises/.../devices/BBB"
CAMS="${CAMS:-}"                      # optional explicit list
DISCOVER="${DISCOVER:-1}"             # 1=discover when CAMS unset; 0=do nothing if CAMS unset
JITTER_MAX="${JITTER_MAX:-7}"         # random stagger per cam start (seconds)

# Common recording knobs passed to each v6 worker:
WINDOW_SECONDS="${WINDOW_SECONDS:-900}"
SEG_SECONDS="${SEG_SECONDS:-120}"
REENCODE="${REENCODE:-1}"
FIRST_EXT_DELAY_SEC="${FIRST_EXT_DELAY_SEC:-120}"
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"

# SDM project for discovery
: "${SDM_PROJECT_ID:?set SDM_PROJECT_ID in env/.env}"

# Root folder for this multi-cam session
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
ROOT="/recordings/diag/${TS}"
mkdir -p "$ROOT"
echo "[v7] root=$ROOT"

# Ensure worker exists and has LF endings + exec bit (no-ops if already fine)
sed -i "s/\r$//" /app/diag_v6.sh || true
chmod +x /app/diag_v6.sh

# ===== Build camera list =====
declare -a CAMS_ARR
if [ -n "$CAMS" ]; then
  # Split CSV on commas without trimming names; users may include spaces in camera names.
  IFS=',' read -r -a CAMS_ARR <<< "$CAMS"
elif [ "$DISCOVER" = "1" ]; then
  # Use your existing token getter; v6 relies on it too. (We only need it for discovery.)
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  DEV_JSON="$ROOT/devices.json"
  curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://smartdevicemanagement.googleapis.com/v1/enterprises/${SDM_PROJECT_ID}/devices" \
    -o "$DEV_JSON"

  # Pull only devices that expose CameraLiveStream; pair "CustomName:DeviceName"
  mapfile -t CAMS_ARR < <(jq -r '
    .devices[]?
    | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
    | (.customName // "cam") + ":" + .name
  ' "$DEV_JSON")
else
  echo "[v7] No CAMS provided and DISCOVER=0 — nothing to do"; exit 0
fi

if [ "${#CAMS_ARR[@]}" -eq 0 ]; then
  echo "[v7] No cameras found."; exit 0
fi

echo "[v7] cameras:"
for pair in "${CAMS_ARR[@]}"; do
  echo "  - $pair"
done

# ===== Per-camera runner (spawns v6) =====
run_one_cam() {
  local pair="$1"

  # Split "Cam Name:enterprises/.../devices/ID"
  local cam="${pair%%:*}"
  local dev="${pair#*:}"

  # Sanitize cam name for folder/filenames
  local cam_id="${cam// /_}"
  cam_id="${cam_id//[^A-Za-z0-9_-]/_}"

  local dir="$ROOT/$cam_id"
  mkdir -p "$dir"
  echo "[v7][$cam_id] starting → $dir"

  # Jitter to avoid API bursts across cams
  sleep $(( RANDOM % (JITTER_MAX + 1) ))

  # Delegate to v6 worker with per-cam env and folder;
  # v6 will honor DIAG_DIR if already set.  :contentReference[oaicite:1]{index=1}
  TZ="${TZ:-America/New_York}" \
  CAM_ID_1="$cam_id" \
  DEVICE_1="$dev" \
  DIAG_DIR="$dir" \
  TS="$TS" \
  WINDOW_SECONDS="$WINDOW_SECONDS" \
  SEG_SECONDS="$SEG_SECONDS" \
  REENCODE="$REENCODE" \
  FIRST_EXT_DELAY_SEC="$FIRST_EXT_DELAY_SEC" \
  EXT_EARLY_SEC="$EXT_EARLY_SEC" \
  bash -lc '/app/diag_v6.sh' || true

  echo "[v7][$cam_id] done."
}

# ===== Fan-out and wait =====
pids=()
for pair in "${CAMS_ARR[@]}"; do
  run_one_cam "$pair" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=$((fail+1)); fi
done

echo "[v7] all cams finished. failures=$fail. root=$ROOT"
exit 0
