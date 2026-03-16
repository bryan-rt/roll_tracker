#!/usr/bin/env bash
set -euo pipefail
SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
if [[ -z "${SDM_PROJECT_ID:-}" && -f "$SECRETS_DIR/project_id.txt" ]]; then
  SDM_PROJECT_ID="$(cat "$SECRETS_DIR/project_id.txt")"
fi
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
: "${SDM_PROJECT_ID:?SDM_PROJECT_ID not set and $SECRETS_DIR/project_id.txt missing}"

# Root folder for this multi-cam session
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
ROOT="/recordings/diag/${TS}"
mkdir -p "$ROOT"
echo "[v7] root=$ROOT"

# Ensure worker exists and has LF endings + exec bit (no-ops if already fine)
sed -i "s/\r$//" /app/diag_v6.sh || true
chmod +x /app/diag_v6.sh

# ===== helpers =====
sanitize_name() {
  # collapse spaces to underscores, strip non [A-Za-z0-9_-]
  local s="${1// /_}"
  s="${s//[^A-Za-z0-9_-]/_}"
  # squeeze multiple underscores
  printf '%s' "$s" | tr -s '_'
}

short_id() {
  # last 6 chars from device id segment
  local devpath="$1"
  local id="${devpath##*/}"        # after last /
  printf '%s' "${id: -6}"
}

# ===== Build camera list =====
# We'll end up with arrays of aligned fields:
# CAM_NAME[i], DEVICE_PATH[i], LABEL[i], OUT_DIR[i]
declare -a CAM_NAME DEVICE_PATH LABEL OUT_DIR

if [ -n "$CAMS" ]; then
  # Split CSV on commas without trimming names; users may include spaces in camera names.
  IFS=',' read -r -a pairs <<< "$CAMS"
  for pair in "${pairs[@]}"; do
    cam="${pair%%:*}"
    dev="${pair#*:}"
    cam="${cam:-cam}"
    CAM_NAME+=("$cam")
    DEVICE_PATH+=("$dev")
  done
elif [ "$DISCOVER" = "1" ]; then
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  DEV_JSON="$ROOT/devices.json"
  curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://smartdevicemanagement.googleapis.com/v1/enterprises/${SDM_PROJECT_ID}/devices" \
    -o "$DEV_JSON"

  # Pull only devices that expose CameraLiveStream
  # Use customName if present; fallback to "cam"
  while IFS=$'\t' read -r name path; do
    CAM_NAME+=("$name")
    DEVICE_PATH+=("$path")
  done < <(jq -r '
    .devices[]?
    | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
    | ((.customName // "cam") + "\t" + .name)
  ' "$DEV_JSON")
else
  echo "[v7] No CAMS provided and DISCOVER=0 — nothing to do"; exit 0
fi

if [ "${#DEVICE_PATH[@]}" -eq 0 ]; then
  echo "[v7] No cameras found."; exit 0
fi

# Build stable per-cam labels and per-cam dirs
declare -A seen_label
for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
  cam="${CAM_NAME[$i]}"
  dev="${DEVICE_PATH[$i]}"

  mat="$(sanitize_name "$cam")"
  sid="$(short_id "$dev")"
  lbl="Cam_${mat}__${sid}"

  # ensure uniqueness even if two cams had same customName and same short suffix (extremely unlikely)
  if [[ -n "${seen_label[$lbl]:-}" ]]; then
    # append a numeric suffix
    n=2
    while [[ -n "${seen_label[${lbl}_$n]:-}" ]]; do n=$((n+1)); done
    lbl="${lbl}_$n"
  fi
  seen_label["$lbl"]=1

  dir="$ROOT/$lbl"
  mkdir -p "$dir"

  LABEL+=("$lbl")
  OUT_DIR+=("$dir")
done

echo "[v7] cameras:"
for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
  echo "  - ${CAM_NAME[$i]}  =>  ${DEVICE_PATH[$i]}  →  ${LABEL[$i]}"
done

# Emit a mapping file for downstream use
{
  echo '# label,customName,devicePath'
  for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    printf '%s,%s,%s\n' "${LABEL[$i]}" "${CAM_NAME[$i]//,/;}" "${DEVICE_PATH[$i]}"
  done
} > "$ROOT/camera_map.csv"

jq -n --arg ts "$TS" --arg root "$ROOT" \
  --argjson cams "$(for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    printf '{"label": "%s", "customName": "%s", "devicePath": "%s"}\n' \
      "${LABEL[$i]}" "${CAM_NAME[$i]//\"/\\\"}" "${DEVICE_PATH[$i]}"
  done | jq -s '.')" \
  '{ts: $ts, root: $root, cameras: $cams}' > "$ROOT/camera_map.json"

# Register discovered cameras in Supabase (non-fatal)
sed -i "s/\r$//" /app/register_cameras.sh 2>/dev/null || true
chmod +x /app/register_cameras.sh 2>/dev/null || true
/app/register_cameras.sh "$ROOT" || true

# ===== Per-camera runner (spawns v6) =====
run_one_cam() {
  local idx="$1"
  local cam="${CAM_NAME[$idx]}"
  local dev="${DEVICE_PATH[$idx]}"
  local lbl="${LABEL[$idx]}"
  local dir="${OUT_DIR[$idx]}"

  echo "[v7][$lbl] starting → $dir"
  # Jitter to avoid API bursts across cams
  sleep $(( RANDOM % (JITTER_MAX + 1) ))

  # Delegate to v6 worker with per-cam env and folder
  TZ="${TZ:-America/New_York}" \
  CAM_ID_1="$lbl" \
  DEVICE_1="$dev" \
  DIAG_DIR="$dir" \
  TS="$TS" \
  WINDOW_SECONDS="$WINDOW_SECONDS" \
  SEG_SECONDS="$SEG_SECONDS" \
  REENCODE="$REENCODE" \
  FIRST_EXT_DELAY_SEC="$FIRST_EXT_DELAY_SEC" \
  EXT_EARLY_SEC="$EXT_EARLY_SEC" \
  bash -lc '/app/diag_v6.sh' || true

  echo "[v7][$lbl] done."
}

# ===== Fan-out and wait =====
pids=()
for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
  run_one_cam "$i" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=$((fail+1)); fi
done

echo "[v7] all cams finished. failures=$fail. root=$ROOT"
exit 0
