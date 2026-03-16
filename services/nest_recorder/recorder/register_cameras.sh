#!/usr/bin/env bash
set -euo pipefail

# register_cameras.sh — upsert discovered cameras into Supabase
# Called by diag_v7_2.sh after camera discovery.
# Non-fatal: if env vars are missing or API fails, logs a warning and exits 0.
#
# Usage: register_cameras.sh <session_root>
#   session_root: path containing devices.json and camera_map.json

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
  echo "[register] ERROR: session root path required as first argument" >&2
  exit 0
fi

# Required env vars — skip gracefully if missing
if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_SERVICE_KEY:-}" ] || [ -z "${GYM_ID:-}" ]; then
  echo "[register] skipping: SUPABASE_URL, SUPABASE_SERVICE_KEY, or GYM_ID not set"
  exit 0
fi

DEVICES_JSON="$ROOT/devices.json"
CAMERA_MAP="$ROOT/camera_map.json"

if [ ! -f "$CAMERA_MAP" ]; then
  echo "[register] skipping: $CAMERA_MAP not found"
  exit 0
fi

# Build a lookup of devicePath -> room name from devices.json
# Falls back to empty string if devices.json is missing or has no parentRelations
declare -A ROOM_MAP
if [ -f "$DEVICES_JSON" ]; then
  while IFS=$'\t' read -r dpath room; do
    ROOM_MAP["$dpath"]="$room"
  done < <(jq -r '
    .devices[]?
    | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
    | (.name + "\t" + (.parentRelations[0].displayName // ""))
  ' "$DEVICES_JSON" 2>/dev/null || true)
fi

# Read camera_map.json and upsert each camera
cam_count=$(jq -r '.cameras | length' "$CAMERA_MAP")
echo "[register] upserting $cam_count cameras to $SUPABASE_URL for gym $GYM_ID"

success=0
fail=0
for (( i=0; i<cam_count; i++ )); do
  device_path=$(jq -r ".cameras[$i].devicePath" "$CAMERA_MAP")

  # cam_id = last 6 chars of device path
  cam_id="${device_path: -6}"

  # display_name: prefer room name from devices.json, fall back to customName from camera_map
  display_name="${ROOM_MAP[$device_path]:-}"
  if [ -z "$display_name" ]; then
    custom=$(jq -r ".cameras[$i].customName // empty" "$CAMERA_MAP")
    if [ -n "$custom" ] && [ "$custom" != "cam" ]; then
      display_name="$custom"
    fi
  fi

  # Build JSON payload
  payload=$(jq -n \
    --arg gym_id "$GYM_ID" \
    --arg cam_id "$cam_id" \
    --arg device_path "$device_path" \
    --arg display_name "$display_name" \
    '{
      gym_id: $gym_id,
      cam_id: $cam_id,
      device_path: $device_path,
      display_name: (if $display_name == "" then null else $display_name end),
      is_active: true,
      last_seen_at: (now | todate)
    }')

  http_code=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "${SUPABASE_URL}/rest/v1/cameras?on_conflict=gym_id,cam_id" \
    -H "apikey: ${SUPABASE_SERVICE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_KEY}" \
    -H "Content-Type: application/json" \
    -H "Prefer: resolution=merge-duplicates" \
    -d "$payload")

  if [ "$http_code" = "201" ] || [ "$http_code" = "200" ]; then
    echo "[register] $cam_id ($display_name) → OK ($http_code)"
    success=$((success + 1))
  else
    echo "[register] $cam_id ($display_name) → FAILED (HTTP $http_code)" >&2
    fail=$((fail + 1))
  fi
done

echo "[register] done: $success ok, $fail failed"
exit 0
