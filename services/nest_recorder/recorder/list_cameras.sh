#!/usr/bin/env bash
set -euo pipefail

SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
if [[ -z "${SDM_PROJECT_ID:-}" && -f "$SECRETS_DIR/project_id.txt" ]]; then
  SDM_PROJECT_ID="$(cat "$SECRETS_DIR/project_id.txt")"
fi
: "${SDM_PROJECT_ID:?SDM_PROJECT_ID not set and $SECRETS_DIR/project_id.txt missing}"

ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
OUT="/recordings/diag/devices-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

# List devices in the SDM project
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://smartdevicemanagement.googleapis.com/v1/enterprises/${SDM_PROJECT_ID}/devices" \
  -o "$OUT/devices.json"

# Extract only devices that expose CameraLiveStream
jq '
  .devices[]?
  | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
  | {
      name: .name,
      type: .type,
      room: (.parentRelations[0].displayName // "Unknown"),
      customName: (.traits."sdm.devices.traits.Info".customName // "cam")
    }
' "$OUT/devices.json" | tee "$OUT/cameras.json"

# Also emit a simple CSV you can paste into an env var for v7
echo "#CAM_ID,DEVICE_NAME"       | tee "$OUT/cameras.csv"
jq -r '
  .devices[]?
  | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
  | ((.traits."sdm.devices.traits.Info".customName // "cam") + "," + .name)
' "$OUT/devices.json" | tee -a "$OUT/cameras.csv"

echo "[list] wrote: $OUT/devices.json  $OUT/cameras.json  $OUT/cameras.csv"