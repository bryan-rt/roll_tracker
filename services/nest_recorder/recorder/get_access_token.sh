#!/usr/bin/env bash
set -euo pipefail

# Where to read/write secrets inside the container
SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
# Writable cache area (token cache + rotated refresh token)
CACHE_DIR="${SDM_CACHE_DIR:-/recordings/secrets}"
mkdir -p "$SECRETS_DIR" "$CACHE_DIR"
umask 077

read_secret() {
  # usage: read_secret VAR_NAME FILEPATH [default]
  local var="$1" file="$2" default="${3:-}"
  if [[ -n "${!var:-}" ]]; then
    printf "%s" "${!var}"
  elif [[ -f "$file" ]]; then
    cat "$file"
  elif [[ -n "$default" ]]; then
    printf "%s" "$default"
  else
    echo "Missing secret: $var or $file" >&2
    return 1
  fi
}

SDM_CLIENT_ID="$(read_secret SDM_CLIENT_ID "$SECRETS_DIR/client_id.txt")"
SDM_CLIENT_SECRET="$(read_secret SDM_CLIENT_SECRET "$SECRETS_DIR/client_secret.txt")"
# Prefer a rotated refresh token in CACHE_DIR; fallback to read-only SECRETS_DIR
if [[ -f "$CACHE_DIR/refresh_token.txt" ]]; then
  SDM_REFRESH_TOKEN="$(cat "$CACHE_DIR/refresh_token.txt")"
else
  SDM_REFRESH_TOKEN="$(read_secret SDM_REFRESH_TOKEN "$SECRETS_DIR/refresh_token.txt")"
fi

CACHE_JSON="$CACHE_DIR/access_token.json"
now="$(date +%s)"

# Use cached access_token if still valid
if [[ -f "$CACHE_JSON" ]]; then
  cached_token="$(jq -r '.access_token // empty' "$CACHE_JSON" 2>/dev/null || true)"
  expires_at="$(jq -r '.expires_at // 0' "$CACHE_JSON" 2>/dev/null || echo 0)"
  if [[ -n "$cached_token" && "$now" -lt "$expires_at" ]]; then
    printf "%s" "$cached_token"
    exit 0
  fi
fi


# Fetch a new access_token using the refresh_token
resp="$(curl -sS -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=${SDM_CLIENT_ID}" \
  -d "client_secret=${SDM_CLIENT_SECRET}" \
  -d "refresh_token=${SDM_REFRESH_TOKEN}" \
  -d "grant_type=refresh_token" \
  https://oauth2.googleapis.com/token)"

token="$(printf "%s" "$resp" | jq -r '.access_token // empty')"
expires_in="$(printf "%s" "$resp" | jq -r '.expires_in // 0')"
new_refresh="$(printf "%s" "$resp" | jq -r '.refresh_token // empty')"

if [[ -z "$token" ]]; then
  echo "[get_access_token] ERROR: no access_token in response" >&2
  printf "%s\n" "$resp" >&2
  exit 1
fi

# Persist a rotated refresh_token if Google sends one
if [[ -n "$new_refresh" && "$new_refresh" != "null" ]]; then
  printf "%s" "$new_refresh" > "$CACHE_DIR/refresh_token.txt"
  chmod 600 "$CACHE_DIR/refresh_token.txt" || true
fi

# Cache the access_token with a small safety margin
expires_at="$(( now + expires_in - 60 ))"
jq -n --arg t "$token" --argjson e "$expires_at" \
  '{access_token:$t, expires_at:$e}' > "$CACHE_JSON"
chmod 600 "$CACHE_JSON" || true

# Output the token for callers
printf "%s" "$token"