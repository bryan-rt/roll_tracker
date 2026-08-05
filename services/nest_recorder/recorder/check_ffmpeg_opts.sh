#!/usr/bin/env bash
# CP-R4: Startup assertion — verify the container's ffmpeg supports every option
# correct recording depends on. Called from entrypoint.sh (production) and
# smoke_test.sh (dev). One implementation, both paths.
#
# Fails hard with exit 1 if any required option is missing. A recorder that
# silently records corrupted footage is worse than one that will not start.

set -euo pipefail

# Log provenance — ffmpeg version and Debian release
echo "[check_ffmpeg] Debian: $(cat /etc/debian_version 2>/dev/null || echo unknown)"
echo "[check_ffmpeg] ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 || echo 'NOT FOUND')"

# Required options:
#   -timeout   : RTSP socket timeout (replaced -stimeout in ffmpeg 7.x; RELIABILITY-1 top fix)
#   -fps_mode  : passthrough mode (CP-R2)
#   -copyts    : source PTS preservation (CP-R2)
REQUIRED_OPTS="-timeout -fps_mode -copyts"

FFHELP=$(mktemp)
ffmpeg -h full >"$FFHELP" 2>&1 || true

fail=0
for opt in $REQUIRED_OPTS; do
  if grep -q -- "$opt" "$FFHELP"; then
    echo "[check_ffmpeg] OK: $opt"
  else
    echo "[check_ffmpeg] MISSING: $opt — recording will break. Check ffmpeg version / base image."
    fail=1
  fi
done

rm -f "$FFHELP"

if [ "$fail" -ne 0 ]; then
  echo "[check_ffmpeg] FATAL: required ffmpeg options missing. Refusing to start."
  exit 1
fi

echo "[check_ffmpeg] All required options present."
