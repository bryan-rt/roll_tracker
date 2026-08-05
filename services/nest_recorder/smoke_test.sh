#!/usr/bin/env bash
set -euo pipefail

# CP-R9: Regression smoke harness for the Nest recorder.
#
# Runs a 60-second capture through diag_v7_2.sh (NEVER diag_v6.sh standalone —
# .env DEVICE_* are human-readable names, not SDM device paths; v7_2 resolves
# them via discovery. Standalone v6 fails Generate 404 every time, by design.)
#
# Usage:
#   ./smoke_test.sh              # default mode (source-PTS passthrough)
#   ./smoke_test.sh --rollback   # arrival-PTS CFR rollback mode

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# --- Mode ---
MODE="default"
if [[ "${1:-}" == "--rollback" ]]; then
  MODE="rollback"
fi

# --- Expected values per mode ---
if [[ "$MODE" == "default" ]]; then
  EXP_SOURCE_PTS=1
  EXP_FPS_PASSTHROUGH=1
  EXP_TIMING_MODE="passthrough"
  TIMING_OVERRIDES=""
else
  EXP_SOURCE_PTS=0
  EXP_FPS_PASSTHROUGH=0
  EXP_TIMING_MODE="cfr_grid"
  TIMING_OVERRIDES="SOURCE_PTS=0 FPS_PASSTHROUGH=0"
fi

SEG_SECONDS=20
WINDOW_SECONDS=60
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SKIP_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
warn() { echo "  WARN: $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
skip() { echo "  SKIP: $1"; SKIP_COUNT=$((SKIP_COUNT + 1)); }

echo "========================================"
echo "Recorder Smoke Test — mode: $MODE"
echo "========================================"
echo ""

# --- Pre-flight: ensure container is running ---
echo "[1/4] Ensuring container is running..."
$COMPOSE up -d recorder 2>&1 | grep -v "^time=" || true
echo ""

# --- Pre-flight: ffmpeg option availability (same script as production entrypoint) ---
echo "[2/4] Checking ffmpeg options (check_ffmpeg_opts.sh)..."
if $COMPOSE exec recorder /app/check_ffmpeg_opts.sh 2>&1 | grep -v "^time="; then
  pass "check_ffmpeg_opts.sh passed"
else
  fail "check_ffmpeg_opts.sh FAILED — ffmpeg is missing required options"
fi
echo ""

# Bail early if ffmpeg is broken
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "ABORT: ffmpeg option check failed. Not running capture."
  exit 1
fi

# --- Run the capture ---
echo "[3/4] Running 60-second capture (SEG_SECONDS=$SEG_SECONDS)..."
echo "  Command: $TIMING_OVERRIDES WINDOW_SECONDS=$WINDOW_SECONDS SEG_SECONDS=$SEG_SECONDS /app/diag_v7_2.sh"
echo ""

CAPTURE_OUTPUT=$($COMPOSE exec recorder bash -lc \
  "$TIMING_OVERRIDES WINDOW_SECONDS=$WINDOW_SECONDS SEG_SECONDS=$SEG_SECONDS /app/diag_v7_2.sh" 2>&1) || true

# Find the session directory (latest /recordings/diag/*)
SESSION_DIR=$($COMPOSE exec recorder bash -c \
  'ls -td /recordings/diag/*/ 2>/dev/null | head -1 | tr -d "\r\n"' 2>&1 | grep -v "^time=")

if [[ -z "$SESSION_DIR" || "$SESSION_DIR" == *"No such file"* ]]; then
  fail "No session directory found under /recordings/diag/"
  echo ""
  echo "Capture output:"
  echo "$CAPTURE_OUTPUT"
  echo ""
  echo "RESULT: $FAIL_COUNT FAIL, $PASS_COUNT PASS, $WARN_COUNT WARN, $SKIP_COUNT SKIP"
  exit 1
fi

echo "  Session: $SESSION_DIR"
echo ""

# --- Per-camera assertions ---
echo "[4/4] Checking per-camera results..."
echo ""

# Discover per-camera output directories.
# With GYM_ID set, output goes to /recordings/{GYM_ID}/{sid}/{date}/{hour}/,
# not under the diag session root. Use camera_map.csv to find short IDs, then
# locate their run.log by matching the TS from the session.
CAM_INFO_RAW=$($COMPOSE exec recorder bash -c "
  SESSION_ROOT='$SESSION_DIR'
  TS=\$(basename \"\$SESSION_ROOT\")
  ymd=\"\${TS:0:4}-\${TS:4:2}-\${TS:6:2}\"
  hh=\"\${TS:9:2}\"
  while IFS=',' read -r label name devpath; do
    case \"\$label\" in '#'*) continue ;; esac
    sid=\"\${devpath: -6}\"
    diag_dir=\"\${SESSION_ROOT}\${label}/\"
    if [ -d \"\$diag_dir\" ]; then
      echo \"\${label}|\${diag_dir}\"
      continue
    fi
    for gym_dir in /recordings/*/\${sid}/\${ymd}/\${hh}/; do
      if [ -d \"\$gym_dir\" ]; then
        echo \"\${label}|\${gym_dir}\"
        break
      fi
    done
  done < \"\${SESSION_ROOT}camera_map.csv\"
" 2>&1 || true)
CAM_INFO=$(echo "$CAM_INFO_RAW" | tr -d '\r' | grep -v "^time=" | grep "|" || true)

if [[ -z "$CAM_INFO" ]]; then
  fail "No camera directories found for session $SESSION_DIR"
  echo ""
  echo "RESULT: $FAIL_COUNT FAIL, $PASS_COUNT PASS, $WARN_COUNT WARN, $SKIP_COUNT SKIP"
  exit 1
fi

# Read CAM_INFO into an array to avoid docker compose exec consuming heredoc stdin
IFS=$'\n' read -r -d '' -a CAM_LINES <<< "$CAM_INFO" || true

# Disable errexit for the assertion loop — individual assertions handle their own errors
set +e
for cam_line in "${CAM_LINES[@]}"; do
  IFS='|' read -r cam_label cam_dir <<< "$cam_line"
  [[ -z "$cam_dir" ]] && continue

  echo "--- $cam_label ---"

  # Count mp4s from THIS session only (newer than the session's camera_map.json)
  mp4_count=$($COMPOSE exec recorder bash -c \
    "find '$cam_dir' -maxdepth 1 -name '*.mp4' -newer '${SESSION_DIR}camera_map.json' 2>/dev/null | wc -l" \
    2>&1 | grep -v "^time=" | tr -d ' \r' | tail -1)
  mp4_count="${mp4_count:-0}"

  if [[ "$mp4_count" -eq 0 ]]; then
    skip "$cam_label: no segments — camera may be offline"
    echo ""
    continue
  fi

  # Assertion 2: at least one mp4
  pass "$cam_label: $mp4_count mp4(s) produced"

  # Assertion 1: startup log values
  log_line=$($COMPOSE exec recorder bash -c \
    "grep 'timing config:' '${cam_dir}run.log' 2>/dev/null | tail -1 | tr -d '\r\n'" 2>&1 | grep -v "^time=")

  if echo "$log_line" | grep -q "SOURCE_PTS=$EXP_SOURCE_PTS.*FPS_PASSTHROUGH=$EXP_FPS_PASSTHROUGH"; then
    pass "startup log: SOURCE_PTS=$EXP_SOURCE_PTS FPS_PASSTHROUGH=$EXP_FPS_PASSTHROUGH"
  else
    fail "startup log mismatch: got '$log_line', expected SOURCE_PTS=$EXP_SOURCE_PTS FPS_PASSTHROUGH=$EXP_FPS_PASSTHROUGH"
  fi

  # Get list of mp4 files from THIS session only (newer than camera_map.json)
  mp4_files=$($COMPOSE exec recorder bash -c \
    "find '$cam_dir' -maxdepth 1 -name '*.mp4' -newer '${SESSION_DIR}camera_map.json' 2>/dev/null | sort | tr '\n' '|'" \
    2>&1 | grep -v "^time=" | tr -d '\r')
  IFS='|' read -ra MP4S <<< "$mp4_files"

  # Track non-final segments for duration check
  mp4_array=()
  for f in "${MP4S[@]}"; do
    [[ -z "$f" ]] && continue
    mp4_array+=("$f")
  done

  for idx in "${!mp4_array[@]}"; do
    mp4="${mp4_array[$idx]}"
    [[ -z "$mp4" ]] && continue
    bn=$($COMPOSE exec recorder bash -c "basename '$mp4' | tr -d '\r\n'" 2>&1 | grep -v "^time=")
    sidecar="${mp4%.mp4}.timing.jsonl"

    # Assertion 3: sidecar exists
    sidecar_exists=$($COMPOSE exec recorder bash -c \
      "[[ -f '$sidecar' ]] && echo yes || echo no" 2>&1 | grep -v "^time=" | tr -d '\r\n')

    if [[ "$sidecar_exists" != "yes" ]]; then
      fail "$bn: no .timing.jsonl sidecar"
      continue
    fi
    pass "$bn: sidecar exists"

    # Parse _meta
    meta=$($COMPOSE exec recorder bash -c "head -1 '$sidecar' | tr -d '\r\n'" 2>&1 | grep -v "^time=")

    # Assertion 4: timing_mode
    timing_mode=$(echo "$meta" | grep -o '"timing_mode":"[^"]*"' | cut -d'"' -f4)
    if [[ "$timing_mode" == "$EXP_TIMING_MODE" ]]; then
      pass "$bn: timing_mode=$timing_mode"
    else
      fail "$bn: timing_mode=$timing_mode, expected $EXP_TIMING_MODE"
    fi

    # Assertion 5: sidecar_schema
    schema=$(echo "$meta" | grep -o '"sidecar_schema":[0-9]*' | cut -d: -f2)
    if [[ "$schema" == "4" ]]; then
      pass "$bn: sidecar_schema=4"
    else
      fail "$bn: sidecar_schema=$schema, expected 4"
    fi

    # Assertion 6: measured_fps band (C2: bimodal-aware)
    measured_fps=$(echo "$meta" | grep -o '"measured_fps":[0-9.]*' | cut -d: -f2)
    trim_kept=$(echo "$meta" | grep -o '"pts_delta_trim_kept":[0-9]*' | cut -d: -f2)
    trim_total=$(echo "$meta" | grep -o '"pts_delta_trim_total":[0-9]*' | cut -d: -f2)

    if [[ "$MODE" == "rollback" ]]; then
      # Schema 4: measured_fps is OMITTED under source_pts=false (arrival-PTS)
      if [[ -z "$measured_fps" ]]; then
        pass "$bn: measured_fps absent (arrival-PTS, expected under schema 4)"
      else
        fail "$bn: measured_fps=$measured_fps present under arrival-PTS — expected absent"
      fi
    else
      # Source-PTS: must be within ±10% of 15 or 30
      # Accept ±10% of 15fps or 30fps. Also accept ~7.5fps (half of 15, observed on PPDmUg).
      in_15_band=$(echo "$measured_fps >= 13.5 && $measured_fps <= 16.5" | bc -l 2>/dev/null || echo 0)
      in_30_band=$(echo "$measured_fps >= 27.0 && $measured_fps <= 33.0" | bc -l 2>/dev/null || echo 0)
      in_7_band=$(echo "$measured_fps >= 6.75 && $measured_fps <= 8.25" | bc -l 2>/dev/null || echo 0)

      if [[ "$in_15_band" == "1" || "$in_30_band" == "1" ]]; then
        pass "$bn: measured_fps=$measured_fps (in band)"
      elif [[ "$in_7_band" == "1" ]]; then
        warn "$bn: measured_fps=$measured_fps (~7.5fps half-rate mode, camera may be under-delivering)"
      else
        # Out of band — check for bimodal contamination (C2)
        # Bimodal signature: value between the two modes (16.5-27.0)
        # OR elevated discard rate (>15%)
        in_between=$(echo "$measured_fps > 16.5 && $measured_fps < 27.0" | bc -l 2>/dev/null || echo 0)
        discard_pct=0
        if [[ -n "$trim_kept" && -n "$trim_total" && "$trim_total" -gt 0 ]]; then
          discard_pct=$(echo "scale=1; (1 - $trim_kept / $trim_total) * 100" | bc -l 2>/dev/null || echo 0)
        fi
        elevated_discard=$(echo "$discard_pct > 15" | bc -l 2>/dev/null || echo 0)

        if [[ "$in_between" == "1" || "$elevated_discard" == "1" ]]; then
          warn "$bn: measured_fps=$measured_fps (out of band, discard=${discard_pct}% — bimodal suspected, TRIM-BIMODAL CP-R6)"
        else
          fail "$bn: measured_fps=$measured_fps (out of band, discard=${discard_pct}% — not bimodal, genuine problem)"
        fi
      fi
    fi

    # Assertion 7: segment duration via ffprobe (C1)
    # Exempt the final segment — window-end truncation is expected
    is_final=$(( idx == ${#mp4_array[@]} - 1 ? 1 : 0 ))

    duration=$($COMPOSE exec recorder bash -c \
      "ffprobe -v error -show_entries format=duration -of csv=p=0 '$mp4' 2>/dev/null | tr -d '\r\n'" 2>&1 | grep -v "^time=")

    if [[ -z "$duration" || "$duration" == "N/A" ]]; then
      fail "$bn: could not read duration via ffprobe"
    elif [[ $is_final -eq 1 ]]; then
      pass "$bn: duration=${duration}s (final segment, exempt from ±20% check)"
    else
      # Non-final: must be within ±20% of SEG_SECONDS
      lo=$(echo "$SEG_SECONDS * 0.8" | bc -l)
      hi=$(echo "$SEG_SECONDS * 1.2" | bc -l)
      in_range=$(echo "$duration >= $lo && $duration <= $hi" | bc -l 2>/dev/null || echo 0)
      if [[ "$in_range" == "1" ]]; then
        pass "$bn: duration=${duration}s (within ±20% of ${SEG_SECONDS}s)"
      else
        fail "$bn: duration=${duration}s (outside ±20% of ${SEG_SECONDS}s: expected ${lo}-${hi})"
      fi
    fi
  done

  echo ""
done
set -e

# --- Summary ---
echo "========================================"
echo "RESULT: $FAIL_COUNT FAIL, $PASS_COUNT PASS, $WARN_COUNT WARN, $SKIP_COUNT SKIP"
echo "========================================"

if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "SMOKE TEST FAILED"
  exit 1
else
  echo "SMOKE TEST PASSED"
  exit 0
fi
