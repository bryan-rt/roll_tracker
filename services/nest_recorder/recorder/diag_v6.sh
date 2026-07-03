#!/usr/bin/env bash
set -euo pipefail

# ========== config (env overridable) ==========
SEG_SECONDS="${SEG_SECONDS:-120}"           # segment length
WINDOW_SECONDS="${WINDOW_SECONDS:-900}"     # total wall-clock window (default 15 min)
FIRST_EXT_DELAY="${FIRST_EXT_DELAY_SEC:-120}"  # first extend ~2 min after start
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"         # schedule next extend at (expiresAt - this)
CAM_ID="${CAM_ID_1:-cam1}"
DEVICE="${DEVICE_1:?missing DEVICE_1 env}"
REENCODE="${REENCODE:-1}"                   # 1 = libx264 (robust), 0 = copy

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
if [ -z "${DIAG_DIR:-}" ]; then
  DIAG_DIR="/recordings/diag/$TS"
fi
mkdir -p "$DIAG_DIR"
LOG="$DIAG_DIR/run.log"
echo "[v6] writing to $DIAG_DIR" | tee -a "$LOG"

# ========== globals ==========
ACCESS_TOKEN="" URL="" EXT_TOKEN="" STOP_TOKEN=""
FFMPEG_PID="" EXT_PID=""
START_EPOCH="$(date -u +%s)"
DEADLINE="$(( START_EPOCH + WINDOW_SECONDS ))"
ATTEMPT=0
BACKOFF=3    # seconds; exponential up to 60s

# ========== helpers ==========
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

cleanup() {
  set +e
  [ -n "$EXT_PID" ]    && kill "$EXT_PID" 2>/dev/null || true
  [ -n "$FFMPEG_PID" ] && kill "$FFMPEG_PID" 2>/dev/null || true
  if [ -n "${STOP_TOKEN:-}" ]; then
    http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/stop.json" -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$STOP_TOKEN\"}}")
    echo "[v6] stop HTTP=$http (ignored if not 200)" | tee -a "$DIAG_DIR/stop_http.txt"
  fi
}
trap cleanup EXIT INT TERM

get_access_token() {
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  printf "%s" "$ACCESS_TOKEN" | wc -c | tr -d ' ' > "$DIAG_DIR/token_len.txt"
}

generate_stream() {
  local out="$DIAG_DIR/generate_${ATTEMPT}.json"
  local http
  http=$(curl -s -w '%{http_code}' -o "$out" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')
  echo "$http" > "$DIAG_DIR/generate_${ATTEMPT}_http.txt"
  if [ "$http" != "200" ]; then
    log "[v6] Generate failed (HTTP=$http)"
    return 1
  fi

  URL="$(jq -r '.results.streamUrls.rtspUrl // empty' "$out")"
  EXT_TOKEN="$(jq -r '.results.streamExtensionToken // empty' "$out")"
  STOP_TOKEN="$(jq -r '.results.streamToken // empty' "$out")"
  if [ -z "$URL" ] || [ -z "$EXT_TOKEN" ] || [ -z "$STOP_TOKEN" ]; then
    log "[v6] Generate missing fields (url/ext/stop)"
    return 1
  fi

  printf "%s\n" "$URL"       > "$DIAG_DIR/rtsp_url.txt"
  printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"
  printf "%s\n" "$STOP_TOKEN"> "$DIAG_DIR/stop_token.txt"
  date -u +%s > "$DIAG_DIR/generated_at_epoch.txt"
  log "[v6] Generated RTSP and tokens (attempt=$ATTEMPT)"
}

extend_loop() {
  local next_sleep="$FIRST_EXT_DELAY"
  while kill -0 "$FFMPEG_PID" 2>/dev/null; do
    sleep "$next_sleep"
    [ ! -e "/proc/$FFMPEG_PID" ] && break

    local stamp http jf new_ext new_stop exp_iso exp_epoch now
    stamp="$(date -u +%s)"
    http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}.json" \
      -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
    echo "$stamp $http" >> "$DIAG_DIR/extend_http.txt"

    if [ "$http" = "200" ]; then
      jf="$DIAG_DIR/extend_${stamp}.json"
      new_ext="$(jq -r '.results.streamExtensionToken // empty' "$jf" 2>/dev/null || true)"
      [ -n "$new_ext" ] && EXT_TOKEN="$new_ext" && printf "%s\n" "$EXT_TOKEN" > "$DIAG_DIR/ext_token.txt"

      new_stop="$(jq -r '.results.streamToken // empty' "$jf" 2>/dev/null || true)"
      [ -n "$new_stop" ] && STOP_TOKEN="$new_stop" && printf "%s\n" "$STOP_TOKEN" > "$DIAG_DIR/stop_token.txt"

      exp_iso="$(jq -r '.results.expiresAt // empty' "$jf" 2>/dev/null || true)"
      if [ -n "$exp_iso" ]; then
        exp_epoch=$(date -u -d "$exp_iso" +%s 2>/dev/null || echo "")
        if [ -n "$exp_epoch" ]; then
          now="$(date -u +%s)"
          next_sleep=$(( exp_epoch - EXT_EARLY_SEC - now ))
          [ "$next_sleep" -lt 60 ] && next_sleep=60
        else
          next_sleep=240
        fi
      else
        next_sleep=240
      fi
    else
      # brief retry
      sleep 3
      stamp="$(date -u +%s)"
      http=$(curl -s -w '%{http_code}' -o "$DIAG_DIR/extend_${stamp}_retry.json" \
        -X POST "https://smartdevicemanagement.googleapis.com/v1/${DEVICE}:executeCommand" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
      echo "$stamp $http (retry)" >> "$DIAG_DIR/extend_http.txt"
      [ "$http" = "200" ] || { log "[v6] extend failing; stopping extend loop"; break; }
      next_sleep=240
    fi
  done
}

build_ffmpeg_opts() {
  # Segment muxer options (array-safe)
  SEG_OPTS=(-f segment -segment_time "$SEG_SECONDS" -strftime 1 -movflags +faststart)
  if ffmpeg -hide_banner -h muxer=segment 2>&1 | grep -qi 'reset_timestamps'; then
    echo "[v6] segment muxer supports -reset_timestamps 1" | tee -a "$LOG"
    SEG_OPTS+=(-reset_timestamps 1)
  else
    echo "[v6] segment muxer lacks -reset_timestamps; proceeding" | tee -a "$LOG"
  fi

  # Defaults — overridden per mode below
  VF_OPTS=()
  FPS_MODE_OPTS=()
  INPUT_FFLAGS="+genpts+igndts"

  # Video path: REENCODE=1|2 (CFR + timing sidecar), 0 (VFR passthrough)
  # Mode 1 and 2 are now identical: CFR re-encode with -vf showinfo for timing
  # sidecar extraction. Video output is byte-identical with and without showinfo
  # (confirmed via MD5 comparison in RECORDER-TIMING-2).
  if [ "$REENCODE" = "1" ] || [ "$REENCODE" = "2" ]; then
    echo "[v6] REENCODE=$REENCODE → libx264 veryfast + timing sidecar (CFR)" | tee -a "$LOG"
    V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
    VF_OPTS=(-vf showinfo)
  elif [ "$REENCODE" = "0" ]; then
    echo "[v6] REENCODE=0 → stream copy (VFR passthrough)" | tee -a "$LOG"
    V_OPTS=(-c:v copy)
    INPUT_FFLAGS="+igndts"
    FPS_MODE_OPTS=(-fps_mode passthrough)
  else
    echo "[v6] REENCODE=$REENCODE unknown, falling back to CFR + timing sidecar" | tee -a "$LOG"
    V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30)
    VF_OPTS=(-vf showinfo)
  fi
}

extract_timing_sidecars() {
  # Post-process ffmpeg stderr to produce per-segment .timing.jsonl sidecars.
  # Each sidecar has one row per OUTPUT frame (keyed on frame_index matching
  # FrameIterator's cap.read() counter), with the real INPUT arrival PTS
  # mapped via nearest-neighbor two-pointer.
  local stderr="$DIAG_DIR/ffmpeg.stderr"
  [ ! -f "$stderr" ] && return 0
  grep -q 'Parsed_showinfo.*pts_time:' "$stderr" || return 0

  # Step 1: Find segment mp4s and their opening-line positions in stderr
  local -a seg_paths seg_lines seg_epochs
  while IFS= read -r gline; do
    local lineno path base ymd_hms epoch=0
    lineno="${gline%%:*}"
    path=$(echo "$gline" | sed "s/.*Opening '//;s/' for writing.*//")
    base=$(basename "$path" .mp4)
    ymd_hms=$(echo "$base" | grep -oE '[0-9]{8}-[0-9]{6}')
    if [ -n "$ymd_hms" ]; then
      epoch=$(date -d "${ymd_hms:0:4}-${ymd_hms:4:2}-${ymd_hms:6:2} ${ymd_hms:9:2}:${ymd_hms:11:2}:${ymd_hms:13:2}" +%s 2>/dev/null || echo 0)
    fi
    seg_paths+=("$path")
    seg_lines+=("$lineno")
    seg_epochs+=("$epoch")
  done < <(grep -n "Opening.*\.mp4.*for writing" "$stderr")

  [ "${#seg_paths[@]}" -eq 0 ] && return 0
  local total_stderr_lines
  total_stderr_lines=$(wc -l < "$stderr")

  # Step 2: For each segment, extract showinfo PTS, get output info, build sidecar
  for (( si=0; si<${#seg_paths[@]}; si++ )); do
    local seg_path="${seg_paths[$si]}"
    local from_line="${seg_lines[$si]}"
    local epoch="${seg_epochs[$si]}"
    local to_line="$total_stderr_lines"
    if (( si + 1 < ${#seg_lines[@]} )); then
      to_line="${seg_lines[$((si+1))]}"
    fi

    local sidecar="${seg_path%.mp4}.timing.jsonl"
    local pts_tmp="$DIAG_DIR/_pts_${si}.tmp"

    # Extract showinfo pts_time values for this segment's stderr range, sort by PTS
    sed -n "${from_line},${to_line}p" "$stderr" \
      | grep 'Parsed_showinfo.*pts_time:' \
      | sed -n 's/.*pts_time:\([0-9.eE+-]*\).*/\1/p' \
      | sort -g \
      > "$pts_tmp"

    local input_count
    input_count=$(wc -l < "$pts_tmp" | tr -d ' ')

    if [ "$input_count" -eq 0 ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — no showinfo data, skipping"
      rm -f "$pts_tmp"
      continue
    fi

    # Get output frame count and fps from the segment mp4
    local output_count=0 output_fps=0
    if [ -f "$seg_path" ]; then
      output_count=$(ffprobe -hide_banner -select_streams v:0 \
        -show_entries stream=nb_frames -of csv=p=0 "$seg_path" 2>/dev/null | tr -d ' ')
      output_fps=$(ffprobe -hide_banner -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "$seg_path" 2>/dev/null | tr -d ' ')
    fi
    [ -z "$output_count" ] || [ "$output_count" = "N/A" ] && output_count=0
    [ -z "$output_fps" ] && output_fps="0/1"

    if [ "$output_count" -eq 0 ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — cannot read output frame count, skipping"
      rm -f "$pts_tmp"
      continue
    fi

    # Mismatch detection
    local mismatch="false"
    if [ "$input_count" -ne "$output_count" ]; then
      mismatch="true"
    fi

    # Two-pointer mapping: for each output frame, find nearest input PTS.
    # Writes one JSONL row per OUTPUT frame (frame_index = join key to Stage A).
    # pts_time_s = the REAL input arrival time (bursty, NOT uniform CFR).
    awk -v output_count="$output_count" \
        -v output_fps="$output_fps" \
        -v epoch="$epoch" \
        -v input_count="$input_count" \
        -v mismatch="$mismatch" \
        -v pts_file="$pts_tmp" \
    '
    BEGIN {
      # Read sorted input PTS into array
      ni = 0
      while ((getline line < pts_file) > 0) {
        input_pts[ni] = line + 0.0
        ni++
      }
      close(pts_file)

      # Normalize PTS: subtract first value so segment-relative starts at ~0
      base_pts = (ni > 0) ? input_pts[0] : 0
      for (k = 0; k < ni; k++) input_pts[k] -= base_pts

      # Parse fractional fps (e.g. "69/4" or "30")
      if (index(output_fps, "/") > 0) {
        split(output_fps, fparts, "/")
        fps_val = fparts[1] / fparts[2]
      } else {
        fps_val = output_fps + 0.0
      }
      if (fps_val <= 0) fps_val = 30.0
      interval = 1.0 / fps_val

      # Metadata line
      printf "{\"_meta\":true,\"segment_start_epoch\":%s,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"mismatch\":%s}\n", \
        epoch, ni, output_count, fps_val, mismatch

      # Two-pointer: map each output frame to nearest input PTS
      j = 0
      for (i = 0; i < output_count; i++) {
        t_out = i * interval
        # Advance j while next input PTS is closer to t_out
        while (j + 1 < ni) {
          d_cur = input_pts[j] - t_out
          if (d_cur < 0) d_cur = -d_cur
          d_next = input_pts[j+1] - t_out
          if (d_next < 0) d_next = -d_next
          if (d_next <= d_cur) j++
          else break
        }
        printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"input_n\":%d}\n", i, input_pts[j], j
      }
    }
    ' > "$sidecar"

    rm -f "$pts_tmp"

    # Summary line with loud mismatch warning
    local sidecar_lines
    sidecar_lines=$(( $(wc -l < "$sidecar") - 1 ))  # subtract metadata line
    if [ "$mismatch" = "true" ]; then
      log "[v6] ⚠ MISMATCH sidecar: $(basename "$sidecar") input=$input_count output=$output_count (frame join may be inaccurate)"
    else
      log "[v6] sidecar: $(basename "$sidecar") $sidecar_lines/$output_count ✓ (epoch=$epoch)"
    fi
  done
}

start_ffmpeg() {
  mkdir -p "$DIAG_DIR"
  local out_tmpl="$DIAG_DIR/${CAM_ID}-%Y%m%d-%H%M%S.mp4"
  log "[v6] recording until $(date -u -d "@$DEADLINE" +%H:%M:%S) in ${SEG_SECONDS}s segments → $out_tmpl"

  # Freshness guard: if URL is old (>60s), re-generate once
  local now age
  now="$(date -u +%s)"
  age=$(( now - $(cat "$DIAG_DIR/generated_at_epoch.txt" 2>/dev/null || echo "$now") ))
  if [ "$age" -gt 60 ]; then
    log "[v6] URL is ${age}s old; regenerating before start"
    generate_stream || log "[v6] pre-start regenerate failed; proceeding with current URL"
  fi

  ffmpeg -hide_banner -loglevel info -nostdin -y \
    -rtsp_transport tcp \
    -use_wallclock_as_timestamps 1 -fflags "$INPUT_FFLAGS" -avoid_negative_ts make_zero \
    -analyzeduration 10M -probesize 10M \
    -i "$URL" \
    -map 0:v:0 -map 0:a:0 \
    "${VF_OPTS[@]}" \
    "${FPS_MODE_OPTS[@]}" \
    "${V_OPTS[@]}" \
    -c:a aac -ar 48000 -ac 1 -b:a 64k \
    -max_muxing_queue_size 1024 \
    -t "$(( DEADLINE - $(date -u +%s) ))" \
    "${SEG_OPTS[@]}" \
    "$out_tmpl" \
    1> "$DIAG_DIR/ffmpeg.stdout" 2> "$DIAG_DIR/ffmpeg.stderr" &
  FFMPEG_PID=$!

  extend_loop & EXT_PID=$!
}

# ========== main ==========
get_access_token

build_ffmpeg_opts

while :; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed"; break; }

  ATTEMPT=$((ATTEMPT+1))
  log "[v6] attempt #$ATTEMPT"

  if ! generate_stream; then
    log "[v6] Generate failed; backoff ${BACKOFF}s"
    sleep "$BACKOFF"
    [ "$BACKOFF" -lt 60 ] && BACKOFF=$(( BACKOFF * 2 ))
    continue
  fi

  start_ffmpeg

  # Wait for ffmpeg to end or window to elapse
  wait "$FFMPEG_PID" || true
  rc=$?
  log "[v6] ffmpeg exited rc=$rc"

  # stop extend loop for this attempt
  [ -n "$EXT_PID" ] && kill "$EXT_PID" 2>/dev/null || true
  EXT_PID=""

  # Extract per-segment timing sidecars from showinfo data in stderr
  extract_timing_sidecars

  # Done if time is up
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed after attempt #$ATTEMPT"; break; }

  # If ffmpeg ended early, back off and try to recover with a fresh stream
  log "[v6] preparing next attempt (backoff ${BACKOFF}s)"
  sleep "$BACKOFF"
  [ "$BACKOFF" -lt 60 ] && BACKOFF=$(( BACKOFF * 2 ))
done

log "[v6] done. Artifacts in $DIAG_DIR"
