#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# diag_timing.sh — Multi-camera timing diagnostic with TRUE capture timestamps
#
# Parallel module for CAPTURE-TIME-2. Does NOT modify the production recorder.
# Captures all discovered cameras concurrently with source PTS preserved
# (no wallclock override), per-frame host-arrival timestamps, and RTCP probing.
#
# Usage: WINDOW_SECONDS=60 /app/diag_timing.sh
# ============================================================================

SECRETS_DIR="${SDM_SECRETS_DIR:-/secrets}"
if [[ -z "${SDM_PROJECT_ID:-}" && -f "$SECRETS_DIR/project_id.txt" ]]; then
  SDM_PROJECT_ID="$(cat "$SECRETS_DIR/project_id.txt")"
fi
: "${SDM_PROJECT_ID:?SDM_PROJECT_ID not set}"

WINDOW_SECONDS="${WINDOW_SECONDS:-90}"
SEG_SECONDS="${SEG_SECONDS:-$((WINDOW_SECONDS + 30))}"  # single segment
FIRST_EXT_DELAY_SEC="${FIRST_EXT_DELAY_SEC:-120}"
EXT_EARLY_SEC="${EXT_EARLY_SEC:-120}"
RTCP_PROBE_SECONDS="${RTCP_PROBE_SECONDS:-10}"

TS="$(date +%Y%m%d-%H%M%S)"
ROOT="/recordings/diag/timing_${TS}"
mkdir -p "$ROOT"
LOGFILE="$ROOT/diag_timing.log"

log() { printf "[%s] %s\n" "$(date -u '+%H:%M:%S.%3N')" "$*" | tee -a "$LOGFILE"; }

# ============================================================================
# Camera discovery (reused from diag_v7_2.sh)
# ============================================================================

sanitize_name() {
  local s="${1// /_}"
  s="${s//[^A-Za-z0-9_-]/_}"
  printf '%s' "$s" | tr -s '_'
}

short_id() {
  local id="${1##*/}"
  printf '%s' "${id: -6}"
}

discover_cameras() {
  local ACCESS_TOKEN
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  local DEV_JSON="$ROOT/devices.json"
  curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://smartdevicemanagement.googleapis.com/v1/enterprises/${SDM_PROJECT_ID}/devices" \
    -o "$DEV_JSON"

  declare -g -a CAM_NAME DEVICE_PATH CAM_LABEL CAM_DIR
  while IFS=$'\t' read -r name path; do
    CAM_NAME+=("$name")
    DEVICE_PATH+=("$path")
  done < <(jq -r '
    .devices[]?
    | select(.traits."sdm.devices.traits.CameraLiveStream"? != null)
    | ((if (.traits."sdm.devices.traits.Info".customName // "" | length) > 0
        then .traits."sdm.devices.traits.Info".customName
        else (.name | split("/") | last | .[-6:])
        end) + "\t" + .name)
  ' "$DEV_JSON")

  if [ "${#DEVICE_PATH[@]}" -eq 0 ]; then
    log "No cameras found"; exit 0
  fi

  declare -A seen_label
  for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    local sid lbl
    sid="$(short_id "${DEVICE_PATH[$i]}")"
    lbl="$sid"
    [[ -n "${seen_label[$lbl]:-}" ]] && lbl="${lbl}_$((++n))"
    seen_label["$lbl"]=1
    CAM_LABEL+=("$lbl")
    CAM_DIR+=("$ROOT/$lbl")
    mkdir -p "$ROOT/$lbl"
  done

  # Build camera_map.json — must use jq --arg for proper escaping
  local cams_arr="[]"
  for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    cams_arr=$(echo "$cams_arr" | jq \
      --arg label "${CAM_LABEL[$i]}" \
      --arg cam_id "$(short_id "${DEVICE_PATH[$i]}")" \
      --arg dev "${DEVICE_PATH[$i]}" \
      '. + [{"label":$label,"cam_id":$cam_id,"devicePath":$dev}]')
  done
  jq -n --arg ts "$TS" --arg root "$ROOT" --argjson cams "$cams_arr" \
    '{ts: $ts, root: $root, cameras: $cams}' > "$ROOT/camera_map.json"

  log "Discovered ${#DEVICE_PATH[@]} cameras"
  for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    log "  ${CAM_LABEL[$i]} → $(short_id "${DEVICE_PATH[$i]}")"
  done
}

# ============================================================================
# Per-camera capture worker (source PTS + timestamped stderr)
# ============================================================================

capture_one_camera() {
  local idx="$1"
  local dev="${DEVICE_PATH[$idx]}"
  local lbl="${CAM_LABEL[$idx]}"
  local sid="$(short_id "$dev")"
  local dir="${CAM_DIR[$idx]}"
  local deadline=$(($(date -u +%s) + WINDOW_SECONDS))

  log "[$lbl] starting capture (${WINDOW_SECONDS}s)"

  # --- Generate RTSP stream ---
  local ACCESS_TOKEN
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  local gen_epoch
  gen_epoch="$EPOCHREALTIME"

  local gen_json="$dir/generate_response.json"
  local http
  http=$(curl -s -w '%{http_code}' -o "$gen_json" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')

  if [ "$http" != "200" ]; then
    log "[$lbl] GenerateRtspStream failed (HTTP $http)"
    return 1
  fi

  local URL EXT_TOKEN STOP_TOKEN
  URL=$(jq -r '.results.streamUrls.rtspUrl // empty' "$gen_json")
  EXT_TOKEN=$(jq -r '.results.streamExtensionToken // empty' "$gen_json")
  STOP_TOKEN=$(jq -r '.results.streamToken // empty' "$gen_json")

  if [ -z "$URL" ]; then
    log "[$lbl] empty RTSP URL"
    return 1
  fi

  log "[$lbl] stream generated (gen_epoch=$gen_epoch)"

  # --- Segment muxer opts ---
  local SEG_OPTS=(-f segment -segment_time "$SEG_SECONDS" -strftime 1 -movflags +faststart)
  ffmpeg -hide_banner -h muxer=segment 2>&1 | grep -qi 'reset_timestamps' \
    && SEG_OPTS+=(-reset_timestamps 1)

  # --- Start ffmpeg with SOURCE PTS (no wallclock override) ---
  local out_tmpl="$dir/${sid}-%Y%m%d-%H%M%S.mp4"
  local stderr_fifo="$dir/_stderr_fifo"
  mkfifo "$stderr_fifo"

  # Timestamp every stderr line with host clock ($EPOCHREALTIME, µs precision)
  while IFS= read -r line; do
    printf "%s %s\n" "$EPOCHREALTIME" "$line"
  done < "$stderr_fifo" > "$dir/ffmpeg_ts.stderr" &
  local TS_PID=$!

  local ffmpeg_start_epoch="$EPOCHREALTIME"

  ffmpeg -hide_banner -loglevel info -nostdin -y \
    -rtsp_transport tcp \
    -copyts \
    -analyzeduration 10M -probesize 10M \
    -i "$URL" \
    -map 0:v:0 -map 0:a:0 \
    -vf showinfo \
    -c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30 \
    -c:a aac -ar 48000 -ac 1 -b:a 64k \
    -max_muxing_queue_size 1024 \
    -t "$((deadline - $(date -u +%s)))" \
    "${SEG_OPTS[@]}" \
    "$out_tmpl" \
    2> "$stderr_fifo" &
  local FFPID=$!

  # --- Extend loop ---
  (
    local next_sleep="$FIRST_EXT_DELAY_SEC"
    while kill -0 "$FFPID" 2>/dev/null; do
      sleep "$next_sleep"
      kill -0 "$FFPID" 2>/dev/null || break
      local at
      at="$(/app/get_access_token.sh | tr -d '\r')"
      local eresp
      eresp=$(curl -s -X POST \
        "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
        -H "Authorization: Bearer $at" \
        -H "Content-Type: application/json" \
        -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.ExtendRtspStream\",\"params\":{\"streamExtensionToken\":\"$EXT_TOKEN\"}}")
      local new_ext
      new_ext=$(echo "$eresp" | jq -r '.results.streamExtensionToken // empty' 2>/dev/null || true)
      [ -n "$new_ext" ] && EXT_TOKEN="$new_ext"
      next_sleep=240
    done
  ) &
  local EXT_PID=$!

  wait "$FFPID" || true
  local rc=$?
  kill "$EXT_PID" 2>/dev/null || true
  wait "$TS_PID" 2>/dev/null || true
  rm -f "$stderr_fifo"

  log "[$lbl] ffmpeg exited rc=$rc"

  # --- Stop stream ---
  if [ -n "${STOP_TOKEN:-}" ]; then
    local at2
    at2="$(/app/get_access_token.sh | tr -d '\r')" 2>/dev/null || true
    curl -s -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
      -H "Authorization: Bearer $at2" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$STOP_TOKEN\"}}" \
      >/dev/null 2>&1 || true
  fi

  # --- Extract sidecar + timing anchors ---
  extract_sidecar "$idx" "$gen_epoch" "$ffmpeg_start_epoch"
}

# ============================================================================
# Sidecar extraction with host-arrival timestamps + lower-envelope offset
# ============================================================================

extract_sidecar() {
  local idx="$1" gen_epoch="$2" ffmpeg_start_epoch="$3"
  local lbl="${CAM_LABEL[$idx]}"
  local sid="$(short_id "${DEVICE_PATH[$idx]}")"
  local dir="${CAM_DIR[$idx]}"
  local ts_stderr="$dir/ffmpeg_ts.stderr"

  [ ! -f "$ts_stderr" ] && { log "[$lbl] no timestamped stderr"; return; }

  # Find segment mp4(s)
  local mp4s_list
  mp4s_list=$(find "$dir" -maxdepth 1 -name "${sid}-*.mp4" -type f 2>/dev/null | sort)
  if [ -z "$mp4s_list" ]; then
    log "[$lbl] no mp4 segments found"; return
  fi
  local -a mp4s
  while IFS= read -r f; do mp4s+=("$f"); done <<< "$mp4s_list"

  # Extract (host_arrival, showinfo_n, source_pts) triples from timestamped stderr
  # Line format: "1783117473.123456 [Parsed_showinfo_0 @ ...] n:   0 pts: ... pts_time:0.033 ..."
  local pairs_file="$dir/_pairs.tmp"
  grep 'Parsed_showinfo.*pts_time:' "$ts_stderr" \
    | sed -n 's/^\([0-9.]*\) .*pts_time:\([0-9.eE+-]*\).*/\1 \2/p' \
    > "$pairs_file"

  local input_count
  input_count=$(wc -l < "$pairs_file" | tr -d ' ')

  if [ "$input_count" -eq 0 ]; then
    log "[$lbl] no showinfo data in stderr"
    rm -f "$pairs_file"
    return
  fi

  # For single-segment case (the expected path)
  local mp4="${mp4s[0]}"
  local output_count output_fps_frac
  output_count=$(ffprobe -hide_banner -select_streams v:0 \
    -show_entries stream=nb_frames -of csv=p=0 "$mp4" 2>/dev/null | tr -d ' ')
  output_fps_frac=$(ffprobe -hide_banner -select_streams v:0 \
    -show_entries stream=r_frame_rate -of csv=p=0 "$mp4" 2>/dev/null | tr -d ' ')
  [ -z "$output_count" ] || [ "$output_count" = "N/A" ] && output_count=0
  [ -z "$output_fps_frac" ] && output_fps_frac="30/1"

  local mismatch="false"
  [ "$input_count" -ne "$output_count" ] && mismatch="true"

  # Extract epoch from segment filename
  local base ymd_hms seg_epoch=0
  base=$(basename "$mp4" .mp4)
  ymd_hms=$(echo "$base" | grep -oE '[0-9]{8}-[0-9]{6}')
  [ -n "$ymd_hms" ] && seg_epoch=$(date -d \
    "${ymd_hms:0:4}-${ymd_hms:4:2}-${ymd_hms:6:2} ${ymd_hms:9:2}:${ymd_hms:11:2}:${ymd_hms:13:2}" \
    +%s 2>/dev/null || echo 0)

  local sidecar="${mp4%.mp4}.timing.jsonl"

  # Build sidecar with per-frame mapping + lower-envelope offset with drift check
  awk -v output_count="$output_count" \
      -v output_fps="$output_fps_frac" \
      -v seg_epoch="$seg_epoch" \
      -v input_count="$input_count" \
      -v mismatch="$mismatch" \
      -v pairs_file="$pairs_file" \
      -v anchors_file="$dir/timing_anchors.json" \
      -v gen_epoch="$gen_epoch" \
      -v ffmpeg_start="$ffmpeg_start_epoch" \
      -v window_sec="$WINDOW_SECONDS" \
  '
  BEGIN {
    # Read (host_arrival, source_pts) pairs
    ni = 0
    while ((getline line < pairs_file) > 0) {
      split(line, parts, " ")
      host_arr[ni] = parts[1] + 0.0
      src_pts[ni]  = parts[2] + 0.0
      ni++
    }
    close(pairs_file)

    # Normalize source PTS (subtract first value for segment-relative)
    base_pts = (ni > 0) ? src_pts[0] : 0
    for (k = 0; k < ni; k++) src_pts[k] -= base_pts

    # Parse fractional fps
    if (index(output_fps, "/") > 0) {
      split(output_fps, fp, "/")
      fps_val = fp[1] / fp[2]
    } else {
      fps_val = output_fps + 0.0
    }
    if (fps_val <= 0) fps_val = 30.0
    interval = 1.0 / fps_val

    # Measured fps
    if (ni > 1) {
      measured_fps = (ni - 1) / (src_pts[ni-1] - src_pts[0])
    } else {
      measured_fps = fps_val
    }

    # --- Lower-envelope offset with windowed drift check ---
    # Offset = host_arrival - source_pts (for each frame)
    # Global lower envelope = min offset
    global_min_offset = 1e18
    for (k = 0; k < ni; k++) {
      off = host_arr[k] - src_pts[k]
      if (off < global_min_offset) global_min_offset = off
    }

    # Windowed offsets (10s windows) for drift detection
    n_windows = 0
    win_size = 10.0
    if (ni > 0) {
      total_dur = src_pts[ni-1]
      if (total_dur < win_size) win_size = total_dur
    }
    # Compute per-window min offset
    max_windows = 30
    for (w = 0; w < max_windows && ni > 0; w++) {
      win_start = w * win_size
      win_end = (w + 1) * win_size
      if (win_start >= src_pts[ni-1]) break
      win_min = 1e18
      win_mid = (win_start + win_end) / 2.0
      found = 0
      for (k = 0; k < ni; k++) {
        if (src_pts[k] >= win_start && src_pts[k] < win_end) {
          off = host_arr[k] - src_pts[k]
          if (off < win_min) win_min = off
          found = 1
        }
      }
      if (found) {
        win_offsets[n_windows] = win_min
        win_mids[n_windows] = win_mid
        n_windows++
      }
    }

    # Drift: linear fit (least squares) of windowed offsets vs time
    drift_rate = 0
    if (n_windows >= 2) {
      sum_x = 0; sum_y = 0; sum_xx = 0; sum_xy = 0
      for (w = 0; w < n_windows; w++) {
        sum_x += win_mids[w]
        sum_y += win_offsets[w]
        sum_xx += win_mids[w] * win_mids[w]
        sum_xy += win_mids[w] * win_offsets[w]
      }
      denom = n_windows * sum_xx - sum_x * sum_x
      if (denom != 0) {
        drift_rate = (n_windows * sum_xy - sum_x * sum_y) / denom
      }
    }
    # drift_rate is seconds/second (ppm = drift_rate * 1e6)
    drift_flat = (n_windows < 2 || (drift_rate > -0.0001 && drift_rate < 0.0001)) ? "true" : "false"

    # PTS uniformity check
    sum_d = 0; sum_d2 = 0; nd = 0
    for (k = 1; k < ni; k++) {
      d = (src_pts[k] - src_pts[k-1]) * 1000
      sum_d += d; sum_d2 += d*d; nd++
    }
    if (nd > 0) {
      mean_d = sum_d / nd
      var_d = sum_d2/nd - mean_d*mean_d
      stdev_d = (var_d > 0) ? sqrt(var_d) : 0
    } else {
      mean_d = 0; stdev_d = 0
    }

    # --- Write metadata line ---
    printf "{\"_meta\":true,\"segment_start_epoch\":%s,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"measured_fps\":%.4f,\"mismatch\":%s,\"pts_wallclock_offset_s\":%.6f,\"offset_method\":\"lower_envelope\",\"drift_rate_s_per_s\":%.9f,\"drift_flat\":%s,\"drift_ppm\":%.3f,\"n_drift_windows\":%d,\"pts_mean_delta_ms\":%.4f,\"pts_stdev_delta_ms\":%.4f}\n", \
      seg_epoch, ni, output_count, fps_val, measured_fps, mismatch, \
      global_min_offset, drift_rate, drift_flat, drift_rate*1e6, n_windows, \
      mean_d, stdev_d

    # --- Two-pointer: map output frames to nearest input PTS ---
    j = 0
    for (i = 0; i < output_count; i++) {
      t_out = i * interval
      while (j + 1 < ni) {
        d_cur = src_pts[j] - t_out; if (d_cur < 0) d_cur = -d_cur
        d_nxt = src_pts[j+1] - t_out; if (d_nxt < 0) d_nxt = -d_nxt
        if (d_nxt <= d_cur) j++
        else break
      }
      printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"host_arrival_s\":%.6f,\"input_n\":%d}\n", \
        i, src_pts[j], host_arr[j], j
    }

    # --- Write timing_anchors.json ---
    printf "{\"generate_epoch\":%s,\"ffmpeg_start_epoch\":%s,\"pts_wallclock_offset_s\":%.6f,\"drift_rate_s_per_s\":%.9f,\"drift_ppm\":%.3f,\"drift_flat\":%s,\"measured_fps\":%.4f,\"pts_stdev_ms\":%.4f,\"input_frames\":%d,\"output_frames\":%d,\"mismatch\":%s,\"windowed_offsets\":[", \
      gen_epoch, ffmpeg_start, global_min_offset, drift_rate, drift_rate*1e6, \
      drift_flat, measured_fps, stdev_d, ni, output_count, mismatch \
      > anchors_file
    for (w = 0; w < n_windows; w++) {
      if (w > 0) printf "," > anchors_file
      printf "{\"t_mid\":%.1f,\"offset\":%.6f}", win_mids[w], win_offsets[w] > anchors_file
    }
    printf "]}\n" > anchors_file
  }
  ' > "$sidecar"

  rm -f "$pairs_file"

  local sidecar_lines
  sidecar_lines=$(( $(wc -l < "$sidecar") - 1 ))
  if [ "$mismatch" = "true" ]; then
    log "[$lbl] ⚠ MISMATCH sidecar: in=$input_count out=$output_count"
  else
    log "[$lbl] sidecar: $sidecar_lines/$output_count ✓"
  fi
  log "[$lbl] timing anchors written"
}

# ============================================================================
# RTCP hunt (separate probe streams — never contaminates anchor data)
# ============================================================================

rtcp_hunt() {
  local idx="$1"
  local dev="${DEVICE_PATH[$idx]}"
  local lbl="${CAM_LABEL[$idx]}"
  local dir="${CAM_DIR[$idx]}"

  log "[$lbl] RTCP hunt starting"

  # --- Probe A: TCP transport (current production mode) ---
  local ACCESS_TOKEN
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  local probe_json="$dir/_rtcp_probe.json"
  local http
  http=$(curl -s -w '%{http_code}' -o "$probe_json" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')

  if [ "$http" != "200" ]; then
    log "[$lbl] RTCP probe: stream generation failed (HTTP $http)"
    echo "stream_generation_failed" > "$dir/rtcp_tcp.log"
    echo "stream_generation_failed" > "$dir/rtcp_udp.log"
    return
  fi

  local PROBE_URL PROBE_STOP
  PROBE_URL=$(jq -r '.results.streamUrls.rtspUrl // empty' "$probe_json")
  PROBE_STOP=$(jq -r '.results.streamToken // empty' "$probe_json")

  # TCP probe
  log "[$lbl] RTCP probe: TCP transport (${RTCP_PROBE_SECONDS}s)"
  ffmpeg -hide_banner -loglevel trace -nostdin -y \
    -rtsp_transport tcp \
    -analyzeduration 5M -probesize 5M \
    -i "$PROBE_URL" \
    -map 0:v:0 -c:v copy \
    -t "$RTCP_PROBE_SECONDS" \
    -f null - \
    2> "$dir/rtcp_tcp.log" || true

  # Check for RTCP mentions
  local tcp_rtcp
  tcp_rtcp=$(grep -ciE 'rtcp|sender.report|SR ' "$dir/rtcp_tcp.log" 2>/dev/null || echo 0)
  log "[$lbl] RTCP TCP: $tcp_rtcp mentions in trace"

  # Stop TCP probe stream
  if [ -n "$PROBE_STOP" ]; then
    local at
    at="$(/app/get_access_token.sh | tr -d '\r')" 2>/dev/null || true
    curl -s -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
      -H "Authorization: Bearer $at" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$PROBE_STOP\"}}" \
      >/dev/null 2>&1 || true
  fi

  # --- Probe B: UDP transport ---
  ACCESS_TOKEN="$(/app/get_access_token.sh | tr -d '\r')"
  http=$(curl -s -w '%{http_code}' -o "$probe_json" \
    -X POST "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"command":"sdm.devices.commands.CameraLiveStream.GenerateRtspStream","params":{}}')

  if [ "$http" != "200" ]; then
    echo "stream_generation_failed_for_udp_probe" > "$dir/rtcp_udp.log"
    log "[$lbl] RTCP UDP: stream generation failed"
    rm -f "$probe_json"
    return
  fi

  PROBE_URL=$(jq -r '.results.streamUrls.rtspUrl // empty' "$probe_json")
  PROBE_STOP=$(jq -r '.results.streamToken // empty' "$probe_json")

  log "[$lbl] RTCP probe: UDP transport (${RTCP_PROBE_SECONDS}s)"
  ffmpeg -hide_banner -loglevel trace -nostdin -y \
    -rtsp_transport udp \
    -analyzeduration 5M -probesize 5M \
    -i "$PROBE_URL" \
    -map 0:v:0 -c:v copy \
    -t "$RTCP_PROBE_SECONDS" \
    -f null - \
    2> "$dir/rtcp_udp.log" || true

  local udp_result
  if grep -qiE 'connection refused|could not|error|fail' "$dir/rtcp_udp.log" 2>/dev/null; then
    udp_result="connection_refused"
    log "[$lbl] RTCP UDP: relay does not offer UDP → RTCP-over-separate-port unavailable by design"
  else
    local udp_rtcp
    udp_rtcp=$(grep -ciE 'rtcp|sender.report|SR ' "$dir/rtcp_udp.log" 2>/dev/null || echo 0)
    udp_result="connected_${udp_rtcp}_mentions"
    log "[$lbl] RTCP UDP: connected, $udp_rtcp RTCP mentions"
  fi

  # Stop UDP probe stream
  if [ -n "$PROBE_STOP" ]; then
    local at2
    at2="$(/app/get_access_token.sh | tr -d '\r')" 2>/dev/null || true
    curl -s -X POST \
      "https://smartdevicemanagement.googleapis.com/v1/${dev}:executeCommand" \
      -H "Authorization: Bearer $at2" \
      -H "Content-Type: application/json" \
      -d "{\"command\":\"sdm.devices.commands.CameraLiveStream.StopRtspStream\",\"params\":{\"streamToken\":\"$PROBE_STOP\"}}" \
      >/dev/null 2>&1 || true
  fi

  rm -f "$probe_json"
}

# ============================================================================
# Session manifest assembly
# ============================================================================

build_session_manifest() {
  local cameras_json="["
  for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
    local dir="${CAM_DIR[$i]}"
    local anchors="$dir/timing_anchors.json"
    local tcp_log="$dir/rtcp_tcp.log"
    local udp_log="$dir/rtcp_udp.log"

    local anchors_content="{}"
    [ -f "$anchors" ] && anchors_content=$(cat "$anchors")

    local tcp_rtcp="not_probed" udp_rtcp="not_probed"
    [ -f "$tcp_log" ] && tcp_rtcp=$(grep -ciE 'rtcp|sender.report' "$tcp_log" 2>/dev/null || echo 0)
    [ -f "$udp_log" ] && {
      if grep -qiE 'connection refused|could not|error|fail' "$udp_log" 2>/dev/null; then
        udp_rtcp="connection_refused"
      else
        udp_rtcp=$(grep -ciE 'rtcp|sender.report' "$udp_log" 2>/dev/null || echo 0)
      fi
    }

    [ "$i" -gt 0 ] && cameras_json+=","
    cameras_json+=$(jq -n \
      --arg label "${CAM_LABEL[$i]}" \
      --arg cam_id "$(short_id "${DEVICE_PATH[$i]}")" \
      --arg dev "${DEVICE_PATH[$i]}" \
      --argjson anchors "$anchors_content" \
      --arg rtcp_tcp "$tcp_rtcp" \
      --arg rtcp_udp "$udp_rtcp" \
      '{label:$label, cam_id:$cam_id, device_path:$dev, timing:$anchors, rtcp:{tcp:$rtcp_tcp, udp:$rtcp_udp}}')
  done
  cameras_json+="]"

  jq -n --arg ts "$TS" --arg root "$ROOT" --argjson cameras "$cameras_json" \
    '{session_ts:$ts, root:$root, cameras:$cameras}' \
    > "$ROOT/session_manifest.json"

  log "Session manifest written"
}

# ============================================================================
# Main orchestrator
# ============================================================================

discover_cameras

# --- Phase 1: Concurrent capture (all cameras, staggered start) ---
log "=== Phase 1: Concurrent capture (${WINDOW_SECONDS}s) ==="
pids=()
for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
  # Stagger stream generation by 2s to avoid SDM API rate-limit (404)
  sleep $(( i * 2 ))
  capture_one_camera "$i" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=$((fail+1)); fi
done
log "All captures complete. failures=$fail"

# --- Phase 2: RTCP hunt (sequential per camera, separate probe streams) ---
log "=== Phase 2: RTCP hunt ==="
for (( i=0; i<${#DEVICE_PATH[@]}; i++ )); do
  rtcp_hunt "$i"
done

# --- Phase 3: Session manifest ---
build_session_manifest

log "=== Done. Session: $ROOT ==="
log "To analyze: python tools/analyze_capture_timing.py analyze-session $ROOT"
