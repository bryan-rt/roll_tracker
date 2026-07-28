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
SOURCE_PTS="${SOURCE_PTS:-0}"               # 1 = preserve camera capture timestamps (no wallclock override)

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
  INPUT_WALLCLOCK=(-use_wallclock_as_timestamps 1)
  COPYTS_OPTS=()

  # SOURCE_PTS: preserve camera's own RTP capture timestamps instead of
  # substituting bursty network-arrival times. Proven in CAPTURE-TIME-1:
  # source PTS = uniform 33ms/67ms true capture cadence.
  if [ "$SOURCE_PTS" = "1" ]; then
    echo "[v6] SOURCE_PTS=1 → preserving camera capture timestamps (-copyts)" | tee -a "$LOG"
    INPUT_WALLCLOCK=()              # remove -use_wallclock_as_timestamps 1
    INPUT_FFLAGS="+igndts"          # drop +genpts — keep source PTS
    COPYTS_OPTS=(-copyts)           # preserve original timestamps
  fi

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
  # FrameIterator's cap.read() counter).
  # When SOURCE_PTS=1: includes host_arrival_s, lower-envelope offset, drift.
  # When SOURCE_PTS=0: nearest-neighbor two-pointer mapping (arrival-PTS).
  local stderr="${1:-$DIAG_DIR/ffmpeg.stderr}"
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
    local pairs_tmp="$DIAG_DIR/_pairs_${ATTEMPT}_${si}.tmp"

    if [ "$SOURCE_PTS" = "1" ]; then
      # SOURCE_PTS mode: extract (host_arrival, source_pts) pairs from timestamped stderr
      # Line format: "1785106319.123456 [Parsed_showinfo_0 @ ...] ... pts_time:0.033 ..."
      sed -n "${from_line},${to_line}p" "$stderr" \
        | grep 'Parsed_showinfo.*pts_time:' \
        | sed -n 's/^\([0-9.]*\) .*pts_time:\([0-9.eE+-]*\).*/\1 \2/p' \
        > "$pairs_tmp"
    else
      # Arrival-PTS mode: extract pts_time only (no host timestamp)
      sed -n "${from_line},${to_line}p" "$stderr" \
        | grep 'Parsed_showinfo.*pts_time:' \
        | sed -n 's/.*pts_time:\([0-9.eE+-]*\).*/0 \1/p' \
        > "$pairs_tmp"
    fi

    local input_count
    input_count=$(wc -l < "$pairs_tmp" | tr -d ' ')

    if [ "$input_count" -eq 0 ]; then
      log "[v6] ⚠ sidecar: $(basename "$seg_path") — no showinfo data, skipping"
      rm -f "$pairs_tmp"
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
      rm -f "$pairs_tmp"
      continue
    fi

    # Mismatch detection
    local mismatch="false"
    [ "$input_count" -ne "$output_count" ] && mismatch="true"

    local use_source_pts="$SOURCE_PTS"

    # Unified awk: handles both modes. In source-PTS mode, includes host_arrival_s,
    # lower-envelope offset, windowed drift check, measured fps.
    awk -v output_count="$output_count" \
        -v output_fps="$output_fps" \
        -v epoch="$epoch" \
        -v input_count="$input_count" \
        -v mismatch="$mismatch" \
        -v pairs_file="$pairs_tmp" \
        -v source_pts_mode="$use_source_pts" \
        -v attempt="$ATTEMPT" \
    '
    BEGIN {
      # Read (host_arrival, source_pts) pairs
      ni = 0
      while ((getline line < pairs_file) > 0) {
        split(line, parts, " ")
        host_arr[ni] = parts[1] + 0.0
        raw_pts[ni]  = parts[2] + 0.0
        ni++
      }
      close(pairs_file)

      # Sort by PTS (insertion sort — handles rare B-frame reorder)
      for (a = 1; a < ni; a++) {
        kp = raw_pts[a]; kh = host_arr[a]
        b = a - 1
        while (b >= 0 && raw_pts[b] > kp) {
          raw_pts[b+1] = raw_pts[b]; host_arr[b+1] = host_arr[b]
          b--
        }
        raw_pts[b+1] = kp; host_arr[b+1] = kh
      }

      # Normalize PTS: segment-relative
      base_pts = (ni > 0) ? raw_pts[0] : 0
      for (k = 0; k < ni; k++) raw_pts[k] -= base_pts

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
      measured_fps = fps_val
      if (ni > 1 && raw_pts[ni-1] > 0) {
        measured_fps = (ni - 1) / raw_pts[ni-1]
      }

      # PTS uniformity stats
      sum_d = 0; sum_d2 = 0; nd = 0
      for (k = 1; k < ni; k++) {
        d = (raw_pts[k] - raw_pts[k-1]) * 1000
        sum_d += d; sum_d2 += d*d; nd++
      }
      mean_d = (nd > 0) ? sum_d / nd : 0
      stdev_d = (nd > 0) ? sqrt(sum_d2/nd - mean_d*mean_d) : 0

      # --- Lower-envelope offset + windowed drift (SOURCE_PTS only) ---
      global_min_offset = 0; drift_rate = 0; drift_flat = "true"; drift_ppm = 0
      n_windows = 0

      if (source_pts_mode == "1" && ni > 0) {
        global_min_offset = 1e18
        for (k = 0; k < ni; k++) {
          off = host_arr[k] - raw_pts[k]
          if (off < global_min_offset) global_min_offset = off
        }

        # Windowed offsets (10s windows)
        win_size = 10.0
        for (w = 0; w < 60 && ni > 0; w++) {
          ws = w * win_size; we = (w + 1) * win_size
          if (ws >= raw_pts[ni-1]) break
          wmin = 1e18; wmid = (ws + we) / 2.0; found = 0
          for (k = 0; k < ni; k++) {
            if (raw_pts[k] >= ws && raw_pts[k] < we) {
              off = host_arr[k] - raw_pts[k]
              if (off < wmin) wmin = off
              found = 1
            }
          }
          if (found) {
            win_off[n_windows] = wmin; win_mid[n_windows] = wmid
            n_windows++
          }
        }

        # Linear fit for drift
        if (n_windows >= 2) {
          sx=0;sy=0;sxx=0;sxy=0
          for (w=0;w<n_windows;w++) {
            sx+=win_mid[w]; sy+=win_off[w]
            sxx+=win_mid[w]*win_mid[w]; sxy+=win_mid[w]*win_off[w]
          }
          den = n_windows*sxx - sx*sx
          if (den != 0) drift_rate = (n_windows*sxy - sx*sy) / den
        }
        drift_ppm = drift_rate * 1e6
        drift_flat = (n_windows<2 || (drift_rate>-0.0001 && drift_rate<0.0001)) ? "true" : "false"
      }

      # --- Metadata line ---
      if (source_pts_mode == "1") {
        printf "{\"_meta\":true,\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"measured_fps\":%.4f,\"mismatch\":%s,\"pts_wallclock_offset_s\":%.6f,\"offset_method\":\"lower_envelope\",\"drift_rate_s_per_s\":%.9f,\"drift_flat\":%s,\"drift_ppm\":%.3f,\"n_drift_windows\":%d,\"pts_mean_delta_ms\":%.4f,\"pts_stdev_delta_ms\":%.4f}\n", \
          epoch, attempt, ni, output_count, fps_val, measured_fps, mismatch, \
          global_min_offset, drift_rate, drift_flat, drift_ppm, n_windows, mean_d, stdev_d
      } else {
        printf "{\"_meta\":true,\"segment_start_epoch\":%s,\"attempt\":%d,\"input_frame_count\":%d,\"output_frame_count\":%d,\"output_fps\":%.4f,\"measured_fps\":%.4f,\"mismatch\":%s}\n", \
          epoch, attempt, ni, output_count, fps_val, measured_fps, mismatch
      }

      # --- Two-pointer: map output frames to nearest input PTS ---
      j = 0
      for (i = 0; i < output_count; i++) {
        t_out = i * interval
        while (j + 1 < ni) {
          d_cur = raw_pts[j] - t_out; if (d_cur < 0) d_cur = -d_cur
          d_nxt = raw_pts[j+1] - t_out; if (d_nxt < 0) d_nxt = -d_nxt
          if (d_nxt <= d_cur) j++
          else break
        }
        if (source_pts_mode == "1") {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"host_arrival_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[j], host_arr[j], j
        } else {
          printf "{\"frame_index\":%d,\"pts_time_s\":%.6f,\"input_n\":%d}\n", \
            i, raw_pts[j], j
        }
      }
    }
    ' > "$sidecar"

    rm -f "$pairs_tmp"

    # Summary line with loud mismatch warning + measured fps
    local sidecar_lines
    sidecar_lines=$(( $(wc -l < "$sidecar") - 1 ))
    local fps_info=""
    fps_info=$(head -1 "$sidecar" | grep -oE '"measured_fps":[0-9.]+' | cut -d: -f2)

    if [ "$mismatch" = "true" ]; then
      log "[v6] ⚠ MISMATCH sidecar: $(basename "$sidecar") input=$input_count output=$output_count fps=${fps_info:-?}"
    else
      log "[v6] sidecar: $(basename "$sidecar") $sidecar_lines/$output_count ✓ fps=${fps_info:-?} (epoch=$epoch)"
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

  # Per-attempt stderr file (supports retry loop — each attempt gets its own file)
  local stderr_file="$DIAG_DIR/ffmpeg_attempt_${ATTEMPT}.stderr"
  TS_PID=""

  if [ "$SOURCE_PTS" = "1" ]; then
    # Host-arrival timestamping: pipe stderr through fifo, prepend $EPOCHREALTIME
    local stderr_fifo="$DIAG_DIR/_stderr_fifo_${ATTEMPT}"
    mkfifo "$stderr_fifo"
    while IFS= read -r line; do
      printf "%s %s\n" "$EPOCHREALTIME" "$line"
    done < "$stderr_fifo" > "$stderr_file" &
    TS_PID=$!
    local stderr_target="$stderr_fifo"
    log "[v6] attempt $ATTEMPT: source-PTS mode, host-arrival timestamping → $(basename "$stderr_file")"
  else
    local stderr_target="$stderr_file"
  fi

  # Record attempt metadata
  printf '{"attempt":%d,"generate_epoch":"%s","ffmpeg_start_epoch":"%s"}\n' \
    "$ATTEMPT" "$(cat "$DIAG_DIR/generated_at_epoch.txt" 2>/dev/null || echo 0)" \
    "$EPOCHREALTIME" \
    >> "$DIAG_DIR/attempt_log.jsonl"

  ffmpeg -hide_banner -loglevel info -nostdin -y \
    -rtsp_transport tcp \
    "${COPYTS_OPTS[@]}" \
    "${INPUT_WALLCLOCK[@]}" -fflags "$INPUT_FFLAGS" -avoid_negative_ts make_zero \
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
    1> "$DIAG_DIR/ffmpeg.stdout" 2> "$stderr_target" &
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

  # Clean up stderr fifo + timestamper for this attempt
  if [ -n "${TS_PID:-}" ] && [ "$TS_PID" != "" ]; then
    wait "$TS_PID" 2>/dev/null || true
    TS_PID=""
  fi
  rm -f "$DIAG_DIR/_stderr_fifo_${ATTEMPT}"

  # Extract per-segment timing sidecars from THIS ATTEMPT's stderr
  extract_timing_sidecars "$DIAG_DIR/ffmpeg_attempt_${ATTEMPT}.stderr"

  # Done if time is up
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "[v6] window elapsed after attempt #$ATTEMPT"; break; }

  # If ffmpeg ended early, back off and try to recover with a fresh stream
  log "[v6] preparing next attempt (backoff ${BACKOFF}s)"
  sleep "$BACKOFF"
  [ "$BACKOFF" -lt 60 ] && BACKOFF=$(( BACKOFF * 2 ))
done

log "[v6] done. Artifacts in $DIAG_DIR"
