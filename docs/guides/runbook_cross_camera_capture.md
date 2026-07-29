# Runbook: Cross-Camera Timing Capture (2026-07-29 evening)

## Purpose
Capture all 3 cameras simultaneously with source-PTS (true capture timestamps) during a
live session. Two alignment signals: (1) people visible in 2+ cameras (continuous
world-coordinate alignment), (2) gym lights out ~20:50–21:30 (instantaneous universal anchor).

## Window
**20:40 → 21:45 EDT** (65 minutes). `WINDOW_SECONDS=3900` covers this.

## Pre-flight Checklist

### 1. Disk space
Estimate: 3 cameras × 65 min × ~5 MB/min ≈ 975 MB. Verify:
```bash
df -h /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker/data/raw/nest/
```
Need at least 2 GB free.

### 2. Scheduler conflict check
The production scheduler (`diag_v8.sh`) must NOT fire during this window — two concurrent
`GenerateRtspStream` requests per camera may contend.
```bash
# Check what SCHED_DAILY_HHMM is set to in .env or diag_v8.sh defaults
grep SCHED_DAILY_HHMM services/nest_recorder/.env 2>/dev/null
# Default is 11:39 — should not conflict with a 20:40 start
# If in doubt, temporarily stop the production container before starting
```

### 3. Camera discovery
```bash
cd services/nest_recorder
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d recorder
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc '/app/list_cameras.sh'
# Expect: 3 cameras (J_EDEw, FP7oJQ, PPDmUg)
```

## Capture Command (DETACHED)

Use `nohup` + background so laptop sleep / terminal close won't kill it:

```bash
cd services/nest_recorder

# Start the container if not already running
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d recorder

# Run the capture DETACHED inside the container
# diag_v7_2.sh discovers all cameras and fans out to parallel diag_v6.sh workers
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -d recorder \
  bash -lc 'WINDOW_SECONDS=3900 SEG_SECONDS=120 SOURCE_PTS=1 /app/diag_v7_2.sh'
```

The `-d` flag detaches the exec so it survives terminal close.

## Verify It's Running

```bash
# Check for running ffmpeg processes inside the container
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'ps aux | grep ffmpeg | grep -v grep'
# Expect: 3 ffmpeg processes (one per camera)

# Check recording output
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'ls -la /recordings/diag/$(ls /recordings/diag/ | tail -1)/'
# Should show camera subdirectories with growing mp4 files

# Tail the log
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'tail -20 /recordings/diag/$(ls /recordings/diag/ | tail -1)/*/run.log 2>/dev/null'
```

## Confirm It Finished

After ~21:45 (or WINDOW_SECONDS elapsed):
```bash
# Check the session directory
ls -la data/raw/nest/diag/
# Find the latest timing_* or timestamped directory

# Check mp4 + sidecar counts per camera
find data/raw/nest/diag/<SESSION_DIR> -name "*.mp4" | wc -l
find data/raw/nest/diag/<SESSION_DIR> -name "*.timing.jsonl" | wc -l
# Expect: ~32 mp4s per camera (65 min / 2 min segments = ~33), same count of sidecars

# Quick validation: source PTS uniform?
head -1 data/raw/nest/diag/<SESSION_DIR>/<CAM>/first_segment.timing.jsonl
# Check: mismatch:false, measured_fps:~15 or ~30, pts_stdev_delta_ms:<5
```

## Manual Step: Note Lights-Out Time
**Write down the wall-clock time (to the second) when the gym lights actually go out.**
This is the discrete anchor for cross-camera alignment validation.

Format: `lights_out_epoch: <unix_timestamp>` or `lights_out: 2026-07-29 20:52:30 EDT`
Save to: `data/raw/nest/diag/<SESSION_DIR>/lights_out.txt`

## Post-Capture Analysis

```bash
cd /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker
source .venv/bin/activate

# Session timing analysis (per-camera offsets, drift, fps)
python tools/analyze_capture_timing.py analyze-session \
  data/raw/nest/diag/<SESSION_DIR>/

# Feed clips through pipeline for world-coordinate alignment
# (separate checkpoint — not part of this runbook)
```

## Troubleshooting

### A camera shows 0 segments
Check its `run.log` and `ffmpeg_attempt_*.stderr` for errors. Common: relay 404 (transient),
stream expired. The retry loop should recover — if ALL attempts failed, the camera was likely
offline.

### Mismatch warnings on sidecars
Under `SOURCE_PTS=1`, expect `mismatch:false`. If `mismatch:true` appears, the stream may
have delivered at a different rate than the encoder expected. The sidecar is still usable
(nearest-neighbor mapping) but note it for analysis.

### Container died
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d recorder
# Re-run with remaining time:
docker compose -f ... exec -d recorder \
  bash -lc 'WINDOW_SECONDS=<remaining_seconds> SEG_SECONDS=120 SOURCE_PTS=1 /app/diag_v7_2.sh'
```
