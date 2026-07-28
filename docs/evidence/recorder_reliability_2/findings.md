# RECORDER-RELIABILITY-2: Findings

Date: 2026-07-28

## Problem

RELIABILITY-1's faster failure detection (10s `-timeout`) + doubled API calls per attempt
(stop + generate) + faster retry cycle increased API traffic from ~0.75 calls/min to ~17
calls/min. This triggered HTTP 429 rate limiting from the SDM API, which the original slow
code never hit.

## SDM API Quota (evidence-based)

Source: developers.google.com/nest/device-access/project/limits

| Limit | Value | Scope |
|-------|-------|-------|
| **ExecuteDeviceCommand** | **10 QPM** | per project, per user (ALL devices, ALL commands) |
| Per-device (CAMERA) | 30 QPM / 100 QPH | per device instance, across projects |
| Per-command per-device | 5 QPM | per project, per user, per device |

**Binding constraint: 10 QPM user-project**, shared across all cameras and command types.

## Today's Test Evidence (hour-09, 45s window, 3 cameras)

**PPDmUg:**
- Attempt 1: 11s, timeout during startup (no data received)
- Attempt 2: 13s, recorded 405/405 exact sidecar at 15.0fps
- Attempt 3: Generate returned **429** ("Rate limited for the ExecuteDeviceCommand API")
- `stop_stream` returned **400** on attempts 2 and 3 (body: `"stream_token contains an invalid value"`)
  → session was already dead, stop was wasting an API call

**J_EDEw:**
- All 3 attempts: 10-11s, `Connection timed out` (camera offline/unreachable)
- `stop_stream` returned 400 on attempts 2 and 3 (same dead-session body)
- Classified as `connect_fail` — over a 30-min window this would be ~100 retries

**FP7oJQ:**
- Attempt 1: 47s, healthy run, sidecar MISMATCH: input=605 output=653 (8% fabricated, 13.85fps)

**Total API calls: 13 in ~46s = ~17 calls/min** (1.7x over 10 QPM budget)

## Fixes Applied

### 1. Optimistic URL Reuse
When ffmpeg exits after a healthy run and the session's extend expiry is still in the future,
the next attempt reuses the SAME RTSP URL with zero API calls. Only falls through to generate
if the reuse attempt fails quickly (<5s).

Extend loop publishes expiry epoch to `_extend_expiry.txt`. Main loop reads it for the reuse
decision.

### 2. Conditional stop_stream
`stop_stream` is ONLY called when we believe the session is still alive AND we're deliberately
abandoning it for a new session. When the session is dead (RTSP 404, session invalidated,
connection timeout with no data received), stop is skipped entirely.

The stop_stream 400 body (`"stream_token contains an invalid value"`) confirms dead sessions
don't need stopping. This eliminates 1 wasted API call per failed retry.

### 3. 429 Backoff Category
429 gets its own backoff starting at 60s, escalating to 300s (5 min cap). Generate captures
response headers (`-D`) and honors `Retry-After` if present. Extend loop also handles 429
with a 60s pause.

### 4. Generate 404 (device not found) — Non-Transient
After 3 consecutive device-not-found responses, the worker exits the main loop instead of
burning the entire window. Counter resets on any successful generate.

### 5. Consecutive Failure Escalation
After 5 consecutive failures of ANY type, backoff escalates to slow-poll mode (120-300s).
This prevents a down camera (like J_EDEw today) from consuming its share of the 10 QPM
quota with rapid retries. Over a 30-min window:
- Before: ~100 retries / ~100 API calls for an offline camera
- After: 5 quick retries → slow-poll at 120-300s → ~12-15 retries / ~12-15 API calls
Reset to normal on any success.

### 6. Cross-Camera Quota Coordination
- `N_CAMERAS` passed from v7_2 to each v6 worker
- Per-camera minimum retry interval computed dynamically: `60 / (10 QPM × 0.7 / N_cameras)`
  → with 3 cameras: min 26s between API calls per camera
- `BACKOFF_INITIAL` raised to 8s (was 3s); `BACKOFF_QUICK_MAX` raised to 25s (was 15s)
- Jitter (0-5s random) added to every backoff via `jittered_sleep()` to prevent
  cross-camera synchronization
- v7_2 already jitters startup (0-7s); this adds jitter to every retry

**Future improvement (not tonight):** shared token-bucket file or named pipe across workers
for hard coordination. Current approach sizes per-camera budgets independently.

### 7. Camera-Dependent Dup/Drop Caveat
FP7oJQ showed input=605 output=653 (8% fabricated frames) at 13.85fps under source-PTS,
while PPDmUg was exact 405/405 at 15.0fps. The RELIABILITY-1 verdict ("dup/drop resolved
by source PTS") is **camera-dependent**: cameras running at nominal 15fps show near-zero
mismatch; FP7oJQ's irregular 13.85fps cadence still produces significant mismatch under
source-PTS.

### Data-Received Heuristic Validation
`Parsed_showinfo` in ffmpeg stderr is the reliable indicator of whether frame data was
received:
- J_EDEw never-connected: 0 showinfo lines, 0 Opening lines
- PPDmUg attempt 1 (died at 11s, no data): 0 showinfo lines, 0 Opening lines
- PPDmUg attempt 2 (data received): 814 showinfo lines, 1 Opening line

`Opening.*\.mp4.*for writing` is NOT reliable — the segment muxer could open a file before
any real data arrives. `Parsed_showinfo` is strictly better.

## Estimated API Calls/Min (Post-Fix)

| Scenario | Calls/min | Within 10 QPM? | Headroom |
|----------|-----------|----------------|----------|
| All 3 healthy | ~1 (extends only) | Yes | 90% |
| 1 failing + 2 healthy | ~3-4 | Yes | 60-70% |
| All 3 failing (first 5 retries) | ~7 | Yes | 30% |
| All 3 failing (after escalation) | ~1-2 (slow-poll) | Yes | 80-90% |
| After 429 fires | drops to <1 (60s+ backoff) | Yes | 90%+ |
| J_EDEw offline 30 min | ~12-15 total calls | ~0.4/min | 96% |

## Tonight's Runbook

### Capture: 20:45 → 21:15 EST (30 minutes)

**Start (detached, from host):**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -d recorder \
  bash -lc 'WINDOW_SECONDS=1800 SEG_SECONDS=120 SOURCE_PTS=1 /app/diag_v7_2.sh'
```

**Verify running:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  ps aux | grep diag
```
Should show `diag_v7_2.sh` plus one `diag_v6.sh` per camera.

**Monitor live (optional):**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  tail -f /recordings/00000000-0000-0000-0000-000000000003/*/2026-07-28/*/run.log
```

**Confirm completion:**
After ~21:20 EST, check each camera's run.log for `[v6] done.`:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  tail -1 /recordings/00000000-0000-0000-0000-000000000003/*/2026-07-28/*/run.log
```

**Note:** Record the wall-clock time when gym lights go out.

**Post-capture: copy artifacts to host:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  ls /recordings/00000000-0000-0000-0000-000000000003/*/2026-07-28/*/run.log
```
Then `docker cp` or volume mount to inspect.

### What to check in the logs
1. "reusing URL, 0 API calls" — confirms reuse path is exercised
2. No `stop_stream HTTP=400` lines — confirms dead-session stop is skipped
3. "API budget: 10 QPM / 3 cameras" — confirms quota awareness
4. If J_EDEw offline: "slow-polling" after 5 failures — confirms escalation
5. No 429 errors — confirms API budget is respected
6. Sidecar mismatch lines per camera — note FP7oJQ vs PPDmUg
