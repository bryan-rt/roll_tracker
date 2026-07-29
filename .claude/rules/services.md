---
paths:
  - "services/**"
---

# Docker Services

## nest_recorder
- OAuth2 → Nest API → MP4 segments + per-segment timing sidecars.
- Production path: `data/raw/nest/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/`. Diag (no GYM_ID):
  `data/raw/nest/diag/{TS}/`. GYM_ID presence is the mode switch.
- Auto-registers cameras to Supabase `cameras` table via REST upsert on discovery.
  `register_cameras.sh` called from `diag_v8.sh` after discovery, before recording.
- `entrypoint.sh` delegates to `diag_v8.sh` scheduler.
- **REENCODE modes:** `1` (default) and `2` = CFR libx264 + `-vf showinfo` timing sidecar
  (video byte-identical with/without showinfo, confirmed via MD5). `0` = VFR stream copy.
- **SOURCE_PTS flag** (default 0): when `SOURCE_PTS=1`, preserves camera's own RTP capture
  timestamps (`-copyts`, no `-use_wallclock_as_timestamps`). Adds per-line host-arrival
  timestamping via `$EPOCHREALTIME` stderr fifo, giving (source_PTS, host_arrival) pairs.
  Sidecar includes `host_arrival_s`, lower-envelope `pts_wallclock_offset_s`, windowed
  drift rate (ppm). Per-attempt stderr files handle retry loop (each stream generation =
  different PTS base + relay session). Passed through v7_2 → v6.
- **Per-segment timing sidecar (RECORDER-SIDECAR-1):** Every CFR segment mp4 gets a
  `.timing.jsonl` sibling. One row per OUTPUT frame, keyed on `frame_index` (matches
  FrameIterator's `cap.read()` counter — join key to Stage A). Each row carries the REAL
  input arrival PTS (bursty, NOT uniform CFR) via nearest-neighbor two-pointer mapping.
  Schema: `{_meta, segment_start_epoch, input_frame_count, output_frame_count, output_fps,
  mismatch}` header + `{frame_index, pts_time_s, input_n}` per frame.
  **MISMATCH is the normal condition** (input != output frame count). The CFR encoder
  always dups/drops when bursty input timing != uniform output spacing. When mismatched,
  `pts_time_s` is a NEAREST-NEIGHBOR APPROXIMATION, not exact input timing per output
  frame. Error grows with gap magnitude: mean ~80ms, P95 ~230ms, max ~500ms during lag
  windows (precisely where accuracy matters most — multiple output frames map to the same
  input PTS during gaps). Consumers must treat pts_time_s as approximate under mismatch;
  the input timing SEQUENCE is reliable for lag detection (gap presence/duration), but
  per-output-frame time attribution has ±500ms worst-case error.
  **MISMATCH warning** emitted loudly to recorder log. Sidecar is COLLECTION ONLY — no
  CV pipeline stage consumes it yet. Uploader is manifest-driven and ignores the sidecar.
- **Source PTS discovery (CAPTURE-TIME-1/2):** The RTSP stream carries TRUE capture
  timestamps via RTP (90kHz clock). The production recorder DISCARDS them with
  `-use_wallclock_as_timestamps 1 +genpts`, substituting bursty arrival times. Under
  source PTS (`-copyts`), timestamps are uniform ~33ms or ~67ms. Adoption would make the
  sidecar EXACT (no mismatch/approximation). Not yet in production.
- **Stream fps VARIES per session** (15fps and 30fps both observed from source PTS). SDP
  reports 30 when delivering 15. Different cameras differ. **Do not hardcode fps.**
- **RTCP absent** — 0 sender reports across all cameras on both TCP and UDP. Absolute
  camera-clock unavailable. Cross-camera sync via Tier-2: source PTS + host-clock lower
  envelope (±14–56ms).
- **Timing diagnostic module:** `diag_timing.sh` — parallel module, does NOT modify
  production recorder. Source PTS capture, per-frame (source_PTS, host_arrival) pairs,
  lower-envelope offset with windowed drift check, RTCP hunt. Analysis:
  `tools/analyze_capture_timing.py`.
- **RECORDER-RELIABILITY-1 (2026-07-28):** Five reliability fixes in `diag_v6.sh`:
  1. **RTSP socket timeout** (`-stimeout`, default 10s): ffmpeg exits within ~10s when
     data stops instead of blocking for 2+ min on OS TCP timeout. Configurable via
     `RTSP_TIMEOUT_SEC` env var. Top fix for recording gaps.
  2. **Stop stream before regenerating**: `stop_stream()` calls StopRtspStream (best-effort,
     5s timeout) before every `generate_stream` in the retry loop. Prevents orphaning
     RTSP sessions at the relay — the root cause of the 404 cascade observed on FP7oJQ
     and J_EDEw (SDM enforces concurrent-stream limit per camera).
  3. **Access token refresh per attempt**: `get_access_token` called before every generate
     and extend (cache hit unless expired — cheap). On Generate 401: auto-refresh + retry.
     Token expires after ~21-25 min; without refresh, the 65-min evening window would fail.
  4. **Failure-type-aware backoff**: healthy exit (>=60s) → immediate reconnect + reset;
     RTSP 404 → 10s→30s cap (moderate, since stop_stream prevents pileup); quick/unknown →
     3s→15s cap. Generate 401 → refresh + immediate retry.
  5. **Sidecar extraction backgrounded**: `extract_timing_sidecars` runs in background,
     PIDs waited at window end. Removes processing-time contribution to gaps.
  Evidence: `docs/evidence/recorder_reliability_1/`.
- **RECORDER-RELIABILITY-2 (2026-07-28):** API traffic reduction + quota awareness.
  RELIABILITY-1 increased API calls from ~0.75/min to ~17/min, triggering 429 rate limits.
  SDM quota: **10 QPM per user per project** (shared across ALL cameras and commands).
  Fixes: (1) **Optimistic URL reuse** — after healthy exit with valid extend expiry, restart
  ffmpeg on the same URL (0 API calls). Falls through to generate on quick failure.
  (2) **Conditional stop_stream** — only called when session is believed alive; dead sessions
  (RTSP 404, invalidated, timeout-no-data) skip stop entirely (confirmed: 400 body = "stream_token
  invalid"). (3) **429 backoff** — 60s start, 300s cap, honors Retry-After header. Applied to
  generate and extend. (4) **Generate 404 fail-fast** — device-not-found exits after 3 retries
  instead of burning the window. (5) **Consecutive failure escalation** — after 5 failures of
  any type, backoff escalates to 120-300s slow-poll; prevents offline cameras from consuming
  shared quota (J_EDEw: ~100 retries/30min → ~12-15). (6) **Cross-camera quota coordination** —
  `N_CAMERAS` passed from v7_2; per-camera min retry interval computed dynamically from
  `70% of 10 QPM / N`; jitter (0-5s) on every backoff; BACKOFF_INITIAL=8, BACKOFF_QUICK_MAX=25.
  Evidence: `docs/evidence/recorder_reliability_2/`.
- **ffmpeg option fragility:** `-stimeout` was removed in ffmpeg 7.x. The correct option is
  `-timeout` (same microsecond units). `debian:stable-slim` silently rolled bookworm→trixie.
  **Pin the Debian version in Dockerfiles** to prevent silent option invalidation on rebuild.
- **Source-PTS dup/drop verdict (TWO mechanisms, partially fixed):**
  1. **Bursty arrival timestamps** → ffmpeg mis-inferred frame rate → dup/drop. **FIXED** by
     source PTS (pixel-identical dups reduced 10-50x).
  2. **CFR encode target ≠ actual capture rate** → encoder pads to fill the grid. **STILL
     PRESENT.** FP7oJQ at 13.85fps against 15fps target → 8.3% fabricated (554→600). PPDmUg
     exact only because rate equals target (15.00fps). Since stream fps varies per session AND
     between cameras, any fixed CFR target will mismatch some camera some of the time.
  **OPEN CONTRADICTION:** RELIABILITY-1 reported FP7oJQ "0.0% typical" pixel-identical dups
  via `mpdecimate`, but sidecar reports 8% fabricated frames. CFR padding produces
  bit-identical frames — one measurement is wrong. Resolve before trusting either.
- **Timing capabilities (consolidated contract):**
  - **Relative per-frame timing: TRUE** (sensor clock, ~33ms/~67ms intervals, 1.21ms stdev).
  - **Absolute per-frame timing: ESTIMATED ±14–56ms** (recorder-side lower-envelope offset +
    per-camera drift correction; FP7oJQ −603 ppm ≈ 181ms/5min, linearly correctable).
  - **RTCP definitively ABSENT** (both TCP and UDP) → no absolute camera clock from stream.
  - **fps varies** per session AND per camera; SDP unreliable. Never hardcode fps.
  - **Sidecar schema:** `frame_index` (join key to Stage A), `pts_time_s`, `host_arrival_s`,
    `input_n` (consecutive same `input_n` = fabricated duplicate). Metadata: measured fps
    (4dp), lower-envelope offset, drift ppm, mismatch flag.
- **Remaining recorder work (open checkpoint):**
  1. Validate reuse path, token refresh past ~25-min barrier, sustained coverage, no 429s.
  2. Resolve 0.0%-vs-8% duplicate-measurement contradiction.
  3. Make `SOURCE_PTS=1` the DEFAULT (currently opt-in).
  4. Decide CFR-target-vs-actual-rate: per-clip CFR target, or accept padding and let
     pipeline flag duplicates via `input_n` (preferred — no recorder change needed).
  5. `N_CAMERAS` div-by-zero guard: `(10*7/10)/N` → 0 at N≥8 → `60/0` crash. Add floor of 1.
  6. Pin Debian version in Dockerfile.

## processor
- Polls `data/raw/nest/` for new MP4s, invokes bjj_pipeline A→F.
- Wall-clock filter: `MAX_CLIP_AGE_HOURS` (default 6) skips stale clips.
- Empty-video failures log as `clip_skipped` (not `clip_error`).
- **CP17 between-pass flow:** After Pass 1 D+E, builds tag evidence + coordinate evidence
  (if `cross_camera.coordinate_evidence.enabled`, default false), merges into overlay,
  re-solves each camera's ILP. Coordinate conflicts logged as `coordinate_conflict` events.
- Session state machine: `SCHEDULE_JSON` groups clips by gym schedule window. Writes
  `.phase1_complete_{cam_id}` / `.session_ready` / `.tag_required` sentinels.
  `.session_completed` prevents Phase 2 re-triggering.
- Config: SCAN_ROOT, OUTPUT_ROOT, POLL_INTERVAL_SECONDS, GYM_ID, MAX_CLIP_AGE_HOURS,
  SCHEDULE_JSON, SESSION_END_BUFFER_MINUTES.
- **Runs natively on Mac** (`run_local.sh`) — Docker ARM64 emulation too slow for YOLO.
  Docker compose processor service commented out; uncomment for Linux.
- MPS auto-detection: `device: "auto"` → MPS > CUDA > CPU. Phase 1 workers use CPU
  (parallel safety), Phase 2 uses MPS.
- Stale worker cleanup in `run_local.sh`: kills orphaned workers at startup and on trap.

## uploader
- Polls `outputs/`, bundles + uploads to Supabase.
- Resolves fighter tag_id → profile_id via active gym check-ins at upload time.
- Writes `global_person_id_a/b` from session export manifest to clips table.
- `.uploaded` sentinel written instead of deleting `export_manifest.jsonl`.
  `discover_manifests()` skips manifests with sentinel. Preserves processor guard.
- Skips `no_matches` manifests. Idempotent — re-runs must not duplicate.

## Contracts
- Processor: `services/processor/contracts/input_output.md`
- Uploader: `services/uploader/contracts/batch_bundle.md`

## Per-camera manifests
- Stage F writes `export_manifest_{cam_id}.jsonl` + `audit_{cam_id}.jsonl`.
- Processor merges per-camera manifests into `export_manifest.jsonl` after Loop 2.
- Stage E writes `match_sessions_{cam_id}.jsonl`, merged into `match_sessions.jsonl` after Loop 1.
