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
