# CLAUDE.md — Roll Tracker

This file is the persistent context bridge for Claude Code CLI. Every `claude` session
in this repo reads this file automatically. Keep it current after significant changes.

---

## Working Methodology

This project uses a "Web-Brain / CLI-Hands" collaboration model. Read this section
before starting any non-trivial task.

### Your Role (CLI)
The CLI owns the *how*. The web session (claude.ai) owns the *what* and *why*.
Every task arrives as a Task Brief from the web session. Do not make architectural
decisions independently — if something in the Task Brief is ambiguous or conflicts
with the codebase, pause and surface it before proceeding.

### The Three-Pass Protocol (required for all non-trivial tasks)

**Always start a new session with Plan Mode: hit `shift+tab` twice before doing anything.**

```
PASS 1 — Explore (Plan Mode)
  Read the Task Brief carefully
  Explore the relevant files in the repo
  Understand the actual current state of the code
  Identify any conflicts with the Task Brief or CLAUDE.md conventions
  ⏸ STOP — summarize findings and wait for user approval before Pass 2

PASS 2 — Specify (Plan Mode continues)
  Plan the exact changes needed
  Verify naming conventions against live code
  Check data contracts (F0 layer) for compatibility
  Resolve edge cases using evidence from the actual files
  ⏸ STOP — present the plan and wait for user approval before Pass 3

PASS 3 — Execute
  Implement the approved plan
  Run tests or validate pipeline output
  Update CLAUDE.md if architecture or conventions changed
  Commit with a descriptive message and push to GitHub
  ⏸ STOP — summarize what was done and wait for user review
```

**Never skip a pause.** User approval gates each pass. Do not run Pass 2 immediately
after Pass 1, and do not run Pass 3 immediately after Pass 2.

### Evidence-Driven Design

We do not code based on assumptions. When behavior is uncertain:
1. Prefer to enhance logging and collect real artifacts first
2. Inspect existing parquet/JSONL outputs before designing a fix
3. Run the pipeline with debug flags and examine the output
4. Plan from evidence, not speculation

If a Task Brief asks you to fix something but the root cause is unclear, say so.
Propose an instrumentation step before a fix step. Speculation is a last resort.

### What to Do With Ambiguity
- Naming conflict with existing code? Surface it in Pass 1, don't resolve it silently.
- Task Brief references a file that doesn't exist? Report it, don't create it unilaterally.
- Architectural question not covered in CLAUDE.md? Pause and flag it — don't guess.

---

## Project Identity

**Name:** Roll Tracker
**Author:** Bryan Thomas
**GitHub:** https://github.com/bryan-rt/roll_tracker
**Status:** POC → MVP transition
**Goal:** Multi-service SaaS pipeline for BJJ gyms. Streams Nest camera footage, aligns
it to a gym mat blueprint via homography, tracks athletes (YOLO + BoT-SORT), uses
AprilTags for online identity labeling, and ILP/MCF for offline match stitching.
Produces per-athlete match clips persisted to Supabase and queryable via a Flutter app.

---

## Monorepo Layout

```
roll_tracker/
├── src/bjj_pipeline/       # Core CV pipeline (Python package, installable via pyproject.toml)
│   ├── config/             # Config loading + Pydantic models
│   ├── contracts/          # F0: manifest, parquet schemas, path layouts, validators
│   ├── core/               # Frame iterator, IO, logging, timebase, shared types
│   ├── stages/
│   │   ├── detect_track/   # Stage A: YOLO detection + BoT-SORT tracklets
│   │   ├── masks/          # Stage B: SAM masks + refined geometry (deferred for POC)
│   │   ├── tags/           # Stage C: AprilTag scheduling, scanning, identity voting
│   │   ├── stitch/         # Stage D: MCF/ILP global identity stitching (D0–D4)
│   │   ├── matches/        # Stage E: Match session detection
│   │   ├── export/         # Stage F: Clip export, ffmpeg, Supabase DB write
│   │   └── orchestration/  # CLI entry point, stage registry, resume logic
│   ├── eval/               # Metrics + sanity checks
│   ├── tools/              # Calibration utilities (homography)
│   └── viz/                # Debug visualizers, overlay writers
├── src/calibration_pipeline/ # Gym setup + maintenance calibration tools (CP16b+)
│   ├── lens_calibration.py  # Interactive lens K+dist estimation from mat edge clicks
│   ├── mat_walk.py          # Stub — grid pattern detection from tagged walker (CP18)
│   ├── drift_detection.py   # Stub — daily baseline comparison for camera drift (CP18)
│   └── inter_camera_sync.py # Stub — cross-camera affine alignment via mat walk (CP18)
├── services/
│   ├── nest_recorder/      # Docker: Google Nest API → MP4 segments to data/raw/
│   ├── processor/          # Docker: placeholder — wraps bjj_pipeline for service context
│   └── uploader/           # Docker: polls outputs/, uploads to Supabase, deletes on confirm
├── backend/
│   └── supabase/supabase/
│       ├── config.toml
│       └── migrations/     # SQL schema (see Database Schema section)
├── app_mobile/             # Flutter mobile app (Supabase + video_player)
├── app_web/                # Web app (Vite + React, Supabase auth, react-router-dom, admin pricing dashboard)
├── bin/run_pipeline.py     # Legacy dev runner (use CLI instead)
├── configs/
│   ├── default.yaml        # Safe mechanical defaults
│   ├── cameras/cam01.yaml  # Per-camera overrides
│   └── cameras/cam0N/
│       └── homography.json # Per-camera homography matrix
├── data/raw/nest/          # Raw MP4 segments, gitignored
├── outputs/                # Pipeline outputs, gitignored
├── tools/                  # Repo-level dev/debug scripts
├── requirements.txt        # Pinned runtime deps (Python 3.12)
└── pyproject.toml          # Package definition (hatchling build)
```

---

## CV Pipeline Stages (A → F)

The pipeline has two phases:

**Phase 1 — Online single-pass (multiplex, per-clip)**
- **Stage A** `detect_track`: YOLO detection + BoT-SORT tracking. Outputs
  `detections.parquet`, `tracklet_frames.parquet`, `tracklet_summaries.parquet`,
  `contact_points.parquet`, `audit.jsonl`.
- **Stage B** `masks`: SAM-based refined masks. Currently deferred for POC.
  Falls back to YOLO bbox masks.
- **Stage C** `tags`: AprilTag identity anchoring.
  - C0: scheduling/cadence — scannability map + gating + trigger logic
  - C1: ROI scan + raw tag observations
  - C2: voting + conflict resolution → identity hints
  - Outputs: `tag_observations.jsonl`, `identity_hints.jsonl`

**Phase 2 — Offline multi-pass**
- **Stage D** `stitch`: Global MCF/ILP identity stitching. Sub-steps D0–D4:
  - D0: tracklet bank tables
  - D1: graph build (merge/split triggers, group spans)
  - D2: constraint generation
  - D3: ILP compile + solve (two solvers: `d3_ilp`, `d3_ilp2`)
  - D4: emit resolved person_tracks
  - Uses Google OR-Tools for ILP solver backend.
- **Stage E** `matches`: Two-layer engagement detection. Sub-steps E0–E6:
  - E0: plumbing + input validation (person_spans optional, person_tracks optional, both missing = error)
  - E1: cap2 GROUP seed extraction (from person_spans)
  - E2: proximity hysteresis state machine (from person_tracks world coords)
  - E3: union seeds + proximity intervals, apply clip buffer
  - E4: buzzer soft gate (optional, from audio_events.jsonl when Stage A.5 built)
  - E5: minimum duration filter + emit (zero matches = valid, no exception)
  - E6: identity enrichment (April tag labels from Stage D)
  - Outputs: `match_sessions.jsonl`, `audit.jsonl`
- **Stage F** `export`: ffmpeg clip cutting, redaction, Supabase DB write, manifest.

**Orchestration CLI:**
```bash
python -m bjj_pipeline.stages.orchestration.cli run   --input data/raw/nest/cam03/... --camera cam03
python -m bjj_pipeline.stages.orchestration.cli status --clip-id <clip_id>
python -m bjj_pipeline.stages.orchestration.cli validate --clip-id <clip_id>
```
Resume logic is config-hash-aware: stages only re-run if required outputs are missing
or config changed.

---

## CV Design Constraints

**AprilTag family: 36h11** (~587 distinct IDs). Family selected to maximize
cell size within an 11x11 inch physical tag printed on athlete apparel. Larger
cells improve detection reliability for fixed Nest cameras operating at gym
distances under real conditions: variable resolution, lens distortion, partial
occlusion common in BJJ. Detection range directly affects the density of
tag observations fed to the Stage D ILP solver — more observations = stronger
identity constraints = better stitching quality.

**Do not upgrade tag family** without re-evaluating detection reliability.
A larger family (e.g. tagStandard41h12) means smaller cells at the same
physical print size, which reduces the effective detection radius per camera.

**Scale beyond 587 athletes:** handled via WiFi-based gym check-in, not tag
family migration. `tag_id` is unique within `(tag_id + gym_id + active session)`.
The schema supports collision gracefully — Stage F uses check-in records to
disambiguate when multiple athletes globally share a `tag_id`.

---

## Data Contracts (F0 Layer)

All inter-stage data lives on disk under `outputs/{gym_id}/{cam_id}/{date}/{hour}/{clip_id}/`
(gym-scoped) or `outputs/legacy/{cam_id}/{date}/{hour}/{clip_id}/` (legacy). The F0 layer enforces this:
- `f0_manifest.py` — `ClipManifest` Pydantic model (includes `gym_id: Optional[str]`), init/load/write, per-stage default registration
- `f0_paths.py` — `ClipOutputLayout`, `SessionOutputLayout`, `StageLetter` — canonical path resolution. `SessionOutputLayout` (CP14a) is a frozen dataclass for session-level (multi-clip, multi-camera) outputs under `outputs/{gym_id}/sessions/{date}/{session_id}/`. CP14f adds `session_cross_camera_identities_jsonl()` → `stage_D/cross_camera_identities.jsonl`.
- `f0_parquet.py` — Parquet read/write helpers
- `f0_models.py` — Shared Pydantic models
- `f0_projection.py` — `project_to_world()` canonical pixel→world projection utility (CP16a). `CameraProjection` NamedTuple, `load_calibration_from_payload()` helper. Strict enforcement: no stage calls homography directly.
- `f0_validate.py` — Post-stage validators

**Ingest path parsing:** `validate_ingest_path()` in `pipeline.py` returns `IngestPathInfo`
(namedtuple: `gym_id`, `cam_id`, `date_str`, `hour_str`). `compute_output_root()` converts
this into the gym-scoped output root path. Both are used by `run_pipeline()`, CLI commands
(`status`, `validate`), and the processor service.

**Rule:** Stages communicate only via the manifest + filesystem. No stage imports another
stage's internals directly.

---

## Docker Services

| Service | Status | Responsibility |
|---|---|---|
| `nest_recorder` | Working | OAuth2 → Nest API → MP4 segments. Production path: `data/raw/nest/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/`. Diag path (no GYM_ID): `data/raw/nest/diag/{TS}/`. Auto-registers cameras to Supabase. `entrypoint.sh` delegates to `diag_v8.sh` scheduler. |
| `processor` | Working | Polls `data/raw/nest/` for new MP4s, invokes `bjj_pipeline` (A→F). Wall-clock filter (`MAX_CLIP_AGE_HOURS`, default 6) skips stale clips. Empty-video failures log as `clip_skipped` (not `clip_error`). Session state machine (CP14a): when `SCHEDULE_JSON` is set, groups clips by gym schedule window, writes `.phase1_complete_{cam_id}` / `.session_ready` / `.tag_required` sentinels under `SessionOutputLayout`. Config: `SCAN_ROOT`, `OUTPUT_ROOT`, `POLL_INTERVAL_SECONDS`, `GYM_ID`, `MAX_CLIP_AGE_HOURS`, `SCHEDULE_JSON`, `SESSION_END_BUFFER_MINUTES`. |
| `uploader` | Working | Polls `outputs/`, bundles + uploads to Supabase, writes `gym_id` to `videos` row from export manifest, resolves fighter tag IDs → profile IDs via active gym check-ins, skips `no_matches` manifests, deletes on confirm |

The processor service has a documented I/O contract at `services/processor/contracts/input_output.md`.
The uploader contract is at `services/uploader/contracts/batch_bundle.md`.
Idempotency is critical for the uploader — re-runs must not duplicate uploads.

---

## Supabase Schema (current migrations)

**Tables:**
- `profiles` — athlete/user records. `auth_user_id` (FK to Supabase Auth), `display_name` (nullable), `email`, `tag_id` (auto-assigned 0–586 via sequence), `tag_assigned_at`, `home_gym_id` FK→gyms
- `videos` — raw video metadata: `camera_id`, `source_path`, `recorded_at`, `status`, `metadata` (jsonb), `gym_id` FK→gyms
- `clips` — processed clips: `video_id` FK, `match_id`, `file_path`, `storage_bucket`, `storage_object_path`, `start_seconds`, `end_seconds`, `fighter_a_tag_id`, `fighter_b_tag_id`, `fighter_a_profile_id`, `fighter_b_profile_id` (nullable FKs→profiles), `global_person_id_a`, `global_person_id_b` (cross-camera identity, CP14f)
- `log_events` — audit log: `clip_id`/`video_id` FK, `event_type`, `event_level`, `message`, `details`
- `gyms` — `name`, `owner_profile_id`, `owner_auth_user_id` (denormalized), `address`, `wifi_ssid`, `wifi_bssid`, `latitude`, `longitude`
- `gym_checkins` — `profile_id`, `gym_id`, `checked_in_at`, `auto_expires_at` (trigger-managed +3hr, slides on upsert), `is_active`, `source` (`manual` or `wifi_auto`). Unique on `(profile_id, gym_id)` — enables upsert for sliding TTL.
- `gym_subscriptions` — `gym_id`, `tier` ENUM, `started_at`, `ended_at`, `is_current`
- `cameras` — `gym_id` FK→gyms, `cam_id` (last 6 chars of SDM device path), `device_path` (full SDM path), `display_name` (nullable, from Google Home room name), `is_active`, `first_seen_at`, `last_seen_at`. Unique on `(gym_id, cam_id)`. Auto-registered by `nest_recorder` on camera discovery via Supabase REST upsert.
- `homography_configs` — `gym_id`, `camera_id`, `config_data` JSONB
- `gym_interest_signals` — `profile_id`, `gym_name_entered`, `owner_email`, `submitted_at`
- `device_tokens` — `profile_id` FK→profiles, `token` (FCM token), `platform` (default `android`). Unique on `(profile_id, token)`. RLS: athletes manage own tokens.

**Storage bucket:** `match-clips` (private, RLS policy allows authenticated reads for signed URLs)

**Auth trigger:** `handle_new_user()` fires on `auth.users` INSERT — auto-creates `profiles` row with `auth_user_id`, `email`, `tag_id` (from cycling sequence 0–586), `tag_assigned_at`.

**Helper functions:**
- `gyms_near(lat, lng, radius_km)` — Haversine proximity search, no PostGIS
- `current_profile_id()` — SECURITY DEFINER helper for RLS policies that need the current user's profile ID without recursion
- `get_claimable_clips(p_tag_id, p_gym_id, p_window_hours)` — SECURITY DEFINER RPC returns clips with unresolved profile_ids for a tag+gym within a time window
- `claim_clip(p_clip_id, p_fighter_side)` — SECURITY DEFINER RPC sets `fighter_{a|b}_profile_id` to current user's profile and updates status to `'uploaded'`. IS NULL guard prevents overwriting existing claims.

**RLS:** Enabled on all 11 tables. Athletes see own profile/clips/check-ins. Gym owners see their gym's data. Service role bypasses all RLS. Note: the gym-owner-reads-checked-in-athlete-profiles policy was dropped due to cross-table RLS recursion (42P17) — will be re-implemented as a SECURITY DEFINER RPC function.

**Pending schema items (not yet migrated):**
- `notification_channel` — TBD (drift alert delivery mechanism)
- Gym owner profile read policy — needs RPC-based approach to avoid RLS recursion
- K + distortion coefficients — per-camera lens calibration parameters (camera matrix K, distortion coefficients k1,k2,p1,p2). Exact schema to be designed in CP16b. Likely extends `homography_configs` or adds a new `camera_calibrations` table.
- Drift scores — per-camera daily drift score, baseline snapshot reference, alert status. Designed in CP16b skeleton.

**Applied migrations (Phase A):**
- `20260311000001_create_gyms.sql` — `gyms` table
- `20260311000002_create_gym_members.sql` — (**superseded by 000007** — dropped)
- `20260311000003_create_gym_subscriptions.sql` — `gym_subscriptions` table
- `20260311000004_create_gym_checkins.sql` — `gym_checkins` table + `set_checkin_expiry` trigger
- `20260311000005_create_homography_configs.sql` — `homography_configs` table
- `20260311000006_add_phase_a_columns.sql` — `tag_id`, `gym_id`, `fighter_*_profile_id` columns
- `20260311000007_phase_a_correction.sql` — drops `gym_members`, adds `home_gym_id`, creates `gym_interest_signals`

**Applied migrations (Phase E + bug fixes):**
- `20260315000001_phase_e_rls_and_trigger.sql` — `display_name` nullable, `owner_email` column, auth trigger, `gyms_near()`, RLS on all tables
- `20260315000002_fix_profiles_update_policy.sql` — adds WITH CHECK to profiles UPDATE policy
- `20260315000003_fix_profiles_select_recursion.sql` — `current_profile_id()` SECURITY DEFINER helper
- `20260315000004_fix_current_profile_id_lang.sql` — switches helper to plpgsql to prevent inlining
- `20260315000005_fix_profiles_recursion_v3.sql` — `owner_auth_user_id` denormalized on gyms
- `20260315000006_drop_recursive_profile_policy.sql` — drops recursive gym-owner profiles policy
- `20260315000007_checkin_source_and_tag_assignment.sql` — `source` column on gym_checkins, `tag_id_seq` cycling sequence (0–586), updated `handle_new_user()` to assign tag_id
- `20260315000008_storage_policies.sql` — storage read policy for `match-clips` bucket

**Applied migrations (cameras + recorder + checkpoint 8):**
- `20260316000001_cameras_table.sql` — `cameras` table with `(gym_id, cam_id)` unique constraint, RLS for gym owner SELECT/UPDATE
- `20260317000001_add_log_events_app_version.sql` — `app_version` text column on `log_events`
- `20260318000001_checkin_upsert_unique.sql` — `UNIQUE(profile_id, gym_id)` on `gym_checkins` for sliding TTL upsert
- `20260318000002_clips_collision_status.sql` — CHECK constraint on `clips.status`: `created`, `exported_local`, `uploaded`, `collision_flagged`
- `20260318000003_claimable_clips_rpc.sql` — `get_claimable_clips()` + `claim_clip()` SECURITY DEFINER RPCs
- `20260318000004_device_tokens.sql` — `device_tokens` table for FCM push notification token storage, RLS for athletes
- `20260318000005_log_events_app_insert_policy.sql` — allows authenticated users to INSERT log_events

**Applied migrations (CP14e + CP14f: session-level export + cross-camera identity — applied to remote 2026-03-25):**
- `20260324000001_clips_source_video_ids.sql` — `source_video_ids text[]` on clips for multi-source session matches, backfill from existing video_id
- `20260325000001_clips_global_person_ids.sql` — `global_person_id_a text`, `global_person_id_b text` on clips for cross-camera identity linking

---

## Tech Stack

| Layer | Technology |
|---|---|
| CV pipeline | Python 3.12, YOLO v8 (ultralytics), BoT-SORT (boxmot) |
| Object detection | YOLOv8n (detection), YOLOv8s-seg (segmentation, optional) |
| Segmentation | SAM (deferred) |
| Tracking | BoT-SORT via boxmot |
| Identity anchoring | AprilTags (apriltag lib) |
| ILP solver | Google OR-Tools 9.12 |
| Data format | Parquet (pyarrow), JSONL for audit |
| Config | YAML + Pydantic v2 |
| Services | Docker (each service is standalone) |
| Backend | Supabase (Postgres + Auth + Storage + Realtime) |
| Mobile app | Flutter + supabase_flutter + geolocator + video_player |
| Web app | Vite + React + react-router-dom + @supabase/supabase-js |
| Audio analysis | librosa 0.10.2 (survey tool + future Stage A.5) |

**Key dependency constraints:**
- NumPy pinned to `1.x` (`<2`) — Torch/NumPy ABI incompatibility with 2.x
- Install `ultralytics` and `boxmot` with `--no-deps` to prevent opencv-python forcing NumPy 2.x
- Python 3.12 target; `>=3.10` minimum per pyproject.toml

---

## Config Resolution Order

1. `configs/default.yaml` (safe defaults)
2. `configs/cameras/<camera_id>.yaml` (per-camera overrides)
3. `configs/cameras/<camera_id>/homography.json` (homography matrix)
4. `--config` CLI overlay (optional, highest priority)

---

## Coding Conventions

- **Modularity first** — this is a SaaS product in development. Correctness and ability
  to change are paramount. Prefer clean module boundaries over cleverness.
- Stage `run()` functions have a stable contract: `run(config: dict, inputs: dict) -> dict`
- No stage imports another stage's internals. All inter-stage data flows through F0 contracts.
- Pydantic v2 for all data models.
- Loguru for logging (not stdlib logging).
- Rich for CLI output.
- Typer for CLI definitions.
- Parquet for all tabular inter-stage data. JSONL for audit/event streams.
- Type hints everywhere. Pyrightconfig is present — keep type coverage clean.
- Debug artifacts go under `outputs/<clip_id>/_debug/`. Never pollute stage output dirs.
- Avoid hardcoding paths. Use `ClipOutputLayout` and environment variables for path resolution.
- **Evidence over assumption** — if a behavior is unclear, add logging and inspect real
  output before writing a fix. Do not guess at root causes.

---

## Key Architectural Decisions

- **Supabase as hub** — no direct service-to-service communication. All clients
  (mobile, web, pipeline) read/write only Supabase. Drift alerts flow:
  CV pipeline detects drift → writes `drift_alert` row → Supabase Realtime
  → push notification to gym owner.
- **Offline-first pipeline** — pipeline runs locally/on-prem, uploads artifacts afterward.
  Not a streaming/real-time inference system.
- **AprilTags for identity** — athletes wear AprilTag IDs. Online pass observes tags,
  offline ILP pass resolves global identities across tracklet fragments.
- **MCF/ILP for stitching** — tracklet identity assignment treated as a min-cost flow
  problem. OR-Tools solver. Two ILP solver variants exist (d3_ilp, d3_ilp2) — d3_ilp2
  is the current preferred path.
- **Session-level Stage D aggregation (CP14c)** — `session_d_run.py` aggregates per-clip
  D0 bank outputs (frames, summaries, detections, identity hints) into a combined session
  bank, namespacing tracklet IDs with `{clip_id}:{tracklet_id}`. Runs existing D1→D4
  unchanged via `SessionStageLayoutAdapter` (duck-typed layout) and `SessionManifest`
  (lightweight manifest with artifact registry for D3 compile). Processor triggers
  session Phase 2 when `.session_ready` sentinel exists. Per-clip Phase 2 is untouched.
- **Stage B (SAM masks) deferred** — POC uses YOLO bbox masks. SAM integration exists
  but is not required for MVP.
- **Three-pass protocol** — Plan Mode (shift+tab x2) for Pass 1+2, execute for Pass 3.
  User approves between each pass. See Working Methodology section above.
- **Processor runs natively on Mac** — Docker ARM64 emulation too slow for YOLO inference.
  `services/processor/run_local.sh` runs the processor natively. Docker compose processor
  service is commented out; uncomment for Linux deployment.
- **Phase 1/Phase 2 parallelism boundary** (NON-NEGOTIABLE) — Phase 1 (Stages A+C) runs
  parallel workers via ProcessPoolExecutor (one per camera). Phase 2 (Stages D+E+F) runs
  sequentially. This boundary is load-bearing for future cross-clip global stitching.
  Do NOT parallelize Stage D+E+F under any circumstances.
- **YOLO segmentation disabled for performance** — `use_seg: false`, `write_yolo_masks: false`
  in default.yaml. Detection-only YOLOv8n used instead of YOLOv8s-seg. Mask code preserved
  behind config flags for future Stage F privacy redaction.
- **MPS auto-detection** — `device: "auto"` selects Apple Silicon MPS > CUDA > CPU.
  Validation step runs before full clip processing. Phase 1 workers use CPU (parallel safety),
  Phase 2 uses MPS (sequential, full GPU).
- **Uploader sentinel pattern** — Uploader writes `.uploaded` sentinel instead of deleting
  `export_manifest.jsonl`. Processor checks both manifest and sentinel for already-processed guard.
- **Two-pass cross-camera solve (CP17+)** — per-camera ILP solves independently in Pass 1, then re-solves with cross-camera priors in Pass 2. Evidence enters as hard constraints (tags, deterministic) or soft constraints (coordinates, confidence-weighted cost modifications). This "track-then-fuse" pattern is standard in multi-camera sports analytics (Second Spectrum, Hawk-Eye). It handles heterogeneous camera setups (variable overlap, imperfect alignment) gracefully — the system degrades to per-camera-only when cross-camera evidence is weak or absent. A single global ILP across all cameras is not used because it requires guaranteed coordinate alignment and camera overlap, which cannot be assumed across gym deployments. The architecture supports iterative message passing (multiple rounds) for coordinate-based evidence, but tag-based evidence converges in one round.

---

## Active Decisions Log

| Decision | Status | Notes |
|---|---|---|
| AprilTag family: 36h11 (~587 IDs) | Decided | Cell size optimized for fixed Nest cameras at gym distances. Larger cells = better detection at range, through occlusion, and at lower resolution. No family migration planned. |
| Check-in mechanism: WiFi SSID+BSSID | Decided | GPS rejected (indoor unreliable, high permission friction). Auto-triggers on WiFi connect in Flutter app. 3hr TTL auto-expiry. gyms table gets wifi_ssid + wifi_bssid columns. |
| profiles.tag_id not globally unique | Decided | tag_id is unique within (tag_id + gym_id + active time window). Handles scale beyond 587 without schema change. Stage F uses check-in to disambiguate if collision exists. |
| Athlete tag assignment: DB-assigned at signup | Decided | `tag_id_seq` cycling sequence (0–586) assigned by `handle_new_user()` trigger on sign-up. Physical merchandise (2 rashguards + 2 gi patches) ships with athlete's distinct tag printed. Replacements available on request. |
| Gym membership: single gym per athlete | Decided | `profiles.home_gym_id` FK (replaced `gym_members` join table). Can relax later. |
| Subscription history: gym_subscriptions table | Decided | Separate table from day one. Fields: gym_id, tier, started_at, ended_at, is_current. |
| Clip identity: denormalized profile IDs on clips | Decided | clips gets fighter_a_profile_id + fighter_b_profile_id (nullable FKs). Stage F writes tag IDs; the uploader service resolves tag → profile via active gym check-ins at upload time. Null = unresolved, backfillable. |
| Camera auto-registration: discovery-derived cam_id | Decided | `cam_id` = last 6 chars of SDM device path. `nest_recorder` auto-registers cameras to `cameras` table via Supabase REST upsert on every discovery run. Replaces manual DEVICE_*/CAM_ID_* env var configuration. `register_cameras.sh` called from `diag_v7_2.sh` after discovery, before recording. |
| Recording file path: gym-scoped production path | Decided | Production: `data/raw/nest/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/{cam_id}-{timestamp}.mp4`. Diag (no GYM_ID): `data/raw/nest/diag/{TS}/`. GYM_ID presence is the mode switch. `entrypoint.sh` delegates to `diag_v8.sh` scheduler (replaces legacy `record_window.sh` call). |
| Pipeline ingest path: gym-scoped, backward compatible | Decided | Pipeline accepts both `data/raw/nest/{gym_id}/{cam_id}/{date}/{hour}/` (new) and `data/raw/nest/{cam_id}/{date}/{hour}/` (legacy). `gym_id` inferred from path structure (date folder position detection), stored in `ClipManifest.gym_id` (None for legacy). No new CLI argument required. |
| Pipeline output path: gym-scoped | Decided | Outputs at `outputs/{gym_id}/{cam_id}/{date}/{hour}/{clip_id}/stage_*/`. Legacy fallback: `outputs/legacy/{cam_id}/{date}/{hour}/{clip_id}/`. `ClipOutputLayout.root` set from `compute_output_root()`. Stage F reads `gym_id` from manifest (fallback to config). |
| Collision detection: uploader tag dedup | Decided | Two signals: Signal A (same april_tag_id on both fighters in manifest `collision_hints`), Signal B (>1 active check-in for same tag+gym at upload time). Colliding clips get `status=collision_flagged`, null profile_ids. Athletes reclaim via `claim_clip()` RPC from Unlinked Clips screen. |
| YOLO masks disabled in Stage A | Decided | `use_seg: false`, `write_yolo_masks: false` in default.yaml. Detection-only YOLOv8n used. Mask code preserved behind config flags — will re-enable selectively in Stage F redaction redesign (CP13). |
| MPS auto-detection | Decided | `device: "auto"` in default.yaml. Detector resolves MPS > CUDA > CPU at construction time. Validated on M1 Air with dummy tensor. Falls back to CPU on validation failure. |
| Phase 1/2 parallelism boundary | Decided (NON-NEGOTIABLE) | A+C parallel via ProcessPoolExecutor (MAX_WORKERS=2, one per camera). D+E+F sequential. This boundary is load-bearing for future cross-clip global stitching — do not parallelize D+E+F under any circumstances. |
| Native processor execution | Decided | `run_local.sh` for Mac (MPS, native ARM). Dockerfile preserved for Linux mini-PC deployment. Docker processor service commented out in root compose. |
| Uploader sentinel pattern | Decided | `.uploaded` file written by uploader instead of deleting `export_manifest.jsonl`. Preserves processor's already-processed guard. Uploader `discover_manifests()` skips manifests with `.uploaded` sentinel. |
| Session pooler URL | Decided | `SUPABASE_DB_URL` uses Supavisor Session pooler (port 5432, `aws-1-us-east-1.pooler.supabase.com`) instead of direct connection (IPv6 only, fails in Docker). |
| Processor Phase 1 worker count | Decided | MAX_WORKERS=2, PARALLEL_DEVICE=mps on M1 Air. QoS P-core pinning via `pthread_set_qos_class_self_np(USER_INITIATED)`. Benchmark: MPS 2w = 7m/4clips, MPS 3w = 7m (GPU saturated), CPU 4w QoS = 15m, CPU 3w QoS = 22m. Validated on 3-camera diverse real footage (PPDmUg, J_EDEw, FP7oJQ) — all 4 clips A→F success including PPDmUg which was 0/12 in first production run. MPS parallel safe after degenerate bbox fix (ab526b7). |
| caffeinate -is for Mac runs | Decided | Prevents idle/display sleep during long MPS workloads on M1 Air. Standard invocation: `caffeinate -is bash -c 'time bash services/processor/run_local.sh'`. Releases automatically on child process exit. |
| Stale worker cleanup in run_local.sh | Decided | ProcessPoolExecutor spawn-mode workers are orphaned on unclean parent exit (Ctrl+C, timeout, CLI kill). run_local.sh now kills stale bjj_pipeline.stages and processor.py processes at startup and on EXIT/INT/TERM trap. Prevents memory/CPU contention on subsequent runs. |
| Session-level Stage D aggregation (CP14c) | Decided | Per-clip D0 banks aggregated into session-level combined bank with `{clip_id}:{tracklet_id}` namespacing. Per-clip 0-based frame indices offset by wall-clock time relative to session start (CP14e fix). D1→D4 run unchanged via `SessionStageLayoutAdapter` + `SessionManifest`. Identity hint frame_index IS offset (CP15 fix — D3 tag ping binding requires hints in same offset frame space as D1 nodes). |
| Gym setup calibration tool | In Progress | `tools/detect_buzzer.py` exists as a standalone audio survey tool. `src/calibration_pipeline/` created (CP16b): functional `lens_calibration.py` + stubs for mat_walk, drift_detection, inter_camera_sync (→ CP18). Two-step calibration chain: (1) lens_calibration estimates K+dist from mat edge clicks, (2) homography_calibrate auto-undistorts frame and produces H valid for undistorted pixels. Buzzer detection runs inside the per-session pipeline (Phase 1, Stage A.5 — future). AprilTag visibility heatmap is a CP18 output surfaced in `app_web`. |
| Session-level stitching: schedule-based clip grouping (CP14a) | Decided | `SCHEDULE_JSON` env var (same one nest_recorder uses) provides gym class windows. Processor groups clips by session (date + start time), writes per-camera `.phase1_complete_{cam_id}` sentinels, then `.session_ready` or `.tag_required` when all-cameras-Phase-1 + wall-clock buffer gates pass. `SessionOutputLayout` in `f0_paths.py` provides canonical session output paths under `outputs/{gym_id}/sessions/{date}/{session_id}/`. Session-level D/E/F invocation is CP14c. |
| Session-level Stage F export (CP14e) | Decided | `session_f_run.py` exports session-level match clips. Multi-source extraction: determines which source MP4s overlap each match's frame range, extracts per-source segments via `export_clip()`, concatenates via ffmpeg concat demuxer. `source_video_paths` in export manifest `clip_row` enables uploader to resolve multiple `source_video_ids`. `clips.source_video_ids text[]` column added via migration. Processor runs full D→E→F per camera: `run_session_d()` returns `SessionManifest`, passed to Stage E via adapter, then `run_session_f()`. `run_session_d()` always returns manifest even if D4 skipped (Stage E handles missing inputs gracefully). Post-review patch: `clip_row` omits `video_id` key (uploader sets it from `resolved_ids[0]`); `_LogProxy.source_match_ids` uses `str(match_id)` for type consistency. Per-camera manifests: each camera writes `export_manifest_{cam_id}.jsonl` + `audit_{cam_id}.jsonl` under `stage_F/`. Processor merges per-camera manifests into `export_manifest.jsonl` after Loop 2 (uploader reads merged file only). |
| Stage E two-layer engagement detection (CP14d) | Decided | Stage E uses two detection layers: (1) cap2 GROUP seeds from person_spans (existing), (2) proximity hysteresis from person_tracks world coords (new). Both layers are optional — either one can produce engagement intervals independently. Unioned per-pair with gap-based merging, buffered, then optionally adjusted by buzzer audio events. Config: `engage_dist_m=0.75`, `disengage_dist_m=2.0`, `engage_min_frames=15`, `hysteresis_frames=450` (~15s@30fps), `min_clip_duration_frames=150` (~5s), `clip_buffer_frames=45` (~1.5s), `buzzer_boundary_window_frames=90` (~3s). Session frame bounds derived from actual data ranges (not assumed 0-based). Zero matches is valid — writes empty JSONL, logs audit event, does not raise. Both inputs missing → PipelineError. |
| Cross-camera identity merge (CP14f) | Decided | `cross_camera_merge.py` links the same athlete across cameras via AprilTag co-observation within a session. Presence-based linking (Option 1): same `tag_id` on 2+ cameras in the same session = same athlete. Filters by `min_tag_observations` (evidence.total_tag_frames >= 2) and `min_assignment_confidence` (>= 0.5). Intra-camera dedup: at most one (cam_id, person_id) per (cam_id, tag_id) — genuine confidence ties skip the tag for THAT camera only (CP15 fix; previously skipped globally). Union-find over cross-camera links → deterministic `gp_` prefixed global IDs (sha256 of sorted member keys). Every (cam_id, person_id) gets a global ID whether linked or standalone. Output: `cross_camera_identities.jsonl` under `SessionOutputLayout.stage_dir("D")`. Processor restructured: Loop 1 (D+E per camera) → cross-camera merge → Loop 2 (F per camera with `global_id_map`). Merge failure logs error and passes empty map — never blocks Stage F export. `clips` table gets `global_person_id_a/b text` columns. `co_observation_window_frames` is a documented no-op config parameter — future hook for buzzer-based clock sync. Config: `cross_camera.clock_sync_method="filename"`, `co_observation_window_frames=90`, `min_tag_observations=2`, `min_assignment_confidence=0.5`. CP14f's post-hoc merge (union-find on shared tags after per-camera ILP) remains as a fallback and baseline. CP17 replaces the primary path: cross-camera evidence is injected INTO the per-camera ILP before solving, enabling the solver to use cross-camera signals to resolve within-camera ambiguities that post-hoc merge cannot fix. |
| Cross-camera ILP enhancement | Planned (CP17) | See CP17 entry. Two-pass architecture with hard constraints (tags) + soft constraints (coordinates). Replaces post-hoc CP14f merge as primary cross-camera identity path. |
| Session export manifest overwrite bug | Fixed | Was: `session_f_run.py` wrote all cameras to shared `export_manifest.jsonl` — `unlink()` + "w" mode meant only last camera survived. Fix: each camera writes `export_manifest_{cam_id}.jsonl` + `audit_{cam_id}.jsonl` (per-camera scoped). Processor merges per-camera manifests into single `export_manifest.jsonl` after Loop 2, filtering out `no_matches` headers. All-no_matches sessions produce a single no_matches merged manifest. Uploader contract unchanged. |
| CP16a: F0 projection utility + call site migration | Completed | `project_to_world()` in `contracts/f0_projection.py` — canonical pixel→world projection with optional `cv2.undistortPoints()` before homography (Option B). Accepts pre-loaded H/K/dist matrices (pure function, no disk I/O). `CameraProjection` NamedTuple bundles H + calibration params. 1 projection call site migrated (`processor.py`), 2 loaders updated (`multiplex_runner.py`, `run.py`) to return `CameraProjection`. 5 `homography.json` files extended with null `camera_matrix` + `dist_coefficients` placeholders. Debug artifact: `_debug/projection_debug.jsonl` + `projection_config` audit event per clip. `quality.py:project_uv_to_xy()` deprecated in place. Known pre-existing issue: `run.py:_load_homography_matrix()` lacks direction correction (isolation path only, not CP16a scope). |
| CP16b: Calibration pipeline skeleton + functional lens calibration | Completed | `src/calibration_pipeline/` created with `lens_calibration.py` (functional), `mat_walk.py`, `drift_detection.py`, `inter_camera_sync.py` (stubs → CP18). Lens calibration: auto-detects mat edge points via 1D gradient analysis along perpendicular profiles (50–100+ points/edge with sub-pixel precision), user can add/delete points manually. Solver: **collinearity optimization** via `scipy.optimize.minimize` (Powell, bounded) — directly minimizes perpendicular distance of undistorted points from fitted lines. 3 free params (f, k1, k2) with bounds (f: 200–3000, k1/k2: ±1.0). Replaces `cv2.calibrateCamera` which over-fitted with 6 params and no pose-distortion decoupling. Per-edge RMS reported. Writes `camera_matrix`, `dist_coefficients`, `lens_calibration` metadata (incl. method, per-edge RMS, auto/manual point counts) to homography.json. `homography_calibrate.py` updated: (1) `_write_homography_json()` read-then-merges to preserve K+dist, (2) auto-undistorts frame at load when K+dist present (both UI modes). `calibration_pipeline` in `pyproject.toml` packages. Two-step calibration chain: lens_calibration → K+dist, then homography_calibrate on undistorted frame → H. |
| CP17: Two-pass cross-camera ILP (tag + coordinate channels) | Planned | Two-pass per-camera solve architecture. Pass 1: each camera solves its ILP independently (existing D3). Pass 2: each camera re-solves its ILP with cross-camera evidence injected as priors from other cameras' Pass 1 results. Evidence enters as: (a) HARD must_link constraints for high-confidence tag co-observations (same tag_id seen on 2+ cameras — deterministic, forces assignment), (b) SOFT cost modifications for coordinate/temporal evidence (modifies edge costs in the flow graph — solver prefers but can override when local evidence is stronger). Every piece of cross-camera evidence carries an explicit confidence score that weights its influence on the second solve. Tag channel (hard constraints) is CP17 scope. Coordinate channel (soft constraints) is stubbed in CP17, activated by CP18 when inter-camera alignment is available. One round of message passing is sufficient for tag-based evidence; coordinate-based evidence may need 2-3 rounds (capped) — architecture supports iteration but CP17 implements single round only. Graceful degradation: system works at three quality tiers — (1) tag-only evidence (no coordinate alignment needed), (2) tag + rough coordinates (after CP16+CP18 with imperfect alignment), (3) tag + precise coordinates (after mat walk calibration). Each tier adds signal without lower tiers breaking. |
| CP18: Calibration pipeline fleshed out + coordinate evidence activation | Planned | Activates the world coordinate evidence channel stubbed in CP17. Three components: (1) Mat walk — known tagged person walks a grid pattern across all camera fields of view. Produces labeled world coordinate correspondences across cameras at same timestamps. Inter-camera affine alignment via least-squares solve on these correspondences gives global coordinate consistency. (2) Drift detection — empty-mat baseline snapshot after successful calibration, daily frame comparison via edge detection, drift score to Supabase. Severe drift alert to gym owner. (3) Scheduled recalibration via nest_recorder secondary pipeline window. Once coordinate alignment is established, CP17's soft constraint channel becomes active — "person at (53, 46) on camera A and (53, 46) on camera B within N frames" becomes a weighted linkage signal in the second ILP pass. Alignment quality score feeds directly into the confidence weight of coordinate evidence — poor alignment = weak soft constraints, good alignment = strong soft constraints. Three correction layers with different update frequencies: (1) Lens calibration — one-time per camera, essentially permanent (CP16b, done). (2) Per-camera homography — nightly recalibration attempt. (3) Inter-camera affine alignment — derived from mat walk, updated when drift detected. |
| Option B undistort-on-projection | Decided | Pixel-to-world projection uses `cv2.undistortPoints()` on pixel coordinates before applying homography rather than undistorting full frames. More efficient than full-frame undistortion at 30fps across multiple cameras. Strict enforcement: `project_to_world()` is the only permitted projection path in `src/bjj_pipeline` — no stage calls homography directly. |
| Calibration pipeline as separate top-level module | Decided | `src/calibration_pipeline/` sits alongside `src/bjj_pipeline`, not inside it. Rationale: mat walk and drift detection are gym initialization and maintenance workflows, not per-session pipeline stages. Outputs (K + distortion coefficients, refined homographies) feed into `src/bjj_pipeline` via shared config files and Supabase. One-time and periodic runs, not triggered by the session processor. |
| Inter-camera homography sync approach | Decided | Mat walk uses single known tagged person walking mat grid. Labeled world coordinate correspondences across cameras at same timestamps. Least-squares affine solve aligns per-camera coordinate systems globally. Requires per-camera lens undistortion first (K + distortion coefficients) before homography computation — undistorted frames only. Three correction layers with different update frequencies: (1) Lens calibration — one-time per camera, essentially permanent. (2) Per-camera homography — nightly recalibration attempt. (3) Inter-camera affine alignment — derived from mat walk. |
| Multipass mode removed (CP16-cleanup) | Decided | `multipass` execution mode removed from CLI and pipeline. `multiplex_AC` is now the only execution path — no `--mode` flag. Phase 1 (A+C) always runs via `run_multiplex_AC()` (single video decode), Phase 2 (D+E+F) runs sequentially. `detect_track/run.py` preserved as standalone Stage A isolation runner with warning comment (does NOT apply homography direction correction). Processor service unchanged (already used multiplex_AC). |

---

## Current Branch & Status

- **Active branch:** `services_uploader`
- **Head commit:** `0f5edd5`
- **Pipeline:** Full pipeline (A→F) verified end-to-end. Session-level pipeline (CP14a→CP15) validated on Alpha BJJ 3-camera data (FP7oJQ, J_EDEw, PPDmUg). Ingest accepts gym-scoped paths (`{gym_id}/{cam_id}/{date}/{hour}/`) and legacy paths (`{cam_id}/{date}/{hour}/`). `gym_id` stored in `ClipManifest`. Stages A, C produce tag observations + identity hints. Stage D (ILP stitching) resolves person tracks. Stage E detects match sessions. Stage F exports clips with privacy redaction.
- **CP15: Closed (2026-03-24).** Seven fixes validated E2E on 3-camera Alpha BJJ data: (1) StageEConfig missing fields, (2) session evaluator clip iteration, (3) `.session_completed` sentinel, (4) session clip_id validation, (5) D2 tag ping field name fix (latent bug — `first_seen_frame` vs `frame_index`), (6) session hint frame offset, (7) cross-camera dedup per-camera scoping. Cross-camera links = 0 on test data due to single-tag ambiguity (all assignments confidence=1.0 with multiple claimants per camera). Fix is structurally correct — will produce links when distinct tags are in use.
- **Post-CP15 bug fix (2026-03-25):** Session export manifest overwrite fixed. Per-camera manifests (`export_manifest_{cam_id}.jsonl`) + explicit merge step in processor. Committed `598ac0f`.
- **Services:** `nest_recorder` working — auto-registers cameras to Supabase on discovery. `uploader` working — resolves fighter tag IDs → profile IDs via active gym check-ins at upload time (Phase C identity bridge). Uploader writes `global_person_id_a/b` from session export manifest to clips table via dynamic INSERT. `processor` working — `.session_completed` sentinel prevents Phase 2 re-triggering.
- **Remote Supabase:** All 23 migrations applied to remote (2026-03-25). CP14e/CP14f migrations (`clips.source_video_ids`, `clips.global_person_id_a/b`) now live on remote.
- **CP16a: Completed (2026-03-25).** F0 projection utility `project_to_world()` in `contracts/f0_projection.py`. 1 call site migrated, 2 loaders extended, 5 homography.json files updated with null calibration placeholders. Debug artifact `_debug/projection_debug.jsonl` confirms code path exercised. Pipeline output unchanged with identity placeholders.
- **CP16b: Completed (2026-03-25).** `src/calibration_pipeline/` created — functional lens calibration tool + 3 stubs. `homography_calibrate.py` updated with read-then-merge save + auto-undistort. Two-step calibration chain: lens_calibration → K+dist, then homography_calibrate on undistorted frame → H. `calibration_pipeline` added to pyproject.toml packages.
- **Next:** CP17 — Two-pass cross-camera ILP. Tag channel (hard constraints) as primary deliverable. Coordinate channel stubbed with confidence-weighted soft constraint infrastructure for CP18 activation.
- **Apps:** Flutter mobile app at `app_mobile/`. End-to-end tested on Pixel 7 Pro against local Supabase.
  - **Auth:** Supabase-native (supabase_flutter). Auth trigger auto-creates profiles with tag_id on sign-up. Biometric login gated behind Settings toggle (default off).
  - **Onboarding:** display name → gym select → invite gym (if not listed). Routes via AuthGate FutureBuilder with profile completeness check.
  - **Clips:** Pull-to-refresh clip list. Tap to play via signed URL + video_player. RLS scopes clips to athlete's profile (fighter_a/b_profile_id match).
  - **Check-in:** WiFi auto check-in (CheckinService) fires after auth + on WiFi changes. Upserts on `(profile_id, gym_id)` — sliding TTL via hourly periodic probe while WiFi connected. Timer cancelled on WiFi disconnect. Manual check-in via Find a Gym screen. SSID-primary matching (BSSID optional refinement). Source tracked as `wifi_auto` or `manual`.
  - **Gym discovery:** Find a Gym screen with GPS proximity via `gyms_near` RPC. Accessible from navigation drawer.
  - **Android:** `usesCleartextTraffic=true` for local HTTP Supabase. `ACCESS_FINE_LOCATION` required for WiFi SSID + GPS.
  - **Local dev:** `supabase_config.dart` has local config commented out (`192.168.0.66:54321`). Remote config active. Signed URLs rewrite `127.0.0.1` → configured host for phone access.
  - **Supabase key format:** Remote Supabase uses new `sb_publishable_`/`sb_secret_` key naming, but PostgREST API requires classic JWT keys (`eyJ...` format). Use JWT keys from Dashboard > Settings > API Keys for all client and service connections.
- **Web app:** Vite + React at `app_web/`. Supabase auth via `@supabase/supabase-js`, client-side routing via `react-router-dom`.
  - `/` — Mat blueprint editor (Konva canvas, drag-and-drop mat sections, import/export JSON)
  - `/admin/pricing` — Admin-only business model pricing simulator (4 tabs: Model, Unit Economics, Sensitivity, Notes). Gated by `AdminGate` component checking session email against `VITE_ADMIN_EMAIL` env var.
  - **Auth:** `AdminGate` wraps protected routes. Email+password sign-in via Supabase. Admin email checked from env, never hardcoded.
  - **Local dev:** `.env.example` provided. Set `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `VITE_ADMIN_EMAIL`.
- **Supabase:** All 23 migrations applied locally and remotely. Remote Supabase linked (project `zwwdduccwrkmkvawwjpc`). Edge Function `send_push_notification` for FCM V1 push delivery. RLS on all 11 tables. Storage read policy on `match-clips` bucket. `cameras` table auto-populated by `nest_recorder`. `gym_checkins` has `UNIQUE(profile_id, gym_id)` for sliding TTL upsert.
- **E2E verified:** 2026-03-17 — nest_recorder → processor → uploader chain tested end-to-end. Tagged clip (FP7oJQ-tag_0-60s.mp4) processed A→F, uploaded to local Supabase, 2 clip rows + 2 log_events inserted. Already-processed guard confirmed working.
- **Performance baseline (final validation run 2026-03-22):**

  | Stage | Before CP11 | After CP11 | Post-fix (CPU 3w) | **Current (MPS 2w QoS)** |
  |---|---|---|---|---|
  | Stage A (2.5min clip) | ~120 min | 4m 37s | ~6-8 min/clip | **~1.9 min/clip** |

  | Phase | Wall-clock (36 clips) | Notes |
  |---|---|---|
  | Phase 1 (A+C, MPS 2w QoS) | **105 min actual / ~69 min representative** | 36/36 completed, 0 skipped. Actual inflated by stale workers from prior run competing for first ~60 min. Post-cleanup rate ~1.9 min/clip is the representative baseline. |
  | Phase 2 (D+E+F, sequential MPS) | **68 min actual** | 35/36 manifests, 1 Stage D error. Also inflated by stale worker memory pressure early in run. |
  | Total | **~173 min actual / ~120 min representative** | Run with `caffeinate -is`. 35 export manifests. |

  **Bug fix history:**
  - Run 1 (2026-03-20): 30/36 failed — degenerate bbox bug. Fixed in `ab526b7`.
  - Run 2 (2026-03-21a): 36/36 Phase 1, 7 Phase 2 errors — Stage D/F bugs. Fixed in `4e825a4`.
  - Run 3 (2026-03-21b): 36/36 Phase 1, 34/36 manifests. 2 remaining Stage D edge cases (FP7oJQ-201022 dt_s, PPDmUg-202751 graph edges).
  - Run 4 (2026-03-22): 36/36 Phase 1, 35/36 manifests. FP7oJQ-201022 now passes. 1 remaining: PPDmUg-202751 (NAType in frame_index). Stale worker contamination inflated timings ~40%.

  **Known open issue:** PPDmUg-20260318-202751 fails consistently at Stage D2 — `int(bank_df["frame_index"].min())` returns NAType. Degenerate clip with extremely sparse tracklets producing all-NaN frame_index column. Requires null-safe integer handling fix in D2 `compute_edge_costs()`. All other 35 clips pass A→F.

- **Last updated:** 2026-03-26 (CP16-cleanup — removed multipass mode, multiplex_AC is now the only execution path)

---

## Common Commands

```bash
# Install (clean)
rm -rf .venv && python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install --no-deps ultralytics boxmot
pip install -e .

# Run pipeline
python -m bjj_pipeline.stages.orchestration.cli run \
  --input data/raw/nest/cam03/2026-01-03/12/<clip>.mp4 \
  --camera cam03

# Run specific stage only (via run_until config overlay)
python -m bjj_pipeline.stages.orchestration.cli run \
  --input <clip> --camera cam03 \
  --config '{"stages": {"stage_D": {"run_until": "D1"}}}'

# Validate outputs
python -m bjj_pipeline.stages.orchestration.cli validate --clip-id <clip_id>

# Supabase local dev (CLI installed via npm, use npx)
cd backend/supabase/supabase
npx supabase start
npx supabase db reset

# Flutter (not on PATH — use full path)
~/development/flutter/bin/flutter pub get
~/development/flutter/bin/flutter analyze
~/development/flutter/bin/flutter run

# Run uploader locally (against local Supabase)
# Set env vars from: npx supabase status (use Secret key for SERVICE_ROLE_KEY)
SUPABASE_URL=http://127.0.0.1:54321 \
SUPABASE_SERVICE_ROLE_KEY=<secret-key-from-supabase-status> \
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \
SUPABASE_STORAGE_BUCKET=match-clips \
UPLOADER_DELETE_LOCAL=false \
python -c "import sys; sys.path.insert(0,'services/uploader'); from uploader.cli import main; sys.argv=['u','--manifest','<path/to/export_manifest.jsonl>']; main()"

# Flutter run on Pixel (device ID may vary)
~/development/flutter/bin/flutter run -d 2A191FDH300C9Z

# Audio landmark survey (buzzer/bell detection)
python tools/detect_buzzer.py --input <mp4_or_dir> --survey
python tools/detect_buzzer.py --input data/raw/nest/gym01/cam03/2026-03-18/20/ --survey --output-dir /tmp/audio_survey

# Docker services
cd services/nest_recorder && docker compose up
cd services/uploader && docker compose up
```

---

## Files Claude Code Should Never Touch

- `data/` — raw video data and secrets
- `outputs/` — pipeline artifacts
- `services/nest_recorder/secrets/` — OAuth credentials
- `.env` files — environment secrets
- Migration files that have already been applied to production Supabase
