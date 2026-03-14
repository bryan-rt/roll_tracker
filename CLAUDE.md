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
├── services/
│   ├── nest_recorder/      # Docker: Google Nest API → MP4 segments to data/raw/
│   ├── processor/          # Docker: placeholder — wraps bjj_pipeline for service context
│   └── uploader/           # Docker: polls outputs/, uploads to Supabase, deletes on confirm
├── backend/
│   └── supabase/supabase/
│       ├── config.toml
│       └── migrations/     # SQL schema (see Database Schema section)
├── apps/                   # Flutter mobile app + web app (empty in current zip)
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
- **Stage E** `matches`: Match session detection from resolved tracks.
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

All inter-stage data lives on disk under `outputs/<clip_id>/`. The F0 layer enforces this:
- `f0_manifest.py` — `ClipManifest` dataclass, init/load/write, per-stage default registration
- `f0_paths.py` — `ClipOutputLayout`, `StageLetter` — canonical path resolution
- `f0_parquet.py` — Parquet read/write helpers
- `f0_models.py` — Shared Pydantic models
- `f0_validate.py` — Post-stage validators

**Rule:** Stages communicate only via the manifest + filesystem. No stage imports another
stage's internals directly.

---

## Docker Services

| Service | Status | Responsibility |
|---|---|---|
| `nest_recorder` | Working | OAuth2 → Nest API → MP4 segments → `data/raw/nest/` |
| `processor` | Scaffold only | Will wrap bjj_pipeline; no implementation yet |
| `uploader` | Working | Polls `outputs/`, bundles + uploads to Supabase, deletes on confirm |

The processor service has a documented I/O contract at `services/processor/contracts/input_output.md`.
The uploader contract is at `services/uploader/contracts/batch_bundle.md`.
Idempotency is critical for the uploader — re-runs must not duplicate uploads.

---

## Supabase Schema (current migrations)

**Tables:**
- `profiles` — athlete/user records. `auth_user_id` (FK to Supabase Auth), `display_name`, `email`
- `videos` — raw video metadata: `camera_id`, `source_path`, `recorded_at`, `status`, `metadata` (jsonb)
- `clips` — processed clips: `video_id` FK, `match_id`, `file_path`, `storage_bucket`,
  `storage_object_path`, `start_seconds`, `end_seconds`, `fighter_a_tag_id`, `fighter_b_tag_id`
- `log_events` — audit log: `clip_id`/`video_id` FK, `event_type`, `event_level`, `message`, `details`

**Storage bucket:** `match-clips` (private)

**Pending schema items (not yet migrated):**
- `notification_channel` — TBD (drift alert delivery mechanism)

**Applied migrations (Phase A):**
- `20260311000001_create_gyms.sql` — `gyms` table: `id`, `name`, `owner_profile_id`, `address`, `wifi_ssid`, `wifi_bssid`, `created_at`, `updated_at`
- `20260311000002_create_gym_members.sql` — `gym_members` table (**superseded by 000007** — table dropped, replaced by `profiles.home_gym_id`)
- `20260311000003_create_gym_subscriptions.sql` — `gym_subscriptions` table: `id`, `gym_id`, `tier` ENUM(`free`, `pro`, `enterprise`), `started_at`, `ended_at`, `is_current`
- `20260311000004_create_gym_checkins.sql` — `gym_checkins` table: `id`, `profile_id`, `gym_id`, `checked_in_at`, `auto_expires_at` (generated, +3hr), `is_active`
- `20260311000005_create_homography_configs.sql` — `homography_configs` table: `id`, `gym_id`, `camera_id`, `config_data` JSONB, `created_at`, `updated_at`
- `20260311000006_add_phase_a_columns.sql` — `profiles` adds `tag_id` (indexed, not unique), `tag_assigned_at`, `starter_pack_sent_at`; `videos` adds `gym_id` FK→gyms; `clips` adds `fighter_a_profile_id`, `fighter_b_profile_id` FK→profiles (nullable)
- `20260311000007_phase_a_correction.sql` — drops `gym_members` table and `gym_role` enum; adds `profiles.home_gym_id` FK→gyms; adds `gyms.latitude`, `gyms.longitude`; creates `gym_interest_signals` table

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
| Mobile app | Flutter (rough draft, TBD migration) |
| Web app | TBD (gym owner blueprint + homography calibration tool) |

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
- **Stage B (SAM masks) deferred** — POC uses YOLO bbox masks. SAM integration exists
  but is not required for MVP.
- **Three-pass protocol** — Plan Mode (shift+tab x2) for Pass 1+2, execute for Pass 3.
  User approves between each pass. See Working Methodology section above.
- **Processor service is a scaffold** — the Python pipeline runs standalone locally.
  Dockerizing it is a near-term MVP task.

---

## Active Decisions Log

| Decision | Status | Notes |
|---|---|---|
| AprilTag family: 36h11 (~587 IDs) | Decided | Cell size optimized for fixed Nest cameras at gym distances. Larger cells = better detection at range, through occlusion, and at lower resolution. No family migration planned. |
| Check-in mechanism: WiFi SSID+BSSID | Decided | GPS rejected (indoor unreliable, high permission friction). Auto-triggers on WiFi connect in Flutter app. 3hr TTL auto-expiry. gyms table gets wifi_ssid + wifi_bssid columns. |
| profiles.tag_id not globally unique | Decided | tag_id is unique within (tag_id + gym_id + active time window). Handles scale beyond 587 without schema change. Stage F uses check-in to disambiguate if collision exists. |
| Athlete tag assignment: backend-assigned at signup | Decided | Backend assigns tag_id sequentially at registration. Physical merchandise (2 rashguards + 2 gi patches) ships with athlete's distinct tag printed. Replacements available on request. |
| Gym membership: single gym per athlete | Decided | `profiles.home_gym_id` FK (replaced `gym_members` join table). Can relax later. |
| Subscription history: gym_subscriptions table | Decided | Separate table from day one. Fields: gym_id, tier, started_at, ended_at, is_current. |
| Clip identity: denormalized profile IDs on clips | Decided | clips gets fighter_a_profile_id + fighter_b_profile_id (nullable FKs). Stage F writes them. Null = unresolved, backfillable. |

---

## Current Branch & Status

- **Active branch:** `services_uploader`
- **Head commit:** `035e464`
- **Pipeline:** Stages A, C, D (D0–D3), E partially implemented. Stage F (export) exists.
- **Services:** `nest_recorder` working. `uploader` working. `processor` scaffold only.
- **Apps:** Flutter mobile app at `mobile_app/`. Auth migrated to Supabase-native (supabase_flutter). Firebase fully removed. Data layer uses profiles/clips/gyms schema. WiFi check-in listener added (CheckinService) — requires ACCESS_FINE_LOCATION (Android) and NSLocationWhenInUseUsageDescription (iOS). Runtime permission request UI deferred.
- **Supabase:** Phase A migrations applied. Phase A correction applied (gym_members → home_gym_id).
- **Last updated:** 2026-03-14 (Phase B2 — WiFi check-in listener)

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
