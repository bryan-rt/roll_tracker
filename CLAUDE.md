# CLAUDE.md — Roll Tracker

This file is the persistent context bridge for Claude Code CLI. Every `claude` session
in this repo reads this file automatically. Keep it current after significant changes.

---

## Project Identity

**Name:** Roll Tracker
**Author:** Bryan Thomas
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
- `gyms` table — gym owner accounts, camera registrations, pricing tier
- `april_tag_assignments` table — `tag_id` ↔ `athlete profile` mapping per gym
- `homography_configs` table — per-camera calibration with version/drift tracking
- `notification_channel` — TBD (drift alert delivery mechanism)

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
- **Processor service is a scaffold** — the Python pipeline runs standalone locally.
  Dockerizing it is a near-term MVP task.

---

## Current Branch & Status

- **Active branch:** `mobile_app`
- **Head commit:** `b1abf24`
- **Pipeline:** Stages A, C, D (D0–D3), E partially implemented. Stage F (export) exists.
- **Services:** `nest_recorder` working. `uploader` working. `processor` scaffold only.
- **Apps:** Empty in current state — Flutter drafts not yet committed to repo.
- **Supabase:** Schema migrations exist. Gym/tag/homography tables not yet migrated.

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

# Supabase local dev
cd backend/supabase/supabase
supabase start
supabase db reset

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