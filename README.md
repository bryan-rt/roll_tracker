# Roll Tracker

Multi-service SaaS pipeline for BJJ gyms. Streams Nest camera footage, tracks
athletes via YOLO + BoT-SORT, anchors identity with AprilTags, stitches global
identities with ILP/MCF, and delivers per-athlete match clips through a mobile app.

## System Topology

```mermaid
flowchart TD
  subgraph External["External"]
    NestAPI["Google Nest SDM API<br/><i>OAuth2 · camera discovery · RTSP streams</i>"]
    SupaAuth["Supabase Auth<br/><i>email+password signup</i>"]
  end

  subgraph Services["Docker Services"]
    Recorder["<b>nest_recorder</b><br/>OAuth2 → MP4 segments<br/>Auto-registers cameras"]
    Processor["<b>processor</b><br/>Polls raw/ → runs pipeline A–F<br/>Outputs export_manifest.jsonl"]
    Uploader["<b>uploader</b><br/>Polls outputs/ → resolves tag→profile<br/>Uploads clips to Supabase"]
  end

  subgraph Pipeline["CV Pipeline  (src/bjj_pipeline)"]
    direction LR
    subgraph Phase1["Phase 1 · Online single-pass"]
      A["<b>Stage A</b><br/>YOLO detect<br/>BoT-SORT track"]
      B["<b>Stage B</b><br/>SAM masks<br/><i>(deferred)</i>"]
      C["<b>Stage C</b><br/>AprilTag scan<br/>Identity voting"]
    end
    subgraph Phase2["Phase 2 · Offline multi-pass"]
      D["<b>Stage D</b><br/>ILP/MCF stitch<br/>Global identity"]
      E["<b>Stage E</b><br/>Match session<br/>detection"]
      F["<b>Stage F</b><br/>ffmpeg export<br/>Privacy redaction"]
    end
    A --> B --> C --> D --> E --> F
  end

  subgraph Storage["Filesystem (Docker volumes)"]
    Raw["data/raw/nest/<br/>{gym_id}/{cam_id}/{date}/{hour}/<br/>*.mp4"]
    Outputs["outputs/{clip_id}/<br/>stage_A/ … stage_F/<br/>export_manifest.jsonl"]
  end

  subgraph Supabase["Supabase"]
    DB[("PostgreSQL<br/><i>10 tables · RLS on all</i>")]
    Bucket["Storage<br/><b>match-clips</b> bucket"]
  end

  subgraph Clients["Client Apps"]
    Mobile["<b>app_mobile</b>  (Flutter)<br/>Clip playback · WiFi check-in<br/>Athlete onboarding"]
    Web["<b>app_web</b>  (Vite + React)<br/>Admin pricing dashboard<br/>Mat blueprint editor"]
  end

  %% Data flow
  NestAPI -->|"RTSP / media"| Recorder
  Recorder -->|"MP4 segments"| Raw
  Recorder -->|"upsert cameras"| DB
  Raw -->|"poll"| Processor
  Processor -->|"stages A–F"| Pipeline
  Pipeline -->|"artifacts"| Outputs
  Outputs -->|"poll"| Uploader
  Uploader -->|"clip MP4"| Bucket
  Uploader -->|"clips + log_events rows<br/>tag_id → profile_id via checkins"| DB

  SupaAuth -->|"signup trigger"| DB
  DB -->|"signed URLs + RLS queries"| Mobile
  DB -->|"admin auth + queries"| Web
  Bucket -->|"signed video URLs"| Mobile
```

## Monorepo Layout

```
roll_tracker/
├── src/bjj_pipeline/          # CV pipeline (Python 3.12, installable package)
│   ├── contracts/             # F0 data contracts: manifest, parquet, paths, validators
│   ├── stages/
│   │   ├── detect_track/      # A: YOLO + BoT-SORT → tracklets + detections
│   │   ├── masks/             # B: SAM masks (deferred, falls back to YOLO bbox)
│   │   ├── tags/              # C: AprilTag scheduling, scanning, identity voting
│   │   ├── stitch/            # D: MCF/ILP global identity (D0–D4, OR-Tools)
│   │   ├── matches/           # E: Match session detection
│   │   ├── export/            # F: ffmpeg clip cutting, redaction, manifest
│   │   └── orchestration/     # CLI entry point, stage registry, resume logic
│   └── config/, core/, eval/, tools/, viz/
│
├── services/
│   ├── nest_recorder/         # Docker: Nest API → MP4 segments + camera registration
│   ├── processor/             # Docker: polls raw/, wraps pipeline A–F
│   └── uploader/              # Docker: polls outputs/, uploads to Supabase
│
├── backend/supabase/supabase/
│   ├── config.toml
│   └── migrations/            # 21 SQL migrations (schema + RLS + triggers)
│
├── app_mobile/                # Flutter: athlete-facing clip viewer + WiFi check-in
├── app_web/                   # Vite + React: gym owner admin dashboard
├── configs/                   # Pipeline YAML configs + per-camera homography
├── docker-compose.yml         # Three-service orchestration
├── data/raw/nest/             # Raw MP4 segments (gitignored)
└── outputs/                   # Pipeline artifacts per clip (gitignored)
```

## Service Architecture

All services communicate through Supabase or the shared filesystem — no direct
service-to-service calls.

| Service | Input | Output | Trigger |
|---|---|---|---|
| **nest_recorder** | Nest SDM API (OAuth2) | MP4 → `data/raw/nest/` | Continuous |
| **processor** | `data/raw/nest/*.mp4` | `outputs/{clip_id}/stage_F/export_manifest.jsonl` | Polls every 30s |
| **uploader** | `export_manifest.jsonl` | Supabase DB rows + Storage upload | Polls outputs/ |

## Database Schema

10 tables, RLS on all. Key tables:

| Table | Role |
|---|---|
| `profiles` | Athletes. Auto-created on signup with `tag_id` (0–586 cycling sequence) |
| `clips` | Processed match clips with `fighter_a/b_tag_id` and resolved `profile_id` |
| `gym_checkins` | WiFi-based attendance. 3hr TTL. Used by uploader for tag→profile resolution |
| `cameras` | Auto-registered by nest_recorder on device discovery |
| `gyms` | Gym metadata, WiFi SSID/BSSID for auto check-in |
| `videos` | Raw video metadata |
| `log_events` | Audit trail |
| `gym_subscriptions` | Billing tier history |
| `homography_configs` | Per-camera calibration matrices |
| `gym_interest_signals` | Lead gen from onboarding flow |

Storage bucket: `match-clips` (private, RLS-gated signed URLs).

## Quick Start

```bash
# Pipeline (local, no Docker)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps ultralytics boxmot
pip install -e .

python -m bjj_pipeline.stages.orchestration.cli run \
  --input data/raw/nest/cam03/2026-01-03/12/clip.mp4 \
  --camera cam03

# Docker services
cp .env.example .env   # fill Supabase + Nest credentials
docker compose up --build

# Local Supabase
cd backend/supabase/supabase && npx supabase start

# Web app
bash app_web/setup.sh && cd app_web && npm run dev

# Mobile app
cd app_mobile && ~/development/flutter/bin/flutter run
```

## Tech Stack

| Layer | Technology |
|---|---|
| CV pipeline | Python 3.12, YOLOv8 (ultralytics), BoT-SORT (boxmot), OR-Tools 9.12 |
| Identity | AprilTags 36h11 (~587 IDs), ILP/MCF stitching |
| Data format | Parquet (pyarrow), JSONL audit streams |
| Services | Docker (standalone containers, shared volumes) |
| Backend | Supabase (Postgres + Auth + Storage + Realtime) |
| Mobile | Flutter + supabase_flutter + video_player + geolocator |
| Web | Vite + React + react-router-dom + @supabase/supabase-js |
