# CLAUDE.md — Roll Tracker

## Project

BJJ gym SaaS pipeline. Nest cameras → YOLO+BoT-SORT tracking → AprilTag identity →
ILP stitching → per-athlete match clips → Supabase → Flutter app.
**Repo:** github.com/bryan-rt/roll_tracker | **Branch:** `services_uploader` | **Python 3.12**

## Working Methodology

**Three-pass protocol (mandatory for all non-trivial tasks):**
1. **Pass 1 — Explore** (Plan Mode: shift+tab ×2): Read Task Brief, explore relevant files,
   identify conflicts. ⏸ STOP — summarize and wait for approval.
2. **Pass 2 — Specify** (Plan Mode continues): Plan exact changes, verify naming/contracts
   against live code. ⏸ STOP — present plan and wait for approval.
3. **Pass 3 — Execute**: Implement, test, update CLAUDE.md if architecture changed,
   commit+push. ⏸ STOP — summarize and wait for review.

**Evidence-driven design:** Do not code from assumptions. When behavior is uncertain:
enhance logging → inspect real output → plan from evidence. Propose instrumentation
before fixes when root cause is unclear.

**Ambiguity protocol:** Surface naming conflicts, missing files, or uncovered architectural
questions in Pass 1. Do not resolve silently or guess.

## Monorepo Layout

```
src/bjj_pipeline/        # CV pipeline package (stages A→F, contracts, config, core)
src/calibration_pipeline/ # Gym setup: lens cal, H refinement, mat line detection
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
docs/                     # Calibration guide, decisions archive, audits
.claude/rules/            # Domain-specific context (auto-loaded by path scope)
```

## Critical Constraints

- **NumPy < 2** — Torch ABI. Install ultralytics/boxmot with `--no-deps`.
- **Supabase is the exclusive integration hub** — no direct service-to-service communication.
- **Phase 1/2 parallelism boundary (NON-NEGOTIABLE)** — A+C parallel, D+E+F sequential.
- **No cross-stage imports** — stages communicate only via F0 contracts + filesystem.
- **Option B undistort-on-projection** — `project_to_world()` is the only permitted
  pixel→world path. No stage calls homography directly.

## Coding Conventions

- Stage contract: `run(config: dict, inputs: dict) -> dict`
- Pydantic v2 for data models. Loguru for logging. Rich for CLI. Typer for CLI defs.
- Parquet for tabular data. JSONL for audit/event streams. Type hints everywhere.
- Debug artifacts → `outputs/<clip_id>/_debug/`. Never pollute stage output dirs.
- Paths via `ClipOutputLayout` and env vars — no hardcoding.

## Config Resolution

`default.yaml` → `cameras/<cam_id>.yaml` → `cameras/<cam_id>/homography.json` → `--config` CLI overlay

## Current Status

*Last updated 2026-04-05.*

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).
**CP20:** YOLOv8n-pose model, isolation gate, HSV color histograms, Tier 3 histogram
cross-camera evidence. Stage A outputs 3 new sidecars: keypoints.parquet,
color_histograms.parquet, tracklet_histogram_summaries.parquet.
- Camera geometry analysis tool complete (v6 pose decomposition, 4-phase)
- Lens calibration bounds fix applied (fixed-f candidate sweep)
- H coordinate space verified as undistorted pixel space
- Calibration wizard re-run for all 3 cameras with updated lens cal
- Cross-camera agreement verified (sub-cm, 9mm worst-case)
- ROI mask union fix: brief written, not yet applied (pending)
**CP22:** Default detection model updated to yolo26n-pose (STAL loss, better small-object
detection). CoreML export validated for both yolov8n-pose and yolo26n-pose — ~2x speedup
over MPS (14.4ms vs 27.4ms per frame). CoreML is a viable future `device` option but
needs a config knob designed before changing the default.
- ultralytics upgraded 8.3.252 → 8.4.33 (`--no-deps`)
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.

See `.claude/rules/` for domain-specific documentation (auto-loaded by path).
See `docs/decisions-archive.md` for full checkpoint history.

## Domain Context (auto-loaded by path)

| Rule file | Scope |
|-----------|-------|
| `calibration.md` | `src/calibration_pipeline/**`, `configs/cameras/**` |
| `cross-camera.md` | `src/bjj_pipeline/stages/stitch/**` |
| `pipeline-stages.md` | `src/bjj_pipeline/**` |
| `services.md` | `services/**` |
| `commands.md` | Common dev commands |
| `apps.md` | `app_mobile/**`, `app_web/**` |
| `supabase.md` | `backend/supabase/**` |

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
