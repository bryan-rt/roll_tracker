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
src/calibration_pipeline/ # Gym setup: lens cal, CP18 calibration (mat lines + footpath + fingerprints)
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
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

- **Head:** `7ee966c` | Pipeline A→F verified E2E. Session pipeline validated (3-camera).
- **CP17 Tier 1 implemented:** Two-pass cross-camera ILP with tag corroboration.
- **CP18 complete:** Calibration pipeline with correction integration into Stage A.
  - **Layer 1:** Footpath fitting (primary, continuous signed distance) + mat line detection
    (21/18/7 matches on FP7oJQ/J_EDEw/PPDmUg). Guard: mat lines fall back to footpath-only
    when combined signal conflicts. Results: FP7oJQ 45→68%, J_EDEw 62→71%, PPDmUg 95%.
  - **Layer 2:** Spatial fingerprint registration (occupancy grid cross-correlation +
    boundary contour stitching). Clock-sync independent.
  - **Integration:** Correction matrix loaded in StageAProcessor, applied after
    `project_to_world()`. Config: `stages.stage_A.calibration_correction.enabled`.
  - **H direction:** On disk = mat→img. `multiplex_runner` inverts to img→mat for
    `project_to_world()`. Projected polylines saved at calibration time via
    `cv2.perspectiveTransform(pts, H_mat_to_img)`.
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.
- **Apps:** Flutter tested on Pixel 7 Pro. Web app has mat editor + admin pricing.
- **Supabase:** 23 migrations applied locally and remotely.

## Domain Context (auto-loaded by path)

See `.claude/rules/` — each file has `paths:` frontmatter scoping it to relevant directories.
Full historical decisions archive: `docs/decisions-archive.md` (not auto-loaded).

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
