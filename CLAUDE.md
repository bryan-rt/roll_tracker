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

- **Head:** `22cd0ae` | Pipeline A→F verified E2E. Session pipeline validated (3-camera).
- **CP17 Tier 1 implemented:** Two-pass cross-camera ILP with tag corroboration.
- **CP17 Tier 2 implemented (2026-04-02):** Coordinate evidence channel. Compares D4
  person track world coordinates across camera pairs via rolling-window spatial proximity.
  Coordinate-corroborated tags merged into `corroborated_tags` (same 10x boost as tag
  evidence). Conflicts logged to session audit JSONL (Signal C — audit-only, no user-facing
  behavior yet). Config: `cross_camera.coordinate_evidence` (disabled by default).
- **Undistortion pipeline audit complete (2026-04-02):** All 9 code paths verified correct.
  Convention: u_px/v_px = raw pixel, x_m/y_m = world via `project_to_world()`. No fixes
  needed. See `docs/undistortion_audit.md`.
- **CP18 calibration pipeline complete:** Layer 1 (footpath + mat line) + Layer 2
  (fingerprint). Affine correction approach abandoned — J_EDEw regression (99.6→87.6%).
- **CP19 unified calibration pipeline implemented (2026-04-01):**
  - **Replaces CP18 affine correction** with direct H refinement from mat-line observations.
  - **Phase A** (polyline lens cal): Detects edge points along projected polylines across
    entire visible mat (~100-170 points, 9-16 edges). Powell optimization of collinearity
    cost (f, k1, k2). Falls back to existing K+dist when optimizer hits bounds or when
    existing calibration available (interactive lens_calibration tool is more reliable
    than automated detection on busy gym frames).
  - **Phase B** (mat-line H refinement): Canny+Hough line detection → match to projected
    polylines → extract dense world↔pixel correspondences → RANSAC homography with
    anchor+line points. Iterative (max 3), converges at <0.1px mean reproj change.
  - **Coordinate space handling:** `_recompute_h_for_space()` transforms anchor points
    between raw/old-undistorted/new-undistorted pixel spaces. Phase A operates on raw
    frame with H_mat_to_raw. Phase B operates on undistorted frame with H in matching space.
  - **Empty frame selection:** `_find_empty_frame()` picks the frame closest to temporal
    median (least activity/people). Used by recalibration script with calibration_test
    videos from `data/raw/nest/calibration_test/{cam_id}/`.
  - **Quality metrics** saved in `homography.json["quality_metrics"]`:
    `h_metrics` (reproj error, inliers, matched lines) + `lens_metrics` + `calibration_mode`.
  - **QA overlay** enhanced: displays metrics text block at top-right.
  - **Results (calibration_test empty-mat frames):**
    FP7oJQ: 17 lines/11 edges, 61% inliers, 1.3px reproj, converged.
    J_EDEw: 11 lines/6 edges, 66% inliers, 1.0px reproj.
    PPDmUg: 8 lines/7 edges, 82% inliers, 1.2px reproj, converged.
  - **Integrated into both save handlers** (clicks + overlay_rect modes). Runs automatically
    after user places anchor corners: Phase A → Phase B → QA with metrics → save.
  - **Batch recalibration:** `tools/cp19_recalibrate.py` re-runs Phase A+B on all cameras
    using existing anchor correspondences + calibration_test videos.
  - **H direction:** On disk = mat→img (unchanged). Projected polylines regenerated from
    refined H at save time.
  - **Calibration wizard** (`calibrate_camera.py`): Unified 3-step CLI
    (initial H → lens cal → final H refinement). Auto-resumes from any interruption.
    `--skip-lens` for H-only recal, `--force` to redo all steps, `--verify` for
    cross-camera agreement check. See `docs/calibration_guide.md`.
  - **Cross-camera verification** (`calibration_verify.py`): Pairwise world-coordinate
    agreement diagnostic. Compares where cameras place shared blueprint edges.
    Thresholds: <5cm excellent, 5-15cm acceptable, >15cm investigate.
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.
- **Apps:** Flutter tested on Pixel 7 Pro. Web app has mat editor + admin pricing.
- **Supabase:** 23 migrations applied locally and remotely.

## Domain Context (auto-loaded by path)

See `.claude/rules/` — each file has `paths:` frontmatter scoping it to relevant directories.
Full historical decisions archive: `docs/decisions-archive.md` (not auto-loaded).

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
