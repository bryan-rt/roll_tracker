---
paths:
  - "src/calibration_pipeline/**"
  - "configs/cameras/**"
---

# Calibration Pipeline

Separate top-level module (`src/calibration_pipeline/`) — gym setup and maintenance
workflows, not per-session pipeline stages. Outputs feed bjj_pipeline via config files
and Supabase.

## CP19 Unified Calibration (current, replaces CP18 affine correction)

### Calibration Wizard (`calibrate_camera.py`)

Unified 3-step CLI: initial H → lens cal → final H refinement. Auto-resumes from any
interruption. `--skip-lens` for H-only recal, `--force` to redo all steps, `--verify`
for cross-camera agreement check. See `docs/calibration_guide.md`.

### Phase A — Polyline Lens Calibration

Detects edge points along projected polylines across entire visible mat (~100-170 points,
9-16 edges). Powell optimization of collinearity cost (f, k1, k2). 3 free params:
f (200–3000), k1 (±1.0), k2 (±1.0). Directly minimizes perpendicular distance from
fitted lines.

Falls back to existing K+dist when optimizer hits bounds or when existing calibration
available (interactive `lens_calibration` tool is more reliable than automated detection
on busy gym frames).

### Phase B — Mat-Line H Refinement

Canny+Hough line detection → match to projected polylines → extract dense world↔pixel
correspondences → RANSAC homography with anchor+line points. Iterative (max 3),
converges at <0.1px mean reproj change.

### Coordinate Space Handling

`_recompute_h_for_space()` transforms anchor points between raw/old-undistorted/
new-undistorted pixel spaces. Phase A operates on raw frame with H_mat_to_raw.
Phase B operates on undistorted frame with H in matching space.

### Empty Frame Selection

`_find_empty_frame()` picks the frame closest to temporal median (least activity/people).
Used by recalibration script with calibration_test videos from
`data/raw/nest/calibration_test/{cam_id}/`.

### Quality Metrics

Saved in `homography.json["quality_metrics"]`: `h_metrics` (reproj error, inliers,
matched lines) + `lens_metrics` + `calibration_mode`. QA overlay displays metrics
text block at top-right.

### Verified Results (calibration_test empty-mat frames)

- FP7oJQ: 17 lines/11 edges, 61% inliers, 1.3px reproj, converged.
- J_EDEw: 11 lines/6 edges, 66% inliers, 1.0px reproj.
- PPDmUg: 8 lines/7 edges, 82% inliers, 1.2px reproj, converged.

### Cross-Camera Verification (`calibration_verify.py`)

Pairwise world-coordinate agreement diagnostic. Compares where cameras place shared
blueprint edges. Thresholds: <5cm excellent, 5-15cm acceptable, >15cm investigate.
**Result: 9mm worst-case pairwise deviation.**

### Batch Recalibration (`cp19_recalibrate.py`)

Re-runs Phase A+B on all cameras using existing anchor correspondences + calibration_test
videos. Intended for daily H drift correction.

### Integration with Save Handlers

Runs automatically after user places anchor corners in both clicks + overlay_rect modes:
Phase A → Phase B → QA with metrics → save. Projected polylines regenerated from
refined H at save time.

## H Direction Convention

- **On disk** (`homography.json`): H = **mat→img** (validated by correspondences).
- **In pipeline** (`multiplex_runner`): inverts H to **img→mat** for `project_to_world()`.
  Auto-detects direction by testing first correspondence point error in both directions.
- **Projected polylines**: generated via `H_mat_to_img` (the on-disk direction) since
  they map blueprint edges to image pixel space.

## Correction Layers (historical context)

1. **Lens calibration** — one-time per camera, essentially permanent. K + dist coefficients.
2. **Per-camera homography** — daily recalibration via `cp19_recalibrate.py`.
3. **Calibration correction** (CP18, superseded) — per-camera affine correction from mat
   walk analysis. Files: `configs/cameras/{cam_id}/calibration_correction.json`. Loaded
   in Stage A when present. **Abandoned** due to J_EDEw regression (99.6→87.6%).
   CP19 direct H refinement replaces this approach entirely.

## Supporting Modules

- `blueprint_geometry.py`: MatBlueprint class — Shapely polygon union of panel rectangles,
  geometric queries (contains, nearest edge, signed distance).
- `mat_line_detection.py`: Detects lines via HoughLinesP, matches against projected
  polylines from `homography.json`. Used by both CP18 mat_walk and CP19 H refinement.
- `tracklet_classifier.py`: Classifies tracklets as cleaning/lingering, detects
  perpendicular vs parallel edge crossings for correspondence quality.
- `lens_calibration.py`: Interactive tool for manual edge point selection + automated
  edge detection. Produces K + dist.

## Pipeline Integration

Stage A `run()` loads correction matrix if present (CP18 path, now inactive):
1. Checks `stages.stage_A.calibration_correction.enabled` (default True).
2. Reads `configs/cameras/{camera_id}/calibration_correction.json`.
3. Applies 2×3 affine transform to `(x_m, y_m)` after `project_to_world()`.
4. With CP19, no correction files are generated — H is refined directly.

## Stubs (future)

- `drift_detection.py` — empty-mat baseline snapshot, daily edge comparison, drift score
  to Supabase. Alert to gym owner on severe drift.
