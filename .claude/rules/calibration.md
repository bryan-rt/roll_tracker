---
paths:
  - "src/calibration_pipeline/**"
  - "configs/cameras/**"
---

# Calibration Pipeline

Separate top-level module (`src/calibration_pipeline/`) — gym setup and maintenance
workflows, not per-session pipeline stages. Outputs feed bjj_pipeline via config files
and Supabase.

## Two-Step Calibration Chain
1. `lens_calibration.py` → estimates K (camera matrix) + dist (distortion coefficients).
   Writes to `homography.json` under `camera_matrix`, `dist_coefficients`, `lens_calibration`
   metadata (method, per-edge RMS, auto/manual point counts).
2. `homography_calibrate.py` → auto-undistorts frame when K+dist present, then computes H.
   Uses read-then-merge save to preserve K+dist fields. Generates `projected_polylines`
   (dense-sampled mat edge points projected via `cv2.perspectiveTransform(pts, H_mat_to_img)`)
   and saves them in `homography.json` for downstream mat line detection.

## H Direction Convention
- **On disk** (`homography.json`): H = **mat→img** (validated by correspondences).
- **In pipeline** (`multiplex_runner`): inverts H to **img→mat** for `project_to_world()`.
  Auto-detects direction by testing first correspondence point error in both directions.
- **Projected polylines**: generated via `H_mat_to_img` (the on-disk direction) since
  they map blueprint edges to image pixel space.

## Three Correction Layers (different update frequencies)
1. **Lens calibration** — one-time per camera, essentially permanent.
2. **Per-camera homography** — nightly recalibration attempt.
3. **Calibration correction** — per-camera affine correction from mat walk analysis.
   Files: `configs/cameras/{cam_id}/calibration_correction.json`. Loaded in Stage A
   `run()` and applied after `project_to_world()`. Enabled by default
   (`stages.stage_A.calibration_correction.enabled`).

## Lens Calibration Details
- Auto-detects mat edge points via 1D gradient analysis (50–100+ points/edge, sub-pixel).
- User can add/delete points manually.
- Solver: collinearity optimization via `scipy.optimize.minimize` (Powell, bounded).
  3 free params: f (200–3000), k1 (±1.0), k2 (±1.0). Directly minimizes perpendicular
  distance from fitted lines. Replaces cv2.calibrateCamera (over-fit with 6 params).

## CP18 — Homography Refinement (implemented)

### Layer 1: Single-Camera Correction (`mat_walk.py`)

Two independent signals from mat cleaning footage:

- **Footpath fitting (primary):** Continuous signed distance from tracklet positions to
  nearest mat edge. Uses RANSAC affine fit on edge-crossing positions. Identity-regularized
  6-param affine prevents wild solutions. Quality gates: >40% coverage, 6+ edge touches,
  2+ distinct edges.
- **Mat line detection (`mat_line_detection.py`):** Detects lines in video frames via
  HoughLinesP, matches against projected polylines from `homography.json`. Results:
  21/18/7 matches on FP7oJQ/J_EDEw/PPDmUg. **Guarded:** mat lines fall back to
  footpath-only when combined signal conflicts with footpath.

Output: `calibration_correction.json` per camera (2×3 affine correction matrix).

### Layer 2: Cross-Camera Alignment (`inter_camera_sync.py`)

Spatial fingerprint registration:
- Occupancy grid cross-correlation + boundary contour stitching.
- Clock-sync independent — uses spatial overlap, not temporal co-observation.
- Opportunistic — never blocks Layer 1.

### Supporting Modules
- `blueprint_geometry.py`: MatBlueprint class — Shapely polygon union of panel rectangles,
  geometric queries (contains, nearest edge, signed distance).
- `tracklet_classifier.py`: Classifies tracklets as cleaning/lingering, detects
  perpendicular vs parallel edge crossings for correspondence quality.
- `calibrate.py`: Orchestrator with CLI. Runs Layer 1 per camera, Layer 2 per pair,
  writes reports + correction JSONs.

## Pipeline Integration

Stage A `run()` in `detect_track/run.py` loads the correction matrix:
1. Checks `stages.stage_A.calibration_correction.enabled` (default True).
2. Reads `configs/cameras/{camera_id}/calibration_correction.json`.
3. Applies 2×3 affine transform to `(x_m, y_m)` after `project_to_world()`.
4. Passes corrected coordinates to `StageAProcessor` for on-mat classification.

`multiplex_runner` logs `CP18 correction loaded for {cam_id}` on successful load.

## A/B Comparison Results (2026-03-30)

Full pipeline re-run (36 clips, 3 cameras) with corrections vs baseline:
- **FP7oJQ:** on_mat 97.3→98.0% (+0.7%). 986 positions moved on-mat, 0 regressed. Good.
- **J_EDEw:** on_mat 99.6→87.6% (**-12%**). 18,902 positions shifted off-mat along east
  edge (x_m > 58). Correction overshoots in x direction. Needs investigation.
- **PPDmUg:** 91.8% unchanged. Correction has no visible effect.

## Planned: Recompute H from Polyline Correspondences

The affine correction layer is inherently limited — it post-adjusts an imprecise H rather
than fixing it. Planned approach: use matched mat line correspondences (image-space lines
↔ blueprint edges) as direct inputs to recompute H, bypassing the correction layer entirely.
The `projected_polylines` in `homography.json` already provide the reference geometry.

## Stubs (future)
- `drift_detection.py` — empty-mat baseline snapshot, daily edge comparison, drift score
  to Supabase. Alert to gym owner on severe drift.
