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
   Uses read-then-merge save to preserve K+dist fields.

## Lens Calibration Details
- Auto-detects mat edge points via 1D gradient analysis (50–100+ points/edge, sub-pixel).
- User can add/delete points manually.
- Solver: collinearity optimization via `scipy.optimize.minimize` (Powell, bounded).
  3 free params: f (200–3000), k1 (±1.0), k2 (±1.0). Directly minimizes perpendicular
  distance from fitted lines. Replaces cv2.calibrateCamera (over-fit with 6 params).

## Three Correction Layers (different update frequencies)
1. **Lens calibration** — one-time per camera, essentially permanent.
2. **Per-camera homography** — nightly recalibration attempt.
3. **Inter-camera affine alignment** — derived from mat walk, updated when drift detected.

## CP18 — Homography Refinement (implemented)

Two-layer calibration from mat cleaning footage:

- **Layer 1** (`mat_walk.py`): Single-camera RANSAC affine correction. Uses tracklet
  birth/death positions near mat edges as correspondences. Identity-regularized 6-param
  affine fit prevents wild solutions. Quality gates: >40% coverage, 6+ edge touches,
  2+ distinct edges. Writes `calibration_correction.json` per camera.
- **Layer 2** (`inter_camera_sync.py`): Cross-camera alignment via overlap (co-temporal
  detections) or handoff (death→birth matching) methods. Opportunistic — never blocks
  Layer 1.
- `blueprint_geometry.py`: MatBlueprint class — Shapely polygon union of panel rectangles,
  geometric queries (contains, nearest edge, signed distance).
- `tracklet_classifier.py`: Classifies tracklets as cleaning/lingering, detects
  perpendicular vs parallel edge crossings for correspondence quality.
- `calibrate.py`: Orchestrator with CLI. Runs Layer 1 per camera, Layer 2 per pair,
  writes reports + correction JSONs.

## Stubs (future)
- `drift_detection.py` — empty-mat baseline snapshot, daily edge comparison, drift score
  to Supabase. Alert to gym owner on severe drift.
