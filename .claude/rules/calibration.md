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

## CP18 Stubs (planned)
- `mat_walk.py` — tagged person walks grid, produces labeled correspondences across cameras.
  Least-squares affine solve for global coordinate consistency.
- `drift_detection.py` — empty-mat baseline snapshot, daily edge comparison, drift score
  to Supabase. Alert to gym owner on severe drift.
- `inter_camera_sync.py` — cross-camera affine alignment from mat walk data.
