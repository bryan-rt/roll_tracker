# Undistortion Pipeline Audit (CP17 Tier 2, Part B)

**Date:** 2026-04-02
**Scope:** All code paths in `src/bjj_pipeline/` and `src/calibration_pipeline/` that touch
homography, pixel↔world projection, or frame processing.

## Coordinate Space Convention

| Field | Space | Description |
|-------|-------|-------------|
| `u_px, v_px` | Raw pixel | YOLO bbox contact point on raw video frame |
| `x1, y1, x2, y2` | Raw pixel | YOLO detection bounding box corners |
| `x_m, y_m` | World (meters) | Ground-plane coordinates via `project_to_world()` |
| `x_m_repaired, y_m_repaired` | World (meters) | D3 ILP-repaired world coordinates |
| Homography H (on disk) | mat→img | Stored in `homography.json`, inverted at load time |
| Homography H (runtime) | img→mat | After direction auto-detection in multiplex_runner |
| H pixel space | Undistorted | H trained on undistorted coordinates (CP19) |
| `projected_polylines` | Undistorted pixel | Pre-computed at calibration time |
| `camera_matrix` (K) | Intrinsics | 3x3 camera matrix from lens calibration |
| `dist_coefficients` | Distortion | OpenCV distortion coefficients (k1, k2, ...) |

## Canonical Projection Path

**`project_to_world(pixel_xy, H, camera_matrix, dist_coefficients)`**
(`src/bjj_pipeline/contracts/f0_projection.py`)

1. If K + dist are provided: `cv2.undistortPoints(pts, K, D, P=K)` — raw pixel → undistorted pixel
2. Apply H (img→mat) to undistorted pixel → world coordinates
3. If K + dist are None: identity (raw pixel treated as undistorted)

This is the **only permitted pixel→world path**. All stages must go through this function.

## Audited Code Paths

### 1. F0 Projection — `project_to_world()` (CORRECT)

**File:** `src/bjj_pipeline/contracts/f0_projection.py`

- Conditionally undistorts via `cv2.undistortPoints` before applying H
- Fallback to identity when K/dist are None
- Returns `(x_m, y_m)` world coordinates

**Verdict:** Correct. Canonical path, no issues.

### 2. Stage A — Detection & Tracking (CORRECT)

**File:** `src/bjj_pipeline/stages/detect_track/processor.py` (lines 349-354)

- YOLO runs on **raw BGR frames** (no pre-undistortion)
- Contact points extracted as `(u_px, v_px)` in raw pixel space
- World projection via `project_to_world(pixel, H, camera_matrix, dist_coefficients)`
- K + dist loaded from `CameraProjection` and passed correctly
- Optional CP18 correction matrix applied **after** world projection (in world space)

**Verdict:** Correct. Detection on raw frames, undistortion at projection time only.

### 3. Multiplex Runner — H Loading (CORRECT)

**File:** `src/bjj_pipeline/stages/orchestration/multiplex_runner.py` (lines 105-138)

- Loads H from `homography.json` (mat→img on disk)
- Auto-detects direction via reprojection error against first correspondence point
- Inverts to img→mat when needed
- Loads `camera_matrix` and `dist_coefficients` from same JSON
- Returns `CameraProjection(H, camera_matrix, dist_coefficients)` to Stage A

**Verdict:** Correct. Direction auto-detection is evidence-based, not assumed.

### 4. Stage C — AprilTag Detection (CORRECT)

**File:** `src/bjj_pipeline/stages/tags/`

- AprilTag detection operates on **raw video frames**
- Tag pixel positions (corners) used for bbox overlap with tracklet detections
- Both tag corners and detection bboxes are in raw pixel space — consistent
- No projection to world space needed (association is pixel-space bbox overlap)

**Verdict:** Correct. Raw↔raw comparison, no space mismatch.

### 5. Mat Line Detection (CORRECT)

**File:** `src/calibration_pipeline/mat_line_detection.py`

- `project_world_to_pixel()` (inverse of `project_to_world`) maps world→undistorted pixel
- Pre-computed `projected_polylines` stored in undistorted pixel space
- Frame is undistorted before Hough line detection (`cv2.undistort(frame, K, dist)`)
- Line matching compares Hough lines (undistorted frame) to polylines (undistorted space)

**Verdict:** Correct. Both sides in undistorted space — consistent.

### 6. CP19 Recalibration (CORRECT)

**File:** `tools/cp19_recalibrate.py`

- Raw frame loaded from video
- Frame undistorted: `cv2.undistort(frame_bgr_raw, K, dist)`
- Anchor points undistorted: `cv2.undistortPoints(anchors, K, dist, P=K)`
- H computed: `cv2.findHomography(mat_pts, anchor_undist)` — in undistorted space
- H refinement: `_refine_h_from_mat_lines()` on undistorted frame with K=None (already undistorted)
- Output H valid for undistorted pixel coordinates
- Polylines regenerated from refined H

**Verdict:** Correct. Entire pipeline operates in undistorted space.

### 7. Stage D — Cost Computation (CORRECT)

**File:** `src/bjj_pipeline/stages/stitch/costs.py` (lines 148-173)

- `u_px, v_px` loaded from bank_frames (Stage A output, raw pixel space)
- Used **only** for border distance: distance from pixel to frame edge
- Border distance is frame-boundary geometry — raw pixel space is correct
- `x_m, y_m` used for world-space cost computation (already projected in Stage A)
- No re-projection or undistortion applied

**Verdict:** Correct. Pixel coords used for frame-relative geometry only.

### 8. Stage F — Export/Cropping (CORRECT)

**File:** `src/bjj_pipeline/stages/export/cropper.py`

- Loads `x1, y1, x2, y2` from person_tracks (raw pixel bbox from Stage A)
- Computes quantile-based crop envelope
- Applied directly to ffmpeg crop filter on raw video
- No projection involved

**Verdict:** Correct. Raw pixel bboxes applied to raw video frames.

### 9. Calibration Evaluation — Tracklet Classifier & Mat Walk (CORRECT)

**Files:** `src/calibration_pipeline/tracklet_classifier.py`, `src/calibration_pipeline/mat_walk.py`

- Operate on `x_m, y_m` world coordinates from Stage A/D output
- Classification uses spatial extent, velocity, on-mat fraction, edge distance
- No pixel coordinates used
- World coordinates already went through `project_to_world()` with undistortion

**Verdict:** Correct. Pure world-coordinate consumers.

## Known Limitations (Not Bugs)

**Stage A standalone runner** (`src/bjj_pipeline/stages/detect_track/run.py`, lines 82-87):
The isolation-path H loader does not auto-detect H direction (unlike multiplex_runner).
This path is never used in production — the pipeline always runs via `multiplex_AC` which
uses multiplex_runner. Documented pre-existing limitation, not a CP17 issue.

## Summary

All 9 code paths correctly handle K + dist:

- Raw frames input to all detectors (YOLO, AprilTag, Hough)
- Undistortion applied **only** inside `project_to_world()` (single canonical path)
- H trained on undistorted coordinates (CP19 calibration pipeline)
- Downstream stages (D/E/F) consume already-projected world coordinates
- Pixel coordinates in output tables remain in raw space for ffmpeg/visualization

**No fixes required.**
