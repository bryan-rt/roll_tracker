---
layout: page
---

# D6 — Homography Calibrator Tool + F1 Interactive Preflight Wiring

Owner: Worker D6 (new)
Status: NOT STARTED
Depends on: F0 (contracts), F1 (orchestration CLI), F2 (config system), Z3 (multiplex mode)
Priority: HIGH (blocks reliable Stage A runs for new cameras)

## 0) Mission

Implement the homography calibration tool that F1 preflight expects when `--interactive` is provided.

Right now:
- Non-interactive mode correctly fail-fast when homography is missing.
- Interactive mode attempts to run `python -m bjj_pipeline.tools.homography_calibrate --camera <camera_id>` but the module does not exist.

This worker will:
1) Create the calibrator module and package path.
2) Implement an interactive UI that lets the user create `configs/cameras/<camera_id>/homography.json`.
3) Ensure the orchestration `--interactive` path works in both multipass and multiplex_ABC modes.

## 1) Canonical Inputs / Outputs (LOCKED)

### Homography file location (LOCKED)
`configs/cameras/<camera_id>/homography.json`

### Inputs required for calibration (LOCKED)
- `mat_blueprint.json` (canonical mat coordinate blueprint; location expected from config)
- First frame of a target clip (or explicit `--clip` for calibration tool)

### Output required (LOCKED)
`homography.json` must minimally contain:
- a `H` 3×3 matrix (list-of-lists or flat list of 9)
- metadata for reproducibility:
  - `camera_id`
  - `created_at`
  - `source_clip` (or `source_image`)
  - `point_pairs` (pixel ↔ mat coords)
  - optional `rmse`/reprojection_error

No new pipeline artifacts. This is a camera config artifact only.

## 2) User Experience Requirements

### Orchestration behavior
If Stage A is in the run window:

1) Check `configs/cameras/<camera_id>/homography.json`
2) If missing:
   - without `--interactive`: fail-fast with clear message (already works)
   - with `--interactive`: launch calibrator, then re-check, then continue or fail

### Calibration UI (interactive)
The calibrator must show a side-by-side view:
- LEFT: video first frame (pixel space)
- RIGHT: rendered mat blueprint (mat coordinates)

User selects 4+ correspondences alternating:
1) click point in video
2) click corresponding point in mat plot
(repeat)

Then tool computes homography and saves the JSON.

Must support:
- Undo last pair
- Clear all
- Save + exit
- Quit without writing

Minimal friction: gym-owner friendly.

## 3) CLI Surface Area

### New module (required)
Create package:
`src/bjj_pipeline/tools/__init__.py`
`src/bjj_pipeline/tools/homography_calibrate.py`

### CLI contract (required)
`python -m bjj_pipeline.tools.homography_calibrate --camera cam01 --clip <path optional> [--out <path>]`

Suggested args:
- `--camera <camera_id>` (required)
- `--clip <path>` (optional but recommended; if omitted, the tool uses latest clip found for that camera under data/raw/** OR fails with instructions)
- `--frame-index 0` (default 0)
- `--mat-blueprint <path>` (optional override; default from config)
- `--out <path>` (default canonical homography.json location)
- `--overwrite` (allow overwrite)
- `--headless` (non-interactive compute-only mode if points provided via JSON for tests)

## 4) Implementation Plan

### 4.1 Load blueprint + frame
- Load resolved config (via F2 loader) to get:
  - mat blueprint path (canonical)
  - mat coordinate extents
- Read first frame using OpenCV `cv2.VideoCapture`.
- Convert to RGB for display.

### 4.2 Render mat blueprint view
- Parse `mat_blueprint.json` and plot with matplotlib:
  - mat boundaries, centerlines, zones if present
  - axes labeled in meters (or blueprint units)
- Allow clicks inside the plot; store (X,Y).

### 4.3 Interactive point collection
Use matplotlib event handlers:
- Two axes in one figure (video axis + mat axis)
- Maintain state machine: expecting_video_point / expecting_mat_point
- Draw markers + labels on both sides as user clicks
- Keyboard shortcuts:
  - `u`: undo last pair
  - `c`: clear
  - `s`: solve+save
  - `q`: quit without saving

### 4.4 Solve homography
- Require >= 4 point pairs
- Compute homography:
  - OpenCV: `cv2.findHomography(src_pts, dst_pts, method=0)` or RANSAC optional
- Compute reprojection error (RMSE) for user feedback.
- Save JSON with fields:
  - `camera_id`
  - `H` (3x3)
  - `pixel_points` and `mat_points`
  - `rmse`
  - timestamps + tool version

### 4.5 Validation helper
Implement a lightweight validator used by orchestration preflight:
- file exists
- JSON parses
- H shape is valid (3x3 numeric)
- determinant not near zero (basic sanity)
- optional: warn if rmse too high

### 4.6 Wiring into F1 preflight
Wherever preflight calls the module:
- ensure it passes BOTH `--camera` and `--clip` if available
- ensure errors are surfaced cleanly
- after calibrator exits, re-check existence/validity
- emit orchestration audit events:
  - homography_preflight_started
  - homography_preflight_missing
  - homography_preflight_calibrator_launch
  - homography_preflight_succeeded
  - homography_preflight_failed

## 5) Test Plan (Required)

### Unit tests
- Homography JSON validation:
  - missing fields -> fail
  - bad H shape -> fail
  - good H -> pass

### Integration tests
- Simulate missing homography + `--interactive`:
  - mock subprocess call to calibrator that writes homography.json
  - pipeline continues

- Failure path:
  - calibrator returns non-zero or doesn’t create file
  - orchestration fails with clear message

### Non-interactive/headless test mode
Support `--headless --points <json>` so CI can test solving without GUI.

## 6) Acceptance Criteria

✅ Running:
`roll-tracker run ... --camera cam01 ...`
with missing homography and no `--interactive`
→ fail-fast (already works)

✅ Running same command WITH `--interactive`
→ launches calibrator, produces `configs/cameras/cam01/homography.json`, resumes pipeline

✅ `python -m bjj_pipeline.tools.homography_calibrate --camera cam01 --clip <path>`
works end-to-end and writes valid JSON

✅ No changes to F0 artifacts, schemas, or outputs/*
✅ Calibrator output is camera-scoped config only
✅ Works in both multipass and multiplex_ABC runs

## 7) Non-Goals (Explicit)

- Drift monitoring / auto-recalibration (future B3/B4 style work)
- Lens distortion calibration
- Multi-camera stitching calibration
- Writing any outputs/<clip_id>/ artifacts

## 8) Notes / Known Failure Currently

Current error:
`ModuleNotFoundError: No module named 'bjj_pipeline.tools'`

Root cause:
The tools package and the calibrator module do not exist yet.
This worker fixes that by adding `src/bjj_pipeline/tools/` and implementing `homography_calibrate.py`.


---

# ✅ D7 Completion Report (Authoritative)

**Worker ID:** D7  
**Title:** Homography Calibrator & Preflight Tool  
**Status:** COMPLETE  
**Sign-off Date:** 2026-01-08  

---

## Executive Summary

Worker D7 is complete and production-ready.  
All responsibilities defined in the original D7 scope have been fully implemented, validated, and integrated into the pipeline without breaking backward compatibility.

This worker now provides the **canonical, auditable, and operator-friendly mechanism** for producing camera-to-mat homographies used throughout the roll_tracker system.

---

## Responsibilities — Final Status

### 1. Canonical Homography Artifact
✅ Implemented  
- Outputs `configs/cameras/<camera_id>/homography.json`
- Enforced 3×3 numeric validity and normalization
- Includes provenance metadata, timestamps, and correspondence data

### 2. Interactive Calibration (Clicks Mode)
✅ Implemented  
- Alternating image ↔ mat blueprint point collection
- Undo / clear / redo support
- RANSAC-based homography solve
- QA overlay confirmation before save

### 3. Overlay-Based Calibration (overlay_rect)
✅ Implemented (Extended beyond original scope)  
- Draggable image-space quad
- Live warped mat-blueprint overlay on frame
- Blueprint tab for selecting anchor rectangles
- Internal rectangle anchoring (not limited to outer perimeter)
- Rotation, flipping, scaling, fine translation controls
- Explicit preservation of corner IDs and mat coordinates

### 4. Preflight Enforcement
✅ Implemented  
- Homography must exist before downstream stages proceed
- Placeholder mode provided for onboarding/testing
- Invalid or missing homographies fail loudly

### 5. QA Visualization
✅ Implemented  
- Union-of-rectangles grid rendering (0.5 m spacing)
- Grid projected via solved homography
- Operator must explicitly accept or reject
- Reject path loops back without writing artifacts

### 6. Determinism & Auditability
✅ Implemented  
- All user inputs recorded
- Calibration UI mode stored
- Anchor rectangle metadata persisted
- QA acceptance explicitly logged

---

## Output Contracts (Locked)

Downstream workers may rely on:

- `homography.json["H"]` maps **mat → image**
- Grid overlays use the same directionality
- Anchor rectangle metadata is optional but stable
- Placeholder homographies are explicitly marked

These contracts are now considered **stable**.

---

## Known Non-Goals (By Design)

The following are intentionally out of scope for D7:

- Continuous homography drift correction
- Automatic calibration without operator involvement
- Multi-camera joint calibration
- Temporal homography changes

These are future workers if needed.

---

## Downstream Impact

D7 unblocks and stabilizes:

- Stage B (contact point homography projection)
- Stage C (AprilTag localization)
- Stage D (global identity stitching)
- Stage E (match/session detection)
- Stage F (QA overlays and exports)

No downstream worker is required to implement calibration logic.

---

## Final Assessment

Worker D7 meets or exceeds all original design goals.  
The additional overlay-based calibration UI significantly improves usability without compromising correctness or determinism.

**D7 is officially complete.**

---

