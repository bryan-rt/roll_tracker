# B3 — Camera Calibration: Homography Preflight + Drift Monitor

## Update: Locked constraints to honor (F0 + F3)

- **F3 ingest contract**: clips live at `data/raw/nest/<camera_id>/YYYY-MM-DD/HH/<camera_id>-YYYYMMDD-HHMMSS.mp4`
- **Processing reads only** from `data/raw/**` and **writes only** to `outputs/<clip_id>/...`
- **F0 contracts are authoritative**: do not invent schemas; use `src/bjj_pipeline/contracts/*`
- **Run anchor**: `outputs/<clip_id>/clip_manifest.json`
- **Stage artifacts are locked** (Parquet/JSONL + masks `.npz`), and paths must be **relative** to `outputs/<clip_id>/`

---

## Why this worker exists
We rely on **homography** to map image pixels to real-world mat coordinates. For a gym-ready workflow, we must:
1) **Check if the camera has a homography already** before running the pipeline.
2) If not, **create one interactively** using the camera’s first frame and a `mat_blueprint.json`.
3) Optionally detect when calibration has drifted (“camera jostle”) and invalidate/refresh.

This worker defines the calibration UX, file formats, and validation rules so downstream stages (B2, D5, E1) can trust the coordinate system.

---

## Scope
### In-scope
- Homography **preflight**: “if missing → run calibrator”
- `mat_blueprint.json` **schema** (minimal, stable)
- Interactive calibration tool:
  - side-by-side: **video frame** vs **blueprint plot**
  - capture **4+** corresponding points (recommend 6–10)
  - compute `H` with RANSAC
  - display reprojection overlay + error
  - save homography artifact for the camera
- File locations + naming conventions (camera-scoped)
- Calibration validation:
  - reprojection RMS threshold
  - sanity checks (e.g., mat corners project inside image)
- Drift/jostle monitoring **design** (POC can implement only the metrics + “warn/fail” behavior)

### Out-of-scope (for now)
- Fully automatic calibration with no user clicks
- Complex lens undistortion calibration (but you can propose “phase 2”)
- Multi-camera calibration

---

## Inputs & Outputs

### Inputs
- `data/raw/nest/.../<clip>.mp4` (first frame only for calibration)
- `configs/mat_blueprints/mat_blueprint.json` (path configured)
- `configs/cameras/<camera_id>.yaml` (camera config)

### Outputs (camera-scoped)
Choose ONE canonical storage and document it:
- Option A: `configs/cameras/<camera_id>/homography.npz`
- Option B: `configs/cameras/<camera_id>/homography.yaml`

Must include:
- `H` (3x3)
- `video_points` (Nx2)
- `mat_points` (Nx2) in meters
- `blueprint_hash`
- `created_at`
- `reprojection_error_px` summary

---

## Required integration points
### With F1 (Orchestration)
- Add a **preflight** step: ensure homography exists for `camera_id`
- If missing: launch calibrator, save artifact, then proceed

### With F2 (Config)
- Camera config must specify:
  - path to homography artifact
  - path to blueprint
  - optional drift thresholds

### With B2 (Contact points + homography)
- B2 must read the homography artifact produced here and apply it to contact points.

### With D5 (MatZone gating)
- D5 will use the blueprint polygon + homography space to define “on mat” vs “off mat”.

---

## Design requirements for `mat_blueprint.json`
Minimum fields:
- `schema_version`
- `units` (must be `"meters"`)
- `mat_name`
- `boundary_polygon_m` (list of [x,y] points, closed or open but documented)
- Optional: `zones` (named polygons), `line_segments` for UI snapping

---

## Calibration UX spec (POC)
1) Load first frame from clip and render it on the left
2) Render blueprint polygon on right (mat top-down)
3) Click flow:
   - user alternates clicks: **video point**, then corresponding **blueprint point**
   - show counters + list
   - require N>=4
4) Compute with OpenCV:
   - `cv2.findHomography(video_pts, mat_pts, cv2.RANSAC)`
5) Validate:
   - project blueprint polygon back onto video to overlay
   - compute reprojection RMS
6) Save artifact and print path

Implementation note:
- use a minimal UI approach first (matplotlib + event handlers is OK).

---

## Drift / jostle monitoring (design)
Define a “health check” metric that can run:
- at pipeline start
- periodically when mat is empty (optional)

Options:
- static background anchors (ORB feature match vs “golden frame”)
- mat line alignment (Hough/LSD) vs projected blueprint edges

POC requirement:
- define the metric + thresholds and write audit events; implement full auto-recal later.

---

## Deliverables back to Manager
1) **Spec**: file formats + locations + config keys
2) **UX**: click workflow + validation overlays
3) **Algorithms**: how H computed, how error computed, thresholds
4) **Copilot prompt pack**: prompts to generate the calibrator module + tests
5) **Acceptance criteria**:
   - create homography for cam03 from a clip
   - reuse existing homography on subsequent runs (no prompts)
   - fail fast if homography missing and non-interactive mode enabled

---

## Kickoff
Please start by drafting:
1) The **exact file paths** and config keys you recommend (camera config + blueprint config).
2) The **mat_blueprint.json schema** (example file).
3) The **calibration UI plan** (what library + how to run from CLI).

End by asking me to approve the plan before implementation.

Also include an explicit bullet confirming alignment with the locked F0/F3 constraints (paths, manifest, artifacts).
