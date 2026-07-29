# Camera Calibration Guide

## Prerequisites

- **Video file** per camera from `data/raw/nest/calibration_test/{cam_id}/` — ideally
  with an empty mat (no people). The wizard auto-selects the emptiest frame.
- **`configs/mat_blueprint.json`** — rectangle layout of your mat panels.
- Python environment with `bjj_pipeline` and `calibration_pipeline` installed.

## Quick Start

```bash
# Calibrate one camera (full 3-step workflow)
python -m bjj_pipeline.tools.calibrate_camera \
  --camera J_EDEw \
  --video data/raw/nest/calibration_test/J_EDEw/2026-03-26/20/J_EDEw-20260326211213-20260326211713.mp4

# Calibrate all three cameras in sequence
python -m bjj_pipeline.tools.calibrate_camera \
  --camera FP7oJQ J_EDEw PPDmUg \
  --video data/raw/nest/calibration_test/FP7oJQ/...mp4 \
         data/raw/nest/calibration_test/J_EDEw/...mp4 \
         data/raw/nest/calibration_test/PPDmUg/...mp4
```

## The 3-Step Workflow

### Step 1 — Initial Homography

**What it does:** You place 4+ corresponding point pairs between the camera frame
and the mat blueprint to establish a rough homography (H).

**UI:** Side-by-side view — camera frame on the left, mat blueprint on the right.
Click a point on the frame, then click the corresponding point on the blueprint.
Repeat for 4+ pairs.

**Tips for good click placement:**
- Spread points across the full visible mat — don't cluster in one area.
- Use mat panel corners and seam intersections as landmarks.
- 4 points is the minimum; 6-8 gives better accuracy via RANSAC.

**Controls:** `[u]` undo | `[c]` clear | `[s]` solve+save | `[q]` quit

**QA overlay:** After pressing `s`, a grid overlay shows the projected mat layout.
Press `[a]` to accept or `[r]` to redo.

### Step 2 — Lens Calibration

**What it does:** Estimates the camera's intrinsic parameters (focal length `f` and
radial distortion `k1`, `k2`) using a collinearity constraint. After this step, barrel
distortion from the wide-angle Nest lens is corrected.

**UI:** The tool auto-detects edge points along the 4 anchor edges using gradient
analysis. You can add or delete points manually. Press `[s]` to solve — the tool
shows the undistorted frame with fitted lines.

**What to check after solving:**
- `f` should be in the 600–1000 range for Nest cameras
- `k1` should be negative (barrel distortion), typically -0.05 to -0.4
- `k2` should be small (< 1.0)
- Per-edge RMS should be < 1px for all edges
- The undistorted frame should show straight mat edges

**Controls:** left-click = add | right-click/`[d]` = delete | `[s]` solve |
`[a]` accept | `[r]` redo | `[q]` quit

### Step 3 — Final H Refinement

**What it does:** With lens distortion corrected, you re-place point pairs on the
now-undistorted frame. The tool then runs automatic mat-line detection (Canny +
Hough) and RANSAC refinement to produce a high-precision homography.

**UI:** Same as Step 1, but the frame now shows straight mat edges (lens corrected).

**What to check in QA overlay:**
- Grid lines should align with mat edges across the full visible mat
- Reproj error < 3px mean
- 5+ matched lines from 3+ distinct edges
- `converged = True`

## Daily H-Only Recalibration

If the camera hasn't moved but you want to refresh the homography (e.g., after a
Nest firmware update shifts the digital crop slightly):

```bash
python -m bjj_pipeline.tools.calibrate_camera \
  --camera J_EDEw --video ... --skip-lens
```

This skips Step 2 (lens parameters are hardware-constant) and only runs Steps 1
and 3.

## Resuming After Interruption

The wizard checks `homography.json` state before each step:
- **After Step 1 interruption** (H exists, no K+dist): resume skips Step 1, runs 2+3
- **After Step 2 interruption** (H + K+dist exist): resume skips 1+2, runs 3 only
- **Force redo:** `--force` re-runs all steps regardless of state

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| k1/k2 hit bounds (-1 or +1) | Insufficient edge points | Add more manual points in Step 2, especially at mat periphery |
| Few matched lines in Step 3 | Poor initial H | Redo Step 1 with better-spread click points |
| Grid stretching at edges | Double-undistortion bug | Ensure you're on latest code (commit 499c6f0+) |
| Reproj error > 5px | Anchor points inaccurate | Zoom in when clicking, use sharp seam intersections |
| `converged = False` | Hough couldn't find enough lines | Try a different calibration_test video with clearer mat seams |

## Advanced: Standalone Tools

The wizard calls these tools internally. They remain available for advanced use:

```bash
# Homography only (clicks or overlay_rect mode)
python -m bjj_pipeline.tools.homography_calibrate \
  --camera J_EDEw interactive --video ... --calibration-ui clicks

# Lens calibration only
python -m calibration_pipeline.lens_calibration \
  --camera J_EDEw --video ...

# Batch H recalibration (all cameras, non-interactive)
python tools/cp19_recalibrate.py
```
