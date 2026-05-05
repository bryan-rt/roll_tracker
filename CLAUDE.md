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
src/calibration_pipeline/ # Gym setup: lens cal, H refinement, mat line detection
src/training_pipeline/    # Active learning: CVAT integration, fine-tuning, evaluation
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
tools/                    # Training, evaluation, comparison, packaging scripts
docs/                     # Calibration guide, decisions archive, audits
.claude/rules/            # Domain-specific context (auto-loaded by path scope)
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

*Last updated 2026-05-05.*

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).

**CP20:** YOLOv8n-pose model, isolation gate, HSV color histograms, Tier 3 histogram
cross-camera evidence. Stage A outputs 3 new sidecars: keypoints.parquet,
color_histograms.parquet, tracklet_histogram_summaries.parquet.
- Camera geometry analysis tool complete (v6 pose decomposition, 4-phase)
- Lens calibration bounds fix applied (fixed-f candidate sweep)
- H coordinate space verified as undistorted pixel space
- Calibration wizard re-run for all 3 cameras with updated lens cal
- Cross-camera agreement verified (sub-cm, 9mm worst-case)
- ROI mask union fix: brief written, not yet applied (parked)

**CP22 (completed):** Default detection model updated to yolo26n-pose (STAL loss, better
small-object detection). ultralytics upgraded 8.3.252 → 8.4.33 (`--no-deps`).
CoreML is now the default inference path (`prefer_coreml: true`). Detector auto-loads
`.mlpackage` sibling when available (78.9 fps CoreML vs 32.5 fps MPS on M1 Air).
Batch predict unsupported with CoreML (ultralytics bug); `infer_batch()` falls back to
sequential. ANE saturated by single stream — threading hurts (2 workers = 0.54x).
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.

**CP23a (completed):** Confidence threshold test. `tools/compare_conf_thresholds.py`
showed model CAN detect grappling pairs at low conf (orange boxes appear) but also misses
some entirely. Conclusion: both confidence AND resolution/classification issues exist.

**CP23b (active — training pipeline + model fine-tuning):**

Training pipeline infrastructure at `src/training_pipeline/` (10 modules, ~3000 lines).
Interactive CLI: `PYTHONPATH=src python -m training_pipeline`. CVAT integration via
cvat-sdk 2.62 with API compatibility fixes. Background models built for all 3 cameras.

*Training rounds completed:*

| Round | Data | Best Box mAP50 | Best Pose mAP50 | Model |
|-------|------|----------------|-----------------|-------|
| R1 | 301 frames FP7oJQ | 0.890 | 0.467 | `models/bjj-pose-r1.pt` |
| R2 | 602 frames FP7oJQ+J_EDEw | 0.891 | 0.209 | `models/bjj-pose-r2.pt` |

R2 detects significantly more people but pose quality degraded vs stock on standing people —
too little data overwrote general COCO pose knowledge.

*Current 3-way comparison (training on Kaggle):*

| Model | Dataset | Status |
|-------|---------|--------|
| bjj-pose-r2_bbox | 602 gym frames, bbox only | Trained |
| bjj-pose-vicos | 12K ViCoS BJJ frames, full keypoints | Training (~85/100 epochs) |
| bjj-pose-hybrid | r2_bbox 20x upsampled + vicos_12k (24K) | Queued |

All train from stock yolo26n-pose.pt, freeze=10, 100 epochs on T4 GPU.

*Key findings:*
- Bbox-only training preserves stock pose quality while improving detection
- Hybrid approach: gym bbox trains detection head, ViCoS trains pose head
- ViCoS dataset (120K frames, smartphone cameras) has domain gap from Nest overhead
- freeze=10 vs freeze=6 probe tied on 602 frames — stay frozen until more diverse data
- CVAT keypoint order differs from COCO — remapping required (see training-pipeline.md)
- MPS training has float64 issue — use CPU locally or GPU on Kaggle/Colab

## Tool Inventory

| Tool | Purpose |
|---|---|
| `tools/compare_model_detections.py` | 2x2 grid comparing YOLO models visually |
| `tools/compare_conf_thresholds.py` | Side-by-side conf=0.25 vs conf=0.05 |
| `tools/merge_cvat_exports.py` | Merge OBBox bboxes + Pose keypoints with CVAT→COCO remap |
| `tools/prepare_round2_dataset.py` | Round 2 data prep (filter, merge, extract, combine) |
| `tools/prepare_vicos_dataset.py` | Convert ViCoS JSON annotations to YOLO format |
| `tools/prepare_3way_datasets.py` | Build r2_bbox, vicos_12k, hybrid datasets |
| `tools/package_vicos_for_colab.py` | Subsample and zip ViCoS data for cloud upload |
| `tools/package_for_colab.py` | Package training data + model for Colab upload |
| `tools/three_way_diff.py` | 2- or 3-panel side-by-side model comparison video |
| `tools/freeze_probe.py` | A/B freeze level comparison (20 epochs each) |
| `tools/colab_training.ipynb` | Jupyter notebook for Colab/Kaggle GPU training |
| `tools/download_vicos.py` | Download ViCoS BJJ dataset (120K frames) |
| `tools/camera_geometry_analysis.py` | 4-phase camera diagnostic (ROI, detectability) |
| `tools/coreml_benchmark.py` | CoreML vs MPS speed comparison |

## Training Data Locations

| Dataset | Location | Description |
|---|---|---|
| Round 1 | `data/training_data/round1/` | 301 frames FP7oJQ, bbox + keypoints |
| Round 2 | `data/training_data/round2/` | 301 frames J_EDEw, bbox + keypoints |
| Combined R1+R2 | `data/training_data/combined/` | 602 frames, both cameras |
| R2 bbox-only | `data/training_data/r2_bbox/` | 602 frames, keypoints zeroed |
| Hybrid | `data/training_data/hybrid/` | r2_bbox 20x upsampled + vicos_12k |
| ViCoS full | `data/vicos_bjj/` | 120K frames, YOLO labels + position labels |
| ViCoS 12K | `data/colab_package/vicos_12k.zip` | Subsampled for cloud training |
| Background models | `data/background_models/` | Per-camera .npy median frames |
| CVAT exports | `data/training_data/training_*.zip` | Raw CVAT export zips |

## Cloud Training Setup

- **Kaggle preferred:** 30 hrs/week free T4 GPU. "Save & Run All" for background execution.
  - Dataset: "roll-tracker-training" by bryanrt
  - Input: `/kaggle/input/datasets/bryanrt/roll-tracker-training/`
- **Colab:** Works but free tier limited (~4-6 hrs before timeout).
- **Notebook:** `tools/colab_training.ipynb` (works on both with path changes)
- All models: batch=16, freeze=10 default, lr0=0.001

## Domain Context (auto-loaded by path)

| Rule file | Scope |
|-----------|-------|
| `calibration.md` | `src/calibration_pipeline/**`, `configs/cameras/**` |
| `cross-camera.md` | `src/bjj_pipeline/stages/stitch/**` |
| `pipeline-stages.md` | `src/bjj_pipeline/**` |
| `training-pipeline.md` | `src/training_pipeline/**`, training tools |
| `model-training.md` | `models/**`, dataset/training tools |
| `cvat-workflow.md` | CVAT integration, annotation workflow |
| `services.md` | `services/**` |
| `commands.md` | Common dev commands |
| `apps.md` | `app_mobile/**`, `app_web/**` |
| `supabase.md` | `backend/supabase/**` |

## Planned Work

**CP23b remaining:**
- Complete 3-way model comparison (r2_bbox vs vicos vs hybrid) via diff video
- Identify winning training approach, potentially test freeze=6 variants
- Bbox-only annotation strategy for faster data collection
- Full pipeline comparison (A→F) with different models on 5-minute window
- CoreML export of winning model for production inference

**CP23c (custom data flywheel):**
- Background subtraction for missed detection discovery
- Cross-camera pseudo-labeling for training data enrichment
- Active learning loop: model pre-fills → human corrects → retrain

**CP23d (position classification):**
- ViCoS 120K position-labeled frames (18 classes) at `data/vicos_bjj/position_labels.json`
- Level 2 classifier on match crop images

**Other pending:**
- CP21: Ankle-based world coordinates (needs better keypoints first)
- CP22c: ROI mask geometry fix (parked)
- Camera temporal jitter investigation
- CVAT XML import debugging (IndexError server-side)

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
