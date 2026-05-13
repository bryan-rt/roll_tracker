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
src/pipeline_validation/  # Evaluation framework: detection, identity, match viz
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
configs/models/           # Per-model training manifests (model_id.yaml)
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

*Last updated 2026-05-08.*

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

**CP23b (completed — training pipeline + model fine-tuning):**

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

*3-way pose comparison (completed on Kaggle):*

| Model | Dataset | Status |
|-------|---------|--------|
| bjj-pose-r2_bbox | 602 gym frames, bbox only | Trained |
| bjj-pose-vicos | 12K ViCoS BJJ frames, full keypoints | Trained |
| bjj-pose-hybrid | r2_bbox 20x upsampled + vicos_12k (24K) | Trained |

All trained from stock yolo26n-pose.pt, freeze=10, 100 epochs on T4 GPU.

*Detection-only model (active in Stage A):*

| Model | Dataset | Metrics | Status |
|-------|---------|---------|--------|
| bjj-detect-all-cameras | 902 frames, 3 cameras, bbox only | mAP@0.5=0.939, mAP@0.50-95=0.669, F1=0.89@0.537 | **Active** |

Base model: stock yolo26n.pt (detection, not pose). freeze=10, 100 epochs on T4 GPU.
Dataset: 10789 annotations across 902 frames (FP7oJQ 301 + J_EDEw 301 + PPDmUg 300).
Train/val: 749/153 (83/17%), per-camera stratified temporal split.
CoreML export: `models/bjj-detect-all-cameras.mlpackage` (active inference path).
Config: `conf: 0.45`, `require_keypoints: false`, `prefer_coreml: true`.

*Dataset v2 fix (2026-05-06):* FP7oJQ frame extraction was misaligned — used
`range(0, 3001, 10)` (every 10th frame across 3000) when annotations covered
frames 0–300 consecutively. Fixed to `range(0, 301, 1)`. Correct source videos:
FP7oJQ `data/cvat_tasks/round1_20260497_FP7oJQ/FP7oJQ-20260318-200014.mp4`,
J_EDEw `data/cvat_tasks/round1_20260497_J_EDEw/J_EDEw-20260318-200015.mp4`,
PPDmUg `data/raw/nest/training_samples/training_PPDmUg_3000.mp4`.
First trained model had FP7oJQ false positives from background memorization.

*Two-clip validation (2026-05-08, J_EDEw clips 200246 + 200517):*
- Stage A avg detections/frame: 9.1 and 10.0 (vs 11.9 at conf=0.25 in comparison video)
- Tracklet counts: 215 and 230 per clip
- Short tracklet ratio (<30 frames): 50.2% and 51.7% — significant fragmentation
- Very short tracklets (<10 frames): 32.6% and 37.8%
- AprilTag 1: 3 observations in clip 200246 (frames 1781–1782), 0 in clip 200517
- Tag 1 stitched to person p0003 (4 tracklets collapsed, 60s span, 1,101 detections)
- Tag 1 matched in 0 match sessions (fragmentation may disrupt proximity signal)
- Person IDs: 22 and 17 per clip; match sessions: 26 and 32 per clip
- Stage F exported all clips successfully (26 + 32 mp4s)
- Bug found and fixed: `prefer_coreml` field missing from DetectorConfig Pydantic model
  (`src/bjj_pipeline/config/models.py`) — latent since CP22d, surfaced by Pydantic
  `extra="forbid"` validation

*Key decisions:*
- Detection-only model preferred over pose model — pose supervision from domain data
  degrades bbox quality due to annotation noise on fisheye ceiling-mount footage
- ViCoS keypoints retained for future pose work but not active in current model
- FP7oJQ false positive root cause confirmed: frame extraction bug (resolved),
  plus zero empty frames across all cameras (pending)

*Key findings:*
- Bbox-only training preserves stock pose quality while improving detection
- Hybrid approach: gym bbox trains detection head, ViCoS trains pose head
- ViCoS dataset (120K frames, smartphone cameras) has domain gap from Nest overhead
- freeze=10 vs freeze=6 probe tied on 602 frames — stay frozen until more diverse data
- CVAT keypoint order differs from COCO — remapping required (see training-pipeline.md)
- MPS training has float64 issue — use CPU locally or GPU on Kaggle/Colab

*Open issues:*
- Tracklet deduplication: ~50% of tracklets <30 frames, ~35% <10 frames
- Empty frame injection: not yet implemented — next data quality step
- Bbox size tier filtering: `tools/visualize_bbox_tiers.py` built, thresholds not applied
- PPDmUg near-zero detections on held-out clip (may be empty mat or model weakness)

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
| `tools/colab_training.ipynb` | Jupyter notebook for Colab/Kaggle GPU pose training |
| `tools/colab_detection_training.ipynb` | Jupyter notebook for Colab/Kaggle GPU detection training |
| `tools/prepare_detection_dataset.py` | 3-camera detection dataset prep (track export → YOLO) |
| `tools/download_vicos.py` | Download ViCoS BJJ dataset (120K frames) |
| `tools/camera_geometry_analysis.py` | 4-phase camera diagnostic (ROI, detectability) |
| `tools/coreml_benchmark.py` | CoreML vs MPS speed comparison |
| `tools/investigate_fp7_annotations.py` | FP7oJQ false positive root cause analysis |
| `tools/visualize_bbox_tiers.py` | Color-coded bbox size tier overlays on training frames |
| `tools/compare_models.py` | Flexible 2x2 grid model comparison video tool |

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
| Detection all cameras | `data/training_data/detection_all_cameras/` | 902 frames, 3 cameras, detection only |
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
- Empty frame injection (~30-50 per camera) to reduce false positives, then retrain
- Bbox size tier filtering review (outputs from `tools/visualize_bbox_tiers.py`)
- Tracklet deduplication strategy (baseline: ~50% short tracklets)
- Full session run with new detection model (all 3 cameras, full clip set)

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

## Pipeline Validation Framework (TB-EVAL series, completed 2026-05-12)

**Module:** `src/pipeline_validation/` — three evaluation layers plus common utilities.
**Entry point:** `PYTHONPATH=src python -m pipeline_validation <stage-a|stage-d|stage-f|discover>`
**Manifest convention:** `configs/models/{model_id}.yaml` per model. Schema: model_id,
weights_path, pipeline_gym_id, training_data entries with annotated_range, splits, resolution.

**annotated_range is authoritative:** GT loader ONLY loads labels for frames defined by
the manifest's annotated_range x split. CVAT auto-interpolated labels outside annotated_range
are NOT trusted GT. Zip contents are advisory; annotated_range is the source of truth.

**GT evaluation surface (bjj-detect-all-cameras):**

| Camera | Annotated | Train (in-dist) | Val (held-out) |
|--------|-----------|-----------------|----------------|
| FP7oJQ | 301 (0-300 stride 1) | 250 | 51 |
| J_EDEw | 301 (0-3000 stride 10) | 250 | 51 |
| PPDmUg | 300 (0-2990 stride 10) | 249 | 51 |

Total: 902 annotated, 153 held-out. Pipeline outputs at `outputs/_eval_gt/` (gym_id `_eval_gt`,
hard links — symlinks don't work due to `Path.resolve()` following them).

### Evaluation Baselines: bjj-detect-all-cameras (val split)

**Stage A Detection (TB-EVAL-1):**

| Camera | Recall@0.5 | Precision@0.5 | Mean IoU | Recall@0.7 | Recall@0.9 |
|--------|-----------|--------------|----------|-----------|-----------|
| FP7oJQ | 0.847 | 0.989 | 0.850 | 0.756 | 0.340 |
| J_EDEw | 0.864 | 0.921 | 0.843 | 0.770 | 0.317 |
| PPDmUg | 0.750 | 0.981 | 0.834 | 0.681 | 0.208 |
| **Aggregate** | **0.832** | **0.959** | | | |

Topology (TB-EVAL-1.1): Duplicate rate 6-69%, true merge 0-28%, true split 0-16% (val).

**Stage D Identity (TB-EVAL-2):**

| Camera | ID Recall | ID Precision | Mean Coverage | Mean Purity |
|--------|-----------|-------------|--------------|-------------|
| FP7oJQ | 0.571 | 1.000 | 0.329 | 0.913 |
| J_EDEw | 0.571 | 0.833 | 0.239 | 0.832 |
| PPDmUg | 0.750 | 0.500 | 0.360 | 0.842 |

Failure mode breakdown (all cameras combined):
- detection_failure: 46% (Stage A missed person)
- tracklet_dropped: 25% (Stage D rejected tracklet)
- sloppy_box: 6% (boxes too loose)
- true_switch: 23% (Stage D mis-stitched)

**Match Preview Visualization (TB-EVAL-3):** Diagnostic mp4 per camera at
`outputs/_eval/stage_f/bjj-detect-all-cameras/{cam}/match_preview.mp4`.
Four layers: all detections (grey), person-assigned (colored), match envelopes
(orange dashed, faithful via `plan_crop_fixed_roi`), tag icons (yellow).

### Known Issues Surfaced by Framework

- **Stage D drops ~56% of detections** across all cameras (25K/44K FP7oJQ,
  33K/49K J_EDEw, 12K/19K PPDmUg), including tag-anchored tracklets.
  J_EDEw t201 (tag:1, frame 2770) was dropped — a Tier 1 identity anchor lost.
  Tracklet acceptance criteria suspected; deferred investigation.
- **Stage C is a placeholder** for everything except tag observations —
  identity_hints.jsonl is empty for FP7oJQ and PPDmUg. Documented drift
  between CLAUDE.md (describes full tag pipeline) and code (placeholder run).
- **PPDmUg training sample** (`training_PPDmUg_3000.mp4`) is not pixel-identical
  to any Nest clip. Provenance unknown. Pipeline output uses clip_id
  `PPDmUg-20260318-training` via manifest's `pipeline_output_clip_id`.
- **Pipeline ingest** uses hard links not symlinks under `data/raw/nest/_eval_gt/`
  because `Path.resolve()` follows symlinks, losing the `nest` path component.

### Open Follow-ups

- Stage D tracklet drop investigation (acceptance threshold tuning)
- Empty frame injection for training data (reduce FP rate)
- Bbox size tier filtering (thresholds not yet applied)
- Stage C full implementation (beyond tag observations)
- PPDmUg training sample provenance

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
