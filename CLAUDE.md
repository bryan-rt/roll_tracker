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

*Last updated 2026-05-23.*

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).

**Evaluation baseline (CP-SPLIT-1 active, CP-EVAL-1 frozen instrument v1.0):**

| Camera | present | misattrib | no_det | untracked | d3_drop |
|--------|---------|-----------|--------|-----------|---------|
| FP7oJQ | 21.0% | 51.0% | 12.2% | 10.7% | 4.6% |
| J_EDEw | 12.2% | 54.0% | 11.7% | 13.5% | 7.9% |
| PPDmUg | 15.0% | 61.4% | 13.4% | 8.1% | 0.0% |

Ceiling without new models: ~35-40% present. Primary blocker: detection
under-segmentation (one box covering two grappling people). See CP7 investigation.

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
`DetectorConfig.iou: Optional[float] = None` — NMS IoU threshold (CP7-pre-6). Default
None = production CoreML path (inert, proven by artifact-diff regression). Setting iou
to any value **bypasses CoreML → .pt** and disables end2end NMS (~32fps vs ~79fps).
See `docs/decisions-archive.md` for the end2end/CoreML double-NMS finding.

*Detection dataset v2 (2026-06-02, not yet trained):*

| Model | Dataset | Metrics | Status |
|-------|---------|---------|--------|
| bjj-detect-all-cameras-v2 | 1352 frames (902 v1 + 450 J_EDEw-200246), bbox only | agg Recall@0.5=0.882 (+0.050 vs v1) | **Evaluated** |

Dataset at `data/training_data/detection_all_cameras_v2/`. 1199 train / 153 val (val
identical to v1). New 450 frames: J_EDEw-200246.mp4, frames 0–4490 stride 10, train only.
Source: `data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/J_EDEw/2026-03-18/20/J_EDEw-20260318-200246.mp4`.
Manifest: `configs/models/bjj-detect-all-cameras-v2.yaml`. Raw CVAT export (with track_id):
`data/training_data/training_J_EDEw_bbox_video2.zip` (4500 labels, stride-10 subset used).
Package: `data/training_data/training_data_detection_all_cameras_v2.zip` (292 MB).

*A/B evaluation v1 vs v2 (2026-06-02, frozen 153-frame val set):*
Original Kaggle training artifact: `bjj-detect-all-cameras_1352.pt`, renamed to
`bjj-detect-all-cameras-v2.pt`. CoreML sibling exported. Symmetric overlay routing
confirmed for both models. Baselines preserved at `outputs/_eval_*_baseline_v1/` and
`outputs/_eval_*_baseline_v2/`.

| Metric | v1 (902) | v2 (1352) | Δ | Signal? |
|--------|----------|-----------|---|---------|
| **Agg Recall@0.5** | 0.832 | **0.882** | **+0.050** | **yes** |
| **Agg Precision@0.5** | 0.959 | 0.935 | -0.024 | yes |
| FP7oJQ present | 21.0% | **26.0%** | **+5.0pp** | **yes** |
| FP7oJQ misattrib | 51.0% | 52.0% | +1.0pp | noise |
| J_EDEw present | 12.2% | 11.3% | -0.9pp | noise |
| J_EDEw misattrib | 54.0% | **59.0%** | **+5.0pp** | **yes (worse)** |
| PPDmUg present | 15.0% | 15.4% | +0.4pp | noise |
| PPDmUg misattrib | 61.4% | 62.3% | +0.9pp | noise |

**Verdict:** v2 is a substantially better detector (+5pp recall), improved FP7oJQ identity
(+5pp present), but did NOT move the misattribution blocker (52-62%, flat or worse).
Confirms CP7: the blocker is detection under-segmentation, not recall. More data recovers
missed people but cannot separate already-merged pairs.

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
| `python -m pipeline_validation evaluate` | Full model evaluation (pipeline + A/D/F eval) |
| `python -m pipeline_validation swap-diagnostic` | GT-oracle swap boundary diagnostic (CP-SWAP-1) |
| `python -m pipeline_validation swap-characterize` | Swap pattern characterization (CP-SWAP-2) |
| `python -m pipeline_validation signal-trace` | Greedy per-GT topology census (CP-TRACE-1) |

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
| Detection all cameras v2 | `data/training_data/detection_all_cameras_v2/` | 1352 frames (902 v1 + 450 J_EDEw-200246), detection only |
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
| `evaluation.md` | `src/pipeline_validation/**` |
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

### Evaluating a new detection model

1. Place weights at `models/{model_id}.pt`
2. Create `configs/models/{model_id}.yaml` (see existing manifest as template)
3. Run: `PYTHONPATH=src python -m pipeline_validation evaluate --model {model_id}`
4. Review outputs at:
   - `outputs/_eval/stage_a/{model_id}/_aggregate.md` (detection quality)
   - `outputs/_eval/stage_d/{model_id}/_aggregate.md` (identity quality)
   - `outputs/_eval/stage_f/{model_id}/*/match_preview.mp4` (visualization)

The `evaluate` command runs the full pipeline rerun + Stage A/D/F evaluation.
Uses direct inference for Stage A evaluation exclusively (not parquet path).
Flags: `--skip-pipeline`, `--skip-stage-a`, `--skip-stage-d`, `--skip-stage-f`
for partial reruns; `--force` to re-run even if outputs exist; `--dry-run` to
preview the plan without executing.

Individual subcommands remain available for debugging:
`PYTHONPATH=src python -m pipeline_validation <stage-a|stage-d|stage-f|discover>`

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

### GT Person Trace Layer (CP6, permanent)

`src/pipeline_validation/gt_person_trace.py` -- runs automatically as part of every
`evaluate` call. Joins five existing artifacts (per_frame_matches, detections,
d1_graph_nodes, d3_solution_ledger, person_tracks) into a per-frame per-GT-person trace.
Identity mapping is derived internally from per_frame_matches + person_tracks (CP-EVAL-1).

**Outputs** (per camera, under `outputs/_eval/stage_d/{model_id}/{camera}/`):
- `gt_person_trace.jsonl` -- one row per (camera, clip, frame, gt_person). Full chain:
  detection -> tracklet -> D1 node/carrier -> D3 status -> D4 person_id -> failure_mode.
- `gt_person_summary.json` -- per-GT-person failure-mode counts.
- `gt_camera_summary.json` / `gt_camera_summary_lite.json` (aggregate).

**Six failure modes** (full): present, stage_a_no_detection, stage_a_untracked,
d3_dropped, d4_unassigned, present_misattributed. Plus missing_canonical (GT track with
no canonical assignment). Lite mode (4) collapses the three Stage D modes into
stage_d_no_person -- used for historical baselines that lack pipeline artifacts.

**This is now the primary Stage D diagnostic.** The six-way breakdown replaces aggregate
coverage as the headline metric. After any intervention, read the per-camera breakdown to
see which mode shifted.

**Schema is a contract.** Adding columns is fine; renaming/removing requires a deliberate
migration since downstream tooling will depend on it.

**CP6 baseline results (current run, full mode):**

| Camera | present | a_miss | untracked | d3_drop | d4_unasgn | misattrib | miss_can |
|--------|---------|--------|-----------|---------|-----------|-----------|----------|
| FP7oJQ | 5.1% | 9.9% | 8.1% | 24.0% | 6.5% | 17.9% | 28.6% |
| J_EDEw | 4.7% | 11.7% | 13.5% | 49.7% | 0.7% | 19.7% | 0.0% |
| PPDmUg | 6.1% | 13.3% | 7.2% | 39.9% | 2.0% | 18.9% | 12.7% |

d3_dropped is the dominant failure mode on J_EDEw and PPDmUg. Parallel-carrier
displacement confirmed as root cause (see `docs/cp6_gt_trace_baseline.md` Section 4).
CP5 (parallel-carrier consolidation) verdict: **resume**.

**Baseline preservation discipline (NEW):** When preserving an eval baseline going forward,
copy BOTH `outputs/_eval/` AND the relevant `outputs/_eval_gt/{camera}/{clip}/` directories.
Pipeline artifacts are required for full-mode trace. The four historical baselines
(penalty_15 through cp4_pre) are lite-mode only because they predate this rule.

### Signal Trace (CP-TRACE-1, completed 2026-06-02)

**Module:** `src/pipeline_validation/signal_trace/` — greedy per-GT matcher + Stage A
topology census. Standalone submodule; does NOT modify the frozen instrument.

**CLI:** `PYTHONPATH=src python -m pipeline_validation signal-trace --model {model_id}`

Greedy matcher (IoU ≥ 0.3, many-to-one): each GT box independently claims its best
detection. Multiple GT people CAN match the same detection (pair-box signature).

**Topology classifications:** tight_match (1:1), pair_box (2+ GT share one detection),
split (GT matched by 2+ detections), miss (no detection at IoU ≥ 0.3).

**Baseline results (bjj-detect-all-cameras, all annotated frames):**

| Camera | tight_match | pair_box | split | miss | total |
|--------|-------------|----------|-------|------|-------|
| FP7oJQ | 2795 (66.3%) | 1010 (24.0%) | 0 (0.0%) | 409 (9.7%) | 4214 |
| J_EDEw | 2727 (64.7%) | 888 (21.1%) | 0 (0.0%) | 599 (14.2%) | 4214 |
| PPDmUg | 1658 (70.2%) | 594 (25.2%) | 0 (0.0%) | 109 (4.6%) | 2361 |
| **Aggregate** | **7180 (66.5%)** | **2492 (23.1%)** | **0** | **1117 (10.4%)** | **10789** |

Consistent with CP7-pre-3: pair_box at 21-25% of GT-person-frames is the dominant
under-segmentation signature. Split is zero (no over-segmentation at IoU ≥ 0.3).

## Stage D Identity Investigation (CP0-CP6, completed 2026-05-19)

A seven-checkpoint investigation into why Stage D coverage was 24-36% despite Stage A
recall of 75-86%. Conclusion: the dominant failure mode is parallel-carrier displacement
in D1 graph construction, not penalty tuning.

**The arc:**
- **CP0** (`docs/stage_d_audit_findings.md`): Config audit. Confirmed 7 of 8 D3 penalty
  fields are dead (never wired from config -> constraints). Only
  `unexplained_tracklet_penalty` is live (via explicit solver.py parameter, bypassing
  the broken constraints path).
- **CP1** (`docs/cp1_evidence.md`): Quantitative evidence. Cost inversion confirmed --
  interior BIRTH+DEATH (20.02) exceeded the flat drop penalty (15.0), so dropping
  interior tracklets was globally optimal.
- **CP2** (`docs/cp2_results.md`): Penalty 15->25. Partial -- helped FP7oJQ marginally,
  no effect on J_EDEw/PPDmUg. Binding constraint is not the cost floor.
- **CP2.5** (`docs/cp2.5_diagnostics.md`): Diagnosed flat penalty as length-agnostic.
  Recommended length-proportional.
- **CP3** (`docs/cp3_results.md`): Pure per-frame penalty. REGRESSION (short tracklets
  became too cheap to drop). Rolled back.
- **CP3b** (`docs/cp3b_results.md`): Floor-protected `max(base, per_frame*n_frames)`.
  No regression but no improvement on long tracklets. Penalty mechanism declared
  saturated.
- **CP4** (`docs/cp4_flow_topology.md`): Root cause found -- parallel-carrier displacement.
  When two tracklets are simultaneous carrier candidates for a merge event, D1 creates
  duplicate GROUP nodes; the solver routes one and orphans the other's entire chain.
  Penalty cannot fix this (it's structural, upstream of cost).
- **CP6** (`docs/cp6_gt_trace_baseline.md`): Built a permanent GT-anchored trace layer in
  pipeline_validation (see below). Confirmed CP4 at the row level AND found the picture is
  bigger than pairwise: J_EDEw has FOUR long carriers dropped (t1, t3, t5, t111), only two
  kept (t108, t2). 100% of d3_dropped frames across all cameras have a concurrent kept
  tracklet on a different GT person. Carrier competition reaches 12 simultaneous carriers
  per frame (J_EDEw, median 7).

**Current config state** (`configs/default.yaml` stages.stage_D.d3):
- `unexplained_tracklet_penalty_base: 25.0`
- `unexplained_tracklet_penalty_per_frame: 0.1`
- Formula: `max(base, per_frame * n_frames)` where n_frames = SINGLE_TRACKLET node frames
- The 7 dead penalty fields from CP0 remain present but unwired (documented, not fixed)

**Two reframings from CP6 that supersede earlier framing:**
1. The old "Stage D drops ~56% of detections / tracklet acceptance criteria suspected"
   framing is RETIRED. The mechanism is parallel-carrier displacement in D1, fully
   characterized.
2. `present_misattributed` (51-61% per camera, CP-SPLIT-1 baseline) is dominantly a
   DETECTION under-segmentation problem: one detection box covers two grappling people,
   so whichever person_id the tracklet receives, it is wrong for the other. On FP7oJQ
   (one 2.5-min clip): ~74% of misattribution is pair-box under-segmentation; of that,
   55.7% is confirmed unbracketed (detection-only-recoverable), the remainder
   indeterminate/partial pending wider-horizon and second-clip confirmation. Not a
   representation problem and not addressable by ReID/pose at the tracking layer. See
   CP7 investigation below.

**Recovery ceiling for CP5** (from CP6 trace analysis): CP5 (parallel-carrier consolidation)
recovers frames lost to d3_dropped. Conservative estimate: J_EDEw 4.7%->14.2% present,
PPDmUg 6.1%->15.7%. Ideal ceiling (every rescued drop attributed correctly where a canonical
slot is free): J_EDEw 37.5%, PPDmUg 42.1%, FP7oJQ 24.8%. All far below the >75% target.
CP5 is a necessary stepping stone, not the destination. Reaching usable coverage requires
detection-level pair separation (see CP7 investigation below).

**CP5 (completed 2026-05-21):** Parallel-carrier consolidation in D1 graph construction.
`_consolidate_parallel_triggers` helper in `d1_graph_build.py` — deterministic N-way
tiebreak (dist -> n_frames -> lexicographic carrier_id). Results (`docs/cp5_results.md`):
d3_dropped collapsed (J_EDEw 49.7% -> 7.9%, PPDmUg 39.9% -> 0.0%, FP7oJQ 24.0% -> 4.6%).
present rose modestly (J_EDEw 7.4%, PPDmUg 10.6%, FP7oJQ 6.4%). present_misattributed
is now the dominant failure mode (59-66% at CP5; 51-61% after CP-SPLIT-1). Solver
OPTIMAL, mergers stable. Next: see CP7 investigation.

**CP7 investigation (completed 2026-05-25, FP7oJQ only):** Eight-checkpoint read-only
investigation into the composition of `present_misattributed`. The arc inverted the
project's understanding:
- **pre-2:** 71-79% "impurity-driven" → sub-tracklet identity recommended.
- **pre-3:** Inverted: 70-78% is detection under-segmentation (one box, two people).
  Sub-tracklet identity targets 0.3-1.5%, not 71-79%.
- **pre-4/pre-6:** NMS-suppressed nested boxes investigated; NMS relaxation ruled out
  (worsened misattribution 4%→25%, fragmentation 1→4.5 tracklets/GT).
- **pre-8:** Axis-1 failure signature — apparent 84% "Branch B" (concurrent-swap node).
  SUPERSEDED by pre-9/pre-10.
- **pre-9:** The 84% was ~93% pair-box under-segmentation in disguise. True concurrent-
  swap margin: 9.9% of misattributed frames.
- **pre-10:** Pair-box spans 0% bracketed at every horizon (30f to full clip). The second
  person is never separately tracked anywhere in this clip → the lever is detection-level
  pair separation, and possibly plain recall on isolated people; the two are not yet
  separated and the separability experiment will distinguish them.

On FP7oJQ (one 2.5-min clip): ~74% of misattribution is pair-box; of that, 55.7% is
confirmed unbracketed (detection-only-recoverable), the remainder indeterminate/partial
pending wider-horizon and second-clip confirmation. 9.9% true Branch-B, 0% bracketed at
all horizons — single-clip, confirmation pending on the buzzer video. Stage D concurrent-
swap node deferred as a ~10% sidecar. Detection-level pair separation is the primary lever.

Integrity caveats:
(a) Pre-10 bracket test uses pipeline-derived GT attribution (majority-vote from
    gt_person_trace). Lean is benign (most reliable at separation points) but not
    ground-truth-verified outside 0-300.
(b) OPEN: the t10→t10_sN fragment map that moved pre-10 indeterminate 39%→13% is
    unverified — spot-check a sample of remapped carriers before treating 13% as hard.

Docs: `cp7_pre8_axis1_signature.md` (SUPERSEDED), `cp7_pre9_branchb_margin.md`,
`cp7_pre10_pairbox_bracketing.md`.

### Known Issues Surfaced by Framework

- **Stage D coverage loss is parallel-carrier displacement** (CP0-CP6, resolved diagnosis).
  See "Stage D Identity Investigation" above. The earlier "tracklet acceptance criteria"
  hypothesis was superseded -- the mechanism is in D1 graph construction. J_EDEw t201
  (tag:1) drop is partially a separate cost-bound case (mostly non-carrier fragments), not
  pure carrier displacement.
- **Stage C is a placeholder** for everything except tag observations —
  identity_hints.jsonl is empty for FP7oJQ and PPDmUg. Documented drift
  between CLAUDE.md (describes full tag pipeline) and code (placeholder run).
- **PPDmUg training sample** (`training_PPDmUg_3000.mp4`) is not pixel-identical
  to any Nest clip. Provenance unknown. Pipeline output uses clip_id
  `PPDmUg-20260318-training` via manifest's `pipeline_output_clip_id`.
- **Pipeline ingest** uses hard links not symlinks under `data/raw/nest/_eval_gt/`
  because `Path.resolve()` follows symlinks, losing the `nest` path component.

### Open Follow-ups

- **CP7 (next):** Detection-level pair separation is the primary lever. On FP7oJQ (one 2.5-min clip): ~74% of misattribution is pair-box under-segmentation (55.7% confirmed unbracketed, remainder indeterminate/partial); 9.9% true Branch-B, 0% bracketed at all horizons — single-clip, confirmation pending on the buzzer video. Stage D concurrent-swap node deferred as ~10% sidecar.
- Empty frame injection for training data (reduce FP rate)
- Bbox size tier filtering (thresholds not yet applied)
- Stage C full implementation (beyond tag observations)
- PPDmUg training sample provenance

## Active Decisions Log

| Decision | Status | Notes |
|----------|--------|-------|
| CP-EVAL-1: Eval instrument freeze — single-path Layer 1/2 | **Active** | Frozen 2026-05-22 (cdf1037). Hungarian IoU 0.5. Identity mapping derived from `per_frame_matches.parquet` + `person_tracks.parquet` inside `gt_person_trace.py`. Spec: `docs/eval_instrument_spec.md` v1.0. |
| CP-REID-1: BoT-SORT ReID experiment | **Rejected** | Generic `osnet_x0_25_msmt17` — negligible improvement, 2-3x runtime. FP7oJQ: zero delta. J_EDEw: +1.5pp present but +5.1pp misattr. PPDmUg: -3.3pp misattr but +2.2pp drops. Config remains `with_reid: false`. |
| CP-SWAP-1: Tracker-swap diagnostic | **Complete** | 167 GT-oracle swaps across 68/562 tracklets. Best single-feature AUC=0.663 (`bbox_aspect_change`). FP7oJQ world_accel 25.8x spike ratio, AUC=0.714. Histogram coverage 100% at swap boundaries. Module: `src/pipeline_validation/tracker_swap/`. |
| CP-SWAP-2: Swap pattern characterization | **Complete** | 47% hop_into_unoccupied, 28% cascade, 2% exchange. 41% transient (50% single-frame flickers). 45% no kinematic spike. 81% within 0.5m. Informed CP-SPLIT-1 design. |
| CP-SPLIT-1: Post-D0 tracklet splitter | **Active** | Tiered: speed cap 48 m/s + spike ratio 5x (min 5 m/s, isolation 3x) + Bhattacharyya 0.15 (kinematic corroboration 2x). Min dwell 5 frames. D0.5 in `d05_split.py` (fce5758). Results vs CP5: present +14.6/+4.8/+4.4pp, misattr -8.0/-7.0/-4.6pp. Config: `stage_D.d05_split`. Validator fix: af258b7. |
| Domain-specific ReID training | **Deferred** | Superseded by CP7 finding: on FP7oJQ (one 2.5-min clip) ~74% of misattribution is detection under-segmentation, not addressable by ReID. Detection pair separation is the primary lever. |
| BoT-SORT parameter tuning | **Deferred** | iou_threshold, track_buffer experiments. After detection pair separation. |
| GROUP node assignment reform | **Deferred** | D4 boundary fix using realized_group_pairings. ~3-5pp potential. Concurrent-swap node deferred as ~10% sidecar (CP7-pre-9). |
| CP7: Misattribution decomposition | **Complete** | Eight-checkpoint investigation (pre-2→pre-10). On FP7oJQ (one 2.5-min clip): ~74% pair-box (55.7% confirmed unbracketed), 9.9% true Branch-B, 0% bracketed — single-clip, confirmation pending on buzzer video. Detection pair separation is the primary lever. See `docs/cp7_pre9_branchb_margin.md`, `docs/cp7_pre10_pairbox_bracketing.md`. |

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
