# Pipeline Validation Discovery Report

Generated: 2026-05-08 18:25

This report scans the repo for model weights, training data, training runs, and answers the in-distribution vs held-out question for each GT clip. Findings are labeled CONFIRMED, INFERRED, or OPEN.

---

## 1. Model Weights Inventory

Status: CONFIRMED (direct filesystem inspection)

| File | Size | Modified | CoreML sibling | Category |
|------|------|----------|----------------|----------|
| `bjj-detect-all-cameras.pt` | 5.1 MB | 2026-05-06 | Yes | domain-tuned |
| `bjj-pose-hybrid.pt` | 17.4 MB | 2026-05-05 | No | domain-tuned |
| `bjj-pose-r1.pt` | 17.6 MB | 2026-04-16 | Yes | domain-tuned |
| `bjj-pose-r2.pt` | 7.4 MB | 2026-05-02 | No | domain-tuned |
| `bjj-pose-r2_bbox.pt` | 7.4 MB | 2026-05-03 | No | domain-tuned |
| `bjj-pose-vicos.pt` | 7.5 MB | 2026-05-03 | No | domain-tuned |
| `yolo11n-pose.pt` | 6.0 MB | 2026-04-03 | No | stock |
| `yolo11s-pose.pt` | 19.4 MB | 2026-04-03 | No | stock |
| `yolo26n-pose.pt` | 7.5 MB | 2026-04-05 | Yes | stock |
| `yolo26n.pt` | 5.3 MB | 2026-05-06 | No | stock |
| `yolov8n-pose.pt` | 6.5 MB | 2026-04-03 | Yes | stock |
| `yolov8n.pt` | 6.2 MB | 2025-09-21 | No | stock |
| `yolov8s-obb.pt` | 22.2 MB | 2026-01-10 | No | stock |
| `yolov8s-pose.pt` | 22.4 MB | 2026-01-10 | No | stock |
| `yolov8s-seg.pt` | 22.8 MB | 2026-01-10 | No | stock |

CoreML packages: 4 (bjj-detect-all-cameras.mlpackage, bjj-pose-r1.mlpackage, yolo26n-pose.mlpackage, yolov8n-pose.mlpackage)

Sidecar metadata files: **none**. No .yaml, .json, .md, .txt, or .log files co-located with model weights.

`models/training_runs/` contents (3 files):
  - `training_runs/.DS_Store`
  - `training_runs/round2_probe_results.json`
  - `training_runs/round_1/best.pt`

## 2. Training-Data Inventory

Status: CONFIRMED (zip content inspection)

### CVAT Export Zips

| Zip | Size | Total labels | Non-empty | Empty |
|-----|------|-------------|-----------|-------|
| `training_COCO_FP7oJQ_clip1_0-300.zip` | 32.2 MB | 0 | 0 | 0 |
| `training_FP7oJQ_clip1_0-300.zip` | 126.7 MB | 0 | 0 | 0 |
| `training_YOLO_obbox_FP7oJQ_clip1_0-300.zip` | 5.2 MB | 4530 | 4530 | 0 |
| `training_YOLO_obbox_J_EDEw_clip1_0-3000.zip` | 5.2 MB | 4530 | 4530 | 0 |
| `training_YOLO_obbox_PPDmUg_clip1_0-2990.zip` | 2.1 MB | 2998 | 2998 | 0 |
| `training_YOLO_pose_FP7oJQ_clip1_0-300.zip` | 23.6 MB | 4530 | 4530 | 0 |
| `training_YOLO_pose_J_EDEw_clip1_0-3000.zip` | 23.6 MB | 4530 | 4530 | 0 |
| `training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip` | 3.2 MB | 4530 | 4530 | 0 |
| `training_YOLO_track_detections_J_EDEw_clip1_0-3000.zip` | 3.2 MB | 4530 | 4530 | 0 |
| `training_YOLO_track_detections_PPDmUg_clip1_0-2990.zip` | 1.4 MB | 2998 | 2998 | 0 |
| `training_data_detection_all_cameras.zip` | 207.7 MB | 0 | 0 | 0 |

### Dataset YAML Files

**`data/training_data/combined/dataset.yaml`**
```yaml
flip_idx:
- 0
- 2
- 1
- 4
- 3
- 6
- 5
- 8
- 7
- 10
- 9
- 12
- 11
- 14
- 13
- 16
- 15
kpt_shape:
- 17
- 3
names:
  0: person
path: /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker/data/training_data/combined
train: train.txt
val: val.txt
```

**`data/training_data/detection_all_cameras/dataset.yaml`**
```yaml
names:
  0: person
nc: 1
path: .
train: train.txt
val: val.txt
```

**`data/training_data/hybrid/dataset.yaml`**
```yaml
flip_idx:
- 0
- 2
- 1
- 4
- 3
- 6
- 5
- 8
- 7
- 10
- 9
- 12
- 11
- 14
- 13
- 16
- 15
kpt_shape:
- 17
- 3
names:
  0: person
path: /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker/data/training_data/hybrid
train: train.txt
val: val.txt
```

**`data/training_data/r2_bbox/dataset.yaml`**
```yaml
flip_idx:
- 0
- 2
- 1
- 4
- 3
- 6
- 5
- 8
- 7
- 10
- 9
- 12
- 11
- 14
- 13
- 16
- 15
kpt_shape:
- 17
- 3
names:
  0: person
path: /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker/data/training_data/r2_bbox
train: train.txt
val: val.txt
```

**`data/training_data/round1/dataset.yaml`**
```yaml
kpt_shape:
- 17
- 3
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
path: /Users/bryanthomas/Desktop/Professional/Projects/roll_tracker/data/training_data/round1
train: train.txt
val: val.txt
```

**`data/training_data/filtered/yolo_obbox_0-300/data.yaml`**
```yaml
names:
  0: Skeleton
  1: bbox
path: .
train: train.txt
```

**`data/training_data/filtered/yolo_pose_0-300/data.yaml`**
```yaml
kpt_shape:
- 17
- 3
names:
  0: Skeleton
path: .
train: train.txt
```

**`data/training_data/round2_unpacked/yolo_obbox/data.yaml`**
```yaml
names:
  0: Skeleton
  1: bbox
path: .
train: train.txt
```

**`data/training_data/round2_unpacked/yolo_pose/data.yaml`**
```yaml
kpt_shape:
- 17
- 3
names:
  0: Skeleton
path: .
train: train.txt
```

**`data/training_data/unpacked_yolo_obbox/data.yaml`**
```yaml
names:
  0: Skeleton
  1: bbox
path: .
train: train.txt
```

**`data/training_data/unpacked_yolo_pose/data.yaml`**
```yaml
kpt_shape:
- 17
- 3
names:
  0: Skeleton
path: .
train: train.txt
```

### Detection Dataset (detection_all_cameras/)

- **train.txt**: 749 entries (fp7: 250, jed: 250, ppd: 249)
- **val.txt**: 153 entries (fp7: 51, jed: 51, ppd: 51)

## 3. Training Run Records

### Local Training Runs (CONFIRMED)

| Run | Task | Data | Epochs | Freeze | Device |
|-----|------|------|--------|--------|--------|
| `pose/models/training_runs/round2_probe_freeze10/train` | pose | `dataset.yaml` | 20 | 10 | cpu |
| `pose/models/training_runs/round2_probe_freeze6/train` | pose | `dataset.yaml` | 20 | 6 | cpu |
| `pose/models/training_runs/round_1/train` | pose | `dataset.yaml` | 100 | 10 | cpu |
| `pose/models/training_runs/round_1/train_continued` | pose | `dataset.yaml` | 70 | 10 | cpu |

- `wandb/`: not found
- `mlruns/`: not found
- `tensorboard/`: not found

### Detection Model Provenance (INFERRED)

`bjj-detect-all-cameras.pt` was trained on Kaggle, not locally. Evidence:
- No detection training runs exist in `runs/` (only pose runs)
- `tools/colab_detection_training.ipynb` references:
  - Base model: `yolo26n.pt`
  - Dataset: `training_data_detection_all_cameras.zip` (uploaded to Kaggle)
  - Config: epochs=100, batch=16, freeze=10, imgsz=640
  - Output saved as `bjj-detect-all-cameras.pt`
- File modified date (2026-05-06) is consistent with CLAUDE.md CP23b timeline
- Active production model: **CONFIRMED** (referenced in `configs/default.yaml`)

### Pose Model Provenance

Pose models (`bjj-pose-r1`, `bjj-pose-r2`, `bjj-pose-r2_bbox`, `bjj-pose-vicos`, `bjj-pose-hybrid`) have partial local training records (R1, R2 probe) but the final models for R2 variants were trained on Kaggle. **No manifest yet, provenance not backfilled.** Out of scope for this brief.

## 4. In-Distribution / Held-Out Status

Status: CONFIRMED (parsed from `tools/prepare_detection_dataset.py`)

**Key rule:** Held-out evaluation frames = val partition only. Frames outside the annotated_range have no ground truth and cannot be used for recall/precision.

### Evaluation Surface

| Camera | Annotated frames | Train (in-dist) | Val (held-out) | Resolution |
|--------|-----------------|-----------------|----------------|------------|
| FP7oJQ | 301 (frames 0--300) | 250 | 51 | 1920x1080 |
| J_EDEw | 301 (frames 0--3000 stride 10) | 250 | 51 | 1280x720 |
| PPDmUg | 300 (frames 0--2990 stride 10) | 249 | 51 | 1280x720 |

**Total:** 902 annotated frames, 153 held-out (val) frames across 3 cameras.

### Zip Content Reconciliation

Cross-references annotated_range (from prep script) against actual zip contents. annotated_range is **authoritative**; zip contents are advisory.

| Camera | annotated_range_count | non_empty_in_zip | extra_non_empty_outside_range | empty_in_zip |
|--------|----------------------|------------------|-----------------------------|-------------|
| FP7oJQ | 301 | 4530 | 4229 | 0 |
| J_EDEw | 301 | 4530 | 4229 | 0 |
| PPDmUg | 300 | 2998 | 2698 | 0 |

**WARNING (FP7oJQ):** 4229 non-empty label files exist outside annotated_range (sample frame indices: [301, 302, 303, 304, 305]...). These are likely CVAT auto-interpolations and are **NOT trusted GT**. Downstream eval code must filter to annotated_range only.

**WARNING (J_EDEw):** 4229 non-empty label files exist outside annotated_range (sample frame indices: [1, 2, 3, 4, 5]...). These are likely CVAT auto-interpolations and are **NOT trusted GT**. Downstream eval code must filter to annotated_range only.

**WARNING (PPDmUg):** 2698 non-empty label files exist outside annotated_range (sample frame indices: [1, 2, 3, 4, 5]...). These are likely CVAT auto-interpolations and are **NOT trusted GT**. Downstream eval code must filter to annotated_range only.

## 5. Existing Manifest Conventions

Status: CONFIRMED (absence confirmed by repo-wide search)

**No existing manifest convention found.** No file in the repo currently tracks which CVAT exports were used to train which models. `configs/models/` does not exist. See Part C below for the proposed schema.

---

## Part C: Manifest Schema Proposal (DRAFT)

Since no existing manifest convention was found, this brief proposes per-model sidecar YAML files in `configs/models/{model_id}.yaml`, following the existing `configs/cameras/{cam_id}.yaml` pattern.

### Schema

```yaml
model_id: <string>           # matches filename stem
weights_path: <string>       # relative to repo root
base_model: <string>         # stock model trained from
trained_at: <date string>    # YYYY-MM-DD
training_config:             # hyperparameters
  epochs: <int>
  batch: <int>
  freeze: <int>
  imgsz: <int>
  lr0: <float>
  platform: <string>         # e.g. kaggle-t4, local-cpu

training_data:               # one entry per CVAT export used
  - export: <filename.zip>   # zip filename in data/training_data/
    source_video: <filename>  # video the annotations cover
    camera_id: <string>
    resolution: [<w>, <h>]   # needed to denormalize GT bboxes
    annotated_range:          # the trusted GT frame coverage
      start: <int>           # first annotated frame index
      stop: <int>            # last annotated frame index (inclusive)
      stride: <int>          # 1 = every frame, 10 = every 10th
      count: <int>           # total annotated frames (checksum)
    splits:
      train:
        start: <int>
        stop: <int>
        stride: <int>        # same as annotated_range stride
        count: <int>
      val:
        start: <int>
        stop: <int>
        stride: <int>
        count: <int>

notes: |                     # free-form provenance notes
  ...
```

### Field Justifications

- **start/stop/stride/count** over `frame_range: [a, b]`: cleanly represents both FP7oJQ's stride-1 and J_EDEw/PPDmUg's stride-10 sampling. `count` is redundant but serves as a cross-check.
- **annotated_range is authoritative**: downstream eval code uses this to determine which frames have trusted GT. Zip contents may include CVAT auto-interpolations outside this range; those must be ignored.
- **resolution per export**: needed to denormalize GT bboxes from [0,1] to pixel space for IoU computation.
- **Two splits only** (train, val): val IS the held-out eval set. No third partition.
- **Per-model sidecar** (not global file): each model's provenance is self-contained, matches `configs/cameras/` convention, and avoids merge conflicts when multiple models are developed in parallel.

### Future Manifest Workflow

**Recommended: hand-author with template emitter.**

Reasoning:
- Auto-generation from prep scripts is fragile -- each prep script has different structure, and Kaggle/Colab training produces no local artifacts to parse.
- A manifest is authored once per model and rarely changes. The cost of hand-authoring is low.
- A template emitter (`python -m pipeline_validation create-manifest --model-id X`) generates an empty YAML with all required fields and inline comments, reducing typo risk.

For this brief, `bjj-detect-all-cameras.yaml` is generated programmatically from `prepare_detection_dataset.py` constants as a one-time bootstrap. Future models should be hand-authored following the template.

The `create-manifest` subcommand is stubbed in this brief's CLI and will be implemented in a future brief.

## Part D: Open Questions (all RESOLVED as of TB-EVAL-1)

1. **PPDmUg training sample provenance** (RESOLVED)

   Path: `data/raw/nest/training_samples/training_PPDmUg_3000.mp4`. No matching
   Nest clip exists; eval runs the model directly on this video via `--run-model`.
   Manifest PPDmUg entry updated with `source_video_path` field pointing at this
   path.

2. **PPDmUg pipeline evaluation path** (RESOLVED)

   Direct inference via `--run-model` on `training_PPDmUg_3000.mp4`, val frames
   only (2490--2990 stride 10, 51 frames). No attempt to locate a matching Nest
   clip. Cross-validation against parquet on FP7oJQ certifies the direct-inference
   path before trusting PPDmUg numbers.

3. **Kaggle training logs (`results.csv`)** (RESOLVED)

   Not stored locally, not needed for the validation pipeline. Future training
   runs may optionally archive `results.csv` for debugging but it is not a
   manifest requirement and not a blocker for evaluation.

4. **Pose model manifests** (RESOLVED)

   Out of scope. Pose models predate the manifest convention and will not be
   backfilled. Going forward, every new model trained must include a manifest
   at training time.

