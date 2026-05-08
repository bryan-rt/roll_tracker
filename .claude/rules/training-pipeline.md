# Training Pipeline Rules

Applies to: `src/training_pipeline/**`, `tools/*vicos*`, `tools/*cvat*`, `tools/*colab*`, `tools/*freeze*`, `tools/*merge*`, `tools/*diff*`, `tools/prepare_*`

## Location: src/training_pipeline/

10 modules (~3000 lines): config.py, state.py, background.py, pseudo_labels.py,
export_to_cvat.py, cvat_integration.py, dataset.py, train.py, evaluate.py, run.py

## Entry point
```bash
PYTHONPATH=src python -m training_pipeline
```

## Key constraints
- No modifications to src/bjj_pipeline/ from training pipeline code
- State file: data/training_data/pipeline_state.json (atomic writes)
- Config file: src/training_pipeline/pipeline_config.yaml (gitignored, has credentials)
- CVAT SDK: cvat-sdk 2.62, use PatchedLabelRequest, getattr for paginated results
- MPS training broken (float64) — use CPU locally or GPU on Kaggle/Colab
- flip_idx required in all dataset.yaml files
- CVAT keypoints are NOT in COCO order — always remap via CVAT_TO_COCO

## CVAT→COCO Keypoint Remapping
CVAT order: nose, right_eye, left_eye, left_ear, right_ear, right_shoulder, left_shoulder, right_elbow, right_wrist, left_elbow, left_wrist, left_hip, right_hip, right_knee, right_ankle, left_ankle, left_knee
COCO order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

Mapping in `tools/merge_cvat_exports.py` (CVAT_TO_COCO dict).

## Training from stock model
Always start experimental models from stock yolo26n-pose.pt for fair comparison.
Progressive fine-tuning (ViCoS → gym data) starts from stock, not from prior rounds.

## Annotation rules
- Annotate ALL people on mat per frame (avoid false negative signal)
- Mark occluded joints as v=1, visible as v=2, missing as v=0
- Export from CVAT: "Ultralytics YOLO Pose 1.0" + "Ultralytics YOLO Oriented Bounding Boxes 1.0"
- Merge with tools/merge_cvat_exports.py

## Cloud training
- Kaggle preferred (30 hrs/week free T4 GPU)
- Use "Save & Run All" for background execution
- Kaggle input: /kaggle/input/datasets/bryanrt/roll-tracker-training/
- Kaggle working: /kaggle/working/
- All models train with batch=16 on T4, freeze=10 default
- Colab also works but free tier GPU quota limited (~4-6 hrs before timeout)
- Notebook: tools/colab_training.ipynb (works on both with path changes)
