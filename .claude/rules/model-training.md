# Model Training Conventions

Applies to: `models/**`, `tools/*train*`, `tools/*probe*`, `tools/*dataset*`

## Models directory: models/
- yolo26n-pose.pt — stock model (never modify)
- bjj-pose-r1.pt — Round 1 (301 frames FP7oJQ)
- bjj-pose-r2.pt — Round 2 (602 frames FP7oJQ + J_EDEw)
- bjj-pose-r2_bbox.pt — Round 2 bbox-only
- bjj-pose-vicos.pt — ViCoS 12K (pending)
- bjj-pose-hybrid.pt — Hybrid r2_bbox + vicos (pending)
- Training run artifacts: runs/pose/models/training_runs/

## Progressive unfreezing thresholds
- <5K frames: freeze=10 (backbone frozen)
- 5-15K frames: freeze=6 (partial backbone)
- 15-50K frames: freeze=3 (mostly unfrozen)
- 50K+ frames: freeze=0 (full fine-tune)
- Always validate with A/B probe (20 epochs each) before committing

## Dataset requirements
- kpt_shape: [17, 3]
- flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
- names: {0: person}
- Absolute paths in train.txt and val.txt
- Val split by session/camera/segment, never random within same video
- Class always 0 (person)

## Bbox-only training
Set all 51 keypoint values to "0.000000 0.000000 0" per annotation.
YOLO skips pose loss for v=0 keypoints, trains detection head only.
Pose head retains stock COCO weights.

## ViCoS dataset
- 120,279 frames, already in COCO keypoint order (no remap needed)
- Annotations at data/vicos_bjj/annotations.json (lowercase keys: image, pose1, pose2, position)
- Images flat in data/vicos_bjj/ as 7-digit filenames (0000001.jpg)
- 18 position classes in position_labels.json
- CC BY-NC-SA 4.0 license — R&D only, need commercial license for deployment
- Train/val split by image number range (>= 1400294 → val)
