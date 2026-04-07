"""Export Stage A outputs + enhancements to CVAT-compatible annotation tasks."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from training_pipeline.background import BBox
from training_pipeline.pseudo_labels import COCO_KEYPOINT_NAMES, PseudoLabel

# COCO skeleton connectivity (joint index pairs)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
]

COCO_PERSON_CATEGORY = {
    "id": 1,
    "name": "person",
    "supercategory": "person",
    "keypoints": COCO_KEYPOINT_NAMES,
    "skeleton": COCO_SKELETON,
}


def _extract_frames(
    clip_path: Path,
    output_dir: Path,
    sample_rate: int = 3,
) -> List[Dict]:
    """Extract frames from video at configured sample rate.

    Returns list of COCO image entries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open clip: {clip_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    images = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            fname = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / fname), frame)
            images.append({
                "id": frame_idx,
                "file_name": fname,
                "width": width,
                "height": height,
            })
            saved += 1
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {saved} frames from {clip_path.name} (every {sample_rate}th)")
    return images


def _load_stage_a_detections(stage_a_path: Path, sampled_frames: set) -> List[Dict]:
    """Load Stage A detections and keypoints, convert to COCO annotation dicts."""
    det_path = stage_a_path / "detections.parquet"
    kp_path = stage_a_path / "keypoints.parquet"

    if not det_path.exists():
        logger.warning(f"No detections.parquet at {stage_a_path}")
        return []

    dets = pd.read_parquet(det_path)
    kps = pd.read_parquet(kp_path) if kp_path.exists() else pd.DataFrame()

    annotations = []
    ann_id = 1

    for _, row in dets.iterrows():
        fi = int(row["frame_index"])
        if fi not in sampled_frames:
            continue

        x1, y1 = float(row["x1"]), float(row["y1"])
        x2, y2 = float(row["x2"]), float(row["y2"])
        w, h = x2 - x1, y2 - y1

        # Build keypoints array (x, y, v) * 17
        keypoints_flat = [0.0] * (17 * 3)
        num_keypoints = 0

        track_id = row.get("tracklet_id")
        if not kps.empty and track_id is not None and "track_id" in kps.columns:
            kp_match = kps[(kps["frame_index"] == fi) & (kps["track_id"] == track_id)]
            if not kp_match.empty:
                kp_row = kp_match.iloc[0]
                for i, name in enumerate(COCO_KEYPOINT_NAMES):
                    kx = kp_row.get(f"kp_{name}_x", 0.0)
                    ky = kp_row.get(f"kp_{name}_y", 0.0)
                    kc = kp_row.get(f"kp_{name}_conf", 0.0)
                    if kc > 0.1:
                        # v=2 means "visible and labeled" in COCO
                        keypoints_flat[i * 3] = float(kx)
                        keypoints_flat[i * 3 + 1] = float(ky)
                        keypoints_flat[i * 3 + 2] = 2
                        num_keypoints += 1

        annotations.append({
            "id": ann_id,
            "image_id": fi,
            "category_id": 1,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
            "keypoints": keypoints_flat,
            "num_keypoints": num_keypoints,
            "source": "stage_a",
        })
        ann_id += 1

    logger.info(f"Loaded {len(annotations)} Stage A annotations")
    return annotations


def _pseudo_labels_to_annotations(
    pseudo_labels: List[PseudoLabel],
    sampled_frames: set,
    start_ann_id: int,
) -> List[Dict]:
    """Convert pseudo-labels to COCO annotation dicts."""
    annotations = []
    ann_id = start_ann_id

    for pl in pseudo_labels:
        if pl.frame_index not in sampled_frames:
            continue

        w = pl.x2 - pl.x1
        h = pl.y2 - pl.y1

        keypoints_flat = [0.0] * (17 * 3)
        num_keypoints = 0

        if pl.keypoints is not None:
            for i in range(17):
                if pl.keypoints[i, 2] > 0:
                    keypoints_flat[i * 3] = float(pl.keypoints[i, 0])
                    keypoints_flat[i * 3 + 1] = float(pl.keypoints[i, 1])
                    keypoints_flat[i * 3 + 2] = 1  # v=1 = estimated
                    num_keypoints += 1

        annotations.append({
            "id": ann_id,
            "image_id": pl.frame_index,
            "category_id": 1,
            "bbox": [pl.x1, pl.y1, w, h],
            "area": w * h,
            "iscrowd": 0,
            "keypoints": keypoints_flat,
            "num_keypoints": num_keypoints,
            "source": "pseudo_label",
            "source_cam": pl.source_cam_id,
        })
        ann_id += 1

    return annotations


def _bg_detections_to_annotations(
    bg_detections: Dict[int, List[BBox]],
    sampled_frames: set,
    existing_annotations: List[Dict],
    start_ann_id: int,
    iou_threshold: float = 0.3,
) -> List[Dict]:
    """Convert background subtraction detections to COCO annotations.

    Only creates annotations for detections that don't overlap existing ones.
    No keypoints — operator places these manually in CVAT.
    """
    annotations = []
    ann_id = start_ann_id

    # Index existing annotations by frame
    existing_by_frame: Dict[int, List] = {}
    for ann in existing_annotations:
        fid = ann["image_id"]
        existing_by_frame.setdefault(fid, []).append(ann["bbox"])

    for frame_idx, bboxes in bg_detections.items():
        if frame_idx not in sampled_frames:
            continue

        existing = existing_by_frame.get(frame_idx, [])

        for bbox in bboxes:
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1
            bg_box = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)

            # Check overlap with existing annotations
            has_overlap = False
            for ex_bbox in existing:
                ex_x1, ex_y1, ex_w, ex_h = ex_bbox
                ex_box = (ex_x1, ex_y1, ex_x1 + ex_w, ex_y1 + ex_h)
                # Inline IoU
                ix1 = max(bg_box[0], ex_box[0])
                iy1 = max(bg_box[1], ex_box[1])
                ix2 = min(bg_box[2], ex_box[2])
                iy2 = min(bg_box[3], ex_box[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = w * h
                area_b = ex_w * ex_h
                union = area_a + area_b - inter
                if union > 0 and inter / union > iou_threshold:
                    has_overlap = True
                    break

            if has_overlap:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": 1,
                "bbox": [float(bbox.x1), float(bbox.y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "keypoints": [0.0] * (17 * 3),
                "num_keypoints": 0,
                "source": "background_subtraction",
            })
            ann_id += 1

    return annotations


def export_task(
    clip_path: str | Path,
    cam_id: str,
    session_id: str,
    stage_a_path: str | Path,
    pseudo_labels: Optional[List[PseudoLabel]] = None,
    bg_detections: Optional[Dict[int, List[BBox]]] = None,
    sample_rate: int = 3,
    output_dir: Optional[Path] = None,
) -> Path:
    """Package Stage A outputs + enhancements into a CVAT-compatible zip.

    Parameters
    ----------
    clip_path : Path to raw video clip.
    cam_id : Camera identifier.
    session_id : Session identifier.
    stage_a_path : Path to Stage A output directory.
    pseudo_labels : Cross-camera pseudo-labels for this camera (optional).
    bg_detections : Background subtraction detections keyed by frame_index (optional).
    sample_rate : Extract every Nth frame.
    output_dir : Output directory. Defaults to data/cvat_tasks/{session_id}_{cam_id}/.

    Returns
    -------
    Path to the created zip file.
    """
    clip_path = Path(clip_path)
    stage_a_path = Path(stage_a_path)
    output_dir = output_dir or Path(f"data/cvat_tasks/{session_id}_{cam_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"

    # Step 1: Extract frames
    images = _extract_frames(clip_path, images_dir, sample_rate)
    sampled_frames = {img["id"] for img in images}

    # Step 2: Load and merge annotations (priority: Stage A > pseudo > bg)
    all_annotations = _load_stage_a_detections(stage_a_path, sampled_frames)

    next_id = max((a["id"] for a in all_annotations), default=0) + 1

    if pseudo_labels:
        pl_anns = _pseudo_labels_to_annotations(pseudo_labels, sampled_frames, next_id)
        all_annotations.extend(pl_anns)
        next_id = max((a["id"] for a in all_annotations), default=0) + 1
        logger.info(f"Added {len(pl_anns)} pseudo-label annotations")

    if bg_detections:
        bg_anns = _bg_detections_to_annotations(
            bg_detections, sampled_frames, all_annotations, next_id
        )
        all_annotations.extend(bg_anns)
        logger.info(f"Added {len(bg_anns)} background subtraction annotations")

    # Step 3: Build COCO JSON
    coco_data = {
        "images": images,
        "annotations": all_annotations,
        "categories": [COCO_PERSON_CATEGORY],
    }

    ann_path = output_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco_data, indent=2))

    # Step 4: Package as zip
    zip_path = output_dir / f"{session_id}_{cam_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ann_path, "annotations.json")
        for img in images:
            img_file = images_dir / img["file_name"]
            zf.write(img_file, f"images/{img['file_name']}")

    logger.info(
        f"CVAT task exported: {zip_path} "
        f"({len(images)} frames, {len(all_annotations)} annotations)"
    )
    return zip_path
