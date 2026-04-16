"""Export Stage A outputs + enhancements to CVAT-compatible annotation tasks.

Supports two modes:
- Image-based (export_task): extracts frames as JPEGs, COCO JSON annotations.
- Video-based (export_video_task): uploads raw video, CVAT XML track annotations
  with keyframe interpolation. Reduces manual correction work by 10-30x.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from training_pipeline.background import BBox
from training_pipeline.pseudo_labels import COCO_KEYPOINT_NAMES, PseudoLabel

# CVAT project keypoint order (matches skeleton label definition in CVAT project)
CVAT_KEYPOINT_ORDER = [
    "nose", "right_eye", "left_eye", "left_ear", "right_ear",
    "right_shoulder", "left_shoulder", "right_elbow", "right_wrist",
    "left_elbow", "left_wrist", "left_hip", "right_hip",
    "right_knee", "right_ankle", "left_ankle", "left_knee",
]

# COCO index → CVAT index (model output order → CVAT project order)
COCO_TO_CVAT = {
    0: 0,    # nose → nose
    1: 2,    # left_eye → cvat index 2
    2: 1,    # right_eye → cvat index 1
    3: 3,    # left_ear → cvat index 3
    4: 4,    # right_ear → cvat index 4
    5: 6,    # left_shoulder → cvat index 6
    6: 5,    # right_shoulder → cvat index 5
    7: 9,    # left_elbow → cvat index 9
    8: 7,    # right_elbow → cvat index 7
    9: 10,   # left_wrist → cvat index 10
    10: 8,   # right_wrist → cvat index 8
    11: 11,  # left_hip → cvat index 11
    12: 12,  # right_hip → cvat index 12
    13: 16,  # left_knee → cvat index 16
    14: 13,  # right_knee → cvat index 13
    15: 15,  # left_ankle → cvat index 15
    16: 14,  # right_ankle → cvat index 14
}

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


# ---------------------------------------------------------------------------
# Shared helpers — used by both image and video export paths
# ---------------------------------------------------------------------------

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


def _get_video_info(clip_path: Path) -> Tuple[int, int, int]:
    """Get total frame count, width, height from a video file."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open clip: {clip_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, w, h


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
        if not kps.empty and track_id is not None:
            # Handle both column names: Stage A pipeline uses "track_id",
            # training pipeline's _run_stage_a_inference also uses "track_id"
            track_col = "tracklet_id" if "tracklet_id" in kps.columns else "track_id"
            kp_match = kps[(kps["frame_index"] == fi) & (kps[track_col] == track_id)]
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
            "tracklet_id": track_id,
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
            "tracklet_id": None,
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
                "tracklet_id": None,
                "source": "background_subtraction",
            })
            ann_id += 1

    return annotations


def _merge_annotations(
    stage_a_path: Path,
    frame_set: set,
    pseudo_labels: Optional[List[PseudoLabel]] = None,
    bg_detections: Optional[Dict[int, List[BBox]]] = None,
) -> List[Dict]:
    """Shared annotation merge logic: Stage A > pseudo-labels > bg detections."""
    all_annotations = _load_stage_a_detections(stage_a_path, frame_set)

    next_id = max((a["id"] for a in all_annotations), default=0) + 1

    if pseudo_labels:
        pl_anns = _pseudo_labels_to_annotations(pseudo_labels, frame_set, next_id)
        all_annotations.extend(pl_anns)
        next_id = max((a["id"] for a in all_annotations), default=0) + 1
        logger.info(f"Added {len(pl_anns)} pseudo-label annotations")

    if bg_detections:
        bg_anns = _bg_detections_to_annotations(
            bg_detections, frame_set, all_annotations, next_id
        )
        all_annotations.extend(bg_anns)
        logger.info(f"Added {len(bg_anns)} background subtraction annotations")

    return all_annotations


# ---------------------------------------------------------------------------
# CVAT for Video XML writer
# ---------------------------------------------------------------------------

def _remap_coco_to_cvat(coco_kps: list) -> list:
    """Remap keypoints from COCO order (model output) to CVAT project order.

    Input: flat list [x0, y0, v0, x1, y1, v1, ...] in COCO order (17*3 = 51 values).
    Output: list of (name, x, y, v) tuples in CVAT project order (17 tuples).
    """
    cvat_kps = [(0.0, 0.0, 0)] * 17
    for coco_idx in range(17):
        cvat_idx = COCO_TO_CVAT[coco_idx]
        kx = coco_kps[coco_idx * 3]
        ky = coco_kps[coco_idx * 3 + 1]
        kv = coco_kps[coco_idx * 3 + 2]
        cvat_kps[cvat_idx] = (kx, ky, kv)
    return [(CVAT_KEYPOINT_ORDER[i], *cvat_kps[i]) for i in range(17)]


def _build_cvat_video_xml(
    annotations: List[Dict],
    total_frames: int,
    keyframe_frames: set,
) -> str:
    """Build CVAT for Video 1.1 XML with track-mode skeleton annotations.

    Annotations are grouped by tracklet_id into tracks. Frames in
    keyframe_frames are marked keyframe="1"; others are keyframe="0"
    (CVAT interpolates between keyframes).

    Keypoints are remapped from COCO order (model output) to CVAT project
    definition order before writing. Label is "Skeleton" matching the
    CVAT project.
    """
    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    # Group annotations by track (tracklet_id or assigned id for non-tracked)
    tracks: Dict[int, List[Dict]] = {}
    next_track_id = 0
    for ann in annotations:
        tid = ann.get("tracklet_id")
        if tid is None:
            tid = 10000 + next_track_id
            next_track_id += 1
        tracks.setdefault(int(tid), []).append(ann)

    for track_id, track_anns in sorted(tracks.items()):
        track_el = SubElement(root, "track", {
            "id": str(track_id),
            "label": "Skeleton",
            "source": "manual",
            "group_id": str(track_id + 1),
        })

        # Sort by frame
        track_anns.sort(key=lambda a: a["image_id"])

        for ann in track_anns:
            frame = ann["image_id"]
            is_kf = "1" if frame in keyframe_frames else "0"

            skel = SubElement(track_el, "skeleton", {
                "frame": str(frame),
                "keyframe": is_kf,
                "z_order": "0",
            })

            # Remap keypoints from COCO to CVAT order
            coco_kps = ann.get("keypoints", [0.0] * 51)
            cvat_kps = _remap_coco_to_cvat(coco_kps)

            for name, kx, ky, kv in cvat_kps:
                occluded = "1" if kv == 1 else "0"
                outside = "1" if kv == 0 else "0"

                pt = SubElement(skel, "points", {
                    "label": name,
                    "keyframe": is_kf,
                    "outside": outside,
                    "occluded": occluded,
                    "points": f"{kx:.2f},{ky:.2f}",
                })
                # Non-self-closing tag — add empty text to force </points>
                pt.text = "\n"

    # Pretty-print XML
    raw_xml = tostring(root, encoding="unicode")
    return parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


# ---------------------------------------------------------------------------
# Image-based export (original)
# ---------------------------------------------------------------------------

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
    """Package Stage A outputs + enhancements into a CVAT-compatible zip (image mode).

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

    # Step 2: Merge annotations
    all_annotations = _merge_annotations(
        stage_a_path, sampled_frames, pseudo_labels, bg_detections
    )

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


# ---------------------------------------------------------------------------
# Video-based export (track interpolation)
# ---------------------------------------------------------------------------

def export_video_task(
    clip_path: str | Path,
    cam_id: str,
    session_id: str,
    stage_a_path: str | Path,
    pseudo_labels: Optional[List[PseudoLabel]] = None,
    bg_detections: Optional[Dict[int, List[BBox]]] = None,
    keyframe_interval: int = 30,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Export video + CVAT XML with track-mode skeleton annotations.

    Keyframe annotations are placed every keyframe_interval frames. CVAT
    linearly interpolates skeletons between keyframes, reducing manual
    correction work by 10-30x compared to per-frame image annotation.

    Parameters
    ----------
    clip_path : Path to raw video clip.
    cam_id : Camera identifier.
    session_id : Session identifier.
    stage_a_path : Path to Stage A output directory.
    pseudo_labels : Cross-camera pseudo-labels for this camera (optional).
    bg_detections : Background subtraction detections keyed by frame_index (optional).
    keyframe_interval : Frames between keyframe annotations (default 30).
    output_dir : Output directory.

    Returns
    -------
    Tuple of (video_path, annotations_xml_path).
    """
    clip_path = Path(clip_path)
    stage_a_path = Path(stage_a_path)
    output_dir = output_dir or Path(f"data/cvat_tasks/{session_id}_{cam_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy video to task directory
    video_dst = output_dir / clip_path.name
    if not video_dst.exists() or video_dst.resolve() != clip_path.resolve():
        shutil.copy2(clip_path, video_dst)

    total_frames, width, height = _get_video_info(clip_path)

    # Step 2: Build keyframe set
    keyframe_frames = set(range(0, total_frames, keyframe_interval))

    # Load only frames that have detections in the parquet (not all video frames).
    # _load_stage_a_detections filters by this set, so passing all detection frames
    # avoids iterating over empty frames.
    det_path = stage_a_path / "detections.parquet"
    if det_path.exists():
        det_df = pd.read_parquet(det_path, columns=["frame_index"])
        all_frame_indices = set(det_df["frame_index"].unique())
    else:
        all_frame_indices = keyframe_frames

    # Step 3: Merge annotations from all sources for all frames
    all_annotations = _merge_annotations(
        stage_a_path, all_frame_indices, pseudo_labels, bg_detections
    )

    # Step 4: Build CVAT for Video XML
    xml_content = _build_cvat_video_xml(
        all_annotations, total_frames, keyframe_frames
    )

    xml_path = output_dir / "annotations.xml"
    xml_path.write_text(xml_content)

    n_keyframe_anns = sum(1 for a in all_annotations if a["image_id"] in keyframe_frames)
    n_interp_anns = len(all_annotations) - n_keyframe_anns
    n_tracks = len({a.get("tracklet_id", id(a)) for a in all_annotations})

    logger.info(
        f"CVAT video task exported: {output_dir.name} "
        f"({total_frames} frames, {n_tracks} tracks, "
        f"{n_keyframe_anns} keyframe anns, {n_interp_anns} interpolated anns)"
    )
    return video_dst, xml_path
