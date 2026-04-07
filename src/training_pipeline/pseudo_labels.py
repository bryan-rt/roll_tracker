"""Cross-camera pseudo-labeling — project detections between cameras."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from bjj_pipeline.contracts.f0_projection import (
    load_calibration_from_payload,
    project_to_world,
)

# COCO 17 keypoint names (must match Stage A keypoints.parquet column order)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class PseudoLabel:
    """A detection projected from one camera into another camera's frame."""

    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    keypoints: Optional[np.ndarray] = None  # (17, 3) — x, y, visibility
    source_cam_id: str = ""
    confidence: float = 0.0


def _project_to_pixel(
    world_xy: Tuple[float, float],
    H_mat_to_img: np.ndarray,
) -> Tuple[float, float]:
    """Project world/mat coordinates to pixel coordinates.

    H on disk (homography.json) is mat→img (world→pixel), so world→pixel
    is simply H @ p with no inversion needed.

    Note: Re-distortion is intentionally skipped since these annotations will be
    human-corrected in CVAT. The undistorted-to-pixel approximation is sufficient.
    """
    p = np.array([world_xy[0], world_xy[1], 1.0], dtype=np.float64)
    q = H_mat_to_img @ p
    w = float(q[2])
    if w == 0.0:
        return (float("nan"), float("nan"))
    return (float(q[0] / w), float(q[1] / w))


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_camera_config(cam_id: str) -> Dict[str, Any]:
    """Load homography.json for a camera."""
    path = Path(f"configs/cameras/{cam_id}/homography.json")
    if not path.exists():
        raise FileNotFoundError(f"No homography config for camera {cam_id}: {path}")
    return json.loads(path.read_text())


def _load_detections(stage_a_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load detections.parquet and keypoints.parquet from a Stage A output directory."""
    det_path = stage_a_path / "detections.parquet"
    kp_path = stage_a_path / "keypoints.parquet"

    dets = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()
    kps = pd.read_parquet(kp_path) if kp_path.exists() else pd.DataFrame()
    return dets, kps


def _extract_keypoints_array(kp_row: pd.Series) -> Optional[np.ndarray]:
    """Extract (17, 3) keypoint array from a keypoints.parquet row."""
    kps = np.zeros((17, 3), dtype=np.float64)
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        x_col = f"kp_{name}_x"
        y_col = f"kp_{name}_y"
        c_col = f"kp_{name}_conf"
        if x_col in kp_row.index and y_col in kp_row.index and c_col in kp_row.index:
            kps[i] = [float(kp_row[x_col]), float(kp_row[y_col]), float(kp_row[c_col])]
    return kps


def generate_pseudo_labels(
    session_detections: Dict[str, Path],
    iou_threshold: float = 0.3,
) -> Dict[str, List[PseudoLabel]]:
    """Generate cross-camera pseudo-labels for missed detections.

    Parameters
    ----------
    session_detections : Mapping of camera_id -> Stage A output directory path.
        Each directory should contain detections.parquet and keypoints.parquet.
    iou_threshold : Minimum IoU overlap to consider a detection as already present
        in the target camera.

    Returns
    -------
    Dict mapping camera_id -> list of PseudoLabel for that camera's missed detections.
    """
    # Load camera configs and detections for all cameras.
    # H on disk (homography.json) is mat→img (world→pixel).
    # project_to_world() expects img→mat, so we store the inverse too.
    cam_configs: Dict[str, Dict[str, Any]] = {}
    cam_H_raw: Dict[str, np.ndarray] = {}      # mat→img (for _project_to_pixel)
    cam_H_img2mat: Dict[str, np.ndarray] = {}   # img→mat (for project_to_world)
    cam_K: Dict[str, Optional[np.ndarray]] = {}
    cam_D: Dict[str, Optional[np.ndarray]] = {}
    cam_dets: Dict[str, pd.DataFrame] = {}
    cam_kps: Dict[str, pd.DataFrame] = {}

    for cam_id, stage_a_path in session_detections.items():
        cfg = _load_camera_config(cam_id)
        cam_configs[cam_id] = cfg
        H_raw = np.array(cfg["H"], dtype=np.float64)
        cam_H_raw[cam_id] = H_raw
        cam_H_img2mat[cam_id] = np.linalg.inv(H_raw)
        K, D = load_calibration_from_payload(cfg)
        cam_K[cam_id] = K
        cam_D[cam_id] = D

        dets, kps = _load_detections(stage_a_path)
        cam_dets[cam_id] = dets
        cam_kps[cam_id] = kps

    cam_ids = list(session_detections.keys())
    result: Dict[str, List[PseudoLabel]] = {c: [] for c in cam_ids}

    # Collect all unique frame indices across all cameras
    all_frames: set = set()
    for dets in cam_dets.values():
        if not dets.empty and "frame_index" in dets.columns:
            all_frames.update(dets["frame_index"].unique())

    logger.info(f"Processing {len(all_frames)} unique frames across {len(cam_ids)} cameras")

    for frame_idx in sorted(all_frames):
        # For each source camera, project detections into every other camera
        for src_cam in cam_ids:
            src_dets = cam_dets[src_cam]
            src_kps = cam_kps[src_cam]

            if src_dets.empty:
                continue

            frame_dets = src_dets[src_dets["frame_index"] == frame_idx]
            if frame_dets.empty:
                continue

            # Get keypoints for this frame if available
            frame_kps = pd.DataFrame()
            if not src_kps.empty and "frame_index" in src_kps.columns:
                frame_kps = src_kps[src_kps["frame_index"] == frame_idx]

            for _, det_row in frame_dets.iterrows():
                # Compute world position of detection center
                cx = (det_row["x1"] + det_row["x2"]) / 2
                cy = (det_row["y1"] + det_row["y2"]) / 2
                bbox_w = det_row["x2"] - det_row["x1"]
                bbox_h = det_row["y2"] - det_row["y1"]

                world_xy = project_to_world(
                    (cx, cy),
                    cam_H_img2mat[src_cam],
                    cam_K[src_cam],
                    cam_D[src_cam],
                )
                if np.isnan(world_xy[0]):
                    continue

                # Project keypoints to world if available
                src_kp_world: Optional[List[Tuple[float, float, float]]] = None
                track_id = det_row.get("tracklet_id")
                if not frame_kps.empty and track_id is not None:
                    # Handle both column names: Stage A pipeline uses "track_id",
                    # training pipeline's _run_stage_a_inference also uses "track_id"
                    kp_track_col = "tracklet_id" if "tracklet_id" in frame_kps.columns else "track_id"
                    kp_match = frame_kps[frame_kps[kp_track_col] == track_id]
                    if not kp_match.empty:
                        kp_arr = _extract_keypoints_array(kp_match.iloc[0])
                        if kp_arr is not None:
                            src_kp_world = []
                            for i in range(17):
                                if kp_arr[i, 2] > 0.1:  # minimum confidence
                                    wx, wy = project_to_world(
                                        (kp_arr[i, 0], kp_arr[i, 1]),
                                        cam_H_img2mat[src_cam],
                                        cam_K[src_cam],
                                        cam_D[src_cam],
                                    )
                                    src_kp_world.append((wx, wy, kp_arr[i, 2]))
                                else:
                                    src_kp_world.append((0.0, 0.0, 0.0))

                # Project into each target camera
                for tgt_cam in cam_ids:
                    if tgt_cam == src_cam:
                        continue

                    # Project world center into target pixel space
                    tgt_px = _project_to_pixel(world_xy, cam_H_raw[tgt_cam])
                    if np.isnan(tgt_px[0]):
                        continue

                    # Approximate bbox in target: use source bbox dimensions
                    # (simplified — no pixel-per-meter scaling)
                    half_w = bbox_w / 2
                    half_h = bbox_h / 2
                    tgt_bbox = (
                        tgt_px[0] - half_w,
                        tgt_px[1] - half_h,
                        tgt_px[0] + half_w,
                        tgt_px[1] + half_h,
                    )

                    # Check overlap with existing target camera detections
                    tgt_dets = cam_dets[tgt_cam]
                    if not tgt_dets.empty:
                        tgt_frame = tgt_dets[tgt_dets["frame_index"] == frame_idx]
                        has_overlap = False
                        for _, tgt_row in tgt_frame.iterrows():
                            existing = (tgt_row["x1"], tgt_row["y1"],
                                        tgt_row["x2"], tgt_row["y2"])
                            if _iou(tgt_bbox, existing) > iou_threshold:
                                has_overlap = True
                                break
                        if has_overlap:
                            continue

                    # Build pseudo-label keypoints in target frame
                    tgt_kps = None
                    if src_kp_world is not None:
                        tgt_kps = np.zeros((17, 3), dtype=np.float64)
                        for i, (wx, wy, conf) in enumerate(src_kp_world):
                            if conf > 0:
                                px, py = _project_to_pixel((wx, wy), cam_H_raw[tgt_cam])
                                if not np.isnan(px):
                                    # v=1 means "estimated" in COCO visibility
                                    tgt_kps[i] = [px, py, 1.0]

                    result[tgt_cam].append(PseudoLabel(
                        frame_index=int(frame_idx),
                        x1=tgt_bbox[0],
                        y1=tgt_bbox[1],
                        x2=tgt_bbox[2],
                        y2=tgt_bbox[3],
                        keypoints=tgt_kps,
                        source_cam_id=src_cam,
                        confidence=float(det_row.get("confidence", 0.5)),
                    ))

    for cam_id, labels in result.items():
        logger.info(f"Camera {cam_id}: {len(labels)} pseudo-labels generated")

    return result
