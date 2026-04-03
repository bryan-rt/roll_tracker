"""CP20 Part B: Isolation gate — per-detection, per-frame is_isolated flag.

A detection is isolated when it represents a cleanly localized individual
(not entangled in a grappling blob). Only isolated frames are trustworthy
for appearance extraction and future spatial scoring.

All four heuristics must pass for is_isolated=True.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# COCO indices for torso keypoints (L/R shoulder + L/R hip)
TORSO_KP_INDICES = [5, 6, 11, 12]


def _compute_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_isolation_flags(
    *,
    bboxes: List[Tuple[float, float, float, float]],
    keypoints_list: List[Optional[np.ndarray]],
    config: Dict[str, Any],
) -> List[bool]:
    """Compute is_isolated for each detection in a single frame.

    Args:
        bboxes: List of (x1, y1, x2, y2) for each detection.
        keypoints_list: List of (17, 3) keypoint arrays (or None) per detection.
        config: isolation_gate config dict.

    Returns:
        List of bool, one per detection.
    """
    if not bboxes:
        return []

    min_aspect = float(config.get("min_aspect_ratio", 0.8))
    max_iou = float(config.get("max_iou_overlap", 0.3))
    min_area = config.get("min_bbox_area", None)
    max_area = config.get("max_bbox_area", None)
    min_torso_kp = int(config.get("min_torso_keypoints", 4))
    min_kp_conf = float(config.get("min_keypoint_conf", 0.3))

    n = len(bboxes)
    flags = [True] * n

    # H1: Aspect ratio
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            flags[i] = False
            continue
        if h / w < min_aspect:
            flags[i] = False

    # H2: IoU overlap — if any pair exceeds threshold, both are flagged
    iou_failed = set()
    for i in range(n):
        for j in range(i + 1, n):
            if _compute_iou(bboxes[i], bboxes[j]) >= max_iou:
                iou_failed.add(i)
                iou_failed.add(j)
    for i in iou_failed:
        flags[i] = False

    # H3: Bbox area bounds
    if min_area is not None or max_area is not None:
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            area = (x2 - x1) * (y2 - y1)
            if min_area is not None and area < float(min_area):
                flags[i] = False
            if max_area is not None and area > float(max_area):
                flags[i] = False

    # H4: Torso keypoint plausibility
    for i, kps in enumerate(keypoints_list):
        if kps is None:
            flags[i] = False
            continue
        n_good = sum(
            1 for idx in TORSO_KP_INDICES
            if idx < kps.shape[0] and float(kps[idx, 2]) > min_kp_conf
        )
        if n_good < min_torso_kp:
            flags[i] = False

    return flags


def auto_derive_bbox_bounds(
    bbox_areas: List[float],
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> Tuple[float, float]:
    """Derive min/max bbox area from distribution percentiles."""
    if not bbox_areas:
        return (0.0, float("inf"))
    arr = np.array(bbox_areas)
    return (float(np.percentile(arr, lower_pct)), float(np.percentile(arr, upper_pct)))
