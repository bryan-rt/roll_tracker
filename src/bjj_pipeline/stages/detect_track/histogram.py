"""CP20 Part C: Color histogram extraction from torso crops.

Extracts a coarse HSV histogram per detection for cross-camera appearance matching.
H channel (18 bins) × S channel (8 bins) = 144-element normalized vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# COCO keypoint indices for torso crop
_TORSO_SHOULDER_L = 5
_TORSO_SHOULDER_R = 6
_TORSO_HIP_L = 11
_TORSO_HIP_R = 12
_TORSO_INDICES = [_TORSO_SHOULDER_L, _TORSO_SHOULDER_R, _TORSO_HIP_L, _TORSO_HIP_R]

HIST_H_BINS = 18
HIST_S_BINS = 8
HIST_SIZE = HIST_H_BINS * HIST_S_BINS  # 144


def _torso_crop_from_keypoints(
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    min_kp_conf: float = 0.3,
    pad_frac: float = 0.15,
) -> Optional[np.ndarray]:
    """Extract torso crop using pose keypoints. Returns BGR crop or None."""
    h, w = frame_bgr.shape[:2]

    # Check all 4 torso keypoints are present with sufficient confidence
    pts = []
    for idx in _TORSO_INDICES:
        if idx >= keypoints.shape[0]:
            return None
        x, y, conf = float(keypoints[idx, 0]), float(keypoints[idx, 1]), float(keypoints[idx, 2])
        if conf < min_kp_conf:
            return None
        pts.append((x, y))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Pad by 15% on each side
    dx = (x_max - x_min) * pad_frac
    dy = (y_max - y_min) * pad_frac
    x1 = max(0, int(x_min - dx))
    y1 = max(0, int(y_min - dy))
    x2 = min(w, int(x_max + dx))
    y2 = min(h, int(y_max + dy))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_bgr[y1:y2, x1:x2]


def _center_crop_from_bbox(
    frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    crop_frac: float = 0.6,
) -> Optional[np.ndarray]:
    """Extract center 60% of bbox (crop 20% from each edge)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    margin_x = bw * (1.0 - crop_frac) / 2.0
    margin_y = bh * (1.0 - crop_frac) / 2.0

    cx1 = max(0, int(x1 + margin_x))
    cy1 = max(0, int(y1 + margin_y))
    cx2 = min(w, int(x2 - margin_x))
    cy2 = min(h, int(y2 - margin_y))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return frame_bgr[cy1:cy2, cx1:cx2]


def compute_hsv_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized 2D HSV histogram (H×S). Returns flat float32 vector of length 144."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [HIST_H_BINS, HIST_S_BINS],
        [0, 180, 0, 256],
    )
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.flatten().astype(np.float32)


def extract_histogram(
    frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[np.ndarray],
    is_isolated: bool,
    min_kp_conf: float = 0.3,
) -> Tuple[Optional[np.ndarray], str]:
    """Extract HSV histogram for a single detection.

    Returns:
        (histogram_144, crop_method) where crop_method is one of:
        "torso_pose", "center_bbox", "not_isolated"
    """
    if not is_isolated:
        return None, "not_isolated"

    # Primary: pose-guided torso crop
    if keypoints is not None:
        crop = _torso_crop_from_keypoints(frame_bgr, keypoints, min_kp_conf=min_kp_conf)
        if crop is not None and crop.size > 0:
            return compute_hsv_histogram(crop), "torso_pose"

    # Fallback: center-cropped bbox
    crop = _center_crop_from_bbox(frame_bgr, bbox)
    if crop is not None and crop.size > 0:
        return compute_hsv_histogram(crop), "center_bbox"

    return None, "crop_failed"


def compute_tracklet_histogram_summary(
    histograms: List[np.ndarray],
    crop_methods: List[str],
) -> Tuple[Optional[np.ndarray], int, Dict[str, int]]:
    """Average isolated-frame histograms for a tracklet.

    Returns:
        (avg_histogram_144, n_isolated_frames, crop_method_distribution)
    """
    if not histograms:
        return None, 0, {}

    stacked = np.stack(histograms, axis=0)
    avg = stacked.mean(axis=0).astype(np.float32)
    # Re-normalize
    total = avg.sum()
    if total > 0:
        avg /= total

    method_dist: Dict[str, int] = {}
    for m in crop_methods:
        method_dist[m] = method_dist.get(m, 0) + 1

    return avg, len(histograms), method_dist


def bhattacharyya_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Bhattacharyya distance between two normalized histograms. Range [0, 1]."""
    return float(cv2.compareHist(
        hist_a.reshape(HIST_H_BINS, HIST_S_BINS).astype(np.float32),
        hist_b.reshape(HIST_H_BINS, HIST_S_BINS).astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))
