"""IoU computation and Hungarian one-to-one matcher.

Vectorized IoU in pixel-space (x1y1x2y2). Hungarian matching via
scipy.optimize.linear_sum_assignment with configurable IoU threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between all pairs of GT and pred boxes.

    Args:
        gt_boxes: (N, 4) array of [x1, y1, x2, y2]
        pred_boxes: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    gt = np.asarray(gt_boxes, dtype=np.float64)
    pred = np.asarray(pred_boxes, dtype=np.float64)

    # Intersection
    x1 = np.maximum(gt[:, 0:1], pred[:, 0:1].T)  # (N, M)
    y1 = np.maximum(gt[:, 1:2], pred[:, 1:2].T)
    x2 = np.minimum(gt[:, 2:3], pred[:, 2:3].T)
    y2 = np.minimum(gt[:, 3:4], pred[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def hungarian_match(
    cost_matrix: np.ndarray, threshold: float = 0.5
) -> list[tuple[int, int]]:
    """One-to-one matching using Hungarian algorithm.

    Args:
        cost_matrix: (N_gt, N_pred) IoU matrix
        threshold: minimum IoU for a valid match

    Returns:
        List of (gt_idx, pred_idx) pairs with IoU >= threshold
    """
    if cost_matrix.size == 0:
        return []

    # linear_sum_assignment minimizes cost, so negate IoU
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            matches.append((int(r), int(c)))

    return matches
