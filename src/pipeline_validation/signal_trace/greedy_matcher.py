"""Per-GT greedy matcher for topology census.

Many-to-one: each GT box independently claims its best-IoU detection.
Multiple GT boxes CAN match the same detection — this is the intended
behavior for detecting pair-box (under-segmentation) topology.

Standalone module. Does NOT modify common/matching.py (frozen instrument).
"""
from __future__ import annotations


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt_boxes: list[tuple[float, float, float, float]],
    det_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """Per-GT greedy matching. Many-to-one allowed.

    Args:
        gt_boxes: list of (x1, y1, x2, y2) ground truth boxes.
        det_boxes: list of (x1, y1, x2, y2) detection boxes.
        iou_threshold: minimum IoU to consider a match.

    Returns:
        List of (gt_idx, det_idx, iou) tuples. Unmatched GT omitted.
    """
    matches: list[tuple[int, int, float]] = []
    for gi, gb in enumerate(gt_boxes):
        best_di = -1
        best_iou = 0.0
        for di, db in enumerate(det_boxes):
            v = _iou(gb, db)
            if v > best_iou:
                best_iou = v
                best_di = di
        if best_di >= 0 and best_iou >= iou_threshold:
            matches.append((gi, best_di, best_iou))
    return matches
