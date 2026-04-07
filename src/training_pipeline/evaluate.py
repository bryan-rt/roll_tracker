"""Evaluation — diff videos and metrics comparison between model checkpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger

# COCO skeleton connectivity
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

OUTPUT_FPS = 15
KP_CONF_DRAW = 0.3
IOU_MATCH_THRESHOLD = 0.5

COLOR_BOTH = (0, 200, 0)       # green — both models detect
COLOR_NEW_ONLY = (0, 140, 255) # orange — only new model
COLOR_REGRESSION = (0, 0, 220) # red — only old model (regression)
SKELETON_COLOR = (235, 206, 135)


@dataclass
class MetricsReport:
    """Validation metrics for a single model."""

    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    model_path: str = ""


@dataclass
class ComparisonReport:
    """Metrics comparison between two training rounds."""

    round_a: int = 0
    round_b: int = 0
    metrics_a: MetricsReport = field(default_factory=MetricsReport)
    metrics_b: MetricsReport = field(default_factory=MetricsReport)
    mAP50_delta: float = 0.0
    mAP50_95_delta: float = 0.0
    improved: bool = False


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run_inference(
    model,
    frame_bgr: np.ndarray,
    conf: float = 0.25,
    device: str = "mps",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run YOLO inference. Returns (boxes_xyxy, confs, keypoints)."""
    preds = model.predict(source=frame_bgr, verbose=False, conf=conf, device=device)
    r0 = preds[0] if preds else None
    boxes_obj = getattr(r0, "boxes", None) if r0 is not None else None

    if boxes_obj is None or len(boxes_obj) == 0:
        return np.empty((0, 4)), np.empty((0,)), None

    xyxy = boxes_obj.xyxy.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    clses = boxes_obj.cls.cpu().numpy()

    keep = clses.astype(int) == 0
    xyxy = xyxy[keep]
    confs = confs[keep]

    kps = None
    kps_obj = getattr(r0, "keypoints", None)
    if kps_obj is not None and hasattr(kps_obj, "data") and kps_obj.data is not None:
        kps_data = kps_obj.data.cpu().numpy()
        kps = kps_data[keep] if kps_data.shape[0] == keep.shape[0] else kps_data

    return xyxy, confs, kps


def _draw_skeleton(out: np.ndarray, kps: np.ndarray) -> None:
    """Draw skeleton lines and keypoint dots."""
    for a, b in COCO_SKELETON:
        if a >= kps.shape[0] or b >= kps.shape[0]:
            continue
        if float(kps[a, 2]) < KP_CONF_DRAW or float(kps[b, 2]) < KP_CONF_DRAW:
            continue
        pt_a = (int(kps[a, 0]), int(kps[a, 1]))
        pt_b = (int(kps[b, 0]), int(kps[b, 1]))
        cv2.line(out, pt_a, pt_b, SKELETON_COLOR, 2, cv2.LINE_AA)

    for j in range(kps.shape[0]):
        if float(kps[j, 2]) < KP_CONF_DRAW:
            continue
        pt = (int(kps[j, 0]), int(kps[j, 1]))
        cv2.circle(out, pt, 4, (0, 220, 0), -1, cv2.LINE_AA)


def _draw_panel(
    frame: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    keypoints: Optional[np.ndarray],
    colors: Dict[int, Tuple[int, int, int]],
    label_text: str,
) -> np.ndarray:
    """Draw detections on a frame panel with per-detection colors."""
    out = frame.copy()
    n_dets = len(boxes)

    for i in range(n_dets):
        x1, y1, x2, y2 = (int(boxes[i, 0]), int(boxes[i, 1]),
                           int(boxes[i, 2]), int(boxes[i, 3]))
        color = colors.get(i, COLOR_BOTH)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        c = float(confs[i])
        label = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if keypoints is not None:
        for i in range(min(n_dets, keypoints.shape[0])):
            _draw_skeleton(out, keypoints[i])

    # Header overlay
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(out, label_text, (8, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def generate_diff_video(
    old_model_path: str | Path,
    new_model_path: str | Path,
    clip_path: str | Path,
    output_path: str | Path,
    max_frames: int = 300,
    device: str = "mps",
    conf: float = 0.25,
) -> None:
    """Generate side-by-side comparison video: old model vs new model.

    Color coding:
    - Green: both models detect (IoU > 0.5 match)
    - Orange: only new model detects (improvement)
    - Red: only old model detects (regression)
    """
    from ultralytics import YOLO

    clip_path = Path(clip_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading old model: {old_model_path}")
    model_old = YOLO(str(old_model_path))
    logger.info(f"Loading new model: {new_model_path}")
    model_new = YOLO(str(new_model_path))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open clip: {clip_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = orig_w // 2
    panel_h = orig_h // 2
    canvas_w = panel_w * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (canvas_w, panel_h))

    frame_idx = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference with both models
        boxes_old, confs_old, kps_old = _run_inference(model_old, frame, conf, device)
        boxes_new, confs_new, kps_new = _run_inference(model_new, frame, conf, device)

        # Match old→new detections
        matched_old: Set[int] = set()
        matched_new: Set[int] = set()
        for i in range(len(boxes_old)):
            for j in range(len(boxes_new)):
                if j in matched_new:
                    continue
                if _iou(boxes_old[i], boxes_new[j]) > IOU_MATCH_THRESHOLD:
                    matched_old.add(i)
                    matched_new.add(j)
                    break

        # Color maps: old panel shows green (shared) + red (regression)
        old_colors = {}
        for i in range(len(boxes_old)):
            old_colors[i] = COLOR_BOTH if i in matched_old else COLOR_REGRESSION

        # New panel shows green (shared) + orange (new only)
        new_colors = {}
        for i in range(len(boxes_new)):
            new_colors[i] = COLOR_BOTH if i in matched_new else COLOR_NEW_ONLY

        n_regression = len(boxes_old) - len(matched_old)
        n_new = len(boxes_new) - len(matched_new)

        old_label = f"OLD | {len(boxes_old)} dets"
        new_label = f"NEW | {len(boxes_new)} dets (+{n_new} new, -{n_regression} lost)"

        panel_old = _draw_panel(frame, boxes_old, confs_old, kps_old, old_colors, old_label)
        panel_new = _draw_panel(frame, boxes_new, confs_new, kps_new, new_colors, new_label)

        left = cv2.resize(panel_old, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(panel_new, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((panel_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :panel_w] = left
        canvas[:, panel_w:] = right
        cv2.line(canvas, (panel_w, 0), (panel_w, panel_h), (100, 100, 100), 1)

        writer.write(canvas)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            logger.info(f"Diff video: {frame_idx}/{max_frames} frames ({fps:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    logger.info(f"Diff video saved: {output_path} ({frame_idx} frames, {elapsed:.1f}s)")


def compute_metrics(
    model_path: str | Path,
    val_dataset_yaml: str | Path,
    device: str = "mps",
    imgsz: int = 640,
) -> MetricsReport:
    """Run YOLO validation and return metrics."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.val(
        data=str(val_dataset_yaml),
        device=device,
        imgsz=imgsz,
        verbose=False,
    )

    report = MetricsReport(model_path=str(model_path))
    if results is not None:
        rd = getattr(results, "results_dict", {})
        report.mAP50 = rd.get("metrics/mAP50(B)", 0.0)
        report.mAP50_95 = rd.get("metrics/mAP50-95(B)", 0.0)
        report.precision = rd.get("metrics/precision(B)", 0.0)
        report.recall = rd.get("metrics/recall(B)", 0.0)

    return report


def compare_rounds(
    round_a_model: str | Path,
    round_b_model: str | Path,
    val_dataset_yaml: str | Path,
    round_a_num: int = 0,
    round_b_num: int = 0,
    device: str = "mps",
) -> ComparisonReport:
    """Compare metrics between two model checkpoints."""
    metrics_a = compute_metrics(round_a_model, val_dataset_yaml, device)
    metrics_b = compute_metrics(round_b_model, val_dataset_yaml, device)

    return ComparisonReport(
        round_a=round_a_num,
        round_b=round_b_num,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        mAP50_delta=metrics_b.mAP50 - metrics_a.mAP50,
        mAP50_95_delta=metrics_b.mAP50_95 - metrics_a.mAP50_95,
        improved=metrics_b.mAP50 >= metrics_a.mAP50,
    )
