"""2- or 3-way model comparison diff video.

Compares stock model vs Round 1 (and optionally Round 2) on an unseen clip.
Color coding per panel (vs stock as baseline):
  - Green:  detection present in both this model AND stock
  - Orange: detection present in this model but NOT stock (improvement)
  - Red:    detection present in stock but NOT this model (regression)

Usage:
    # 2-way (stock vs round1)
    python tools/three_way_diff.py \
      --stock models/yolo26n-pose.pt \
      --round1 models/bjj-pose-r1.pt \
      --clip data/raw/nest/training_samples/training_PPDmUg_3000.mp4

    # 3-way (stock vs round1 vs round2)
    python tools/three_way_diff.py \
      --stock models/yolo26n-pose.pt \
      --round1 models/bjj-pose-r1.pt \
      --round2 models/bjj-pose-r2.pt \
      --clip data/raw/nest/training_samples/training_PPDmUg_3000.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

COLOR_BOTH = (0, 200, 0)        # green — shared with stock
COLOR_NEW_ONLY = (0, 140, 255)  # orange — improvement over stock
COLOR_REGRESSION = (0, 0, 220)  # red — lost vs stock
SKELETON_COLOR = (235, 206, 135)

# ---------------------------------------------------------------------------
# Helpers (adapted from training_pipeline/evaluate.py)
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
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


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _match_vs_stock(
    stock_boxes: np.ndarray,
    model_boxes: np.ndarray,
) -> Tuple[Set[int], Set[int]]:
    """Match model detections against stock. Returns (matched_stock, matched_model)."""
    matched_stock: Set[int] = set()
    matched_model: Set[int] = set()
    for i in range(len(stock_boxes)):
        for j in range(len(model_boxes)):
            if j in matched_model:
                continue
            if _iou(stock_boxes[i], model_boxes[j]) > IOU_MATCH_THRESHOLD:
                matched_stock.add(i)
                matched_model.add(j)
                break
    return matched_stock, matched_model


def _color_map_vs_stock(
    model_boxes: np.ndarray,
    matched_model: Set[int],
) -> Dict[int, Tuple[int, int, int]]:
    """Green if matched with stock, orange if new."""
    return {
        i: COLOR_BOTH if i in matched_model else COLOR_NEW_ONLY
        for i in range(len(model_boxes))
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_diff_video(
    stock_path: str | Path,
    round1_path: str | Path,
    clip_path: str | Path,
    output_path: str | Path,
    round2_path: str | Path | None = None,
    max_frames: int = 300,
    device: str = "mps",
    conf: float = 0.25,
) -> None:
    from ultralytics import YOLO

    clip_path = Path(clip_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = 3 if round2_path else 2

    print(f"Loading stock model: {stock_path}")
    model_stock = YOLO(str(stock_path))
    print(f"Loading Round 1 model: {round1_path}")
    model_r1 = YOLO(str(round1_path))
    model_r2 = None
    if round2_path:
        print(f"Loading Round 2 model: {round2_path}")
        model_r2 = YOLO(str(round2_path))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open clip: {clip_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale panels so canvas isn't too wide
    panel_w = orig_w // n_panels
    panel_h = orig_h // n_panels
    canvas_w = panel_w * n_panels

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (canvas_w, panel_h))

    frame_idx = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        boxes_s, confs_s, kps_s = _run_inference(model_stock, frame, conf, device)
        boxes_r1, confs_r1, kps_r1 = _run_inference(model_r1, frame, conf, device)

        # Match R1 vs Stock
        matched_s_r1, matched_r1 = _match_vs_stock(boxes_s, boxes_r1)
        n_new_r1 = len(boxes_r1) - len(matched_r1)
        n_lost_r1 = len(boxes_s) - len(matched_s_r1)

        # Stock panel: all green
        stock_colors = {i: COLOR_BOTH for i in range(len(boxes_s))}
        stock_label = f"STOCK | {len(boxes_s)} dets"

        # R1 panel
        r1_colors = _color_map_vs_stock(boxes_r1, matched_r1)
        r1_label = f"ROUND 1 | {len(boxes_r1)} dets (+{n_new_r1} -{n_lost_r1})"

        panel_stock = _draw_panel(frame, boxes_s, confs_s, kps_s, stock_colors, stock_label)
        panel_r1 = _draw_panel(frame, boxes_r1, confs_r1, kps_r1, r1_colors, r1_label)

        panels = [
            cv2.resize(panel_stock, (panel_w, panel_h), interpolation=cv2.INTER_AREA),
            cv2.resize(panel_r1, (panel_w, panel_h), interpolation=cv2.INTER_AREA),
        ]

        if model_r2 is not None:
            boxes_r2, confs_r2, kps_r2 = _run_inference(model_r2, frame, conf, device)
            matched_s_r2, matched_r2 = _match_vs_stock(boxes_s, boxes_r2)
            n_new_r2 = len(boxes_r2) - len(matched_r2)
            n_lost_r2 = len(boxes_s) - len(matched_s_r2)

            r2_colors = _color_map_vs_stock(boxes_r2, matched_r2)
            r2_label = f"ROUND 2 | {len(boxes_r2)} dets (+{n_new_r2} -{n_lost_r2})"

            panel_r2 = _draw_panel(frame, boxes_r2, confs_r2, kps_r2, r2_colors, r2_label)
            panels.append(
                cv2.resize(panel_r2, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            )

        canvas = np.hstack(panels)
        # Draw divider lines
        for p in range(1, n_panels):
            x = panel_w * p
            cv2.line(canvas, (x, 0), (x, panel_h), (100, 100, 100), 1)

        writer.write(canvas)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            print(f"  {frame_idx}/{max_frames} frames ({fps:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    mode = "3-way" if model_r2 else "2-way"
    print(f"\n{mode} diff video saved: {output_path} ({frame_idx} frames, {elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="2- or 3-way model comparison diff video")
    parser.add_argument("--stock", required=True, help="Stock model path")
    parser.add_argument("--round1", required=True, help="Round 1 model path")
    parser.add_argument("--round2", default=None, help="Round 2 model path (optional)")
    parser.add_argument("--clip", required=True, help="Test clip path")
    parser.add_argument("--output", default="outputs/_benchmarks/three_way_diff_PPDmUg.mp4",
                        help="Output video path")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", default="mps", help="Inference device")
    args = parser.parse_args()

    if args.round2:
        print(f"Generating 3-way diff: stock vs round1 vs round2")
    else:
        print(f"Generating 2-way diff: stock vs round1")

    generate_diff_video(
        stock_path=args.stock,
        round1_path=args.round1,
        clip_path=args.clip,
        output_path=args.output,
        round2_path=args.round2,
        max_frames=args.max_frames,
        device=args.device,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
