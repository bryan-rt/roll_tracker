"""CP23a: Side-by-side confidence threshold comparison video.

Produces one MP4 per camera with a 1x2 layout:
  Left panel:  detections at conf >= 0.25 (current pipeline threshold)
  Right panel: detections at conf >= 0.05 (low threshold)

Color coding on the right panel:
  Green  = detection present at both thresholds (IoU > 0.5 match)
  Orange = detection present ONLY at low threshold (what we're missing)

Both panels show keypoint wireframes on all detections.

Usage:
    python tools/compare_conf_thresholds.py
    python tools/compare_conf_thresholds.py --max-frames 300
    python tools/compare_conf_thresholds.py --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Set, Tuple

import cv2
import numpy as np

MODEL_PATH = "models/yolo26n-pose.pt"

CONF_HIGH = 0.25
CONF_LOW = 0.05

TEST_CLIPS = {
    "FP7oJQ": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014.mp4",
    "J_EDEw": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/J_EDEw/2026-03-18/20/J_EDEw-20260318-200015.mp4",
    "PPDmUg": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/PPDmUg/2026-03-18/20/PPDmUg-20260318-200019.mp4",
}

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
DEFAULT_MAX_FRAMES = 200

KP_CONF_DRAW = 0.3
KP_CONF_GREEN = 0.5
SKELETON_COLOR = (235, 206, 135)  # light blue (BGR)
IOU_MATCH_THRESHOLD = 0.5

COLOR_SHARED = (0, 200, 0)       # green (BGR)
COLOR_LOW_ONLY = (0, 140, 255)   # orange (BGR)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_detections(boxes_high: np.ndarray, boxes_low: np.ndarray) -> Set[int]:
    """Return indices into boxes_low that match a high-conf detection (IoU > threshold)."""
    matched: Set[int] = set()
    if len(boxes_high) == 0 or len(boxes_low) == 0:
        return matched
    for i in range(len(boxes_low)):
        for j in range(len(boxes_high)):
            if _iou(boxes_low[i], boxes_high[j]) > IOU_MATCH_THRESHOLD:
                matched.add(i)
                break
    return matched


def _kp_color(conf: float) -> Tuple[int, int, int]:
    if conf >= KP_CONF_GREEN:
        return (0, 220, 0)
    return (0, 220, 220)


def _draw_skeleton(out: np.ndarray, kps: np.ndarray) -> None:
    """Draw skeleton lines and keypoint dots for one detection."""
    for a, b in COCO_SKELETON:
        if a >= kps.shape[0] or b >= kps.shape[0]:
            continue
        ca, cb = float(kps[a, 2]), float(kps[b, 2])
        if ca < KP_CONF_DRAW or cb < KP_CONF_DRAW:
            continue
        pt_a = (int(kps[a, 0]), int(kps[a, 1]))
        pt_b = (int(kps[b, 0]), int(kps[b, 1]))
        cv2.line(out, pt_a, pt_b, SKELETON_COLOR, 2, cv2.LINE_AA)

    for j in range(kps.shape[0]):
        kc = float(kps[j, 2])
        if kc < KP_CONF_DRAW:
            continue
        pt = (int(kps[j, 0]), int(kps[j, 1]))
        cv2.circle(out, pt, 4, _kp_color(kc), -1, cv2.LINE_AA)


def _draw_panel_high(
    frame: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    keypoints: Optional[np.ndarray],
) -> np.ndarray:
    """Draw the high-threshold (left) panel. All detections in green."""
    out = frame.copy()
    n_dets = len(boxes)

    for i in range(n_dets):
        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        c = float(confs[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_SHARED, 2)
        label = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_SHARED, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if keypoints is not None:
        for i in range(min(n_dets, keypoints.shape[0])):
            _draw_skeleton(out, keypoints[i])

    overlay = f"conf >= {CONF_HIGH} | {n_dets} dets"
    (tw, th), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(out, overlay, (8, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def _draw_panel_low(
    frame: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    keypoints: Optional[np.ndarray],
    matched: Set[int],
) -> np.ndarray:
    """Draw the low-threshold (right) panel. Green=shared, orange=low-only."""
    out = frame.copy()
    n_dets = len(boxes)
    n_new = n_dets - len(matched)

    for i in range(n_dets):
        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        c = float(confs[i])
        color = COLOR_SHARED if i in matched else COLOR_LOW_ONLY
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if keypoints is not None:
        for i in range(min(n_dets, keypoints.shape[0])):
            _draw_skeleton(out, keypoints[i])

    overlay = f"conf >= {CONF_LOW} | {n_dets} dets (+{n_new} new)"
    (tw, th), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(out, overlay, (8, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def _run_inference(
    model, frame_bgr: np.ndarray, conf: float, device: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run YOLO inference at a given confidence threshold. Returns (boxes, confs, keypoints)."""
    preds = model.predict(source=frame_bgr, verbose=False, conf=conf, device=device)
    r0 = preds[0] if preds else None
    boxes_obj = getattr(r0, "boxes", None) if r0 is not None else None

    if boxes_obj is None or len(boxes_obj) == 0:
        return np.empty((0, 4)), np.empty((0,)), None

    xyxy = boxes_obj.xyxy.cpu().numpy()
    confs_all = boxes_obj.conf.cpu().numpy()
    clses = boxes_obj.cls.cpu().numpy()

    keep = clses.astype(int) == 0
    xyxy = xyxy[keep]
    confs_all = confs_all[keep]

    kps = None
    kps_obj = getattr(r0, "keypoints", None)
    if kps_obj is not None and hasattr(kps_obj, "data") and kps_obj.data is not None:
        kps_data = kps_obj.data.cpu().numpy()
        kps = kps_data[keep] if kps_data.shape[0] == keep.shape[0] else kps_data

    return xyxy, confs_all, kps


def process_clip(
    model,
    clip_path: str,
    camera_id: str,
    output_path: Path,
    max_frames: int,
    device: str,
) -> None:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {clip_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Each panel is half the original width so the stitched frame = original resolution
    panel_w = orig_w // 2
    panel_h = orig_h // 2
    canvas_w = panel_w * 2
    canvas_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (canvas_w, canvas_h))

    frame_idx = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Run inference at both thresholds
        boxes_h, confs_h, kps_h = _run_inference(model, frame_bgr, CONF_HIGH, device)
        boxes_l, confs_l, kps_l = _run_inference(model, frame_bgr, CONF_LOW, device)

        # Match low-conf detections to high-conf ones
        matched = _match_detections(boxes_h, boxes_l)

        # Draw panels
        panel_left = _draw_panel_high(frame_bgr, boxes_h, confs_h, kps_h)
        panel_right = _draw_panel_low(frame_bgr, boxes_l, confs_l, kps_l, matched)

        # Resize and stitch
        left_resized = cv2.resize(panel_left, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(panel_right, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :panel_w] = left_resized
        canvas[:, panel_w:] = right_resized

        # Divider line
        cv2.line(canvas, (panel_w, 0), (panel_w, canvas_h), (100, 100, 100), 1)

        writer.write(canvas)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            print(f"    {frame_idx}/{max_frames} frames ({fps:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  {camera_id}: {frame_idx} frames in {elapsed:.1f}s -> {output_path.name}")


def main():
    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="CP23a: Confidence threshold comparison video")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--device", default="auto",
                        help="Device for inference (mps/cpu/cuda). CoreML NOT used.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Device: {device}")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Missing model {MODEL_PATH}")
        raise SystemExit(1)
    for cam_id, clip in TEST_CLIPS.items():
        if not Path(clip).exists():
            print(f"ERROR: Missing clip for {cam_id}: {clip}")
            raise SystemExit(1)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    out_dir = Path("outputs/_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, clip_path in TEST_CLIPS.items():
        print(f"\n{cam_id}:")
        output_path = out_dir / f"conf_threshold_{cam_id}.mp4"
        process_clip(
            model=model,
            clip_path=clip_path,
            camera_id=cam_id,
            output_path=output_path,
            max_frames=args.max_frames,
            device=device,
        )

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
