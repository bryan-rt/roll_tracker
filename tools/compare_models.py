"""2×2 grid model comparison video.

Runs 1-4 YOLO models on the same video and tiles outputs into a grid with
bbox + skeleton overlays and model name labels.

# NOTE: tools/compare_model_detections.py is preserved intentionally — it is
# the CP20/CP22 historical comparison tool with hardcoded models and clips.
# This script is the flexible CLI replacement for ad-hoc comparisons.

Usage:
    python tools/compare_models.py \
      --video data/raw/nest/training_samples/training_PPDmUg_3000.mp4 \
      --models models/yolo26n-pose.pt models/bjj-pose-r2_bbox.pt \
              models/bjj-pose-vicos.pt models/bjj-pose-hybrid.pt \
      --labels "Base" "R2-BBox" "ViCoS" "Hybrid" \
      --output outputs/_benchmarks/model_comparison.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

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
KP_CONF_GREEN = 0.5
SKELETON_COLOR = (235, 206, 135)


def _bbox_color(conf: float) -> Tuple[int, int, int]:
    if conf > 0.7:
        return (0, 200, 0)
    if conf > 0.4:
        return (0, 220, 220)
    return (0, 0, 220)


def _kp_color(conf: float) -> Tuple[int, int, int]:
    if conf >= KP_CONF_GREEN:
        return (0, 220, 0)
    return (0, 220, 220)


def _draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    keypoints: Optional[np.ndarray],
    model_label: str,
    is_pose: bool,
) -> np.ndarray:
    out = frame.copy()
    n_dets = len(boxes)

    for i in range(n_dets):
        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        c = float(confs[i])
        color = _bbox_color(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if is_pose and keypoints is not None:
        for i in range(min(n_dets, keypoints.shape[0])):
            kps = keypoints[i]

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

    overlay_text = f"{model_label} | {n_dets} dets"
    (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(out, overlay_text, (8, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def run_models_on_frame(
    models: list,
    frame_bgr: np.ndarray,
    device: str,
    conf: float = 0.25,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    results = []
    for model in models:
        preds = model.predict(source=frame_bgr, verbose=False, conf=conf, device=device)
        r0 = preds[0] if preds else None
        boxes_obj = getattr(r0, "boxes", None) if r0 is not None else None

        if boxes_obj is None or len(boxes_obj) == 0:
            results.append((np.empty((0, 4)), np.empty((0,)), None))
            continue

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

        results.append((xyxy, confs_all, kps))
    return results


def process_clip(
    models: list,
    model_labels: list[str],
    clip_path: str,
    output_path: Path,
    max_frames: int,
    device: str,
    conf: float,
) -> None:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {clip_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = len(models)
    if n <= 2:
        cols, rows = n, 1
    else:
        cols, rows = 2, 2

    panel_w = orig_w // cols
    panel_h = orig_h // rows
    grid_w = panel_w * cols
    grid_h = panel_h * rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (grid_w, grid_h))

    frame_idx = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        detections = run_models_on_frame(models, frame_bgr, device, conf)

        # Initialize grid to black (handles 3-model case cleanly)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, (boxes, confs_arr, kps) in enumerate(detections):
            col = i % cols
            row = i // cols
            annotated = _draw_detections(frame_bgr, boxes, confs_arr, kps, model_labels[i], True)
            resized = cv2.resize(annotated, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            x_off = col * panel_w
            y_off = row * panel_h
            grid[y_off:y_off + panel_h, x_off:x_off + panel_w] = resized

        # Divider lines
        if cols == 2:
            cv2.line(grid, (panel_w, 0), (panel_w, grid_h), (100, 100, 100), 1)
        if rows == 2:
            cv2.line(grid, (0, panel_h), (grid_w, panel_h), (100, 100, 100), 1)

        writer.write(grid)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            print(f"  {frame_idx}/{max_frames} frames ({fps:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\n{n}-model comparison saved: {output_path} ({frame_idx} frames, {elapsed:.1f}s)")


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="2×2 grid model comparison video")
    parser.add_argument("--video", required=True, help="Source video clip path")
    parser.add_argument("--models", nargs="+", required=True, help="1-4 model .pt paths")
    parser.add_argument("--labels", nargs="+", default=None, help="Display names (defaults to filenames)")
    parser.add_argument("--output", default="outputs/_benchmarks/model_comparison.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if len(args.models) > 4:
        print("ERROR: Maximum 4 models supported for 2×2 grid")
        raise SystemExit(1)

    if args.labels is None:
        args.labels = [Path(m).stem for m in args.models]

    if len(args.labels) != len(args.models):
        print(f"ERROR: {len(args.labels)} labels provided for {len(args.models)} models")
        raise SystemExit(1)

    for m in args.models:
        if not Path(m).exists():
            print(f"ERROR: Missing model {m}")
            raise SystemExit(1)

    if not Path(args.video).exists():
        print(f"ERROR: Missing video {args.video}")
        raise SystemExit(1)

    print(f"Loading {len(args.models)} models...")
    loaded_models = [YOLO(m) for m in args.models]

    print(f"Processing {args.video}...")
    process_clip(
        models=loaded_models,
        model_labels=args.labels,
        clip_path=args.video,
        output_path=Path(args.output),
        max_frames=args.max_frames,
        device=args.device,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
