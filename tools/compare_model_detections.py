"""CP20: Side-by-side model detection comparison video.

Produces one MP4 per camera with a 2x2 grid showing detections from 4 models
on the exact same frames. Used to visually verify that detection count changes
(e.g. J_EDEw -51%) are false positive removal, not missed athletes.

Layout:
    yolov8n (baseline) | yolov8n-pose
    -------------------|-------------
    yolov8s-pose       | yolo11n-pose

Usage:
    python tools/compare_model_detections.py
    python tools/compare_model_detections.py --max-frames 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

MODELS = [
    "models/yolov8n.pt",
    "models/yolov8n-pose.pt",
    "models/yolov8s-pose.pt",
    "models/yolo11n-pose.pt",
]

MODEL_LABELS = [
    "yolov8n (baseline)",
    "yolov8n-pose",
    "yolov8s-pose",
    "yolo11n-pose",
]

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

CONF_THRESHOLD = 0.25
OUTPUT_FPS = 15
KP_CONF_DRAW = 0.3
KP_CONF_GREEN = 0.5
SKELETON_COLOR = (235, 206, 135)  # light blue (BGR)

DEFAULT_MAX_FRAMES = 200


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
    h, w = out.shape[:2]
    n_dets = len(boxes)

    # Draw boxes
    for i in range(n_dets):
        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        c = float(confs[i])
        color = _bbox_color(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw skeletons (pose models only)
    if is_pose and keypoints is not None:
        for i in range(min(n_dets, keypoints.shape[0])):
            kps = keypoints[i]  # (17, 3)

            # Skeleton lines
            for a, b in COCO_SKELETON:
                if a >= kps.shape[0] or b >= kps.shape[0]:
                    continue
                ca, cb = float(kps[a, 2]), float(kps[b, 2])
                if ca < KP_CONF_DRAW or cb < KP_CONF_DRAW:
                    continue
                pt_a = (int(kps[a, 0]), int(kps[a, 1]))
                pt_b = (int(kps[b, 0]), int(kps[b, 1]))
                cv2.line(out, pt_a, pt_b, SKELETON_COLOR, 2, cv2.LINE_AA)

            # Keypoint dots
            for j in range(kps.shape[0]):
                kc = float(kps[j, 2])
                if kc < KP_CONF_DRAW:
                    continue
                pt = (int(kps[j, 0]), int(kps[j, 1]))
                cv2.circle(out, pt, 4, _kp_color(kc), -1, cv2.LINE_AA)

    # Model name + count overlay
    overlay_text = f"{model_label} | {n_dets} dets"
    (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(out, overlay_text, (8, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def run_models_on_frame(
    models: list,
    frame_bgr: np.ndarray,
    device: str,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Run all models on a single frame. Returns list of (boxes, confs, keypoints)."""
    results = []
    for model in models:
        preds = model.predict(source=frame_bgr, verbose=False, conf=CONF_THRESHOLD, device=device)
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

    quad_w = orig_w // 2
    quad_h = orig_h // 2
    grid_w = quad_w * 2
    grid_h = quad_h * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (grid_w, grid_h))

    frame_idx = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Run all 4 models on this exact frame
        detections = run_models_on_frame(models, frame_bgr, device)

        # Build 2x2 grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        positions = [(0, 0), (quad_w, 0), (0, quad_h), (quad_w, quad_h)]

        for i, (boxes, confs, kps) in enumerate(detections):
            is_pose = "pose" in MODELS[i]
            annotated = _draw_detections(frame_bgr, boxes, confs, kps, MODEL_LABELS[i], is_pose)
            resized = cv2.resize(annotated, (quad_w, quad_h), interpolation=cv2.INTER_AREA)
            x_off, y_off = positions[i]
            grid[y_off:y_off + quad_h, x_off:x_off + quad_w] = resized

        # Divider lines
        cv2.line(grid, (quad_w, 0), (quad_w, grid_h), (100, 100, 100), 1)
        cv2.line(grid, (0, quad_h), (grid_w, quad_h), (100, 100, 100), 1)

        writer.write(grid)
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

    parser = argparse.ArgumentParser(description="CP20: Model detection comparison video")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--device", default="auto")
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

    # Validate inputs
    for m in MODELS:
        if not Path(m).exists():
            print(f"ERROR: Missing model {m}")
            raise SystemExit(1)
    for cam_id, clip in TEST_CLIPS.items():
        if not Path(clip).exists():
            print(f"ERROR: Missing clip for {cam_id}: {clip}")
            raise SystemExit(1)

    # Load models once
    print("Loading models...")
    loaded_models = [YOLO(m) for m in MODELS]

    out_dir = Path("outputs/_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, clip_path in TEST_CLIPS.items():
        print(f"\n{cam_id}:")
        output_path = out_dir / f"model_comparison_{cam_id}.mp4"
        process_clip(
            models=loaded_models,
            clip_path=clip_path,
            camera_id=cam_id,
            output_path=output_path,
            max_frames=args.max_frames,
            device=device,
        )

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
