"""Merge CVAT YOLO OBBox (manual bboxes) + YOLO Pose (keypoints) exports.

Remaps keypoints from CVAT definition order to COCO standard order.
Extracts frames from source video and produces a training-ready dataset.

Usage:
    python tools/merge_cvat_exports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Keypoint remapping: CVAT definition order → COCO standard order
# ---------------------------------------------------------------------------

CVAT_KEYPOINT_NAMES = [
    "nose", "right_eye", "left_eye", "left_ear", "right_ear",
    "right_shoulder", "left_shoulder", "right_elbow", "right_wrist",
    "left_elbow", "left_wrist", "left_hip", "right_hip",
    "right_knee", "right_ankle", "left_ankle", "left_knee",
]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# CVAT index → COCO index
CVAT_TO_COCO = {
    0: 0,    # nose → nose
    1: 2,    # right_eye → index 2
    2: 1,    # left_eye → index 1
    3: 3,    # left_ear → index 3
    4: 4,    # right_ear → index 4
    5: 6,    # right_shoulder → index 6
    6: 5,    # left_shoulder → index 5
    7: 8,    # right_elbow → index 8
    8: 10,   # right_wrist → index 10
    9: 7,    # left_elbow → index 7
    10: 9,   # left_wrist → index 9
    11: 11,  # left_hip → index 11
    12: 12,  # right_hip → index 12
    13: 14,  # right_knee → index 14
    14: 16,  # right_ankle → index 16
    15: 15,  # left_ankle → index 15
    16: 13,  # left_knee → index 13
}

# COCO skeleton for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def parse_obbox_line(line: str):
    """Parse OBBox line: class x1 y1 x2 y2 x3 y3 x4 y4 → (x_c, y_c, w, h)."""
    parts = line.strip().split()
    # Skip class (index 0), take 8 corner coords
    coords = [float(p) for p in parts[1:9]]
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    x_c = (min(xs) + max(xs)) / 2
    y_c = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return x_c, y_c, w, h


def parse_pose_line(line: str):
    """Parse YOLO Pose line → (class, bbox, keypoints_17x3)."""
    parts = line.strip().split()
    cls = int(parts[0])
    x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    kps = []
    for i in range(17):
        base = 5 + i * 3
        kx = float(parts[base])
        ky = float(parts[base + 1])
        kv = int(float(parts[base + 2]))
        kps.append((kx, ky, kv))
    return cls, (x_c, y_c, w, h), kps


def remap_keypoints(cvat_kps):
    """Remap 17 keypoints from CVAT order to COCO standard order."""
    coco_kps = [(0.0, 0.0, 0)] * 17
    for cvat_idx, (kx, ky, kv) in enumerate(cvat_kps):
        coco_idx = CVAT_TO_COCO[cvat_idx]
        coco_kps[coco_idx] = (kx, ky, kv)
    return coco_kps


def format_merged_line(x_c, y_c, w, h, coco_kps):
    """Format a merged YOLO Pose label line."""
    parts = [f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"]
    for kx, ky, kv in coco_kps:
        parts.append(f"{kx:.6f} {ky:.6f} {kv}")
    return " ".join(parts)


def merge_frame(pose_lines, obbox_lines):
    """Merge one frame: OBBox bboxes + YOLO Pose keypoints with COCO remapping."""
    merged = []
    for pose_line, obbox_line in zip(pose_lines, obbox_lines):
        # Get bbox from OBBox (manual, full person coverage)
        x_c, y_c, w, h = parse_obbox_line(obbox_line)
        # Get keypoints from YOLO Pose
        _, _, cvat_kps = parse_pose_line(pose_line)
        # Remap to COCO order
        coco_kps = remap_keypoints(cvat_kps)
        merged.append(format_merged_line(x_c, y_c, w, h, coco_kps))
    return merged


def extract_frames(video_path, max_frame, output_dir):
    """Extract frames 0..max_frame from video as JPEGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    count = 0
    for i in range(max_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"frame_{i:06d}.jpg"), frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to {output_dir}")
    return count


def verify_and_visualize(
    merged_label_path, obbox_label_path, pose_label_path, image_path, output_path
):
    """Verify merge and visualize one frame with bbox + remapped keypoints."""
    img = cv2.imread(str(image_path))
    h_img, w_img = img.shape[:2]

    merged_lines = merged_label_path.read_text().strip().split("\n")
    obbox_lines = obbox_label_path.read_text().strip().split("\n")
    pose_lines = pose_label_path.read_text().strip().split("\n")

    print(f"\nVerification on {image_path.name}:")
    print(f"  Annotations: merged={len(merged_lines)}, obbox={len(obbox_lines)}, pose={len(pose_lines)}")

    # Verify first annotation
    m_parts = merged_lines[0].strip().split()
    mx_c, my_c, mw, mh = float(m_parts[1]), float(m_parts[2]), float(m_parts[3]), float(m_parts[4])
    ox_c, oy_c, ow, oh = parse_obbox_line(obbox_lines[0])
    print(f"  Merged bbox:  ({mx_c:.4f}, {my_c:.4f}, {mw:.4f}, {mh:.4f})")
    print(f"  OBBox bbox:   ({ox_c:.4f}, {oy_c:.4f}, {ow:.4f}, {oh:.4f})")
    bbox_match = abs(mx_c - ox_c) < 1e-4 and abs(my_c - oy_c) < 1e-4
    print(f"  Bbox match: {bbox_match}")

    # Print keypoint names in output order
    print(f"\n  Output keypoint order (COCO standard):")
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        kx = float(m_parts[5 + i * 3])
        ky = float(m_parts[5 + i * 3 + 1])
        kv = int(float(m_parts[5 + i * 3 + 2]))
        vis = {0: "unlabeled", 1: "occluded", 2: "visible"}[kv]
        print(f"    [{i:2d}] {name:<20s} ({kx:.3f}, {ky:.3f}) {vis}")

    # Visualize all annotations
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
        (200, 200, 0), (200, 0, 200),
    ]

    for ann_idx, line in enumerate(merged_lines):
        parts = line.strip().split()
        x_c = float(parts[1]) * w_img
        y_c = float(parts[2]) * h_img
        w = float(parts[3]) * w_img
        h = float(parts[4]) * h_img

        # Draw bbox
        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)
        color = colors[ann_idx % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw keypoints
        kps_px = []
        for i in range(17):
            kx = float(parts[5 + i * 3]) * w_img
            ky = float(parts[5 + i * 3 + 1]) * h_img
            kv = int(float(parts[5 + i * 3 + 2]))
            kps_px.append((int(kx), int(ky), kv))
            if kv > 0:
                pt_color = (0, 220, 0) if kv == 2 else (0, 220, 220)
                cv2.circle(img, (int(kx), int(ky)), 3, pt_color, -1)

        # Draw skeleton
        for a, b in COCO_SKELETON:
            if kps_px[a][2] > 0 and kps_px[b][2] > 0:
                cv2.line(img, (kps_px[a][0], kps_px[a][1]),
                         (kps_px[b][0], kps_px[b][1]), (235, 206, 135), 1)

        # Label left/right shoulders for visual confirmation
        for idx, name in [(5, "L_sh"), (6, "R_sh")]:
            if kps_px[idx][2] > 0:
                cv2.putText(img, name, (kps_px[idx][0] + 5, kps_px[idx][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"\n  Verification image saved: {output_path}")


def main():
    pose_dir = Path("data/training_data/filtered/yolo_pose_0-300/labels/train")
    obbox_dir = Path("data/training_data/filtered/yolo_obbox_0-300/labels/train")
    video_path = Path("data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014.mp4")
    out_dir = Path("data/training_data/round1")
    max_frame = 300

    labels_out = out_dir / "labels"
    images_out = out_dir / "images"
    labels_out.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge labels
    print("=== Step 1: Merge OBBox bboxes + Pose keypoints (COCO remapped) ===")
    total_anns = 0
    for i in range(max_frame + 1):
        fname = f"frame_{i:06d}.txt"
        pose_file = pose_dir / fname
        obbox_file = obbox_dir / fname

        pose_lines = pose_file.read_text().strip().split("\n") if pose_file.exists() else []
        obbox_lines = obbox_file.read_text().strip().split("\n") if obbox_file.exists() else []

        if not pose_lines or not obbox_lines:
            (labels_out / fname).write_text("")
            continue

        merged = merge_frame(pose_lines, obbox_lines)
        (labels_out / fname).write_text("\n".join(merged) + "\n")
        total_anns += len(merged)

    print(f"  Merged {total_anns} annotations across {max_frame + 1} frames")

    # Step 2: Extract frames
    print("\n=== Step 2: Extract frames from video ===")
    extract_frames(video_path, max_frame, images_out)

    # Step 3: Train/val split
    print("\n=== Step 3: Train/val split ===")
    train_frames = list(range(0, 250))
    val_frames = list(range(250, max_frame + 1))

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    train_txt.write_text(
        "\n".join(f"images/frame_{i:06d}.jpg" for i in train_frames) + "\n"
    )
    val_txt.write_text(
        "\n".join(f"images/frame_{i:06d}.jpg" for i in val_frames) + "\n"
    )
    print(f"  Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")

    # Step 4: dataset.yaml
    import yaml
    dataset_config = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "person"},
        "kpt_shape": [17, 3],
    }
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_config, default_flow_style=False))
    print(f"  dataset.yaml written: {yaml_path}")

    # Step 5: Verification
    print("\n=== Step 4: Verification ===")
    verify_and_visualize(
        labels_out / "frame_000000.txt",
        obbox_dir / "frame_000000.txt",
        pose_dir / "frame_000000.txt",
        images_out / "frame_000000.jpg",
        Path("outputs/_benchmarks/round1_verification.png"),
    )

    print("\n=== Done ===")
    print(f"Dataset ready at: {out_dir}")
    print(f"  Images: {len(list(images_out.glob('*.jpg')))}")
    print(f"  Labels: {len(list(labels_out.glob('*.txt')))}")
    print(f"  Train: {len(train_frames)}, Val: {len(val_frames)}")


if __name__ == "__main__":
    main()
