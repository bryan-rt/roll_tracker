"""Convert ViCoS BJJ Positions Dataset to YOLO Pose format.

Reads the custom ViCoS annotations.json and produces:
  - YOLO pose label files in data/vicos_bjj/yolo_labels/
  - Train/val split by video sequence (01-05 train, 06 val)
  - dataset.yaml with flip_idx
  - Position labels JSON for future classifier work
  - Verification images in outputs/_benchmarks/

Prerequisites:
    python tools/download_vicos.py   # download + extract first

Usage:
    python tools/prepare_vicos_dataset.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VICOS_DIR = Path("data/vicos_bjj")
ANNOTATIONS_PATH = VICOS_DIR / "annotations.json"
IMAGES_DIR = VICOS_DIR / "images"
LABELS_DIR = VICOS_DIR / "yolo_labels"
VERIFY_DIR = Path("outputs/_benchmarks")

# COCO skeleton for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def find_image_path(image_name: str) -> Path | None:
    """Find an image file by its ViCoS name (e.g. '01_00001').

    Images may be in subdirectories or flat. Try common patterns.
    """
    for ext in [".jpg", ".png", ".jpeg"]:
        # Flat
        p = IMAGES_DIR / f"{image_name}{ext}"
        if p.exists():
            return p
        # Subdirectory by sequence
        seq = image_name.split("_")[0]
        p = IMAGES_DIR / seq / f"{image_name}{ext}"
        if p.exists():
            return p
        # Just the frame part in a subdirectory
        p = IMAGES_DIR / seq / f"{image_name.split('_', 1)[1]}{ext}"
        if p.exists():
            return p
    return None


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Read image dimensions without loading full image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height


def convert_entry(entry: dict, img_w: int, img_h: int) -> list[str]:
    """Convert one ViCoS annotation entry to YOLO pose format lines."""
    lines = []
    for pose_key in ["Pose1", "Pose2"]:
        pose = entry[pose_key]  # [[x,y,c], ...] 17 joints

        # Skip if fewer than 3 valid keypoints
        valid_kps = [(x, y, c) for x, y, c in pose if c > 0]
        if len(valid_kps) < 3:
            continue

        # Compute bbox from visible keypoints with 15% padding
        xs = [x for x, y, c in valid_kps]
        ys = [y for x, y, c in valid_kps]
        padding_x = (max(xs) - min(xs)) * 0.15
        padding_y = (max(ys) - min(ys)) * 0.15

        x1 = max(0, min(xs) - padding_x)
        y1 = max(0, min(ys) - padding_y)
        x2 = min(img_w, max(xs) + padding_x)
        y2 = min(img_h, max(ys) + padding_y)

        # Normalize to [0, 1]
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # Build keypoint string (already in COCO order — no remapping)
        kp_parts = []
        for x, y, c in pose:
            if c > 0:
                kp_parts.append(f"{x / img_w:.6f} {y / img_h:.6f} 2")
            else:
                kp_parts.append("0.000000 0.000000 0")

        line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {' '.join(kp_parts)}"
        lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def draw_verification(image_path: Path, label_path: Path, output_path: Path) -> None:
    """Draw bbox + keypoints on an image for visual verification."""
    img = cv2.imread(str(image_path))
    h_img, w_img = img.shape[:2]

    lines = [l for l in label_path.read_text().strip().split("\n") if l.strip()]

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
    ]

    for ann_idx, line in enumerate(lines):
        parts = line.strip().split()
        x_c = float(parts[1]) * w_img
        y_c = float(parts[2]) * h_img
        w = float(parts[3]) * w_img
        h = float(parts[4]) * h_img

        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)
        color = colors[ann_idx % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw keypoints + skeleton
        kps_px = []
        for i in range(17):
            kx = float(parts[5 + i * 3]) * w_img
            ky = float(parts[5 + i * 3 + 1]) * h_img
            kv = int(float(parts[5 + i * 3 + 2]))
            kps_px.append((int(kx), int(ky), kv))
            if kv > 0:
                pt_color = (0, 220, 0) if kv == 2 else (0, 220, 220)
                cv2.circle(img, (int(kx), int(ky)), 3, pt_color, -1)

        for a, b in COCO_SKELETON:
            if kps_px[a][2] > 0 and kps_px[b][2] > 0:
                cv2.line(img, (kps_px[a][0], kps_px[a][1]),
                         (kps_px[b][0], kps_px[b][1]), (235, 206, 135), 1)

        # Label shoulders for left/right verification
        for idx, name in [(5, "L_sh"), (6, "R_sh")]:
            if kps_px[idx][2] > 0:
                cv2.putText(img, name, (kps_px[idx][0] + 5, kps_px[idx][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== ViCoS BJJ → YOLO Pose Conversion ===\n")

    # Load annotations
    print("Loading annotations...")
    annotations = json.loads(ANNOTATIONS_PATH.read_text())
    print(f"  Total entries: {len(annotations)}")

    # Discover image structure
    print("\nDiscovering image file structure...")
    sample = annotations[0]
    sample_path = find_image_path(sample["Image"])
    if sample_path is None:
        # Try listing what's actually in the images dir
        contents = list(IMAGES_DIR.iterdir())[:10]
        print(f"  Could not find image for '{sample['Image']}'")
        print(f"  Images dir contains: {[c.name for c in contents]}")
        raise FileNotFoundError(
            f"Cannot locate image files. Check data/vicos_bjj/images/ structure."
        )
    print(f"  Sample image found: {sample_path}")

    # Cache image dimensions per unique image
    print("\nReading image dimensions (first pass)...")
    dim_cache: dict[str, tuple[int, int]] = {}
    image_paths: dict[str, Path] = {}
    missing = 0

    for i, entry in enumerate(annotations):
        img_name = entry["Image"]
        if img_name in image_paths:
            continue
        img_path = find_image_path(img_name)
        if img_path is None:
            missing += 1
            continue
        image_paths[img_name] = img_path
        if i < 5 or i % 20000 == 0:
            # Read dimensions for first few and periodically
            dim_cache[img_name] = get_image_dimensions(img_path)

        if (i + 1) % 20000 == 0:
            print(f"  Scanned {i + 1}/{len(annotations)} entries...")

    print(f"  Unique images found: {len(image_paths)}, missing: {missing}")

    # If we don't have all dims cached, read a representative sample to check
    # if all images are the same size
    if len(dim_cache) > 0:
        dims = list(dim_cache.values())
        all_same = all(d == dims[0] for d in dims)
        if all_same:
            print(f"  All sampled images are {dims[0][0]}x{dims[0][1]} — assuming uniform")
            default_dim = dims[0]
        else:
            print(f"  Images have varying dimensions — will read each")
            default_dim = None
    else:
        default_dim = None

    # Convert annotations to YOLO format
    print("\nConverting to YOLO pose format...")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    total_annotations = 0
    total_images = 0
    position_labels = {}
    kp_visibility = Counter()
    position_counts = Counter()

    for i, entry in enumerate(annotations):
        img_name = entry["Image"]
        if img_name not in image_paths:
            continue

        # Get dimensions
        if img_name in dim_cache:
            img_w, img_h = dim_cache[img_name]
        elif default_dim is not None:
            img_w, img_h = default_dim
        else:
            img_w, img_h = get_image_dimensions(image_paths[img_name])
            dim_cache[img_name] = (img_w, img_h)

        # Convert
        lines = convert_entry(entry, img_w, img_h)
        label_path = LABELS_DIR / f"{img_name}.txt"
        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        total_annotations += len(lines)
        total_images += 1

        # Position labels
        position_labels[img_name] = entry.get("Position", "unknown")
        position_counts[entry.get("Position", "unknown")] += 1

        # Keypoint visibility stats
        for pose_key in ["Pose1", "Pose2"]:
            for j, (x, y, c) in enumerate(entry[pose_key]):
                if c > 0:
                    kp_visibility[COCO_KEYPOINT_NAMES[j]] += 1

        if (i + 1) % 20000 == 0:
            print(f"  Converted {i + 1}/{len(annotations)} entries...")

    print(f"  Total images: {total_images}")
    print(f"  Total person annotations: {total_annotations}")

    # Save position labels
    pos_path = VICOS_DIR / "position_labels.json"
    pos_path.write_text(json.dumps(position_labels, indent=2))
    print(f"\nPosition labels saved: {pos_path}")

    # Train/val split by sequence
    print("\nCreating train/val split by video sequence...")
    train_images = []
    val_images = []

    for img_name, img_path in sorted(image_paths.items()):
        seq = img_name.split("_")[0]
        label_path = LABELS_DIR / f"{img_name}.txt"
        if not label_path.exists():
            continue
        abs_path = str(img_path.resolve())
        if seq == "06":
            val_images.append(abs_path)
        else:
            train_images.append(abs_path)

    train_txt = VICOS_DIR / "train.txt"
    val_txt = VICOS_DIR / "val.txt"
    train_txt.write_text("\n".join(train_images) + "\n")
    val_txt.write_text("\n".join(val_images) + "\n")
    print(f"  Train: {len(train_images)} images (sequences 01-05)")
    print(f"  Val: {len(val_images)} images (sequence 06)")

    # dataset.yaml
    dataset_config = {
        "path": str(VICOS_DIR.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "person"},
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    yaml_path = VICOS_DIR / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_config, default_flow_style=False))
    print(f"  dataset.yaml written: {yaml_path}")

    # Verification images
    print("\nGenerating verification images...")
    labeled_images = [
        name for name in image_paths
        if (LABELS_DIR / f"{name}.txt").exists()
        and (LABELS_DIR / f"{name}.txt").read_text().strip()
    ]
    random.seed(42)
    verify_samples = random.sample(labeled_images, min(5, len(labeled_images)))

    for idx, img_name in enumerate(verify_samples):
        img_path = image_paths[img_name]
        label_path = LABELS_DIR / f"{img_name}.txt"
        out_path = VERIFY_DIR / f"vicos_verification_{idx}.png"
        draw_verification(img_path, label_path, out_path)
        print(f"  Saved: {out_path}")

    # Print stats
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total images:      {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Train images:      {len(train_images)}")
    print(f"Val images:        {len(val_images)}")

    print(f"\nPosition distribution:")
    for pos, count in sorted(position_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos:<20s} {count:>6d} ({count / total_images * 100:.1f}%)")

    print(f"\nKeypoint visibility (across all person annotations):")
    for kp_name in COCO_KEYPOINT_NAMES:
        count = kp_visibility[kp_name]
        pct = count / (total_annotations * 1.0) * 100 if total_annotations else 0
        print(f"  {kp_name:<20s} {count:>8d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
