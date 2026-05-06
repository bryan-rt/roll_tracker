"""Prepare combined 3-camera detection dataset for yolo26n.pt retraining.

Extracts selected frames from CVAT tracking detection exports across all three
gym cameras, strips track IDs, remaps class 1→0, extracts video frames, builds
train/val split, and packages for Kaggle/Colab upload.

Usage:
    python tools/prepare_detection_dataset.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path("data/training_data")
OUTPUT_DIR = BASE / "detection_all_cameras"
ZIP_OUTPUT = BASE / "training_data_detection_all_cameras.zip"

CAMERAS = {
    # FP7oJQ: range(0, 301, 1) — all 301 frames, no stride.
    # Previously incorrectly used range(0, 3001, 10) which misaligned
    # annotations with frames beyond the annotated clip window.
    "FP7oJQ": {
        "zip": BASE / "training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip",
        "video": Path("data/cvat_tasks/round1_20260497_FP7oJQ/FP7oJQ-20260318-200014.mp4"),
        "frames": list(range(0, 301, 1)),  # 301 frames, every frame
        "prefix": "fp7",
    },
    "J_EDEw": {
        "zip": BASE / "training_YOLO_track_detections_J_EDEw_clip1_0-3000.zip",
        "video": Path("data/cvat_tasks/round1_20260497_J_EDEw/J_EDEw-20260318-200015.mp4"),
        "frames": list(range(0, 3001, 10)),  # 301 frames
        "prefix": "jed",
    },
    "PPDmUg": {
        "zip": BASE / "training_YOLO_track_detections_PPDmUg_clip1_0-2990.zip",
        "video": Path("data/raw/nest/training_samples/training_PPDmUg_3000.mp4"),
        "frames": list(range(0, 2991, 10)),  # 300 frames
        "prefix": "ppd",
    },
}

# Train/val split: ~83/17 temporal split per camera (tail goes to val)
VAL_COUNT_PER_CAMERA = 51


# ---------------------------------------------------------------------------
# Step 1: Unzip + filter labels to every-10th frames
# ---------------------------------------------------------------------------

def step1_extract_labels():
    """Unzip each tracking export and copy labels for selected frames."""
    print("=== Step 1: Extract labels for selected frames ===")

    labels_out = OUTPUT_DIR / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = {}

    for cam_name, cam in CAMERAS.items():
        found = 0
        missing = 0

        with zipfile.ZipFile(cam["zip"]) as zf:
            for frame_num in cam["frames"]:
                fname = f"labels/train/frame_{frame_num:06d}.txt"
                try:
                    data = zf.read(fname)
                    out_name = f"{cam['prefix']}_frame_{frame_num:06d}.txt"
                    (labels_out / out_name).write_bytes(data)
                    found += 1
                except KeyError:
                    print(f"  WARNING: {cam_name} missing {fname}")
                    missing += 1

        stats[cam_name] = {"found": found, "missing": missing, "expected": len(cam["frames"])}
        print(f"  {cam_name}: {found}/{len(cam['frames'])} labels extracted"
              f"{f' ({missing} missing)' if missing else ''}")

    return stats


# ---------------------------------------------------------------------------
# Step 2: Strip track ID + remap class
# ---------------------------------------------------------------------------

def step2_fix_labels():
    """Strip track_id field and remap class 1→0 in all label files."""
    print("\n=== Step 2: Strip track IDs + remap class ===")

    labels_dir = OUTPUT_DIR / "labels"
    total_annotations = 0
    total_warnings = 0

    for label_file in sorted(labels_dir.glob("*.txt")):
        lines = label_file.read_text().strip().split("\n")
        new_lines = []

        for line in lines:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 6:
                print(f"  WARNING: {label_file.name} unexpected field count ({len(parts)}): {line}")
                total_warnings += 1
                continue

            cls, cx, cy, w, h, _track_id = parts

            # Remap class: CVAT bbox class 1 → person class 0
            if cls not in ("0", "1"):
                print(f"  WARNING: {label_file.name} unexpected class {cls}: {line}")
                total_warnings += 1
                continue

            # Validate bbox coordinates
            try:
                coords = [float(cx), float(cy), float(w), float(h)]
                if not all(0.0 < v <= 1.0 for v in coords):
                    print(f"  WARNING: {label_file.name} coords out of range: {line}")
                    total_warnings += 1
                    continue
            except ValueError:
                print(f"  WARNING: {label_file.name} non-numeric coords: {line}")
                total_warnings += 1
                continue

            new_lines.append(f"0 {cx} {cy} {w} {h}")
            total_annotations += 1

        label_file.write_text("\n".join(new_lines) + "\n" if new_lines else "")

    print(f"  {total_annotations} annotations processed across {len(list(labels_dir.glob('*.txt')))} files")
    if total_warnings:
        print(f"  {total_warnings} lines flagged (see warnings above)")
    else:
        print("  Validation clean — no warnings")

    return total_annotations, total_warnings


# ---------------------------------------------------------------------------
# Step 3: Extract frames from source videos
# ---------------------------------------------------------------------------

def step3_extract_frames():
    """Extract selected frames from source videos for all cameras."""
    print("\n=== Step 3: Extract frames from source videos ===")

    images_out = OUTPUT_DIR / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    for cam_name, cam in CAMERAS.items():
        cap = cv2.VideoCapture(str(cam["video"]))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {cam['video']}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {cam_name}: {cam['video'].name} ({total_frames} frames)")

        count = 0
        for frame_num in cam["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"    WARNING: could not read frame {frame_num}")
                continue
            out_name = f"{cam['prefix']}_frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(images_out / out_name), frame)
            count += 1

        cap.release()
        print(f"    Extracted {count} frames")


# ---------------------------------------------------------------------------
# Step 4: Train/val split (per-camera stratified, temporal)
# ---------------------------------------------------------------------------

def step4_split():
    """Build train/val split — 83/17 temporal split, stratified per camera."""
    print("\n=== Step 4: Train/val split ===")

    train_paths = []
    val_paths = []

    for cam_name, cam in CAMERAS.items():
        # Sorted frame list for this camera
        cam_images = sorted(
            (OUTPUT_DIR / "images").glob(f"{cam['prefix']}_frame_*.jpg")
        )

        n_val = min(VAL_COUNT_PER_CAMERA, len(cam_images))
        n_train = len(cam_images) - n_val

        cam_train = [f"images/{f.name}" for f in cam_images[:n_train]]
        cam_val = [f"images/{f.name}" for f in cam_images[n_train:]]

        train_paths.extend(cam_train)
        val_paths.extend(cam_val)

        print(f"  {cam_name}: {len(cam_train)} train / {len(cam_val)} val")

    (OUTPUT_DIR / "train.txt").write_text("\n".join(train_paths) + "\n")
    (OUTPUT_DIR / "val.txt").write_text("\n".join(val_paths) + "\n")

    print(f"  Total: {len(train_paths)} train / {len(val_paths)} val = {len(train_paths) + len(val_paths)}")
    return len(train_paths), len(val_paths)


# ---------------------------------------------------------------------------
# Step 5: Write dataset.yaml
# ---------------------------------------------------------------------------

def step5_dataset_yaml():
    """Write detection-only dataset.yaml."""
    print("\n=== Step 5: Write dataset.yaml ===")

    config = {
        "path": ".",
        "train": "train.txt",
        "val": "val.txt",
        "nc": 1,
        "names": {0: "person"},
    }
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml.dump(config, default_flow_style=False))
    print(f"  Written: {yaml_path}")


# ---------------------------------------------------------------------------
# Step 6: Zip for upload
# ---------------------------------------------------------------------------

def step6_package():
    """Zip the dataset for Kaggle/Colab upload."""
    print("\n=== Step 6: Package for upload ===")

    if ZIP_OUTPUT.exists():
        ZIP_OUTPUT.unlink()

    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(OUTPUT_DIR)
                zf.write(f, arcname)

    size_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  {ZIP_OUTPUT} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 7: Summary
# ---------------------------------------------------------------------------

def step7_summary(label_stats, total_annotations, total_warnings, n_train, n_val):
    """Print final sanity-check summary."""
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print("\nPer-camera frame counts:")
    total_frames = 0
    total_missing = 0
    for cam_name, s in label_stats.items():
        total_frames += s["found"]
        total_missing += s["missing"]
        status = "" if s["missing"] == 0 else f"  ({s['missing']} MISSING)"
        print(f"  {cam_name}: {s['found']} frames{status}")

    print(f"\nTotal frames: {total_frames}")
    print(f"Total annotations: {total_annotations}")
    if total_missing:
        print(f"Missing labels: {total_missing}")
    if total_warnings:
        print(f"Validation warnings: {total_warnings}")

    print(f"\nTrain/val split:")
    print(f"  Train: {n_train}")
    print(f"  Val:   {n_val}")
    print(f"  Total: {n_train + n_val}")
    print(f"  Ratio: {n_train / (n_train + n_val) * 100:.1f}% / {n_val / (n_train + n_val) * 100:.1f}%")

    print(f"\nOutput: {ZIP_OUTPUT}")
    print(f"Upload to Google Drive → roll_tracker_training/ for Colab")
    print(f"  or Kaggle dataset bryanrt/roll-tracker-training")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    label_stats = step1_extract_labels()
    total_annotations, total_warnings = step2_fix_labels()
    step3_extract_frames()
    n_train, n_val = step4_split()
    step5_dataset_yaml()
    step6_package()
    step7_summary(label_stats, total_annotations, total_warnings, n_train, n_val)


if __name__ == "__main__":
    main()
