"""Prepare Round 2 training data and build combined Round 1 + Round 2 dataset.

Unpacks CVAT exports, filters to annotated frames, merges OBBox + Pose with
COCO keypoint remapping, extracts frames from source video, and builds a
combined dataset ready for YOLO training.

Usage:
    python tools/prepare_round2_dataset.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import yaml

# Add tools/ to path so we can import from merge_cvat_exports
sys.path.insert(0, str(Path(__file__).parent))
from merge_cvat_exports import merge_frame, verify_and_visualize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path("data/training_data")
R2_POSE_ZIP = BASE / "training_YOLO_pose_J_EDEw_clip1_0-3000.zip"
R2_OBBOX_ZIP = BASE / "training_YOLO_obbox_J_EDEw_clip1_0-3000.zip"
R2_UNPACKED = BASE / "round2_unpacked"
R2_OUT = BASE / "round2"
R1_DIR = BASE / "round1"
COMBINED_DIR = BASE / "combined"
VIDEO_PATH = Path("data/cvat_tasks/round1_20260497_J_EDEw/J_EDEw-20260318-200015.mp4")
VERIFY_IMG = Path("outputs/_benchmarks/round2_verification.png")

# Annotated frames: every 10th frame from 0 to 3000
ANNOTATED_FRAMES = list(range(0, 3001, 10))  # 301 frames


def step1_unpack():
    """Unpack Round 2 CVAT exports."""
    print("=== Step 1: Unpack Round 2 exports ===")

    pose_dir = R2_UNPACKED / "yolo_pose"
    obbox_dir = R2_UNPACKED / "yolo_obbox"

    for zip_path, out_dir in [(R2_POSE_ZIP, pose_dir), (R2_OBBOX_ZIP, obbox_dir)]:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        label_dir = out_dir / "labels" / "train"
        count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        print(f"  {out_dir.name}: {count} label files")

    return pose_dir, obbox_dir


def step2_filter(pose_dir, obbox_dir):
    """Filter to annotated frames only (every 10th, 0-3000)."""
    print("\n=== Step 2: Filter to annotated frames ===")

    pose_labels = pose_dir / "labels" / "train"
    obbox_labels = obbox_dir / "labels" / "train"

    pose_count = 0
    obbox_count = 0
    mismatches = []

    for frame_num in ANNOTATED_FRAMES:
        fname = f"frame_{frame_num:06d}.txt"
        p_file = pose_labels / fname
        o_file = obbox_labels / fname

        if not p_file.exists():
            print(f"  WARNING: missing pose label {fname}")
            continue
        if not o_file.exists():
            print(f"  WARNING: missing obbox label {fname}")
            continue

        p_lines = [l for l in p_file.read_text().strip().split("\n") if l.strip()]
        o_lines = [l for l in o_file.read_text().strip().split("\n") if l.strip()]

        if len(p_lines) != len(o_lines):
            mismatches.append((fname, len(p_lines), len(o_lines)))

        pose_count += 1
        obbox_count += 1

    print(f"  Pose frames found: {pose_count}/301")
    print(f"  OBBox frames found: {obbox_count}/301")
    if mismatches:
        print(f"  WARNING: {len(mismatches)} frames with annotation count mismatch:")
        for fname, pc, oc in mismatches[:5]:
            print(f"    {fname}: pose={pc}, obbox={oc}")
    else:
        print("  All frames have matching annotation counts between pose and obbox.")

    return pose_labels, obbox_labels


def step3_merge(pose_labels, obbox_labels):
    """Merge OBBox bboxes + Pose keypoints with COCO remapping."""
    print("\n=== Step 3: Merge OBBox + Pose with COCO keypoint remapping ===")

    labels_out = R2_OUT / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    total_anns = 0
    merged_count = 0

    for frame_num in ANNOTATED_FRAMES:
        fname = f"frame_{frame_num:06d}.txt"
        pose_file = pose_labels / fname
        obbox_file = obbox_labels / fname

        if not pose_file.exists() or not obbox_file.exists():
            continue

        pose_lines = [l for l in pose_file.read_text().strip().split("\n") if l.strip()]
        obbox_lines = [l for l in obbox_file.read_text().strip().split("\n") if l.strip()]

        if not pose_lines or not obbox_lines:
            (labels_out / fname).write_text("")
            merged_count += 1
            continue

        merged = merge_frame(pose_lines, obbox_lines)
        (labels_out / fname).write_text("\n".join(merged) + "\n")
        total_anns += len(merged)
        merged_count += 1

    print(f"  Merged {total_anns} annotations across {merged_count} frames")
    return labels_out


def step4_extract_frames():
    """Extract annotated frames from source video."""
    print("\n=== Step 4: Extract frames from source video ===")

    images_out = R2_OUT / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {VIDEO_PATH.name} ({total_frames} frames)")

    count = 0
    for frame_num in ANNOTATED_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: could not read frame {frame_num}")
            continue
        cv2.imwrite(str(images_out / f"frame_{frame_num:06d}.jpg"), frame)
        count += 1

    cap.release()
    print(f"  Extracted {count} frames to {images_out}")
    return images_out


def step5_verify(labels_out, images_out, pose_labels, obbox_labels):
    """Verify Round 2 data and generate verification image."""
    print("\n=== Step 5: Verification ===")

    label_count = len(list(labels_out.glob("*.txt")))
    image_count = len(list(images_out.glob("*.jpg")))
    print(f"  Labels: {label_count}, Images: {image_count}")

    if label_count != image_count:
        print(f"  WARNING: count mismatch!")
    if label_count != 301:
        print(f"  WARNING: expected 301, got {label_count}")

    # Find a frame with annotations for verification
    verify_frame = None
    for frame_num in [100, 200, 300, 0, 10]:
        fname = f"frame_{frame_num * 10 if frame_num <= 300 else frame_num:06d}.txt"
        # Use actual annotated frame numbers
        test_num = frame_num
        test_fname = f"frame_{test_num:06d}.txt"
        label_file = labels_out / test_fname
        if label_file.exists() and label_file.read_text().strip():
            verify_frame = test_num
            break

    if verify_frame is None:
        # Find any non-empty label
        for f in sorted(labels_out.glob("*.txt")):
            if f.read_text().strip():
                verify_frame = int(f.stem.replace("frame_", ""))
                break

    if verify_frame is not None:
        fname = f"frame_{verify_frame:06d}"
        print(f"\n  Verifying frame {verify_frame}...")
        verify_and_visualize(
            labels_out / f"{fname}.txt",
            obbox_labels / f"{fname}.txt",
            pose_labels / f"{fname}.txt",
            images_out / f"{fname}.jpg",
            VERIFY_IMG,
        )
    else:
        print("  WARNING: no non-empty label found for verification")


def step6_build_combined():
    """Build combined Round 1 + Round 2 dataset."""
    print("\n=== Step 6: Build combined dataset ===")

    combined_images = COMBINED_DIR / "images"
    combined_labels = COMBINED_DIR / "labels"
    combined_images.mkdir(parents=True, exist_ok=True)
    combined_labels.mkdir(parents=True, exist_ok=True)

    # Copy Round 1 files with r1_ prefix
    r1_images = sorted((R1_DIR / "images").glob("*.jpg"))
    r1_labels = sorted((R1_DIR / "labels").glob("*.txt"))
    print(f"  Round 1: {len(r1_images)} images, {len(r1_labels)} labels")

    for img in r1_images:
        shutil.copy2(img, combined_images / f"r1_{img.name}")
    for lbl in r1_labels:
        shutil.copy2(lbl, combined_labels / f"r1_{lbl.name}")

    # Copy Round 2 files with r2_ prefix
    r2_images = sorted((R2_OUT / "images").glob("*.jpg"))
    r2_labels = sorted((R2_OUT / "labels").glob("*.txt"))
    print(f"  Round 2: {len(r2_images)} images, {len(r2_labels)} labels")

    for img in r2_images:
        shutil.copy2(img, combined_images / f"r2_{img.name}")
    for lbl in r2_labels:
        shutil.copy2(lbl, combined_labels / f"r2_{lbl.name}")

    # Train/val split
    # R1: first 250 train, last 51 val (frames 0-249 train, 250-300 val)
    r1_train = [f"r1_{f.name}" for f in r1_images[:250]]
    r1_val = [f"r1_{f.name}" for f in r1_images[250:]]

    # R2: first 250 train, last 51 val (sorted by frame number)
    r2_train = [f"r2_{f.name}" for f in r2_images[:250]]
    r2_val = [f"r2_{f.name}" for f in r2_images[250:]]

    abs_images = str(combined_images.resolve())

    train_paths = [f"{abs_images}/{f}" for f in r1_train + r2_train]
    val_paths = [f"{abs_images}/{f}" for f in r1_val + r2_val]

    train_txt = COMBINED_DIR / "train.txt"
    val_txt = COMBINED_DIR / "val.txt"
    train_txt.write_text("\n".join(train_paths) + "\n")
    val_txt.write_text("\n".join(val_paths) + "\n")

    print(f"  Train: {len(train_paths)} ({len(r1_train)} R1 + {len(r2_train)} R2)")
    print(f"  Val: {len(val_paths)} ({len(r1_val)} R1 + {len(r2_val)} R2)")

    # dataset.yaml
    dataset_config = {
        "path": str(COMBINED_DIR.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "person"},
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    yaml_path = COMBINED_DIR / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_config, default_flow_style=False))
    print(f"  dataset.yaml written: {yaml_path}")

    return len(r1_images), len(r2_images), len(train_paths), len(val_paths)


def step7_summary(r1_count, r2_count, train_count, val_count):
    """Print final summary."""
    print("\n=== Step 7: Summary ===")

    # Count annotations
    r1_anns = 0
    for f in (R1_DIR / "labels").glob("*.txt"):
        lines = [l for l in f.read_text().strip().split("\n") if l.strip()]
        r1_anns += len(lines)

    r2_anns = 0
    for f in (R2_OUT / "labels").glob("*.txt"):
        lines = [l for l in f.read_text().strip().split("\n") if l.strip()]
        r2_anns += len(lines)

    print(f"  Round 1 (FP7oJQ): {r1_count} frames, {r1_anns} annotations")
    print(f"  Round 2 (J_EDEw): {r2_count} frames, {r2_anns} annotations")
    print(f"  Combined: {r1_count + r2_count} frames, {r1_anns + r2_anns} annotations")
    print(f"  Train: {train_count} frames")
    print(f"  Val: {val_count} frames")
    print(f"  Cameras: FP7oJQ (1920x1080), J_EDEw (1280x720)")


def main():
    pose_dir, obbox_dir = step1_unpack()
    pose_labels, obbox_labels = step2_filter(pose_dir, obbox_dir)
    labels_out = step3_merge(pose_labels, obbox_labels)
    images_out = step4_extract_frames()
    step5_verify(labels_out, images_out, pose_labels, obbox_labels)
    r1_count, r2_count, train_count, val_count = step6_build_combined()
    step7_summary(r1_count, r2_count, train_count, val_count)
    print("\n=== Done — ready for training ===")


if __name__ == "__main__":
    main()
