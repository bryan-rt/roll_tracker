"""Prepare 3 training datasets for A/B/C Colab comparison.

1. r2_bbox  — 602 gym frames, bbox only (keypoints zeroed)
2. vicos_12k — 12K subsampled ViCoS (full keypoints)
3. hybrid   — r2_bbox upsampled 20x + vicos_12k

Usage:
    python tools/prepare_3way_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMBINED_DIR = Path("data/training_data/combined")
R2_BBOX_DIR = Path("data/training_data/r2_bbox")
HYBRID_DIR = Path("data/training_data/hybrid")
VICOS_DIR = Path("data/vicos_bjj")
COLAB_PKG = Path("data/colab_package")

ZERO_KP = " ".join(["0.000000 0.000000 0"] * 17)
GYM_UPSAMPLE = 20


def dataset_yaml(path: Path) -> None:
    config = {
        "path": str(path.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "person"},
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    (path / "dataset.yaml").write_text(yaml.dump(config, default_flow_style=False))


# ---------------------------------------------------------------------------
# Step 1: r2_bbox
# ---------------------------------------------------------------------------

def step1_r2_bbox():
    print("=== Step 1: r2_bbox (602 gym frames, keypoints zeroed) ===")

    images_out = R2_BBOX_DIR / "images"
    labels_out = R2_BBOX_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Copy images
    src_images = sorted((COMBINED_DIR / "images").glob("*.jpg"))
    for img in src_images:
        shutil.copy2(img, images_out / img.name)

    # Zero keypoints in labels
    src_labels = sorted((COMBINED_DIR / "labels").glob("*.txt"))
    for lbl in src_labels:
        lines = lbl.read_text().strip().split("\n")
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            bbox = " ".join(parts[:5])  # class x_c y_c w h
            new_lines.append(f"{bbox} {ZERO_KP}")
        (labels_out / lbl.name).write_text("\n".join(new_lines) + "\n" if new_lines else "")

    # Train/val from combined, rewrite paths
    abs_images = str(images_out.resolve())
    for split in ["train.txt", "val.txt"]:
        src_lines = (COMBINED_DIR / split).read_text().strip().split("\n")
        new_lines = [f"{abs_images}/{Path(l).name}" for l in src_lines if l.strip()]
        (R2_BBOX_DIR / split).write_text("\n".join(new_lines) + "\n")

    dataset_yaml(R2_BBOX_DIR)

    # Verify
    sample_lbl = next(labels_out.glob("*.txt"))
    sample_line = sample_lbl.read_text().strip().split("\n")[0]
    parts = sample_line.split()
    print(f"  Images: {len(src_images)}, Labels: {len(src_labels)}")
    print(f"  Sample bbox: {' '.join(parts[:5])}")
    kp_vals = [float(parts[5 + i * 3 + 2]) for i in range(17)]
    print(f"  All keypoint visibility=0: {all(v == 0 for v in kp_vals)}")


# ---------------------------------------------------------------------------
# Step 2: vicos_12k
# ---------------------------------------------------------------------------

def step2_vicos_12k():
    print("\n=== Step 2: vicos_12k (subsampled ViCoS) ===")

    zip_path = COLAB_PKG / "vicos_12k.zip"
    if zip_path.exists():
        print(f"  Already exists: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
        return

    import subprocess
    result = subprocess.run(
        ["python", "tools/package_vicos_for_colab.py", "--subsample", "10"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return

    if zip_path.exists():
        print(f"  Created: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Step 3: hybrid
# ---------------------------------------------------------------------------

def step3_hybrid():
    print("\n=== Step 3: hybrid (gym 20x upsampled + vicos_12k) ===")

    images_out = HYBRID_DIR / "images"
    labels_out = HYBRID_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # --- ViCoS portion: subsample from full dataset ---
    vicos_train = [l.strip() for l in (VICOS_DIR / "train.txt").read_text().strip().split("\n") if l.strip()]
    vicos_val = [l.strip() for l in (VICOS_DIR / "val.txt").read_text().strip().split("\n") if l.strip()]
    vicos_train_sampled = vicos_train[::10]
    vicos_val_sampled = vicos_val[::10]
    vicos_labels_dir = VICOS_DIR / "yolo_labels"

    vicos_train_out = []
    vicos_val_out = []

    for img_list, out_list in [(vicos_train_sampled, vicos_train_out),
                                (vicos_val_sampled, vicos_val_out)]:
        for abs_path in img_list:
            img_path = Path(abs_path)
            img_name = img_path.stem  # e.g. "0000001"
            label_src = vicos_labels_dir / f"{img_name}.txt"

            if not img_path.exists() or not label_src.exists():
                continue

            out_img = images_out / f"vicos_{img_path.name}"
            out_lbl = labels_out / f"vicos_{img_name}.txt"
            shutil.copy2(img_path, out_img)
            shutil.copy2(label_src, out_lbl)
            out_list.append(str(out_img.resolve()))

    print(f"  ViCoS: {len(vicos_train_out)} train + {len(vicos_val_out)} val")

    # --- Gym portion: 20x upsample with zeroed keypoints ---
    gym_train_src = [l.strip() for l in (COMBINED_DIR / "train.txt").read_text().strip().split("\n") if l.strip()]
    gym_val_src = [l.strip() for l in (COMBINED_DIR / "val.txt").read_text().strip().split("\n") if l.strip()]

    gym_train_out = []
    gym_val_out = []

    for img_list, out_list in [(gym_train_src, gym_train_out),
                                (gym_val_src, gym_val_out)]:
        for abs_path in img_list:
            img_path = Path(abs_path)
            img_name = img_path.stem  # e.g. "r1_frame_000000"
            label_src = COMBINED_DIR / "labels" / f"{img_name}.txt"

            if not img_path.exists() or not label_src.exists():
                continue

            # Zero keypoints
            lines = label_src.read_text().strip().split("\n")
            zeroed_lines = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split()
                bbox = " ".join(parts[:5])
                zeroed_lines.append(f"{bbox} {ZERO_KP}")
            zeroed_text = "\n".join(zeroed_lines) + "\n" if zeroed_lines else ""

            # Write N copies
            for copy_idx in range(GYM_UPSAMPLE):
                prefix = f"gym_{copy_idx:02d}_{img_name}"
                out_img = images_out / f"{prefix}.jpg"
                out_lbl = labels_out / f"{prefix}.txt"
                shutil.copy2(img_path, out_img)
                out_lbl.write_text(zeroed_text)
                out_list.append(str(out_img.resolve()))

    print(f"  Gym:   {len(gym_train_out)} train + {len(gym_val_out)} val (×{GYM_UPSAMPLE})")

    # Write train/val
    all_train = vicos_train_out + gym_train_out
    all_val = vicos_val_out + gym_val_out
    (HYBRID_DIR / "train.txt").write_text("\n".join(all_train) + "\n")
    (HYBRID_DIR / "val.txt").write_text("\n".join(all_val) + "\n")

    dataset_yaml(HYBRID_DIR)

    total = len(all_train) + len(all_val)
    print(f"\n  Hybrid dataset:")
    print(f"    ViCoS images: {len(vicos_train_out) + len(vicos_val_out)} (with keypoints)")
    print(f"    Gym images:   {len(gym_train_out) + len(gym_val_out)} (602 × {GYM_UPSAMPLE}, bbox only)")
    print(f"    Total:        {total}")
    print(f"    Train:        {len(all_train)}")
    print(f"    Val:          {len(all_val)}")


# ---------------------------------------------------------------------------
# Step 4: Package for Colab
# ---------------------------------------------------------------------------

def step4_package():
    print("\n=== Step 4: Package all 3 for Colab ===")

    COLAB_PKG.mkdir(parents=True, exist_ok=True)

    # r2_bbox
    r2_zip = COLAB_PKG / "training_data_r2bbox"
    print(f"  Zipping r2_bbox...")
    shutil.make_archive(str(r2_zip), "zip", R2_BBOX_DIR.parent, R2_BBOX_DIR.name)
    r2_zip_file = r2_zip.with_suffix(".zip")
    print(f"    {r2_zip_file.name}: {r2_zip_file.stat().st_size / 1e6:.1f} MB")

    # vicos_12k — already created in step 2
    vicos_zip = COLAB_PKG / "vicos_12k.zip"
    if vicos_zip.exists():
        print(f"    {vicos_zip.name}: {vicos_zip.stat().st_size / 1e6:.1f} MB")

    # hybrid
    hybrid_zip = COLAB_PKG / "training_data_hybrid"
    print(f"  Zipping hybrid...")
    shutil.make_archive(str(hybrid_zip), "zip", HYBRID_DIR.parent, HYBRID_DIR.name)
    hybrid_zip_file = hybrid_zip.with_suffix(".zip")
    print(f"    {hybrid_zip_file.name}: {hybrid_zip_file.stat().st_size / 1e6:.1f} MB")

    # Copy stock model
    stock_model = Path("models/yolo26n-pose.pt")
    stock_dst = COLAB_PKG / stock_model.name
    shutil.copy2(stock_model, stock_dst)
    print(f"    {stock_dst.name}: {stock_dst.stat().st_size / 1e6:.1f} MB")

    # Print instructions
    print(f"""
Upload these to Google Drive at 'roll_tracker_training/':
  1. {r2_zip_file} ({r2_zip_file.stat().st_size / 1e6:.0f} MB)
  2. {vicos_zip} ({vicos_zip.stat().st_size / 1e6:.0f} MB)
  3. {hybrid_zip_file} ({hybrid_zip_file.stat().st_size / 1e6:.0f} MB)
  4. {stock_dst} (base model for all 3)

Colab Cell 2 — change TRAINING_ZIP to:
  Run 1: f"{{DRIVE_PATH}}/training_data_r2bbox.zip"
  Run 2: f"{{DRIVE_PATH}}/vicos_12k.zip"
  Run 3: f"{{DRIVE_PATH}}/training_data_hybrid.zip"

Colab Cell 3 — MODEL_NAME for all 3:
  MODEL_NAME = "yolo26n-pose.pt"

Colab Cell 5 — change ROUND_NAME to:
  Run 1: ROUND_NAME = "r2_bbox"
  Run 2: ROUND_NAME = "vicos"
  Run 3: ROUND_NAME = "hybrid"

All other settings identical:
  BASE_MODEL = "/content/yolo26n-pose.pt"
  FREEZE = 10
  EPOCHS = 100
  LR0 = 0.001
""")


def main():
    step1_r2_bbox()
    step2_vicos_12k()
    step3_hybrid()
    step4_package()
    print("=== Done ===")


if __name__ == "__main__":
    main()
