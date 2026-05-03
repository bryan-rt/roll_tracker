"""Package a subsampled ViCoS dataset for Colab upload.

Takes every Nth image to create a manageable zip for Google Drive upload.

Usage:
    python tools/package_vicos_for_colab.py                  # default: every 10th
    python tools/package_vicos_for_colab.py --subsample 5    # every 5th (~24K)
    python tools/package_vicos_for_colab.py --subsample 1    # full dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Package subsampled ViCoS for Colab")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth image (default: 10 → ~12K images)")
    parser.add_argument("--vicos-dir", default="data/vicos_bjj",
                        help="ViCoS dataset directory")
    parser.add_argument("--output", default="data/colab_package",
                        help="Output directory")
    args = parser.parse_args()

    vicos_dir = Path(args.vicos_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the full train/val lists
    train_txt = vicos_dir / "train.txt"
    val_txt = vicos_dir / "val.txt"

    if not train_txt.exists() or not val_txt.exists():
        print("ERROR: Run tools/prepare_vicos_dataset.py first to generate train/val split")
        return

    train_paths = [l.strip() for l in train_txt.read_text().strip().split("\n") if l.strip()]
    val_paths = [l.strip() for l in val_txt.read_text().strip().split("\n") if l.strip()]

    # Subsample
    train_sampled = train_paths[::args.subsample]
    val_sampled = val_paths[::args.subsample]

    n_total = len(train_sampled) + len(val_sampled)
    suffix = f"{n_total // 1000}k" if n_total >= 1000 else str(n_total)
    zip_name = f"vicos_{suffix}"

    print(f"Subsampling every {args.subsample}th image:")
    print(f"  Train: {len(train_paths)} → {len(train_sampled)}")
    print(f"  Val: {len(val_paths)} → {len(val_sampled)}")
    print(f"  Total: {n_total}")

    # Build zip with images + labels + dataset.yaml
    labels_dir = vicos_dir / "yolo_labels"
    zip_path = output_dir / f"{zip_name}.zip"

    print(f"\nCreating {zip_path}...")

    # Write subsampled train/val lists with Colab-relative paths
    # (will be rewritten by Colab notebook Cell 4, but include for reference)
    all_sampled = train_sampled + val_sampled
    included = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_abs_path in all_sampled:
            img_path = Path(img_abs_path)
            img_name = img_path.stem  # e.g. "01_00001"

            if not img_path.exists():
                skipped += 1
                continue

            label_path = labels_dir / f"{img_name}.txt"
            if not label_path.exists():
                skipped += 1
                continue

            # Add image and label to zip
            zf.write(img_path, f"images/{img_path.name}")
            zf.write(label_path, f"labels/{label_path.name}")
            included += 1

        # Write train.txt / val.txt with relative paths
        train_lines = []
        for p in train_sampled:
            name = Path(p).name
            train_lines.append(f"images/{name}")
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

        val_lines = []
        for p in val_sampled:
            name = Path(p).name
            val_lines.append(f"images/{name}")
        zf.writestr("val.txt", "\n".join(val_lines) + "\n")

        # Write dataset.yaml
        dataset_config = {
            "path": "/content/training_data",  # Colab path
            "train": "train.txt",
            "val": "val.txt",
            "names": {0: "person"},
            "kpt_shape": [17, 3],
            "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        }
        zf.writestr("dataset.yaml", yaml.dump(dataset_config, default_flow_style=False))

    zip_size = zip_path.stat().st_size
    print(f"  Included: {included} images+labels, skipped: {skipped}")
    print(f"  Zip size: {zip_size / 1e9:.2f} GB")

    print(f"\nUpload to Google Drive at 'roll_tracker_training/':")
    print(f"  {zip_path}")
    print(f"\nColab training (Phase 1 — ViCoS fine-tune):")
    print(f"  BASE_MODEL = stock yolo26n-pose.pt (NOT bjj-r1/r2)")
    print(f"  EPOCHS = 50")
    print(f"  FREEZE = 10")
    print(f"  Output: bjj-pose-vicos.pt")


if __name__ == "__main__":
    main()
