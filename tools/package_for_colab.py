"""Package training data for Colab upload.

Usage: python tools/package_for_colab.py [--dataset path] [--model path]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Package training data for Colab upload")
    parser.add_argument("--dataset", default="data/training_data/combined",
                        help="Path to dataset directory")
    parser.add_argument("--model", default="models/bjj-pose-r1.pt",
                        help="Base model to upload")
    parser.add_argument("--output", default="data/colab_package",
                        help="Output directory for zip + model")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zip the training data
    dataset_path = Path(args.dataset)
    zip_path = output_dir / "training_data"
    print(f"Zipping {dataset_path}...")
    shutil.make_archive(str(zip_path), "zip", dataset_path.parent, dataset_path.name)
    zip_file = zip_path.with_suffix(".zip")
    print(f"Created: {zip_file} ({zip_file.stat().st_size / 1e6:.1f} MB)")

    # Copy model
    model_src = Path(args.model)
    model_dst = output_dir / model_src.name
    shutil.copy2(model_src, model_dst)
    print(f"Copied model: {model_dst}")

    print(f"\nUpload these to Google Drive at 'roll_tracker_training/':")
    print(f"  1. {zip_file}")
    print(f"  2. {model_dst}")


if __name__ == "__main__":
    main()
