"""Download ViCoS BJJ Positions Dataset.

Source: https://vicos.si/resources/jiujitsu/
License: CC BY-NC-SA 4.0

Run manually (not through CLI — large download):
    source .venv/bin/activate
    python tools/download_vicos.py
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

IMAGES_URL = "http://data.vicos.si/datasets/JuiJuitsu/images.zip"
ANNOTATIONS_URL = "http://data.vicos.si/datasets/JuiJuitsu/annotations.json"
OUTPUT_DIR = Path("data/vicos_bjj")


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ann_path = OUTPUT_DIR / "annotations.json"
    if not ann_path.exists():
        print("Downloading annotations.json...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, ann_path)
        print(f"Saved: {ann_path} ({ann_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"annotations.json already exists, skipping download")

    zip_path = OUTPUT_DIR / "images.zip"
    if not zip_path.exists():
        print("Downloading images.zip (this may take a while — 120K images)...")
        urllib.request.urlretrieve(IMAGES_URL, zip_path)
        print(f"Saved: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"images.zip already exists, skipping download")

    # Unzip images
    images_dir = OUTPUT_DIR / "images"
    if not images_dir.exists():
        print("Extracting images...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(OUTPUT_DIR)
        print(f"Extracted to: {images_dir}")
    else:
        print(f"images/ already exists, skipping extraction")

    print(f"\nDone. Dataset at: {OUTPUT_DIR}")


if __name__ == "__main__":
    download()
