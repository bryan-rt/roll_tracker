"""Visualize training bbox size tiers on sample frames.

Draws color-coded bboxes (Red/Yellow/Green) based on min(pixel_w, pixel_h)
to help choose a principled size-filtering threshold for training data cleanup.

Tier definitions:
  Red    — min dim <= 32px
  Yellow — 32px < min dim <= 50px
  Green  — min dim > 50px

Usage:
    python tools/visualize_bbox_tiers.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LABELS_DIR = Path("data/training_data/detection_all_cameras/labels")
IMAGES_DIR = Path("data/training_data/detection_all_cameras/images")
OUTPUT_BASE = Path("outputs/_debug/bbox_tiers")

CAMERAS = {"FP7oJQ": "fp7", "J_EDEw": "jed", "PPDmUg": "ppd"}

TIERS = [
    ("Red",    32, (0, 0, 255)),
    ("Yellow", 50, (0, 215, 255)),
    ("Green",  None, (0, 200, 0)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tier_for(min_dim: float) -> tuple[str, tuple[int, int, int]]:
    """Return (tier_name, color) for a given min pixel dimension."""
    if min_dim <= 32:
        return "Red", (0, 0, 255)
    if min_dim <= 50:
        return "Yellow", (0, 215, 255)
    return "Green", (0, 200, 0)


def draw_label(img, text: str, x: int, y: int, color: tuple[int, int, int]):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Background rectangle
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, color, thickness, cv2.LINE_AA)


def draw_legend(img):
    """Draw a tier legend in the top-left corner."""
    x0, y0 = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    entries = [
        ("Red: min dim <= 32px", (0, 0, 255)),
        ("Yellow: 33-50px", (0, 215, 255)),
        ("Green: > 50px", (0, 200, 0)),
    ]
    line_h = 22
    pad = 6
    box_h = line_h * len(entries) + pad * 2
    box_w = 260
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    for i, (text, color) in enumerate(entries):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(img, text, (x0 + pad, ty), font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Training Bbox Size Tier Visualization")
    print("=" * 70)

    for cam_id, prefix in CAMERAS.items():
        label_files = sorted(LABELS_DIR.glob(f"{prefix}_frame_*.txt"))
        n_frames = len(label_files)

        # Read resolution from first image
        first_img_path = IMAGES_DIR / label_files[0].name.replace(".txt", ".jpg")
        sample = cv2.imread(str(first_img_path))
        img_h, img_w = sample.shape[:2]
        print(f"\n  {cam_id}: {n_frames} frames, {img_w}x{img_h}")

        # Collect all min-dims for stats
        all_min_dims: list[float] = []
        tier_counts = {"Red": 0, "Yellow": 0, "Green": 0}

        # Pre-scan all labels for stats
        for lf in label_files:
            for line in lf.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                w_norm, h_norm = float(parts[3]), float(parts[4])
                px_w = w_norm * img_w
                px_h = h_norm * img_h
                min_dim = min(px_w, px_h)
                all_min_dims.append(min_dim)
                name, _ = tier_for(min_dim)
                tier_counts[name] += 1

        total = len(all_min_dims)
        arr = np.array(all_min_dims)

        # Print stats
        print(f"    min(pixel_w, pixel_h) stats:")
        print(f"      min={arr.min():.1f}  max={arr.max():.1f}  "
              f"mean={arr.mean():.1f}  median={np.median(arr):.1f}")
        for tname in ["Red", "Yellow", "Green"]:
            cnt = tier_counts[tname]
            pct = cnt / total * 100
            print(f"    {tname:>6s}: {cnt:>5d} ({pct:>5.1f}%)")

        # Select 10 evenly-spaced frames
        indices = np.linspace(0, n_frames - 1, 10, dtype=int)
        out_dir = OUTPUT_BASE / cam_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for fi in indices:
            lf = label_files[fi]
            img_path = IMAGES_DIR / lf.name.replace(".txt", ".jpg")
            img = cv2.imread(str(img_path))

            for line in lf.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cx, cy = float(parts[1]), float(parts[2])
                w_norm, h_norm = float(parts[3]), float(parts[4])
                px_w = w_norm * img_w
                px_h = h_norm * img_h
                min_dim = min(px_w, px_h)
                _, color = tier_for(min_dim)

                x1 = int((cx - w_norm / 2) * img_w)
                y1 = int((cy - h_norm / 2) * img_h)
                x2 = int((cx + w_norm / 2) * img_w)
                y2 = int((cy + h_norm / 2) * img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = f"{int(px_w)}x{int(px_h)}"
                draw_label(img, label, x1, y1 - 1, color)

            draw_legend(img)

            # Use original frame number from filename
            frame_num = lf.stem.split("_")[-1]
            out_path = out_dir / f"frame_{frame_num}.png"
            cv2.imwrite(str(out_path), img)

        print(f"    Saved 10 overlay images to {out_dir}/")

    # Final summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"\n  {'Camera':>8s}  {'Total':>6s}  {'Red (<=32px)':>14s}  "
          f"{'Yellow (33-50)':>16s}  {'Green (>50px)':>15s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*15}")

    for cam_id, prefix in CAMERAS.items():
        label_files = sorted(LABELS_DIR.glob(f"{prefix}_frame_*.txt"))
        first_img_path = IMAGES_DIR / label_files[0].name.replace(".txt", ".jpg")
        sample = cv2.imread(str(first_img_path))
        img_h, img_w = sample.shape[:2]

        counts = {"Red": 0, "Yellow": 0, "Green": 0}
        total = 0
        for lf in label_files:
            for line in lf.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                px_w = float(parts[3]) * img_w
                px_h = float(parts[4]) * img_h
                name, _ = tier_for(min(px_w, px_h))
                counts[name] += 1
                total += 1

        def fmt(c):
            return f"{c} ({c/total*100:.1f}%)"

        print(f"  {cam_id:>8s}  {total:>6d}  {fmt(counts['Red']):>14s}  "
              f"{fmt(counts['Yellow']):>16s}  {fmt(counts['Green']):>15s}")

    print()


if __name__ == "__main__":
    main()
