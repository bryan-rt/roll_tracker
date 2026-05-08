"""FP7oJQ false positive investigation — read-only analysis.

Analyzes the detection_all_cameras training dataset to understand why
bjj-detect-all-cameras.pt produces high-confidence false positives on FP7oJQ.

Four objectives:
  1. Spatial distribution analysis (annotation heatmap per camera)
  2. Temporal persistence check (how often each region is occupied)
  3. False positive location mapping (inference on test clip)
  4. Per-camera annotation coverage summary

Writes debug images to outputs/_debug/ only. No other files modified.

Usage:
    python tools/investigate_fp7_annotations.py
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LABELS_DIR = Path("data/training_data/detection_all_cameras/labels")
IMAGES_DIR = Path("data/training_data/detection_all_cameras/images")
DEBUG_ANNOTATION = Path("outputs/_debug/fp7_annotation_check")
DEBUG_FP = Path("outputs/_debug/fp7_false_positive_check")

PREFIXES = {"FP7oJQ": "fp7", "J_EDEw": "jed", "PPDmUg": "ppd"}
GRID = 10  # 10x10 spatial grid

TEST_CLIP = Path(
    "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77"
    "/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-202748.mp4"
)
MODEL_PATH = Path("models/bjj-detect-all-cameras.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels(prefix: str) -> dict[str, list[tuple[float, float, float, float]]]:
    """Load all label files for a prefix. Returns {filename: [(cx,cy,w,h), ...]}."""
    result = {}
    for f in sorted(LABELS_DIR.glob(f"{prefix}_frame_*.txt")):
        boxes = []
        for line in f.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cx, cy, w, h))
        result[f.name] = boxes
    return result


def to_grid(cx: float, cy: float) -> tuple[int, int]:
    """Map normalized (cx, cy) to a grid cell (row, col)."""
    row = min(int(cy * GRID), GRID - 1)
    col = min(int(cx * GRID), GRID - 1)
    return row, col


def print_grid(grid: list[list[int]], title: str):
    """Print a 10x10 grid as formatted ASCII."""
    print(f"\n  {title}")
    print("     " + "".join(f"{c:>6d}" for c in range(GRID)))
    print("     " + "-" * (GRID * 6))
    for r in range(GRID):
        print(f"  {r:>2d} |" + "".join(f"{grid[r][c]:>6d}" for c in range(GRID)))


# ---------------------------------------------------------------------------
# Objective 1: Spatial Distribution Analysis
# ---------------------------------------------------------------------------

def objective1_spatial_distribution() -> dict[str, set[tuple[int, int]]]:
    """Analyze annotation spatial distribution per camera.

    Returns high-density cells per camera for cross-referencing.
    """
    print("\n" + "=" * 70)
    print("OBJECTIVE 1: Annotation Spatial Distribution Analysis")
    print("=" * 70)

    high_density_cells: dict[str, set[tuple[int, int]]] = {}

    for cam, prefix in PREFIXES.items():
        labels = load_labels(prefix)
        all_cx, all_cy = [], []
        grid = [[0] * GRID for _ in range(GRID)]

        for boxes in labels.values():
            for cx, cy, _w, _h in boxes:
                all_cx.append(cx)
                all_cy.append(cy)
                r, c = to_grid(cx, cy)
                grid[r][c] += 1

        total = len(all_cx)
        mean_cx = statistics.mean(all_cx)
        mean_cy = statistics.mean(all_cy)
        std_cx = statistics.stdev(all_cx)
        std_cy = statistics.stdev(all_cy)

        print(f"\n  {cam} ({prefix}): {total} annotations across {len(labels)} frames")
        print(f"    cx: mean={mean_cx:.4f}  std={std_cx:.4f}")
        print(f"    cy: mean={mean_cy:.4f}  std={std_cy:.4f}")

        print_grid(grid, f"{cam} annotation density (10x10 grid)")

        # Flag cells > 3x mean
        flat = [grid[r][c] for r in range(GRID) for c in range(GRID)]
        occupied = [v for v in flat if v > 0]
        mean_cell = statistics.mean(occupied) if occupied else 0
        threshold = mean_cell * 3

        hot_cells: set[tuple[int, int]] = set()
        if threshold > 0:
            flagged = []
            for r in range(GRID):
                for c in range(GRID):
                    if grid[r][c] > threshold:
                        hot_cells.add((r, c))
                        cy_range = f"{r / GRID:.1f}-{(r + 1) / GRID:.1f}"
                        cx_range = f"{c / GRID:.1f}-{(c + 1) / GRID:.1f}"
                        flagged.append(
                            f"    cell ({r},{c}): {grid[r][c]:>5d} annotations "
                            f"(cx={cx_range}, cy={cy_range}) — {grid[r][c] / mean_cell:.1f}x mean"
                        )
            if flagged:
                print(f"\n  HIGH-DENSITY cells (>{threshold:.0f}, >3x mean of {mean_cell:.0f}):")
                for line in flagged:
                    print(line)
            else:
                print(f"\n  No cells exceed 3x mean ({mean_cell:.0f}).")

        high_density_cells[cam] = hot_cells

    return high_density_cells


# ---------------------------------------------------------------------------
# Objective 2: Temporal Persistence Check
# ---------------------------------------------------------------------------

def objective2_temporal_persistence():
    """Check how often each spatial region has annotations across frames."""
    print("\n" + "=" * 70)
    print("OBJECTIVE 2: Temporal Persistence Check")
    print("=" * 70)

    for cam, prefix in PREFIXES.items():
        labels = load_labels(prefix)
        total_frames = len(labels)

        # Track which frames each grid cell appears in
        cell_frames: dict[tuple[int, int], set[str]] = defaultdict(set)
        for fname, boxes in labels.items():
            for cx, cy, _w, _h in boxes:
                r, c = to_grid(cx, cy)
                cell_frames[(r, c)].add(fname)

        print(f"\n  {cam} ({prefix}): {total_frames} frames")

        # Persistence ratio per cell
        persistent_70 = []
        persistent_100 = []
        for (r, c), frames in sorted(cell_frames.items()):
            ratio = len(frames) / total_frames
            if ratio >= 0.70:
                cy_range = f"{r / GRID:.1f}-{(r + 1) / GRID:.1f}"
                cx_range = f"{c / GRID:.1f}-{(c + 1) / GRID:.1f}"
                persistent_70.append(
                    f"    cell ({r},{c}): {len(frames)}/{total_frames} frames "
                    f"({ratio:.0%}) cx={cx_range} cy={cy_range}"
                )
                if ratio == 1.0:
                    persistent_100.append((r, c))

        print(f"  Cells with >70% persistence: {len(persistent_70)}")
        for line in persistent_70:
            print(line)
        print(f"  Cells with 100% persistence (never empty): {len(persistent_100)}")

        # Overlay images: first 3 and last 3 frames
        sorted_fnames = sorted(labels.keys())
        picks = sorted_fnames[:3] + sorted_fnames[-3:]
        DEBUG_ANNOTATION.mkdir(parents=True, exist_ok=True)

        for i, fname in enumerate(picks):
            tag = "first" if i < 3 else "last"
            idx = i if i < 3 else i - 3
            img_name = fname.replace(".txt", ".jpg")
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                print(f"    WARNING: {img_path} not found, skipping overlay")
                continue

            img = cv2.imread(str(img_path))
            h_img, w_img = img.shape[:2]

            for cx, cy, w, h in labels[fname]:
                x1 = int((cx - w / 2) * w_img)
                y1 = int((cy - h / 2) * h_img)
                x2 = int((cx + w / 2) * w_img)
                y2 = int((cy + h / 2) * h_img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out_path = DEBUG_ANNOTATION / f"{prefix}_{tag}_{idx}.png"
            cv2.imwrite(str(out_path), img)

        frame_nums = [sorted_fnames[0], sorted_fnames[-1]]
        print(f"  Overlay images saved to {DEBUG_ANNOTATION}/")
        print(f"    Frames: {sorted_fnames[0]}..{sorted_fnames[2]} and {sorted_fnames[-3]}..{sorted_fnames[-1]}")


# ---------------------------------------------------------------------------
# Objective 3: False Positive Location Mapping
# ---------------------------------------------------------------------------

def objective3_false_positive_mapping(high_density_cells: dict[str, set[tuple[int, int]]]):
    """Run inference on test clip and cross-reference with training density."""
    print("\n" + "=" * 70)
    print("OBJECTIVE 3: False Positive Location Mapping")
    print("=" * 70)

    if not TEST_CLIP.exists():
        print(f"  ERROR: Test clip not found: {TEST_CLIP}")
        return
    if not MODEL_PATH.exists():
        print(f"  ERROR: Model not found: {MODEL_PATH}")
        return

    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    fp7_hot = high_density_cells.get("FP7oJQ", set())

    cap = cv2.VideoCapture(str(TEST_CLIP))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n  Test clip: {TEST_CLIP.name}")
    print(f"  Resolution: {frame_w}x{frame_h}, frames: {total_frames}")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  FP7oJQ high-density cells from Obj 1: {len(fp7_hot)}")

    sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    DEBUG_FP.mkdir(parents=True, exist_ok=True)

    all_detections = []
    high_conf_in_hot = 0
    high_conf_total = 0

    print(f"\n  {'Frame':>8s}  {'Total':>5s}  {'Conf>0.50':>9s}  {'In Hot Zone':>11s}")
    print(f"  {'-' * 8}  {'-' * 5}  {'-' * 9}  {'-' * 11}")

    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            print(f"  {fi:>8d}  FAILED TO READ")
            continue

        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        det_count = len(boxes)
        high_conf_count = 0
        hot_zone_count = 0

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cx_norm = ((x1 + x2) / 2) / frame_w
            cy_norm = ((y1 + y2) / 2) / frame_h
            r, c = to_grid(cx_norm, cy_norm)
            in_hot = (r, c) in fp7_hot

            # Draw all detections
            color = (0, 255, 0)  # green default
            if conf > 0.50:
                high_conf_count += 1
                high_conf_total += 1
                color = (0, 200, 255)  # orange for high-conf
                if in_hot:
                    hot_zone_count += 1
                    high_conf_in_hot += 1
                    color = (0, 0, 255)  # red for high-conf in hot zone

                all_detections.append({
                    "frame": int(fi),
                    "cx": round(cx_norm, 4),
                    "cy": round(cy_norm, 4),
                    "conf": round(conf, 3),
                    "grid": (r, c),
                    "in_hot_zone": in_hot,
                })

            ix1, iy1 = int(x1), int(y1)
            ix2, iy2 = int(x2), int(y2)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            label = f"{conf:.2f}"
            if in_hot and conf > 0.50:
                label += " HOT"
            cv2.putText(
                frame, label, (ix1, iy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        print(f"  {fi:>8d}  {det_count:>5d}  {high_conf_count:>9d}  {hot_zone_count:>11d}")

        out_path = DEBUG_FP / f"frame_{fi:06d}.png"
        cv2.imwrite(str(out_path), frame)

    cap.release()

    # Summary table of all conf>0.50 detections
    if all_detections:
        print(f"\n  All detections with conf > 0.50:")
        print(f"  {'Frame':>8s}  {'cx':>6s}  {'cy':>6s}  {'Conf':>6s}  {'Grid':>8s}  {'Hot?':>5s}")
        print(f"  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 5}")
        for d in all_detections:
            hot_flag = "YES" if d["in_hot_zone"] else "no"
            print(
                f"  {d['frame']:>8d}  {d['cx']:>6.4f}  {d['cy']:>6.4f}  "
                f"{d['conf']:>6.3f}  {str(d['grid']):>8s}  {hot_flag:>5s}"
            )

        pct = high_conf_in_hot / high_conf_total * 100 if high_conf_total else 0
        print(f"\n  Summary: {high_conf_in_hot}/{high_conf_total} conf>0.50 detections "
              f"({pct:.0f}%) fall in FP7oJQ high-density training zones")
    else:
        print("\n  No detections with conf > 0.50 found.")

    print(f"\n  Annotated frames saved to {DEBUG_FP}/")


# ---------------------------------------------------------------------------
# Objective 4: Per-Camera Coverage Summary
# ---------------------------------------------------------------------------

def objective4_coverage_summary():
    """Summarize annotation coverage per camera."""
    print("\n" + "=" * 70)
    print("OBJECTIVE 4: Per-Camera Annotation Count and Coverage Summary")
    print("=" * 70)

    rows = []
    for cam, prefix in PREFIXES.items():
        labels = load_labels(prefix)
        total_frames = len(labels)

        counts = []
        areas = []
        occupied_cells: set[tuple[int, int]] = set()
        empty_frames = 0

        for boxes in labels.values():
            counts.append(len(boxes))
            if len(boxes) == 0:
                empty_frames += 1
            for cx, cy, w, h in boxes:
                areas.append(w * h)
                occupied_cells.add(to_grid(cx, cy))

        total_ann = sum(counts)
        unique_cells = len(occupied_cells)
        density = total_ann / unique_cells if unique_cells else 0

        area_q = np.quantile(areas, [0.25, 0.50, 0.75]) if areas else [0, 0, 0]
        tiny_pct = sum(1 for a in areas if a < 0.005) / len(areas) * 100 if areas else 0

        rows.append({
            "cam": cam,
            "frames": total_frames,
            "annotations": total_ann,
            "ann_per_frame_mean": statistics.mean(counts),
            "ann_per_frame_std": statistics.stdev(counts) if len(counts) > 1 else 0,
            "unique_cells": unique_cells,
            "density": density,
            "empty_frames": empty_frames,
            "area_q25": area_q[0],
            "area_q50": area_q[1],
            "area_q75": area_q[2],
            "tiny_pct": tiny_pct,
        })

    # Print table
    print(f"\n  {'Camera':>8s}  {'Frames':>6s}  {'Annot':>6s}  {'Per-Frame':>10s}  "
          f"{'Cells':>5s}  {'Density':>7s}  {'Empty':>5s}  {'Tiny%':>6s}  "
          f"{'Area Q25':>8s}  {'Area Q50':>8s}  {'Area Q75':>8s}")
    print(f"  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 10}  "
          f"{'-' * 5}  {'-' * 7}  {'-' * 5}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 8}")

    for r in rows:
        pf = f"{r['ann_per_frame_mean']:.1f}+/-{r['ann_per_frame_std']:.1f}"
        print(
            f"  {r['cam']:>8s}  {r['frames']:>6d}  {r['annotations']:>6d}  {pf:>10s}  "
            f"{r['unique_cells']:>5d}  {r['density']:>7.1f}  {r['empty_frames']:>5d}  "
            f"{r['tiny_pct']:>5.1f}%  "
            f"{r['area_q25']:>8.5f}  {r['area_q50']:>8.5f}  {r['area_q75']:>8.5f}"
        )

    print("\n  Key observations:")
    print("  - 'Tiny%' = fraction of annotations with area < 0.005 (roughly <10x50 px)")
    print("  - 'Cells' = unique 10x10 grid cells occupied (out of 100)")
    print("  - 'Density' = annotations per occupied cell (higher = more concentrated)")
    print("  - 'Empty' = frames with zero annotations (background-only training signal)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FP7oJQ False Positive Investigation")
    print("=" * 70)

    high_density = objective1_spatial_distribution()
    objective2_temporal_persistence()
    objective3_false_positive_mapping(high_density)
    objective4_coverage_summary()

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    print("Debug images saved to:")
    print(f"  {DEBUG_ANNOTATION}/")
    print(f"  {DEBUG_FP}/")


if __name__ == "__main__":
    main()
