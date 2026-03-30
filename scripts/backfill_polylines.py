"""Backfill projected_polylines into existing homography.json files.

One-off utility. Generates dense projected polylines from the existing H
(mat->img) and blueprint, then merges into homography.json.

Usage:
    python scripts/backfill_polylines.py \
        --cameras FP7oJQ J_EDEw PPDmUg \
        --configs-root configs \
        --blueprint configs/mat_blueprint.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def _extract_contiguous_runs(items: list) -> list:
    runs = []
    current = []
    for item in items:
        if item is not None:
            current.append(item)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def generate_projected_polylines(
    H_mat_to_img: np.ndarray,
    rects: list[tuple],
    image_wh: tuple[int, int],
    sample_spacing: float = 0.25,
    frame_margin: float = 50.0,
) -> dict:
    width, height = image_wh

    # All unique panel edges
    edges_set: set = set()
    for (x, y, w, h, _) in rects:
        corners = [(float(x), float(y)), (float(x + w), float(y)),
                   (float(x + w), float(y + h)), (float(x), float(y + h))]
        for i in range(4):
            c1, c2 = corners[i], corners[(i + 1) % 4]
            normalized = tuple(sorted([c1, c2]))
            edges_set.add(normalized)
    all_edges = [((e[0][0], e[0][1]), (e[1][0], e[1][1])) for e in edges_set]

    polylines = []
    H64 = np.asarray(H_mat_to_img, dtype=np.float64)

    for edge_idx, ((wx1, wy1), (wx2, wy2)) in enumerate(all_edges):
        edge_len = math.sqrt((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2)
        n_samples = max(2, int(math.ceil(edge_len / sample_spacing)))

        projected_points: list = []
        for k in range(n_samples):
            t = k / max(1, n_samples - 1)
            wx = wx1 + t * (wx2 - wx1)
            wy = wy1 + t * (wy2 - wy1)

            world_pt = np.array([[[wx, wy]]], dtype=np.float32)
            pixel_pt = cv2.perspectiveTransform(world_pt, H64)
            u, v = float(pixel_pt[0, 0, 0]), float(pixel_pt[0, 0, 1])

            if (math.isfinite(u) and math.isfinite(v)
                    and -frame_margin <= u <= width + frame_margin
                    and -frame_margin <= v <= height + frame_margin):
                projected_points.append([round(u, 1), round(v, 1)])
            else:
                projected_points.append(None)

        for run in _extract_contiguous_runs(projected_points):
            polylines.append({
                "edge_index": edge_idx,
                "world_start": [wx1, wy1],
                "world_end": [wx2, wy2],
                "pixel_points": run,
            })

    return {
        "image_wh": [width, height],
        "sample_spacing": sample_spacing,
        "polylines": polylines,
        "n_polylines": len(polylines),
        "n_edges_total": len(all_edges),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill projected_polylines into homography.json")
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--blueprint", type=Path, default=Path("configs/mat_blueprint.json"))
    args = parser.parse_args()

    # Load blueprint panels
    with open(args.blueprint) as f:
        blueprint_data = json.load(f)
    rects = [
        (r["x"], r["y"], r["width"], r["height"], r.get("label", ""))
        for r in blueprint_data
        if isinstance(r, dict) and all(k in r for k in ("x", "y", "width", "height"))
    ]
    print(f"Blueprint: {len(rects)} panels")

    for cam in args.cameras:
        path = args.configs_root / "cameras" / cam / "homography.json"
        if not path.exists():
            print(f"{cam}: homography.json not found — skipping")
            continue

        with open(path) as f:
            payload = json.load(f)

        H = np.asarray(payload["H"], dtype=np.float64).reshape((3, 3))

        # Get image dimensions
        image_size = payload.get("lens_calibration", {}).get("image_size")
        if image_size is None:
            print(f"{cam}: no lens_calibration.image_size — skipping")
            continue
        image_wh = (int(image_size[0]), int(image_size[1]))

        # Generate polylines
        polyline_data = generate_projected_polylines(
            H_mat_to_img=H, rects=rects, image_wh=image_wh,
        )

        # Merge into existing payload
        payload["projected_polylines"] = polyline_data

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"{cam}: {polyline_data['n_polylines']} polylines from "
              f"{polyline_data['n_edges_total']} edges ({image_wh[0]}x{image_wh[1]})")


if __name__ == "__main__":
    main()
