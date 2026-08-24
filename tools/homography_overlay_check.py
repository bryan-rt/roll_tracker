#!/usr/bin/env python3
"""Homography validation overlay — forward-projected metre grid on raw frames.

Calls project_to_world() from the production path. Contains no reimplemented
projection, undistortion, or homography math, and no inversion.
Does not read projected_polylines.

Usage:
    PYTHONPATH=src python tools/homography_overlay_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bjj_pipeline.stages.orchestration.multiplex_runner import _load_homography_matrix
from bjj_pipeline.contracts.f0_projection import project_to_world


# ---------------------------------------------------------------------------
# World bounds — from the mat blueprint extent, NOT from projected data.
# ---------------------------------------------------------------------------
X_LEVELS = list(range(42, 59))   # integer metre x contours
Y_LEVELS = list(range(34, 59))   # integer metre y contours
X_BOUNDS = (41.0, 59.0)          # mask: discard projections outside these
Y_BOUNDS = (33.0, 60.0)


def _load_quad_from_h_file(h_path: Path) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Read calibrated quad extents from a specific homography file."""
    with open(h_path) as f:
        d = json.load(f)
    mp = d.get("correspondences", {}).get("mat_points", [])
    if not mp:
        raise ValueError(f"No correspondences.mat_points in {h_path}")
    xs = [pt[0] for pt in mp]
    ys = [pt[1] for pt in mp]
    return (min(xs), max(xs)), (min(ys), max(ys))


def render_overlay(
    mp4_path: Path,
    proj,
    out_path: Path,
    label: str,
    quad_x: Tuple[float, float],
    quad_y: Tuple[float, float],
    frame_index: int = 100,
    grid_step: int = 6,
) -> dict:
    """Render metre-grid contours on a raw frame using project_to_world forward.

    Returns dict with grid stats (total samples, masked count).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {mp4_path}")

    h_px, w_px = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Subsample pixel grid and project forward
    us = np.arange(0, w_px, grid_step, dtype=float)
    vs = np.arange(0, h_px, grid_step, dtype=float)
    x_grid = np.full((len(vs), len(us)), np.nan)
    y_grid = np.full((len(vs), len(us)), np.nan)

    total_samples = len(vs) * len(us)
    for vi, v in enumerate(vs):
        for ui, u in enumerate(us):
            x_m, y_m = project_to_world(
                (float(u), float(v)),
                proj.H,
                camera_matrix=proj.camera_matrix,
                dist_coefficients=proj.dist_coefficients,
            )
            x_grid[vi, ui] = x_m
            y_grid[vi, ui] = y_m

    # Mask out-of-bounds and NaN (vanishing line, degenerate homography)
    oob = (
        np.isnan(x_grid) | np.isnan(y_grid)
        | (x_grid < X_BOUNDS[0]) | (x_grid > X_BOUNDS[1])
        | (y_grid < Y_BOUNDS[0]) | (y_grid > Y_BOUNDS[1])
    )
    masked_count = int(np.sum(oob))
    x_grid[oob] = np.nan
    y_grid[oob] = np.nan

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)
    ax.imshow(frame_rgb, extent=[0, w_px, h_px, 0], aspect="auto")

    # x_m contours (cyan) — general grid
    cs_x = ax.contour(us, vs, x_grid, levels=X_LEVELS, colors="cyan", linewidths=0.5, alpha=0.6)
    ax.clabel(cs_x, inline=True, fontsize=6, fmt="x=%.0f")

    # y_m contours (yellow) — general grid
    cs_y = ax.contour(us, vs, y_grid, levels=Y_LEVELS, colors="yellow", linewidths=0.5, alpha=0.6)
    ax.clabel(cs_y, inline=True, fontsize=6, fmt="y=%.0f")

    # Calibrated quad contours — thicker, distinct colour
    ax.contour(us, vs, x_grid, levels=[quad_x[0], quad_x[1]], colors="red", linewidths=2.0)
    ax.contour(us, vs, y_grid, levels=[quad_y[0], quad_y[1]], colors="red", linewidths=2.0)

    ax.set_title(f"{mp4_path.stem} — {label}\nframe_index={frame_index}, "
                 f"grid_step={grid_step}, masked={masked_count}/{total_samples}", fontsize=10)
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return {"total_samples": total_samples, "masked": masked_count}


def main():
    """Example: compare two H versions on Aug 22 footage.

    The default H and a prior version are loaded through the production
    resolver. Each version's calibrated quad is read from its own file.
    """
    base = Path("data/raw/nest/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13")
    out_dir = Path("docs/evidence/homography_validate_1")
    segments = ["130229", "131129", "132650"]
    camera_id = "FP7oJQ"

    # Version (a): default resolution
    default_h_path = Path("configs/cameras") / camera_id / "homography.json"
    proj_default = _load_homography_matrix({}, camera_id)
    quad_default = _load_quad_from_h_file(default_h_path)

    # Version (b): prior H via homography_path override — caller must
    # extract it to a scratch path first (e.g. git show <ref>:... > /tmp/...)
    prior_h_path = Path("/tmp/homography_old_2edce38.json")
    if prior_h_path.exists():
        proj_prior = _load_homography_matrix({"homography_path": str(prior_h_path)}, camera_id)
        quad_prior = _load_quad_from_h_file(prior_h_path)
    else:
        proj_prior = None
        quad_prior = None
        print(f"NOTE: {prior_h_path} not found — skipping prior-H overlays")

    for seg in segments:
        mp4 = base / f"FP7oJQ-20260822-{seg}.mp4"
        if not mp4.exists():
            print(f"SKIP {seg}: {mp4} not found")
            continue

        # Default H
        stats_a = render_overlay(
            mp4, proj_default,
            out_dir / f"FP7oJQ-20260822-{seg}_new.png",
            label=f"default ({default_h_path.name})",
            quad_x=quad_default[0], quad_y=quad_default[1],
        )
        print(f"{seg} default: masked={stats_a['masked']}/{stats_a['total_samples']}")

        # Prior H
        if proj_prior is not None:
            stats_b = render_overlay(
                mp4, proj_prior,
                out_dir / f"FP7oJQ-20260822-{seg}_old.png",
                label=f"prior ({prior_h_path.name})",
                quad_x=quad_prior[0], quad_y=quad_prior[1],
            )
            print(f"{seg} prior: masked={stats_b['masked']}/{stats_b['total_samples']}")


if __name__ == "__main__":
    main()
