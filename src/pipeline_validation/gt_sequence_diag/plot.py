"""Timeline plot for GT-DIAG-1.

One row per GT track, x-axis = frame index.
Three bands per track: tracklet purity (top), D1 node (mid), person_id (bot).
Group spans crosshatched. Low-confidence tracks marked with *.

Tracklet band uses purity-semantic colours (not per-id):
  Green:  pure (purity >= 0.9)
  Orange: impure (0.5 <= purity < 0.9)
  Red:    heavily impure (purity < 0.5)
  Grey:   no detection
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def _purity_color(purity: float | None) -> str:
    if purity is None:
        return "#E0E0E0"
    if purity >= 0.9:
        return "#2E8B57"  # sea green — pure
    if purity >= 0.5:
        return "#FF8C00"  # dark orange — impure
    return "#DC143C"      # crimson — heavily impure


def render_timeline(
    seq_df: pd.DataFrame,
    output_path: Path,
    total_frames: int = 1764,
) -> None:
    """Render the three-layer timeline plot."""
    gt_tracks = sorted(seq_df["gt_track_id"].unique())
    n_tracks = len(gt_tracks)

    node_solo_color = "#4682B4"
    node_group_color = "#FF8C00"
    no_det_color = "#E0E0E0"

    fig, axes = plt.subplots(n_tracks, 1, figsize=(24, 2.0 * n_tracks),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    band_height = 0.25

    for ax_idx, gt_id in enumerate(gt_tracks):
        ax = axes[ax_idx]
        gt_segs = seq_df[seq_df["gt_track_id"] == gt_id].sort_values("seg_index")

        meta = gt_segs.iloc[0]
        on_mat_str = "ON MAT" if meta["on_mat"] else "OFF MAT"
        low_conf_str = " *" if meta["low_confidence"] else ""
        in_quad = meta.get("in_quad_pct")
        quad_str = f" [{in_quad}% quad]" if in_quad is not None else ""
        label = f"GT {gt_id} ({on_mat_str}{quad_str}){low_conf_str}"
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")

        band_positions = {"tracklet": 0.6, "node": 0.3, "person": 0.0}

        for _, seg in gt_segs.iterrows():
            x_start = seg["frame_start"]
            width = seg["n_frames"]

            tid = seg["tracklet_id"]
            nid = seg["d1_node_id"]
            in_group = seg["in_group_span"]
            purity = seg["tracklet_purity"]

            # Tracklet band — purity-semantic colour
            tc = _purity_color(purity) if tid else no_det_color
            ax.barh(band_positions["tracklet"], width, left=x_start,
                    height=band_height, color=tc, edgecolor="none", linewidth=0)

            # Node band
            if nid:
                nc = node_group_color if in_group else node_solo_color
                hatch = "///" if in_group else None
                ax.barh(band_positions["node"], width, left=x_start,
                        height=band_height, color=nc, edgecolor="grey",
                        linewidth=0.3, hatch=hatch, alpha=0.8)
            else:
                ax.barh(band_positions["node"], width, left=x_start,
                        height=band_height, color=no_det_color, edgecolor="none")

            # Person band
            pid = seg["person_id"]
            if pid:
                agrees = seg["agrees_with_canonical"]
                if agrees is True:
                    pc = "#2E8B57"
                elif agrees is False:
                    pc = "#DC143C"
                else:
                    pc = "#999999"
                ax.barh(band_positions["person"], width, left=x_start,
                        height=band_height, color=pc, edgecolor="none")
            else:
                ax.barh(band_positions["person"], width, left=x_start,
                        height=band_height, color=no_det_color, edgecolor="none")

        ax.set_xlim(0, total_frames)
        ax.set_ylim(-0.1, 1.0)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)

        for bname, by in band_positions.items():
            ax.text(total_frames + 10, by + band_height / 2, bname,
                    fontsize=6, va="center", ha="left", color="#666")

    axes[-1].set_xlabel("Frame index", fontsize=9)

    legend_patches = [
        mpatches.Patch(color="#2E8B57", label="Tracklet pure (>=0.9)"),
        mpatches.Patch(color="#FF8C00", label="Tracklet impure (0.5-0.9)"),
        mpatches.Patch(color="#DC143C", label="Tracklet heavily impure (<0.5)"),
        mpatches.Patch(color=node_solo_color, label="SOLO node"),
        mpatches.Patch(facecolor=node_group_color, hatch="///",
                       edgecolor="grey", label="GROUP node"),
        mpatches.Patch(color="#2E8B57", label="Correct person_id"),
        mpatches.Patch(color="#DC143C", label="Misattributed person_id"),
        mpatches.Patch(color=no_det_color, label="No detection"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4,
               fontsize=8, frameon=False)

    fig.suptitle("GT-DIAG-1: GT-to-pipeline sequence diagnostic\n"
                 "Three layers per GT track: tracklet purity (top), D1 node (mid), person_id (bot)\n"
                 "* = low confidence (coverage < 50%)",
                 fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
