"""Timeline plot for GT-DIAG-1.

One row per GT track, x-axis = frame index.
Three bands per track: tracklet_id, d1_node_id, person_id.
Group spans crosshatched. Low-confidence tracks marked with *.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def _color_map(values: list) -> dict:
    """Assign distinct colours to a list of unique values."""
    unique = sorted(set(v for v in values if v is not None))
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    return {v: cmap(i) for i, v in enumerate(unique)}


def render_timeline(
    seq_df: pd.DataFrame,
    output_path: Path,
    total_frames: int = 1764,
) -> None:
    """Render the three-layer timeline plot."""
    gt_tracks = sorted(seq_df["gt_track_id"].unique())
    n_tracks = len(gt_tracks)

    # Collect all unique values for colour maps
    all_tracklets = seq_df["tracklet_id"].dropna().unique().tolist()
    all_persons = seq_df["person_id"].dropna().unique().tolist()

    tracklet_colors = _color_map(all_tracklets)
    person_colors = _color_map(all_persons)

    # Colour for nodes: SOLO = steel blue family, GROUP = orange family
    node_solo_color = "#4682B4"
    node_group_color = "#FF8C00"
    no_det_color = "#E0E0E0"

    fig, axes = plt.subplots(n_tracks, 1, figsize=(24, 2.0 * n_tracks),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    band_height = 0.25
    band_gap = 0.05

    for ax_idx, gt_id in enumerate(gt_tracks):
        ax = axes[ax_idx]
        gt_segs = seq_df[seq_df["gt_track_id"] == gt_id].sort_values("seg_index")

        meta = gt_segs.iloc[0]
        on_mat_str = "ON MAT" if meta["on_mat"] else "OFF MAT"
        low_conf_str = " *" if meta["low_confidence"] else ""
        label = f"GT {gt_id} ({on_mat_str}){low_conf_str}"
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")

        # Three bands: top=tracklet (y=0.6), mid=node (y=0.3), bot=person (y=0.0)
        band_positions = {"tracklet": 0.6, "node": 0.3, "person": 0.0}

        for _, seg in gt_segs.iterrows():
            x_start = seg["frame_start"]
            width = seg["n_frames"]

            tid = seg["tracklet_id"]
            nid = seg["d1_node_id"]
            pid = seg["person_id"]
            in_group = seg["in_group_span"]

            # Tracklet band
            tc = tracklet_colors.get(tid, no_det_color) if tid else no_det_color
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
            if pid:
                agrees = seg["agrees_with_canonical"]
                if agrees is True:
                    pc = "#2E8B57"  # sea green = correct
                elif agrees is False:
                    pc = "#DC143C"  # crimson = misattributed
                else:
                    pc = "#999999"  # grey = unknown
                ax.barh(band_positions["person"], width, left=x_start,
                        height=band_height, color=pc, edgecolor="none")
            else:
                ax.barh(band_positions["person"], width, left=x_start,
                        height=band_height, color=no_det_color, edgecolor="none")

        ax.set_xlim(0, total_frames)
        ax.set_ylim(-0.1, 1.0)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)

        # Band labels on right side
        for bname, by in band_positions.items():
            ax.text(total_frames + 10, by + band_height / 2, bname,
                    fontsize=6, va="center", ha="left", color="#666")

    axes[-1].set_xlabel("Frame index", fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(color=node_solo_color, label="SOLO node"),
        mpatches.Patch(facecolor=node_group_color, hatch="///",
                       edgecolor="grey", label="GROUP node"),
        mpatches.Patch(color="#2E8B57", label="Correct person_id"),
        mpatches.Patch(color="#DC143C", label="Misattributed person_id"),
        mpatches.Patch(color=no_det_color, label="No detection"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=5,
               fontsize=8, frameon=False)

    fig.suptitle("GT-DIAG-1: GT-to-pipeline sequence diagnostic\n"
                 "Three layers per GT track: tracklet (top), D1 node (mid), person_id (bot)\n"
                 "* = low confidence (coverage < 50%)",
                 fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
