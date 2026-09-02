"""CLI runner for GT-DIAG-1.

Usage:
    PYTHONPATH=src python -m pipeline_validation.gt_sequence_diag.run \
        --trace outputs/_eval/stage_d/gt-eval-fp7oJQ-132650/FP7oJQ/gt_person_trace.parquet \
        --pipeline-dir outputs/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/FP7oJQ-20260822-132650 \
        --pfm outputs/_eval/stage_a/gt-eval-fp7oJQ-132650/FP7oJQ/per_frame_matches.parquet \
        --output docs/evidence/gt_diag_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_validation.gt_sequence_diag.sequence import build_sequence_table
from pipeline_validation.gt_sequence_diag.edge_join import build_edge_analysis
from pipeline_validation.gt_sequence_diag.plot import render_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-DIAG-1: GT-to-pipeline sequence diagnostic")
    parser.add_argument("--trace", type=Path, required=True,
                        help="Path to gt_person_trace.parquet")
    parser.add_argument("--pipeline-dir", type=Path, required=True,
                        help="Path to clip pipeline output directory")
    parser.add_argument("--pfm", type=Path, required=True,
                        help="Path to per_frame_matches.parquet")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for evidence")
    parser.add_argument("--total-frames", type=int, default=1764,
                        help="Total frames in clip")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    nodes_path = args.pipeline_dir / "stage_D" / "d1_graph_nodes.parquet"
    detections_path = args.pipeline_dir / "stage_A" / "detections.parquet"
    edge_costs_path = args.pipeline_dir / "stage_D" / "d2_edge_costs.parquet"
    selected_edges_path = args.pipeline_dir / "_debug" / "d3_selected_edges.parquet"

    for p in [args.trace, nodes_path, detections_path, args.pfm, edge_costs_path, selected_edges_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    print("Building sequence table...")
    seq_df = build_sequence_table(
        trace_path=args.trace,
        nodes_path=nodes_path,
        detections_path=detections_path,
        pfm_path=args.pfm,
        total_clip_frames=args.total_frames,
    )

    seq_df.to_parquet(args.output / "gt_sequence_table.parquet", index=False)
    seq_df.to_csv(args.output / "gt_sequence_table.csv", index=False)
    print(f"  {len(seq_df)} segments across {seq_df['gt_track_id'].nunique()} GT tracks")

    print("Building edge-cost analysis...")
    edge_df = build_edge_analysis(seq_df, edge_costs_path, selected_edges_path)
    if not edge_df.empty:
        edge_df.to_csv(args.output / "edge_cost_analysis.csv", index=False)
        print(f"  {len(edge_df)} node boundaries analysed")

        # Population summary
        pop_counts = edge_df["population"].value_counts()
        print("  Population counts:")
        for pop, count in pop_counts.items():
            print(f"    {pop}: {count}")

        # Gate vs cost for chosen_wrong
        wrong = edge_df[edge_df["population"] == "chosen_wrong"]
        if not wrong.empty:
            gate_fail = wrong[wrong["is_allowed"] == False]
            cost_fail = wrong[wrong["is_allowed"] == True]
            print(f"  chosen_wrong breakdown: {len(gate_fail)} gate failures, {len(cost_fail)} cost failures")
    else:
        print("  No node boundaries found")

    print("Rendering timeline...")
    render_timeline(seq_df, args.output / "timeline.png", total_frames=args.total_frames)
    print(f"  Saved to {args.output / 'timeline.png'}")

    # Print summary stats
    print("\n=== Per-GT-track summary ===")
    for gt_id in sorted(seq_df["gt_track_id"].unique()):
        gt = seq_df[seq_df["gt_track_id"] == gt_id]
        meta = gt.iloc[0]
        n_segs = len(gt)
        n_tracklets = gt["tracklet_id"].dropna().nunique()
        n_persons = gt["person_id"].dropna().nunique()
        n_group = int(gt["in_group_span"].sum())
        on_mat = "ON MAT" if meta["on_mat"] else "OFF MAT"
        low = " *LOW" if meta["low_confidence"] else ""
        print(f"  GT {gt_id} ({on_mat}{low}): "
              f"{meta['gt_matched_frames']} matched/{meta['coverage_clip_pct']:.1f}% clip "
              f"({meta['coverage_presence_pct']:.1f}% presence), "
              f"area={meta['median_box_area']}, "
              f"{n_segs} segs, {n_tracklets} tracklets, {n_persons} persons, "
              f"{n_group} group segs")

    print("\nDone.")


if __name__ == "__main__":
    main()
