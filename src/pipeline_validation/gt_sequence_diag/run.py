"""CLI runner for GT-DIAG-1 / GT-VERIFY-1.

Usage:
    PYTHONPATH=src python -m pipeline_validation.gt_sequence_diag.run \
        --trace outputs/_eval/stage_d/gt-eval-fp7oJQ-132650/FP7oJQ/gt_person_trace.parquet \
        --pipeline-dir outputs/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/FP7oJQ-20260822-132650 \
        --pfm outputs/_eval/stage_a/gt-eval-fp7oJQ-132650/FP7oJQ/per_frame_matches.parquet \
        --video data/raw/nest/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/FP7oJQ-20260822-132650.mp4 \
        --camera FP7oJQ \
        --output docs/evidence/gt_diag_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_validation.gt_sequence_diag.sequence import build_sequence_table
from pipeline_validation.gt_sequence_diag.edge_join import build_edge_analysis
from pipeline_validation.gt_sequence_diag.plot import render_timeline


def _load_projection(camera_id: str):
    from bjj_pipeline.contracts.f0_projection import load_calibration_from_payload, CameraProjection
    from bjj_pipeline.stages.orchestration.multiplex_runner import _homography_to_img_to_mat, _load_json
    cam_dir = Path("configs") / "cameras" / camera_id
    j = _load_json(cam_dir / "homography.json")
    H_raw = np.asarray(j.get("H", j.get("homography")), dtype=np.float64)
    cm, dc = load_calibration_from_payload(j)
    H = _homography_to_img_to_mat(H_raw, j)
    return CameraProjection(H=H, camera_matrix=cm, dist_coefficients=dc)


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-DIAG-1: GT-to-pipeline sequence diagnostic")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--pfm", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--camera", type=str, default="FP7oJQ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-frames", type=int, default=1764)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    nodes_path = args.pipeline_dir / "stage_D" / "d1_graph_nodes.parquet"
    detections_path = args.pipeline_dir / "stage_A" / "detections.parquet"
    edge_costs_path = args.pipeline_dir / "stage_D" / "d2_edge_costs.parquet"
    selected_edges_path = args.pipeline_dir / "_debug" / "d3_selected_edges.parquet"
    person_tracks_path = args.pipeline_dir / "stage_D" / "person_tracks.parquet"

    for p in [args.trace, nodes_path, detections_path, args.pfm, edge_costs_path, selected_edges_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    projection = _load_projection(args.camera)

    print("Building sequence table...")
    seq_df = build_sequence_table(
        trace_path=args.trace,
        nodes_path=nodes_path,
        detections_path=detections_path,
        pfm_path=args.pfm,
        total_clip_frames=args.total_frames,
        projection=projection,
    )

    seq_df.to_parquet(args.output / "gt_sequence_table.parquet", index=False)
    seq_df.to_csv(args.output / "gt_sequence_table.csv", index=False)
    print(f"  {len(seq_df)} segments across {seq_df['gt_track_id'].nunique()} GT tracks")

    print("Building edge-cost analysis...")
    edge_df = build_edge_analysis(seq_df, edge_costs_path, selected_edges_path)
    if not edge_df.empty:
        edge_df.to_csv(args.output / "edge_cost_analysis.csv", index=False)
        print(f"  {len(edge_df)} node boundaries analysed")
        pop_counts = edge_df["population"].value_counts()
        for pop, count in pop_counts.items():
            print(f"    {pop}: {count}")
    else:
        print("  No node boundaries found")

    print("Rendering timeline...")
    render_timeline(seq_df, args.output / "timeline.png", total_frames=args.total_frames)
    print(f"  Saved to {args.output / 'timeline.png'}")

    # Compact view
    if args.video and args.video.exists():
        print("Building compact view...")
        from pipeline_validation.gt_sequence_diag.compact_view import build_compact_view
        build_compact_view(seq_df, edge_df, args.video, args.output / "compact_view.md")
        print(f"  Saved to {args.output / 'compact_view.md'}")

    # Videos
    if args.video and args.video.exists() and not args.skip_videos:
        print("Rendering annotated_gt.mp4...")
        from pipeline_validation.gt_sequence_diag.render_videos import (
            render_annotated_gt, render_mat_view_gt,
        )
        render_annotated_gt(
            video_path=args.video,
            pfm_path=args.pfm,
            person_tracks_path=person_tracks_path,
            output_path=args.output / "annotated_gt.mp4",
        )
        print(f"  Saved to {args.output / 'annotated_gt.mp4'}")

        print("Rendering mat_view_gt.mp4...")
        render_mat_view_gt(
            video_path=args.video,
            pfm_path=args.pfm,
            person_tracks_path=person_tracks_path,
            output_path=args.output / "mat_view_gt.mp4",
            camera_id=args.camera,
        )
        print(f"  Saved to {args.output / 'mat_view_gt.mp4'}")

    # Summary
    print("\n=== Per-GT-track summary ===")
    for gt_id in sorted(seq_df["gt_track_id"].unique()):
        gt = seq_df[seq_df["gt_track_id"] == gt_id]
        meta = gt.iloc[0]
        n_segs = len(gt)
        n_tracklets = gt["tracklet_id"].dropna().nunique()
        n_persons = gt["person_id"].dropna().nunique()
        n_group = int(gt["in_group_span"].sum())
        on_mat_col = "on_mat_blueprint" if "on_mat_blueprint" in meta.index else "on_mat"
        on_mat = "ON MAT" if meta[on_mat_col] else "OFF MAT"
        low = " *LOW" if meta["low_confidence"] else ""
        quad = f" [{meta.get('in_quad_pct', '?')}% quad]" if meta.get("in_quad_pct") is not None else ""
        print(f"  GT {gt_id} ({on_mat}{quad}{low}): "
              f"{meta['gt_matched_frames']} matched/{meta['coverage_clip_pct']:.1f}% clip "
              f"({meta['coverage_presence_pct']:.1f}% presence), "
              f"area={meta['median_box_area']}, "
              f"{n_segs} segs, {n_tracklets} tracklets, {n_persons} persons, "
              f"{n_group} group segs")

    print("\nDone.")


if __name__ == "__main__":
    main()
