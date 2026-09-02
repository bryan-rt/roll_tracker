"""Compact sequence view — chronological text summary per GT track.

Two paths side by side: CORRECT (what GT maps to) and ACTUAL (what the solver assigned).
Time as mm:ss from sidecar pts_time_s. Inline edge costs and population classification.

Collapse rule: segments shorter than MIN_SEGMENT_FRAMES are collapsed with a count.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bjj_pipeline.contracts.f0_sidecar import load_sidecar

MIN_SEGMENT_FRAMES = 3


def _frame_to_mmss(frame_index: int, sidecar) -> str:
    t = sidecar.pts_time_s(frame_index)
    mins = int(t // 60)
    secs = int(t % 60)
    return f"{mins:02d}:{secs:02d}"


def _collapse_short_segments(segs: list[dict]) -> list[dict]:
    """Collapse runs of consecutive short segments into a single summary entry."""
    result = []
    short_run = []

    def flush_short():
        if short_run:
            total_frames = sum(s["n_frames"] for s in short_run)
            result.append({
                "collapsed": True,
                "count": len(short_run),
                "n_frames": total_frames,
                "frame_start": short_run[0]["frame_start"],
                "frame_end": short_run[-1]["frame_end"],
            })
            short_run.clear()

    for seg in segs:
        if seg["n_frames"] < MIN_SEGMENT_FRAMES:
            short_run.append(seg)
        else:
            flush_short()
            seg["collapsed"] = False
            result.append(seg)

    flush_short()
    return result


def build_compact_view(
    seq_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    video_path: Path,
    output_path: Path,
) -> None:
    """Write the compact sequence view as markdown."""
    sidecar = load_sidecar(video_path)

    # Build edge lookup: (gt_track_id, boundary_frame) -> edge info
    edge_lookup: dict[tuple[int, int], dict] = {}
    if not edge_df.empty:
        for _, e in edge_df.iterrows():
            key = (int(e["gt_track_id"]), int(e["boundary_frame"]))
            edge_lookup[key] = e.to_dict()

    lines = []
    lines.append("# GT-DIAG-1: Compact Sequence View")
    lines.append("")
    lines.append(f"Collapse rule: segments shorter than {MIN_SEGMENT_FRAMES} frames are "
                 f"collapsed into [Nx short] with a count.")
    lines.append("Time: mm:ss from sidecar pts_time_s (clip start = 00:00).")
    lines.append("Edge populations: chosen_correct, chosen_wrong, available_not_chosen, no_edge_exists.")
    lines.append("")

    for gt_id in sorted(seq_df["gt_track_id"].unique()):
        gt_segs = seq_df[seq_df["gt_track_id"] == gt_id].sort_values("seg_index")
        meta = gt_segs.iloc[0]

        on_mat = "ON MAT" if meta["on_mat_blueprint"] else "OFF MAT"
        low = " *LOW-CONF" if meta["low_confidence"] else ""
        quad_pct = meta.get("in_quad_pct", "?")

        lines.append(f"## GT {gt_id}  [{on_mat}, {meta['gt_matched_frames']}/{1764} frames, "
                     f"{meta['coverage_clip_pct']}%, area={meta['median_box_area']}{low}]")
        lines.append(f"  Calibrated quad: {quad_pct}% in-quad")
        lines.append("")

        # Build segment list as dicts for collapse
        seg_list = []
        for _, seg in gt_segs.iterrows():
            seg_list.append(seg.to_dict())

        collapsed = _collapse_short_segments(seg_list)

        # CORRECT path (what GT maps to: tracklet + node)
        lines.append("  CORRECT path (tracklet + D1 node the GT boxes map to):")
        for entry in collapsed:
            if entry.get("collapsed"):
                t0 = _frame_to_mmss(entry["frame_start"], sidecar)
                t1 = _frame_to_mmss(entry["frame_end"], sidecar)
                lines.append(f"    {t0}-{t1}  [{entry['count']}x short, {entry['n_frames']}f]")
            else:
                t0 = _frame_to_mmss(int(entry["frame_start"]), sidecar)
                t1 = _frame_to_mmss(int(entry["frame_end"]), sidecar)
                tid = entry["tracklet_id"] or "—"
                nid = entry["d1_node_id"] or "—"
                seg_type = entry.get("d1_segment_type", "")
                group_tag = " [GROUP]" if entry.get("in_group_span") else ""
                purity = entry.get("tracklet_purity")
                pur_str = f" pur={purity:.2f}" if purity is not None else ""
                lines.append(f"    {t0}-{t1}  {tid}{pur_str}  {nid}{group_tag}  ({entry['n_frames']}f)")

        lines.append("")

        # ACTUAL path (person_id assignment + edge info at boundaries)
        lines.append("  ACTUAL path (solver assignment):")
        for i, entry in enumerate(collapsed):
            if entry.get("collapsed"):
                t0 = _frame_to_mmss(entry["frame_start"], sidecar)
                t1 = _frame_to_mmss(entry["frame_end"], sidecar)
                lines.append(f"    {t0}-{t1}  [{entry['count']}x short, {entry['n_frames']}f]")
            else:
                t0 = _frame_to_mmss(int(entry["frame_start"]), sidecar)
                t1 = _frame_to_mmss(int(entry["frame_end"]), sidecar)
                pid = entry.get("person_id") or "—"
                agrees = entry.get("agrees_with_canonical")
                if agrees is True:
                    tag = "correct"
                elif agrees is False:
                    tag = "WRONG"
                else:
                    tag = "?"
                mode = entry.get("failure_mode", "")
                lines.append(f"    {t0}-{t1}  {pid}  [{tag}]  {mode}  ({entry['n_frames']}f)")

        lines.append("")

        # Divergence summary — list boundaries
        lines.append("  Divergence points:")
        divergence_count = 0
        prev_entry = None
        for entry in collapsed:
            if entry.get("collapsed") or prev_entry is None:
                prev_entry = entry
                continue

            if prev_entry.get("collapsed"):
                prev_entry = entry
                continue

            prev_node = prev_entry.get("d1_node_id")
            curr_node = entry.get("d1_node_id")

            if prev_node and curr_node and prev_node != curr_node:
                bf = int(entry["frame_start"])
                edge_info = edge_lookup.get((int(gt_id), bf))
                if edge_info:
                    pop = edge_info.get("population", "?")
                    cost = edge_info.get("total_cost")
                    cost_str = f" cost={cost:.3f}" if cost is not None else ""
                    t = _frame_to_mmss(bf, sidecar)
                    lines.append(f"    {t} (frame {bf}): {prev_node} -> {curr_node}  [{pop}{cost_str}]")
                    divergence_count += 1

            prev_entry = entry

        if divergence_count == 0:
            lines.append("    (none)")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
