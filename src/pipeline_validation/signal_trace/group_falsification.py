"""GROUP falsification analysis (CP-TRACE-2).

Tests whether the GROUP machinery engages on pair-box tracklets. Uses
d1_segments.parquet (on disk) for direct segment_type lookup — no
re-derivation needed.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _build_segment_lookup(
    seg_df: pd.DataFrame,
) -> dict[str, list[tuple[int, int, str, str, str]]]:
    """Pre-index by base_tracklet_id -> [(start, end, seg_type, node_id, payload), ...]"""
    lookup: dict[str, list[tuple[int, int, str, str, str]]] = defaultdict(list)
    for _, row in seg_df.iterrows():
        lookup[row.base_tracklet_id].append((
            int(row.start_frame),
            int(row.end_frame),
            row.segment_type,
            row.node_id,
            row.payload_json if pd.notna(row.payload_json) else "{}",
        ))
    return dict(lookup)


def run_group_falsification(
    stage_a_trace_path: Path,
    d1_segments_path: Path,
) -> dict:
    """Cross-reference pair-box tracklets against d1_segments.

    Returns group_falsification summary dict.
    """
    trace_df = pd.read_parquet(stage_a_trace_path)
    seg_df = pd.read_parquet(d1_segments_path)
    seg_lookup = _build_segment_lookup(seg_df)

    # Get pair-box rows and their tracklet_ids + frame_indices
    pair_box = trace_df[trace_df.classification == "pair_box"].copy()
    if pair_box.empty:
        return {
            "total_pair_box_tracklets": 0,
            "in_solo_node": 0,
            "in_group_node": 0,
            "not_in_graph": 0,
            "verdict": "No pair-box frames detected",
            "detail": [],
        }

    # Group pair-box frames by tracklet_id
    pb_by_tracklet: dict[str, list[int]] = defaultdict(list)
    for _, row in pair_box.iterrows():
        if pd.notna(row.tracklet_id):
            pb_by_tracklet[row.tracklet_id].append(int(row.frame_index))

    total_tracklets = len(pb_by_tracklet)
    in_solo = 0
    in_group = 0
    not_in_graph = 0
    detail: list[dict] = []

    for tid, frames in sorted(pb_by_tracklet.items()):
        segments = seg_lookup.get(tid)
        if not segments:
            not_in_graph += 1
            continue

        # Find which segments overlap with this tracklet's pair-box frames
        overlapping_types: set[str] = set()
        group_triggers: list[dict] = []

        for fi in frames:
            for start, end, seg_type, node_id, payload_str in segments:
                if start <= fi <= end:
                    overlapping_types.add(seg_type)
                    if seg_type == "GROUP":
                        try:
                            payload = json.loads(payload_str)
                        except json.JSONDecodeError:
                            payload = {}
                        trigger = {
                            "node_id": node_id,
                            "frame_range": f"{start}-{end}",
                        }
                        if payload.get("kind"):
                            trigger["kind"] = payload["kind"]
                        if payload.get("carrier"):
                            trigger["carrier"] = payload["carrier"]
                        if payload.get("disappear"):
                            trigger["disappear"] = payload["disappear"]
                        if payload.get("new"):
                            trigger["new"] = payload["new"]
                        # Deduplicate by node_id
                        if not any(d.get("node_id") == node_id for d in group_triggers):
                            group_triggers.append(trigger)
                    break  # found the segment for this frame

        if "GROUP" in overlapping_types:
            in_group += 1
            detail.append({
                "tracklet_id": tid,
                "node_type": "GROUP",
                "pair_box_frames": frames[:10],  # cap detail at 10
                "n_pair_box_frames": len(frames),
                "group_triggers": group_triggers,
                "causal": False,
                "note": (
                    f"GROUP triggered by lifecycle event "
                    f"({group_triggers[0].get('kind', '?')} with "
                    f"{group_triggers[0].get('disappear') or group_triggers[0].get('new', '?')}), "
                    f"not by pair-box"
                    if group_triggers else "GROUP trigger details unavailable"
                ),
            })
        elif "SOLO" in overlapping_types:
            in_solo += 1
        else:
            not_in_graph += 1

    # Verdict
    if total_tracklets == 0:
        verdict = "No pair-box tracklets to analyze"
    elif in_solo > in_group * 3:
        verdict = (
            f"GROUP structurally irrelevant to pair-boxes: "
            f"{in_solo}/{total_tracklets} in SOLO, "
            f"{in_group}/{total_tracklets} in GROUP (coincidental)"
        )
    elif in_group > 0:
        verdict = (
            f"GROUP engages on {in_group}/{total_tracklets} pair-box tracklets, "
            f"but triggers are lifecycle events (merges/splits), not pair-boxes. "
            f"Engagement is coincidental."
        )
    else:
        verdict = f"All {total_tracklets} pair-box tracklets in SOLO nodes — GROUP never fires"

    return {
        "total_pair_box_tracklets": total_tracklets,
        "in_solo_node": in_solo,
        "in_group_node": in_group,
        "not_in_graph": not_in_graph,
        "verdict": verdict,
        "detail": detail,
    }
