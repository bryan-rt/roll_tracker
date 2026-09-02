"""Edge-cost join at node-sequence boundaries.

For every boundary where the D1 node changes between consecutive segments of a
GT track, joins to d2_edge_costs.parquet and classifies into four populations:

  1. Chosen and correct — solver picked an edge GT agrees with
  2. Chosen and wrong — solver picked an edge GT disagrees with
  3. Available but not chosen, though correct — the right edge existed and lost
  4. No edge exists — the right connection was never a candidate (graph-construction)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


COST_TERM_COLS = [
    "term_env", "term_time", "term_vreq", "term_missing_geom",
    "term_flags", "term_group_coherence", "term_birth_prior",
    "term_death_prior", "term_merge_prior", "term_split_prior",
]

CONTEXT_COLS = [
    "dt_s", "dist_m", "v_req_mps", "is_allowed", "disallow_reasons_json",
    "total_cost",
]


def build_edge_analysis(
    seq_df: pd.DataFrame,
    edge_costs_path: Path,
    selected_edges_path: Path,
) -> pd.DataFrame:
    """Classify node-sequence boundaries into four populations."""
    edges = pd.read_parquet(edge_costs_path)
    selected = pd.read_parquet(selected_edges_path)
    selected_ids = set(selected["edge_id"])

    # Build edge lookup: (src_node_id, dst_node_id) -> list of edge rows
    edge_lookup: dict[tuple[str, str], list[dict]] = {}
    for _, e in edges.iterrows():
        key = (e["src_node_id"], e["dst_node_id"])
        edge_lookup.setdefault(key, []).append(e.to_dict())

    rows = []

    for gt_id in sorted(seq_df["gt_track_id"].unique()):
        gt_segs = seq_df[seq_df["gt_track_id"] == gt_id].sort_values("seg_index")
        seg_list = list(gt_segs.itertuples(index=False))

        for i in range(len(seg_list) - 1):
            curr = seg_list[i]
            nxt = seg_list[i + 1]

            curr_node = curr.d1_node_id
            nxt_node = nxt.d1_node_id

            # Skip if either side has no node (undetected segments)
            if curr_node is None or nxt_node is None:
                continue

            # Skip if same node (no boundary)
            if curr_node == nxt_node:
                continue

            # This is a node-sequence boundary
            boundary_frame = nxt.frame_start

            # Is the transition GT-correct? Same canonical person on both sides.
            curr_agrees = curr.agrees_with_canonical
            nxt_agrees = nxt.agrees_with_canonical

            # Look for the CHOSEN edge at this boundary
            # The solver may have chosen an edge from curr_node to nxt_node,
            # or from curr_node to some other node (wrong), or from some other
            # node to nxt_node.

            # Check if an edge curr_node -> nxt_node exists
            direct_edges = edge_lookup.get((curr_node, nxt_node), [])
            direct_selected = [e for e in direct_edges if e["edge_id"] in selected_ids]

            # Check what edge was actually chosen FROM curr_node
            chosen_from_curr = []
            for (src, dst), elist in edge_lookup.items():
                if src == curr_node:
                    for e in elist:
                        if e["edge_id"] in selected_ids:
                            chosen_from_curr.append(e)

            # Check what edge was actually chosen TO nxt_node
            chosen_to_nxt = []
            for (src, dst), elist in edge_lookup.items():
                if dst == nxt_node:
                    for e in elist:
                        if e["edge_id"] in selected_ids:
                            chosen_to_nxt.append(e)

            # Classify the boundary
            if direct_selected:
                # An edge curr->nxt was selected
                edge = direct_selected[0]
                gt_correct = bool(curr_agrees and nxt_agrees)
                pop = "chosen_correct" if gt_correct else "chosen_wrong"
                rows.append(_make_edge_row(
                    gt_id, boundary_frame, curr_node, nxt_node,
                    pop, edge, gt_correct, curr, nxt,
                ))
            else:
                # No direct edge was selected
                if direct_edges:
                    # Edge existed but was not chosen
                    # Find the best (lowest cost) available correct edge
                    best = min(direct_edges, key=lambda e: e["total_cost"])
                    rows.append(_make_edge_row(
                        gt_id, boundary_frame, curr_node, nxt_node,
                        "available_not_chosen", best, True, curr, nxt,
                    ))
                else:
                    # No edge exists between these nodes at all
                    rows.append(_make_edge_row(
                        gt_id, boundary_frame, curr_node, nxt_node,
                        "no_edge_exists", None, None, curr, nxt,
                    ))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _make_edge_row(
    gt_id: int,
    boundary_frame: int,
    src_node: str,
    dst_node: str,
    population: str,
    edge: dict | None,
    gt_correct: bool | None,
    curr_seg,
    nxt_seg,
) -> dict:
    row = {
        "gt_track_id": gt_id,
        "boundary_frame": boundary_frame,
        "src_node_id": src_node,
        "dst_node_id": dst_node,
        "population": population,
        "gt_correct": gt_correct,
        "curr_person_id": curr_seg.person_id,
        "nxt_person_id": nxt_seg.person_id,
        "curr_in_group": curr_seg.in_group_span,
        "nxt_in_group": nxt_seg.in_group_span,
    }

    if edge is not None:
        row["edge_id"] = edge["edge_id"]
        row["edge_type"] = edge.get("edge_type")
        row["is_allowed"] = edge.get("is_allowed")
        row["disallow_reasons"] = edge.get("disallow_reasons_json")
        row["total_cost"] = edge.get("total_cost")
        for col in COST_TERM_COLS:
            row[col] = edge.get(col)
        for col in ["dt_s", "dist_m", "v_req_mps"]:
            row[col] = edge.get(col)
    else:
        row["edge_id"] = None
        row["edge_type"] = None
        row["is_allowed"] = None
        row["disallow_reasons"] = None
        row["total_cost"] = None
        for col in COST_TERM_COLS:
            row[col] = None
        for col in ["dt_s", "dist_m", "v_req_mps"]:
            row[col] = None

    return row
