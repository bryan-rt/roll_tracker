"""REACH-1: Graph reachability analysis for GT persons.

For each GT person, determines whether a connected path exists through
the D1 graph that visits all their tracklets and yields one identity.

Classifies each required hop into:
  EDGE_EXISTS_SELECTED     — edge present and chosen by D3
  EDGE_EXISTS_NOT_SELECTED — edge present, solver chose otherwise
  EDGE_ABSENT_IN_WINDOW    — no edge, but gap/speed within D1 limits
  UNREACHABLE_BY_WINDOW    — no edge, gap or speed exceeds D1 limits
  SHARED_NODE              — node needed by multiple GT people (capacity analysis)

Runs against both production and dedup-ceiling artifacts.

Output: docs/evidence/reach_1/
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

PROD_STAGE_D = (
    ROOT / "outputs" / "00000000-0000-0000-0000-000000000003"
    / "FP7oJQ" / "2026-08-22" / "13" / "FP7oJQ-20260822-132650" / "stage_D"
)
PROD_GPT = (
    ROOT / "outputs" / "_eval" / "stage_d"
    / "gt-eval-fp7oJQ-132650" / "FP7oJQ" / "gt_person_trace.parquet"
)

DEDUP_BASE = ROOT / "outputs" / "_dedup_ceiling" / "FP7oJQ-20260822-132650"
DEDUP_STAGE_D = DEDUP_BASE / "stage_D"
DEDUP_PFM = (
    ROOT / "docs" / "evidence" / "dedup_ceiling_1"
    / "gt_matching" / "per_frame_matches.parquet"
)

OUT_DIR = ROOT / "docs" / "evidence" / "reach_1"

# ── D1 parameters (from configs/default.yaml) ─────────────────────────────

RECONNECT_MAX_GAP_FRAMES = 250
V_MAX_MPS = 8.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_graph(stage_d: Path) -> dict:
    """Load D1 nodes, D2 edge costs, person_spans, tracklet_bank_frames."""
    nodes = pd.read_parquet(stage_d / "d1_graph_nodes.parquet")
    edges = pd.read_parquet(stage_d / "d2_edge_costs.parquet")
    spans = pd.read_parquet(stage_d / "person_spans.parquet")
    tbf = pd.read_parquet(stage_d / "tracklet_bank_frames.parquet")
    return dict(nodes=nodes, edges=edges, spans=spans, tbf=tbf)


def _build_edge_lookup(edges: pd.DataFrame) -> dict[tuple[str, str], list[dict]]:
    """(src_node_id, dst_node_id) -> list of edge rows."""
    lookup: dict[tuple[str, str], list[dict]] = {}
    for _, e in edges.iterrows():
        key = (e["src_node_id"], e["dst_node_id"])
        lookup.setdefault(key, []).append(e.to_dict())
    return lookup


def _build_selected_pairs(spans: pd.DataFrame) -> set[tuple[str, str]]:
    """Set of (node_A, node_B) pairs that the solver selected.

    Inferred from person_spans: consecutive nodes on the same person_id path
    imply the solver selected the connecting edge. This is an inference from
    spans, not a recorded edge selection — person_spans records which nodes
    are on which person path but not which edges connect them.
    """
    pairs: set[tuple[str, str]] = set()
    for pid in spans["person_id"].unique():
        path = spans[spans["person_id"] == pid].sort_values("start_frame")
        node_ids = path["node_id"].tolist()
        for i in range(len(node_ids) - 1):
            pairs.add((node_ids[i], node_ids[i + 1]))
    return pairs


def _build_node_map(nodes_df: pd.DataFrame) -> dict[str, dict]:
    """node_id -> {capacity, segment_type, start_frame, end_frame, base_tracklet_id}."""
    nm: dict[str, dict] = {}
    for _, r in nodes_df.iterrows():
        nm[r["node_id"]] = {
            "capacity": int(r["capacity"]),
            "segment_type": r.get("segment_type", "UNKNOWN"),
            "start_frame": int(r["start_frame"]) if pd.notna(r["start_frame"]) else None,
            "end_frame": int(r["end_frame"]) if pd.notna(r["end_frame"]) else None,
            "base_tracklet_id": r.get("base_tracklet_id"),
        }
    return nm


def _build_d3_flow(spans: pd.DataFrame) -> dict[str, list[str]]:
    """node_id -> list of person_ids routed through it."""
    flow: dict[str, list[str]] = defaultdict(list)
    for _, r in spans.iterrows():
        flow[r["node_id"]].append(r["person_id"])
    return dict(flow)


def _get_endpoint_positions(
    tbf: pd.DataFrame,
    node_map: dict[str, dict],
    node_id: str,
    end: str,  # "start" or "end"
) -> tuple[float, float] | None:
    """Get (x_m, y_m) at the start or end of a node's base tracklet."""
    info = node_map.get(node_id)
    if not info or not info["base_tracklet_id"]:
        return None
    tid = info["base_tracklet_id"]
    frame = info["start_frame"] if end == "start" else info["end_frame"]
    if frame is None:
        return None
    row = tbf[(tbf["tracklet_id"] == tid) & (tbf["frame_index"] == frame)]
    if row.empty:
        # Try nearest frame
        t_rows = tbf[tbf["tracklet_id"] == tid].copy()
        if t_rows.empty:
            return None
        t_rows["_dist"] = (t_rows["frame_index"] - frame).abs()
        row = t_rows.nsmallest(1, "_dist")
    x = row.iloc[0].get("x_m")
    y = row.iloc[0].get("y_m")
    if pd.isna(x) or pd.isna(y):
        return None
    return (float(x), float(y))


# ── Node-frame occupancy (frame-level, for shared-node and joint checks) ──

def _build_node_frame_occupancy_from_gpt(
    gpt: pd.DataFrame,
) -> dict[str, dict[int, set[int]]]:
    """node_id -> {gt_person_id -> set of frame indices}."""
    occ: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for _, row in gpt.iterrows():
        if pd.isna(row["d1_node_ids"]):
            continue
        nodes = json.loads(row["d1_node_ids"])
        gid = int(row["gt_person_id"])
        frame = int(row["frame_idx"])
        for n in nodes:
            occ[n][gid].add(frame)
    return dict(occ)


def _build_node_frame_occupancy_from_dedup(
    matched: pd.DataFrame,
) -> dict[str, dict[int, set[int]]]:
    """node_id -> {gt_person_id -> set of frame indices} from dedup join."""
    occ: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for _, row in matched.iterrows():
        node = row.get("node_id")
        if node is None or (isinstance(node, float) and math.isnan(node)):
            continue
        gid = int(row["gt_track_id"])
        frame = int(row["frame_index"])
        occ[node][gid].add(frame)
    return dict(occ)


# ── Target path building ──────────────────────────────────────────────────

def _build_target_paths_from_gpt(gpt: pd.DataFrame) -> dict[int, list[dict]]:
    """Build target paths from gt_person_trace.parquet.

    Returns: {gt_person_id: [{node_id, first_frame, last_frame}, ...]}
    Each entry is a contiguous run of the same node, None gaps removed.
    """
    paths: dict[int, list[dict]] = {}
    for gid in sorted(gpt["gt_person_id"].unique()):
        sub = gpt[gpt["gt_person_id"] == gid].sort_values("frame_idx")
        runs: list[dict] = []
        current_node = None
        current_start = None

        for _, row in sub.iterrows():
            nids_raw = row["d1_node_ids"]
            if pd.isna(nids_raw):
                node = None
            else:
                parsed = json.loads(nids_raw)
                node = parsed[0] if parsed else None

            if node is None:
                if current_node is not None:
                    runs.append({"node_id": current_node, "first_frame": current_start,
                                 "last_frame": int(row["frame_idx"]) - 1})
                    current_node = None
                continue

            if node == current_node:
                continue  # extend current run
            else:
                if current_node is not None:
                    runs.append({"node_id": current_node, "first_frame": current_start,
                                 "last_frame": int(row["frame_idx"]) - 1})
                current_node = node
                current_start = int(row["frame_idx"])

        # Close last run
        if current_node is not None:
            last_frame = int(sub["frame_idx"].iloc[-1])
            runs.append({"node_id": current_node, "first_frame": current_start,
                         "last_frame": last_frame})

        paths[gid] = runs
    return paths


def _find_node_from_segs(
    d1_segments: pd.DataFrame, tid: str, frame: int,
) -> str | None:
    """Look up D1 node_id for (tracklet_id, frame_index) from d1_segments."""
    for _, s in d1_segments.iterrows():
        if s["base_tracklet_id"] == tid and int(s["start_frame"]) <= frame <= int(s["end_frame"]):
            return s["node_id"]
    return None


def _build_target_paths_from_dedup(
    pfm: pd.DataFrame,
    detections: pd.DataFrame,
    d1_segments: pd.DataFrame,
) -> dict[int, list[dict]]:
    """Build target paths for dedup-ceiling by joining GT→detection→tracklet→node."""
    # Step 1: GT match → detection_id
    matched = pfm[pfm["match_status"] == "matched"].copy()
    matched = matched[["frame_index", "gt_track_id", "pred_detection_id"]].rename(
        columns={"pred_detection_id": "detection_id"}
    )

    # Step 2: detection_id → tracklet_id
    det_tid = detections[["detection_id", "tracklet_id"]].drop_duplicates()
    matched = matched.merge(det_tid, on="detection_id", how="left")

    # Step 3: tracklet_id + frame_index → d1 node_id
    # Build a lookup from (base_tracklet_id, frame_index) → node_id
    seg_lookup: list[tuple[str, int, int, str]] = []
    for _, s in d1_segments.iterrows():
        seg_lookup.append((
            s["base_tracklet_id"],
            int(s["start_frame"]),
            int(s["end_frame"]),
            s["node_id"],
        ))

    def _find_node(tid: str, frame: int) -> str | None:
        for btid, sf, ef, nid in seg_lookup:
            if btid == tid and sf <= frame <= ef:
                return nid
        return None

    matched["node_id"] = matched.apply(
        lambda r: _find_node(r["tracklet_id"], r["frame_index"])
        if pd.notna(r["tracklet_id"]) else None,
        axis=1,
    )

    # Step 4: Build contiguous runs per GT
    paths: dict[int, list[dict]] = {}
    for gid in sorted(matched["gt_track_id"].unique()):
        sub = matched[matched["gt_track_id"] == gid].sort_values("frame_index")
        runs: list[dict] = []
        current_node = None
        current_start = None

        for _, row in sub.iterrows():
            node = row["node_id"]
            frame = int(row["frame_index"])

            if node is None or (isinstance(node, float) and math.isnan(node)):
                if current_node is not None:
                    runs.append({"node_id": current_node, "first_frame": current_start,
                                 "last_frame": frame - 1})
                    current_node = None
                continue

            if node == current_node:
                continue
            else:
                if current_node is not None:
                    runs.append({"node_id": current_node, "first_frame": current_start,
                                 "last_frame": frame - 1})
                current_node = node
                current_start = frame

        if current_node is not None:
            last_frame = int(sub["frame_index"].iloc[-1])
            runs.append({"node_id": current_node, "first_frame": current_start,
                         "last_frame": last_frame})

        paths[int(gid)] = runs
    return paths


# ── Detection gap stats ───────────────────────────────────────────────────

def _detection_stats(gpt: pd.DataFrame) -> dict[int, dict]:
    """Per GT person: detected frames, total frames, gap-separated hop ratio."""
    stats: dict[int, dict] = {}
    for gid in sorted(gpt["gt_person_id"].unique()):
        sub = gpt[gpt["gt_person_id"] == gid].sort_values("frame_idx")
        total = len(sub)
        detected = sub["d1_node_ids"].notna().sum()
        stats[gid] = {
            "total_frames": int(total),
            "detected_frames": int(detected),
            "detection_rate": round(detected / total, 4) if total > 0 else 0,
        }
    return stats


# ── Hop classification ────────────────────────────────────────────────────

def _classify_hops(
    paths: dict[int, list[dict]],
    edge_lookup: dict[tuple[str, str], list[dict]],
    selected_pairs: set[tuple[str, str]],
    node_map: dict[str, dict],
    d3_flow: dict[str, list[str]],
    tbf: pd.DataFrame,
) -> list[dict]:
    """Classify every hop in every GT person's target path."""
    hop_rows: list[dict] = []

    for gid, runs in sorted(paths.items()):
        for i in range(len(runs) - 1):
            a = runs[i]
            b = runs[i + 1]
            node_a = a["node_id"]
            node_b = b["node_id"]

            row = {
                "gt_person_id": gid,
                "hop_index": i,
                "src_node": node_a,
                "dst_node": node_b,
                "src_last_frame": a["last_frame"],
                "dst_first_frame": b["first_frame"],
            }

            # Frame gap between nodes
            a_info = node_map.get(node_a, {})
            b_info = node_map.get(node_b, {})
            a_end = a_info.get("end_frame", a["last_frame"])
            b_start = b_info.get("start_frame", b["first_frame"])
            frame_gap = b_start - a_end if (a_end is not None and b_start is not None) else None
            row["frame_gap"] = frame_gap

            # Is there a detection gap between runs?
            row["has_detection_gap"] = (b["first_frame"] - a["last_frame"]) > 1

            # Check edge
            direct_edges = edge_lookup.get((node_a, node_b), [])

            if direct_edges:
                is_selected = (node_a, node_b) in selected_pairs
                if is_selected:
                    row["outcome"] = "EDGE_EXISTS_SELECTED"
                else:
                    row["outcome"] = "EDGE_EXISTS_NOT_SELECTED"
                    # Capacity check on endpoints
                    a_cap = a_info.get("capacity", 1)
                    b_cap = b_info.get("capacity", 1)
                    a_flow = len(d3_flow.get(node_a, []))
                    b_flow = len(d3_flow.get(node_b, []))
                    row["src_capacity"] = a_cap
                    row["dst_capacity"] = b_cap
                    row["src_flow"] = a_flow
                    row["dst_flow"] = b_flow
                    row["capacity_blocked"] = (a_flow >= a_cap) or (b_flow >= b_cap)

                best = min(direct_edges, key=lambda e: e.get("total_cost", 999))
                row["edge_id"] = best.get("edge_id")
                row["is_allowed"] = best.get("is_allowed")
                row["total_cost"] = best.get("total_cost")
                row["dist_m"] = best.get("dist_m")
                row["v_req_mps"] = best.get("v_req_mps")
                row["dt_s"] = best.get("dt_s")
            else:
                # No edge — compute gap metrics
                pos_a = _get_endpoint_positions(tbf, node_map, node_a, "end")
                pos_b = _get_endpoint_positions(tbf, node_map, node_b, "start")

                dist_m = None
                if pos_a and pos_b:
                    dist_m = math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                    row["dist_m"] = round(dist_m, 3)

                # Implied speed
                implied_speed = None
                if dist_m is not None and frame_gap is not None and frame_gap > 0:
                    # Approximate time: frame_gap * nominal_dt (~67ms at 15fps)
                    dt_approx = frame_gap * 0.067
                    implied_speed = dist_m / dt_approx if dt_approx > 0 else None
                    row["implied_speed_mps"] = round(implied_speed, 2) if implied_speed else None
                    row["dt_approx_s"] = round(dt_approx, 3)

                # Classify: concurrent vs in-window vs unreachable
                # Negative or zero frame gap means nodes overlap in time —
                # D1 edges are temporal transitions and cannot connect
                # simultaneous nodes. This is concurrent-node flicker from
                # detection under-segmentation, not a candidate generation gap.
                if frame_gap is not None and frame_gap <= 0:
                    row["outcome"] = "CONCURRENT_NODES"
                else:
                    gap_exceeds = frame_gap is not None and frame_gap > RECONNECT_MAX_GAP_FRAMES
                    speed_exceeds = implied_speed is not None and implied_speed > V_MAX_MPS

                    if gap_exceeds or speed_exceeds:
                        row["outcome"] = "UNREACHABLE_BY_WINDOW"
                        if gap_exceeds:
                            row["gap_margin"] = frame_gap - RECONNECT_MAX_GAP_FRAMES
                        if speed_exceeds:
                            row["speed_margin"] = round(implied_speed - V_MAX_MPS, 2)
                    else:
                        row["outcome"] = "EDGE_ABSENT_IN_WINDOW"

            hop_rows.append(row)

    return hop_rows


# ── Shared node analysis ──────────────────────────────────────────────────

def _shared_node_analysis(
    node_frame_occ: dict[str, dict[int, set[int]]],
    node_map: dict[str, dict],
    d3_flow: dict[str, list[str]],
) -> list[dict]:
    """Identify nodes where multiple GT people co-occupy at the same frame.

    Uses frame-level occupancy, not run-level range envelopes. Two GT people
    on the same node at different frames is sequential use, not contention.
    Contention only exists when two GT people need the same node at the same
    frame AND the node's capacity cannot accommodate both.

    With Hungarian matching (IoU 0.5), two GT people never match the same
    detection at the same frame. On a SOLO (capacity-1) node representing
    one detection box, the matcher assigns it to one GT person per frame.
    Co-occupancy at the frame level requires two distinct detections both
    mapping to the same node — possible only on GROUP nodes.
    """
    shared_rows: list[dict] = []
    for node_id, gt_frames in sorted(node_frame_occ.items()):
        gt_ids = sorted(gt_frames.keys())
        if len(gt_ids) < 2:
            continue

        info = node_map.get(node_id, {})
        cap = info.get("capacity", 1)
        seg_type = info.get("segment_type", "UNKNOWN")
        d3_persons = d3_flow.get(node_id, [])

        for g1, g2 in combinations(gt_ids, 2):
            co_frames = gt_frames[g1] & gt_frames[g2]
            if not co_frames:
                # Both use this node but never at the same frame — sequential, not contention
                shared_rows.append({
                    "node_id": node_id,
                    "gt_person_a": g1,
                    "gt_person_b": g2,
                    "co_occupied_frames": 0,
                    "gt_a_frame_count": len(gt_frames[g1]),
                    "gt_b_frame_count": len(gt_frames[g2]),
                    "capacity": cap,
                    "segment_type": seg_type,
                    "d3_persons_routed": list(set(d3_persons)),
                    "d3_flow_count": len(set(d3_persons)),
                    "contention": "sequential",
                })
                continue

            # Co-occupied frames exist — check capacity
            contention = "none"
            if len(co_frames) > 0 and cap < 2:
                contention = "structural_impossibility"
            elif len(co_frames) > 0 and cap >= 2:
                contention = "group_handles_it"

            shared_rows.append({
                "node_id": node_id,
                "gt_person_a": g1,
                "gt_person_b": g2,
                "co_occupied_frames": len(co_frames),
                "gt_a_frame_count": len(gt_frames[g1]),
                "gt_b_frame_count": len(gt_frames[g2]),
                "capacity": cap,
                "segment_type": seg_type,
                "d3_persons_routed": list(set(d3_persons)),
                "d3_flow_count": len(set(d3_persons)),
                "contention": contention,
            })

    return shared_rows


# ── Joint feasibility ─────────────────────────────────────────────────────

def _joint_feasibility(
    node_frame_occ: dict[str, dict[int, set[int]]],
    node_map: dict[str, dict],
    gt_ids: list[int],
) -> dict:
    """Determine maximum number of GT people whose paths can coexist.

    Uses frame-level occupancy: for each node, at each frame, count how many
    GT people from the subset occupy it. If that count exceeds the node's
    capacity at any frame, the subset is infeasible.

    With Hungarian matching, two GT people never match the same detection at
    the same frame. On SOLO nodes (one detection box), frame-level
    co-occupancy is zero by construction. Contention at the frame level
    requires two detections both mapping to the same node — only possible
    on GROUP nodes where it is handled by capacity=2.

    Method: exact search (2^8 = 256 subsets).
    """
    n = len(gt_ids)

    def _check_subset(subset: list[int]) -> bool:
        subset_set = set(subset)
        for node_id, gt_frames in node_frame_occ.items():
            relevant = {g: frames for g, frames in gt_frames.items() if g in subset_set}
            if len(relevant) <= 1:
                continue
            cap = node_map.get(node_id, {}).get("capacity", 1)
            # Find maximum simultaneous occupancy at any frame
            all_frames: set[int] = set()
            for frames in relevant.values():
                all_frames |= frames
            for frame in all_frames:
                count = sum(1 for g, frames in relevant.items() if frame in frames)
                if count > cap:
                    return False
        return True

    # Check full set first
    if _check_subset(gt_ids):
        return {
            "all_feasible": True,
            "max_coexisting": n,
            "best_subset": gt_ids,
            "blocking_nodes": [],
        }

    # Find maximum feasible subset by decreasing size
    from itertools import combinations as combs
    best_size = 0
    best_subset: list[int] = []
    for size in range(n - 1, 0, -1):
        found = False
        for subset in combs(gt_ids, size):
            if _check_subset(list(subset)):
                best_size = size
                best_subset = list(subset)
                found = True
                break
        if found:
            break

    # Identify blocking nodes (nodes where full set exceeds capacity)
    blocking = []
    for node_id, gt_frames in node_frame_occ.items():
        relevant = {g: frames for g, frames in gt_frames.items() if g in set(gt_ids)}
        if len(relevant) <= 1:
            continue
        cap = node_map.get(node_id, {}).get("capacity", 1)
        all_frames: set[int] = set()
        for frames in relevant.values():
            all_frames |= frames
        max_sim = 0
        for frame in all_frames:
            count = sum(1 for g, frames in relevant.items() if frame in frames)
            max_sim = max(max_sim, count)
        if max_sim > cap:
            blocking.append({
                "node_id": node_id,
                "capacity": cap,
                "max_simultaneous_gt": max_sim,
                "gt_people": sorted(relevant.keys()),
                "segment_type": node_map.get(node_id, {}).get("segment_type", "UNKNOWN"),
            })

    return {
        "all_feasible": False,
        "max_coexisting": best_size,
        "best_subset": best_subset,
        "excluded": [g for g in gt_ids if g not in best_subset],
        "blocking_nodes": blocking,
    }


# ── Verdicts ──────────────────────────────────────────────────────────────

def _independent_reachability(
    paths: dict[int, list[dict]],
    hop_rows: list[dict],
) -> dict[int, dict]:
    """Per GT person: is a fully connected path available (ignoring contention)?"""
    verdicts: dict[int, dict] = {}
    for gid in sorted(paths.keys()):
        hops = [h for h in hop_rows if h["gt_person_id"] == gid]
        n_hops = len(hops)
        n_path_nodes = len(paths[gid])

        if n_hops == 0:
            verdicts[gid] = {
                "reachable": True,
                "n_path_nodes": n_path_nodes,
                "n_hops": 0,
                "reason": "single node or no hops needed",
            }
            continue

        edge_exists = all(
            h["outcome"] in ("EDGE_EXISTS_SELECTED", "EDGE_EXISTS_NOT_SELECTED")
            for h in hops
        )
        breaks = [h for h in hops if h["outcome"] in (
            "EDGE_ABSENT_IN_WINDOW", "UNREACHABLE_BY_WINDOW", "CONCURRENT_NODES",
        )]

        verdicts[gid] = {
            "reachable": edge_exists,
            "n_path_nodes": n_path_nodes,
            "n_hops": n_hops,
            "n_edge_exists_selected": sum(1 for h in hops if h["outcome"] == "EDGE_EXISTS_SELECTED"),
            "n_edge_exists_not_selected": sum(1 for h in hops if h["outcome"] == "EDGE_EXISTS_NOT_SELECTED"),
            "n_concurrent": sum(1 for h in hops if h["outcome"] == "CONCURRENT_NODES"),
            "n_absent_in_window": sum(1 for h in hops if h["outcome"] == "EDGE_ABSENT_IN_WINDOW"),
            "n_unreachable": sum(1 for h in hops if h["outcome"] == "UNREACHABLE_BY_WINDOW"),
            "breaks": [
                {"hop": h["hop_index"], "src": h["src_node"], "dst": h["dst_node"],
                 "outcome": h["outcome"], "frame_gap": h.get("frame_gap")}
                for h in breaks
            ],
        }
    return verdicts


def _aggregate_by_owner(hop_rows: list[dict]) -> dict[str, int]:
    """Count hops in each outcome class."""
    counts: dict[str, int] = defaultdict(int)
    for h in hop_rows:
        counts[h["outcome"]] += 1
    return dict(counts)


# ── Reporting ─────────────────────────────────────────────────────────────

def _write_findings(
    mode: str,
    paths: dict[int, list[dict]],
    hop_rows: list[dict],
    shared_rows: list[dict],
    detection_stats: dict[int, dict],
    indep_verdicts: dict[int, dict],
    joint: dict,
    agg: dict[str, int],
) -> str:
    """Generate findings markdown."""
    lines: list[str] = []
    w = lines.append

    w(f"# REACH-1: Graph Reachability Analysis — {mode.upper()}")
    w("")
    w("## Method")
    w("")
    w("For each of the 8 GT persons on FP7oJQ-20260822-132650, this analysis:")
    w("1. Builds the **target path** — the ordered sequence of D1 nodes the GT person")
    w("   occupies, with detection gaps removed and contiguous runs collapsed.")
    w("2. Classifies each **hop** (consecutive node pair) in the target path.")
    w("3. Identifies **shared nodes** — D1 nodes needed by multiple GT people simultaneously.")
    w("4. Checks **independent reachability** — can each GT person's path be walked ignoring others?")
    w("5. Solves **joint feasibility** — maximum GT people whose paths coexist given capacity.")
    w("")
    w("**Selected edges are inferred from `person_spans.parquet`:** consecutive nodes on the")
    w("same person_id path imply the solver selected the connecting edge. This is an inference")
    w("from the D4 output, not a recorded edge-selection artifact.")
    w("")
    w(f"**D1 parameters:** `reconnect_max_gap_frames` = {RECONNECT_MAX_GAP_FRAMES}, "
      f"`v_max_mps` = {V_MAX_MPS}")
    w("")

    # Detection gap stats
    w("## 1. Detection Coverage per GT Person")
    w("")
    w("| GT | Total frames | Detected | Rate | Path nodes | Hops | Gap-separated hops |")
    w("|---|---|---|---|---|---|---|")
    for gid in sorted(paths.keys()):
        ds = detection_stats.get(gid, {})
        n_nodes = len(paths[gid])
        hops = [h for h in hop_rows if h["gt_person_id"] == gid]
        n_hops = len(hops)
        gap_hops = sum(1 for h in hops if h.get("has_detection_gap"))
        w(f"| {gid} | {ds.get('total_frames', '?')} | {ds.get('detected_frames', '?')} "
          f"| {ds.get('detection_rate', 0):.1%} | {n_nodes} | {n_hops} | {gap_hops} ({gap_hops}/{n_hops} hops) |")
    w("")

    # Hop classification
    w("## 2. Hop Classification")
    w("")
    w("| Outcome | Count | % |")
    w("|---|---|---|")
    total_hops = len(hop_rows)
    for outcome in ["EDGE_EXISTS_SELECTED", "EDGE_EXISTS_NOT_SELECTED",
                     "CONCURRENT_NODES", "EDGE_ABSENT_IN_WINDOW",
                     "UNREACHABLE_BY_WINDOW"]:
        c = agg.get(outcome, 0)
        pct = c / total_hops * 100 if total_hops > 0 else 0
        w(f"| {outcome} | {c} | {pct:.1f}% |")
    w(f"| **Total** | **{total_hops}** | |")
    w("")

    # CONCURRENT_NODES details
    concurrent = [h for h in hop_rows if h["outcome"] == "CONCURRENT_NODES"]
    if concurrent:
        w("### CONCURRENT_NODES — overlapping nodes, no temporal edge possible")
        w("")
        w("These hops are between D1 nodes whose frame ranges overlap (frame_gap <= 0).")
        w("D1 edges represent temporal transitions; they cannot connect simultaneous nodes.")
        w("This is the NOEDGE-1 finding: concurrent-node flicker from detection under-segmentation.")
        w("")
        # Summarize by GT person rather than listing all (can be very long)
        by_gt: dict[int, int] = defaultdict(int)
        for h in concurrent:
            by_gt[h["gt_person_id"]] += 1
        w("| GT | Concurrent hops | Example src→dst |")
        w("|---|---|---|")
        for gid in sorted(by_gt.keys()):
            example = next(h for h in concurrent if h["gt_person_id"] == gid)
            w(f"| {gid} | {by_gt[gid]} | `{example['src_node'][:35]}` → `{example['dst_node'][:35]}` |")
        w("")

    # EDGE_ABSENT_IN_WINDOW details
    absent_in = [h for h in hop_rows if h["outcome"] == "EDGE_ABSENT_IN_WINDOW"]
    if absent_in:
        w("### EDGE_ABSENT_IN_WINDOW — D1 should have generated these")
        w("")
        w("| GT | Hop | Src node | Dst node | Frame gap | Dist (m) | Speed (m/s) |")
        w("|---|---|---|---|---|---|---|")
        for h in absent_in:
            w(f"| {h['gt_person_id']} | {h['hop_index']} | `{h['src_node'][:40]}` "
              f"| `{h['dst_node'][:40]}` | {h.get('frame_gap', '?')} "
              f"| {h.get('dist_m', '?')} | {h.get('implied_speed_mps', '?')} |")
        w("")

    # UNREACHABLE_BY_WINDOW details
    unreach = [h for h in hop_rows if h["outcome"] == "UNREACHABLE_BY_WINDOW"]
    if unreach:
        w("### UNREACHABLE_BY_WINDOW — correctly excluded by D1 limits")
        w("")
        w("| GT | Hop | Frame gap | Gap margin | Speed (m/s) | Speed margin |")
        w("|---|---|---|---|---|---|")
        for h in unreach:
            w(f"| {h['gt_person_id']} | {h['hop_index']} | {h.get('frame_gap', '?')} "
              f"| +{h.get('gap_margin', '?')} | {h.get('implied_speed_mps', '?')} "
              f"| +{h.get('speed_margin', '?')} |")
        w("")

    # EDGE_EXISTS_NOT_SELECTED details
    not_sel = [h for h in hop_rows if h["outcome"] == "EDGE_EXISTS_NOT_SELECTED"]
    if not_sel:
        w("### EDGE_EXISTS_NOT_SELECTED — edge available, solver chose otherwise")
        w("")
        w("| GT | Hop | Src node | Dst node | Cost | Capacity blocked? |")
        w("|---|---|---|---|---|---|")
        for h in not_sel:
            w(f"| {h['gt_person_id']} | {h['hop_index']} | `{h['src_node'][:40]}` "
              f"| `{h['dst_node'][:40]}` | {h.get('total_cost', '?')} "
              f"| {h.get('capacity_blocked', '?')} |")
        w("")
        cap_blocked = sum(1 for h in not_sel if h.get("capacity_blocked"))
        cost_beaten = len(not_sel) - cap_blocked
        w(f"**Capacity-blocked:** {cap_blocked} | **Cost-beaten:** {cost_beaten}")
        w("")

    # Shared nodes
    w("## 3. Shared Node Analysis")
    w("")
    if not shared_rows:
        w("No shared nodes found.")
    else:
        # Classify by contention type
        structural = [s for s in shared_rows if s["contention"] == "structural_impossibility"]
        group_ok = [s for s in shared_rows if s["contention"] == "group_handles_it"]
        sequential = [s for s in shared_rows if s["contention"] == "sequential"]

        w(f"**Frame-level co-occupancy (structural impossibility):** {len(structural)}")
        w(f"**Frame-level co-occupancy (GROUP handles it):** {len(group_ok)}")
        w(f"**Sequential use (same node, different frames — no contention):** {len(sequential)}")
        w("")

        if structural:
            w("### Structural impossibility — co-occupied SOLO nodes")
            w("")
            w("| Node | GT A | GT B | Co-frames | Capacity | Seg type |")
            w("|---|---|---|---|---|---|")
            for s in structural:
                w(f"| `{s['node_id'][:50]}` | {s['gt_person_a']} | {s['gt_person_b']} "
                  f"| {s['co_occupied_frames']} | {s['capacity']} | {s['segment_type']} |")
            w("")

        if group_ok:
            w("### GROUP nodes correctly serving co-occupied GT people")
            w("")
            w(f"{len(group_ok)} GROUP nodes (capacity >= 2) where two GT people co-occupy at the same frame.")
            w("This is correct behavior — GROUP nodes exist to represent two people on one tracklet.")
            w("")

        if sequential:
            w("### Sequential use — same node, interleaved frames, no capacity conflict")
            w("")
            w(f"{len(sequential)} node-pairs where two GT people use the same node at different frames.")
            w("With Hungarian matching, two GT people never match the same detection at the same frame.")
            w("A capacity-1 SOLO node can serve both people sequentially — one gets correct attribution")
            w("per frame, the other gets misattribution. This is not a structural impossibility;")
            w("it is the detection under-segmentation problem expressed as misattribution, not as")
            w("a graph capacity limit.")
            w("")
            # Summary table
            w("| Node | GT A (frames) | GT B (frames) | Capacity | Seg type |")
            w("|---|---|---|---|---|")
            for s in sequential:
                w(f"| `{s['node_id'][:45]}` | {s['gt_person_a']} ({s['gt_a_frame_count']}f) "
                  f"| {s['gt_person_b']} ({s['gt_b_frame_count']}f) "
                  f"| {s['capacity']} | {s['segment_type']} |")
            w("")

    # Independent reachability
    w("## 4a. Independent Reachability (ignoring contention)")
    w("")
    w("| GT | Reachable? | Path nodes | Hops | Selected | Not selected | Concurrent | Absent | Unreachable |")
    w("|---|---|---|---|---|---|---|---|---|")
    n_indep_reachable = 0
    for gid in sorted(indep_verdicts.keys()):
        v = indep_verdicts[gid]
        tag = "YES" if v["reachable"] else "NO"
        if v["reachable"]:
            n_indep_reachable += 1
        w(f"| {gid} | {tag} | {v['n_path_nodes']} | {v['n_hops']} "
          f"| {v.get('n_edge_exists_selected', 0)} | {v.get('n_edge_exists_not_selected', 0)} "
          f"| {v.get('n_concurrent', 0)} | {v.get('n_absent_in_window', 0)} "
          f"| {v.get('n_unreachable', 0)} |")
    w("")
    w(f"**Independent reachability: {n_indep_reachable} / {len(indep_verdicts)} GT people**")
    w("")

    # Joint feasibility
    w("## 4b. Joint Feasibility (respecting node capacities)")
    w("")
    w(f"**Method:** Exhaustive search over all 2^{len(paths)} = {2**len(paths)} subsets.")
    w(f"For each subset, verify that every shared node has capacity >= number of GT people needing it simultaneously.")
    w("")
    if joint["all_feasible"]:
        w(f"**Result: ALL {len(paths)} GT people can coexist.** No capacity contention.")
    else:
        w(f"**Result: Maximum {joint['max_coexisting']} of {len(paths)} GT people can coexist.**")
        w(f"- Best feasible subset: GT {joint['best_subset']}")
        w(f"- Excluded: GT {joint['excluded']}")
        w("")
        if joint["blocking_nodes"]:
            w("**Blocking nodes (capacity < simultaneous GT demand):**")
            w("")
            w("| Node | Capacity | Seg type | Max simultaneous GT | GT people |")
            w("|---|---|---|---|---|")
            for bn in joint["blocking_nodes"]:
                w(f"| `{bn['node_id'][:50]}` | {bn['capacity']} | {bn['segment_type']} "
                  f"| {bn['max_simultaneous_gt']} | {bn['gt_people']} |")
            w("")
    w("")

    # Aggregate
    w("## 5. Aggregate by Owner")
    w("")
    w("| Owner | Category | Count |")
    w("|---|---|---|")
    w(f"| Working correctly | EDGE_EXISTS_SELECTED | {agg.get('EDGE_EXISTS_SELECTED', 0)} |")
    w(f"| D2 cost / D3 solve | EDGE_EXISTS_NOT_SELECTED | {agg.get('EDGE_EXISTS_NOT_SELECTED', 0)} |")
    not_sel_cap = sum(1 for h in hop_rows if h["outcome"] == "EDGE_EXISTS_NOT_SELECTED" and h.get("capacity_blocked"))
    not_sel_cost = agg.get("EDGE_EXISTS_NOT_SELECTED", 0) - not_sel_cap
    w(f"|   of which: capacity-blocked | | {not_sel_cap} |")
    w(f"|   of which: cost-beaten | | {not_sel_cost} |")
    w(f"| Detection (concurrent nodes) | CONCURRENT_NODES | {agg.get('CONCURRENT_NODES', 0)} |")
    w(f"| D1 candidate generation | EDGE_ABSENT_IN_WINDOW | {agg.get('EDGE_ABSENT_IN_WINDOW', 0)} |")
    w(f"| D1 parameters / detection | UNREACHABLE_BY_WINDOW | {agg.get('UNREACHABLE_BY_WINDOW', 0)} |")
    structural_count = len([s for s in shared_rows if s["contention"] == "structural_impossibility"])
    sequential_count = len([s for s in shared_rows if s["contention"] == "sequential"])
    w(f"| Detection (co-occupied SOLO) | SHARED_NODE structural impossibility | {structural_count} node-pairs |")
    w(f"| Detection (sequential use) | Same node, interleaved frames | {sequential_count} node-pairs |")
    w("")

    # Summary verdict
    w("## Summary Verdict")
    w("")
    w(f"- **Independent reachability (a):** {n_indep_reachable} / {len(paths)} GT people have a connected path")
    jmax = joint["max_coexisting"] if not joint["all_feasible"] else len(paths)
    w(f"- **Joint feasibility (b):** {jmax} / {len(paths)} GT people can coexist given capacity")
    w("")
    if n_indep_reachable == len(paths) and joint["all_feasible"]:
        w("**Finding:** Every GT person has a connected path AND all can coexist.")
        w("The ceiling is not connectivity or capacity — it is D2 cost / D3 solve decisions.")
    elif n_indep_reachable < len(paths):
        w(f"**Finding:** {len(paths) - n_indep_reachable} GT people lack connected paths.")
        w("The ceiling includes connectivity (edge generation), not just solver decisions.")
    if not joint["all_feasible"]:
        w(f"**Finding:** Contention limits joint feasibility to {jmax}.")
        w("Under-segmentation propagates into the graph — one detection covering two grapplers")
        w("becomes one SOLO node covering two people. No stitching or cost work can fix this.")
    w("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def run_production() -> tuple[str, dict]:
    """Run analysis on production artifacts."""
    print("=== PRODUCTION ===")
    graph = _load_graph(PROD_STAGE_D)
    gpt = pd.read_parquet(PROD_GPT)

    paths = _build_target_paths_from_gpt(gpt)
    det_stats = _detection_stats(gpt)

    edge_lookup = _build_edge_lookup(graph["edges"])
    selected_pairs = _build_selected_pairs(graph["spans"])
    node_map = _build_node_map(graph["nodes"])
    d3_flow = _build_d3_flow(graph["spans"])

    node_frame_occ = _build_node_frame_occupancy_from_gpt(gpt)

    hop_rows = _classify_hops(paths, edge_lookup, selected_pairs, node_map, d3_flow, graph["tbf"])
    shared_rows = _shared_node_analysis(node_frame_occ, node_map, d3_flow)
    indep = _independent_reachability(paths, hop_rows)
    joint = _joint_feasibility(node_frame_occ, node_map, sorted(paths.keys()))
    agg = _aggregate_by_owner(hop_rows)

    md = _write_findings("production", paths, hop_rows, shared_rows, det_stats, indep, joint, agg)

    summary = {
        "mode": "production",
        "n_gt_people": len(paths),
        "total_hops": len(hop_rows),
        "aggregate": agg,
        "independent_reachability": {str(k): v["reachable"] for k, v in indep.items()},
        "n_independent_reachable": sum(1 for v in indep.values() if v["reachable"]),
        "joint_feasibility": {
            "all_feasible": joint["all_feasible"],
            "max_coexisting": int(joint["max_coexisting"]) if not joint["all_feasible"] else len(paths),
        },
        "shared_structural": len([s for s in shared_rows if s["contention"] == "structural_impossibility"]),
        "shared_group_ok": len([s for s in shared_rows if s["contention"] == "group_handles_it"]),
        "shared_sequential": len([s for s in shared_rows if s["contention"] == "sequential"]),
    }

    return md, summary


def run_dedup() -> tuple[str, dict]:
    """Run analysis on dedup-ceiling artifacts."""
    print("=== DEDUP-CEILING ===")
    graph = _load_graph(DEDUP_STAGE_D)

    # Build target paths from per_frame_matches + dedup detections + d1_segments
    pfm = pd.read_parquet(DEDUP_PFM)
    det = pd.read_parquet(DEDUP_BASE / "stage_A" / "detections.parquet")
    d1_segs = pd.read_parquet(DEDUP_STAGE_D / "d1_segments.parquet")

    paths = _build_target_paths_from_dedup(pfm, det, d1_segs)

    # Detection stats from pfm
    det_stats: dict[int, dict] = {}
    for gid in sorted(pfm["gt_track_id"].dropna().unique()):
        sub = pfm[pfm["gt_track_id"] == gid]
        total = len(sub)
        detected = (sub["match_status"] == "matched").sum()
        det_stats[int(gid)] = {
            "total_frames": int(total),
            "detected_frames": int(detected),
            "detection_rate": round(detected / total, 4) if total > 0 else 0,
        }

    edge_lookup = _build_edge_lookup(graph["edges"])
    selected_pairs = _build_selected_pairs(graph["spans"])
    node_map = _build_node_map(graph["nodes"])
    d3_flow = _build_d3_flow(graph["spans"])

    # Build node-frame occupancy from the matched join (before path builder drops frame info)
    # Re-join to get node_id per matched row
    matched = pfm[pfm["match_status"] == "matched"].copy()
    det_tid = det[["detection_id", "tracklet_id"]].drop_duplicates()
    matched = matched.merge(det_tid, left_on="pred_detection_id", right_on="detection_id", how="left")
    matched["node_id"] = matched.apply(
        lambda r: _find_node_from_segs(d1_segs, r["tracklet_id"], r["frame_index"])
        if pd.notna(r["tracklet_id"]) else None,
        axis=1,
    )
    node_frame_occ = _build_node_frame_occupancy_from_dedup(matched)

    hop_rows = _classify_hops(paths, edge_lookup, selected_pairs, node_map, d3_flow, graph["tbf"])
    shared_rows = _shared_node_analysis(node_frame_occ, node_map, d3_flow)
    indep = _independent_reachability(paths, hop_rows)
    joint = _joint_feasibility(node_frame_occ, node_map, sorted(paths.keys()))
    agg = _aggregate_by_owner(hop_rows)

    md = _write_findings("dedup-ceiling", paths, hop_rows, shared_rows, det_stats, indep, joint, agg)

    summary = {
        "mode": "dedup-ceiling",
        "n_gt_people": len(paths),
        "total_hops": len(hop_rows),
        "aggregate": agg,
        "independent_reachability": {str(k): v["reachable"] for k, v in indep.items()},
        "n_independent_reachable": sum(1 for v in indep.values() if v["reachable"]),
        "joint_feasibility": {
            "all_feasible": joint["all_feasible"],
            "max_coexisting": int(joint["max_coexisting"]) if not joint["all_feasible"] else len(paths),
        },
        "shared_structural": len([s for s in shared_rows if s["contention"] == "structural_impossibility"]),
        "shared_group_ok": len([s for s in shared_rows if s["contention"] == "group_handles_it"]),
        "shared_sequential": len([s for s in shared_rows if s["contention"] == "sequential"]),
    }

    return md, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Production
    prod_md, prod_summary = run_production()

    # Dedup-ceiling
    dedup_md, dedup_summary = run_dedup()

    # Combined findings
    combined = prod_md + "\n\n---\n\n" + dedup_md

    # Comparison section
    combined += "\n\n---\n\n# Cross-Check: Production vs Dedup-Ceiling\n\n"
    combined += "| Metric | Production | Dedup-Ceiling |\n"
    combined += "|---|---|---|\n"
    combined += f"| Total hops | {prod_summary['total_hops']} | {dedup_summary['total_hops']} |\n"
    for outcome in ["EDGE_EXISTS_SELECTED", "EDGE_EXISTS_NOT_SELECTED",
                     "CONCURRENT_NODES", "EDGE_ABSENT_IN_WINDOW",
                     "UNREACHABLE_BY_WINDOW"]:
        combined += (f"| {outcome} | {prod_summary['aggregate'].get(outcome, 0)} "
                     f"| {dedup_summary['aggregate'].get(outcome, 0)} |\n")
    combined += (f"| Independent reachability | {prod_summary['n_independent_reachable']}/8 "
                 f"| {dedup_summary['n_independent_reachable']}/8 |\n")
    combined += (f"| Joint feasibility | {prod_summary['joint_feasibility']['max_coexisting']}/8 "
                 f"| {dedup_summary['joint_feasibility']['max_coexisting']}/8 |\n")
    combined += f"| Shared: structural impossibility | {prod_summary.get('shared_structural', 0)} | {dedup_summary.get('shared_structural', 0)} |\n"
    combined += f"| Shared: GROUP handles it | {prod_summary.get('shared_group_ok', 0)} | {dedup_summary.get('shared_group_ok', 0)} |\n"
    combined += f"| Shared: sequential (no contention) | {prod_summary.get('shared_sequential', 0)} | {dedup_summary.get('shared_sequential', 0)} |\n"

    # Write outputs
    (OUT_DIR / "findings.md").write_text(combined)

    with open(OUT_DIR / "reach_summary.json", "w") as f:
        json.dump({"production": prod_summary, "dedup_ceiling": dedup_summary}, f, indent=2)

    print(f"\nOutputs written to {OUT_DIR}")
    print(f"  findings.md")
    print(f"  reach_summary.json")


if __name__ == "__main__":
    main()
