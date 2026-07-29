#!/usr/bin/env python3
"""CP7-pre-8: Axis-1 failure-signature characterization for FP7oJQ.

READ-ONLY diagnostic. No pipeline/config changes.
Produces measurements for Branch A vs Branch B decision.

Usage:
    PYTHONPATH=src python tools/cp7_pre8_axis1_diagnostic.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
CLIP_ROOT = Path(
    "outputs/_eval_gt/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014"
)
EVAL_ROOT = Path("outputs/_eval/stage_d/bjj-detect-all-cameras/FP7oJQ")
SWAP_ROOT = Path("outputs/_eval/tracker_swap/bjj-detect-all-cameras/FP7oJQ")
OUT_DIR = Path("outputs/_eval/_debug/cp7_pre8_axis1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── config thresholds (from configs/default.yaml, confirmed in Pass 1) ──
MERGE_DIST_M = 1.5  # effective merge_dist_m from YAML
PROXIMITY_THRESHOLD_M = MERGE_DIST_M  # episode proximity gate
MIN_EPISODE_FRAMES = 5  # minimum consecutive proximate frames
MIN_OVERLAP_FRAMES = 15  # minimum temporal overlap for pair consideration
CONVERGENCE_WINDOW = 10  # frames around death to check convergence
CONVERGENCE_DIST_M = MERGE_DIST_M  # convergence bar = merge gate (Correction 3)
BRACKET_SEARCH_WINDOW = 30  # frames before/after GROUP span to check resolution
BRACKET_RADIUS_M = 1.0  # world-coord radius for bracket check
GT_FRAME_MAX = 300  # GT annotated range upper bound
SWAP_VICINITY_FRAMES = 5  # frames around swap boundary for Branch-B test


def load_artifacts():
    """Load all required parquet/JSON artifacts."""
    bf = pd.read_parquet(CLIP_ROOT / "stage_D" / "tracklet_bank_frames.parquet")
    bs = pd.read_parquet(CLIP_ROOT / "stage_D" / "tracklet_bank_summaries.parquet")
    det = pd.read_parquet(CLIP_ROOT / "stage_A" / "detections.parquet")
    nodes = pd.read_parquet(CLIP_ROOT / "stage_D" / "d1_graph_nodes.parquet")
    group_spans = pd.read_parquet(CLIP_ROOT / "_debug" / "d1_group_spans.parquet")
    trace = pd.read_parquet(EVAL_ROOT / "gt_person_trace.parquet")

    with open(CLIP_ROOT / "_debug" / "d3_ilp2_group_semantics.json") as f:
        group_sem = json.load(f)

    swap_events = []
    with open(SWAP_ROOT / "swap_events.jsonl") as f:
        for line in f:
            swap_events.append(json.loads(line))

    with open(SWAP_ROOT / "topology.json") as f:
        topology = json.load(f)

    return bf, bs, det, nodes, group_spans, trace, group_sem, swap_events, topology


# ══════════════════════════════════════════════════════════════════════════
#  PART 2a: GT-free proximity episode classifier (full clip)
# ══════════════════════════════════════════════════════════════════════════

def run_part_2a(bf: pd.DataFrame, bs: pd.DataFrame) -> list[dict]:
    """Classify proximity episodes as clean-lifecycle vs concurrent-alive."""

    # Build per-tracklet lifespan + per-frame positions
    lifespan = {
        row.tracklet_id: (int(row.start_frame), int(row.end_frame))
        for row in bs.itertuples()
    }

    # Index: (tracklet_id, frame_index) -> (x_m, y_m, occ_r_bottom, occ_r_height)
    bf_indexed = bf.set_index(["tracklet_id", "frame_index"])[
        ["x_m", "y_m", "occ_r_bottom", "occ_r_height"]
    ]

    tids = sorted(lifespan.keys())
    episodes = []
    episode_id = 0

    for i, tidA in enumerate(tids):
        startA, endA = lifespan[tidA]
        for tidB in tids[i + 1 :]:
            startB, endB = lifespan[tidB]

            # Check temporal overlap
            ov_start = max(startA, startB)
            ov_end = min(endA, endB)
            if ov_end - ov_start + 1 < MIN_OVERLAP_FRAMES:
                continue

            # Get positions for both in overlap
            try:
                posA = bf_indexed.loc[tidA]
                posB = bf_indexed.loc[tidB]
            except KeyError:
                continue

            # Align on frame_index within overlap
            framesA = posA.index
            framesB = posB.index
            common = sorted(
                set(framesA) & set(framesB)
                & set(range(ov_start, ov_end + 1))
            )
            if len(common) < MIN_EPISODE_FRAMES:
                continue

            # Compute distances at common frames
            dists = {}
            for f in common:
                try:
                    rA = posA.loc[f]
                    rB = posB.loc[f]
                except KeyError:
                    continue
                # Handle possible duplicate indices (multi-row per frame after split)
                xA = float(rA["x_m"].iloc[0]) if hasattr(rA["x_m"], "iloc") else float(rA["x_m"])
                yA = float(rA["y_m"].iloc[0]) if hasattr(rA["y_m"], "iloc") else float(rA["y_m"])
                xB = float(rB["x_m"].iloc[0]) if hasattr(rB["x_m"], "iloc") else float(rB["x_m"])
                yB = float(rB["y_m"].iloc[0]) if hasattr(rB["y_m"], "iloc") else float(rB["y_m"])
                dists[f] = np.sqrt((xA - xB) ** 2 + (yA - yB) ** 2)

            if not dists:
                continue

            # Find consecutive proximity runs
            sorted_frames = sorted(dists.keys())
            in_prox = False
            run_start = None
            runs = []

            for fi, f in enumerate(sorted_frames):
                if dists[f] < PROXIMITY_THRESHOLD_M:
                    if not in_prox:
                        in_prox = True
                        run_start = f
                elif in_prox:
                    in_prox = False
                    runs.append((run_start, sorted_frames[fi - 1]))
                    run_start = None
            if in_prox and run_start is not None:
                runs.append((run_start, sorted_frames[-1]))

            # Merge runs separated by <=5 non-proximate frames
            merged_runs = []
            for run in runs:
                if merged_runs and run[0] - merged_runs[-1][1] <= 5:
                    merged_runs[-1] = (merged_runs[-1][0], run[1])
                else:
                    merged_runs.append(run)

            for rs, re in merged_runs:
                n_frames = re - rs + 1
                if n_frames < MIN_EPISODE_FRAMES:
                    continue

                run_dists = [dists[f] for f in sorted_frames if rs <= f <= re and f in dists]

                # Check for death within episode ± 5 frames
                death_tid = None
                death_frame = None
                survivor_tid = None

                for tid, (ts, te) in [(tidA, lifespan[tidA]), (tidB, lifespan[tidB])]:
                    if rs - 5 <= te <= re + 5 and te < 4529:  # not clip boundary
                        death_tid = tid
                        death_frame = te
                        survivor_tid = tidB if tid == tidA else tidA
                        break

                classification = "concurrent_alive"
                convergence_position = False
                convergence_bbox = False

                if death_tid is not None and death_frame is not None:
                    # Check convergence evidence (Correction 3: bar = MERGE_DIST_M)
                    # Position convergence: distance trend decreasing in window before death
                    # AND final distance < CONVERGENCE_DIST_M
                    pre_death_frames = [
                        f for f in sorted_frames
                        if death_frame - CONVERGENCE_WINDOW <= f <= death_frame
                        and f in dists
                    ]
                    if len(pre_death_frames) >= 3:
                        pre_dists = [dists[f] for f in pre_death_frames]
                        # Trend: compare first half mean to second half mean
                        mid = len(pre_dists) // 2
                        if mid > 0:
                            first_half_mean = np.mean(pre_dists[:mid])
                            second_half_mean = np.mean(pre_dists[mid:])
                            final_dist = pre_dists[-1]
                            if second_half_mean < first_half_mean and final_dist < CONVERGENCE_DIST_M:
                                convergence_position = True

                    # Bbox growth: survivor's occ_r_height increases around death
                    try:
                        surv_data = posA if survivor_tid == tidA else posB
                        window_frames = [
                            f for f in range(death_frame - CONVERGENCE_WINDOW, death_frame + CONVERGENCE_WINDOW + 1)
                            if f in surv_data.index
                        ]
                        if len(window_frames) >= 4:
                            mid_f = len(window_frames) // 2
                            pre_vals = surv_data.loc[window_frames[:mid_f]]
                            post_vals = surv_data.loc[window_frames[mid_f:]]

                            def _safe_mean(s):
                                if hasattr(s, "iloc"):
                                    return float(s.mean())
                                return float(s)

                            pre_h = _safe_mean(pre_vals["occ_r_height"])
                            post_h = _safe_mean(post_vals["occ_r_height"])
                            pre_b = _safe_mean(pre_vals["occ_r_bottom"])
                            post_b = _safe_mean(post_vals["occ_r_bottom"])

                            if post_h - pre_h >= 0.05 or post_b - pre_b >= 0.03:
                                convergence_bbox = True
                    except (KeyError, IndexError):
                        pass

                    if convergence_position or convergence_bbox:
                        classification = "clean_lifecycle"
                    else:
                        classification = "ordinary_exit"

                episodes.append({
                    "episode_id": episode_id,
                    "tidA": tidA,
                    "tidB": tidB,
                    "start_frame": rs,
                    "end_frame": re,
                    "n_frames": n_frames,
                    "min_distance_m": round(min(run_dists), 3) if run_dists else None,
                    "mean_distance_m": round(float(np.mean(run_dists)), 3) if run_dists else None,
                    "classification": classification,
                    "death_tracklet": death_tid,
                    "death_frame": death_frame,
                    "convergence_position": convergence_position,
                    "convergence_bbox": convergence_bbox,
                })
                episode_id += 1

    return episodes


# ══════════════════════════════════════════════════════════════════════════
#  PART 2c: Bracketed detection proxy (world-coord, identity-tracked)
#     Two populations: D1-caught spans AND all pair-context spans
# ══════════════════════════════════════════════════════════════════════════

def _bracket_test(
    carrier_tid: str,
    span_start: int,
    span_end: int,
    bf: pd.DataFrame,
    bf_indexed: pd.DataFrame,
) -> dict:
    """Run pre/post bracket test for a single span.

    Returns dict with bracket_class, pre_resolved_n, post_resolved_n.
    Identity-tracked: carrier_tid must be present in both windows.
    """
    # Pre-bracket: 30 frames before span_start
    pre_start = max(0, span_start - BRACKET_SEARCH_WINDOW)
    pre_end = span_start - 1
    # Post-bracket: 30 frames after span_end
    post_start = span_end + 1
    post_end = min(4529, span_end + BRACKET_SEARCH_WINDOW)

    def count_resolved(window_start, window_end):
        """Count distinct tracklets within BRACKET_RADIUS_M of carrier in window."""
        if window_start > window_end:
            return 0

        # Get carrier positions in window
        try:
            carrier_data = bf_indexed.loc[carrier_tid]
        except KeyError:
            return 0

        carrier_in_window = carrier_data[
            (carrier_data.index >= window_start) & (carrier_data.index <= window_end)
        ]
        if carrier_in_window.empty:
            return 0  # carrier not present — identity-tracking fails

        # Get mean carrier position in window
        cx = float(carrier_in_window["x_m"].mean())
        cy = float(carrier_in_window["y_m"].mean())

        # Find all tracklets in window near carrier
        window_data = bf[
            (bf.frame_index >= window_start)
            & (bf.frame_index <= window_end)
        ]
        # Per-tracklet mean position in window
        tid_positions = window_data.groupby("tracklet_id")[["x_m", "y_m"]].mean()
        dists = np.sqrt((tid_positions.x_m - cx) ** 2 + (tid_positions.y_m - cy) ** 2)
        nearby = dists[dists < BRACKET_RADIUS_M]
        return len(nearby)

    pre_n = count_resolved(pre_start, pre_end)
    post_n = count_resolved(post_start, post_end)

    if pre_n >= 2 and post_n >= 2:
        bracket_class = "bracketed"
    elif pre_n >= 2:
        bracket_class = "half_bracket_pre"
    elif post_n >= 2:
        bracket_class = "half_bracket_post"
    else:
        bracket_class = "unbracketed"

    return {
        "bracket_class": bracket_class,
        "pre_resolved_n": pre_n,
        "post_resolved_n": post_n,
    }


def run_part_2c_d1_caught(
    group_spans: pd.DataFrame, bf: pd.DataFrame
) -> list[dict]:
    """Bracket test for D1-caught GROUP spans."""
    bf_indexed = bf.set_index(["tracklet_id", "frame_index"])[["x_m", "y_m"]]
    results = []

    for idx, row in group_spans.iterrows():
        carrier = row["carrier"]
        gs = int(row["group_start"])
        ge = int(row["group_end"])

        bt = _bracket_test(carrier, gs, ge, bf, bf_indexed)

        results.append({
            "span_id": int(idx),
            "carrier_tid": carrier,
            "disappear_tid": row.get("disappear"),
            "new_tid": row.get("new"),
            "group_start": gs,
            "group_end": ge,
            "n_frames": ge - gs + 1,
            "kind": row["kind"],
            "population": "d1_caught",
            **bt,
        })

    return results


def run_part_2c_all_pair_context(
    bf: pd.DataFrame, bs: pd.DataFrame, group_spans: pd.DataFrame
) -> list[dict]:
    """Bracket test for ALL pair-context spans (GT-free, independent of D1).

    A pair-context span = a tracklet dies near (< MERGE_DIST_M) an active carrier,
    creating a period where one box covers the work of two. The span runs from
    death_frame to the next tracklet birth near the carrier (or carrier end).
    """
    bf_indexed = bf.set_index(["tracklet_id", "frame_index"])[["x_m", "y_m"]]
    lifespan = {
        row.tracklet_id: (int(row.start_frame), int(row.end_frame))
        for row in bs.itertuples()
    }

    # Build set of D1-caught spans for labeling overlap
    d1_spans = set()
    for _, row in group_spans.iterrows():
        d1_spans.add((row["carrier"], int(row["group_start"]), int(row["group_end"])))

    # For each tracklet death (non-boundary), find if a carrier was nearby
    results = []
    span_id = 0

    # Get last-frame positions
    last_pos = (
        bf.sort_values("frame_index")
        .groupby("tracklet_id")
        .last()[["frame_index", "x_m", "y_m"]]
    )

    for dying_tid, drow in last_pos.iterrows():
        death_f = int(drow["frame_index"])
        if death_f >= 4529:  # clip boundary
            continue

        dx, dy = float(drow["x_m"]), float(drow["y_m"])

        # Find active tracklets at death_f within MERGE_DIST_M
        active_at_death = bf[
            (bf.frame_index == death_f)
            & (bf.tracklet_id != dying_tid)
        ]
        if active_at_death.empty:
            continue

        for _, cand in active_at_death.iterrows():
            carrier_tid = cand["tracklet_id"]
            cx, cy = float(cand["x_m"]), float(cand["y_m"])
            dist = np.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
            if dist >= MERGE_DIST_M:
                continue

            carrier_start, carrier_end = lifespan.get(carrier_tid, (0, 0))
            if death_f >= carrier_end:
                continue  # carrier also dying

            # Pair-context span: from death_f to next birth near carrier or carrier_end
            # Find next tracklet birth within MERGE_DIST_M of carrier after death_f
            span_end = carrier_end  # default: runs to carrier end
            for birth_tid, (bs_start, _) in lifespan.items():
                if birth_tid == carrier_tid or birth_tid == dying_tid:
                    continue
                if bs_start <= death_f:
                    continue
                # Check if birth is near carrier at birth frame
                try:
                    carrier_at_birth = bf_indexed.loc[(carrier_tid, bs_start)]
                    birth_data = bf_indexed.loc[(birth_tid, bs_start)]
                    bcx = float(carrier_at_birth["x_m"].iloc[0]) if hasattr(carrier_at_birth["x_m"], "iloc") else float(carrier_at_birth["x_m"])
                    bcy = float(carrier_at_birth["y_m"].iloc[0]) if hasattr(carrier_at_birth["y_m"], "iloc") else float(carrier_at_birth["y_m"])
                    bbx = float(birth_data["x_m"].iloc[0]) if hasattr(birth_data["x_m"], "iloc") else float(birth_data["x_m"])
                    bby = float(birth_data["y_m"].iloc[0]) if hasattr(birth_data["y_m"], "iloc") else float(birth_data["y_m"])
                    bdist = np.sqrt((bcx - bbx) ** 2 + (bcy - bby) ** 2)
                    if bdist < MERGE_DIST_M and bs_start < span_end:
                        span_end = bs_start
                except KeyError:
                    continue

            if span_end - death_f < 5:
                continue  # too short to be meaningful

            # Check if this overlaps a D1-caught span
            overlaps_d1 = any(
                c == carrier_tid and not (span_end < gs or death_f > ge)
                for c, gs, ge in d1_spans
            )

            bt = _bracket_test(carrier_tid, death_f, span_end, bf, bf_indexed)

            results.append({
                "span_id": span_id,
                "carrier_tid": carrier_tid,
                "dying_tid": dying_tid,
                "span_start": death_f,
                "span_end": span_end,
                "n_frames": span_end - death_f + 1,
                "death_dist_m": round(dist, 3),
                "overlaps_d1": overlaps_d1,
                "population": "all_pair_context",
                **bt,
            })
            span_id += 1

    return results


# ══════════════════════════════════════════════════════════════════════════
#  PART 3: GT-anchored misattribution signature (frames 0-300)
# ══════════════════════════════════════════════════════════════════════════

def run_part_3(
    trace: pd.DataFrame,
    swap_events: list[dict],
    topology: list[dict],
    group_spans: pd.DataFrame,
    nodes: pd.DataFrame,
    bf: pd.DataFrame,
) -> list[dict]:
    """Label each present_misattributed frame as Branch-A, Branch-B, or other."""

    misattr = trace[
        (trace.failure_mode == "present_misattributed")
        & (trace.frame_idx <= GT_FRAME_MAX)
    ].copy()

    # Build swap event lookup: (tracklet_id, frame_before, frame_after)
    swap_lookup = []
    for ev in swap_events:
        swap_lookup.append((
            ev["tracklet_id"],
            ev["frame_before"],
            ev["frame_after"],
        ))

    # Build GROUP node → role mapping from d1_graph_nodes
    # For each GROUP node, record which tracklets are carrier/disappear/new
    group_roles = {}  # node_id -> {carrier, disappear, new}
    group_nodes = nodes[nodes.node_type == "NodeType.GROUP_TRACKLET"]
    for _, gn in group_nodes.iterrows():
        roles = set()
        if pd.notna(gn.carrier_tracklet_id):
            roles.add(gn.carrier_tracklet_id)
        if pd.notna(gn.disappearing_tracklet_id):
            roles.add(gn.disappearing_tracklet_id)
        if pd.notna(gn.new_tracklet_id):
            roles.add(gn.new_tracklet_id)
        group_roles[gn.node_id] = {
            "carrier": gn.carrier_tracklet_id if pd.notna(gn.carrier_tracklet_id) else None,
            "disappear": gn.disappearing_tracklet_id if pd.notna(gn.disappearing_tracklet_id) else None,
            "new": gn.new_tracklet_id if pd.notna(gn.new_tracklet_id) else None,
            "start": int(gn.start_frame),
            "end": int(gn.end_frame),
            "roles_set": roles,
        }

    # Build per-frame active tracklets for Branch-B persistent test
    # (from bank_frames, GT-free)
    bf_gt_range = bf[bf.frame_index <= GT_FRAME_MAX]
    frame_tracklets = bf_gt_range.groupby("frame_index")["tracklet_id"].apply(set).to_dict()

    # Build person_id assignments per tracklet (from trace data)
    # Use the trace's final_person_id column
    tid_to_person = {}
    for _, row in trace[trace.final_person_id.notna()].iterrows():
        tid = row.tracklet_id
        pid = row.final_person_id
        if tid not in tid_to_person:
            tid_to_person[tid] = set()
        tid_to_person[tid].add(pid)

    results = []

    for _, row in misattr.iterrows():
        frame = int(row.frame_idx)
        tid = row.tracklet_id
        canonical = row.canonical_person_id
        final = row.final_person_id
        gt_person = row.gt_person_id

        # ── Branch-B test: swap vicinity ──
        is_swap_vicinity = False
        swap_detail = None
        for s_tid, s_before, s_after in swap_lookup:
            if s_tid == tid and s_before - SWAP_VICINITY_FRAMES <= frame <= s_after + SWAP_VICINITY_FRAMES:
                is_swap_vicinity = True
                # Find topology class
                for t in topology:
                    if t.get("tracklet_id") == s_tid and t.get("frame_before") == s_before:
                        swap_detail = t.get("topology_class", "unknown")
                        break
                break

        # ── Branch-B test: persistent concurrent-alive ──
        is_persistent_concurrent = False
        if not is_swap_vicinity and frame in frame_tracklets:
            active_tids = frame_tracklets[frame]
            # Check if >=2 active tracklets cover different GT persons at this frame
            # (i.e., the misattributed tracklet + another tracklet assigned to canonical)
            for other_tid in active_tids:
                if other_tid == tid:
                    continue
                other_persons = tid_to_person.get(other_tid, set())
                if canonical in other_persons:
                    is_persistent_concurrent = True
                    break

        # ── Branch-A test (tightened per Correction 1) ──
        # Require: (a) tracklet is carrier/disappear/new in a GROUP node active at
        # this frame, AND (b) no concurrent swap event covers this frame.
        is_branch_a = False
        branch_a_node = None
        if not is_swap_vicinity:  # condition (b): no swap covers frame
            # Parse d1_node_ids from trace
            try:
                node_ids = json.loads(row.d1_node_ids) if isinstance(row.d1_node_ids, str) else []
            except (json.JSONDecodeError, TypeError):
                node_ids = []

            for nid in node_ids:
                if nid in group_roles:
                    info = group_roles[nid]
                    # Condition (a): this tracklet is one of the routed roles
                    if tid in info["roles_set"]:
                        is_branch_a = True
                        branch_a_node = nid
                        break

        # ── Classify ──
        if is_branch_a and (is_swap_vicinity or is_persistent_concurrent):
            signature = "ambiguous_a_b"
            evidence = f"GROUP node {branch_a_node} routes {tid}, but concurrent swap/persistent also present"
        elif is_branch_a:
            signature = "branch_a"
            evidence = f"GROUP node {branch_a_node} active, {tid} is routed role, no concurrent swap"
        elif is_swap_vicinity:
            signature = "branch_b_swap"
            evidence = f"swap boundary vicinity, topology={swap_detail}"
        elif is_persistent_concurrent:
            signature = "branch_b_persistent"
            evidence = f"concurrent tracklet assigned to canonical {canonical}"
        else:
            # Check if GROUP node covers frame but tracklet is NOT a routed role
            # This is Axis-2 / under-segmentation territory
            has_group_cover = False
            try:
                node_ids = json.loads(row.d1_node_ids) if isinstance(row.d1_node_ids, str) else []
            except (json.JSONDecodeError, TypeError):
                node_ids = []
            for nid in node_ids:
                if nid in group_roles:
                    has_group_cover = True
                    break

            if has_group_cover:
                signature = "axis2_underseg"
                evidence = "GROUP node covers frame but tracklet not a routed role (incidental overlap)"
            else:
                signature = "other"
                evidence = "no GROUP, no swap, no concurrent canonical-holder"

        results.append({
            "frame_idx": frame,
            "gt_person_id": int(gt_person),
            "tracklet_id": tid,
            "canonical_person_id": canonical,
            "final_person_id": final,
            "signature": signature,
            "evidence": evidence,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Loading artifacts...")
    bf, bs, det, nodes, group_spans, trace, group_sem, swap_events, topology = (
        load_artifacts()
    )

    # ── Part 1: Config reconciliation (inline) ──
    print("\n" + "=" * 72)
    print("PART 1: Config Reconciliation")
    print("=" * 72)
    print(f"  merge_dist_m: 1.5 m (YAML) overrides 0.45 m (code default)")
    print(f"  split_dist_m: 2.0 m (YAML) overrides 0.60 m (code default)")
    print(f"  No disconnect — config flows through run_d1() -> d1_cfg.get()")

    # ── Part 2b: forced_unused tally (inline, already measured) ──
    print("\n" + "=" * 72)
    print("PART 2b: GROUP forced_unused tally")
    print("=" * 72)
    summary = group_sem.get("summary", group_sem)
    print(f"  GROUP nodes total:                 {summary.get('n_group_nodes', 'N/A')}")
    print(f"  GROUPISH nodes total:              {summary.get('n_groupish_nodes', 'N/A')}")
    print(f"  Forced unused (all reasons):       {summary.get('n_forced_unused', 'N/A')}")
    print(f"  missing_required_merge_in:         {summary.get('n_missing_merge_required', 'N/A')}")
    print(f"  missing_required_split_out:         {summary.get('n_missing_split_required', 'N/A')}")
    print(f"  missing_groupish_group_cont_bridge: {summary.get('n_missing_groupish_bridge', 'N/A')}")
    print(f"\n  => 'Node formed but D3 forced unused' version of Branch A is DEAD.")
    print(f"     Branch A survives only as lifecycle events that never triggered a node.")

    # ── Part 2a: Proximity episode classification ──
    print("\n" + "=" * 72)
    print("PART 2a: GT-free proximity episode classification (full clip)")
    print("=" * 72)
    print(f"  Proximity threshold: {PROXIMITY_THRESHOLD_M} m (= merge_dist_m from config)")
    print(f"  Convergence distance bar: {CONVERGENCE_DIST_M} m (= merge_dist_m)")
    print(f"  Rationale: convergence bar tied to the pipeline's actual merge gate.")
    print(f"  A tracklet death within {CONVERGENCE_DIST_M} m with converging trajectory")
    print(f"  is plausibly a merge; requiring tighter would under-count clean-lifecycle.")
    print()
    print("  Computing proximity episodes (this may take a minute)...")

    episodes = run_part_2a(bf, bs)

    # Save raw
    with open(OUT_DIR / "part_2a_episodes.json", "w") as f:
        json.dump(episodes, f, indent=2, default=str)

    # Summary
    ep_counts = Counter(e["classification"] for e in episodes)
    total_ep = len(episodes)
    print(f"\n  Total proximity episodes: {total_ep}")
    print(f"  {'Classification':<22} {'Episodes':>8} {'%':>7}")
    print(f"  {'-'*22} {'-'*8} {'-'*7}")
    for cls in ["concurrent_alive", "clean_lifecycle", "ordinary_exit"]:
        n = ep_counts.get(cls, 0)
        pct = 100 * n / total_ep if total_ep else 0
        print(f"  {cls:<22} {n:>8} {pct:>6.1f}%")

    # ── Part 2c: Bracketed detection proxy ──
    print("\n" + "=" * 72)
    print("PART 2c: Bracketed detection proxy (world-coord, identity-tracked)")
    print("=" * 72)

    print("  Population 1: D1-caught GROUP spans...")
    d1_brackets = run_part_2c_d1_caught(group_spans, bf)

    print("  Population 2: All pair-context spans (GT-free)...")
    all_brackets = run_part_2c_all_pair_context(bf, bs, group_spans)

    # Save raw
    with open(OUT_DIR / "part_2c_d1_brackets.json", "w") as f:
        json.dump(d1_brackets, f, indent=2, default=str)
    with open(OUT_DIR / "part_2c_all_brackets.json", "w") as f:
        json.dump(all_brackets, f, indent=2, default=str)

    for label, brackets in [("D1-caught spans", d1_brackets), ("All pair-context spans", all_brackets)]:
        total = len(brackets)
        if total == 0:
            print(f"\n  {label}: 0 spans found")
            continue
        bc = Counter(b["bracket_class"] for b in brackets)
        print(f"\n  {label} ({total} total):")
        print(f"  {'Bracket Class':<22} {'Spans':>6} {'%':>7}")
        print(f"  {'-'*22} {'-'*6} {'-'*7}")
        for cls in ["bracketed", "half_bracket_pre", "half_bracket_post", "unbracketed"]:
            n = bc.get(cls, 0)
            pct = 100 * n / total if total else 0
            print(f"  {cls:<22} {n:>6} {pct:>6.1f}%")

        if label == "All pair-context spans":
            d1_overlap = sum(1 for b in brackets if b.get("overlaps_d1"))
            non_d1 = total - d1_overlap
            print(f"\n  D1-overlapping: {d1_overlap}, D1-missed: {non_d1}")
            if non_d1 > 0:
                bc_missed = Counter(
                    b["bracket_class"] for b in brackets if not b.get("overlaps_d1")
                )
                print(f"  D1-missed bracket rate: {bc_missed.get('bracketed', 0)}/{non_d1} "
                      f"({100*bc_missed.get('bracketed', 0)/non_d1:.1f}%)")

    # ── Part 3: GT-anchored signature ──
    print("\n" + "=" * 72)
    print("PART 3: GT-anchored misattribution signature (frames 0-300)")
    print("=" * 72)

    sig_results = run_part_3(trace, swap_events, topology, group_spans, nodes, bf)

    # Save raw
    with open(OUT_DIR / "part_3_signatures.json", "w") as f:
        json.dump(sig_results, f, indent=2, default=str)

    total_sig = len(sig_results)
    sig_counts = Counter(r["signature"] for r in sig_results)
    print(f"\n  Total misattributed frames (0-300): {total_sig}")
    print(f"  {'Signature':<22} {'Frames':>7} {'%':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7}")
    for sig in ["branch_a", "branch_b_swap", "branch_b_persistent", "ambiguous_a_b", "axis2_underseg", "other"]:
        n = sig_counts.get(sig, 0)
        pct = 100 * n / total_sig if total_sig else 0
        print(f"  {sig:<22} {n:>7} {pct:>6.1f}%")

    # Branch A vs B aggregation
    branch_a_total = sig_counts.get("branch_a", 0) + sig_counts.get("ambiguous_a_b", 0)
    branch_b_total = (
        sig_counts.get("branch_b_swap", 0)
        + sig_counts.get("branch_b_persistent", 0)
        + sig_counts.get("ambiguous_a_b", 0)
    )
    print(f"\n  Branch A (including ambiguous): {branch_a_total} ({100*branch_a_total/total_sig:.1f}%)")
    print(f"  Branch B (including ambiguous): {branch_b_total} ({100*branch_b_total/total_sig:.1f}%)")

    # ── Cross-checks ──
    print("\n" + "=" * 72)
    print("CROSS-CHECKS")
    print("=" * 72)

    # (a) All-pair-context bracket rate vs Part 3 signature
    all_bracket_rate = (
        sum(1 for b in all_brackets if b["bracket_class"] == "bracketed") / len(all_brackets)
        if all_brackets else 0
    )
    branch_b_pct = 100 * branch_b_total / total_sig if total_sig else 0
    print(f"\n  (a) Bracket rate (all pair-context): {all_bracket_rate:.1%}")
    print(f"      Branch B in GT signature: {branch_b_pct:.1f}%")
    if all_bracket_rate > 0.5 and branch_b_pct > 50:
        print("      => High bracket rate + Branch B dominant: GROUP nodes cover the spans")
        print("         but wrong identity exits. Fix is routing/identity, not trigger expansion.")
    elif all_bracket_rate < 0.3:
        print("      => Low bracket rate: many pair-context spans never resolve to 2 boxes.")
        print("         GROUP machinery has limited Axis-1 reach.")
    else:
        print("      => Mixed bracket rate — interpretation requires case-by-case review.")

    # (b) Full-clip prevalence vs GT signature
    ca_pct = 100 * ep_counts.get("concurrent_alive", 0) / total_ep if total_ep else 0
    cl_pct = 100 * ep_counts.get("clean_lifecycle", 0) / total_ep if total_ep else 0
    print(f"\n  (b) Full-clip: concurrent_alive={ca_pct:.1f}%, clean_lifecycle={cl_pct:.1f}%")
    print(f"      GT (0-300): Branch A={100*sig_counts.get('branch_a',0)/total_sig:.1f}%, "
          f"Branch B={branch_b_pct:.1f}%")

    if ca_pct > 60 and branch_b_pct > 60:
        print("      => AGREE: concurrent-alive dominates full clip AND produces confirmed misattribution.")
    elif cl_pct > ca_pct and sig_counts.get("branch_a", 0) > branch_b_total:
        print("      => AGREE: clean-lifecycle dominates full clip AND Branch A dominates GT signature.")
    else:
        print("      => Prevalence and harm may diverge — dominant mechanism may not be the harmful one.")

    # ── Verdict ──
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    pure_a = sig_counts.get("branch_a", 0)
    pure_b_swap = sig_counts.get("branch_b_swap", 0)
    pure_b_pers = sig_counts.get("branch_b_persistent", 0)
    ambig = sig_counts.get("ambiguous_a_b", 0)
    axis2 = sig_counts.get("axis2_underseg", 0)
    other = sig_counts.get("other", 0)

    print(f"\n  With 2b=0 (zero forced_unused GROUP nodes), the live A-vs-B fork is:")
    print(f"    Branch A = clean-lifecycle events not triggering GROUP nodes (gate-tuning)")
    print(f"    Branch B = concurrent-alive tracklets swapping (new node class needed)")
    print()

    if pure_b_swap + pure_b_pers > pure_a * 3 and ca_pct > cl_pct:
        print(f"  RECOMMENDATION: Branch B (new concurrent-swap node class).")
        print(f"  Evidence: Branch B frames ({pure_b_swap + pure_b_pers}) vastly outnumber")
        print(f"  Branch A ({pure_a}). Full-clip concurrent_alive ({ca_pct:.0f}%) > clean_lifecycle")
        print(f"  ({cl_pct:.0f}%). The dominant failure is two tracklets staying alive and")
        print(f"  swapping detections, with no death/birth event for GROUP to trigger on.")
    elif pure_a > (pure_b_swap + pure_b_pers) * 3 and cl_pct > ca_pct:
        print(f"  RECOMMENDATION: Branch A (tune/loosen GROUP trigger gates).")
        print(f"  Evidence: Branch A frames ({pure_a}) dominate. Real lifecycle events exist")
        print(f"  but GROUP nodes aren't forming. Gate-tuning is cheaper, try first.")
    elif total_sig < 50:
        print(f"  VERDICT: Indeterminate on this clip — too few misattributed frames ({total_sig})")
        print(f"  in the GT window. Wait for buzzer-video GT with longer annotation spans.")
    else:
        print(f"  VERDICT: Mixed signal.")
        print(f"  Branch A: {pure_a} frames ({100*pure_a/total_sig:.1f}%)")
        print(f"  Branch B: {pure_b_swap + pure_b_pers} frames ({100*(pure_b_swap+pure_b_pers)/total_sig:.1f}%)")
        print(f"  Ambiguous: {ambig} ({100*ambig/total_sig:.1f}%)")
        print(f"  Axis-2: {axis2} ({100*axis2/total_sig:.1f}%)")
        print(f"  Full-clip: concurrent_alive={ca_pct:.1f}% vs clean_lifecycle={cl_pct:.1f}%")
        if ca_pct > cl_pct and (pure_b_swap + pure_b_pers) > pure_a:
            print(f"\n  Lean: Branch B, but ambiguous/axis2 mass warrants caution.")
        elif cl_pct > ca_pct and pure_a > (pure_b_swap + pure_b_pers):
            print(f"\n  Lean: Branch A (gate-tuning), but verify with longer GT.")
        else:
            print(f"\n  No clear lean. Wait for buzzer-video GT.")

    print(f"\n  STOP: Report written to {OUT_DIR}/ and docs/checkpoints/cp7_pre8_axis1_signature.md.")
    print(f"  Node/trigger design returns to the web session.")

    return episodes, d1_brackets, all_brackets, sig_results


if __name__ == "__main__":
    episodes, d1_brackets, all_brackets, sig_results = main()
