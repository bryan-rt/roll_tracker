#!/usr/bin/env python3
"""CP7-pre-9: Disambiguate ambiguous_a_b and branch_b_persistent buckets.

Measures the TRUE recoverable Branch-B share by running CP7-pre-3's containment
test on the two suspect buckets from CP7-pre-8.

READ-ONLY diagnostic. No pipeline/config changes.

Usage:
    PYTHONPATH=src python tools/cp7_pre9_branchb_margin.py
"""

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
CLIP_ROOT = Path(
    "outputs/_eval_gt/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014"
)
EVAL_ROOT = Path("outputs/_eval/stage_d/bjj-detect-all-cameras/FP7oJQ")
PRE8_DIR = Path("outputs/_eval/_debug/cp7_pre8_axis1")
GT_ZIP_PATH = Path(
    "data/training_data/training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip"
)
OUT_DIR = Path("outputs/_eval/_debug/cp7_pre9_branchb_margin")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1920, 1080
TAU_HEADLINE = 0.5
TAU_SWEEP = [0.3, 0.5, 0.7, 0.9]
# Dense annotated range for FP7oJQ (from model manifest)
ANNOTATED_RANGE_START = 0
ANNOTATED_RANGE_END = 300  # inclusive, stride 1


def load_gt_boxes_from_zip(zip_path: Path) -> dict[int, list[dict]]:
    """Load all GT boxes from YOLO track-detection zip.

    Returns {frame_idx: [{x1, y1, x2, y2, track_id}, ...]}.
    """
    gt_by_frame: dict[int, list[dict]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt") or "labels" not in name:
                continue
            # Parse frame index from filename: frame_000123.txt
            stem = name.split("/")[-1].replace(".txt", "")
            if not stem.startswith("frame_"):
                continue
            frame_idx = int(stem.replace("frame_", ""))

            content = zf.read(name).decode().strip()
            if not content:
                gt_by_frame[frame_idx] = []
                continue

            boxes = []
            for line in content.split("\n"):
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                # Format: class x_center y_center width height track_id
                xc = float(parts[1]) * IMG_W
                yc = float(parts[2]) * IMG_H
                w = float(parts[3]) * IMG_W
                h = float(parts[4]) * IMG_H
                track_id = int(parts[5])
                boxes.append({
                    "x1": xc - w / 2,
                    "y1": yc - h / 2,
                    "x2": xc + w / 2,
                    "y2": yc + h / 2,
                    "track_id": track_id,
                })
            gt_by_frame[frame_idx] = boxes

    return gt_by_frame


def containment(det: dict, gt: dict) -> float:
    """Pre-3 containment: intersection(D, G') / area(G')."""
    ix1 = max(det["x1"], gt["x1"])
    iy1 = max(det["y1"], gt["y1"])
    ix2 = min(det["x2"], gt["x2"])
    iy2 = min(det["y2"], gt["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gt_area = (gt["x2"] - gt["x1"]) * (gt["y2"] - gt["y1"])
    if gt_area <= 0:
        return 0.0
    return inter / gt_area


def main():
    print("Loading artifacts...")

    # ── Load pre-8 signatures ──
    with open(PRE8_DIR / "part_3_signatures.json") as f:
        all_sigs = json.load(f)

    target_sigs = ["ambiguous_a_b", "branch_b_persistent"]
    passthrough_sigs = ["branch_a", "branch_b_swap", "other"]

    target_rows = [s for s in all_sigs if s["signature"] in target_sigs]
    passthrough_rows = [s for s in all_sigs if s["signature"] in passthrough_sigs]
    print(f"Target rows (ambiguous_a_b + branch_b_persistent): {len(target_rows)}")
    print(f"Passthrough rows: {len(passthrough_rows)}")

    # ── Load detections ──
    det_df = pd.read_parquet(CLIP_ROOT / "stage_A" / "detections.parquet")
    det_lookup = {}
    for _, d in det_df.iterrows():
        det_lookup[d["detection_id"]] = {
            "x1": float(d["x1"]),
            "y1": float(d["y1"]),
            "x2": float(d["x2"]),
            "y2": float(d["y2"]),
        }

    # ── Load GT trace for pred_detection_id + concurrent tracklet lookup ──
    trace = pd.read_parquet(EVAL_ROOT / "gt_person_trace.parquet")

    # Build frame-level index: {frame_idx: [{tracklet_id, final_person_id, pred_detection_id}, ...]}
    trace_by_frame: dict[int, list[dict]] = {}
    for _, r in trace[trace.frame_idx <= ANNOTATED_RANGE_END].iterrows():
        fidx = int(r.frame_idx)
        if fidx not in trace_by_frame:
            trace_by_frame[fidx] = []
        trace_by_frame[fidx].append({
            "gt_person_id": int(r.gt_person_id),
            "tracklet_id": r.tracklet_id,
            "final_person_id": r.final_person_id,
            "pred_detection_id": r.pred_detection_id if pd.notna(r.get("pred_detection_id")) else None,
        })

    # ── Load GT boxes ──
    print("Loading GT boxes from zip...")
    gt_by_frame = load_gt_boxes_from_zip(GT_ZIP_PATH)

    # Verify dense coverage in 0-300
    missing_frames = []
    for f in range(ANNOTATED_RANGE_START, ANNOTATED_RANGE_END + 1):
        if f not in gt_by_frame:
            missing_frames.append(f)
    print(f"GT frames in 0-300: {ANNOTATED_RANGE_END + 1 - len(missing_frames)}/{ANNOTATED_RANGE_END + 1}")
    if missing_frames:
        print(f"  WARNING: missing GT at frames: {missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}")

    # ── Load D1 graph nodes for concurrent_role test ──
    nodes = pd.read_parquet(CLIP_ROOT / "stage_D" / "d1_graph_nodes.parquet")
    group_nodes = nodes[nodes.node_type == "NodeType.GROUP_TRACKLET"]

    # Build GROUP node index: for each (tracklet_id, frame) -> list of GROUP nodes
    # where that tracklet is a role (carrier/disappear/new)
    # Also store role sets per node for same-node check
    group_node_roles: dict[str, dict] = {}  # node_id -> {carrier, disappear, new, start, end}
    tid_frame_to_group_nodes: dict[tuple[str, int], list[str]] = {}

    for _, gn in group_nodes.iterrows():
        nid = gn.node_id
        roles = set()
        carrier = gn.carrier_tracklet_id if pd.notna(gn.carrier_tracklet_id) else None
        disappear = gn.disappearing_tracklet_id if pd.notna(gn.disappearing_tracklet_id) else None
        new = gn.new_tracklet_id if pd.notna(gn.new_tracklet_id) else None
        if carrier:
            roles.add(carrier)
        if disappear:
            roles.add(disappear)
        if new:
            roles.add(new)

        group_node_roles[nid] = {
            "carrier": carrier,
            "disappear": disappear,
            "new": new,
            "start": int(gn.start_frame),
            "end": int(gn.end_frame),
            "roles_set": roles,
        }

        # Index each role tracklet at each frame in the node's span
        for tid in roles:
            for f in range(int(gn.start_frame), int(gn.end_frame) + 1):
                key = (tid, f)
                if key not in tid_frame_to_group_nodes:
                    tid_frame_to_group_nodes[key] = []
                tid_frame_to_group_nodes[key].append(nid)

    # ══════════════════════════════════════════════════════════════════════
    #  Run containment test on target rows
    # ══════════════════════════════════════════════════════════════════════

    def classify_row(row: dict, tau: float) -> dict:
        """Classify a single target row at given tau."""
        frame_idx = row["frame_idx"]
        gt_person_id = row["gt_person_id"]
        tracklet_id = row["tracklet_id"]
        canonical_pid = row["canonical_person_id"]

        result = {
            "frame_idx": frame_idx,
            "gt_person_id": gt_person_id,
            "tracklet_id": tracklet_id,
            "original_signature": row["signature"],
            "outcome": None,
            "max_other_containment": 0.0,
            "contained_gt_track_id": None,
            "concurrent_canonical_tracklet": None,
            "concurrent_is_group_role": None,
            "same_node": None,
        }

        # ── Check frame is in dense annotated range ──
        if frame_idx < ANNOTATED_RANGE_START or frame_idx > ANNOTATED_RANGE_END:
            result["outcome"] = "indeterminate"
            return result

        # ── Check GT coverage ──
        if frame_idx not in gt_by_frame or not gt_by_frame[frame_idx]:
            result["outcome"] = "indeterminate"
            return result

        # ── Get detection box ──
        # Find pred_detection_id from trace
        frame_trace = trace_by_frame.get(frame_idx, [])
        pred_det_id = None
        for tr in frame_trace:
            if tr["gt_person_id"] == gt_person_id:
                pred_det_id = tr["pred_detection_id"]
                break

        if pred_det_id is None or pred_det_id not in det_lookup:
            result["outcome"] = "indeterminate"
            return result

        det_box = det_lookup[pred_det_id]
        gt_boxes = gt_by_frame[frame_idx]

        # ── PRECEDENCE RULE: pair_box first (Correction 2) ──
        max_contain = 0.0
        max_contain_tid = None
        for gt in gt_boxes:
            if gt["track_id"] == gt_person_id:
                continue
            c = containment(det_box, gt)
            if c > max_contain:
                max_contain = c
                max_contain_tid = gt["track_id"]

        result["max_other_containment"] = round(max_contain, 4)

        if max_contain >= tau:
            result["outcome"] = "pair_box"
            result["contained_gt_track_id"] = max_contain_tid
            return result

        # ── concurrent_role check (Correction 3: reuse frozen identity mapping) ──
        # Find concurrent tracklet holding canonical_person_id
        concurrent_tid = None
        for tr in frame_trace:
            if tr["tracklet_id"] != tracklet_id and tr["final_person_id"] == canonical_pid:
                concurrent_tid = tr["tracklet_id"]
                break

        if concurrent_tid is not None:
            result["concurrent_canonical_tracklet"] = concurrent_tid

            # Strict same-node check (tightening):
            # Both misattrib tracklet AND canonical-holder must be roles in the SAME GROUP node
            misattr_groups = tid_frame_to_group_nodes.get((tracklet_id, frame_idx), [])
            concurrent_groups = tid_frame_to_group_nodes.get((concurrent_tid, frame_idx), [])

            # Find shared GROUP nodes
            shared_nodes = set(misattr_groups) & set(concurrent_groups)

            if shared_nodes:
                result["outcome"] = "concurrent_role"
                result["concurrent_is_group_role"] = True
                result["same_node"] = True
                return result

            # Loose check: concurrent tracklet is a GROUP role at this frame (any node)
            if concurrent_groups:
                result["concurrent_is_group_role"] = True
                result["same_node"] = False
                # Per tightening: tag but don't classify as concurrent_role
                # Fall through to single_person
            else:
                result["concurrent_is_group_role"] = False

        # ── single_person: genuine Axis-1, recoverable ──
        result["outcome"] = "single_person"
        return result

    print("\nRunning containment test (headline tau=0.5)...")
    headline_results = [classify_row(r, TAU_HEADLINE) for r in target_rows]

    # Save raw results
    with open(OUT_DIR / "containment_results.json", "w") as f:
        json.dump(headline_results, f, indent=2, default=str)

    # ── Per-bucket breakdown ──
    print("\n" + "=" * 72)
    print("PER-BUCKET OUTCOMES (tau=0.5)")
    print("=" * 72)

    for bucket in target_sigs:
        bucket_results = [r for r in headline_results if r["original_signature"] == bucket]
        total = len(bucket_results)
        outcomes = Counter(r["outcome"] for r in bucket_results)
        print(f"\n  {bucket} ({total} frames):")
        print(f"  {'Outcome':<22} {'Frames':>7} {'%':>7}")
        print(f"  {'-'*22} {'-'*7} {'-'*7}")
        for out in ["single_person", "pair_box", "concurrent_role", "indeterminate"]:
            n = outcomes.get(out, 0)
            pct = 100 * n / total if total else 0
            print(f"  {out:<22} {n:>7} {pct:>6.1f}%")

        # concurrent_role same_node split
        cr_rows = [r for r in bucket_results if r["outcome"] == "concurrent_role"]
        if cr_rows:
            same = sum(1 for r in cr_rows if r.get("same_node"))
            diff = len(cr_rows) - same
            print(f"    concurrent_role same_node=true: {same}, false: {diff}")

    # ── Tau sweep ──
    print("\n" + "=" * 72)
    print("TAU SWEEP STABILITY")
    print("=" * 72)

    sweep_data = []
    for tau in TAU_SWEEP:
        tau_results = [classify_row(r, tau) for r in target_rows]
        for bucket in target_sigs:
            bucket_results = [r for r in tau_results if r["original_signature"] == bucket]
            n_pair = sum(1 for r in bucket_results if r["outcome"] == "pair_box")
            sweep_data.append({
                "tau": tau,
                "bucket": bucket,
                "pair_box": n_pair,
                "total": len(bucket_results),
            })

    print(f"\n  {'tau':<6} {'ambig_pb':>10} {'bpers_pb':>10} {'total_pb':>10} {'% targets':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for tau in TAU_SWEEP:
        ambig_pb = next(s["pair_box"] for s in sweep_data if s["tau"] == tau and s["bucket"] == "ambiguous_a_b")
        bpers_pb = next(s["pair_box"] for s in sweep_data if s["tau"] == tau and s["bucket"] == "branch_b_persistent")
        total_pb = ambig_pb + bpers_pb
        pct = 100 * total_pb / len(target_rows) if target_rows else 0
        print(f"  {tau:<6.1f} {ambig_pb:>10} {bpers_pb:>10} {total_pb:>10} {pct:>9.1f}%")

    with open(OUT_DIR / "tau_sweep.json", "w") as f:
        json.dump(sweep_data, f, indent=2)

    # ── Corrected aggregation ──
    print("\n" + "=" * 72)
    print("CORRECTED GT-0-300 AGGREGATION")
    print("=" * 72)

    outcomes_all = Counter(r["outcome"] for r in headline_results)
    n_single = outcomes_all.get("single_person", 0)
    n_pair = outcomes_all.get("pair_box", 0)
    n_concurrent = outcomes_all.get("concurrent_role", 0)
    n_indet_target = outcomes_all.get("indeterminate", 0)

    n_branch_b_swap = sum(1 for s in all_sigs if s["signature"] == "branch_b_swap")
    n_branch_a = sum(1 for s in all_sigs if s["signature"] == "branch_a")
    n_other = sum(1 for s in all_sigs if s["signature"] == "other")
    total_all = len(all_sigs)

    true_branch_b = n_single + n_branch_b_swap
    scorable = total_all - n_indet_target

    categories = [
        ("true_branch_b", true_branch_b, "single_person from targets + branch_b_swap passthrough"),
        ("axis2_in_disguise", n_pair, "pair_box from targets"),
        ("ab_co_causation", n_concurrent, "concurrent_role (same GROUP node)"),
        ("pure_branch_a", n_branch_a, "passthrough"),
        ("other", n_other, "passthrough"),
        ("indeterminate", n_indet_target, "missing det/GT"),
    ]

    print(f"\n  Total misattributed frames: {total_all}")
    print(f"  Scorable (excl. indeterminate): {scorable}")
    print()
    print(f"  {'Category':<22} {'Frames':>7} {'% scorable':>11} {'% of 2259':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*11} {'-'*10}")
    for cat, n, note in categories:
        pct_scorable = 100 * n / scorable if scorable and cat != "indeterminate" else 0
        pct_all = 100 * n / total_all if total_all else 0
        if cat == "indeterminate":
            print(f"  {cat:<22} {n:>7} {'---':>11} {pct_all:>9.1f}%")
        else:
            print(f"  {cat:<22} {n:>7} {pct_scorable:>10.1f}% {pct_all:>9.1f}%")

    # ── TRUE Branch-B margin ──
    print(f"\n  TRUE BRANCH-B MARGIN: {true_branch_b}/{scorable} = "
          f"{100*true_branch_b/scorable:.1f}% of scorable frames")
    print(f"  (replaces the asserted 84.3% from CP7-pre-8)")

    # ── Cross-reference with pre-3 ──
    print("\n" + "=" * 72)
    print("CROSS-REFERENCE WITH CP7-PRE-3")
    print("=" * 72)

    print(f"""
  CP7-pre-3 (pre-CP-SPLIT-1 run state):
    Total misattributed: 2,765 (FP7oJQ)
    Under-segmentation:  1,950 (70.5%)

  CP7-pre-9 (post-CP-SPLIT-1 run state):
    Total misattributed: {total_all} (FP7oJQ)
    axis2_in_disguise (pair_box): {n_pair} ({100*n_pair/total_all:.1f}% of all, tested on {len(target_rows)} target rows)

  Run-state difference: CP-SPLIT-1 split tracklets at swap boundaries,
  changing the tracklet population. Pre-3 measured ALL misattributed frames;
  pre-9 measured only the ambiguous_a_b + branch_b_persistent buckets
  ({len(target_rows)} of {total_all} frames). The remaining {total_all - len(target_rows)} frames
  (branch_a={n_branch_a}, branch_b_swap={n_branch_b_swap}, other={n_other}) were not
  re-tested — pre-3's under-seg mass may reside partly there.

  Qualitative check: {"Material under-seg found in suspect buckets." if n_pair > len(target_rows) * 0.1 else "Under-seg is NOT the dominant mechanism in the suspect buckets." if n_pair < len(target_rows) * 0.05 else "Modest under-seg presence in suspect buckets."}
  This is a run-state difference, not a contradiction with pre-3.""")

    # ── Verdicts ──
    print("\n" + "=" * 72)
    print("VERDICTS")
    print("=" * 72)

    # Risk (1): Is ambiguous_a_b genuine co-causation or incidental?
    ambig_results = [r for r in headline_results if r["original_signature"] == "ambiguous_a_b"]
    ambig_concurrent = sum(1 for r in ambig_results if r["outcome"] == "concurrent_role")
    ambig_single = sum(1 for r in ambig_results if r["outcome"] == "single_person")
    ambig_pair = sum(1 for r in ambig_results if r["outcome"] == "pair_box")

    print(f"\n  Risk (1): Is ambiguous_a_b genuine co-causation or incidental GROUP overlap?")
    print(f"    concurrent_role (same GROUP node): {ambig_concurrent}/{len(ambig_results)}")
    print(f"    single_person (GROUP incidental):  {ambig_single}/{len(ambig_results)}")
    print(f"    pair_box (Axis-2):                 {ambig_pair}/{len(ambig_results)}")
    if ambig_concurrent < len(ambig_results) * 0.1:
        print(f"    VERDICT: Incidental. <10% genuine co-causation. GROUP overlap is")
        print(f"    a tiling artifact. pre-8's argument holds: ambiguous_a_b is functionally Branch B.")
    elif ambig_concurrent > len(ambig_results) * 0.3:
        print(f"    VERDICT: Material co-causation. Node design must address both GROUP")
        print(f"    routing AND concurrent swaps.")
    else:
        print(f"    VERDICT: Modest co-causation. Primarily Branch B but co-causation is non-negligible.")

    # Risk (2): Is there material under-seg hiding in branch_b_persistent?
    bpers_results = [r for r in headline_results if r["original_signature"] == "branch_b_persistent"]
    bpers_pair = sum(1 for r in bpers_results if r["outcome"] == "pair_box")

    print(f"\n  Risk (2): Is there material under-seg hiding in branch_b_persistent?")
    print(f"    pair_box (Axis-2):    {bpers_pair}/{len(bpers_results)} ({100*bpers_pair/len(bpers_results):.1f}%)")
    print(f"    single_person:        {sum(1 for r in bpers_results if r['outcome']=='single_person')}/{len(bpers_results)}")
    if bpers_pair > len(bpers_results) * 0.3:
        print(f"    VERDICT: YES. Material under-seg ({100*bpers_pair/len(bpers_results):.0f}%) hides")
        print(f"    in branch_b_persistent. A concurrent-swap node CANNOT recover these frames.")
        print(f"    The 84.3% headline from pre-8 was inflated by Axis-2 mass.")
    elif bpers_pair > len(bpers_results) * 0.1:
        print(f"    VERDICT: Moderate. Under-seg is present but not dominant.")
    else:
        print(f"    VERDICT: NO. Under-seg is minimal in branch_b_persistent.")
        print(f"    The concurrent-tracklet confusion is genuine Axis-1.")

    print(f"\n  TRUE BRANCH-B MARGIN: {true_branch_b}/{scorable} = "
          f"{100*true_branch_b/scorable:.1f}% of scorable misattributed frames.")
    print(f"  This replaces the asserted 84.3% from CP7-pre-8.")

    print(f"\n  STOP: Report at docs/cp7_pre9_branchb_margin.md.")
    print(f"  Node design returns to the web session.")


if __name__ == "__main__":
    main()
