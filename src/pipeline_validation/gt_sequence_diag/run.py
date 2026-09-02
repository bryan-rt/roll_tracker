"""CLI runner for GT-DIAG-1 / GT-VERIFY-1 / GT-VERIFY-2.

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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_validation.gt_sequence_diag.sequence import (
    build_sequence_table, _box_iou,
)
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


# ---------------------------------------------------------------------------
# GT engagement derivation via Stage E's own functions
# ---------------------------------------------------------------------------

def _build_gt_world_coords(pfm_path: Path, projection, matched_only: bool = False):
    """Build GT world-coordinate table from per_frame_matches.

    When matched_only=True, only include detection-matched GT rows (simulates
    what the pipeline would see). When False, include ALL GT rows.
    """
    from bjj_pipeline.stages.detect_track.quality import contact_point_from_bbox
    from bjj_pipeline.contracts.f0_projection import project_to_world

    pfm = pd.read_parquet(pfm_path)
    gt_rows = pfm[pfm["gt_track_id"].notna() & pfm["gt_x1"].notna()]
    if matched_only:
        gt_rows = gt_rows[gt_rows["match_status"] == "matched"]

    rows = []
    for _, r in gt_rows.iterrows():
        u, v, _, _ = contact_point_from_bbox(
            (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])
        )
        x_m, y_m = project_to_world(
            (u, v), projection.H, projection.camera_matrix,
            projection.dist_coefficients,
        )
        if np.isnan(x_m) or np.isnan(y_m):
            continue
        rows.append({
            "frame_index": int(r["frame_index"]),
            "person_id": f"gt{int(r['gt_track_id'])}",
            "x_m": x_m,
            "y_m": y_m,
        })

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["frame_index", "person_id"]).reset_index(drop=True)


def _derive_gt_engagements(pfm_path, projection, config, total_frames):
    """Derive GT engagements using Stage E's own functions on GT tracks."""
    from bjj_pipeline.stages.matches.pairing import compute_pair_distances
    from bjj_pipeline.stages.matches.hysteresis import run_proximity_hysteresis

    gt_all = _build_gt_world_coords(pfm_path, projection, matched_only=False)
    gt_matched = _build_gt_world_coords(pfm_path, projection, matched_only=True)

    end_frame = total_frames - 1

    def _run_engagement(gt_df):
        pair_dists = compute_pair_distances(gt_df)
        intervals = run_proximity_hysteresis(
            pair_dists,
            session_start_frame=0,
            session_end_frame=end_frame,
            **config,
        )
        return pair_dists, intervals

    pair_dists_all, intervals_all = _run_engagement(gt_all)
    pair_dists_matched, intervals_matched = _run_engagement(gt_matched)

    return {
        "gt_all": gt_all,
        "gt_matched": gt_matched,
        "pair_dists_all": pair_dists_all,
        "pair_dists_matched": pair_dists_matched,
        "intervals_all": intervals_all,
        "intervals_matched": intervals_matched,
        "config": config,
    }


def _engagement_config():
    """Production engagement thresholds from configs/default.yaml."""
    return {
        "engage_dist_m": 0.75,
        "disengage_dist_m": 2.0,
        "engage_min_frames": 15,
        "hysteresis_frames": 450,
        "min_clip_duration_frames": 150,
    }


# ---------------------------------------------------------------------------
# Recall-gating analysis
# ---------------------------------------------------------------------------

def _analyze_recall_gating(eng_result, pfm_path, total_frames):
    """For each GT engagement, report whether it survives on matched-only."""
    pfm = pd.read_parquet(pfm_path)
    gt_rows = pfm[pfm["gt_track_id"].notna() & pfm["gt_x1"].notna()]

    # Per-GT recall
    gt_recall = {}
    for gt_id in sorted(gt_rows["gt_track_id"].unique()):
        sub = gt_rows[gt_rows["gt_track_id"] == gt_id]
        total = len(sub.drop_duplicates("frame_index"))
        matched = len(sub[sub["match_status"] == "matched"].drop_duplicates("frame_index"))
        gt_recall[int(gt_id)] = {"matched": matched, "total": total,
                                  "recall": matched / total if total > 0 else 0.0}

    gt_all = eng_result["gt_all"]
    gt_matched = eng_result["gt_matched"]
    pair_dists_matched = eng_result["pair_dists_matched"]
    config = eng_result["config"]

    engaged_pairs = set()
    for iv in eng_result["intervals_all"]:
        engaged_pairs.add((iv.person_id_a, iv.person_id_b))

    results = []
    for iv in eng_result["intervals_all"]:
        pa, pb = iv.person_id_a, iv.person_id_b
        pa_id, pb_id = int(pa[2:]), int(pb[2:])

        # Co-detection
        frames_a_all = set(gt_all[gt_all["person_id"] == pa]["frame_index"])
        frames_b_all = set(gt_all[gt_all["person_id"] == pb]["frame_index"])
        frames_a_m = set(gt_matched[gt_matched["person_id"] == pa]["frame_index"])
        frames_b_m = set(gt_matched[gt_matched["person_id"] == pb]["frame_index"])
        co_all = frames_a_all & frames_b_all
        co_matched = frames_a_m & frames_b_m
        co_rate = len(co_matched) / len(co_all) if co_all else 0.0

        # Max consecutive co-detected frames below engage threshold
        sub_m = pair_dists_matched[
            (pair_dists_matched["person_id_a"] == pa)
            & (pair_dists_matched["person_id_b"] == pb)
        ].sort_values("frame_index")

        max_consec = 0
        if not sub_m.empty:
            close = (sub_m["dist_m"] < config["engage_dist_m"]).values
            fis = sub_m["frame_index"].values
            cur = 0
            for i, c in enumerate(close):
                if c and (i == 0 or fis[i] == fis[i - 1] + 1):
                    cur += 1
                    max_consec = max(max_consec, cur)
                elif c:
                    cur = 1
                    max_consec = max(max_consec, cur)
                else:
                    cur = 0

        # Did it survive?
        survived = any(
            iv2.person_id_a == pa and iv2.person_id_b == pb
            for iv2 in eng_result["intervals_matched"]
        )

        # Matched interval details
        matched_iv = None
        for iv2 in eng_result["intervals_matched"]:
            if iv2.person_id_a == pa and iv2.person_id_b == pb:
                matched_iv = iv2
                break

        results.append({
            "pair": f"{pa}<->{pb}",
            "gt_frames": f"{iv.start_frame}-{iv.end_frame}",
            "gt_duration": iv.end_frame - iv.start_frame,
            "recall_a": gt_recall[pa_id]["recall"],
            "recall_b": gt_recall[pb_id]["recall"],
            "co_presence": len(co_all),
            "co_detected": len(co_matched),
            "co_detection_rate": co_rate,
            "product_recall": gt_recall[pa_id]["recall"] * gt_recall[pb_id]["recall"],
            "max_consecutive_below_engage": max_consec,
            "engage_min_frames": config["engage_min_frames"],
            "survived_matched_only": survived,
            "matched_frames": f"{matched_iv.start_frame}-{matched_iv.end_frame}" if matched_iv else None,
        })

    return {"per_pair": results, "gt_recall": gt_recall}


# ---------------------------------------------------------------------------
# Threshold flapping analysis
# ---------------------------------------------------------------------------

def _analyze_threshold_flapping(eng_result):
    """For each GT engaged pair, analyze dist_m crossings above disengage threshold."""
    pair_dists = eng_result["pair_dists_all"]
    config = eng_result["config"]
    results = []

    for iv in eng_result["intervals_all"]:
        pa, pb = iv.person_id_a, iv.person_id_b
        sub = pair_dists[
            (pair_dists["person_id_a"] == pa)
            & (pair_dists["person_id_b"] == pb)
        ].sort_values("frame_index")

        dists = sub["dist_m"].values
        frames = sub["frame_index"].values

        below_engage = int(np.sum(dists < config["engage_dist_m"]))
        between = int(np.sum((dists >= config["engage_dist_m"]) & (dists <= config["disengage_dist_m"])))
        above_disengage = int(np.sum(dists > config["disengage_dist_m"]))

        # Runs above disengage threshold
        above = dists > config["disengage_dist_m"]
        runs = []
        run_len = 0
        run_start = None
        for i, a in enumerate(above):
            if a:
                if run_len == 0:
                    run_start = int(frames[i])
                run_len += 1
            else:
                if run_len > 0:
                    runs.append({"start_frame": run_start, "length": run_len})
                run_len = 0
        if run_len > 0:
            runs.append({"start_frame": run_start, "length": run_len})

        would_emit_disengagement = any(r["length"] >= config["hysteresis_frames"] for r in runs)

        results.append({
            "pair": f"{pa}<->{pb}",
            "n_frames": len(dists),
            "dist_mean": round(float(np.mean(dists)), 3),
            "dist_median": round(float(np.median(dists)), 3),
            "dist_max": round(float(np.max(dists)), 3),
            "below_engage": below_engage,
            "between": between,
            "above_disengage": above_disengage,
            "runs_above_disengage": len(runs),
            "max_run_length": max((r["length"] for r in runs), default=0),
            "would_emit_disengagement": would_emit_disengagement,
            "runs_detail": sorted(runs, key=lambda r: -r["length"])[:5],
        })

    return results


# ---------------------------------------------------------------------------
# Stage E evaluation against GT
# ---------------------------------------------------------------------------

def _evaluate_stage_e(eng_result, pipeline_dir, trace_path):
    """Map Stage E sessions to GT tracks, classify into three buckets."""
    sessions_path = pipeline_dir / "stage_E" / "match_sessions.jsonl"
    if not sessions_path.exists():
        return None

    sessions = []
    with open(sessions_path) as f:
        for line in f:
            sessions.append(json.loads(line))

    trace = pd.read_parquet(trace_path)

    # GT engaged pairs
    gt_engaged = set()
    for iv in eng_result["intervals_all"]:
        a_id = int(iv.person_id_a[2:])
        b_id = int(iv.person_id_b[2:])
        gt_engaged.add(frozenset({a_id, b_id}))

    results = []
    for s in sessions:
        pa, pb = s["person_id_a"], s["person_id_b"]
        sf, ef = s["start_frame"], s["end_frame"]

        trace_in_range = trace[(trace["frame_idx"] >= sf) & (trace["frame_idx"] <= ef)]

        gts_a = trace_in_range[trace_in_range["final_person_id"] == pa]["gt_person_id"].value_counts()
        gts_b = trace_in_range[trace_in_range["final_person_id"] == pb]["gt_person_id"].value_counts()

        gt_pairs_present = set()
        for ga in gts_a.index:
            for gb in gts_b.index:
                if ga != gb:
                    gt_pairs_present.add(frozenset({int(ga), int(gb)}))

        engaged_present = gt_pairs_present & gt_engaged

        dom_a = int(gts_a.index[0]) if len(gts_a) > 0 else None
        dom_b = int(gts_b.index[0]) if len(gts_b) > 0 else None
        dominant_pair = frozenset({dom_a, dom_b}) if dom_a is not None and dom_b is not None and dom_a != dom_b else None

        if dominant_pair is not None and dominant_pair in gt_engaged:
            category = "CORRECT_ENGAGED"
        elif engaged_present:
            category = "CONTAMINATED"
        else:
            category = "PHANTOM"

        # Weighted frame contribution for engaged pairs present
        engaged_weight = {}
        for ep in engaged_present:
            ep_list = sorted(ep)
            fa0 = set(trace_in_range[(trace_in_range["final_person_id"] == pa) & (trace_in_range["gt_person_id"] == ep_list[0])]["frame_idx"])
            fa1 = set(trace_in_range[(trace_in_range["final_person_id"] == pa) & (trace_in_range["gt_person_id"] == ep_list[1])]["frame_idx"])
            fb0 = set(trace_in_range[(trace_in_range["final_person_id"] == pb) & (trace_in_range["gt_person_id"] == ep_list[0])]["frame_idx"])
            fb1 = set(trace_in_range[(trace_in_range["final_person_id"] == pb) & (trace_in_range["gt_person_id"] == ep_list[1])]["frame_idx"])
            co_frames = (fa0 & fb1) | (fa1 & fb0)
            total_session = ef - sf + 1
            engaged_weight[f"GT{ep_list[0]}<->GT{ep_list[1]}"] = {
                "frames": len(co_frames),
                "total": total_session,
                "pct": round(100 * len(co_frames) / total_session, 1) if total_session > 0 else 0,
            }

        comp_a = {int(k): int(v) for k, v in gts_a.items()} if len(gts_a) > 0 else {}
        comp_b = {int(k): int(v) for k, v in gts_b.items()} if len(gts_b) > 0 else {}

        results.append({
            "person_id_a": pa,
            "person_id_b": pb,
            "start_frame": sf,
            "end_frame": ef,
            "duration": ef - sf,
            "dominant_gt_a": dom_a,
            "dominant_gt_b": dom_b,
            "category": category,
            "engaged_weight": engaged_weight,
            "composition_a": comp_a,
            "composition_b": comp_b,
        })

    from collections import Counter
    cats = Counter(r["category"] for r in results)
    gt_pairs_covered = set()
    for r in results:
        if r["category"] == "CORRECT_ENGAGED" and r["dominant_gt_a"] is not None:
            gt_pairs_covered.add(frozenset({r["dominant_gt_a"], r["dominant_gt_b"]}))

    return {
        "sessions": results,
        "summary": {
            "total": len(results),
            "correct_engaged": cats.get("CORRECT_ENGAGED", 0),
            "contaminated": cats.get("CONTAMINATED", 0),
            "phantom": cats.get("PHANTOM", 0),
            "gt_target": len(gt_engaged),
            "gt_pairs_covered": len(gt_pairs_covered),
            "gt_pairs_missing": [
                f"GT{sorted(p)[0]}<->GT{sorted(p)[1]}"
                for p in gt_engaged - gt_pairs_covered
            ],
        },
    }


# ---------------------------------------------------------------------------
# Partner-tolerant scoring
# ---------------------------------------------------------------------------

def _compute_partner_tolerant(trace_path, eng_result, total_frames):
    """Compute strict and partner-tolerant correct_id.

    Partner-tolerant: a misattribution between two GT tracks that are engaged
    with each other at that frame counts as correct.
    """
    trace = pd.read_parquet(trace_path)

    # Build per-frame engagement lookup: frame -> set of engaged GT pairs
    engaged_at_frame: dict[int, set] = {}
    for iv in eng_result["intervals_all"]:
        a_id = int(iv.person_id_a[2:])
        b_id = int(iv.person_id_b[2:])
        pair = frozenset({a_id, b_id})
        for f in range(iv.start_frame, iv.end_frame + 1):
            engaged_at_frame.setdefault(f, set()).add(pair)

    # Build person_id -> dominant GT mapping
    pid_gt = {}
    for pid in trace["final_person_id"].dropna().unique():
        sub = trace[trace["final_person_id"] == pid]
        gt_counts = sub["gt_person_id"].value_counts()
        pid_gt[pid] = int(gt_counts.index[0])

    strict_correct = 0
    tolerant_correct = 0
    total = len(trace)

    for _, row in trace.iterrows():
        fi = int(row["frame_idx"])
        gt_id = int(row["gt_person_id"])
        mode = row["failure_mode"]

        if mode == "present":
            strict_correct += 1
            tolerant_correct += 1
        elif mode == "present_misattributed":
            # Check: is the assigned person_id's dominant GT engaged with this GT?
            assigned_pid = row.get("final_person_id")
            if assigned_pid is not None and assigned_pid in pid_gt:
                assigned_gt = pid_gt[assigned_pid]
                pair = frozenset({gt_id, assigned_gt})
                frame_pairs = engaged_at_frame.get(fi, set())
                if pair in frame_pairs:
                    tolerant_correct += 1

    return {
        "strict_correct_id": strict_correct,
        "partner_tolerant_correct_id": tolerant_correct,
        "total": total,
        "strict_pct": round(100 * strict_correct / total, 1) if total > 0 else 0,
        "tolerant_pct": round(100 * tolerant_correct / total, 1) if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Occlusion partition
# ---------------------------------------------------------------------------

def _analyze_occlusion(trace_path, pfm_path, total_frames):
    """Partition stage_a_no_detection frames by GT-box overlap."""
    trace = pd.read_parquet(trace_path)
    pfm = pd.read_parquet(pfm_path)

    no_det = trace[trace["failure_mode"] == "stage_a_no_detection"].copy()
    if no_det.empty:
        return None

    # Build per-frame GT bbox lookup from pfm (all GT rows, not just matched)
    gt_by_frame: dict[int, dict[int, list]] = {}
    gt_rows = pfm[pfm["gt_track_id"].notna() & pfm["gt_x1"].notna()]
    for _, r in gt_rows.iterrows():
        fi = int(r["frame_index"])
        gt_id = int(r["gt_track_id"])
        bbox = [float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"])]
        gt_by_frame.setdefault(fi, {})[gt_id] = bbox

    # Per GT track stats
    per_gt = {}
    overlapping_total = 0
    no_overlap_total = 0
    in_group_total = 0
    not_in_group_total = 0

    # No-overlap characterization
    no_overlap_areas = []
    no_overlap_in_quad = 0
    no_overlap_out_quad = 0
    no_overlap_flicker = 0  # detected in adjacent frame
    no_overlap_sustained = 0  # not detected in either adjacent frame

    # Build set of detected frames per GT track
    detected_frames: dict[int, set] = {}
    for gt_id in trace["gt_person_id"].unique():
        det = trace[(trace["gt_person_id"] == gt_id) & (trace["failure_mode"] != "stage_a_no_detection")]
        detected_frames[int(gt_id)] = set(det["frame_idx"].values)

    for gt_id in sorted(no_det["gt_person_id"].unique()):
        gt_sub = no_det[no_det["gt_person_id"] == gt_id]
        gt_int = int(gt_id)
        overlapping = 0
        no_overlap = 0
        in_group_span = 0

        for _, row in gt_sub.iterrows():
            fi = int(row["frame_idx"])
            frame_bboxes = gt_by_frame.get(fi, {})
            my_bbox = frame_bboxes.get(gt_int)
            if my_bbox is None:
                continue

            # Check overlap with any other GT track at this frame
            has_overlap = False
            for other_id, other_bbox in frame_bboxes.items():
                if other_id == gt_int:
                    continue
                iou = _box_iou(my_bbox, other_bbox)
                if iou > 0:
                    has_overlap = True
                    break

            if has_overlap:
                overlapping += 1
            else:
                no_overlap += 1
                # Characterize
                area = (my_bbox[2] - my_bbox[0]) * (my_bbox[3] - my_bbox[1])
                no_overlap_areas.append(area)

                # Adjacent-frame detection (flicker vs sustained)
                det_set = detected_frames.get(gt_int, set())
                adjacent_detected = (fi - 1 in det_set) or (fi + 1 in det_set)
                if adjacent_detected:
                    no_overlap_flicker += 1
                else:
                    no_overlap_sustained += 1

            # GROUP span check
            node_types_raw = row.get("d1_node_types")
            if node_types_raw and isinstance(node_types_raw, str):
                try:
                    types = json.loads(node_types_raw)
                    if types and "GROUP" in types:
                        in_group_span += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        per_gt[gt_int] = {
            "total_no_det": len(gt_sub),
            "overlapping": overlapping,
            "no_overlap": no_overlap,
        }
        overlapping_total += overlapping
        no_overlap_total += no_overlap

    return {
        "total_no_detection": len(no_det),
        "overlapping": overlapping_total,
        "no_overlap": no_overlap_total,
        "per_gt": per_gt,
        "no_overlap_flicker": no_overlap_flicker,
        "no_overlap_sustained": no_overlap_sustained,
        "no_overlap_area_median": round(float(np.median(no_overlap_areas)), 1) if no_overlap_areas else None,
        "no_overlap_area_mean": round(float(np.mean(no_overlap_areas)), 1) if no_overlap_areas else None,
        "no_overlap_area_min": round(float(np.min(no_overlap_areas)), 1) if no_overlap_areas else None,
        "no_overlap_area_max": round(float(np.max(no_overlap_areas)), 1) if no_overlap_areas else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # GT-VERIFY-2: Engagement, scoring, occlusion
    # -----------------------------------------------------------------------
    config = _engagement_config()
    print(f"\nDeriving GT engagements (thresholds: {config})...")
    eng_result = _derive_gt_engagements(
        args.pfm, projection, config, args.total_frames,
    )

    print(f"  GT engagements (all GT frames): {len(eng_result['intervals_all'])}")
    gt_engaged_pairs = []
    for iv in eng_result["intervals_all"]:
        dur = iv.end_frame - iv.start_frame
        ps = " [partial_start]" if iv.partial_start else ""
        pe = " [partial_end]" if iv.partial_end else ""
        print(f"    {iv.person_id_a} <-> {iv.person_id_b}: "
              f"frames {iv.start_frame}-{iv.end_frame} ({dur}f){ps}{pe}")
        gt_engaged_pairs.append({
            "person_id_a": iv.person_id_a,
            "person_id_b": iv.person_id_b,
            "start_frame": iv.start_frame,
            "end_frame": iv.end_frame,
            "partial_start": iv.partial_start,
            "partial_end": iv.partial_end,
        })

    print(f"  GT engagements (matched-only): {len(eng_result['intervals_matched'])}")
    for iv in eng_result["intervals_matched"]:
        dur = iv.end_frame - iv.start_frame
        print(f"    {iv.person_id_a} <-> {iv.person_id_b}: "
              f"frames {iv.start_frame}-{iv.end_frame} ({dur}f)")

    # Recall-gating
    print("\nRecall-gating analysis...")
    recall_gating = _analyze_recall_gating(eng_result, args.pfm, args.total_frames)
    for r in recall_gating["per_pair"]:
        status = "SURVIVED" if r["survived_matched_only"] else "KILLED"
        print(f"  {r['pair']}: co-detected {r['co_detected']}/{r['co_presence']} "
              f"({r['co_detection_rate']:.1%}), "
              f"max {r['max_consecutive_below_engage']} consecutive <{config['engage_dist_m']}m "
              f"(need {config['engage_min_frames']}): {status}")
        if r["matched_frames"]:
            print(f"    matched interval: {r['matched_frames']} (GT: {r['gt_frames']})")

    # Threshold flapping
    print("\nThreshold flapping analysis...")
    flapping = _analyze_threshold_flapping(eng_result)
    for f_res in flapping:
        print(f"  {f_res['pair']}: {f_res['above_disengage']} frames >{config['disengage_dist_m']}m, "
              f"{f_res['runs_above_disengage']} runs, "
              f"max run {f_res['max_run_length']}f "
              f"(need {config['hysteresis_frames']} to disengage): "
              f"{'WOULD FLAP' if f_res['would_emit_disengagement'] else 'stable'}")

    # Stage E evaluation
    print("\nStage E evaluation against GT...")
    stage_e_eval = _evaluate_stage_e(eng_result, args.pipeline_dir, args.trace)
    if stage_e_eval:
        s = stage_e_eval["summary"]
        print(f"  Stage E sessions: {s['total']} (GT target: {s['gt_target']})")
        print(f"  CORRECT_ENGAGED: {s['correct_engaged']}")
        print(f"  CONTAMINATED: {s['contaminated']}")
        print(f"  PHANTOM: {s['phantom']}")
        print(f"  GT pairs covered: {s['gt_pairs_covered']}/{s['gt_target']}")
        if s["gt_pairs_missing"]:
            print(f"  Missing: {', '.join(s['gt_pairs_missing'])}")
    else:
        print("  No match_sessions.jsonl found")

    # Partner-tolerant scoring
    print("\nPartner-tolerant scoring...")
    tolerance = _compute_partner_tolerant(args.trace, eng_result, args.total_frames)
    print(f"  Strict correct_id:   {tolerance['strict_correct_id']}/{tolerance['total']} "
          f"({tolerance['strict_pct']}%)")
    print(f"  Partner-tolerant:    {tolerance['partner_tolerant_correct_id']}/{tolerance['total']} "
          f"({tolerance['tolerant_pct']}%)")
    delta = tolerance["tolerant_pct"] - tolerance["strict_pct"]
    print(f"  Delta: +{delta:.1f}pp")

    # Occlusion partition
    print("\nOcclusion partition...")
    occlusion = _analyze_occlusion(args.trace, args.pfm, args.total_frames)
    if occlusion:
        total_nd = occlusion["total_no_detection"]
        print(f"  Total no_detection: {total_nd}")
        print(f"  Overlapping (IoU>0 with another GT): {occlusion['overlapping']} "
              f"({100 * occlusion['overlapping'] / total_nd:.1f}%)")
        print(f"  No overlap: {occlusion['no_overlap']} "
              f"({100 * occlusion['no_overlap'] / total_nd:.1f}%)")
        print(f"    Flicker (detected in adjacent frame): {occlusion['no_overlap_flicker']}")
        print(f"    Sustained (not in adjacent): {occlusion['no_overlap_sustained']}")
        if occlusion["no_overlap_area_median"] is not None:
            print(f"    Area: median={occlusion['no_overlap_area_median']}, "
                  f"mean={occlusion['no_overlap_area_mean']}")
        print(f"\n  Per GT track:")
        for gt_id in sorted(occlusion["per_gt"].keys()):
            g = occlusion["per_gt"][gt_id]
            print(f"    GT {gt_id}: {g['total_no_det']} no_det, "
                  f"{g['overlapping']} overlapping, {g['no_overlap']} no-overlap")

    # Save all analysis results
    analysis = {
        "engagement_config": config,
        "gt_engagements": gt_engaged_pairs,
        "recall_gating": recall_gating,
        "threshold_flapping": flapping,
        "stage_e_evaluation": stage_e_eval,
        "partner_tolerant": tolerance,
        "occlusion": occlusion,
        "frame_based_threshold_note": (
            "hysteresis_frames=450 and min_clip_duration_frames=150 are frame-count-based. "
            "At ~15fps these are ~30s and ~10s; at 30fps they would be ~15s and ~5s. "
            "These should arguably be time-based (analogous to the variable-dt Kalman work) "
            "but are NOT changed here — one clip is not a basis, and the dominant cause of "
            "the 23-vs-3 session gap is upstream person_id fragmentation, not threshold tuning."
        ),
    }
    (args.output / "gt_verify_2_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nSaved analysis to {args.output / 'gt_verify_2_analysis.json'}")

    # Videos
    if args.video and args.video.exists() and not args.skip_videos:
        print("\nRendering annotated_gt.mp4...")
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
            gt_engagements=eng_result["intervals_all"],
            stage_e_sessions=stage_e_eval["sessions"] if stage_e_eval else None,
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
        on_mat = "ON MAT" if meta["on_mat"] else "OFF MAT"
        low = " *LOW" if meta["low_confidence"] else ""
        quad = f" [{meta.get('in_quad_pct', '?')}% quad]" if meta.get("in_quad_pct") is not None else ""
        bp = f" [{meta.get('in_blueprint_pct', '?')}% bp]" if meta.get("in_blueprint_pct") is not None else ""
        print(f"  GT {gt_id} ({on_mat}{quad}{bp}{low}): "
              f"{meta['gt_matched_frames']} matched/{meta['coverage_clip_pct']:.1f}% clip "
              f"({meta['coverage_presence_pct']:.1f}% presence), "
              f"area={meta['median_box_area']}, "
              f"{n_segs} segs, {n_tracklets} tracklets, {n_persons} persons, "
              f"{n_group} group segs")

    print("\nDone.")


if __name__ == "__main__":
    main()
