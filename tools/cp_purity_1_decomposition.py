"""CP-PURITY-1: Decomposition of tagged-athlete identity quality.

Measurement-only script. No production code changes.
Runs all 9 angles on both J_EDEw clips (per-clip and session-level).

Usage:
    PYTHONPATH=src python tools/cp_purity_1_decomposition.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline_validation.common.manifest import (
    enumerate_annotated_frames,
    load_manifest,
)
from pipeline_validation.signal_trace.greedy_matcher import greedy_match
from pipeline_validation.signal_trace.stage_a_census import _load_gt_all_annotated

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_purity_1"

VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_GT_TRACK = 24
VID2_GT_TRACK = 8
CAM_ID = "J_EDEw"
GYM_ID = "_eval_gt"
TAG_PERSON_ID = "p0022"  # Session-level tag:1 assignment from CP-TAG-4a

VID1_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID1_CLIP_ID
VID2_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID2_CLIP_ID
SESSION_DIR = OUTPUTS_DIR / GYM_ID / "sessions" / "2026-03-18" / "cp_tag_3_baseline"

# Vid2 frame offset in session space
VID2_FRAME_OFFSET = 4530

MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-v2.yaml"

# Proximity sweep (meters)
PROX_THRESHOLDS = {"tight": 0.5, "close": 1.0, "engage": 1.5, "loose": 2.0}
# Duration sweep (consecutive annotated frames at stride-10)
DUR_THRESHOLDS = {"instant": 3, "brief": 8, "sustained": 15, "long": 30}


# ---------------------------------------------------------------------------
# Helpers: World projection
# ---------------------------------------------------------------------------


def _load_projection():
    """Load J_EDEw homography (inverted to pixel->world) + lens params."""
    h_path = REPO_ROOT / "configs" / "cameras" / CAM_ID / "homography.json"
    with open(h_path) as f:
        payload = json.load(f)
    H_stored = np.array(payload["H"], dtype=np.float64)
    K = np.array(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(payload["dist_coefficients"], dtype=np.float64).ravel()
    # Stored H is world->pixel; invert for pixel->world
    H_inv = np.linalg.inv(H_stored)
    return H_inv, K, D


def _project_bbox_foot(bbox_xyxy, H_inv, K, D):
    """Project bbox bottom-center to world coordinates.

    Returns (x_m, y_m) or (nan, nan) on failure.
    """
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) / 2.0
    v = y2  # bottom center

    # Undistort
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    u2 = float(undistorted[0, 0, 0])
    v2 = float(undistorted[0, 0, 1])

    # Apply H_inv
    p = np.array([u2, v2, 1.0], dtype=np.float64)
    q = H_inv @ p
    w = q[2]
    if abs(w) < 1e-12:
        return (float("nan"), float("nan"))
    return (q[0] / w, q[1] / w)


# ---------------------------------------------------------------------------
# Helpers: Data loading
# ---------------------------------------------------------------------------


def _load_session_person_tracks() -> pd.DataFrame:
    """Load session-level person_tracks (both clips, unified person_ids)."""
    return pd.read_parquet(SESSION_DIR / "stage_D" / "person_tracks_J_EDEw.parquet")


def _load_clip_detections(clip_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")


def _load_split_audit(clip_dir: Path) -> dict[str, list[str]]:
    """Load d05_split_audit: original_tid -> [product_tids]."""
    audit_path = clip_dir / "stage_D" / "d05_split_audit.jsonl"
    split_map: dict[str, list[str]] = {}
    if not audit_path.exists():
        return split_map
    with open(audit_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("artifact_type") == "d05_split_event":
                orig = obj["original_tracklet_id"]
                new = obj["new_tracklet_id"]
                split_map.setdefault(orig, []).append(new)
    return split_map


def _load_d1_segments(clip_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(clip_dir / "stage_D" / "d1_segments.parquet")


def _load_match_sessions(clip_dir: Path) -> list[dict]:
    ms_path = clip_dir / "stage_E" / "match_sessions.jsonl"
    with open(ms_path) as f:
        return [json.loads(line) for line in f]


def _load_gt_for_clips():
    """Load GT boxes for both clips from manifest."""
    manifest = load_manifest(MANIFEST_PATH)
    gt_data = {}
    for exp in manifest.training_data:
        if exp.camera_id != CAM_ID:
            continue
        src = exp.source_video.replace(".mp4", "")
        clip_id = VID1_CLIP_ID if "200015" in src else VID2_CLIP_ID
        zip_path = REPO_ROOT / "data" / "training_data" / exp.export
        gt_by_frame = _load_gt_all_annotated(zip_path, exp)
        annotated_frames = sorted(enumerate_annotated_frames(exp))
        gt_data[clip_id] = {
            "gt_by_frame": gt_by_frame,
            "annotated_frames": annotated_frames,
            "export": exp,
        }
    return gt_data


# ---------------------------------------------------------------------------
# Helpers: GT-to-detection matching
# ---------------------------------------------------------------------------


def _match_gt_to_detections(gt_by_frame, det_df, annotated_frames, iou_threshold=0.3):
    """Match GT to pipeline detections on annotated frames.

    Returns list of dicts: {frame_index, gt_track_id, detection_id, tracklet_id, iou,
                            classification, gt_bbox, det_bbox}
    """
    records = []
    for fi in annotated_frames:
        gt_boxes_raw = gt_by_frame.get(fi, [])
        if not gt_boxes_raw:
            continue

        gt_tuples = [(b.x1, b.y1, b.x2, b.y2) for b in gt_boxes_raw]
        gt_track_ids = [b.track_id for b in gt_boxes_raw]

        frame_dets = det_df[det_df.frame_index == fi]
        if frame_dets.empty:
            for i, tid in enumerate(gt_track_ids):
                records.append({
                    "frame_index": fi,
                    "gt_track_id": tid,
                    "detection_id": None,
                    "tracklet_id": None,
                    "iou": 0.0,
                    "classification": "miss",
                    "gt_bbox": gt_tuples[i],
                    "det_bbox": None,
                })
            continue

        det_tuples = list(zip(
            frame_dets.x1.values, frame_dets.y1.values,
            frame_dets.x2.values, frame_dets.y2.values,
        ))
        det_ids = frame_dets.detection_id.values.tolist()
        det_tids = frame_dets.tracklet_id.values.tolist()

        matches = greedy_match(gt_tuples, det_tuples, iou_threshold=iou_threshold)

        # Build map: gt_idx -> (det_idx, iou)
        gt_matched = {}
        det_gt_count: Counter = Counter()
        for gt_idx, det_idx, iou in matches:
            gt_matched[gt_idx] = (det_idx, iou)
            det_gt_count[det_idx] += 1

        for gt_idx, tid in enumerate(gt_track_ids):
            if gt_idx not in gt_matched:
                records.append({
                    "frame_index": fi,
                    "gt_track_id": tid,
                    "detection_id": None,
                    "tracklet_id": None,
                    "iou": 0.0,
                    "classification": "miss",
                    "gt_bbox": gt_tuples[gt_idx],
                    "det_bbox": None,
                })
            else:
                det_idx, iou = gt_matched[gt_idx]
                n_sharing = det_gt_count[det_idx]
                classification = "pair_box" if n_sharing >= 2 else "tight_match"
                records.append({
                    "frame_index": fi,
                    "gt_track_id": tid,
                    "detection_id": det_ids[det_idx],
                    "tracklet_id": det_tids[det_idx],
                    "iou": iou,
                    "classification": classification,
                    "gt_bbox": gt_tuples[gt_idx],
                    "det_bbox": det_tuples[det_idx],
                })
    return records


# ---------------------------------------------------------------------------
# Helpers: Dominant person_id (reuses signal_trace logic)
# ---------------------------------------------------------------------------


def _build_det_frame_pids(person_tracks_df, clip_id):
    """Pre-build (detection_id, frame_index) -> [person_ids] lookup for a clip.

    Session person_tracks uses clip-namespaced tracklet_ids ({clip_id}:{tid}),
    but detection_ids are consistent between Stage A and session person_tracks.
    Join on (detection_id, frame_index) to avoid tracklet namespace issues.
    """
    clip_pt = person_tracks_df[person_tracks_df.clip_id == clip_id]
    det_frame_pids: dict[tuple, list[str]] = defaultdict(list)
    for det_id, fi, pid in zip(clip_pt.detection_id.values, clip_pt.frame_index.values, clip_pt.person_id.values):
        det_frame_pids[(det_id, int(fi))].append(pid)
    return det_frame_pids


def _lookup_pids_by_detection(detection_id, frame_index, frame_offset, det_frame_pids):
    """Look up person_ids for a detection at a frame."""
    if detection_id is None:
        return []
    session_fi = frame_index + frame_offset
    pids = det_frame_pids.get((detection_id, session_fi), [])
    if not pids:
        # Fallback: try clip-local frame
        pids = det_frame_pids.get((detection_id, frame_index), [])
    return pids


def _compute_dominant_pid_per_gt(match_records, det_frame_pids, frame_offset=0):
    """Compute majority-vote person_id per GT track from session person_tracks.

    Returns: {gt_track_id: dominant_person_id}
    """
    gt_pid_counts: dict[int, Counter] = defaultdict(Counter)
    for rec in match_records:
        if rec["detection_id"] is None:
            continue
        pids_found = _lookup_pids_by_detection(
            rec["detection_id"], rec["frame_index"], frame_offset, det_frame_pids
        )
        for pid in pids_found:
            gt_pid_counts[rec["gt_track_id"]][pid] += 1

    dominant = {}
    for gt_tid, counter in gt_pid_counts.items():
        if counter:
            dominant[gt_tid] = counter.most_common(1)[0][0]
    return dominant


# ---------------------------------------------------------------------------
# Angle 1: Matched metric re-baseline
# ---------------------------------------------------------------------------


def angle_1(gt_data, session_pt):
    """Correct_id re-baseline for tagged athletes + aggregate no-regression."""
    logger.info("=== Angle 1: Matched metric re-baseline ===")
    results = {}

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        det_frame_pids = _build_det_frame_pids(session_pt, clip_id)

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        dominant = _compute_dominant_pid_per_gt(
            match_records, det_frame_pids, clip_info["offset"]
        )

        # Per-GT-track correct_id
        per_track = {}
        for gt_tid in set(r["gt_track_id"] for r in match_records):
            track_recs = [r for r in match_records if r["gt_track_id"] == gt_tid]
            dom_pid = dominant.get(gt_tid)
            correct = 0
            wrong = 0
            no_id = 0
            no_det = 0
            for r in track_recs:
                if r["detection_id"] is None:
                    no_det += 1
                    continue
                # Get person_ids at this frame for this detection
                pids = _get_pids_at_frame(
                    r["detection_id"], r["frame_index"], clip_info["offset"],
                    det_frame_pids
                )
                if not pids:
                    no_id += 1
                elif dom_pid and dom_pid in pids:
                    correct += 1
                else:
                    wrong += 1

            total = len(track_recs)
            per_track[gt_tid] = {
                "correct_id": correct,
                "wrong_id": wrong,
                "no_id": no_id,
                "no_detection": no_det,
                "total": total,
                "correct_pct": correct / total * 100 if total else 0,
                "dominant_pid": dom_pid,
            }

        tagged_track = clip_info["gt_track"]
        tagged_result = per_track.get(tagged_track, {})

        # Aggregate across all tracks
        agg_correct = sum(t["correct_id"] for t in per_track.values())
        agg_total = sum(t["total"] for t in per_track.values())
        agg_pct = agg_correct / agg_total * 100 if agg_total else 0

        results[clip_id] = {
            "tagged_athlete": tagged_result,
            "aggregate_correct_id_pct": agg_pct,
            "aggregate_total_frames": agg_total,
            "n_gt_tracks": len(per_track),
            "per_track_summary": {
                str(k): {"correct_pct": v["correct_pct"], "dominant_pid": v["dominant_pid"]}
                for k, v in per_track.items()
            },
        }

        logger.info(
            f"  {clip_id}: tagged={tagged_result.get('correct_pct', 0):.1f}% correct_id, "
            f"aggregate={agg_pct:.1f}% (n={agg_total}, {len(per_track)} tracks)"
        )

    results["metric_definition"] = (
        "correct_id = GT frames where dominant_person_id (majority-vote from session "
        "person_tracks) is present in the frame's person_id set. Matcher: greedy IoU>=0.3. "
        "Dominant computed per-GT-track from session-level person_tracks_J_EDEw.parquet."
    )
    results["baseline_comparison"] = {
        "cp_tag_3": {"vid1_gt24": 25.6, "vid2_gt8": 22.2},
        "cp_tag_4a_verify": {"vid1_gt24": 17.6, "vid2_gt8": 19.1},
        "note": "CP-TAG-3/4a used per-clip person_tracks; this uses session-level. "
                "Expected reference: ~58.7% aggregate (signal-trace baseline).",
    }
    return results


def _get_pids_at_frame(detection_id, frame_index, frame_offset, det_frame_pids):
    """Get person_ids for a detection at a frame using pre-built lookup."""
    return _lookup_pids_by_detection(detection_id, frame_index, frame_offset, det_frame_pids)


# ---------------------------------------------------------------------------
# Angle 2: Tracklet purity distribution
# ---------------------------------------------------------------------------


def angle_2(gt_data, session_pt):
    """Tracklet purity: per tracklet/split-product, fraction of frames owned by majority GT."""
    logger.info("=== Angle 2: Tracklet purity distribution ===")
    results = {}

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # Group by tracklet_id: which GT tracks does each tracklet carry?
        tracklet_gt_counts: dict[str, Counter] = defaultdict(Counter)
        tracklet_total: dict[str, int] = defaultdict(int)
        for r in match_records:
            tid = r["tracklet_id"]
            if tid is None:
                continue
            tracklet_gt_counts[tid][r["gt_track_id"]] += 1
            tracklet_total[tid] += 1

        purities = []
        impure_tracklets = []
        for tid, counter in tracklet_gt_counts.items():
            total = tracklet_total[tid]
            majority_count = counter.most_common(1)[0][1]
            purity = majority_count / total if total else 0
            majority_gt = counter.most_common(1)[0][0]
            purities.append(purity)
            if purity < 0.8:
                impure_tracklets.append({
                    "tracklet_id": tid,
                    "purity": round(purity, 3),
                    "majority_gt": majority_gt,
                    "total_gt_frames": total,
                    "gt_distribution": dict(counter),
                })

        # Histogram (0.0-1.0, 10 bins)
        hist, bin_edges = np.histogram(purities, bins=10, range=(0.0, 1.0))

        results[clip_id] = {
            "n_tracklets_measured": len(purities),
            "mean_purity": float(np.mean(purities)) if purities else 0,
            "median_purity": float(np.median(purities)) if purities else 0,
            "histogram": {f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(hist[i]) for i in range(10)},
            "n_impure_below_0.8": len(impure_tracklets),
            "impure_tracklets": sorted(impure_tracklets, key=lambda x: x["purity"])[:20],
        }
        logger.info(
            f"  {clip_id}: {len(purities)} tracklets, mean purity={np.mean(purities):.3f}, "
            f"{len(impure_tracklets)} impure (<0.8)"
        )

    results["metric_definition"] = (
        "Tracklet purity = for a tracklet, fraction of its GT-annotated detection frames "
        "whose GT-matched track is the tracklet's plurality GT track. "
        "Matcher: greedy IoU>=0.3."
    )
    return results


# ---------------------------------------------------------------------------
# Angle 3: D0.5 help or hurt
# ---------------------------------------------------------------------------


def angle_3(gt_data):
    """Compare purity of pre-split tracklets vs their post-split products."""
    logger.info("=== Angle 3: D0.5 split impact on purity ===")
    results = {}

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR}),
        (VID2_CLIP_ID, {"dir": VID2_DIR}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        split_map = _load_split_audit(clip_info["dir"])

        if not split_map:
            results[clip_id] = {"n_splits": 0, "note": "No splits in this clip"}
            logger.info(f"  {clip_id}: No splits")
            continue

        # Load tracklet_bank_summaries for product frame ranges
        bank_path = clip_info["dir"] / "stage_D" / "tracklet_bank_summaries.parquet"
        bank_df = pd.read_parquet(bank_path)
        # Build product -> (first_frame, last_frame)
        product_ranges: dict[str, tuple[int, int]] = {}
        for _, row in bank_df.iterrows():
            product_ranges[row.tracklet_id] = (int(row.start_frame), int(row.end_frame))

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # For each split: partition the original tracklet's frames by product ranges
        comparisons = []
        for orig_tid, products in split_map.items():
            # Get all GT-matched frames for this original tracklet
            orig_frames = [r for r in match_records if r["tracklet_id"] == orig_tid]
            if not orig_frames:
                continue

            # Original purity (unsplit)
            orig_gt_counts: Counter = Counter()
            for r in orig_frames:
                orig_gt_counts[r["gt_track_id"]] += 1
            orig_total = len(orig_frames)
            orig_purity = orig_gt_counts.most_common(1)[0][1] / orig_total

            # Partition frames by product ranges
            product_gt_counts: dict[str, Counter] = {p: Counter() for p in products}
            product_totals: dict[str, int] = {p: 0 for p in products}

            for r in orig_frames:
                fi = r["frame_index"]
                assigned = False
                for p in products:
                    if p in product_ranges:
                        pf, pl = product_ranges[p]
                        if pf <= fi <= pl:
                            product_gt_counts[p][r["gt_track_id"]] += 1
                            product_totals[p] += 1
                            assigned = True
                            break
                # Frame might not fall in any product range (gap at split boundary)

            prod_purities = []
            for p in products:
                pt = product_totals[p]
                if pt > 0:
                    pp = product_gt_counts[p].most_common(1)[0][1] / pt
                    prod_purities.append(pp)

            if not prod_purities:
                continue

            # Weighted average product purity
            total_assigned = sum(product_totals[p] for p in products)
            if total_assigned == 0:
                continue
            weighted_purity = sum(
                product_gt_counts[p].most_common(1)[0][1]
                for p in products if product_totals[p] > 0
            ) / total_assigned

            comparisons.append({
                "original_tid": orig_tid,
                "n_products": len(products),
                "original_purity": round(orig_purity, 3),
                "product_purities": [round(p, 3) for p in prod_purities],
                "weighted_product_purity": round(weighted_purity, 3),
                "delta": round(weighted_purity - orig_purity, 3),
                "helped": weighted_purity > orig_purity + 0.01,
                "hurt": weighted_purity < orig_purity - 0.01,
                "orig_gt_frames": orig_total,
            })

        helped = sum(1 for c in comparisons if c["helped"])
        hurt = sum(1 for c in comparisons if c["hurt"])
        neutral = len(comparisons) - helped - hurt

        results[clip_id] = {
            "n_splits": len(split_map),
            "n_measured": len(comparisons),
            "helped": helped,
            "hurt": hurt,
            "neutral": neutral,
            "mean_delta": float(np.mean([c["delta"] for c in comparisons])) if comparisons else 0,
            "comparisons": sorted(comparisons, key=lambda x: x["delta"])[:10]
            + sorted(comparisons, key=lambda x: -x["delta"])[:10],
        }
        logger.info(
            f"  {clip_id}: {len(split_map)} splits, {len(comparisons)} measured, "
            f"{helped} helped, {hurt} hurt, {neutral} neutral"
        )

    results["metric_definition"] = (
        "Compares purity of pre-split parent tracklet (all GT-matched frames on detections.parquet) "
        "vs weighted-average purity when partitioned by D0.5 product frame ranges from "
        "tracklet_bank_summaries.parquet. Delta > 0 = split helped purity."
    )
    return results


# ---------------------------------------------------------------------------
# Angle 4: Entity purity distribution
# ---------------------------------------------------------------------------


def angle_4(gt_data, session_pt):
    """Per person_id purity: fraction of GT-matched frames owned by majority GT track."""
    logger.info("=== Angle 4: Entity purity distribution ===")

    # Build: for each person_id + frame, which GT track is the detection matched to?
    # We need to join session_pt -> detections -> GT match

    all_pid_gt_counts: dict[str, Counter] = defaultdict(Counter)
    all_pid_totals: dict[str, int] = defaultdict(int)
    all_pid_frame_counts: dict[str, int] = defaultdict(int)

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # Build detection_id -> gt_track_id at each frame (for matched detections)
        det_to_gt: dict[tuple, int] = {}  # (detection_id, frame_index) -> gt_track_id
        for r in match_records:
            if r["detection_id"] is not None:
                det_to_gt[(r["detection_id"], r["frame_index"])] = r["gt_track_id"]

        # Filter session_pt to this clip's annotated frames using vectorized ops
        clip_pt = session_pt[session_pt.clip_id == clip_id]
        annotated_set = set(gt["annotated_frames"])
        offset = clip_info["offset"]

        clip_frames = clip_pt.frame_index.values - offset
        mask = np.isin(clip_frames, list(annotated_set))
        clip_pt_annotated = clip_pt[mask]

        for det_id, fi_session, pid in zip(
            clip_pt_annotated.detection_id.values,
            clip_pt_annotated.frame_index.values,
            clip_pt_annotated.person_id.values,
        ):
            clip_frame = int(fi_session) - offset
            key = (det_id, clip_frame)
            if key in det_to_gt:
                gt_tid = det_to_gt[key]
                all_pid_gt_counts[pid][gt_tid] += 1
                all_pid_totals[pid] += 1
            all_pid_frame_counts[pid] += 1

    # Compute purity per person_id
    entity_purities = {}
    for pid, counter in all_pid_gt_counts.items():
        total = all_pid_totals[pid]
        if total == 0:
            continue
        majority = counter.most_common(1)[0]
        purity = majority[1] / total
        entity_purities[pid] = {
            "purity": round(purity, 3),
            "majority_gt": majority[0],
            "gt_matched_frames": total,
            "total_session_frames": all_pid_frame_counts.get(pid, 0),
            "gt_distribution": dict(counter),
        }

    purities_list = [v["purity"] for v in entity_purities.values()]
    hist, bin_edges = np.histogram(purities_list, bins=10, range=(0.0, 1.0))

    # Highlight tagged entity
    tagged_entity = entity_purities.get(TAG_PERSON_ID, {})

    results = {
        "n_entities": len(entity_purities),
        "mean_purity": float(np.mean(purities_list)) if purities_list else 0,
        "median_purity": float(np.median(purities_list)) if purities_list else 0,
        "histogram": {f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(hist[i]) for i in range(10)},
        "tagged_entity_p0022": tagged_entity,
        "diluted_entities": {
            pid: info for pid, info in sorted(entity_purities.items(), key=lambda x: x[1]["purity"])
            if info["purity"] < 0.5
        },
        "metric_definition": (
            "Entity purity = per person_id, fraction of GT-annotated frames where the "
            "detection's GT match is the entity's majority GT track. Session-level."
        ),
    }
    logger.info(
        f"  {len(entity_purities)} entities, mean purity={np.mean(purities_list):.3f}, "
        f"p0022 purity={tagged_entity.get('purity', 'N/A')}"
    )
    return results


# ---------------------------------------------------------------------------
# Angle 5: Through-line integrity
# ---------------------------------------------------------------------------


def angle_5(gt_data, session_pt):
    """Trace tagged athlete's through-line: does p0022 follow GT track 24/8?"""
    logger.info("=== Angle 5: Through-line integrity ===")

    # For each GT frame of the tagged athlete, find which person_id(s) the detection carries
    timeline = []  # [{session_frame, clip_id, clip_frame, person_ids, gt_track_id}]

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        tagged_gt = clip_info["gt_track"]
        det_frame_pids = _build_det_frame_pids(session_pt, clip_id)

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # Filter to tagged athlete's frames
        tagged_recs = [r for r in match_records if r["gt_track_id"] == tagged_gt]

        for r in tagged_recs:
            pids = _get_pids_at_frame_fast(
                r["detection_id"], r["frame_index"], clip_info["offset"],
                det_frame_pids
            )
            timeline.append({
                "session_frame": r["frame_index"] + clip_info["offset"],
                "clip_id": clip_id,
                "clip_frame": r["frame_index"],
                "person_ids": pids,
                "has_detection": r["detection_id"] is not None,
                "classification": r["classification"],
            })

    timeline.sort(key=lambda x: x["session_frame"])

    # Analyze through-line: track the "active" dominant pid
    # A teleport = the dominant pid at frame N is different from frame N-1
    # (excluding frames with no detection or no person_id)
    teleport_events = []
    prev_pid = None
    identity_sequence = []  # list of (pid, start_frame, end_frame)

    for entry in timeline:
        if not entry["has_detection"] or not entry["person_ids"]:
            continue

        # Which pid is "active" — prefer TAG_PERSON_ID if present, else first
        if TAG_PERSON_ID in entry["person_ids"]:
            active = TAG_PERSON_ID
        else:
            active = entry["person_ids"][0]  # arbitrary

        if prev_pid is not None and active != prev_pid:
            teleport_events.append({
                "session_frame": entry["session_frame"],
                "clip_id": entry["clip_id"],
                "from_pid": prev_pid,
                "to_pid": active,
                "person_ids_at_frame": entry["person_ids"],
            })
            if identity_sequence:
                identity_sequence[-1]["end_frame"] = entry["session_frame"]
            identity_sequence.append({
                "pid": active,
                "start_frame": entry["session_frame"],
                "end_frame": None,
            })
        elif prev_pid is None:
            identity_sequence.append({
                "pid": active,
                "start_frame": entry["session_frame"],
                "end_frame": None,
            })

        prev_pid = active

    if identity_sequence:
        identity_sequence[-1]["end_frame"] = timeline[-1]["session_frame"]

    # Count frames where TAG_PERSON_ID is present vs absent
    frames_with_tag_pid = sum(
        1 for t in timeline if t["has_detection"] and TAG_PERSON_ID in t.get("person_ids", [])
    )
    frames_with_detection = sum(1 for t in timeline if t["has_detection"])
    frames_with_any_pid = sum(
        1 for t in timeline if t["has_detection"] and t.get("person_ids")
    )

    results = {
        "total_gt_frames": len(timeline),
        "frames_with_detection": frames_with_detection,
        "frames_with_any_pid": frames_with_any_pid,
        "frames_with_tag_pid": frames_with_tag_pid,
        "tag_pid_coverage_pct": frames_with_tag_pid / frames_with_detection * 100 if frames_with_detection else 0,
        "n_teleport_events": len(teleport_events),
        "identity_sequence_length": len(identity_sequence),
        "identity_sequence": identity_sequence[:30],  # truncate for readability
        "teleport_events": teleport_events[:30],
        "metric_definition": (
            "Through-line: traces GT track 24 (vid1) / 8 (vid2) frame-by-frame through "
            "session person_tracks. Teleport = active person_id changes between consecutive "
            "GT frames. TAG_PERSON_ID (p0022) preferred when present in frame's pid set."
        ),
    }
    logger.info(
        f"  {len(timeline)} GT frames, {frames_with_tag_pid}/{frames_with_detection} "
        f"have p0022, {len(teleport_events)} teleports, "
        f"{len(identity_sequence)} identity segments"
    )
    return results


def _get_pids_at_frame_fast(detection_id, frame_index, frame_offset, det_frame_pids):
    """Fast pid lookup using pre-built detection dict."""
    return _lookup_pids_by_detection(detection_id, frame_index, frame_offset, det_frame_pids)


# ---------------------------------------------------------------------------
# Angle 6: Intra-match vs cross-match decomposition
# ---------------------------------------------------------------------------


def angle_6(gt_data, session_pt):
    """Sweep GT match windows and decompose non-self frames into intra/cross-match."""
    logger.info("=== Angle 6: Intra-match vs cross-match decomposition ===")

    H_inv, K, D = _load_projection()

    # Step 1: Compute world positions for ALL GT boxes at annotated frames
    gt_world_positions: dict[str, dict[int, dict[int, tuple]]] = {}  # clip -> frame -> gt_track -> (x,y)

    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        gt = gt_data[clip_id]
        gt_world_positions[clip_id] = {}
        n_projected = 0
        n_fallback = 0
        for fi, boxes in gt["gt_by_frame"].items():
            gt_world_positions[clip_id][fi] = {}
            for b in boxes:
                wx, wy = _project_bbox_foot((b.x1, b.y1, b.x2, b.y2), H_inv, K, D)
                if np.isnan(wx):
                    n_fallback += 1
                else:
                    n_projected += 1
                gt_world_positions[clip_id][fi][b.track_id] = (wx, wy)
        logger.info(f"  {clip_id}: {n_projected} projected, {n_fallback} fallback")

    # Step 2: For each proximity x duration threshold, define GT match windows
    # between the tagged athlete and every other GT track
    tagged_tracks = {VID1_CLIP_ID: VID1_GT_TRACK, VID2_CLIP_ID: VID2_GT_TRACK}

    sweep_results = {}
    for prox_label, prox_thresh in PROX_THRESHOLDS.items():
        for dur_label, dur_thresh in DUR_THRESHOLDS.items():
            key = f"{prox_label}_{dur_label}"

            # Find GT match windows for tagged athlete
            gt_match_windows = _find_gt_match_windows(
                gt_data, gt_world_positions, tagged_tracks, prox_thresh, dur_thresh
            )

            # Build frame-level lookup: which GT tracks are "in match" with tagged at each frame
            in_match_frames = _build_in_match_lookup(gt_match_windows)

            # Score the tagged entity's non-self frames
            intra, cross, self_frames, no_det, no_pid = _score_entity_frames(
                gt_data, session_pt, tagged_tracks, in_match_frames
            )

            total_non_self = intra + cross
            sweep_results[key] = {
                "proximity_m": prox_thresh,
                "duration_frames": dur_thresh,
                "n_gt_match_windows": sum(len(v) for v in gt_match_windows.values()),
                "self_frames": self_frames,
                "intra_match": intra,
                "cross_match": cross,
                "no_detection": no_det,
                "no_pid": no_pid,
                "intra_pct": intra / total_non_self * 100 if total_non_self else 0,
                "cross_pct": cross / total_non_self * 100 if total_non_self else 0,
            }

    # Assess stability
    intra_pcts = [v["intra_pct"] for v in sweep_results.values()]
    stable = (max(intra_pcts) - min(intra_pcts)) < 15 if intra_pcts else True

    results = {
        "sweep_grid": sweep_results,
        "stability_assessment": {
            "verdict_stable": stable,
            "intra_pct_range": [round(min(intra_pcts), 1), round(max(intra_pcts), 1)] if intra_pcts else [],
            "note": "Stable = intra% varies <15pp across grid" if stable else
                    "UNSTABLE: verdict is threshold-sensitive, requires confirmation",
        },
        "metric_definition": (
            "GT match window = sustained interval where tagged athlete's GT track and "
            "another GT track have world-distance <= threshold for >= duration consecutive "
            "annotated frames. Intra-match = non-self frame on entity where the GT person IS "
            "in a GT match window with tagged athlete. Cross-match = NOT engaged."
        ),
    }
    logger.info(f"  Sweep complete: stable={stable}, intra range={results['stability_assessment']['intra_pct_range']}")
    return results


def _find_gt_match_windows(gt_data, gt_world_positions, tagged_tracks, prox_thresh, dur_thresh):
    """Find GT match windows between tagged athlete and all other GT tracks."""
    # Returns: {clip_id: [{opponent_gt, start_frame, end_frame}]}
    windows: dict[str, list[dict]] = {}

    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        windows[clip_id] = []
        gt = gt_data[clip_id]
        tagged_gt = tagged_tracks[clip_id]
        frames = sorted(gt["annotated_frames"])
        world_pos = gt_world_positions[clip_id]

        # Get all opponent GT tracks
        all_tracks = set()
        for boxes in gt["gt_by_frame"].values():
            for b in boxes:
                all_tracks.add(b.track_id)
        opponents = all_tracks - {tagged_gt}

        for opp in opponents:
            # Find consecutive runs where distance <= prox_thresh
            run_start = None
            run_length = 0

            for i, fi in enumerate(frames):
                pos_fi = world_pos.get(fi, {})
                tagged_pos = pos_fi.get(tagged_gt)
                opp_pos = pos_fi.get(opp)

                in_proximity = False
                if tagged_pos and opp_pos:
                    tx, ty = tagged_pos
                    ox, oy = opp_pos
                    if not (np.isnan(tx) or np.isnan(ox)):
                        dist = np.sqrt((tx - ox) ** 2 + (ty - oy) ** 2)
                        in_proximity = dist <= prox_thresh

                if in_proximity:
                    if run_start is None:
                        run_start = fi
                    run_length += 1
                else:
                    if run_length >= dur_thresh:
                        windows[clip_id].append({
                            "opponent_gt": opp,
                            "start_frame": run_start,
                            "end_frame": frames[i - 1] if i > 0 else fi,
                        })
                    run_start = None
                    run_length = 0

            # Close trailing run
            if run_length >= dur_thresh:
                windows[clip_id].append({
                    "opponent_gt": opp,
                    "start_frame": run_start,
                    "end_frame": frames[-1],
                })

    return windows


def _build_in_match_lookup(gt_match_windows):
    """Pass-through: scorer uses windows directly with set-based lookup."""
    return gt_match_windows


def _score_entity_frames(gt_data, session_pt, tagged_tracks, gt_match_windows):
    """Score p0022's entity frames: self, intra-match, cross-match."""
    intra = 0
    cross = 0
    self_frames = 0
    no_det = 0
    no_pid = 0

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        tagged_gt = clip_info["gt_track"]
        offset = clip_info["offset"]

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # Build detection_id -> gt_track_id map
        det_to_gt: dict[tuple, int] = {}
        for r in match_records:
            if r["detection_id"]:
                det_to_gt[(r["detection_id"], r["frame_index"])] = r["gt_track_id"]

        # Get p0022's frames in this clip from session person_tracks
        clip_pt = session_pt[
            (session_pt.clip_id == clip_id) & (session_pt.person_id == TAG_PERSON_ID)
        ]

        # Only score at annotated frames
        annotated_set = set(gt["annotated_frames"])
        windows = gt_match_windows.get(clip_id, [])

        # Build window lookup: (opponent_gt, frame) -> bool for speed
        window_lookup: set[tuple] = set()
        for w in windows:
            for fi in gt["annotated_frames"]:
                if w["start_frame"] <= fi <= w["end_frame"]:
                    window_lookup.add((w["opponent_gt"], fi))

        for det_id, fi_session in zip(clip_pt.detection_id.values, clip_pt.frame_index.values):
            clip_frame = int(fi_session) - offset
            if clip_frame not in annotated_set:
                continue

            key = (det_id, clip_frame)
            if key not in det_to_gt:
                no_det += 1
                continue

            gt_track = det_to_gt[key]
            if gt_track == tagged_gt:
                self_frames += 1
            else:
                if (gt_track, clip_frame) in window_lookup:
                    intra += 1
                else:
                    cross += 1

    return intra, cross, self_frames, no_det, no_pid


def _is_in_match(opponent_gt, frame_index, windows):
    """Check if opponent_gt is in a GT match window at frame_index (fallback)."""
    for w in windows:
        if w["opponent_gt"] == opponent_gt and w["start_frame"] <= frame_index <= w["end_frame"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Angle 7: Match window recovery
# ---------------------------------------------------------------------------


def angle_7(gt_data, session_pt):
    """Compare Stage E match sessions (on p0022) to GT match windows."""
    logger.info("=== Angle 7: Match window recovery ===")

    H_inv, K, D = _load_projection()
    gt_world_positions: dict[str, dict[int, dict[int, tuple]]] = {}
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        gt = gt_data[clip_id]
        gt_world_positions[clip_id] = {}
        for fi, boxes in gt["gt_by_frame"].items():
            gt_world_positions[clip_id][fi] = {}
            for b in boxes:
                wx, wy = _project_bbox_foot((b.x1, b.y1, b.x2, b.y2), H_inv, K, D)
                gt_world_positions[clip_id][fi][b.track_id] = (wx, wy)

    tagged_tracks = {VID1_CLIP_ID: VID1_GT_TRACK, VID2_CLIP_ID: VID2_GT_TRACK}

    # Use "engage" + "sustained" as the reference threshold for window recovery
    gt_match_windows = _find_gt_match_windows(
        gt_data, gt_world_positions, tagged_tracks,
        prox_thresh=PROX_THRESHOLDS["engage"],
        dur_thresh=DUR_THRESHOLDS["sustained"],
    )

    # Load Stage E sessions involving p0022
    stage_e_sessions = []
    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "offset": VID2_FRAME_OFFSET}),
    ]:
        sessions = _load_match_sessions(clip_info["dir"])
        for s in sessions:
            if s.get("person_id_a") == TAG_PERSON_ID or s.get("person_id_b") == TAG_PERSON_ID:
                stage_e_sessions.append({
                    "clip_id": clip_id,
                    "match_id": s["match_id"],
                    "start_frame": s["start_frame"],
                    "end_frame": s["end_frame"],
                    "opponent_pid": s["person_id_b"] if s["person_id_a"] == TAG_PERSON_ID else s["person_id_a"],
                    "session_start": s["start_frame"] + clip_info["offset"],
                    "session_end": s["end_frame"] + clip_info["offset"],
                })

    # Match GT windows to Stage E sessions (overlap-based)
    all_gt_windows = []
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        offset = 0 if clip_id == VID1_CLIP_ID else VID2_FRAME_OFFSET
        for w in gt_match_windows.get(clip_id, []):
            all_gt_windows.append({
                **w,
                "clip_id": clip_id,
                "session_start": w["start_frame"] + offset,
                "session_end": w["end_frame"] + offset,
            })

    # For each GT window, check if any Stage E session overlaps
    gt_recovered = 0
    gt_missed = 0
    gt_window_details = []
    for gw in all_gt_windows:
        overlapping = [
            s for s in stage_e_sessions
            if s["session_start"] <= gw["session_end"] and s["session_end"] >= gw["session_start"]
        ]
        recovered = len(overlapping) > 0
        if recovered:
            gt_recovered += 1
        else:
            gt_missed += 1
        gt_window_details.append({
            "clip_id": gw["clip_id"],
            "opponent_gt": gw["opponent_gt"],
            "start_frame": gw["start_frame"],
            "end_frame": gw["end_frame"],
            "recovered": recovered,
            "n_overlapping_sessions": len(overlapping),
        })

    # Order check: are recovered windows in the right temporal sequence?
    recovered_windows = [w for w in gt_window_details if w["recovered"]]
    if len(recovered_windows) > 1:
        session_starts = [w["start_frame"] for w in recovered_windows]
        order_correct = all(session_starts[i] <= session_starts[i + 1] for i in range(len(session_starts) - 1))
    else:
        order_correct = True

    total_gt = len(all_gt_windows)
    results = {
        "reference_threshold": {"proximity_m": 1.5, "duration_frames": 15},
        "n_gt_match_windows": total_gt,
        "n_stage_e_sessions_on_entity": len(stage_e_sessions),
        "gt_recall": gt_recovered / total_gt if total_gt else 0,
        "gt_recovered": gt_recovered,
        "gt_missed": gt_missed,
        "order_correct": order_correct,
        "gt_window_details": gt_window_details,
        "stage_e_sessions": stage_e_sessions[:20],
        "metric_definition": (
            "Recall of GT engagement windows: for each GT match the tagged athlete was in, "
            "does an overlapping Stage E session exist on her entity (p0022)? "
            "Reference threshold: engage (1.5m) x sustained (15 frames)."
        ),
    }
    logger.info(
        f"  {total_gt} GT windows, {gt_recovered} recovered ({gt_recovered/total_gt*100:.0f}% recall), "
        f"order={'correct' if order_correct else 'WRONG'}, "
        f"{len(stage_e_sessions)} Stage E sessions on entity"
    )
    return results


# ---------------------------------------------------------------------------
# Angle 8: Unfixable floor (pair-box + miss)
# ---------------------------------------------------------------------------


def angle_8(gt_data):
    """For tagged athlete's GT frames: how many have no own detection?"""
    logger.info("=== Angle 8: Unfixable floor ===")
    results = {}

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        tagged_gt = clip_info["gt_track"]

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        tagged_recs = [r for r in match_records if r["gt_track_id"] == tagged_gt]
        total = len(tagged_recs)
        miss = sum(1 for r in tagged_recs if r["classification"] == "miss")
        pair_box = sum(1 for r in tagged_recs if r["classification"] == "pair_box")
        tight = sum(1 for r in tagged_recs if r["classification"] == "tight_match")

        results[clip_id] = {
            "total_gt_frames": total,
            "tight_match": tight,
            "pair_box": pair_box,
            "miss": miss,
            "tight_pct": tight / total * 100 if total else 0,
            "pair_box_pct": pair_box / total * 100 if total else 0,
            "miss_pct": miss / total * 100 if total else 0,
            "unfixable_floor_pct": (pair_box + miss) / total * 100 if total else 0,
            "note": "pair_box = identity ceiling (detection covers 2+ people); "
                    "miss = no detection at all. Neither fixable by identity logic.",
        }
        logger.info(
            f"  {clip_id}: {tight}/{total} tight ({tight/total*100:.1f}%), "
            f"{pair_box} pair_box ({pair_box/total*100:.1f}%), "
            f"{miss} miss ({miss/total*100:.1f}%)"
        )

    results["metric_definition"] = (
        "Unfixable floor = GT frames where tagged athlete has no own 1:1 detection "
        "(pair_box + miss). No identity-layer fix can recover these."
    )
    return results


# ---------------------------------------------------------------------------
# Angle 9: Approximate stage attribution of impurity
# ---------------------------------------------------------------------------


def angle_9(gt_data, session_pt):
    """Approximate attribution of non-self frames on p0022's entity to pipeline stages."""
    logger.info("=== Angle 9: Approximate stage attribution ===")

    attribution = {
        "group_over_attribution": 0,
        "tracklet_impurity": 0,
        "ilp_stitch": 0,
        "unattributed": 0,
    }
    total_non_self = 0

    for clip_id, clip_info in [
        (VID1_CLIP_ID, {"dir": VID1_DIR, "gt_track": VID1_GT_TRACK, "offset": 0}),
        (VID2_CLIP_ID, {"dir": VID2_DIR, "gt_track": VID2_GT_TRACK, "offset": VID2_FRAME_OFFSET}),
    ]:
        gt = gt_data[clip_id]
        det_df = _load_clip_detections(clip_info["dir"])
        tagged_gt = clip_info["gt_track"]

        match_records = _match_gt_to_detections(
            gt["gt_by_frame"], det_df, gt["annotated_frames"]
        )

        # Build detection_id -> gt_track_id
        det_to_gt: dict[tuple, int] = {}
        for r in match_records:
            if r["detection_id"]:
                det_to_gt[(r["detection_id"], r["frame_index"])] = r["gt_track_id"]

        # Build tracklet_id -> majority GT track (tracklet-level owner)
        # Store both bare and namespaced keys, plus split products
        split_map_local = _load_split_audit(clip_info["dir"])
        tracklet_gt_counts: dict[str, Counter] = defaultdict(Counter)
        for r in match_records:
            if r["tracklet_id"]:
                tracklet_gt_counts[r["tracklet_id"]][r["gt_track_id"]] += 1
        tracklet_owner: dict[str, int] = {}
        for tid, counter in tracklet_gt_counts.items():
            owner = counter.most_common(1)[0][0]
            tracklet_owner[tid] = owner
            tracklet_owner[f"{clip_id}:{tid}"] = owner
            # Also map split products to parent's owner
            if tid in split_map_local:
                for prod in split_map_local[tid]:
                    tracklet_owner[f"{clip_id}:{prod}"] = owner

        # Load D1 segments for GROUP detection
        try:
            d1_seg = _load_d1_segments(clip_info["dir"])
        except FileNotFoundError:
            d1_seg = pd.DataFrame()

        # Get p0022's frames
        clip_pt = session_pt[
            (session_pt.clip_id == clip_id) & (session_pt.person_id == TAG_PERSON_ID)
        ]
        annotated_set = set(gt["annotated_frames"])
        offset = clip_info["offset"]

        # Pre-build frame -> count of p0022 rows (GROUP detection)
        frame_pid_count: Counter = Counter()
        for fi in clip_pt.frame_index.values:
            frame_pid_count[int(fi)] += 1

        for det_id, fi_session, tid in zip(
            clip_pt.detection_id.values, clip_pt.frame_index.values, clip_pt.tracklet_id.values
        ):
            clip_frame = int(fi_session) - offset
            if clip_frame not in annotated_set:
                continue

            key = (det_id, clip_frame)
            if key not in det_to_gt:
                continue

            gt_track = det_to_gt[key]
            if gt_track == tagged_gt:
                continue  # self frame, skip

            total_non_self += 1

            # Check GROUP over-attribution: multiple p0022 rows at same session frame
            if frame_pid_count[int(fi_session)] > 1:
                attribution["group_over_attribution"] += 1
            elif tid in tracklet_owner and tracklet_owner[tid] != tagged_gt:
                attribution["ilp_stitch"] += 1
            elif tid in tracklet_owner and tracklet_owner[tid] == tagged_gt:
                attribution["tracklet_impurity"] += 1
            else:
                attribution["unattributed"] += 1

    results = {
        "total_non_self_frames": total_non_self,
        "attribution": attribution,
        "attribution_pct": {
            k: round(v / total_non_self * 100, 1) if total_non_self else 0
            for k, v in attribution.items()
        },
        "metric_definition": (
            "APPROXIMATE stage attribution of non-self frames on p0022's entity. "
            "group_over_attribution = multiple pids at same frame (GROUP node). "
            "ilp_stitch = tracklet's majority GT is someone else (solver connected wrong tracklet). "
            "tracklet_impurity = tracklet 'belongs' to tagged but frame-level GT differs (tracker swap). "
            "This is approximate — angles 5+6 are the load-bearing discriminators."
        ),
        "caveat": "APPROXIMATE. Overlapping categories possible. Use angles 5+6 for decisions.",
    }
    logger.info(
        f"  {total_non_self} non-self frames: "
        + ", ".join(f"{k}={v}" for k, v in attribution.items())
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("CP-PURITY-1: Decomposition of tagged-athlete identity quality")
    logger.info(f"Output: {EVIDENCE_DIR}")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared data
    logger.info("Loading data...")
    gt_data = _load_gt_for_clips()
    session_pt = _load_session_person_tracks()
    logger.info(
        f"Session person_tracks: {len(session_pt)} rows, "
        f"{session_pt.person_id.nunique()} person_ids"
    )

    # Check frame space: does session_pt use clip-local or session-offset frames?
    vid2_pt = session_pt[session_pt.clip_id == VID2_CLIP_ID]
    vid2_min_frame = vid2_pt.frame_index.min()
    logger.info(f"Vid2 min frame in session_pt: {vid2_min_frame} (expect ~4530 if offset, ~0 if clip-local)")

    # Run all angles
    results = {}
    results["angle_1"] = angle_1(gt_data, session_pt)
    results["angle_2"] = angle_2(gt_data, session_pt)
    results["angle_3"] = angle_3(gt_data)
    results["angle_4"] = angle_4(gt_data, session_pt)
    results["angle_5"] = angle_5(gt_data, session_pt)
    results["angle_6"] = angle_6(gt_data, session_pt)
    results["angle_7"] = angle_7(gt_data, session_pt)
    results["angle_8"] = angle_8(gt_data)
    results["angle_9"] = angle_9(gt_data, session_pt)

    # Write per-angle JSON files
    for angle_name, data in results.items():
        out_path = EVIDENCE_DIR / f"{angle_name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Wrote {out_path.name}")

    # Write synthesis report
    _write_report(results)

    logger.info("CP-PURITY-1 complete.")


def _write_report(results):
    """Write the decomposition report markdown."""
    r = results
    a1 = r["angle_1"]
    a5 = r["angle_5"]
    a6 = r["angle_6"]
    a8 = r["angle_8"]
    a9 = r["angle_9"]

    # Extract key numbers
    vid1_tagged = a1.get(VID1_CLIP_ID, {}).get("tagged_athlete", {})
    vid2_tagged = a1.get(VID2_CLIP_ID, {}).get("tagged_athlete", {})
    vid1_agg = a1.get(VID1_CLIP_ID, {}).get("aggregate_correct_id_pct", 0)
    vid2_agg = a1.get(VID2_CLIP_ID, {}).get("aggregate_correct_id_pct", 0)

    sweep = a6.get("sweep_grid", {})
    stability = a6.get("stability_assessment", {})

    # Pick a representative sweep cell for headline
    rep_key = "engage_sustained"
    rep = sweep.get(rep_key, {})

    report = f"""# CP-PURITY-1: Decomposition Report

## Headline Discriminators

### Angle 5 — Through-line integrity
- Tagged athlete GT frames: {a5.get('total_gt_frames', '?')}
- Frames with p0022 present: {a5.get('frames_with_tag_pid', '?')}/{a5.get('frames_with_detection', '?')} ({a5.get('tag_pid_coverage_pct', 0):.1f}%)
- Teleport events: {a5.get('n_teleport_events', '?')}
- Identity segments: {a5.get('identity_sequence_length', '?')}

### Angle 6 — Intra-match vs cross-match (headline: engage x sustained)
- Intra-match (expected dilution): {rep.get('intra_match', '?')} frames ({rep.get('intra_pct', 0):.1f}%)
- Cross-match (bug/over-reach): {rep.get('cross_match', '?')} frames ({rep.get('cross_pct', 0):.1f}%)
- Self frames: {rep.get('self_frames', '?')}
- GT match windows found: {rep.get('n_gt_match_windows', '?')}
- Sweep stability: {'STABLE' if stability.get('verdict_stable') else 'UNSTABLE'} (intra% range: {stability.get('intra_pct_range', [])})

**Diagnosis: {'(a) Mostly intended intra-match ambiguity' if rep.get('intra_pct', 0) > 60 else '(b) Mostly cross-match over-reach' if rep.get('cross_pct', 0) > 60 else 'Mixed — see sweep grid'}**

---

## Angle 1 — Matched metric re-baseline

| Clip | Tagged correct_id | Aggregate correct_id | CP-TAG-3 baseline |
|------|------------------|---------------------|-------------------|
| vid1 (200015) | {vid1_tagged.get('correct_pct', 0):.1f}% | {vid1_agg:.1f}% | 25.6% |
| vid2 (200246) | {vid2_tagged.get('correct_pct', 0):.1f}% | {vid2_agg:.1f}% | 22.2% |

Reference aggregate (signal-trace): ~58.7%

## Angle 2 — Tracklet purity distribution

"""
    a2 = r["angle_2"]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        clip_data = a2.get(clip_id, {})
        report += f"**{clip_id}:** {clip_data.get('n_tracklets_measured', '?')} tracklets, "
        report += f"mean={clip_data.get('mean_purity', 0):.3f}, median={clip_data.get('median_purity', 0):.3f}, "
        report += f"{clip_data.get('n_impure_below_0.8', '?')} impure (<0.8)\n\n"

    report += """## Angle 3 -- D0.5 split impact

"""
    a3 = r["angle_3"]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        clip_data = a3.get(clip_id, {})
        report += f"**{clip_id}:** {clip_data.get('n_splits', 0)} splits, "
        report += f"helped={clip_data.get('helped', 0)}, hurt={clip_data.get('hurt', 0)}, "
        report += f"neutral={clip_data.get('neutral', 0)}, mean delta={clip_data.get('mean_delta', 0):.3f}\n\n"

    report += """## Angle 4 -- Entity purity

"""
    a4 = r["angle_4"]
    report += f"Entities measured: {a4.get('n_entities', '?')}, "
    report += f"mean purity={a4.get('mean_purity', 0):.3f}, median={a4.get('median_purity', 0):.3f}\n\n"
    tagged_ent = a4.get("tagged_entity_p0022", {})
    report += f"**p0022 (tagged):** purity={tagged_ent.get('purity', '?')}, "
    report += f"majority GT={tagged_ent.get('majority_gt', '?')}, "
    report += f"GT-matched frames={tagged_ent.get('gt_matched_frames', '?')}\n\n"

    report += """## Angle 7 -- Match window recovery

"""
    a7 = r["angle_7"]
    report += f"GT match windows (engage x sustained): {a7.get('n_gt_match_windows', '?')}\n"
    report += f"Stage E sessions on p0022: {a7.get('n_stage_e_sessions_on_entity', '?')}\n"
    report += f"GT recall: {a7.get('gt_recall', 0):.1%} ({a7.get('gt_recovered', 0)}/{a7.get('n_gt_match_windows', 0)})\n"
    report += f"Order correct: {a7.get('order_correct', '?')}\n\n"

    report += """## Angle 8 -- Unfixable floor

"""
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        clip_data = a8.get(clip_id, {})
        report += f"**{clip_id}:** tight={clip_data.get('tight_pct', 0):.1f}%, "
        report += f"pair_box={clip_data.get('pair_box_pct', 0):.1f}%, "
        report += f"miss={clip_data.get('miss_pct', 0):.1f}% "
        report += f"(floor={clip_data.get('unfixable_floor_pct', 0):.1f}%)\n\n"

    report += """## Angle 9 -- Approximate stage attribution

"""
    report += f"Total non-self frames on p0022: {a9.get('total_non_self_frames', '?')}\n\n"
    for k, v in a9.get("attribution_pct", {}).items():
        report += f"- {k}: {v}%\n"
    report += f"\n**Caveat:** {a9.get('caveat', '')}\n\n"

    report += """## Angle 6 -- Full sweep grid

| Proximity | Duration | GT Windows | Intra | Cross | Intra% | Cross% |
|-----------|----------|-----------|-------|-------|--------|--------|
"""
    for key, cell in sorted(sweep.items()):
        prox, dur = key.split("_")
        report += (
            f"| {prox} ({cell['proximity_m']}m) | {dur} ({cell['duration_frames']}f) | "
            f"{cell['n_gt_match_windows']} | {cell['intra_match']} | {cell['cross_match']} | "
            f"{cell['intra_pct']:.1f}% | {cell['cross_pct']:.1f}% |\n"
        )

    report += f"""
## Metric Definitions (locked)

- **Tracklet purity** = fraction of GT-matched frames whose GT track is the tracklet's plurality GT track (greedy IoU>=0.3)
- **Entity purity** = same, per emitted person_id over its session person_tracks frames
- **Through-line dominant id** = majority-vote person_id per GT track from session person_tracks
- **Coverage** = of tagged athlete's GT frames, fraction with detection carrying dominant_id
- **Anchor-correctness** = of dominant_id's frames, fraction matching tagged GT track (= entity purity for p0022)
- **GT match window** = sustained interval where two GT tracks' world-projected foot-points are within proximity threshold for >= duration consecutive annotated frames
- **Intra-match frame** = non-self frame on entity whose GT person IS in GT match window with tagged athlete
- **Cross-match frame** = non-self frame whose GT person is NOT in GT match window at that frame

## Options for Web Session

1. **If diagnosis is (a) mostly intra-match:** The entity is doing its job; the "impurity" is opponent frames during real matches. Fix is in metric design (exclude intra-match from purity) and/or Stage E window extraction (crop to athlete's own tracklet within the match window).

2. **If diagnosis is (b) mostly cross-match:** The entity wanders into matches it shouldn't be in. Fix is in ILP pairing/emission (HSV-assisted GROUP node ownership, hard ping connectivity, is_isolated gate).

3. **If mixed:** Prioritize the bigger bucket first. Cross-match requires architectural fix; intra-match may be acceptable with metric adjustment.
"""

    report_path = EVIDENCE_DIR / "decomposition_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"  Wrote {report_path.name}")


if __name__ == "__main__":
    main()
