"""CP-PURITY-2: Aggregate reconciliation + unfixable-floor decomposition.

Measurement-only script. No production code changes.
Extends CP-PURITY-1 with:
  M1: Aggregate correct_id reconciliation (clip-level val-split vs 40.5% baseline)
  M2: Pair-box floor split (correct-group vs mishandled, proximity sweep)
  M3: Miss floor split (proxy-occluded / edge-ROI / detector-fail, CVAT cross-check)
  M4: True addressable ceiling partition

Usage:
    PYTHONPATH=src python tools/cp_purity_2_floor.py
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

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
# Constants (shared with CP-PURITY-1)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_purity_2"

VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_GT_TRACK = 24
VID2_GT_TRACK = 8
CAM_ID = "J_EDEw"
GYM_ID = "_eval_gt"
TAG_PERSON_ID = "p0022"

VID1_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID1_CLIP_ID
VID2_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID2_CLIP_ID
SESSION_DIR = OUTPUTS_DIR / GYM_ID / "sessions" / "2026-03-18" / "cp_tag_3_baseline"
VID2_FRAME_OFFSET = 4530

MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-v2.yaml"
CVAT_XML_PATH = REPO_ROOT / "data" / "cvat_tasks" / "round1_20260497_J_EDEw" / "annotations.xml"
ROI_MASK_PATH = REPO_ROOT / "configs" / "cameras" / CAM_ID / "roi_mask.png"

# Proximity thresholds for M2 sweep (matching CP-PURITY-1)
PROX_THRESHOLDS = {"tight": 0.5, "close": 1.0, "engage": 1.5}

# IoU threshold for greedy matcher
IOU_THRESHOLD = 0.3
# Proxy occlusion: max GT-GT IoU threshold
PROXY_OCCLUDED_IOU = 0.15
# CVAT keypoint occlusion fraction threshold
CVAT_OCCLUDED_FRAC = 0.5


# ---------------------------------------------------------------------------
# Helpers: World projection (same as CP-PURITY-1)
# ---------------------------------------------------------------------------


def _load_projection():
    """Load J_EDEw homography (inverted to pixel->world) + lens params."""
    h_path = REPO_ROOT / "configs" / "cameras" / CAM_ID / "homography.json"
    with open(h_path) as f:
        payload = json.load(f)
    H_stored = np.array(payload["H"], dtype=np.float64)
    K = np.array(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(payload["dist_coefficients"], dtype=np.float64).ravel()
    H_inv = np.linalg.inv(H_stored)
    return H_inv, K, D


def _project_bbox_foot(bbox_xyxy, H_inv, K, D):
    """Project bbox bottom-center to world coordinates."""
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) / 2.0
    v = y2
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    u2 = float(undistorted[0, 0, 0])
    v2 = float(undistorted[0, 0, 1])
    p = np.array([u2, v2, 1.0], dtype=np.float64)
    q = H_inv @ p
    w = q[2]
    if abs(w) < 1e-12:
        return (float("nan"), float("nan"))
    return (q[0] / w, q[1] / w)


# ---------------------------------------------------------------------------
# Helpers: Data loading
# ---------------------------------------------------------------------------


def _load_clip_detections(clip_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")


def _load_clip_person_tracks(clip_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(clip_dir / "stage_D" / "person_tracks.parquet")


def _load_session_person_tracks() -> pd.DataFrame:
    return pd.read_parquet(SESSION_DIR / "stage_D" / "person_tracks_J_EDEw.parquet")


def _load_d1_segments(clip_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(clip_dir / "stage_D" / "d1_segments.parquet")


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


def _load_roi_mask() -> np.ndarray:
    """Load ROI mask as binary array (True = valid)."""
    mask = cv2.imread(str(ROI_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    return mask > 127


# ---------------------------------------------------------------------------
# Helpers: GT matching (same as CP-PURITY-1)
# ---------------------------------------------------------------------------


def _match_gt_to_detections(gt_by_frame, det_df, annotated_frames):
    """Match GT to pipeline detections. Returns list of record dicts."""
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
                    "frame_index": fi, "gt_track_id": tid,
                    "detection_id": None, "tracklet_id": None,
                    "iou": 0.0, "classification": "miss",
                    "gt_bbox": gt_tuples[i], "det_bbox": None,
                })
            continue

        det_tuples = list(zip(
            frame_dets.x1.values, frame_dets.y1.values,
            frame_dets.x2.values, frame_dets.y2.values,
        ))
        det_ids = frame_dets.detection_id.values.tolist()
        det_tids = frame_dets.tracklet_id.values.tolist()
        matches = greedy_match(gt_tuples, det_tuples, iou_threshold=IOU_THRESHOLD)

        gt_matched = {}
        det_gt_count: Counter = Counter()
        for gt_idx, det_idx, iou in matches:
            gt_matched[gt_idx] = (det_idx, iou)
            det_gt_count[det_idx] += 1

        for gt_idx, tid in enumerate(gt_track_ids):
            if gt_idx not in gt_matched:
                records.append({
                    "frame_index": fi, "gt_track_id": tid,
                    "detection_id": None, "tracklet_id": None,
                    "iou": 0.0, "classification": "miss",
                    "gt_bbox": gt_tuples[gt_idx], "det_bbox": None,
                })
            else:
                det_idx, iou = gt_matched[gt_idx]
                n_sharing = det_gt_count[det_idx]
                classification = "pair_box" if n_sharing >= 2 else "tight_match"
                records.append({
                    "frame_index": fi, "gt_track_id": tid,
                    "detection_id": det_ids[det_idx], "tracklet_id": det_tids[det_idx],
                    "iou": iou, "classification": classification,
                    "gt_bbox": gt_tuples[gt_idx], "det_bbox": det_tuples[det_idx],
                })
    return records


def _build_det_frame_pids(person_tracks_df, clip_id):
    """Pre-build (detection_id, frame_index) -> [person_ids] lookup."""
    clip_pt = person_tracks_df[person_tracks_df.clip_id == clip_id]
    det_frame_pids: dict[tuple, list[str]] = defaultdict(list)
    for det_id, fi, pid in zip(
        clip_pt.detection_id.values, clip_pt.frame_index.values, clip_pt.person_id.values
    ):
        det_frame_pids[(det_id, int(fi))].append(pid)
    return det_frame_pids


def _compute_dominant_pid_per_gt(match_records, det_frame_pids, frame_offset=0):
    """Majority-vote person_id per GT track. Returns {gt_track_id: dominant_pid}."""
    gt_pid_counts: dict[int, Counter] = defaultdict(Counter)
    for rec in match_records:
        if rec["detection_id"] is None:
            continue
        session_fi = rec["frame_index"] + frame_offset
        pids = det_frame_pids.get((rec["detection_id"], session_fi), [])
        if not pids:
            pids = det_frame_pids.get((rec["detection_id"], rec["frame_index"]), [])
        for pid in pids:
            gt_pid_counts[rec["gt_track_id"]][pid] += 1
    dominant = {}
    for gt_tid, counter in gt_pid_counts.items():
        if counter:
            dominant[gt_tid] = counter.most_common(1)[0][0]
    return dominant


def _compute_correct_id(match_records, det_frame_pids, frame_offset=0):
    """Compute per-GT-track correct_id breakdown. Returns (per_track, aggregate_pct)."""
    dominant = _compute_dominant_pid_per_gt(match_records, det_frame_pids, frame_offset)
    per_track = {}
    for gt_tid in set(r["gt_track_id"] for r in match_records):
        track_recs = [r for r in match_records if r["gt_track_id"] == gt_tid]
        dom_pid = dominant.get(gt_tid)
        correct = wrong = no_id = no_det = 0
        for r in track_recs:
            if r["detection_id"] is None:
                no_det += 1
                continue
            session_fi = r["frame_index"] + frame_offset
            pids = det_frame_pids.get((r["detection_id"], session_fi), [])
            if not pids:
                pids = det_frame_pids.get((r["detection_id"], r["frame_index"]), [])
            if not pids:
                no_id += 1
            elif dom_pid and dom_pid in pids:
                correct += 1
            else:
                wrong += 1
        total = len(track_recs)
        per_track[gt_tid] = {
            "correct": correct, "wrong": wrong, "no_id": no_id,
            "no_det": no_det, "total": total,
            "correct_pct": correct / total * 100 if total else 0,
            "dominant_pid": dom_pid,
        }
    agg_correct = sum(t["correct"] for t in per_track.values())
    agg_total = sum(t["total"] for t in per_track.values())
    agg_pct = agg_correct / agg_total * 100 if agg_total else 0
    return per_track, agg_pct


# ---------------------------------------------------------------------------
# Helpers: IoU between GT boxes
# ---------------------------------------------------------------------------


def _iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


# ---------------------------------------------------------------------------
# M1: Aggregate reconciliation
# ---------------------------------------------------------------------------


def measurement_1(gt_data):
    """Aggregate correct_id reconciliation: clip-level val-split vs 40.5% baseline."""
    logger.info("=== M1: Aggregate reconciliation ===")
    manifest = load_manifest(MANIFEST_PATH)
    results = {}

    # Find the J_EDEw vid1 export entry for val-split frames
    vid1_export = None
    for exp in manifest.training_data:
        if exp.camera_id == CAM_ID and "200015" in exp.source_video:
            vid1_export = exp
            break

    if vid1_export is None or vid1_export.splits.val is None:
        logger.error("No val split found for J_EDEw vid1")
        return {"error": "no val split"}

    val_frames = list(range(
        vid1_export.splits.val.start,
        vid1_export.splits.val.stop + 1,
        vid1_export.splits.val.stride,
    ))
    logger.info(f"  Val-split: {len(val_frames)} frames ({val_frames[0]}-{val_frames[-1]})")

    gt = gt_data[VID1_CLIP_ID]
    det_df = _load_clip_detections(VID1_DIR)

    # --- A: Clip-level, val-split only (apples-to-apples with 40.5%) ---
    clip_pt = _load_clip_person_tracks(VID1_DIR)
    clip_det_pids = _build_det_frame_pids(clip_pt, VID1_CLIP_ID)

    val_records = _match_gt_to_detections(gt["gt_by_frame"], det_df, val_frames)
    per_track_val, agg_val = _compute_correct_id(val_records, clip_det_pids)

    results["val_split_clip_level"] = {
        "correct_id_pct": round(agg_val, 1),
        "baseline_pct": 40.5,
        "delta": round(agg_val - 40.5, 1),
        "n_frames": sum(t["total"] for t in per_track_val.values()),
        "n_tracks": len(per_track_val),
        "note": "Clip-level person_tracks, val-split only (frames 2500-3000). "
                "Direct comparison to signal-trace J_EDEw baseline.",
    }
    logger.info(f"  Val-split clip-level: {agg_val:.1f}% (baseline 40.5%, delta {agg_val-40.5:+.1f}pp)")

    # --- B: Clip-level, full range (vid1) ---
    full_records = _match_gt_to_detections(gt["gt_by_frame"], det_df, gt["annotated_frames"])
    _, agg_full_clip = _compute_correct_id(full_records, clip_det_pids)
    results["full_range_clip_level_vid1"] = {"correct_id_pct": round(agg_full_clip, 1)}
    logger.info(f"  Full-range clip-level vid1: {agg_full_clip:.1f}%")

    # --- C: Session-level, full range (vid1) ---
    session_pt = _load_session_person_tracks()
    session_det_pids = _build_det_frame_pids(session_pt, VID1_CLIP_ID)
    _, agg_full_session = _compute_correct_id(full_records, session_det_pids)
    results["full_range_session_level_vid1"] = {"correct_id_pct": round(agg_full_session, 1)}
    logger.info(f"  Full-range session-level vid1: {agg_full_session:.1f}%")

    # --- D: Vid2 full range, clip-level ---
    gt2 = gt_data[VID2_CLIP_ID]
    det_df2 = _load_clip_detections(VID2_DIR)
    clip_pt2 = _load_clip_person_tracks(VID2_DIR)
    clip_det_pids2 = _build_det_frame_pids(clip_pt2, VID2_CLIP_ID)
    full_records2 = _match_gt_to_detections(gt2["gt_by_frame"], det_df2, gt2["annotated_frames"])
    _, agg_vid2_clip = _compute_correct_id(full_records2, clip_det_pids2)
    results["full_range_clip_level_vid2"] = {"correct_id_pct": round(agg_vid2_clip, 1)}
    logger.info(f"  Full-range clip-level vid2: {agg_vid2_clip:.1f}%")

    # --- E: Vid2 full range, session-level ---
    session_det_pids2 = _build_det_frame_pids(session_pt, VID2_CLIP_ID)
    _, agg_vid2_session = _compute_correct_id(
        full_records2, session_det_pids2, frame_offset=VID2_FRAME_OFFSET
    )
    results["full_range_session_level_vid2"] = {"correct_id_pct": round(agg_vid2_session, 1)}
    logger.info(f"  Full-range session-level vid2: {agg_vid2_session:.1f}%")

    # Verdict
    delta = agg_val - 40.5
    if abs(delta) < 3.0:
        verdict = "NO REGRESSION: within noise margin of baseline"
    elif delta < -3.0:
        verdict = f"REGRESSION: {delta:+.1f}pp below baseline"
    else:
        verdict = f"IMPROVEMENT: {delta:+.1f}pp above baseline"
    results["verdict"] = verdict
    results["canonical_definition"] = (
        "Canonical aggregate correct_id: clip-level person_tracks, greedy IoU>=0.3, "
        "majority-vote dominant_pid per GT track, val-split frames only. "
        "J_EDEw baseline: 40.5% (signal-trace, bjj-detect-all-cameras). "
        "Three-camera aggregate (58.7%) is NOT comparable to single-camera numbers."
    )

    logger.info(f"  Verdict: {verdict}")
    return results


# ---------------------------------------------------------------------------
# M2: Pair-box floor split
# ---------------------------------------------------------------------------


def measurement_2(gt_data):
    """Pair-box floor split: correct-group vs mishandled, with proximity sweep."""
    logger.info("=== M2: Pair-box floor split ===")
    H_inv, K, D = _load_projection()

    sweep_results = {}

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

        # Get tagged athlete's pair_box frames
        pair_box_recs = [
            r for r in match_records
            if r["gt_track_id"] == tagged_gt and r["classification"] == "pair_box"
        ]

        if not pair_box_recs:
            sweep_results[clip_id] = {"n_pair_box": 0}
            continue

        # Load D1 segments for GROUP check
        try:
            d1_seg = _load_d1_segments(clip_info["dir"])
        except FileNotFoundError:
            d1_seg = pd.DataFrame()

        # Pre-compute world positions for all GT boxes
        gt_world = {}
        for fi, boxes in gt["gt_by_frame"].items():
            gt_world[fi] = {}
            for b in boxes:
                wx, wy = _project_bbox_foot((b.x1, b.y1, b.x2, b.y2), H_inv, K, D)
                gt_world[fi][b.track_id] = (wx, wy)

        # For each pair_box frame, check (a) pipeline GROUP and (b) GT proximity
        clip_sweep = {}
        for prox_label, prox_thresh in PROX_THRESHOLDS.items():
            correct_group = 0
            mishandled = 0
            spurious_group = 0
            neither = 0

            for r in pair_box_recs:
                fi = r["frame_index"]
                tid = r["tracklet_id"]

                # (a) Is the detection's tracklet in a GROUP segment?
                in_group = False
                if not d1_seg.empty and tid is not None:
                    seg_match = d1_seg[
                        (d1_seg.base_tracklet_id == tid) &
                        (d1_seg.start_frame <= fi) &
                        (d1_seg.end_frame >= fi) &
                        (d1_seg.segment_type == "GROUP")
                    ]
                    in_group = len(seg_match) > 0

                # (b) GT-should-group: is tagged athlete within proximity of another GT?
                should_group = False
                tagged_pos = gt_world.get(fi, {}).get(tagged_gt)
                if tagged_pos and not np.isnan(tagged_pos[0]):
                    for other_tid, other_pos in gt_world.get(fi, {}).items():
                        if other_tid == tagged_gt:
                            continue
                        if np.isnan(other_pos[0]):
                            continue
                        dist = np.sqrt(
                            (tagged_pos[0] - other_pos[0]) ** 2 +
                            (tagged_pos[1] - other_pos[1]) ** 2
                        )
                        if dist <= prox_thresh:
                            should_group = True
                            break

                if in_group and should_group:
                    correct_group += 1
                elif should_group and not in_group:
                    mishandled += 1
                elif in_group and not should_group:
                    spurious_group += 1
                else:
                    neither += 1

            total = len(pair_box_recs)
            clip_sweep[prox_label] = {
                "proximity_m": prox_thresh,
                "n_pair_box": total,
                "correct_group": correct_group,
                "mishandled": mishandled,
                "spurious_group": spurious_group,
                "neither": neither,
                "correct_group_pct": round(correct_group / total * 100, 1) if total else 0,
                "mishandled_pct": round(mishandled / total * 100, 1) if total else 0,
                "group_error_rate": round(
                    (mishandled + spurious_group) / total * 100, 1
                ) if total else 0,
            }

        sweep_results[clip_id] = {
            "n_pair_box": len(pair_box_recs),
            "proximity_sweep": clip_sweep,
        }

        for prox_label, data in clip_sweep.items():
            logger.info(
                f"  {clip_id} @ {prox_label}: {data['correct_group']} correct-group, "
                f"{data['mishandled']} mishandled, {data['spurious_group']} spurious, "
                f"{data['neither']} neither"
            )

    # Stability check
    all_mishandled_pcts = []
    for clip_data in sweep_results.values():
        if isinstance(clip_data, dict) and "proximity_sweep" in clip_data:
            for prox_data in clip_data["proximity_sweep"].values():
                all_mishandled_pcts.append(prox_data["mishandled_pct"])
    stable = (max(all_mishandled_pcts) - min(all_mishandled_pcts)) < 15 if all_mishandled_pcts else True

    results = {
        "per_clip": sweep_results,
        "stability": {
            "verdict_stable": stable,
            "mishandled_pct_range": [
                round(min(all_mishandled_pcts), 1),
                round(max(all_mishandled_pcts), 1),
            ] if all_mishandled_pcts else [],
        },
        "metric_definition": (
            "Pair-box floor split: for tagged athlete's pair_box frames "
            "(shared detection, no own bbox), cross-tabulate (a) pipeline GROUP segment "
            "(from d1_segments.parquet) vs (b) GT world-distance proximity. "
            "correct-group = a AND b. mishandled = b AND NOT a. "
            "Group error rate = (mishandled + spurious) / total."
        ),
    }
    return results


# ---------------------------------------------------------------------------
# M3: Miss floor split
# ---------------------------------------------------------------------------


def _parse_cvat_keypoint_occlusion(xml_path: Path):
    """Parse CVAT XML to get per-(track_id, frame) keypoint occlusion fraction.

    Returns: {(track_id, frame): (n_occluded, n_total, fraction)}
    Also returns per-track bboxes for join verification: {(track_id, frame): (x1,y1,x2,y2)}
    """
    logger.info("  Parsing CVAT XML for keypoint occlusion...")
    result: dict[tuple, tuple] = {}
    bboxes: dict[tuple, tuple] = {}

    current_track_id = None
    current_frame = None
    current_keypoints: list[tuple] = []  # (x, y, occluded)

    for event, elem in ET.iterparse(xml_path, events=["start", "end"]):
        if event == "start" and elem.tag == "track":
            current_track_id = int(elem.get("id", -1))

        if event == "start" and elem.tag == "skeleton":
            current_frame = int(elem.get("frame", -1))
            current_keypoints = []

        if event == "start" and elem.tag == "points":
            occ = elem.get("occluded", "0")
            pts = elem.get("points", "0,0")
            try:
                x, y = pts.split(",")
                current_keypoints.append((float(x), float(y), occ == "1"))
            except (ValueError, AttributeError):
                pass

        if event == "end" and elem.tag == "skeleton":
            if current_track_id is not None and current_frame is not None and current_keypoints:
                n_occ = sum(1 for _, _, o in current_keypoints if o)
                n_total = len(current_keypoints)
                frac = n_occ / n_total if n_total > 0 else 0
                result[(current_track_id, current_frame)] = (n_occ, n_total, frac)

                # Compute bounding box from keypoints for join verification
                xs = [kp[0] for kp in current_keypoints]
                ys = [kp[1] for kp in current_keypoints]
                bboxes[(current_track_id, current_frame)] = (
                    min(xs), min(ys), max(xs), max(ys)
                )
            current_frame = None
            current_keypoints = []

        elem.clear()

    logger.info(f"  CVAT XML: {len(result)} (track, frame) entries parsed")
    return result, bboxes


def _verify_cvat_gt_join(cvat_bboxes, gt_by_frame, annotated_frames):
    """Verify CVAT track IDs align with GT track IDs by spatial overlap.

    Returns: {cvat_track_id: gt_track_id} or None if join can't be established.
    """
    # Strategy: for each annotated frame, match CVAT bboxes to GT bboxes by IoU
    # Build consensus: for each CVAT track, which GT track does it most often match?
    cvat_to_gt_votes: dict[int, Counter] = defaultdict(Counter)

    # Sample frames spread across the range for better coverage
    sample_frames = annotated_frames[::3]  # Every 3rd annotated frame
    for fi in sample_frames:
        gt_boxes = gt_by_frame.get(fi, [])
        if not gt_boxes:
            continue

        for cvat_key, cvat_bbox in cvat_bboxes.items():
            cvat_tid, cvat_frame = cvat_key
            if cvat_frame != fi:
                continue

            best_iou = 0
            best_gt = None
            for b in gt_boxes:
                iou_val = _iou(cvat_bbox, (b.x1, b.y1, b.x2, b.y2))
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = b.track_id

            if best_gt is not None and best_iou >= 0.3:
                cvat_to_gt_votes[cvat_tid][best_gt] += 1

    # Build mapping: each CVAT track -> its most-voted GT track
    cvat_to_gt = {}
    for cvat_tid, votes in cvat_to_gt_votes.items():
        best_gt, count = votes.most_common(1)[0]
        total = sum(votes.values())
        if count / total >= 0.7:  # 70% consensus threshold
            cvat_to_gt[cvat_tid] = best_gt

    # Check bijectivity: each GT track should map to at most one CVAT track
    gt_to_cvat: dict[int, list[int]] = defaultdict(list)
    for cvat_tid, gt_tid in cvat_to_gt.items():
        gt_to_cvat[gt_tid].append(cvat_tid)

    conflicts = {gt: cvats for gt, cvats in gt_to_cvat.items() if len(cvats) > 1}

    n_gt_tracks = len(set(b.track_id for boxes in gt_by_frame.values() for b in boxes))
    coverage = len(set(cvat_to_gt.values())) / n_gt_tracks if n_gt_tracks else 0

    logger.info(
        f"  CVAT-GT join: {len(cvat_to_gt)} CVAT tracks mapped, "
        f"{len(set(cvat_to_gt.values()))}/{n_gt_tracks} GT tracks covered, "
        f"{len(conflicts)} conflicts"
    )

    if coverage < 0.5 or len(conflicts) > 2:
        return None  # Join not reliable

    return cvat_to_gt


def measurement_3(gt_data):
    """Miss floor split: proxy-occluded / edge-ROI / detector-fail + CVAT cross-check."""
    logger.info("=== M3: Miss floor split ===")
    roi_mask = _load_roi_mask()
    results = {}

    # Parse CVAT XML for vid1 cross-check
    cvat_occ = None
    cvat_bboxes = None
    cvat_to_gt = None
    if CVAT_XML_PATH.exists():
        cvat_occ, cvat_bboxes = _parse_cvat_keypoint_occlusion(CVAT_XML_PATH)
        # Verify join
        gt1 = gt_data[VID1_CLIP_ID]
        cvat_to_gt = _verify_cvat_gt_join(
            cvat_bboxes, gt1["gt_by_frame"], gt1["annotated_frames"]
        )
        if cvat_to_gt is None:
            logger.warning(
                "  CVAT-GT join could NOT be established — cross-check disabled. "
                "CVAT skeleton bboxes (min/max of all 17 keypoints including defaults) "
                "are much larger than GT detection bboxes, making IoU matching unreliable."
            )
        else:
            # Invert: gt_track_id -> cvat_track_id
            gt_to_cvat = {}
            for cvat_tid, gt_tid in cvat_to_gt.items():
                gt_to_cvat.setdefault(gt_tid, []).append(cvat_tid)

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

        miss_recs = [
            r for r in match_records
            if r["gt_track_id"] == tagged_gt and r["classification"] == "miss"
        ]

        if not miss_recs:
            results[clip_id] = {"n_miss": 0}
            continue

        proxy_occluded = []
        edge_roi = []
        detector_fail = []
        cross_check_pairs = []  # (proxy_says_occluded, cvat_says_occluded)

        for r in miss_recs:
            fi = r["frame_index"]
            gt_bbox = r["gt_bbox"]

            # --- Proxy occlusion: max IoU with other GT boxes ---
            other_boxes = [
                (b.x1, b.y1, b.x2, b.y2)
                for b in gt["gt_by_frame"].get(fi, [])
                if b.track_id != tagged_gt
            ]
            max_gt_iou = max(
                (_iou(gt_bbox, ob) for ob in other_boxes), default=0
            )
            is_proxy_occluded = max_gt_iou >= PROXY_OCCLUDED_IOU

            # --- Edge-ROI: is bbox center outside ROI mask? ---
            cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
            cy = (gt_bbox[1] + gt_bbox[3]) / 2.0
            ix, iy = int(round(cx)), int(round(cy))
            h_mask, w_mask = roi_mask.shape
            if 0 <= ix < w_mask and 0 <= iy < h_mask:
                inside_roi = bool(roi_mask[iy, ix])
            else:
                inside_roi = False
            is_edge_roi = not inside_roi

            # --- Classify ---
            if is_proxy_occluded:
                proxy_occluded.append({
                    "frame": fi, "max_gt_iou": round(max_gt_iou, 3),
                })
            elif is_edge_roi:
                edge_roi.append({"frame": fi, "bbox_center": (round(cx, 1), round(cy, 1))})
            else:
                detector_fail.append({
                    "frame": fi, "max_gt_iou": round(max_gt_iou, 3),
                })

            # --- CVAT cross-check (vid1 only) ---
            if clip_id == VID1_CLIP_ID and cvat_to_gt is not None and cvat_occ is not None:
                cvat_tids = gt_to_cvat.get(tagged_gt, [])
                cvat_frac = None
                for ctid in cvat_tids:
                    entry = cvat_occ.get((ctid, fi))
                    if entry is not None:
                        cvat_frac = entry[2]  # fraction occluded
                        break
                if cvat_frac is not None:
                    cross_check_pairs.append((
                        is_proxy_occluded,
                        cvat_frac >= CVAT_OCCLUDED_FRAC,
                        round(cvat_frac, 3),
                        round(max_gt_iou, 3),
                    ))

        total = len(miss_recs)
        clip_result = {
            "n_miss": total,
            "proxy_occluded": len(proxy_occluded),
            "edge_roi": len(edge_roi),
            "detector_fail": len(detector_fail),
            "proxy_occluded_pct": round(len(proxy_occluded) / total * 100, 1) if total else 0,
            "edge_roi_pct": round(len(edge_roi) / total * 100, 1) if total else 0,
            "detector_fail_pct": round(len(detector_fail) / total * 100, 1) if total else 0,
            "proxy_occluded_frames": proxy_occluded[:20],
            "edge_roi_frames": edge_roi[:20],
            "detector_fail_frames": detector_fail[:20],
        }

        # CVAT cross-check (vid1 only)
        if clip_id == VID1_CLIP_ID and cross_check_pairs:
            # Agreement matrix
            both_occ = sum(1 for p, c, _, _ in cross_check_pairs if p and c)
            proxy_only = sum(1 for p, c, _, _ in cross_check_pairs if p and not c)
            cvat_only = sum(1 for p, c, _, _ in cross_check_pairs if not p and c)
            neither_occ = sum(1 for p, c, _, _ in cross_check_pairs if not p and not c)
            total_pairs = len(cross_check_pairs)
            agreement = (both_occ + neither_occ) / total_pairs if total_pairs else 0

            clip_result["cvat_cross_check"] = {
                "n_pairs": total_pairs,
                "both_occluded": both_occ,
                "proxy_only": proxy_only,
                "cvat_only": cvat_only,
                "neither_occluded": neither_occ,
                "agreement_pct": round(agreement * 100, 1),
                "divergence_pct": round((1 - agreement) * 100, 1),
                "proxy_threshold": PROXY_OCCLUDED_IOU,
                "cvat_threshold": CVAT_OCCLUDED_FRAC,
                "detail": [
                    {"proxy_occ": p, "cvat_occ": c, "cvat_frac": cf, "max_gt_iou": gi}
                    for p, c, cf, gi in cross_check_pairs
                ],
                "note": "Join verified by spatial overlap (IoU>=0.3, 70% consensus).",
            }
            if (1 - agreement) > 0.20:
                clip_result["cvat_cross_check"]["WARNING"] = (
                    f"Divergence {(1-agreement)*100:.0f}% exceeds 20% threshold. "
                    "Proxy IoU threshold may need revisiting."
                )
            logger.info(
                f"  {clip_id} CVAT cross-check: {total_pairs} pairs, "
                f"agreement={agreement*100:.0f}%, divergence={(1-agreement)*100:.0f}%"
            )
        elif clip_id == VID1_CLIP_ID:
            clip_result["cvat_cross_check"] = {
                "status": "cross-check not established" if cvat_to_gt is None
                else "no cross-check pairs found",
            }

        results[clip_id] = clip_result
        logger.info(
            f"  {clip_id}: {total} miss -> {len(proxy_occluded)} proxy-occluded, "
            f"{len(edge_roi)} edge-ROI, {len(detector_fail)} detector-fail"
        )

    results["metric_definition"] = (
        "Miss floor split for tagged athlete's GT frames with no detection. "
        f"Proxy-occluded: max IoU with other GT boxes >= {PROXY_OCCLUDED_IOU} (PROXY). "
        "Edge-ROI: GT bbox center outside roi_mask.png. "
        "Detector-fail-on-visible: not proxy-occluded AND inside ROI."
    )
    return results


# ---------------------------------------------------------------------------
# M4: Synthesis — true addressable ceiling
# ---------------------------------------------------------------------------


def measurement_4(gt_data, m2_results, m3_results):
    """Partition tagged athlete's GT frames into six buckets."""
    logger.info("=== M4: True addressable ceiling ===")

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

        tight = sum(1 for r in tagged_recs if r["classification"] == "tight_match")
        pair_box = sum(1 for r in tagged_recs if r["classification"] == "pair_box")
        miss = sum(1 for r in tagged_recs if r["classification"] == "miss")

        # Get M2 pair-box split at "engage" threshold
        m2_clip = m2_results.get("per_clip", {}).get(clip_id, {})
        m2_engage = m2_clip.get("proximity_sweep", {}).get("engage", {})
        pb_correct_group = m2_engage.get("correct_group", 0)
        pb_mishandled = m2_engage.get("mishandled", 0)
        pb_other = pair_box - pb_correct_group - pb_mishandled

        # Get M3 miss split
        m3_clip = m3_results.get(clip_id, {})
        miss_proxy_occ = m3_clip.get("proxy_occluded", 0)
        miss_edge_roi = m3_clip.get("edge_roi", 0)
        miss_detector_fail = m3_clip.get("detector_fail", 0)

        partition = {
            "1_addressable_ceiling": {
                "count": tight,
                "pct": round(tight / total * 100, 1) if total else 0,
                "description": "Own clean detection (tight_match). UPPER BOUND on what "
                               "appearance-stitch (CP21) can address. Realized gain bounded "
                               "below by ILP mis-stitch rate on these tracklets (CP-PURITY-1 "
                               "showed 100% of entity corruption is ILP stitch on clean tracklets).",
            },
            "2_designed_group_ambiguity": {
                "count": pb_correct_group,
                "pct": round(pb_correct_group / total * 100, 1) if total else 0,
                "description": "Pair-box with correct GROUP node. Designed ambiguity, "
                               "recoverable via match-window delivery.",
            },
            "3_group_formation_defect": {
                "count": pb_mishandled,
                "pct": round(pb_mishandled / total * 100, 1) if total else 0,
                "description": "Pair-box that should have been grouped but wasn't. "
                               "D1 graph construction arc.",
            },
            "4_pair_box_other": {
                "count": pb_other,
                "pct": round(pb_other / total * 100, 1) if total else 0,
                "description": "Pair-box frames not classified as correct-group or mishandled "
                               "(spurious group or neither at engage threshold).",
            },
            "5_miss_accept_occluded": {
                "count": miss_proxy_occ,
                "pct": round(miss_proxy_occ / total * 100, 1) if total else 0,
                "description": "No detection, proxy-occluded (high GT-GT overlap). "
                               "Accept and move on.",
            },
            "6_miss_edge_roi": {
                "count": miss_edge_roi,
                "pct": round(miss_edge_roi / total * 100, 1) if total else 0,
                "description": "No detection, outside ROI mask. Geometry/config arc.",
            },
            "7_miss_detector_fail": {
                "count": miss_detector_fail,
                "pct": round(miss_detector_fail / total * 100, 1) if total else 0,
                "description": "No detection, visible and inside ROI. "
                               "Trainable detector miss (CP23 arc).",
            },
        }

        # Verify partition sums to total
        partition_sum = sum(p["count"] for p in partition.values())
        assert partition_sum == total, f"Partition {partition_sum} != total {total}"

        results[clip_id] = {
            "total_gt_frames": total,
            "partition": partition,
        }

        logger.info(f"  {clip_id}: total={total}")
        for k, v in partition.items():
            logger.info(f"    {k}: {v['count']} ({v['pct']}%)")

    results["metric_definition"] = (
        "Six-bucket partition of tagged athlete's GT frames. "
        "Bucket 1 (tight_match) is the UPPER BOUND on appearance-stitch addressable "
        "gap — NOT a prediction of realized gain. "
        "Pair-box split uses 'engage' (1.5m) proximity threshold. "
        "Miss split uses GT-overlap proxy for occlusion (PROXY, not ground truth)."
    )
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_report(m1, m2, m3, m4):
    """Write synthesis report."""
    report = "# CP-PURITY-2: Aggregate Reconciliation + Floor Decomposition\n\n"

    # M1
    report += "## M1: Aggregate Reconciliation\n\n"
    report += "| Basis | correct_id | Notes |\n"
    report += "|-------|-----------|-------|\n"
    val = m1.get("val_split_clip_level", {})
    report += f"| J_EDEw val-split, clip-level (CP-TAG-4a) | **{val.get('correct_id_pct', '?')}%** | Apples-to-apples comparison |\n"
    report += f"| J_EDEw val-split, clip-level (baseline) | **40.5%** | Signal-trace, pre-CP-TAG-4a |\n"
    report += f"| Delta | **{val.get('delta', '?'):+}pp** | |\n"
    fr_c1 = m1.get("full_range_clip_level_vid1", {})
    fr_s1 = m1.get("full_range_session_level_vid1", {})
    fr_c2 = m1.get("full_range_clip_level_vid2", {})
    fr_s2 = m1.get("full_range_session_level_vid2", {})
    report += f"| Vid1 full-range, clip-level | {fr_c1.get('correct_id_pct', '?')}% | |\n"
    report += f"| Vid1 full-range, session-level | {fr_s1.get('correct_id_pct', '?')}% | |\n"
    report += f"| Vid2 full-range, clip-level | {fr_c2.get('correct_id_pct', '?')}% | |\n"
    report += f"| Vid2 full-range, session-level | {fr_s2.get('correct_id_pct', '?')}% | |\n"
    report += f"\n**Verdict:** {m1.get('verdict', '?')}\n\n"
    report += f"**Canonical definition:** {m1.get('canonical_definition', '')}\n\n"

    # M2
    report += "## M2: Pair-box Floor Split\n\n"
    report += "| Clip | Proximity | Pair-box | Correct-group | Mishandled | Spurious | Neither |\n"
    report += "|------|-----------|----------|--------------|------------|----------|--------|\n"
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        clip_data = m2.get("per_clip", {}).get(clip_id, {})
        for prox_label, data in clip_data.get("proximity_sweep", {}).items():
            short_clip = "vid1" if "200015" in clip_id else "vid2"
            report += (
                f"| {short_clip} | {prox_label} ({data['proximity_m']}m) | {data['n_pair_box']} | "
                f"{data['correct_group']} ({data['correct_group_pct']}%) | "
                f"{data['mishandled']} ({data['mishandled_pct']}%) | "
                f"{data['spurious_group']} | {data['neither']} |\n"
            )
    stab = m2.get("stability", {})
    report += f"\nMishandled% stability: {'STABLE' if stab.get('verdict_stable') else 'UNSTABLE'} "
    report += f"(range: {stab.get('mishandled_pct_range', [])})\n\n"

    # M3
    report += "## M3: Miss Floor Split\n\n"
    report += "| Clip | Miss | Proxy-occluded | Edge-ROI | Detector-fail |\n"
    report += "|------|------|---------------|----------|---------------|\n"
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        m3c = m3.get(clip_id, {})
        short_clip = "vid1" if "200015" in clip_id else "vid2"
        report += (
            f"| {short_clip} | {m3c.get('n_miss', 0)} | "
            f"{m3c.get('proxy_occluded', 0)} ({m3c.get('proxy_occluded_pct', 0)}%) | "
            f"{m3c.get('edge_roi', 0)} ({m3c.get('edge_roi_pct', 0)}%) | "
            f"{m3c.get('detector_fail', 0)} ({m3c.get('detector_fail_pct', 0)}%) |\n"
        )

    # CVAT cross-check
    vid1_cc = m3.get(VID1_CLIP_ID, {}).get("cvat_cross_check", {})
    if "agreement_pct" in vid1_cc:
        report += f"\n### CVAT Cross-check (vid1 only)\n\n"
        report += f"Join verified by spatial overlap (IoU>=0.3, 70% consensus).\n\n"
        report += "| | CVAT-occluded | CVAT-not-occluded |\n"
        report += "|---|---|---|\n"
        report += f"| Proxy-occluded | {vid1_cc['both_occluded']} | {vid1_cc['proxy_only']} |\n"
        report += f"| Proxy-not-occluded | {vid1_cc['cvat_only']} | {vid1_cc['neither_occluded']} |\n"
        report += f"\nAgreement: {vid1_cc['agreement_pct']}%\n"
        if "WARNING" in vid1_cc:
            report += f"\n**WARNING:** {vid1_cc['WARNING']}\n"
    elif "status" in vid1_cc:
        report += f"\n### CVAT Cross-check: {vid1_cc['status']}\n"
    report += "\n"

    # M4
    report += "## M4: True Addressable Ceiling\n\n"
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        m4c = m4.get(clip_id, {})
        short_clip = "vid1" if "200015" in clip_id else "vid2"
        report += f"### {short_clip} ({m4c.get('total_gt_frames', '?')} GT frames)\n\n"
        report += "| Bucket | Count | % | Owner |\n"
        report += "|--------|-------|---|-------|\n"
        for k, v in m4c.get("partition", {}).items():
            owner = ""
            if "addressable" in k:
                owner = "Appearance-stitch (CP21) UPPER BOUND"
            elif "designed" in k:
                owner = "Window delivery"
            elif "group_formation" in k:
                owner = "D1 graph (structural)"
            elif "accept" in k:
                owner = "Accept"
            elif "edge_roi" in k:
                owner = "Geometry/config"
            elif "detector" in k:
                owner = "Detection model (CP23)"
            elif "other" in k:
                owner = "Mixed/uncategorized"
            report += f"| {k} | {v['count']} | {v['pct']}% | {owner} |\n"
        report += "\n"

    report += "## Options for Web Session\n\n"
    report += "1. **Appearance-stitch (CP21):** Bucket 1 sets the upper bound. "
    report += "Realized gain depends on how much ILP mis-stitch appearance evidence can prevent.\n\n"
    report += "2. **Detection model (CP23):** Bucket 7 is trainable detector misses — "
    report += "more training data or architecture changes.\n\n"
    report += "3. **Geometry/config:** Bucket 6 — ROI mask application or camera repositioning.\n\n"
    report += "4. **D1 graph:** Bucket 3 — group-formation defects on pair-box frames.\n\n"
    report += "5. **Accept:** Bucket 5 — occluded frames, no fix possible.\n\n"

    out = EVIDENCE_DIR / "decomposition_report.md"
    with open(out, "w") as f:
        f.write(report)
    logger.info(f"  Wrote {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("CP-PURITY-2: Aggregate reconciliation + floor decomposition")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    gt_data = _load_gt_for_clips()

    m1 = measurement_1(gt_data)
    m2 = measurement_2(gt_data)
    m3 = measurement_3(gt_data)
    m4 = measurement_4(gt_data, m2, m3)

    # Write JSON outputs
    for name, data in [("m1_reconciliation", m1), ("m2_pairbox", m2),
                        ("m3_miss", m3), ("m4_ceiling", m4)]:
        out = EVIDENCE_DIR / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Wrote {out.name}")

    _write_report(m1, m2, m3, m4)
    logger.info("CP-PURITY-2 complete.")


if __name__ == "__main__":
    main()
