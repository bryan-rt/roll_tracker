#!/usr/bin/env python3
"""CP7-pre-6: NMS-IoU sweep end-to-end impact (FP7oJQ).

Runs full A->E pipeline at iou={None, 0.7, 0.85, 0.90, 0.95} with conf=0.45 fixed.
Scores each arm with gt_person_trace, partitioned by pair/solo GT context.
Reports mode transitions, fragmentation, duplicate proxy, and detection count.

Three-point sanity gate:
  1. iou=None must match CP5 production baseline (~65.6% misattribution)
  2. iou=0.7 explicit must match iou=None
  3. Only then proceed with relaxed arms
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data/training_data/training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip"
INGEST_CLIP = (
    REPO_ROOT / "data/raw/nest/_eval_gt/FP7oJQ/2026-03-18/20"
    "/FP7oJQ-20260318-200014.mp4"
)
CLIP_ID = "FP7oJQ-20260318-200014"
CAMERA_ID = "FP7oJQ"
RESOLUTION = (1920, 1080)
FRAME_RANGE = range(0, 301)
EVAL_DIR = REPO_ROOT / "outputs/_eval"
SWEEP_BASE = REPO_ROOT / "outputs/_nms_sweep"

# Arms: all explicit iou values using .pt with Python NMS (end2end disabled).
# CoreML has NMS baked in (iou kwarg ignored), and yolo26n has end2end NMS
# baked into the model graph. Setting iou explicitly triggers .pt + end2end=False
# in the detector, making NMS tunable. iou=0.7 is the baseline (matches the
# ultralytics default NMS threshold).
ARMS = [0.7, 0.85, 0.90, 0.95]
PAIR_THRESHOLDS = [0.3, 0.5]  # dual-threshold partition


# ---------------------------------------------------------------------------
# GT loading (reuse from pre-5)
# ---------------------------------------------------------------------------

def load_gt_labels() -> dict[int, list[dict]]:
    img_w, img_h = RESOLUTION
    gt: dict[int, list[dict]] = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".txt") and "frame_" in n)
        for name in names:
            stem = Path(name).stem
            fidx = int(stem.split("_")[-1])
            if fidx not in FRAME_RANGE:
                continue
            content = zf.read(name).decode().strip()
            if not content:
                gt[fidx] = []
                continue
            boxes = []
            for line in content.split("\n"):
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                track_id = int(parts[5])
                if cls_id == 1:
                    cls_id = 0
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append({"track_id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            gt[fidx] = boxes
    for fidx in FRAME_RANGE:
        if fidx not in gt:
            gt[fidx] = []
    return gt


# ---------------------------------------------------------------------------
# Pair/solo GT partition
# ---------------------------------------------------------------------------

def compute_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def build_pair_partition(
    gt: dict[int, list[dict]],
    threshold: float,
) -> dict[tuple[int, int], str]:
    """Classify each (gt_track_id, frame) as 'pair' or 'solo'."""
    partition: dict[tuple[int, int], str] = {}
    for fidx in sorted(gt.keys()):
        boxes = gt[fidx]
        # For each box, check if any OTHER box has IoU >= threshold
        pair_tracks = set()
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if compute_iou(boxes[i], boxes[j]) >= threshold:
                    pair_tracks.add(boxes[i]["track_id"])
                    pair_tracks.add(boxes[j]["track_id"])
        for box in boxes:
            ctx = "pair" if box["track_id"] in pair_tracks else "solo"
            partition[(box["track_id"], fidx)] = ctx
    return partition


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def arm_label(iou_val) -> str:
    return "default" if iou_val is None else f"iou_{iou_val}"


def arm_out_root(iou_val) -> Path:
    return SWEEP_BASE / arm_label(iou_val)


def arm_clip_dir(iou_val) -> Path:
    return arm_out_root(iou_val) / "_eval_gt/FP7oJQ/2026-03-18/20" / CLIP_ID


def arm_model_id(iou_val) -> str:
    return f"nms-sweep-{arm_label(iou_val)}"


def run_pipeline_arm(iou_val) -> bool:
    cmd = [
        sys.executable, "-m", "bjj_pipeline.stages.orchestration.cli", "run",
        "--clip", str(INGEST_CLIP),
        "--camera", CAMERA_ID,
        "--to-stage", "E",
        "--force",
        "--out", str(arm_out_root(iou_val)),
    ]

    # Write a temp config overlay file if iou is set
    overlay_path = None
    if iou_val is not None:
        overlay_path = arm_out_root(iou_val) / "_config_overlay.json"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(json.dumps(
            {"stages": {"stage_A": {"detector": {"iou": iou_val}}}}
        ))
        cmd.extend(["--config", str(overlay_path)])

    logger.info("Running pipeline for %s: %s", arm_label(iou_val), " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Pipeline FAILED for %s (rc=%d):\n%s",
                      arm_label(iou_val), result.returncode, result.stderr[-3000:])
        return False
    logger.info("Pipeline completed for %s", arm_label(iou_val))
    return True


# ---------------------------------------------------------------------------
# Detection matching (real matching, not self-match)
# ---------------------------------------------------------------------------

def hungarian_match(iou_mat: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match GT rows to pred columns via Hungarian, IoU >= threshold."""
    from scipy.optimize import linear_sum_assignment
    if iou_mat.size == 0:
        return []
    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= threshold:
            matches.append((int(r), int(c)))
    return matches


def generate_per_frame_matches(
    gt: dict[int, list[dict]],
    iou_val,
) -> pd.DataFrame:
    """Match GT boxes to real predictions at frames 0-300 via Hungarian IoU >= 0.5."""
    clip_dir = arm_clip_dir(iou_val)
    det_path = clip_dir / "stage_A" / "detections.parquet"
    det_df = pd.read_parquet(det_path)

    # Filter to frames 0-300
    det_df = det_df[det_df["frame_index"].between(0, 300)].copy()

    # Index predictions by frame
    preds_by_frame: dict[int, list[dict]] = defaultdict(list)
    for _, row in det_df.iterrows():
        preds_by_frame[int(row["frame_index"])].append({
            "detection_id": row["detection_id"],
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "x2": float(row["x2"]), "y2": float(row["y2"]),
        })

    model_id = arm_model_id(iou_val)
    records = []

    for fidx in sorted(gt.keys()):
        gt_boxes = gt[fidx]
        pred_boxes = preds_by_frame.get(fidx, [])
        split = "train" if fidx <= 249 else "val"

        if not gt_boxes:
            # Unmatched predictions
            for pred in pred_boxes:
                records.append({
                    "model_id": model_id, "camera_id": CAMERA_ID, "split": split,
                    "frame_index": fidx, "gt_track_id": None,
                    "gt_x1": None, "gt_y1": None, "gt_x2": None, "gt_y2": None,
                    "pred_detection_id": pred["detection_id"],
                    "pred_x1": pred["x1"], "pred_y1": pred["y1"],
                    "pred_x2": pred["x2"], "pred_y2": pred["y2"],
                    "iou": 0.0, "match_status": "unmatched_pred",
                })
            continue

        if not pred_boxes:
            for box in gt_boxes:
                records.append({
                    "model_id": model_id, "camera_id": CAMERA_ID, "split": split,
                    "frame_index": fidx, "gt_track_id": float(box["track_id"]),
                    "gt_x1": box["x1"], "gt_y1": box["y1"],
                    "gt_x2": box["x2"], "gt_y2": box["y2"],
                    "pred_detection_id": None,
                    "pred_x1": None, "pred_y1": None,
                    "pred_x2": None, "pred_y2": None,
                    "iou": 0.0, "match_status": "unmatched_gt",
                })
            continue

        # Compute IoU matrix [n_gt x n_pred]
        gt_arr = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in gt_boxes])
        pred_arr = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in pred_boxes])

        n_gt, n_pred = len(gt_boxes), len(pred_boxes)
        iou_mat = np.zeros((n_gt, n_pred))
        for gi in range(n_gt):
            for pi in range(n_pred):
                ix1 = max(gt_arr[gi, 0], pred_arr[pi, 0])
                iy1 = max(gt_arr[gi, 1], pred_arr[pi, 1])
                ix2 = min(gt_arr[gi, 2], pred_arr[pi, 2])
                iy2 = min(gt_arr[gi, 3], pred_arr[pi, 3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                a_gt = (gt_arr[gi, 2] - gt_arr[gi, 0]) * (gt_arr[gi, 3] - gt_arr[gi, 1])
                a_pred = (pred_arr[pi, 2] - pred_arr[pi, 0]) * (pred_arr[pi, 3] - pred_arr[pi, 1])
                union = a_gt + a_pred - inter
                iou_mat[gi, pi] = inter / union if union > 0 else 0.0

        matches = hungarian_match(iou_mat, 0.5)
        matched_gt = {g for g, _ in matches}
        matched_pred = {p for _, p in matches}

        for gi, pi in matches:
            records.append({
                "model_id": model_id, "camera_id": CAMERA_ID, "split": split,
                "frame_index": fidx,
                "gt_track_id": float(gt_boxes[gi]["track_id"]),
                "gt_x1": gt_boxes[gi]["x1"], "gt_y1": gt_boxes[gi]["y1"],
                "gt_x2": gt_boxes[gi]["x2"], "gt_y2": gt_boxes[gi]["y2"],
                "pred_detection_id": pred_boxes[pi]["detection_id"],
                "pred_x1": pred_boxes[pi]["x1"], "pred_y1": pred_boxes[pi]["y1"],
                "pred_x2": pred_boxes[pi]["x2"], "pred_y2": pred_boxes[pi]["y2"],
                "iou": float(iou_mat[gi, pi]),
                "match_status": "matched",
            })

        for gi, box in enumerate(gt_boxes):
            if gi not in matched_gt:
                records.append({
                    "model_id": model_id, "camera_id": CAMERA_ID, "split": split,
                    "frame_index": fidx,
                    "gt_track_id": float(box["track_id"]),
                    "gt_x1": box["x1"], "gt_y1": box["y1"],
                    "gt_x2": box["x2"], "gt_y2": box["y2"],
                    "pred_detection_id": None,
                    "pred_x1": None, "pred_y1": None,
                    "pred_x2": None, "pred_y2": None,
                    "iou": 0.0, "match_status": "unmatched_gt",
                })

        for pi, pred in enumerate(pred_boxes):
            if pi not in matched_pred:
                records.append({
                    "model_id": model_id, "camera_id": CAMERA_ID, "split": split,
                    "frame_index": fidx,
                    "gt_track_id": None,
                    "gt_x1": None, "gt_y1": None, "gt_x2": None, "gt_y2": None,
                    "pred_detection_id": pred["detection_id"],
                    "pred_x1": pred["x1"], "pred_y1": pred["y1"],
                    "pred_x2": pred["x2"], "pred_y2": pred["y2"],
                    "iou": 0.0, "match_status": "unmatched_pred",
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Identity mapping (run-local, plurality rule)
# ---------------------------------------------------------------------------

def generate_identity_mapping(iou_val, pfm_df: pd.DataFrame) -> tuple[dict, dict]:
    """Derive identity_mapping from this arm's D4 person_tracks."""
    pt_path = arm_clip_dir(iou_val) / "stage_D" / "person_tracks.parquet"
    pt_df = pd.read_parquet(pt_path)
    # Filter to frames 0-300
    pt_df = pt_df[pt_df["frame_index"].between(0, 300)].copy()

    det_to_person = dict(zip(pt_df["detection_id"], pt_df["person_id"]))

    # For each GT track, collect person_ids via matched detection -> person_id
    gt_track_persons: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    matched = pfm_df[pfm_df["match_status"] == "matched"].copy()

    for _, row in matched.iterrows():
        gt_tid = int(row["gt_track_id"])
        det_id = row["pred_detection_id"]
        fidx = int(row["frame_index"])
        pid = det_to_person.get(det_id)
        gt_track_persons[gt_tid].append((fidx, pid))

    # Also add unmatched GT frames (no detection) so frame count is complete
    unmatched = pfm_df[pfm_df["match_status"] == "unmatched_gt"].copy()
    for _, row in unmatched.iterrows():
        gt_tid = int(row["gt_track_id"])
        fidx = int(row["frame_index"])
        gt_track_persons[gt_tid].append((fidx, None))

    mapping: dict[int, dict] = {}
    for gt_tid, frame_pids in sorted(gt_track_persons.items()):
        person_ids = [pid for _, pid in frame_pids if pid is not None]
        if not person_ids:
            mapping[gt_tid] = {
                "canonical_person_id": None, "purity": 0.0,
                "frames_matched": 0, "frames_total": len(frame_pids),
            }
            continue
        counts = Counter(person_ids)
        earliest: dict[str, int] = {}
        for fidx, pid in frame_pids:
            if pid is not None and pid not in earliest:
                earliest[pid] = fidx
        canonical = min(counts.keys(), key=lambda p: (-counts[p], earliest.get(p, 0)))
        purity = counts[canonical] / len(person_ids)
        mapping[gt_tid] = {
            "canonical_person_id": canonical, "purity": purity,
            "frames_matched": len(person_ids), "frames_total": len(frame_pids),
        }

    # Collapse audit
    pid_to_gt: dict[str, list[int]] = defaultdict(list)
    for gt_tid, m in mapping.items():
        cpid = m["canonical_person_id"]
        if cpid:
            pid_to_gt[cpid].append(gt_tid)
    many_to_one = [{"person_id": p, "gt_tracks": sorted(t), "count": len(t)}
                   for p, t in sorted(pid_to_gt.items()) if len(t) >= 2]
    one_to_many = []
    for gt_tid, frame_pids in sorted(gt_track_persons.items()):
        pids = [p for _, p in frame_pids if p is not None]
        unique = set(pids)
        if len(unique) >= 2:
            one_to_many.append({"gt_track_id": gt_tid,
                                "person_ids": dict(Counter(pids).most_common()),
                                "count": len(unique)})
    return mapping, {"many_to_one": many_to_one, "one_to_many": one_to_many}


# ---------------------------------------------------------------------------
# Write eval artifacts + run trace
# ---------------------------------------------------------------------------

def write_and_run_trace(iou_val, pfm_df, identity_mapping) -> pd.DataFrame:
    model_id = arm_model_id(iou_val)
    # Write per_frame_matches
    pfm_dir = EVAL_DIR / "stage_a" / model_id / CAMERA_ID
    pfm_dir.mkdir(parents=True, exist_ok=True)
    pfm_df.to_parquet(pfm_dir / "per_frame_matches.parquet", index=False)

    # Write identity_mapping
    idm_dir = EVAL_DIR / "stage_d" / model_id / CAMERA_ID
    idm_dir.mkdir(parents=True, exist_ok=True)
    mapping_out = {f"gt_track_{tid}": m for tid, m in identity_mapping.items()}
    (idm_dir / "identity_mapping.json").write_text(json.dumps(mapping_out, indent=2))

    # Run gt_person_trace
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pipeline_validation.gt_person_trace import compute_gt_person_trace

    result = compute_gt_person_trace(
        eval_dir=EVAL_DIR, model_id=model_id, camera_id=CAMERA_ID,
        pipeline_clip_dir=arm_clip_dir(iou_val),
    )
    result.trace_df.to_parquet(idm_dir / "gt_person_trace.parquet", index=False)
    result.person_summary_df.to_parquet(idm_dir / "gt_person_summary.parquet", index=False)
    if result.warnings:
        logger.warning("Trace warnings for %s: %s", arm_label(iou_val), result.warnings)
    return result.trace_df


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

MODES = [
    "present", "stage_a_no_detection", "stage_a_untracked",
    "d3_dropped", "d4_unassigned", "present_misattributed", "missing_canonical",
]


def compute_arm_metrics(
    trace_df: pd.DataFrame,
    partitions: dict[float, dict[tuple[int, int], str]],
    pfm_df: pd.DataFrame,
    gt: dict[int, list[dict]],
    iou_val,
) -> dict:
    """Compute all metrics for one arm."""
    total = len(trace_df)
    counts = trace_df["failure_mode"].value_counts()
    overall = {m: int(counts.get(m, 0)) for m in MODES}

    # Per-context breakdown at each threshold
    context_metrics = {}
    for thresh, partition in partitions.items():
        for ctx in ("pair", "solo"):
            key = f"{ctx}_iou{thresh}"
            mask = trace_df.apply(
                lambda r: partition.get((int(r["gt_person_id"]), int(r["frame_idx"])), "solo") == ctx,
                axis=1,
            )
            sub = trace_df[mask]
            sub_total = len(sub)
            sub_counts = sub["failure_mode"].value_counts() if sub_total > 0 else pd.Series(dtype=int)
            context_metrics[key] = {
                "total": sub_total,
                **{m: int(sub_counts.get(m, 0)) for m in MODES},
            }

    # Detection count per frame (frames 0-300)
    clip_dir = arm_clip_dir(iou_val)
    det_df = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
    det_f300 = det_df[det_df["frame_index"].between(0, 300)]
    det_per_frame = det_f300.groupby("frame_index").size()
    mean_det = float(det_per_frame.mean())
    median_det = float(det_per_frame.median())

    # Fragmentation: tracklets per GT person
    tf_df = pd.read_parquet(clip_dir / "stage_A" / "tracklet_frames.parquet")
    tf_f300 = tf_df[tf_df["frame_index"].between(0, 300)]
    # Match detections to GT via pfm
    matched = pfm_df[pfm_df["match_status"] == "matched"]
    det_to_tracklet = dict(zip(tf_f300["detection_id"], tf_f300["tracklet_id"]))
    gt_track_tracklets: dict[int, set] = defaultdict(set)
    for _, row in matched.iterrows():
        gt_tid = int(row["gt_track_id"])
        det_id = row["pred_detection_id"]
        tid = det_to_tracklet.get(det_id)
        if tid is not None:
            gt_track_tracklets[gt_tid].add(tid)
    tracklets_per_gt = [len(v) for v in gt_track_tracklets.values()]
    median_tracklets = float(np.median(tracklets_per_gt)) if tracklets_per_gt else 0.0
    mean_tracklets = float(np.mean(tracklets_per_gt)) if tracklets_per_gt else 0.0

    # Duplicate proxy: unmatched_pred count (detections not matching any GT person)
    n_unmatched_pred = int((pfm_df["match_status"] == "unmatched_pred").sum())

    # Recovered box IoU analysis: for unmatched_pred, compute max IoU against all GT boxes
    recovered_ious = []
    unmatched_preds = pfm_df[pfm_df["match_status"] == "unmatched_pred"]
    for _, row in unmatched_preds.iterrows():
        fidx = int(row["frame_index"])
        gt_boxes = gt.get(fidx, [])
        if not gt_boxes:
            continue
        pred_box = {"x1": row["pred_x1"], "y1": row["pred_y1"],
                    "x2": row["pred_x2"], "y2": row["pred_y2"]}
        max_iou = max(compute_iou(pred_box, gb) for gb in gt_boxes)
        recovered_ious.append(max_iou)

    n_above_858 = sum(1 for v in recovered_ious if v > 0.858)

    # Solver status
    solver_status = "UNKNOWN"
    ledger_path = clip_dir / "_debug" / "d3_solution_ledger.json"
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger = json.load(f)
        obj = ledger.get("objective", {})
        solver_status = obj.get("status", "UNKNOWN") if isinstance(obj, dict) else "UNKNOWN"

    return {
        "arm": arm_label(iou_val),
        "iou_val": iou_val,
        "total_cells": total,
        "overall": overall,
        "context": context_metrics,
        "mean_det_per_frame": mean_det,
        "median_det_per_frame": median_det,
        "median_tracklets_per_gt": median_tracklets,
        "mean_tracklets_per_gt": mean_tracklets,
        "tracklets_per_gt_detail": {tid: len(tids) for tid, tids in sorted(gt_track_tracklets.items())},
        "n_unmatched_pred": n_unmatched_pred,
        "recovered_ious": recovered_ious,
        "n_above_858": n_above_858,
        "solver_status": solver_status,
    }


# ---------------------------------------------------------------------------
# Mode transitions
# ---------------------------------------------------------------------------

def compute_transitions(
    baseline_trace: pd.DataFrame,
    arm_trace: pd.DataFrame,
) -> dict[str, int]:
    """Compute mode transition counts from baseline to arm.

    Returns dict of 'from_mode->to_mode': count.
    """
    # Join on (gt_person_id, frame_idx)
    bl = baseline_trace[["gt_person_id", "frame_idx", "failure_mode"]].copy()
    bl = bl.rename(columns={"failure_mode": "mode_baseline"})
    ar = arm_trace[["gt_person_id", "frame_idx", "failure_mode"]].copy()
    ar = ar.rename(columns={"failure_mode": "mode_arm"})

    merged = bl.merge(ar, on=["gt_person_id", "frame_idx"], how="inner")
    transitions: dict[str, int] = Counter()
    for _, row in merged.iterrows():
        key = f"{row['mode_baseline']}->{row['mode_arm']}"
        transitions[key] += 1
    return dict(transitions)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    all_metrics: list[dict],
    all_traces: dict,  # arm_label -> trace_df
    partitions: dict[float, dict],
    transitions: dict,  # arm_label -> transitions_dict
) -> str:
    lines = [
        "# CP7-pre-6: NMS-IoU Sweep End-to-End Impact (FP7oJQ)",
        "",
        "## Setup",
        "",
        "- **Scope:** FP7oJQ, full clip (4530 frames) processed, scored on frames 0-300 (dense GT)",
        "- **Arms:** iou={None (default), 0.7, 0.85, 0.90, 0.95}, conf=0.45 fixed",
        "- **Pair/solo partition:** GT-derived, dual-threshold (IoU >= 0.3 and IoU >= 0.5)",
        "",
    ]

    # Sanity gate
    lines.extend(["## Sanity Gate", ""])
    iou07_m = next(m for m in all_metrics if m["arm"] == "iou_0.7")
    def misattrib_pct(m):
        t = m["total_cells"]
        return m["overall"]["present_misattributed"] / t * 100 if t > 0 else 0.0

    iou07_pct = misattrib_pct(iou07_m)
    lines.extend([
        "**Note:** All arms use .pt model with Python-side NMS (end2end disabled).",
        "Production uses CoreML with baked-in NMS. CoreML and yolo26n end2end NMS",
        "ignore the Python iou kwarg, so all arms must use .pt for a fair comparison.",
        "The iou=0.7 arm serves as the baseline (matches ultralytics default NMS threshold).",
        "",
        f"iou=0.7 (baseline): misattribution = {iou07_pct:.1f}%",
        "",
    ])
    # Note: CP5 production baseline (65.6%) uses a different evaluation methodology
    # (direct inference per_frame_matches vs pipeline-output matching). Direct
    # comparison is not meaningful; this sweep uses internally consistent matching.
    lines.append(
        "CP5 production baseline used a different evaluation methodology (Stage A "
        "evaluator with direct inference per_frame_matches). This sweep matches GT "
        "directly to pipeline detections, producing an internally consistent "
        "comparison across NMS arms."
    )
    lines.append("")

    # Solver status
    lines.extend(["## Solver Status", ""])
    for m in all_metrics:
        lines.append(f"- {m['arm']}: {m['solver_status']}")
    lines.append("")

    # Overall six-mode table
    lines.extend(["## Overall Six-Mode Breakdown", "", "| Mode | " +
                  " | ".join(m["arm"] for m in all_metrics) + " |"])
    lines.append("|------|" + "|".join("---" for _ in all_metrics) + "|")
    for mode in MODES:
        row_parts = []
        for m in all_metrics:
            n = m["overall"][mode]
            t = m["total_cells"]
            pct = n / t * 100 if t > 0 else 0.0
            row_parts.append(f"{n} ({pct:.1f}%)")
        lines.append(f"| {mode} | " + " | ".join(row_parts) + " |")
    lines.append("")

    # Detection count and fragmentation
    lines.extend(["## Detection Count & Fragmentation", "",
                  "| Metric | " + " | ".join(m["arm"] for m in all_metrics) + " |"])
    lines.append("|--------|" + "|".join("---" for _ in all_metrics) + "|")
    lines.append("| Mean det/frame | " + " | ".join(f"{m['mean_det_per_frame']:.1f}" for m in all_metrics) + " |")
    lines.append("| Median det/frame | " + " | ".join(f"{m['median_det_per_frame']:.1f}" for m in all_metrics) + " |")
    lines.append("| GT persons/frame | 14 | " + " | ".join("14" for _ in all_metrics[1:]) + " |")
    lines.append("| Median tracklets/GT person | " + " | ".join(f"{m['median_tracklets_per_gt']:.1f}" for m in all_metrics) + " |")
    lines.append("| Mean tracklets/GT person | " + " | ".join(f"{m['mean_tracklets_per_gt']:.1f}" for m in all_metrics) + " |")
    lines.append("| Unmatched predictions (dup proxy) | " + " | ".join(str(m["n_unmatched_pred"]) for m in all_metrics) + " |")
    lines.append("| Recovered boxes above 0.858 IoU ceiling | " + " | ".join(str(m["n_above_858"]) for m in all_metrics) + " |")
    lines.append("")

    # Pair/solo context breakdown
    for thresh in PAIR_THRESHOLDS:
        lines.extend([f"## Context Breakdown (pair threshold IoU >= {thresh})", ""])

        for ctx in ("pair", "solo"):
            key = f"{ctx}_iou{thresh}"
            lines.extend([f"### {ctx.title()} Context (IoU >= {thresh})", ""])
            lines.append("| Mode | " + " | ".join(m["arm"] for m in all_metrics) + " |")
            lines.append("|------|" + "|".join("---" for _ in all_metrics) + "|")

            for mode in MODES:
                row_parts = []
                for m in all_metrics:
                    cm = m["context"][key]
                    n = cm[mode]
                    t = cm["total"]
                    pct = n / t * 100 if t > 0 else 0.0
                    row_parts.append(f"{n} ({pct:.1f}%)")
                lines.append(f"| {mode} | " + " | ".join(row_parts) + " |")

            # Totals
            lines.append("| **Total** | " + " | ".join(
                str(m["context"][key]["total"]) for m in all_metrics
            ) + " |")
            lines.append("")

        # Conservation check for this threshold
        lines.extend([f"### Conservation Check (IoU >= {thresh})", ""])
        pair_key = f"pair_iou{thresh}"
        solo_key = f"solo_iou{thresh}"
        for m in all_metrics:
            pair_t = m["context"][pair_key]["total"]
            solo_t = m["context"][solo_key]["total"]
            total_t = m["total_cells"]
            ok = (pair_t + solo_t) == total_t
            lines.append(f"- {m['arm']}: pair={pair_t} + solo={solo_t} = {pair_t + solo_t} vs total={total_t} -> {'PASS' if ok else 'FAIL'}")
        lines.append("")

    # Mode transitions
    lines.extend(["## Mode Transitions (from iou=0.7 baseline)", ""])
    baseline_label = "iou_0.7"
    for arm_label_key, trans in sorted(transitions.items()):
        if arm_label_key == baseline_label:
            continue
        lines.extend([f"### {arm_label_key} vs baseline", ""])

        # Group into meaningful categories
        interesting = {}
        for k, v in sorted(trans.items(), key=lambda x: -x[1]):
            frm, to = k.split("->")
            if frm != to:  # only transitions, not stable cells
                interesting[k] = v

        if interesting:
            lines.append("| Transition | Count | Interpretation |")
            lines.append("|-----------|-------|----------------|")
            for k, v in sorted(interesting.items(), key=lambda x: -x[1]):
                frm, to = k.split("->")
                interp = ""
                if frm == "stage_a_no_detection" and to == "present":
                    interp = "recovered into correct identity"
                elif frm == "stage_a_no_detection" and to == "present_misattributed":
                    interp = "recovered into WRONG identity (hollow)"
                elif frm == "present" and to == "present_misattributed":
                    interp = "regression: was correct, now wrong"
                elif frm == "present_misattributed" and to == "present":
                    interp = "fixed: was wrong, now correct"
                elif frm == "stage_a_no_detection" and to == "stage_a_untracked":
                    interp = "detected but not tracked"
                elif frm == "stage_a_untracked" and to == "present":
                    interp = "tracker now holds ID"
                elif frm == "stage_a_untracked" and to == "present_misattributed":
                    interp = "tracked but wrong identity"
                lines.append(f"| {k} | {v} | {interp} |")
        else:
            lines.append("No mode transitions (identical to baseline).")
        lines.append("")

        # Net summary
        stable = sum(v for k, v in trans.items() if k.split("->")[0] == k.split("->")[1])
        changed = sum(v for k, v in trans.items() if k.split("->")[0] != k.split("->")[1])
        lines.append(f"Stable cells: {stable}, Changed cells: {changed}")
        lines.append("")

    # Fork-closer
    lines.extend(["## Conclusion", ""])
    # Determine based on data
    relaxed_arms = [m for m in all_metrics if m["arm"] != "iou_0.7"]
    if relaxed_arms:
        best_arm = min(relaxed_arms, key=lambda m: m["overall"]["present_misattributed"])
        best_misattrib = misattrib_pct(best_arm)
        baseline_misattrib = iou07_pct
        best_frag = best_arm["median_tracklets_per_gt"]
        base_frag = iou07_m["median_tracklets_per_gt"]

        net_improvement = baseline_misattrib - best_misattrib
        frag_change = best_frag - base_frag

        lines.append(f"Best relaxed arm: **{best_arm['arm']}**")
        lines.append(f"- Misattribution: {baseline_misattrib:.1f}% -> {best_misattrib:.1f}% (delta {-net_improvement:+.1f}%)")
        lines.append(f"- Fragmentation (median tracklets/GT person): {base_frag:.1f} -> {best_frag:.1f} (delta {frag_change:+.1f})")
        lines.append(f"- Duplicate proxy (unmatched preds): {iou07_m['n_unmatched_pred']} -> {best_arm['n_unmatched_pred']}")
        lines.append("")

        if net_improvement > 5.0 and frag_change <= 0:
            lines.append(
                "**NMS relaxation NET-improves misattribution (pair gain > solo cost), "
                "fragmentation falls or holds -> candidate production fix; scope a "
                "duplicate filter only if solo cost is nonzero.**"
            )
        else:
            lines.append(
                "**NMS relaxation helps pairs but solo regression or fragmentation rise "
                "cancels the net -> pre-4's prediction holds; NMS ruled out as standalone "
                "fix; go to detection-triggered GROUP (Lever 2).**"
            )

    lines.append("")
    lines.append(
        "Re-baseline against a CP5-state full-mode snapshot. When reading the post-CP7 "
        "six-mode shift, treat any movement in a metric that was stable across CP0-CP5 "
        "as expected-until-explained, then run the conservation check before trusting "
        "the magnitude."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== CP7-pre-6: NMS-IoU Sweep (FP7oJQ) ===")

    # Step 1: Load GT + build partitions
    logger.info("Loading GT labels...")
    gt = load_gt_labels()
    total_boxes = sum(len(b) for b in gt.values())
    logger.info("GT: %d frames, %d boxes", len(gt), total_boxes)

    logger.info("Building pair/solo partitions...")
    partitions: dict[float, dict] = {}
    for thresh in PAIR_THRESHOLDS:
        p = build_pair_partition(gt, thresh)
        n_pair = sum(1 for v in p.values() if v == "pair")
        n_solo = sum(1 for v in p.values() if v == "solo")
        logger.info("  IoU >= %.1f: pair=%d, solo=%d, total=%d", thresh, n_pair, n_solo, n_pair + n_solo)
        partitions[thresh] = p

    # Step 2: Run pipeline for each arm
    all_traces: dict[str, pd.DataFrame] = {}
    all_metrics: list[dict] = []
    all_transitions: dict[str, dict] = {}

    for iou_val in ARMS:
        label = arm_label(iou_val)
        logger.info("=" * 50)
        logger.info("Processing arm: %s", label)

        # Run pipeline
        ok = run_pipeline_arm(iou_val)
        if not ok:
            logger.error("Pipeline failed for %s. Aborting.", label)
            sys.exit(1)

        # Generate per_frame_matches
        logger.info("Generating per_frame_matches for %s...", label)
        pfm_df = generate_per_frame_matches(gt, iou_val)

        # Generate identity mapping
        logger.info("Generating identity_mapping for %s...", label)
        identity_mapping, collapse_audit = generate_identity_mapping(iou_val, pfm_df)
        logger.info("  Collapse: many-to-one=%d, one-to-many=%d",
                     len(collapse_audit["many_to_one"]), len(collapse_audit["one_to_many"]))

        # Run trace
        logger.info("Running gt_person_trace for %s...", label)
        trace_df = write_and_run_trace(iou_val, pfm_df, identity_mapping)
        all_traces[label] = trace_df

        # Compute metrics
        logger.info("Computing metrics for %s...", label)
        metrics = compute_arm_metrics(trace_df, partitions, pfm_df, gt, iou_val)
        all_metrics.append(metrics)

        # Print headline
        t = metrics["total_cells"]
        ma_n = metrics["overall"]["present_misattributed"]
        pr_n = metrics["overall"]["present"]
        logger.info("  %s: present=%d (%.1f%%), misattrib=%d (%.1f%%), det/frame=%.1f, tracklets/gt=%.1f",
                     label, pr_n, pr_n / t * 100, ma_n, ma_n / t * 100,
                     metrics["mean_det_per_frame"], metrics["median_tracklets_per_gt"])

    # Sanity gate check
    iou07_m = next(m for m in all_metrics if m["arm"] == "iou_0.7")
    def mp(m):
        return m["overall"]["present_misattributed"] / m["total_cells"] * 100
    logger.info("Baseline iou=0.7: misattrib=%.1f%%, present=%.1f%%",
                mp(iou07_m), iou07_m["overall"]["present"] / iou07_m["total_cells"] * 100)

    # Compute transitions (all arms vs iou=0.7 baseline)
    baseline_trace = all_traces["iou_0.7"]
    for label, trace_df in all_traces.items():
        all_transitions[label] = compute_transitions(baseline_trace, trace_df)

    # Generate report
    logger.info("Generating report...")
    report = generate_report(all_metrics, all_traces, partitions, all_transitions)
    doc_path = REPO_ROOT / "docs" / "cp7_pre6_nms_sweep.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    logger.info("Report written to %s", doc_path)

    # Print summary
    print("\n" + "=" * 70)
    print("NMS SWEEP SUMMARY")
    print("=" * 70)
    for m in all_metrics:
        t = m["total_cells"]
        print(f"  {m['arm']:15s}  present={m['overall']['present']:4d} ({m['overall']['present']/t*100:5.1f}%)  "
              f"misattrib={m['overall']['present_misattributed']:4d} ({m['overall']['present_misattributed']/t*100:5.1f}%)  "
              f"det/f={m['mean_det_per_frame']:5.1f}  trk/gt={m['median_tracklets_per_gt']:4.1f}  "
              f"dup={m['n_unmatched_pred']:3d}  solver={m['solver_status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
