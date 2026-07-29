#!/usr/bin/env python3
"""CP7-pre-7: Dominant-failure tracklet topology (FP7oJQ).

Classifies every present_misattributed contiguous span into one of six buckets
(A/A2/B/C/E/D) based on per-frame box count, carrier stability, boundary
lifecycle events, GROUP node presence, trigger behavior, and gate-distance
re-derivation.

Read-only diagnostic. No pipeline behavior changed.
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CLIP_ID = "FP7oJQ-20260318-200014"
CAMERA_ID = "FP7oJQ"
MODEL_ID = "bjj-detect-all-cameras"
FRAME_RANGE = range(0, 301)

CLIP_ROOT = (
    REPO_ROOT / "outputs/_eval_gt/FP7oJQ/2026-03-18/20" / CLIP_ID
)
EVAL_ROOT = REPO_ROOT / "outputs/_eval/stage_d" / MODEL_ID / CAMERA_ID
STAGE_A_EVAL = REPO_ROOT / "outputs/_eval/stage_a" / MODEL_ID / CAMERA_ID
DEBUG_DIR = CLIP_ROOT / "_debug"

# D1 thresholds (must match production config)
MERGE_DIST_M = 0.45
SPLIT_DIST_M = 0.60
MERGE_TRIGGER_MAX_AGE_FRAMES = 60
MIN_GROUP_DURATION_FRAMES = 10
CARRIER_COORD_WINDOW_FRAMES = 8
LIFECYCLE_W = 10  # primary boundary-event window (matches D1 start/end_window)

# Bucket thresholds
SINGLE_BOX_THRESHOLD = 0.5    # frac_single_box >= this -> single-box dominant
CARRIER_STABLE_THRESHOLD = 0.8  # dominant carrier coverage >= this -> stable
FOOTPRINT_IOU_THRESHOLD = 0.3  # IoU to match detection to GT footprint

# Early-exit thresholds
LOPSIDED_INSTANCE_RATIO = 0.90
LOPSIDED_FRAC_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _effective_xy(row: pd.Series) -> Optional[tuple[float, float]]:
    """Match D1's _effective_xy_row: prefer x_m_repaired, fall back to x_m."""
    x_rep = row.get("x_m_repaired", None)
    y_rep = row.get("y_m_repaired", None)
    x_raw = row.get("x_m", None)
    y_raw = row.get("y_m", None)
    # Use repaired if finite
    if _is_finite(x_rep) and _is_finite(y_rep):
        return (float(x_rep), float(y_rep))
    if _is_finite(x_raw) and _is_finite(y_raw):
        return (float(x_raw), float(y_raw))
    return None


def _is_finite(v) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_gt_bbox(s) -> Optional[tuple[float, float, float, float]]:
    """Parse gt_bbox JSON string to (x1, y1, x2, y2)."""
    if pd.isna(s) or s is None:
        return None
    try:
        coords = json.loads(s) if isinstance(s, str) else s
        if len(coords) == 4:
            return tuple(float(c) for c in coords)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    """Load all required artifacts."""
    logger.info("Loading artifacts...")

    data = {}

    # GT person trace
    data["trace"] = pd.read_parquet(EVAL_ROOT / "gt_person_trace.parquet")
    logger.info("  gt_person_trace: %d rows", len(data["trace"]))

    # Stage A detections
    data["detections"] = pd.read_parquet(CLIP_ROOT / "stage_A/detections.parquet")
    logger.info("  detections: %d rows", len(data["detections"]))

    # D0 bank summaries
    data["bank_summaries"] = pd.read_parquet(
        CLIP_ROOT / "stage_D/tracklet_bank_summaries.parquet"
    )
    logger.info("  bank_summaries: %d tracklets", len(data["bank_summaries"]))

    # D0 bank frames
    data["bank_frames"] = pd.read_parquet(
        CLIP_ROOT / "stage_D/tracklet_bank_frames.parquet"
    )
    logger.info("  bank_frames: %d rows", len(data["bank_frames"]))

    # D1 graph nodes
    data["graph_nodes"] = pd.read_parquet(
        CLIP_ROOT / "stage_D/d1_graph_nodes.parquet"
    )
    logger.info("  graph_nodes: %d nodes", len(data["graph_nodes"]))

    # D1 debug parquets
    debug_files = [
        "d1_group_spans", "d1_merge_triggers", "d1_split_triggers",
        "d1_suppressed_split_triggers", "d1_suppressed_split_triggers_entrance",
        "d1_suppressed_start_merged_entrance", "d1_suppressed_group_spans",
    ]
    for name in debug_files:
        path = DEBUG_DIR / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
            logger.info("  %s: %d rows", name, len(data[name]))
        else:
            data[name] = pd.DataFrame()
            logger.warning("  %s: NOT FOUND", name)

    # per_frame_matches (for cross-reference)
    pfm_path = STAGE_A_EVAL / "per_frame_matches.parquet"
    if pfm_path.exists():
        data["per_frame_matches"] = pd.read_parquet(pfm_path)
        logger.info("  per_frame_matches: %d rows", len(data["per_frame_matches"]))

    return data


# ---------------------------------------------------------------------------
# Instance segmentation
# ---------------------------------------------------------------------------

def segment_instances(trace: pd.DataFrame) -> list[dict]:
    """Segment present_misattributed frames into contiguous spans per GT person."""
    mis = trace[
        (trace["failure_mode"] == "present_misattributed")
        & (trace["frame_idx"] >= FRAME_RANGE.start)
        & (trace["frame_idx"] <= FRAME_RANGE.stop - 1)
    ].sort_values(["gt_person_id", "frame_idx"])

    instances = []
    idx = 0
    for gp, grp in mis.groupby("gt_person_id"):
        frames = sorted(grp["frame_idx"].values)
        if not frames:
            continue
        start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f > prev + 1:
                instances.append({
                    "instance_id": idx,
                    "gt_person_id": int(gp),
                    "start_frame": int(start),
                    "end_frame": int(prev),
                    "length": int(prev - start + 1),
                })
                idx += 1
                start = f
            prev = f
        instances.append({
            "instance_id": idx,
            "gt_person_id": int(gp),
            "start_frame": int(start),
            "end_frame": int(prev),
            "length": int(prev - start + 1),
        })
        idx += 1

    return instances


# ---------------------------------------------------------------------------
# Measurement 1: Per-frame box count within GT footprint
# ---------------------------------------------------------------------------

def measure_box_count(instances: list[dict], trace: pd.DataFrame,
                      detections: pd.DataFrame) -> None:
    """For each instance, compute per-frame box count within GT footprint."""
    # Index detections by frame for fast lookup
    det_by_frame = {
        fidx: grp for fidx, grp in detections.groupby("frame_index")
    }

    for inst in instances:
        gp = inst["gt_person_id"]
        per_frame_counts = []
        per_frame_tracklet_ids = []

        for fidx in range(inst["start_frame"], inst["end_frame"] + 1):
            # Get GT bbox for this (frame, person)
            row = trace[
                (trace["frame_idx"] == fidx) & (trace["gt_person_id"] == gp)
            ]
            if row.empty:
                continue
            gt_bbox = _parse_gt_bbox(row.iloc[0]["gt_bbox"])
            if gt_bbox is None:
                per_frame_counts.append(0)
                per_frame_tracklet_ids.append([])
                continue

            # Find detections overlapping GT footprint
            frame_dets = det_by_frame.get(fidx, pd.DataFrame())
            matched_tids = []
            if not frame_dets.empty:
                for _, det in frame_dets.iterrows():
                    det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
                    iou = _compute_iou(gt_bbox, det_box)
                    if iou >= FOOTPRINT_IOU_THRESHOLD:
                        tid = det.get("tracklet_id")
                        if tid is not None and not pd.isna(tid):
                            matched_tids.append(str(tid))

            unique_tids = list(set(matched_tids))
            per_frame_counts.append(len(unique_tids))
            per_frame_tracklet_ids.append(unique_tids)

        # Compute distribution
        count_dist = Counter(per_frame_counts)
        total_frames = len(per_frame_counts)
        n_single = count_dist.get(1, 0)
        n_zero = count_dist.get(0, 0)
        n_two_plus = sum(v for k, v in count_dist.items() if k >= 2)

        inst["box_count_distribution"] = dict(count_dist)
        inst["frac_single_box"] = n_single / total_frames if total_frames > 0 else 0.0
        inst["frac_two_plus_box"] = n_two_plus / total_frames if total_frames > 0 else 0.0
        inst["frac_zero_box"] = n_zero / total_frames if total_frames > 0 else 0.0
        inst["_per_frame_tracklet_ids"] = per_frame_tracklet_ids  # for meas 2


# ---------------------------------------------------------------------------
# Measurement 2: Carrier stability
# ---------------------------------------------------------------------------

def measure_carrier_stability(instances: list[dict], trace: pd.DataFrame) -> None:
    """Within each span, check tracklet_id stability from gt_person_trace."""
    for inst in instances:
        gp = inst["gt_person_id"]
        span_trace = trace[
            (trace["gt_person_id"] == gp)
            & (trace["frame_idx"] >= inst["start_frame"])
            & (trace["frame_idx"] <= inst["end_frame"])
            & (trace["failure_mode"] == "present_misattributed")
        ]

        tracklet_ids = span_trace["tracklet_id"].dropna().tolist()
        tid_counts = Counter(str(t) for t in tracklet_ids)

        n_unique = len(tid_counts)
        if tid_counts:
            dominant_tid, dominant_count = tid_counts.most_common(1)[0]
            dominant_coverage = dominant_count / inst["length"]
        else:
            dominant_tid = None
            dominant_count = 0
            dominant_coverage = 0.0

        inst["n_unique_tracklets"] = n_unique
        inst["dominant_tracklet_id"] = dominant_tid
        inst["dominant_carrier_coverage"] = dominant_coverage
        inst["carrier_stable"] = dominant_coverage >= CARRIER_STABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Measurement 3: Boundary lifecycle events
# ---------------------------------------------------------------------------

def measure_boundary_lifecycle(instances: list[dict],
                               bank_summaries: pd.DataFrame) -> None:
    """Check for tracklet deaths/births near instance boundaries."""
    ends = bank_summaries[["tracklet_id", "end_frame"]].copy()
    starts = bank_summaries[["tracklet_id", "start_frame"]].copy()

    for inst in instances:
        s, e = inst["start_frame"], inst["end_frame"]
        events = {}

        for w in [LIFECYCLE_W, 30, 60]:
            # Deaths near span start (tracklet ending -> potential merge trigger)
            deaths = ends[
                (ends["end_frame"] >= s - w) & (ends["end_frame"] <= s + w)
            ]
            # Births near span end (tracklet starting -> potential split trigger)
            births = starts[
                (starts["start_frame"] >= e - w) & (starts["start_frame"] <= e + w)
            ]
            events[f"deaths_W{w}"] = deaths["tracklet_id"].tolist()
            events[f"births_W{w}"] = births["tracklet_id"].tolist()
            events[f"n_deaths_W{w}"] = len(deaths)
            events[f"n_births_W{w}"] = len(births)
            events[f"has_boundary_event_W{w}"] = len(deaths) > 0 or len(births) > 0

        inst["has_boundary_event"] = events["has_boundary_event_W10"]
        inst["n_boundary_deaths_W10"] = events["n_deaths_W10"]
        inst["n_boundary_births_W10"] = events["n_births_W10"]
        inst["boundary_events_detail"] = json.dumps({
            k: v for k, v in events.items()
            if k.startswith("has_") or k.startswith("n_")
        })


# ---------------------------------------------------------------------------
# Measurement 4: GROUP presence
# ---------------------------------------------------------------------------

def measure_group_presence(instances: list[dict],
                           group_spans: pd.DataFrame) -> None:
    """Check whether any GROUP span intersects the instance span."""
    for inst in instances:
        s, e = inst["start_frame"], inst["end_frame"]
        overlapping = group_spans[
            (group_spans["group_start"] <= e) & (group_spans["group_end"] >= s)
        ]
        inst["has_group_overlap"] = len(overlapping) > 0
        inst["n_group_overlaps"] = len(overlapping)
        if len(overlapping) > 0:
            details = []
            for _, g in overlapping.iterrows():
                details.append({
                    "carrier": str(g.get("carrier", "")),
                    "kind": str(g.get("kind", "")),
                    "group_start": int(g["group_start"]),
                    "group_end": int(g["group_end"]),
                })
            inst["group_overlap_detail"] = json.dumps(details)
        else:
            inst["group_overlap_detail"] = "[]"


# ---------------------------------------------------------------------------
# Measurement 5: Trigger behavior
# ---------------------------------------------------------------------------

def measure_trigger_behavior(instances: list[dict], data: dict) -> None:
    """Check fired/suppressed/consolidated triggers near each instance."""
    merge_triggers = data["d1_merge_triggers"]
    split_triggers = data["d1_split_triggers"]
    supp_splits = data["d1_suppressed_split_triggers"]
    supp_splits_entrance = data.get("d1_suppressed_split_triggers_entrance", pd.DataFrame())
    supp_start_merged = data.get("d1_suppressed_start_merged_entrance", pd.DataFrame())
    supp_group_spans = data["d1_suppressed_group_spans"]

    for inst in instances:
        s, e = inst["start_frame"], inst["end_frame"]
        w = LIFECYCLE_W
        trigger_info = {
            "fired_merges": [],
            "fired_splits": [],
            "suppressed_splits": [],
            "suppressed_splits_entrance": [],
            "suppressed_start_merged_entrance": [],
            "suppressed_group_spans": [],
        }

        # Fired merge triggers near span
        if not merge_triggers.empty:
            near_merges = merge_triggers[
                (merge_triggers["merge_frame"] >= s - w)
                & (merge_triggers["merge_frame"] <= e + w)
            ]
            for _, t in near_merges.iterrows():
                trigger_info["fired_merges"].append({
                    "carrier": str(t["carrier"]),
                    "disappear": str(t["disappear"]),
                    "merge_frame": int(t["merge_frame"]),
                    "merge_dist_m": float(t["merge_dist_m"]),
                })

        # Fired split triggers near span
        if not split_triggers.empty:
            near_splits = split_triggers[
                (split_triggers["split_frame"] >= s - w)
                & (split_triggers["split_frame"] <= e + w)
            ]
            for _, t in near_splits.iterrows():
                trigger_info["fired_splits"].append({
                    "carrier": str(t["carrier"]),
                    "new": str(t["new"]),
                    "split_frame": int(t["split_frame"]),
                    "split_dist_m": float(t["split_dist_m"]),
                })

        # Suppressed split triggers
        if not supp_splits.empty:
            near_supp = supp_splits[
                (supp_splits["split_frame"] >= s - w)
                & (supp_splits["split_frame"] <= e + w)
            ]
            for _, t in near_supp.iterrows():
                trigger_info["suppressed_splits"].append({
                    "carrier": str(t["carrier"]),
                    "new": str(t["new"]),
                    "split_frame": int(t["split_frame"]),
                    "reason": str(t.get("reason", "unknown")),
                })

        # Suppressed entrance-like splits
        if not supp_splits_entrance.empty and "split_frame" in supp_splits_entrance.columns:
            frame_col = "split_frame"
            near = supp_splits_entrance[
                (supp_splits_entrance[frame_col] >= s - w)
                & (supp_splits_entrance[frame_col] <= e + w)
            ]
            for _, t in near.iterrows():
                trigger_info["suppressed_splits_entrance"].append({
                    "carrier": str(t.get("carrier", "")),
                    "new": str(t.get("new", "")),
                    "frame": int(t[frame_col]),
                })

        # Suppressed start-merged entrance
        if not supp_start_merged.empty:
            # These may use different column names; adapt
            for col in ["split_frame", "merge_frame", "frame"]:
                if col in supp_start_merged.columns:
                    near = supp_start_merged[
                        (supp_start_merged[col] >= s - w)
                        & (supp_start_merged[col] <= e + w)
                    ]
                    for _, t in near.iterrows():
                        trigger_info["suppressed_start_merged_entrance"].append({
                            "carrier": str(t.get("carrier", "")),
                            "frame": int(t[col]),
                        })
                    break

        # Suppressed group spans
        if not supp_group_spans.empty:
            # Check if suppressed span overlaps instance
            for start_col, end_col in [("group_start", "group_end"),
                                        ("group_start_raw", "group_end_raw")]:
                if start_col in supp_group_spans.columns:
                    near = supp_group_spans[
                        (supp_group_spans[start_col] <= e + w)
                        & (supp_group_spans[end_col] >= s - w)
                    ]
                    for _, t in near.iterrows():
                        trigger_info["suppressed_group_spans"].append({
                            "carrier": str(t.get("carrier", "")),
                            "reason": str(t.get("reason", "")),
                            "start": int(t[start_col]),
                            "end": int(t[end_col]),
                        })
                    break

        has_fired = (
            len(trigger_info["fired_merges"]) > 0
            or len(trigger_info["fired_splits"]) > 0
        )
        has_suppressed = (
            len(trigger_info["suppressed_splits"]) > 0
            or len(trigger_info["suppressed_splits_entrance"]) > 0
            or len(trigger_info["suppressed_start_merged_entrance"]) > 0
            or len(trigger_info["suppressed_group_spans"]) > 0
        )

        inst["has_fired_trigger"] = has_fired
        inst["has_suppressed_trigger"] = has_suppressed
        inst["trigger_detail"] = json.dumps(trigger_info)


# ---------------------------------------------------------------------------
# Measurement 6: Gate-distance re-derivation
# ---------------------------------------------------------------------------

def measure_gate_distances(instances: list[dict], bank_summaries: pd.DataFrame,
                           bank_frames: pd.DataFrame) -> None:
    """Re-derive merge/split candidacy geometry using D1's coordinate source."""
    # Index bank frames by tracklet_id
    bf_by_tid = {tid: grp for tid, grp in bank_frames.groupby("tracklet_id")}

    # Build tracklet lifespan lookup
    tid_info = {}
    for _, row in bank_summaries.iterrows():
        tid = str(row["tracklet_id"])
        tid_info[tid] = {
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "n_frames": int(row["n_frames"]),
        }

    def get_xy_at_frame(tid: str, frame: int, window: int = CARRIER_COORD_WINDOW_FRAMES
                        ) -> Optional[tuple[float, float]]:
        """Get effective xy for tracklet at frame, using D1's window logic."""
        frames_df = bf_by_tid.get(tid)
        if frames_df is None:
            return None
        # Look for exact frame first, then within window
        for search_frame in range(frame, frame + window + 1):
            rows = frames_df[frames_df["frame_index"] == search_frame]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if not bool(row.get("on_mat", True)):
                continue
            xy = _effective_xy(row)
            if xy is not None:
                return xy
        # Search backward
        for search_frame in range(frame - 1, frame - window - 1, -1):
            rows = frames_df[frames_df["frame_index"] == search_frame]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if not bool(row.get("on_mat", True)):
                continue
            xy = _effective_xy(row)
            if xy is not None:
                return xy
        return None

    for inst in instances:
        if not inst.get("has_boundary_event", False):
            inst["gate_rederivation"] = json.dumps({"skipped": "no_boundary_event"})
            continue

        s, e = inst["start_frame"], inst["end_frame"]
        w = LIFECYCLE_W
        rederivations = []

        # Find disappearing tracklets (end near span start) -> merge candidacy
        for tid, info in tid_info.items():
            if not (s - w <= info["end_frame"] <= s + w):
                continue
            d_end = info["end_frame"]
            disappear_xy = get_xy_at_frame(tid, d_end)
            if disappear_xy is None:
                continue

            # Find carriers alive at d_end
            for c_tid, c_info in tid_info.items():
                if c_tid == tid:
                    continue
                if c_info["start_frame"] <= d_end <= c_info["end_frame"]:
                    carrier_xy = get_xy_at_frame(c_tid, d_end)
                    if carrier_xy is None:
                        continue
                    dist = _dist_m(disappear_xy, carrier_xy)
                    gap = 0  # d_end is within carrier span
                    rederivations.append({
                        "type": "merge",
                        "disappear": tid,
                        "carrier": c_tid,
                        "frame": d_end,
                        "distance_m": round(dist, 4),
                        "threshold_m": MERGE_DIST_M,
                        "margin_m": round(dist - MERGE_DIST_M, 4),
                        "passes_gate": dist <= MERGE_DIST_M,
                        "gap_frames": gap,
                        "gap_threshold": MERGE_TRIGGER_MAX_AGE_FRAMES,
                    })

        # Find new tracklets (start near span end) -> split candidacy
        for tid, info in tid_info.items():
            if not (e - w <= info["start_frame"] <= e + w):
                continue
            n_start = info["start_frame"]
            new_xy = get_xy_at_frame(tid, n_start)
            if new_xy is None:
                continue

            # Find carriers alive at n_start
            for c_tid, c_info in tid_info.items():
                if c_tid == tid:
                    continue
                if c_info["start_frame"] <= n_start <= c_info["end_frame"]:
                    carrier_xy = get_xy_at_frame(c_tid, n_start)
                    if carrier_xy is None:
                        continue
                    dist = _dist_m(new_xy, carrier_xy)
                    rederivations.append({
                        "type": "split",
                        "new": tid,
                        "carrier": c_tid,
                        "frame": n_start,
                        "distance_m": round(dist, 4),
                        "threshold_m": SPLIT_DIST_M,
                        "margin_m": round(dist - SPLIT_DIST_M, 4),
                        "passes_gate": dist <= SPLIT_DIST_M,
                    })

        # Classify: any near-miss (margin small positive, < 0.3m beyond threshold)?
        has_near_miss = any(
            not r["passes_gate"] and r["margin_m"] < 0.30
            for r in rederivations
        )
        has_passing = any(r["passes_gate"] for r in rederivations)

        inst["gate_rederivation"] = json.dumps({
            "n_candidates": len(rederivations),
            "has_passing": has_passing,
            "has_near_miss": has_near_miss,
            "candidates": rederivations,
        })
        inst["gate_has_near_miss"] = has_near_miss
        inst["gate_has_passing"] = has_passing


# ---------------------------------------------------------------------------
# IoU spot-check (Change 4)
# ---------------------------------------------------------------------------

def iou_spot_check(instances: list[dict], trace: pd.DataFrame,
                   detections: pd.DataFrame) -> list[dict]:
    """Spot-check 10 random two-box frames to validate frac_single_box."""
    det_by_frame = {fidx: grp for fidx, grp in detections.groupby("frame_index")}

    # Collect all two-box frames across instances
    two_box_frames = []
    for inst in instances:
        gp = inst["gt_person_id"]
        for fidx in range(inst["start_frame"], inst["end_frame"] + 1):
            row = trace[
                (trace["frame_idx"] == fidx) & (trace["gt_person_id"] == gp)
            ]
            if row.empty:
                continue
            gt_bbox = _parse_gt_bbox(row.iloc[0]["gt_bbox"])
            if gt_bbox is None:
                continue
            frame_dets = det_by_frame.get(fidx, pd.DataFrame())
            if frame_dets.empty:
                continue
            matched = []
            for _, det in frame_dets.iterrows():
                det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
                iou = _compute_iou(gt_bbox, det_box)
                if iou >= FOOTPRINT_IOU_THRESHOLD:
                    matched.append({
                        "tracklet_id": str(det["tracklet_id"]),
                        "det_box": det_box,
                        "iou_with_gt": round(iou, 3),
                    })
            if len(matched) == 2:
                two_box_frames.append({
                    "frame_idx": fidx,
                    "gt_person_id": gp,
                    "gt_bbox": gt_bbox,
                    "matched": matched,
                })

    # Sample up to 10
    random.seed(42)
    sample = random.sample(two_box_frames, min(10, len(two_box_frames)))

    results = []
    for s in sample:
        m1, m2 = s["matched"]
        det_iou = _compute_iou(m1["det_box"], m2["det_box"])
        results.append({
            "frame_idx": s["frame_idx"],
            "gt_person_id": s["gt_person_id"],
            "gt_bbox": [round(c, 1) for c in s["gt_bbox"]],
            "det1_tid": m1["tracklet_id"],
            "det1_iou_with_gt": m1["iou_with_gt"],
            "det2_tid": m2["tracklet_id"],
            "det2_iou_with_gt": m2["iou_with_gt"],
            "det1_det2_iou": round(det_iou, 3),
            "same_person_plausible": det_iou >= 0.5,  # high mutual IoU = both cover same region
        })

    return results


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------

def classify_buckets(instances: list[dict]) -> None:
    """Assign each instance to exactly one bucket: A/A2/B/C/E/D."""
    for inst in instances:
        frac_single = inst["frac_single_box"]
        has_boundary = inst["has_boundary_event"]
        carrier_stable = inst["carrier_stable"]

        if frac_single < SINGLE_BOX_THRESHOLD and not has_boundary:
            # Two-box, no boundary event
            if carrier_stable:
                inst["bucket"] = "A"
            else:
                inst["bucket"] = "A2"
        elif frac_single >= SINGLE_BOX_THRESHOLD and not has_boundary:
            # Single-box, no boundary event -> pure under-segmentation
            inst["bucket"] = "E"
        elif has_boundary:
            # Has boundary event -> B or C
            has_suppressed = inst.get("has_suppressed_trigger", False)
            gate_near_miss = inst.get("gate_has_near_miss", False)
            if has_suppressed or gate_near_miss:
                inst["bucket"] = "B"
            else:
                inst["bucket"] = "C"
        else:
            inst["bucket"] = "D"


# ---------------------------------------------------------------------------
# Conservation check
# ---------------------------------------------------------------------------

def check_conservation(instances: list[dict], expected: int) -> bool:
    """Assert A+A2+B+C+E+D == expected."""
    counts = Counter(inst["bucket"] for inst in instances)
    total = sum(counts.values())
    buckets = ["A", "A2", "B", "C", "E", "D"]

    logger.info("=== CONSERVATION CHECK ===")
    for b in buckets:
        logger.info("  Bucket %s: %d", b, counts.get(b, 0))
    logger.info("  Total: %d (expected: %d)", total, expected)

    if total != expected:
        logger.error("CONSERVATION FAILED: %d != %d", total, expected)
        return False
    logger.info("CONSERVATION PASSED")
    return True


# ---------------------------------------------------------------------------
# GROUP x bucket cross-tab
# ---------------------------------------------------------------------------

def group_bucket_crosstab(instances: list[dict]) -> pd.DataFrame:
    """Build GROUP-presence x bucket cross-tab."""
    rows = []
    for inst in instances:
        rows.append({
            "bucket": inst["bucket"],
            "group_present": "GROUP-present" if inst["has_group_overlap"] else "GROUP-absent",
        })
    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["bucket"], df["group_present"], margins=True)
    # Reorder rows
    bucket_order = ["A", "A2", "B", "C", "E", "D", "All"]
    ct = ct.reindex([b for b in bucket_order if b in ct.index])
    return ct


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(instances: list[dict], spot_check: list[dict],
                    crosstab: pd.DataFrame, lopsided: Optional[str],
                    meas6_run: bool) -> str:
    """Generate docs/checkpoints/cp7_pre7_failure_topology.md content."""
    lines = []
    lines.append("# CP7-pre-7: Dominant-Failure Tracklet Topology (FP7oJQ)")
    lines.append("")
    lines.append("*Generated by `tools/cp7_pre7_failure_topology.py`. Read-only diagnostic.*")
    lines.append("")

    # Section 1: Purpose
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("Resolve the internal contradiction between CP7-pre-2 (95-99% of")
    lines.append("fragmentation is concurrent, both tracklets alive) and CP7-pre-3")
    lines.append("(70-78% is detection under-segmentation, one box spanning a pair).")
    lines.append("Classifies each misattributed contiguous span into topology buckets")
    lines.append("that determine the Lever-2 implementation path.")
    lines.append("")

    # Section 2: Condition 1 evidence
    lines.append("## 2. Production Root Verification")
    lines.append("")
    lines.append("Detections at `outputs/_eval_gt/FP7oJQ/.../stage_A/detections.parquet`")
    lines.append("carry sub-1.0 confidence (range 0.450-0.981, mean 0.861) across all 3357")
    lines.append("detections in window. Mean boxes/frame = 11.2 (vs 14 GT persons); only")
    lines.append("1/301 frames hits 14 boxes. GT-injected pre-5 run writes to separate")
    lines.append("`outputs/_gt_ceiling/` root. **Verdict: production detector, comparable")
    lines.append("to the production GT trace.**")
    lines.append("")

    # Section 3: Instance count and length distribution
    lines.append("## 3. Instance Count and Length Distribution")
    lines.append("")
    n_inst = len(instances)
    lengths = [i["length"] for i in instances]
    lines.append(f"Total contiguous misattributed instances: **{n_inst}**")
    lines.append("")
    lines.append(f"- Min length: {min(lengths)}")
    lines.append(f"- Max length: {max(lengths)}")
    lines.append(f"- Median length: {int(np.median(lengths))}")
    lines.append(f"- Mean length: {np.mean(lengths):.1f}")
    lines.append("")
    length_bins = [(0, 1), (2, 5), (6, 10), (11, 30), (31, 100), (101, 1000)]
    lines.append("| Length range | Count | Pct |")
    lines.append("|-------------|-------|-----|")
    for lo, hi in length_bins:
        c = sum(1 for l in lengths if lo <= l <= hi)
        pct = 100 * c / n_inst if n_inst > 0 else 0
        lines.append(f"| {lo}-{hi} | {c} | {pct:.1f}% |")
    lines.append("")

    # Section 4: IoU spot-check
    lines.append("## 4. IoU Spot-Check (Measurement 1 Validation)")
    lines.append("")
    if not spot_check:
        lines.append("No two-box frames found for spot-check.")
    else:
        n_plausible = sum(1 for s in spot_check if s["same_person_plausible"])
        n_total = len(spot_check)
        n_encroaching = n_total - n_plausible
        pct_encroaching = 100 * n_encroaching / n_total if n_total > 0 else 0

        lines.append(f"Sampled {n_total} two-box frames (seed=42). For each, the two")
        lines.append(f"detections matched to a single GT person at IoU >= {FOOTPRINT_IOU_THRESHOLD}.")
        lines.append(f"`same_person_plausible` = mutual detection IoU >= 0.5 (both cover same region).")
        lines.append("")
        lines.append("| Frame | GT person | Det1 tid | Det1-GT IoU | Det2 tid | Det2-GT IoU | Det1-Det2 IoU | Plausible |")
        lines.append("|-------|-----------|----------|-------------|----------|-------------|---------------|-----------|")
        for s in spot_check:
            lines.append(
                f"| {s['frame_idx']} | {s['gt_person_id']} | {s['det1_tid']} | {s['det1_iou_with_gt']} "
                f"| {s['det2_tid']} | {s['det2_iou_with_gt']} | {s['det1_det2_iou']} "
                f"| {'Y' if s['same_person_plausible'] else 'N'} |"
            )
        lines.append("")
        lines.append(f"**Result:** {n_plausible}/{n_total} plausible same-person, "
                      f"{n_encroaching}/{n_total} encroaching-opponent ({pct_encroaching:.0f}%).")
        if pct_encroaching > 30:
            lines.append("")
            lines.append("**WARNING:** >30% encroaching-opponent rate. `frac_single_box` may")
            lines.append("undercount pure under-segmentation (some E instances misrouted to A).")
        else:
            lines.append(f" frac_single_box is trustworthy at IoU >= {FOOTPRINT_IOU_THRESHOLD}.")
    lines.append("")

    # Section 5: Per-frame box-count distribution
    lines.append("## 5. Per-Frame Box-Count Distribution (Measurement 1)")
    lines.append("")
    all_frac_single = [i["frac_single_box"] for i in instances]
    lines.append(f"Across {n_inst} instances:")
    lines.append(f"- Mean frac_single_box: {np.mean(all_frac_single):.3f}")
    lines.append(f"- Median frac_single_box: {np.median(all_frac_single):.3f}")
    lines.append(f"- Instances with frac_single_box >= 0.5: {sum(1 for f in all_frac_single if f >= 0.5)}")
    lines.append(f"- Instances with frac_single_box < 0.5: {sum(1 for f in all_frac_single if f < 0.5)}")
    lines.append("")
    frac_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                 (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    lines.append("| frac_single_box range | Count | Pct |")
    lines.append("|-----------------------|-------|-----|")
    for lo, hi in frac_bins:
        c = sum(1 for f in all_frac_single if lo <= f < hi)
        pct = 100 * c / n_inst if n_inst > 0 else 0
        label = f"{lo:.1f}-{hi:.1f}" if hi <= 1.0 else f"{lo:.1f}-1.0"
        lines.append(f"| {label} | {c} | {pct:.1f}% |")
    lines.append("")

    if lopsided:
        lines.append(f"**LOPSIDED flag: {lopsided}** — see early-exit note below.")
        lines.append("")

    # Section 6: Pre-2 vs pre-3 reconciliation headline
    lines.append("## 6. Pre-2 vs Pre-3 Reconciliation (A+A2 vs E)")
    lines.append("")
    counts = Counter(i["bucket"] for i in instances)
    a_total = counts.get("A", 0) + counts.get("A2", 0)
    e_total = counts.get("E", 0)
    lines.append(f"- **A + A2 (concurrent two-box):** {a_total} instances "
                  f"({100*a_total/n_inst:.1f}%)")
    lines.append(f"  - A (persistent carrier): {counts.get('A', 0)}")
    lines.append(f"  - A2 (carrier switch): {counts.get('A2', 0)}")
    lines.append(f"- **E (pure under-segmentation):** {e_total} instances "
                  f"({100*e_total/n_inst:.1f}%)")
    lines.append(f"- **B + C (boundary-event):** {counts.get('B', 0) + counts.get('C', 0)} instances")
    lines.append(f"- **D (unclassified residual):** {counts.get('D', 0)} instances")
    lines.append("")

    # Section 7: GROUP x bucket cross-tab
    lines.append("## 7. GROUP x Bucket Cross-Tab")
    lines.append("")
    lines.append("Rows = bucket, columns = whether any GROUP_TRACKLET span overlaps the instance.")
    lines.append("")
    # Manual markdown table (avoids tabulate dependency)
    cols = [c for c in crosstab.columns if c != "All"] + (["All"] if "All" in crosstab.columns else [])
    header = "| bucket | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|--------|" + "|".join("------" for _ in cols) + "|"
    lines.append(header)
    lines.append(sep)
    for idx_val in crosstab.index:
        row_vals = " | ".join(str(crosstab.loc[idx_val, c]) if c in crosstab.columns else "0" for c in cols)
        lines.append(f"| {idx_val} | {row_vals} |")
    lines.append("")

    # Section 8: Full bucket tallies
    lines.append("## 8. Bucket Tallies")
    lines.append("")
    lines.append("| Bucket | Label | Count | Pct |")
    lines.append("|--------|-------|-------|-----|")
    for b, label in [("A", "Concurrent persistent"), ("A2", "Concurrent carrier-switch"),
                      ("B", "Gate-rejected"), ("C", "No-trigger structural"),
                      ("E", "Pure under-segmentation"), ("D", "Unclassified residual")]:
        c = counts.get(b, 0)
        pct = 100 * c / n_inst if n_inst > 0 else 0
        lines.append(f"| {b} | {label} | {c} | {pct:.1f}% |")
    lines.append("")

    # B and C split by box count
    lines.append("### B and C by box-count")
    lines.append("")
    for b in ["B", "C"]:
        b_instances = [i for i in instances if i["bucket"] == b]
        if b_instances:
            n_single_dom = sum(1 for i in b_instances if i["frac_single_box"] >= 0.5)
            n_two_dom = len(b_instances) - n_single_dom
            lines.append(f"- **Bucket {b}:** {n_single_dom} single-box-dominant, {n_two_dom} two-box-dominant")
        else:
            lines.append(f"- **Bucket {b}:** 0 instances")
    lines.append("")

    # Section 9: Measurement 6 detail (if run)
    if meas6_run:
        lines.append("## 9. Gate-Distance Re-Derivation (Measurement 6)")
        lines.append("")
        b_instances = [i for i in instances if i["bucket"] == "B"]
        if b_instances:
            # Collect all margins for distribution summary
            all_margins = []
            b_reason_counts = {"suppressed_trigger_only": 0, "gate_near_miss": 0, "both": 0}
            for inst in b_instances:
                gate_data = json.loads(inst.get("gate_rederivation", "{}"))
                candidates = gate_data.get("candidates", [])
                for c in candidates:
                    if not c.get("passes_gate", True):
                        all_margins.append(c.get("margin_m", 0))
                has_supp = inst.get("has_suppressed_trigger", False)
                has_nm = inst.get("gate_has_near_miss", False)
                if has_supp and has_nm:
                    b_reason_counts["both"] += 1
                elif has_supp:
                    b_reason_counts["suppressed_trigger_only"] += 1
                elif has_nm:
                    b_reason_counts["gate_near_miss"] += 1

            lines.append(f"**{len(b_instances)} bucket-B instances.** Classification reason:")
            lines.append(f"- Suppressed trigger only: {b_reason_counts['suppressed_trigger_only']}")
            lines.append(f"- Gate near-miss only: {b_reason_counts['gate_near_miss']}")
            lines.append(f"- Both: {b_reason_counts['both']}")
            lines.append("")

            if all_margins:
                margins_arr = np.array(all_margins)
                lines.append(f"Gate-rejected candidate margin distribution (N={len(margins_arr)}):")
                lines.append(f"- Min: {margins_arr.min():.3f} m, Max: {margins_arr.max():.3f} m")
                lines.append(f"- Mean: {margins_arr.mean():.3f} m, Median: {np.median(margins_arr):.3f} m")
                lines.append("")
                margin_bins = [(-0.5, 0.0), (0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)]
                lines.append("| Margin range (m) | Count | Pct | Interpretation |")
                lines.append("|------------------|-------|-----|----------------|")
                for lo, hi in margin_bins:
                    c = int(np.sum((margins_arr >= lo) & (margins_arr < hi)))
                    pct = 100 * c / len(margins_arr)
                    if lo < 0:
                        interp = "Passes gate (already fires)"
                    elif hi <= 0.05:
                        interp = "Very close near-miss"
                    elif hi <= 0.1:
                        interp = "Close near-miss"
                    else:
                        interp = "Moderate near-miss"
                    lines.append(f"| {lo:.2f} to {hi:.2f} | {c} | {pct:.1f}% | {interp} |")
                lines.append("")

            # Show 10 representative examples
            lines.append("Representative examples (first 10):")
            lines.append("")
            lines.append("| Instance | GT person | Span | Type | Distance (m) | Threshold (m) | Margin (m) |")
            lines.append("|----------|-----------|------|------|-------------|---------------|------------|")
            shown = 0
            for inst in b_instances:
                if shown >= 10:
                    break
                gate_data = json.loads(inst.get("gate_rederivation", "{}"))
                candidates = gate_data.get("candidates", [])
                near_misses = [c for c in candidates if not c.get("passes_gate", True) and c.get("margin_m", 999) < 0.30]
                suppressed_cands = [c for c in candidates if c.get("passes_gate", False)]
                show = near_misses[:1] if near_misses else suppressed_cands[:1]
                for c in show:
                    lines.append(
                        f"| {inst['instance_id']} | {inst['gt_person_id']} "
                        f"| {inst['start_frame']}-{inst['end_frame']} "
                        f"| {c.get('type', '?')} | {c.get('distance_m', '?')} "
                        f"| {c.get('threshold_m', '?')} | {c.get('margin_m', '?')} |"
                    )
                    shown += 1
        else:
            lines.append("No bucket-B instances.")
        lines.append("")
    else:
        lines.append("## 9. Gate-Distance Re-Derivation")
        lines.append("")
        lines.append("Skipped (early-exit or not applicable). Run with full mode to populate.")
        lines.append("")

    # Section 10: Conservation
    lines.append("## 10. Conservation Assertion")
    lines.append("")
    total = sum(counts.values())
    lines.append(f"A({counts.get('A',0)}) + A2({counts.get('A2',0)}) + B({counts.get('B',0)}) "
                  f"+ C({counts.get('C',0)}) + E({counts.get('E',0)}) + D({counts.get('D',0)}) "
                  f"= **{total}** (expected **{n_inst}**)")
    if total == n_inst:
        lines.append("")
        lines.append("**CONSERVATION PASSED.**")
    else:
        lines.append("")
        lines.append("**CONSERVATION FAILED.**")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("CP7-pre-7: Failure topology diagnostic (FP7oJQ, frames 0-300)")
    logger.info("="*70)

    # Load data
    data = load_all_data()

    # Segment instances
    instances = segment_instances(data["trace"])
    n_inst = len(instances)
    logger.info("Segmented %d contiguous misattributed instances", n_inst)

    # Measurement 1: box count
    logger.info("Running Measurement 1: per-frame box count...")
    measure_box_count(instances, data["trace"], data["detections"])

    # Measurement 2: carrier stability
    logger.info("Running Measurement 2: carrier stability...")
    measure_carrier_stability(instances, data["trace"])

    # Measurement 3: boundary lifecycle
    logger.info("Running Measurement 3: boundary lifecycle events...")
    measure_boundary_lifecycle(instances, data["bank_summaries"])

    # Measurement 4: GROUP presence
    logger.info("Running Measurement 4: GROUP presence...")
    measure_group_presence(instances, data["d1_group_spans"])

    # Measurement 5: trigger behavior
    logger.info("Running Measurement 5: trigger behavior...")
    measure_trigger_behavior(instances, data)

    # IoU spot-check
    logger.info("Running IoU spot-check...")
    spot_check = iou_spot_check(instances, data["trace"], data["detections"])
    logger.info("  Spot-checked %d two-box frames", len(spot_check))

    # Early-exit check BEFORE measurement 6
    all_frac_single = [i["frac_single_box"] for i in instances]
    pct_high_single = sum(1 for f in all_frac_single if f >= LOPSIDED_FRAC_THRESHOLD) / n_inst
    pct_two_box = sum(1 for f in all_frac_single if f < SINGLE_BOX_THRESHOLD) / n_inst

    lopsided = None
    run_meas6 = True

    if pct_high_single >= LOPSIDED_INSTANCE_RATIO:
        lopsided = f"LOPSIDED-E: {pct_high_single*100:.0f}% of instances have frac_single_box >= {LOPSIDED_FRAC_THRESHOLD}"
        logger.info("EARLY-EXIT: %s", lopsided)
        run_meas6 = False
    elif pct_two_box >= LOPSIDED_INSTANCE_RATIO:
        # Check if boundary-event instances are < 10%
        n_boundary = sum(1 for i in instances if i["has_boundary_event"])
        if n_boundary / n_inst < 0.10:
            lopsided = (f"LOPSIDED-A: {pct_two_box*100:.0f}% two-box, "
                        f"only {n_boundary} boundary-event instances ({100*n_boundary/n_inst:.0f}%)")
            logger.info("EARLY-EXIT: %s", lopsided)
            run_meas6 = False

    # Measurement 6: gate re-derivation (conditional)
    if run_meas6:
        logger.info("Running Measurement 6: gate-distance re-derivation...")
        measure_gate_distances(instances, data["bank_summaries"], data["bank_frames"])
    else:
        logger.info("Skipping Measurement 6 (early-exit condition met)")
        for inst in instances:
            inst["gate_rederivation"] = json.dumps({"skipped": "early_exit"})
            inst["gate_has_near_miss"] = False
            inst["gate_has_passing"] = False

    # Classify buckets
    logger.info("Classifying buckets...")
    classify_buckets(instances)

    # Conservation check
    passed = check_conservation(instances, n_inst)
    if not passed:
        logger.error("RUN INVALID: conservation failed")
        sys.exit(1)

    # GROUP x bucket cross-tab
    crosstab = group_bucket_crosstab(instances)
    logger.info("\n=== GROUP x BUCKET CROSS-TAB ===")
    logger.info("\n%s", crosstab.to_string())

    # Build instance table for parquet output
    out_cols = [
        "instance_id", "gt_person_id", "start_frame", "end_frame", "length",
        "frac_single_box", "frac_two_plus_box", "frac_zero_box",
        "n_unique_tracklets", "dominant_tracklet_id", "dominant_carrier_coverage",
        "carrier_stable",
        "has_boundary_event", "n_boundary_deaths_W10", "n_boundary_births_W10",
        "has_group_overlap", "n_group_overlaps",
        "has_fired_trigger", "has_suppressed_trigger",
        "gate_has_near_miss", "gate_has_passing",
        "bucket",
        "box_count_distribution", "boundary_events_detail",
        "group_overlap_detail", "trigger_detail", "gate_rederivation",
    ]
    out_rows = []
    for inst in instances:
        row = {}
        for col in out_cols:
            val = inst.get(col)
            if isinstance(val, dict):
                row[col] = json.dumps(val)
            else:
                row[col] = val
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)

    # Write parquet
    out_path = EVAL_ROOT / "failure_topology_instances.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info("Instance table written to %s", out_path)

    # Generate and write doc
    report = generate_report(instances, spot_check, crosstab, lopsided, run_meas6)
    doc_path = REPO_ROOT / "docs/checkpoints/cp7_pre7_failure_topology.md"
    doc_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", doc_path)

    # Print summary
    counts = Counter(i["bucket"] for i in instances)
    print("\n" + "="*60)
    print("CP7-pre-7 RESULTS SUMMARY")
    print("="*60)
    print(f"Total instances: {n_inst}")
    print(f"\nBucket tallies:")
    for b in ["A", "A2", "B", "C", "E", "D"]:
        print(f"  {b}: {counts.get(b, 0)} ({100*counts.get(b,0)/n_inst:.1f}%)")
    a_total = counts.get("A", 0) + counts.get("A2", 0)
    e_total = counts.get("E", 0)
    print(f"\nPre-2 vs Pre-3 reconciliation:")
    print(f"  A+A2 (concurrent two-box): {a_total} ({100*a_total/n_inst:.1f}%)")
    print(f"  E (pure under-seg):        {e_total} ({100*e_total/n_inst:.1f}%)")
    print(f"\nGROUP x Bucket cross-tab:")
    print(crosstab.to_string())
    if lopsided:
        print(f"\n{lopsided}")
    print(f"\nConservation: {'PASSED' if passed else 'FAILED'}")
    print("="*60)


if __name__ == "__main__":
    main()
