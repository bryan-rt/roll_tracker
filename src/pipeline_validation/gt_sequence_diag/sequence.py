"""Build contiguous-segment table from gt_person_trace.parquet.

Consumes the existing CP6 trace (Hungarian IoU 0.5, CP-EVAL-1 frozen instrument)
rather than recomputing any joins. Adds:
  - Contiguous-segment decomposition (RLE of tracklet/node/person)
  - Per-tracklet purity
  - Group-span GT-box IoU (false-group check)
  - Coverage floor flagging
  - Median GT box area per track
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Mat blueprint bounds (x=[42,58], y=[34,58]) — the actual mat.
# Calibrated quad (x=[51.01,57.00], y=[33.96,56.02]) — the homography fit region.
# Positions outside the quad are homography extrapolations — less reliable but not off-mat.
MAT_BLUEPRINT_X = (42.0, 58.0)
MAT_BLUEPRINT_Y = (34.0, 58.0)
CALIBRATED_QUAD_X = (51.01, 57.00)
CALIBRATED_QUAD_Y = (33.96, 56.02)
COVERAGE_FLOOR_PCT = 50.0


def _parse_node_id(val) -> str | None:
    """Extract single node_id from the JSON-encoded list in gt_person_trace."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        lst = json.loads(val)
        if lst and len(lst) > 0:
            return lst[0]
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return None


def _parse_node_type(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        lst = json.loads(val)
        if lst and len(lst) > 0:
            return lst[0]
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return None


def _compute_tracklet_purity(trace: pd.DataFrame) -> dict[tuple[str, int], float]:
    """For each (tracklet_id, gt_person_id), compute purity = count / total tracklet matches."""
    has_tracklet = trace["tracklet_id"].notna()
    sub = trace[has_tracklet][["tracklet_id", "gt_person_id"]].copy()
    if sub.empty:
        return {}

    per_gt = sub.groupby(["tracklet_id", "gt_person_id"]).size().reset_index(name="count")
    totals = sub.groupby("tracklet_id").size().reset_index(name="total")
    merged = per_gt.merge(totals, on="tracklet_id")
    merged["purity"] = merged["count"] / merged["total"]

    result = {}
    for _, r in merged.iterrows():
        result[(r["tracklet_id"], int(r["gt_person_id"]))] = {
            "purity": float(r["purity"]),
            "total_matched": int(r["total"]),
        }
    return result


def _compute_group_gt_box_iou(
    trace: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> dict[str, float]:
    """For each GROUP node, compute mean pairwise GT-box IoU across its frames.

    A high IoU means the GT people genuinely overlap (valid group).
    A low IoU means well-separated people were grouped (false group).
    """
    group_nodes = nodes_df[nodes_df["segment_type"] == "GROUP"]
    if group_nodes.empty:
        return {}

    # Build frame -> gt_bbox map from trace (only detected rows)
    trace_detected = trace[trace["tracklet_id"].notna()].copy()
    trace_detected["_node_id"] = trace_detected["d1_node_ids"].apply(_parse_node_id)

    result = {}
    for _, node in group_nodes.iterrows():
        node_id = node["node_id"]
        start = int(node["start_frame"])
        end = int(node["end_frame"])

        # Find trace rows matching this node
        in_node = trace_detected[
            (trace_detected["_node_id"] == node_id)
            & (trace_detected["frame_idx"] >= start)
            & (trace_detected["frame_idx"] <= end)
        ]

        if in_node.empty:
            continue

        gt_ids_in_node = in_node["gt_person_id"].unique()
        if len(gt_ids_in_node) < 2:
            # Only one GT person matched to this group — no pairwise IoU
            result[node_id] = None
            continue

        # Compute pairwise GT-box IoU per frame
        ious = []
        frames = in_node["frame_idx"].unique()
        for frame in frames:
            frame_rows = in_node[in_node["frame_idx"] == frame]
            if len(frame_rows) < 2:
                continue
            # Get GT bboxes for different GT people
            bboxes_by_gt = {}
            for _, r in frame_rows.iterrows():
                bbox = r["gt_bbox"]
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                if bbox and len(bbox) == 4:
                    bboxes_by_gt[int(r["gt_person_id"])] = bbox

            gt_keys = list(bboxes_by_gt.keys())
            for i in range(len(gt_keys)):
                for j in range(i + 1, len(gt_keys)):
                    b1 = bboxes_by_gt[gt_keys[i]]
                    b2 = bboxes_by_gt[gt_keys[j]]
                    iou = _box_iou(b1, b2)
                    ious.append(iou)

        result[node_id] = float(np.mean(ious)) if ious else None

    return result


def _box_iou(b1: list[float], b2: list[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _compute_gt_presence(trace: pd.DataFrame) -> dict[int, int]:
    """Total frames each GT track is present in GT (not just matched)."""
    return trace.groupby("gt_person_id").size().to_dict()


def _compute_on_mat_status(
    pfm_path: Path,
    projection,
) -> dict[int, dict]:
    """Compute on-mat classification for each GT track using production projection.

    Returns per GT track: on_mat_blueprint, in_quad_pct.
    Uses contact_point_from_bbox + project_to_world — same path as Stage A.
    """
    from bjj_pipeline.stages.detect_track.quality import contact_point_from_bbox
    from bjj_pipeline.contracts.f0_projection import project_to_world

    pfm = pd.read_parquet(pfm_path)
    matched = pfm[pfm["match_status"] == "matched"].copy()

    result = {}
    for gt_id in sorted(matched["gt_track_id"].unique()):
        gt = matched[matched["gt_track_id"] == gt_id]
        in_bp = 0
        in_quad = 0
        total = 0
        for _, r in gt.iterrows():
            u, v, _, _ = contact_point_from_bbox(
                (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])
            )
            x_m, y_m = project_to_world(
                (u, v), projection.H, projection.camera_matrix,
                projection.dist_coefficients,
            )
            if np.isnan(x_m) or np.isnan(y_m):
                continue
            total += 1
            if (MAT_BLUEPRINT_X[0] <= x_m <= MAT_BLUEPRINT_X[1]
                    and MAT_BLUEPRINT_Y[0] <= y_m <= MAT_BLUEPRINT_Y[1]):
                in_bp += 1
            if (CALIBRATED_QUAD_X[0] <= x_m <= CALIBRATED_QUAD_X[1]
                    and CALIBRATED_QUAD_Y[0] <= y_m <= CALIBRATED_QUAD_Y[1]):
                in_quad += 1

        pct_bp = 100.0 * in_bp / total if total > 0 else 0.0
        pct_quad = 100.0 * in_quad / total if total > 0 else 0.0
        result[int(gt_id)] = {
            "on_mat_blueprint": pct_bp >= 50.0,
            "in_quad_pct": round(pct_quad, 1),
        }

    return result


def _compute_median_box_area(pfm_path: Path) -> dict[int, float]:
    """Median GT box area per GT track from per_frame_matches."""
    pfm = pd.read_parquet(pfm_path)
    matched = pfm[pfm["match_status"] == "matched"].copy()
    matched["area"] = (matched["gt_x2"] - matched["gt_x1"]) * (matched["gt_y2"] - matched["gt_y1"])
    return matched.groupby("gt_track_id")["area"].median().to_dict()


def build_sequence_table(
    trace_path: Path,
    nodes_path: Path,
    detections_path: Path,
    pfm_path: Path,
    total_clip_frames: int = 1764,
    projection=None,
) -> pd.DataFrame:
    """Build the contiguous-segment table from gt_person_trace.parquet.

    Each row represents a contiguous run of frames for one GT track where
    (tracklet_id, d1_node_id, final_person_id) are all the same.
    """
    trace = pd.read_parquet(trace_path)
    nodes_df = pd.read_parquet(nodes_path)

    # Parse d1_node_ids (always len 0 or 1 on this clip — verified)
    trace["_node_id"] = trace["d1_node_ids"].apply(_parse_node_id)
    trace["_node_type"] = trace["d1_node_types"].apply(_parse_node_type)

    # Build node metadata lookup
    node_meta = {}
    for _, n in nodes_df.iterrows():
        node_meta[n["node_id"]] = {
            "segment_type": n["segment_type"],
            "capacity": int(n["capacity"]),
        }

    # Tracklet purity
    purity_map = _compute_tracklet_purity(trace)

    # Group GT-box IoU
    group_iou_map = _compute_group_gt_box_iou(trace, nodes_df)

    # GT presence (total frames per GT track in trace)
    gt_presence = _compute_gt_presence(trace)

    # Matched frames per GT track (frames with a detection)
    gt_matched = trace[trace["tracklet_id"].notna()].groupby("gt_person_id").size().to_dict()

    # Median box area
    median_area = _compute_median_box_area(pfm_path)

    # On-mat classification via production projection
    if projection is not None:
        on_mat_status = _compute_on_mat_status(pfm_path, projection)
    else:
        # Fallback: all on-mat (cannot classify without projection)
        on_mat_status = {int(gt): {"on_mat_blueprint": True, "in_quad_pct": None}
                         for gt in trace["gt_person_id"].unique()}

    # Canonical person_id per GT track
    canonical = {}
    for gt_id in trace["gt_person_id"].unique():
        vals = trace[trace["gt_person_id"] == gt_id]["canonical_person_id"].dropna().unique()
        canonical[gt_id] = vals[0] if len(vals) > 0 else None

    segments = []

    for gt_id in sorted(trace["gt_person_id"].unique()):
        gt_sub = trace[trace["gt_person_id"] == gt_id].sort_values("frame_idx")
        gt_int = int(gt_id)

        matched_count = gt_matched.get(gt_int, 0)
        presence_count = gt_presence.get(gt_int, 0)
        coverage_clip = 100.0 * matched_count / total_clip_frames
        coverage_presence = 100.0 * matched_count / presence_count if presence_count > 0 else 0.0
        low_conf = coverage_presence < COVERAGE_FLOOR_PCT

        canon_pid = canonical.get(gt_int)
        mat_info = on_mat_status.get(gt_int, {"on_mat_blueprint": True, "in_quad_pct": None})

        # RLE: group consecutive frames with same (tracklet_id, _node_id, final_person_id)
        seg_index = 0
        prev_key = None
        seg_frames = []

        for _, row in gt_sub.iterrows():
            tid = row["tracklet_id"] if pd.notna(row.get("tracklet_id")) else None
            nid = row["_node_id"]
            pid = row["final_person_id"] if pd.notna(row.get("final_person_id")) else None
            key = (tid, nid, pid)
            fi = int(row["frame_idx"])

            if key != prev_key:
                if seg_frames:
                    segments.append(_make_segment_row(
                        gt_int, seg_index, seg_frames, prev_key,
                        purity_map, node_meta, group_iou_map,
                        canon_pid, matched_count, coverage_clip, coverage_presence,
                        low_conf, median_area.get(float(gt_int)),
                        gt_sub, mat_info,
                    ))
                    seg_index += 1
                seg_frames = [(fi, row)]
                prev_key = key
            else:
                seg_frames.append((fi, row))

        # Last segment
        if seg_frames:
            segments.append(_make_segment_row(
                gt_int, seg_index, seg_frames, prev_key,
                purity_map, node_meta, group_iou_map,
                canon_pid, matched_count, coverage_clip, coverage_presence,
                low_conf, median_area.get(float(gt_int)),
                gt_sub, mat_info,
            ))

    return pd.DataFrame(segments)


def _make_segment_row(
    gt_id: int,
    seg_index: int,
    seg_frames: list,
    key: tuple,
    purity_map: dict,
    node_meta: dict,
    group_iou_map: dict,
    canon_pid: str | None,
    matched_count: int,
    coverage_clip: float,
    coverage_presence: float,
    low_conf: bool,
    median_area: float | None,
    gt_sub: pd.DataFrame,
    mat_info: dict | None = None,
) -> dict:
    tid, nid, pid = key
    frame_start = seg_frames[0][0]
    frame_end = seg_frames[-1][0]
    n_frames = len(seg_frames)

    # Purity
    purity_info = purity_map.get((tid, gt_id)) if tid else None
    tracklet_purity = purity_info["purity"] if purity_info else None
    tracklet_total_matched = purity_info["total_matched"] if purity_info else None

    # Node metadata
    nm = node_meta.get(nid, {}) if nid else {}
    seg_type = nm.get("segment_type")
    capacity = nm.get("capacity")
    in_group = seg_type == "GROUP"

    # Group GT-box IoU
    g_iou = group_iou_map.get(nid) if in_group else None

    # Agreement
    agrees = (pid == canon_pid) if (pid is not None and canon_pid is not None) else None

    # Dominant failure mode in this segment
    modes = [r["failure_mode"] for _, r in seg_frames]
    mode_counts = pd.Series(modes).value_counts()
    dominant_mode = mode_counts.index[0] if len(mode_counts) > 0 else None

    mi = mat_info or {"on_mat_blueprint": True, "in_quad_pct": None}
    return {
        "gt_track_id": gt_id,
        "on_mat_blueprint": mi["on_mat_blueprint"],
        "in_quad_pct": mi["in_quad_pct"],
        "gt_matched_frames": matched_count,
        "coverage_clip_pct": round(coverage_clip, 1),
        "coverage_presence_pct": round(coverage_presence, 1),
        "low_confidence": low_conf,
        "median_box_area": round(median_area, 1) if median_area is not None else None,
        "seg_index": seg_index,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "n_frames": n_frames,
        "tracklet_id": tid,
        "tracklet_purity": round(tracklet_purity, 4) if tracklet_purity is not None else None,
        "tracklet_total_matched": tracklet_total_matched,
        "d1_node_id": nid,
        "d1_segment_type": seg_type,
        "d1_capacity": capacity,
        "in_group_span": in_group,
        "group_gt_box_iou": round(g_iou, 4) if g_iou is not None else None,
        "person_id": pid,
        "canonical_person_id": canon_pid,
        "agrees_with_canonical": agrees,
        "failure_mode": dominant_mode,
    }
