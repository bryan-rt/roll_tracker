"""GT person trace — per-frame per-GT-person failure mode classification.

Joins Stage A evaluation matches, Stage A detections, D1 graph nodes,
D3 solution ledger, and Stage D person tracks into a single trace that
classifies every (frame, GT-person) into one of six failure modes (full)
or four failure modes (lite, when pipeline artifacts are unavailable).

Failure mode resolution order matters — first match wins. Do not reorder.

CP6 — permanent validation layer.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FAILURE_MODES = (
    "present",
    "stage_a_no_detection",
    "stage_a_untracked",
    "d3_dropped",
    "d4_unassigned",
    "present_misattributed",
    "missing_canonical",
)

FAILURE_MODES_LITE = (
    "present",
    "stage_a_no_detection",
    "stage_d_no_person",
    "present_misattributed",
    "missing_canonical",
)

FULL_TO_LITE = {
    "present": "present",
    "stage_a_no_detection": "stage_a_no_detection",
    "stage_a_untracked": "stage_d_no_person",
    "d3_dropped": "stage_d_no_person",
    "d4_unassigned": "stage_d_no_person",
    "present_misattributed": "present_misattributed",
    "missing_canonical": "missing_canonical",
}


@dataclass
class TraceResult:
    camera_id: str
    model_id: str
    trace_df: pd.DataFrame              # gt_person_trace.parquet schema
    person_summary_df: pd.DataFrame     # gt_person_summary.parquet schema
    camera_summary: dict                # single row of gt_camera_summary
    warnings: list[str] = field(default_factory=list)
    mode: str = "full"                  # "full" or "lite"


# ---------------------------------------------------------------------------
# Identity mapping loader
# ---------------------------------------------------------------------------

def _load_identity_mapping(path: Path) -> dict[int, str]:
    """Load identity_mapping.json -> {gt_track_id_int: canonical_person_id}.

    Entries with canonical_person_id == None are omitted (they represent
    GT tracks with zero matched frames).  Raises ValueError if the file
    is missing or empty.
    """
    if not path.exists():
        raise ValueError(f"identity_mapping.json not found: {path}")
    raw = json.loads(path.read_text())
    if not raw:
        raise ValueError(f"identity_mapping.json is empty: {path}")

    mapping: dict[int, str] = {}
    purities: list[float] = []
    for k, v in raw.items():
        tid = int(k.removeprefix("gt_track_"))
        cpid = v.get("canonical_person_id")
        if cpid is not None:
            mapping[tid] = cpid
        purities.append(v.get("purity", 0.0))

    if purities:
        logger.info(
            "identity_mapping loaded: %d GT tracks, min purity %.2f, mean purity %.2f",
            len(raw),
            min(purities),
            sum(purities) / len(purities),
        )

    return mapping


# ---------------------------------------------------------------------------
# D1 lookup
# ---------------------------------------------------------------------------

def _build_d1_lookup(
    nodes_df: pd.DataFrame,
) -> dict[tuple[str, int], list[dict]]:
    """Explode D1 nodes into (tracklet_id, frame_idx) -> [node_info] map.

    Indexes SINGLE_TRACKLET by base_tracklet_id; GROUP_TRACKLET by
    base + carrier + disappearing + new tracklet_ids (all roles).
    Stores list of node_info to surface overlap.
    """
    lookup: dict[tuple[str, int], list[dict]] = {}
    for _, row in nodes_df.iterrows():
        node_type = str(row["node_type"])
        if "SOURCE" in node_type or "SINK" in node_type:
            continue
        sf, ef = int(row["start_frame"]), int(row["end_frame"])
        base_tid = str(row.get("base_tracklet_id", ""))
        tids_to_index = {base_tid}
        if "GROUP" in node_type:
            for col in [
                "carrier_tracklet_id",
                "disappearing_tracklet_id",
                "new_tracklet_id",
            ]:
                v = row.get(col)
                if isinstance(v, str) and v and v not in ("None", "none"):
                    tids_to_index.add(v)
        info = {
            "node_id": str(row["node_id"]),
            "node_type": node_type,
            "segment_type": str(row.get("segment_type", "")),
            "carrier_tracklet_id": row.get("carrier_tracklet_id"),
        }
        for tid in tids_to_index:
            for fi in range(sf, ef + 1):
                lookup.setdefault((tid, fi), []).append(info)
    return lookup


# ---------------------------------------------------------------------------
# D3 status
# ---------------------------------------------------------------------------

def _build_d3_status(ledger_path: Path) -> dict[str, str]:
    """Load d3_solution_ledger.json -> {tracklet_id: 'explained'|'dropped'}."""
    if not ledger_path.exists():
        return {}
    with open(ledger_path) as f:
        ledger = json.load(f)
    status: dict[str, str] = {}
    for tid in ledger.get("explained_tracklets", []):
        status[str(tid)] = "explained"
    for item in ledger.get("dropped_tracklets", []):
        if isinstance(item, dict):
            status[item["base_tracklet_id"]] = "dropped"
        else:
            status[str(item)] = "dropped"
    return status


# ---------------------------------------------------------------------------
# Person tracks index
# ---------------------------------------------------------------------------

def _build_person_ids_for_detection(
    pt_df: pd.DataFrame,
) -> dict[str, set[str]]:
    """Build detection_id -> set[person_id] from person_tracks."""
    result: dict[str, set[str]] = {}
    for det_id, pid in zip(pt_df["detection_id"], pt_df["person_id"]):
        result.setdefault(det_id, set()).add(pid)
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(trace_df: pd.DataFrame) -> None:
    """Assert sum(failure_mode counts) == total_frames per GT person."""
    for gt_pid, grp in trace_df.groupby("gt_person_id"):
        total = len(grp)
        mode_counts = int(grp["failure_mode"].value_counts().sum())
        assert total == mode_counts, (
            f"Smoke test failed for gt_person_id={gt_pid}: "
            f"{total} frames but {mode_counts} failure_mode entries"
        )


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_person_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    """GROUP BY (camera_id, clip_id, gt_person_id) -> per-person summary."""
    if trace_df.empty:
        return pd.DataFrame()

    mode = trace_df["mode"].iloc[0]
    modes_list = FAILURE_MODES if mode == "full" else FAILURE_MODES_LITE

    summaries = []
    for (cam, clip, gt_pid), grp in trace_df.groupby(
        ["camera_id", "clip_id", "gt_person_id"]
    ):
        total = len(grp)
        row: dict = {
            "camera_id": cam,
            "clip_id": clip,
            "gt_track_id": gt_pid,
            "total_frames": total,
            "canonical_person_id": grp["canonical_person_id"].iloc[0],
        }
        counts = grp["failure_mode"].value_counts()
        for fm in modes_list:
            n = int(counts.get(fm, 0))
            row[f"n_{fm}"] = n
            row[f"pct_{fm}"] = n / total if total > 0 else 0.0

        if mode == "full":
            tracklets = grp["tracklet_id"].dropna().unique()
            row["n_unique_tracklets"] = len(tracklets)
            row["tracklet_ids"] = sorted(tracklets.tolist())

        summaries.append(row)

    return pd.DataFrame(summaries)


def _compute_camera_summary_row(
    trace_df: pd.DataFrame,
    camera_id: str,
    clip_id: str,
) -> dict:
    """Single summary dict for one camera."""
    total = len(trace_df)
    mode = trace_df["mode"].iloc[0] if total > 0 else "full"
    modes_list = FAILURE_MODES if mode == "full" else FAILURE_MODES_LITE

    counts = trace_df["failure_mode"].value_counts() if total > 0 else pd.Series(dtype=int)
    summary: dict = {
        "camera_id": camera_id,
        "clip_id": clip_id,
        "total_frames": total,
        "mode": mode,
    }
    for fm in modes_list:
        n = int(counts.get(fm, 0))
        summary[f"n_{fm}"] = n
        summary[f"pct_{fm}"] = n / total if total > 0 else 0.0

    return summary


# ---------------------------------------------------------------------------
# Full trace (6 failure modes)
# ---------------------------------------------------------------------------

def _compute_full_trace(
    eval_dir: Path,
    model_id: str,
    camera_id: str,
    pipeline_clip_dir: Path,
    identity_mapping: dict[int, str],
    warnings: list[str],
) -> TraceResult:
    """Full 6-mode trace joining all pipeline artifacts."""
    # 1. Load per_frame_matches
    pfm_path = eval_dir / "stage_a" / model_id / camera_id / "per_frame_matches.parquet"
    pfm = pd.read_parquet(pfm_path)

    # 2. Filter unmatched_pred rows (NULL gt_track_id)
    unmatched_pred = pfm["gt_track_id"].isna()
    n_unmatched = int(unmatched_pred.sum())
    if n_unmatched > 0:
        warnings.append(
            f"Filtered {n_unmatched} unmatched_pred rows (no gt_track_id)"
        )
    pfm = pfm[~unmatched_pred].copy()

    # 3. Cast gt_track_id to int
    pfm["gt_track_id"] = pfm["gt_track_id"].astype(int)

    # 4. Load detections.parquet -> map pred_detection_id -> tracklet_id
    det_df = pd.read_parquet(pipeline_clip_dir / "stage_A" / "detections.parquet")
    det_tracklet_map = dict(zip(det_df["detection_id"], det_df["tracklet_id"]))
    pfm["tracklet_id"] = pfm["pred_detection_id"].map(det_tracklet_map)

    # 5. Count detection_id mismatches
    has_pred = pfm["pred_detection_id"].notna()
    no_tracklet = pfm["tracklet_id"].isna()
    mismatch_count = int((has_pred & no_tracklet).sum())
    if mismatch_count > 0:
        warnings.append(
            f"Detection ID mismatch: {mismatch_count} of {len(pfm)} rows "
            f"had no tracklet_id in detections.parquet"
        )

    # 6. Build D1 lookup
    d1_nodes = pd.read_parquet(
        pipeline_clip_dir / "stage_D" / "d1_graph_nodes.parquet"
    )
    d1_lookup = _build_d1_lookup(d1_nodes)

    # 7. Build D3 status
    d3_status = _build_d3_status(
        pipeline_clip_dir / "_debug" / "d3_solution_ledger.json"
    )

    # 8. Build person_ids_for_detection
    pt_df = pd.read_parquet(
        pipeline_clip_dir / "stage_D" / "person_tracks.parquet"
    )
    person_ids_for_det = _build_person_ids_for_detection(pt_df)

    # 9. Get clip_id
    clip_id = det_df["clip_id"].iloc[0] if len(det_df) > 0 else "unknown"

    # 10. Build trace rows — resolution order matters, do not reorder
    rows = []
    for _, r in pfm.iterrows():
        gt_tid = int(r["gt_track_id"])
        frame_idx = int(r["frame_index"])
        pred_det_id = r["pred_detection_id"] if pd.notna(r.get("pred_detection_id")) else None
        tracklet_id = r["tracklet_id"] if pd.notna(r.get("tracklet_id")) else None
        iou_val = float(r["iou"]) if pd.notna(r.get("iou")) else 0.0
        gt_bbox = [
            float(r[c]) for c in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")
        ]

        canonical_pid = identity_mapping.get(gt_tid)

        # D1 info
        d1_info_list = (
            d1_lookup.get((tracklet_id, frame_idx), []) if tracklet_id else []
        )
        d1_node_ids = [info["node_id"] for info in d1_info_list] or None
        d1_node_types = [info["node_type"] for info in d1_info_list] or None
        d1_carrier_status = None
        if d1_info_list:
            for info in d1_info_list:
                carrier_tid = info.get("carrier_tracklet_id")
                if carrier_tid and carrier_tid == tracklet_id:
                    d1_carrier_status = "carrier"
                    break
            if d1_carrier_status is None:
                d1_carrier_status = "non_carrier"
        d1_node_overlap = len(d1_info_list) > 1 if d1_info_list else None

        # D3 status
        d3_st = d3_status.get(tracklet_id) if tracklet_id else None

        # Person IDs for this detection
        det_person_ids = (
            person_ids_for_det.get(pred_det_id, set()) if pred_det_id else set()
        )

        # --- Failure mode resolution (order matters!) ---
        if canonical_pid is None:
            failure_mode = "missing_canonical"
            final_person_id = None
        elif pred_det_id is None:
            failure_mode = "stage_a_no_detection"
            final_person_id = None
        elif tracklet_id is None:
            failure_mode = "stage_a_untracked"
            final_person_id = None
        elif d3_st == "dropped":
            failure_mode = "d3_dropped"
            final_person_id = None
        elif pred_det_id not in person_ids_for_det:
            failure_mode = "d4_unassigned"
            final_person_id = None
        elif canonical_pid in det_person_ids:
            failure_mode = "present"
            final_person_id = canonical_pid
        else:
            failure_mode = "present_misattributed"
            final_person_id = min(det_person_ids) if det_person_ids else None

        rows.append({
            "camera_id": camera_id,
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "gt_person_id": gt_tid,
            "gt_bbox": gt_bbox,
            "pred_detection_id": pred_det_id,
            "iou": iou_val,
            "tracklet_id": tracklet_id,
            "d1_node_ids": d1_node_ids,
            "d1_node_types": d1_node_types,
            "d1_carrier_status": d1_carrier_status,
            "d1_node_overlap": d1_node_overlap,
            "d3_status": d3_st,
            "d3_path_type": None,
            "final_person_id": final_person_id,
            "final_person_ids_all": sorted(det_person_ids) if det_person_ids else None,
            "is_group_ambiguous": len(det_person_ids) > 1 if det_person_ids else None,
            "failure_mode": failure_mode,
            "canonical_person_id": canonical_pid,
            "mode": "full",
        })

    trace_df = pd.DataFrame(rows)
    _smoke_test(trace_df)

    person_summary_df = _compute_person_summary(trace_df)
    camera_summary = _compute_camera_summary_row(trace_df, camera_id, clip_id)

    return TraceResult(
        camera_id=camera_id,
        model_id=model_id,
        trace_df=trace_df,
        person_summary_df=person_summary_df,
        camera_summary=camera_summary,
        warnings=warnings,
        mode="full",
    )


# ---------------------------------------------------------------------------
# Lite trace (4 failure modes)
# ---------------------------------------------------------------------------

def _compute_lite_trace(
    eval_dir: Path,
    model_id: str,
    camera_id: str,
    identity_mapping: dict[int, str],
    warnings: list[str],
) -> TraceResult:
    """Lite 4-mode trace from gt_track_sequences.jsonl + identity_mapping."""
    seq_path = eval_dir / "stage_d" / model_id / camera_id / "gt_track_sequences.jsonl"
    if not seq_path.exists():
        raise ValueError(f"gt_track_sequences.jsonl not found: {seq_path}")

    sequences = []
    with open(seq_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sequences.append(json.loads(line))

    rows = []
    for seq in sequences:
        gt_tid = int(seq["gt_track_id"])
        canonical_pid = identity_mapping.get(gt_tid)

        for frame in seq["frames"]:
            frame_idx = int(frame["frame_index"])
            person_id = frame.get("person_id")
            match_status = frame.get("match_status", "")
            iou_val = float(frame.get("iou", 0.0))

            # Resolution order matters — first match wins
            if canonical_pid is None:
                failure_mode = "missing_canonical"
            elif match_status == "unmatched_gt":
                failure_mode = "stage_a_no_detection"
            elif person_id is None:
                failure_mode = "stage_d_no_person"
            elif person_id == canonical_pid:
                failure_mode = "present"
            else:
                failure_mode = "present_misattributed"

            final_pid = None
            if failure_mode == "present":
                final_pid = canonical_pid
            elif failure_mode == "present_misattributed":
                final_pid = person_id

            rows.append({
                "camera_id": camera_id,
                "clip_id": camera_id,
                "frame_idx": frame_idx,
                "gt_person_id": gt_tid,
                "gt_bbox": None,
                "pred_detection_id": None,
                "iou": iou_val,
                "tracklet_id": None,
                "d1_node_ids": None,
                "d1_node_types": None,
                "d1_carrier_status": None,
                "d1_node_overlap": None,
                "d3_status": None,
                "d3_path_type": None,
                "final_person_id": final_pid,
                "final_person_ids_all": None,
                "is_group_ambiguous": None,
                "failure_mode": failure_mode,
                "canonical_person_id": canonical_pid,
                "mode": "lite",
            })

    trace_df = pd.DataFrame(rows)
    _smoke_test(trace_df)

    person_summary_df = _compute_person_summary(trace_df)
    camera_summary = _compute_camera_summary_row(trace_df, camera_id, "all")

    return TraceResult(
        camera_id=camera_id,
        model_id=model_id,
        trace_df=trace_df,
        person_summary_df=person_summary_df,
        camera_summary=camera_summary,
        warnings=warnings,
        mode="lite",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gt_person_trace(
    eval_dir: Path,
    model_id: str,
    camera_id: str,
    pipeline_clip_dir: Path | None = None,
) -> TraceResult:
    """Compute GT-anchored per-frame failure mode trace.

    If pipeline_clip_dir is provided and contains needed artifacts ->
    full 6-mode trace.  If None or missing -> lite 4-mode trace from
    gt_track_sequences.jsonl + identity_mapping.json.

    Raises ValueError if identity_mapping.json is empty or missing.
    """
    warnings: list[str] = []

    # Always need identity_mapping
    id_map_path = (
        eval_dir / "stage_d" / model_id / camera_id / "identity_mapping.json"
    )
    identity_mapping = _load_identity_mapping(id_map_path)

    # Determine mode
    can_full = (
        pipeline_clip_dir is not None
        and (pipeline_clip_dir / "stage_A" / "detections.parquet").exists()
        and (pipeline_clip_dir / "stage_D" / "person_tracks.parquet").exists()
        and (pipeline_clip_dir / "stage_D" / "d1_graph_nodes.parquet").exists()
    )

    pfm_path = (
        eval_dir / "stage_a" / model_id / camera_id / "per_frame_matches.parquet"
    )
    if can_full and not pfm_path.exists():
        can_full = False
        warnings.append(
            f"per_frame_matches.parquet not found at {pfm_path}, "
            f"falling back to lite mode"
        )

    if can_full:
        return _compute_full_trace(
            eval_dir, model_id, camera_id, pipeline_clip_dir,
            identity_mapping, warnings,
        )
    else:
        return _compute_lite_trace(
            eval_dir, model_id, camera_id, identity_mapping, warnings,
        )


def rollup_full_to_lite(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Map 6-mode failure_mode column to 4-mode for longitudinal comparison.

    Mapping:
      present -> present
      stage_a_no_detection -> stage_a_no_detection
      stage_a_untracked -> stage_d_no_person
      d3_dropped -> stage_d_no_person
      d4_unassigned -> stage_d_no_person
      present_misattributed -> present_misattributed
      missing_canonical -> missing_canonical
    """
    df = trace_df.copy()
    df["failure_mode"] = df["failure_mode"].map(FULL_TO_LITE)
    df["mode"] = "lite"
    return df


def write_trace_artifacts(result: TraceResult, output_dir: Path) -> None:
    """Write gt_person_trace.parquet and gt_person_summary.parquet."""
    out = output_dir / "stage_d" / result.model_id / result.camera_id
    out.mkdir(parents=True, exist_ok=True)

    # Serialize list columns to JSON strings for parquet compatibility
    trace_pq = result.trace_df.copy()
    for col in ("d1_node_ids", "d1_node_types", "final_person_ids_all", "gt_bbox"):
        if col in trace_pq.columns:
            trace_pq[col] = trace_pq[col].apply(
                lambda x: json.dumps(x) if x is not None else None
            )

    summary_pq = result.person_summary_df.copy()
    if "tracklet_ids" in summary_pq.columns:
        summary_pq["tracklet_ids"] = summary_pq["tracklet_ids"].apply(
            lambda x: json.dumps(x) if x is not None else None
        )

    trace_pq.to_parquet(out / "gt_person_trace.parquet", index=False)
    summary_pq.to_parquet(out / "gt_person_summary.parquet", index=False)
    logger.info("Wrote trace artifacts to %s", out)


def write_camera_summary(
    results: list[TraceResult],
    output_dir: Path,
    model_id: str,
) -> None:
    """Aggregate across cameras. Writes both full and lite summary parquets."""
    out = output_dir / "stage_d" / model_id
    out.mkdir(parents=True, exist_ok=True)

    # Full camera summary (6-mode columns where full, lite columns where lite)
    full_rows = [r.camera_summary for r in results]
    pd.DataFrame(full_rows).to_parquet(
        out / "gt_camera_summary.parquet", index=False
    )

    # Lite rollup summary (all rows rolled to 4 lite modes)
    lite_rows = []
    for r in results:
        if r.mode == "full":
            rolled = rollup_full_to_lite(r.trace_df)
        else:
            rolled = r.trace_df
        lite_summary = _compute_camera_summary_row(
            rolled, r.camera_id, r.camera_summary.get("clip_id", "all"),
        )
        lite_rows.append(lite_summary)
    pd.DataFrame(lite_rows).to_parquet(
        out / "gt_camera_summary_lite.parquet", index=False
    )
    logger.info("Wrote camera summaries to %s", out)
