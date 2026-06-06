"""No-ID root cause diagnosis (CP-TRACE-3).

For each GT-person-frame classified as no_id in the D-trace, determines
WHERE in D0->D4 the tracklet was lost: d0_filtered, d1_excluded,
d3_solver_drop, or d4_frame_trim.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_no_id_diagnosis(
    d_trace_path: Path,
    stage_d_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Classify drop reason for every no_id frame.

    Returns (detail_df with drop_reason column, diagnosis_summary).
    """
    d_trace = pd.read_parquet(d_trace_path)

    # Load Stage D artifacts for lookup sets
    bank = pd.read_parquet(stage_d_dir / "tracklet_bank_summaries.parquet")
    bank_tids = set(bank.tracklet_id.unique())

    gn = pd.read_parquet(stage_d_dir / "d1_graph_nodes.parquet")
    d1_tids = set(gn[gn.base_tracklet_id.notna()].base_tracklet_id.unique())

    pt = pd.read_parquet(stage_d_dir / "person_tracks.parquet")
    pt_tids = set(pt.tracklet_id.unique())

    # Classify each no_id frame
    drop_reasons: list[str | None] = []
    for _, row in d_trace.iterrows():
        if row.d_classification != "no_id":
            drop_reasons.append(None)
            continue

        tid = row.tracklet_id
        if pd.isna(tid):
            # Shouldn't happen (no_id requires a tracklet), but defensive
            drop_reasons.append(None)
            continue

        if tid not in bank_tids:
            drop_reasons.append("d0_filtered")
        elif tid not in d1_tids:
            drop_reasons.append("d1_excluded")
        elif tid not in pt_tids:
            drop_reasons.append("d3_solver_drop")
        else:
            drop_reasons.append("d4_frame_trim")

    detail_df = d_trace.copy()
    detail_df["drop_reason"] = drop_reasons

    # Build summary
    no_id_rows = detail_df[detail_df.d_classification == "no_id"]
    total_no_id = len(no_id_rows)
    reason_counts = no_id_rows.drop_reason.value_counts().to_dict()

    reasons = ["d0_filtered", "d1_excluded", "d3_solver_drop", "d4_frame_trim"]
    summary: dict = {
        "total_no_id_frames": total_no_id,
    }
    for r in reasons:
        c = reason_counts.get(r, 0)
        summary[r] = {
            "count": c,
            "pct": round(c / total_no_id, 4) if total_no_id else 0,
        }

    # Per-tracklet detail
    per_tracklet: dict[str, dict] = {}
    for tid, grp in no_id_rows.groupby("tracklet_id"):
        reason = grp.drop_reason.iloc[0]  # same for all frames of a tracklet
        per_tracklet[str(tid)] = {
            "drop_reason": reason,
            "n_no_id_frames": len(grp),
        }
    summary["per_tracklet_detail"] = per_tracklet

    return detail_df, summary
