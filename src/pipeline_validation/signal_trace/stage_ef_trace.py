"""E/F signal extension trace (CP-TRACE-3).

Follows each GT person's correctly-identified signal through Stage E
(match sessions) and Stage F (clip export). Produces per-GT-person
end-to-end classification.

Gracefully degrades when Stage F artifacts don't exist (pipeline ran
--to-stage E).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_match_sessions(stage_e_dir: Path) -> list[dict]:
    """Load match_sessions.jsonl, filtering to match_session artifacts."""
    sessions = []
    ms_path = stage_e_dir / "match_sessions.jsonl"
    if not ms_path.exists():
        return sessions
    with open(ms_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("artifact_type") == "match_session":
                sessions.append(ev)
    return sessions


def run_ef_trace(
    d_trace_path: Path,
    stage_e_dir: Path | None,
    stage_f_dir: Path | None,
) -> tuple[pd.DataFrame, dict]:
    """Per-GT-person E/F trace.

    Returns (per_gt_person_df, ef_summary).
    """
    d_trace = pd.read_parquet(d_trace_path)

    # Aggregate per GT track from D-trace
    per_track_records: list[dict] = []
    for gt_tid, grp in d_trace.groupby("gt_track_id"):
        dc = grp.d_classification.value_counts().to_dict()
        dominant = grp.dominant_person_id.dropna().iloc[0] if grp.dominant_person_id.notna().any() else None
        correct = dc.get("correct_id", 0)
        total = len(grp)
        per_track_records.append({
            "gt_track_id": gt_tid,
            "dominant_person_id": dominant,
            "n_correct_id_frames": correct,
            "n_wrong_id_frames": dc.get("wrong_id", 0),
            "n_no_id_frames": dc.get("no_id", 0),
            "n_no_detection_frames": dc.get("no_detection", 0),
            "purity": round(correct / total, 4) if total else 0,
        })

    per_gt = pd.DataFrame(per_track_records)

    # Load match sessions
    sessions: list[dict] = []
    if stage_e_dir and stage_e_dir.exists():
        sessions = _load_match_sessions(stage_e_dir)
    logger.info("Loaded %d match sessions", len(sessions))

    # Build person_id -> list of session info
    pid_to_sessions: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        pid_a = s.get("person_id_a")
        pid_b = s.get("person_id_b")
        match_id = s.get("match_id", "")
        info = {
            "match_id": match_id,
            "start_frame": s.get("start_frame"),
            "end_frame": s.get("end_frame"),
        }
        if pid_a:
            pid_to_sessions[pid_a].append({**info, "partner": pid_b})
        if pid_b:
            pid_to_sessions[pid_b].append({**info, "partner": pid_a})

    # Stage F check
    f_available = stage_f_dir is not None and stage_f_dir.exists()

    # Classify each GT person
    match_ids_col: list[str] = []
    n_sessions_col: list[int] = []
    has_clip_col: list[bool] = []
    e2e_col: list[str] = []
    clip_files_col: list[str] = []

    for _, row in per_gt.iterrows():
        dominant = row.dominant_person_id
        if dominant is None:
            match_ids_col.append("[]")
            n_sessions_col.append(0)
            has_clip_col.append(False)
            e2e_col.append("lost_at_d")
            clip_files_col.append("[]")
            continue

        person_sessions = pid_to_sessions.get(dominant, [])
        session_ids = [s["match_id"] for s in person_sessions]

        if not session_ids:
            match_ids_col.append("[]")
            n_sessions_col.append(0)
            has_clip_col.append(False)
            e2e_col.append("no_match")
            clip_files_col.append("[]")
        else:
            match_ids_col.append(json.dumps(session_ids))
            n_sessions_col.append(len(session_ids))
            has_clip_col.append(False)  # F not available
            e2e_col.append("in_match_session")
            clip_files_col.append("[]")

    per_gt["match_session_ids"] = match_ids_col
    per_gt["n_match_sessions"] = n_sessions_col
    per_gt["has_exported_clip"] = has_clip_col
    per_gt["e2e_classification"] = e2e_col
    per_gt["clip_filenames"] = clip_files_col

    # Build summary
    e2e_counts = per_gt.e2e_classification.value_counts().to_dict()
    total_gt = len(per_gt)
    summary = {
        "total_gt_tracks": total_gt,
        "stage_f_available": f_available,
        "e2e_classification": {},
    }
    for cls in ("in_match_session", "no_match", "lost_at_d"):
        c = e2e_counts.get(cls, 0)
        summary["e2e_classification"][cls] = {
            "count": c,
            "pct": round(c / total_gt, 4) if total_gt else 0,
        }

    if not f_available:
        summary["stage_f_note"] = (
            "Stage F not available (pipeline ran --to-stage E). "
            "E2E classification caps at in_match_session."
        )

    return per_gt, summary
