#!/usr/bin/env python3
"""Compare pipeline outputs between baseline and experiment runs.

Reads per-clip Stage A outputs and session-level D/E outputs from two
output directories, extracts key metrics, and prints a side-by-side
comparison table. Also writes a JSON report for programmatic consumption.

Usage:
    python tools/compare_runs.py
    python tools/compare_runs.py --baseline outputs --experiment outputs_cross_camera
    python tools/compare_runs.py --gym-id c8a592a4-2bca-400a-80e1-fec0e5cbea77
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, returning list of dicts. Empty list if missing."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _read_parquet_safe(path: Path) -> "pd.DataFrame | None":
    """Read a parquet file, returning None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _discover_cameras(gym_root: Path) -> list[str]:
    """Discover camera IDs from top-level directories under gym root."""
    if not gym_root.exists():
        return []
    return sorted(
        d.name for d in gym_root.iterdir()
        if d.is_dir() and d.name != "sessions"
    )


def _discover_clips(cam_root: Path) -> list[Path]:
    """Find all clip directories under a camera root (date/hour/clip_id/)."""
    clips = []
    for date_dir in sorted(cam_root.iterdir()) if cam_root.exists() else []:
        if not date_dir.is_dir():
            continue
        for hour_dir in sorted(date_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            for clip_dir in sorted(hour_dir.iterdir()):
                if clip_dir.is_dir():
                    clips.append(clip_dir)
    return clips


def _find_session_dir(gym_root: Path) -> Path | None:
    """Find the (first) session directory under gym_root/sessions/."""
    sessions_root = gym_root / "sessions"
    if not sessions_root.exists():
        return None
    for date_dir in sorted(sessions_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir()):
            if session_dir.is_dir():
                return session_dir
    return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(output_root: Path, gym_id: str) -> dict[str, Any]:
    """Extract all comparison metrics from an output directory."""
    gym_root = output_root / gym_id
    cameras = _discover_cameras(gym_root)
    session_dir = _find_session_dir(gym_root)

    metrics: dict[str, Any] = {
        "cameras": cameras,
        "clips_per_camera": {},
        "clips_failed": {},
        "on_mat_pct": {},
        "tracklet_count": {},
        "person_id_count": {},
        "person_track_rows": {},
        "identity_assignments": {},
        "unique_tags": {},
        "mean_confidence": {},
        "global_person_ids": 0,
        "linked_ids": 0,
        "standalone_ids": 0,
        "total_matches": 0,
        "matches_per_camera": {},
        "cp17_corroborated_tags": 0,
        "cp17_tag_corroborated": 0,
        "cp17_coordinate_corroborated": 0,
        "coordinate_conflicts": 0,
        "ilp_runtimes_ms": {},
        "identity_assignments_total": 0,
        "mean_confidence_all": 0.0,
    }

    # --- Per-clip Stage A metrics ---
    for cam_id in cameras:
        cam_root = gym_root / cam_id
        clips = _discover_clips(cam_root)
        metrics["clips_per_camera"][cam_id] = len(clips)

        all_on_mat = 0
        all_total = 0
        tracklet_ids: set[str] = set()
        failed = 0

        for clip_dir in clips:
            tf_path = clip_dir / "stage_A" / "tracklet_frames.parquet"
            df = _read_parquet_safe(tf_path)
            if df is None:
                failed += 1
                continue
            if "on_mat" in df.columns:
                on_mat_series = df["on_mat"].dropna()
                all_on_mat += int(on_mat_series.sum())
                all_total += len(on_mat_series)
            if "tracklet_id" in df.columns:
                tracklet_ids.update(df["tracklet_id"].unique())

        metrics["clips_failed"][cam_id] = failed
        metrics["on_mat_pct"][cam_id] = (
            round(100.0 * all_on_mat / all_total, 2) if all_total > 0 else 0.0
        )
        metrics["tracklet_count"][cam_id] = len(tracklet_ids)

    # --- Session-level Stage D metrics ---
    if session_dir:
        stage_d = session_dir / "stage_D"
        stage_e = session_dir / "stage_E"

        for cam_id in cameras:
            # Person tracks
            pt_path = stage_d / f"person_tracks_{cam_id}.parquet"
            pt_df = _read_parquet_safe(pt_path)
            if pt_df is not None and "person_id" in pt_df.columns:
                metrics["person_id_count"][cam_id] = int(pt_df["person_id"].nunique())
                metrics["person_track_rows"][cam_id] = len(pt_df)
            else:
                metrics["person_id_count"][cam_id] = 0
                metrics["person_track_rows"][cam_id] = 0

            # Identity assignments
            ia_path = stage_d / f"identity_assignments_{cam_id}.jsonl"
            ia_records = _read_jsonl(ia_path)
            metrics["identity_assignments"][cam_id] = len(ia_records)
            tag_ids = {r.get("tag_id") for r in ia_records if r.get("tag_id")}
            metrics["unique_tags"][cam_id] = len(tag_ids)
            confidences = [
                r["assignment_confidence"]
                for r in ia_records
                if r.get("assignment_confidence") is not None
            ]
            metrics["mean_confidence"][cam_id] = (
                round(sum(confidences) / len(confidences), 4)
                if confidences else 0.0
            )

        # Aggregate identity stats
        total_assignments = sum(metrics["identity_assignments"].values())
        all_confidences = []
        for cam_id in cameras:
            ia_path = stage_d / f"identity_assignments_{cam_id}.jsonl"
            for r in _read_jsonl(ia_path):
                if r.get("assignment_confidence") is not None:
                    all_confidences.append(r["assignment_confidence"])
        metrics["identity_assignments_total"] = total_assignments
        metrics["mean_confidence_all"] = (
            round(sum(all_confidences) / len(all_confidences), 4)
            if all_confidences else 0.0
        )

        # Cross-camera identities
        cc_path = stage_d / "cross_camera_identities.jsonl"
        cc_records = _read_jsonl(cc_path)
        # Group by global_person_id → set of cam_ids
        gp_cams: dict[str, set[str]] = defaultdict(set)
        for r in cc_records:
            gp_id = r.get("global_person_id", "")
            cam = r.get("cam_id", "")
            if gp_id and cam:
                gp_cams[gp_id].add(cam)
        metrics["global_person_ids"] = len(gp_cams)
        metrics["linked_ids"] = sum(1 for cams in gp_cams.values() if len(cams) >= 2)
        metrics["standalone_ids"] = sum(1 for cams in gp_cams.values() if len(cams) == 1)

        # Audit events
        audit_path = stage_d / "audit.jsonl"
        audit_records = _read_jsonl(audit_path)
        for r in audit_records:
            et = r.get("event_type", "")
            if et == "cp17_pass2_evidence":
                metrics["cp17_corroborated_tags"] += r.get("n_corroborated_tags", 0)
                metrics["cp17_tag_corroborated"] += r.get("n_corroborated_tags", 0)
            if et == "cp17_coordinate_evidence":
                metrics["cp17_coordinate_corroborated"] += r.get("n_coordinate_corroborated", 0)
            if et == "coordinate_conflict":
                metrics["coordinate_conflicts"] += 1
            if et == "d3_ilp_summary":
                cam = r.get("camera_id", "unknown")
                metrics["ilp_runtimes_ms"][cam] = r.get("runtime_ms", 0)

        # Match sessions
        ms_path = stage_e / "match_sessions.jsonl"
        ms_records = _read_jsonl(ms_path)
        metrics["total_matches"] = len(ms_records)
        cam_match_counts: dict[str, int] = defaultdict(int)
        for r in ms_records:
            cam = r.get("camera_id", "unknown")
            cam_match_counts[cam] += 1
        metrics["matches_per_camera"] = dict(cam_match_counts)

    return metrics


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def _fmt_int(val: int) -> str:
    return str(val)


def _fmt_float(val: float) -> str:
    return f"{val:.4f}"


def _delta_pct(base: float, exp: float) -> str:
    d = exp - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def _delta_int(base: int, exp: int) -> str:
    d = exp - base
    sign = "+" if d > 0 else ""
    return f"{sign}{d}" if d != 0 else "0"


def _delta_float(base: float, exp: float) -> str:
    d = exp - base
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.4f}" if abs(d) > 0.00005 else "0"


def print_comparison(base: dict, exp: dict) -> list[dict]:
    """Print side-by-side comparison table. Returns rows for JSON report."""
    rows: list[dict] = []

    def row(label: str, b_val: Any, e_val: Any, fmt_fn, delta_fn):
        b_str = fmt_fn(b_val)
        e_str = fmt_fn(e_val)
        d_str = delta_fn(b_val, e_val)
        print(f"  {label:<40s} {b_str:>12s} {e_str:>12s} {d_str:>12s}")
        rows.append({
            "metric": label,
            "baseline": b_val,
            "experiment": e_val,
            "delta": e_val - b_val if isinstance(e_val, (int, float)) else None,
        })

    all_cams = sorted(set(base.get("cameras", [])) | set(exp.get("cameras", [])))

    print()
    print("Pipeline Comparison Report")
    print("=" * 80)
    print(f"  {'Metric':<40s} {'Baseline':>12s} {'Experiment':>12s} {'Delta':>12s}")
    print(f"  {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12}")

    # On-mat %
    for cam in all_cams:
        row(
            f"On-mat % ({cam})",
            base.get("on_mat_pct", {}).get(cam, 0.0),
            exp.get("on_mat_pct", {}).get(cam, 0.0),
            _fmt_pct, _delta_pct,
        )

    # Tracklets
    for cam in all_cams:
        row(
            f"Tracklets ({cam})",
            base.get("tracklet_count", {}).get(cam, 0),
            exp.get("tracklet_count", {}).get(cam, 0),
            _fmt_int, _delta_int,
        )

    # Person IDs
    for cam in all_cams:
        row(
            f"Person IDs ({cam})",
            base.get("person_id_count", {}).get(cam, 0),
            exp.get("person_id_count", {}).get(cam, 0),
            _fmt_int, _delta_int,
        )

    # Identity assignments
    row(
        "Identity assignments (total)",
        base.get("identity_assignments_total", 0),
        exp.get("identity_assignments_total", 0),
        _fmt_int, _delta_int,
    )
    row(
        "Mean assignment confidence",
        base.get("mean_confidence_all", 0.0),
        exp.get("mean_confidence_all", 0.0),
        _fmt_float, _delta_float,
    )

    # Global IDs
    row(
        "Global person IDs",
        base.get("global_person_ids", 0),
        exp.get("global_person_ids", 0),
        _fmt_int, _delta_int,
    )
    row(
        "  Linked (multi-camera)",
        base.get("linked_ids", 0),
        exp.get("linked_ids", 0),
        _fmt_int, _delta_int,
    )
    row(
        "  Standalone (single-camera)",
        base.get("standalone_ids", 0),
        exp.get("standalone_ids", 0),
        _fmt_int, _delta_int,
    )

    # Matches
    row(
        "Matches detected",
        base.get("total_matches", 0),
        exp.get("total_matches", 0),
        _fmt_int, _delta_int,
    )
    for cam in all_cams:
        row(
            f"  Matches ({cam})",
            base.get("matches_per_camera", {}).get(cam, 0),
            exp.get("matches_per_camera", {}).get(cam, 0),
            _fmt_int, _delta_int,
        )

    # CP17 evidence
    row(
        "CP17 corroborated tags",
        base.get("cp17_corroborated_tags", 0),
        exp.get("cp17_corroborated_tags", 0),
        _fmt_int, _delta_int,
    )
    row(
        "  Tag-corroborated",
        base.get("cp17_tag_corroborated", 0),
        exp.get("cp17_tag_corroborated", 0),
        _fmt_int, _delta_int,
    )
    row(
        "  Coordinate-corroborated",
        base.get("cp17_coordinate_corroborated", 0),
        exp.get("cp17_coordinate_corroborated", 0),
        _fmt_int, _delta_int,
    )
    row(
        "Coordinate conflicts",
        base.get("coordinate_conflicts", 0),
        exp.get("coordinate_conflicts", 0),
        _fmt_int, _delta_int,
    )

    # ILP runtimes
    for cam in all_cams:
        b_rt = base.get("ilp_runtimes_ms", {}).get(cam, 0)
        e_rt = exp.get("ilp_runtimes_ms", {}).get(cam, 0)
        row(f"ILP runtime ms ({cam})", b_rt, e_rt, _fmt_int, _delta_int)

    # Failed clips
    print()
    print("  Clip status:")
    for cam in all_cams:
        b_total = base.get("clips_per_camera", {}).get(cam, 0)
        e_total = exp.get("clips_per_camera", {}).get(cam, 0)
        b_fail = base.get("clips_failed", {}).get(cam, 0)
        e_fail = exp.get("clips_failed", {}).get(cam, 0)
        print(f"    {cam}: baseline {b_total - b_fail}/{b_total} ok, "
              f"experiment {e_total - e_fail}/{e_total} ok")

    # Interpretation
    print()
    print("  Interpretation guidance:")
    print("    - Fewer person_ids = better (fewer false splits)")
    print("    - More linked global IDs = better cross-camera association")
    print("    - Fewer standalone IDs = better (more athletes linked across cameras)")
    print("    - Higher assignment confidence = better tag evidence")
    print("    - Higher on-mat % = better homography")
    print("    - Coordinate-corroborated tags > 0 = CP17 Tier 2 is contributing")
    print()

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    baseline: str = "outputs",
    experiment: str = "outputs_cross_camera",
    gym_id: str = "c8a592a4-2bca-400a-80e1-fec0e5cbea77",
) -> None:
    if pd is None:
        print("ERROR: pandas is required. Install with: pip install pandas", file=sys.stderr)
        sys.exit(1)

    baseline_path = Path(baseline)
    experiment_path = Path(experiment)

    if not baseline_path.exists():
        print(f"ERROR: Baseline path not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    if not experiment_path.exists():
        print(f"ERROR: Experiment path not found: {experiment_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting baseline metrics from: {baseline_path}")
    base_metrics = extract_metrics(baseline_path, gym_id)

    print(f"Extracting experiment metrics from: {experiment_path}")
    exp_metrics = extract_metrics(experiment_path, gym_id)

    rows = print_comparison(base_metrics, exp_metrics)

    # Write JSON report
    report = {
        "baseline_path": str(baseline_path),
        "experiment_path": str(experiment_path),
        "gym_id": gym_id,
        "baseline": base_metrics,
        "experiment": exp_metrics,
        "comparison_rows": rows,
    }
    report_path = experiment_path / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"JSON report written to: {report_path}")


if __name__ == "__main__":
    if typer is not None:
        typer.run(main)
    else:
        # Fallback: parse args manually
        import argparse
        parser = argparse.ArgumentParser(description="Compare pipeline runs")
        parser.add_argument("--baseline", default="outputs")
        parser.add_argument("--experiment", default="outputs_cross_camera")
        parser.add_argument("--gym-id", default="c8a592a4-2bca-400a-80e1-fec0e5cbea77")
        args = parser.parse_args()
        main(baseline=args.baseline, experiment=args.experiment, gym_id=args.gym_id)
