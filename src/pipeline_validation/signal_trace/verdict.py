"""Synthesis verdict for CP-TRACE series (CP-TRACE-3).

Reads all signal_trace artifacts and generates _verdict.md — the main
deliverable of the trace series.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TRACE_DIR = OUTPUTS_DIR / "_eval" / "signal_trace"


def generate_verdict(model_id: str, cameras: list[str]) -> Path:
    """Read all signal_trace artifacts and generate _verdict.md."""
    base = TRACE_DIR / model_id
    lines: list[str] = [
        f"# CP-TRACE Synthesis Verdict: {model_id}",
        "",
    ]

    # ---------------------------------------------------------------
    # 1. Signal flow waterfall
    # ---------------------------------------------------------------
    lines.append("## 1. Signal Flow Waterfall")
    lines.append("")

    # Aggregate Stage A topology
    total_frames = 0
    agg_tight = 0
    agg_pair = 0
    agg_miss = 0
    agg_split = 0
    # Aggregate D-trace
    agg_correct = 0
    agg_wrong = 0
    agg_no_id = 0
    agg_no_det = 0
    # Aggregate no-ID diagnosis
    agg_diag: dict[str, int] = {}
    # Aggregate E/F
    agg_e2e: dict[str, int] = {}
    total_gt_tracks = 0

    for cam in cameras:
        cam_dir = base / cam

        # Stage A topology
        topo_path = cam_dir / "topology_summary.json"
        if topo_path.exists():
            with open(topo_path) as f:
                topo = json.load(f)
            total_frames += topo["total_gt_person_frames"]
            agg_tight += topo["tight_match"]["count"]
            agg_pair += topo["pair_box"]["count"]
            agg_miss += topo["miss"]["count"]
            agg_split += topo["split"]["count"]

        # D-trace preservation
        pres_path = cam_dir / "signal_preservation_summary.json"
        if pres_path.exists():
            with open(pres_path) as f:
                pres = json.load(f)
            agg_correct += pres["correct_id"]["count"]
            agg_wrong += pres["wrong_id"]["count"]
            agg_no_id += pres["no_id"]["count"]
            agg_no_det += pres["no_detection"]["count"]

        # No-ID diagnosis
        diag_path = cam_dir / "no_id_diagnosis.json"
        if diag_path.exists():
            with open(diag_path) as f:
                diag = json.load(f)
            for reason in ("d0_filtered", "d1_excluded", "d3_solver_drop", "d4_frame_trim"):
                agg_diag[reason] = agg_diag.get(reason, 0) + diag.get(reason, {}).get("count", 0)

        # E/F
        ef_path = cam_dir / "ef_summary.json"
        if ef_path.exists():
            with open(ef_path) as f:
                ef = json.load(f)
            total_gt_tracks += ef["total_gt_tracks"]
            for cls, v in ef.get("e2e_classification", {}).items():
                agg_e2e[cls] = agg_e2e.get(cls, 0) + v["count"]

    detected = total_frames - agg_miss
    det_pct = f"{detected/total_frames:.1%}" if total_frames else "?"
    tight_pct = f"{agg_tight/total_frames:.1%}" if total_frames else "?"
    correct_pct = f"{agg_correct/total_frames:.1%}" if total_frames else "?"

    lines.extend([
        "```",
        f"{total_frames:,} GT-person-frames",
        f"  -> {detected:,} detected ({det_pct})                  [miss: {agg_miss:,}]",
        f"  -> {agg_tight:,} tight_match ({tight_pct})            [pair_box: {agg_pair:,}]",
        f"  -> {agg_correct:,} correct_id ({correct_pct})         [wrong: {agg_wrong:,}, no_id: {agg_no_id:,}]",
    ])
    if agg_e2e:
        in_ms = agg_e2e.get("in_match_session", 0)
        no_match = agg_e2e.get("no_match", 0)
        lost = agg_e2e.get("lost_at_d", 0)
        lines.append(
            f"  -> {in_ms}/{total_gt_tracks} GT people in match sessions  "
            f"[no_match: {no_match}, lost_at_d: {lost}]"
        )
    lines.extend([
        "  -> Stage F not available (pipeline ran --to-stage E)",
        "```",
        "",
    ])

    # ---------------------------------------------------------------
    # 2. No-ID root cause
    # ---------------------------------------------------------------
    lines.append("## 2. No-ID Root Cause Breakdown")
    lines.append("")
    lines.append(f"Total no_id frames: {agg_no_id:,}")
    lines.append("")
    lines.append("| Drop point | Count | Pct of no_id |")
    lines.append("|---|---:|---:|")
    for reason in ("d0_filtered", "d1_excluded", "d3_solver_drop", "d4_frame_trim"):
        c = agg_diag.get(reason, 0)
        pct = f"{c/agg_no_id:.1%}" if agg_no_id else "0.0%"
        lines.append(f"| {reason} | {c:,} | {pct} |")
    lines.extend([
        "",
        "**d4_frame_trim dominates:** tracklets ARE in person_tracks at other frames,",
        "but the annotated frame falls outside their solver-coverage window. This is",
        "a graph-lifetime/coverage issue, not solver rejection.",
        "",
    ])

    # ---------------------------------------------------------------
    # 3. Root cause ranking
    # ---------------------------------------------------------------
    lines.append("## 3. Root Cause Ranking (by frame impact)")
    lines.append("")

    failure_modes = [
        ("pair_box (Stage A under-segmentation)", agg_pair),
        ("no_id / d4_frame_trim (graph coverage gap)", agg_diag.get("d4_frame_trim", 0)),
        ("wrong_id (identity misattribution)", agg_wrong),
        ("miss (Stage A no detection)", agg_miss),
        ("no_id / d3_solver_drop", agg_diag.get("d3_solver_drop", 0)),
        ("no_id / d0_filtered", agg_diag.get("d0_filtered", 0)),
        ("no_id / d1_excluded", agg_diag.get("d1_excluded", 0)),
    ]
    failure_modes.sort(key=lambda x: x[1], reverse=True)

    lines.append("| Rank | Failure mode | Frames | Pct of total |")
    lines.append("|---:|---|---:|---:|")
    for i, (mode, count) in enumerate(failure_modes, 1):
        if count == 0:
            continue
        pct = f"{count/total_frames:.1%}" if total_frames else "?"
        lines.append(f"| {i} | {mode} | {count:,} | {pct} |")
    lines.append("")

    # ---------------------------------------------------------------
    # 4. Intervention recommendation
    # ---------------------------------------------------------------
    lines.append("## 4. Intervention Recommendation")
    lines.append("")

    top_mode = failure_modes[0][0] if failure_modes else ""
    if "pair_box" in top_mode:
        lines.extend([
            "**Primary lever: detection-level pair separation.**",
            "",
            f"pair_box accounts for {agg_pair:,} GT-person-frames ({agg_pair/total_frames:.1%}).",
            "These are frames where one detection covers two grappling people.",
            "No downstream fix (solver tuning, ReID, GROUP reform) can recover signal",
            "that was never separated at detection time. A model that outputs two",
            "boxes per grappling pair would convert pair_box -> tight_match and",
            "directly reduce wrong_id.",
            "",
            "**Secondary levers:**",
            "",
        ])
    else:
        lines.append("Primary lever determined by ranking above.")
        lines.append("")

    if agg_diag.get("d4_frame_trim", 0) > 0:
        trim_pct = agg_diag["d4_frame_trim"] / total_frames if total_frames else 0
        lines.extend([
            f"- **Graph coverage extension** ({agg_diag['d4_frame_trim']:,} frames, {trim_pct:.1%}):",
            "  d4_frame_trim is NOT solver rejection — tracklets are accepted but their",
            "  graph coverage ends before the annotated frame. Extending D1 graph lifetimes",
            "  or allowing longer SOLO tails could recover these frames.",
            "",
        ])

    if agg_wrong > 0:
        wrong_pct = agg_wrong / total_frames if total_frames else 0
        lines.extend([
            f"- **Identity collision reduction** ({agg_wrong:,} frames, {wrong_pct:.1%}):",
            "  wrong_id arises when pair_box tracklets carry the wrong person_id.",
            "  Reducing pair_box at the detector level also reduces wrong_id.",
            "",
        ])

    if agg_miss > 0:
        miss_pct = agg_miss / total_frames if total_frames else 0
        lines.extend([
            f"- **Detection recall** ({agg_miss:,} frames, {miss_pct:.1%}):",
            "  Missed detections. v2 model already improved recall +5pp; diminishing",
            "  returns expected from more training data of the same type.",
            "",
        ])

    lines.extend([
        "**Low priority:**",
        "- GROUP node redesign: falsified in CP-TRACE-2 — GROUP engagement on pair-box",
        "  tracklets is coincidental, not causal.",
        "- Solver penalty tuning: d3_solver_drop accounts for "
        f"{agg_diag.get('d3_solver_drop', 0)} frames.",
        "",
    ])

    # ---------------------------------------------------------------
    # 5. Worst frames
    # ---------------------------------------------------------------
    lines.append("## 5. Worst Frames (most lost signal)")
    lines.append("")

    # Find frames with most wrong_id + no_id
    all_traces = []
    for cam in cameras:
        d_path = base / cam / "gt_signal_trace_d.parquet"
        if d_path.exists():
            df = pd.read_parquet(d_path)
            df["camera_id"] = cam
            all_traces.append(df)

    if all_traces:
        combined = pd.concat(all_traces, ignore_index=True)
        lost = combined[combined.d_classification.isin(["wrong_id", "no_id"])]
        frame_loss = lost.groupby(["camera_id", "frame_index"]).size().reset_index(name="lost_count")
        worst = frame_loss.nlargest(10, "lost_count")

        lines.append("| Rank | Camera | Frame | Lost GT-persons | Breakdown |")
        lines.append("|---:|---|---:|---:|---|")
        for i, (_, row) in enumerate(worst.iterrows(), 1):
            cam = row.camera_id
            fi = int(row.frame_index)
            frame_data = combined[(combined.camera_id == cam) & (combined.frame_index == fi)]
            dc = frame_data.d_classification.value_counts().to_dict()
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(dc.items()))
            lines.append(f"| {i} | {cam} | {fi} | {int(row.lost_count)} | {breakdown} |")
        lines.append("")
    else:
        lines.append("No trace data available.")
        lines.append("")

    # Write
    out_path = base / "_verdict.md"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Verdict written to %s", out_path)
    return out_path
