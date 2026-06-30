#!/usr/bin/env python3
"""End-to-end sweep point orchestrator.

Ties replay_tracker -> run_stage_d -> run_gt2actuals into a single
invocation per sweep point. Appends results to outputs/_sweep/results.jsonl.

Usage:
    python tools/sweep/sweep_runner.py --run-id baseline \
        --clip-id J_EDEw-20260318-200015 --params '{}'
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP_BASE = REPO_ROOT / "outputs" / "_sweep"
RESULTS_JSONL = SWEEP_BASE / "results.jsonl"

CLIPS = [
    "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246",
]


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_sweep_point(
    run_id: str, clip_id: str, camera_id: str, params: dict
) -> dict:
    """Run one sweep point: replay -> Stage D -> GT2ACTUALS."""
    t0 = time.monotonic()
    python = sys.executable

    # Step 1: Replay tracker
    print(f"\n[sweep] Step 1/3: Replay tracker for {clip_id}")
    cmd_replay = [
        python, str(REPO_ROOT / "tools" / "sweep" / "replay_tracker.py"),
        "--clip-id", clip_id,
        "--camera-id", camera_id,
        "--params", json.dumps(params),
        "--run-id", run_id,
    ]
    result = subprocess.run(cmd_replay, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"replay_tracker failed: {result.stderr[-500:]}")

    # Read replay metadata for tag_hint_dropped
    replay_meta_path = SWEEP_BASE / "runs" / run_id / clip_id / "run_metadata.json"
    with open(replay_meta_path) as f:
        replay_meta = json.load(f)

    # Step 2: Run Stage D
    print(f"[sweep] Step 2/3: Stage D for {clip_id}")
    cmd_stage_d = [
        python, str(REPO_ROOT / "tools" / "sweep" / "run_stage_d.py"),
        "--run-id", run_id,
        "--clip-id", clip_id,
        "--camera-id", camera_id,
    ]
    result = subprocess.run(cmd_stage_d, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"run_stage_d failed: {result.stderr[-500:]}")
    # Print Stage D stdout (contains progress info)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")

    # Step 3: GT2ACTUALS measurement (subprocess for clean interpreter)
    print(f"[sweep] Step 3/3: GT2ACTUALS measurement for {clip_id}")
    baseline_json = str(SWEEP_BASE / "baseline.json")
    cmd_gt2a = [
        python, str(REPO_ROOT / "tools" / "sweep" / "run_gt2actuals.py"),
        "--run-id", run_id,
        "--clip-id", clip_id,
        "--camera-id", camera_id,
        "--baseline-json", baseline_json,
    ]
    result = subprocess.run(cmd_gt2a, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"run_gt2actuals failed: {result.stderr[-500:]}")

    # Parse metrics from stdout (last line is JSON)
    stdout_lines = result.stdout.strip().split("\n")
    metrics_line = stdout_lines[-1] if stdout_lines else "{}"
    try:
        metrics = json.loads(metrics_line)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse GT2ACTUALS output: {metrics_line[:200]}")
        metrics = {}

    wall_time = time.monotonic() - t0

    metrics["wall_time_seconds"] = round(wall_time, 1)
    metrics["tag_hint_dropped"] = replay_meta.get("tag_hint_dropped", False)
    metrics["n_tracklets"] = replay_meta.get("n_tracklets", 0)
    metrics["mean_tracklet_length"] = replay_meta.get("mean_tracklet_length", 0.0)
    metrics["short_tracklet_ratio_lt30"] = replay_meta.get("short_tracklet_ratio_lt30", 0.0)
    metrics["short_tracklet_ratio_lt10"] = replay_meta.get("short_tracklet_ratio_lt10", 0.0)

    return metrics


def print_summary(run_id: str, params: dict, clip_metrics: list[dict], combined: dict):
    """Print one-line summary with flags."""
    pct = combined["pct_correct"] * 100
    delta = combined["delta_vs_baseline"] * 100
    drift = combined["jump_counts"].get("tracklet_drift", 0)
    misstitch = combined["jump_counts"].get("ilp_misstitch", 0)
    bl_drift = combined["baseline_jump_counts"].get("tracklet_drift", 0)
    bl_misstitch = combined["baseline_jump_counts"].get("ilp_misstitch", 0)

    sign = "+" if delta >= 0 else ""
    line = (
        f"[{run_id}] correct_id: {combined['baseline_pct_correct']*100:.1f}% -> {pct:.1f}% "
        f"({sign}{delta:.1f}pp) | "
        f"drift: {bl_drift} -> {drift} ({drift - bl_drift:+d}) | "
        f"misstitch: {bl_misstitch} -> {misstitch} ({misstitch - bl_misstitch:+d})"
    )

    flags = []
    if combined.get("solver_starvation_signal"):
        flags.append("BOTH drift AND misstitch rose -- possible over-fragmentation / solver starvation")
    elif combined.get("ilp_misstitch_rose"):
        flags.append("misstitch_rose")

    # Check tag_hint_dropped across any clip
    if any(m.get("tag_hint_dropped") for m in clip_metrics):
        flags.append("tag_hint_dropped")

    if flags:
        line += " | " + " | ".join(f"WARNING {f}" for f in flags)

    print(f"\n{'='*80}")
    print(f"  {line}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end sweep point")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id", nargs="*", default=None,
                        help="Clip IDs (default: both J_EDEw clips)")
    parser.add_argument("--camera-id", default="J_EDEw")
    parser.add_argument("--params", default="{}", help="JSON string of tracker params")
    args = parser.parse_args()

    params = json.loads(args.params)
    clip_ids = args.clip_id or CLIPS

    all_metrics = []
    for clip_id in clip_ids:
        metrics = run_sweep_point(args.run_id, clip_id, args.camera_id, params)
        all_metrics.append(metrics)

    # Compute combined metrics
    total_correct = sum(m.get("n_correct", 0) for m in all_metrics)
    total_rows = sum(m.get("total_rows", 0) for m in all_metrics)
    combined_pct = total_correct / total_rows if total_rows > 0 else 0.0

    # Load baseline for combined comparison
    baseline_path = SWEEP_BASE / "baseline.json"
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    baseline_combined = baseline.get("combined", {})
    baseline_pct = baseline_combined.get("pct_correct", 0.0)
    baseline_jumps = baseline_combined.get("jump_counts", {})

    combined_jumps = {}
    for jt in ["tracklet_drift", "ilp_misstitch", "false_split",
               "group_boundary_jump", "group_membership_drift"]:
        combined_jumps[jt] = sum(m.get("jump_counts", {}).get(jt, 0) for m in all_metrics)

    ilp_rose = combined_jumps.get("ilp_misstitch", 0) > baseline_jumps.get("ilp_misstitch", 0)
    drift_rose = combined_jumps.get("tracklet_drift", 0) > baseline_jumps.get("tracklet_drift", 0)

    combined = {
        "pct_correct": round(combined_pct, 4),
        "delta_vs_baseline": round(combined_pct - baseline_pct, 4),
        "total_rows": total_rows,
        "n_correct": total_correct,
        "jump_counts": combined_jumps,
        "baseline_pct_correct": baseline_pct,
        "baseline_jump_counts": dict(baseline_jumps),
        "ilp_misstitch_rose": ilp_rose,
        "tracklet_drift_rose": drift_rose,
        "solver_starvation_signal": ilp_rose and drift_rose,
    }

    print_summary(args.run_id, params, all_metrics, combined)

    # Append to results.jsonl
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    result_record = {
        "run_id": args.run_id,
        "params": params,
        "git_sha": get_git_sha(),
        "clips": {m["clip_id"]: m for m in all_metrics},
        "combined": combined,
        "basis": {
            "instrument": "gt2actuals_dense",
            "denominator": "all_rows",
            "frame_range": "full_annotated_dense_stride1",
        },
    }

    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(result_record) + "\n")
    print(f"\n  Appended to {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
