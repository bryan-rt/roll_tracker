#!/usr/bin/env python3
"""Measure sweep run via GT2ACTUALS dense join.

Runs the gt2actuals dense join against sweep Stage D outputs, writing
results to the sweep directory (never to outputs/_eval/).

Designed to be invoked as a subprocess from sweep_runner.py so that
module-level overrides don't leak across sweep points.

Usage:
    python tools/sweep/run_gt2actuals.py --run-id baseline \
        --clip-id J_EDEw-20260318-200015 --baseline-json outputs/_sweep/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SWEEP_BASE = REPO_ROOT / "outputs" / "_sweep" / "runs"
MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"


def measure(run_id: str, clip_id: str, camera_id: str, baseline_path: str) -> dict:
    """Run gt2actuals against sweep outputs and compute metrics."""
    import pipeline_validation.gt2actuals.dense_join as dense_join
    from pipeline_validation.common.manifest import load_manifest

    sweep_clip_dir = SWEEP_BASE / run_id / clip_id

    # Verify Stage D outputs exist
    stage_d = sweep_clip_dir / "stage_D"
    for f in ["person_tracks.parquet", "d1_graph_nodes.parquet",
              "tracklet_bank_frames.parquet", "tracklet_bank_summaries.parquet"]:
        if not (stage_d / f).exists():
            raise FileNotFoundError(f"Missing {f} in {stage_d}. Run run_stage_d.py first.")

    # Override module-level path constants so _resolve_clip_dir and _output_dir
    # point at the sweep directory. This process exits after one measurement,
    # so the override's blast radius is this single invocation.
    orig_outputs_dir = dense_join.OUTPUTS_DIR
    orig_eval_dir = dense_join.EVAL_DIR
    try:
        # _resolve_clip_dir globs: OUTPUTS_DIR / f"{gym_id}/{cam}/**/{clip_id}"
        # We use a run_id-scoped gym_id so the glob matches ONLY this run's clip,
        # not other sweep runs' clips (glob collision bug discovered in SWEEP-3).
        # Directory: outputs/_sweep/_gt2a/<run_id>/<cam>/<clip_id>/ -> sweep clip
        import os
        gt2a_base = REPO_ROOT / "outputs" / "_sweep" / "_gt2a"
        gt2a_run_dir = gt2a_base / run_id / camera_id
        gt2a_run_dir.mkdir(parents=True, exist_ok=True)
        link_path = gt2a_run_dir / clip_id
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_target = os.path.relpath(sweep_clip_dir, link_path.parent)
        link_path.symlink_to(link_target)

        # OUTPUTS_DIR scoped to _gt2a base; gym_id = run_id so glob is unique
        dense_join.OUTPUTS_DIR = gt2a_base

        # _output_dir writes to EVAL_DIR / camera_id / clip_id
        sweep_gt2a_dir = sweep_clip_dir / "gt2actuals"
        dense_join.EVAL_DIR = sweep_gt2a_dir

        # gym_id = run_id ensures glob only matches this run
        gym_id = run_id
        manifest = load_manifest(MANIFEST_PATH)

        # Find the matching export entry for this clip
        target_export = None
        for export in manifest.training_data:
            exp_clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
            if exp_clip_id == clip_id and export.camera_id == camera_id:
                target_export = export
                break

        if target_export is None:
            raise ValueError(f"No manifest entry for clip_id={clip_id}, camera_id={camera_id}")

        resolved = dense_join._resolve_clip_dir(target_export, gym_id)
        if resolved is None:
            raise RuntimeError(
                f"Sweep clip dir not found by glob pattern. "
                f"Expected: {dense_join.OUTPUTS_DIR}/{gym_id}/{camera_id}/**/{clip_id} "
                f"Symlink at: {link_path} -> {link_target}"
            )

        result_path = dense_join._build_one_camera(
            manifest, target_export, gym_id,
            str(MANIFEST_PATH), iou_threshold=0.3,
        )

    finally:
        dense_join.OUTPUTS_DIR = orig_outputs_dir
        dense_join.EVAL_DIR = orig_eval_dir

    # Read results and compute metrics
    import pandas as pd
    df = pd.read_parquet(result_path)

    total = len(df)
    n_correct = int((df["state"] == "correct").sum())
    pct_correct = n_correct / total if total > 0 else 0.0

    state_counts = {k: int(v) for k, v in df["state"].value_counts().to_dict().items()}

    jump_types = ["tracklet_drift", "ilp_misstitch", "false_split",
                  "group_boundary_jump", "group_membership_drift"]
    jump_counts = {}
    for jt in jump_types:
        jump_counts[jt] = int((df["jump_type"] == jt).sum())

    # Load baseline for delta computation
    baseline = {}
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    baseline_clip = baseline.get("clips", {}).get(clip_id, {})
    baseline_pct = baseline_clip.get("pct_correct", 0.0)
    baseline_jumps = baseline_clip.get("jump_counts", {})

    delta_pct = round(pct_correct - baseline_pct, 4)
    ilp_misstitch_rose = jump_counts.get("ilp_misstitch", 0) > baseline_jumps.get("ilp_misstitch", 0)
    drift_rose = jump_counts.get("tracklet_drift", 0) > baseline_jumps.get("tracklet_drift", 0)

    metrics = {
        "clip_id": clip_id,
        "total_rows": total,
        "n_correct": n_correct,
        "pct_correct": round(pct_correct, 4),
        "delta_vs_baseline": delta_pct,
        "state_counts": state_counts,
        "jump_counts": jump_counts,
        "baseline_pct_correct": baseline_pct,
        "baseline_jump_counts": dict(baseline_jumps),
        "ilp_misstitch_rose": ilp_misstitch_rose,
        "tracklet_drift_rose": drift_rose,
        "solver_starvation_signal": ilp_misstitch_rose and drift_rose,
        "gt2actuals_parquet": str(result_path),
    }

    # Write metrics
    metrics_path = sweep_clip_dir / "gt2actuals_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print to stdout for subprocess capture
    print(json.dumps(metrics))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Measure sweep run via GT2ACTUALS")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", default="J_EDEw")
    parser.add_argument("--baseline-json", default="outputs/_sweep/baseline.json")
    args = parser.parse_args()

    measure(args.run_id, args.clip_id, args.camera_id, args.baseline_json)


if __name__ == "__main__":
    main()
