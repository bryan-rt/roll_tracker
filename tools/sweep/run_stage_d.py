#!/usr/bin/env python3
"""Re-run Stage D (D0->D4) on sweep tracklet artifacts.

Reads remapped Stage A artifacts from a sweep run directory and runs
the full Stage D pipeline (D0 -> D0.5 -> D1 -> D2 -> D3 -> D4).

Usage:
    python tools/sweep/run_stage_d.py --run-id baseline --clip-id J_EDEw-20260318-200015
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bjj_pipeline.contracts.f0_manifest import ClipManifest, write_manifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout

BASELINE_BASE = REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
SWEEP_BASE = REPO_ROOT / "outputs" / "_sweep" / "runs"


def run_stage_d(run_id: str, clip_id: str, camera_id: str):
    sweep_root = SWEEP_BASE / run_id
    clip_dir = sweep_root / clip_id

    # Verify Stage A artifacts exist
    stage_a = clip_dir / "stage_A"
    for f in ["tracklet_frames.parquet", "tracklet_summaries.parquet",
              "detections.parquet", "color_histograms.parquet"]:
        if not (stage_a / f).exists():
            raise FileNotFoundError(f"Missing {f} in {stage_a}. Run replay_tracker.py first.")

    # Load baseline manifest to get clip metadata
    baseline_manifest_path = BASELINE_BASE / clip_id / "clip_manifest.json"
    baseline_manifest = ClipManifest.model_validate_json(
        baseline_manifest_path.read_text(encoding="utf-8")
    )

    # Create a sweep manifest with same clip metadata but clean stage registry
    manifest = ClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        gym_id="_sweep",
        input_video_path=baseline_manifest.input_video_path,
        fps=baseline_manifest.fps,
        frame_count=baseline_manifest.frame_count,
        duration_ms=baseline_manifest.duration_ms,
        pipeline_version="sweep",
        created_at_ms=int(time.time() * 1000),
    )

    # Write manifest to sweep dir
    manifest_path = clip_dir / "clip_manifest.json"
    write_manifest(manifest, manifest_path)

    # Build layout pointing at sweep dir
    layout = ClipOutputLayout(clip_id=clip_id, root=sweep_root)

    # Load config
    config_path = REPO_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Ensure D4 run
    config.setdefault("stages", {}).setdefault("stage_D", {})["run_until"] = "D4"

    print(f"[sweep] Running Stage D for {clip_id} (run_id={run_id})")
    t0 = time.monotonic()

    from bjj_pipeline.stages.stitch.run import run as stage_d_run
    stage_d_run(config=config, inputs={"layout": layout, "manifest": manifest})

    wall_time = time.monotonic() - t0

    # Read solver status from audit
    solver_status = "unknown"
    audit_path = clip_dir / "stage_D" / "audit.jsonl"
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event_type") == "d3_solve_result":
                solver_status = rec.get("status", "unknown")

    # Count persons
    import pandas as pd
    pt_path = clip_dir / "stage_D" / "person_tracks.parquet"
    n_persons = 0
    if pt_path.exists():
        pt = pd.read_parquet(pt_path, columns=["person_id"])
        n_persons = pt["person_id"].nunique()

    metadata = {
        "run_id": run_id,
        "clip_id": clip_id,
        "wall_time_seconds": round(wall_time, 1),
        "solver_status": solver_status,
        "n_persons": n_persons,
    }

    meta_path = clip_dir / "run_stage_d_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[sweep] Stage D done: {n_persons} persons, solver={solver_status}, wall={wall_time:.1f}s")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Run Stage D on sweep artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", default="J_EDEw")
    args = parser.parse_args()

    run_stage_d(args.run_id, args.clip_id, args.camera_id)


if __name__ == "__main__":
    main()
