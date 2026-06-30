#!/usr/bin/env python3
"""Step 1: Freshen outputs/_eval_gt/ Stage D artifacts and re-measure.

Backs up stale D2-D4 artifacts, re-runs Stage D (D0->D4) using current code
and config, then runs the standard gt2actuals CLI to confirm ~30.7%.

This is the independent confirmation that the sweep's 30.7% is reproducible
via the standard production tooling, not just the sweep harness.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

EVAL_GT_BASE = REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
BACKUP_BASE = REPO_ROOT / "outputs" / "_eval_gt_stale_backup_20260630"

CLIPS = [
    "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246",
]

# Stale files to back up (D2-D4 from Jun 7, plus audit which spans both runs)
STALE_FILES = [
    "d2_constraints.json",
    "d2_edge_costs.parquet",
    "person_tracks.parquet",
    "identity_assignments.jsonl",
    "person_spans.parquet",
]


def backup_stale(clip_id: str):
    """Back up stale D2-D4 artifacts before overwriting."""
    src_dir = EVAL_GT_BASE / clip_id / "stage_D"
    dst_dir = BACKUP_BASE / clip_id / "stage_D"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fname in STALE_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            print(f"  Backed up: {fname}")

    # Also back up audit.jsonl (mixed provenance)
    audit = src_dir / "audit.jsonl"
    if audit.exists():
        shutil.copy2(audit, dst_dir / "audit.jsonl")
        print(f"  Backed up: audit.jsonl")


def rerun_stage_d(clip_id: str):
    """Re-run Stage D (D0->D4) on existing Stage A/C artifacts."""
    from bjj_pipeline.contracts.f0_manifest import ClipManifest, load_manifest, write_manifest
    from bjj_pipeline.contracts.f0_paths import ClipOutputLayout

    clip_dir = EVAL_GT_BASE / clip_id
    manifest_path = clip_dir / "clip_manifest.json"
    manifest = load_manifest(manifest_path)

    layout = ClipOutputLayout(
        clip_id=clip_id,
        root=EVAL_GT_BASE,
    )

    config_path = REPO_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault("stages", {}).setdefault("stage_D", {})["run_until"] = "D4"

    t0 = time.monotonic()
    from bjj_pipeline.stages.stitch.run import run as stage_d_run
    stage_d_run(config=config, inputs={"layout": layout, "manifest": manifest})
    wall = time.monotonic() - t0

    print(f"  Stage D done in {wall:.1f}s")


def run_gt2actuals():
    """Run the standard gt2actuals CLI against freshened outputs/_eval_gt/."""
    manifest_path = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"
    cmd = [
        sys.executable, "-m", "pipeline_validation", "gt2actuals",
        "--manifest-path", str(manifest_path),
        "--camera", "J_EDEw",
    ]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    print("  Running standard gt2actuals CLI...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[-500:]}")
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError("gt2actuals CLI failed")
    for line in result.stdout.strip().split("\n")[-10:]:
        print(f"    {line}")


def measure_freshened():
    """Read freshened gt2actuals parquets and report."""
    import pandas as pd

    gt2a_dir = REPO_ROOT / "outputs" / "_eval" / "gt2actuals" / "J_EDEw"
    results = []

    for clip_id in CLIPS:
        pq = gt2a_dir / clip_id / "gt2actuals_dense.parquet"
        if not pq.exists():
            print(f"  WARNING: {pq} not found")
            continue
        df = pd.read_parquet(pq, columns=["state", "jump_type"])
        n_correct = int((df["state"] == "correct").sum())
        total = len(df)
        pct = n_correct / total if total > 0 else 0.0
        results.append({"clip_id": clip_id, "n_correct": n_correct, "total": total, "pct": pct})
        print(f"  {clip_id}: {pct*100:.1f}% ({n_correct}/{total})")

    if len(results) == 2:
        combined_correct = sum(r["n_correct"] for r in results)
        combined_total = sum(r["total"] for r in results)
        combined_pct = combined_correct / combined_total
        print(f"  Combined: {combined_pct*100:.1f}% ({combined_correct}/{combined_total})")
        delta = abs(combined_pct - 0.307)
        status = "PASS" if delta <= 0.005 else "FAIL"
        print(f"  Gate (vs 30.7% ±0.5pp): {status} (delta={delta*100:.1f}pp)")
        return combined_pct, status

    return None, "FAIL"


def main():
    print("=" * 60)
    print("  Step 1: Freshen outputs/_eval_gt/ Stage D artifacts")
    print("=" * 60)

    # Back up stale artifacts
    print("\nBacking up stale D2-D4 artifacts...")
    for clip_id in CLIPS:
        print(f"\n  {clip_id}:")
        backup_stale(clip_id)
    print(f"\n  Backups at: {BACKUP_BASE}")

    # Re-run Stage D
    print("\nRe-running Stage D (D0->D4)...")
    for clip_id in CLIPS:
        print(f"\n  {clip_id}:")
        rerun_stage_d(clip_id)

    # Run standard gt2actuals CLI
    print("\nRunning standard gt2actuals CLI...")
    run_gt2actuals()

    # Measure
    print("\nFreshened results:")
    combined_pct, status = measure_freshened()

    print(f"\n{'=' * 60}")
    if status == "PASS":
        print("  CONFIRMED: Standard tooling reproduces ~30.7%")
        print("  The 34.7% pre-existing baseline was purely a staleness artifact.")
    else:
        print("  WARNING: Third number — does not match 30.7%")
        print("  Further investigation needed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
