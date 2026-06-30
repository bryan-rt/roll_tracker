#!/usr/bin/env python3
"""Step 1: Passthrough test — run Stage D + GT2ACTUALS on unmodified baseline artifacts.

Tests hypothesis 3: is run_stage_d.py / run_gt2actuals.py faithful to the
original pipeline invocation? If the passthrough reproduces 34.7% (±0.5pp),
Stage D invocation is confirmed faithful.

Copies baseline stage_A/ and stage_C/ into a sweep run directory with NO
remapping, then runs the exact same subprocess invocation path as sweep_runner.py.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASELINE_BASE = REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
SWEEP_BASE = REPO_ROOT / "outputs" / "_sweep" / "runs"
DIAG_DIR = REPO_ROOT / "tools" / "sweep" / "diagnostics"

CLIPS = [
    "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246",
]

RUN_ID = "baseline_passthrough"

# Pre-existing baseline numbers from baseline.json
EXPECTED = {
    "J_EDEw-20260318-200015": 0.4032,
    "J_EDEw-20260318-200246": 0.3063,
    "combined": 0.3471,
}


def setup_passthrough(clip_id: str):
    """Copy baseline stage_A and stage_C into sweep run dir."""
    dest = SWEEP_BASE / RUN_ID / clip_id
    if dest.exists():
        shutil.rmtree(dest)

    src_a = BASELINE_BASE / clip_id / "stage_A"
    src_c = BASELINE_BASE / clip_id / "stage_C"

    shutil.copytree(src_a, dest / "stage_A")
    if src_c.exists():
        shutil.copytree(src_c, dest / "stage_C")
    else:
        (dest / "stage_C").mkdir(parents=True)

    print(f"  Copied baseline artifacts for {clip_id}")


def run_stage_d(clip_id: str):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "sweep" / "run_stage_d.py"),
        "--run-id", RUN_ID,
        "--clip-id", clip_id,
        "--camera-id", "J_EDEw",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"run_stage_d failed for {clip_id}")
    for line in result.stdout.strip().split("\n"):
        print(f"  {line}")


def run_gt2actuals(clip_id: str) -> dict:
    baseline_json = str(REPO_ROOT / "outputs" / "_sweep" / "baseline.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "sweep" / "run_gt2actuals.py"),
        "--run-id", RUN_ID,
        "--clip-id", clip_id,
        "--camera-id", "J_EDEw",
        "--baseline-json", baseline_json,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"run_gt2actuals failed for {clip_id}")

    stdout_lines = result.stdout.strip().split("\n")
    return json.loads(stdout_lines[-1])


def main():
    report_lines = []
    report_lines.append("# Step 1: Passthrough Test Results\n")
    report_lines.append("Tests hypothesis 3: is run_stage_d.py / run_gt2actuals.py faithful?\n")

    all_metrics = []

    for clip_id in CLIPS:
        print(f"\n{'='*60}")
        print(f"  {clip_id}")
        print(f"{'='*60}")

        print("  Setting up passthrough...")
        setup_passthrough(clip_id)

        print("  Running Stage D...")
        run_stage_d(clip_id)

        print("  Running GT2ACTUALS...")
        metrics = run_gt2actuals(clip_id)
        all_metrics.append(metrics)

        pct = metrics["pct_correct"] * 100
        expected = EXPECTED[clip_id] * 100
        delta = pct - expected
        status = "PASS" if abs(delta) <= 0.5 else "FAIL"

        print(f"\n  Result: {pct:.1f}% (expected {expected:.1f}%, delta {delta:+.1f}pp) [{status}]")
        print(f"  Jump counts: {metrics['jump_counts']}")

        report_lines.append(f"## {clip_id}\n")
        report_lines.append(f"- Passthrough correct_id: **{pct:.1f}%**")
        report_lines.append(f"- Pre-existing baseline:  **{expected:.1f}%**")
        report_lines.append(f"- Delta: **{delta:+.1f}pp**")
        report_lines.append(f"- Gate (±0.5pp): **{status}**")
        report_lines.append(f"- Jump counts: {metrics['jump_counts']}\n")

    # Combined
    total_correct = sum(m["n_correct"] for m in all_metrics)
    total_rows = sum(m["total_rows"] for m in all_metrics)
    combined_pct = total_correct / total_rows if total_rows > 0 else 0.0
    expected_combined = EXPECTED["combined"] * 100
    delta_combined = combined_pct * 100 - expected_combined
    status_combined = "PASS" if abs(delta_combined) <= 0.5 else "FAIL"

    print(f"\n{'='*60}")
    print(f"  COMBINED")
    print(f"{'='*60}")
    print(f"  Result: {combined_pct*100:.1f}% (expected {expected_combined:.1f}%, delta {delta_combined:+.1f}pp) [{status_combined}]")

    report_lines.append("## Combined\n")
    report_lines.append(f"- Passthrough correct_id: **{combined_pct*100:.1f}%**")
    report_lines.append(f"- Pre-existing baseline:  **{expected_combined:.1f}%**")
    report_lines.append(f"- Delta: **{delta_combined:+.1f}pp**")
    report_lines.append(f"- Gate (±0.5pp): **{status_combined}**\n")

    if status_combined == "PASS":
        report_lines.append("**Verdict: Hypothesis 3 RULED OUT.** Stage D invocation is faithful.")
        report_lines.append("Proceed to Step 2.\n")
    else:
        report_lines.append("**Verdict: Hypothesis 3 CONFIRMED.** Bug in run_stage_d.py or run_gt2actuals.py.")
        report_lines.append("STOP — diagnose before proceeding.\n")

    # Write report
    report_path = DIAG_DIR / "step1_results.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Report written: {report_path}")

    return status_combined == "PASS"


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
