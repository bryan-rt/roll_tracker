#!/usr/bin/env python3
"""Reproduce baseline correct_id from gt2actuals parquets.

Reads existing gt2actuals_dense.parquet for J_EDEw vid1+vid2 and computes:
- pct_correct (state == "correct" / total rows)
- jump counts by type
Writes outputs/_sweep/baseline.json with exact numbers + provenance stamps.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

GT2ACTUALS_DIR = REPO_ROOT / "outputs" / "_eval" / "gt2actuals" / "J_EDEw"

CLIPS = [
    "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246",
]

JUMP_TYPES = [
    "tracklet_drift",
    "ilp_misstitch",
    "false_split",
    "group_boundary_jump",
    "group_membership_drift",
]


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def compute_clip_stats(clip_id: str) -> dict:
    pq_path = GT2ACTUALS_DIR / clip_id / "gt2actuals_dense.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing: {pq_path}")

    df = pd.read_parquet(pq_path, columns=["state", "jump_type"])
    total = len(df)
    n_correct = int((df["state"] == "correct").sum())
    pct_correct = n_correct / total if total > 0 else 0.0

    state_counts = df["state"].value_counts().to_dict()
    state_counts = {k: int(v) for k, v in state_counts.items()}

    jump_counts = {}
    for jt in JUMP_TYPES:
        jump_counts[jt] = int((df["jump_type"] == jt).sum())

    return {
        "clip_id": clip_id,
        "parquet_path": str(pq_path),
        "parquet_mtime": os.path.getmtime(pq_path),
        "total_rows": total,
        "n_correct": n_correct,
        "pct_correct": round(pct_correct, 4),
        "state_counts": state_counts,
        "jump_counts": jump_counts,
    }


def main():
    results = {}
    combined_correct = 0
    combined_total = 0

    for clip_id in CLIPS:
        stats = compute_clip_stats(clip_id)
        results[clip_id] = stats
        combined_correct += stats["n_correct"]
        combined_total += stats["total_rows"]

        pct = stats["pct_correct"] * 100
        print(f"\n{'='*60}")
        print(f"  {clip_id}")
        print(f"{'='*60}")
        print(f"  Total rows:  {stats['total_rows']:,}")
        print(f"  Correct:     {stats['n_correct']:,} ({pct:.1f}%)")
        print(f"\n  State breakdown:")
        for state, count in sorted(stats["state_counts"].items(), key=lambda x: -x[1]):
            print(f"    {state:30s} {count:>7,} ({count/stats['total_rows']*100:5.1f}%)")
        print(f"\n  Jump counts:")
        for jt in JUMP_TYPES:
            count = stats["jump_counts"][jt]
            print(f"    {jt:30s} {count:>5}")

    combined_pct = combined_correct / combined_total if combined_total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"  COMBINED (vid1 + vid2)")
    print(f"{'='*60}")
    print(f"  Total rows:  {combined_total:,}")
    print(f"  Correct:     {combined_correct:,} ({combined_pct*100:.1f}%)")

    # Gate check
    if combined_pct < 0.10 or combined_pct > 0.50:
        print(f"\n  *** GATE FAILED: combined pct_correct={combined_pct:.3f} outside [0.10, 0.50] ***")
        exit(1)
    else:
        print(f"\n  Gate: PASS (within expected 10-50% range)")

    combined_jumps = {}
    for jt in JUMP_TYPES:
        combined_jumps[jt] = sum(results[c]["jump_counts"][jt] for c in CLIPS)
    print(f"\n  Combined jump counts:")
    for jt in JUMP_TYPES:
        print(f"    {jt:30s} {combined_jumps[jt]:>5}")

    output = {
        "git_sha": get_git_sha(),
        "clips": results,
        "combined": {
            "total_rows": combined_total,
            "n_correct": combined_correct,
            "pct_correct": round(combined_pct, 4),
            "jump_counts": combined_jumps,
        },
    }

    out_path = REPO_ROOT / "outputs" / "_sweep" / "baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Written: {out_path}")


if __name__ == "__main__":
    main()
