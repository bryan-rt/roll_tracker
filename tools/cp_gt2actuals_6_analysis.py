#!/usr/bin/env python3
"""CP-GT2ACTUALS-6: Signal-shape + stage-attribution analysis.

Read-only consumer of the validated jump artifact. Produces findings for
docs/evidence/cp_gt2actuals_6/.

Usage:
    PYTHONPATH=src python tools/cp_gt2actuals_6_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
EVIDENCE_DIR = REPO / "docs" / "evidence" / "cp_gt2actuals_6"


def load_clip(label: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(f"{path}/gt2actuals_dense.parquet")


def load_split_classifications(path: str) -> list[dict]:
    meta = json.load(open(f"{path}/metadata.json"))
    return meta.get("d05_net_effect", {})


# ============================================================
# 1. Stage-attribution of through-line damage
# ============================================================

def analyze_stage_attribution(df: pd.DataFrame, label: str) -> dict:
    """Partition jump events by owning stage."""
    stage_map = {
        "tracklet_drift": "Stage A (tracker)",
        "false_split": "D0.5 (splitter)",
        "ilp_misstitch": "Solver (D3)",
        "group_boundary_jump": "Group handling (D1/solver)",
        "group_membership_drift": "Group handling (D1/solver)",
    }

    jumps = df[df["jump_type"].notna()]
    total_jumps = len(jumps)

    by_stage: dict[str, int] = Counter()
    by_type: dict[str, int] = Counter()
    for _, r in jumps.iterrows():
        jt = r["jump_type"]
        by_type[jt] += 1
        by_stage[stage_map.get(jt, "unknown")] += 1

    # How much damage is upstream of the solver?
    upstream = by_stage.get("Stage A (tracker)", 0) + by_stage.get("D0.5 (splitter)", 0)
    solver = by_stage.get("Solver (D3)", 0)
    group = by_stage.get("Group handling (D1/solver)", 0)

    return {
        "label": label,
        "total_jumps": total_jumps,
        "by_type": dict(by_type),
        "by_stage": dict(by_stage),
        "upstream_of_solver": upstream,
        "upstream_pct": upstream / total_jumps * 100 if total_jumps else 0,
        "solver_pct": solver / total_jumps * 100 if total_jumps else 0,
        "group_pct": group / total_jumps * 100 if total_jumps else 0,
    }


# ============================================================
# 2. Signal-shape distributions
# ============================================================

def analyze_signal_shapes(df: pd.DataFrame, split_events_path: str, label: str) -> dict:
    """Compare velocity, HSV, is_isolated across event classes."""
    hist_cols = [c for c in df.columns if c.startswith("hist_")]

    # --- Population definitions ---
    # Real swaps: tracklet_drift events (GT identity genuinely changed)
    real_swaps = df[df["jump_type"] == "tracklet_drift"]

    # Non-split calm: correct state, no jump
    calm = df[(df["jump_type"].isna()) & (df["state"] == "correct")]

    # False splits at split-event level: window frames around each split_frame
    # Load split events from the stage_D audit
    stage_d_dir = Path(split_events_path)
    audit_path = stage_d_dir / "d05_split_audit.jsonl"
    false_split_frames = set()
    correct_split_frames = set()
    if audit_path.exists():
        # Load the classifications from metadata
        meta_dir = Path(str(stage_d_dir).replace("_eval_gt", "_eval/gt2actuals")
                        .replace("/stage_D", ""))
        # Find the gt2actuals metadata
        from pipeline_validation.gt2actuals.jumps import load_split_events, classify_split_events
        sevts = load_split_events(stage_d_dir)
        split_map = defaultdict(list)
        for ev in sevts:
            split_map[ev["original_tracklet_id"]].append(ev["new_tracklet_id"])
        classifications = classify_split_events(sevts, df, dict(split_map))

        for ev, cl in zip(sevts, classifications):
            sf = ev.get("split_frame", -1)
            if cl["classification"] == "false_split":
                false_split_frames.add(sf)
            elif cl["classification"] == "correct_split":
                correct_split_frames.add(sf)

    # Window: frames within +-2 of split events
    def window_frames(frame_set, half_window=2):
        result = set()
        for f in frame_set:
            for df_offset in range(-half_window, half_window + 1):
                result.add(f + df_offset)
        return result

    fs_window = window_frames(false_split_frames)
    cs_window = window_frames(correct_split_frames)

    false_split_rows = df[df["frame_index"].isin(fs_window) & df["tracklet_id"].notna()]
    correct_split_rows = df[df["frame_index"].isin(cs_window) & df["tracklet_id"].notna()]

    def speed_stats(subset: pd.DataFrame) -> dict:
        speeds = subset["speed_mps_k"].dropna()
        if len(speeds) == 0:
            return {"n": 0}
        return {
            "n": len(speeds),
            "mean": float(speeds.mean()),
            "median": float(speeds.median()),
            "p75": float(speeds.quantile(0.75)),
            "p95": float(speeds.quantile(0.95)),
            "max": float(speeds.max()),
        }

    def isolation_rate(subset: pd.DataFrame) -> dict:
        iso = subset["is_isolated"]
        n = iso.notna().sum()
        if n == 0:
            return {"n": 0}
        return {
            "n": int(n),
            "isolated_pct": float((iso == True).sum() / n * 100),
        }

    def hist_distance_pre_post(subset: pd.DataFrame, split_frames: set) -> dict:
        """For frames near split boundaries, compute pre vs post histogram distance."""
        if not hist_cols or not split_frames:
            return {"n": 0}
        distances = []
        for sf in split_frames:
            pre = subset[(subset["frame_index"] < sf) & (subset["frame_index"] >= sf - 5)]
            post = subset[(subset["frame_index"] >= sf) & (subset["frame_index"] < sf + 5)]
            pre_hists = pre[hist_cols].dropna(how="all")
            post_hists = post[hist_cols].dropna(how="all")
            if len(pre_hists) == 0 or len(post_hists) == 0:
                continue
            pre_avg = pre_hists.mean().values.astype(np.float32)
            post_avg = post_hists.mean().values.astype(np.float32)
            # Bhattacharyya distance
            pre_n = pre_avg / (pre_avg.sum() + 1e-10)
            post_n = post_avg / (post_avg.sum() + 1e-10)
            bc = float(np.sum(np.sqrt(pre_n * post_n)))
            dist = float(-np.log(bc + 1e-10)) if bc < 1.0 else 0.0
            distances.append(dist)
        if not distances:
            return {"n": 0}
        return {
            "n": len(distances),
            "mean_bhatt_dist": float(np.mean(distances)),
            "median_bhatt_dist": float(np.median(distances)),
        }

    results = {
        "label": label,
        "populations": {
            "real_swaps": {"n_events": len(real_swaps)},
            "false_split_window": {"n_frames": len(false_split_rows), "n_split_events": len(false_split_frames)},
            "correct_split_window": {"n_frames": len(correct_split_rows), "n_split_events": len(correct_split_frames)},
            "calm": {"n_frames": len(calm)},
        },
        "speed": {
            "real_swaps": speed_stats(real_swaps),
            "false_split_window": speed_stats(false_split_rows),
            "correct_split_window": speed_stats(correct_split_rows),
            "calm": speed_stats(calm),
        },
        "isolation_rate": {
            "real_swaps": isolation_rate(real_swaps),
            "false_split_window": isolation_rate(false_split_rows),
            "correct_split_window": isolation_rate(correct_split_rows),
            "calm": isolation_rate(calm),
        },
        "hist_pre_post": {
            "false_split": hist_distance_pre_post(df, false_split_frames),
            "correct_split": hist_distance_pre_post(df, correct_split_frames),
        },
    }
    return results


# ============================================================
# 4. Appearance-in-solver: false-split sibling color agreement
# ============================================================

def analyze_appearance_in_solver(df: pd.DataFrame, stage_d_dir: str, label: str) -> dict:
    """Do false-split sibling fragments have agreeing color?"""
    from pipeline_validation.gt2actuals.jumps import load_split_events, classify_split_events

    stage_d = Path(stage_d_dir)
    sevts = load_split_events(stage_d)
    if not sevts:
        return {"label": label, "n_events": 0}

    split_map = defaultdict(list)
    for ev in sevts:
        split_map[ev["original_tracklet_id"]].append(ev["new_tracklet_id"])
    classifications = classify_split_events(sevts, df, dict(split_map))

    hist_cols = [c for c in df.columns if c.startswith("hist_")]

    # For each false split: check is_isolated rate at the boundary
    fs_iso_at_boundary = []
    fs_hist_available = 0
    fs_total = 0
    for ev, cl in zip(sevts, classifications):
        if cl["classification"] != "false_split":
            continue
        fs_total += 1
        sf = ev.get("split_frame", -1)
        # Check +-3 frames around split
        boundary = df[
            (df["frame_index"] >= sf - 3) &
            (df["frame_index"] <= sf + 3) &
            (df["tracklet_id"].notna())
        ]
        if len(boundary) == 0:
            continue
        iso_rate = (boundary["is_isolated"] == True).sum() / len(boundary)
        fs_iso_at_boundary.append(iso_rate)
        if hist_cols:
            has_hist = boundary[hist_cols[0]].notna().sum()
            if has_hist > 0:
                fs_hist_available += 1

    return {
        "label": label,
        "n_false_splits": fs_total,
        "n_with_boundary_frames": len(fs_iso_at_boundary),
        "mean_iso_rate_at_boundary": float(np.mean(fs_iso_at_boundary)) if fs_iso_at_boundary else None,
        "median_iso_rate_at_boundary": float(np.median(fs_iso_at_boundary)) if fs_iso_at_boundary else None,
        "n_with_histogram_at_boundary": fs_hist_available,
        "hist_available_pct": fs_hist_available / len(fs_iso_at_boundary) * 100 if fs_iso_at_boundary else 0,
    }


# ============================================================
# Main
# ============================================================

def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    clips = [
        ("vid2 (authoritative)", "outputs/_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200246",
         "outputs/_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200246/stage_D"),
        ("vid1 (corroboration)", "outputs/_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200015",
         "outputs/_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200015/stage_D"),
    ]

    all_results = {}

    for label, gt2a_path, stage_d_path in clips:
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print(f"{'='*60}")

        df = load_clip(label, gt2a_path)

        # 1. Stage attribution
        attr = analyze_stage_attribution(df, label)
        print(f"\n--- Stage Attribution ---")
        print(f"Total jumps: {attr['total_jumps']}")
        for stage, count in sorted(attr["by_stage"].items(), key=lambda x: -x[1]):
            print(f"  {stage}: {count} ({count/attr['total_jumps']:.0%})")
        print(f"Upstream of solver (Stage A + D0.5): {attr['upstream_pct']:.0%}")
        print(f"Solver (D3): {attr['solver_pct']:.0%}")
        print(f"Group handling: {attr['group_pct']:.0%}")

        # 2. Signal shapes
        sig = analyze_signal_shapes(df, stage_d_path, label)
        print(f"\n--- Signal Shapes ---")
        for pop in ["real_swaps", "false_split_window", "correct_split_window", "calm"]:
            sp = sig["speed"].get(pop, {})
            iso = sig["isolation_rate"].get(pop, {})
            print(f"  {pop}: speed median={sp.get('median','N/A'):.2f} p95={sp.get('p95','N/A'):.2f} | isolated={iso.get('isolated_pct','N/A'):.0f}%" if sp.get("n", 0) > 0 and iso.get("n", 0) > 0 else f"  {pop}: insufficient data")

        # Pre/post histogram distance
        for pop in ["false_split", "correct_split"]:
            hp = sig["hist_pre_post"].get(pop, {})
            if hp.get("n", 0) > 0:
                print(f"  {pop} pre/post Bhatt dist: mean={hp['mean_bhatt_dist']:.4f} median={hp['median_bhatt_dist']:.4f} (n={hp['n']})")

        # 4. Appearance in solver
        app = analyze_appearance_in_solver(df, stage_d_path, label)
        print(f"\n--- Appearance in Solver ---")
        print(f"  False splits: {app['n_false_splits']}")
        if app.get("mean_iso_rate_at_boundary") is not None:
            print(f"  Mean is_isolated at boundary: {app['mean_iso_rate_at_boundary']:.0%}")
            print(f"  Histogram available at boundary: {app['hist_available_pct']:.0f}%")

        all_results[label] = {
            "stage_attribution": attr,
            "signal_shapes": sig,
            "appearance_in_solver": app,
        }

    # Write results
    results_path = EVIDENCE_DIR / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
