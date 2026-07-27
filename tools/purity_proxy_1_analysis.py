#!/usr/bin/env python3
"""PURITY-PROXY-1: Does any tracklet-aggregate signal separate pure from impure tracklets?

Read-only analysis. Loads gt2actuals_dense.parquet and scores candidate purity proxies
against GT-derived purity labels.

Usage:
    PYTHONPATH=src python tools/purity_proxy_1_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO / "docs" / "evidence" / "purity_proxy_1"

GT2A_BASE = REPO / "outputs" / "_eval" / "gt2actuals" / "J_EDEw"
VID2_PATH = GT2A_BASE / "J_EDEw-20260318-200246" / "gt2actuals_dense.parquet"
VID1_PATH = GT2A_BASE / "J_EDEw-20260318-200015" / "gt2actuals_dense.parquet"

HIST_COLS = [f"hist_{i}" for i in range(864)]

# Thresholds
MIN_ISOLATED_FOR_APPEARANCE = 10  # appearance proxy coverage gate
TELEPORT_THRESHOLD_M = 1.0  # per-frame displacement threshold (meters)
FPS = 15  # nominal frame rate for velocity context
SAME_COLOR_BHATT_THRESHOLD = 0.15  # below this = "same color"


# ============================================================
# Bhattacharyya distance
# ============================================================

def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance between two normalized histograms."""
    h1 = np.asarray(h1, dtype=np.float64).ravel()
    h2 = np.asarray(h2, dtype=np.float64).ravel()
    # Normalize
    s1, s2 = h1.sum(), h2.sum()
    if s1 < 1e-12 or s2 < 1e-12:
        return 1.0
    h1 = h1 / s1
    h2 = h2 / s2
    bc = np.sum(np.sqrt(h1 * h2))
    bc = min(bc, 1.0)
    return -np.log(max(bc, 1e-12))


# ============================================================
# Step 1: Derive GT purity labels
# ============================================================

def derive_purity_labels(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Derive per-tracklet GT purity label.

    Args:
        df: gt2actuals_dense filtered to assigned rows (correct + wrong_id)
        id_col: 'tracklet_id' (raw) or 'resolved_tracklet_id' (post-D0.5)

    Returns:
        DataFrame with one row per tracklet: id_col, n_gt_people, gt_people_set,
        is_pure, is_temporal_impure, is_spatial_only_impure, is_both,
        n_frames, n_pair_box_frames, n_unambiguous_frames, drift_frames.
    """
    records = []

    for tid, grp in df.groupby(id_col):
        if pd.isna(tid) or str(tid).lower() == "none":
            continue

        n_frames = grp["frame_index"].nunique()

        # Identify pair-box frames: same (tracklet, frame) with multiple GT people
        frame_gt = grp.groupby("frame_index")["gt_track_id"].apply(set)
        pair_box_frames = set()
        for fi, gt_set in frame_gt.items():
            if len(gt_set) > 1:
                pair_box_frames.add(fi)

        n_pair_box = len(pair_box_frames)
        n_unambiguous = n_frames - n_pair_box

        # All GT people this tracklet ever touches
        all_gt = set()
        for gt_set in frame_gt:
            all_gt.update(gt_set)

        # Temporal drift: on UNAMBIGUOUS frames only, does the sole GT person change?
        unambig_frames = sorted(fi for fi, gt_set in frame_gt.items() if len(gt_set) == 1)
        unambig_gt_by_frame = {}
        for fi in unambig_frames:
            unambig_gt_by_frame[fi] = list(frame_gt[fi])[0]

        temporal_gt_people = set(unambig_gt_by_frame.values())
        is_temporal_impure = len(temporal_gt_people) > 1

        # Drift points: frames where unambiguous GT identity changes
        drift_frames = []
        if len(unambig_frames) >= 2:
            prev_gt = unambig_gt_by_frame[unambig_frames[0]]
            for fi in unambig_frames[1:]:
                cur_gt = unambig_gt_by_frame[fi]
                if cur_gt != prev_gt:
                    drift_frames.append(fi)
                prev_gt = cur_gt

        # Classify
        has_spatial = n_pair_box > 0
        is_pure = len(all_gt) == 1
        is_both = is_temporal_impure and has_spatial
        is_spatial_only = has_spatial and not is_temporal_impure

        records.append({
            id_col: tid,
            "n_gt_people": len(all_gt),
            "n_temporal_gt_people": len(temporal_gt_people),
            "gt_people_set": sorted(all_gt),
            "temporal_gt_people_set": sorted(temporal_gt_people),
            "is_pure": is_pure,
            "is_temporal_impure": is_temporal_impure,
            "is_spatial_only_impure": is_spatial_only,
            "is_both": is_both,
            "n_frames": n_frames,
            "n_pair_box_frames": n_pair_box,
            "n_unambiguous_frames": n_unambiguous,
            "n_drift_points": len(drift_frames),
            "drift_frames": drift_frames,
        })

    return pd.DataFrame(records)


# ============================================================
# Step 2: Proxy 1 — Appearance multi-modality
# ============================================================

def compute_appearance_proxy(df: pd.DataFrame, id_col: str, purity_df: pd.DataFrame) -> pd.DataFrame:
    """Compute tracklet-level appearance multi-modality metrics.

    Uses only is_isolated=True frames. Marks tracklets with < MIN_ISOLATED_FOR_APPEARANCE
    isolated frames as 'appearance_na'.
    """
    # Filter to isolated frames with valid histograms
    iso = df[df["is_isolated"] == True].copy()
    iso_hist = iso[iso[HIST_COLS[0]].notna()]

    results = []
    for tid in purity_df[id_col]:
        tid_rows = iso_hist[iso_hist[id_col] == tid].sort_values("frame_index")
        n_isolated = len(tid_rows)

        rec = {
            id_col: tid,
            "n_isolated_frames": n_isolated,
            "appearance_na": n_isolated < MIN_ISOLATED_FOR_APPEARANCE,
        }

        if n_isolated < MIN_ISOLATED_FOR_APPEARANCE:
            rec.update({
                "max_pairwise_bhatt": np.nan,
                "max_deviation_from_mean": np.nan,
                "n_appearance_modes": np.nan,
            })
            results.append(rec)
            continue

        hists = tid_rows[HIST_COLS].values  # (n_isolated, 864)
        mean_hist = hists.mean(axis=0)

        # Max deviation of any frame from tracklet mean
        deviations = [bhattacharyya(hists[i], mean_hist) for i in range(n_isolated)]
        rec["max_deviation_from_mean"] = float(np.max(deviations))

        # Max pairwise Bhattacharyya between windows across the tracklet
        # Use non-overlapping windows of ~10 frames, compute mean hist per window,
        # then max pairwise distance between windows
        window_size = min(10, n_isolated // 2)
        if window_size < 3:
            # Too few frames for windowed analysis, use frame-level max pairwise
            # Sample up to 50 frames to keep computation tractable
            sample_idx = np.linspace(0, n_isolated - 1, min(50, n_isolated), dtype=int)
            max_pw = 0.0
            for i_idx in range(len(sample_idx)):
                for j_idx in range(i_idx + 1, len(sample_idx)):
                    d = bhattacharyya(hists[sample_idx[i_idx]], hists[sample_idx[j_idx]])
                    max_pw = max(max_pw, d)
            rec["max_pairwise_bhatt"] = float(max_pw)
        else:
            n_windows = n_isolated // window_size
            window_means = []
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size
                window_means.append(hists[start:end].mean(axis=0))
            # Also capture the tail
            if n_isolated % window_size >= 3:
                window_means.append(hists[n_windows * window_size:].mean(axis=0))

            max_pw = 0.0
            for i in range(len(window_means)):
                for j in range(i + 1, len(window_means)):
                    d = bhattacharyya(window_means[i], window_means[j])
                    max_pw = max(max_pw, d)
            rec["max_pairwise_bhatt"] = float(max_pw)

        # Mode count: simple change-point detection on time-ordered histograms
        # Use a rolling comparison: if Bhattacharyya between consecutive windows
        # exceeds a threshold, count as a mode boundary
        mode_threshold = 0.10  # empirical — Bhattacharyya between different people ~0.3+
        if window_size >= 3 and len(window_means) >= 2:  # type: ignore[possibly-undefined]
            n_modes = 1
            for i in range(1, len(window_means)):
                d = bhattacharyya(window_means[i], window_means[i - 1])
                if d > mode_threshold:
                    n_modes += 1
            rec["n_appearance_modes"] = n_modes
        else:
            # Fallback: frame-level mode detection with larger stride
            stride = max(1, n_isolated // 20)
            sampled = hists[::stride]
            n_modes = 1
            for i in range(1, len(sampled)):
                d = bhattacharyya(sampled[i], sampled[i - 1])
                if d > mode_threshold:
                    n_modes += 1
            rec["n_appearance_modes"] = n_modes

        results.append(rec)

    return pd.DataFrame(results)


# ============================================================
# Step 3: Proxy 2 — Path discontinuity (teleport detection)
# ============================================================

def compute_path_proxy(df: pd.DataFrame, id_col: str, purity_df: pd.DataFrame) -> pd.DataFrame:
    """Compute max single-frame displacement and teleport count per tracklet."""
    # Use only rows with valid world coords
    has_coords = df[df["x_m_eff"].notna()].copy()

    results = []
    for tid in purity_df[id_col]:
        tid_rows = has_coords[has_coords[id_col] == tid].sort_values("frame_index")

        rec = {id_col: tid}

        if len(tid_rows) < 2:
            rec.update({
                "max_displacement_m": np.nan,
                "n_teleports": 0,
                "mean_displacement_m": np.nan,
            })
            results.append(rec)
            continue

        # Deduplicate: if pair-box, same (tracklet, frame) appears multiple times
        # with same coords. Keep one row per (tracklet, frame).
        tid_rows = tid_rows.drop_duplicates(subset=[id_col, "frame_index"]).sort_values("frame_index")

        frames = tid_rows["frame_index"].values
        x = tid_rows["x_m_eff"].values
        y = tid_rows["y_m_eff"].values

        # Frame gaps for normalization (avoid false teleports on non-consecutive frames)
        frame_gaps = np.diff(frames).astype(float)
        frame_gaps[frame_gaps < 1] = 1  # safety

        dx = np.diff(x)
        dy = np.diff(y)
        raw_displacement = np.sqrt(dx**2 + dy**2)

        # Normalize: displacement per single frame (accounts for stride gaps)
        per_frame_displacement = raw_displacement / frame_gaps

        rec["max_displacement_m"] = float(np.max(per_frame_displacement))
        rec["n_teleports"] = int(np.sum(per_frame_displacement > TELEPORT_THRESHOLD_M))
        rec["mean_displacement_m"] = float(np.mean(per_frame_displacement))

        results.append(rec)

    return pd.DataFrame(results)


# ============================================================
# Step 4: Proxy 3 — Within-tracklet tag contradiction
# ============================================================

def compute_tag_proxy(df: pd.DataFrame, id_col: str, purity_df: pd.DataFrame) -> dict:
    """Check for within-tracklet tag contradictions."""
    tag_rows = df[df["has_tag_obs"] == True].copy()

    # Group by tracklet, collect distinct tag_ids
    tracklet_tags = tag_rows.groupby(id_col)["tag_id"].apply(lambda s: set(s.dropna()))

    contradictions = {tid: tags for tid, tags in tracklet_tags.items() if len(tags) > 1}

    # Precision: of tracklets with contradiction, what % truly impure?
    n_contradictions = len(contradictions)
    n_truly_impure = 0
    for tid in contradictions:
        row = purity_df[purity_df[id_col] == tid]
        if not row.empty and row.iloc[0]["is_temporal_impure"]:
            n_truly_impure += 1

    # Recall: of impure tracklets, what % have a tag contradiction?
    impure_tids = set(purity_df[purity_df["is_temporal_impure"]][id_col])
    n_impure_with_contradiction = len(set(contradictions.keys()) & impure_tids)

    return {
        "n_tracklets_with_tags": len(tracklet_tags),
        "n_contradictions": n_contradictions,
        "contradiction_tracklets": list(contradictions.keys()),
        "precision": n_truly_impure / n_contradictions if n_contradictions > 0 else None,
        "recall": n_impure_with_contradiction / len(impure_tids) if impure_tids else None,
        "n_impure_total": len(impure_tids),
    }


# ============================================================
# Step 5: Same-color caveat
# ============================================================

def compute_same_color_caveat(df: pd.DataFrame, purity_df: pd.DataFrame, id_col: str) -> dict:
    """For impure tracklets, check if the GT people they drift between have similar color."""
    iso = df[(df["is_isolated"] == True) & (df[HIST_COLS[0]].notna())]

    # Compute per-GT-person mean histogram
    gt_mean_hists = {}
    for gt_id, grp in iso.groupby("gt_track_id"):
        gt_mean_hists[gt_id] = grp[HIST_COLS].values.mean(axis=0)

    impure = purity_df[purity_df["is_temporal_impure"]]
    n_impure = len(impure)
    n_same_color = 0
    n_diff_color = 0
    n_no_hist = 0
    distances = []

    for _, row in impure.iterrows():
        gt_people = row["temporal_gt_people_set"]
        if len(gt_people) < 2:
            continue

        # Check all pairs of GT people in this tracklet
        has_pair = False
        min_dist = float("inf")
        for i in range(len(gt_people)):
            for j in range(i + 1, len(gt_people)):
                g1, g2 = gt_people[i], gt_people[j]
                if g1 in gt_mean_hists and g2 in gt_mean_hists:
                    d = bhattacharyya(gt_mean_hists[g1], gt_mean_hists[g2])
                    min_dist = min(min_dist, d)
                    has_pair = True

        if not has_pair:
            n_no_hist += 1
        elif min_dist < SAME_COLOR_BHATT_THRESHOLD:
            n_same_color += 1
            distances.append(min_dist)
        else:
            n_diff_color += 1
            distances.append(min_dist)

    return {
        "n_impure": n_impure,
        "n_same_color": n_same_color,
        "n_diff_color": n_diff_color,
        "n_no_hist": n_no_hist,
        "same_color_frac": n_same_color / n_impure if n_impure else None,
        "diff_color_frac": n_diff_color / n_impure if n_impure else None,
        "mean_inter_gt_bhatt": float(np.mean(distances)) if distances else None,
        "median_inter_gt_bhatt": float(np.median(distances)) if distances else None,
    }


# ============================================================
# Step 6: Score proxies against labels (AUC + distributions)
# ============================================================

def score_proxy(merged: pd.DataFrame, proxy_col: str, label_col: str = "is_temporal_impure") -> dict:
    """Score a single proxy column against the purity label."""
    valid = merged[[proxy_col, label_col]].dropna()
    if len(valid) < 5:
        return {"proxy": proxy_col, "n": len(valid), "auc": None, "note": "too few samples"}

    labels = valid[label_col].astype(int).values
    scores = valid[proxy_col].astype(float).values

    # Check for constant labels or scores
    if len(set(labels)) < 2:
        return {"proxy": proxy_col, "n": len(valid), "auc": None, "note": "single class"}
    if np.std(scores) < 1e-12:
        return {"proxy": proxy_col, "n": len(valid), "auc": 0.5, "note": "constant score"}

    auc = roc_auc_score(labels, scores)

    # Distribution stats
    pure_vals = scores[labels == 0]
    impure_vals = scores[labels == 1]

    return {
        "proxy": proxy_col,
        "n": len(valid),
        "n_pure": int(len(pure_vals)),
        "n_impure": int(len(impure_vals)),
        "auc": float(auc),
        "pure_mean": float(np.mean(pure_vals)),
        "pure_median": float(np.median(pure_vals)),
        "pure_std": float(np.std(pure_vals)),
        "pure_p95": float(np.percentile(pure_vals, 95)),
        "impure_mean": float(np.mean(impure_vals)),
        "impure_median": float(np.median(impure_vals)),
        "impure_std": float(np.std(impure_vals)),
        "impure_p95": float(np.percentile(impure_vals, 95)),
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_clip(parquet_path: Path, label: str) -> dict:
    """Run full purity proxy analysis on one clip."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label}: {parquet_path.name}")
    print(f"{'='*60}")

    df = pd.read_parquet(parquet_path)
    assigned = df[df["state"].isin(["correct", "wrong_id"])].copy()
    print(f"  Total rows: {len(df)}, assigned (correct+wrong_id): {len(assigned)}")

    results = {}

    # --- Level A: Raw tracklet purity ---
    print("\n--- Level A: Raw Stage A tracklet purity ---")
    purity_a = derive_purity_labels(assigned, "tracklet_id")
    n_pure_a = purity_a["is_pure"].sum()
    n_temp_a = purity_a["is_temporal_impure"].sum()
    n_spatial_a = purity_a["is_spatial_only_impure"].sum()
    n_both_a = purity_a["is_both"].sum()
    n_total_a = len(purity_a)
    print(f"  Tracklets: {n_total_a}")
    print(f"  Pure: {n_pure_a} ({100*n_pure_a/n_total_a:.1f}%)")
    print(f"  Temporal impure: {n_temp_a} ({100*n_temp_a/n_total_a:.1f}%)")
    print(f"  Spatial-only impure: {n_spatial_a} ({100*n_spatial_a/n_total_a:.1f}%)")
    print(f"  Both: {n_both_a} ({100*n_both_a/n_total_a:.1f}%)")

    # k-distribution for temporal impure
    k_dist = Counter(purity_a[purity_a["is_temporal_impure"]]["n_temporal_gt_people"].values)
    print(f"  k-distribution (temporal): {dict(sorted(k_dist.items()))}")

    results["level_a"] = {
        "id_col": "tracklet_id",
        "n_tracklets": n_total_a,
        "n_pure": int(n_pure_a),
        "n_temporal_impure": int(n_temp_a),
        "n_spatial_only_impure": int(n_spatial_a),
        "n_both": int(n_both_a),
        "k_distribution": {int(k): int(v) for k, v in sorted(k_dist.items())},
    }

    # --- Level B: Post-D0.5 product purity ---
    print("\n--- Level B: Post-D0.5 product purity (fix-relevant) ---")
    purity_b = derive_purity_labels(assigned, "resolved_tracklet_id")
    n_pure_b = purity_b["is_pure"].sum()
    n_temp_b = purity_b["is_temporal_impure"].sum()
    n_spatial_b = purity_b["is_spatial_only_impure"].sum()
    n_both_b = purity_b["is_both"].sum()
    n_total_b = len(purity_b)
    print(f"  Products: {n_total_b}")
    print(f"  Pure: {n_pure_b} ({100*n_pure_b/n_total_b:.1f}%)")
    print(f"  Temporal impure: {n_temp_b} ({100*n_temp_b/n_total_b:.1f}%)")
    print(f"  Spatial-only impure: {n_spatial_b} ({100*n_spatial_b/n_total_b:.1f}%)")
    print(f"  Both: {n_both_b} ({100*n_both_b/n_total_b:.1f}%)")

    k_dist_b = Counter(purity_b[purity_b["is_temporal_impure"]]["n_temporal_gt_people"].values)
    print(f"  k-distribution (temporal): {dict(sorted(k_dist_b.items()))}")

    results["level_b"] = {
        "id_col": "resolved_tracklet_id",
        "label": "FIX-RELEVANT (penalty charges post-D0.5 products)",
        "n_products": n_total_b,
        "n_pure": int(n_pure_b),
        "n_temporal_impure": int(n_temp_b),
        "n_spatial_only_impure": int(n_spatial_b),
        "n_both": int(n_both_b),
        "k_distribution": {int(k): int(v) for k, v in sorted(k_dist_b.items())},
    }

    # --- D0.5 effectiveness comparison ---
    print("\n--- D0.5 effectiveness: did splitting fix impure raw tracklets? ---")
    # For each impure raw tracklet, check if its split products are pure
    impure_raw = purity_a[purity_a["is_temporal_impure"]]
    n_fixed = 0
    n_still_impure = 0
    n_partially_fixed = 0
    for _, raw_row in impure_raw.iterrows():
        raw_tid = raw_row["tracklet_id"]
        # Find all resolved products that came from this raw tracklet
        products = assigned[assigned["tracklet_id"] == raw_tid]["resolved_tracklet_id"].unique()
        product_purity = purity_b[purity_b["resolved_tracklet_id"].isin(products)]
        if product_purity.empty:
            continue
        n_prod_impure = product_purity["is_temporal_impure"].sum()
        if n_prod_impure == 0:
            n_fixed += 1
        elif n_prod_impure < len(product_purity):
            n_partially_fixed += 1
        else:
            n_still_impure += 1

    print(f"  Impure raw tracklets: {len(impure_raw)}")
    print(f"  Fully fixed by D0.5: {n_fixed}")
    print(f"  Partially fixed: {n_partially_fixed}")
    print(f"  Still impure: {n_still_impure}")

    results["d05_effectiveness"] = {
        "n_impure_raw": len(impure_raw),
        "n_fully_fixed": n_fixed,
        "n_partially_fixed": n_partially_fixed,
        "n_still_impure": n_still_impure,
    }

    # ============================
    # Proxy scoring (on Level A — raw tracklets, where drift physically happens)
    # ============================
    print("\n--- Proxy 1: Appearance multi-modality (raw tracklets) ---")
    appearance = compute_appearance_proxy(assigned, "tracklet_id", purity_a)
    merged_a = purity_a.merge(appearance, on="tracklet_id", how="left")

    n_na = merged_a["appearance_na"].sum()
    n_scoreable = (~merged_a["appearance_na"]).sum()
    print(f"  Appearance N/A (< {MIN_ISOLATED_FOR_APPEARANCE} isolated frames): {n_na} ({100*n_na/len(merged_a):.1f}%)")
    print(f"  Scoreable: {n_scoreable} ({100*n_scoreable/len(merged_a):.1f}%)")

    scoreable = merged_a[~merged_a["appearance_na"]]
    for col in ["max_pairwise_bhatt", "max_deviation_from_mean", "n_appearance_modes"]:
        result = score_proxy(scoreable, col)
        results[f"proxy1_{col}"] = result
        auc_str = f"{result['auc']:.3f}" if result["auc"] is not None else "N/A"
        print(f"  {col}: AUC={auc_str}, pure_mean={result.get('pure_mean', 'N/A'):.4f}, "
              f"impure_mean={result.get('impure_mean', 'N/A'):.4f}" if result["auc"] is not None else
              f"  {col}: {result.get('note', 'N/A')}")

    results["proxy1_coverage"] = {
        "n_na": int(n_na),
        "n_scoreable": int(n_scoreable),
        "na_frac": float(n_na / len(merged_a)) if len(merged_a) else 0,
    }

    print("\n--- Proxy 2: Path discontinuity (raw tracklets) ---")
    path = compute_path_proxy(assigned, "tracklet_id", purity_a)
    merged_a = merged_a.merge(path, on="tracklet_id", how="left")

    for col in ["max_displacement_m", "n_teleports", "mean_displacement_m"]:
        result = score_proxy(merged_a, col)
        results[f"proxy2_{col}"] = result
        auc_str = f"{result['auc']:.3f}" if result["auc"] is not None else "N/A"
        print(f"  {col}: AUC={auc_str}, pure_mean={result.get('pure_mean', 'N/A'):.4f}, "
              f"impure_mean={result.get('impure_mean', 'N/A'):.4f}" if result["auc"] is not None else
              f"  {col}: {result.get('note', 'N/A')}")

    print("\n--- Proxy 3: Within-tracklet tag contradiction ---")
    tag_result = compute_tag_proxy(assigned, "tracklet_id", purity_a)
    results["proxy3_tag"] = tag_result
    print(f"  Tracklets with any tag obs: {tag_result['n_tracklets_with_tags']}")
    print(f"  Contradictions: {tag_result['n_contradictions']}")
    print(f"  Precision: {tag_result['precision']}")
    print(f"  Recall: {tag_result['recall']}")

    print("\n--- Proxy 4: Tracker-internal confidence ---")
    print("  local_track_conf is 100% NULL in BoT-SORT output.")
    print("  UNAVAILABLE: would require instrumenting tracker association step.")
    results["proxy4_tracker"] = {
        "status": "unavailable",
        "reason": "local_track_conf is 100% NULL; BoT-SORT does not expose per-association confidence",
        "action": "would require instrumenting tracker (separate future option)",
    }

    # --- Same-color caveat ---
    print("\n--- Same-color caveat ---")
    same_color = compute_same_color_caveat(assigned, purity_a, "tracklet_id")
    results["same_color_caveat"] = same_color
    print(f"  Impure tracklets: {same_color['n_impure']}")
    print(f"  Same-color (Bhatt < {SAME_COLOR_BHATT_THRESHOLD}): {same_color['n_same_color']} "
          f"({100*same_color['same_color_frac']:.1f}%)" if same_color['same_color_frac'] else "")
    print(f"  Different-color: {same_color['n_diff_color']} "
          f"({100*same_color['diff_color_frac']:.1f}%)" if same_color['diff_color_frac'] else "")
    print(f"  No histogram: {same_color['n_no_hist']}")
    print(f"  Mean inter-GT Bhattacharyya: {same_color['mean_inter_gt_bhatt']:.4f}" if same_color['mean_inter_gt_bhatt'] else "")

    # ============================
    # Also score proxies on Level B (post-D0.5, fix-relevant)
    # ============================
    print("\n--- Proxy scoring on Level B (post-D0.5 products, fix-relevant) ---")
    appearance_b = compute_appearance_proxy(assigned, "resolved_tracklet_id", purity_b)
    path_b = compute_path_proxy(assigned, "resolved_tracklet_id", purity_b)
    merged_b = purity_b.merge(appearance_b, on="resolved_tracklet_id", how="left")
    merged_b = merged_b.merge(path_b, on="resolved_tracklet_id", how="left")

    scoreable_b = merged_b[~merged_b["appearance_na"]]
    for col in ["max_pairwise_bhatt", "max_deviation_from_mean", "n_appearance_modes",
                "max_displacement_m", "n_teleports"]:
        result = score_proxy(scoreable_b if col.startswith("max_p") or col.startswith("n_app") else merged_b, col)
        results[f"level_b_{col}"] = result
        auc_str = f"{result['auc']:.3f}" if result["auc"] is not None else "N/A"
        print(f"  {col}: AUC={auc_str}" + (
            f", pure_mean={result.get('pure_mean', 0):.4f}, impure_mean={result.get('impure_mean', 0):.4f}"
            if result["auc"] is not None else f" ({result.get('note', '')})"))

    # Save per-tracklet table
    results["_merged_a"] = merged_a
    results["_merged_b"] = merged_b

    return results


def write_verdict(vid2_results: dict, vid1_results: Optional[dict], evidence_dir: Path):
    """Write the final verdict to evidence directory."""
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Save per-tracklet tables
    for label, res in [("vid2", vid2_results), ("vid1", vid1_results)]:
        if res is None:
            continue
        merged_a = res.pop("_merged_a")
        merged_b = res.pop("_merged_b")
        # Drop list columns for parquet compatibility
        for col in ["gt_people_set", "temporal_gt_people_set", "drift_frames"]:
            if col in merged_a.columns:
                merged_a[col] = merged_a[col].apply(json.dumps)
            if col in merged_b.columns:
                merged_b[col] = merged_b[col].apply(json.dumps)
        merged_a.to_parquet(evidence_dir / f"{label}_tracklet_purity_raw.parquet", index=False)
        merged_b.to_parquet(evidence_dir / f"{label}_tracklet_purity_resolved.parquet", index=False)

    # Save JSON results
    json_results = {}
    for label, res in [("vid2", vid2_results), ("vid1", vid1_results)]:
        if res is None:
            continue
        json_results[label] = res

    with open(evidence_dir / "proxy_scores.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # Build verdict text
    lines = [
        "# PURITY-PROXY-1 Verdict",
        "",
        "## Question",
        "Does any TRACKLET-AGGREGATE signal separate GT-pure from GT-impure tracklets?",
        "",
        "## Entity confirmation",
        "d3_ilp2's unexplained_tracklet_penalty charges against post-D0.5 products",
        "(base_tracklet_id from SINGLE_TRACKLET nodes). Level B is fix-relevant.",
        "",
    ]

    for clip_label, res in [("vid2 (authoritative)", vid2_results), ("vid1 (corroboration)", vid1_results)]:
        if res is None:
            continue
        lines.append(f"## {clip_label}")
        lines.append("")

        # Purity summary
        la = res["level_a"]
        lb = res["level_b"]
        lines.append(f"### Level A: Raw Stage A tracklets")
        lines.append(f"- Total: {la['n_tracklets']}, Pure: {la['n_pure']} ({100*la['n_pure']/la['n_tracklets']:.1f}%), "
                     f"Temporal impure: {la['n_temporal_impure']} ({100*la['n_temporal_impure']/la['n_tracklets']:.1f}%), "
                     f"Spatial-only: {la['n_spatial_only_impure']}, Both: {la['n_both']}")
        lines.append(f"- k-distribution: {la['k_distribution']}")
        lines.append("")

        lines.append(f"### Level B: Post-D0.5 products (FIX-RELEVANT)")
        lines.append(f"- Total: {lb['n_products']}, Pure: {lb['n_pure']} ({100*lb['n_pure']/lb['n_products']:.1f}%), "
                     f"Temporal impure: {lb['n_temporal_impure']} ({100*lb['n_temporal_impure']/lb['n_products']:.1f}%), "
                     f"Spatial-only: {lb['n_spatial_only_impure']}, Both: {lb['n_both']}")
        lines.append(f"- k-distribution: {lb['k_distribution']}")
        lines.append("")

        # D0.5 effectiveness
        d05 = res["d05_effectiveness"]
        lines.append(f"### D0.5 effectiveness")
        lines.append(f"- Impure raw tracklets: {d05['n_impure_raw']}")
        lines.append(f"- Fully fixed: {d05['n_fully_fixed']}, Partially: {d05['n_partially_fixed']}, "
                     f"Still impure: {d05['n_still_impure']}")
        lines.append("")

        # Proxy scores
        lines.append("### Proxy scores (Level A — raw tracklets)")
        lines.append("")
        lines.append("| Proxy | AUC | Pure mean | Impure mean | N |")
        lines.append("|-------|-----|-----------|-------------|---|")
        for key in sorted(res.keys()):
            if key.startswith("proxy1_") or key.startswith("proxy2_"):
                if key.endswith("_coverage"):
                    continue
                r = res[key]
                if r.get("auc") is not None:
                    lines.append(f"| {r['proxy']} | {r['auc']:.3f} | {r['pure_mean']:.4f} | "
                                f"{r['impure_mean']:.4f} | {r['n']} |")
                else:
                    lines.append(f"| {r['proxy']} | N/A | - | - | {r.get('n', 0)} |")

        lines.append("")

        # Appearance coverage
        cov = res.get("proxy1_coverage", {})
        lines.append(f"Appearance coverage: {cov.get('n_scoreable', '?')} scoreable, "
                     f"{cov.get('n_na', '?')} N/A ({100*cov.get('na_frac', 0):.1f}%)")
        lines.append("")

        # Level B proxy scores
        lines.append("### Proxy scores (Level B — post-D0.5, fix-relevant)")
        lines.append("")
        lines.append("| Proxy | AUC | Pure mean | Impure mean | N |")
        lines.append("|-------|-----|-----------|-------------|---|")
        for key in sorted(res.keys()):
            if key.startswith("level_b_"):
                r = res[key]
                if r.get("auc") is not None:
                    lines.append(f"| {r['proxy']} | {r['auc']:.3f} | {r['pure_mean']:.4f} | "
                                f"{r['impure_mean']:.4f} | {r['n']} |")
                else:
                    lines.append(f"| {r['proxy']} | N/A | - | - | {r.get('n', 0)} |")

        lines.append("")

        # Tag proxy
        tp = res.get("proxy3_tag", {})
        lines.append(f"### Proxy 3: Tag contradiction")
        lines.append(f"- Tracklets with tags: {tp.get('n_tracklets_with_tags', 0)}")
        lines.append(f"- Contradictions: {tp.get('n_contradictions', 0)}")
        lines.append(f"- Precision: {tp.get('precision')}, Recall: {tp.get('recall')}")
        lines.append("")

        # Proxy 4
        lines.append("### Proxy 4: Tracker-internal confidence")
        lines.append("- UNAVAILABLE: local_track_conf is 100% NULL in BoT-SORT output.")
        lines.append("- Would require instrumenting the tracker association step.")
        lines.append("")

        # Same-color caveat
        sc = res.get("same_color_caveat", {})
        lines.append(f"### Same-color caveat")
        lines.append(f"- Impure tracklets: {sc.get('n_impure', 0)}")
        sf = sc.get('same_color_frac')
        df_frac = sc.get('diff_color_frac')
        lines.append(f"- Same-color (undetectable by appearance): {sc.get('n_same_color', 0)} "
                     f"({100*sf:.1f}%)" if sf is not None else "- Same-color: N/A")
        lines.append(f"- Different-color (detectable): {sc.get('n_diff_color', 0)} "
                     f"({100*df_frac:.1f}%)" if df_frac is not None else "- Different-color: N/A")
        mb = sc.get('mean_inter_gt_bhatt')
        lines.append(f"- Mean inter-GT Bhattacharyya: {mb:.4f}" if mb is not None else "- Mean inter-GT Bhatt: N/A")
        lines.append("")

    with open(evidence_dir / "verdict.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nEvidence written to {evidence_dir}/")


def main():
    print("PURITY-PROXY-1: Tracklet-aggregate purity proxy analysis")
    print(f"  vid2: {VID2_PATH}")
    print(f"  vid1: {VID1_PATH}")

    # Vid2 (authoritative)
    vid2_results = analyze_clip(VID2_PATH, "vid2")

    # Vid1 (corroboration)
    vid1_results = None
    if VID1_PATH.exists():
        vid1_results = analyze_clip(VID1_PATH, "vid1")
    else:
        print(f"\nvid1 not found at {VID1_PATH}, skipping corroboration.")

    # Write evidence
    write_verdict(vid2_results, vid1_results, EVIDENCE_DIR)


if __name__ == "__main__":
    main()
