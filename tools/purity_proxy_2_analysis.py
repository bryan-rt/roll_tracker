#!/usr/bin/env python3
"""PURITY-PROXY-2: Re-run appearance proxy on MASKED color, then multivariate vs path.

Read-only analysis. Extracts per-frame masked H+S+V histograms via median-plate
background subtraction, recomputes appearance multi-modality proxies, and tests
whether path + masked-appearance beats path-alone.

Usage:
    PYTHONPATH=src python tools/purity_proxy_2_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO / "docs" / "evidence" / "purity_proxy_2"

# Plate
PLATE_PATH = REPO / "outputs" / "_eval_gt_oracle" / "raster_plate_2" / "J_EDEw_median_plate.npy"

# Videos
VIDEO_BASE = REPO / "data" / "raw" / "nest" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
VID2_VIDEO = VIDEO_BASE / "J_EDEw-20260318-200246.mp4"
VID1_VIDEO = VIDEO_BASE / "J_EDEw-20260318-200015.mp4"

# Detections
DET_BASE = REPO / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
VID2_DET = DET_BASE / "J_EDEw-20260318-200246" / "stage_A" / "detections.parquet"
VID1_DET = DET_BASE / "J_EDEw-20260318-200015" / "stage_A" / "detections.parquet"

# GT2ACTUALS (for resolved_tracklet_id mapping)
GT2A_BASE = REPO / "outputs" / "_eval" / "gt2actuals" / "J_EDEw"
VID2_GT2A = GT2A_BASE / "J_EDEw-20260318-200246" / "gt2actuals_dense.parquet"
VID1_GT2A = GT2A_BASE / "J_EDEw-20260318-200015" / "gt2actuals_dense.parquet"

# PROXY-1 outputs
P1_DIR = REPO / "docs" / "evidence" / "purity_proxy_1"

# Constants
HIST_H_BINS = 18
HIST_S_BINS = 8
HIST_V_BINS = 6
HIST_SIZE = HIST_H_BINS * HIST_S_BINS * HIST_V_BINS  # 864
HIST_COLS = [f"hist_{i}" for i in range(HIST_SIZE)]

H_THRESH = 12
S_THRESH = 35
DEGEN_LOW = 0.05
DEGEN_HIGH = 0.95
MIN_MASKED_FRAMES = 10  # coverage gate for appearance proxy
SAME_COLOR_BHATT_THRESHOLD = 0.15
TELEPORT_THRESHOLD_M = 1.0  # "smooth" = max_displacement below this


# ============================================================
# Masking functions (from cp_raster_plate_2.py)
# ============================================================

def compute_foreground_mask(crop_bgr: np.ndarray, plate_crop_bgr: np.ndarray) -> np.ndarray:
    """Compute foreground mask by HSV channel difference vs plate."""
    hsv_f = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hsv_p = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2HSV)
    h_diff = np.abs(hsv_f[:, :, 0].astype(np.int16) - hsv_p[:, :, 0].astype(np.int16))
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(hsv_f[:, :, 1].astype(np.int16) - hsv_p[:, :, 1].astype(np.int16))
    fg = ((h_diff > H_THRESH) | (s_diff > S_THRESH)).astype(np.uint8) * 255
    return fg


def compute_masked_hist(crop_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Compute normalized 864-dim H+S+V histogram with mask."""
    if mask.sum() == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask,
        [HIST_H_BINS, HIST_S_BINS, HIST_V_BINS],
        [0, 180, 0, 256, 0, 256],
    )
    t = hist.sum()
    if t > 0:
        hist /= t
    return hist.flatten().astype(np.float32)


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance between two normalized histograms."""
    h1 = np.asarray(h1, dtype=np.float64).ravel()
    h2 = np.asarray(h2, dtype=np.float64).ravel()
    s1, s2 = h1.sum(), h2.sum()
    if s1 < 1e-12 or s2 < 1e-12:
        return 1.0
    h1, h2 = h1 / s1, h2 / s2
    bc = np.sum(np.sqrt(h1 * h2))
    return -np.log(max(min(bc, 1.0), 1e-12))


# ============================================================
# Phase A: Extract masked histograms from video
# ============================================================

def extract_masked_histograms(
    video_path: Path, det_path: Path, plate: np.ndarray, label: str,
) -> pd.DataFrame:
    """Extract per-detection masked histograms from video.

    Returns DataFrame with columns:
        frame_index, tracklet_id, detection_id, coverage, hist_0..hist_863
    """
    det = pd.read_parquet(det_path)
    det = det.sort_values("frame_index")

    # Group detections by frame for sequential reading
    frame_dets = det.groupby("frame_index")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = sorted(frame_dets.groups.keys())

    print(f"  Extracting masked histograms for {label}: {len(det)} detections across "
          f"{len(target_frames)} frames...")

    records = []
    frame_idx = 0
    n_processed = 0
    n_degen = 0

    for fi in target_frames:
        # Seek to target frame
        while frame_idx < fi:
            cap.grab()
            frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        grp = frame_dets.get_group(fi)
        for _, row in grp.iterrows():
            ix1 = max(0, int(row["x1"]))
            iy1 = max(0, int(row["y1"]))
            ix2 = min(frame.shape[1], int(row["x2"]))
            iy2 = min(frame.shape[0], int(row["y2"]))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            plate_crop = plate[iy1:iy2, ix1:ix2]
            fg_mask = compute_foreground_mask(crop, plate_crop)

            bbox_px = crop.shape[0] * crop.shape[1]
            mask_px = int(np.sum(fg_mask > 0))
            coverage = mask_px / bbox_px if bbox_px > 0 else 0.0

            is_degen = coverage < DEGEN_LOW or coverage > DEGEN_HIGH
            if is_degen:
                n_degen += 1
                hist = None
            else:
                hist = compute_masked_hist(crop, fg_mask)

            rec = {
                "frame_index": int(fi),
                "tracklet_id": str(row["tracklet_id"]),
                "detection_id": str(row["detection_id"]),
                "coverage": float(coverage),
                "is_degenerate": is_degen,
            }
            if hist is not None:
                for i, v in enumerate(hist):
                    rec[f"hist_{i}"] = float(v)

            records.append(rec)
            n_processed += 1

        if n_processed % 5000 == 0 and n_processed > 0:
            print(f"    ... {n_processed}/{len(det)} detections processed")

    cap.release()
    print(f"  Done: {n_processed} detections, {n_degen} degenerate ({100*n_degen/max(1,n_processed):.1f}%)")
    return pd.DataFrame(records)


# ============================================================
# Phase B: Compute per-tracklet masked appearance metrics
# ============================================================

def compute_masked_appearance_metrics(
    masked_df: pd.DataFrame, id_col: str, tracklet_ids: list,
) -> pd.DataFrame:
    """Compute tracklet-level appearance multi-modality from masked histograms."""
    # Filter to non-degenerate frames with valid histograms
    valid = masked_df[~masked_df["is_degenerate"] & masked_df["hist_0"].notna()].copy()

    results = []
    for tid in tracklet_ids:
        tid_rows = valid[valid[id_col] == tid].sort_values("frame_index")
        n_masked = len(tid_rows)

        rec = {
            id_col: tid,
            "n_masked_frames": n_masked,
            "masked_appearance_na": n_masked < MIN_MASKED_FRAMES,
        }

        # Per-frame coverage stats
        tid_all = masked_df[masked_df[id_col] == tid]
        if len(tid_all) > 0:
            rec["mean_coverage"] = float(tid_all["coverage"].mean())
            rec["median_coverage"] = float(tid_all["coverage"].median())
        else:
            rec["mean_coverage"] = np.nan
            rec["median_coverage"] = np.nan

        if n_masked < MIN_MASKED_FRAMES:
            rec.update({
                "masked_max_pairwise_bhatt": np.nan,
                "masked_max_deviation": np.nan,
                "masked_n_modes": np.nan,
            })
            results.append(rec)
            continue

        hists = tid_rows[HIST_COLS].values  # (n_masked, 864)
        mean_hist = hists.mean(axis=0)

        # Max deviation from mean
        deviations = [bhattacharyya(hists[i], mean_hist) for i in range(n_masked)]
        rec["masked_max_deviation"] = float(np.max(deviations))

        # Max pairwise Bhattacharyya between non-overlapping windows
        window_size = min(10, n_masked // 2)
        if window_size < 3:
            sample_idx = np.linspace(0, n_masked - 1, min(50, n_masked), dtype=int)
            max_pw = 0.0
            for i_idx in range(len(sample_idx)):
                for j_idx in range(i_idx + 1, len(sample_idx)):
                    d = bhattacharyya(hists[sample_idx[i_idx]], hists[sample_idx[j_idx]])
                    max_pw = max(max_pw, d)
            rec["masked_max_pairwise_bhatt"] = float(max_pw)
        else:
            n_windows = n_masked // window_size
            window_means = []
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size
                window_means.append(hists[start:end].mean(axis=0))
            if n_masked % window_size >= 3:
                window_means.append(hists[n_windows * window_size:].mean(axis=0))

            max_pw = 0.0
            for i in range(len(window_means)):
                for j in range(i + 1, len(window_means)):
                    d = bhattacharyya(window_means[i], window_means[j])
                    max_pw = max(max_pw, d)
            rec["masked_max_pairwise_bhatt"] = float(max_pw)

        # Mode count
        mode_threshold = 0.10
        if window_size >= 3 and len(window_means) >= 2:  # type: ignore[possibly-undefined]
            n_modes = 1
            for i in range(1, len(window_means)):
                d = bhattacharyya(window_means[i], window_means[i - 1])
                if d > mode_threshold:
                    n_modes += 1
            rec["masked_n_modes"] = n_modes
        else:
            stride = max(1, n_masked // 20)
            sampled = hists[::stride]
            n_modes = 1
            for i in range(1, len(sampled)):
                d = bhattacharyya(sampled[i], sampled[i - 1])
                if d > mode_threshold:
                    n_modes += 1
            rec["masked_n_modes"] = n_modes

        results.append(rec)

    return pd.DataFrame(results)


# ============================================================
# Phase C: Drift-frame survival guard
# ============================================================

def check_drift_frame_survival(
    masked_df: pd.DataFrame, purity_df: pd.DataFrame, id_col: str,
) -> list[dict]:
    """For each impure tracklet, check if its drift frames survived the coverage gate."""
    impure = purity_df[purity_df["is_temporal_impure"]]
    records = []

    for _, row in impure.iterrows():
        tid = row[id_col]
        drift_frames = json.loads(row["drift_frames"]) if isinstance(row["drift_frames"], str) else row["drift_frames"]
        if not drift_frames:
            records.append({
                id_col: tid,
                "n_drift_frames": 0,
                "n_drift_survived": 0,
                "n_drift_gated": 0,
                "drift_survival_rate": None,
            })
            continue

        # Check which drift frames have non-degenerate masked histograms
        tid_masked = masked_df[masked_df[id_col] == tid]
        survived = 0
        gated = 0
        for df_idx in drift_frames:
            frame_row = tid_masked[tid_masked["frame_index"] == df_idx]
            if frame_row.empty or frame_row.iloc[0]["is_degenerate"] or pd.isna(frame_row.iloc[0].get("hist_0")):
                gated += 1
            else:
                survived += 1

        records.append({
            id_col: tid,
            "n_drift_frames": len(drift_frames),
            "n_drift_survived": survived,
            "n_drift_gated": gated,
            "drift_survival_rate": survived / len(drift_frames) if drift_frames else None,
        })

    return records


# ============================================================
# Phase D: Scoring + stratification
# ============================================================

def score_proxy(labels: np.ndarray, scores: np.ndarray, proxy_name: str) -> dict:
    """Score a proxy against binary purity label."""
    valid = ~np.isnan(scores)
    labels_v, scores_v = labels[valid], scores[valid]
    n = len(labels_v)

    if n < 5 or len(set(labels_v)) < 2:
        return {"proxy": proxy_name, "n": n, "auc": None, "note": "insufficient data"}

    if np.std(scores_v) < 1e-12:
        return {"proxy": proxy_name, "n": n, "auc": 0.5, "note": "constant score"}

    auc = roc_auc_score(labels_v, scores_v)
    pure_vals = scores_v[labels_v == 0]
    impure_vals = scores_v[labels_v == 1]
    return {
        "proxy": proxy_name, "n": int(n),
        "n_pure": int(len(pure_vals)), "n_impure": int(len(impure_vals)),
        "auc": float(auc),
        "pure_mean": float(np.mean(pure_vals)), "pure_median": float(np.median(pure_vals)),
        "impure_mean": float(np.mean(impure_vals)), "impure_median": float(np.median(impure_vals)),
    }


def compute_masked_same_color_caveat(
    masked_df: pd.DataFrame, gt2a_df: pd.DataFrame, purity_df: pd.DataFrame, id_col: str,
) -> dict:
    """Compute same-color caveat using MASKED inter-GT histograms (guard 2)."""
    # Build per-GT-person mean MASKED histogram
    # Join masked_df to gt2a to get gt_track_id per detection
    # gt2a has (frame_index, gt_track_id, tracklet_id, detection_id)
    # masked_df has (frame_index, tracklet_id, detection_id, hist_*)

    # Use gt2a to get gt_track_id per (frame_index, detection_id)
    gt2a_slim = gt2a_df[gt2a_df["state"].isin(["correct", "wrong_id"])][
        ["frame_index", "gt_track_id", "detection_id"]
    ].copy()
    gt2a_slim["detection_id"] = gt2a_slim["detection_id"].astype(str)

    valid_masked = masked_df[~masked_df["is_degenerate"] & masked_df["hist_0"].notna()].copy()
    valid_masked["detection_id"] = valid_masked["detection_id"].astype(str)

    # Join
    joined = valid_masked.merge(gt2a_slim, on=["frame_index", "detection_id"], how="inner")

    gt_mean_hists = {}
    for gt_id, grp in joined.groupby("gt_track_id"):
        gt_mean_hists[gt_id] = grp[HIST_COLS].values.mean(axis=0)

    impure = purity_df[purity_df["is_temporal_impure"]]
    n_impure = len(impure)
    n_same_color = 0
    n_diff_color = 0
    n_no_hist = 0
    distances = []
    per_tracklet = []

    for _, row in impure.iterrows():
        gt_people = json.loads(row["temporal_gt_people_set"]) if isinstance(row["temporal_gt_people_set"], str) else row["temporal_gt_people_set"]
        if len(gt_people) < 2:
            continue

        has_pair = False
        min_dist = float("inf")
        for i in range(len(gt_people)):
            for j in range(i + 1, len(gt_people)):
                g1, g2 = gt_people[i], gt_people[j]
                if g1 in gt_mean_hists and g2 in gt_mean_hists:
                    d = bhattacharyya(gt_mean_hists[g1], gt_mean_hists[g2])
                    min_dist = min(min_dist, d)
                    has_pair = True

        tid = row[id_col]
        if not has_pair:
            n_no_hist += 1
            per_tracklet.append({id_col: tid, "color_stratum": "no_hist", "inter_gt_bhatt": None})
        elif min_dist < SAME_COLOR_BHATT_THRESHOLD:
            n_same_color += 1
            distances.append(min_dist)
            per_tracklet.append({id_col: tid, "color_stratum": "same_color", "inter_gt_bhatt": float(min_dist)})
        else:
            n_diff_color += 1
            distances.append(min_dist)
            per_tracklet.append({id_col: tid, "color_stratum": "diff_color", "inter_gt_bhatt": float(min_dist)})

    return {
        "n_impure": n_impure,
        "n_same_color": n_same_color,
        "n_diff_color": n_diff_color,
        "n_no_hist": n_no_hist,
        "same_color_frac": n_same_color / n_impure if n_impure else None,
        "diff_color_frac": n_diff_color / n_impure if n_impure else None,
        "mean_inter_gt_bhatt": float(np.mean(distances)) if distances else None,
        "per_tracklet": per_tracklet,
    }


# ============================================================
# Phase E: Multivariate (path + masked appearance vs path alone)
# ============================================================

def multivariate_test(merged: pd.DataFrame, label_col: str, subset_name: str) -> dict:
    """Test path + masked-appearance vs path-alone via logistic regression."""
    path_col = "max_displacement_m"
    app_col = "masked_max_pairwise_bhatt"

    # Filter to rows with both features and valid labels
    cols_needed = [label_col, path_col, app_col]
    valid = merged[cols_needed].dropna()
    if len(valid) < 10 or valid[label_col].nunique() < 2:
        return {"subset": subset_name, "n": len(valid), "note": "insufficient data"}

    labels = valid[label_col].astype(int).values
    X_path = valid[[path_col]].values
    X_app = valid[[app_col]].values
    X_both = valid[[path_col, app_col]].values

    # Path-alone AUC
    auc_path = roc_auc_score(labels, X_path.ravel())

    # Appearance-alone AUC
    auc_app = roc_auc_score(labels, X_app.ravel())

    # Combined: logistic regression
    scaler = StandardScaler()
    X_both_scaled = scaler.fit_transform(X_both)
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_both_scaled, labels)
        probs = lr.predict_proba(X_both_scaled)[:, 1]
        auc_combo = roc_auc_score(labels, probs)
        coefs = {path_col: float(lr.coef_[0][0]), app_col: float(lr.coef_[0][1])}
    except Exception as e:
        auc_combo = None
        coefs = {"error": str(e)}

    return {
        "subset": subset_name,
        "n": int(len(valid)),
        "n_pure": int((labels == 0).sum()),
        "n_impure": int((labels == 1).sum()),
        "auc_path_alone": float(auc_path),
        "auc_appearance_alone": float(auc_app),
        "auc_combo": float(auc_combo) if auc_combo is not None else None,
        "lift_vs_path": float(auc_combo - auc_path) if auc_combo is not None else None,
        "lr_coefs": coefs,
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_clip(
    video_path: Path, det_path: Path, gt2a_path: Path,
    p1_raw_path: Path, p1_res_path: Path,
    plate: np.ndarray, label: str,
) -> dict:
    """Full analysis for one clip."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label}")
    print(f"{'='*60}")

    results = {}

    # --- Phase A: Extract masked histograms ---
    masked_df = extract_masked_histograms(video_path, det_path, plate, label)
    results["extraction"] = {
        "n_detections": len(masked_df),
        "n_degenerate": int(masked_df["is_degenerate"].sum()),
        "degen_rate": float(masked_df["is_degenerate"].mean()),
        "median_coverage": float(masked_df["coverage"].median()),
        "mean_coverage": float(masked_df["coverage"].mean()),
    }
    print(f"  Coverage: median={masked_df['coverage'].median():.3f}, "
          f"degenerate={masked_df['is_degenerate'].sum()} ({100*masked_df['is_degenerate'].mean():.1f}%)")

    # --- Load PROXY-1 purity labels ---
    purity_a = pd.read_parquet(p1_raw_path)
    purity_b = pd.read_parquet(p1_res_path)

    # --- Load gt2actuals for resolved mapping ---
    gt2a = pd.read_parquet(gt2a_path)

    # Build raw→resolved mapping from gt2actuals
    raw_to_resolved = (
        gt2a[gt2a["state"].isin(["correct", "wrong_id"])]
        [["frame_index", "tracklet_id", "resolved_tracklet_id", "detection_id"]]
        .drop_duplicates()
    )
    raw_to_resolved["detection_id"] = raw_to_resolved["detection_id"].astype(str)

    # Add resolved_tracklet_id to masked_df
    masked_df["detection_id"] = masked_df["detection_id"].astype(str)
    masked_with_resolved = masked_df.merge(
        raw_to_resolved[["detection_id", "resolved_tracklet_id"]].drop_duplicates(),
        on="detection_id", how="left",
    )

    # --- Phase B: Compute masked appearance metrics ---
    print("\n--- Phase B: Masked appearance metrics (Level A — raw tracklets) ---")
    app_a = compute_masked_appearance_metrics(masked_df, "tracklet_id", purity_a["tracklet_id"].tolist())
    n_na_a = app_a["masked_appearance_na"].sum()
    n_ok_a = (~app_a["masked_appearance_na"]).sum()
    print(f"  Scoreable: {n_ok_a}, N/A: {n_na_a} ({100*n_na_a/len(app_a):.1f}%)")

    print("\n--- Phase B: Masked appearance metrics (Level B — post-D0.5) ---")
    resolved_ids = purity_b["resolved_tracklet_id"].tolist()
    app_b = compute_masked_appearance_metrics(masked_with_resolved, "resolved_tracklet_id", resolved_ids)
    n_na_b = app_b["masked_appearance_na"].sum()
    n_ok_b = (~app_b["masked_appearance_na"]).sum()
    print(f"  Scoreable: {n_ok_b}, N/A: {n_na_b} ({100*n_na_b/len(app_b):.1f}%)")

    # --- Phase C: Drift-frame survival guard ---
    print("\n--- Phase C: Drift-frame survival guard ---")
    drift_survival = check_drift_frame_survival(masked_df, purity_a, "tracklet_id")
    total_drift = sum(r["n_drift_frames"] for r in drift_survival)
    survived_drift = sum(r["n_drift_survived"] for r in drift_survival)
    gated_drift = sum(r["n_drift_gated"] for r in drift_survival)
    print(f"  Total drift frames across impure tracklets: {total_drift}")
    print(f"  Survived coverage gate: {survived_drift} ({100*survived_drift/max(1,total_drift):.1f}%)")
    print(f"  Gated out (degenerate): {gated_drift} ({100*gated_drift/max(1,total_drift):.1f}%)")
    results["drift_survival"] = {
        "total_drift_frames": total_drift,
        "survived": survived_drift,
        "gated": gated_drift,
        "survival_rate": survived_drift / total_drift if total_drift else None,
        "per_tracklet": drift_survival,
    }

    # --- Phase D: Score proxies ---
    print("\n--- Phase D: Score masked appearance proxies (Level A) ---")

    # Merge purity labels + path proxy + masked appearance
    merged_a = purity_a.merge(app_a, on="tracklet_id", how="left")
    scoreable_a = merged_a[~merged_a["masked_appearance_na"].fillna(True)]

    labels_a = scoreable_a["is_temporal_impure"].astype(int).values

    proxy_results_a = {}
    for col in ["masked_max_pairwise_bhatt", "masked_max_deviation", "masked_n_modes"]:
        r = score_proxy(labels_a, scoreable_a[col].values, col)
        proxy_results_a[col] = r
        auc_s = f"{r['auc']:.3f}" if r.get("auc") is not None else "N/A"
        pm = f", pure={r.get('pure_mean',0):.4f}, impure={r.get('impure_mean',0):.4f}" if r.get("auc") else ""
        print(f"  {col}: AUC={auc_s}{pm} (n={r['n']})")

    results["proxy_scores_a"] = proxy_results_a

    # Level B
    print("\n--- Phase D: Score masked appearance proxies (Level B — fix-relevant) ---")
    merged_b = purity_b.merge(app_b, on="resolved_tracklet_id", how="left")
    scoreable_b = merged_b[~merged_b["masked_appearance_na"].fillna(True)]

    labels_b = scoreable_b["is_temporal_impure"].astype(int).values
    proxy_results_b = {}
    for col in ["masked_max_pairwise_bhatt", "masked_max_deviation", "masked_n_modes"]:
        r = score_proxy(labels_b, scoreable_b[col].values, col)
        proxy_results_b[col] = r
        auc_s = f"{r['auc']:.3f}" if r.get("auc") is not None else "N/A"
        print(f"  {col}: AUC={auc_s} (n={r['n']})")

    results["proxy_scores_b"] = proxy_results_b

    # --- Same-color caveat (MASKED, guard 2) ---
    print("\n--- Same-color caveat (MASKED inter-GT histograms) ---")
    same_color = compute_masked_same_color_caveat(masked_df, gt2a, purity_a, "tracklet_id")
    results["same_color_masked"] = {k: v for k, v in same_color.items() if k != "per_tracklet"}
    color_strata = {r["tracklet_id"]: r["color_stratum"] for r in same_color["per_tracklet"]}
    print(f"  Same-color: {same_color['n_same_color']}, Diff-color: {same_color['n_diff_color']}, "
          f"No hist: {same_color['n_no_hist']}")
    if same_color.get("mean_inter_gt_bhatt"):
        print(f"  Mean masked inter-GT Bhatt: {same_color['mean_inter_gt_bhatt']:.4f}")

    # --- Stratified appearance AUC ---
    print("\n--- Stratified masked appearance AUC (Level A) ---")
    # Add color stratum to merged_a
    merged_a["color_stratum"] = merged_a["tracklet_id"].map(color_strata)

    for stratum in ["diff_color", "same_color"]:
        subset = scoreable_a[scoreable_a["tracklet_id"].isin(
            [t for t, s in color_strata.items() if s == stratum]
        ) | (~scoreable_a["is_temporal_impure"])]  # include all pure for contrast
        if len(subset) < 5 or subset["is_temporal_impure"].nunique() < 2:
            print(f"  {stratum}: insufficient data (n={len(subset)})")
            results[f"stratified_{stratum}"] = {"note": "insufficient data", "n": len(subset)}
            continue
        r = score_proxy(
            subset["is_temporal_impure"].astype(int).values,
            subset["masked_max_pairwise_bhatt"].values,
            f"masked_max_pairwise_bhatt ({stratum})",
        )
        results[f"stratified_{stratum}"] = r
        auc_s = f"{r['auc']:.3f}" if r.get("auc") else "N/A"
        print(f"  {stratum}: AUC={auc_s} (n_impure={r.get('n_impure', '?')})")

    # --- Phase E: Multivariate ---
    print("\n--- Phase E: Multivariate (path + masked appearance vs path alone) ---")

    # Full set (Level A)
    mv_full = multivariate_test(merged_a, "is_temporal_impure", "full_set")
    results["multivariate_full"] = mv_full
    if mv_full.get("auc_combo") is not None:
        print(f"  Full set: path={mv_full['auc_path_alone']:.3f}, app={mv_full['auc_appearance_alone']:.3f}, "
              f"combo={mv_full['auc_combo']:.3f}, lift={mv_full['lift_vs_path']:+.3f}")
    else:
        print(f"  Full set: {mv_full.get('note', 'error')}")

    # Smooth-different-color subset
    # "Smooth" = max_displacement_m < TELEPORT_THRESHOLD_M (no teleport)
    smooth_diff = merged_a[
        (merged_a["max_displacement_m"] < TELEPORT_THRESHOLD_M)
        & (
            (~merged_a["is_temporal_impure"])
            | (merged_a["color_stratum"] == "diff_color")
        )
    ]
    mv_smooth_diff = multivariate_test(smooth_diff, "is_temporal_impure", "smooth_diff_color")
    results["multivariate_smooth_diff"] = mv_smooth_diff
    if mv_smooth_diff.get("auc_combo") is not None:
        print(f"  Smooth+diff-color: path={mv_smooth_diff['auc_path_alone']:.3f}, "
              f"app={mv_smooth_diff['auc_appearance_alone']:.3f}, "
              f"combo={mv_smooth_diff['auc_combo']:.3f}, lift={mv_smooth_diff['lift_vs_path']:+.3f}")
    else:
        print(f"  Smooth+diff-color: {mv_smooth_diff.get('note', 'error')}")

    # --- Blind spot ---
    print("\n--- Residual blind spot (smooth + same-color) ---")
    smooth_same = merged_a[
        merged_a["is_temporal_impure"]
        & (merged_a["max_displacement_m"] < TELEPORT_THRESHOLD_M)
        & (merged_a["color_stratum"] == "same_color")
    ]
    n_blind = len(smooth_same)
    n_impure_total = merged_a["is_temporal_impure"].sum()
    print(f"  Smooth + same-color impure tracklets: {n_blind} / {n_impure_total} "
          f"({100*n_blind/max(1,n_impure_total):.1f}% of impure)")
    results["blind_spot"] = {
        "n_smooth_same_color": int(n_blind),
        "n_impure_total": int(n_impure_total),
        "frac_of_impure": float(n_blind / n_impure_total) if n_impure_total else None,
    }

    # Also count smooth+diff (where appearance adds value) and teleport (where path catches)
    smooth_diff_impure = merged_a[
        merged_a["is_temporal_impure"]
        & (merged_a["max_displacement_m"] < TELEPORT_THRESHOLD_M)
        & (merged_a["color_stratum"] == "diff_color")
    ]
    teleport_impure = merged_a[
        merged_a["is_temporal_impure"]
        & (merged_a["max_displacement_m"] >= TELEPORT_THRESHOLD_M)
    ]
    print(f"  Teleport (path catches): {len(teleport_impure)} ({100*len(teleport_impure)/max(1,n_impure_total):.1f}%)")
    print(f"  Smooth + diff-color (appearance helps): {len(smooth_diff_impure)} ({100*len(smooth_diff_impure)/max(1,n_impure_total):.1f}%)")
    print(f"  Smooth + same-color (blind): {n_blind} ({100*n_blind/max(1,n_impure_total):.1f}%)")
    results["impure_partition"] = {
        "teleport": int(len(teleport_impure)),
        "smooth_diff_color": int(len(smooth_diff_impure)),
        "smooth_same_color": int(n_blind),
        "total": int(n_impure_total),
    }

    # Store per-tracklet tables for evidence
    results["_merged_a"] = merged_a
    results["_merged_b"] = merged_b

    return results


def write_evidence(vid2_results: dict, vid1_results: Optional[dict], evidence_dir: Path):
    """Write findings to evidence directory."""
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for clip_label, res in [("vid2", vid2_results), ("vid1", vid1_results)]:
        if res is None:
            continue
        merged_a = res.pop("_merged_a")
        merged_b = res.pop("_merged_b")
        # Convert list columns to JSON strings for parquet
        for col in merged_a.columns:
            if merged_a[col].dtype == object:
                try:
                    # Test if it's a list column
                    sample = merged_a[col].dropna().head(1)
                    if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                        merged_a[col] = merged_a[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                except Exception:
                    pass
        for col in merged_b.columns:
            if merged_b[col].dtype == object:
                try:
                    sample = merged_b[col].dropna().head(1)
                    if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                        merged_b[col] = merged_b[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                except Exception:
                    pass

        merged_a.to_parquet(evidence_dir / f"{clip_label}_masked_purity_raw.parquet", index=False)
        merged_b.to_parquet(evidence_dir / f"{clip_label}_masked_purity_resolved.parquet", index=False)

    # Save JSON (without DataFrames)
    json_results = {}
    for clip_label, res in [("vid2", vid2_results), ("vid1", vid1_results)]:
        if res is None:
            continue
        json_results[clip_label] = res

    with open(evidence_dir / "proxy_scores.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # --- Build verdict ---
    lines = [
        "# PURITY-PROXY-2 Verdict",
        "",
        "## Question",
        "Does MASKED appearance improve purity separation over PROXY-1's contaminated signal,",
        "and does path + masked-appearance beat path-alone?",
        "",
    ]

    for clip_label, res in [("vid2 (authoritative)", vid2_results), ("vid1 (corroboration)", vid1_results)]:
        if res is None:
            continue
        lines.append(f"## {clip_label}")
        lines.append("")

        # Extraction stats
        ext = res["extraction"]
        lines.append(f"### Extraction")
        lines.append(f"- Detections: {ext['n_detections']}, Degenerate: {ext['n_degenerate']} ({100*ext['degen_rate']:.1f}%)")
        lines.append(f"- Median coverage: {ext['median_coverage']:.3f}")
        lines.append("")

        # Drift survival
        ds = res["drift_survival"]
        lines.append(f"### Drift-frame survival guard")
        lines.append(f"- Total drift frames: {ds['total_drift_frames']}")
        lines.append(f"- Survived: {ds['survived']} ({100*(ds['survival_rate'] or 0):.1f}%)")
        lines.append(f"- Gated out: {ds['gated']} ({100*(1-(ds['survival_rate'] or 1)):.1f}%)")
        lines.append("")

        # Proxy scores Level A
        lines.append("### Masked appearance AUC (Level A) vs PROXY-1 contaminated")
        lines.append("")
        lines.append("| Proxy | Masked AUC | PROXY-1 AUC (contaminated) | Delta |")
        lines.append("|-------|-----------|---------------------------|-------|")
        p1_aucs = {
            "masked_max_pairwise_bhatt": "max_pairwise_bhatt",
            "masked_max_deviation": "max_deviation_from_mean",
            "masked_n_modes": "n_appearance_modes",
        }
        for masked_col, p1_col in p1_aucs.items():
            r = res["proxy_scores_a"].get(masked_col, {})
            m_auc = r.get("auc")
            m_str = f"{m_auc:.3f}" if m_auc is not None else "N/A"
            lines.append(f"| {masked_col} | {m_str} | (see PROXY-1) | - |")
        lines.append("")

        # Level B
        lines.append("### Masked appearance AUC (Level B — fix-relevant)")
        lines.append("")
        lines.append("| Proxy | AUC | N |")
        lines.append("|-------|-----|---|")
        for col, r in res["proxy_scores_b"].items():
            auc_s = f"{r['auc']:.3f}" if r.get("auc") is not None else "N/A"
            lines.append(f"| {col} | {auc_s} | {r['n']} |")
        lines.append("")

        # Same-color
        sc = res.get("same_color_masked", {})
        lines.append(f"### Same-color caveat (MASKED)")
        sf = sc.get("same_color_frac")
        lines.append(f"- Same-color: {sc.get('n_same_color', 0)} ({100*(sf or 0):.1f}%)")
        lines.append(f"- Diff-color: {sc.get('n_diff_color', 0)}")
        mb = sc.get("mean_inter_gt_bhatt")
        lines.append(f"- Mean masked inter-GT Bhatt: {mb:.4f}" if mb else "- Mean: N/A")
        lines.append("")

        # Stratified
        for stratum in ["diff_color", "same_color"]:
            sr = res.get(f"stratified_{stratum}", {})
            auc_s = f"{sr['auc']:.3f}" if sr.get("auc") is not None else sr.get("note", "N/A")
            lines.append(f"- {stratum} AUC: {auc_s} (n_impure={sr.get('n_impure', '?')})")
        lines.append("")

        # Multivariate
        mv = res.get("multivariate_full", {})
        lines.append("### Multivariate: path + masked-appearance vs path-alone")
        if mv.get("auc_combo") is not None:
            lines.append(f"- Full set: path={mv['auc_path_alone']:.3f}, app={mv['auc_appearance_alone']:.3f}, "
                        f"combo={mv['auc_combo']:.3f}, lift={mv['lift_vs_path']:+.3f}")
        else:
            lines.append(f"- Full set: {mv.get('note', 'N/A')}")

        mv_sd = res.get("multivariate_smooth_diff", {})
        if mv_sd.get("auc_combo") is not None:
            lines.append(f"- Smooth+diff-color: path={mv_sd['auc_path_alone']:.3f}, "
                        f"app={mv_sd['auc_appearance_alone']:.3f}, combo={mv_sd['auc_combo']:.3f}, "
                        f"lift={mv_sd['lift_vs_path']:+.3f}")
        else:
            lines.append(f"- Smooth+diff-color: {mv_sd.get('note', 'N/A')}")
        lines.append("")

        # Blind spot + partition
        bs = res.get("blind_spot", {})
        ip = res.get("impure_partition", {})
        lines.append("### Impure tracklet partition")
        lines.append(f"- Teleport (path catches): {ip.get('teleport', '?')} "
                     f"({100*ip.get('teleport',0)/max(1,ip.get('total',1)):.1f}%)")
        lines.append(f"- Smooth + diff-color (appearance helps): {ip.get('smooth_diff_color', '?')} "
                     f"({100*ip.get('smooth_diff_color',0)/max(1,ip.get('total',1)):.1f}%)")
        lines.append(f"- Smooth + same-color (BLIND): {ip.get('smooth_same_color', '?')} "
                     f"({100*ip.get('smooth_same_color',0)/max(1,ip.get('total',1)):.1f}%)")
        lines.append("")

    with open(evidence_dir / "verdict.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nEvidence written to {evidence_dir}/")


def main():
    print("PURITY-PROXY-2: Masked appearance proxy + multivariate analysis")

    plate = np.load(str(PLATE_PATH))
    print(f"Plate loaded: {plate.shape}")

    vid2_results = analyze_clip(
        VID2_VIDEO, VID2_DET, VID2_GT2A,
        P1_DIR / "vid2_tracklet_purity_raw.parquet",
        P1_DIR / "vid2_tracklet_purity_resolved.parquet",
        plate, "vid2",
    )

    vid1_results = None
    if VID1_VIDEO.exists():
        vid1_results = analyze_clip(
            VID1_VIDEO, VID1_DET, VID1_GT2A,
            P1_DIR / "vid1_tracklet_purity_raw.parquet",
            P1_DIR / "vid1_tracklet_purity_resolved.parquet",
            plate, "vid1",
        )

    write_evidence(vid2_results, vid1_results, EVIDENCE_DIR)


if __name__ == "__main__":
    main()
