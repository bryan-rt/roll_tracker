"""Tracker-swap diagnostic (CP-SWAP-1).

Three-phase diagnostic that uses GT as an oracle to identify tracker swap
boundaries, measures GT-free pipeline signals at those boundaries, and
computes separability metrics to assess whether a GT-free splitter is viable.

IoU threshold for GT assignment is 0.3 (not the frozen 0.5 used by the
Layer 1/2 evaluation instrument). The lower threshold is intentional —
swap detection needs to see weaker overlaps where the tracker's decision
is ambiguous. This does NOT affect the frozen evaluation instrument.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline_validation.common.matching import iou_matrix

logger = logging.getLogger(__name__)

GT_ASSIGN_IOU_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SwapEvent:
    tracklet_id: str
    camera_id: str
    frame_before: int
    frame_after: int
    gt_person_before: int
    gt_person_after: int
    iou_before: float
    iou_after: float
    gap_frames: int

    def to_dict(self) -> dict:
        return {
            "tracklet_id": self.tracklet_id,
            "camera_id": self.camera_id,
            "frame_before": self.frame_before,
            "frame_after": self.frame_after,
            "gt_person_before": self.gt_person_before,
            "gt_person_after": self.gt_person_after,
            "iou_before": self.iou_before,
            "iou_after": self.iou_after,
            "gap_frames": self.gap_frames,
        }


@dataclass
class DiagnosticResult:
    camera_id: str
    swap_events: list[SwapEvent]
    frame_features: pd.DataFrame
    separability: dict
    n_tracklets: int = 0
    n_tracklets_with_swaps: int = 0


# ---------------------------------------------------------------------------
# Phase 1 — GT-oracle swap identification
# ---------------------------------------------------------------------------

def identify_swaps(
    detections_df: pd.DataFrame,
    gt_by_frame: dict[int, list],
    camera_id: str,
    swap_window: int = 2,
) -> tuple[list[SwapEvent], pd.DataFrame]:
    """Identify frames where a tracklet's detection switches GT person.

    Returns (swap_events, labels_df) where labels_df has columns:
        tracklet_id, frame_index, gt_person_id, is_swap_boundary
    """
    swap_events: list[SwapEvent] = []
    label_rows: list[dict] = []

    # Pre-build GT boxes array per frame for vectorized IoU
    gt_arrays: dict[int, tuple[np.ndarray, list[int]]] = {}
    for fidx, gt_boxes in gt_by_frame.items():
        if gt_boxes:
            boxes = np.array([[b.x1, b.y1, b.x2, b.y2] for b in gt_boxes])
            track_ids = [b.track_id for b in gt_boxes]
            gt_arrays[fidx] = (boxes, track_ids)

    grouped = detections_df.groupby("tracklet_id")

    for tid, grp in grouped:
        grp_sorted = grp.sort_values("frame_index")
        frames = grp_sorted["frame_index"].values
        bboxes = grp_sorted[["x1", "y1", "x2", "y2"]].values

        # Assign GT person at each frame
        assignments: list[tuple[int, int | None, float]] = []  # (frame, gt_id|None, iou)
        for i, fidx in enumerate(frames):
            fidx_int = int(fidx)
            if fidx_int not in gt_arrays:
                assignments.append((fidx_int, None, 0.0))
                continue
            gt_boxes_arr, gt_tids = gt_arrays[fidx_int]
            det_box = bboxes[i : i + 1]  # (1, 4)
            ious = iou_matrix(gt_boxes_arr, det_box)  # (N_gt, 1)
            best_idx = int(np.argmax(ious[:, 0]))
            best_iou = float(ious[best_idx, 0])
            if best_iou >= GT_ASSIGN_IOU_THRESHOLD:
                assignments.append((fidx_int, gt_tids[best_idx], best_iou))
            else:
                assignments.append((fidx_int, None, best_iou))

        # Find swap boundaries
        tracklet_swap_frames: set[int] = set()
        prev_valid = None  # (index_in_assignments, gt_id, iou)
        for j, (fidx_int, gt_id, iou) in enumerate(assignments):
            if gt_id is None:
                continue
            if prev_valid is not None and gt_id != prev_valid[1]:
                swap_events.append(SwapEvent(
                    tracklet_id=str(tid),
                    camera_id=camera_id,
                    frame_before=assignments[prev_valid[0]][0],
                    frame_after=fidx_int,
                    gt_person_before=prev_valid[1],
                    gt_person_after=gt_id,
                    iou_before=prev_valid[2],
                    iou_after=iou,
                    gap_frames=fidx_int - assignments[prev_valid[0]][0],
                ))
                tracklet_swap_frames.add(assignments[prev_valid[0]][0])
                tracklet_swap_frames.add(fidx_int)
            prev_valid = (j, gt_id, iou)

        # Expand swap boundary labels within ±swap_window
        frame_list = [a[0] for a in assignments]
        swap_boundary_frames: set[int] = set()
        for sf in tracklet_swap_frames:
            try:
                sf_pos = frame_list.index(sf)
            except ValueError:
                continue
            for offset in range(-swap_window, swap_window + 1):
                pos = sf_pos + offset
                if 0 <= pos < len(frame_list):
                    swap_boundary_frames.add(frame_list[pos])

        for fidx_int, gt_id, _iou in assignments:
            label_rows.append({
                "tracklet_id": str(tid),
                "frame_index": fidx_int,
                "gt_person_id": gt_id if gt_id is not None else -1,
                "is_swap_boundary": fidx_int in swap_boundary_frames,
            })

    labels_df = pd.DataFrame(label_rows)
    if labels_df.empty:
        labels_df = pd.DataFrame(columns=[
            "tracklet_id", "frame_index", "gt_person_id", "is_swap_boundary",
        ])

    return swap_events, labels_df


# ---------------------------------------------------------------------------
# Phase 2 — Signal measurement at every tracklet frame
# ---------------------------------------------------------------------------

def compute_frame_features(
    labels_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    bank_frames_df: pd.DataFrame | None = None,
    histograms_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute GT-free features at each tracklet frame.

    All features are derived purely from pipeline artifacts. GT is only used
    for the swap boundary labels (carried in labels_df).
    """
    if labels_df.empty:
        return labels_df.copy()

    # Merge detection bbox data
    det_cols = ["tracklet_id", "frame_index", "x1", "y1", "x2", "y2", "confidence"]
    available_cols = [c for c in det_cols if c in detections_df.columns]
    df = labels_df.merge(
        detections_df[available_cols],
        on=["tracklet_id", "frame_index"],
        how="left",
    )

    # Sort within each tracklet for consecutive-frame computations
    df = df.sort_values(["tracklet_id", "frame_index"]).reset_index(drop=True)

    # Bbox-derived features (computed per tracklet)
    cx = (df["x1"] + df["x2"]) / 2
    cy = (df["y1"] + df["y2"]) / 2
    w = df["x2"] - df["x1"]
    h = df["y2"] - df["y1"]
    area = w * h
    aspect = w / h.replace(0, np.nan)

    # Group-aware diffs (within same tracklet)
    g = df.groupby("tracklet_id", sort=False)
    df["bbox_center_dx_px"] = g.apply(
        lambda x: pd.Series(cx.loc[x.index]).diff(), include_groups=False
    ).reset_index(level=0, drop=True)
    df["bbox_center_dy_px"] = g.apply(
        lambda x: pd.Series(cy.loc[x.index]).diff(), include_groups=False
    ).reset_index(level=0, drop=True)
    df["bbox_center_speed_px"] = np.sqrt(
        df["bbox_center_dx_px"] ** 2 + df["bbox_center_dy_px"] ** 2
    )
    df["bbox_area_ratio"] = g.apply(
        lambda x: pd.Series(area.loc[x.index] / area.loc[x.index].shift(1)),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    df["bbox_aspect_change"] = g.apply(
        lambda x: pd.Series(aspect.loc[x.index]).diff().abs(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Consecutive bbox IoU (per tracklet)
    iou_vals = np.full(len(df), np.nan)
    for _tid, grp in g:
        idxs = grp.index.values
        for i in range(1, len(idxs)):
            prev_box = df.loc[idxs[i - 1], ["x1", "y1", "x2", "y2"]].values.reshape(1, 4)
            curr_box = df.loc[idxs[i], ["x1", "y1", "x2", "y2"]].values.reshape(1, 4)
            iou_val = iou_matrix(prev_box, curr_box)
            iou_vals[idxs[i]] = float(iou_val[0, 0])
    df["bbox_iou_consecutive"] = iou_vals

    # Bank frames features (world-coordinate kinematics)
    if bank_frames_df is not None and not bank_frames_df.empty:
        bank_cols = ["tracklet_id", "frame_index"]
        rename_map = {}
        if "speed_mps_k" in bank_frames_df.columns:
            bank_cols.append("speed_mps_k")
            rename_map["speed_mps_k"] = "world_speed_mps"
        if "accel_mps2_k" in bank_frames_df.columns:
            bank_cols.append("accel_mps2_k")
            rename_map["accel_mps2_k"] = "world_accel_mps2"
        for col in ["speed_is_implausible", "accel_is_implausible", "contact_conf"]:
            if col in bank_frames_df.columns:
                bank_cols.append(col)

        bank_subset = bank_frames_df[bank_cols].rename(columns=rename_map)
        df = df.merge(bank_subset, on=["tracklet_id", "frame_index"], how="left")
    else:
        logger.warning("tracklet_bank_frames.parquet not available — world features will be NaN")
        for col in ["world_speed_mps", "world_accel_mps2",
                     "speed_is_implausible", "accel_is_implausible", "contact_conf"]:
            df[col] = np.nan

    # Color histogram features
    if histograms_df is not None and not histograms_df.empty:
        hist_bin_cols = [c for c in histograms_df.columns if c.startswith("hist_")]
        # Mark histogram availability per (tracklet, frame)
        hist_keys = histograms_df[["track_id", "frame_index"]].copy()
        hist_keys = hist_keys.rename(columns={"track_id": "tracklet_id"})
        hist_keys["histogram_available"] = True
        df = df.merge(hist_keys, on=["tracklet_id", "frame_index"], how="left")
        df["histogram_available"] = df["histogram_available"].fillna(False)

        # Bhattacharyya distance between consecutive histograms
        if hist_bin_cols:
            hist_data = histograms_df[["track_id", "frame_index"] + hist_bin_cols].copy()
            hist_data = hist_data.rename(columns={"track_id": "tracklet_id"})
            hist_data = hist_data.sort_values(["tracklet_id", "frame_index"])

            bhatt_map: dict[tuple[str, int], float] = {}
            for tid_h, hgrp in hist_data.groupby("tracklet_id"):
                hvals = hgrp[hist_bin_cols].values  # (n_frames, 144)
                hframes = hgrp["frame_index"].values
                for i in range(1, len(hframes)):
                    h1 = hvals[i - 1]
                    h2 = hvals[i]
                    # Bhattacharyya distance: 1 - sum(sqrt(h1 * h2))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bc = float(np.sum(np.sqrt(np.maximum(h1 * h2, 0.0))))
                    bhatt_map[(str(tid_h), int(hframes[i]))] = 1.0 - bc

            if bhatt_map:
                df["histogram_bhattacharyya_vs_prev"] = df.apply(
                    lambda row: bhatt_map.get(
                        (row["tracklet_id"], int(row["frame_index"])), np.nan
                    ),
                    axis=1,
                )
            else:
                df["histogram_bhattacharyya_vs_prev"] = np.nan
        else:
            df["histogram_bhattacharyya_vs_prev"] = np.nan
    else:
        logger.warning("color_histograms.parquet not available — histogram features will be NaN")
        df["histogram_available"] = False
        df["histogram_bhattacharyya_vs_prev"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Phase 3 — Signal separability analysis
# ---------------------------------------------------------------------------

def _auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC. Uses sklearn if available, else Mann-Whitney U."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except ImportError:
        pass
    # Mann-Whitney U fallback (AUC = U / (n1 * n2))
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count how often a positive score exceeds a negative score
    u = 0.0
    for p in pos:
        u += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return u / (len(pos) * len(neg))


def compute_separability(frame_features: pd.DataFrame) -> dict:
    """Compute per-feature separability between swap-boundary and non-swap frames.

    Returns dict keyed by feature name -> metrics dict.
    """
    if frame_features.empty or "is_swap_boundary" not in frame_features.columns:
        return {}

    skip_cols = {
        "tracklet_id", "frame_index", "gt_person_id", "is_swap_boundary",
        "x1", "y1", "x2", "y2", "confidence", "histogram_available",
        "speed_is_implausible", "accel_is_implausible",
    }

    swap_mask = frame_features["is_swap_boundary"].astype(bool)
    results = {}

    for col in frame_features.columns:
        if col in skip_cols:
            continue
        if not pd.api.types.is_numeric_dtype(frame_features[col]):
            continue

        vals = frame_features[col].values.astype(float)
        valid = np.isfinite(vals)
        if valid.sum() < 10:
            continue

        swap_vals = vals[valid & swap_mask.values]
        nonswap_vals = vals[valid & ~swap_mask.values]

        if len(swap_vals) < 5 or len(nonswap_vals) < 5:
            continue

        # Medians and IQR
        swap_med = float(np.median(swap_vals))
        swap_q1, swap_q3 = float(np.percentile(swap_vals, 25)), float(np.percentile(swap_vals, 75))
        nonswap_med = float(np.median(nonswap_vals))
        nonswap_q1, nonswap_q3 = float(np.percentile(nonswap_vals, 25)), float(np.percentile(nonswap_vals, 75))

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(swap_vals) - 1) * np.var(swap_vals, ddof=1)
             + (len(nonswap_vals) - 1) * np.var(nonswap_vals, ddof=1))
            / (len(swap_vals) + len(nonswap_vals) - 2)
        )
        cohens_d = float((np.mean(swap_vals) - np.mean(nonswap_vals)) / pooled_std) if pooled_std > 0 else 0.0

        # AUC
        y_true = np.concatenate([np.ones(len(swap_vals)), np.zeros(len(nonswap_vals))])
        scores = np.concatenate([swap_vals, nonswap_vals])
        finite_mask = np.isfinite(scores)
        auc = _auc_score(y_true[finite_mask], scores[finite_mask])
        # Use max(auc, 1-auc) since feature direction is unknown
        auc = max(auc, 1.0 - auc)

        # Threshold fraction: swap frames exceeding 95th percentile of non-swap
        p95 = float(np.percentile(nonswap_vals, 95))
        threshold_frac = float(np.mean(swap_vals > p95)) if len(swap_vals) > 0 else 0.0

        results[col] = {
            "swap_median": swap_med,
            "swap_q1": swap_q1,
            "swap_q3": swap_q3,
            "nonswap_median": nonswap_med,
            "nonswap_q1": nonswap_q1,
            "nonswap_q3": nonswap_q3,
            "cohens_d": cohens_d,
            "auc": auc,
            "threshold_frac_above_p95": threshold_frac,
            "n_swap": int(len(swap_vals)),
            "n_nonswap": int(len(nonswap_vals)),
        }

    return results


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_diagnostic(
    camera_id: str,
    detections_df: pd.DataFrame,
    gt_by_frame: dict[int, list],
    bank_frames_df: pd.DataFrame | None = None,
    histograms_df: pd.DataFrame | None = None,
    swap_window: int = 2,
) -> DiagnosticResult:
    """Run all three phases of the tracker-swap diagnostic for one camera."""
    logger.info("Phase 1: identifying swaps for %s", camera_id)
    swap_events, labels_df = identify_swaps(
        detections_df, gt_by_frame, camera_id, swap_window=swap_window,
    )

    n_tracklets = detections_df["tracklet_id"].nunique()
    tracklets_with_swaps = len({e.tracklet_id for e in swap_events})
    logger.info(
        "%s: %d swap events across %d/%d tracklets",
        camera_id, len(swap_events), tracklets_with_swaps, n_tracklets,
    )

    logger.info("Phase 2: computing frame features for %s", camera_id)
    frame_features = compute_frame_features(
        labels_df, detections_df,
        bank_frames_df=bank_frames_df,
        histograms_df=histograms_df,
    )

    logger.info("Phase 3: computing separability for %s", camera_id)
    separability = compute_separability(frame_features)

    return DiagnosticResult(
        camera_id=camera_id,
        swap_events=swap_events,
        frame_features=frame_features,
        separability=separability,
        n_tracklets=n_tracklets,
        n_tracklets_with_swaps=tracklets_with_swaps,
    )
