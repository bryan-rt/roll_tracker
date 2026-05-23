"""Report writer for tracker-swap diagnostic (CP-SWAP-1).

Produces per-camera artifacts (swap_events.jsonl, frame_features.parquet,
separability.json) and a cross-camera aggregate markdown report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_validation.tracker_swap.diagnostic import DiagnosticResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_BASE = REPO_ROOT / "outputs" / "_eval" / "tracker_swap"


def write_reports(model_id: str, results: list[DiagnosticResult]) -> Path:
    """Write all per-camera and aggregate reports. Returns aggregate path."""
    model_dir = OUTPUT_BASE / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        cam_dir = model_dir / res.camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        # swap_events.jsonl
        events_path = cam_dir / "swap_events.jsonl"
        with open(events_path, "w") as f:
            for ev in res.swap_events:
                f.write(json.dumps(ev.to_dict()) + "\n")
        logger.info("Wrote %d events to %s", len(res.swap_events), events_path)

        # frame_features.parquet
        features_path = cam_dir / "frame_features.parquet"
        if not res.frame_features.empty:
            res.frame_features.to_parquet(features_path, index=False)
        logger.info("Wrote %d rows to %s", len(res.frame_features), features_path)

        # separability.json
        sep_path = cam_dir / "separability.json"
        with open(sep_path, "w") as f:
            json.dump(res.separability, f, indent=2)
        logger.info("Wrote separability to %s", sep_path)

    # Aggregate markdown
    agg_path = model_dir / "_aggregate.md"
    _write_aggregate_md(agg_path, model_id, results)
    logger.info("Wrote aggregate report to %s", agg_path)
    return agg_path


def _write_aggregate_md(
    path: Path, model_id: str, results: list[DiagnosticResult]
) -> None:
    lines: list[str] = []
    lines.append(f"# Tracker-Swap Diagnostic: {model_id}\n")

    # --- Swap Event Summary ---
    lines.append("## Swap Event Summary\n")
    lines.append("| Camera | Tracklets | Tracklets w/ swaps | Total swaps | Swaps/tracklet (mean) |")
    lines.append("|--------|-----------|-------------------|-------------|----------------------|")
    for r in results:
        mean_swaps = (
            len(r.swap_events) / r.n_tracklets_with_swaps
            if r.n_tracklets_with_swaps > 0 else 0.0
        )
        lines.append(
            f"| {r.camera_id} | {r.n_tracklets} | {r.n_tracklets_with_swaps} "
            f"| {len(r.swap_events)} | {mean_swaps:.2f} |"
        )

    # --- Swap Context ---
    lines.append("\n## Swap Context\n")
    lines.append("| Camera | Mean IoU at swap | Mean gap (frames) | Median tracklet length (frames) |")
    lines.append("|--------|-----------------|-------------------|-------------------------------|")
    for r in results:
        if r.swap_events:
            mean_iou = np.mean(
                [(e.iou_before + e.iou_after) / 2 for e in r.swap_events]
            )
            mean_gap = np.mean([e.gap_frames for e in r.swap_events])
        else:
            mean_iou = 0.0
            mean_gap = 0.0
        if not r.frame_features.empty:
            tracklet_lengths = r.frame_features.groupby("tracklet_id").size()
            med_len = float(tracklet_lengths.median())
        else:
            med_len = 0.0
        lines.append(
            f"| {r.camera_id} | {mean_iou:.3f} | {mean_gap:.1f} | {med_len:.0f} |"
        )

    # --- Histogram coverage at swap boundaries ---
    lines.append("\n## Histogram Data Availability at Swap Boundaries\n")
    lines.append("(Expected: low coverage — confirms histograms are unavailable during grappling)\n")
    lines.append("| Camera | Swap-boundary frames | Frames with histogram | Coverage |")
    lines.append("|--------|---------------------|----------------------|----------|")
    for r in results:
        if not r.frame_features.empty and "histogram_available" in r.frame_features.columns:
            swap_mask = r.frame_features["is_swap_boundary"].astype(bool)
            n_swap = int(swap_mask.sum())
            n_hist = int(
                r.frame_features.loc[swap_mask, "histogram_available"].sum()
            )
            cov = f"{n_hist / n_swap:.1%}" if n_swap > 0 else "N/A"
        else:
            n_swap = 0
            n_hist = 0
            cov = "N/A"
        lines.append(f"| {r.camera_id} | {n_swap} | {n_hist} | {cov} |")

    # --- Feature Separability ---
    # Collect all features across cameras
    all_features: set[str] = set()
    for r in results:
        all_features.update(r.separability.keys())

    if all_features:
        # Compute mean AUC across cameras for ranking
        feature_aucs: dict[str, dict[str, float]] = {}
        for feat in sorted(all_features):
            feature_aucs[feat] = {}
            for r in results:
                if feat in r.separability:
                    feature_aucs[feat][r.camera_id] = r.separability[feat]["auc"]

        # Sort by mean AUC descending
        ranked = sorted(
            feature_aucs.keys(),
            key=lambda f: np.mean(list(feature_aucs[f].values())),
            reverse=True,
        )

        cam_ids = [r.camera_id for r in results]
        lines.append("\n## Feature Separability (ranked by AUC)\n")
        header = "| Feature | " + " | ".join(f"{c} AUC" for c in cam_ids) + " | Mean AUC | Cohen's d |"
        sep = "|" + "|".join(["--------"] * (len(cam_ids) + 3)) + "|"
        lines.append(header)
        lines.append(sep)

        best_mean_auc = 0.0
        for feat in ranked:
            auc_vals = []
            cells = []
            for cid in cam_ids:
                if cid in feature_aucs[feat]:
                    v = feature_aucs[feat][cid]
                    cells.append(f"{v:.3f}")
                    auc_vals.append(v)
                else:
                    cells.append("—")
            mean_auc = float(np.mean(auc_vals)) if auc_vals else 0.0
            best_mean_auc = max(best_mean_auc, mean_auc)
            # Cohen's d: average across cameras
            d_vals = []
            for r in results:
                if feat in r.separability:
                    d_vals.append(r.separability[feat]["cohens_d"])
            mean_d = float(np.mean(d_vals)) if d_vals else 0.0
            row = f"| {feat} | " + " | ".join(cells) + f" | {mean_auc:.3f} | {mean_d:+.3f} |"
            lines.append(row)

        # --- Verdict ---
        lines.append("\n## Verdict\n")
        if best_mean_auc > 0.7:
            lines.append(
                f"Best mean AUC = {best_mean_auc:.3f} (> 0.7). "
                "**GT-free swap detection appears viable.** "
                "A post-hoc tracklet splitter using the top-ranked features is worth pursuing."
            )
        elif best_mean_auc >= 0.6:
            lines.append(
                f"Best mean AUC = {best_mean_auc:.3f} (0.6–0.7 range). "
                "**Marginal separability.** A multi-feature detector might work "
                "but single-feature thresholding is likely insufficient."
            )
        else:
            lines.append(
                f"Best mean AUC = {best_mean_auc:.3f} (< 0.6). "
                "**GT-free swap detection is likely not feasible with current signals.** "
                "Additional signals (e.g., pose keypoints, deeper appearance features) "
                "may be needed."
            )
    else:
        lines.append("\n## Feature Separability\n")
        lines.append("No features had sufficient data for separability analysis.\n")
        lines.append("\n## Verdict\n")
        lines.append("Insufficient data for verdict.")

    # Note on IoU threshold
    lines.append(
        "\n---\n*GT assignment uses IoU >= 0.3 (not the frozen 0.5 threshold "
        "from the Layer 1/2 evaluation instrument). The lower threshold captures "
        "weaker overlaps at swap boundaries where tracker decisions are ambiguous.*"
    )

    path.write_text("\n".join(lines) + "\n")
