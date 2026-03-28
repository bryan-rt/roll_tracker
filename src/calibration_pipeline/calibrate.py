"""CP18 calibration orchestrator.

Runs the full calibration pipeline:
  1. Load mat blueprint
  2. For each camera: classify tracklets, run Layer 1 correction
  3. For each camera pair: run Layer 2 cross-camera alignment
  4. Write results + report

Can consume either:
  - Pre-computed D0 bank parquet files (from pipeline outputs)
  - Raw tracklet frame JSONL files (from calibration_test/output/)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import TrackletFeatures, classify_tracklets
from calibration_pipeline.mat_walk import (
    CalibrationResult,
    calibrate_single_camera,
    write_correction_json,
)
from calibration_pipeline.inter_camera_sync import (
    CrossCameraResult,
    align_camera_pair,
    write_alignment_json,
)


def run_calibration(
    camera_data: dict[str, pd.DataFrame],
    blueprint_path: Path,
    output_dir: Path,
    fps: float = 30.0,
) -> dict:
    """Full calibration pipeline.

    Parameters
    ----------
    camera_data : dict[str, DataFrame]
        camera_id → DataFrame with tracklet frame data. Must contain at minimum:
        tracklet_id, frame_index, x_m, y_m. Prefers x_m_repaired, y_m_repaired.
    blueprint_path : Path
        Path to mat_blueprint.json.
    output_dir : Path
        Where to write all results.
    fps : float
        Frames per second.

    Returns
    -------
    dict with keys: "layer1" (per-camera results), "layer2" (pairwise results),
    "summary" (report data).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load blueprint
    blueprint = MatBlueprint.from_json(blueprint_path)

    # 2. Layer 1: per-camera
    layer1_results: dict[str, CalibrationResult] = {}
    all_features: dict[str, list[TrackletFeatures]] = {}

    for camera_id, df in camera_data.items():
        print(f"[CP18] Layer 1 — {camera_id}: classifying {df.tracklet_id.nunique()} tracklets...")
        features = classify_tracklets(df, blueprint, fps=fps)
        all_features[camera_id] = features

        cleaning = [f for f in features if f.classification == "cleaning"]
        print(f"  {len(features)} tracklets, {len(cleaning)} cleaning-like")

        result = calibrate_single_camera(features, blueprint, camera_id)
        layer1_results[camera_id] = result

        # Write correction JSON
        path = write_correction_json(result, output_dir)

        # Write tracklet features JSONL
        _write_features_jsonl(features, output_dir, camera_id)

        print(f"  Confidence: {result.confidence}")
        if result.correction_matrix is not None:
            print(f"  Inside-mat: {result.inside_mat_fraction_before:.1%} → {result.inside_mat_fraction_after:.1%}")
            print(f"  Edge residual: {result.mean_edge_residual_before:.3f} → {result.mean_edge_residual_after:.3f}")
        else:
            print(f"  No correction applied: {result.details.get('reason', 'unknown')}")

    # 3. Layer 2: cross-camera
    layer2_results: dict[tuple[str, str], CrossCameraResult] = {}
    camera_ids = sorted(camera_data.keys())

    for cam_a, cam_b in combinations(camera_ids, 2):
        if cam_a not in all_features or cam_b not in all_features:
            continue
        if cam_a not in layer1_results or cam_b not in layer1_results:
            continue

        print(f"[CP18] Layer 2 — {cam_a} ↔ {cam_b}...")
        try:
            pair_result = align_camera_pair(
                all_features[cam_a], all_features[cam_b],
                layer1_results[cam_a], layer1_results[cam_b],
                blueprint,
            )
            layer2_results[(cam_a, cam_b)] = pair_result
            write_alignment_json(pair_result, output_dir)
            print(f"  Method: {pair_result.method}, confidence: {pair_result.confidence}, "
                  f"correspondences: {pair_result.n_correspondences}")
        except Exception as e:
            print(f"  Layer 2 failed (non-fatal): {e}")
            layer2_results[(cam_a, cam_b)] = CrossCameraResult(
                camera_pair=(cam_a, cam_b), method="none",
                pairwise_correction=None, confidence="none",
                details={"error": str(e)},
            )

    # 4. Write reports
    summary = _build_summary(layer1_results, layer2_results, blueprint)
    _write_report_md(summary, layer1_results, layer2_results, output_dir)
    _write_report_json(summary, layer1_results, layer2_results, output_dir)
    _write_before_after_json(layer1_results, output_dir)

    print(f"\n[CP18] Done. Results at: {output_dir}")
    return {
        "layer1": layer1_results,
        "layer2": layer2_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_summary(
    l1: dict[str, CalibrationResult],
    l2: dict[tuple[str, str], CrossCameraResult],
    blueprint: MatBlueprint,
) -> dict:
    """Build summary data for reports."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cameras": len(l1),
        "blueprint_area": blueprint.area,
        "blueprint_n_edges": blueprint.n_edges,
        "per_camera": {
            cam: {
                "confidence": r.confidence,
                "inside_before": round(r.inside_mat_fraction_before, 4),
                "inside_after": round(r.inside_mat_fraction_after, 4),
                "edge_residual_before": round(r.mean_edge_residual_before, 4),
                "edge_residual_after": round(r.mean_edge_residual_after, 4),
                "n_edge_touches": r.n_edge_touches,
                "n_cleaning": r.n_cleaning_tracklets,
                "coverage": round(r.coverage_fraction, 4),
            }
            for cam, r in l1.items()
        },
        "cross_camera": {
            f"{a}_{b}": {
                "method": r.method,
                "confidence": r.confidence,
                "n_correspondences": r.n_correspondences,
                "mean_residual": round(r.mean_residual, 4),
            }
            for (a, b), r in l2.items()
        },
    }


def _write_report_md(
    summary: dict,
    l1: dict[str, CalibrationResult],
    l2: dict[tuple[str, str], CrossCameraResult],
    output_dir: Path,
) -> None:
    """Write human-readable calibration report."""
    lines = ["# CP18 Calibration Report\n"]
    lines.append(f"**Generated:** {summary['timestamp']}")
    lines.append(f"**Cameras:** {summary['n_cameras']}")
    lines.append(f"**Blueprint area:** {summary['blueprint_area']:.1f} sq units, "
                 f"{summary['blueprint_n_edges']} boundary edges\n")

    lines.append("## Layer 1 — Per-Camera Corrections\n")
    lines.append("| Camera | Confidence | Inside-mat Before | Inside-mat After | Edge Residual Before | Edge Residual After | Edge Touches | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cam, r in sorted(l1.items()):
        lines.append(
            f"| {cam} | {r.confidence} | {r.inside_mat_fraction_before:.1%} | "
            f"{r.inside_mat_fraction_after:.1%} | {r.mean_edge_residual_before:.3f} | "
            f"{r.mean_edge_residual_after:.3f} | {r.n_edge_touches} | "
            f"{r.coverage_fraction:.1%} |"
        )
    lines.append("")

    for cam, r in sorted(l1.items()):
        lines.append(f"### {cam}")
        lines.append(f"- Cleaning tracklets: {r.n_cleaning_tracklets}")
        lines.append(f"- Distinct edges: {r.n_distinct_edges}")
        lines.append(f"- RANSAC inliers/outliers: {r.n_ransac_inliers}/{r.n_ransac_outliers}")
        lines.append(f"- Off-mat walkers: {r.n_off_mat_walkers}")
        if r.details:
            for k, v in r.details.items():
                if k != "affine_params":
                    lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Layer 2 — Cross-Camera Alignment\n")
    if l2:
        lines.append("| Pair | Method | Confidence | Correspondences | Mean Residual |")
        lines.append("|---|---|---|---|---|")
        for (a, b), r in sorted(l2.items()):
            lines.append(
                f"| {a} ↔ {b} | {r.method} | {r.confidence} | "
                f"{r.n_correspondences} | {r.mean_residual:.3f}m |"
            )
    else:
        lines.append("No camera pairs evaluated.")
    lines.append("")

    (output_dir / "calibration_report.md").write_text("\n".join(lines))


def _write_report_json(
    summary: dict,
    l1: dict[str, CalibrationResult],
    l2: dict[tuple[str, str], CrossCameraResult],
    output_dir: Path,
) -> None:
    """Write machine-readable calibration report."""
    with open(output_dir / "calibration_report.json", "w") as f:
        json.dump(summary, f, indent=2)


def _write_before_after_json(
    l1: dict[str, CalibrationResult], output_dir: Path
) -> None:
    """Write before/after comparison JSON."""
    data = {}
    for cam, r in l1.items():
        data[cam] = {
            "inside_mat_before": round(r.inside_mat_fraction_before, 4),
            "inside_mat_after": round(r.inside_mat_fraction_after, 4),
            "edge_residual_before": round(r.mean_edge_residual_before, 4),
            "edge_residual_after": round(r.mean_edge_residual_after, 4),
        }
    with open(output_dir / "before_after_comparison.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_features_jsonl(
    features: list[TrackletFeatures], output_dir: Path, camera_id: str
) -> None:
    """Write per-camera tracklet features JSONL."""
    cam_dir = output_dir / "per_camera" / camera_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    path = cam_dir / "tracklet_features.jsonl"

    with open(path, "w") as f:
        for feat in features:
            rec = {
                "tracklet_id": feat.tracklet_id,
                "camera_id": feat.camera_id,
                "clip_id": feat.clip_id,
                "classification": feat.classification,
                "spatial_extent_m2": feat.spatial_extent_m2,
                "total_distance_m": feat.total_distance_m,
                "duration_s": feat.duration_s,
                "avg_speed_mps": feat.avg_speed_mps,
                "stillness_fraction": feat.stillness_fraction,
                "birth_position": list(feat.birth_position),
                "death_position": list(feat.death_position),
                "birth_edge_distance": feat.birth_edge_distance,
                "death_edge_distance": feat.death_edge_distance,
                "birth_is_perpendicular": feat.birth_is_perpendicular,
                "death_is_perpendicular": feat.death_is_perpendicular,
                "n_edge_touches": feat.n_edge_touches,
                "on_mat_fraction": feat.on_mat_fraction,
            }
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_parquet_data(
    outputs_dir: Path, cameras: list[str]
) -> dict[str, pd.DataFrame]:
    """Load D0 tracklet_bank_frames.parquet from pipeline outputs.

    Searches for stage_D/tracklet_bank_frames.parquet under each camera's
    clip output directories. Concatenates across clips.
    """
    camera_data = {}
    for cam in cameras:
        frames = []
        for parquet in sorted(outputs_dir.rglob(
            f"{cam}/**/stage_D/tracklet_bank_frames.parquet"
        )):
            df = pd.read_parquet(parquet)
            frames.append(df)

        if frames:
            camera_data[cam] = pd.concat(frames, ignore_index=True)
            print(f"  Loaded {cam}: {len(camera_data[cam])} frames from {len(frames)} clips")
        else:
            print(f"  Warning: no D0 bank data found for {cam}")

    return camera_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CP18: Single-camera homography refinement + cross-camera alignment"
    )
    parser.add_argument(
        "--outputs-dir", type=Path, required=True,
        help="Pipeline outputs directory containing per-camera stage_D outputs",
    )
    parser.add_argument(
        "--blueprint", type=Path, default=Path("configs/mat_blueprint.json"),
        help="Path to mat_blueprint.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("calibration_results"),
        help="Where to write calibration results",
    )
    parser.add_argument(
        "--cameras", nargs="+", required=True,
        help="Camera IDs to calibrate (e.g., FP7oJQ J_EDEw PPDmUg)",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Frames per second",
    )
    args = parser.parse_args()

    print(f"[CP18] Loading data from {args.outputs_dir}...")
    camera_data = load_parquet_data(args.outputs_dir, args.cameras)

    if not camera_data:
        print("Error: no camera data loaded")
        sys.exit(1)

    run_calibration(camera_data, args.blueprint, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
