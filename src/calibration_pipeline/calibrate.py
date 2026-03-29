"""CP18 calibration orchestrator.

Runs the full calibration pipeline with iterative refinement:
  1. Load mat blueprint + homography data
  2. Detect mat lines in video frames (if video provided)
  3. Iterative refinement loop:
     a. Layer 1: per-camera correction (mat lines + footpath)
     b. Layer 2: cross-camera fingerprint alignment
     c. Check convergence
  4. Write results + report

Can consume either:
  - Pre-computed D0 bank parquet files (from pipeline outputs)
  - Raw tracklet frame JSONL files (from calibration_test/output/)

CP18 v2: Adds video-based mat line detection, iterative refinement loop,
and spatial fingerprint registration.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import TrackletFeatures, classify_tracklets
from calibration_pipeline.mat_line_detection import detect_mat_lines, MatLineResult, save_diagnostic_image
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
    video_paths: dict[str, Path] | None = None,
    homography_dir: Path | None = None,
    max_iterations: int = 3,
    convergence_threshold: float = 0.001,
) -> dict:
    """Full calibration pipeline with iterative refinement.

    Parameters
    ----------
    camera_data : dict[str, DataFrame]
        camera_id → DataFrame with tracklet frame data.
    blueprint_path : Path
        Path to mat_blueprint.json.
    output_dir : Path
        Where to write all results.
    fps : float
        Frames per second.
    video_paths : dict[str, Path] | None
        camera_id → video file path for mat line detection.
    homography_dir : Path | None
        Directory containing per-camera homography.json files.
        Expected structure: homography_dir/camera_id/homography.json
        Defaults to configs/cameras/.
    max_iterations : int
        Maximum iterative refinement iterations (Layer 1 + Layer 2).
    convergence_threshold : float
        Stop iterating when inside-mat improvement < this threshold.

    Returns
    -------
    dict with keys: "layer1", "layer2", "summary".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if homography_dir is None:
        homography_dir = Path("configs/cameras")

    # 1. Load blueprint
    blueprint = MatBlueprint.from_json(blueprint_path)

    # 2. Classify tracklets for all cameras
    all_features: dict[str, list[TrackletFeatures]] = {}
    for camera_id, df in camera_data.items():
        print(f"[CP18] Classifying {camera_id}: {df.tracklet_id.nunique()} tracklets...")
        features = classify_tracklets(df, blueprint, fps=fps)
        all_features[camera_id] = features
        cleaning = [f for f in features if f.classification == "cleaning"]
        print(f"  {len(features)} tracklets, {len(cleaning)} cleaning-like")

    # 3. Run mat line detection per camera (if video + homography available)
    mat_line_results: dict[str, MatLineResult] = {}
    if video_paths:
        for camera_id, video_path in video_paths.items():
            if camera_id not in camera_data:
                continue
            homography_data = _load_homography(homography_dir, camera_id)
            if homography_data is None:
                print(f"  [CP18] {camera_id}: no homography.json — skipping mat line detection")
                continue

            print(f"[CP18] Mat line detection — {camera_id}...")
            df = camera_data[camera_id]
            mlr = detect_mat_lines(
                video_path=video_path,
                H=homography_data["H"],
                camera_matrix=homography_data.get("camera_matrix"),
                dist_coefficients=homography_data.get("dist_coefficients"),
                blueprint=blueprint,
                tracklet_frames_df=df,
            )
            mat_line_results[camera_id] = mlr
            print(f"  Frames: {mlr.n_frames_analyzed}, "
                  f"lines detected: {mlr.n_lines_detected}, "
                  f"matched: {mlr.n_lines_matched}")

            # Save diagnostic image
            diag_path = output_dir / "diagnostics" / f"{camera_id}_mat_lines.png"
            save_diagnostic_image(
                video_path,
                homography_data["H"],
                homography_data.get("camera_matrix"),
                homography_data.get("dist_coefficients"),
                blueprint,
                mlr,
                diag_path,
            )
            print(f"  Diagnostic image: {diag_path}")

    # 4. Iterative refinement loop
    layer1_results: dict[str, CalibrationResult] = {}
    layer2_results: dict[tuple[str, str], CrossCameraResult] = {}
    camera_ids = sorted(camera_data.keys())

    for iteration in range(max_iterations):
        print(f"\n[CP18] === Iteration {iteration + 1}/{max_iterations} ===")
        prev_inside_fractions = {
            cam: r.inside_mat_fraction_after
            for cam, r in layer1_results.items()
            if r.correction_matrix is not None
        }

        # Layer 1: per-camera correction
        for camera_id in camera_ids:
            if camera_id not in all_features:
                continue

            mlr = mat_line_results.get(camera_id)
            result = calibrate_single_camera(
                all_features[camera_id],
                blueprint,
                camera_id,
                mat_line_result=mlr,
            )
            layer1_results[camera_id] = result

            write_correction_json(result, output_dir)
            _write_features_jsonl(all_features[camera_id], output_dir, camera_id)

            if result.correction_matrix is not None:
                print(f"  {camera_id}: {result.confidence} ({result.signal_type}) "
                      f"inside-mat {result.inside_mat_fraction_before:.1%} → {result.inside_mat_fraction_after:.1%}")
            else:
                print(f"  {camera_id}: {result.confidence} — {result.details.get('reason', 'unknown')}")

        # Layer 2: cross-camera alignment
        for cam_a, cam_b in combinations(camera_ids, 2):
            if cam_a not in all_features or cam_b not in all_features:
                continue
            if cam_a not in layer1_results or cam_b not in layer1_results:
                continue

            try:
                pair_result = align_camera_pair(
                    all_features[cam_a], all_features[cam_b],
                    layer1_results[cam_a], layer1_results[cam_b],
                    blueprint,
                )
                layer2_results[(cam_a, cam_b)] = pair_result
                write_alignment_json(pair_result, output_dir)
                print(f"  {cam_a} ↔ {cam_b}: {pair_result.method}, "
                      f"{pair_result.confidence}, "
                      f"correspondences: {pair_result.n_correspondences}")
            except Exception as e:
                print(f"  {cam_a} ↔ {cam_b}: Layer 2 failed (non-fatal): {e}")
                layer2_results[(cam_a, cam_b)] = CrossCameraResult(
                    camera_pair=(cam_a, cam_b), method="none",
                    pairwise_correction=None, confidence="none",
                    details={"error": str(e)},
                )

        # Check convergence
        if iteration > 0 and prev_inside_fractions:
            max_improvement = 0.0
            for cam, r in layer1_results.items():
                if cam in prev_inside_fractions and r.correction_matrix is not None:
                    diff = abs(r.inside_mat_fraction_after - prev_inside_fractions[cam])
                    max_improvement = max(max_improvement, diff)

            if max_improvement < convergence_threshold:
                print(f"\n[CP18] Converged (max improvement {max_improvement:.4f} < {convergence_threshold})")
                break

    # 5. Write reports
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
# Homography loading
# ---------------------------------------------------------------------------


def _load_homography(
    homography_dir: Path, camera_id: str
) -> Optional[dict]:
    """Load homography.json for a camera.

    Returns dict with H, camera_matrix, dist_coefficients (numpy arrays),
    or None if not found.
    """
    path = homography_dir / camera_id / "homography.json"
    if not path.exists():
        return None

    with open(path) as f:
        payload = json.load(f)

    result = {
        "H": np.asarray(payload["H"], dtype=np.float64).reshape((3, 3)),
    }

    if "camera_matrix" in payload and payload["camera_matrix"] is not None:
        result["camera_matrix"] = np.asarray(
            payload["camera_matrix"], dtype=np.float64
        ).reshape((3, 3))

    if "dist_coefficients" in payload and payload["dist_coefficients"] is not None:
        result["dist_coefficients"] = np.asarray(
            payload["dist_coefficients"], dtype=np.float64
        ).ravel()

    return result


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
                "signal_type": r.signal_type,
                "inside_before": round(r.inside_mat_fraction_before, 4),
                "inside_after": round(r.inside_mat_fraction_after, 4),
                "edge_residual_before": round(r.mean_edge_residual_before, 4),
                "edge_residual_after": round(r.mean_edge_residual_after, 4),
                "n_matched_lines": r.n_matched_lines,
                "n_cleaning_positions": r.n_cleaning_positions,
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
    lines = ["# CP18 Calibration Report (v2)\n"]
    lines.append(f"**Generated:** {summary['timestamp']}")
    lines.append(f"**Cameras:** {summary['n_cameras']}")
    lines.append(f"**Blueprint area:** {summary['blueprint_area']:.1f} sq units, "
                 f"{summary['blueprint_n_edges']} boundary edges\n")

    lines.append("## Layer 1 — Per-Camera Corrections\n")
    lines.append("| Camera | Confidence | Signal | Inside Before | Inside After | Mat Lines | Cleaning Pos | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cam, r in sorted(l1.items()):
        lines.append(
            f"| {cam} | {r.confidence} | {r.signal_type} | "
            f"{r.inside_mat_fraction_before:.1%} | {r.inside_mat_fraction_after:.1%} | "
            f"{r.n_matched_lines} | {r.n_cleaning_positions} | "
            f"{r.coverage_fraction:.1%} |"
        )
    lines.append("")

    for cam, r in sorted(l1.items()):
        lines.append(f"### {cam}")
        lines.append(f"- Signal type: {r.signal_type}")
        lines.append(f"- Cleaning tracklets: {r.n_cleaning_tracklets}")
        lines.append(f"- Matched mat lines: {r.n_matched_lines} ({r.n_matched_line_edges} edges)")
        lines.append(f"- Cleaning positions: {r.n_cleaning_positions}")
        lines.append(f"- Edge touches (diagnostic): {r.n_edge_touches} ({r.n_distinct_edges} edges)")
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
            "signal_type": r.signal_type,
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
# Video path resolution
# ---------------------------------------------------------------------------


def _resolve_video_paths(
    video_dir: Path, cameras: list[str]
) -> dict[str, Path]:
    """Scan video_dir for mp4 files matching camera_id prefix.

    Searches for files like {camera_id}*.mp4 or {camera_id}/*.mp4.
    """
    video_paths = {}
    for cam in cameras:
        # Direct match: video_dir/cam_id*.mp4
        candidates = sorted(video_dir.glob(f"{cam}*.mp4"))
        if candidates:
            video_paths[cam] = candidates[0]
            continue

        # Subdirectory match: video_dir/cam_id/*.mp4
        candidates = sorted((video_dir / cam).glob("*.mp4")) if (video_dir / cam).is_dir() else []
        if candidates:
            video_paths[cam] = candidates[0]
            continue

        # Recursive search
        candidates = sorted(video_dir.rglob(f"{cam}*.mp4"))
        if candidates:
            video_paths[cam] = candidates[0]

    return video_paths


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
        description="CP18: Homography refinement + cross-camera alignment (v2)"
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
    parser.add_argument(
        "--video-dir", type=Path, default=None,
        help="Directory containing calibration video files (scanned by camera_id prefix)",
    )
    parser.add_argument(
        "--homography-dir", type=Path, default=None,
        help="Directory containing per-camera homography.json (default: configs/cameras/)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum refinement iterations",
    )
    args = parser.parse_args()

    print(f"[CP18] Loading data from {args.outputs_dir}...")
    camera_data = load_parquet_data(args.outputs_dir, args.cameras)

    if not camera_data:
        print("Error: no camera data loaded")
        sys.exit(1)

    # Resolve video paths if --video-dir provided
    video_paths = None
    if args.video_dir:
        video_paths = _resolve_video_paths(args.video_dir, args.cameras)
        if video_paths:
            print(f"[CP18] Found videos for: {', '.join(video_paths.keys())}")
        else:
            print("[CP18] Warning: --video-dir provided but no matching videos found")

    run_calibration(
        camera_data,
        args.blueprint,
        args.output_dir,
        args.fps,
        video_paths=video_paths,
        homography_dir=args.homography_dir,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
