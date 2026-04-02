# src/bjj_pipeline/tools/calibration_verify.py
"""Cross-camera calibration agreement diagnostic.

Measures pairwise world-coordinate agreement between calibrated cameras by
comparing where each camera places shared blueprint edges in world space.
Read-only — does not modify any homography.json files.

Usage::

    python -m bjj_pipeline.tools.calibration_verify --configs-root configs
    python -m bjj_pipeline.tools.calibration_verify --camera FP7oJQ J_EDEw PPDmUg
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_camera_data(homography_json: Path) -> Optional[Dict[str, Any]]:
    """Load calibration data from a single camera's homography.json."""
    if not homography_json.exists():
        return None
    try:
        data = json.loads(homography_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if "H" not in data or data["H"] is None:
        return None

    H = np.asarray(data["H"], dtype=np.float64)
    pp = data.get("projected_polylines", {})
    image_wh = tuple(pp.get("image_wh", [0, 0]))

    # Index polylines by edge_index
    polylines_by_edge: Dict[int, Dict[str, Any]] = {}
    for pl in pp.get("polylines", []):
        eidx = pl["edge_index"]
        # Keep the polyline with more pixel points if duplicates
        if eidx not in polylines_by_edge or len(pl["pixel_points"]) > len(
            polylines_by_edge[eidx]["pixel_points"]
        ):
            polylines_by_edge[eidx] = pl

    return {
        "H": H,
        "H_inv": np.linalg.inv(H),
        "image_wh": image_wh,
        "polylines_by_edge": polylines_by_edge,
        "quality_metrics": data.get("quality_metrics", {}),
    }


def _pixel_to_world(H_inv: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    """Project a single pixel point to world coords via H_inv."""
    p = H_inv @ np.array([px, py, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-12:
        return (float("nan"), float("nan"))
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def _world_point_on_edge(
    world_start: List[float], world_end: List[float], t: float
) -> Tuple[float, float]:
    """Interpolate a point along a blueprint edge at parameter t in [0, 1]."""
    wx = world_start[0] + t * (world_end[0] - world_start[0])
    wy = world_start[1] + t * (world_end[1] - world_start[1])
    return (wx, wy)


def _compute_edge_disagreement(
    pl_a: Dict[str, Any],
    pl_b: Dict[str, Any],
    H_inv_a: np.ndarray,
    H_inv_b: np.ndarray,
    n_samples: int = 20,
) -> Dict[str, Any]:
    """Compute world-coordinate disagreement between two cameras on a shared edge.

    For each of n_samples points along the blueprint edge, project the
    corresponding pixel point from each camera through H_inv to world coords.
    Measure the distance between the two world estimates.
    """
    ws = pl_a["world_start"]
    we = pl_a["world_end"]

    pts_a = pl_a["pixel_points"]
    pts_b = pl_b["pixel_points"]

    if len(pts_a) < 2 or len(pts_b) < 2:
        return {"n_samples": 0, "errors_m": []}

    errors: List[float] = []

    for i in range(n_samples):
        t = i / max(1, n_samples - 1)

        # Find the pixel point closest to parameter t along each polyline
        idx_a = min(int(t * (len(pts_a) - 1) + 0.5), len(pts_a) - 1)
        idx_b = min(int(t * (len(pts_b) - 1) + 0.5), len(pts_b) - 1)

        wx_a, wy_a = _pixel_to_world(H_inv_a, pts_a[idx_a][0], pts_a[idx_a][1])
        wx_b, wy_b = _pixel_to_world(H_inv_b, pts_b[idx_b][0], pts_b[idx_b][1])

        if np.isfinite(wx_a) and np.isfinite(wy_a) and np.isfinite(wx_b) and np.isfinite(wy_b):
            err = float(np.hypot(wx_a - wx_b, wy_a - wy_b))
            errors.append(err)

    return {
        "n_samples": len(errors),
        "errors_m": errors,
    }


def compute_pairwise_agreement(
    cameras: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute pairwise agreement metrics for all camera pairs."""
    cam_ids = sorted(cameras.keys())
    results: List[Dict[str, Any]] = []

    for cam_a, cam_b in combinations(cam_ids, 2):
        data_a = cameras[cam_a]
        data_b = cameras[cam_b]

        edges_a = set(data_a["polylines_by_edge"].keys())
        edges_b = set(data_b["polylines_by_edge"].keys())
        shared = sorted(edges_a & edges_b)

        if not shared:
            results.append({
                "camera_a": cam_a,
                "camera_b": cam_b,
                "shared_edges": 0,
                "shared_edge_indices": [],
                "mean_m": None,
                "p95_m": None,
                "max_m": None,
                "per_edge": [],
            })
            continue

        all_errors: List[float] = []
        per_edge: List[Dict[str, Any]] = []

        for eidx in shared:
            pl_a = data_a["polylines_by_edge"][eidx]
            pl_b = data_b["polylines_by_edge"][eidx]

            edge_result = _compute_edge_disagreement(
                pl_a, pl_b, data_a["H_inv"], data_b["H_inv"],
            )

            if edge_result["n_samples"] > 0:
                errs = edge_result["errors_m"]
                all_errors.extend(errs)
                per_edge.append({
                    "edge_index": eidx,
                    "world_start": pl_a["world_start"],
                    "world_end": pl_a["world_end"],
                    "n_samples": edge_result["n_samples"],
                    "mean_m": round(float(np.mean(errs)), 4),
                    "max_m": round(float(np.max(errs)), 4),
                })

        if all_errors:
            arr = np.array(all_errors)
            mean_m = round(float(np.mean(arr)), 4)
            p95_m = round(float(np.percentile(arr, 95)), 4)
            max_m = round(float(np.max(arr)), 4)
        else:
            mean_m = p95_m = max_m = None

        results.append({
            "camera_a": cam_a,
            "camera_b": cam_b,
            "shared_edges": len(shared),
            "shared_edge_indices": shared,
            "mean_m": mean_m,
            "p95_m": p95_m,
            "max_m": max_m,
            "per_edge": per_edge,
        })

    return results


def _status_label(max_m: Optional[float]) -> str:
    if max_m is None:
        return "-- No data"
    if max_m < 0.05:
        return "Excellent"
    if max_m < 0.15:
        return "Acceptable"
    return "Investigate"


def print_report(results: List[Dict[str, Any]]) -> None:
    """Print a summary table to stdout."""
    print()
    print("Cross-Camera Agreement Report")
    print("=" * 72)
    print(f"{'Camera pair':<22} {'Shared':>6}  {'Mean (m)':>8}  {'P95 (m)':>8}  {'Max (m)':>8}  Status")
    print("-" * 72)

    worst_max = 0.0
    worst_pair = ""

    for r in results:
        pair = f"{r['camera_a']} <> {r['camera_b']}"
        shared = r["shared_edges"]
        mean_s = f"{r['mean_m']:.3f}" if r["mean_m"] is not None else "  --"
        p95_s = f"{r['p95_m']:.3f}" if r["p95_m"] is not None else "  --"
        max_s = f"{r['max_m']:.3f}" if r["max_m"] is not None else "  --"
        status = _status_label(r["max_m"])

        print(f"{pair:<22} {shared:>6}  {mean_s:>8}  {p95_s:>8}  {max_s:>8}  {status}")

        if r["max_m"] is not None and r["max_m"] > worst_max:
            worst_max = r["max_m"]
            worst_pair = pair

    print("-" * 72)
    if worst_pair:
        print(f"Overall worst-pair: {worst_max:.3f}m ({worst_pair})")
    print()


def write_json_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write detailed JSON report."""
    report = {
        "pairs": results,
        "summary": {},
    }

    maxes = [r["max_m"] for r in results if r["max_m"] is not None]
    if maxes:
        report["summary"] = {
            "n_pairs": len(results),
            "n_pairs_with_data": len(maxes),
            "worst_max_m": round(max(maxes), 4),
            "mean_max_m": round(float(np.mean(maxes)), 4),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report to {output_path}")


def run_verify(
    configs_root: Path,
    camera_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run cross-camera verification. Returns pairwise results."""
    cameras_dir = configs_root / "cameras"
    if not cameras_dir.exists():
        print(f"Error: cameras directory not found: {cameras_dir}", file=sys.stderr)
        return []

    # Discover cameras
    if camera_ids:
        cam_dirs = [cameras_dir / c for c in camera_ids]
    else:
        cam_dirs = sorted(
            [d for d in cameras_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    # Load data
    cameras: Dict[str, Dict[str, Any]] = {}
    for cam_dir in cam_dirs:
        hj = cam_dir / "homography.json"
        data = _load_camera_data(hj)
        if data is not None:
            cameras[cam_dir.name] = data
            n_edges = len(data["polylines_by_edge"])
            print(f"  Loaded {cam_dir.name}: {n_edges} edges")
        else:
            print(f"  Skipped {cam_dir.name}: no valid homography.json")

    if len(cameras) < 2:
        print("Need at least 2 calibrated cameras for cross-camera verification.")
        return []

    results = compute_pairwise_agreement(cameras)
    print_report(results)

    report_path = configs_root / "calibration_report.json"
    write_json_report(results, report_path)

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m bjj_pipeline.tools.calibration_verify",
        description=(
            "Cross-camera calibration agreement diagnostic. "
            "Measures pairwise world-coordinate agreement between calibrated cameras."
        ),
    )
    p.add_argument(
        "--camera", nargs="*", default=None,
        help="Camera id(s) to verify (default: all cameras with homography.json)",
    )
    p.add_argument(
        "--configs-root", default="configs",
        help="Repo configs root (default: ./configs)",
    )

    args = p.parse_args()
    run_verify(
        configs_root=Path(args.configs_root),
        camera_ids=args.camera,
    )


if __name__ == "__main__":
    main()
