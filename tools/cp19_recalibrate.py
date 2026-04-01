"""CP19-revised: Re-calibrate all cameras — H-only recalibration.

Uses existing anchor correspondences + permanent K+dist from homography.json.
Only recomputes H via mat-line detection on the undistorted frame.
K+dist are treated as permanent (from interactive setup or manual lens_calibration).

Usage:
    python tools/cp19_recalibrate.py [--dry-run] [--camera CAM_ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bjj_pipeline.tools.homography_calibrate import (
    _ensure_3x3,
    _find_empty_frame,
    _generate_projected_polylines,
    _load_lens_calibration,
    _parse_rects_from_blueprint,
    _refine_h_from_mat_lines,
    _write_homography_json,
)


CONFIGS_ROOT = Path("configs")
CALIBRATION_TEST_ROOT = Path("data/raw/nest/calibration_test")
CAMERAS = ["FP7oJQ", "J_EDEw", "PPDmUg"]


def _load_homography_json(cam_id: str) -> dict:
    path = CONFIGS_ROOT / "cameras" / cam_id / "homography.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_blueprint() -> list:
    bp_path = CONFIGS_ROOT / "mat_blueprint.json"
    return json.loads(bp_path.read_text(encoding="utf-8"))


def _find_calibration_video(cam_id: str) -> str:
    """Find the best calibration video for a camera.

    Prefers calibration_test videos (empty-mat recordings) over the
    source video from homography.json (which may have people on the mat).
    """
    cal_dir = CALIBRATION_TEST_ROOT / cam_id
    if cal_dir.is_dir():
        mp4s = sorted(cal_dir.rglob("*.mp4"))
        if mp4s:
            return str(mp4s[0])
    # Fall back to source video from homography.json
    hom_data = _load_homography_json(cam_id)
    return hom_data["source"]["video"]


def recalibrate_camera(cam_id: str, *, dry_run: bool = False) -> dict:
    """H-only recalibration: loads permanent K+dist, only recomputes H."""
    print(f"\n{'='*60}")
    print(f"  CP19 H-Only Recalibration: {cam_id}")
    print(f"{'='*60}")

    hom_data = _load_homography_json(cam_id)
    blueprint_raw = _load_blueprint()
    rects = _parse_rects_from_blueprint(blueprint_raw)

    # Load permanent K+dist
    out_path = CONFIGS_ROOT / "cameras" / cam_id / "homography.json"
    K, dist = _load_lens_calibration(out_path)
    if K is None:
        print("  ERROR: No lens calibration found. Run interactive setup first.")
        return {"error": "no lens calibration"}

    f_val = float(K[0, 0])
    k1_val = float(dist[0]) if len(dist) > 0 else 0.0
    k2_val = float(dist[1]) if len(dist) > 1 else 0.0
    print(f"  Permanent K+dist: f={f_val:.1f} k1={k1_val:.4f} k2={k2_val:.4f}")

    # Load existing correspondences (in raw pixel space)
    corr = hom_data["correspondences"]
    img_pts = np.array(corr["image_points_px"], dtype=np.float64)
    mat_pts = np.array(corr["mat_points"], dtype=np.float64)
    corner_ids = corr.get("corner_ids", ["tl", "tr", "br", "bl"])

    # Find calibration video
    video_path = _find_calibration_video(cam_id)
    is_cal_test = "calibration_test" in video_path
    print(f"  Video: {video_path}" + (" (calibration_test)" if is_cal_test else " (source)"))

    # Find the emptiest frame
    frame_bgr_raw, frame_idx = _find_empty_frame(video_path)
    print(f"  Selected frame {frame_idx} (lowest activity)")

    img_h, img_w = frame_bgr_raw.shape[:2]
    image_wh = (img_w, img_h)

    print(f"  Frame: {img_w}x{img_h}")
    print(f"  Anchor: {len(img_pts)} points, corner_ids={corner_ids}")
    print(f"  Blueprint: {len(rects)} rectangles")

    # Undistort frame + anchor points with permanent K+dist
    frame_undist = cv2.undistort(frame_bgr_raw, K, dist)
    anchor_undist = cv2.undistortPoints(
        img_pts.reshape(-1, 1, 2).astype(np.float64),
        K, dist, P=K,
    ).reshape(-1, 2)

    # H₀ from anchor in undistorted space
    H_initial, _ = cv2.findHomography(mat_pts, anchor_undist, method=0)
    H_initial = _ensure_3x3(H_initial)

    # H refinement from mat lines on undistorted frame (no K/dist — already undistorted)
    print(f"\n  H refinement from mat lines...")
    H_refined, h_metrics = _refine_h_from_mat_lines(
        H_initial=H_initial,
        frame_bgr=frame_undist,
        rects=rects,
        anchor_img_pts=anchor_undist,
        anchor_mat_pts=mat_pts,
        camera_matrix=None,
        dist_coefficients=None,
    )

    print(f"    Mean reproj error: {h_metrics['mean_reproj_error_px']:.2f} px")
    print(f"    Max reproj error:  {h_metrics['max_reproj_error_px']:.2f} px")
    print(f"    Anchor reproj:     {h_metrics['anchor_reproj_error_px']:.2f} px")
    print(f"    Matched lines:     {h_metrics['n_matched_lines']}")
    print(f"    Detected lines:    {h_metrics['n_detected_lines']}")
    print(f"    Distinct edges:    {h_metrics['n_distinct_edges_matched']}")
    print(f"    Inliers:           {h_metrics['n_inliers']}/{h_metrics['n_total_correspondences']} "
          f"({h_metrics['inlier_ratio']:.1%})")
    print(f"    Iterations:        {h_metrics['refinement_iterations']}")
    print(f"    Converged:         {h_metrics['converged']}")

    # Check if H changed
    h_diff = np.abs(H_refined - H_initial).max()
    print(f"    H max element change: {h_diff:.6f}")

    quality_metrics = {
        "h_refinement_metrics": h_metrics,
        "calibration_mode": "unified_v2",
    }

    if dry_run:
        print(f"\n  [DRY RUN] Would save to {out_path}")
    else:
        # Generate polylines from refined H
        polyline_data = _generate_projected_polylines(
            H_mat_to_img=H_refined, rects=rects, image_wh=image_wh,
        )
        print(f"\n  Generated {polyline_data['n_polylines']} polylines from "
              f"{polyline_data['n_edges_total']} edges")

        # Preserve existing lens_calibration metadata, K+dist untouched
        extra = {
            "correspondences": {
                "image_points_px": img_pts.tolist(),
                "mat_points": mat_pts.tolist(),
                "corner_ids": corner_ids,
            },
            "ui": hom_data.get("ui", {}),
            "qa": hom_data.get("qa", {}),
            "projected_polylines": polyline_data,
            "quality_metrics": quality_metrics,
            "camera_matrix": K.tolist(),
            "dist_coefficients": dist.ravel().tolist(),
        }
        # Preserve existing lens_calibration block (from interactive setup or manual tool)
        if "lens_calibration" in hom_data:
            extra["lens_calibration"] = hom_data["lens_calibration"]

        _write_homography_json(
            out_path=out_path,
            camera_id=cam_id,
            H=H_refined,
            source=hom_data.get("source", {"type": "cp19_recalibrate"}),
            extra=extra,
        )
        print(f"  Saved to {out_path}")

    return quality_metrics


def main():
    parser = argparse.ArgumentParser(description="CP19 H-only recalibration")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("--camera", default=None, help="Single camera to recalibrate")
    args = parser.parse_args()

    cameras = [args.camera] if args.camera else CAMERAS
    results = {}

    for cam in cameras:
        try:
            results[cam] = recalibrate_camera(cam, dry_run=args.dry_run)
        except Exception as e:
            print(f"\n  ERROR on {cam}: {e}")
            import traceback; traceback.print_exc()
            results[cam] = {"error": str(e)}

    # Summary table
    print(f"\n\n{'='*60}")
    print("  CP19 Recalibration Summary")
    print(f"{'='*60}")
    print(f"{'Camera':<10} {'Lines':>6} {'Reproj':>8} {'Inliers':>10} {'Converged':>10}")
    print(f"{'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

    for cam in cameras:
        r = results.get(cam, {})
        if "error" in r:
            print(f"{cam:<10} ERROR: {r['error']}")
            continue
        hm = r.get("h_refinement_metrics", {})
        lines = str(hm.get("n_matched_lines", 0))
        reproj = f"{hm.get('mean_reproj_error_px', 0):.1f}px"
        inliers = f"{hm.get('n_inliers', 0)}/{hm.get('n_total_correspondences', 0)}"
        converged = str(hm.get("converged", "?"))
        print(f"{cam:<10} {lines:>6} {reproj:>8} {inliers:>10} {converged:>10}")


if __name__ == "__main__":
    main()
