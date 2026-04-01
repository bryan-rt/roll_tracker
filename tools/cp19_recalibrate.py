"""CP19: Re-calibrate all cameras using unified polyline lens cal + H refinement.

Uses existing anchor correspondences from homography.json (no GUI needed).
Runs Phase A (polyline lens cal) + Phase B (mat-line H refinement) and
saves updated homography.json with quality metrics.

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
    _detect_edge_points_along_polylines,
    _ensure_3x3,
    _generate_projected_polylines,
    _parse_rects_from_blueprint,
    _polyline_lens_calibration,
    _recompute_h_for_space,
    _redistort_points,
    _refine_h_from_mat_lines,
    _write_homography_json,
)


CONFIGS_ROOT = Path("configs")
CAMERAS = ["FP7oJQ", "J_EDEw", "PPDmUg"]


def _load_homography_json(cam_id: str) -> dict:
    path = CONFIGS_ROOT / "cameras" / cam_id / "homography.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_blueprint() -> list:
    bp_path = CONFIGS_ROOT / "mat_blueprint.json"
    return json.loads(bp_path.read_text(encoding="utf-8"))


def recalibrate_camera(cam_id: str, *, dry_run: bool = False) -> dict:
    """Run CP19 unified calibration on a single camera. Returns quality_metrics."""
    print(f"\n{'='*60}")
    print(f"  CP19 Recalibration: {cam_id}")
    print(f"{'='*60}")

    hom_data = _load_homography_json(cam_id)
    blueprint_raw = _load_blueprint()
    rects = _parse_rects_from_blueprint(blueprint_raw)

    # Load existing correspondences
    corr = hom_data["correspondences"]
    img_pts = np.array(corr["image_points_px"], dtype=np.float64)
    mat_pts = np.array(corr["mat_points"], dtype=np.float64)
    corner_ids = corr.get("corner_ids", ["tl", "tr", "br", "bl"])

    # Load existing K+dist (anchor img_pts are in this undistorted space)
    old_K = None
    old_dist = None
    cm = hom_data.get("camera_matrix")
    dc = hom_data.get("dist_coefficients")
    if cm is not None and dc is not None:
        old_K = np.asarray(cm, dtype=np.float64).reshape(3, 3)
        old_dist = np.asarray(dc, dtype=np.float64).ravel()

    # Load source video, grab first frame
    video_path = hom_data["source"]["video"]
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr_raw = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame from: {video_path}")

    img_h, img_w = frame_bgr_raw.shape[:2]
    image_wh = (img_w, img_h)
    frame_gray = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2GRAY)

    print(f"  Frame: {img_w}x{img_h}")
    print(f"  Anchor: {len(img_pts)} points, corner_ids={corner_ids}")
    print(f"  Blueprint: {len(rects)} rectangles")
    if old_K is not None:
        print(f"  Existing K: f={old_K[0,0]:.1f}, dist=[{old_dist[0]:.4f}, {old_dist[1]:.4f}]")

    # --- Coordinate space handling ---
    # img_pts are in old-undistorted pixel space (if old K+dist exist).
    # Phase A needs H that maps mat → RAW pixel space.
    # Phase B needs H that maps mat → new-undistorted pixel space.

    # Compute H_mat_to_raw for Phase A
    H_mat_to_raw, raw_anchor_pts = _recompute_h_for_space(
        mat_pts, img_pts,
        from_K=old_K, from_dist=old_dist,
        to_K=None, to_dist=None,  # target = raw pixel space
    )
    print(f"  H_mat_to_raw computed (anchor pts re-distorted to raw space)")

    # --- Phase A: Polyline-based lens calibration ---
    print(f"\n  Phase A: Polyline lens calibration...")
    K_new, dist_new, lens_metrics = _polyline_lens_calibration(
        H_mat_to_img=H_mat_to_raw,  # projects to RAW pixel space
        frame_bgr=frame_bgr_raw,
        frame_gray=frame_gray,
        rects=rects,
        image_wh=image_wh,
    )

    if K_new is not None:
        print(f"    f = {lens_metrics['f']:.1f}")
        print(f"    k1 = {lens_metrics['k1']:.4f}")
        print(f"    k2 = {lens_metrics['k2']:.4f}")
        print(f"    Collinearity cost = {lens_metrics['collinearity_cost']:.1f}")
        print(f"    Edge points: {lens_metrics['n_edge_points']} across {lens_metrics['n_edges_with_points']} edges")
        print(f"    Converged: {lens_metrics.get('converged', '?')}")
    else:
        print(f"    SKIPPED: {lens_metrics.get('reason', 'unknown')}")
        # Fall back to existing K+dist
        if old_K is not None:
            K_new, dist_new = old_K, old_dist
            print(f"    Using existing K+dist from homography.json")
        else:
            print(f"    No existing K+dist either — Phase B will run without undistortion")

    # Compare with old lens cal
    old_lens = hom_data.get("lens_calibration", {})
    if old_lens and K_new is not None and lens_metrics.get("f") is not None:
        print(f"\n    Comparison with previous lens calibration:")
        print(f"      f:  {old_lens.get('f', '?')} → {lens_metrics['f']:.1f}")
        print(f"      k1: {old_lens.get('k1', '?')} → {lens_metrics['k1']:.4f}")
        print(f"      k2: {old_lens.get('k2', '?')} → {lens_metrics['k2']:.4f}")
        print(f"      pts: {old_lens.get('num_points', '?')} → {lens_metrics['n_edge_points']}")

    # --- Phase B: Mat-line H refinement ---
    # Recompute H and anchor pts for the new-undistorted pixel space
    H_for_phase_b, new_undist_anchor = _recompute_h_for_space(
        mat_pts, img_pts,
        from_K=old_K, from_dist=old_dist,
        to_K=K_new, to_dist=dist_new,  # target = new-undistorted space
    )
    print(f"\n  Phase B: Mat-line H refinement...")
    print(f"    H recomputed for new-undistorted space")
    H_refined, h_metrics = _refine_h_from_mat_lines(
        H_initial=H_for_phase_b,
        frame_bgr=frame_bgr_raw,
        rects=rects,
        anchor_img_pts=new_undist_anchor,
        anchor_mat_pts=mat_pts,
        camera_matrix=K_new,
        dist_coefficients=dist_new,
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
    h_diff = np.abs(H_refined - H_for_phase_b).max()
    print(f"    H max element change: {h_diff:.6f}")

    quality_metrics = {
        "h_metrics": h_metrics,
        "lens_metrics": lens_metrics,
        "calibration_mode": "unified",
    }

    if dry_run:
        print(f"\n  [DRY RUN] Would save to configs/cameras/{cam_id}/homography.json")
    else:
        # Generate polylines from refined H
        polyline_data = _generate_projected_polylines(
            H_mat_to_img=H_refined, rects=rects, image_wh=image_wh,
        )
        print(f"\n  Generated {polyline_data['n_polylines']} polylines from "
              f"{polyline_data['n_edges_total']} edges")

        out_path = CONFIGS_ROOT / "cameras" / cam_id / "homography.json"

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
        }
        if K_new is not None:
            extra["camera_matrix"] = K_new.tolist()
            extra["dist_coefficients"] = dist_new.tolist()
        # Preserve lens_calibration metadata with updated values
        if K_new is not None and lens_metrics.get("f") is not None:
            extra["lens_calibration"] = {
                "method": "cp19_polyline_collinearity",
                **{k: v for k, v in lens_metrics.items()
                   if k not in ("per_edge_rms", "points_per_edge")},
                "image_size": [img_w, img_h],
            }

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
    parser = argparse.ArgumentParser(description="CP19 re-calibration")
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
    print(f"{'Camera':<10} {'f':>8} {'k1':>8} {'k2':>8} {'EdgePts':>8} {'Lines':>6} {'Reproj':>8} {'Inliers':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*10}")

    for cam in cameras:
        r = results.get(cam, {})
        if "error" in r:
            print(f"{cam:<10} ERROR: {r['error']}")
            continue
        lm = r.get("lens_metrics", {})
        hm = r.get("h_metrics", {})
        f_val = f"{lm.get('f', 0):.0f}" if lm.get("f") else "skip"
        k1_val = f"{lm.get('k1', 0):.3f}" if lm.get("k1") is not None else "skip"
        k2_val = f"{lm.get('k2', 0):.3f}" if lm.get("k2") is not None else "skip"
        pts = str(lm.get("n_edge_points", "-"))
        lines = str(hm.get("n_matched_lines", 0))
        reproj = f"{hm.get('mean_reproj_error_px', 0):.1f}px"
        inliers = f"{hm.get('n_inliers', 0)}/{hm.get('n_total_correspondences', 0)}"
        print(f"{cam:<10} {f_val:>8} {k1_val:>8} {k2_val:>8} {pts:>8} {lines:>6} {reproj:>8} {inliers:>10}")


if __name__ == "__main__":
    main()
