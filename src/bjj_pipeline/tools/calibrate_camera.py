# src/bjj_pipeline/tools/calibrate_camera.py
"""Unified per-camera calibration wizard.

Orchestrates the full 3-step calibration workflow:
  Step 1 — Initial Homography (clicks UI on raw frame)
  Step 2 — Manual Lens Calibration (gradient edge detection + collinearity solve)
  Step 3 — Final H Refinement (clicks UI on undistorted frame, Mode B)

Each step checks current state in homography.json, enabling resume from any
interruption point.

Usage::

    # Full calibration from scratch
    python -m bjj_pipeline.tools.calibrate_camera \\
      --camera J_EDEw \\
      --video data/raw/nest/calibration_test/J_EDEw/video.mp4

    # Recalibrate H only (lens cal already done)
    python -m bjj_pipeline.tools.calibrate_camera \\
      --camera J_EDEw --video ... --skip-lens

    # All three cameras
    python -m bjj_pipeline.tools.calibrate_camera \\
      --camera FP7oJQ J_EDEw PPDmUg \\
      --video fp7.mp4 jed.mp4 ppd.mp4

    # Force redo all steps
    python -m bjj_pipeline.tools.calibrate_camera \\
      --camera J_EDEw --video ... --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _load_state(homography_json: Path) -> Tuple[bool, bool]:
    """Check calibration state from homography.json.

    Returns (has_h, has_lens) where:
      has_h: True if homography.json exists and contains a valid H matrix
      has_lens: True if camera_matrix and dist_coefficients are present and non-null
    """
    if not homography_json.exists():
        return False, False
    try:
        data = json.loads(homography_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, False

    has_h = "H" in data and data["H"] is not None
    has_lens = (
        data.get("camera_matrix") is not None
        and data.get("dist_coefficients") is not None
    )
    return has_h, has_lens


def _print_summary(camera_id: str, homography_json: Path) -> None:
    """Print calibration summary from the final homography.json."""
    if not homography_json.exists():
        print(f"\n  [{camera_id}] No homography.json found — calibration incomplete.")
        return

    try:
        data = json.loads(homography_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"\n  [{camera_id}] Could not read homography.json.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Calibration summary: {camera_id}")
    print(f"{'=' * 60}")

    cm = data.get("camera_matrix")
    dc = data.get("dist_coefficients")
    if cm is not None and dc is not None:
        K = np.asarray(cm)
        dist = np.asarray(dc).ravel()
        f = float(K[0, 0])
        k1 = float(dist[0]) if len(dist) > 0 else 0.0
        k2 = float(dist[1]) if len(dist) > 1 else 0.0
        print(f"  Lens:  f={f:.1f}  k1={k1:.4f}  k2={k2:.4f}")
    else:
        print("  Lens:  not calibrated")

    qm = data.get("quality_metrics", {})
    hm = qm.get("h_refinement_metrics", {})
    if hm:
        print(
            f"  H refinement:  reproj={hm.get('mean_reproj_error_px', '?')}px  "
            f"lines={hm.get('n_matched_lines', '?')}  "
            f"edges={hm.get('n_distinct_edges_matched', '?')}  "
            f"inliers={hm.get('n_inliers', '?')}/{hm.get('n_total_correspondences', '?')}  "
            f"converged={hm.get('converged', '?')}"
        )
    else:
        mode = qm.get("calibration_mode", "unknown")
        print(f"  H mode: {mode}")

    print(f"{'=' * 60}\n")


def run_wizard(
    camera_id: str,
    video_path: Path,
    configs_root: Path,
    mat_blueprint_path: Path,
    *,
    skip_lens: bool = False,
    force: bool = False,
) -> None:
    """Run the full calibration wizard for one camera."""
    from bjj_pipeline.tools.homography_calibrate import (
        _default_homography_json_path,
        _interactive_calibrate,
        _load_lens_calibration,
    )
    from calibration_pipeline.lens_calibration import _run_interactive as _run_lens_cal

    homography_json = _default_homography_json_path(configs_root, camera_id)
    has_h, has_lens = _load_state(homography_json)

    print(f"\n{'=' * 60}")
    print(f"  Calibration wizard: {camera_id}")
    print(f"  Video: {video_path}")
    print(f"  State: H={'yes' if has_h else 'no'}  K+dist={'yes' if has_lens else 'no'}")
    if force:
        print("  Mode: --force (redo all steps)")
    if skip_lens:
        print("  Mode: --skip-lens (skip Step 2)")
    print(f"{'=' * 60}\n")

    # ── Step 1: Initial Homography ─────────────────────────────
    if has_h and not force:
        print("[Step 1/3] Initial homography — SKIPPED (H already exists)")
    else:
        print("[Step 1/3] Initial homography — place 4+ point pairs on RAW frame")
        print("  Controls: click IMAGE then MAT (repeat). [u]=undo [c]=clear [s]=solve+save [q]=quit")
        print()
        _interactive_calibrate(
            camera_id=camera_id,
            out_path=homography_json,
            video_path=video_path,
            mat_blueprint_path=mat_blueprint_path,
        )
        # Re-check state after Step 1
        has_h, has_lens = _load_state(homography_json)
        if not has_h:
            print("\n[!] Step 1 did not produce an H. Aborting wizard.")
            return
        print("\n[Step 1/3] Complete.\n")

    # ── Step 2: Lens Calibration ───────────────────────────────
    if skip_lens:
        print("[Step 2/3] Lens calibration — SKIPPED (--skip-lens)")
    elif has_lens and not force:
        print("[Step 2/3] Lens calibration — SKIPPED (K+dist already exist)")
    else:
        if not has_h:
            print("[!] Cannot run lens calibration without initial H. Aborting.")
            return
        print("[Step 2/3] Lens calibration — verify/add edge points, then solve")
        print("  Controls: left-click=add  right-click/d=delete  [s]=solve  [a]=accept  [r]=redo  [q]=quit")
        print()
        _run_lens_cal(
            camera_id=camera_id,
            homography_json_path=homography_json,
            video_path=video_path,
        )
        # Re-check state after Step 2
        has_h, has_lens = _load_state(homography_json)
        if not has_lens:
            print("\n[!] Step 2 did not produce K+dist. Continuing to Step 3 without lens cal.")
        else:
            print("\n[Step 2/3] Complete.\n")

    # ── Step 3: Final H Refinement ─────────────────────────────
    # Always runs — this is the final output.
    # If K+dist exist, display frame is undistorted and Mode B fires automatically.
    # If K+dist don't exist (lens cal skipped/failed), Mode A fires.
    has_h, has_lens = _load_state(homography_json)
    if has_lens:
        print("[Step 3/3] Final H refinement — place 4+ point pairs on UNDISTORTED frame")
        print("  The display frame has lens distortion corrected. Mat edges should look straight.")
    else:
        print("[Step 3/3] Final H calibration — place 4+ point pairs")
        print("  No lens calibration found — working on raw frame.")
    print("  Controls: click IMAGE then MAT (repeat). [u]=undo [c]=clear [s]=solve+save [q]=quit")
    print()
    _interactive_calibrate(
        camera_id=camera_id,
        out_path=homography_json,
        video_path=video_path,
        mat_blueprint_path=mat_blueprint_path,
    )

    # Final summary
    _print_summary(camera_id, homography_json)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m bjj_pipeline.tools.calibrate_camera",
        description=(
            "Unified per-camera calibration wizard. "
            "Runs all 3 steps (initial H, lens cal, final H refinement) "
            "in sequence, with automatic resume from any interruption point."
        ),
    )
    p.add_argument(
        "--camera", required=True, nargs="+",
        help="Camera id(s) to calibrate (e.g. J_EDEw, or FP7oJQ J_EDEw PPDmUg)",
    )
    p.add_argument(
        "--video", required=True, nargs="+",
        help="Video file(s) for calibration (one per camera, same order)",
    )
    p.add_argument(
        "--configs-root", default="configs",
        help="Repo configs root (default: ./configs)",
    )
    p.add_argument(
        "--mat-blueprint", default="configs/mat_blueprint.json",
        help="Path to mat blueprint JSON (default: configs/mat_blueprint.json)",
    )
    p.add_argument(
        "--skip-lens", action="store_true",
        help="Skip Step 2 (lens calibration). Use when K+dist are already correct and you only need to recalibrate H.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Redo all steps even if prior calibration exists.",
    )

    args = p.parse_args()

    cameras = args.camera
    videos = args.video

    if len(cameras) != len(videos):
        print(
            f"Error: --camera has {len(cameras)} entries but --video has {len(videos)}. "
            "Provide one video per camera.",
            file=sys.stderr,
        )
        sys.exit(1)

    configs_root = Path(args.configs_root)
    mat_blueprint = Path(args.mat_blueprint)

    for cam_id, vid in zip(cameras, videos):
        vid_path = Path(vid)
        if not vid_path.exists():
            print(f"Error: video not found: {vid_path}", file=sys.stderr)
            sys.exit(1)

        run_wizard(
            camera_id=cam_id,
            video_path=vid_path,
            configs_root=configs_root,
            mat_blueprint_path=mat_blueprint,
            skip_lens=args.skip_lens,
            force=args.force,
        )

    if len(cameras) > 1:
        print(f"\nAll {len(cameras)} cameras calibrated.")


if __name__ == "__main__":
    main()
