"""Lens calibration — estimate K + distortion from mat edge clicks.

Interactive matplotlib tool. Uses existing 4-corner correspondences from
homography.json as seed data. The user clicks additional points along mat
edges in the raw distorted frame. Mat edges are straight in world space,
so deviation from collinearity in pixel space IS the lens distortion signal.

Two-step calibration chain (CP16b):
  1. Run this tool  → produces K + dist in homography.json
  2. Re-run homography_calibrate.py → auto-undistorts frame, user aligns
     overlay on straight geometry → produces H valid for undistorted pixels

Usage::

    python -m calibration_pipeline.lens_calibration \\
      --camera J_EDEw \\
      --video data/raw/nest/gym01/J_EDEw/2026-03-18/20/J_EDEw-20260318-201503.mp4 \\
      --configs-root configs

Controls (matplotlib key bindings):
  click   — add edge point (auto-snaps to nearest mat edge)
  u       — undo last edge point
  c       — clear all edge points (keep original 4 corners)
  s       — solve (run calibrateCamera, display undistorted result)
  a       — accept (write K + dist to homography.json)
  r       — redo (back to point clicking)
  q       — quit without saving
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Edge-snapping geometry
# ---------------------------------------------------------------------------

def _point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> Tuple[float, float]:
    """Distance from point (px,py) to segment (A,B). Returns (distance, t).

    t is the projection parameter along AB, clamped to [0,1].
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return float(np.hypot(apx, apy)), 0.0
    t = (apx * abx + apy * aby) / ab_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(np.hypot(px - cx, py - cy)), t


def _snap_to_nearest_edge(
    click_xy: Tuple[float, float],
    corners_img: List[Tuple[float, float]],
    corners_mat: List[Tuple[float, float]],
) -> Tuple[int, float, Tuple[float, float]]:
    """Find the nearest mat edge to a click and compute the world coordinate.

    Parameters
    ----------
    click_xy : (u, v) pixel coordinates of the click
    corners_img : 4 corner pixel coords, ordered [tl, tr, br, bl]
    corners_mat : 4 corner world coords, same ordering

    Returns
    -------
    (edge_idx, t, world_xy)
      edge_idx: index of nearest edge (0=tl-tr, 1=tr-br, 2=br-bl, 3=bl-tl)
      t: parameter along edge [0..1]
      world_xy: interpolated world coordinate on that edge
    """
    cx, cy = click_xy
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # tl-tr, tr-br, br-bl, bl-tl
    best_dist = float("inf")
    best_edge = 0
    best_t = 0.0
    for i, (a_idx, b_idx) in enumerate(edges):
        ax, ay = corners_img[a_idx]
        bx, by = corners_img[b_idx]
        dist, t = _point_to_segment_distance(cx, cy, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best_edge = i
            best_t = t

    # Interpolate world coordinate along the best edge
    a_idx, b_idx = edges[best_edge]
    mx_a, my_a = corners_mat[a_idx]
    mx_b, my_b = corners_mat[b_idx]
    world_x = mx_a + best_t * (mx_b - mx_a)
    world_y = my_a + best_t * (my_b - my_a)

    return best_edge, best_t, (world_x, world_y)


# ---------------------------------------------------------------------------
# Interactive calibration
# ---------------------------------------------------------------------------

def _run_interactive(
    camera_id: str,
    homography_json_path: Path,
    video_path: Path,
) -> None:
    """Interactive lens calibration UI."""
    import cv2
    import matplotlib.pyplot as plt

    # --- Load existing homography.json ---
    if not homography_json_path.exists():
        raise FileNotFoundError(
            f"homography.json not found: {homography_json_path}\n"
            "Run homography_calibrate.py first to create the initial calibration."
        )
    hom_data = json.loads(homography_json_path.read_text(encoding="utf-8"))
    corr = hom_data.get("correspondences", {})
    img_pts_raw = corr.get("image_points_px")
    mat_pts_raw = corr.get("mat_points")
    corner_ids = corr.get("corner_ids", ["tl", "tr", "br", "bl"])

    if not img_pts_raw or not mat_pts_raw or len(img_pts_raw) != 4 or len(mat_pts_raw) != 4:
        raise ValueError(
            "homography.json must have exactly 4 correspondences (image_points_px + mat_points). "
            f"Found {len(img_pts_raw or [])} image points, {len(mat_pts_raw or [])} mat points."
        )

    corners_img: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in img_pts_raw]
    corners_mat: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in mat_pts_raw]

    # --- Load first frame ---
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame_rgb.shape[:2]
    print(f"[lens_cal] Frame size: {img_w}x{img_h}")
    print(f"[lens_cal] Loaded 4 corners from {homography_json_path}")
    for cid, ip, mp in zip(corner_ids, corners_img, corners_mat):
        print(f"  {cid}: pixel=({ip[0]:.1f}, {ip[1]:.1f})  world=({mp[0]:.1f}, {mp[1]:.1f})")

    # --- State ---
    edge_points_img: List[Tuple[float, float]] = []
    edge_points_mat: List[Tuple[float, float]] = []
    edge_artists: List[Any] = []
    state: Dict[str, Any] = {"mode": "click", "solved": False}

    # --- Setup matplotlib figure ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_title(
        f"Lens Calibration: {camera_id}  |  Click mat edges  |  "
        "u=undo  c=clear  s=solve  q=quit",
        fontsize=10,
    )
    ax.imshow(frame_rgb)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    # Draw existing 4 corners
    for cid, (cx, cy) in zip(corner_ids, corners_img):
        ax.plot(cx, cy, "rs", markersize=10)
        ax.annotate(
            cid, (cx, cy), textcoords="offset points", xytext=(5, 5),
            fontsize=8, color="red", fontweight="bold",
        )

    # Draw edge lines (straight lines between corners — these show the
    # "expected" straight edges; deviation of the actual mat edge from these
    # lines is the distortion signal)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_labels = ["top", "right", "bottom", "left"]
    for (a_idx, b_idx), label in zip(edge_pairs, edge_labels):
        ax.plot(
            [corners_img[a_idx][0], corners_img[b_idx][0]],
            [corners_img[a_idx][1], corners_img[b_idx][1]],
            "r--", linewidth=1, alpha=0.5,
        )

    fig.canvas.draw()

    def _redraw_edge_points() -> None:
        for a in edge_artists:
            a.remove()
        edge_artists.clear()
        for i, (ex, ey) in enumerate(edge_points_img):
            (dot,) = ax.plot(ex, ey, "co", markersize=6)
            edge_artists.append(dot)
            txt = ax.annotate(
                f"{i}", (ex, ey), textcoords="offset points", xytext=(3, 3),
                fontsize=7, color="cyan",
            )
            edge_artists.append(txt)
        fig.canvas.draw_idle()

    # --- Click handler ---
    def on_click(event: Any) -> None:
        if state["mode"] != "click":
            return
        if event.inaxes != ax or event.button != 1:
            return
        cx, cy = float(event.xdata), float(event.ydata)
        edge_idx, t, world_xy = _snap_to_nearest_edge(
            (cx, cy), corners_img, corners_mat,
        )
        edge_points_img.append((cx, cy))
        edge_points_mat.append(world_xy)
        edge_name = edge_labels[edge_idx]
        print(
            f"  #{len(edge_points_img)-1} edge={edge_name} t={t:.3f} "
            f"pixel=({cx:.1f},{cy:.1f}) world=({world_xy[0]:.2f},{world_xy[1]:.2f})"
        )
        _redraw_edge_points()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # --- Key handler ---
    def on_key(event: Any) -> None:
        k = event.key
        if k == "q":
            print("[lens_cal] Quit without saving.")
            plt.close(fig)
            return

        if k == "u" and state["mode"] == "click":
            if edge_points_img:
                edge_points_img.pop()
                edge_points_mat.pop()
                print(f"[lens_cal] Undo → {len(edge_points_img)} edge points remain")
                _redraw_edge_points()
            return

        if k == "c" and state["mode"] == "click":
            edge_points_img.clear()
            edge_points_mat.clear()
            print("[lens_cal] Cleared all edge points (4 corners retained)")
            _redraw_edge_points()
            return

        if k == "s" and state["mode"] == "click":
            total = 4 + len(edge_points_img)
            if total < 8:
                print(f"[lens_cal] Need >= 8 total points (have {total}). Add more edge points.")
                return
            print(f"[lens_cal] Solving with {total} points ({len(edge_points_img)} edge + 4 corners)...")
            _solve(fig, ax)
            return

        if k == "a" and state["mode"] == "verify":
            _accept()
            plt.close(fig)
            return

        if k == "r" and state["mode"] == "verify":
            print("[lens_cal] Redo — returning to click mode")
            state["mode"] = "click"
            edge_points_img.clear()
            edge_points_mat.clear()
            # Restore original frame
            ax.clear()
            ax.set_title(
                f"Lens Calibration: {camera_id}  |  Click mat edges  |  "
                "u=undo  c=clear  s=solve  q=quit",
                fontsize=10,
            )
            ax.imshow(frame_rgb)
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            for cid, (ccx, ccy) in zip(corner_ids, corners_img):
                ax.plot(ccx, ccy, "rs", markersize=10)
                ax.annotate(
                    cid, (ccx, ccy), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="red", fontweight="bold",
                )
            for (a_idx, b_idx), _label in zip(edge_pairs, edge_labels):
                ax.plot(
                    [corners_img[a_idx][0], corners_img[b_idx][0]],
                    [corners_img[a_idx][1], corners_img[b_idx][1]],
                    "r--", linewidth=1, alpha=0.5,
                )
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Solve callback ---
    def _solve(fig_: Any, ax_: Any) -> None:
        # Build point arrays: 4 corners + N edge points
        all_img = list(corners_img) + list(edge_points_img)
        all_mat = list(corners_mat) + list(edge_points_mat)

        obj_pts = np.array(
            [[mx, my, 0.0] for (mx, my) in all_mat], dtype=np.float32,
        )
        img_pts = np.array(
            [[ux, uy] for (ux, uy) in all_img], dtype=np.float32,
        )

        ret, K, dist_full, rvecs, tvecs = cv2.calibrateCamera(
            [obj_pts],
            [img_pts],
            imageSize=(img_w, img_h),
            cameraMatrix=None,
            distCoeffs=None,
            flags=cv2.CALIB_FIX_TANGENT_DIST,
        )

        dist_4 = dist_full.ravel()[:4]  # [k1, k2, p1, p2]
        state["K"] = K
        state["dist"] = dist_full
        state["dist_4"] = dist_4
        state["rms"] = ret
        state["num_points"] = len(all_img)

        print(f"[lens_cal] RMS reprojection error: {ret:.4f} px")
        print(f"[lens_cal] K:\n{K}")
        print(f"[lens_cal] dist (k1,k2,p1,p2): {dist_4}")

        # Show undistorted frame
        undistorted_bgr = cv2.undistort(frame_bgr, K, dist_full)
        undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

        ax_.clear()
        ax_.set_title(
            f"Undistorted: {camera_id}  |  RMS={ret:.3f}px  |  "
            "a=accept  r=redo  q=quit",
            fontsize=10,
        )
        ax_.imshow(undistorted_rgb)
        ax_.set_xlim(0, img_w)
        ax_.set_ylim(img_h, 0)

        # Re-project all points onto undistorted frame for visual check
        proj_pts, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3),
            rvecs[0], tvecs[0], K, dist_full,
        )
        proj_pts = proj_pts.reshape(-1, 2)
        # Draw original corners
        for i, cid in enumerate(corner_ids):
            ax_.plot(proj_pts[i, 0], proj_pts[i, 1], "gs", markersize=10)
            ax_.annotate(
                cid, (proj_pts[i, 0], proj_pts[i, 1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="lime", fontweight="bold",
            )
        # Draw edge points
        for i in range(4, len(proj_pts)):
            ax_.plot(proj_pts[i, 0], proj_pts[i, 1], "co", markersize=5)

        # Draw straight edge lines between re-projected corners
        for a_idx, b_idx in edge_pairs:
            ax_.plot(
                [proj_pts[a_idx, 0], proj_pts[b_idx, 0]],
                [proj_pts[a_idx, 1], proj_pts[b_idx, 1]],
                "g-", linewidth=1, alpha=0.7,
            )

        fig_.canvas.draw_idle()
        state["mode"] = "verify"
        state["solved"] = True

    # --- Accept callback ---
    def _accept() -> None:
        K = state["K"]
        dist_4 = state["dist_4"]
        rms = state["rms"]
        num_pts = state["num_points"]

        existing = json.loads(homography_json_path.read_text(encoding="utf-8"))
        existing["camera_matrix"] = K.tolist()
        existing["dist_coefficients"] = dist_4.tolist()
        existing["lens_calibration"] = {
            "rms_reprojection_error": float(rms),
            "num_points": num_pts,
            "flags": "CALIB_FIX_TANGENT_DIST",
            "image_size": [img_w, img_h],
            "created_at": _iso_utc_now(),
        }

        with homography_json_path.open("w") as f:
            json.dump(existing, f, indent=2)

        print(f"[lens_cal] Wrote K + dist to {homography_json_path}")
        print(f"[lens_cal] RMS={rms:.4f}px, {num_pts} points, flags=CALIB_FIX_TANGENT_DIST")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m calibration_pipeline.lens_calibration",
        description=(
            "Interactive lens calibration tool (CP16b). "
            "Estimates K + distortion coefficients from mat edge clicks. "
            "Writes results to configs/cameras/<camera>/homography.json."
        ),
    )
    p.add_argument("--camera", required=True, help="Camera id (e.g. J_EDEw)")
    p.add_argument("--video", required=True, help="Path to mp4 to grab the first frame from")
    p.add_argument(
        "--configs-root",
        default="configs",
        help="Repo configs root (default: ./configs)",
    )

    args = p.parse_args()
    configs_root = Path(args.configs_root)
    homography_json = configs_root / "cameras" / args.camera / "homography.json"

    _run_interactive(
        camera_id=args.camera,
        homography_json_path=homography_json,
        video_path=Path(args.video),
    )


if __name__ == "__main__":
    main()
