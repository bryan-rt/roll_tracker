"""Lens calibration — estimate K + distortion from mat edge auto-detection.

Interactive matplotlib tool. Uses existing 4-corner correspondences from
homography.json as seed data. Auto-detects additional edge points via 1D
gradient analysis along perpendicular profiles. Manual clicks supplement
auto-detected points where needed.

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
  left-click  — add manual edge point (auto-snaps to nearest mat edge)
  right-click — delete nearest point (auto or manual)
  d           — delete nearest point to last mouse position
  u           — undo last manual addition
  c           — clear all points + re-run auto-detection
  s           — solve (run calibrateCamera, display undistorted result)
  a           — accept (write K + dist to homography.json)
  r           — redo (back to point clicking)
  q           — quit without saving
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

    Returns (edge_idx, t, world_xy).
    """
    cx, cy = click_xy
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
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

    a_idx, b_idx = edges[best_edge]
    mx_a, my_a = corners_mat[a_idx]
    mx_b, my_b = corners_mat[b_idx]
    world_x = mx_a + best_t * (mx_b - mx_a)
    world_y = my_a + best_t * (my_b - my_a)

    return best_edge, best_t, (world_x, world_y)


# ---------------------------------------------------------------------------
# Auto edge detection
# ---------------------------------------------------------------------------

_EDGE_NAMES = ["top", "right", "bottom", "left"]


def _auto_detect_edge_points(
    frame_gray: "np.ndarray",
    corners_img: List[Tuple[float, float]],
    corners_mat: List[Tuple[float, float]],
    *,
    sample_spacing_px: int = 15,
    profile_half_width_px: int = 30,
    min_gradient_strength: float = 15.0,
    max_deviation_px: float = 40.0,
    edge_margin: float = 0.05,
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[int],
    Dict[str, int],
]:
    """Auto-detect mat edge points via perpendicular gradient profiles.

    Parameters
    ----------
    frame_gray : grayscale frame (uint8, H x W)
    corners_img : 4 corner pixel coords [tl, tr, br, bl]
    corners_mat : 4 corner world coords, same ordering
    sample_spacing_px : interval between sample points along each edge
    profile_half_width_px : half-width of perpendicular profile (±N pixels)
    min_gradient_strength : reject detections below this gradient magnitude
    max_deviation_px : reject detections farther than this from the straight
        edge line (generous to allow for distortion)
    edge_margin : skip samples within this fraction of each edge endpoint
        (avoids corners where two edges meet)

    Returns
    -------
    (detected_img, detected_mat, detected_edge_idx, stats)
    """
    img_h, img_w = frame_gray.shape[:2]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    detected_img: List[Tuple[float, float]] = []
    detected_mat: List[Tuple[float, float]] = []
    detected_edge_idx: List[int] = []
    per_edge_counts: Dict[str, int] = {}
    total_rejected = 0

    for edge_i, (a_idx, b_idx) in enumerate(edges):
        edge_name = _EDGE_NAMES[edge_i]
        ax, ay = corners_img[a_idx]
        bx, by = corners_img[b_idx]

        # Edge direction and perpendicular
        dx, dy = bx - ax, by - ay
        edge_len = float(np.hypot(dx, dy))
        if edge_len < 1.0:
            per_edge_counts[edge_name] = 0
            continue
        # Unit tangent and normal
        tx, ty = dx / edge_len, dy / edge_len
        # Perpendicular (rotate 90 degrees CCW — points "outward" from mat for
        # top/left edges, "inward" for bottom/right. Direction doesn't matter
        # for finding the gradient peak, only the magnitude.)
        nx, ny = -ty, tx

        # Sample along the edge, skipping margins near corners
        n_samples = max(1, int(edge_len / sample_spacing_px))
        edge_count = 0
        for si in range(n_samples):
            t_frac = edge_margin + (1.0 - 2 * edge_margin) * (si + 0.5) / n_samples
            # Point on the straight edge line
            sx = ax + t_frac * dx
            sy = ay + t_frac * dy

            # Extract 1D profile along the perpendicular
            hw = profile_half_width_px
            profile = np.zeros(2 * hw + 1, dtype=np.float64)
            valid = True
            for pi in range(-hw, hw + 1):
                px_f = sx + pi * nx
                py_f = sy + pi * ny
                px_i = int(round(px_f))
                py_i = int(round(py_f))
                if px_i < 0 or px_i >= img_w or py_i < 0 or py_i >= img_h:
                    valid = False
                    break
                profile[pi + hw] = float(frame_gray[py_i, px_i])

            if not valid:
                total_rejected += 1
                continue

            # Compute gradient and find strongest peak
            grad = np.gradient(profile)
            abs_grad = np.abs(grad)
            peak_idx = int(np.argmax(abs_grad))
            peak_strength = float(abs_grad[peak_idx])

            if peak_strength < min_gradient_strength:
                total_rejected += 1
                continue

            # Sub-pixel refinement via parabola fit around peak
            sub_offset = 0.0
            if 1 <= peak_idx <= len(grad) - 2:
                g_m1 = abs_grad[peak_idx - 1]
                g_0 = abs_grad[peak_idx]
                g_p1 = abs_grad[peak_idx + 1]
                denom = 2.0 * (2.0 * g_0 - g_m1 - g_p1)
                if abs(denom) > 1e-9:
                    sub_offset = (g_m1 - g_p1) / denom

            # Convert profile index to perpendicular offset from edge line
            perp_offset = (peak_idx + sub_offset) - hw

            # Actual pixel location of detected edge point
            det_x = sx + perp_offset * nx
            det_y = sy + perp_offset * ny

            # Quality filter: reject if too far from the straight edge line
            dist_from_line = abs(perp_offset)
            if dist_from_line > max_deviation_px:
                total_rejected += 1
                continue

            # Compute world coordinate via parametric t
            # The detected point's t is approximately the same as the sample's
            # t_frac (the perpendicular shift doesn't change position along edge)
            mx_a, my_a = corners_mat[a_idx]
            mx_b, my_b = corners_mat[b_idx]
            world_x = mx_a + t_frac * (mx_b - mx_a)
            world_y = my_a + t_frac * (my_b - my_a)

            detected_img.append((float(det_x), float(det_y)))
            detected_mat.append((world_x, world_y))
            detected_edge_idx.append(edge_i)
            edge_count += 1

        per_edge_counts[edge_name] = edge_count

    stats = {
        "per_edge": per_edge_counts,
        "total_detected": len(detected_img),
        "total_rejected": total_rejected,
    }
    return detected_img, detected_mat, detected_edge_idx, stats


# ---------------------------------------------------------------------------
# Interactive calibration
# ---------------------------------------------------------------------------

def _run_interactive(
    camera_id: str,
    homography_json_path: Path,
    video_path: Path,
) -> None:
    """Interactive lens calibration UI with auto edge detection."""
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
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = frame_rgb.shape[:2]
    print(f"[lens_cal] Frame size: {img_w}x{img_h}")
    print(f"[lens_cal] Loaded 4 corners from {homography_json_path}")
    for cid, ip, mp in zip(corner_ids, corners_img, corners_mat):
        print(f"  {cid}: pixel=({ip[0]:.1f}, {ip[1]:.1f})  world=({mp[0]:.1f}, {mp[1]:.1f})")

    # --- State ---
    # Auto-detected points (can be deleted but not individually undone)
    auto_img: List[Tuple[float, float]] = []
    auto_mat: List[Tuple[float, float]] = []
    auto_edge_idx: List[int] = []
    # Manual points (support undo)
    manual_img: List[Tuple[float, float]] = []
    manual_mat: List[Tuple[float, float]] = []
    edge_artists: List[Any] = []
    state: Dict[str, Any] = {
        "mode": "click",
        "solved": False,
        "last_mouse_xy": None,
        "auto_stats": {},
    }

    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def _run_auto_detect() -> None:
        """Run auto-detection and populate auto_* lists."""
        auto_img.clear()
        auto_mat.clear()
        auto_edge_idx.clear()
        det_img, det_mat, det_eidx, stats = _auto_detect_edge_points(
            frame_gray, corners_img, corners_mat,
        )
        auto_img.extend(det_img)
        auto_mat.extend(det_mat)
        auto_edge_idx.extend(det_eidx)
        state["auto_stats"] = stats
        print(
            f"[lens_cal] Auto-detected {stats['total_detected']} points "
            f"(top: {stats['per_edge'].get('top', 0)}, "
            f"right: {stats['per_edge'].get('right', 0)}, "
            f"bottom: {stats['per_edge'].get('bottom', 0)}, "
            f"left: {stats['per_edge'].get('left', 0)}). "
            f"{stats['total_rejected']} rejected."
        )

    # Initial auto-detection
    _run_auto_detect()

    # --- Setup matplotlib figure ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    def _setup_base_view() -> None:
        """Draw frame, corners, edge lines."""
        ax.clear()
        ax.set_title(
            f"Lens Calibration: {camera_id}  |  "
            f"{len(auto_img)} auto + {len(manual_img)} manual pts  |  "
            "click=add  right-click/d=del  c=reset  s=solve  q=quit",
            fontsize=9,
        )
        ax.imshow(frame_rgb)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        for cid, (cx, cy) in zip(corner_ids, corners_img):
            ax.plot(cx, cy, "rs", markersize=10)
            ax.annotate(
                cid, (cx, cy), textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="red", fontweight="bold",
            )
        for (a_idx, b_idx) in edge_pairs:
            ax.plot(
                [corners_img[a_idx][0], corners_img[b_idx][0]],
                [corners_img[a_idx][1], corners_img[b_idx][1]],
                "r--", linewidth=1, alpha=0.5,
            )

    _setup_base_view()

    def _redraw_all_points() -> None:
        for a in edge_artists:
            a.remove()
        edge_artists.clear()
        # Draw auto-detected points (cyan, smaller)
        for ex, ey in auto_img:
            (dot,) = ax.plot(ex, ey, "co", markersize=4, alpha=0.7)
            edge_artists.append(dot)
        # Draw manual points (cyan, larger, with index)
        for i, (ex, ey) in enumerate(manual_img):
            (dot,) = ax.plot(ex, ey, "c^", markersize=8)
            edge_artists.append(dot)
            txt = ax.annotate(
                f"m{i}", (ex, ey), textcoords="offset points", xytext=(3, 3),
                fontsize=7, color="yellow",
            )
            edge_artists.append(txt)
        # Update title with counts
        ax.set_title(
            f"Lens Calibration: {camera_id}  |  "
            f"{len(auto_img)} auto + {len(manual_img)} manual pts  |  "
            "click=add  right-click/d=del  c=reset  s=solve  q=quit",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    _redraw_all_points()
    fig.canvas.draw()

    # --- Click handler ---
    def on_click(event: Any) -> None:
        if state["mode"] != "click":
            return
        if event.inaxes != ax:
            return
        cx, cy = float(event.xdata), float(event.ydata)
        state["last_mouse_xy"] = (cx, cy)

        if event.button == 1:  # Left click — add manual point
            edge_idx, t, world_xy = _snap_to_nearest_edge(
                (cx, cy), corners_img, corners_mat,
            )
            manual_img.append((cx, cy))
            manual_mat.append(world_xy)
            print(
                f"  manual #{len(manual_img)-1} edge={_EDGE_NAMES[edge_idx]} "
                f"t={t:.3f} pixel=({cx:.1f},{cy:.1f}) "
                f"world=({world_xy[0]:.2f},{world_xy[1]:.2f})"
            )
            _redraw_all_points()

        elif event.button == 3:  # Right click — delete nearest point
            _delete_nearest_point(cx, cy)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Motion handler to track mouse position for 'd' key
    def on_motion(event: Any) -> None:
        if event.inaxes == ax and event.xdata is not None:
            state["last_mouse_xy"] = (float(event.xdata), float(event.ydata))

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    def _delete_nearest_point(mx: float, my: float) -> None:
        """Delete the auto or manual point nearest to (mx, my)."""
        best_dist = float("inf")
        best_src = ""  # "auto" or "manual"
        best_idx = -1

        for i, (px, py) in enumerate(auto_img):
            d = float(np.hypot(mx - px, my - py))
            if d < best_dist:
                best_dist = d
                best_src = "auto"
                best_idx = i

        for i, (px, py) in enumerate(manual_img):
            d = float(np.hypot(mx - px, my - py))
            if d < best_dist:
                best_dist = d
                best_src = "manual"
                best_idx = i

        if best_idx < 0 or best_dist > 30.0:
            print("[lens_cal] No point near cursor to delete.")
            return

        if best_src == "auto":
            px, py = auto_img.pop(best_idx)
            auto_mat.pop(best_idx)
            auto_edge_idx.pop(best_idx)
            print(f"[lens_cal] Deleted auto point at ({px:.1f},{py:.1f})")
        else:
            px, py = manual_img.pop(best_idx)
            manual_mat.pop(best_idx)
            print(f"[lens_cal] Deleted manual point at ({px:.1f},{py:.1f})")

        _redraw_all_points()

    # --- Key handler ---
    def on_key(event: Any) -> None:
        k = event.key
        if k == "q":
            print("[lens_cal] Quit without saving.")
            plt.close(fig)
            return

        if k == "d" and state["mode"] == "click":
            xy = state.get("last_mouse_xy")
            if xy:
                _delete_nearest_point(xy[0], xy[1])
            return

        if k == "u" and state["mode"] == "click":
            if manual_img:
                manual_img.pop()
                manual_mat.pop()
                print(f"[lens_cal] Undo manual → {len(manual_img)} manual points remain")
                _redraw_all_points()
            return

        if k == "c" and state["mode"] == "click":
            manual_img.clear()
            manual_mat.clear()
            _run_auto_detect()
            _setup_base_view()
            _redraw_all_points()
            return

        if k == "s" and state["mode"] == "click":
            total = 4 + len(auto_img) + len(manual_img)
            if total < 8:
                print(f"[lens_cal] Need >= 8 total points (have {total}). Add more edge points.")
                return
            print(
                f"[lens_cal] Solving with {total} points "
                f"(4 corners + {len(auto_img)} auto + {len(manual_img)} manual)..."
            )
            _solve(fig, ax)
            return

        if k == "a" and state["mode"] == "verify":
            _accept()
            plt.close(fig)
            return

        if k == "r" and state["mode"] == "verify":
            print("[lens_cal] Redo — returning to click mode")
            state["mode"] = "click"
            manual_img.clear()
            manual_mat.clear()
            _run_auto_detect()
            _setup_base_view()
            _redraw_all_points()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Solve callback ---
    def _solve(fig_: Any, ax_: Any) -> None:
        all_img = list(corners_img) + list(auto_img) + list(manual_img)
        all_mat = list(corners_mat) + list(auto_mat) + list(manual_mat)

        obj_pts = np.array(
            [[mx, my, 0.0] for (mx, my) in all_mat], dtype=np.float32,
        )
        img_pts = np.array(
            [[ux, uy] for (ux, uy) in all_img], dtype=np.float32,
        )

        calib_flags = (
            cv2.CALIB_FIX_TANGENT_DIST
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_ASPECT_RATIO
        )

        ret, K, dist_full, rvecs, tvecs = cv2.calibrateCamera(
            [obj_pts],
            [img_pts],
            imageSize=(img_w, img_h),
            cameraMatrix=None,
            distCoeffs=None,
            flags=calib_flags,
        )

        dist_4 = dist_full.ravel()[:4]  # [k1, k2, p1, p2]
        state["K"] = K
        state["dist"] = dist_full
        state["dist_4"] = dist_4
        state["rms"] = ret
        state["num_points"] = len(all_img)
        state["num_auto"] = len(auto_img)
        state["num_manual"] = len(manual_img)

        print(f"[lens_cal] RMS reprojection error: {ret:.4f} px")
        print(f"[lens_cal] K:\n{K}")
        print(f"[lens_cal] dist (k1,k2,p1,p2): {dist_4}")

        # Show undistorted frame
        undistorted_bgr = cv2.undistort(frame_bgr, K, dist_full)
        undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

        ax_.clear()
        ax_.set_title(
            f"Undistorted: {camera_id}  |  RMS={ret:.3f}px  |  "
            f"{len(all_img)} pts (4+{len(auto_img)}auto+{len(manual_img)}man)  |  "
            "a=accept  r=redo  q=quit",
            fontsize=9,
        )
        ax_.imshow(undistorted_rgb)
        ax_.set_xlim(0, img_w)
        ax_.set_ylim(img_h, 0)

        # Re-project all points
        proj_pts, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3),
            rvecs[0], tvecs[0], K, dist_full,
        )
        proj_pts = proj_pts.reshape(-1, 2)
        for i, cid in enumerate(corner_ids):
            ax_.plot(proj_pts[i, 0], proj_pts[i, 1], "gs", markersize=10)
            ax_.annotate(
                cid, (proj_pts[i, 0], proj_pts[i, 1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="lime", fontweight="bold",
            )
        for i in range(4, len(proj_pts)):
            ax_.plot(proj_pts[i, 0], proj_pts[i, 1], "co", markersize=4)

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
        num_auto = state.get("num_auto", 0)
        num_manual = state.get("num_manual", 0)
        auto_stats = state.get("auto_stats", {})

        existing = json.loads(homography_json_path.read_text(encoding="utf-8"))
        existing["camera_matrix"] = K.tolist()
        existing["dist_coefficients"] = dist_4.tolist()
        existing["lens_calibration"] = {
            "rms_reprojection_error": float(rms),
            "num_points": num_pts,
            "auto_detected_points": num_auto,
            "manual_points": num_manual,
            "rejected_points": auto_stats.get("total_rejected", 0),
            "points_per_edge": auto_stats.get("per_edge", {}),
            "flags": "CALIB_FIX_TANGENT_DIST|CALIB_FIX_PRINCIPAL_POINT|CALIB_FIX_ASPECT_RATIO",
            "image_size": [img_w, img_h],
            "created_at": _iso_utc_now(),
        }

        with homography_json_path.open("w") as f:
            json.dump(existing, f, indent=2)

        print(f"[lens_cal] Wrote K + dist to {homography_json_path}")
        print(
            f"[lens_cal] RMS={rms:.4f}px, {num_pts} pts "
            f"({num_auto} auto + {num_manual} manual + 4 corners)"
        )

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m calibration_pipeline.lens_calibration",
        description=(
            "Interactive lens calibration tool (CP16b). "
            "Auto-detects mat edge points via gradient analysis, "
            "then estimates K + distortion coefficients. "
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
