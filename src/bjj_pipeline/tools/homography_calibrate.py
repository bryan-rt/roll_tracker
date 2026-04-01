# src/bjj_pipeline/tools/homography_calibrate.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from math import isfinite, cos, sin, pi, sqrt, hypot

# CP19: Import mat line detection functions for H refinement.
# These are stable pure functions (CP18); leading underscores are internal to
# that module but the API is frozen and well-tested.
from calibration_pipeline.mat_line_detection import (
    _detect_lines_in_frame,
    _merge_collinear_segments,
    _match_lines_to_polylines,
    DetectedMatLine,
)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_3x3(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3,3), got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("Homography contains non-finite values.")
    return H


def _normalize_h(H: np.ndarray) -> np.ndarray:
    """Normalize so H[2,2] == 1 when possible (common convention)."""
    H = _ensure_3x3(H)
    denom = H[2, 2]
    if abs(denom) > 1e-12:
        H = H / denom
    return H


def _default_homography_json_path(configs_root: Path, camera_id: str) -> Path:
    return configs_root / "cameras" / camera_id / "homography.json"


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _load_existing_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_lens_calibration(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load K + dist from existing homography.json if present and non-null."""
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cm = data.get("camera_matrix")
        dc = data.get("dist_coefficients")
        if cm is not None and dc is not None:
            return np.asarray(cm, dtype=np.float64), np.asarray(dc, dtype=np.float64)
    except (json.JSONDecodeError, OSError):
        pass
    return None, None


def _write_homography_json(
    out_path: Path,
    camera_id: str,
    H: np.ndarray,
    source: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read-then-merge: preserve fields from existing file (e.g. camera_matrix,
    # dist_coefficients written by lens_calibration).
    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    payload: Dict[str, Any] = {
        **existing,
        "H": _normalize_h(H).astype(float).tolist(),
        "camera_id": camera_id,
        "source": source,
        "created_at": _iso_utc_now(),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = v

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"✔ Wrote homography → {out_path}")


# -------------------------
# Optional: interactive mode
# -------------------------
@dataclass
class ClickPairs:
    image_points_px: List[Tuple[float, float]]
    mat_points: List[Tuple[float, float]]


def _try_load_mat_blueprint(mat_blueprint_path: Path) -> Optional[Dict[str, Any]]:
    if not mat_blueprint_path.exists():
        return None
    try:
        with mat_blueprint_path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_rects_from_blueprint(blueprint: Any) -> List[Tuple[float, float, float, float, str]]:
    """
    Evidence-based parser for configs/mat_blueprint.json:
      the file is a JSON array of rectangle specs:
        [{"label": str, "x": number, "y": number, "width": number, "height": number}, ...]
    Returns: [(x, y, w, h, label), ...] with numeric validity checks applied.
    """
    if not isinstance(blueprint, list):
        return []
    rects: List[Tuple[float, float, float, float, str]] = []
    for r in blueprint:
        if not isinstance(r, dict):
            continue
        x = r.get("x"); y = r.get("y"); w = r.get("width"); h = r.get("height")
        if not all(isinstance(v, (int, float)) and isfinite(float(v)) for v in (x, y, w, h)):
            continue
        x = float(x); y = float(y); w = float(w); h = float(h)
        if w <= 0 or h <= 0:
            continue
        label = r.get("label") if isinstance(r.get("label"), str) else ""
        rects.append((x, y, w, h, label))
    return rects

def _render_mat_blueprint_rects(ax, blueprint: Any) -> None:
    """
    Evidence-based renderer for configs/mat_blueprint.json:
    the file is a JSON array of rectangle specs:
      [{"label": str, "x": number, "y": number, "width": number, "height": number}, ...]
    """
    try:
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError:
        # If matplotlib isn't available, caller will already error in interactive mode.
        return

    if not isinstance(blueprint, list):
        print("[D7] mat_blueprint: expected a JSON list of rectangles; got", type(blueprint).__name__)
        return []
    rects = _parse_rects_from_blueprint(blueprint)

    if not rects:
        print("[D7] mat_blueprint: no valid rectangles found to render.")
        return []

    # Draw rectangles
    for (x, y, w, h, label) in rects:
        ax.add_patch(Rectangle((x, y), w, h, fill=False))
        if label.strip():
            ax.text(x + 0.5 * w, y + 0.5 * h, label, ha="center", va="center")

    # Fit view to rectangles with padding
    xs = [x for (x, _, w, _, _) in rects] + [x + w for (x, _, w, _, _) in rects]
    ys = [y for (_, y, _, h, _) in rects] + [y + h for (_, y, _, h, _) in rects]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = max(1.0, 0.05 * (xmax - xmin))
    pad_y = max(1.0, 0.05 * (ymax - ymin))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    return rects


def _rect_union_bounds(rects: List[Tuple[float, float, float, float, str]]) -> Tuple[float, float, float, float]:
    xs = [x for (x, _, w, _, _) in rects] + [x + w for (x, _, w, _, _) in rects]
    ys = [y for (_, y, _, h, _) in rects] + [y + h for (_, y, _, h, _) in rects]
    return (min(xs), min(ys), max(xs), max(ys))


def _make_union_mask(
    rects: List[Tuple[float, float, float, float, str]],
    *,
    step_m: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize the UNION of rectangles into a boolean mask (no shapely dependency).
    Returns: mask, xs, ys where:
      mask shape = (len(ys), len(xs)) and mask[j,i] indicates point (xs[i], ys[j]) is inside union.
    """
    xs = np.arange(xmin, xmax + 1e-9, step_m, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, step_m, dtype=float)
    mask = np.zeros((ys.size, xs.size), dtype=bool)
    for (x, y, w, h, _) in rects:
        x2 = x + w
        y2 = y + h
        xi0 = int(np.searchsorted(xs, x, side="left"))
        xi1 = int(np.searchsorted(xs, x2, side="right"))
        yi0 = int(np.searchsorted(ys, y, side="left"))
        yi1 = int(np.searchsorted(ys, y2, side="right"))
        mask[yi0:yi1, xi0:xi1] = True
    return mask, xs, ys


def _iter_masked_polylines_constant_x(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x_idx: int,
) -> List[np.ndarray]:
    """
    For a fixed x column index, return a list of polyline point arrays (mat coords)
    representing contiguous segments where mask==True.
    """
    col = mask[:, x_idx]
    segments: List[np.ndarray] = []
    start = None
    for j, inside in enumerate(col):
        if inside and start is None:
            start = j
        if (not inside) and start is not None:
            js = np.arange(start, j, dtype=int)
            pts = np.column_stack([np.full(js.shape, xs[x_idx]), ys[js]])
            segments.append(pts)
            start = None
    if start is not None:
        js = np.arange(start, col.size, dtype=int)
        pts = np.column_stack([np.full(js.shape, xs[x_idx]), ys[js]])
        segments.append(pts)
    return segments


def _iter_masked_polylines_constant_y(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    y_idx: int,
) -> List[np.ndarray]:
    row = mask[y_idx, :]
    segments: List[np.ndarray] = []
    start = None
    for i, inside in enumerate(row):
        if inside and start is None:
            start = i
        if (not inside) and start is not None:
            is_ = np.arange(start, i, dtype=int)
            pts = np.column_stack([xs[is_], np.full(is_.shape, ys[y_idx])])
            segments.append(pts)
            start = None
    if start is not None:
        is_ = np.arange(start, row.size, dtype=int)
        pts = np.column_stack([xs[is_], np.full(is_.shape, ys[y_idx])])
        segments.append(pts)
    return segments


def _extract_contiguous_runs(items: list) -> list:
    """Split a list with None gaps into sublists of consecutive non-None items."""
    runs = []
    current = []
    for item in items:
        if item is not None:
            current.append(item)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _generate_projected_polylines(
    H_mat_to_img: np.ndarray,
    rects: List[Tuple[float, float, float, float, str]],
    image_wh: Tuple[int, int],
    sample_spacing: float = 0.25,
    frame_margin: float = 50.0,
) -> Dict[str, Any]:
    """Generate densely sampled projected polylines for all panel edges.

    Projects world-space panel edge points through H_mat_to_img to pixel space.
    Filters to in-frame points, extracts contiguous visible segments.
    """
    import cv2 as _cv2  # noqa

    width, height = image_wh

    # Compute all unique panel edges
    edges_set: set = set()
    for (x, y, w, h, _) in rects:
        corners = [(float(x), float(y)), (float(x + w), float(y)),
                   (float(x + w), float(y + h)), (float(x), float(y + h))]
        for i in range(4):
            c1, c2 = corners[i], corners[(i + 1) % 4]
            normalized = tuple(sorted([c1, c2]))
            edges_set.add(normalized)
    all_edges = [((e[0][0], e[0][1]), (e[1][0], e[1][1])) for e in edges_set]

    polylines = []
    H64 = np.asarray(H_mat_to_img, dtype=np.float64)

    for edge_idx, ((wx1, wy1), (wx2, wy2)) in enumerate(all_edges):
        edge_len = ((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2) ** 0.5
        n_samples = max(2, int(edge_len / sample_spacing))

        projected_points: list = []
        for k in range(n_samples):
            t = k / max(1, n_samples - 1)
            wx = wx1 + t * (wx2 - wx1)
            wy = wy1 + t * (wy2 - wy1)

            world_pt = np.array([[[wx, wy]]], dtype=np.float32)
            pixel_pt = _cv2.perspectiveTransform(world_pt, H64)
            u, v = float(pixel_pt[0, 0, 0]), float(pixel_pt[0, 0, 1])

            if (isfinite(u) and isfinite(v)
                    and -frame_margin <= u <= width + frame_margin
                    and -frame_margin <= v <= height + frame_margin):
                projected_points.append([round(u, 1), round(v, 1)])
            else:
                projected_points.append(None)

        for run in _extract_contiguous_runs(projected_points):
            polylines.append({
                "edge_index": edge_idx,
                "world_start": [wx1, wy1],
                "world_end": [wx2, wy2],
                "pixel_points": run,
            })

    return {
        "image_wh": [width, height],
        "sample_spacing": sample_spacing,
        "polylines": polylines,
        "n_polylines": len(polylines),
        "n_edges_total": len(all_edges),
        "created_at": _iso_utc_now(),
    }


# ---------------------------------------------------------------------------
# CP19: Unified calibration — Phase A (polyline lens cal) + Phase B (H refinement)
# ---------------------------------------------------------------------------


def _redistort_points(
    pts_undistorted: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Convert undistorted pixel coords back to raw (distorted) pixel coords.

    Applies the radial distortion model: given a point in undistorted pixel
    space, computes where it would appear in the raw (distorted) image.
    Assumes tangential distortion = 0 (p1=p2=0), which matches our calibration.

    Args:
        pts_undistorted: (N, 2) points in undistorted pixel space
        K: 3x3 camera matrix
        dist: [k1, k2, ...] distortion coefficients (only k1, k2 used)

    Returns: (N, 2) points in raw pixel space
    """
    pts = np.asarray(pts_undistorted, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist, dtype=np.float64).ravel()
    k1 = float(d[0]) if len(d) > 0 else 0.0
    k2 = float(d[1]) if len(d) > 1 else 0.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize to camera coordinates
    x_n = (pts[:, 0] - cx) / fx
    y_n = (pts[:, 1] - cy) / fy

    # Apply radial distortion
    r2 = x_n ** 2 + y_n ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d = x_n * radial
    y_d = y_n * radial

    # Back to pixel coordinates
    raw = np.column_stack([x_d * fx + cx, y_d * fy + cy])
    return raw


def _recompute_h_for_space(
    mat_pts: np.ndarray,
    anchor_img_pts: np.ndarray,
    from_K: Optional[np.ndarray],
    from_dist: Optional[np.ndarray],
    to_K: Optional[np.ndarray],
    to_dist: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute H and anchor points for a different undistortion space.

    anchor_img_pts are in the pixel space defined by from_K/from_dist
    (undistorted with those params, or raw if both are None).

    Returns (H_mat_to_target, target_anchor_pts) where the target pixel
    space is defined by to_K/to_dist (raw if both None).
    """
    import cv2 as _cv2

    pts = np.asarray(anchor_img_pts, dtype=np.float64).reshape(-1, 2)

    # Step 1: Convert from current space to raw pixel space
    if from_K is not None and from_dist is not None:
        raw_pts = _redistort_points(pts, from_K, from_dist)
    else:
        raw_pts = pts.copy()

    # Step 2: Convert from raw to target space
    if to_K is not None and to_dist is not None:
        target_pts = _cv2.undistortPoints(
            raw_pts.reshape(-1, 1, 2).astype(np.float64),
            np.asarray(to_K, dtype=np.float64).reshape(3, 3),
            np.asarray(to_dist, dtype=np.float64).ravel(),
            P=np.asarray(to_K, dtype=np.float64).reshape(3, 3),
        ).reshape(-1, 2)
    else:
        target_pts = raw_pts

    # Compute new H for the target space
    H, _ = _cv2.findHomography(
        np.asarray(mat_pts, dtype=np.float64),
        target_pts.astype(np.float64),
        method=0,
    )
    return _ensure_3x3(H), target_pts


def _detect_edge_points_along_polylines(
    frame_gray: np.ndarray,
    polyline_data: Dict[str, Any],
    *,
    sample_spacing_px: int = 15,
    profile_half_width_px: int = 30,
    min_gradient_strength: float = 15.0,
    max_deviation_px: float = 40.0,
    edge_margin_frac: float = 0.05,
) -> Tuple[
    List[Tuple[float, float]],   # detected_img (pixel coords)
    List[Tuple[float, float]],   # detected_mat (world coords)
    List[int],                    # detected_edge_idx
    Dict[str, Any],               # stats
]:
    """Detect mat edge points via perpendicular gradient profiles along projected polylines.

    Generalizes _auto_detect_edge_points from lens_calibration.py: instead of 4
    straight edges from corner pairs, works with N projected polylines covering
    the full visible mat surface.
    """
    img_h, img_w = frame_gray.shape[:2]
    detected_img: List[Tuple[float, float]] = []
    detected_mat: List[Tuple[float, float]] = []
    detected_edge_idx: List[int] = []
    per_edge_counts: Dict[int, int] = {}
    total_rejected = 0

    for pl in polyline_data.get("polylines", []):
        edge_idx = pl["edge_index"]
        pixel_pts = pl["pixel_points"]
        ws = pl["world_start"]
        we = pl["world_end"]
        wx1, wy1 = float(ws[0]), float(ws[1])
        wx2, wy2 = float(we[0]), float(we[1])

        if len(pixel_pts) < 3:
            continue

        # Compute polyline arc-length in pixels for sample spacing
        arc_len = 0.0
        for i in range(1, len(pixel_pts)):
            dx = float(pixel_pts[i][0]) - float(pixel_pts[i - 1][0])
            dy = float(pixel_pts[i][1]) - float(pixel_pts[i - 1][1])
            arc_len += hypot(dx, dy)

        if arc_len < 2.0 * sample_spacing_px:
            continue

        n_samples = max(1, int(arc_len / sample_spacing_px))
        edge_count = 0
        hw = profile_half_width_px

        for si in range(n_samples):
            # Fractional position along polyline, skipping margins
            t_frac = edge_margin_frac + (1.0 - 2 * edge_margin_frac) * (si + 0.5) / n_samples

            # Find the polyline point at this fractional arc-length
            target_dist = t_frac * arc_len
            cum_dist = 0.0
            seg_i = 0
            for j in range(1, len(pixel_pts)):
                dx = float(pixel_pts[j][0]) - float(pixel_pts[j - 1][0])
                dy = float(pixel_pts[j][1]) - float(pixel_pts[j - 1][1])
                seg_len = hypot(dx, dy)
                if cum_dist + seg_len >= target_dist and seg_len > 1e-6:
                    seg_i = j - 1
                    break
                cum_dist += seg_len
            else:
                seg_i = max(0, len(pixel_pts) - 2)

            # Interpolate position on this segment
            p0x, p0y = float(pixel_pts[seg_i][0]), float(pixel_pts[seg_i][1])
            p1x, p1y = float(pixel_pts[seg_i + 1][0]), float(pixel_pts[seg_i + 1][1])
            seg_dx, seg_dy = p1x - p0x, p1y - p0y
            seg_len = hypot(seg_dx, seg_dy)
            if seg_len < 1e-6:
                total_rejected += 1
                continue
            local_t = (target_dist - cum_dist) / seg_len
            local_t = max(0.0, min(1.0, local_t))
            sx = p0x + local_t * seg_dx
            sy = p0y + local_t * seg_dy

            # Compute local tangent via central difference on nearby polyline points
            # Use the segment direction as tangent (robust for curved polylines)
            # For better tangent at interior points, use adjacent polyline segments
            if seg_i > 0 and seg_i + 1 < len(pixel_pts) - 1:
                # Central difference across two segments
                prev_x, prev_y = float(pixel_pts[seg_i][0]), float(pixel_pts[seg_i][1])
                next_x, next_y = float(pixel_pts[seg_i + 1][0]), float(pixel_pts[seg_i + 1][1])
                tx, ty = next_x - prev_x, next_y - prev_y
            else:
                tx, ty = seg_dx, seg_dy

            t_len = hypot(tx, ty)
            if t_len < 1e-6:
                total_rejected += 1
                continue
            tx, ty = tx / t_len, ty / t_len
            # Perpendicular (rotate 90°)
            nx, ny = -ty, tx

            # Extract 1D intensity profile along perpendicular
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

            # Gradient and strongest peak
            grad = np.gradient(profile)
            abs_grad = np.abs(grad)
            peak_idx = int(np.argmax(abs_grad))
            peak_strength = float(abs_grad[peak_idx])

            if peak_strength < min_gradient_strength:
                total_rejected += 1
                continue

            # Sub-pixel refinement via parabola fit
            sub_offset = 0.0
            if 1 <= peak_idx <= len(grad) - 2:
                g_m1 = abs_grad[peak_idx - 1]
                g_0 = abs_grad[peak_idx]
                g_p1 = abs_grad[peak_idx + 1]
                denom = 2.0 * (2.0 * g_0 - g_m1 - g_p1)
                if abs(denom) > 1e-9:
                    sub_offset = (g_m1 - g_p1) / denom

            perp_offset = (peak_idx + sub_offset) - hw

            # Detected pixel location
            det_x = sx + perp_offset * nx
            det_y = sy + perp_offset * ny

            if abs(perp_offset) > max_deviation_px:
                total_rejected += 1
                continue

            # World coordinate via parametric t along the blueprint edge
            world_x = wx1 + t_frac * (wx2 - wx1)
            world_y = wy1 + t_frac * (wy2 - wy1)

            detected_img.append((float(det_x), float(det_y)))
            detected_mat.append((world_x, world_y))
            detected_edge_idx.append(edge_idx)
            edge_count += 1

        per_edge_counts[edge_idx] = per_edge_counts.get(edge_idx, 0) + edge_count

    stats = {
        "per_edge": per_edge_counts,
        "total_detected": len(detected_img),
        "total_rejected": total_rejected,
        "n_edges_with_points": sum(1 for c in per_edge_counts.values() if c > 0),
    }
    return detected_img, detected_mat, detected_edge_idx, stats


def _polyline_lens_calibration(
    H_mat_to_img: np.ndarray,
    frame_bgr: np.ndarray,
    frame_gray: np.ndarray,
    rects: List[Tuple[float, float, float, float, str]],
    image_wh: Tuple[int, int],
    *,
    f_bounds: Tuple[float, float] = (200.0, 5000.0),
    k_bounds: Tuple[float, float] = (-10.0, 10.0),
    min_edge_points: int = 20,
    min_edges_with_points: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Lens calibration using edge points detected along projected polylines.

    Returns (K, dist_4, lens_metrics) or (None, None, metrics) if insufficient data.
    K is 3x3 camera matrix. dist_4 is [k1, k2, 0, 0].
    """
    import cv2 as _cv2
    from scipy.optimize import minimize as _sp_minimize

    img_w, img_h = image_wh

    # Generate projected polylines from H on the RAW (distorted) frame
    polyline_data = _generate_projected_polylines(
        H_mat_to_img=H_mat_to_img, rects=rects, image_wh=image_wh,
    )

    # Detect edge points along polylines
    det_img, det_mat, det_eidx, stats = _detect_edge_points_along_polylines(
        frame_gray, polyline_data,
    )

    n_pts = stats["total_detected"]
    n_edges = stats["n_edges_with_points"]

    if n_pts < min_edge_points or n_edges < min_edges_with_points:
        return None, None, {
            "reason": f"insufficient detections ({n_pts} pts, {n_edges} edges)",
            "n_edge_points": n_pts,
            "n_edges_with_points": n_edges,
        }

    # Build edge groups: edge_idx -> list of point indices
    edge_groups: Dict[int, List[int]] = {}
    for i, eidx in enumerate(det_eidx):
        edge_groups.setdefault(eidx, []).append(i)

    all_img_arr = np.array(det_img, dtype=np.float64)
    cx = img_w / 2.0
    cy = img_h / 2.0

    # Collinearity cost: undistort points, fit line per edge, sum squared residuals
    def _collinearity_cost(params: np.ndarray) -> float:
        f, k1, k2 = float(params[0]), float(params[1]), float(params[2])
        K_trial = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist_trial = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)

        pts = all_img_arr.reshape(-1, 1, 2)
        undist = _cv2.undistortPoints(pts, K_trial, dist_trial, P=K_trial)
        undist_2d = undist.reshape(-1, 2)

        total = 0.0
        for eidx, idxs in edge_groups.items():
            if len(idxs) < 2:
                continue
            edge_pts = undist_2d[idxs]
            centroid = edge_pts.mean(axis=0)
            centered = edge_pts - centroid
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            normal = vt[1]
            resid = centered @ normal
            total += float(np.sum(resid ** 2))
        return total

    # Powell optimization
    f0 = float(max(img_w, img_h))
    result = _sp_minimize(
        _collinearity_cost,
        x0=np.array([f0, 0.0, 0.0]),
        method="Powell",
        bounds=[f_bounds, k_bounds, k_bounds],
        options={"maxiter": 5000, "ftol": 1e-6},
    )

    f_opt, k1_opt, k2_opt = float(result.x[0]), float(result.x[1]), float(result.x[2])
    K_opt = np.array([[f_opt, 0, cx], [0, f_opt, cy], [0, 0, 1]], dtype=np.float64)
    dist_opt = np.array([k1_opt, k2_opt, 0.0, 0.0], dtype=np.float64)

    # Compute per-edge RMS for diagnostics
    pts = all_img_arr.reshape(-1, 1, 2)
    undist = _cv2.undistortPoints(pts, K_opt, dist_opt, P=K_opt).reshape(-1, 2)
    per_edge_rms: Dict[str, float] = {}
    per_edge_npts: Dict[str, int] = {}
    for eidx, idxs in edge_groups.items():
        if len(idxs) < 2:
            per_edge_rms[str(eidx)] = 0.0
            per_edge_npts[str(eidx)] = len(idxs)
            continue
        edge_pts = undist[idxs]
        centroid = edge_pts.mean(axis=0)
        centered = edge_pts - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        resid = centered @ vt[1]
        per_edge_rms[str(eidx)] = float(np.sqrt(np.mean(resid ** 2)))
        per_edge_npts[str(eidx)] = len(idxs)

    lens_metrics = {
        "f": f_opt,
        "k1": k1_opt,
        "k2": k2_opt,
        "collinearity_cost": float(result.fun),
        "n_edge_points": n_pts,
        "n_edges_with_points": n_edges,
        "per_edge_rms": per_edge_rms,
        "points_per_edge": per_edge_npts,
        "converged": bool(result.success),
        "iterations": int(result.nit),
    }

    return K_opt, dist_opt, lens_metrics


def _refine_h_from_mat_lines(
    H_initial: np.ndarray,
    frame_bgr: np.ndarray,
    rects: List[Tuple[float, float, float, float, str]],
    anchor_img_pts: np.ndarray,
    anchor_mat_pts: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coefficients: Optional[np.ndarray] = None,
    max_iterations: int = 3,
    ransac_reproj_threshold: float = 5.0,
    min_matched_lines: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine H using observed mat lines. Returns (refined_H_mat_to_img, quality_metrics).

    If K+dist provided, undistorts frame internally before detection.
    All pixel coordinates in the output are in undistorted space.
    """
    import cv2 as _cv2

    img_h, img_w = frame_bgr.shape[:2]
    image_wh = (img_w, img_h)

    # Undistort frame once if K+dist available
    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        frame_work = _cv2.undistort(frame_bgr, K, D)
    else:
        frame_work = frame_bgr

    H_current = np.array(H_initial, dtype=np.float64)
    prev_reproj_error = float("inf")

    metrics: Dict[str, Any] = {
        "mean_reproj_error_px": 0.0,
        "max_reproj_error_px": 0.0,
        "anchor_reproj_error_px": 0.0,
        "n_inliers": 0,
        "n_total_correspondences": 0,
        "inlier_ratio": 0.0,
        "n_matched_lines": 0,
        "n_detected_lines": 0,
        "n_distinct_edges_matched": 0,
        "refinement_iterations": 0,
        "ransac_reproj_threshold": ransac_reproj_threshold,
        "converged": False,
    }

    for iteration in range(max_iterations):
        # Generate projected polylines from current H
        polyline_data = _generate_projected_polylines(
            H_mat_to_img=H_current, rects=rects, image_wh=image_wh,
        )

        # Detect lines — pass None for K/dist (frame already undistorted or raw)
        detected_lines = _detect_lines_in_frame(
            frame_work, None, None, 50, 150, 80, 50, 10,
        )

        # Merge collinear segments
        merged = _merge_collinear_segments(detected_lines)
        metrics["n_detected_lines"] = len(merged)

        # Convert polyline_data to format _match_lines_to_polylines expects
        polylines_as_tuples: List[List[Tuple[float, float]]] = []
        edge_indices: List[int] = []
        for pl in polyline_data.get("polylines", []):
            pts = [(float(p[0]), float(p[1])) for p in pl["pixel_points"]]
            if len(pts) >= 2:
                polylines_as_tuples.append(pts)
                edge_indices.append(pl["edge_index"])

        # Reconstruct all_edges from polyline data (option i — safe)
        edges_by_idx: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        for pl in polyline_data.get("polylines", []):
            idx = pl["edge_index"]
            if idx not in edges_by_idx:
                ws, we = pl["world_start"], pl["world_end"]
                edges_by_idx[idx] = (
                    (float(ws[0]), float(ws[1])),
                    (float(we[0]), float(we[1])),
                )
        max_edge_idx = max(edges_by_idx.keys()) if edges_by_idx else -1
        all_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
            edges_by_idx.get(i, ((0.0, 0.0), (0.0, 0.0)))
            for i in range(max_edge_idx + 1)
        ]

        # Match detected lines to polylines
        H_img_to_world = np.linalg.inv(H_current)
        matched = _match_lines_to_polylines(
            merged, polylines_as_tuples, edge_indices,
            H_img_to_world, None, None, 80.0,
        )
        metrics["n_matched_lines"] = len(matched)

        if len(matched) < min_matched_lines:
            metrics["converged"] = False
            metrics["refinement_iterations"] = iteration + 1
            if iteration == 0:
                return H_initial, metrics
            return H_current, metrics

        # Extract correspondences from matched lines
        world_corr: List[Tuple[float, float]] = []
        pixel_corr: List[Tuple[float, float]] = []
        matched_edge_set: set = set()

        for ml in matched:
            eidx = ml.matched_edge_index
            if eidx < 0 or eidx >= len(all_edges):
                continue
            matched_edge_set.add(eidx)
            (ewx1, ewy1), (ewx2, ewy2) = all_edges[eidx]

            # Sample 15 world points along the blueprint edge
            for k in range(15):
                t = k / 14.0
                w_x = ewx1 + t * (ewx2 - ewx1)
                w_y = ewy1 + t * (ewy2 - ewy1)

                # Find closest point on detected pixel segment
                px1, py1 = ml.pixel_start
                px2, py2 = ml.pixel_end
                seg_dx = px2 - px1
                seg_dy = py2 - py1
                seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy

                # Project world point to pixel using current H to get approximate pixel location
                wp = np.array([w_x, w_y, 1.0], dtype=np.float64)
                proj = H_current @ wp
                if abs(proj[2]) < 1e-12:
                    continue
                approx_px = proj[0] / proj[2]
                approx_py = proj[1] / proj[2]

                # Find nearest point on detected segment to this projected point
                if seg_len_sq < 1e-12:
                    near_x, near_y = px1, py1
                else:
                    t_seg = ((approx_px - px1) * seg_dx + (approx_py - py1) * seg_dy) / seg_len_sq
                    t_seg = max(0.0, min(1.0, t_seg))
                    near_x = px1 + t_seg * seg_dx
                    near_y = py1 + t_seg * seg_dy

                world_corr.append((w_x, w_y))
                pixel_corr.append((near_x, near_y))

        metrics["n_distinct_edges_matched"] = len(matched_edge_set)

        # Combine anchor + mat-line correspondences
        all_world = list(anchor_mat_pts.tolist()) + [(w[0], w[1]) for w in world_corr]
        all_pixel = list(anchor_img_pts.tolist()) + [(p[0], p[1]) for p in pixel_corr]

        all_world_arr = np.array(all_world, dtype=np.float64)
        all_pixel_arr = np.array(all_pixel, dtype=np.float64)
        metrics["n_total_correspondences"] = len(all_world)

        # RANSAC homography
        H_new, inlier_mask = _cv2.findHomography(
            all_world_arr, all_pixel_arr, _cv2.RANSAC, ransac_reproj_threshold,
        )

        if H_new is None or inlier_mask is None:
            metrics["refinement_iterations"] = iteration + 1
            if iteration == 0:
                return H_initial, metrics
            return H_current, metrics

        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / max(1, len(all_world))
        metrics["n_inliers"] = inlier_count
        metrics["inlier_ratio"] = round(inlier_ratio, 3)

        if inlier_ratio < 0.3:
            metrics["refinement_iterations"] = iteration + 1
            if iteration == 0:
                return H_initial, metrics
            return H_current, metrics

        H_new = _ensure_3x3(H_new)

        # Compute reprojection errors for inliers
        inlier_idxs = np.where(inlier_mask.ravel())[0]
        reproj_errors = []
        for idx in inlier_idxs:
            wp = np.array([all_world_arr[idx, 0], all_world_arr[idx, 1], 1.0])
            proj = H_new @ wp
            if abs(proj[2]) < 1e-12:
                continue
            pred_x, pred_y = proj[0] / proj[2], proj[1] / proj[2]
            err = hypot(pred_x - all_pixel_arr[idx, 0], pred_y - all_pixel_arr[idx, 1])
            reproj_errors.append(err)

        if reproj_errors:
            mean_err = float(np.mean(reproj_errors))
            max_err = float(np.max(reproj_errors))
        else:
            mean_err, max_err = 0.0, 0.0

        # Anchor-only reprojection error
        anchor_errors = []
        for i in range(len(anchor_mat_pts)):
            wp = np.array([anchor_mat_pts[i, 0], anchor_mat_pts[i, 1], 1.0])
            proj = H_new @ wp
            if abs(proj[2]) < 1e-12:
                continue
            pred_x, pred_y = proj[0] / proj[2], proj[1] / proj[2]
            err = hypot(pred_x - anchor_img_pts[i, 0], pred_y - anchor_img_pts[i, 1])
            anchor_errors.append(err)

        metrics["mean_reproj_error_px"] = round(mean_err, 2)
        metrics["max_reproj_error_px"] = round(max_err, 2)
        metrics["anchor_reproj_error_px"] = round(float(np.mean(anchor_errors)) if anchor_errors else 0.0, 2)

        H_current = H_new
        metrics["refinement_iterations"] = iteration + 1

        # Convergence check
        if abs(prev_reproj_error - mean_err) < 0.1:
            metrics["converged"] = True
            break
        prev_reproj_error = mean_err

    return H_current, metrics


def _project_polyline_mat_to_img(H: np.ndarray, pts_mat: np.ndarray) -> np.ndarray:
    """
    pts_mat: (N,2) in mat coords. Returns (N,2) image coords.
    """
    import cv2  # local optional dep
    pts = np.asarray(pts_mat, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return proj.reshape(-1, 2)


def _bbox_from_rects(rects: List[Tuple[float, float, float, float, str]]) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) from validated blueprint rectangles."""
    if not rects:
        raise ValueError("No rectangles available to compute mat union bbox.")
    xs = [x for (x, _, w, _, _) in rects] + [x + w for (x, _, w, _, _) in rects]
    ys = [y for (_, y, _, h, _) in rects] + [y + h for (_, y, _, h, _) in rects]
    return (min(xs), min(ys), max(xs), max(ys))


def _mat_bbox_corners(xmin: float, ymin: float, xmax: float, ymax: float) -> List[Tuple[float, float]]:
    """Corners in mat coordinates, ordered [tl, tr, br, bl]."""
    return [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]


def _qa_overlay_dialog(
    *,
    camera_id: str,
    frame_rgb: np.ndarray,
    H_mat_to_img: np.ndarray,
    rects: List[Tuple[float, float, float, float, str]],
    grid_spacing_m: float = 0.5,
    sample_step_m: float = 0.05,
    quality_metrics: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Show a QA overlay window: mat-union grid (0.5m spacing) projected onto the frame via H.
    Returns True if accepted, False if redo requested.
    """
    import matplotlib.pyplot as plt  # noqa

    if not rects:
        print("[D7][QA] No blueprint rects available; cannot render grid QA.")
        return True

    xmin, ymin, xmax, ymax = _rect_union_bounds(rects)
    mask, xs, ys = _make_union_mask(rects, step_m=sample_step_m, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    # Determine which raster columns/rows correspond to the requested grid spacing.
    x0 = np.ceil(xmin / grid_spacing_m) * grid_spacing_m
    y0 = np.ceil(ymin / grid_spacing_m) * grid_spacing_m
    grid_xs = np.arange(x0, xmax + 1e-9, grid_spacing_m)
    grid_ys = np.arange(y0, ymax + 1e-9, grid_spacing_m)

    # Map grid positions to nearest raster indices.
    x_idxs = [int(np.argmin(np.abs(xs - gx))) for gx in grid_xs]
    y_idxs = [int(np.argmin(np.abs(ys - gy))) for gy in grid_ys]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame_rgb)
    ax.set_axis_off()
    ax.set_title(f"QA Overlay — {camera_id} (grid {grid_spacing_m}m). Keys: [a]=accept  [r]=redo")
    # IMPORTANT: lock axes to image pixel coordinates to prevent autoscale from hiding the image.
    h, w = frame_rgb.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # y-down to match image convention
    ax.set_autoscale_on(False)

    def _plot_polyline_img(pts_img: np.ndarray) -> None:
        pts_img = np.asarray(pts_img, dtype=float)
        if pts_img.ndim != 2 or pts_img.shape[1] != 2:
            return
        ok = np.isfinite(pts_img).all(axis=1)
        pts_img = pts_img[ok]
        if pts_img.shape[0] < 2:
            return
        ax.plot(pts_img[:, 0], pts_img[:, 1], linewidth=1.0, clip_on=True)

    # Vertical grid lines (constant x)
    for xi in x_idxs:
        segs = _iter_masked_polylines_constant_x(mask, xs, ys, x_idx=xi)
        for pts_mat in segs:
            if pts_mat.shape[0] < 2:
                continue
            pts_img = _project_polyline_mat_to_img(H_mat_to_img, pts_mat)
            _plot_polyline_img(pts_img)

    # Horizontal grid lines (constant y)
    for yi in y_idxs:
        segs = _iter_masked_polylines_constant_y(mask, xs, ys, y_idx=yi)
        for pts_mat in segs:
            if pts_mat.shape[0] < 2:
                continue
            pts_img = _project_polyline_mat_to_img(H_mat_to_img, pts_mat)
            _plot_polyline_img(pts_img)

    # CP19: Render quality metrics text block at top-right
    if quality_metrics is not None:
        hm = quality_metrics.get("h_metrics", {})
        lm = quality_metrics.get("lens_metrics", {})
        lines = []
        if hm:
            lines.append(
                f"Reproj: {hm.get('mean_reproj_error_px', 0):.1f}px mean / "
                f"{hm.get('max_reproj_error_px', 0):.1f}px max "
                f"({hm.get('n_inliers', 0)}/{hm.get('n_total_correspondences', 0)} inliers)"
            )
            lines.append(
                f"Lines: {hm.get('n_matched_lines', 0)} matched from "
                f"{hm.get('n_distinct_edges_matched', 0)} edges"
            )
        if lm and lm.get("f") is not None:
            lines.append(
                f"Lens: f={lm.get('f', 0):.0f} k1={lm.get('k1', 0):.2f} "
                f"k2={lm.get('k2', 0):.2f} ({lm.get('n_edge_points', 0)} pts / "
                f"{lm.get('n_edges_with_points', 0)} edges)"
            )
        if lines:
            ax.text(
                0.99, 0.99, "\n".join(lines),
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
                color="white",
            )

    decision = {"accept": False, "redo": False}

    def on_key(event):
        k = (event.key or "").lower()
        if k == "a":
            decision["accept"] = True
            plt.close(fig)
        elif k == "r":
            decision["redo"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return bool(decision["accept"]) and not bool(decision["redo"])


def _interactive_calibrate(
    camera_id: str,
    out_path: Path,
    video_path: Path,
    mat_blueprint_path: Path,
) -> None:
    """
    Minimal interactive calibrator skeleton:
      - shows first frame + mat blueprint plot
      - collects alternating clicks (frame -> mat) until >= 4 pairs
      - solves H and saves canonical JSON

    Notes:
      - Keeps dependencies optional: only imports cv2/matplotlib when invoked.
      - You can later make this match the full D7 UI spec (undo/clear/save keys, overlays, etc).
    """
    import cv2  # noqa
    import matplotlib.pyplot as plt  # noqa

    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    frame_bgr_raw = frame_bgr.copy()  # CP19: preserve raw for Phase A

    # Auto-undistort if lens calibration exists (CP16b)
    _K, _dist = _load_lens_calibration(out_path)
    if _K is not None and _dist is not None:
        frame_bgr = cv2.undistort(frame_bgr, _K, _dist)
        print(f"[D7] Applied lens undistortion (K diag: {_K[0,0]:.1f}, {_K[1,1]:.1f})")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    blueprint = _try_load_mat_blueprint(mat_blueprint_path)
    if blueprint is None:
        print(f"[D7] mat_blueprint not found or unreadable: {mat_blueprint_path}")
    else:
        print(f"[D7] loaded mat_blueprint: {mat_blueprint_path}")

    pairs = ClickPairs(image_points_px=[], mat_points=[])

    fig, (ax_img, ax_mat) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Homography Calibrator — {camera_id}")

    ax_img.imshow(frame_rgb)
    ax_img.set_title("Image (px) — click points here")
    ax_img.set_axis_off()

    ax_mat.set_title("Mat blueprint — click corresponding points here")
    ax_mat.set_xlabel("mat_x")
    ax_mat.set_ylabel("mat_y")
    ax_mat.grid(True)

    # Evidence-based rendering for your rectangle-list blueprint schema
    rects: List[Tuple[float, float, float, float, str]] = []
    if blueprint is not None:
        rects = _render_mat_blueprint_rects(ax_mat, blueprint)

    state = {"expect": "img"}  # img then mat alternating
    # --- Persistent point/label artists (avoid ax.lines.clear(), which breaks on newer mpl) ---
    img_point_artists: List[Any] = []
    img_text_artists: List[Any] = []
    mat_point_artists: List[Any] = []
    mat_text_artists: List[Any] = []

    def _add_point(ax, x: float, y: float, *, color: str, label: str):
        # plot() returns a Line2D; keep references so we can remove on undo/clear
        pt = ax.plot([x], [y], marker="o", linestyle="none", color=color)[0]
        txt = ax.text(x, y, label, color=color, fontsize=10, ha="left", va="bottom")
        return pt, txt

    def _undo_last():
        # If we have an unmatched image point (img > mat), remove that image point.
        if len(pairs.image_points_px) > len(pairs.mat_points):
            pairs.image_points_px.pop()
            if img_point_artists:
                img_point_artists.pop().remove()
            if img_text_artists:
                img_text_artists.pop().remove()
            state["expect"] = "img"
            return
        # Otherwise remove the last complete pair.
        if pairs.image_points_px and pairs.mat_points:
            pairs.image_points_px.pop()
            pairs.mat_points.pop()
            if img_point_artists:
                img_point_artists.pop().remove()
            if img_text_artists:
                img_text_artists.pop().remove()
            if mat_point_artists:
                mat_point_artists.pop().remove()
            if mat_text_artists:
                mat_text_artists.pop().remove()
            state["expect"] = "img"

    def _clear_all():
        pairs.image_points_px.clear()
        pairs.mat_points.clear()
        for a in img_point_artists:
            a.remove()
        for a in img_text_artists:
            a.remove()
        for a in mat_point_artists:
            a.remove()
        for a in mat_text_artists:
            a.remove()
        img_point_artists.clear()
        img_text_artists.clear()
        mat_point_artists.clear()
        mat_text_artists.clear()
        state["expect"] = "img"

    def on_click(event):
        if event.inaxes not in (ax_img, ax_mat):
            return
        # Ignore clicks with no data coords (can happen on some backends)
        if event.xdata is None or event.ydata is None:
            return

        if state["expect"] == "img":
            if event.inaxes is not ax_img:
                print("Click on IMAGE (left) first.")
                return
            x = float(event.xdata); y = float(event.ydata)
            pairs.image_points_px.append((x, y))
            n = len(pairs.image_points_px)
            label = f"r{n}"
            pt, txt = _add_point(ax_img, x, y, color="red", label=label)
            img_point_artists.append(pt)
            img_text_artists.append(txt)
            state["expect"] = "mat"
            print(f"Image point #{n} = ({x:.1f}, {y:.1f}) [{label}]")
        else:
            if event.inaxes is not ax_mat:
                print("Click on MAT (right) next.")
                return
            x = float(event.xdata); y = float(event.ydata)
            pairs.mat_points.append((x, y))
            n = len(pairs.mat_points)
            label = f"b{n}"
            pt, txt = _add_point(ax_mat, x, y, color="blue", label=label)
            mat_point_artists.append(pt)
            mat_text_artists.append(txt)
            state["expect"] = "img"
            print(f"Mat point   #{n} = ({x:.3f}, {y:.3f}) [{label}]")

        fig.canvas.draw_idle()

    def on_key(event):
        k = (event.key or "").lower()

        if k == "u":
            _undo_last()
            print("Undo.")
            fig.canvas.draw_idle()

        elif k == "c":
            _clear_all()
            print("Cleared.")
            fig.canvas.draw_idle()

        elif k == "s":
            if len(pairs.image_points_px) != len(pairs.mat_points) or len(pairs.image_points_px) < 4:
                print("Need >= 4 complete point pairs before solving.")
                return

            img_pts = np.array(pairs.image_points_px, dtype=float)
            mat_pts = np.array(pairs.mat_points, dtype=float)
            import cv2  # noqa

            # 1. Compute initial H from click pairs (mat → old-undistorted img space)
            H, mask = cv2.findHomography(mat_pts, img_pts, method=cv2.RANSAC)
            H = _ensure_3x3(H)
            inliers = int(mask.sum()) if mask is not None else None
            img_h_px, img_w_px = frame_rgb.shape[:2]

            # 2. CP19 Phase A — Polyline-based lens calibration (on raw frame)
            # img_pts are in old-undistorted space; recompute H for raw pixel space
            H_mat_to_raw, _ = _recompute_h_for_space(
                mat_pts, img_pts,
                from_K=_K, from_dist=_dist,
                to_K=None, to_dist=None,
            )
            print("[CP19] Phase A: polyline lens calibration...")
            K_new, dist_new, lens_metrics = _polyline_lens_calibration(
                H_mat_to_img=H_mat_to_raw,
                frame_bgr=frame_bgr_raw,
                frame_gray=cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2GRAY),
                rects=rects,
                image_wh=(img_w_px, img_h_px),
            )
            if K_new is not None:
                print(f"[CP19] Phase A: f={lens_metrics['f']:.1f} k1={lens_metrics['k1']:.4f} "
                      f"k2={lens_metrics['k2']:.4f} ({lens_metrics['n_edge_points']} pts / "
                      f"{lens_metrics['n_edges_with_points']} edges)")
            else:
                print(f"[CP19] Phase A skipped: {lens_metrics.get('reason', 'insufficient detections')}. "
                      "Using existing lens calibration.")
                K_new, dist_new = _K, _dist

            # 3. CP19 Phase B — Refine H using mat lines
            # Recompute H + anchor pts for new-undistorted space
            H_for_phase_b, new_undist_anchor = _recompute_h_for_space(
                mat_pts, img_pts,
                from_K=_K, from_dist=_dist,
                to_K=K_new, to_dist=dist_new,
            )
            print("[CP19] Phase B: mat-line H refinement...")
            H_refined, h_metrics = _refine_h_from_mat_lines(
                H_initial=H_for_phase_b,
                frame_bgr=frame_bgr_raw,
                rects=rects,
                anchor_img_pts=new_undist_anchor,
                anchor_mat_pts=mat_pts,
                camera_matrix=K_new,
                dist_coefficients=dist_new,
            )
            print(f"[CP19] Phase B: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                  f"{h_metrics['n_matched_lines']} lines matched")

            quality_metrics = {
                "h_metrics": h_metrics,
                "lens_metrics": lens_metrics,
                "calibration_mode": "unified",
            }

            # 4. Re-undistort display frame with new K+dist for QA
            if K_new is not None:
                frame_bgr_display = cv2.undistort(frame_bgr_raw, K_new, dist_new)
            else:
                frame_bgr_display = frame_bgr
            frame_rgb_display = cv2.cvtColor(frame_bgr_display, cv2.COLOR_BGR2RGB)

            # 5. QA with refined H + metrics
            print("[D7] Launching QA overlay... (accept=a, redo=r)")
            accepted = _qa_overlay_dialog(
                camera_id=camera_id,
                frame_rgb=frame_rgb_display,
                H_mat_to_img=H_refined,
                rects=rects,
                grid_spacing_m=0.5,
                sample_step_m=0.05,
                quality_metrics=quality_metrics,
            )
            if not accepted:
                print("[D7] QA requested redo. Clearing points; please re-select correspondences.")
                _clear_all()
                fig.canvas.draw_idle()
                return

            # 6. Generate polylines from refined H and save
            polyline_data = _generate_projected_polylines(
                H_mat_to_img=H_refined, rects=rects,
                image_wh=(img_w_px, img_h_px),
            )
            print(f"[D7] Generated {polyline_data['n_polylines']} projected polylines "
                  f"from {polyline_data['n_edges_total']} panel edges")

            extra: Dict[str, Any] = {
                "correspondences": {
                    "image_points_px": pairs.image_points_px,
                    "mat_points": pairs.mat_points,
                },
                "fit": {
                    "method": "cv2.findHomography_ransac",
                    "num_points": len(pairs.image_points_px),
                    "inliers": inliers,
                },
                "qa": {
                    "grid_spacing_m": 0.5,
                    "sample_step_m": 0.05,
                    "accepted": True,
                },
                "projected_polylines": polyline_data,
                "quality_metrics": quality_metrics,
            }
            if K_new is not None:
                extra["camera_matrix"] = K_new.tolist()
                extra["dist_coefficients"] = dist_new.tolist()

            _write_homography_json(
                out_path=out_path,
                camera_id=camera_id,
                H=H_refined,
                source={"type": "interactive_clicks", "video": str(video_path)},
                extra=extra,
            )
            print("[D7] Saved homography (accepted). Closing calibrator and returning to pipeline...")
            plt.close(fig)

        elif k == "q":
            print("Quit without saving.")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: click IMAGE then MAT (repeat). Keys: [u]=undo [c]=clear [s]=solve+save [q]=quit")
    plt.show()


def _interactive_calibrate_overlay_rect_fixed(
    camera_id: str,
    out_path: Path,
    video_path: Path,
    mat_blueprint_path: Path,
    *,
    grid_spacing_m: float = 0.5,
    sample_step_m: float = 0.05,
) -> None:
    """Overlay-rect UI with tabs."""
    # Frame tab:
    #   - Draggable quad represents the selected anchor rectangle corners in image pixels.
    #   - Homography computed from anchor mat corners -> image corners.
    #   - Whole mat blueprint preview-warped using that homography.
    # Blueprint tab:
    #   - Click a rectangle to select anchor (prefers smallest-area under cursor).
    import cv2  # noqa
    import matplotlib.pyplot as plt  # noqa
    from matplotlib.patches import Polygon  # noqa

    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    frame_bgr_raw = frame_bgr.copy()  # CP19: preserve raw for Phase A

    # Auto-undistort if lens calibration exists (CP16b)
    _K, _dist = _load_lens_calibration(out_path)
    if _K is not None and _dist is not None:
        frame_bgr = cv2.undistort(frame_bgr, _K, _dist)
        print(f"[D7] Applied lens undistortion (K diag: {_K[0,0]:.1f}, {_K[1,1]:.1f})")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame_rgb.shape[:2]

    blueprint = _try_load_mat_blueprint(mat_blueprint_path)
    if blueprint is None:
        raise RuntimeError(f"[D7] mat_blueprint not found or unreadable: {mat_blueprint_path}")
    rects = _parse_rects_from_blueprint(blueprint)
    if not rects:
        raise RuntimeError("[D7] mat_blueprint had no valid rectangles; cannot build overlay_rect UI.")

    corner_ids = ["tl", "tr", "br", "bl"]

    fig = plt.figure(figsize=(12, 7))
    ax_img = fig.add_subplot(1, 1, 1)
    fig.suptitle(f"Homography Calibrator (overlay_rect) — {camera_id}")
    # Base frame image
    ax_img.imshow(frame_rgb)
    ax_img.set_axis_off()
    # Allow the quad to extend beyond the frame by padding the view.
    pad = max(50, int(0.08 * max(img_w, img_h)))
    ax_img.set_xlim(-pad, img_w + pad)
    ax_img.set_ylim(img_h + pad, -pad)
    ax_img.set_autoscale_on(False)

    init_w = 0.65 * img_w
    init_h = 0.65 * img_h
    cx0 = 0.5 * img_w
    cy0 = 0.5 * img_h
    img_pts = np.array(
        [
            [cx0 - 0.5 * init_w, cy0 - 0.5 * init_h],  # tl
            [cx0 + 0.5 * init_w, cy0 - 0.5 * init_h],  # tr
            [cx0 + 0.5 * init_w, cy0 + 0.5 * init_h],  # br
            [cx0 - 0.5 * init_w, cy0 + 0.5 * init_h],  # bl
        ],
        dtype=float,
    )

    state = {
        # Draggable quad in image pixels, ordered [tl,tr,br,bl].
        # IMPORTANT: these represent the selected ANCHOR rectangle corners.
        "img_pts": img_pts,
        "step_px": 8.0,
        "step_ang": 2.0 * (pi / 180.0),
        "step_scale": 1.05,
        "drag_idx": None,
        "selected_idx": None,
        "pick_tol_px": 18.0,
        "blueprint_alpha": 0.35,
        "page": "frame",     # "frame" or "blueprint"
        "anchor_rect": None,  # dict with x,y,width,height,label
    }

    # This polygon shows the ANCHOR quad (not the union bbox).
    poly = Polygon(state["img_pts"], closed=True, fill=False, linewidth=2)
    ax_img.add_patch(poly)

    corner_text: List[Any] = []
    for (u, v), cid in zip(state["img_pts"], corner_ids):
        corner_text.append(ax_img.text(u, v, cid, fontsize=10, ha="left", va="bottom"))

    corner_handle_artists: List[Any] = []
    for (u, v) in state["img_pts"]:
        hdl = ax_img.plot([u], [v], marker="o", linestyle="none")[0]
        corner_handle_artists.append(hdl)

    # --- Build a blueprint raster (source image) and warp it into the draggable quad ---
    # The raster is a clean line-drawing of your mat rectangles in "mat space",
    # then we perspective-warp it into the image quad to visually align seams/lines.
    def _build_blueprint_raster(
        rects_: List[Tuple[float, float, float, float, str]],
        *,
        canvas_px: int = 900,
        margin_px: int = 24,
        line_px: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          blueprint_rgb: (H,W,3) uint8, white background with black outlines
                    H_src_to_mat: (3,3) float64 mapping blueprint pixel coords -> mat coords
        """
        xs = [x for (x, _, w, _, _) in rects_] + [x + w for (x, _, w, _, _) in rects_]
        ys = [y for (_, y, _, h, _) in rects_] + [y + h for (_, y, _, h, _) in rects_]
        bx0, by0, bx1, by1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        bw = max(1e-6, bx1 - bx0)
        bh = max(1e-6, by1 - by0)

        inner = max(50, canvas_px - 2 * margin_px)
        scale = min(inner / bw, inner / bh)
        out_w = int(round(bw * scale)) + 2 * margin_px
        out_h = int(round(bh * scale)) + 2 * margin_px
        out_w = max(out_w, 2 * margin_px + 50)
        out_h = max(out_h, 2 * margin_px + 50)

        img = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        def to_px(mx: float, my: float) -> Tuple[int, int]:
            # Map mat coords into blueprint image pixels.
            # Invert Y so that "mat +y up" maps naturally into image with y-down.
            u = margin_px + (mx - bx0) * scale
            v = margin_px + (by1 - my) * scale
            return int(round(u)), int(round(v))

        for (x, y, w, h, _) in rects_:
            x0, y0 = float(x), float(y)
            x1, y1 = x0 + float(w), y0 + float(h)
            p_tl = to_px(x0, y1)
            p_br = to_px(x1, y0)
            cv2.rectangle(img, p_tl, p_br, color=(0, 0, 0), thickness=line_px)

        # Inverse of to_px mapping:
        # mx = bx0 + (u - margin)/scale
        # my = by1 - (v - margin)/scale
        # So [mx,my,1]^T = H_src_to_mat * [u,v,1]^T
        inv_s = 1.0 / float(scale)
        H_src_to_mat = np.array(
            [
                [inv_s, 0.0, bx0 - float(margin_px) * inv_s],
                [0.0, -inv_s, by1 + float(margin_px) * inv_s],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return img, H_src_to_mat

    blueprint_rgb, H_src_to_mat = _build_blueprint_raster(rects)

    # This artist displays a pre-blended frame (frame + warped blueprint).
    # It starts invisible until first redraw.
    overlay_artist = ax_img.imshow(frame_rgb, alpha=0.0)
    overlay_artist.set_zorder(2)  # above base frame, below polygon/handles/text
    poly.set_zorder(5)
    for h in corner_handle_artists:
        h.set_zorder(6)
    for t in corner_text:
        t.set_zorder(7)
    ax_img.text(
        0.01,
        0.01,
        "TAB=toggle (frame/blueprint) | Frame: drag corners, arrows=move, +/- scale, j/l scaleX, i/k scaleY, r/e rot, t turn90, f flip, s save, q quit",
        transform=ax_img.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    def _centroid(pts: np.ndarray) -> np.ndarray:
        return np.mean(np.asarray(pts, dtype=float), axis=0)

    # Helpers used by blueprint overlay and continuity placement
    def _mat_rect_corners(x: float, y: float, w: float, h: float) -> np.ndarray:
        # Order: [tl,tr,br,bl] in MAT coordinates
        return np.array([[x, y + h], [x + w, y + h], [x + w, y], [x, y]], dtype=float)

    def _project_pts(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
        return proj.reshape(-1, 2).astype(float)

    def _compute_current_H_mat_to_img(mat_pts: np.ndarray) -> Optional[np.ndarray]:
        """Compute H from current anchor mat_pts -> current draggable img_pts."""
        img_pts2 = np.asarray(state["img_pts"], dtype=float)
        if img_pts2.shape != (4, 2) or mat_pts.shape != (4, 2):
            return None
        if not (np.isfinite(img_pts2).all() and np.isfinite(mat_pts).all()):
            return None
        H, _ = cv2.findHomography(np.asarray(mat_pts, dtype=float), np.asarray(img_pts2, dtype=float), method=0)
        if H is None:
            return None
        return _ensure_3x3(H)

    def _update_blueprint_overlay() -> None:
        """
        Warp blueprint_rgb into the frame using the current H derived from the selected anchor.
        """
        if state.get("anchor_rect") is None:
            overlay_artist.set_alpha(0.0)
            return

        ar = state["anchor_rect"]
        mat_pts = _mat_rect_corners(float(ar["x"]), float(ar["y"]), float(ar["width"]), float(ar["height"]))
        H_mat_to_img = _compute_current_H_mat_to_img(mat_pts)
        if H_mat_to_img is None:
            overlay_artist.set_alpha(0.0)
            return

        # Compose: src(px)->mat then mat->img
        H_src_to_img = (np.asarray(H_mat_to_img, dtype=np.float64) @ np.asarray(H_src_to_mat, dtype=np.float64))
        H_src_to_img = _ensure_3x3(H_src_to_img)

        warped = cv2.warpPerspective(blueprint_rgb, H_src_to_img, (img_w, img_h))

        # Mask: keep non-white pixels (lines) from the warped blueprint.
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        mask = gray < 250

        a = float(state.get("blueprint_alpha", 0.35))
        a = max(0.0, min(1.0, a))
        if a <= 0.0:
            overlay_artist.set_alpha(0.0)
            return

        out = frame_rgb.copy()
        out[mask] = (
            out[mask].astype(np.float32) * (1.0 - a) + warped[mask].astype(np.float32) * a
        ).astype(np.uint8)

        overlay_artist.set_data(out)
        overlay_artist.set_alpha(1.0)

    def _redraw():
        pts = state["img_pts"]
        poly.set_xy(pts)
        for t, (u, v) in zip(corner_text, pts):
            t.set_position((u, v))
        for hdl, (u, v) in zip(corner_handle_artists, pts):
            hdl.set_data([u], [v])
        _update_blueprint_overlay()
        fig.canvas.draw_idle()

    def _translate(dx: float, dy: float, *, only_selected: bool = False) -> None:
        if only_selected and state["selected_idx"] is not None:
            i = int(state["selected_idx"])
            state["img_pts"][i, 0] += dx
            state["img_pts"][i, 1] += dy
        else:
            state["img_pts"][:, 0] += dx
            state["img_pts"][:, 1] += dy

    def _scale(sx: float, sy: float) -> None:
        c = _centroid(state["img_pts"])
        pts = state["img_pts"] - c
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        state["img_pts"] = pts + c

    def _rotate(theta: float) -> None:
        c = _centroid(state["img_pts"])
        pts = state["img_pts"] - c
        ct = cos(theta)
        st = sin(theta)
        R = np.array([[ct, -st], [st, ct]], dtype=float)
        state["img_pts"] = (pts @ R.T) + c

    def _flip_y_axis() -> None:
        c = _centroid(state["img_pts"])
        pts = state["img_pts"].copy()
        pts[:, 0] = 2.0 * c[0] - pts[:, 0]
        state["img_pts"] = pts

    # -------------------------
    # Blueprint "tab" (second axes) + anchor selection
    # -------------------------
    from matplotlib.patches import Rectangle  # noqa

    ax_blueprint = fig.add_subplot(1, 1, 1)
    ax_blueprint.set_visible(False)
    ax_blueprint.set_title("Blueprint — click a rectangle to set the anchor (TAB to return to frame)")
    ax_blueprint.set_xlabel("mat_x")
    ax_blueprint.set_ylabel("mat_y")
    ax_blueprint.grid(True)
    _render_mat_blueprint_rects(ax_blueprint, blueprint)

    sel_patch = Rectangle((0, 0), 1, 1, fill=False, linewidth=3)
    sel_patch.set_visible(False)
    ax_blueprint.add_patch(sel_patch)

    def _update_blueprint_selection_patch(x: float, y: float, w: float, h: float) -> None:
        sel_patch.set_xy((x, y))
        sel_patch.set_width(w)
        sel_patch.set_height(h)
        sel_patch.set_visible(True)

    def _default_anchor_rect(rects_: List[Tuple[float, float, float, float, str]]) -> Tuple[float, float, float, float, str]:
        best = None
        best_a = -1.0
        for (x, y, w, h, label) in rects_:
            a = float(w) * float(h)
            if a > best_a:
                best_a = a
                best = (float(x), float(y), float(w), float(h), str(label))
        assert best is not None
        return best

    def _update_blueprint_selection_patch(x: float, y: float, w: float, h: float) -> None:
        sel_patch.set_xy((x, y))
        sel_patch.set_width(w)
        sel_patch.set_height(h)
        sel_patch.set_visible(True)

    def _pick_rect_at(mx: float, my: float) -> Optional[Tuple[float, float, float, float, str]]:
        # IMPORTANT: prefer *smallest-area* rect containing the click
        # so inner rectangles are selectable even when an outer rect contains them.
        candidates: List[Tuple[float, float, float, float, str]] = []
        for (x, y, w, h, label) in rects:
            if float(x) <= mx <= float(x + w) and float(y) <= my <= float(y + h):
                candidates.append((float(x), float(y), float(w), float(h), str(label)))
        if not candidates:
            return None
        candidates.sort(key=lambda r: float(r[2]) * float(r[3]))  # area ascending
        return candidates[0]

    # Anchor rect + canonical mat points used to compute H.
    anchor = _default_anchor_rect(rects)
    state["anchor_rect"] = {"x": anchor[0], "y": anchor[1], "width": anchor[2], "height": anchor[3], "label": anchor[4]}
    _update_blueprint_selection_patch(anchor[0], anchor[1], anchor[2], anchor[3])

    def _refresh_title() -> None:
        lbl = (state.get("anchor_rect") or {}).get("label", "")
        if isinstance(lbl, str) and lbl.strip():
            fig.suptitle(f"Homography Calibrator (overlay_rect) — {camera_id} | anchor: {lbl}")
        else:
            fig.suptitle(f"Homography Calibrator (overlay_rect) — {camera_id} | anchor: (unnamed)")

    def on_blueprint_click(event):
        if state.get("page", "frame") != "blueprint":
            return
        if event.inaxes is not ax_blueprint or event.xdata is None or event.ydata is None:
            return
        pick = _pick_rect_at(float(event.xdata), float(event.ydata))
        if pick is None:
            return
        x, y, w, h, label = pick
        # Keep continuity: use current H (from old anchor) to place the new anchor corners in image space.
        prev_ar = state.get("anchor_rect")
        if prev_ar is not None:
            prev_mat = _mat_rect_corners(float(prev_ar["x"]), float(prev_ar["y"]), float(prev_ar["width"]), float(prev_ar["height"]))
            H_prev = _compute_current_H_mat_to_img(prev_mat)
        else:
            H_prev = None

        state["anchor_rect"] = {"x": x, "y": y, "width": w, "height": h, "label": label}
        _update_blueprint_selection_patch(x, y, w, h)

        # Move the draggable quad to the newly-selected anchor's corners (projected via current H).
        if H_prev is not None:
            new_mat = _mat_rect_corners(x, y, w, h)
            state["img_pts"] = _project_pts(H_prev, new_mat)
            state["selected_idx"] = None
            state["drag_idx"] = None

        _refresh_title()
        _redraw()

    def on_key(event):
        k = (event.key or "").lower()
        if k in {"tab", "b", "v"}:
            page = state.get("page", "frame")
            if k == "b":
                page = "blueprint"
            elif k == "v":
                page = "frame"
            else:
                page = "blueprint" if page == "frame" else "frame"
            state["page"] = page
            ax_blueprint.set_visible(page == "blueprint")
            ax_img.set_visible(page == "frame")
            # When returning to frame, refresh overlay immediately (mat_pts may have changed).
            if page == "frame":
                _redraw()
            else:
                fig.canvas.draw_idle()
            return

        if state.get("page", "frame") == "blueprint":
            if k == "q":
                print("Quit without saving."); plt.close(fig); return
            return
        if k in {"left", "right", "up", "down"}:
            dx = (-state["step_px"] if k == "left" else (state["step_px"] if k == "right" else 0.0))
            dy = (-state["step_px"] if k == "up" else (state["step_px"] if k == "down" else 0.0))
            _translate(dx, dy, only_selected=(state["selected_idx"] is not None))
            _redraw()
            return
        if k in {"+", "="}:
            _scale(state["step_scale"], state["step_scale"])
            _redraw(); return
        if k == "-":
            _scale(1.0 / state["step_scale"], 1.0 / state["step_scale"])
            _redraw(); return
        if k == "j":
            _scale(1.0 / state["step_scale"], 1.0)
            _redraw(); return
        if k == "l":
            _scale(state["step_scale"], 1.0)
            _redraw(); return
        if k == "i":
            _scale(1.0, state["step_scale"])
            _redraw(); return
        if k == "k":
            _scale(1.0, 1.0 / state["step_scale"])
            _redraw(); return
        if k == "r":
            _rotate(state["step_ang"])
            _redraw(); return
        if k == "e":
            _rotate(-state["step_ang"])
            _redraw(); return
        if k == "t":
            _rotate(pi / 2.0)
            _redraw(); return
        if k == "f":
            _flip_y_axis()
            _redraw(); return
        if k == "q":
            print("Quit without saving."); plt.close(fig); return
        if k == "s":
            if state.get("anchor_rect") is None:
                print("[D7] No anchor selected."); return
            ar = state["anchor_rect"]
            mat_pts = _mat_rect_corners(float(ar["x"]), float(ar["y"]), float(ar["width"]), float(ar["height"]))
            img_pts2 = np.asarray(state["img_pts"], dtype=float)
            import cv2  # noqa

            # 1. Compute initial H from anchor (mat → old-undistorted img space)
            H, _ = cv2.findHomography(np.asarray(mat_pts, dtype=float), np.asarray(img_pts2, dtype=float), method=0)
            H = _ensure_3x3(H)
            mat_pts_f = np.asarray(mat_pts, dtype=float)

            # 2. CP19 Phase A — Polyline-based lens calibration (on raw frame)
            # img_pts2 are in old-undistorted space; recompute H for raw pixel space
            H_mat_to_raw, _ = _recompute_h_for_space(
                mat_pts_f, img_pts2,
                from_K=_K, from_dist=_dist,
                to_K=None, to_dist=None,
            )
            print("[CP19] Phase A: polyline lens calibration...")
            K_new, dist_new, lens_metrics = _polyline_lens_calibration(
                H_mat_to_img=H_mat_to_raw,
                frame_bgr=frame_bgr_raw,
                frame_gray=cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2GRAY),
                rects=rects,
                image_wh=(img_w, img_h),
            )
            if K_new is not None:
                print(f"[CP19] Phase A: f={lens_metrics['f']:.1f} k1={lens_metrics['k1']:.4f} "
                      f"k2={lens_metrics['k2']:.4f} ({lens_metrics['n_edge_points']} pts / "
                      f"{lens_metrics['n_edges_with_points']} edges)")
            else:
                print(f"[CP19] Phase A skipped: {lens_metrics.get('reason', 'insufficient detections')}. "
                      "Using existing lens calibration.")
                K_new, dist_new = _K, _dist

            # 3. CP19 Phase B — Refine H using mat lines
            # Recompute H + anchor pts for new-undistorted space
            H_for_phase_b, new_undist_anchor = _recompute_h_for_space(
                mat_pts_f, img_pts2,
                from_K=_K, from_dist=_dist,
                to_K=K_new, to_dist=dist_new,
            )
            print("[CP19] Phase B: mat-line H refinement...")
            H_refined, h_metrics = _refine_h_from_mat_lines(
                H_initial=H_for_phase_b,
                frame_bgr=frame_bgr_raw,
                rects=rects,
                anchor_img_pts=new_undist_anchor,
                anchor_mat_pts=mat_pts_f,
                camera_matrix=K_new,
                dist_coefficients=dist_new,
            )
            print(f"[CP19] Phase B: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                  f"{h_metrics['n_matched_lines']} lines matched")

            quality_metrics = {
                "h_metrics": h_metrics,
                "lens_metrics": lens_metrics,
                "calibration_mode": "unified",
            }

            # 4. Re-undistort display frame with new K+dist for QA
            if K_new is not None:
                frame_bgr_display = cv2.undistort(frame_bgr_raw, K_new, dist_new)
            else:
                frame_bgr_display = frame_bgr
            frame_rgb_display = cv2.cvtColor(frame_bgr_display, cv2.COLOR_BGR2RGB)

            # 5. QA with refined H + metrics
            accepted = _qa_overlay_dialog(
                camera_id=camera_id,
                frame_rgb=frame_rgb_display,
                rects=rects,
                H_mat_to_img=H_refined,
                grid_spacing_m=grid_spacing_m,
                sample_step_m=sample_step_m,
                quality_metrics=quality_metrics,
            )
            if not accepted:
                print("[D7] QA rejected. Continue adjusting overlay, then press 's' again."); return

            # 6. Generate polylines from refined H and save
            polyline_data = _generate_projected_polylines(
                H_mat_to_img=H_refined, rects=rects, image_wh=(img_w, img_h),
            )
            print(f"[D7] Generated {polyline_data['n_polylines']} projected polylines "
                  f"from {polyline_data['n_edges_total']} panel edges")

            extra_save: Dict[str, Any] = {
                "correspondences": {
                    "image_points_px": img_pts2.tolist(),
                    "mat_points": mat_pts.tolist(),
                    "corner_ids": corner_ids,
                },
                "ui": {
                    "calibration_ui": "overlay_rect",
                    "note": "overlay_rect stores direct dragged image-space corners; corner_ids preserve mapping to chosen anchor rectangle corners (in mat coords).",
                    "blueprint_alpha": float(state.get("blueprint_alpha", 0.35)),
                    "anchor_rect": dict(state.get("anchor_rect", {})) if state.get("anchor_rect") else None,
                },
                "qa": {
                    "grid_spacing_m": float(grid_spacing_m),
                    "sample_step_m": float(sample_step_m),
                    "accepted": True,
                },
                "projected_polylines": polyline_data,
                "quality_metrics": quality_metrics,
            }
            if K_new is not None:
                extra_save["camera_matrix"] = K_new.tolist()
                extra_save["dist_coefficients"] = dist_new.tolist()

            _write_homography_json(
                out_path=out_path,
                camera_id=camera_id,
                H=H_refined,
                source={"type": "overlay_rect", "video": str(video_path)},
                extra=extra_save,
            )
            print("[D7] Saved homography. Closing calibrator and returning to pipeline...")
            plt.close(fig)

    def _nearest_corner_idx(x: float, y: float) -> Optional[int]:
        pts = np.asarray(state["img_pts"], dtype=float)
        d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return i if float(np.sqrt(d2[i])) <= float(state["pick_tol_px"]) else None

    def on_mouse_press(event):
        if state.get("page", "frame") != "frame":
            return
        if event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return
        i = _nearest_corner_idx(float(event.xdata), float(event.ydata))
        if i is None:
            return
        state["drag_idx"] = i
        state["selected_idx"] = i

    def on_mouse_release(event):
        state["drag_idx"] = None

    def on_mouse_move(event):
        i = state["drag_idx"]
        if state.get("page", "frame") != "frame":
            return
        if i is None or event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return
        state["img_pts"][int(i), 0] = float(event.xdata)
        state["img_pts"][int(i), 1] = float(event.ydata)
        _redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_press_event", on_blueprint_click)

    # Initial overlay render.
    _refresh_title()
    _redraw()

    print(
        "Controls (frame tab): drag corners (warp) | arrows=move(all or selected)  +/-=scale  j/l=scaleX  i/k=scaleY  "
        "r/e=rotate  t=turn90  f=flip  s=save  q=quit | tab=browse tabs (frame/blueprint)"
    )
    plt.show()
def _interactive_calibrate_overlay_rect(
    camera_id: str,
    out_path: Path,
    video_path: Path,
    mat_blueprint_path: Path,
    *,
    grid_spacing_m: float = 0.5,
    sample_step_m: float = 0.05,
) -> None:
    """Overlay-rect UI: user aligns a mat bounding-rectangle directly on the image (draggable corners)."""
    return _interactive_calibrate_overlay_rect_fixed(
        camera_id=camera_id,
        out_path=out_path,
        video_path=video_path,
        mat_blueprint_path=mat_blueprint_path,
        grid_spacing_m=grid_spacing_m,
        sample_step_m=sample_step_m,
    )


def main():
    p = argparse.ArgumentParser(
        prog="python -m bjj_pipeline.tools.homography_calibrate",
        description="Homography calibration/import tool (D7). Writes configs/cameras/<camera>/homography.json",
    )
    p.add_argument("--camera", required=True, help="Camera id (e.g. cam03)")
    p.add_argument(
        "--configs-root",
        default="configs",
        help="Repo configs root (default: ./configs)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Override output JSON path (default: configs/cameras/<camera>/homography.json)",
    )

    sub = p.add_subparsers(dest="mode", required=True)

    p_import = sub.add_parser("import", help="Import an existing matrix (.npy or .json) and write canonical homography.json")
    p_import.add_argument("--npy", default=None, help="Path to homography_matrix.npy")
    p_import.add_argument("--json", default=None, help="Path to existing homography.json containing top-level 'H'")

    p_interactive = sub.add_parser("interactive", help="Interactive click-based calibration from a video")
    p_interactive.add_argument("--video", required=True, help="Path to an mp4 to grab the first frame from")
    p_interactive.add_argument(
        "--mat-blueprint",
        default="configs/mat_blueprint.json",
        help="Path to mat blueprint json (default: configs/mat_blueprint.json)",
    )
    p_interactive.add_argument(
        "--calibration-ui",
        choices=["clicks", "overlay_rect"],
        default="clicks",
        help="Calibration UI mode (default: clicks). overlay_rect lets you align a mat rectangle directly on the image.",
    )

    p_placeholder = sub.add_parser(
        "placeholder",
        help="Write a placeholder identity homography.json marked as needing calibration (useful for tests/onboarding)",
    )

    args = p.parse_args()

    configs_root = Path(args.configs_root)
    out_path = Path(args.out) if args.out else _default_homography_json_path(configs_root, args.camera)

    if args.mode == "import":
        if bool(args.npy) == bool(args.json):
            raise SystemExit("Provide exactly one of --npy or --json")

        if args.npy:
            npy_path = Path(args.npy)
            H = _load_npy(npy_path)
            _write_homography_json(
                out_path=out_path,
                camera_id=args.camera,
                H=H,
                source={"type": "imported_npy", "path": str(npy_path)},
            )
        else:
            json_path = Path(args.json)
            payload = _load_existing_json(json_path)
            if "H" not in payload:
                raise ValueError(f"Input JSON missing top-level 'H': {json_path}")
            H = np.array(payload["H"], dtype=float)
            _write_homography_json(
                out_path=out_path,
                camera_id=args.camera,
                H=H,
                source={"type": "imported_json", "path": str(json_path)},
                extra={k: v for k, v in payload.items() if k not in {"H", "camera_id", "source", "created_at"}},
            )

    elif args.mode == "interactive":
        if args.calibration_ui == "overlay_rect":
            _interactive_calibrate_overlay_rect_fixed(
                camera_id=args.camera,
                out_path=out_path,
                video_path=Path(args.video),
                mat_blueprint_path=Path(args.mat_blueprint),
            )
        else:
            _interactive_calibrate(
                camera_id=args.camera,
                out_path=out_path,
                video_path=Path(args.video),
                mat_blueprint_path=Path(args.mat_blueprint),
            )
    elif args.mode == "placeholder":
        H = np.eye(3, dtype=float)
        _write_homography_json(
            out_path=out_path,
            camera_id=args.camera,
            H=H,
            source={"type": "placeholder_identity"},
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
