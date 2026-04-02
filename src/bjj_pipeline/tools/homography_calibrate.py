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


def _find_empty_frame(
    video_path: str,
    *,
    n_candidates: int = 20,
) -> Tuple[np.ndarray, int]:
    """Find the frame with least activity (closest to temporal median).

    Samples n_candidates frames evenly across the video, computes the
    per-pixel median, and returns the frame with the smallest mean
    absolute deviation from that median. This selects for empty-mat
    frames where no people or movement are present.

    Returns (frame_bgr, frame_index).
    """
    import cv2 as _cv2

    cap = _cv2.VideoCapture(video_path)
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read any frame from: {video_path}")
        return frame, 0

    # Sample evenly spaced frames
    indices = np.linspace(0, total - 1, min(n_candidates, total), dtype=int)
    samples: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (idx, gray, bgr)
    for fi in indices:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok and frame is not None:
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            samples.append((int(fi), gray, frame))
    cap.release()

    if not samples:
        raise RuntimeError(f"Could not read frames from: {video_path}")
    if len(samples) == 1:
        return samples[0][2], samples[0][0]

    # Compute temporal median
    grays = np.array([g for _, g, _ in samples], dtype=np.float32)
    median = np.median(grays, axis=0)

    # Pick frame closest to median (least activity)
    best_idx = 0
    best_diff = float("inf")
    for i, (fi, gray, bgr) in enumerate(samples):
        diff = float(np.mean(np.abs(gray.astype(np.float32) - median)))
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    fi, _, bgr = samples[best_idx]
    return bgr, fi


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


def _detect_lines_near_polylines(
    frame_bgr: np.ndarray,
    polyline_data: Dict[str, Any],
    *,
    proximity_px: float = 25.0,
    canny_low: int = 40,
    canny_high: int = 120,
    hough_threshold: int = 50,
    hough_min_length: int = 40,
    hough_max_gap: int = 15,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Detect Hough line segments spatially filtered to areas near projected polylines.

    Builds a binary proximity mask from projected polylines, runs Canny on the full
    frame, ANDs edges with the mask (rejecting benches/walls/fixtures), then runs
    HoughLinesP on the filtered edge map.

    Returns line segments in the same format as _detect_lines_in_frame.
    """
    import cv2 as _cv2

    gray = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    # Build proximity mask from projected polylines
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    thickness = max(1, int(round(2 * proximity_px)))
    for pl in polyline_data.get("polylines", []):
        pts = pl.get("pixel_points", [])
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            p1 = (int(round(float(pts[i][0]))), int(round(float(pts[i][1]))))
            p2 = (int(round(float(pts[i + 1][0]))), int(round(float(pts[i + 1][1]))))
            _cv2.line(mask, p1, p2, 255, thickness)

    # Canny on full frame, then AND with proximity mask
    edges = _cv2.Canny(gray, canny_low, canny_high)
    filtered_edges = _cv2.bitwise_and(edges, mask)

    # HoughLinesP on filtered edges
    raw_lines = _cv2.HoughLinesP(
        filtered_edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=hough_min_length, maxLineGap=hough_max_gap,
    )

    if raw_lines is None:
        return []

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segments


def _polyline_lens_calibration_v2(
    frame_bgr: np.ndarray,
    polyline_data: Dict[str, Any],
    image_wh: Tuple[int, int],
    *,
    proximity_px: float = 10.0,
    canny_low: int = 40,
    canny_high: int = 120,
    min_edge_pixels: int = 20,
    min_edges_with_pixels: int = 4,
    f_bounds: Optional[Tuple[float, float]] = None,
    k_bounds: Tuple[float, float] = (-1.0, 1.0),
    core_max_dist_px: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Lens calibration using Canny edge pixels grouped by polyline proximity.

    No world coordinates needed — only collinearity constraint per edge group.
    Each group of Canny pixels near a projected polyline should lie on a straight
    line after correct undistortion.

    Two-stage filtering:
      1. Proximity mask groups Canny pixels by nearest polyline edge
      2. Core extraction: fit line to each group, keep only pixels within
         core_max_dist_px of the fitted line — removes parallel shadow/texture
         edges that inflate RMS and mislead the optimizer

    Returns (K, dist_4, lens_metrics) or (None, None, metrics) if insufficient data.
    """
    import cv2 as _cv2
    from scipy.optimize import minimize as sp_minimize

    img_w, img_h = image_wh
    gray = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2GRAY)

    # --- Canny on full frame ---
    edges = _cv2.Canny(gray, canny_low, canny_high)

    # --- Group Canny pixels by nearest polyline (per-polyline mask approach) ---
    polylines = polyline_data.get("polylines", [])
    thickness = max(1, int(round(2 * proximity_px)))

    # Track which pixels are already claimed to avoid double-assignment
    claimed = np.zeros((img_h, img_w), dtype=np.uint8)
    edge_groups: Dict[int, np.ndarray] = {}  # edge_index -> (N, 2) pixel coords

    for pl in polylines:
        pts = pl.get("pixel_points", [])
        if len(pts) < 2:
            continue
        eidx = pl["edge_index"]

        # Draw this polyline's proximity mask
        pl_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(len(pts) - 1):
            p1 = (int(round(float(pts[i][0]))), int(round(float(pts[i][1]))))
            p2 = (int(round(float(pts[i + 1][0]))), int(round(float(pts[i + 1][1]))))
            _cv2.line(pl_mask, p1, p2, 255, thickness)

        # AND with Canny edges, exclude already-claimed pixels
        hits = _cv2.bitwise_and(edges, pl_mask)
        hits = _cv2.bitwise_and(hits, _cv2.bitwise_not(claimed))

        ys, xs = np.nonzero(hits)
        if len(xs) == 0:
            continue

        coords = np.column_stack([xs, ys]).astype(np.float64)

        if eidx in edge_groups:
            edge_groups[eidx] = np.vstack([edge_groups[eidx], coords])
        else:
            edge_groups[eidx] = coords

        # Mark these pixels as claimed
        claimed[ys, xs] = 255

    # --- Pre-filter + core extraction ---
    def _line_fit_residuals(pts_2d: np.ndarray) -> np.ndarray:
        """Signed perpendicular distances from a fitted line (SVD)."""
        if pts_2d.shape[0] < 2:
            return np.zeros(pts_2d.shape[0], dtype=np.float64)
        centroid = pts_2d.mean(axis=0)
        centered = pts_2d - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[1]

    filtered_groups: Dict[int, np.ndarray] = {}
    rejected_reasons: List[str] = []
    for eidx, coords in edge_groups.items():
        if coords.shape[0] < min_edge_pixels:
            rejected_reasons.append(f"edge {eidx}: {coords.shape[0]} pixels < {min_edge_pixels}")
            continue
        # Core extraction: fit line, keep only pixels within core_max_dist_px
        resid = _line_fit_residuals(coords)
        core_mask = np.abs(resid) <= core_max_dist_px
        core = coords[core_mask]
        if core.shape[0] < min_edge_pixels:
            rejected_reasons.append(f"edge {eidx}: {core.shape[0]} core pixels < {min_edge_pixels}")
            continue
        filtered_groups[eidx] = core

    n_groups = len(filtered_groups)
    total_pixels = sum(c.shape[0] for c in filtered_groups.values())

    if n_groups < min_edges_with_pixels:
        return None, None, {
            "reason": f"too few edge groups ({n_groups} < {min_edges_with_pixels})",
            "n_edge_groups": n_groups,
            "n_total_pixels": total_pixels,
            "rejected": rejected_reasons,
            "method": "canny_pixel_collinearity",
        }

    # --- Build flat arrays for optimizer ---
    all_pts = np.vstack(list(filtered_groups.values()))  # (N, 2)
    group_indices: Dict[int, List[int]] = {}
    offset = 0
    for eidx, coords in filtered_groups.items():
        n = coords.shape[0]
        group_indices[eidx] = list(range(offset, offset + n))
        offset += n

    cx = img_w / 2.0
    cy = img_h / 2.0

    def _collinearity_cost(params: np.ndarray) -> float:
        f, k1, k2 = float(params[0]), float(params[1]), float(params[2])
        K_trial = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist_trial = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)

        pts = all_pts.reshape(-1, 1, 2).astype(np.float64)
        undist = _cv2.undistortPoints(pts, K_trial, dist_trial, P=K_trial).reshape(-1, 2)

        total = 0.0
        for idxs in group_indices.values():
            if len(idxs) < 2:
                continue
            edge_pts = undist[idxs]
            centroid = edge_pts.mean(axis=0)
            centered = edge_pts - centroid
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            resid = centered @ vt[1]
            total += float(np.sum(resid ** 2))
        return total

    # --- Powell optimization ---
    # Compute f bounds from image dimensions if not provided.
    # Range 0.4x-1.0x of max(w,h) covers wide-angle to moderate FOV consumer cameras
    # and prevents f/k degeneracy where high-f + weak-k mimics the correct solution.
    max_dim = float(max(img_w, img_h))
    if f_bounds is None:
        f_bounds = (0.4 * max_dim, 1.0 * max_dim)
    f0 = max(f_bounds[0], min(0.7 * max_dim, f_bounds[1]))
    result = sp_minimize(
        _collinearity_cost,
        x0=np.array([f0, 0.0, 0.0]),
        method="Powell",
        bounds=[f_bounds, k_bounds, k_bounds],
        options={"maxiter": 5000, "ftol": 1e-6},
    )

    f_opt, k1_opt, k2_opt = float(result.x[0]), float(result.x[1]), float(result.x[2])

    # --- Validation: reject if any param hits its bound ---
    tol = 1e-3
    if (abs(f_opt - f_bounds[0]) < tol or abs(f_opt - f_bounds[1]) < tol
            or abs(k1_opt - k_bounds[0]) < tol or abs(k1_opt - k_bounds[1]) < tol
            or abs(k2_opt - k_bounds[0]) < tol or abs(k2_opt - k_bounds[1]) < tol):
        return None, None, {
            "reason": "parameter hit bound",
            "f": round(f_opt, 1),
            "k1": round(k1_opt, 6),
            "k2": round(k2_opt, 6),
            "f_bounds": list(f_bounds),
            "k_bounds": list(k_bounds),
            "collinearity_cost": round(float(result.fun), 1),
            "n_edge_groups": n_groups,
            "n_total_pixels": total_pixels,
            "method": "canny_pixel_collinearity",
        }

    # --- Per-edge RMS after optimization ---
    K = np.array([[f_opt, 0, cx], [0, f_opt, cy], [0, 0, 1]], dtype=np.float64)
    dist_4 = np.array([k1_opt, k2_opt, 0.0, 0.0], dtype=np.float64)
    pts_undist = _cv2.undistortPoints(
        all_pts.reshape(-1, 1, 2).astype(np.float64), K, dist_4, P=K,
    ).reshape(-1, 2)

    per_edge_rms: Dict[int, float] = {}
    for eidx, idxs in group_indices.items():
        edge_pts = pts_undist[idxs]
        resid = _line_fit_residuals(edge_pts)
        per_edge_rms[eidx] = float(np.sqrt(np.mean(resid ** 2)))

    mean_cost_per_pt = float(result.fun) / max(1, total_pixels)

    metrics: Dict[str, Any] = {
        "method": "canny_pixel_collinearity",
        "f": round(f_opt, 1),
        "k1": round(k1_opt, 6),
        "k2": round(k2_opt, 6),
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "collinearity_cost": round(float(result.fun), 1),
        "mean_cost_per_pixel": round(mean_cost_per_pt, 4),
        "n_edge_groups": n_groups,
        "n_total_pixels": total_pixels,
        "per_edge_rms": {str(k): round(v, 3) for k, v in per_edge_rms.items()},
        "converged": bool(result.success),
        "optimizer_iterations": int(result.nit),
        "image_size": [img_w, img_h],
    }

    return K, dist_4, metrics


def _extract_correspondences_from_matches(
    matched_lines: List[DetectedMatLine],
    all_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    H_current: np.ndarray,
    n_samples_per_line: int = 15,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], set]:
    """Extract world↔pixel correspondences from matched lines.

    For each matched line, samples points along the blueprint edge and finds
    the closest point on the detected pixel segment.

    Returns (world_corr, pixel_corr, matched_edge_set).
    """
    world_corr: List[Tuple[float, float]] = []
    pixel_corr: List[Tuple[float, float]] = []
    matched_edge_set: set = set()

    for ml in matched_lines:
        eidx = ml.matched_edge_index
        if eidx < 0 or eidx >= len(all_edges):
            continue
        matched_edge_set.add(eidx)
        (ewx1, ewy1), (ewx2, ewy2) = all_edges[eidx]

        for k in range(n_samples_per_line):
            t = k / max(1, n_samples_per_line - 1)
            w_x = ewx1 + t * (ewx2 - ewx1)
            w_y = ewy1 + t * (ewy2 - ewy1)

            px1, py1 = ml.pixel_start
            px2, py2 = ml.pixel_end
            seg_dx = px2 - px1
            seg_dy = py2 - py1
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy

            # Project world point to pixel using current H
            wp = np.array([w_x, w_y, 1.0], dtype=np.float64)
            proj = H_current @ wp
            if abs(proj[2]) < 1e-12:
                continue
            approx_px = proj[0] / proj[2]
            approx_py = proj[1] / proj[2]

            if seg_len_sq < 1e-12:
                continue
            t_seg = ((approx_px - px1) * seg_dx + (approx_py - py1) * seg_dy) / seg_len_sq
            if t_seg < 0.02 or t_seg > 0.98:
                continue
            near_x = px1 + t_seg * seg_dx
            near_y = py1 + t_seg * seg_dy

            world_corr.append((w_x, w_y))
            pixel_corr.append((near_x, near_y))

    return world_corr, pixel_corr, matched_edge_set


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
        # Generate projected polylines from current H (strict in-frame for matching)
        polyline_data = _generate_projected_polylines(
            H_mat_to_img=H_current, rects=rects, image_wh=image_wh,
            frame_margin=0.0,
        )

        # Detect lines with spatial filtering near projected polylines
        filtered_lines = _detect_lines_near_polylines(frame_work, polyline_data)

        # Merge collinear segments
        merged = _merge_collinear_segments(filtered_lines)
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
        world_corr, pixel_corr, matched_edge_set = _extract_correspondences_from_matches(
            matched, all_edges, H_current,
        )
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

        if inlier_ratio < 0.15:
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
    projected_polylines: Optional[Dict[str, Any]] = None,
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

    # CP19: Render projected panel edge polylines in green
    if projected_polylines is not None:
        for pl in projected_polylines.get("polylines", []):
            pts = pl.get("pixel_points", [])
            if len(pts) >= 2:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                ax.plot(xs, ys, color="lime", linewidth=1.5, alpha=0.8, clip_on=True)

    # CP19: Render quality metrics text block at top-right
    if quality_metrics is not None:
        # Support both old ("h_metrics"/"lens_metrics") and new key names
        hm = quality_metrics.get("h_refinement_metrics", quality_metrics.get("h_metrics", {}))
        cm = quality_metrics.get("calibration_metrics", quality_metrics.get("lens_metrics", {}))
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
        if cm and cm.get("f") is not None:
            lines.append(
                f"Lens: f={cm.get('f', 0):.0f} k1={cm.get('k1', 0):.4f} "
                f"k2={cm.get('k2', 0):.4f} RMS={cm.get('rms_reproj_error', 0):.2f}px "
                f"({cm.get('n_correspondences', 0)} pts)"
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

    # CP19: Select emptiest frame via temporal median comparison
    frame_bgr, frame_idx = _find_empty_frame(str(video_path))
    print(f"[D7] Selected frame {frame_idx} (lowest activity)")

    # CP19: Work on RAW frame — no startup undistortion.
    frame_bgr_raw = frame_bgr

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

    state = {"expect": "img", "qa_active": False}  # img then mat alternating
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
        if state.get("qa_active"):
            return
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

            # H₀ maps mat → raw pixel (user placed points on raw frame)
            H_raw, mask = cv2.findHomography(mat_pts, img_pts, method=cv2.RANSAC)
            H_raw = _ensure_3x3(H_raw)
            inliers = int(mask.sum()) if mask is not None else None
            img_h_px, img_w_px = frame_rgb.shape[:2]
            image_wh = (img_w_px, img_h_px)

            # Check if K+dist exist (from prior lens_calibration.py run)
            K_existing, dist_existing = _load_lens_calibration(out_path)

            if K_existing is not None:
                # === MODE B: K+dist exist — undistort + refine H ===
                print("[CP19] Mode B: K+dist found, running mat-line H refinement...")

                undist_anchor = cv2.undistortPoints(
                    img_pts.reshape(-1, 1, 2).astype(np.float64),
                    K_existing, dist_existing, P=K_existing,
                ).reshape(-1, 2)
                H_undist, _ = cv2.findHomography(mat_pts, undist_anchor, method=0)
                H_undist = _ensure_3x3(H_undist)

                frame_undist = cv2.undistort(frame_bgr_raw, K_existing, dist_existing)

                H_refined, h_metrics = _refine_h_from_mat_lines(
                    H_initial=H_undist,
                    frame_bgr=frame_undist,
                    rects=rects,
                    anchor_img_pts=undist_anchor,
                    anchor_mat_pts=mat_pts,
                    camera_matrix=None,
                    dist_coefficients=None,
                )
                print(f"[CP19] H refinement: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                      f"{h_metrics['n_matched_lines']} lines matched")

                H_final = H_refined
                frame_display = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
                quality_metrics = {
                    "h_refinement_metrics": h_metrics,
                    "calibration_mode": "unified_v2",
                }
            else:
                # === MODE A: No K+dist — try automated lens cal ===
                print("[CP19] Mode A: No lens calibration found. Attempting automated lens cal...")

                polyline_data_raw = _generate_projected_polylines(
                    H_mat_to_img=H_raw, rects=rects, image_wh=image_wh, frame_margin=0.0,
                )
                K_auto, dist_auto, lens_metrics = _polyline_lens_calibration_v2(
                    frame_bgr=frame_bgr_raw,
                    polyline_data=polyline_data_raw,
                    image_wh=image_wh,
                )

                if K_auto is not None:
                    print(f"[CP19] Auto lens cal: f={lens_metrics['f']:.1f} "
                          f"k1={lens_metrics['k1']:.4f} k2={lens_metrics['k2']:.4f} "
                          f"cost={lens_metrics['collinearity_cost']:.1f}")

                    # Transition to Mode B flow with auto-calibrated K+dist
                    K_existing, dist_existing = K_auto, dist_auto

                    undist_anchor = cv2.undistortPoints(
                        img_pts.reshape(-1, 1, 2).astype(np.float64),
                        K_existing, dist_existing, P=K_existing,
                    ).reshape(-1, 2)
                    H_undist, _ = cv2.findHomography(mat_pts, undist_anchor, method=0)
                    H_undist = _ensure_3x3(H_undist)

                    frame_undist = cv2.undistort(frame_bgr_raw, K_existing, dist_existing)

                    H_refined, h_metrics = _refine_h_from_mat_lines(
                        H_initial=H_undist,
                        frame_bgr=frame_undist,
                        rects=rects,
                        anchor_img_pts=undist_anchor,
                        anchor_mat_pts=mat_pts,
                        camera_matrix=None,
                        dist_coefficients=None,
                    )
                    print(f"[CP19] H refinement: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                          f"{h_metrics['n_matched_lines']} lines matched")

                    H_final = H_refined
                    frame_display = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
                    quality_metrics = {
                        "h_refinement_metrics": h_metrics,
                        "lens_metrics": lens_metrics,
                        "calibration_mode": "unified_v2",
                    }
                else:
                    print(f"[CP19] Auto lens cal failed: {lens_metrics.get('reason', 'unknown')}")
                    print("[CP19] Saving 4-point H. Run lens_calibration.py manually for better results.")

                    H_final = H_raw
                    frame_display = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2RGB)
                    quality_metrics = {
                        "calibration_mode": "initial_4point",
                    }

            # QA overlay
            qa_polylines = _generate_projected_polylines(
                H_mat_to_img=H_final, rects=rects, image_wh=image_wh, frame_margin=0.0,
            )
            print("[D7] Launching QA overlay... (accept=a, redo=r)")
            state["qa_active"] = True
            accepted = _qa_overlay_dialog(
                camera_id=camera_id,
                frame_rgb=frame_display,
                H_mat_to_img=H_final,
                rects=rects,
                grid_spacing_m=0.5,
                sample_step_m=0.05,
                quality_metrics=quality_metrics,
                projected_polylines=qa_polylines,
            )
            state["qa_active"] = False
            if not accepted:
                print("[D7] QA requested redo. Clearing points; please re-select correspondences.")
                _clear_all()
                fig.canvas.draw_idle()
                return

            # Generate polylines from final H and save
            polyline_data = _generate_projected_polylines(
                H_mat_to_img=H_final, rects=rects, image_wh=image_wh,
            )
            print(f"[D7] Generated {polyline_data['n_polylines']} projected polylines "
                  f"from {polyline_data['n_edges_total']} panel edges")

            extra: Dict[str, Any] = {
                "correspondences": {
                    "image_points_px": pairs.image_points_px,  # always raw pixel space
                    "mat_points": pairs.mat_points,
                },
                "fit": {
                    "method": "cv2.findHomography",
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

            if K_existing is not None:
                extra["camera_matrix"] = K_existing.tolist()
                extra["dist_coefficients"] = dist_existing.tolist()
            else:
                extra["camera_matrix"] = None
                extra["dist_coefficients"] = None

            _write_homography_json(
                out_path=out_path,
                camera_id=camera_id,
                H=H_final,
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

    # CP19: Select emptiest frame via temporal median comparison
    frame_bgr, frame_idx = _find_empty_frame(str(video_path))
    print(f"[D7] Selected frame {frame_idx} (lowest activity)")

    # CP19: Work on RAW frame — no startup undistortion.
    # Phase A computes K+dist from scratch; no dependency on prior calibration.
    frame_bgr_raw = frame_bgr  # raw frame for Phase A + Phase B

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame_rgb.shape[:2]

    blueprint = _try_load_mat_blueprint(mat_blueprint_path)
    if blueprint is None:
        raise RuntimeError(f"[D7] mat_blueprint not found or unreadable: {mat_blueprint_path}")
    rects = _parse_rects_from_blueprint(blueprint)
    if not rects:
        raise RuntimeError("[D7] mat_blueprint had no valid rectangles; cannot build overlay_rect UI.")

    corner_ids = ["tl", "tr", "br", "bl"]

    fig = plt.figure(figsize=(16, 10))
    ax_img = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    fig.suptitle(f"Homography Calibrator (overlay_rect) — {camera_id}")
    # Base frame image
    ax_img.imshow(frame_rgb)
    ax_img.set_axis_off()
    # Allow the quad to extend beyond the frame by padding the view.
    pad = max(20, int(0.03 * max(img_w, img_h)))
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
    # Semi-transparent so the frame beneath is visible for precise placement.
    poly = Polygon(state["img_pts"], closed=True, fill=False, linewidth=2, alpha=0.4)
    ax_img.add_patch(poly)

    corner_text: List[Any] = []
    for (u, v), cid in zip(state["img_pts"], corner_ids):
        corner_text.append(ax_img.text(u, v, cid, fontsize=10, ha="left", va="bottom", alpha=0.5))

    corner_handle_artists: List[Any] = []
    for (u, v) in state["img_pts"]:
        hdl = ax_img.plot([u], [v], marker="o", linestyle="none", alpha=0.4)[0]
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
        # Ignore key events when CP19 QA dialog is active (prevents key leaks)
        if state.get("qa_active"):
            return
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

            # H₀ from anchor on raw frame (always)
            H_raw, _ = cv2.findHomography(np.asarray(mat_pts, dtype=float), np.asarray(img_pts2, dtype=float), method=0)
            H_raw = _ensure_3x3(H_raw)
            image_wh = (img_w, img_h)

            # Check if K+dist exist (from prior lens_calibration.py run)
            K_existing, dist_existing = _load_lens_calibration(out_path)

            if K_existing is not None:
                # === MODE B: K+dist exist — undistort + refine H ===
                print("[CP19] Mode B: K+dist found, running mat-line H refinement...")

                undist_anchor = cv2.undistortPoints(
                    img_pts2.reshape(-1, 1, 2).astype(np.float64),
                    K_existing, dist_existing, P=K_existing,
                ).reshape(-1, 2)
                H_undist, _ = cv2.findHomography(np.asarray(mat_pts, dtype=float), undist_anchor, method=0)
                H_undist = _ensure_3x3(H_undist)

                frame_undist = cv2.undistort(frame_bgr_raw, K_existing, dist_existing)

                H_refined, h_metrics = _refine_h_from_mat_lines(
                    H_initial=H_undist,
                    frame_bgr=frame_undist,
                    rects=rects,
                    anchor_img_pts=undist_anchor,
                    anchor_mat_pts=np.asarray(mat_pts, dtype=float),
                    camera_matrix=None,
                    dist_coefficients=None,
                )
                print(f"[CP19] H refinement: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                      f"{h_metrics['n_matched_lines']} lines matched")

                H_final = H_refined
                frame_display = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
                quality_metrics = {
                    "h_refinement_metrics": h_metrics,
                    "calibration_mode": "unified_v2",
                }
            else:
                # === MODE A: No K+dist — try automated lens cal ===
                print("[CP19] Mode A: No lens calibration found. Attempting automated lens cal...")

                polyline_data_raw = _generate_projected_polylines(
                    H_mat_to_img=H_raw, rects=rects, image_wh=image_wh, frame_margin=0.0,
                )
                K_auto, dist_auto, lens_metrics = _polyline_lens_calibration_v2(
                    frame_bgr=frame_bgr_raw,
                    polyline_data=polyline_data_raw,
                    image_wh=image_wh,
                )

                if K_auto is not None:
                    print(f"[CP19] Auto lens cal: f={lens_metrics['f']:.1f} "
                          f"k1={lens_metrics['k1']:.4f} k2={lens_metrics['k2']:.4f} "
                          f"cost={lens_metrics['collinearity_cost']:.1f}")

                    # Transition to Mode B flow with auto-calibrated K+dist
                    K_existing, dist_existing = K_auto, dist_auto

                    undist_anchor = cv2.undistortPoints(
                        img_pts2.reshape(-1, 1, 2).astype(np.float64),
                        K_existing, dist_existing, P=K_existing,
                    ).reshape(-1, 2)
                    H_undist, _ = cv2.findHomography(np.asarray(mat_pts, dtype=float), undist_anchor, method=0)
                    H_undist = _ensure_3x3(H_undist)

                    frame_undist = cv2.undistort(frame_bgr_raw, K_existing, dist_existing)

                    H_refined, h_metrics = _refine_h_from_mat_lines(
                        H_initial=H_undist,
                        frame_bgr=frame_undist,
                        rects=rects,
                        anchor_img_pts=undist_anchor,
                        anchor_mat_pts=np.asarray(mat_pts, dtype=float),
                        camera_matrix=None,
                        dist_coefficients=None,
                    )
                    print(f"[CP19] H refinement: reproj={h_metrics['mean_reproj_error_px']:.1f}px, "
                          f"{h_metrics['n_matched_lines']} lines matched")

                    H_final = H_refined
                    frame_display = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
                    quality_metrics = {
                        "h_refinement_metrics": h_metrics,
                        "lens_metrics": lens_metrics,
                        "calibration_mode": "unified_v2",
                    }
                else:
                    print(f"[CP19] Auto lens cal failed: {lens_metrics.get('reason', 'unknown')}")
                    print("[CP19] Saving 4-point H. Run lens_calibration.py manually for better results.")

                    H_final = H_raw
                    frame_display = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2RGB)
                    quality_metrics = {
                        "calibration_mode": "initial_4point",
                    }

            # QA overlay
            qa_polylines = _generate_projected_polylines(
                H_mat_to_img=H_final, rects=rects, image_wh=image_wh, frame_margin=0.0,
            )
            state["qa_active"] = True
            accepted = _qa_overlay_dialog(
                camera_id=camera_id,
                frame_rgb=frame_display,
                rects=rects,
                H_mat_to_img=H_final,
                grid_spacing_m=grid_spacing_m,
                sample_step_m=sample_step_m,
                quality_metrics=quality_metrics,
                projected_polylines=qa_polylines,
            )
            state["qa_active"] = False
            if not accepted:
                print("[D7] QA rejected. Continue adjusting overlay, then press 's' again."); return

            # Generate polylines from final H and save
            polyline_data = _generate_projected_polylines(
                H_mat_to_img=H_final, rects=rects, image_wh=image_wh,
            )
            print(f"[D7] Generated {polyline_data['n_polylines']} projected polylines "
                  f"from {polyline_data['n_edges_total']} panel edges")

            extra_save: Dict[str, Any] = {
                "correspondences": {
                    "image_points_px": img_pts2.tolist(),  # always raw pixel space
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

            if K_existing is not None:
                extra_save["camera_matrix"] = K_existing.tolist()
                extra_save["dist_coefficients"] = dist_existing.tolist()
            else:
                extra_save["camera_matrix"] = None
                extra_save["dist_coefficients"] = None

            _write_homography_json(
                out_path=out_path,
                camera_id=camera_id,
                H=H_final,
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
