"""Layer 2 — Cross-camera alignment via spatial fingerprint registration.

After Layer 1 per-camera corrections, cameras with overlapping or adjacent
views are aligned using spatial occupancy patterns — no clock sync needed.

Two methods:
  - Overlap: occupancy grid cross-correlation (shared coverage area)
  - Adjacent: boundary contour stitching (exit vectors ↔ entry vectors)

CP18 v2: Replaces temporal point matching (v1) with spatial fingerprints.
Clock-sync independent. Handles overlap via cross-correlation and adjacency
via boundary contour stitching.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.mat_walk import (
    CalibrationResult,
    _affine_params_to_matrix,
    _apply_affine_single,
    _apply_affine_batch,
    _identity_params,
)
from calibration_pipeline.tracklet_classifier import TrackletFeatures


@dataclass
class SpatialFingerprint:
    """Occupancy grid from cleaning tracklet positions."""

    camera_id: str
    grid: np.ndarray  # 2D binary array (occupied/not)
    cell_size: float  # meters per cell
    origin: tuple[float, float]  # world-space origin (x_min, y_min)
    n_occupied_cells: int
    bounding_box: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


@dataclass
class CrossCameraResult:
    camera_pair: tuple[str, str]
    method: str  # "overlap" | "adjacent" | "none"
    pairwise_correction: Optional[np.ndarray]  # 2x3 affine: cam_b → cam_a
    confidence: str  # "high" | "medium" | "low" | "none"
    n_correspondences: int = 0
    mean_residual: float = 0.0
    details: dict = field(default_factory=dict)


def align_camera_pair(
    features_a: list[TrackletFeatures],
    features_b: list[TrackletFeatures],
    correction_a: CalibrationResult,
    correction_b: CalibrationResult,
    blueprint: MatBlueprint,
    time_window_s: float = 30.0,
    spatial_threshold_m: float = 3.0,
    ransac_iterations: int = 100,
    ransac_inlier_threshold: float = 0.5,
) -> CrossCameraResult:
    """Align a pair of cameras using spatial fingerprint registration.

    Parameters
    ----------
    features_a, features_b : list[TrackletFeatures]
        Classified tracklets for each camera.
    correction_a, correction_b : CalibrationResult
        Layer 1 correction results (applied before fingerprinting).
    blueprint : MatBlueprint
        Mat blueprint for reference.
    time_window_s : float
        Unused in v2 (kept for API compat). Was handoff time window.
    spatial_threshold_m : float
        Max distance for boundary vector matching.
    ransac_iterations : int
        RANSAC iterations for affine fit.
    ransac_inlier_threshold : float
        Maximum residual for RANSAC inlier.
    """
    cam_a = correction_a.camera_id
    cam_b = correction_b.camera_id

    result = CrossCameraResult(
        camera_pair=(cam_a, cam_b),
        method="none",
        pairwise_correction=None,
        confidence="none",
    )

    # Build spatial fingerprints
    fp_a = build_fingerprint(features_a, correction_a)
    fp_b = build_fingerprint(features_b, correction_b)

    if fp_a is None or fp_b is None:
        result.details = {"reason": "insufficient data for fingerprint"}
        return result

    # Detect relationship
    relationship = detect_camera_relationship(fp_a, fp_b)
    result.details["relationship"] = relationship

    if relationship == "overlap":
        overlap_result = align_overlapping(fp_a, fp_b)
        if overlap_result.pairwise_correction is not None:
            return overlap_result

    if relationship in ("overlap", "adjacent"):
        adjacent_result = align_adjacent(
            fp_a, fp_b, features_a, features_b,
            correction_a, correction_b,
            spatial_threshold=spatial_threshold_m,
            ransac_iterations=ransac_iterations,
            ransac_inlier_threshold=ransac_inlier_threshold,
        )
        if adjacent_result.pairwise_correction is not None:
            return adjacent_result

    result.details["reason"] = f"no alignment found ({relationship})"
    return result


def build_fingerprint(
    features: list[TrackletFeatures],
    correction: CalibrationResult,
    cell_size: float = 0.25,
) -> Optional[SpatialFingerprint]:
    """Build occupancy grid from corrected cleaning positions.

    Parameters
    ----------
    features : Classified tracklets.
    correction : Layer 1 result (correction_matrix applied to positions).
    cell_size : Grid resolution in meters.

    Returns
    -------
    SpatialFingerprint or None if insufficient data.
    """
    affine = (
        correction.correction_matrix
        if correction.correction_matrix is not None
        else np.eye(3)[:2]
    )

    # Collect corrected cleaning positions
    cleaning_pts = []
    for f in features:
        if f.classification == "cleaning":
            for x, y in f.positions:
                cx, cy = _apply_affine_single(x, y, affine)
                cleaning_pts.append((cx, cy))

    if len(cleaning_pts) < 10:
        return None

    pts = np.array(cleaning_pts)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Build grid
    n_cols = max(1, int(math.ceil((x_max - x_min) / cell_size)))
    n_rows = max(1, int(math.ceil((y_max - y_min) / cell_size)))
    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)

    for cx, cy in cleaning_pts:
        col = min(int((cx - x_min) / cell_size), n_cols - 1)
        row = min(int((cy - y_min) / cell_size), n_rows - 1)
        grid[row, col] = 1

    n_occupied = int(grid.sum())

    return SpatialFingerprint(
        camera_id=correction.camera_id,
        grid=grid,
        cell_size=cell_size,
        origin=(float(x_min), float(y_min)),
        n_occupied_cells=n_occupied,
        bounding_box=(float(x_min), float(y_min), float(x_max), float(y_max)),
    )


def detect_camera_relationship(
    fp_a: SpatialFingerprint,
    fp_b: SpatialFingerprint,
) -> str:
    """Determine if two cameras overlap, are adjacent, or separated.

    Returns "overlap", "adjacent", or "separated".
    """
    bbox_a = fp_a.bounding_box
    bbox_b = fp_b.bounding_box

    intersection = _bbox_intersection(bbox_a, bbox_b)

    if intersection is not None:
        int_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
        area_a = max(1e-6, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
        area_b = max(1e-6, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
        overlap_frac = max(int_area / area_a, int_area / area_b)

        if overlap_frac > 0.1:
            return "overlap"

    # Check adjacency: gap < 2 meters in any direction
    gap_x = max(0, max(bbox_a[0], bbox_b[0]) - min(bbox_a[2], bbox_b[2]))
    gap_y = max(0, max(bbox_a[1], bbox_b[1]) - min(bbox_a[3], bbox_b[3]))
    gap = math.sqrt(gap_x**2 + gap_y**2)

    if gap < 2.0:
        return "adjacent"

    return "separated"


def align_overlapping(
    fp_a: SpatialFingerprint,
    fp_b: SpatialFingerprint,
) -> CrossCameraResult:
    """Occupancy grid cross-correlation for overlapping cameras.

    Slides grid B over grid A at integer cell offsets.
    Score = count of co-occupied cells at each offset.
    Best offset = translation correction.
    """
    cam_a = fp_a.camera_id
    cam_b = fp_b.camera_id
    result = CrossCameraResult(
        camera_pair=(cam_a, cam_b),
        method="overlap",
        pairwise_correction=None,
        confidence="none",
    )

    # Compute the offset between grid origins in cell units
    cell_size = fp_a.cell_size
    base_offset_x = (fp_b.origin[0] - fp_a.origin[0]) / cell_size
    base_offset_y = (fp_b.origin[1] - fp_a.origin[1]) / cell_size

    # Search range: +/- search_radius cells
    search_radius = 10
    best_score = 0
    best_dx = 0
    best_dy = 0

    grid_a = fp_a.grid
    grid_b = fp_b.grid

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            # Effective offset for grid B onto grid A's coordinate system
            off_y = int(round(base_offset_y)) + dy
            off_x = int(round(base_offset_x)) + dx

            # Compute overlap region
            a_y_start = max(0, off_y)
            a_y_end = min(grid_a.shape[0], grid_b.shape[0] + off_y)
            a_x_start = max(0, off_x)
            a_x_end = min(grid_a.shape[1], grid_b.shape[1] + off_x)

            b_y_start = max(0, -off_y)
            b_y_end = b_y_start + (a_y_end - a_y_start)
            b_x_start = max(0, -off_x)
            b_x_end = b_x_start + (a_x_end - a_x_start)

            if a_y_end <= a_y_start or a_x_end <= a_x_start:
                continue

            overlap = grid_a[a_y_start:a_y_end, a_x_start:a_x_end].astype(int)
            b_slice = grid_b[b_y_start:b_y_end, b_x_start:b_x_end].astype(int)

            if overlap.shape != b_slice.shape:
                continue

            score = int((overlap * b_slice).sum())
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    if best_score < 3:
        result.details = {
            "reason": "insufficient overlap co-occupancy",
            "best_score": best_score,
        }
        return result

    # Convert cell offset correction to world-space translation
    correction_x = best_dx * cell_size
    correction_y = best_dy * cell_size

    # Build 2x3 affine: translate cam_b positions to align with cam_a
    pairwise = np.array([
        [1.0, 0.0, -correction_x],
        [0.0, 1.0, -correction_y],
    ])

    result.pairwise_correction = pairwise
    result.n_correspondences = best_score
    result.mean_residual = cell_size  # resolution-limited

    # Confidence based on co-occupied cell count
    if best_score >= 20:
        result.confidence = "high"
    elif best_score >= 10:
        result.confidence = "medium"
    else:
        result.confidence = "low"

    result.details = {
        "method": "grid_cross_correlation",
        "best_offset_cells": (best_dx, best_dy),
        "correction_meters": (round(correction_x, 3), round(correction_y, 3)),
        "co_occupied_cells": best_score,
        "search_radius": search_radius,
    }

    return result


def align_adjacent(
    fp_a: SpatialFingerprint,
    fp_b: SpatialFingerprint,
    features_a: list[TrackletFeatures],
    features_b: list[TrackletFeatures],
    correction_a: CalibrationResult,
    correction_b: CalibrationResult,
    spatial_threshold: float = 3.0,
    ransac_iterations: int = 100,
    ransac_inlier_threshold: float = 0.5,
) -> CrossCameraResult:
    """Boundary contour stitching for adjacent cameras.

    Extracts exit/entry vectors near FOV edges:
    - Exit vectors: last N positions of tracklets dying near FOV boundary
    - Entry vectors: first N positions of tracklets born near FOV boundary

    At the shared boundary, exit vectors from A should spatially continue
    as entry vectors from B (and vice versa).
    """
    cam_a = correction_a.camera_id
    cam_b = correction_b.camera_id

    result = CrossCameraResult(
        camera_pair=(cam_a, cam_b),
        method="adjacent",
        pairwise_correction=None,
        confidence="none",
    )

    affine_a = (
        correction_a.correction_matrix
        if correction_a.correction_matrix is not None
        else np.eye(3)[:2]
    )
    affine_b = (
        correction_b.correction_matrix
        if correction_b.correction_matrix is not None
        else np.eye(3)[:2]
    )

    # Extract boundary crossing vectors
    exits_a = _extract_boundary_vectors(features_a, affine_a, fp_a, is_exit=True)
    entries_b = _extract_boundary_vectors(features_b, affine_b, fp_b, is_exit=False)
    exits_b = _extract_boundary_vectors(features_b, affine_b, fp_b, is_exit=True)
    entries_a = _extract_boundary_vectors(features_a, affine_a, fp_a, is_exit=False)

    # Match A exits → B entries
    correspondences = _match_boundary_vectors(
        exits_a, entries_b, spatial_threshold
    )
    # Also match B exits → A entries
    reverse_corr = _match_boundary_vectors(
        exits_b, entries_a, spatial_threshold
    )
    # For reverse correspondences, swap pos_a/pos_b
    for c in reverse_corr:
        correspondences.append({
            "pos_a": c["pos_b"],
            "pos_b": c["pos_a"],
            "distance": c["distance"],
        })

    result.n_correspondences = len(correspondences)

    if len(correspondences) < 3:
        result.details = {
            "reason": f"insufficient boundary correspondences ({len(correspondences)} < 3)",
            "exits_a": len(exits_a),
            "entries_b": len(entries_b),
            "exits_b": len(exits_b),
            "entries_a": len(entries_a),
        }
        return result

    # RANSAC affine fit: cam_b → cam_a
    affine, inliers, residual = _ransac_pairwise(
        correspondences, ransac_iterations, ransac_inlier_threshold,
    )

    if affine is None or len(inliers) < 3:
        result.details = {"reason": "RANSAC failed on boundary vectors"}
        return result

    result.pairwise_correction = affine
    result.mean_residual = residual

    if len(inliers) >= 10 and residual < 0.3:
        result.confidence = "high"
    elif len(inliers) >= 5 and residual < 0.5:
        result.confidence = "medium"
    elif len(inliers) >= 3:
        result.confidence = "low"

    result.details = {
        "method": "boundary_contour_stitching",
        "n_inliers": len(inliers),
        "n_outliers": len(correspondences) - len(inliers),
        "mean_residual_m": round(residual, 4),
    }

    return result


def write_alignment_json(
    result: CrossCameraResult, output_dir: Path
) -> Optional[Path]:
    """Write cross-camera alignment JSON."""
    cross_dir = output_dir / "cross_camera"
    cross_dir.mkdir(parents=True, exist_ok=True)

    cam_a, cam_b = result.camera_pair
    path = cross_dir / f"{cam_a}_{cam_b}_alignment.json"

    payload = {
        "camera_a": cam_a,
        "camera_b": cam_b,
        "method": result.method,
        "confidence": result.confidence,
        "n_correspondences": result.n_correspondences,
        "mean_residual_m": round(result.mean_residual, 4),
        "pairwise_correction": (
            result.pairwise_correction.tolist()
            if result.pairwise_correction is not None
            else None
        ),
        "details": result.details,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_boundary_vectors(
    features: list[TrackletFeatures],
    affine: np.ndarray,
    fingerprint: SpatialFingerprint,
    is_exit: bool,
    n_positions: int = 5,
    boundary_margin: float = 1.0,
) -> list[dict]:
    """Extract exit or entry vectors near the FOV boundary.

    For exits: last N corrected positions of tracklets dying near boundary.
    For entries: first N corrected positions of tracklets born near boundary.
    """
    x_min, y_min, x_max, y_max = fingerprint.bounding_box
    vectors = []

    for f in features:
        if len(f.positions) < n_positions:
            continue

        if is_exit:
            # Check death position near FOV boundary
            cx, cy = _apply_affine_single(*f.death_position, affine)
            near_boundary = (
                abs(cx - x_min) < boundary_margin
                or abs(cx - x_max) < boundary_margin
                or abs(cy - y_min) < boundary_margin
                or abs(cy - y_max) < boundary_margin
            )
            if not near_boundary:
                continue

            # Last N positions
            tail = f.positions[-n_positions:]
            corrected = [_apply_affine_single(x, y, affine) for x, y in tail]
            position = corrected[-1]  # exit point
        else:
            # Check birth position near FOV boundary
            cx, cy = _apply_affine_single(*f.birth_position, affine)
            near_boundary = (
                abs(cx - x_min) < boundary_margin
                or abs(cx - x_max) < boundary_margin
                or abs(cy - y_min) < boundary_margin
                or abs(cy - y_max) < boundary_margin
            )
            if not near_boundary:
                continue

            # First N positions
            head = f.positions[:n_positions]
            corrected = [_apply_affine_single(x, y, affine) for x, y in head]
            position = corrected[0]  # entry point

        # Compute direction vector
        if len(corrected) >= 2:
            dx = corrected[-1][0] - corrected[0][0]
            dy = corrected[-1][1] - corrected[0][1]
            mag = math.sqrt(dx**2 + dy**2)
            if mag > 0.01:
                direction = (dx / mag, dy / mag)
            else:
                direction = (0.0, 0.0)
        else:
            direction = (0.0, 0.0)

        vectors.append({
            "tracklet_id": f.tracklet_id,
            "position": position,
            "direction": direction,
            "positions": corrected,
        })

    return vectors


def _match_boundary_vectors(
    exits: list[dict],
    entries: list[dict],
    spatial_threshold: float,
) -> list[dict]:
    """Match exit vectors to entry vectors by spatial proximity."""
    correspondences = []
    used_entries = set()

    for ex in exits:
        best_dist = float("inf")
        best_entry = None
        best_idx = -1

        for i, en in enumerate(entries):
            if i in used_entries:
                continue
            d = math.sqrt(
                (ex["position"][0] - en["position"][0]) ** 2
                + (ex["position"][1] - en["position"][1]) ** 2
            )
            if d < spatial_threshold and d < best_dist:
                best_dist = d
                best_entry = en
                best_idx = i

        if best_entry is not None:
            correspondences.append({
                "pos_a": ex["position"],
                "pos_b": best_entry["position"],
                "distance": best_dist,
            })
            used_entries.add(best_idx)

    return correspondences


def _bbox_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> Optional[tuple[float, float, float, float]]:
    """Return intersection bbox or None if no overlap."""
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    if x_min < x_max and y_min < y_max:
        return (x_min, y_min, x_max, y_max)
    return None


def _ransac_pairwise(
    correspondences: list[dict],
    n_iterations: int = 100,
    inlier_threshold: float = 0.5,
) -> tuple[Optional[np.ndarray], list[int], float]:
    """RANSAC affine fit: cam_b positions → cam_a positions."""
    n = len(correspondences)
    if n < 3:
        return None, [], float("inf")

    best_affine = None
    best_inliers: list[int] = []
    best_residual = float("inf")

    for _ in range(n_iterations):
        sample = random.sample(range(n), 3)

        src = np.array([correspondences[i]["pos_b"] for i in sample])
        dst = np.array([correspondences[i]["pos_a"] for i in sample])

        affine = _fit_affine_lsq(src, dst)
        if affine is None:
            continue

        inliers = []
        residuals = []
        for i, c in enumerate(correspondences):
            cx, cy = _apply_affine_single(*c["pos_b"], affine)
            d = math.sqrt((cx - c["pos_a"][0]) ** 2 + (cy - c["pos_a"][1]) ** 2)
            if d < inlier_threshold:
                inliers.append(i)
                residuals.append(d)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_residual = sum(residuals) / max(1, len(residuals))
            src_all = np.array([correspondences[i]["pos_b"] for i in inliers])
            dst_all = np.array([correspondences[i]["pos_a"] for i in inliers])
            best_affine = _fit_affine_lsq(src_all, dst_all)

    return best_affine, best_inliers, best_residual


def _fit_affine_lsq(
    src: np.ndarray, dst: np.ndarray
) -> Optional[np.ndarray]:
    """Least-squares affine fit: src → dst.

    src, dst: Nx2 arrays.
    Returns 2x3 affine matrix or None if degenerate.
    """
    n = len(src)
    if n < 3:
        return None

    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        A[2 * i, 0] = src[i, 0]
        A[2 * i, 1] = src[i, 1]
        A[2 * i, 2] = 1.0
        b[2 * i] = dst[i, 0]

        A[2 * i + 1, 3] = src[i, 0]
        A[2 * i + 1, 4] = src[i, 1]
        A[2 * i + 1, 5] = 1.0
        b[2 * i + 1] = dst[i, 1]

    try:
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return _affine_params_to_matrix(params)
    except np.linalg.LinAlgError:
        return None
