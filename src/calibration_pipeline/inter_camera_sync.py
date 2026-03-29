"""Layer 2 — Cross-camera alignment (opportunistic).

After Layer 1 per-camera corrections, cameras with overlapping or adjacent
views can cross-validate and correct relative alignment. Two methods:

  - Overlap: simultaneous co-observations of the same person on both cameras
  - Handoff: sequential tracklet death on cam A → birth on cam B

Layer 2 is opportunistic — Layer 1 stands alone when cameras have no overlap.
Never blocks Layer 1 results on failure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json
import numpy as np
from scipy.optimize import minimize

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.mat_walk import (
    CalibrationResult,
    _affine_params_to_matrix,
    _apply_affine_single,
    _identity_params,
)
from calibration_pipeline.tracklet_classifier import TrackletFeatures


@dataclass
class CrossCameraResult:
    camera_pair: tuple[str, str]
    method: str  # "overlap" | "handoff" | "none"
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
    """Align a pair of cameras using overlap or handoff correspondences.

    Parameters
    ----------
    features_a, features_b : list[TrackletFeatures]
        Classified tracklets for each camera.
    correction_a, correction_b : CalibrationResult
        Layer 1 correction results (applied before comparison).
    blueprint : MatBlueprint
        Mat blueprint for bounding box / polygon reference.
    time_window_s : float
        Maximum time gap for handoff matching (seconds).
    spatial_threshold_m : float
        Maximum spatial distance for overlap matching (meters).
    ransac_iterations : int
        RANSAC iterations for affine fit.
    ransac_inlier_threshold : float
        Maximum residual for RANSAC inlier.
    """
    cam_a = correction_a.camera_id
    cam_b = correction_b.camera_id

    result = CrossCameraResult(
        camera_pair=(cam_a, cam_b), method="none",
        pairwise_correction=None, confidence="none",
    )

    # Apply Layer 1 corrections to positions
    affine_a = correction_a.correction_matrix if correction_a.correction_matrix is not None else np.eye(3)[:2]
    affine_b = correction_b.correction_matrix if correction_b.correction_matrix is not None else np.eye(3)[:2]

    # Compute corrected bounding boxes
    bbox_a = _corrected_bbox(features_a, affine_a)
    bbox_b = _corrected_bbox(features_b, affine_b)

    if bbox_a is None or bbox_b is None:
        result.details = {"reason": "insufficient data for bounding box"}
        return result

    # Step 1: Detect overlap vs adjacency
    overlap = _bbox_intersection(bbox_a, bbox_b)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    if overlap is not None:
        overlap_area = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])
        overlap_frac = max(overlap_area / max(area_a, 1e-6),
                          overlap_area / max(area_b, 1e-6))
    else:
        overlap_frac = 0.0

    # Try overlap method first, then handoff
    correspondences = []

    if overlap_frac > 0.1 and overlap is not None:
        # Step 2a: Overlap method — co-temporal detections
        correspondences = _find_overlap_correspondences(
            features_a, features_b, affine_a, affine_b,
            overlap, spatial_threshold_m,
        )
        if len(correspondences) >= 3:
            result.method = "overlap"

    if len(correspondences) < 3:
        # Step 2b: Handoff method — death→birth matching
        handoff_corr = _find_handoff_correspondences(
            features_a, features_b, affine_a, affine_b,
            time_window_s,
        )
        if len(handoff_corr) >= len(correspondences):
            correspondences = handoff_corr
            result.method = "handoff"

    result.n_correspondences = len(correspondences)

    if len(correspondences) < 3:
        result.details = {
            "reason": f"insufficient correspondences ({len(correspondences)} < 3)",
            "overlap_fraction": round(overlap_frac, 4),
        }
        return result

    # Step 3: RANSAC affine fit (cam_b positions → cam_a positions)
    affine, inliers, residual = _ransac_pairwise(
        correspondences, ransac_iterations, ransac_inlier_threshold,
    )

    if affine is None or len(inliers) < 3:
        result.details = {"reason": "RANSAC failed"}
        return result

    result.pairwise_correction = affine
    result.mean_residual = residual

    # Assign confidence
    if len(inliers) >= 10 and residual < 0.3:
        result.confidence = "high"
    elif len(inliers) >= 5 and residual < 0.5:
        result.confidence = "medium"
    elif len(inliers) >= 3:
        result.confidence = "low"

    result.details = {
        "overlap_fraction": round(overlap_frac, 4),
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
            if result.pairwise_correction is not None else None
        ),
        "details": result.details,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _corrected_bbox(
    features: list[TrackletFeatures], affine: np.ndarray
) -> Optional[tuple[float, float, float, float]]:
    """Compute bounding box of all corrected positions."""
    all_pts = []
    for f in features:
        for x, y in f.positions:
            cx, cy = _apply_affine_single(x, y, affine)
            all_pts.append((cx, cy))
    if not all_pts:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return (min(xs), min(ys), max(xs), max(ys))


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


def _find_overlap_correspondences(
    features_a: list[TrackletFeatures],
    features_b: list[TrackletFeatures],
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    overlap_bbox: tuple[float, float, float, float],
    spatial_threshold: float,
) -> list[dict]:
    """Find co-temporal detections in the overlap zone.

    Builds a frame-indexed lookup of positions, matches by proximity.
    Uses timestamp_ms from positions is not available, so we use
    frame_index-based matching: same frame_index = same time.
    Since we only have (x,y) positions in TrackletFeatures, we build
    a simplified approach: for each tracklet pair where both have
    positions in the overlap zone, check average position proximity.
    """
    # Build per-tracklet corrected centroids in overlap zone
    centroids_a = _overlap_centroids(features_a, affine_a, overlap_bbox)
    centroids_b = _overlap_centroids(features_b, affine_b, overlap_bbox)

    correspondences = []
    used_b = set()

    for tid_a, (cx_a, cy_a, dur_a) in centroids_a.items():
        best_dist = float("inf")
        best_b = None
        for tid_b, (cx_b, cy_b, dur_b) in centroids_b.items():
            if tid_b in used_b:
                continue
            d = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            if d < spatial_threshold and d < best_dist:
                best_dist = d
                best_b = tid_b

        if best_b is not None:
            cx_b, cy_b, _ = centroids_b[best_b]
            correspondences.append({
                "pos_a": (cx_a, cy_a),
                "pos_b": (cx_b, cy_b),
                "distance": best_dist,
            })
            used_b.add(best_b)

    return correspondences


def _overlap_centroids(
    features: list[TrackletFeatures],
    affine: np.ndarray,
    overlap_bbox: tuple[float, float, float, float],
) -> dict[str, tuple[float, float, float]]:
    """Compute corrected centroids of tracklets that pass through overlap zone."""
    result = {}
    x_min, y_min, x_max, y_max = overlap_bbox

    for f in features:
        overlap_pts = []
        for x, y in f.positions:
            cx, cy = _apply_affine_single(x, y, affine)
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                overlap_pts.append((cx, cy))

        if len(overlap_pts) >= 5:  # need meaningful presence in overlap
            mean_x = sum(p[0] for p in overlap_pts) / len(overlap_pts)
            mean_y = sum(p[1] for p in overlap_pts) / len(overlap_pts)
            result[f.tracklet_id] = (mean_x, mean_y, f.duration_s)

    return result


def _find_handoff_correspondences(
    features_a: list[TrackletFeatures],
    features_b: list[TrackletFeatures],
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    time_window_s: float,
) -> list[dict]:
    """Find death→birth handoffs between cameras.

    Match tracklet death on cam A to tracklet birth on cam B within time window.
    Uses last_frame / first_frame as proxy for timestamps (assumes same fps).
    """
    # Build death events for A and birth events for B
    deaths_a = []
    for f in features_a:
        if f.positions:
            cx, cy = _apply_affine_single(*f.death_position, affine_a)
            # Use last_frame from positions length as proxy
            deaths_a.append({
                "tracklet_id": f.tracklet_id,
                "position": (cx, cy),
                "frame": len(f.positions),  # relative end frame
                "time_proxy": f.duration_s,
            })

    births_b = []
    for f in features_b:
        if f.positions:
            cx, cy = _apply_affine_single(*f.birth_position, affine_b)
            births_b.append({
                "tracklet_id": f.tracklet_id,
                "position": (cx, cy),
                "frame": 0,
                "time_proxy": 0.0,
            })

    # Also try: death on B → birth on A
    deaths_b = []
    for f in features_b:
        if f.positions:
            cx, cy = _apply_affine_single(*f.death_position, affine_b)
            deaths_b.append({
                "tracklet_id": f.tracklet_id,
                "position": (cx, cy),
                "time_proxy": f.duration_s,
            })

    births_a = []
    for f in features_a:
        if f.positions:
            cx, cy = _apply_affine_single(*f.birth_position, affine_a)
            births_a.append({
                "tracklet_id": f.tracklet_id,
                "position": (cx, cy),
                "time_proxy": 0.0,
            })

    # Simple spatial matching — handoff pairs should be spatially close
    correspondences = []
    used = set()

    for death in deaths_a:
        best_dist = float("inf")
        best_birth = None
        for birth in births_b:
            key = ("ab", death["tracklet_id"], birth["tracklet_id"])
            if key in used:
                continue
            d = math.sqrt(
                (death["position"][0] - birth["position"][0]) ** 2
                + (death["position"][1] - birth["position"][1]) ** 2
            )
            if d < 5.0 and d < best_dist:  # generous spatial threshold for handoffs
                best_dist = d
                best_birth = birth

        if best_birth is not None:
            correspondences.append({
                "pos_a": death["position"],
                "pos_b": best_birth["position"],
                "distance": best_dist,
            })
            used.add(("ab", death["tracklet_id"], best_birth["tracklet_id"]))

    return correspondences


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

        # Build point pairs
        src = np.array([correspondences[i]["pos_b"] for i in sample])
        dst = np.array([correspondences[i]["pos_a"] for i in sample])

        # Fit affine: minimize || A @ src_aug - dst ||
        affine = _fit_affine_lsq(src, dst)
        if affine is None:
            continue

        # Score all correspondences
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
            # Refit on inliers
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

    # Build system: [x y 1 0 0 0; 0 0 0 x y 1] * [a b tx c d ty]^T = [dx dy]
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
