"""Layer 1 — Single-camera homography refinement via mat cleaning footage.

Uses tracklet birth/death positions near mat edges as correspondences to fit
an affine correction on top of the existing homography. The correction is small
(translation + minor rotation) — the base homography from corner overlay is
assumed roughly correct.

Replaces the CP16b mat_walk stub (grid pattern detection from tagged walker).
The cleaning-footage RANSAC approach was validated by evidence-driven design
in the CP18 exploration phase.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from shapely.geometry import Point
from shapely.prepared import prep as shapely_prep

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import TrackletFeatures


@dataclass
class CalibrationResult:
    camera_id: str
    correction_matrix: Optional[np.ndarray]  # 2x3 affine, None if inconclusive
    confidence: str  # "high" | "medium" | "low" | "inconclusive"

    # Quality metrics
    n_cleaning_tracklets: int = 0
    n_edge_touches: int = 0
    n_distinct_edges: int = 0
    coverage_fraction: float = 0.0

    # Before/after comparison
    inside_mat_fraction_before: float = 0.0
    inside_mat_fraction_after: float = 0.0
    mean_edge_residual_before: float = 0.0
    mean_edge_residual_after: float = 0.0

    # Diagnostics
    n_ransac_inliers: int = 0
    n_ransac_outliers: int = 0
    n_off_mat_walkers: int = 0

    # Audit trail
    details: dict = field(default_factory=dict)


def calibrate_single_camera(
    tracklet_features: list[TrackletFeatures],
    blueprint: MatBlueprint,
    camera_id: str,
    min_coverage_fraction: float = 0.4,
    min_edge_touches: int = 6,
    min_distinct_edges: int = 2,
    ransac_iterations: int = 200,
    ransac_inlier_threshold: float = 0.5,
    w_edge: float = 1.0,
    w_interior: float = 0.1,
    w_negative: float = 0.1,
    w_regularization: float = 0.5,
    cell_size: float = 0.5,
) -> CalibrationResult:
    """Run Layer 1 single-camera homography refinement.

    Parameters
    ----------
    tracklet_features : list[TrackletFeatures]
        Classified tracklets for this camera.
    blueprint : MatBlueprint
        Parsed mat blueprint.
    camera_id : str
        Camera identifier.
    min_coverage_fraction : float
        Minimum fraction of visible area covered by cleaning tracklets.
    min_edge_touches : int
        Minimum valid (perpendicular) edge touches required.
    min_distinct_edges : int
        Minimum distinct blueprint edges touched.
    ransac_iterations : int
        Number of RANSAC iterations.
    ransac_inlier_threshold : float
        Maximum edge residual (m) for a correspondence to be an inlier.
    w_edge, w_interior, w_negative : float
        Cost function weights.
    w_regularization : float
        Penalty weight for deviation from identity transform. Prevents wild
        solutions from noisy 3-point samples — expected corrections are small
        (a few feet translation, 1-2° rotation, negligible scale).
    cell_size : float
        Grid cell size (m) for coverage computation.
    """
    result = CalibrationResult(camera_id=camera_id, correction_matrix=None,
                               confidence="inconclusive")

    if not tracklet_features:
        result.details = {"reason": "no tracklet features provided"}
        return result

    # --- Step 1: Data sufficiency check ---
    cleaning = [f for f in tracklet_features if f.classification == "cleaning"]
    result.n_cleaning_tracklets = len(cleaning)

    # Valid edge touches: perpendicular crossings within distance threshold
    edge_correspondences = _build_edge_correspondences(tracklet_features, blueprint)
    result.n_edge_touches = len(edge_correspondences)
    touched_edges = set(ec["edge_index"] for ec in edge_correspondences)
    result.n_distinct_edges = len(touched_edges)

    # Coverage: grid cells occupied by cleaning tracklets / total visible cells
    all_positions = []
    cleaning_positions = []
    for f in tracklet_features:
        all_positions.extend(f.positions)
        if f.classification == "cleaning":
            cleaning_positions.extend(f.positions)

    if not all_positions:
        result.details = {"reason": "no valid positions"}
        return result

    all_cells = set(
        (int(x / cell_size), int(y / cell_size)) for x, y in all_positions
    )
    clean_cells = set(
        (int(x / cell_size), int(y / cell_size)) for x, y in cleaning_positions
    )
    result.coverage_fraction = len(clean_cells & all_cells) / max(1, len(all_cells))

    # Off-mat walkers: lingering tracklets that are mostly outside the polygon
    off_mat_walkers = [
        f for f in tracklet_features
        if f.classification == "lingering" and f.on_mat_fraction < 0.3
    ]
    result.n_off_mat_walkers = len(off_mat_walkers)

    # Gate check
    gates_passed = True
    gate_failures = []
    if result.n_edge_touches < min_edge_touches:
        gate_failures.append(
            f"edge_touches={result.n_edge_touches} < {min_edge_touches}"
        )
        gates_passed = False
    if result.n_distinct_edges < min_distinct_edges:
        gate_failures.append(
            f"distinct_edges={result.n_distinct_edges} < {min_distinct_edges}"
        )
        gates_passed = False
    if result.coverage_fraction < min_coverage_fraction:
        gate_failures.append(
            f"coverage={result.coverage_fraction:.2f} < {min_coverage_fraction}"
        )
        gates_passed = False

    if not gates_passed:
        result.details = {"reason": "quality gates failed", "failures": gate_failures}
        return result

    # --- Step 2-4: Build constraint sets ---
    # Interior positions (cleaning tracklets, on-mat)
    interior_points = []
    for f in cleaning:
        for x, y in f.positions:
            if blueprint.contains_point(x, y):
                interior_points.append((x, y))

    # Negative constraint positions (off-mat walkers)
    negative_points = []
    for f in off_mat_walkers:
        for x, y in f.positions:
            if not blueprint.contains_point(x, y):
                negative_points.append((x, y))

    # Subsample interior/negative to keep cost function fast
    if len(interior_points) > 50:
        interior_points = random.sample(interior_points, 50)
    if len(negative_points) > 30:
        negative_points = random.sample(negative_points, 30)

    # --- Before metrics ---
    result.inside_mat_fraction_before = _compute_inside_fraction(
        all_positions, blueprint
    )
    result.mean_edge_residual_before = _mean_edge_residual(
        edge_correspondences, blueprint, np.eye(3)[:2]
    )

    # --- Step 5: RANSAC affine fit ---
    best_affine, best_inliers, best_score = _ransac_affine(
        edge_correspondences=edge_correspondences,
        interior_points=interior_points,
        negative_points=negative_points,
        blueprint=blueprint,
        n_iterations=ransac_iterations,
        inlier_threshold=ransac_inlier_threshold,
        w_edge=w_edge,
        w_interior=w_interior,
        w_negative=w_negative,
        w_regularization=w_regularization,
    )

    if best_affine is None:
        result.details = {"reason": "RANSAC failed to find valid solution"}
        return result

    result.n_ransac_inliers = len(best_inliers)
    result.n_ransac_outliers = len(edge_correspondences) - len(best_inliers)

    # Refit on all inliers for final solution
    final_affine = _fit_affine_on_inliers(
        inlier_correspondences=[edge_correspondences[i] for i in best_inliers],
        interior_points=interior_points,
        negative_points=negative_points,
        blueprint=blueprint,
        w_edge=w_edge,
        w_interior=w_interior,
        w_negative=w_negative,
        w_regularization=w_regularization,
    )

    # --- Step 6: Validate correction ---
    corrected_positions = _apply_affine(all_positions, final_affine)
    result.inside_mat_fraction_after = _compute_inside_fraction(
        corrected_positions, blueprint
    )
    result.mean_edge_residual_after = _mean_edge_residual(
        edge_correspondences, blueprint, final_affine
    )

    # If correction made things worse, discard
    if result.inside_mat_fraction_after < result.inside_mat_fraction_before:
        result.confidence = "low"
        result.correction_matrix = None
        result.details = {
            "reason": "correction decreased inside-mat fraction",
            "before": result.inside_mat_fraction_before,
            "after": result.inside_mat_fraction_after,
        }
        return result

    result.correction_matrix = final_affine

    # Assign confidence
    improvement = (
        result.inside_mat_fraction_after - result.inside_mat_fraction_before
    )
    if (
        result.coverage_fraction > 0.6
        and result.n_edge_touches >= 10
        and result.n_distinct_edges >= 3
        and improvement > 0.20
    ):
        result.confidence = "high"
    elif (
        result.coverage_fraction > 0.4
        and result.n_edge_touches >= 6
        and result.n_distinct_edges >= 2
        and improvement > 0.0
    ):
        result.confidence = "medium"
    else:
        result.confidence = "low"

    result.details = {
        "n_interior_constraints": len(interior_points),
        "n_negative_constraints": len(negative_points),
        "improvement": round(improvement, 4),
        "affine_params": {
            "a": float(final_affine[0, 0]),
            "b": float(final_affine[0, 1]),
            "tx": float(final_affine[0, 2]),
            "c": float(final_affine[1, 0]),
            "d": float(final_affine[1, 1]),
            "ty": float(final_affine[1, 2]),
        },
    }

    return result


def write_correction_json(
    result: CalibrationResult, output_dir: Path
) -> Optional[Path]:
    """Write calibration correction JSON for a camera."""
    if result.correction_matrix is None:
        return None

    cam_dir = output_dir / "per_camera" / result.camera_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    path = cam_dir / "calibration_correction.json"

    payload = {
        "camera_id": result.camera_id,
        "correction_matrix": result.correction_matrix.tolist(),
        "confidence": result.confidence,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "n_edge_touches": result.n_edge_touches,
        "n_cleaning_tracklets": result.n_cleaning_tracklets,
        "coverage_fraction": round(result.coverage_fraction, 4),
        "inside_mat_before": round(result.inside_mat_fraction_before, 4),
        "inside_mat_after": round(result.inside_mat_fraction_after, 4),
        "mean_edge_residual_before": round(result.mean_edge_residual_before, 4),
        "mean_edge_residual_after": round(result.mean_edge_residual_after, 4),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_edge_correspondences(
    features: list[TrackletFeatures],
    blueprint: MatBlueprint,
    edge_touch_distance: float = 1.0,
) -> list[dict]:
    """Extract edge correspondences from tracklet birth/death positions.

    Only includes perpendicular crossings within edge_touch_distance.
    """
    correspondences = []
    for f in features:
        if f.birth_edge_distance < edge_touch_distance and f.birth_is_perpendicular:
            correspondences.append({
                "position": f.birth_position,
                "edge_index": f.birth_edge_index,
                "edge_segment": blueprint.boundary_edges[f.birth_edge_index],
                "tracklet_id": f.tracklet_id,
                "type": "birth",
            })
        if f.death_edge_distance < edge_touch_distance and f.death_is_perpendicular:
            correspondences.append({
                "position": f.death_position,
                "edge_index": f.death_edge_index,
                "edge_segment": blueprint.boundary_edges[f.death_edge_index],
                "tracklet_id": f.tracklet_id,
                "type": "death",
            })
    return correspondences


def _affine_params_to_matrix(params: np.ndarray) -> np.ndarray:
    """Convert 6-element parameter vector to 2x3 affine matrix.

    params = [a, b, tx, c, d, ty]
    Matrix = [[a, b, tx], [c, d, ty]]
    """
    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]]])


def _identity_params() -> np.ndarray:
    """Identity affine as parameter vector: [1, 0, 0, 0, 1, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _apply_affine_single(
    x: float, y: float, affine: np.ndarray
) -> tuple[float, float]:
    """Apply 2x3 affine to a single point."""
    cx = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]
    cy = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]
    return (cx, cy)


def _apply_affine(
    positions: list[tuple[float, float]], affine: np.ndarray
) -> list[tuple[float, float]]:
    """Apply 2x3 affine to a list of points."""
    if not positions:
        return []
    pts = np.array(positions)
    ones = np.ones((len(pts), 1))
    augmented = np.hstack([pts, ones])  # Nx3
    result = (affine @ augmented.T).T  # Nx2
    return [(float(r[0]), float(r[1])) for r in result]


def _apply_affine_batch(
    pts: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """Apply 2x3 affine to Nx2 numpy array. Returns Nx2."""
    ones = np.ones((len(pts), 1))
    augmented = np.hstack([pts, ones])
    return (affine @ augmented.T).T


def _point_to_edge_distance(
    x: float, y: float, edge: tuple[tuple[float, float], tuple[float, float]]
) -> float:
    """Perpendicular distance from point to edge segment."""
    (x1, y1), (x2, y2) = edge
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


def _cost_function(
    params: np.ndarray,
    edge_correspondences: list[dict],
    interior_points_arr: np.ndarray,
    negative_points_arr: np.ndarray,
    prepared_polygon,
    w_edge: float,
    w_interior: float,
    w_negative: float,
    w_regularization: float,
) -> float:
    """Cost function for affine correction optimization.

    Minimizes:
    - Edge residuals (corrected edge touch should lie on matched edge)
    - Interior violations (corrected on-mat points should stay inside)
    - Negative violations (corrected off-mat points should stay outside)
    - Regularization (deviation from identity transform — prevents wild
      solutions from noisy 3-point samples)

    Uses vectorized numpy for interior/negative constraints instead of
    per-point Shapely calls for performance.
    """
    affine = _affine_params_to_matrix(params)

    cost = 0.0

    # Edge correspondences: minimize distance to matched edge (primary signal)
    for ec in edge_correspondences:
        cx, cy = _apply_affine_single(*ec["position"], affine)
        d = _point_to_edge_distance(cx, cy, ec["edge_segment"])
        cost += w_edge * d * d

    # Interior constraint: penalize count of on-mat points that end up outside
    # Uses a simple penalty per violation rather than expensive signed_distance
    if len(interior_points_arr) > 0 and w_interior > 0:
        corrected = _apply_affine_batch(interior_points_arr, affine)
        violations = sum(
            1 for cx, cy in corrected
            if not prepared_polygon.contains(Point(cx, cy))
        )
        # Penalty proportional to violation fraction
        cost += w_interior * violations

    # Negative constraint: penalize count of off-mat points that end up inside
    if len(negative_points_arr) > 0 and w_negative > 0:
        corrected = _apply_affine_batch(negative_points_arr, affine)
        violations = sum(
            1 for cx, cy in corrected
            if prepared_polygon.contains(Point(cx, cy))
        )
        cost += w_negative * violations

    # Regularization: penalize deviation from identity
    identity = _identity_params()
    deviation = params - identity
    cost += w_regularization * float(np.sum(deviation**2))

    return cost


def _ransac_affine(
    edge_correspondences: list[dict],
    interior_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
    blueprint: MatBlueprint,
    n_iterations: int = 200,
    inlier_threshold: float = 0.5,
    w_edge: float = 1.0,
    w_interior: float = 0.1,
    w_negative: float = 0.1,
    w_regularization: float = 0.5,
) -> tuple[Optional[np.ndarray], list[int], float]:
    """RANSAC loop: sample 3 correspondences, fit affine, score all.

    Returns (best_affine_2x3, inlier_indices, best_score) or (None, [], inf).
    """
    n = len(edge_correspondences)
    if n < 3:
        return None, [], float("inf")

    # Pre-compute for cost function performance
    prepared = shapely_prep(blueprint.polygon)
    int_arr = np.array(interior_points) if interior_points else np.empty((0, 2))
    neg_arr = np.array(negative_points) if negative_points else np.empty((0, 2))

    best_affine = None
    best_inliers: list[int] = []
    best_score = float("inf")

    for _ in range(n_iterations):
        sample_indices = random.sample(range(n), min(3, n))
        sample = [edge_correspondences[i] for i in sample_indices]

        # Fit affine starting from identity — regularization prevents wild jumps
        result = minimize(
            _cost_function,
            _identity_params(),
            args=(
                sample, int_arr, neg_arr, prepared,
                w_edge, w_interior, w_negative, w_regularization,
            ),
            method="Powell",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success and result.fun > 1e6:
            continue

        affine = _affine_params_to_matrix(result.x)

        # Score: count inliers (edge residual < threshold)
        inliers = []
        for i, ec in enumerate(edge_correspondences):
            cx, cy = _apply_affine_single(*ec["position"], affine)
            d = _point_to_edge_distance(cx, cy, ec["edge_segment"])
            if d < inlier_threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and result.fun < best_score
        ):
            best_affine = affine
            best_inliers = inliers
            best_score = result.fun

    return best_affine, best_inliers, best_score


def _fit_affine_on_inliers(
    inlier_correspondences: list[dict],
    interior_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
    blueprint: MatBlueprint,
    w_edge: float = 1.0,
    w_interior: float = 0.1,
    w_negative: float = 0.1,
    w_regularization: float = 0.5,
) -> np.ndarray:
    """Refit affine on all inlier correspondences for final solution."""
    prepared = shapely_prep(blueprint.polygon)
    int_arr = np.array(interior_points) if interior_points else np.empty((0, 2))
    neg_arr = np.array(negative_points) if negative_points else np.empty((0, 2))

    result = minimize(
        _cost_function,
        _identity_params(),
        args=(
            inlier_correspondences, int_arr, neg_arr, prepared,
            w_edge, w_interior, w_negative, w_regularization,
        ),
        method="Powell",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    return _affine_params_to_matrix(result.x)


def _compute_inside_fraction(
    positions: list[tuple[float, float]], blueprint: MatBlueprint
) -> float:
    """Fraction of positions inside the mat polygon."""
    if not positions:
        return 0.0
    inside = sum(1 for x, y in positions if blueprint.contains_point(x, y))
    return inside / len(positions)


def _mean_edge_residual(
    edge_correspondences: list[dict],
    blueprint: MatBlueprint,
    affine: np.ndarray,
) -> float:
    """Mean distance of corrected edge touch positions to their matched edges."""
    if not edge_correspondences:
        return 0.0
    total = 0.0
    for ec in edge_correspondences:
        cx, cy = _apply_affine_single(*ec["position"], affine)
        d = _point_to_edge_distance(cx, cy, ec["edge_segment"])
        total += d
    return total / len(edge_correspondences)
