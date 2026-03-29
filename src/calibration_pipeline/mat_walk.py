"""Layer 1 — Single-camera homography refinement.

Primary signal: mat line correspondences (detected mat edges vs blueprint edges).
Secondary signal: footpath fitting (cleaning positions inside polygon).
Edge touches are computed for diagnostics only — not in the cost function.

CP18 v2: Replaces edge-touch RANSAC (v1) with mat-line + footpath cost function.
Mat lines are dense, geometric, behavior-independent. Footpath fitting uses
thousands of independent "should be inside polygon" constraints.
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
from calibration_pipeline.mat_line_detection import MatLineResult
from calibration_pipeline.tracklet_classifier import TrackletFeatures


@dataclass
class CalibrationResult:
    camera_id: str
    correction_matrix: Optional[np.ndarray]  # 2x3 affine, None if inconclusive
    confidence: str  # "high" | "medium" | "low" | "inconclusive"

    # Quality metrics
    n_cleaning_tracklets: int = 0
    n_edge_touches: int = 0  # diagnostic only (v2)
    n_distinct_edges: int = 0  # diagnostic only (v2)
    coverage_fraction: float = 0.0

    # Mat line metrics (v2)
    n_matched_lines: int = 0
    n_matched_line_edges: int = 0  # distinct blueprint edges matched by lines
    n_cleaning_positions: int = 0
    signal_type: str = ""  # "mat_lines+footpath" | "mat_lines_only" | "footpath_only"

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
    mat_line_result: Optional[MatLineResult] = None,
    min_coverage_fraction: float = 0.4,
    ransac_iterations: int = 200,
    ransac_inlier_threshold: float = 0.5,
    w_mat_lines: float = 1.0,
    w_footpath: float = 0.3,
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
    mat_line_result : Optional[MatLineResult]
        Detected mat lines (primary signal). None = footpath-only fallback.
    min_coverage_fraction : float
        Minimum coverage for footpath-only mode.
    ransac_iterations : int
        RANSAC iterations (used with mat lines).
    ransac_inlier_threshold : float
        Maximum residual for RANSAC inlier (world meters).
    w_mat_lines, w_footpath, w_negative : float
        Cost function weights.
    w_regularization : float
        Near-identity penalty.
    cell_size : float
        Grid cell size (m) for coverage computation.
    """
    result = CalibrationResult(
        camera_id=camera_id, correction_matrix=None, confidence="inconclusive"
    )

    # --- Collect positions and classify ---
    all_positions: list[tuple[float, float]] = []
    cleaning_positions: list[tuple[float, float]] = []
    cleaning = []

    if tracklet_features:
        cleaning = [f for f in tracklet_features if f.classification == "cleaning"]
        result.n_cleaning_tracklets = len(cleaning)

        for f in tracklet_features:
            all_positions.extend(f.positions)
            if f.classification == "cleaning":
                cleaning_positions.extend(f.positions)

    result.n_cleaning_positions = len(cleaning_positions)

    # Diagnostic: edge touch counts (not used in cost function)
    if tracklet_features:
        edge_correspondences = _build_edge_correspondences(tracklet_features, blueprint)
        result.n_edge_touches = len(edge_correspondences)
        touched_edges = set(ec["edge_index"] for ec in edge_correspondences)
        result.n_distinct_edges = len(touched_edges)
    else:
        edge_correspondences = []

    # Coverage
    if all_positions:
        all_cells = set(
            (int(x / cell_size), int(y / cell_size)) for x, y in all_positions
        )
        clean_cells = set(
            (int(x / cell_size), int(y / cell_size)) for x, y in cleaning_positions
        )
        result.coverage_fraction = len(clean_cells & all_cells) / max(1, len(all_cells))

    # Off-mat walkers (negative constraints)
    off_mat_walkers = [
        f for f in (tracklet_features or [])
        if f.classification == "lingering" and f.on_mat_fraction < 0.3
    ]
    result.n_off_mat_walkers = len(off_mat_walkers)

    # --- Determine signal availability ---
    has_mat_lines = (
        mat_line_result is not None
        and mat_line_result.n_lines_matched >= 3
    )
    has_footpath = len(cleaning_positions) >= 100

    if has_mat_lines:
        mat_line_correspondences = _build_mat_line_correspondences(
            mat_line_result, blueprint
        )
        matched_edge_indices = set(
            ml.matched_edge_index for ml in mat_line_result.matched_lines
        )
        result.n_matched_lines = mat_line_result.n_lines_matched
        result.n_matched_line_edges = len(matched_edge_indices)
        # Need at least 2 distinct edges for a good constraint
        if result.n_matched_line_edges < 2:
            has_mat_lines = False
            mat_line_correspondences = []
    else:
        mat_line_correspondences = []

    if has_mat_lines and has_footpath:
        result.signal_type = "mat_lines+footpath"
    elif has_mat_lines:
        result.signal_type = "mat_lines_only"
    elif has_footpath:
        result.signal_type = "footpath_only"
    else:
        result.details = {
            "reason": "insufficient signals",
            "n_matched_lines": mat_line_result.n_lines_matched if mat_line_result else 0,
            "n_cleaning_positions": len(cleaning_positions),
        }
        return result

    # Footpath-only quality gate
    if not has_mat_lines and result.coverage_fraction < min_coverage_fraction:
        result.details = {
            "reason": "footpath-only but insufficient coverage",
            "coverage": result.coverage_fraction,
        }
        return result

    # --- Build constraint sets ---
    # Interior positions: ALL cleaning positions (should be inside after correction).
    # Includes positions currently outside the polygon — those are exactly what the
    # optimizer needs to push inside. Using only already-inside points gives zero
    # gradient at identity.
    interior_points = list(cleaning_positions)

    # Negative constraint positions: ALL off-mat walker positions (should stay outside).
    negative_points = []
    for f in off_mat_walkers:
        negative_points.extend(f.positions)

    # Subsample for performance — continuous signed distance is more expensive
    # per point than discrete counting, so keep samples smaller
    if len(interior_points) > 50:
        interior_points = random.sample(interior_points, 50)
    if len(negative_points) > 30:
        negative_points = random.sample(negative_points, 30)

    # --- Before metrics ---
    if all_positions:
        result.inside_mat_fraction_before = _compute_inside_fraction(
            all_positions, blueprint
        )
    if edge_correspondences:
        result.mean_edge_residual_before = _mean_edge_residual(
            edge_correspondences, blueprint, np.eye(3)[:2]
        )

    # --- Optimization ---
    prepared = shapely_prep(blueprint.polygon)
    int_arr = np.array(interior_points) if interior_points else np.empty((0, 2))
    neg_arr = np.array(negative_points) if negative_points else np.empty((0, 2))

    if has_mat_lines:
        # RANSAC with mat line correspondences
        best_affine, best_inliers, best_score = _ransac_mat_lines(
            mat_line_correspondences=mat_line_correspondences,
            interior_points_arr=int_arr,
            negative_points_arr=neg_arr,
            prepared_polygon=prepared,
            blueprint=blueprint,
            n_iterations=ransac_iterations,
            inlier_threshold=ransac_inlier_threshold,
            w_mat_lines=w_mat_lines,
            w_footpath=w_footpath,
            w_negative=w_negative,
            w_regularization=w_regularization,
        )

        if best_affine is None:
            # Fall through to footpath-only if available
            if has_footpath:
                has_mat_lines = False
                result.signal_type = "footpath_only"
            else:
                result.details = {"reason": "RANSAC failed on mat lines"}
                return result

        if best_affine is not None:
            result.n_ransac_inliers = len(best_inliers)
            result.n_ransac_outliers = len(mat_line_correspondences) - len(best_inliers)
            final_affine = best_affine

    if not has_mat_lines and has_footpath:
        # Direct optimization from identity (no RANSAC needed — smooth surface)
        opt_result = minimize(
            _cost_function,
            _identity_params(),
            args=(
                [], int_arr, neg_arr, prepared, blueprint,
                w_mat_lines, w_footpath, w_negative, w_regularization,
            ),
            method="Powell",
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        final_affine = _affine_params_to_matrix(opt_result.x)

    # --- Validate correction ---
    corrected_positions = _apply_affine(all_positions, final_affine) if all_positions else []
    if corrected_positions:
        result.inside_mat_fraction_after = _compute_inside_fraction(
            corrected_positions, blueprint
        )
    if edge_correspondences:
        result.mean_edge_residual_after = _mean_edge_residual(
            edge_correspondences, blueprint, final_affine
        )

    # Discard if correction made things worse
    if all_positions and result.inside_mat_fraction_after < result.inside_mat_fraction_before:
        result.confidence = "low"
        result.correction_matrix = None
        result.details = {
            "reason": "correction decreased inside-mat fraction",
            "before": result.inside_mat_fraction_before,
            "after": result.inside_mat_fraction_after,
            "signal_type": result.signal_type,
        }
        return result

    result.correction_matrix = final_affine

    # --- Confidence grading ---
    improvement = result.inside_mat_fraction_after - result.inside_mat_fraction_before

    if result.signal_type == "mat_lines+footpath":
        if (
            result.n_matched_line_edges >= 3
            and result.n_cleaning_positions >= 200
            and improvement > 0.20
        ):
            result.confidence = "high"
        elif improvement > 0.0:
            result.confidence = "medium"
        else:
            result.confidence = "low"
    elif result.signal_type == "mat_lines_only":
        if result.n_matched_line_edges >= 3 and improvement > 0.10:
            result.confidence = "high"
        elif result.n_matched_line_edges >= 2 and improvement > 0.0:
            result.confidence = "medium"
        else:
            result.confidence = "low"
    elif result.signal_type == "footpath_only":
        if (
            result.coverage_fraction > 0.6
            and result.n_cleaning_positions >= 500
            and improvement > 0.10
        ):
            result.confidence = "medium"
        elif improvement > 0.0:
            result.confidence = "low"
        else:
            result.confidence = "low"

    result.details = {
        "signal_type": result.signal_type,
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
        "signal_type": result.signal_type,
        "n_matched_lines": result.n_matched_lines,
        "n_matched_line_edges": result.n_matched_line_edges,
        "n_cleaning_positions": result.n_cleaning_positions,
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


def _build_mat_line_correspondences(
    mat_line_result: MatLineResult,
    blueprint: MatBlueprint,
    n_sample_points: int = 10,
) -> list[dict]:
    """Convert matched mat lines into world-space correspondences for the cost function.

    Each correspondence dict has:
    - blueprint_sample_points: list of (x,y) along the blueprint edge
    - detected_world_line: ((x1,y1), (x2,y2)) detected line in world space
    - edge_index: which blueprint edge
    """
    correspondences = []
    for ml in mat_line_result.matched_lines:
        if ml.matched_edge_index < 0:
            continue

        edge = blueprint.boundary_edges[ml.matched_edge_index]
        (ex1, ey1), (ex2, ey2) = edge

        # Sample points along the blueprint edge
        sample_pts = []
        for k in range(n_sample_points):
            t = k / max(1, n_sample_points - 1)
            sample_pts.append((
                ex1 + t * (ex2 - ex1),
                ey1 + t * (ey2 - ey1),
            ))

        correspondences.append({
            "blueprint_sample_points": sample_pts,
            "detected_world_line": (ml.world_start, ml.world_end),
            "edge_index": ml.matched_edge_index,
        })

    return correspondences


def _build_edge_correspondences(
    features: list[TrackletFeatures],
    blueprint: MatBlueprint,
    edge_touch_distance: float = 1.0,
) -> list[dict]:
    """Extract edge correspondences from tracklet birth/death positions.

    Diagnostic only in v2 — used for before/after reporting, not cost function.
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


def _point_to_line_distance_world(
    x: float, y: float,
    line: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Distance from point to a line segment in world space."""
    (lx1, ly1), (lx2, ly2) = line
    return _point_to_edge_distance(x, y, ((lx1, ly1), (lx2, ly2)))


def _cost_function(
    params: np.ndarray,
    mat_line_correspondences: list[dict],
    interior_points_arr: np.ndarray,
    negative_points_arr: np.ndarray,
    prepared_polygon,
    blueprint: MatBlueprint,
    w_mat_lines: float,
    w_footpath: float,
    w_negative: float,
    w_regularization: float,
) -> float:
    """Cost function for affine correction optimization.

    Combines:
    1. Mat line alignment (primary): corrected blueprint edge points should
       lie close to the detected line in world space.
    2. Footpath fitting (secondary): corrected cleaning positions should
       remain inside the polygon.
    3. Negative constraints: off-mat positions should stay outside.
    4. Regularization: near-identity penalty.
    """
    affine = _affine_params_to_matrix(params)
    cost = 0.0

    # 1. Mat line alignment (PRIMARY)
    for corr in mat_line_correspondences:
        detected_line = corr["detected_world_line"]
        for world_pt in corr["blueprint_sample_points"]:
            cx, cy = _apply_affine_single(*world_pt, affine)
            d = _point_to_line_distance_world(cx, cy, detected_line)
            cost += w_mat_lines * d * d

    # 2. Footpath fitting (SECONDARY) — continuous signed distance
    # Uses nearest_edge_distance for positions outside polygon, giving
    # a smooth gradient that Powell can follow (vs discrete violation counts).
    if len(interior_points_arr) > 0 and w_footpath > 0:
        corrected = _apply_affine_batch(interior_points_arr, affine)
        for cx, cy in corrected:
            if not prepared_polygon.contains(Point(cx, cy)):
                d, _ = blueprint.nearest_edge_distance(cx, cy)
                cost += w_footpath * d * d

    # 3. Negative constraints — continuous signed distance
    if len(negative_points_arr) > 0 and w_negative > 0:
        corrected = _apply_affine_batch(negative_points_arr, affine)
        for cx, cy in corrected:
            if prepared_polygon.contains(Point(cx, cy)):
                d, _ = blueprint.nearest_edge_distance(cx, cy)
                cost += w_negative * d * d

    # 4. Regularization toward identity
    identity = _identity_params()
    deviation = params - identity
    cost += w_regularization * float(np.sum(deviation**2))

    return cost


def _ransac_mat_lines(
    mat_line_correspondences: list[dict],
    interior_points_arr: np.ndarray,
    negative_points_arr: np.ndarray,
    prepared_polygon,
    blueprint: MatBlueprint,
    n_iterations: int = 200,
    inlier_threshold: float = 0.5,
    w_mat_lines: float = 1.0,
    w_footpath: float = 0.3,
    w_negative: float = 0.1,
    w_regularization: float = 0.5,
) -> tuple[Optional[np.ndarray], list[int], float]:
    """RANSAC: sample 3 mat line correspondences, fit affine, score all.

    Returns (best_affine_2x3, inlier_indices, best_score) or (None, [], inf).
    """
    n = len(mat_line_correspondences)
    if n < 3:
        # Not enough for RANSAC — try direct optimization with all
        if n > 0:
            result = minimize(
                _cost_function,
                _identity_params(),
                args=(
                    mat_line_correspondences, interior_points_arr,
                    negative_points_arr, prepared_polygon, blueprint,
                    w_mat_lines, w_footpath, w_negative, w_regularization,
                ),
                method="Powell",
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            affine = _affine_params_to_matrix(result.x)
            return affine, list(range(n)), result.fun
        return None, [], float("inf")

    best_affine = None
    best_inliers: list[int] = []
    best_score = float("inf")

    for _ in range(n_iterations):
        sample_indices = random.sample(range(n), min(3, n))
        sample = [mat_line_correspondences[i] for i in sample_indices]

        result = minimize(
            _cost_function,
            _identity_params(),
            args=(
                sample, interior_points_arr, negative_points_arr,
                prepared_polygon, blueprint,
                w_mat_lines, w_footpath, w_negative, w_regularization,
            ),
            method="Powell",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success and result.fun > 1e6:
            continue

        affine = _affine_params_to_matrix(result.x)

        # Score: inlier = all sample points on a correspondence are within threshold
        inliers = []
        for i, corr in enumerate(mat_line_correspondences):
            detected_line = corr["detected_world_line"]
            residuals = []
            for world_pt in corr["blueprint_sample_points"]:
                cx, cy = _apply_affine_single(*world_pt, affine)
                d = _point_to_line_distance_world(cx, cy, detected_line)
                residuals.append(d)
            if residuals and (sum(residuals) / len(residuals)) < inlier_threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and result.fun < best_score
        ):
            best_affine = affine
            best_inliers = inliers
            best_score = result.fun

    # Refit on all inliers
    if best_affine is not None and len(best_inliers) > 0:
        inlier_corrs = [mat_line_correspondences[i] for i in best_inliers]
        refit = minimize(
            _cost_function,
            _identity_params(),
            args=(
                inlier_corrs, interior_points_arr, negative_points_arr,
                prepared_polygon, blueprint,
                w_mat_lines, w_footpath, w_negative, w_regularization,
            ),
            method="Powell",
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        best_affine = _affine_params_to_matrix(refit.x)

    return best_affine, best_inliers, best_score


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
    """Mean distance of corrected edge touch positions to their matched edges.

    Diagnostic metric — not used in cost function.
    """
    if not edge_correspondences:
        return 0.0
    total = 0.0
    for ec in edge_correspondences:
        cx, cy = _apply_affine_single(*ec["position"], affine)
        d = _point_to_edge_distance(cx, cy, ec["edge_segment"])
        total += d
    return total / len(edge_correspondences)
