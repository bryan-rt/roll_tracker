"""Mat line detection for CP18 calibration.

Detects visible mat panel seams/edges in video frames via Canny + HoughLinesP,
projects blueprint edges to pixel space via dense point sampling, and matches
detected lines to expected positions. Provides the primary signal for Layer 1
homography refinement.

The inverse projection (world->pixel) lives here in the calibration pipeline,
NOT in f0_projection.py (which is the main pipeline's pixel->world path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from calibration_pipeline.blueprint_geometry import MatBlueprint


@dataclass
class DetectedMatLine:
    """A line segment detected in a video frame."""

    pixel_start: tuple[float, float]  # (u, v) in pixel space
    pixel_end: tuple[float, float]
    world_start: tuple[float, float]  # projected to world via current H
    world_end: tuple[float, float]
    matched_edge_index: int  # index into MatLineResult.all_edges (-1 if none)
    match_distance: float  # avg perpendicular distance to matched polyline (px)
    length_px: float  # segment length in pixels


@dataclass
class MatLineResult:
    """Result of mat line detection on a set of frames."""

    n_frames_analyzed: int
    n_lines_detected: int
    n_lines_matched: int
    matched_lines: list[DetectedMatLine]
    # Projected polylines: each is a list of pixel points from dense sampling
    projected_polylines: list[list[tuple[float, float]]]
    # Index into all_edges for each projected polyline
    projected_edge_indices: list[int]
    # All world-space edges used (boundary + panel), indexed by matched_edge_index
    all_edges: list[tuple[tuple[float, float], tuple[float, float]]]
    details: dict = field(default_factory=dict)


def project_world_to_pixel(
    world_xy: tuple[float, float],
    H: np.ndarray,
    camera_matrix: np.ndarray | None = None,
    dist_coefficients: np.ndarray | None = None,
) -> tuple[float, float]:
    """Inverse of project_to_world: world coords -> pixel coords.

    H maps pixel->world, so H_inv maps world->pixel.

    We work in undistorted pixel space (no re-distortion needed) because
    video frames are undistorted before edge detection.

    Returns (nan, nan) for degenerate or wildly out-of-bounds projections.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"))
    p = np.array([world_xy[0], world_xy[1], 1.0], dtype=np.float64)
    q = H_inv @ p
    w = q[2]
    if abs(w) < 1e-6:
        return (float("nan"), float("nan"))
    u, v = q[0] / w, q[1] / w
    if abs(u) > 10000 or abs(v) > 10000:
        return (float("nan"), float("nan"))
    return (u, v)


def detect_mat_lines(
    video_path: Path,
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    blueprint: MatBlueprint,
    tracklet_frames_df: pd.DataFrame | None = None,
    n_frames: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    hough_min_length: int = 50,
    hough_max_gap: int = 10,
    match_distance_threshold: float = 80.0,
) -> MatLineResult:
    """Detect mat edges in video frames and match to blueprint.

    Projects all panel edges (boundary + internal seams) via dense point
    sampling, keeping only in-frame portions as polylines. Matches detected
    Hough lines to these polylines by distance + angle compatibility.
    """
    # Get all panel edges (boundary + internal seams)
    all_edges = _get_all_panel_edges(blueprint)

    # Select frames to analyze
    frame_indices = _select_low_occupancy_frames(
        video_path, tracklet_frames_df, n_frames
    )

    if not frame_indices:
        polylines, poly_indices = _project_edges_dense(
            all_edges, H, camera_matrix, dist_coefficients
        )
        return MatLineResult(
            n_frames_analyzed=0, n_lines_detected=0, n_lines_matched=0,
            matched_lines=[], projected_polylines=polylines,
            projected_edge_indices=poly_indices, all_edges=all_edges,
            details={"reason": "no frames selected"},
        )

    # Detect lines in each frame and get image dimensions
    all_detected: list[tuple[tuple[float, float], tuple[float, float]]] = []
    n_frames_read = 0
    image_wh: tuple[int, int] | None = None

    cap = cv2.VideoCapture(str(video_path))
    try:
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            n_frames_read += 1
            if image_wh is None:
                h, w = frame_bgr.shape[:2]
                image_wh = (w, h)

            lines = _detect_lines_in_frame(
                frame_bgr, camera_matrix, dist_coefficients,
                canny_low, canny_high,
                hough_threshold, hough_min_length, hough_max_gap,
            )
            all_detected.extend(lines)
    finally:
        cap.release()

    # Project edges via dense sampling, clipped to frame
    polylines, poly_indices = _project_edges_dense(
        all_edges, H, camera_matrix, dist_coefficients, image_wh=image_wh,
    )

    if not all_detected:
        return MatLineResult(
            n_frames_analyzed=n_frames_read, n_lines_detected=0, n_lines_matched=0,
            matched_lines=[], projected_polylines=polylines,
            projected_edge_indices=poly_indices, all_edges=all_edges,
            details={"reason": "no lines detected in any frame"},
        )

    # Merge collinear segments across frames
    merged = _merge_collinear_segments(all_detected)

    # Match detected lines to projected polylines
    matched = _match_lines_to_polylines(
        merged, polylines, poly_indices, H,
        camera_matrix, dist_coefficients, match_distance_threshold,
    )

    return MatLineResult(
        n_frames_analyzed=n_frames_read,
        n_lines_detected=len(merged),
        n_lines_matched=len(matched),
        matched_lines=matched,
        projected_polylines=polylines,
        projected_edge_indices=poly_indices,
        all_edges=all_edges,
        details={
            "frame_indices": frame_indices,
            "raw_detections_per_frame": len(all_detected) / max(1, n_frames_read),
            "merged_segments": len(merged),
            "raw_detected_lines": all_detected,
            "n_projected_polylines": len(polylines),
            "n_all_edges": len(all_edges),
        },
    )


# ---------------------------------------------------------------------------
# Panel edge extraction
# ---------------------------------------------------------------------------


def _get_all_panel_edges(
    blueprint: MatBlueprint,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Get all edges of each panel rectangle, including internal seams.

    Deduplicates shared edges between adjacent panels. Returns more edges
    than the outer boundary alone — internal seams are physically visible
    and project well near the anchor rectangle.
    """
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for p in blueprint.panels:
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        corners = [
            (float(x), float(y)),
            (float(x + w), float(y)),
            (float(x + w), float(y + h)),
            (float(x), float(y + h)),
        ]
        for i in range(4):
            c1 = corners[i]
            c2 = corners[(i + 1) % 4]
            # Normalize direction for deduplication
            normalized = (min(c1, c2), max(c1, c2))
            edges.add(normalized)
    return list(edges)


# ---------------------------------------------------------------------------
# Dense projection
# ---------------------------------------------------------------------------


def _project_edges_dense(
    edges: list[tuple[tuple[float, float], tuple[float, float]]],
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    image_wh: tuple[int, int] | None = None,
    sample_spacing: float = 0.25,
    frame_margin: float = 50.0,
) -> tuple[list[list[tuple[float, float]]], list[int]]:
    """Project edges to pixel space via dense point sampling.

    For each edge, samples points every sample_spacing world units, projects
    each independently, keeps only in-frame points, and extracts contiguous
    runs as polylines.

    Returns (polylines, edge_indices) where each polyline is a list of pixel
    points and edge_indices[i] is the index into `edges` it came from.
    """
    polylines: list[list[tuple[float, float]]] = []
    edge_indices: list[int] = []

    for ei, ((wx1, wy1), (wx2, wy2)) in enumerate(edges):
        edge_len = math.sqrt((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2)
        n_samples = max(2, int(math.ceil(edge_len / sample_spacing)))

        # Sample and project
        projected_pts: list[tuple[float, float] | None] = []
        for k in range(n_samples):
            t = k / max(1, n_samples - 1)
            wx = wx1 + t * (wx2 - wx1)
            wy = wy1 + t * (wy2 - wy1)
            px = project_world_to_pixel((wx, wy), H, camera_matrix, dist_coefficients)

            if math.isnan(px[0]):
                projected_pts.append(None)
                continue

            # Check if in frame (with margin)
            if image_wh is not None:
                w, h = image_wh
                if (px[0] < -frame_margin or px[0] > w + frame_margin
                        or px[1] < -frame_margin or px[1] > h + frame_margin):
                    projected_pts.append(None)
                    continue

            projected_pts.append(px)

        # Extract contiguous runs of surviving points
        runs = _extract_contiguous_runs(projected_pts)
        for run in runs:
            if len(run) >= 2:
                polylines.append(run)
                edge_indices.append(ei)

    return polylines, edge_indices


def _extract_contiguous_runs(
    points: list[tuple[float, float] | None],
) -> list[list[tuple[float, float]]]:
    """Extract contiguous sequences of non-None points."""
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for pt in points:
        if pt is not None:
            current.append(pt)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []

    if len(current) >= 2:
        runs.append(current)

    return runs


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def _select_low_occupancy_frames(
    video_path: Path,
    tracklet_frames_df: pd.DataFrame | None,
    n_frames: int,
) -> list[int]:
    """Select frames with fewest people visible.

    Uses tracklet_frames_df frame_index value counts if available.
    Falls back to evenly spaced frames across the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return []

    if tracklet_frames_df is not None and "frame_index" in tracklet_frames_df.columns:
        counts = tracklet_frames_df["frame_index"].value_counts()
        all_frames = pd.Series(0, index=range(total_frames))
        for fi, cnt in counts.items():
            fi_int = int(fi)
            if 0 <= fi_int < total_frames:
                all_frames.iloc[fi_int] = cnt

        segment_size = max(1, total_frames // n_frames)
        selected = []
        for seg_start in range(0, total_frames, segment_size):
            seg_end = min(seg_start + segment_size, total_frames)
            seg = all_frames.iloc[seg_start:seg_end]
            if len(seg) > 0:
                selected.append(int(seg.idxmin()))
            if len(selected) >= n_frames:
                break
        return selected

    if total_frames <= n_frames:
        return list(range(total_frames))
    step = total_frames // n_frames
    return [i * step for i in range(n_frames)]


# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------


def _detect_lines_in_frame(
    frame_bgr: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    canny_low: int,
    canny_high: int,
    hough_threshold: int,
    hough_min_length: int,
    hough_max_gap: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Detect line segments in a single frame via Canny + HoughLinesP."""
    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        frame_bgr = cv2.undistort(frame_bgr, K, D)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    raw_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=hough_min_length, maxLineGap=hough_max_gap,
    )

    if raw_lines is None:
        return []

    segments = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segments


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------


def _merge_collinear_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    angle_tolerance: float = 10.0,
    distance_tolerance: float = 20.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Merge approximately collinear, nearby segments."""
    if not segments:
        return []

    annotated = []
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nx, ny = -dy / length, dx / length
        perp_offset = mx * nx + my * ny
        annotated.append({
            "start": (x1, y1), "end": (x2, y2),
            "angle": angle, "perp_offset": perp_offset, "length": length,
        })

    groups: list[list[dict]] = []
    used = [False] * len(annotated)
    for i, a in enumerate(annotated):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j in range(i + 1, len(annotated)):
            if used[j]:
                continue
            b = annotated[j]
            angle_diff = abs(a["angle"] - b["angle"])
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff < angle_tolerance and abs(a["perp_offset"] - b["perp_offset"]) < distance_tolerance:
                group.append(b)
                used[j] = True
        groups.append(group)

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append((group[0]["start"], group[0]["end"]))
            continue

        longest = max(group, key=lambda g: g["length"])
        dx = longest["end"][0] - longest["start"][0]
        dy = longest["end"][1] - longest["start"][1]
        length = longest["length"]
        ux, uy = dx / length, dy / length

        ref_x, ref_y = longest["start"]
        projections = []
        for g in group:
            for pt in [g["start"], g["end"]]:
                t = (pt[0] - ref_x) * ux + (pt[1] - ref_y) * uy
                projections.append(t)

        t_min = min(projections)
        t_max = max(projections)
        merged.append((
            (ref_x + t_min * ux, ref_y + t_min * uy),
            (ref_x + t_max * ux, ref_y + t_max * uy),
        ))

    return merged


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------


def _angle_of_segment(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Angle in degrees [0, 180) of a line segment."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 180


def _angle_compatible(
    angle1: float, angle2: float, threshold: float = 20.0
) -> bool:
    """Check if two angles (in [0,180)) are within threshold degrees."""
    diff = abs(angle1 - angle2)
    if diff > 90:
        diff = 180 - diff
    return diff < threshold


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _polyline_direction_angle(polyline: list[tuple[float, float]]) -> float:
    """Compute overall direction angle of a polyline from first to last point."""
    if len(polyline) < 2:
        return 0.0
    return _angle_of_segment(polyline[0], polyline[-1])


def _point_to_polyline_dist(
    px: float, py: float,
    polyline: list[tuple[float, float]],
) -> float:
    """Minimum distance from a point to any segment in a polyline."""
    min_d = float("inf")
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        d = _point_to_segment_dist_px(px, py, x1, y1, x2, y2)
        if d < min_d:
            min_d = d
    return min_d


def _match_lines_to_polylines(
    detected_lines: list[tuple[tuple[float, float], tuple[float, float]]],
    polylines: list[list[tuple[float, float]]],
    edge_indices: list[int],
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    match_distance_threshold: float,
) -> list[DetectedMatLine]:
    """Match detected line segments to projected polylines.

    For each detected line, finds the nearest polyline by average distance
    (sampled along the detected segment). Only considers polylines with
    compatible orientation (within 20 degrees).
    """
    if not detected_lines or not polylines:
        return []

    # Pre-compute angles for polylines
    poly_angles = [_polyline_direction_angle(pl) for pl in polylines]

    matched = []
    for (px1, py1), (px2, py2) in detected_lines:
        seg_len = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
        if seg_len < 10:
            continue

        det_angle = _angle_of_segment((px1, py1), (px2, py2))

        best_pi = -1
        best_dist = float("inf")

        for pi, polyline in enumerate(polylines):
            if not _angle_compatible(det_angle, poly_angles[pi]):
                continue

            # Average distance from samples along detected line to polyline
            n_samples = max(3, int(seg_len / 20))
            total_d = 0.0
            for k in range(n_samples):
                t = k / max(1, n_samples - 1)
                sx = px1 + t * (px2 - px1)
                sy = py1 + t * (py2 - py1)
                d = _point_to_polyline_dist(sx, sy, polyline)
                total_d += d
            avg_d = total_d / n_samples

            if avg_d < best_dist:
                best_dist = avg_d
                best_pi = pi

        if best_pi >= 0 and best_dist < match_distance_threshold:
            edge_idx = edge_indices[best_pi]

            w_start = _pixel_to_world(px1, py1, H, camera_matrix, dist_coefficients)
            w_end = _pixel_to_world(px2, py2, H, camera_matrix, dist_coefficients)

            matched.append(DetectedMatLine(
                pixel_start=(px1, py1),
                pixel_end=(px2, py2),
                world_start=w_start,
                world_end=w_end,
                matched_edge_index=edge_idx,
                match_distance=best_dist,
                length_px=seg_len,
            ))

    return matched


# ---------------------------------------------------------------------------
# Pixel → World
# ---------------------------------------------------------------------------


def _pixel_to_world(
    u: float, v: float,
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
) -> tuple[float, float]:
    """Project pixel to world using the forward homography."""
    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, K, D, P=K)
        u = float(undistorted[0, 0, 0])
        v = float(undistorted[0, 0, 1])

    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    w = q[2]
    if abs(w) < 1e-12:
        return (float("nan"), float("nan"))
    return (q[0] / w, q[1] / w)


# ---------------------------------------------------------------------------
# Diagnostic visualization
# ---------------------------------------------------------------------------


def save_diagnostic_image(
    video_path: Path,
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    blueprint: MatBlueprint,
    mat_line_result: MatLineResult,
    output_path: Path,
    frame_index: int = 0,
) -> None:
    """Save annotated diagnostic image.

    Layers (drawn in order):
    - Video frame (undistorted if K available)
    - Detected Hough lines in RED (1px)
    - Projected polylines in GREEN with black outline (3px)
    - Matched lines in YELLOW with distance labels
    - Legend
    """
    if mat_line_result.details.get("frame_indices"):
        frame_index = mat_line_result.details["frame_indices"][0]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return

    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        frame_bgr = cv2.undistort(frame_bgr, K, D)

    canvas = frame_bgr.copy()

    # Layer 1: Detected lines in RED
    raw_lines = mat_line_result.details.get("raw_detected_lines", [])
    for (rx1, ry1), (rx2, ry2) in raw_lines:
        cv2.line(canvas,
                 (int(round(rx1)), int(round(ry1))),
                 (int(round(rx2)), int(round(ry2))),
                 (0, 0, 255), 1)

    # Layer 2: Projected polylines in GREEN with black outline
    polylines = mat_line_result.projected_polylines
    poly_indices = mat_line_result.projected_edge_indices
    for i, polyline in enumerate(polylines):
        pts = [(int(round(x)), int(round(y))) for x, y in polyline]
        for j in range(len(pts) - 1):
            cv2.line(canvas, pts[j], pts[j + 1], (0, 0, 0), 5)
            cv2.line(canvas, pts[j], pts[j + 1], (0, 255, 0), 3)
        # Vertex dots at endpoints
        if pts:
            cv2.circle(canvas, pts[0], 5, (0, 255, 0), -1)
            cv2.circle(canvas, pts[-1], 5, (0, 255, 0), -1)
        # Edge index label at midpoint
        if len(pts) >= 2:
            mid_idx = len(pts) // 2
            mx, my = pts[mid_idx]
            edge_label = str(poly_indices[i]) if i < len(poly_indices) else "?"
            cv2.putText(canvas, f"e{edge_label}", (mx + 5, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(canvas, f"e{edge_label}", (mx + 5, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Layer 3: Matched lines in YELLOW
    for ml in mat_line_result.matched_lines:
        p1 = (int(round(ml.pixel_start[0])), int(round(ml.pixel_start[1])))
        p2 = (int(round(ml.pixel_end[0])), int(round(ml.pixel_end[1])))
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.putText(canvas, f"{ml.match_distance:.0f}px e{ml.matched_edge_index}",
                    (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Legend
    y0 = 30
    cv2.putText(canvas, f"Frame {frame_index}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    n_edges = mat_line_result.details.get("n_all_edges", len(mat_line_result.all_edges))
    cv2.putText(canvas, f"GREEN: {len(polylines)} polylines from {n_edges} edges",
                (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f"RED: {len(raw_lines)} detected lines",
                (10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, f"YELLOW: {mat_line_result.n_lines_matched} matched",
                (10, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _point_to_segment_dist_px(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Point-to-segment distance in pixel space."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
