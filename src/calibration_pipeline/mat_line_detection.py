"""Mat line detection for CP18 calibration.

Detects visible mat panel seams/edges in video frames via Canny + HoughLinesP,
projects blueprint edges to pixel space, and matches detected lines to expected
positions. Provides the primary signal for Layer 1 homography refinement.

The inverse projection (world→pixel) lives here in the calibration pipeline,
NOT in f0_projection.py (which is the main pipeline's pixel→world path).
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
    matched_edge_index: int  # which blueprint edge it matches (-1 if none)
    match_distance: float  # avg perpendicular distance to matched edge (px)
    length_px: float  # segment length in pixels


@dataclass
class MatLineResult:
    """Result of mat line detection on a set of frames."""

    n_frames_analyzed: int
    n_lines_detected: int
    n_lines_matched: int
    matched_lines: list[DetectedMatLine]
    projected_blueprint_edges_px: list[
        tuple[tuple[float, float], tuple[float, float]]
    ]
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
    video frames are undistorted before edge detection. This keeps the
    pipeline simple: undistort frame, then all pixel-space comparisons
    are in the same undistorted coordinate system.

    Parameters
    ----------
    world_xy : (x, y) in world/mat coordinates.
    H : 3x3 homography (pixel -> world).
    camera_matrix, dist_coefficients : unused for inverse direction
        (we skip re-distortion by design — see docstring).

    Returns
    -------
    (u, v) in undistorted pixel coordinates, or (nan, nan) if degenerate.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"))
    p = np.array([world_xy[0], world_xy[1], 1.0], dtype=np.float64)
    q = H_inv @ p
    w = q[2]
    if abs(w) < 1e-12:
        return (float("nan"), float("nan"))
    return (q[0] / w, q[1] / w)


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
    match_distance_threshold: float = 30.0,
) -> MatLineResult:
    """Detect mat edges in video frames and match to blueprint.

    Parameters
    ----------
    video_path : Path to video file (mp4).
    H : 3x3 homography (pixel -> world).
    camera_matrix, dist_coefficients : lens calibration (optional).
    blueprint : Parsed mat blueprint.
    tracklet_frames_df : If provided, used to select low-occupancy frames.
    n_frames : Number of frames to analyze.
    canny_low, canny_high : Canny edge detector thresholds.
    hough_threshold, hough_min_length, hough_max_gap : HoughLinesP params.
    match_distance_threshold : Max avg pixel distance to count as a match.

    Returns
    -------
    MatLineResult with matched lines and diagnostics.
    """
    # Project blueprint edges to pixel space
    projected_edges = _project_blueprint_edges_to_pixel(
        blueprint, H, camera_matrix, dist_coefficients
    )

    # Select frames to analyze
    frame_indices = _select_low_occupancy_frames(
        video_path, tracklet_frames_df, n_frames
    )

    if not frame_indices:
        return MatLineResult(
            n_frames_analyzed=0,
            n_lines_detected=0,
            n_lines_matched=0,
            matched_lines=[],
            projected_blueprint_edges_px=projected_edges,
            details={"reason": "no frames selected"},
        )

    # Detect lines in each frame and accumulate matches
    all_detected: list[tuple[tuple[float, float], tuple[float, float]]] = []
    n_frames_read = 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            n_frames_read += 1

            lines = _detect_lines_in_frame(
                frame_bgr, camera_matrix, dist_coefficients,
                canny_low, canny_high,
                hough_threshold, hough_min_length, hough_max_gap,
            )
            all_detected.extend(lines)
    finally:
        cap.release()

    if not all_detected:
        return MatLineResult(
            n_frames_analyzed=n_frames_read,
            n_lines_detected=0,
            n_lines_matched=0,
            matched_lines=[],
            projected_blueprint_edges_px=projected_edges,
            details={"reason": "no lines detected in any frame"},
        )

    # Merge collinear segments across frames
    merged = _merge_collinear_segments(all_detected)

    # Match detected lines to projected blueprint edges
    matched = _match_lines_to_edges(
        merged, projected_edges, H, camera_matrix, dist_coefficients,
        match_distance_threshold,
    )

    return MatLineResult(
        n_frames_analyzed=n_frames_read,
        n_lines_detected=len(merged),
        n_lines_matched=len(matched),
        matched_lines=matched,
        projected_blueprint_edges_px=projected_edges,
        details={
            "frame_indices": frame_indices,
            "raw_detections_per_frame": len(all_detected) / max(1, n_frames_read),
            "merged_segments": len(merged),
            "raw_detected_lines": all_detected,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
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
        # Count detections per frame
        counts = tracklet_frames_df["frame_index"].value_counts()

        # Consider all frames, defaulting unobserved frames to 0
        all_frames = pd.Series(0, index=range(total_frames))
        for fi, cnt in counts.items():
            fi_int = int(fi)
            if 0 <= fi_int < total_frames:
                all_frames.iloc[fi_int] = cnt

        # Pick frames with fewest detections, spread across the video
        # Divide video into n_frames segments, pick lowest-count from each
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

    # Fallback: evenly spaced frames
    if total_frames <= n_frames:
        return list(range(total_frames))
    step = total_frames // n_frames
    return [i * step for i in range(n_frames)]


def _project_blueprint_edges_to_pixel(
    blueprint: MatBlueprint,
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Project all blueprint boundary edges to pixel space."""
    projected = []
    for (wx1, wy1), (wx2, wy2) in blueprint.boundary_edges:
        px1 = project_world_to_pixel((wx1, wy1), H, camera_matrix, dist_coefficients)
        px2 = project_world_to_pixel((wx2, wy2), H, camera_matrix, dist_coefficients)
        if not (math.isnan(px1[0]) or math.isnan(px2[0])):
            projected.append((px1, px2))
    return projected


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
    """Detect line segments in a single frame.

    Undistorts if K+dist available, then Canny + HoughLinesP.
    """
    # Optional lens undistortion
    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        frame_bgr = cv2.undistort(frame_bgr, K, D)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap,
    )

    if raw_lines is None:
        return []

    segments = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segments


def _merge_collinear_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    angle_tolerance: float = 10.0,
    distance_tolerance: float = 20.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Merge approximately collinear, nearby segments.

    Groups by similar angle and perpendicular offset, then merges
    overlapping segments within each group into their bounding segment.
    """
    if not segments:
        return []

    # Compute angle and midpoint-perpendicular-distance for each segment
    annotated = []
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180  # normalize to [0, 180)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # Perpendicular offset from origin along the line's normal
        nx, ny = -dy / length, dx / length
        perp_offset = mx * nx + my * ny
        annotated.append({
            "start": (x1, y1), "end": (x2, y2),
            "angle": angle, "perp_offset": perp_offset, "length": length,
        })

    # Group by similar angle + perpendicular offset
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

    # Merge each group: project all endpoints onto the group's primary axis,
    # take the min/max extent
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append((group[0]["start"], group[0]["end"]))
            continue

        # Use the longest segment's direction as the axis
        longest = max(group, key=lambda g: g["length"])
        dx = longest["end"][0] - longest["start"][0]
        dy = longest["end"][1] - longest["start"][1]
        length = longest["length"]
        ux, uy = dx / length, dy / length

        # Project all endpoints onto this axis
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


def _match_lines_to_edges(
    detected_lines: list[tuple[tuple[float, float], tuple[float, float]]],
    projected_edges: list[tuple[tuple[float, float], tuple[float, float]]],
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
    match_distance_threshold: float,
) -> list[DetectedMatLine]:
    """Match detected line segments to projected blueprint edges.

    For each detected line, find the nearest projected edge by average
    perpendicular distance (sampled along the detected segment).
    """
    if not detected_lines or not projected_edges:
        return []

    matched = []
    for (px1, py1), (px2, py2) in detected_lines:
        seg_len = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
        if seg_len < 10:  # skip very short segments
            continue

        # Find nearest projected edge
        best_idx = -1
        best_dist = float("inf")

        for ei, ((ex1, ey1), (ex2, ey2)) in enumerate(projected_edges):
            # Average perpendicular distance: sample points along detected segment
            n_samples = max(3, int(seg_len / 20))
            total_d = 0.0
            for k in range(n_samples):
                t = k / max(1, n_samples - 1)
                sx = px1 + t * (px2 - px1)
                sy = py1 + t * (py2 - py1)
                d = _point_to_segment_dist_px(sx, sy, ex1, ey1, ex2, ey2)
                total_d += d
            avg_d = total_d / n_samples

            if avg_d < best_dist:
                best_dist = avg_d
                best_idx = ei

        if best_idx >= 0 and best_dist < match_distance_threshold:
            # Project detected line endpoints to world space
            w_start = _pixel_to_world(px1, py1, H, camera_matrix, dist_coefficients)
            w_end = _pixel_to_world(px2, py2, H, camera_matrix, dist_coefficients)

            matched.append(DetectedMatLine(
                pixel_start=(px1, py1),
                pixel_end=(px2, py2),
                world_start=w_start,
                world_end=w_end,
                matched_edge_index=best_idx,
                match_distance=best_dist,
                length_px=seg_len,
            ))

    return matched


def _pixel_to_world(
    u: float, v: float,
    H: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coefficients: np.ndarray | None,
) -> tuple[float, float]:
    """Project pixel to world using the forward homography.

    If K+dist available, undistort the point first (same as f0_projection).
    """
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
    """Save annotated diagnostic image showing frame + projected edges + detected lines.

    Layers:
    - Video frame (undistorted if K available) as background
    - Projected blueprint boundary edges in GREEN (2px solid)
    - All detected Hough lines in RED (1px)
    - Matched lines (if any) in YELLOW with distance labels
    - Legend in top-left corner
    """
    # Use frame_index from detection if available
    if mat_line_result.details.get("frame_indices"):
        frame_index = mat_line_result.details["frame_indices"][0]

    # Read frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return

    # Undistort if K available
    if camera_matrix is not None and dist_coefficients is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
        D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
        frame_bgr = cv2.undistort(frame_bgr, K, D)

    canvas = frame_bgr.copy()

    # Draw projected blueprint edges in GREEN
    projected = _project_blueprint_edges_to_pixel(
        blueprint, H, camera_matrix, dist_coefficients
    )
    for i, ((px1, py1), (px2, py2)) in enumerate(projected):
        p1 = (int(round(px1)), int(round(py1)))
        p2 = (int(round(px2)), int(round(py2)))
        cv2.line(canvas, p1, p2, (0, 255, 0), 2)
        cv2.circle(canvas, p1, 4, (0, 255, 0), -1)
        cv2.circle(canvas, p2, 4, (0, 255, 0), -1)
        # Edge index label at midpoint
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(canvas, str(i), (mx, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw all detected lines (raw, pre-merge) in RED
    raw_lines = mat_line_result.details.get("raw_detected_lines", [])
    for (rx1, ry1), (rx2, ry2) in raw_lines:
        cv2.line(canvas,
                 (int(round(rx1)), int(round(ry1))),
                 (int(round(rx2)), int(round(ry2))),
                 (0, 0, 255), 1)

    # Draw matched lines in YELLOW with distance labels
    for ml in mat_line_result.matched_lines:
        p1 = (int(round(ml.pixel_start[0])), int(round(ml.pixel_start[1])))
        p2 = (int(round(ml.pixel_end[0])), int(round(ml.pixel_end[1])))
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.putText(canvas, f"{ml.match_distance:.0f}px e{ml.matched_edge_index}",
                    (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Legend
    h = canvas.shape[0]
    y0 = 30
    cv2.putText(canvas, f"Frame {frame_index}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"GREEN: projected blueprint ({len(projected)} edges)",
                (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f"RED: detected lines ({len(raw_lines)} raw)",
                (10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, f"YELLOW: matched ({mat_line_result.n_lines_matched})",
                (10, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


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
