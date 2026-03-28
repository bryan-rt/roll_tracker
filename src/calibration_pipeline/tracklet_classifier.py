"""Tracklet classification and feature extraction for CP18 calibration.

Classifies tracklets from D0 bank frames as cleaning-like vs lingering,
extracts edge correspondences and movement quality metrics used by the
single-camera homography refinement (mat_walk.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from calibration_pipeline.blueprint_geometry import MatBlueprint


@dataclass
class TrackletFeatures:
    tracklet_id: str
    camera_id: str
    clip_id: str
    classification: str  # "cleaning" | "lingering"

    # Movement quality metrics
    spatial_extent_m2: float
    total_distance_m: float
    duration_s: float
    avg_speed_mps: float
    stillness_fraction: float

    # Edge correspondence features
    birth_position: tuple[float, float]
    death_position: tuple[float, float]
    birth_edge_distance: float
    death_edge_distance: float
    birth_edge_index: int
    death_edge_index: int
    birth_is_perpendicular: bool
    death_is_perpendicular: bool
    n_edge_touches: int

    # Positions
    positions: list[tuple[float, float]] = field(repr=False)
    on_mat_fraction: float = 0.0


def classify_tracklets(
    tracklet_frames_df: pd.DataFrame,
    blueprint: MatBlueprint,
    fps: float = 30.0,
    extent_threshold: float = 1.0,
    speed_range: tuple[float, float] = (0.3, 3.0),
    stillness_threshold: float = 0.5,
    edge_touch_distance: float = 1.0,
    perpendicular_angle_threshold: float = 45.0,
    approach_window: int = 5,
) -> list[TrackletFeatures]:
    """Classify all tracklets and extract calibration features.

    Parameters
    ----------
    tracklet_frames_df : DataFrame
        Must contain: tracklet_id, frame_index, x_m, y_m (or x_m_repaired, y_m_repaired).
        Also uses: camera_id, clip_id if present.
    blueprint : MatBlueprint
        Parsed mat blueprint for edge distance computation.
    fps : float
        Frames per second for duration/speed calculation.
    extent_threshold : float
        Minimum convex hull area (m²) for cleaning classification.
    speed_range : tuple
        (min, max) average speed (m/s) for cleaning classification.
    stillness_threshold : float
        Maximum fraction of still frames for cleaning classification.
    edge_touch_distance : float
        Maximum distance to a boundary edge for an edge touch.
    perpendicular_angle_threshold : float
        Degrees from edge normal within which a crossing is "perpendicular".
    approach_window : int
        Number of frames to average for approach/departure direction.
    """
    # Prefer repaired coordinates
    x_col = "x_m_repaired" if "x_m_repaired" in tracklet_frames_df.columns else "x_m"
    y_col = "y_m_repaired" if "y_m_repaired" in tracklet_frames_df.columns else "y_m"

    # Fallback: if repaired cols exist but are all NaN, use raw
    if x_col == "x_m_repaired" and tracklet_frames_df[x_col].isna().all():
        x_col, y_col = "x_m", "y_m"

    features = []
    for tid, grp in tracklet_frames_df.groupby("tracklet_id"):
        grp = grp.sort_values("frame_index")
        valid = grp.dropna(subset=[x_col, y_col])
        if len(valid) < 2:
            continue

        xs = valid[x_col].values
        ys = valid[y_col].values
        frames = valid["frame_index"].values

        camera_id = grp["camera_id"].iloc[0] if "camera_id" in grp.columns else ""
        clip_id = grp["clip_id"].iloc[0] if "clip_id" in grp.columns else ""

        # Movement metrics
        dx = np.diff(xs)
        dy = np.diff(ys)
        dists = np.sqrt(dx**2 + dy**2)
        total_dist = float(dists.sum())
        dur_frames = int(frames[-1] - frames[0])
        dur_s = max(dur_frames / fps, 1.0 / fps)
        avg_speed = total_dist / dur_s
        speeds = dists * fps
        stillness = float((speeds < 0.1).sum() / max(1, len(speeds)))

        # Convex hull area
        try:
            pts = np.column_stack([xs, ys])
            unique_pts = np.unique(pts, axis=0)
            if len(unique_pts) >= 3:
                spatial_extent = float(ConvexHull(unique_pts).volume)
            else:
                spatial_extent = 0.0
        except Exception:
            spatial_extent = 0.0

        # Positions list
        positions = list(zip(xs.tolist(), ys.tolist()))

        # On-mat fraction
        inside_count = sum(1 for x, y in positions if blueprint.contains_point(x, y))
        on_mat_fraction = inside_count / len(positions) if positions else 0.0

        # Birth/death positions
        birth_pos = (float(xs[0]), float(ys[0]))
        death_pos = (float(xs[-1]), float(ys[-1]))

        # Edge distances
        birth_dist, birth_edge_idx = blueprint.nearest_edge_distance(*birth_pos)
        death_dist, death_edge_idx = blueprint.nearest_edge_distance(*death_pos)

        # Perpendicular vs parallel detection
        birth_perp = _is_perpendicular_crossing(
            xs, ys, is_birth=True, edge_idx=birth_edge_idx,
            blueprint=blueprint, angle_threshold=perpendicular_angle_threshold,
            window=approach_window,
        )
        death_perp = _is_perpendicular_crossing(
            xs, ys, is_birth=False, edge_idx=death_edge_idx,
            blueprint=blueprint, angle_threshold=perpendicular_angle_threshold,
            window=approach_window,
        )

        # Count valid edge touches (perpendicular + within distance)
        n_edge_touches = 0
        if birth_dist < edge_touch_distance and birth_perp:
            n_edge_touches += 1
        if death_dist < edge_touch_distance and death_perp:
            n_edge_touches += 1

        # Classification
        is_cleaning = (
            spatial_extent > extent_threshold
            and speed_range[0] < avg_speed < speed_range[1]
            and stillness < stillness_threshold
        )
        classification = "cleaning" if is_cleaning else "lingering"

        features.append(TrackletFeatures(
            tracklet_id=str(tid),
            camera_id=camera_id,
            clip_id=clip_id,
            classification=classification,
            spatial_extent_m2=round(spatial_extent, 4),
            total_distance_m=round(total_dist, 4),
            duration_s=round(dur_s, 2),
            avg_speed_mps=round(avg_speed, 4),
            stillness_fraction=round(stillness, 4),
            birth_position=birth_pos,
            death_position=death_pos,
            birth_edge_distance=round(birth_dist, 4),
            death_edge_distance=round(death_dist, 4),
            birth_edge_index=birth_edge_idx,
            death_edge_index=death_edge_idx,
            birth_is_perpendicular=birth_perp,
            death_is_perpendicular=death_perp,
            n_edge_touches=n_edge_touches,
            positions=positions,
            on_mat_fraction=round(on_mat_fraction, 4),
        ))

    return features


def _is_perpendicular_crossing(
    xs: np.ndarray,
    ys: np.ndarray,
    is_birth: bool,
    edge_idx: int,
    blueprint: MatBlueprint,
    angle_threshold: float = 45.0,
    window: int = 5,
) -> bool:
    """Check if a tracklet's approach/departure is perpendicular to the edge.

    For birth: direction from position[0] to position[window].
    For death: direction from position[-window] to position[-1].
    Perpendicular means the angle between the tracklet direction and the
    edge direction is within `angle_threshold` degrees of 90°.
    """
    n = len(xs)
    if n < 2:
        return False

    if is_birth:
        end_idx = min(window, n - 1)
        dir_x = xs[end_idx] - xs[0]
        dir_y = ys[end_idx] - ys[0]
    else:
        start_idx = max(0, n - 1 - window)
        dir_x = xs[-1] - xs[start_idx]
        dir_y = ys[-1] - ys[start_idx]

    tracklet_len = math.sqrt(dir_x**2 + dir_y**2)
    if tracklet_len < 1e-6:
        return False

    edge_dx, edge_dy = blueprint.edge_direction(edge_idx)

    # Angle between tracklet direction and edge direction
    cos_angle = abs(dir_x * edge_dx + dir_y * edge_dy) / tracklet_len
    cos_angle = min(1.0, cos_angle)  # clamp for numerical safety
    angle_from_edge = math.degrees(math.acos(cos_angle))

    # Perpendicular means angle_from_edge is close to 90°
    return abs(angle_from_edge - 90.0) <= angle_threshold
