"""Tests for calibration_pipeline.tracklet_classifier.

Tests classification logic and perpendicular/parallel detection.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import (
    TrackletFeatures,
    classify_tracklets,
    _is_perpendicular_crossing,
)


BLUEPRINT_PATH = Path("configs/mat_blueprint.json")


@pytest.fixture
def blueprint():
    return MatBlueprint.from_json(BLUEPRINT_PATH)


def _make_tracklet_df(
    tracklet_id: str,
    positions: list[tuple[float, float]],
    camera_id: str = "test_cam",
    clip_id: str = "test_clip",
) -> pd.DataFrame:
    """Create a minimal tracklet DataFrame."""
    return pd.DataFrame({
        "tracklet_id": [tracklet_id] * len(positions),
        "frame_index": list(range(len(positions))),
        "x_m": [p[0] for p in positions],
        "y_m": [p[1] for p in positions],
        "camera_id": [camera_id] * len(positions),
        "clip_id": [clip_id] * len(positions),
    })


class TestClassification:
    def test_cleaning_tracklet(self, blueprint):
        """A tracklet sweeping across the mat with good extent and speed."""
        # Zigzag: 300 frames @ 30fps = 10s. Keep speed in 0.3-3.0 m/s.
        import math
        n = 300
        positions = [
            (50 + 2 * i / n, 45 + 1.5 * math.sin(i * 0.1) + 1.5 * i / n)
            for i in range(n)
        ]
        df = _make_tracklet_df("t1", positions)
        features = classify_tracklets(df, blueprint, fps=30.0)
        assert len(features) == 1
        assert features[0].classification == "cleaning"

    def test_lingering_tracklet(self, blueprint):
        """A stationary tracklet (small extent, low speed)."""
        n = 100
        positions = [(52, 46)] * n  # same position
        df = _make_tracklet_df("t1", positions)
        features = classify_tracklets(df, blueprint, fps=30.0)
        assert len(features) == 1
        assert features[0].classification == "lingering"

    def test_too_fast_is_lingering(self, blueprint):
        """A tracklet moving too fast (> 3.0 m/s) is classified as lingering."""
        n = 30
        # 30 frames = 1 second, distance = 10m → speed = 10 m/s
        positions = [(50 + 10 * i / n, 45) for i in range(n)]
        df = _make_tracklet_df("t1", positions)
        features = classify_tracklets(df, blueprint, fps=30.0)
        assert features[0].classification == "lingering"

    def test_empty_df(self, blueprint):
        """Empty DataFrame produces no features."""
        df = pd.DataFrame(columns=["tracklet_id", "frame_index", "x_m", "y_m",
                                   "camera_id", "clip_id"])
        features = classify_tracklets(df, blueprint)
        assert len(features) == 0

    def test_single_frame_tracklet_skipped(self, blueprint):
        """Tracklet with only one frame is skipped."""
        df = _make_tracklet_df("t1", [(52, 46)])
        features = classify_tracklets(df, blueprint)
        assert len(features) == 0

    def test_on_mat_fraction(self, blueprint):
        """On-mat fraction computed correctly."""
        # All positions inside mat
        positions = [(53, 53), (54, 54), (55, 55)] * 10
        df = _make_tracklet_df("t1", positions)
        features = classify_tracklets(df, blueprint)
        assert features[0].on_mat_fraction > 0.5

    def test_multiple_tracklets(self, blueprint):
        """Multiple tracklets classified independently."""
        import math
        df1 = _make_tracklet_df("t1", [(50 + 2*i/300, 45 + 1.5*math.sin(i*0.1) + 1.5*i/300) for i in range(300)])
        df2 = _make_tracklet_df("t2", [(52, 46)] * 100)
        df = pd.concat([df1, df2], ignore_index=True)
        features = classify_tracklets(df, blueprint)
        assert len(features) == 2
        classifications = {f.tracklet_id: f.classification for f in features}
        assert classifications["t1"] == "cleaning"
        assert classifications["t2"] == "lingering"


class TestPerpendicularDetection:
    def test_perpendicular_approach_to_horizontal_edge(self, blueprint):
        """Tracklet approaching a horizontal edge from above (perpendicular)."""
        # Bottom edge of blueprint is at y=34 (horizontal)
        # Approaching from y=36 → y=34.5 (moving down = perpendicular to horizontal)
        n = 20
        xs = np.full(n, 46.0)
        ys = np.linspace(36, 34.5, n)
        # Find the edge at y=34
        _, edge_idx = blueprint.nearest_edge_distance(46, 34.5)
        result = _is_perpendicular_crossing(xs, ys, is_birth=True,
                                            edge_idx=edge_idx, blueprint=blueprint)
        assert result is True

    def test_parallel_walk_along_edge(self, blueprint):
        """Tracklet walking parallel to a horizontal edge."""
        n = 20
        xs = np.linspace(44, 48, n)
        ys = np.full(n, 34.5)  # walking along bottom edge
        _, edge_idx = blueprint.nearest_edge_distance(46, 34.5)
        result = _is_perpendicular_crossing(xs, ys, is_birth=True,
                                            edge_idx=edge_idx, blueprint=blueprint)
        assert result is False

    def test_short_tracklet_returns_false(self, blueprint):
        """Single-point tracklet can't determine direction."""
        xs = np.array([46.0])
        ys = np.array([34.5])
        result = _is_perpendicular_crossing(xs, ys, is_birth=True,
                                            edge_idx=0, blueprint=blueprint)
        assert result is False
