"""Tests for mat line detection module (CP18 v2)."""

import math
from pathlib import Path

import numpy as np
import pytest

from calibration_pipeline.mat_line_detection import (
    DetectedMatLine,
    MatLineResult,
    project_world_to_pixel,
    _project_blueprint_edges_to_pixel,
    _match_lines_to_edges,
    _merge_collinear_segments,
    _point_to_segment_dist_px,
)
from calibration_pipeline.blueprint_geometry import MatBlueprint


BLUEPRINT_PATH = Path("configs/mat_blueprint.json")


@pytest.fixture
def blueprint():
    if not BLUEPRINT_PATH.exists():
        pytest.skip("Blueprint not available")
    return MatBlueprint.from_json(BLUEPRINT_PATH)


@pytest.fixture
def known_H():
    """A simple known homography for testing roundtrips.

    This H maps pixel (0,0) → world (50, 50) with scale ~0.01 m/px.
    """
    # Simple translation + scale homography
    # world = H @ pixel: world_x = 0.01*u + 50, world_y = 0.01*v + 50
    return np.array([
        [0.01, 0.0, 50.0],
        [0.0, 0.01, 50.0],
        [0.0, 0.0, 1.0],
    ])


class TestProjectWorldToPixel:
    """Test world-to-pixel inverse projection."""

    def test_roundtrip_no_distortion(self, known_H):
        """project_to_world → project_world_to_pixel ≈ identity."""
        # Forward: pixel (100, 200) → world
        u, v = 100.0, 200.0
        p = np.array([u, v, 1.0])
        q = known_H @ p
        world_x, world_y = q[0] / q[2], q[1] / q[2]

        # Inverse: world → pixel
        u2, v2 = project_world_to_pixel((world_x, world_y), known_H)

        assert abs(u2 - u) < 1e-6
        assert abs(v2 - v) < 1e-6

    def test_multiple_points(self, known_H):
        """Roundtrip works for multiple points."""
        test_pixels = [(0, 0), (640, 480), (320, 240), (1920, 1080)]
        for u, v in test_pixels:
            p = np.array([u, v, 1.0])
            q = known_H @ p
            wx, wy = q[0] / q[2], q[1] / q[2]

            u2, v2 = project_world_to_pixel((wx, wy), known_H)
            assert abs(u2 - u) < 1e-6, f"Failed for pixel ({u}, {v})"
            assert abs(v2 - v) < 1e-6

    def test_degenerate_H(self):
        """Degenerate homography returns nan."""
        H_degen = np.zeros((3, 3))
        H_degen[2, 2] = 1e-15  # near-singular
        result = project_world_to_pixel((50.0, 50.0), H_degen)
        assert math.isnan(result[0]) or math.isnan(result[1])


class TestPointToSegmentDistance:
    def test_point_on_segment(self):
        assert _point_to_segment_dist_px(5, 0, 0, 0, 10, 0) < 1e-6

    def test_perpendicular_distance(self):
        d = _point_to_segment_dist_px(5, 3, 0, 0, 10, 0)
        assert abs(d - 3.0) < 1e-6

    def test_endpoint_distance(self):
        d = _point_to_segment_dist_px(12, 0, 0, 0, 10, 0)
        assert abs(d - 2.0) < 1e-6


class TestMergeCollinearSegments:
    def test_single_segment(self):
        segs = [((0, 0), (100, 0))]
        merged = _merge_collinear_segments(segs)
        assert len(merged) == 1

    def test_collinear_merge(self):
        """Two collinear overlapping segments should merge."""
        segs = [
            ((0, 0), (50, 0)),
            ((30, 0), (100, 0)),
        ]
        merged = _merge_collinear_segments(segs)
        assert len(merged) == 1
        # Merged should span from ~0 to ~100
        start, end = merged[0]
        span = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        assert span > 90  # should be ~100

    def test_perpendicular_no_merge(self):
        """Perpendicular segments should not merge."""
        segs = [
            ((0, 0), (100, 0)),  # horizontal
            ((50, 0), (50, 100)),  # vertical
        ]
        merged = _merge_collinear_segments(segs)
        assert len(merged) == 2


class TestLineMatching:
    def test_synthetic_match(self, known_H, blueprint):
        """Synthetic lines near projected blueprint edges should match."""
        projected = _project_blueprint_edges_to_pixel(blueprint, known_H, None, None)
        assert len(projected) > 0, "No projected edges"

        # Create synthetic detected lines very close to projected edges
        detected = []
        for (px1, py1), (px2, py2) in projected[:3]:
            # Offset by 5 pixels (should be within 30px threshold)
            detected.append(((px1, py1 + 5), (px2, py2 + 5)))

        matched = _match_lines_to_edges(
            detected, projected, known_H, None, None,
            match_distance_threshold=30.0,
        )

        assert len(matched) >= 1, "Should match at least one synthetic line"
        for ml in matched:
            assert ml.matched_edge_index >= 0
            assert ml.match_distance < 30.0

    def test_far_lines_no_match(self, known_H, blueprint):
        """Lines far from any blueprint edge should not match."""
        projected = _project_blueprint_edges_to_pixel(blueprint, known_H, None, None)

        # Lines far away from any edge
        detected = [((10000, 10000), (10100, 10000))]

        matched = _match_lines_to_edges(
            detected, projected, known_H, None, None,
            match_distance_threshold=30.0,
        )
        assert len(matched) == 0


class TestBlueprintProjection:
    def test_edges_project_to_reasonable_pixels(self, known_H, blueprint):
        """Projected edges should be at reasonable pixel coordinates."""
        projected = _project_blueprint_edges_to_pixel(blueprint, known_H, None, None)
        assert len(projected) > 0

        for (px1, py1), (px2, py2) in projected:
            # With our H (scale 0.01, offset 50), world coords ~42-58 map to
            # pixel coords in range [-800, 800]
            assert -2000 < px1 < 2000
            assert -2000 < py1 < 2000

    def test_edge_count_matches_blueprint(self, known_H, blueprint):
        """Should project same number of edges as blueprint has."""
        projected = _project_blueprint_edges_to_pixel(blueprint, known_H, None, None)
        # All edges should project (no degenerate points with this H)
        assert len(projected) == blueprint.n_edges
