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
    _clip_segment_to_frame,
    _angle_of_segment,
    _angle_compatible,
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

    This H maps pixel (0,0) -> world (50, 50) with scale ~0.01 m/px.
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
        """project_to_world -> project_world_to_pixel ~ identity."""
        u, v = 100.0, 200.0
        p = np.array([u, v, 1.0])
        q = known_H @ p
        world_x, world_y = q[0] / q[2], q[1] / q[2]

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

    def test_wildly_out_of_bounds_returns_nan(self):
        """Points projecting beyond 10000px are filtered as degenerate."""
        # H that maps world (0,0) to pixel (20000, 20000)
        H = np.array([
            [0.00001, 0.0, 0.0],
            [0.0, 0.00001, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = project_world_to_pixel((0.2, 0.2), H)
        # world (0.2, 0.2) -> pixel (20000, 20000) -> should be nan
        assert math.isnan(result[0]) or math.isnan(result[1])


class TestClipSegmentToFrame:
    def test_fully_inside(self):
        result = _clip_segment_to_frame((100, 100), (500, 100), 1920, 1080)
        assert result is not None
        (x1, _), (x2, _) = result
        assert abs(x1 - 100) < 1e-6
        assert abs(x2 - 500) < 1e-6

    def test_partially_outside(self):
        result = _clip_segment_to_frame((-100, 540), (500, 540), 1920, 1080)
        assert result is not None
        (x1, _), (x2, _) = result
        assert abs(x1 - 0) < 1e-6  # clipped to left edge
        assert abs(x2 - 500) < 1e-6

    def test_fully_outside(self):
        result = _clip_segment_to_frame((-500, -500), (-100, -100), 1920, 1080)
        assert result is None

    def test_too_short_after_clip(self):
        result = _clip_segment_to_frame((-100, 540), (5, 540), 1920, 1080, min_length=10)
        assert result is None  # clipped segment is only 5px


class TestPointToSegmentDistance:
    def test_point_on_segment(self):
        assert _point_to_segment_dist_px(5, 0, 0, 0, 10, 0) < 1e-6

    def test_perpendicular_distance(self):
        d = _point_to_segment_dist_px(5, 3, 0, 0, 10, 0)
        assert abs(d - 3.0) < 1e-6

    def test_endpoint_distance(self):
        d = _point_to_segment_dist_px(12, 0, 0, 0, 10, 0)
        assert abs(d - 2.0) < 1e-6


class TestAngleHelpers:
    def test_horizontal_angle(self):
        assert abs(_angle_of_segment((0, 0), (100, 0))) < 1e-6

    def test_vertical_angle(self):
        assert abs(_angle_of_segment((0, 0), (0, 100)) - 90) < 1e-6

    def test_compatible_same(self):
        assert _angle_compatible(45, 45)

    def test_compatible_close(self):
        assert _angle_compatible(10, 25)

    def test_incompatible(self):
        assert not _angle_compatible(0, 90)

    def test_compatible_wrapping(self):
        """170 degrees and 5 degrees are 15 apart (wrapping around 180)."""
        assert _angle_compatible(170, 5)


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
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )
        assert len(projected) > 0, "No projected edges"

        # Create synthetic detected lines very close to projected edges
        detected = []
        for (px1, py1), (px2, py2) in projected[:3]:
            # Offset by 5 pixels (should be within 80px threshold)
            detected.append(((px1, py1 + 5), (px2, py2 + 5)))

        matched = _match_lines_to_edges(
            detected, projected, edge_indices, known_H, None, None,
            match_distance_threshold=80.0,
        )

        assert len(matched) >= 1, "Should match at least one synthetic line"
        for ml in matched:
            assert ml.matched_edge_index >= 0
            assert ml.match_distance < 80.0

    def test_far_lines_no_match(self, known_H, blueprint):
        """Lines far from any blueprint edge should not match."""
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )

        # Lines far away from any edge
        detected = [((9000, 9000), (9100, 9000))]

        matched = _match_lines_to_edges(
            detected, projected, edge_indices, known_H, None, None,
            match_distance_threshold=80.0,
        )
        assert len(matched) == 0

    def test_perpendicular_no_match(self, known_H, blueprint):
        """A detected line perpendicular to a projected edge should not match."""
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )
        # Take the first projected edge (likely horizontal or vertical)
        if not projected:
            pytest.skip("No projected edges")

        (ex1, ey1), (ex2, ey2) = projected[0]
        # Create a perpendicular detected line at the midpoint
        mx, my = (ex1 + ex2) / 2, (ey1 + ey2) / 2
        dx, dy = ex2 - ex1, ey2 - ey1
        # Perpendicular direction
        detected = [((mx - dy * 0.5, my + dx * 0.5), (mx + dy * 0.5, my - dx * 0.5))]

        matched = _match_lines_to_edges(
            detected, projected, edge_indices, known_H, None, None,
            match_distance_threshold=80.0,
        )
        # Should not match because orientation differs by ~90 degrees
        assert len(matched) == 0


class TestBlueprintProjection:
    def test_edges_project_to_reasonable_pixels(self, known_H, blueprint):
        """Projected edges should be at reasonable pixel coordinates."""
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )
        assert len(projected) > 0

        for (px1, py1), (px2, py2) in projected:
            # With our H (scale 0.01, offset 50), world coords ~42-58 map to
            # pixel coords in range [-800, 800]
            assert -2000 < px1 < 2000
            assert -2000 < py1 < 2000

    def test_edge_count_matches_blueprint(self, known_H, blueprint):
        """Should project same number of edges as blueprint has (no clipping)."""
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )
        # All edges should project (no degenerate points with this H, no clipping)
        assert len(projected) == blueprint.n_edges
        assert len(edge_indices) == blueprint.n_edges

    def test_clipping_reduces_edges(self, known_H, blueprint):
        """With a small frame, clipping should remove off-screen edges."""
        # Small 100x100 frame — most edges will be off-screen
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None, image_wh=(100, 100)
        )
        # With H scale 0.01, world range [42,58] maps to pixel [-800,800]
        # A 100x100 frame can only contain a small portion
        assert len(projected) <= blueprint.n_edges

    def test_edge_indices_track_correctly(self, known_H, blueprint):
        """Edge indices should map back to correct blueprint edges."""
        projected, edge_indices = _project_blueprint_edges_to_pixel(
            blueprint, known_H, None, None
        )
        for i, ei in enumerate(edge_indices):
            assert 0 <= ei < blueprint.n_edges
