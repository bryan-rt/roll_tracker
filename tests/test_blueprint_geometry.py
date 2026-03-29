"""Tests for calibration_pipeline.blueprint_geometry.

Tests the MatBlueprint class against the actual L-shaped mat blueprint.
"""

import math
from pathlib import Path

import pytest

from calibration_pipeline.blueprint_geometry import MatBlueprint, _point_to_segment_dist


BLUEPRINT_PATH = Path("configs/mat_blueprint.json")


@pytest.fixture
def blueprint():
    """Load the actual mat blueprint."""
    return MatBlueprint.from_json(BLUEPRINT_PATH)


class TestMatBlueprintLoading:
    def test_loads_8_panels(self, blueprint):
        assert len(blueprint.panels) == 8

    def test_bounding_box(self, blueprint):
        x_min, y_min, x_max, y_max = blueprint.bounding_box
        assert x_min == 42.0
        assert y_min == 34.0
        assert x_max == 58.0
        assert y_max == 58.0

    def test_area_less_than_bounding_box(self, blueprint):
        # L-shape: union area < bounding box area (16 * 24 = 384)
        # Union of overlapping rectangles should be less than sum of panels (418)
        assert blueprint.area < 418
        # But more than any single panel
        assert blueprint.area > 64

    def test_polygon_is_valid(self, blueprint):
        assert blueprint.polygon.is_valid

    def test_boundary_edges_form_closed_loop(self, blueprint):
        # Edges should form a closed polygon — n_edges > 4 for L-shape
        assert blueprint.n_edges >= 4


class TestContainsPoint:
    def test_center_of_panel_0_is_inside(self, blueprint):
        # Panel 0: x=50, y=50, w=8, h=8 → center (54, 54)
        assert blueprint.contains_point(54, 54)

    def test_center_of_panel_3_is_inside(self, blueprint):
        # Panel 3: x=42, y=34, w=8, h=8 → center (46, 38)
        assert blueprint.contains_point(46, 38)

    def test_far_outside_is_outside(self, blueprint):
        assert not blueprint.contains_point(0, 0)
        assert not blueprint.contains_point(100, 100)

    def test_l_shaped_cutout_is_outside(self, blueprint):
        # The L-shape means the top-left corner area should be outside.
        # Bounding box is [42,58] x [34,58], but panels don't cover
        # the region around (43, 52) IF there's no panel there.
        # Panel 3 covers (42,34)-(50,42), Panel 0 covers (50,50)-(58,58)
        # Panel 7 covers (43,35)-(50,41), Panel 4 covers (51,51)-(57,57)
        # Point (43, 53) should be outside — no panel covers x<50, y>42
        assert not blueprint.contains_point(43, 53)

    def test_overlapping_panel_region_is_inside(self, blueprint):
        # Panels 0 and 4 overlap: panel 0 (50,50)-(58,58), panel 4 (51,51)-(57,57)
        # Point (53, 53) should be inside
        assert blueprint.contains_point(53, 53)


class TestNearestEdgeDistance:
    def test_point_on_edge_has_zero_distance(self, blueprint):
        # Bottom edge of panel 3: y=34, x∈[42,50]
        dist, _ = blueprint.nearest_edge_distance(46, 34)
        assert dist < 0.01

    def test_interior_point_has_positive_distance(self, blueprint):
        dist, _ = blueprint.nearest_edge_distance(54, 54)
        assert dist > 0

    def test_returns_valid_edge_index(self, blueprint):
        _, idx = blueprint.nearest_edge_distance(54, 54)
        assert 0 <= idx < blueprint.n_edges


class TestEdgeDirection:
    def test_unit_vector(self, blueprint):
        for i in range(blueprint.n_edges):
            dx, dy = blueprint.edge_direction(i)
            length = math.sqrt(dx * dx + dy * dy)
            assert abs(length - 1.0) < 1e-6


class TestSignedDistance:
    def test_inside_is_negative(self, blueprint):
        assert blueprint.signed_distance(54, 54) < 0

    def test_outside_is_positive(self, blueprint):
        assert blueprint.signed_distance(0, 0) > 0


class TestPointToSegmentDist:
    def test_point_on_segment(self):
        assert _point_to_segment_dist(5, 0, 0, 0, 10, 0) == 0.0

    def test_perpendicular_distance(self):
        d = _point_to_segment_dist(5, 3, 0, 0, 10, 0)
        assert abs(d - 3.0) < 1e-9

    def test_distance_to_endpoint(self):
        d = _point_to_segment_dist(12, 0, 0, 0, 10, 0)
        assert abs(d - 2.0) < 1e-9

    def test_degenerate_segment(self):
        d = _point_to_segment_dist(3, 4, 0, 0, 0, 0)
        assert abs(d - 5.0) < 1e-9
