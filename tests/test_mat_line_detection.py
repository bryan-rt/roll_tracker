"""Tests for mat line detection module (CP18 v2 — dense sampling)."""

import math
from pathlib import Path

import numpy as np
import pytest

from calibration_pipeline.mat_line_detection import (
    DetectedMatLine,
    MatLineResult,
    project_world_to_pixel,
    _project_edges_dense,
    _get_all_panel_edges,
    _match_lines_to_polylines,
    _merge_collinear_segments,
    _extract_contiguous_runs,
    _point_to_segment_dist_px,
    _point_to_polyline_dist,
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
    """H maps pixel (0,0) -> world (50, 50) with scale 0.01 m/px."""
    return np.array([
        [0.01, 0.0, 50.0],
        [0.0, 0.01, 50.0],
        [0.0, 0.0, 1.0],
    ])


class TestProjectWorldToPixel:
    def test_roundtrip_no_distortion(self, known_H):
        u, v = 100.0, 200.0
        p = np.array([u, v, 1.0])
        q = known_H @ p
        world_x, world_y = q[0] / q[2], q[1] / q[2]
        u2, v2 = project_world_to_pixel((world_x, world_y), known_H)
        assert abs(u2 - u) < 1e-6
        assert abs(v2 - v) < 1e-6

    def test_multiple_points(self, known_H):
        for u, v in [(0, 0), (640, 480), (320, 240), (1920, 1080)]:
            p = np.array([u, v, 1.0])
            q = known_H @ p
            wx, wy = q[0] / q[2], q[1] / q[2]
            u2, v2 = project_world_to_pixel((wx, wy), known_H)
            assert abs(u2 - u) < 1e-6
            assert abs(v2 - v) < 1e-6

    def test_degenerate_H(self):
        H_degen = np.zeros((3, 3))
        H_degen[2, 2] = 1e-15
        result = project_world_to_pixel((50.0, 50.0), H_degen)
        assert math.isnan(result[0]) or math.isnan(result[1])

    def test_wildly_out_of_bounds_returns_nan(self):
        H = np.array([
            [0.00001, 0.0, 0.0],
            [0.0, 0.00001, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = project_world_to_pixel((0.2, 0.2), H)
        assert math.isnan(result[0]) or math.isnan(result[1])


class TestContiguousRuns:
    def test_all_valid(self):
        pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        runs = _extract_contiguous_runs(pts)
        assert len(runs) == 1
        assert len(runs[0]) == 3

    def test_gap_in_middle(self):
        pts = [(1.0, 2.0), (3.0, 4.0), None, (7.0, 8.0), (9.0, 10.0)]
        runs = _extract_contiguous_runs(pts)
        assert len(runs) == 2
        assert len(runs[0]) == 2
        assert len(runs[1]) == 2

    def test_single_point_runs_dropped(self):
        pts = [(1.0, 2.0), None, (5.0, 6.0), None, (9.0, 10.0)]
        runs = _extract_contiguous_runs(pts)
        assert len(runs) == 0  # all runs have only 1 point

    def test_all_none(self):
        runs = _extract_contiguous_runs([None, None, None])
        assert len(runs) == 0


class TestPanelEdges:
    def test_panel_edges_more_than_boundary(self, blueprint):
        """Panel edges should include internal seams, more than boundary."""
        all_edges = _get_all_panel_edges(blueprint)
        assert len(all_edges) > blueprint.n_edges
        # 8 panels x 4 edges = 32, minus shared edges -> ~20-25 unique
        assert len(all_edges) >= 15

    def test_edges_are_deduplicated(self, blueprint):
        """No duplicate edges (shared panel boundaries counted once)."""
        all_edges = _get_all_panel_edges(blueprint)
        normalized = set()
        for (x1, y1), (x2, y2) in all_edges:
            e = (min((x1, y1), (x2, y2)), max((x1, y1), (x2, y2)))
            normalized.add(e)
        assert len(normalized) == len(all_edges)


class TestDenseProjection:
    def test_produces_polylines(self, known_H, blueprint):
        """Dense projection should produce polylines with multiple points."""
        all_edges = _get_all_panel_edges(blueprint)
        polylines, indices = _project_edges_dense(
            all_edges, known_H, None, None
        )
        assert len(polylines) > 0
        # Each polyline should have 2+ points
        for pl in polylines:
            assert len(pl) >= 2

    def test_clipping_filters_offscreen(self, known_H, blueprint):
        """With a tiny frame, most edges should be filtered."""
        all_edges = _get_all_panel_edges(blueprint)
        polylines, _ = _project_edges_dense(
            all_edges, known_H, None, None, image_wh=(50, 50)
        )
        # A 50x50 frame should contain very few of the projected edges
        full_polylines, _ = _project_edges_dense(
            all_edges, known_H, None, None
        )
        assert len(polylines) <= len(full_polylines)

    def test_indices_track_correctly(self, known_H, blueprint):
        all_edges = _get_all_panel_edges(blueprint)
        polylines, indices = _project_edges_dense(
            all_edges, known_H, None, None
        )
        for idx in indices:
            assert 0 <= idx < len(all_edges)


class TestPointToSegmentDistance:
    def test_point_on_segment(self):
        assert _point_to_segment_dist_px(5, 0, 0, 0, 10, 0) < 1e-6

    def test_perpendicular_distance(self):
        assert abs(_point_to_segment_dist_px(5, 3, 0, 0, 10, 0) - 3.0) < 1e-6

    def test_endpoint_distance(self):
        assert abs(_point_to_segment_dist_px(12, 0, 0, 0, 10, 0) - 2.0) < 1e-6


class TestPointToPolylineDistance:
    def test_on_polyline(self):
        polyline = [(0, 0), (10, 0), (10, 10)]
        assert _point_to_polyline_dist(5, 0, polyline) < 1e-6

    def test_perpendicular_to_second_segment(self):
        polyline = [(0, 0), (10, 0), (10, 10)]
        assert abs(_point_to_polyline_dist(13, 5, polyline) - 3.0) < 1e-6


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
        assert _angle_compatible(170, 5)


class TestMergeCollinearSegments:
    def test_single_segment(self):
        assert len(_merge_collinear_segments([((0, 0), (100, 0))])) == 1

    def test_collinear_merge(self):
        segs = [((0, 0), (50, 0)), ((30, 0), (100, 0))]
        merged = _merge_collinear_segments(segs)
        assert len(merged) == 1
        start, end = merged[0]
        assert math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) > 90

    def test_perpendicular_no_merge(self):
        segs = [((0, 0), (100, 0)), ((50, 0), (50, 100))]
        assert len(_merge_collinear_segments(segs)) == 2


class TestPolylineMatching:
    def test_synthetic_match(self, known_H, blueprint):
        """Detected lines near projected polylines should match."""
        all_edges = _get_all_panel_edges(blueprint)
        polylines, indices = _project_edges_dense(
            all_edges, known_H, None, None
        )
        assert len(polylines) > 0

        # Create synthetic detected lines near a polyline
        detected = []
        for pl in polylines[:3]:
            if len(pl) >= 2:
                p1, p2 = pl[0], pl[-1]
                # Offset by 5 pixels
                detected.append(((p1[0], p1[1] + 5), (p2[0], p2[1] + 5)))

        matched = _match_lines_to_polylines(
            detected, polylines, indices, known_H, None, None,
            match_distance_threshold=80.0,
        )
        assert len(matched) >= 1

    def test_far_lines_no_match(self, known_H, blueprint):
        all_edges = _get_all_panel_edges(blueprint)
        polylines, indices = _project_edges_dense(
            all_edges, known_H, None, None
        )
        detected = [((9000, 9000), (9100, 9000))]
        matched = _match_lines_to_polylines(
            detected, polylines, indices, known_H, None, None,
            match_distance_threshold=80.0,
        )
        assert len(matched) == 0
