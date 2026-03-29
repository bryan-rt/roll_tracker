"""Mat blueprint geometry utilities.

Parses the mat blueprint JSON (array of {x, y, width, height} panels) into
a Shapely polygon union and provides geometric queries used by CP18 calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


class MatBlueprint:
    """Parsed mat blueprint with geometric query methods.

    The blueprint is stored as a Shapely polygon (union of all panel rectangles).
    Boundary edges are extracted from the polygon exterior as line segments.
    """

    def __init__(
        self,
        panels: list[dict],
        polygon: Polygon,
        boundary_edges: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> None:
        self._panels = panels
        self._polygon = polygon
        self._boundary_edges = boundary_edges

    @classmethod
    def from_json(cls, path: Path) -> MatBlueprint:
        """Load blueprint JSON (array of {x, y, width, height} panels)."""
        with open(path) as f:
            panels = json.load(f)

        # Build Shapely boxes for each panel and union them
        boxes = []
        for p in panels:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            boxes.append(box(x, y, x + w, y + h))

        polygon = unary_union(boxes)

        # Extract boundary edges from exterior ring
        if polygon.geom_type == "MultiPolygon":
            # Shouldn't happen for a connected mat, but handle gracefully
            coords_list = []
            for geom in polygon.geoms:
                coords_list.extend(list(geom.exterior.coords))
            # Fall back to convex hull boundary
            polygon = polygon.convex_hull

        exterior_coords = list(polygon.exterior.coords)
        boundary_edges = []
        for i in range(len(exterior_coords) - 1):
            p1 = exterior_coords[i]
            p2 = exterior_coords[i + 1]
            # Skip degenerate zero-length edges
            if abs(p1[0] - p2[0]) > 1e-9 or abs(p1[1] - p2[1]) > 1e-9:
                boundary_edges.append((p1, p2))

        return cls(panels=panels, polygon=polygon, boundary_edges=boundary_edges)

    def contains_point(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the mat polygon (union of all panels)."""
        return self._polygon.contains(Point(x, y))

    def nearest_edge_distance(self, x: float, y: float) -> tuple[float, int]:
        """Return (perpendicular distance, edge_index) to nearest boundary edge."""
        min_dist = float("inf")
        min_idx = -1
        for i, (p1, p2) in enumerate(self._boundary_edges):
            d = _point_to_segment_dist(x, y, p1[0], p1[1], p2[0], p2[1])
            if d < min_dist:
                min_dist = d
                min_idx = i
        return (min_dist, min_idx)

    def nearest_edge_segment(
        self, x: float, y: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the (start, end) of the nearest boundary edge segment."""
        _, idx = self.nearest_edge_distance(x, y)
        return self._boundary_edges[idx]

    def edge_direction(self, edge_index: int) -> tuple[float, float]:
        """Return unit vector along edge."""
        p1, p2 = self._boundary_edges[edge_index]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def signed_distance(self, x: float, y: float) -> float:
        """Signed distance: negative inside polygon, positive outside."""
        d = self._polygon.exterior.distance(Point(x, y))
        if self._polygon.contains(Point(x, y)):
            return -d
        return d

    @property
    def panels(self) -> list[dict]:
        """Raw panel definitions."""
        return self._panels

    @property
    def polygon(self) -> Polygon:
        """The Shapely polygon (union of all panels)."""
        return self._polygon

    @property
    def boundary_edges(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Outer boundary as list of (start, end) line segments."""
        return self._boundary_edges

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max)"""
        return self._polygon.bounds

    @property
    def area(self) -> float:
        """Total area of the mat polygon."""
        return self._polygon.area

    @property
    def n_edges(self) -> int:
        """Number of boundary edge segments."""
        return len(self._boundary_edges)


def _point_to_segment_dist(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Perpendicular distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
