from __future__ import annotations

import pandas as pd
import pytest

from bjj_pipeline.stages.stitch.d3_compile import (
	_assert_edge_costs_cover_edges,
	_prune_disallowed_edges,
	_validate_constraints_canonical,
)


def test_d3_compile_fails_missing_cost_row() -> None:
	edges = pd.DataFrame([
		{"edge_id": "e0"},
		{"edge_id": "e1"},
	])
	costs = pd.DataFrame([
		{"edge_id": "e0", "is_allowed": True},
	])
	with pytest.raises(ValueError) as ei:
		_assert_edge_costs_cover_edges(edges, costs)
	assert "missing" in str(ei.value).lower()


def test_d3_compile_fails_duplicate_cost_row() -> None:
	edges = pd.DataFrame([
		{"edge_id": "e0"},
	])
	costs = pd.DataFrame([
		{"edge_id": "e0", "is_allowed": True},
		{"edge_id": "e0", "is_allowed": True},
	])
	with pytest.raises(ValueError) as ei:
		_assert_edge_costs_cover_edges(edges, costs)
	assert "duplicate" in str(ei.value).lower() or "duplicates" in str(ei.value).lower()


def test_d3_compile_prune_drops_disallowed_edges_deterministically() -> None:
	edges = pd.DataFrame([
		{"edge_id": "e1", "edge_type": "EdgeType.CONTINUE"},
		{"edge_id": "e0", "edge_type": "EdgeType.SPLIT"},
	])
	costs = pd.DataFrame([
		{"edge_id": "e0", "is_allowed": True},
		{"edge_id": "e1", "is_allowed": False, "disallow_reasons_json": "[\"x\"]"},
	])
	edges_p, costs_p, stats = _prune_disallowed_edges(edges_df=edges, costs_df=costs)
	assert stats["n_edges_before"] == 2
	assert stats["n_edges_after"] == 1
	assert edges_p["edge_id"].tolist() == ["e0"]
	assert costs_p["edge_id"].tolist() == ["e0"]


def test_d3_compile_constraints_validation_accepts_canonical() -> None:
	constraints = {
		"cannot_link_pairs": [["t0", "t9"], ["t1", "t2"]],
		"must_link_groups": [
			{"anchor_key": "tag:1", "tracklet_ids": ["t5", "t6"]},
			{"anchor_key": "tag:2", "tracklet_ids": ["t1"]},
		],
		"stats": {"n_events_read": 0},
	}
	_validate_constraints_canonical(constraints)


def test_d3_compile_constraints_validation_rejects_noncanonical() -> None:
	constraints = {
		"cannot_link_pairs": [["t9", "t0"]],  # non-canonical ordering
		"must_link_groups": [
			{"anchor_key": "tag:2", "tracklet_ids": ["t1"]},
			{"anchor_key": "tag:1", "tracklet_ids": ["t6", "t5"]},  # unsorted
		],
		"stats": {"n_events_read": 0},
	}
	with pytest.raises(ValueError):
		_validate_constraints_canonical(constraints)

