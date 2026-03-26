from __future__ import annotations

import pandas as pd
import pytest

cp_model = pytest.importorskip("ortools.sat.python.cp_model")

from bjj_pipeline.stages.stitch.d3_ilp2 import solve_structure_ilp2_core


def test_d3_poc1_ilp_selects_unique_path_and_is_deterministic() -> None:
	# Tiny DAG with a single feasible path S -> n1 -> n2 -> T
	nodes = pd.DataFrame(
		[
			{"node_id": "S", "node_type": "NodeType.SOURCE", "capacity": 2},
			{"node_id": "n1", "node_type": "NodeType.SINGLE_TRACKLET", "capacity": 1},
			{"node_id": "n2", "node_type": "NodeType.SINGLE_TRACKLET", "capacity": 1},
			{"node_id": "T", "node_type": "NodeType.SINK", "capacity": 2},
		]
	)

	edges = pd.DataFrame(
		[
			{"edge_id": "e0", "u": "S", "v": "n1", "edge_type": "EdgeType.BIRTH", "capacity": 1},
			{"edge_id": "e1", "u": "n1", "v": "n2", "edge_type": "EdgeType.CONTINUE", "capacity": 1},
			{"edge_id": "e2", "u": "n2", "v": "T", "edge_type": "EdgeType.DEATH", "capacity": 1},
		]
	)

	costs = pd.DataFrame(
		[
			{"edge_id": "e0", "total_cost": 2.0, "is_allowed": True},
			{"edge_id": "e1", "total_cost": 0.5, "is_allowed": True},
			{"edge_id": "e2", "total_cost": 2.0, "is_allowed": True},
		]
	)

	res1, _, _ = solve_structure_ilp2_core(nodes_df=nodes, edges_df=edges, costs_df=costs)
	res2, _, _ = solve_structure_ilp2_core(nodes_df=nodes, edges_df=edges, costs_df=costs)

	assert res1.status in ("OPTIMAL", "FEASIBLE")
	assert res2.status in ("OPTIMAL", "FEASIBLE")
	assert res1.selected_edge_ids == ["e0", "e1", "e2"]
	assert res2.selected_edge_ids == ["e0", "e1", "e2"]
	assert res1.objective_value == pytest.approx(4.5)
	assert res2.objective_value == pytest.approx(4.5)
