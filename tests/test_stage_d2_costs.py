from __future__ import annotations

import json

import pandas as pd

from bjj_pipeline.stages.stitch.costs import DISALLOWED_EDGE_COST, compute_edge_costs


def _tiny_tables():
	d1_nodes = pd.DataFrame(
		[
			{"node_id": "n0", "base_tracklet_id": "t0", "start_frame": 0, "end_frame": 0, "segment_type": "SOLO"},
			{"node_id": "n1", "base_tracklet_id": "t1", "start_frame": 1, "end_frame": 1, "segment_type": "SOLO"},
		]
	)
	d1_edges = pd.DataFrame(
		[
			{"edge_id": "e0", "edge_type": "EdgeType.CONTINUE", "u": "n0", "v": "n1", "capacity": 1, "dt_frames": 1},
		]
	)
	bank_frames = pd.DataFrame(
		[
			{
				"tracklet_id": "t0",
				"frame_index": 0,
				"x_m": 0.0,
				"y_m": 0.0,
				"x_m_repaired": None,
				"y_m_repaired": None,
				"contact_conf": 0.9,
				"speed_is_implausible": False,
				"accel_is_implausible": False,
			},
			{
				"tracklet_id": "t1",
				"frame_index": 1,
				"x_m": 1.0,
				"y_m": 0.0,
				"x_m_repaired": None,
				"y_m_repaired": None,
				"contact_conf": 0.9,
				"speed_is_implausible": False,
				"accel_is_implausible": False,
			},
		]
	)
	return d1_edges, d1_nodes, bank_frames


def test_d2_missing_geom_disallow_logs_reason() -> None:
	d1_edges, d1_nodes, bank_frames = _tiny_tables()
	# remove one endpoint row -> missing geom
	bank_frames = bank_frames[bank_frames["tracklet_id"] != "t1"].reset_index(drop=True)

	cfg = {"dt_max_s": 1.0, "missing_geom_policy": "disallow"}
	out = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_frames,
			fps=10.0,
			cfg=cfg,
			v_cost_scale_mps_resolved=8.0,
			v_hinge_mps_resolved=8.0,
	)
	assert bool(out.loc[0, "is_allowed"]) is False
	reasons = json.loads(out.loc[0, "disallow_reasons_json"])
	assert "missing_geom" in reasons
	assert float(out.loc[0, "total_cost"]) == DISALLOWED_EDGE_COST


def test_d2_cost_increases_with_distance() -> None:
	d1_edges, d1_nodes, bank_frames = _tiny_tables()
	cfg = {
		"dt_max_s": 1.0,
		"missing_geom_policy": "disallow",
		"w_time": 0.0,
		"w_vreq": 1.0,
		"base_env_cost": 0.0,
		"use_contact_rel": False,
		"use_flags": False,
	}
	# baseline: dist=1.0
	out1 = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_frames,
			fps=10.0,
			cfg=cfg,
			v_cost_scale_mps_resolved=8.0,
			v_hinge_mps_resolved=0.5,  # make hinge active
	)

	# increase distance by moving t1 to x=2.0
	bank_frames2 = bank_frames.copy()
	bank_frames2.loc[bank_frames2["tracklet_id"] == "t1", "x_m"] = 2.0

	out2 = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_frames2,
			fps=10.0,
			cfg=cfg,
			v_cost_scale_mps_resolved=8.0,
			v_hinge_mps_resolved=0.5,
	)

	assert float(out2.loc[0, "total_cost"]) > float(out1.loc[0, "total_cost"])


def test_d2_contact_rel_gentle_scaling() -> None:
	d1_edges, d1_nodes, bank_frames = _tiny_tables()
	cfg = {
		"dt_max_s": 1.0,
		"missing_geom_policy": "disallow",
		"w_time": 0.0,
		"w_vreq": 1.0,
		"base_env_cost": 0.0,
		"use_contact_rel": True,
		"contact_conf_floor": 0.25,
		"contact_rel_alpha": 0.35,
		"use_flags": False,
	}

	out_hi = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_frames,
			fps=10.0,
			cfg=cfg,
			v_cost_scale_mps_resolved=8.0,
			v_hinge_mps_resolved=0.5,
	)

	bank_low = bank_frames.copy()
	bank_low.loc[:, "contact_conf"] = 0.25

	out_lo = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_low,
			fps=10.0,
			cfg=cfg,
			v_cost_scale_mps_resolved=8.0,
			v_hinge_mps_resolved=0.5,
	)

	# lower contact confidence should weakly increase the vreq term
	assert float(out_lo.loc[0, "term_vreq"]) >= float(out_hi.loc[0, "term_vreq"])


def test_d2_disallow_reasons_canonical_order() -> None:
	d1_edges, d1_nodes, bank_frames = _tiny_tables()
	# Force two reasons deterministically: missing geom + dt_too_large
	bank_frames = bank_frames[bank_frames["tracklet_id"] != "t1"].reset_index(drop=True)
	# Make dt_s exceed dt_max_s (fps=10 => dt_s=2.0)
	d1_edges = d1_edges.copy()
	d1_edges.loc[0, "dt_frames"] = 20

	cfg = {"dt_max_s": 1.0, "missing_geom_policy": "disallow"}
	out = compute_edge_costs(
		d1_edges=d1_edges,
		d1_nodes=d1_nodes,
		bank_frames=bank_frames,
		fps=10.0,
		cfg=cfg,
		v_cost_scale_mps_resolved=8.0,
		v_hinge_mps_resolved=8.0,
	)

	assert bool(out.loc[0, "is_allowed"]) is False
	reasons = json.loads(out.loc[0, "disallow_reasons_json"])
	assert reasons == sorted(reasons)
	assert set(reasons) == {"dt_too_large", "missing_geom"}
