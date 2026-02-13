"""Stage D2 runner — compute per-edge costs + normalized constraint spec.

This stage is *solver-agnostic*:
  - D1 defines candidate edges (graph structure).
  - D2 assigns scalar costs and emits normalized constraints.
  - D3 enforces feasibility and runs the solver.

Artifacts:
  - stage_D/d2_edge_costs.parquet
  - stage_D/d2_constraints.json
  - stage_D/audit.jsonl (events: d2_config_resolved, d2_inputs_observed, d2_costs_summary, d2_constraints_summary)

Policy (locked):
  - missing geometry => disallow (log reason)
  - must-link vs disallow conflict => fail loudly (enforced by D3; D2 does not override disallow)
  - contact reliability weighting ON by default (gentle scaling)
  - fps source of truth: clip manifest
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from bjj_pipeline.contracts import f0_validate as v
from bjj_pipeline.stages.stitch.costs import compute_edge_costs
from bjj_pipeline.stages.stitch.d2_constraints import normalize_identity_constraints


def _now_ms() -> int:
	return int(time.time() * 1000)


def _write_audit_event(audit_path: Path, event: Dict[str, Any]) -> None:
	audit_path.parent.mkdir(parents=True, exist_ok=True)
	with audit_path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(event, sort_keys=True) + "\n")


def _stage_d_dict(config: Dict[str, Any]) -> Dict[str, Any]:
	# Mirror Stage D0 behavior (supports both nested styles).
	if isinstance(config.get("stages"), dict) and isinstance(config["stages"].get("stage_D"), dict):
		return config["stages"]["stage_D"]
	if isinstance(config.get("stage_D"), dict):
		return config["stage_D"]
	return {}


def _quantiles(arr: np.ndarray, ps=(0.5, 0.9, 0.99)) -> Dict[str, float]:
	if arr.size == 0:
		return {}
	qs = np.quantile(arr, ps)
	out: Dict[str, float] = {}
	for p, q in zip(ps, qs):
		out[f"p{int(p*100):02d}"] = float(q)
	return out


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
	missing = sorted(required - set(df.columns))
	if missing:
		observed = sorted(df.columns.tolist())
		raise ValueError(f"D2 missing required columns for {label}: missing={missing} observed={observed}")


def run_d2(*, config: Dict[str, Any], inputs: Dict[str, Any]) -> None:
	layout = inputs["layout"]
	manifest = inputs["manifest"]
	audit_path = layout.audit_jsonl("D")

	stage_d = _stage_d_dict(config)
	d0 = stage_d.get("d0", {}) if isinstance(stage_d, dict) else {}
	d0_kin = d0.get("kinematics", {}) if isinstance(d0, dict) else {}
	d1_cfg = stage_d.get("d1", {}) if isinstance(stage_d, dict) else {}

	d2_cfg = stage_d.get("d2_costs", {}) if isinstance(stage_d, dict) else {}
	if not isinstance(d2_cfg, dict):
		raise ValueError("stage_D.d2_costs must be a dict if provided")

	# Surface D1 merge/split distance thresholds into D2 cfg so that
	# MERGE/SPLIT geometry costs can be normalized by the same scales.
	if isinstance(d1_cfg, dict):
		for key in ("merge_dist_m", "split_dist_m"):
			if key in d1_cfg and key not in d2_cfg:
				d2_cfg[key] = d1_cfg[key]

	# Resolve endpoint_search_window_frames for D2 endpoint geometry lookup.
	# Prefer explicit D2 config; otherwise mirror D1 carrier_coord_window_frames.
	endpoint_window_source: str | None = None
	if "endpoint_search_window_frames" in d2_cfg and d2_cfg.get("endpoint_search_window_frames") is not None:
		d2_cfg["endpoint_search_window_frames"] = int(d2_cfg["endpoint_search_window_frames"])
		endpoint_window_source = "stage_D.d2_costs.endpoint_search_window_frames"
	elif isinstance(d1_cfg, dict) and d1_cfg.get("carrier_coord_window_frames") is not None:
		d2_cfg["endpoint_search_window_frames"] = int(d1_cfg["carrier_coord_window_frames"])
		endpoint_window_source = "stage_D.d1.carrier_coord_window_frames"
	else:
		raise ValueError(
			"D2 requires endpoint_search_window_frames; set stage_D.d2_costs.endpoint_search_window_frames "
			"or stage_D.d1.carrier_coord_window_frames"
		)
	endpoint_search_window_frames_resolved = int(d2_cfg["endpoint_search_window_frames"])

	enabled = bool(d2_cfg.get("enabled", True))
	if not enabled:
		_write_audit_event(
					audit_path,
					{"artifact_type": "d2_skipped", "created_at_ms": _now_ms(), "reason": "stage_D.d2_costs.enabled=false"},
		)
		return

	fps = float(getattr(manifest, "fps", None))
	if fps <= 0:
		raise ValueError(f"Manifest fps must be >0 (got {fps!r})")

	# Resolve non-redundant speed params: default to D0 kinematics v_max_mps
	d0_vmax = d0_kin.get("v_max_mps", None)
	if d0_vmax is None:
		raise ValueError("stage_D.d0.kinematics.v_max_mps is required to derive D2 motion normalization defaults")
	d0_vmax = float(d0_vmax)

	v_cost_scale_mps = d2_cfg.get("v_cost_scale_mps", None)
	v_hinge_mps = d2_cfg.get("v_hinge_mps", None)

	if v_cost_scale_mps is None:
		v_cost_scale_mps_resolved = d0_vmax
		v_cost_scale_source = "stage_D.d0.kinematics.v_max_mps"
	else:
		v_cost_scale_mps_resolved = float(v_cost_scale_mps)
		v_cost_scale_source = "stage_D.d2_costs.v_cost_scale_mps"

	if v_hinge_mps is None:
		v_hinge_mps_resolved = v_cost_scale_mps_resolved
		v_hinge_source = "resolved(v_cost_scale_mps)"
	else:
		v_hinge_mps_resolved = float(v_hinge_mps)
		v_hinge_source = "stage_D.d2_costs.v_hinge_mps"

	# Resolve reconnect speed threshold (do not assume v_cost_scale_mps is a threshold).
	reconnect_v_max_mps = d2_cfg.get("reconnect_v_max_mps", None)
	if reconnect_v_max_mps is None:
		reconnect_v_max_mps_resolved = d0_vmax
		reconnect_v_max_source = "stage_D.d0.kinematics.v_max_mps"
	else:
		reconnect_v_max_mps_resolved = float(reconnect_v_max_mps)
		reconnect_v_max_source = "stage_D.d2_costs.reconnect_v_max_mps"
	# Ensure downstream cost code sees a concrete value (back-compat: key already exists in config model).
	d2_cfg["reconnect_v_max_mps"] = reconnect_v_max_mps_resolved

	# Audit resolved config + derivations
	_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_config_resolved",
				"created_at_ms": _now_ms(),
				"d2_costs": d2_cfg,
				"v_cost_scale_mps_resolved": v_cost_scale_mps_resolved,
				"v_cost_scale_mps_source": v_cost_scale_source,
				"v_hinge_mps_resolved": v_hinge_mps_resolved,
				"reconnect_v_max_mps_resolved": reconnect_v_max_mps_resolved,
				"reconnect_v_max_source": reconnect_v_max_source,
				"v_hinge_mps_source": v_hinge_source,
				"endpoint_search_window_frames_resolved": endpoint_search_window_frames_resolved,
				"endpoint_search_window_frames_source": endpoint_window_source,
				"fps": fps,
			},
	)

	# Load required inputs (D1 is canonical left table)
	d1_nodes_path = layout.d1_graph_nodes_parquet()
	d1_edges_path = layout.d1_graph_edges_parquet()
	bank_frames_path = layout.tracklet_bank_frames_parquet()

	if not d1_nodes_path.exists():
		raise FileNotFoundError(f"Missing D1 nodes parquet: {d1_nodes_path}")
	if not d1_edges_path.exists():
		raise FileNotFoundError(f"Missing D1 edges parquet: {d1_edges_path}")
	if not bank_frames_path.exists():
		raise FileNotFoundError(f"Missing D0 bank frames parquet: {bank_frames_path}")

	d1_nodes = pd.read_parquet(d1_nodes_path)
	d1_edges = pd.read_parquet(d1_edges_path)
	bank_frames = pd.read_parquet(bank_frames_path)

	# Fail-fast required columns (evidence-based): only require what is needed
	# based on the actual edge types present in this run.
	_require_columns(d1_nodes, {"node_id", "base_tracklet_id", "segment_type"}, "d1_graph_nodes")
	_require_columns(d1_edges, {"edge_id", "edge_type", "u", "v", "dt_frames"}, "d1_graph_edges")
	_require_columns(
		bank_frames,
		{"tracklet_id", "frame_index", "contact_conf", "speed_is_implausible", "accel_is_implausible"},
		"tracklet_bank_frames",
	)

	def _edge_type_suffixes(edges_df: pd.DataFrame) -> set[str]:
		vals = edges_df.get("edge_type", pd.Series([], dtype="string"))
		out: set[str] = set()
		for s in vals.astype(str).tolist():
			out.add(s.split(".")[-1])
		return out

	etype_suffixes = _edge_type_suffixes(d1_edges)
	if "CONTINUE" in etype_suffixes:
		_require_columns(d1_nodes, {"start_frame", "end_frame"}, "d1_graph_nodes (CONTINUE)")
		_require_columns(
			bank_frames,
			{"x_m", "y_m", "x_m_repaired", "y_m_repaired"},
			"tracklet_bank_frames (CONTINUE)",
		)
	if "MERGE" in etype_suffixes or "SPLIT" in etype_suffixes:
		_require_columns(
			d1_nodes,
			{"disappearing_tracklet_id", "new_tracklet_id", "carrier_tracklet_id"},
			"d1_graph_nodes (MERGE/SPLIT coherence)",
		)

	identity_hints_path = layout.identity_hints_jsonl()
	constraints = normalize_identity_constraints(identity_hints_path)

	_write_audit_event(
		audit_path,
		{
			"artifact_type": "d2_inputs_observed",
			"created_at_ms": _now_ms(),
			"fps": fps,
			"n_nodes": int(len(d1_nodes)),
			"n_edges": int(len(d1_edges)),
			"n_bank_frames": int(len(bank_frames)),
			"n_identity_hint_events_read": int(constraints.get("stats", {}).get("n_events_read", 0)),
			"n_identity_hint_events_used": int(constraints.get("stats", {}).get("n_events_used", 0)),
			"d1_nodes_columns": sorted(d1_nodes.columns.tolist()),
			"d1_edges_columns": sorted(d1_edges.columns.tolist()),
			"d0_bank_frames_columns": sorted(bank_frames.columns.tolist()),
		},
	)

	# Compute per-edge costs
	costs_df, endpoint_stats = compute_edge_costs(
			d1_edges=d1_edges,
			d1_nodes=d1_nodes,
			bank_frames=bank_frames,
			fps=fps,
			cfg=d2_cfg,
			v_cost_scale_mps_resolved=v_cost_scale_mps_resolved,
			v_hinge_mps_resolved=v_hinge_mps_resolved,
	)

	# Summarize endpoint lookup behavior (contract-safe: audit only; do not add parquet columns).
	_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_endpoint_lookup_summary",
				"created_at_ms": _now_ms(),
				**endpoint_stats,
			},
	)

	# Soft Option B summary: shadowed reconnect candidates
	try:
		allowed_by_edge_id = costs_df.set_index("edge_id")["is_allowed"].to_dict()
		n_reconnect_edges_total = 0
		n_shadowed_reconnect_edges_total = 0
		n_shadowed_reconnect_edges_allowed = 0
		n_shadowed_reconnect_edges_disallowed = 0
		for _, er in d1_edges.iterrows():
			raw = er.get("payload_json", None)
			try:
				p = json.loads(str(raw)) if raw is not None and str(raw) != "nan" else {}
			except Exception:
				p = {}
			if not isinstance(p, dict):
				continue
			if bool(p.get("reconnect", False)):
				n_reconnect_edges_total += 1
				if bool(p.get("shadowed_by_group_chain", False)):
					n_shadowed_reconnect_edges_total += 1
					is_allowed_edge = bool(allowed_by_edge_id.get(str(er.get("edge_id")), False))
					if is_allowed_edge:
						n_shadowed_reconnect_edges_allowed += 1
					else:
						n_shadowed_reconnect_edges_disallowed += 1
		_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_shadowed_reconnect_summary",
				"created_at_ms": _now_ms(),
				"n_reconnect_edges_total": int(n_reconnect_edges_total),
				"n_shadowed_reconnect_edges_total": int(n_shadowed_reconnect_edges_total),
				"n_shadowed_reconnect_edges_allowed": int(n_shadowed_reconnect_edges_allowed),
				"n_shadowed_reconnect_edges_disallowed": int(n_shadowed_reconnect_edges_disallowed),
				"shadowed_reconnect_policy": str(d2_cfg.get("shadowed_reconnect_policy", "disallow")),
				"shadowed_reconnect_penalty": float(d2_cfg.get("shadowed_reconnect_penalty", 10.0)),
			},
		)
	except Exception as e:
		_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_shadowed_reconnect_summary",
				"created_at_ms": _now_ms(),
				"error": str(e),
			},
		)

	# Validate before writing
	v.validate_d2_edge_costs_df(costs_df)
	v.validate_d2_constraints_json(constraints)

	# Write artifacts
	out_costs = layout.d2_edge_costs_parquet()
	out_constraints = layout.d2_constraints_json()
	out_costs.parent.mkdir(parents=True, exist_ok=True)
	costs_df.to_parquet(out_costs, index=False)
	out_constraints.write_text(json.dumps(constraints, sort_keys=True, indent=2), encoding="utf-8")

	# Summary audit
	allowed = costs_df[costs_df["is_allowed"] == True]  # noqa: E712
	disallowed = costs_df[costs_df["is_allowed"] == False]  # noqa: E712

	# parse disallow reasons (small; row count is candidate edges, not frames)
	reason_counts: Dict[str, int] = {}
	for s in disallowed["disallow_reasons_json"].tolist():
		try:
			rs = json.loads(s)
		except Exception:
			rs = ["invalid_json"]
		for r in rs:
			reason_counts[str(r)] = reason_counts.get(str(r), 0) + 1

	def _q(col: str) -> Dict[str, float]:
		if col not in allowed.columns:
			return {}
		arr = allowed[col].dropna().to_numpy(dtype=float)
		return _quantiles(arr)

	def _edge_type_counts(df: pd.DataFrame) -> Dict[str, int]:
		if df.empty or "edge_type" not in df.columns:
			return {}
		counts = df.groupby("edge_type").size().to_dict()
		return {str(k): int(v) for k, v in counts.items()}

	edge_type_counts_total = _edge_type_counts(costs_df)
	edge_type_counts_allowed = _edge_type_counts(allowed)
	edge_type_counts_disallowed = _edge_type_counts(disallowed)

	_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_costs_summary",
				"created_at_ms": _now_ms(),
				"n_edges_total": int(len(costs_df)),
				"n_edges_allowed": int(len(allowed)),
				"n_edges_disallowed": int(len(disallowed)),
				"edge_type_counts_total": edge_type_counts_total,
				"edge_type_counts_allowed": edge_type_counts_allowed,
				"edge_type_counts_disallowed": edge_type_counts_disallowed,
				"disallow_reason_counts": reason_counts,
				"dt_s_q": _q("dt_s"),
				"dist_m_q": _q("dist_m"),
				"total_cost_q": _q("total_cost"),
				"term_vreq_q": _q("term_vreq"),
				"term_group_coherence_q": _q("term_group_coherence"),
			},
	)

	_write_audit_event(
			audit_path,
			{
				"artifact_type": "d2_constraints_summary",
				"created_at_ms": _now_ms(),
				**constraints.get("stats", {}),
			},
	)
