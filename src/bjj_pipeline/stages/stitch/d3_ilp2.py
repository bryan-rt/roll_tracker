from __future__ import annotations
import math
"""Stage D3 — ILP structure solve (POC_2_TAGS_MCF).

This module is an *alternate* D3 solver implementation intended to replace the
current POC_2_TAGS "node label" formulation with a multi-commodity flow (MCF)
overlay per AprilTag.

Current goal:
	- Provide a standalone ILP2 solver for Stage D3 with no ILP1 delegation.
	- Solve identity flow on the D1 graph, then overlay binary per-tag commodity
		threads constrained by identity flow capacity.
	- Allow sparse AprilTag pings to bind to SINGLE / GROUP / GROUPISH nodes so
		group-derived observations remain available to the solver and debug artifacts.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ortools.sat.python import cp_model  # type: ignore

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event
from bjj_pipeline.stages.stitch.d3_compile import CompiledInputs
from bjj_pipeline.stages.stitch.d3_common import (
	_debug_dir,
	_find_unique_node_id,
	_require_columns,
	_write_entities_format_a,
	_write_solution_ledger_json,
)

# Breadcrumb constants (written into audit + debug ledger copy).
_SOLVER_IMPL: str = "ilp2"
_SOLVER_MODULE: str = __name__
_SOLVER_VERSION: str = "mcf3b_group_pair_steering"


@dataclass(frozen=True)
class ILPResult:
	"""Public result contract consumed by Stage D4.

	Stage D4 only relies on a subset of fields (selected_edge_ids, flow_by_edge_id).
	"""

	status: str
	objective_scaled: int | None
	objective_value: float | None
	runtime_ms: int
	selected_edge_ids: List[str]
	flow_by_edge_id: Dict[str, int]
	cost_scale: int
	# Transparency/debugging for objective discretization and model constraints.
	enforced_min_one_path: bool
	rounding_n_edges: int
	rounding_n_edges_nonzero: int
	rounding_max_abs_scaled_error: float
	rounding_max_abs_cost_error: float
	# Tracklet "explain-or-penalize" diagnostics
	unexplained_tracklet_penalty: float | None
	n_tracklets_total: int
	n_tracklets_explained: int
	n_tracklets_unexplained: int
	# Deterministic lists for full transparency
	dropped_tracklet_ids: List[str]
	explained_tracklet_ids: List[str]
	# D4 handoff: realized local in/out pairings through GROUP/GROUPISH nodes
	realized_group_pairings: List[Dict[str, Any]]


def _write_json_atomic(*, path: Path, payload: Dict[str, Any]) -> None:
	import json

	tmp = path.parent / (path.name + ".tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)
	tmp.replace(path)


def _write_solver_breadcrumbs_json(*, debug_dir: Path, payload: Dict[str, Any]) -> Path:
	"""Write a tiny json file to make it provable which solver module ran."""
	out = debug_dir / "d3_solver_breadcrumbs.json"
	_write_json_atomic(path=out, payload=payload)
	return out


def _cost_scale_for(_: pd.Series | None = None) -> int:
	"""Deterministic integer scale for CP-SAT objective."""
	return 1000


def _scaled_costs(costs_df: pd.DataFrame, *, scale: int) -> tuple[Dict[str, int], Dict[str, float | int]]:
	_require_columns(costs_df, name="d2_edge_costs", cols=["edge_id", "total_cost"])
	out: Dict[str, int] = {}
	n_edges = 0
	n_nonzero = 0
	max_abs_scaled_err = 0.0
	for _, row in costs_df.iterrows():
		n_edges += 1
		edge_id = str(row["edge_id"])
		c = float(row["total_cost"])
		s = c * float(scale)
		rounded = int(round(s))
		err = abs(s - float(rounded))
		if err > 0.0:
			n_nonzero += 1
			if err > max_abs_scaled_err:
				max_abs_scaled_err = err
		out[edge_id] = rounded
	stats: Dict[str, float | int] = {
		"rounding_n_edges": int(n_edges),
		"rounding_n_edges_nonzero": int(n_nonzero),
		"rounding_max_abs_scaled_error": float(max_abs_scaled_err),
		"rounding_max_abs_cost_error": float(max_abs_scaled_err) / float(scale) if scale > 0 else float("nan"),
	}
	return out, stats


def _parse_payload_desired_capacity(payload_json: Any) -> Optional[int]:
	"""Extract desired_capacity from a payload_json cell (dict or JSON string)."""
	if payload_json is None:
		return None
	obj: Any = None
	if isinstance(payload_json, dict):
		obj = payload_json
	elif isinstance(payload_json, str):
		s = payload_json.strip()
		if not s:
			return None
		try:
			import json

			obj = json.loads(s)
		except Exception:
			return None
	else:
		return None
	if not isinstance(obj, dict):
		return None
	dc = obj.get("desired_capacity")
	try:
		return int(dc) if dc is not None else None
	except Exception:
		return None


def _compute_edge_capacity_eff(edges_df: pd.DataFrame) -> Dict[str, int]:
	_require_columns(edges_df, name="d1_graph_edges", cols=["edge_id", "capacity"])
	out: Dict[str, int] = {}
	for _, row in edges_df.iterrows():
		eid = str(row["edge_id"])
		cap = row["capacity"]
		try:
			cap_i = int(cap)
		except Exception:
			cap_i = 1
		cap_eff = max(1, cap_i)
		# Optional: capacity_eff column precomputed upstream
		if "capacity_eff" in edges_df.columns:
			try:
				cap_eff = max(cap_eff, int(row["capacity_eff"]))
			except Exception:
				pass
		# Optional: payload_json.desired_capacity
		if "payload_json" in edges_df.columns:
			dc = _parse_payload_desired_capacity(row.get("payload_json"))
			if dc is not None:
				cap_eff = max(cap_eff, int(dc))
		out[eid] = int(cap_eff)
	return out


def _compute_node_capacity_eff(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, *, edge_cap_eff: Dict[str, int]) -> Dict[str, int]:
	_require_columns(nodes_df, name="d1_graph_nodes", cols=["node_id", "capacity"])
	_require_columns(edges_df, name="d1_graph_edges", cols=["edge_id", "u", "v"])
	# Precompute incident max edge cap per node
	inc_max: Dict[str, int] = {}
	for _, row in edges_df.iterrows():
		eid = str(row["edge_id"])
		cap = int(edge_cap_eff.get(eid, 1))
		u = str(row["u"])
		v = str(row["v"])
		inc_max[u] = max(int(inc_max.get(u, 1)), cap)
		inc_max[v] = max(int(inc_max.get(v, 1)), cap)
	out: Dict[str, int] = {}
	for _, row in nodes_df.iterrows():
		nid = str(row["node_id"])
		try:
			base = int(row["capacity"])
		except Exception:
			base = 1
		out[nid] = int(max(1, base, int(inc_max.get(nid, 1))))
	return out


def _safe_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
			return None
		return int(x)
	except Exception:
		return None


def _get_str(row: Any, key: str) -> Optional[str]:
	try:
		v = row[key]
	except Exception:
		try:
			v = getattr(row, key)
		except Exception:
			return None
	if v is None:
		return None
	s = str(v)
	if s.strip() == "" or s.strip().lower() == "none":
		return None
	return s


def _node_span(row: Any) -> tuple[Optional[int], Optional[int]]:
	sf = _safe_int(getattr(row, "start_frame", None) if hasattr(row, "start_frame") else (row.get("start_frame") if isinstance(row, dict) else None))
	ef = _safe_int(getattr(row, "end_frame", None) if hasattr(row, "end_frame") else (row.get("end_frame") if isinstance(row, dict) else None))
	return sf, ef


def _is_carrier_cont_edge(*, edge_id: str, edge_type: str) -> bool:
	return edge_type == "EdgeType.CONTINUE" and edge_id.startswith("E:CONT:")


def _is_group_cont_edge(*, edge_row: pd.Series, edge_cap_eff: Dict[str, int]) -> bool:
	# Best-effort payload_json predicate:
	# desired_capacity==2 AND dest_groupish==True (or equivalent)
	try:
		eid = str(edge_row["edge_id"])
	except Exception:
		return False
	if int(edge_cap_eff.get(eid, 1)) < 2:
		return False
	if "payload_json" not in edge_row.index:
		return False
	obj: Any = None
	pj = edge_row.get("payload_json")
	if isinstance(pj, dict):
		obj = pj
	elif isinstance(pj, str):
		s = pj.strip()
		if not s:
			return False
		try:
			import json

			obj = json.loads(s)
		except Exception:
			return False
	else:
		return False
	if not isinstance(obj, dict):
		return False
	dc = obj.get("desired_capacity")
	try:
		dc_i = int(dc) if dc is not None else None
	except Exception:
		dc_i = None
	if dc_i is not None and dc_i < 2:
		return False
	# Accept a few possible key names for "dest_groupish"
	dg = obj.get("dest_groupish", obj.get("dest_is_groupish", obj.get("to_groupish")))
	if dg is True:
		return True
	# Sometimes these may be encoded as 1/0 or "true"
	if isinstance(dg, (int, float)) and int(dg) == 1:
		return True
	if isinstance(dg, str) and dg.strip().lower() in ("true", "1", "yes", "y"):
		return True
	return False


def _pick_unique_edge(edge_ids: List[str]) -> Optional[str]:
	if not edge_ids:
		return None
	return sorted([str(e) for e in edge_ids])[0]


def _emit_ilp2_group_semantics_debug(*, debug_dir: Path, payload: Dict[str, Any]) -> Path:
	out = debug_dir / "d3_ilp2_group_semantics.json"
	_write_json_atomic(path=out, payload=payload)
	return out


def _emit_ilp2_explain_or_penalize_debug(*, debug_dir: Path, payload: Dict[str, Any]) -> Path:
	"""Emit `_debug/d3_ilp2_explain_or_penalize.json` (dev-only evidence artifact)."""
	out = debug_dir / "d3_ilp2_explain_or_penalize.json"
	_write_json_atomic(path=out, payload=payload)
	return out


def _emit_ilp2_tag_pair_preferences_debug(*, debug_dir: Path, payload: Dict[str, Any]) -> Path:
	"""Emit `_debug/d3_ilp2_tag_pair_preferences.json` for local tag-consistent pair steering."""
	out = debug_dir / "d3_ilp2_tag_pair_preferences.json"
	_write_json_atomic(path=out, payload=payload)
	return out


def _apply_group_semantics_constraints(
	*,
	model: Any,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	edge_u: Dict[str, str],
	edge_v: Dict[str, str],
	edge_type: Dict[str, str],
	in_edges_by_node: Dict[str, List[str]],
	out_edges_by_node: Dict[str, List[str]],
	flow_var: Dict[str, Any],
	used_var: Dict[str, Any],
	edge_cap_eff: Dict[str, int],
	node_cap_eff: Dict[str, int],
	group_boundary_window_frames: int,
	continue_edges_by_base_pair: Dict[tuple[str, str], List[str]],
	node_to_base: Dict[str, str],
) -> Dict[str, Any]:
	"""Apply GROUP/GROUPISH semantics constraints on top of baseline flow model.

	Safety rule: if required structural edges are missing, force the node unused.
	"""
	from ortools.sat.python import cp_model  # type: ignore

	# Determine clip span for boundary tests
	clip_last_frame: Optional[int] = None
	if "end_frame" in nodes_df.columns:
		try:
			m = pd.to_numeric(nodes_df["end_frame"], errors="coerce")
			if m.notna().any():
				clip_last_frame = int(m.max())
		except Exception:
			clip_last_frame = None
	if clip_last_frame is None:
		clip_last_frame = 0

	window = int(group_boundary_window_frames)

	def is_start_boundary(sf: Optional[int]) -> bool:
		if sf is None:
			return False
		return int(sf) <= (0 + window - 1)

	def is_end_boundary(ef: Optional[int]) -> bool:
		if ef is None:
			return False
		return int(ef) >= (int(clip_last_frame) - window + 1)

	nodes_out: List[Dict[str, Any]] = []
	n_forced_unused = 0
	n_missing_merge_required = 0
	n_missing_split_required = 0
	n_missing_groupish_bridge = 0
	groupish_overlay_nodes: List[Dict[str, Any]] = []

	# Iterate over group and groupish nodes
	for _, row in nodes_df.iterrows():
		nt = str(row["node_type"])
		if nt not in ("NodeType.GROUP_TRACKLET", "NodeType.GROUPISH_TRACKLET"):
			continue
		nid = str(row["node_id"])
		sf = _safe_int(row.get("start_frame")) if "start_frame" in row.index else None
		ef = _safe_int(row.get("end_frame")) if "end_frame" in row.index else None

		# Create used_node var
		used_n = model.NewBoolVar(f"used_node[{nid}]")

		# 0-or-2 on inbound flow
		ins = in_edges_by_node.get(nid, [])
		in_sum = sum(flow_var[e] for e in ins) if ins else 0
		model.Add(in_sum == 2 * used_n)

		outs = out_edges_by_node.get(nid, [])

		# Classify incident edges
		carrier_cont_in = []
		carrier_cont_out = []
		merge_in = []
		split_out = []
		group_cont_in = []
		group_cont_out = []

		# Build fast row lookup for edges_df by edge_id for payload predicate
		# (Edges_df is small enough; O(E) scan is acceptable per node.)
		in_set = set(str(e) for e in ins)
		out_set = set(str(e) for e in outs)
		for _, er in edges_df.iterrows():
			eid = str(er["edge_id"])
			et = str(er["edge_type"])
			if eid in in_set:
				if _is_carrier_cont_edge(edge_id=eid, edge_type=et):
					carrier_cont_in.append(eid)
				if et == "EdgeType.MERGE":
					merge_in.append(eid)
				if _is_group_cont_edge(edge_row=er, edge_cap_eff=edge_cap_eff):
					group_cont_in.append(eid)
			if eid in out_set:
				if _is_carrier_cont_edge(edge_id=eid, edge_type=et):
					carrier_cont_out.append(eid)
				if et == "EdgeType.SPLIT":
					split_out.append(eid)
				if _is_group_cont_edge(edge_row=er, edge_cap_eff=edge_cap_eff):
					group_cont_out.append(eid)

		carrier_cont_in_e = _pick_unique_edge(carrier_cont_in)
		carrier_cont_out_e = _pick_unique_edge(carrier_cont_out)
		merge_in = sorted(merge_in)
		split_out = sorted(split_out)
		group_cont_in = sorted(group_cont_in)
		group_cont_out = sorted(group_cont_out)

		# Helper: enforce int cap-2 edge to be {0,2} and <= 2*used_n
		def enforce_zero_or_two(eid: str) -> None:
			# Must be an int var with cap>=2 (else no-op)
			if int(edge_cap_eff.get(eid, 1)) < 2:
				return
			# f <= 2*used_n (forces 0 when unused)
			model.Add(flow_var[eid] <= 2 * used_n)
			# force domain {0,2} via selector z
			z = model.NewBoolVar(f"z2[{nid}:{eid}]")
			# If z==1 -> flow=2, else flow=0
			model.Add(flow_var[eid] == 2).OnlyEnforceIf(z)
			model.Add(flow_var[eid] == 0).OnlyEnforceIf(z.Not())
			# If used_n==0 then z must be 0 (already implied by f<=0, but make explicit)
			model.Add(z == 0).OnlyEnforceIf(used_n.Not())

		# Carrier chain saturation (best-effort)
		if carrier_cont_in_e is not None:
			model.Add(flow_var[carrier_cont_in_e] == used_n)
		if carrier_cont_out_e is not None:
			model.Add(flow_var[carrier_cont_out_e] == used_n)

		# Metadata
		carrier_tid = _get_str(row, "carrier_tracklet_id") if "carrier_tracklet_id" in row.index else None
		disp_tid = _get_str(row, "disappearing_tracklet_id") if "disappearing_tracklet_id" in row.index else None
		new_tid = _get_str(row, "new_tracklet_id") if "new_tracklet_id" in row.index else None

		# GROUP_TRACKLET: require merge/split based on metadata
		forced_unused_reason: Optional[str] = None
		if nt == "NodeType.GROUP_TRACKLET":
			if disp_tid is not None:
				if not merge_in:
					model.Add(used_n == 0)
					forced_unused_reason = "missing_required_merge_in"
					n_forced_unused += 1
					n_missing_merge_required += 1
				else:
					# choose first merge edge and force it to used_n, others off
					keep = merge_in[0]
					model.Add(flow_var[keep] == used_n)
					for e in merge_in[1:]:
						model.Add(used_var[e] == 0)
			else:
				# No disappearing => no merge edges should be used
				for e in merge_in:
					model.Add(used_var[e] == 0)

			if new_tid is not None:
				if not split_out:
					model.Add(used_n == 0)
					if forced_unused_reason is None:
						forced_unused_reason = "missing_required_split_out"
						n_forced_unused += 1
					n_missing_split_required += 1
				else:
					keep = split_out[0]
					model.Add(flow_var[keep] == used_n)
					for e in split_out[1:]:
						model.Add(used_var[e] == 0)
			else:
				for e in split_out:
					model.Add(used_var[e] == 0)

			# Group continuation edges: constrain to {0,2} when present
			for e in group_cont_in:
				enforce_zero_or_two(e)
			for e in group_cont_out:
				enforce_zero_or_two(e)

		# GROUPISH_TRACKLET: require cap-2 bridge integrity if such edges exist
		if nt == "NodeType.GROUPISH_TRACKLET":
			# Enforce group-cont edges are {0,2} when present
			for e in group_cont_in:
				enforce_zero_or_two(e)
			for e in group_cont_out:
				enforce_zero_or_two(e)

			if (len(group_cont_in) == 0) or (len(group_cont_out) == 0):
				# Structural integrity: groupish without cap-2 bridge is meaningless
				model.Add(used_n == 0)
				forced_unused_reason = "missing_groupish_group_cont_bridge"
				n_forced_unused += 1
				n_missing_groupish_bridge += 1
			else:
				# If used, require at least one cap-2 bridge in and out carries the second unit (>=1 implies 2)
				model.Add(sum(flow_var[e] for e in group_cont_in) >= used_n)
				model.Add(sum(flow_var[e] for e in group_cont_out) >= used_n)

		# Group-derived tightening (carrier vs disp/new/opponent) for SINGLE<->SINGLE CONTINUE edges
		tightening_pairs: List[Dict[str, Any]] = []

		def gate_pair(a: Optional[str], b: Optional[str]) -> None:
			if a is None or b is None:
				return
			x = str(a)
			y = str(b)
			k = (x, y) if x < y else (y, x)
			edges_to_gate = continue_edges_by_base_pair.get(k, [])
			if not edges_to_gate:
				return
			for eid in edges_to_gate:
				# Disable that CONTINUE edge when this group/groupish node is used
				model.Add(used_var[eid] == 0).OnlyEnforceIf(used_n)
			tightening_pairs.append({"pair": [x, y], "n_edges_gated": int(len(edges_to_gate))})

		# For GROUP: carrier vs disappearing/new
		if carrier_tid is not None:
			gate_pair(carrier_tid, disp_tid)
			gate_pair(carrier_tid, new_tid)

			# For GROUPISH: try best-effort "paired/opponent" fields if present
			if nt == "NodeType.GROUPISH_TRACKLET":
				opp = None
				for key in ("paired_tracklet_id", "opponent_tracklet_id", "partner_tracklet_id", "other_tracklet_id"):
					if key in row.index:
						opp = _get_str(row, key)
						if opp is not None:
							break
				gate_pair(carrier_tid, opp)

		nodes_out.append(
			{
				"node_id": nid,
				"node_type": nt,
				"span": {"start": sf, "end": ef},
				"boundary": {"is_start": bool(is_start_boundary(sf)), "is_end": bool(is_end_boundary(ef)), "window_frames": int(window)},
				"metadata": {
					"carrier_tracklet_id": carrier_tid,
					"disappearing_tracklet_id": disp_tid,
					"new_tracklet_id": new_tid,
				},
				"edges": {
					"carrier_cont_in": carrier_cont_in_e,
					"carrier_cont_out": carrier_cont_out_e,
					"merge_in": list(merge_in),
					"split_out": list(split_out),
					"group_cont_in": list(group_cont_in),
					"group_cont_out": list(group_cont_out),
				},
				"constraints_applied": {
					"enforce_0_or_2": True,
					"carrier_chain_saturation": bool((carrier_cont_in_e is not None) or (carrier_cont_out_e is not None)),
					"merge_required": bool(disp_tid is not None),
					"split_required": bool(new_tid is not None),
					"group_cont_0_or_2": bool((len(group_cont_in) + len(group_cont_out)) > 0),
					"tightening_pairs": tightening_pairs,
				},
				"forced_unused": {"value": bool(forced_unused_reason is not None), "reason": forced_unused_reason},
			}
		)

	# --- Derived "groupish overlay" nodes (implicit groupish via cap_eff>=2 on SINGLE nodes) ---
	try:
		singles = nodes_df[nodes_df["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		if len(singles) > 0:
			singles["node_id"] = singles["node_id"].astype(str)
			if "base_tracklet_id" in singles.columns:
				singles["base_tracklet_id"] = singles["base_tracklet_id"].astype(str)
			for _, rr in singles.iterrows():
				nid = str(rr["node_id"])
				cap = int(node_cap_eff.get(nid, 1))
				if cap < 2:
					continue
				sf = _safe_int(rr.get("start_frame")) if "start_frame" in rr.index else None
				ef = _safe_int(rr.get("end_frame")) if "end_frame" in rr.index else None
				groupish_overlay_nodes.append(
					{
						"node_id": nid,
						"node_type": "NodeType.SINGLE_TRACKLET",
						"base_tracklet_id": (str(rr.get("base_tracklet_id")) if "base_tracklet_id" in rr.index else None),
						"cap_eff": int(cap),
						"span": {"start": sf, "end": ef},
					}
				)
	except Exception:
		groupish_overlay_nodes = []

	payload: Dict[str, Any] = {
		"schema_version": "ilp2_group_semantics_v0.1.0",
		"summary": {
			"n_group_nodes": int((nodes_df["node_type"].astype(str) == "NodeType.GROUP_TRACKLET").sum()),
			"n_groupish_nodes": int((nodes_df["node_type"].astype(str) == "NodeType.GROUPISH_TRACKLET").sum()),
			"n_forced_unused": int(n_forced_unused),
			"n_missing_merge_required": int(n_missing_merge_required),
			"n_missing_split_required": int(n_missing_split_required),
			"n_missing_groupish_bridge": int(n_missing_groupish_bridge),
			"n_groupish_overlay_nodes": int(len(groupish_overlay_nodes)),
			"groupish_overlay_node_ids": sorted([str(x.get("node_id")) for x in groupish_overlay_nodes if x.get("node_id") is not None]),
		},
		"nodes": nodes_out,
		"overlay": {"groupish_overlay_nodes": groupish_overlay_nodes},
	}
	return payload


def _normalize_tag_key(x: Any) -> Optional[str]:
	if x is None:
		return None
	s = str(x)
	if s.startswith("tag:"):
		return s
	try:
		int(s)
		return f"tag:{s}"
	except Exception:
		return s


def _extract_tag_pings_from_constraints(constraints: Dict[str, Any] | None) -> List[Dict[str, Any]]:
	if not constraints or not isinstance(constraints, dict):
		return []
	if isinstance(constraints.get("tag_pings"), list):
		return list(constraints["tag_pings"])  # type: ignore
	if isinstance(constraints.get("identity_hints"), list):
		return list(constraints["identity_hints"])  # type: ignore
	return []


def _extract_must_link_groups_from_constraints(constraints: Dict[str, Any] | None) -> Dict[str, List[str]]:
	if not constraints or not isinstance(constraints, dict):
		return {}
	mg = constraints.get("must_link_groups")
	out: Dict[str, List[str]] = {}
	if isinstance(mg, list):
		# D2 canonical format: list of {anchor_key, tracklet_ids}
		for g in mg:
			if not isinstance(g, dict):
				continue
			ak = g.get("anchor_key")
			tk = _normalize_tag_key(ak) if isinstance(ak, str) else None
			if tk is None:
				continue
			tids = g.get("tracklet_ids", [])
			if isinstance(tids, list):
				out.setdefault(tk, []).extend(str(x) for x in tids)
	elif isinstance(mg, dict):
		# Legacy dict format: {anchor_key: [tracklet_ids]}
		for k, v in mg.items():
			tk = _normalize_tag_key(k)
			if tk is None:
				continue
			if isinstance(v, list):
				out[tk] = [str(x) for x in v]
			else:
				out[tk] = [str(v)]
	for tk in list(out.keys()):
		vals = []
		for x in out.get(tk, []):
			sx = str(x)
			if sx.strip() == "" or sx.strip().lower() == "none":
				continue
			vals.append(sx)
		out[tk] = sorted(set(vals))
	return out

def _infer_ping_miss_penalty_unscaled(*, constraints: Dict[str, Any] | None) -> float:
	"""Best-effort ping miss penalty (unscaled float, before cost_scale).

	Preference order (if present in constraints dict):
	  1) solo_ping_miss_penalty_abs
	  2) solo_ping_miss_penalty_mult * penalty_ref_edge_cost
	  3) default 50.0

	This keeps ILP2 runnable even when compile doesn't yet forward all config-derived fields.
	"""
	if not constraints or not isinstance(constraints, dict):
		return 50.0
	abs_v = constraints.get("solo_ping_miss_penalty_abs")
	if abs_v is not None:
		try:
			return float(abs_v)
		except Exception:
			pass
	mult_v = constraints.get("solo_ping_miss_penalty_mult")
	ref_v = constraints.get("penalty_ref_edge_cost")
	if mult_v is not None and ref_v is not None:
		try:
			return float(mult_v) * float(ref_v)
		except Exception:
			pass
	return 50.0

def _extract_bound_ping_records(tag_inputs: Dict[str, Any] | None) -> List[Dict[str, Any]]:
	"""Return list of bound pings with (tag_key, ping_id, node_id).

	Only pings with binding.status == "bound" and binding.chosen.node_id are returned.
	"""
	out: List[Dict[str, Any]] = []
	if not isinstance(tag_inputs, dict):
		return out
	pings = tag_inputs.get("pings")
	if not isinstance(pings, list):
		return out
	for p in pings:
		if not isinstance(p, dict):
			continue
		b = p.get("binding") or {}
		if not isinstance(b, dict):
			continue
		if str(b.get("status")) != "bound":
			continue
		ch = b.get("chosen") or {}
		if not isinstance(ch, dict):
			continue
		node_id = ch.get("node_id")
		if node_id is None:
			continue
		tag_key = p.get("tag_key")
		ping_id = p.get("ping_id")
		if tag_key is None or ping_id is None:
			continue
		source = p.get("source") or {}
		if not isinstance(source, dict):
			source = {}
		chosen_node_type = ch.get("node_type")
		match_role = ch.get("match_role")
		frame_index = p.get("frame_index")
		try:
			frame_index_i = int(frame_index) if frame_index is not None else None
		except Exception:
			frame_index_i = None
		out.append(
			{
				"tag_key": str(tag_key),
				"ping_id": str(ping_id),
				"node_id": str(node_id),
				"node_type": (str(chosen_node_type) if chosen_node_type is not None else None),
				"match_role": (str(match_role) if match_role is not None else None),
				"frame_index": frame_index_i,
				"raw_index": source.get("raw_index"),
			}
		)
	# Deterministic ordering
	out.sort(key=lambda r: (str(r["tag_key"]), str(r["ping_id"]), str(r["node_id"])))
	return out


def _build_tag_evidence_by_node(
	*,
	bound_pings: List[Dict[str, Any]],
	tag_must_link_resolved_nodes: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[str]]]:
	"""Collect graph-grounded tag evidence on nodes.

	Returns:
	  node_id -> tag_key -> sorted list of evidence source labels
	"""
	node_to_tags: Dict[str, Dict[str, List[str]]] = {}
	for bp in bound_pings:
		nid = bp.get("node_id")
		tk = bp.get("tag_key")
		if nid is None or tk is None:
			continue
		node_to_tags.setdefault(str(nid), {}).setdefault(str(tk), []).append("bound_ping")
	for tk, node_ids in tag_must_link_resolved_nodes.items():
		for nid in node_ids:
			node_to_tags.setdefault(str(nid), {}).setdefault(str(tk), []).append("resolved_must_link")
	for nid in list(node_to_tags.keys()):
		for tk in list(node_to_tags[nid].keys()):
			node_to_tags[nid][tk] = sorted(set(str(x) for x in node_to_tags[nid][tk]))
	return node_to_tags


def _build_group_tag_pair_preferences(
	*,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	in_edges_by_node: Dict[str, List[str]],
	out_edges_by_node: Dict[str, List[str]],
	edge_u: Dict[str, str],
	edge_v: Dict[str, str],
	edge_type: Dict[str, str],
	node_tag_evidence: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
	"""Build preferred local in/out pairings through GROUP/GROUPISH nodes.

	A preferred pairing is created when the source node of an inbound edge and the
	destination node of an outbound edge both carry evidence for the same tag.
	"""
	node_type_by_id: Dict[str, str] = {}
	for _, row in nodes_df.iterrows():
		node_type_by_id[str(row["node_id"])] = str(row["node_type"])

	records: List[Dict[str, Any]] = []
	seen: set[tuple[str, str, str, str]] = set()
	for _, row in nodes_df.iterrows():
		group_nid = str(row["node_id"])
		group_nt = str(row["node_type"])
		if group_nt not in ("NodeType.GROUP_TRACKLET", "NodeType.GROUPISH_TRACKLET"):
			continue
		in_edges = sorted(set(str(e) for e in (in_edges_by_node.get(group_nid, []) or [])))
		out_edges = sorted(set(str(e) for e in (out_edges_by_node.get(group_nid, []) or [])))
		for in_eid in in_edges:
			in_src = str(edge_u.get(in_eid, ""))
			if in_src in ("", "SOURCE", "SINK", group_nid):
				continue
			in_src_nt = node_type_by_id.get(in_src, "UNKNOWN")
			if in_src_nt == "UNKNOWN":
				continue
			in_tags = node_tag_evidence.get(in_src, {})
			if not in_tags:
				continue
			for out_eid in out_edges:
				out_dst = str(edge_v.get(out_eid, ""))
				if out_dst in ("", "SOURCE", "SINK", group_nid):
					continue
				out_dst_nt = node_type_by_id.get(out_dst, "UNKNOWN")
				if out_dst_nt == "UNKNOWN":
					continue
				out_tags = node_tag_evidence.get(out_dst, {})
				if not out_tags:
					continue
				shared_tags = sorted(set(in_tags.keys()) & set(out_tags.keys()))
				for tk in shared_tags:
					key = (group_nid, tk, in_eid, out_eid)
					if key in seen:
						continue
					seen.add(key)
					records.append(
						{
							"pair_id": f"{group_nid}|{tk}|{in_eid}|{out_eid}",
							"group_node_id": group_nid,
							"group_node_type": group_nt,
							"tag_key": tk,
							"in_edge_id": in_eid,
							"in_edge_type": str(edge_type.get(in_eid, "")),
							"in_node_id": in_src,
							"in_node_type": in_src_nt,
							"out_edge_id": out_eid,
							"out_edge_type": str(edge_type.get(out_eid, "")),
							"out_node_id": out_dst,
							"out_node_type": out_dst_nt,
							"in_sources": list(in_tags.get(tk, [])),
							"out_sources": list(out_tags.get(tk, [])),
							"preference_kind": "tag_consistent_pair",
						}
					)
	return sorted(records, key=lambda r: (str(r["group_node_id"]), str(r["tag_key"]), str(r["in_edge_id"]), str(r["out_edge_id"])))


def _emit_mcf_tag_inputs(
	*,
	debug_dir: Path,
	manifest: ClipManifest,
	checkpoint: str,
	nodes_df: pd.DataFrame,
	constraints: Dict[str, Any] | None,
) -> Dict[str, Any]:
	"""MCF inputs snapshot (non-behavioral).

	Writes `_debug/d3_mcf_tag_inputs.json` using the normalized TagPing schema.
	"""
	pings_raw = _extract_tag_pings_from_constraints(constraints)
	must_link = _extract_must_link_groups_from_constraints(constraints)

	# Best-effort node index for binding (by tracklet_id + span).
	#
	# Goal: allow pings to bind not only to SINGLE_TRACKLET, but also GROUP_TRACKLET and
	# GROUPISH_TRACKLET nodes so tags can anchor through group/groupish observations.
	index_rows: List[Dict[str, Any]] = []
	try:
		group_member_roles_seen: Dict[str, int] = {}

		df = nodes_df.copy()
		df["node_id"] = df["node_id"].astype(str)
		df["node_type"] = df["node_type"].astype(str)

		# Support both historical and current column names.
		start_col = "start_frame" if "start_frame" in df.columns else ("frame_start" if "frame_start" in df.columns else None)
		end_col = "end_frame" if "end_frame" in df.columns else ("frame_end" if "frame_end" in df.columns else None)
		if start_col is not None:
			df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
		if end_col is not None:
			df[end_col] = pd.to_numeric(df[end_col], errors="coerce")

		def _sf_ef(row_obj: Any) -> tuple[Optional[int], Optional[int]]:
			sf = getattr(row_obj, start_col) if start_col is not None and hasattr(row_obj, start_col) else None
			ef = getattr(row_obj, end_col) if end_col is not None and hasattr(row_obj, end_col) else None
			sfi = int(sf) if sf is not None and pd.notna(sf) else None
			efi = int(ef) if ef is not None and pd.notna(ef) else None
			return sfi, efi

		# SINGLE_TRACKLET: base_tracklet_id
		single = df[df["node_type"] == "NodeType.SINGLE_TRACKLET"].copy()
		if "base_tracklet_id" in single.columns:
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)
		for r in single.itertuples(index=False):
			sfi, efi = _sf_ef(r)
			index_rows.append(
				{
					"node_id": str(getattr(r, "node_id")),
					"node_type": "NodeType.SINGLE_TRACKLET",
					"member_tracklet_id": (str(getattr(r, "base_tracklet_id")) if hasattr(r, "base_tracklet_id") else None),
					"member_role": "base_tracklet_id",
					"frame_start": sfi,
					"frame_end": efi,
				}
			)

		# GROUP/GROUPISH: allow binding by carrier/opponent/etc metadata
		group = df[df["node_type"].isin(["NodeType.GROUP_TRACKLET", "NodeType.GROUPISH_TRACKLET"])].copy()
		for r in group.itertuples(index=False):
			sfi, efi = _sf_ef(r)
			nt = str(getattr(r, "node_type"))
			cands: List[tuple[str, Optional[str]]] = []
			for key in (
				"carrier_tracklet_id",
				"disappearing_tracklet_id",
				"new_tracklet_id",
				"paired_tracklet_id",
				"opponent_tracklet_id",
				"partner_tracklet_id",
				"other_tracklet_id",
			):
				if hasattr(r, key):
					val = getattr(r, key)
					if val is None:
						continue
					vs = str(val).strip()
					if vs == "" or vs.lower() == "none":
						continue
					group_member_roles_seen[str(key)] = int(group_member_roles_seen.get(str(key), 0)) + 1
					cands.append((key, vs))
			for role, tid in cands:
				index_rows.append(
					{
						"node_id": str(getattr(r, "node_id")),
						"node_type": nt,
						"member_tracklet_id": str(tid),
						"member_role": str(role),
						"frame_start": sfi,
						"frame_end": efi,
					}
				)
	except Exception:
		index_rows = []
		group_member_roles_seen = {}

	normalized_pings: List[Dict[str, Any]] = []
	tag_to_ping_ids: Dict[str, List[str]] = {}
	tag_summaries: Dict[str, Dict[str, Any]] = {}
	for i, p in enumerate(pings_raw):
		if not isinstance(p, dict):
			continue
		tag_key = _normalize_tag_key(p.get("tag_id") or p.get("tag") or p.get("anchor_key"))
		if tag_key is None:
			continue
		frame = p.get("frame_index", p.get("frame"))
		try:
			frame_i = int(frame) if frame is not None else None
		except Exception:
			frame_i = None
		tid = p.get("tracklet_id") or p.get("base_tracklet_id") or p.get("tid")
		tid_s = str(tid) if tid is not None else None
		ping_id = f"{tag_key}@frame:{frame_i if frame_i is not None else 'na'}#{i}"

		tag_summaries.setdefault(
			tag_key,
			{
				"n_pings": 0,
				"n_bound_pings": 0,
				"n_bound_single_pings": 0,
				"n_bound_group_pings": 0,
				"n_bound_groupish_pings": 0,
				"n_multi_candidate_pings": 0,
				"n_tiebreak_bound_pings": 0,
			},
		)
		tag_summaries[tag_key]["n_pings"] = int(tag_summaries[tag_key]["n_pings"]) + 1
		binding: Dict[str, Any] = {"status": "unbound", "candidates": [], "chosen": None, "notes": []}
		if frame_i is not None and tid_s is not None and index_rows:
			cands = []
			for b in index_rows:
				if b.get("member_tracklet_id") != tid_s:
					continue
				fs = b.get("frame_start")
				fe = b.get("frame_end")
				if fs is None or fe is None:
					continue
				if int(fs) <= frame_i <= int(fe):
					cands.append(
						{
							"node_id": str(b.get("node_id")),
							"node_type": str(b.get("node_type") or "NodeType.SINGLE_TRACKLET"),
							"span": {"start": int(fs), "end": int(fe)},
							"reason": "contains_frame",
							"match_role": str(b.get("member_role") or ""),
						}
					)
			binding["candidates"] = cands
			if len(cands) >= 1:
				# Deterministic tie-break:
				# Prefer GROUPISH -> GROUP -> SINGLE, then shortest span, then node_id.
				prio = {"NodeType.GROUPISH_TRACKLET": 0, "NodeType.GROUP_TRACKLET": 1, "NodeType.SINGLE_TRACKLET": 2}
				def _score(c: Dict[str, Any]) -> tuple[int, int, str]:
					nt = str(c.get("node_type"))
					sp = c.get("span") or {}
					try:
						span_len = int(sp.get("end")) - int(sp.get("start"))
					except Exception:
						span_len = 10**9
					return (int(prio.get(nt, 9)), int(span_len), str(c.get("node_id")))
				cands_sorted = sorted(cands, key=_score)
				chosen = cands_sorted[0]
				chosen_nt = str(chosen.get("node_type") or "NodeType.SINGLE_TRACKLET")
				chosen_reason = "unique_candidate" if len(cands) == 1 else "tiebreak"
				binding["status"] = "bound"
				binding["chosen"] = {
					"node_id": chosen["node_id"],
					"node_type": chosen_nt,
					"match_role": chosen.get("match_role"),
					"span": dict(chosen.get("span") or {}),
					"reason": chosen_reason,
				}
				binding["candidate_count"] = int(len(cands))
				binding["chosen_node_type"] = chosen_nt
				binding["chosen_match_role"] = chosen.get("match_role")
				tag_summaries[tag_key]["n_bound_pings"] = int(tag_summaries[tag_key]["n_bound_pings"]) + 1
				if len(cands) > 1:
					tag_summaries[tag_key]["n_multi_candidate_pings"] = int(tag_summaries[tag_key]["n_multi_candidate_pings"]) + 1
					tag_summaries[tag_key]["n_tiebreak_bound_pings"] = int(tag_summaries[tag_key]["n_tiebreak_bound_pings"]) + 1
				if len(cands) > 1:
					binding["notes"].append("multiple candidates contain frame; chose by priority+span")
				if chosen_nt == "NodeType.GROUPISH_TRACKLET":
					tag_summaries[tag_key]["n_bound_groupish_pings"] = int(tag_summaries[tag_key]["n_bound_groupish_pings"]) + 1
				elif chosen_nt == "NodeType.GROUP_TRACKLET":
					tag_summaries[tag_key]["n_bound_group_pings"] = int(tag_summaries[tag_key]["n_bound_group_pings"]) + 1
				else:
					tag_summaries[tag_key]["n_bound_single_pings"] = int(tag_summaries[tag_key]["n_bound_single_pings"]) + 1
			else:
				binding["candidate_count"] = 0
		else:
			binding["candidate_count"] = 0

		norm = {
			"ping_id": ping_id,
			"tag_key": tag_key,
			"frame_index": frame_i,
			"source": {"kind": "compiled_constraints", "raw_index": i, "raw": p},
			"observed": {"tracklet_id": tid_s, "node_id": None},
			"binding": binding,
		}
		normalized_pings.append(norm)
		tag_to_ping_ids.setdefault(tag_key, []).append(ping_id)

	tags: Dict[str, Any] = {}
	all_tags = sorted(set(list(tag_to_ping_ids.keys()) + list(must_link.keys())))
	for tk in all_tags:
		tag_summary = dict(tag_summaries.get(tk, {}))
		ml = sorted(set(must_link.get(tk, [])))
		tags[tk] = {
			"tag_key": tk,
			"must_link_tracklets": ml,
			"has_must_link_tracklets": bool(len(ml) > 0),
			"pings": tag_to_ping_ids.get(tk, []),
			"summary": {
				"n_pings": int(tag_summary.get("n_pings", 0)),
				"n_bound_pings": int(tag_summary.get("n_bound_pings", 0)),
				"n_bound_single_pings": int(tag_summary.get("n_bound_single_pings", 0)),
				"n_bound_group_pings": int(tag_summary.get("n_bound_group_pings", 0)),
				"n_bound_groupish_pings": int(tag_summary.get("n_bound_groupish_pings", 0)),
				"n_multi_candidate_pings": int(tag_summary.get("n_multi_candidate_pings", 0)),
				"n_tiebreak_bound_pings": int(tag_summary.get("n_tiebreak_bound_pings", 0)),
			},
		}

	n_bound_single = 0
	n_bound_group = 0
	n_bound_groupish = 0
	n_unbound = 0
	n_multi_candidate_pings = 0
	n_tiebreak_bound = 0
	for x in normalized_pings:
		if not isinstance(x, dict):
			continue
		b = x.get("binding") or {}
		if not isinstance(b, dict):
			continue
		status = str(b.get("status"))
		if status != "bound":
			n_unbound += 1
			continue
		chosen = b.get("chosen") or {}
		if not isinstance(chosen, dict):
			n_unbound += 1
			continue
		nt = str(chosen.get("node_type") or "")
		if nt == "NodeType.GROUPISH_TRACKLET":
			n_bound_groupish += 1
		elif nt == "NodeType.GROUP_TRACKLET":
			n_bound_group += 1
		else:
			n_bound_single += 1
		if int(b.get("candidate_count", 0)) > 1:
			n_multi_candidate_pings += 1
		if str(chosen.get("reason")) == "tiebreak":
			n_tiebreak_bound += 1

	payload: Dict[str, Any] = {
		"schema_version": "mcf_tag_inputs_v0.1.0",
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"checkpoint": checkpoint,
		"summary": {
			"n_tags": int(len(tags)),
			"n_pings": int(len(normalized_pings)),
			"n_bound_pings": int(sum(1 for x in normalized_pings if x.get("binding", {}).get("status") == "bound")),
			"n_bound_single": int(n_bound_single),
			"n_bound_group": int(n_bound_group),
			"n_bound_groupish": int(n_bound_groupish),
			"n_unbound": int(n_unbound),
			"n_multi_candidate_pings": int(n_multi_candidate_pings),
			"n_tiebreak_bound": int(n_tiebreak_bound),
			"group_member_roles_seen": dict(sorted(group_member_roles_seen.items(), key=lambda kv: kv[0])),
		},
		"tags": tags,
		"pings": normalized_pings,
	}
	_write_json_atomic(path=(debug_dir / "d3_mcf_tag_inputs.json"), payload=payload)
	return payload


def _emit_mcf_tag_paths(
	*,
	debug_dir: Path,
	manifest: ClipManifest,
	checkpoint: str,
	tag_inputs: Dict[str, Any] | None,
	identity_flow_by_edge: Dict[str, int],
	tag_flow_by_tag_edge: Dict[str, Dict[str, int]],
	ping_statuses_by_tag: Dict[str, List[Dict[str, Any]]] | None = None,
	) -> Path:
	"""Emit `_debug/d3_mcf_tag_paths.json` (schema v0.1.0).

	Reports current tag flow, ping satisfaction, must-link support, and local pair steering summaries.
	"""
	tags_out: Dict[str, Any] = {}
	tags = (tag_inputs or {}).get("tags", {}) if isinstance(tag_inputs, dict) else {}
	pings = (tag_inputs or {}).get("pings", []) if isinstance(tag_inputs, dict) else []

	ping_statuses_by_tag = ping_statuses_by_tag or {}
	tag_activation_info: Dict[str, Dict[str, Any]] = {}
	if isinstance(tags, dict):
		for tk, info in tags.items():
			tk_s = str(tk)
			meta_rec: Optional[Dict[str, Any]] = None
			for rec in ping_statuses_by_tag.get(tk_s, []) or []:
				if isinstance(rec, dict) and str(rec.get("record_type", "ping")) == "tag_meta":
					meta_rec = rec
					break
			if not isinstance(info, dict):
				tag_activation_info[tk_s] = {
					"is_active": False,
					"activation_reason": "no_tag_info",
					"has_must_link_tracklets": False,
					"must_link_tracklets": [],
					"must_link_resolved_nodes": [],
					"must_link_support_applicable": False,
					"must_link_support_satisfied": False,
					"must_link_support_missed": False,
					"must_link_support_nodes": [],
					"must_link_support_edges": [],
					"n_must_link_support_edges": 0,
					"n_tag_pair_preferences": 0,
					"n_tag_pair_preferences_realized": 0,
					"tag_pair_preference_ids_realized": [],
				}
				continue
			summary = info.get("summary") or {}
			if not isinstance(summary, dict):
				summary = {}
			n_bound_pings = int(summary.get("n_bound_pings", 0))
			must_link_tracklets = list(info.get("must_link_tracklets", [])) if isinstance(info.get("must_link_tracklets"), list) else []
			must_link_resolved_nodes = []
			if isinstance(meta_rec, dict):
				must_link_resolved_nodes = list(meta_rec.get("must_link_resolved_nodes", [])) if isinstance(meta_rec.get("must_link_resolved_nodes"), list) else []
				is_active = bool(int(meta_rec.get("is_active", 0)) > 0)
				activation_reason = str(meta_rec.get("activation_reason", "no_resolved_evidence"))
				ml_support_applicable = bool(int(meta_rec.get("must_link_support_applicable", 0)) > 0)
				ml_support_satisfied = bool(int(meta_rec.get("must_link_support_visit", 0)) > 0)
				ml_support_missed = bool(int(meta_rec.get("must_link_support_miss", 0)) > 0)
				ml_support_nodes = list(meta_rec.get("must_link_support_nodes", [])) if isinstance(meta_rec.get("must_link_support_nodes"), list) else []
				ml_support_edges = list(meta_rec.get("must_link_support_edges", [])) if isinstance(meta_rec.get("must_link_support_edges"), list) else []
				n_pair_prefs = int(meta_rec.get("n_tag_pair_preferences", 0))
				n_pair_realized = int(meta_rec.get("n_tag_pair_preferences_realized", 0))
				pair_ids_realized = list(meta_rec.get("tag_pair_preference_ids_realized", [])) if isinstance(meta_rec.get("tag_pair_preference_ids_realized"), list) else []
			else:
				has_resolved_must_link = False
				is_active = bool(n_bound_pings > 0)
				if n_bound_pings > 0:
					activation_reason = "bound_ping"
				else:
					activation_reason = "no_resolved_evidence"
				if has_resolved_must_link:
					is_active = True
					activation_reason = "resolved_must_link_only"
				ml_support_applicable = False
				ml_support_satisfied = False
				ml_support_missed = False
				ml_support_nodes = []
				ml_support_edges = []
				n_pair_prefs = 0
				n_pair_realized = 0
				pair_ids_realized = []
			tag_activation_info[tk_s] = {
				"is_active": is_active,
				"activation_reason": activation_reason,
				"has_must_link_tracklets": bool(len(must_link_tracklets) > 0),
				"must_link_tracklets": must_link_tracklets,
				"must_link_resolved_nodes": must_link_resolved_nodes,
				"must_link_support_applicable": ml_support_applicable,
				"must_link_support_satisfied": ml_support_satisfied,
				"must_link_support_missed": ml_support_missed,
				"must_link_support_nodes": ml_support_nodes,
				"must_link_support_edges": ml_support_edges,
				"n_tag_pair_preferences": n_pair_prefs,
				"n_tag_pair_preferences_realized": n_pair_realized,
				"tag_pair_preference_ids_realized": pair_ids_realized,
			}

	bound_nodes_by_tag: Dict[str, List[str]] = {}
	bound_node_types_by_tag: Dict[str, Dict[str, int]] = {}
	if isinstance(pings, list):
		for p in pings:
			if not isinstance(p, dict):
				continue
			tk = str(p.get("tag_key"))
			chosen = (p.get("binding") or {}).get("chosen") or {}
			nid = chosen.get("node_id")
			nt = chosen.get("node_type")
			if tk and nid:
				bound_nodes_by_tag.setdefault(tk, []).append(str(nid))
			if tk and nt:
				d = bound_node_types_by_tag.setdefault(tk, {})
				nt_s = str(nt)
				d[nt_s] = int(d.get(nt_s, 0)) + 1

	# Ping enforcement summaries (bound pings only; solver-provided)
	ping_statuses_by_tag = ping_statuses_by_tag or {}
	n_pings_enforced_bound = 0
	n_pings_satisfied = 0
	n_pings_missed = 0
	n_bound_single_pings_satisfied = 0
	n_bound_group_pings_satisfied = 0
	n_bound_groupish_pings_satisfied = 0
	n_tags_with_must_link_support_applicable = 0
	n_tags_with_must_link_support_satisfied = 0
	n_tags_with_must_link_support_missed = 0
	n_tag_pair_preferences_total = 0
	n_tag_pair_preferences_realized_total = 0
	for _tk, sts in ping_statuses_by_tag.items():
		if not isinstance(sts, list):
			continue
		for s in sts:
			if not isinstance(s, dict):
				continue
			if str(s.get("record_type", "ping")) != "ping":
				continue
			n_pings_enforced_bound += 1
			node_type = str(s.get("node_type") or "")
			if int(s.get("visit", 0)) > 0:
				n_pings_satisfied += 1
				if node_type == "NodeType.GROUPISH_TRACKLET":
					n_bound_groupish_pings_satisfied += 1
				elif node_type == "NodeType.GROUP_TRACKLET":
					n_bound_group_pings_satisfied += 1
				else:
					n_bound_single_pings_satisfied += 1
			if int(s.get("miss", 0)) > 0:
				n_pings_missed += 1

	n_active_tags = 0
	n_inactive_tags = 0
	n_tags_with_nonzero_flow = 0
	for tk in sorted(tags.keys()):
		info = tags.get(tk, {})
		pids = info.get("pings", []) if isinstance(info, dict) else []
		activation = tag_activation_info.get(str(tk), {})
		is_active = bool(activation.get("is_active", False))
		if is_active:
			n_active_tags += 1
		else:
			n_inactive_tags += 1
		if bool(activation.get("must_link_support_applicable", False)):
			n_tags_with_must_link_support_applicable += 1
		if bool(activation.get("must_link_support_satisfied", False)):
			n_tags_with_must_link_support_satisfied += 1
		if bool(activation.get("must_link_support_missed", False)):
			n_tags_with_must_link_support_missed += 1
		n_tag_pair_preferences_total += int(activation.get("n_tag_pair_preferences", 0))
		n_tag_pair_preferences_realized_total += int(activation.get("n_tag_pair_preferences_realized", 0))
		status = "inactive" if not pids else "present"
		notes = ["mcf3b_group_pair_steering"]
		tag_edges = tag_flow_by_tag_edge.get(tk, {})
		selected_edges = []
		for eid, fval in sorted(tag_edges.items()):
			if int(fval) <= 0:
				continue
			selected_edges.append({"edge_id": str(eid), "identity_flow": int(identity_flow_by_edge.get(str(eid), 0)), "tag_flow": int(fval)})
		if len(selected_edges) > 0:
			n_tags_with_nonzero_flow += 1
		ping_statuses = ping_statuses_by_tag.get(tk, [])
		n_satisfied_tag = int(sum(1 for s in ping_statuses if isinstance(s, dict) and str(s.get("record_type", "ping")) == "ping" and int(s.get("visit", 0)) > 0))
		n_missed_tag = int(sum(1 for s in ping_statuses if isinstance(s, dict) and str(s.get("record_type", "ping")) == "ping" and int(s.get("miss", 0)) > 0))
		tags_out[tk] = {
			"status": status,
			"is_active": is_active,
			"activation_reason": activation.get("activation_reason"),
			"pings": list(pids) if isinstance(pids, list) else [],
			"bound_nodes": sorted(set(bound_nodes_by_tag.get(tk, []))),
			"bound_node_types": dict(sorted((bound_node_types_by_tag.get(tk, {}) or {}).items(), key=lambda kv: kv[0])),
			"must_link_tracklets": list(activation.get("must_link_tracklets", [])),
			"must_link_resolved_nodes": list(activation.get("must_link_resolved_nodes", [])),
			"must_link_support_applicable": bool(activation.get("must_link_support_applicable", False)),
			"must_link_support_satisfied": bool(activation.get("must_link_support_satisfied", False)),
			"must_link_support_missed": bool(activation.get("must_link_support_missed", False)),
			"must_link_support_nodes": list(activation.get("must_link_support_nodes", [])),
			"must_link_support_edges": list(activation.get("must_link_support_edges", [])),
			"n_tag_pair_preferences": int(activation.get("n_tag_pair_preferences", 0)),
			"n_tag_pair_preferences_realized": int(activation.get("n_tag_pair_preferences_realized", 0)),
			"tag_pair_preference_ids_realized": list(activation.get("tag_pair_preference_ids_realized", [])),
			"selected_edges": selected_edges,
			"n_selected_edges": int(len(selected_edges)),
			"visited_nodes": [],
			"notes": notes,
			"ping_statuses": ping_statuses,
			"summary": {
				"n_bound_pings": int(len(bound_nodes_by_tag.get(tk, []))),
				"n_satisfied_pings": int(n_satisfied_tag),
				"n_missed_pings": int(n_missed_tag),
				"n_must_link_resolved_nodes": int(len(activation.get("must_link_resolved_nodes", []))),
				"n_must_link_support_edges": int(len(activation.get("must_link_support_edges", []))),
				"n_tag_pair_preferences": int(activation.get("n_tag_pair_preferences", 0)),
				"n_tag_pair_preferences_realized": int(activation.get("n_tag_pair_preferences_realized", 0)),
			},
		}

	n_nonzero = int(
			sum(
				1
				for tk in tag_flow_by_tag_edge
				for _eid, v in (tag_flow_by_tag_edge.get(tk, {}) or {}).items()
				if int(v) > 0
			)
		)
	payload: Dict[str, Any] = {
		"schema_version": "mcf_tag_paths_v0.1.0",
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"checkpoint": checkpoint,
		"solver_impl": _SOLVER_IMPL,
		"solver_module": _SOLVER_MODULE,
		"solver_version": _SOLVER_VERSION,
		"summary": {
			"mcf_checkpoint": "MCF-3b",
			"n_tag_flow_nonzero_edges_total": n_nonzero,
			"n_active_tags": int(n_active_tags),
			"n_inactive_tags": int(n_inactive_tags),
			"n_tags_with_nonzero_flow": int(n_tags_with_nonzero_flow),
			"n_tags_with_must_link_support_applicable": int(n_tags_with_must_link_support_applicable),
			"n_tags_with_must_link_support_satisfied": int(n_tags_with_must_link_support_satisfied),
			"n_tags_with_must_link_support_missed": int(n_tags_with_must_link_support_missed),
			"n_tag_pair_preferences_total": int(n_tag_pair_preferences_total),
			"n_tag_pair_preferences_realized_total": int(n_tag_pair_preferences_realized_total),
			"n_bound_single_pings_satisfied": int(n_bound_single_pings_satisfied),
			"n_bound_group_pings_satisfied": int(n_bound_group_pings_satisfied),
			"n_bound_groupish_pings_satisfied": int(n_bound_groupish_pings_satisfied),
			"n_tags": int(len(tags_out)),
			"n_pings": int(len(pings) if isinstance(pings, list) else 0),
			"n_bound_pings": int(sum(1 for x in (pings or []) if isinstance(x, dict) and (x.get("binding") or {}).get("status") == "bound")),
			"n_pings_enforced_bound": int(n_pings_enforced_bound),
			"n_pings_satisfied": int(n_pings_satisfied),
			"n_pings_missed": int(n_pings_missed),
			"n_edges_used_by_identity": int(sum(1 for _, v in identity_flow_by_edge.items() if int(v) > 0)),
		},
		"tags": tags_out,
	}
	out_path = debug_dir / "d3_mcf_tag_paths.json"
	_write_json_atomic(path=out_path, payload=payload)
	return out_path


def _solve_identity_ilp2_identity_only(
	*,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	costs_df: pd.DataFrame,
	constraints: Dict[str, Any] | None,
	debug_dir: Path | None,
	emit_transparency: bool,
	unexplained_tracklet_penalty: float | None,
	group_boundary_window_frames: int,
	tag_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[ILPResult, Dict[str, Dict[str, int]], Dict[str, List[Dict[str, Any]]]]:
	"""ILP2-A identity-only solver (standalone; no ILP1 delegation).

	Model:
	  - Edge flow variables x_e (cap=1) or f_e (0..cap) for cap>1
	  - Node conservation: sum_in == sum_out (except SOURCE/SINK)
	  - Node capacity: sum_in <= cap(node)
	  - SOURCE/SINK balance + enforce at least one path when graph is non-empty
	  - cannot-link pruning on CONTINUE edges (if constraints provided)
	  - Edge costs from D2 (scaled to int)
	  - A1: group/groupish semantics constraints (applied via _apply_group_semantics_constraints)
	  - A2: explain-or-penalize coverage pressure per base_tracklet_id
	  - MCF overlay uses binary per-tag threads constrained by identity flow capacity
	  - Sparse AprilTag pings may bind to SINGLE / GROUP / GROUPISH nodes and remain
	    the strongest local identity anchors
	  - Active tags remain a constraint layer over identity flow
	  - Local tag-consistent pairings through GROUP/GROUPISH nodes are rewarded so
	    ambiguous identity edge choices become tag-consistent when evidence supports them
	  - Resolved must-link-supported nodes contribute a soft support preference
	"""
	# Defensive: this checkpoint ignores tags.
	_ = emit_transparency

	# Normalize + validate inputs.
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	costs = costs_df.copy()

	# Cost scale MUST be defined before any downstream logic references it
	# (e.g., MCF miss penalties, explain-or-penalize penalties).
	#
	# We keep this deterministic and consistent with the rest of ILP2.
	scale = _cost_scale_for(None)
	if not isinstance(scale, int) or scale <= 0:
		# absolute safety fallback; should never happen
		scale = 1000

	# --- Tag inputs (MCF) pre-parse + scalar precompute ---
	# Ordering requirement: do not reference model / adjacency / tag vars before they exist.
	tag_flow_vars: Dict[str, Dict[str, Any]] = {}
	tag_flow_by_tag_edge: Dict[str, Dict[str, int]] = {}
	ping_statuses_by_tag: Dict[str, List[Dict[str, Any]]] = {}

	tag_keys: List[str] = []
	if isinstance(tag_inputs, dict):
		tags_obj = tag_inputs.get("tags", {})
		if isinstance(tags_obj, dict):
			for tk_raw, info in tags_obj.items():
				tag_keys.append(str(tk_raw))
	tag_keys = sorted(set(tag_keys))

	# MCF-2a scalar precompute only (constraints are added later after model/adjacency/tag vars exist).
	bound_pings = _extract_bound_ping_records(tag_inputs)
	miss_penalty_unscaled = _infer_ping_miss_penalty_unscaled(constraints=constraints)
	try:
		miss_penalty_scaled = int(round(float(miss_penalty_unscaled) * float(scale)))
	except Exception:
		# deterministic fallback
		miss_penalty_scaled = int(round(50.0 * float(scale)))

	# CP17: extract cross-camera corroboration evidence for per-tag penalty boost
	cross_camera = (constraints or {}).get("cross_camera_evidence", {})
	corroborated_tags: Dict[str, Any] = cross_camera.get("corroborated_tags", {})
	corroboration_multiplier = float(
		(constraints or {}).get("corroboration_miss_multiplier", 10.0)
	)

	# CP20: read histogram cost_modifiers (logged for audit; direct ILP integration
	# deferred to CP21 when both histogram + spatial signals are available)
	cost_modifiers = (constraints or {}).get("cost_modifiers", {})
	if cost_modifiers:
		n_pairs = len(cost_modifiers.get("cross_camera_pairs", []))
		logger.info("CP20: cost_modifiers present with {} cross-camera pairs", n_pairs)

	# These are created later (after model exists) in the MCF-2a block.
	visit_vars: Dict[str, Any] = {}
	miss_vars: Dict[str, Any] = {}
	miss_var_tag_key: Dict[str, str] = {}  # ping_id -> tag_key (for per-tag penalty)
	must_link_visit_vars: Dict[str, Any] = {}
	must_link_miss_vars: Dict[str, Any] = {}
	tag_pair_pref_vars: Dict[str, Any] = {}
	tag_pair_preference_records: List[Dict[str, Any]] = []

	# Resolve must-link tracklet evidence against the current graph so activation can be
	# based on graph-grounded evidence rather than raw strings.
	tag_active_const: Dict[str, int] = {}
	tag_activation_reason: Dict[str, str] = {}
	bound_ping_ids_by_tag: Dict[str, List[str]] = {}
	bound_ping_nodes_by_tag: Dict[str, List[str]] = {}
	tag_must_link_tracklets: Dict[str, List[str]] = {}
	tag_must_link_resolved_nodes: Dict[str, List[str]] = {}
	if isinstance(tag_inputs, dict):
		tags_obj = tag_inputs.get("tags", {})
		if isinstance(tags_obj, dict):
			# Build best-effort node membership index keyed by tracklet id.
			tracklet_to_nodes: Dict[str, List[str]] = {}
			for _, nr in nodes.iterrows():
				nid = str(nr["node_id"]) if "node_id" in nr.index else None
				if nid is None:
					continue
				for key in (
					"base_tracklet_id",
					"carrier_tracklet_id",
					"disappearing_tracklet_id",
					"new_tracklet_id",
					"paired_tracklet_id",
					"opponent_tracklet_id",
					"partner_tracklet_id",
					"other_tracklet_id",
				):
					if key not in nr.index:
						continue
					val = nr.get(key)
					if val is None:
						continue
					sv = str(val).strip()
					if sv == "" or sv.lower() == "none":
						continue
					tracklet_to_nodes.setdefault(sv, []).append(nid)
			for tk, info in tags_obj.items():
				ml = list(info.get("must_link_tracklets", [])) if isinstance(info, dict) and isinstance(info.get("must_link_tracklets"), list) else []
				ml_sorted = sorted(set(str(x) for x in ml if str(x).strip() != "" and str(x).strip().lower() != "none"))
				tag_must_link_tracklets[str(tk)] = ml_sorted
				resolved: List[str] = []
				for tid in ml_sorted:
					resolved.extend(tracklet_to_nodes.get(str(tid), []))
				tag_must_link_resolved_nodes[str(tk)] = sorted(set(str(x) for x in resolved))

	for bp in bound_pings:
		tk = str(bp.get("tag_key"))
		pid = bp.get("ping_id")
		nid = bp.get("node_id")
		if pid is not None:
			bound_ping_ids_by_tag.setdefault(tk, []).append(str(pid))
		if nid is not None:
			bound_ping_nodes_by_tag.setdefault(tk, []).append(str(nid))
	for tk in list(bound_ping_ids_by_tag.keys()):
		bound_ping_ids_by_tag[tk] = sorted(set(bound_ping_ids_by_tag[tk]))
	for tk in list(bound_ping_nodes_by_tag.keys()):
		bound_ping_nodes_by_tag[tk] = sorted(set(bound_ping_nodes_by_tag[tk]))

	for tk in tag_keys:
		n_bound = int(len(bound_ping_ids_by_tag.get(tk, [])))
		n_resolved_ml = int(len(tag_must_link_resolved_nodes.get(tk, [])))
		is_active = bool((n_bound > 0) or (n_resolved_ml > 0))
		tag_active_const[tk] = 1 if is_active else 0
		if n_bound > 0 and n_resolved_ml > 0:
			tag_activation_reason[tk] = "bound_ping_and_resolved_must_link"
		elif n_bound > 0:
			tag_activation_reason[tk] = "bound_ping"
		elif n_resolved_ml > 0:
			tag_activation_reason[tk] = "resolved_must_link_only"
		else:
			tag_activation_reason[tk] = "no_resolved_evidence"

	# Penalty scales for soft evidence support.
	must_link_penalty_unscaled = float(miss_penalty_unscaled) * 2.0
	try:
		must_link_penalty_scaled = int(round(float(must_link_penalty_unscaled) * float(scale)))
	except Exception:
		must_link_penalty_scaled = int(max(1, 2 * int(miss_penalty_scaled)))

	try:
		tag_pair_reward_scaled = int(max(1, miss_penalty_scaled))
	except Exception:
		tag_pair_reward_scaled = 50000

	_require_columns(nodes, name="d1_graph_nodes", cols=["node_id", "node_type", "capacity"])
	_require_columns(edges, name="d1_graph_edges", cols=["edge_id", "u", "v", "edge_type", "capacity"])
	_require_columns(costs, name="d2_edge_costs", cols=["edge_id", "total_cost"])

	for c in ("node_id", "node_type"):
		nodes[c] = nodes[c].astype(str)
	for c in ("edge_id", "u", "v", "edge_type"):
		edges[c] = edges[c].astype(str)
	costs["edge_id"] = costs["edge_id"].astype(str)

	edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	# Capacity bookkeeping
	edge_cap_eff = _compute_edge_capacity_eff(edges)
	node_cap_eff = _compute_node_capacity_eff(nodes, edges, edge_cap_eff=edge_cap_eff)

	# Costs
	scaled_cost, rounding_stats = _scaled_costs(costs, scale=scale)

	# Identify terminals
	source_id = _find_unique_node_id(nodes, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes, node_type="NodeType.SINK")

	# Build adjacency lists
	in_edges_by_node: Dict[str, List[str]] = {}
	out_edges_by_node: Dict[str, List[str]] = {}
	edge_u: Dict[str, str] = {}
	edge_v: Dict[str, str] = {}
	edge_type: Dict[str, str] = {}

	for _, r in edges.iterrows():
		eid = str(r["edge_id"])
		u = str(r["u"])
		v = str(r["v"])
		edge_u[eid] = u
		edge_v[eid] = v
		edge_type[eid] = str(r["edge_type"])
		out_edges_by_node.setdefault(u, []).append(eid)
		in_edges_by_node.setdefault(v, []).append(eid)

	# Precompute SINGLE<->SINGLE CONTINUE edges keyed by base_tracklet_id pairs (for group tightening)
	continue_edges_by_base_pair: Dict[tuple[str, str], List[str]] = {}
	node_to_base: Dict[str, str] = {}
	if "base_tracklet_id" in nodes.columns:
		# node_to_base populated here (best-effort); reused below for cannot-link
		tmp = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		if "base_tracklet_id" in tmp.columns:
			tmp["base_tracklet_id"] = tmp["base_tracklet_id"].astype(str)
			for _, rr in tmp.iterrows():
				node_to_base[str(rr["node_id"])] = str(rr["base_tracklet_id"])
	for _, er in edges.iterrows():
		eid = str(er["edge_id"])
		if str(er["edge_type"]) != "EdgeType.CONTINUE":
			continue
		u2 = str(er["u"])
		v2 = str(er["v"])
		bu = node_to_base.get(u2)
		bv = node_to_base.get(v2)
		if bu is None or bv is None:
			continue
		k = (bu, bv) if bu < bv else (bv, bu)
		continue_edges_by_base_pair.setdefault(k, []).append(eid)
	for k in list(continue_edges_by_base_pair.keys()):
		continue_edges_by_base_pair[k] = sorted(set(continue_edges_by_base_pair[k]))

	# OR-Tools model
	model = cp_model.CpModel()

	flow_var: Dict[str, Any] = {}
	used_var: Dict[str, Any] = {}
	explained_var_by_tid: Dict[str, Any] = {}
	edge_ids_sorted: List[str] = []

	edge_ids_sorted = sorted(edges["edge_id"].astype(str).tolist())
	for eid in edge_ids_sorted:
		cap = int(edge_cap_eff.get(eid, 1))
		if cap <= 1:
			x = model.NewBoolVar(f"x[{eid}]")
			flow_var[eid] = x
			used_var[eid] = x
		else:
			f = model.NewIntVar(0, cap, f"f[{eid}]")
			u = model.NewBoolVar(f"used[{eid}]")
			# Link f to used
			model.Add(f >= u)
			model.Add(f <= cap * u)
			flow_var[eid] = f
			used_var[eid] = u

	# Node constraints: conservation + capacity
	enforced_min_one_path = False
	track_node_types = {"NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET", "NodeType.GROUPISH_TRACKLET"}
	has_track_nodes = bool((nodes["node_type"].astype(str).isin(track_node_types)).any())
	has_source_out = source_id in out_edges_by_node and len(out_edges_by_node[source_id]) > 0

	for _, nr in nodes.iterrows():
		nid = str(nr["node_id"])
		if nid in (source_id, sink_id):
			continue
		ins = in_edges_by_node.get(nid, [])
		outs = out_edges_by_node.get(nid, [])
		in_sum = sum(flow_var[eid] for eid in ins) if ins else 0
		out_sum = sum(flow_var[eid] for eid in outs) if outs else 0
		model.Add(in_sum == out_sum)
		cap_n = int(node_cap_eff.get(nid, 1))
		model.Add(in_sum <= cap_n)
		model.Add(out_sum <= cap_n)

	# SOURCE/SINK balance
	sum_out_source = sum(flow_var[eid] for eid in out_edges_by_node.get(source_id, [])) if has_source_out else 0
	sum_in_sink = sum(flow_var[eid] for eid in in_edges_by_node.get(sink_id, [])) if in_edges_by_node.get(sink_id) else 0
	model.Add(sum_out_source == sum_in_sink)

	# Prevent trivial all-zero solution when graph is non-empty.
	if has_track_nodes and has_source_out:
		model.Add(sum_out_source >= 1)
		enforced_min_one_path = True

	# cannot-link pruning (identity-only)
	cannot_pairs = []
	if constraints and isinstance(constraints, dict) and isinstance(constraints.get("cannot_link_pairs"), list):
		cannot_pairs = list(constraints["cannot_link_pairs"])  # type: ignore
	# node_to_base populated above (best-effort); keep defensive fill if empty
	if "base_tracklet_id" in nodes.columns and not node_to_base:
		m = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		m["base_tracklet_id"] = m["base_tracklet_id"].astype(str)
		for _, rr in m.iterrows():
			node_to_base[str(rr["node_id"])] = str(rr["base_tracklet_id"])

	for pair in cannot_pairs:
		if not isinstance(pair, (list, tuple)) or len(pair) != 2:
			continue
		a = str(pair[0])
		b = str(pair[1])
		for eid in edges["edge_id"].astype(str).tolist():
			if edge_type.get(eid) != "EdgeType.CONTINUE":
				continue
			u = edge_u.get(eid)
			v = edge_v.get(eid)
			if u is None or v is None:
				continue
			bu = node_to_base.get(str(u))
			bv = node_to_base.get(str(v))
			if bu is None or bv is None:
				continue
			if (bu == a and bv == b) or (bu == b and bv == a):
				model.Add(used_var[eid] == 0)

	# --- ILP2-A1: Group + Groupish semantics constraints ---
	group_semantics_debug: Optional[Dict[str, Any]] = None
	try:
		group_semantics_debug = _apply_group_semantics_constraints(
			model=model,
			nodes_df=nodes,
			edges_df=edges,
			edge_u=edge_u,
			edge_v=edge_v,
			edge_type=edge_type,
			in_edges_by_node=in_edges_by_node,
			out_edges_by_node=out_edges_by_node,
			flow_var=flow_var,
			used_var=used_var,
			edge_cap_eff=edge_cap_eff,
			node_cap_eff=node_cap_eff,
			group_boundary_window_frames=int(group_boundary_window_frames),
			continue_edges_by_base_pair=continue_edges_by_base_pair,
			node_to_base=node_to_base,
		)
		if debug_dir is not None:
			_emit_ilp2_group_semantics_debug(debug_dir=Path(debug_dir), payload=group_semantics_debug)
	except Exception:
		group_semantics_debug = None

	# Build graph-grounded tag evidence on nodes, then derive preferred local tag-consistent
	# in/out pairings through GROUP/GROUPISH nodes. These preferences will steer the
	# underlying identity edge choices, not just the overlay path reporting.
	node_tag_evidence = _build_tag_evidence_by_node(
		bound_pings=bound_pings,
		tag_must_link_resolved_nodes=tag_must_link_resolved_nodes,
	)
	tag_pair_preference_records = _build_group_tag_pair_preferences(
		nodes_df=nodes,
		edges_df=edges,
		in_edges_by_node=in_edges_by_node,
		out_edges_by_node=out_edges_by_node,
		edge_u=edge_u,
		edge_v=edge_v,
		edge_type=edge_type,
		node_tag_evidence=node_tag_evidence,
	)
	for rec in tag_pair_preference_records:
		pid = str(rec["pair_id"])
		in_eid = str(rec["in_edge_id"])
		out_eid = str(rec["out_edge_id"])
		if in_eid not in used_var or out_eid not in used_var:
			continue
		pv = model.NewBoolVar(f"pair_pref[{pid}]")
		model.Add(pv <= used_var[in_eid])
		model.Add(pv <= used_var[out_eid])
		model.Add(pv >= used_var[in_eid] + used_var[out_eid] - 1)
		tag_pair_pref_vars[pid] = pv

	tag_pair_debug_payload: Optional[Dict[str, Any]] = None
	if tag_pair_preference_records:
		tag_pair_debug_payload = {
			"schema_version": "ilp2_tag_pair_preferences_v0.1.0",
			"summary": {
				"n_pair_preferences_total": int(len(tag_pair_preference_records)),
			},
			"pair_preferences": [dict(r) for r in tag_pair_preference_records],
		}

	# --- MCF-1: tag flow variables (gated only) ---
	# Create tf[tag, edge] in {0,1} and:
	#  - couple tag capacity to identity flow capacity: sum_t tf[tag, e] <= identity_flow(e)
	#  - enforce per-tag conservation to form a single-thread path from SOURCE to SINK
	#
	# Create vars only if tags are present.
	if tag_keys:
		tag_index = {tk: i for i, tk in enumerate(tag_keys)}
		edge_index = {eid: i for i, eid in enumerate(edge_ids_sorted)}
		for tk in tag_keys:
			tag_flow_vars[tk] = {}
			tag_flow_by_tag_edge[tk] = {}
			ki = int(tag_index[tk])
			for eid in edge_ids_sorted:
				ei = int(edge_index[eid])
				tf = model.NewBoolVar(f"tf[k{ki},e{ei}]")
				# Gating: tag flow can only ride on used identity edges
				model.Add(tf <= used_var[eid])
				tag_flow_vars[tk][eid] = tf

		# Shared edge capacity across tags: sum_t tf[tag, e] <= identity_flow(e)
		for eid in edge_ids_sorted:
			model.Add(sum(tag_flow_vars[tk][eid] for tk in tag_keys) <= flow_var[eid])

		# Per-tag conservation: retain the current overlay continuity scaffold, but let
		# local tag-consistent pair preferences steer the underlying identity choices.
		for tk in tag_keys:
			active = int(tag_active_const.get(tk, 0))
			# SOURCE/SINK participation remains optional up to the active gate.
			out_s = out_edges_by_node.get(source_id, []) or []
			in_t = in_edges_by_node.get(sink_id, []) or []
			model.Add(sum(tag_flow_vars[tk][eid] for eid in out_s) <= active)
			model.Add(sum(tag_flow_vars[tk][eid] for eid in in_t) <= active)
			if active <= 0:
				continue
			# Conservation + anti-branching at all intermediate nodes
			for _, nr in nodes.iterrows():
				nid = str(nr["node_id"])
				if nid in (source_id, sink_id):
					continue
				ins = in_edges_by_node.get(nid, []) or []
				outs = out_edges_by_node.get(nid, []) or []
				in_sum = sum(tag_flow_vars[tk][eid] for eid in ins) if ins else 0
				out_sum = sum(tag_flow_vars[tk][eid] for eid in outs) if outs else 0
				model.Add(in_sum == out_sum)
				# Single-tag thread: cannot split/merge for a single tag
				model.Add(in_sum <= 1)

	# --- MCF-2a: bound ping enforcement with slack miss ---
	# For each bound ping p on node n:
	#   visit[p] + miss[p] == 1
	#   sum_incident_tf[tag, e] >= visit[p]
	# Objective: + miss_penalty * miss[p]  (added later)
	#
	# Notes:
	# - Uses tag flow vars (not identity flow) so single-thread semantics apply naturally.
	# - Uses incident edges of the *bound* node (SINGLE/GROUP/GROUPISH).
	if bound_pings and tag_keys and tag_flow_vars:
		for bp in bound_pings:
			tk = bp["tag_key"]
			pid = bp["ping_id"]
			nid = bp["node_id"]
			if tk not in tag_flow_vars:
				# Tag key exists in pings but tag vars were not constructed (unexpected); skip defensively.
				continue

			visit = model.NewBoolVar(f"visit[{pid}]")
			miss = model.NewBoolVar(f"miss[{pid}]")
			model.Add(visit + miss == 1)

			inc = (in_edges_by_node.get(nid, []) or []) + (out_edges_by_node.get(nid, []) or [])
			inc = sorted(set(str(e) for e in inc))
			if inc:
				model.Add(sum(tag_flow_vars[tk][eid] for eid in inc) >= visit)
			else:
				# If node has no incident edges, it can never be visited.
				model.Add(visit == 0)
				model.Add(miss == 1)

			visit_vars[pid] = visit
			miss_vars[pid] = miss
			miss_var_tag_key[pid] = tk

	# --- MCF-3a: soft must-link support from resolved nodes ---
	# If a tag has resolved must-link-supported nodes in the current graph, prefer that
	# the tag thread touches at least one such node, but do not make the whole solve fail
	# if continuity/support cannot be justified.
	must_link_support_nodes_by_tag: Dict[str, List[str]] = {}
	must_link_support_edges_by_tag: Dict[str, List[str]] = {}
	if tag_keys and tag_flow_vars:
		for tk in tag_keys:
			if int(tag_active_const.get(tk, 0)) <= 0:
				continue
			support_nodes = sorted(set(str(n) for n in tag_must_link_resolved_nodes.get(tk, []) if str(n).strip() != ""))
			must_link_support_nodes_by_tag[tk] = support_nodes
			support_edges: List[str] = []
			for nid in support_nodes:
				support_edges.extend(in_edges_by_node.get(nid, []) or [])
				support_edges.extend(out_edges_by_node.get(nid, []) or [])
			support_edges = sorted(set(str(e) for e in support_edges))
			must_link_support_edges_by_tag[tk] = support_edges
			if support_nodes:
				ml_visit = model.NewBoolVar(f"ml_visit[{tk}]")
				ml_miss = model.NewBoolVar(f"ml_miss[{tk}]")
				model.Add(ml_visit + ml_miss == 1)
				if support_edges:
					model.Add(sum(tag_flow_vars[tk][eid] for eid in support_edges) >= ml_visit)
				else:
					model.Add(ml_visit == 0)
					model.Add(ml_miss == 1)
				must_link_visit_vars[tk] = ml_visit
				must_link_miss_vars[tk] = ml_miss

	# --- ILP2-A2: Explain-or-penalize (coverage) ---
	# If a base_tracklet_id appears in SINGLE_TRACKLET nodes, either explain it (use any incident edge)
	# or pay a penalty once per base_tracklet_id.
	explain_debug: Optional[Dict[str, Any]] = None
	penalty_scaled: Optional[int] = None
	unexplained_var_by_tid: Dict[str, Any] = {}

	if unexplained_tracklet_penalty is not None:
		try:
			penalty_scaled = int(round(float(unexplained_tracklet_penalty) * float(scale)))
		except Exception:
			penalty_scaled = None

		# Universe of base tracklet ids (deterministic)
		base_tids: List[str] = []
		base_tid_to_nodes: Dict[str, List[str]] = {}
		if "node_type" in nodes.columns and "base_tracklet_id" in nodes.columns:
			single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			if len(single) > 0:
				single["node_id"] = single["node_id"].astype(str)
				single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)
				for _, rr in single.iterrows():
					tid = str(rr["base_tracklet_id"]).strip()
					if tid == "" or tid.lower() == "none":
						continue
					base_tid_to_nodes.setdefault(tid, []).append(str(rr["node_id"]))
				base_tids = sorted(base_tid_to_nodes.keys())

		# Build incident edge lists per tid using node->incident edges
		# (use used_var[e] for cap==1 and cap>1 consistently)
		tid_records: List[Dict[str, Any]] = []

		for tid in base_tids:
			nids = sorted(set(base_tid_to_nodes.get(tid, [])))
			incident_edges: List[str] = []
			for nid in nids:
				incident_edges.extend(in_edges_by_node.get(nid, []))
				incident_edges.extend(out_edges_by_node.get(nid, []))
			incident_edges = sorted(set(str(e) for e in incident_edges))

			expl = model.NewBoolVar(f"explained[{tid}]")
			unexpl = model.NewBoolVar(f"unexplained[{tid}]")
			# unexpl = 1 - expl
			model.Add(expl + unexpl == 1)

			if incident_edges:
				# explained >= used_e for all incident edges
				for eid in incident_edges:
					if eid in used_var:
						model.Add(expl >= used_var[eid])
				# sum(used_e) >= explained
				model.Add(sum(used_var[eid] for eid in incident_edges if eid in used_var) >= expl)
			else:
				# No incident edges: cannot be explained
				model.Add(expl == 0)
				model.Add(unexpl == 1)

			explained_var_by_tid[tid] = expl
			unexplained_var_by_tid[tid] = unexpl

			tid_records.append(
				{
					"base_tracklet_id": tid,
					"n_single_nodes": int(len(nids)),
					"n_incident_edges": int(len(incident_edges)),
					"note": ("no_incident_edges" if not incident_edges else None),
				}
			)

		# Attach penalty to objective terms later (below) by emitting penalty vars now.
		explain_debug = {
			"schema_version": "ilp2_explain_or_penalize_v0.1.0",
			"summary": {
				"n_tracklets_total": int(len(base_tids)),
				"penalty_unscaled": float(unexplained_tracklet_penalty),
				"cost_scale": int(scale),
				"penalty_scaled": int(penalty_scaled or 0),
			},
			"tracklets": tid_records,
		}

	# Objective: minimize scaled costs * flow + penalties
	terms: List[Any] = []
	for eid in edge_ids_sorted:
		ci = int(scaled_cost.get(str(eid), 0))
		if ci == 0:
			continue
		terms.append(ci * flow_var[eid])

	# Explain-or-penalize penalty terms
	if unexplained_tracklet_penalty is not None and penalty_scaled is not None and penalty_scaled > 0:
		for tid, uvar in unexplained_var_by_tid.items():
			terms.append(int(penalty_scaled) * uvar)

	# MCF-2a: miss penalties (soft enforcement)
	# CP17: corroborated tags get boosted miss penalty
	if bound_pings and miss_vars and miss_penalty_scaled > 0:
		for pid, mv in miss_vars.items():
			tk = miss_var_tag_key.get(pid, "")
			if tk in corroborated_tags:
				terms.append(int(miss_penalty_scaled * corroboration_multiplier) * mv)
			else:
				terms.append(int(miss_penalty_scaled) * mv)

	# MCF-3a: must-link support miss penalties (soft behavioral preference)
	# CP17: corroborated tags get boosted must-link penalty
	if must_link_miss_vars and must_link_penalty_scaled > 0:
		for tk, mv in must_link_miss_vars.items():
			if tk in corroborated_tags:
				terms.append(int(must_link_penalty_scaled * corroboration_multiplier) * mv)
			else:
				terms.append(int(must_link_penalty_scaled) * mv)

	# MCF-3b: reward local tag-consistent pair realization through GROUP/GROUPISH nodes.
	# This is the primary hypothesis-steering mechanism that should make the identity
	# decomposition prefer tag-consistent ambiguous transitions.
	if tag_pair_pref_vars and tag_pair_reward_scaled > 0:
		for _pid, pv in tag_pair_pref_vars.items():
			terms.append(-int(tag_pair_reward_scaled) * pv)

	# MCF determinism: minimize all tag flow vars so unsupported propagation is not forced.
	if tag_keys:
		for tk in tag_keys:
			for eid in edge_ids_sorted:
				terms.append(tag_flow_vars[tk][eid])  # coefficient 1

	model.Minimize(sum(terms) if terms else 0)

	# Solve
	solver = cp_model.CpSolver()
	# solver.parameters.num_search_workers = 1

	t0 = time.time()
	status = solver.Solve(model)
	runtime_ms = int(round((time.time() - t0) * 1000.0))

	status_map = {
		cp_model.OPTIMAL: "OPTIMAL",
		cp_model.FEASIBLE: "FEASIBLE",
		cp_model.INFEASIBLE: "INFEASIBLE",
		cp_model.MODEL_INVALID: "MODEL_INVALID",
		cp_model.UNKNOWN: "UNKNOWN",
	}
	status_s = status_map.get(status, str(status))

	flow_by_edge: Dict[str, int] = {}
	selected: List[str] = []
	if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		for eid in edge_ids_sorted:
			val = int(solver.Value(flow_var[eid]))
			if val > 0:
				flow_by_edge[str(eid)] = int(val)
				selected.append(str(eid))

	# Post-solve: explain-or-penalize stats + debug artifact
	explained_ids: List[str] = []
	dropped_ids: List[str] = []
	if unexplained_tracklet_penalty is not None and explained_var_by_tid:
		for tid in sorted(explained_var_by_tid.keys()):
			try:
				is_expl = int(solver.Value(explained_var_by_tid[tid])) == 1 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else False
			except Exception:
				is_expl = False
			if is_expl:
				explained_ids.append(tid)
			else:
				dropped_ids.append(tid)

		if explain_debug is not None:
			_explain = dict(explain_debug)
			_explain["summary"] = dict(_explain.get("summary", {}))
			_explain["summary"].update(
				{
					"n_tracklets_explained": int(len(explained_ids)),
					"n_tracklets_unexplained": int(len(dropped_ids)),
					"total_unexplained_penalty_scaled": int((penalty_scaled or 0) * len(dropped_ids)),
					"solution_status": status_s,
				}
			)
			recs: List[Dict[str, Any]] = []
			for r in _explain.get("tracklets", []):
				if not isinstance(r, dict):
					continue
				tid = str(r.get("base_tracklet_id", ""))
				rr = dict(r)
				rr["explained"] = bool(tid in set(explained_ids))
				recs.append(rr)
			_explain["tracklets"] = recs
			if debug_dir is not None:
				_emit_ilp2_explain_or_penalize_debug(debug_dir=Path(debug_dir), payload=_explain)

	selected = sorted(selected)
	obj_scaled: Optional[int] = int(round(solver.ObjectiveValue())) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
	obj_value: Optional[float] = (float(obj_scaled) / float(scale)) if obj_scaled is not None and scale > 0 else None

	# Read back tag flows (sparse: only store non-zeros)
	if tag_keys and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		for tk in tag_keys:
			for eid in edge_ids_sorted:
				try:
					v = int(solver.Value(tag_flow_vars[tk][eid]))
				except Exception:
					v = 0
				if v > 0:
					tag_flow_by_tag_edge.setdefault(tk, {})[eid] = int(v)

	# Read back realized local tag-consistent pair preferences.
	realized_pair_ids: List[str] = []
	pair_counts_by_tag: Dict[str, int] = {}
	pair_realized_counts_by_tag: Dict[str, int] = {}
	realized_pair_ids_by_tag: Dict[str, List[str]] = {}
	for rec in tag_pair_preference_records:
		tk = str(rec.get("tag_key"))
		pair_counts_by_tag[tk] = int(pair_counts_by_tag.get(tk, 0)) + 1
		pid = str(rec.get("pair_id"))
		val = 0
		if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
			try:
				val = int(solver.Value(tag_pair_pref_vars.get(pid))) if pid in tag_pair_pref_vars else 0
			except Exception:
				val = 0
		rec["realized"] = int(val)
		if val > 0:
			realized_pair_ids.append(pid)
			pair_realized_counts_by_tag[tk] = int(pair_realized_counts_by_tag.get(tk, 0)) + 1
			realized_pair_ids_by_tag.setdefault(tk, []).append(pid)

	# Compact D4 handoff for realized local pairings. Keep only fields D4 needs.
	realized_group_pairings: List[Dict[str, Any]] = []
	for rec in tag_pair_preference_records:
		if int(rec.get("realized", 0)) <= 0:
			continue
		realized_group_pairings.append(
			{
				"group_node_id": str(rec.get("group_node_id")),
				"group_node_type": str(rec.get("group_node_type")),
				"tag_key": str(rec.get("tag_key")),
				"in_edge_id": str(rec.get("in_edge_id")),
				"out_edge_id": str(rec.get("out_edge_id")),
				"in_node_id": str(rec.get("in_node_id")),
				"out_node_id": str(rec.get("out_node_id")),
			}
		)

	# Post-solve: record ping statuses (bound pings only)
	if bound_pings:
		for bp in bound_pings:
			tk = bp["tag_key"]
			pid = bp["ping_id"]
			nid = bp["node_id"]
			rec = {
				"ping_id": pid,
				"node_id": nid,
				"node_type": bp.get("node_type"),
				"match_role": bp.get("match_role"),
				"frame_index": bp.get("frame_index"),
				"visit": 0,
				"miss": 0,
			}
			if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
				try:
					rec["visit"] = int(solver.Value(visit_vars.get(pid))) if pid in visit_vars else 0
				except Exception:
					rec["visit"] = 0
				try:
					rec["miss"] = int(solver.Value(miss_vars.get(pid))) if pid in miss_vars else 0
				except Exception:
					rec["miss"] = 0
			ping_statuses_by_tag.setdefault(tk, []).append(rec)
		for tk in list(ping_statuses_by_tag.keys()):
			ping_statuses_by_tag[tk] = sorted(ping_statuses_by_tag[tk], key=lambda r: (str(r.get("ping_id")), str(r.get("node_id"))))

	# Append logging-only meta status records for tags so downstream debug artifacts can show
	# activation reason, must-link grounding, and support behavior.
	for tk in tag_keys:
		ping_statuses_by_tag.setdefault(tk, [])
		ml_visit_val = 0
		ml_miss_val = 0
		if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
			try:
				ml_visit_val = int(solver.Value(must_link_visit_vars.get(tk))) if tk in must_link_visit_vars else 0
			except Exception:
				ml_visit_val = 0
			try:
				ml_miss_val = int(solver.Value(must_link_miss_vars.get(tk))) if tk in must_link_miss_vars else 0
			except Exception:
				ml_miss_val = 0
		ping_statuses_by_tag[tk].append(
			{
				"record_type": "tag_meta",
				"tag_key": tk,
				"is_active": int(tag_active_const.get(tk, 0)),
				"activation_reason": tag_activation_reason.get(tk, "unknown"),
				"bound_ping_ids": list(bound_ping_ids_by_tag.get(tk, [])),
				"bound_ping_node_ids": list(bound_ping_nodes_by_tag.get(tk, [])),
				"must_link_tracklets": list(tag_must_link_tracklets.get(tk, [])),
				"must_link_resolved_nodes": list(tag_must_link_resolved_nodes.get(tk, [])),
				"must_link_support_applicable": int(1 if len(must_link_support_nodes_by_tag.get(tk, [])) > 0 else 0),
				"must_link_support_visit": int(ml_visit_val),
				"must_link_support_miss": int(ml_miss_val),
				"must_link_support_nodes": list(must_link_support_nodes_by_tag.get(tk, [])),
				"must_link_support_edges": list(must_link_support_edges_by_tag.get(tk, [])),
				"n_bound_pings": int(len(bound_ping_ids_by_tag.get(tk, []))),
				"n_must_link_tracklets": int(len(tag_must_link_tracklets.get(tk, []))),
				"n_must_link_resolved_nodes": int(len(tag_must_link_resolved_nodes.get(tk, []))),
				"n_must_link_support_edges": int(len(must_link_support_edges_by_tag.get(tk, []))),
				"n_tag_pair_preferences": int(pair_counts_by_tag.get(tk, 0)),
				"n_tag_pair_preferences_realized": int(pair_realized_counts_by_tag.get(tk, 0)),
				"tag_pair_preference_ids_realized": list(sorted(realized_pair_ids_by_tag.get(tk, []))),
			}
		)
		ping_statuses_by_tag[tk] = sorted(
			ping_statuses_by_tag[tk],
			key=lambda r: (
				0 if str(r.get("record_type", "ping")) == "tag_meta" else 1,
				str(r.get("ping_id", "")),
				str(r.get("node_id", "")),
			),
		)

	if tag_pair_debug_payload is not None and debug_dir is not None:
		_tag_pair_debug = dict(tag_pair_debug_payload)
		_tag_pair_debug["summary"] = dict(_tag_pair_debug.get("summary", {}))
		_tag_pair_debug["summary"].update(
			{
				"n_pair_preferences_realized": int(len(realized_pair_ids)),
				"solution_status": status_s,
			}
		)
		_emit_ilp2_tag_pair_preferences_debug(
			debug_dir=Path(debug_dir),
			payload=_tag_pair_debug,
		)

	return ILPResult(
		status=status_s,
		objective_scaled=obj_scaled,
		objective_value=obj_value,
		runtime_ms=runtime_ms,
		selected_edge_ids=selected,
		flow_by_edge_id=flow_by_edge,
		cost_scale=int(scale),
		enforced_min_one_path=bool(enforced_min_one_path),
		rounding_n_edges=int(rounding_stats.get("rounding_n_edges", 0)),
		rounding_n_edges_nonzero=int(rounding_stats.get("rounding_n_edges_nonzero", 0)),
		rounding_max_abs_scaled_error=float(rounding_stats.get("rounding_max_abs_scaled_error", 0.0)),
		rounding_max_abs_cost_error=float(rounding_stats.get("rounding_max_abs_cost_error", 0.0)),
		unexplained_tracklet_penalty=float(unexplained_tracklet_penalty) if unexplained_tracklet_penalty is not None else None,
		n_tracklets_total=int(len(explained_var_by_tid)),
		n_tracklets_explained=int(len(explained_ids)),
		n_tracklets_unexplained=int(len(dropped_ids)),
		dropped_tracklet_ids=list(sorted(dropped_ids)),
		explained_tracklet_ids=list(sorted(explained_ids)),
		realized_group_pairings=realized_group_pairings,
	), tag_flow_by_tag_edge, ping_statuses_by_tag


def solve_structure_ilp2_core(
	*,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	costs_df: pd.DataFrame,
	constraints: Dict[str, Any] | None = None,
	debug_dir: Path | None = None,
	emit_transparency: bool = True,
	unexplained_tracklet_penalty: float | None = None,
	group_boundary_window_frames: int = 10,
	tag_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[ILPResult, Dict[str, Dict[str, int]], Dict[str, List[Dict[str, Any]]]]:
	"""Core solver for ILP2.

	NOTE: This module is ILP2-only. ILP1 delegation is intentionally removed.
	"""
	return _solve_identity_ilp2_identity_only(
		nodes_df=nodes_df,
		edges_df=edges_df,
		costs_df=costs_df,
		constraints=constraints,
		debug_dir=debug_dir,
		emit_transparency=emit_transparency,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		group_boundary_window_frames=int(group_boundary_window_frames),
		tag_inputs=tag_inputs,
	)


def solve_structure_ilp2(
	*,
	compiled: CompiledInputs,
	layout: ClipOutputLayout,
	manifest: ClipManifest,
	checkpoint: str,
	unexplained_tracklet_penalty: float | None = None,
	group_boundary_window_frames: int = 10,
) -> ILPResult:
	"""Wrapper: solve + write the standard debug/audit outputs."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	# MCF inputs snapshot (non-behavioral)
	try:
		tag_inputs = _emit_mcf_tag_inputs(
			debug_dir=dbg,
			manifest=manifest,
			checkpoint=checkpoint,
			nodes_df=compiled.nodes_df,
			constraints=compiled.constraints,
		)
	except Exception:
		tag_inputs = None

	t0 = time.time()
	res, tag_flow_by_tag_edge, ping_statuses_by_tag = solve_structure_ilp2_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		constraints=compiled.constraints,
		debug_dir=dbg,
		emit_transparency=True,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		group_boundary_window_frames=int(group_boundary_window_frames),
		tag_inputs=tag_inputs,
	)
	elapsed_ms = int(round((time.time() - t0) * 1000.0))

	# --- Standard debug outputs ---
	# Selected edges parquet
	edges = compiled.edges_df.copy()
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges = edges[edges["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges) > 0:
		edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
		costs = compiled.costs_df[["edge_id", "total_cost"]].copy()
		costs["edge_id"] = costs["edge_id"].astype(str)
		edges = edges.merge(costs, on="edge_id", how="left", validate="1:1")
		edges["flow"] = edges["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))
	else:
		edges["total_cost"] = []
		edges["flow"] = []

	out_sel = dbg / "d3_selected_edges.parquet"
	edges.to_parquet(out_sel, index=False)

	out_entities = _write_entities_format_a(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)
	out_ledger = _write_solution_ledger_json(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)

	def rel(p: Path) -> str:
		return str(p.relative_to(layout.clip_root))

	# --- Breadcrumbs (provable evidence ilp2 ran) ---
	# We write:
	#  1) a tiny json file under _debug/
	#  2) fields in the audit event (and optionally in the ledger in a later refactor)
	breadcrumb_payload: Dict[str, Any] = {
		"solver_impl": _SOLVER_IMPL,
		"solver_module": _SOLVER_MODULE,
		"solver_version": _SOLVER_VERSION,
		"checkpoint": checkpoint,
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"elapsed_ms_wrapper": elapsed_ms,
	}
	out_breadcrumbs = _write_solver_breadcrumbs_json(debug_dir=dbg, payload=breadcrumb_payload)

	# MCF tag paths (schema exists even if empty)
	try:
		out_mcf_paths = _emit_mcf_tag_paths(
			debug_dir=dbg,
			manifest=manifest,
			checkpoint=checkpoint,
			tag_inputs=tag_inputs,
			identity_flow_by_edge=res.flow_by_edge_id,
			tag_flow_by_tag_edge=(tag_flow_by_tag_edge or {}),
			ping_statuses_by_tag=(ping_statuses_by_tag or {}),
		)
	except Exception:
		out_mcf_paths = None

	# Audit summary (parity)
	edge_type_counts: Dict[str, int] = {}
	if "edge_type" in edges.columns:
		for k, v in edges["edge_type"].astype(str).value_counts().items():
			edge_type_counts[str(k)] = int(v)

	k_paths = None
	try:
		source_id = _find_unique_node_id(compiled.nodes_df, node_type="NodeType.SOURCE")
		out_ids = compiled.edges_df[compiled.edges_df["u"].astype(str) == str(source_id)]["edge_id"].astype(str).tolist()
		k_paths = int(sum(res.flow_by_edge_id.get(eid, 0) for eid in out_ids))
	except Exception:
		k_paths = None

	tag_inputs_summary = (tag_inputs or {}).get("summary", {}) if isinstance(tag_inputs, dict) else {}
	tag_paths_summary: Dict[str, Any] = {}
	try:
		if out_mcf_paths is not None:
			import json
			with open(out_mcf_paths, "r", encoding="utf-8") as f:
				_mcf_payload = json.load(f)
			if isinstance(_mcf_payload, dict) and isinstance(_mcf_payload.get("summary"), dict):
				tag_paths_summary = dict(_mcf_payload.get("summary", {}))
	except Exception:
		tag_paths_summary = {}

	append_audit_event(
		layout=layout,
		event={
			"event_type": "d3_ilp_summary",
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"checkpoint": checkpoint,
			"solver_impl": _SOLVER_IMPL,
			"solver_module": _SOLVER_MODULE,
			"solver_version": _SOLVER_VERSION,
			"status": res.status,
			"objective_value": res.objective_value,
			"objective_scaled": res.objective_scaled,
			"cost_scale": res.cost_scale,
			"enforced_min_one_path": res.enforced_min_one_path,
			"rounding": {
				"n_edges": res.rounding_n_edges,
				"n_edges_nonzero": res.rounding_n_edges_nonzero,
				"max_abs_scaled_error": res.rounding_max_abs_scaled_error,
				"max_abs_cost_error": res.rounding_max_abs_cost_error,
			},
			"runtime_ms": res.runtime_ms,
			"n_selected_edges": len(res.selected_edge_ids),
			"selected_edge_type_counts": dict(sorted(edge_type_counts.items(), key=lambda kv: kv[0])),
			"explain_or_penalize": {
				"unexplained_tracklet_penalty": res.unexplained_tracklet_penalty,
				"n_tracklets_total": res.n_tracklets_total,
				"n_tracklets_explained": res.n_tracklets_explained,
				"n_tracklets_unexplained": res.n_tracklets_unexplained,
			},
			"tag_evidence": {
				"n_tags": int(tag_inputs_summary.get("n_tags", 0)),
				"n_pings": int(tag_inputs_summary.get("n_pings", 0)),
				"n_bound_pings": int(tag_inputs_summary.get("n_bound_pings", 0)),
				"n_bound_single": int(tag_inputs_summary.get("n_bound_single", 0)),
				"n_bound_group": int(tag_inputs_summary.get("n_bound_group", 0)),
				"n_bound_groupish": int(tag_inputs_summary.get("n_bound_groupish", 0)),
				"n_active_tags": int(tag_paths_summary.get("n_active_tags", 0)),
				"n_tags_with_nonzero_flow": int(tag_paths_summary.get("n_tags_with_nonzero_flow", 0)),
			},
			"n_paths": k_paths,
			"debug_outputs": {
				"d3_selected_edges_parquet": rel(out_sel),
				"d3_entities_format_a_json": rel(out_entities),
				"d3_solution_ledger_json": rel(out_ledger),
				"d3_solver_breadcrumbs_json": rel(out_breadcrumbs),
				**({"d3_mcf_tag_inputs_json": rel(dbg / "d3_mcf_tag_inputs.json")} if (dbg / "d3_mcf_tag_inputs.json").exists() else {}),
				**({"d3_mcf_tag_paths_json": rel(out_mcf_paths)} if out_mcf_paths is not None else {}),
				**(
					{"d3_ilp2_group_semantics_json": rel(dbg / "d3_ilp2_group_semantics.json")}
					if (dbg / "d3_ilp2_group_semantics.json").exists()
					else {}
				),
				**(
					{"d3_ilp2_explain_or_penalize_json": rel(dbg / "d3_ilp2_explain_or_penalize.json")}
					if (dbg / "d3_ilp2_explain_or_penalize.json").exists()
					else {}
				),
				**({"d3_ilp2_tag_pair_preferences_json": rel(dbg / "d3_ilp2_tag_pair_preferences.json")} if (dbg / "d3_ilp2_tag_pair_preferences.json").exists() else {}),
			},
		},
	)

	return res
