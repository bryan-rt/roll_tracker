from __future__ import annotations
import math
"""Stage D3 — ILP structure solve (POC_2_TAGS_MCF).

This module is an *alternate* D3 solver implementation intended to replace the
current POC_2_TAGS "node label" formulation with a multi-commodity flow (MCF)
overlay per AprilTag.

Initial scaffolding goal:
	- Provide a drop-in ILPResult and wrappers compatible with Stage D3/D4 wiring.
	- Delegate to the existing d3_ilp.solve_structure_ilp_core for now, so we can
		toggle between implementations without behavior changes.
	- Keep this file intentionally small; port only what is needed as we migrate.
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

# Breadcrumb constants (written into audit + debug ledger copy).
_SOLVER_IMPL: str = "ilp2"
_SOLVER_MODULE: str = __name__
_SOLVER_VERSION: str = "mcf1_gating_only"


@dataclass(frozen=True)
class ILPResult:
	"""Public result contract consumed by Stage D4.

	Stage D4 only relies on a subset of fields (selected_edge_ids, flow_by_edge_id),
	but we keep parity with d3_ilp.ILPResult to make A/B comparisons easy.
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


def _debug_dir(layout: ClipOutputLayout) -> Path:
	return layout.clip_root / "_debug"


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


def _require_columns(df: pd.DataFrame, *, name: str, cols: List[str]) -> None:
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


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


def _find_unique_node_id(nodes_df: pd.DataFrame, *, node_type: str) -> str:
	_require_columns(nodes_df, name="d1_graph_nodes", cols=["node_id", "node_type"])
	m = nodes_df[nodes_df["node_type"].astype(str) == node_type]
	if len(m) != 1:
		raise ValueError(f"Expected exactly 1 node with node_type={node_type}, found {len(m)}")
	return str(m.iloc[0]["node_id"])


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
	if not isinstance(mg, dict):
		return {}
	out: Dict[str, List[str]] = {}
	for k, v in mg.items():
		tk = _normalize_tag_key(k)
		if tk is None:
			continue
		if isinstance(v, list):
			out[tk] = [str(x) for x in v]
		else:
			out[tk] = [str(v)]
	return out


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

	# Best-effort SINGLE_TRACKLET index for binding (by base_tracklet_id + span).
	index_rows: List[Dict[str, Any]] = []
	try:
		df = nodes_df.copy()
		df["node_id"] = df["node_id"].astype(str)
		df["node_type"] = df["node_type"].astype(str)
		single = df[df["node_type"] == "NodeType.SINGLE_TRACKLET"].copy()
		if "base_tracklet_id" in single.columns:
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)
		if "frame_start" in single.columns:
			single["frame_start"] = pd.to_numeric(single["frame_start"], errors="coerce")
		if "frame_end" in single.columns:
			single["frame_end"] = pd.to_numeric(single["frame_end"], errors="coerce")
		for r in single.itertuples(index=False):
			index_rows.append(
				{
					"node_id": str(getattr(r, "node_id")),
					"base_tracklet_id": str(getattr(r, "base_tracklet_id", "")) if hasattr(r, "base_tracklet_id") else None,
					"frame_start": int(getattr(r, "frame_start")) if hasattr(r, "frame_start") and pd.notna(getattr(r, "frame_start")) else None,
					"frame_end": int(getattr(r, "frame_end")) if hasattr(r, "frame_end") and pd.notna(getattr(r, "frame_end")) else None,
				}
			)
	except Exception:
		index_rows = []

	normalized_pings: List[Dict[str, Any]] = []
	tag_to_ping_ids: Dict[str, List[str]] = {}
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

		binding: Dict[str, Any] = {"status": "unbound", "candidates": [], "chosen": None, "notes": []}
		if frame_i is not None and tid_s is not None and index_rows:
			cands = []
			for b in index_rows:
				if b.get("base_tracklet_id") != tid_s:
					continue
				fs = b.get("frame_start")
				fe = b.get("frame_end")
				if fs is None or fe is None:
					continue
				if int(fs) <= frame_i <= int(fe):
					cands.append(
						{
							"node_id": str(b.get("node_id")),
							"node_type": "NodeType.SINGLE_TRACKLET",
							"span": {"start": int(fs), "end": int(fe)},
							"reason": "contains_frame",
						}
					)
			binding["candidates"] = cands
			if len(cands) == 1:
				binding["status"] = "bound"
				binding["chosen"] = {"node_id": cands[0]["node_id"], "reason": "unique_candidate"}
			elif len(cands) > 1:
				binding["status"] = "ambiguous"
				binding["notes"].append("multiple candidates contain frame")

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
		tags[tk] = {"tag_key": tk, "must_link_tracklets": sorted(set(must_link.get(tk, []))), "pings": tag_to_ping_ids.get(tk, [])}

	payload: Dict[str, Any] = {
		"schema_version": "mcf_tag_inputs_v0.1.0",
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"checkpoint": checkpoint,
		"summary": {
			"n_tags": int(len(tags)),
			"n_pings": int(len(normalized_pings)),
			"n_bound_pings": int(sum(1 for x in normalized_pings if x.get("binding", {}).get("status") == "bound")),
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
) -> Path:
	"""Emit `_debug/d3_mcf_tag_paths.json` (schema v0.1.0).

	MCF-1: tag flow vars exist but are gated and minimized to zero (no enforcement yet).
	"""
	tags_out: Dict[str, Any] = {}
	tags = (tag_inputs or {}).get("tags", {}) if isinstance(tag_inputs, dict) else {}
	pings = (tag_inputs or {}).get("pings", []) if isinstance(tag_inputs, dict) else []

	bound_nodes_by_tag: Dict[str, List[str]] = {}
	if isinstance(pings, list):
		for p in pings:
			if not isinstance(p, dict):
				continue
			tk = str(p.get("tag_key"))
			chosen = (p.get("binding") or {}).get("chosen") or {}
			nid = chosen.get("node_id")
			if tk and nid:
				bound_nodes_by_tag.setdefault(tk, []).append(str(nid))

	for tk in sorted(tags.keys()):
		info = tags.get(tk, {})
		pids = info.get("pings", []) if isinstance(info, dict) else []
		status = "inactive" if not pids else "present"
		notes = ["mcf1_gating_only", "tag_flow_minimized_to_zero"]
		tag_edges = tag_flow_by_tag_edge.get(tk, {})
		selected_edges = []
		for eid, fval in sorted(tag_edges.items()):
			if int(fval) <= 0:
				continue
			selected_edges.append({"edge_id": str(eid), "identity_flow": int(identity_flow_by_edge.get(str(eid), 0)), "tag_flow": int(fval)})
		tags_out[tk] = {
			"status": status,
			"pings": list(pids) if isinstance(pids, list) else [],
			"bound_nodes": sorted(set(bound_nodes_by_tag.get(tk, []))),
			"selected_edges": selected_edges,
			"visited_nodes": [],
			"notes": notes,
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
			"mcf_checkpoint": "MCF-1",
			"n_tag_flow_nonzero_edges_total": n_nonzero,
			"n_tags": int(len(tags_out)),
			"n_pings": int(len(pings) if isinstance(pings, list) else 0),
			"n_bound_pings": int(sum(1 for x in (pings or []) if isinstance(x, dict) and (x.get("binding") or {}).get("status") == "bound")),
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
) -> Tuple[ILPResult, Dict[str, Dict[str, int]]]:
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
	"""
	# Defensive: this checkpoint ignores tags.
	_ = emit_transparency

	# Normalize + validate inputs.
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	costs = costs_df.copy()

	tag_flow_vars: Dict[str, Dict[str, Any]] = {}
	tag_flow_by_tag_edge: Dict[str, Dict[str, int]] = {}

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
	scale = _cost_scale_for(None)
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

	# --- MCF-1: tag flow variables (gated only) ---
	# Create tf[tag, edge] in [0..cap_eff(edge)] and enforce tf <= identity_flow(edge).
	# Add tiny objective tiebreaker to minimize all tf to 0 for determinism (no behavior change).
	tag_keys: List[str] = []
	if isinstance(tag_inputs, dict):
		tags_obj = tag_inputs.get("tags")
		if isinstance(tags_obj, dict):
			tag_keys = sorted([str(k) for k in tags_obj.keys()])

	# Create vars only if tags are present.
	if tag_keys:
		tag_index = {tk: i for i, tk in enumerate(tag_keys)}
		edge_index = {eid: i for i, eid in enumerate(edge_ids_sorted)}
		for tk in tag_keys:
			tag_flow_vars[tk] = {}
			tag_flow_by_tag_edge[tk] = {}
			ki = int(tag_index[tk])
			for eid in edge_ids_sorted:
				cap = int(edge_cap_eff.get(eid, 1))
				ei = int(edge_index[eid])
				tf = model.NewIntVar(0, cap, f"tf[k{ki},e{ei}]")
				# Gating: tag flow can only ride on identity flow
				model.Add(tf <= flow_var[eid])
				tag_flow_vars[tk][eid] = tf

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

	# MCF-1 determinism: minimize all tag flow vars (keeps identity solution unchanged)
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
	), tag_flow_by_tag_edge


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
) -> Tuple[ILPResult, Dict[str, Dict[str, int]]]:
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
	"""Wrapper: solve + write the standard debug/audit outputs.

	This keeps the same external behavior as d3_ilp.solve_structure_ilp so Stage D3/D4
	do not care which solver module is used.
	"""
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
	res, tag_flow_by_tag_edge = solve_structure_ilp2_core(
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

	# --- Standard debug outputs (parity with d3_ilp) ---
	# Keep these written by reusing d3_ilp helpers for now; when d3_ilp2 diverges,
	# we will port only what we need.
	from bjj_pipeline.stages.stitch.d3_ilp import (
		_write_entities_format_a as _write_entities_format_a,
		_write_solution_ledger_json as _write_solution_ledger_json,
		_find_unique_node_id as _find_unique_node_id,
	)

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
			},
		},
	)

	return res
