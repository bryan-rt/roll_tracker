"""Stage D3 — ILP structure solve (POC_1).

This module solves the *structure-only* stitching problem on the pruned D1/D2 graph
produced by d3_compile.compile_solver_inputs().

POC_1 scope:
  - No must_link / cannot_link enforcement yet
  - No person_id extraction yet
  - Emit debug artifacts + audit summary only
"""

from __future__ import annotations

import time
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event
from bjj_pipeline.stages.stitch.d3_compile import CompiledInputs


@dataclass(frozen=True)
class ILPResult:
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


def _debug_dir(layout: ClipOutputLayout) -> Path:
	return layout.clip_root / "_debug"



def _extract_entity_paths_format_a(
	*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, flow_by_edge_id: Dict[str, int]
) -> List[Dict[str, Any]]:
	"""Decompose selected edge flows into per-entity SOURCE->SINK paths (Format A).

	Format A is intended to be human-auditable and temporally monotone:
	  - Each entity is a single path through the DAG from SOURCE to SINK.
	  - Steps are ordered by traversal; we include node frame ranges to sanity-check monotonicity.

	This is a debug/POC artifact only (not an F0 contract).
	"""
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	nodes["node_id"] = nodes["node_id"].astype(str)
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges["u"] = edges["u"].astype(str)
	edges["v"] = edges["v"].astype(str)

	# Index nodes for metadata lookup
	nodes_ix = nodes.set_index("node_id", drop=False)

	# Build remaining-capacity adjacency from flow
	remaining: Dict[str, int] = {str(k): int(v) for k, v in flow_by_edge_id.items() if int(v) > 0}
	out_by_u: Dict[str, List[str]] = {}
	edge_row: Dict[str, Any] = {}
	for _, e in edges.sort_values(["edge_id"], kind="mergesort").iterrows():
		eid = str(e["edge_id"])
		edge_row[eid] = e.to_dict()
		if remaining.get(eid, 0) <= 0:
			continue
		u = str(e["u"])
		out_by_u.setdefault(u, []).append(eid)

	def node_meta(nid: str) -> Dict[str, Any]:
		if nid not in nodes_ix.index:
			return {"node_id": nid}
		r = nodes_ix.loc[nid]
		out: Dict[str, Any] = {"node_id": str(r.get("node_id")), "node_type": str(r.get("node_type"))}
		# Optional temporal hints
		for k in ("start_frame", "end_frame", "base_tracklet_id", "carrier_tracklet_id", "disappearing_tracklet_id", "new_tracklet_id"):
			if k in r.index and pd.notna(r[k]):
				out[k] = int(r[k]) if isinstance(r[k], (int, float)) and str(r[k]).isdigit() else str(r[k])
		return out

	# Identify SOURCE/SINK ids (robust to missing columns already validated upstream)
	source_id = _find_unique_node_id(nodes, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes, node_type="NodeType.SINK")

	total_flow = 0
	for eid in out_by_u.get(source_id, []):
		total_flow += remaining.get(eid, 0)

	entities: List[Dict[str, Any]] = []
	for ent_i in range(total_flow):
		cur = source_id
		steps: List[Dict[str, Any]] = []
		visited_guard = 0
		tracklets_in_order: List[str] = []

		while cur != sink_id:
			visited_guard += 1
			if visited_guard > 5000:
				# Defensive guard: should never happen in a valid acyclic graph.
				raise RuntimeError("Path extraction exceeded step limit; possible cycle in selected edges.")

			choices = out_by_u.get(cur, [])
			# pick first edge with remaining flow (deterministic due to sorted edge_id order)
			next_eid = None
			for eid in choices:
				if remaining.get(eid, 0) > 0:
					next_eid = eid
					break
			if next_eid is None:
				raise RuntimeError(f"Failed to extract full path: stuck at node_id={cur}")

			remaining[next_eid] -= 1
			if remaining[next_eid] <= 0:
				remaining.pop(next_eid, None)

			e = edge_row[next_eid]
			u = str(e["u"])
			v = str(e["v"])
			step = {
				"edge_id": str(e.get("edge_id")),
				"edge_type": str(e.get("edge_type")),
				"u": u,
				"v": v,
			}
			for k in ("dt_frames", "merge_end", "split_start", "capacity"):
				if k in e and pd.notna(e[k]):
					step[k] = int(e[k]) if isinstance(e[k], (int, float)) else e[k]
			step["u_node"] = node_meta(u)
			step["v_node"] = node_meta(v)

			# Tracklets (best-effort): use base_tracklet_id if present
			for nid in (u, v):
				if nid in nodes_ix.index and "base_tracklet_id" in nodes_ix.columns:
					bt = nodes_ix.loc[nid].get("base_tracklet_id")
					if pd.notna(bt):
						bt_s = str(bt)
						if (len(tracklets_in_order) == 0) or (tracklets_in_order[-1] != bt_s):
							tracklets_in_order.append(bt_s)

			steps.append(step)
			cur = v

		# Temporal monotonicity check (best-effort)
		start_frames = []
		for s in steps:
			vnode = s.get("v_node", {})
			sf = vnode.get("start_frame")
			if isinstance(sf, int):
				start_frames.append(sf)
		temporal_monotone = start_frames == sorted(start_frames)

		entities.append(
			{
				"entity_id": ent_i + 1,
				"steps": steps,
				"tracklets_in_order": tracklets_in_order,
				"temporal_monotone_by_start_frame": bool(temporal_monotone),
				"april_tag_found_in": None,
			}
		)

	return entities


def _write_entities_format_a(
	*, layout: ClipOutputLayout, compiled: CompiledInputs, res: ILPResult, checkpoint: str, manifest: ClipManifest
) -> Path:
	"""Write Format A entity paths to _debug and return output path."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	entities = _extract_entity_paths_format_a(
		nodes_df=compiled.nodes_df, edges_df=compiled.edges_df, flow_by_edge_id=res.flow_by_edge_id
	)
	out = dbg / "d3_entities_format_a.json"
	payload = {
		"schema_version": 1,
		"checkpoint": checkpoint,
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"n_entities": len(entities),
		"entities": entities,
	}
	out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
	return out


def _cost_scale_for(costs: pd.Series) -> int:
	"""Choose a deterministic integer scale so we can use CP-SAT's integer objective.

	We pick 1000 which preserves 0.001 precision; D2 costs commonly include 0.01 increments.
	We also validate that costs * scale are close to integers.
	"""
	return 1000



def _scaled_costs(costs_df: pd.DataFrame, *, scale: int) -> tuple[Dict[str, int], Dict[str, float | int]]:
	if "edge_id" not in costs_df.columns or "total_cost" not in costs_df.columns:
		raise ValueError("d2_edge_costs missing required columns: edge_id, total_cost")
	out: Dict[str, int] = {}
	n_edges = 0
	n_nonzero = 0
	max_abs_scaled_err = 0.0
	for _, row in costs_df.iterrows():
		n_edges += 1
		edge_id = str(row["edge_id"])
		c = float(row["total_cost"])
		s = c * scale
		rounded = int(round(s))
		err = abs(s - float(rounded))
		if err > 0.0:
			nonlocal_max = max_abs_scaled_err
			if err > nonlocal_max:
				max_abs_scaled_err = err
			n_nonzero += 1
		out[edge_id] = rounded
	stats: Dict[str, float | int] = {
		"rounding_n_edges": int(n_edges),
		"rounding_n_edges_nonzero": int(n_nonzero),
		"rounding_max_abs_scaled_error": float(max_abs_scaled_err),
		"rounding_max_abs_cost_error": float(max_abs_scaled_err) / float(scale) if scale > 0 else float("nan"),
	}
	return out, stats


def _find_unique_node_id(nodes_df: pd.DataFrame, *, node_type: str) -> str:
	if "node_type" not in nodes_df.columns or "node_id" not in nodes_df.columns:
		raise ValueError("d1_graph_nodes missing required columns: node_id, node_type")
	m = nodes_df[nodes_df["node_type"].astype(str) == node_type]
	if len(m) != 1:
		raise ValueError(f"Expected exactly 1 node with node_type={node_type}, found {len(m)}")
	return str(m.iloc[0]["node_id"])


def solve_structure_ilp_core(
	*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, costs_df: pd.DataFrame, unexplained_tracklet_penalty: float | None = None
) -> ILPResult:
	"""Pure core solver used by POC_1 (unit-test friendly; no I/O)."""
	start = time.time()

	# Required columns
	for col in ("node_id", "node_type", "capacity"):
		if col not in nodes_df.columns:
			raise ValueError(f"d1_graph_nodes missing required column: {col}")
	for col in ("edge_id", "u", "v", "edge_type", "capacity"):
		if col not in edges_df.columns:
			raise ValueError(f"d1_graph_edges missing required column: {col}")

	# Normalize ids
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	nodes["node_id"] = nodes["node_id"].astype(str)
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges["u"] = edges["u"].astype(str)
	edges["v"] = edges["v"].astype(str)

	if unexplained_tracklet_penalty is not None and unexplained_tracklet_penalty < 0:
		raise ValueError("unexplained_tracklet_penalty must be >= 0 when provided")

	node_ids = set(nodes["node_id"].tolist())
	if not set(edges["u"]).issubset(node_ids) or not set(edges["v"]).issubset(node_ids):
		missing_u = sorted(set(edges["u"]) - node_ids)[:25]
		missing_v = sorted(set(edges["v"]) - node_ids)[:25]
		raise ValueError(f"Edges reference unknown node_id(s): missing_u={missing_u} missing_v={missing_v}")

	source_id = _find_unique_node_id(nodes, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes, node_type="NodeType.SINK")

	# Deterministic edge ordering
	edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	# Determine variable type: bool if all capacities are 1, else integer flow
	max_edge_cap = int(pd.to_numeric(edges["capacity"], errors="raise").max())
	max_node_cap = int(pd.to_numeric(nodes["capacity"], errors="raise").max())
	use_flow_int = (max_edge_cap > 1) or (max_node_cap > 1)

	scale = _cost_scale_for(costs_df["total_cost"] if "total_cost" in costs_df.columns else pd.Series([], dtype=float))
	cost_int, rounding_stats = _scaled_costs(costs_df, scale=scale)

	model = cp_model.CpModel()
	var_by_edge: Dict[str, Any] = {}

	for _, e in edges.iterrows():
		edge_id = str(e["edge_id"])
		cap_e = int(e["capacity"])
		if use_flow_int:
			var_by_edge[edge_id] = model.NewIntVar(0, cap_e, f"f_{edge_id}")
		else:
			# With binary selection, edge capacity should be 1; we still validate.
			if cap_e != 1:
				raise ValueError(f"Binary edge selection requires capacity=1, got {cap_e} for edge_id={edge_id}")
			var_by_edge[edge_id] = model.NewBoolVar(f"x_{edge_id}")

	# Build adjacency lists
	in_edges: Dict[str, List[str]] = {nid: [] for nid in node_ids}
	out_edges: Dict[str, List[str]] = {nid: [] for nid in node_ids}
	for _, e in edges.iterrows():
		u = str(e["u"])
		v = str(e["v"])
		eid = str(e["edge_id"])
		out_edges[u].append(eid)
		in_edges[v].append(eid)

	# Pre-index node rows for caps / tracklet grouping
	nodes_ix = nodes.set_index("node_id", drop=False)

	# Non-terminal nodes: conservation + capacity
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		ntype = str(n["node_type"])
		if ntype in ("NodeType.SOURCE", "NodeType.SINK"):
			continue
		cap_n = int(n["capacity"])
		ins = [var_by_edge[eid] for eid in in_edges[nid]]
		outs = [var_by_edge[eid] for eid in out_edges[nid]]
		model.Add(sum(ins) == sum(outs))
		model.Add(sum(ins) <= cap_n)
		model.Add(sum(outs) <= cap_n)

	# Terminals: balance total flow (let K be decided by costs)
	src_out = [var_by_edge[eid] for eid in out_edges[source_id]]
	snk_in = [var_by_edge[eid] for eid in in_edges[sink_id]]
	model.Add(sum(src_out) == sum(snk_in))
	# Optional: require >=1 only when the graph actually contains track nodes.
	# This avoids forcing infeasible/meaningless solutions on empty clips.
	has_track_nodes = nodes["node_type"].astype(str).isin(
		["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"]
	).any()
	enforced_min_one_path = bool(has_track_nodes and len(src_out) > 0)
	if enforced_min_one_path:
		model.Add(sum(src_out) >= 1)

	# ------------------------------------------------------------
	# "Explain each tracklet OR pay a penalty"
	#
	# We define "explained" over base_tracklet_id appearing in SINGLE_TRACKLET nodes.
	# A tracklet is explained iff the solution routes >0 flow through ANY node with that base_tracklet_id.
	# Otherwise, we pay unexplained_tracklet_penalty once per base_tracklet_id.
	#
	# This makes the number of paths K emerge from data: the solver will create additional SOURCE→SINK
	# paths when it’s cheaper than leaving tracklets unexplained.
	# ------------------------------------------------------------
	tracklet_penalty_scaled: int | None = None
	explained_var_by_tid: Dict[str, cp_model.IntVar] = {}
	unexplained_var_by_tid: Dict[str, cp_model.IntVar] = {}

	if unexplained_tracklet_penalty is not None and unexplained_tracklet_penalty > 0:
		tracklet_penalty_scaled = int(round(float(unexplained_tracklet_penalty) * float(scale)))

		if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
			single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)

			# group nodes by base_tracklet_id
			for tid, grp in single.groupby("base_tracklet_id", sort=True):
				node_list = [str(x) for x in grp["node_id"].astype(str).tolist()]
				# total flow through tid: sum of inflows over all nodes in this tid
				flow_terms = []
				M = 0
				for nid in node_list:
					# Node cap upper-bounds flow through node
					try:
						cap_n = int(nodes_ix.loc[nid]["capacity"])
					except Exception:
						cap_n = 1
					M += max(1, cap_n)
					for eid in in_edges.get(nid, []):
						flow_terms.append(var_by_edge[eid])

				# If no inflow terms (shouldn’t happen in valid graph), skip
				if len(flow_terms) == 0:
					continue

				explained = model.NewBoolVar(f"tid_explained_{tid}")
				unexplained = model.NewBoolVar(f"tid_unexplained_{tid}")
				model.Add(explained + unexplained == 1)

				# If explained=1 => total_flow >= 1 ; if explained=0 => total_flow == 0
				total_flow = sum(flow_terms)
				model.Add(total_flow >= 1).OnlyEnforceIf(explained)
				model.Add(total_flow == 0).OnlyEnforceIf(explained.Not())

				explained_var_by_tid[str(tid)] = explained
				unexplained_var_by_tid[str(tid)] = unexplained

	# Objective
	terms = []
	for _, e in edges.iterrows():
		eid = str(e["edge_id"])
		coef = int(cost_int[eid])
		terms.append(coef * var_by_edge[eid])
	# Add unexplained-tracklet penalties (if enabled)
	if tracklet_penalty_scaled is not None and tracklet_penalty_scaled > 0:
		for tid, unex in unexplained_var_by_tid.items():
			terms.append(int(tracklet_penalty_scaled) * unex)
	model.Minimize(sum(terms))

	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.random_seed = 0
	solver.parameters.max_time_in_seconds = 30.0
	solver.parameters.log_search_progress = False

	status_code = solver.Solve(model)
	status = solver.StatusName(status_code)

	runtime_ms = int(round((time.time() - start) * 1000))

	selected: List[str] = []
	flow_by_edge: Dict[str, int] = {}
	if status in ("OPTIMAL", "FEASIBLE"):
		for _, e in edges.iterrows():
			eid = str(e["edge_id"])
			val = int(solver.Value(var_by_edge[eid]))
			flow_by_edge[eid] = val
			if val > 0:
				selected.append(eid)

	obj_scaled = None
	obj_value = None
	if status in ("OPTIMAL", "FEASIBLE"):
		obj_scaled = int(round(solver.ObjectiveValue()))
		obj_value = float(obj_scaled) / float(scale)

	n_tracklets_total = int(len(unexplained_var_by_tid))
	n_tracklets_explained = 0
	n_tracklets_unexplained = 0
	if status in ("OPTIMAL", "FEASIBLE") and n_tracklets_total > 0:
		for tid, ex in explained_var_by_tid.items():
			if int(solver.Value(ex)) == 1:
				n_tracklets_explained += 1
		for tid, unex in unexplained_var_by_tid.items():
			if int(solver.Value(unex)) == 1:
				n_tracklets_unexplained += 1

	return ILPResult(
		status=status,
		objective_scaled=obj_scaled,
		objective_value=obj_value,
		runtime_ms=runtime_ms,
		selected_edge_ids=sorted(selected),
		flow_by_edge_id=flow_by_edge,
		cost_scale=scale,
		enforced_min_one_path=enforced_min_one_path,
		rounding_n_edges=int(rounding_stats.get("rounding_n_edges", 0)),
		rounding_n_edges_nonzero=int(rounding_stats.get("rounding_n_edges_nonzero", 0)),
		rounding_max_abs_scaled_error=float(rounding_stats.get("rounding_max_abs_scaled_error", 0.0)),
		rounding_max_abs_cost_error=float(rounding_stats.get("rounding_max_abs_cost_error", 0.0)),
		unexplained_tracklet_penalty=float(unexplained_tracklet_penalty) if unexplained_tracklet_penalty is not None else None,
		n_tracklets_total=int(n_tracklets_total),
		n_tracklets_explained=int(n_tracklets_explained),
		n_tracklets_unexplained=int(n_tracklets_unexplained),
	)


def solve_structure_ilp(
	*, compiled: CompiledInputs, layout: ClipOutputLayout, manifest: ClipManifest, checkpoint: str, unexplained_tracklet_penalty: float | None = None
) -> ILPResult:
	"""POC_1 wrapper: solve + write debug outputs + audit summary."""
	res = solve_structure_ilp_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
	)

	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	# Selected edges parquet
	edges = compiled.edges_df.copy()
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges = edges[edges["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges) > 0:
		edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
		# Merge costs (already aligned/1:1 by edge_id), add flow
		costs = compiled.costs_df[["edge_id", "total_cost"]].copy()
		costs["edge_id"] = costs["edge_id"].astype(str)
		edges = edges.merge(costs, on="edge_id", how="left", validate="1:1")
		edges["flow"] = edges["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))
	else:
		edges["total_cost"] = []
		edges["flow"] = []

	out_sel = dbg / "d3_selected_edges.parquet"
	edges.to_parquet(out_sel, index=False)

	# Entity paths (Format A JSON)
	out_entities = _write_entities_format_a(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)

	def rel(p: Path) -> str:
		return str(p.relative_to(layout.clip_root))

	# Audit summary
	edge_type_counts: Dict[str, int] = {}
	if "edge_type" in edges.columns:
		for k, v in edges["edge_type"].astype(str).value_counts().items():
			edge_type_counts[str(k)] = int(v)

	# Total paths (K) inferred from SOURCE outflow (if available)
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
			},
		},
	)

	return res
