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
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
	*,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	costs_df: pd.DataFrame,
	constraints: Dict[str, Any] | None = None,
	unexplained_tracklet_penalty: float | None = None,
	group_boundary_window_frames: int = 10,
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

	# Default unexplained-tracklet penalty (stable, config-overridable).
	# Manager-locked rule: do NOT derive from max edge costs.
	# Use a stable default ≈ 3–5×(birth + death); we pick 4× as midpoint.
	if unexplained_tracklet_penalty is None:
		# Only compute a default when the clip contains track nodes (otherwise keep disabled).
		has_track_nodes_tmp = nodes["node_type"].astype(str).isin(
			["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"]
		).any()
		if has_track_nodes_tmp:
			birth_costs: List[float] = []
			death_costs: List[float] = []
			if "edge_id" in costs_df.columns and "total_cost" in costs_df.columns:
				for _, r in costs_df.iterrows():
					eid = str(r["edge_id"])
					c = float(r["total_cost"])
					if eid.startswith("E:BIRTH:"):
						birth_costs.append(c)
					elif eid.startswith("E:DEATH:"):
						death_costs.append(c)
			birth_med = float(statistics.median(birth_costs)) if len(birth_costs) > 0 else 0.0
			death_med = float(statistics.median(death_costs)) if len(death_costs) > 0 else 0.0
			base = birth_med + death_med
			if base > 0.0:
				unexplained_tracklet_penalty = 4.0 * base
			else:
				# Conservative fallback if birth/death edges are absent in this graph.
				unexplained_tracklet_penalty = 1000.0

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

	# Tracklet usage vars (base_tracklet_id). Used for coverage + identity constraints.
	use_tid: Dict[str, cp_model.IntVar] = {}
	# Index SINGLE_TRACKLET nodes by base_tracklet_id
	single_nodes: List[str] = []
	if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		if "base_tracklet_id" in single.columns:
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)
			for tid in sorted(single["base_tracklet_id"].unique().tolist()):
				use_tid[str(tid)] = model.NewBoolVar(f"tid_used_{tid}")
			single_nodes = [str(x) for x in single["node_id"].astype(str).tolist()]

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

	# Inflow expressions (useful for coverage / identity constraints).
	flow_in_by_node: Dict[str, Any] = {}
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		ins = [var_by_edge[eid] for eid in in_edges[nid]]
		flow_in_by_node[nid] = sum(ins) if len(ins) > 0 else 0

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
	# GROUP_TRACKLET semantics (Worker G)
	#
	# Hard constraints that restore semantic meaning of overlap episodes:
	#  - GROUP_TRACKLET usage is 0-or-2 (never 1).
	#  - When a GROUP_TRACKLET is used, the carrier participant must traverse via the
	#    deterministic chain CONT edge(s), and the second participant must traverse via
	#    the group MERGE/SPLIT structure, except for boundary substitutes:
	#      - Start boundary: second participant already present at t=0 (extra BIRTH capacity).
	#      - End boundary: still merged at end (extra DEATH capacity).
	#
	# If required MERGE/SPLIT/BIRTH/DEATH/CONT edges are missing for a group episode,
	# we force group usage to 0 (never make the model infeasible).
	# ------------------------------------------------------------
	if group_boundary_window_frames < 0:
		raise ValueError("group_boundary_window_frames must be >= 0")

	# Clip boundary frames (for boundary substitutes)
	track_nodes = nodes[nodes["node_type"].astype(str).isin(["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"])].copy()
	clip_first_frame = 0
	clip_last_frame = 0
	if len(track_nodes) > 0 and "end_frame" in track_nodes.columns:
		end_frames = pd.to_numeric(track_nodes["end_frame"], errors="coerce").dropna()
		if len(end_frames) > 0:
			clip_last_frame = int(end_frames.max())

	# Edge-type lookup for adjacency filtering
	edge_type_by_id: Dict[str, str] = {}
	for _, e in edges.iterrows():
		edge_type_by_id[str(e["edge_id"])] = str(e.get("edge_type"))

	# ------------------------------------------------------------
	# Segment connectivity constraint (Worker I)
	# For structural SINGLE_TRACKLET→SINGLE_TRACKLET edges that connect
	# segments of the SAME base_tracklet_id, enforce directed joint-usage
	# along time: flow_in[v] >= flow_in[u].
	# (Equality is too strong and can over-constrain valid graphs.)
	# ------------------------------------------------------------
	if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		single_ix = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].set_index("node_id", drop=False)
		# Map node_id -> base_tracklet_id for SINGLE_TRACKLET nodes
		node_base: Dict[str, str] = {}
		for nid, r in single_ix.iterrows():
			bt = r.get("base_tracklet_id", None)
			if bt is None or (isinstance(bt, float) and math.isnan(bt)):
				continue
			node_base[str(r["node_id"])] = str(bt)
		for _, e in edges.iterrows():
			u = str(e["u"])
			v = str(e["v"])
			etype = str(e.get("edge_type"))
			# "structural (non-group)" constraint applied only on CONTINUE edges
			# where both endpoints are SINGLE_TRACKLET and share base_tracklet_id.
			if etype != "EdgeType.CONTINUE":
				continue
			bu = node_base.get(u, None)
			bv = node_base.get(v, None)
			if bu is None or bv is None:
				continue
			if bu != bv:
				continue
			# Directed implication along time.
			model.Add(flow_in_by_node[v] >= flow_in_by_node[u])

	# ------------------------------------------------------------
	# Cannot-link enforcement (Worker I)
	# Semantics (PM-confirmed): cannot-link means "must not be the same entity".
	# Implemented structurally by forbidding identity-continuation edges (EdgeType.CONTINUE)
	# that stitch across the cannot-link pair.
	# ------------------------------------------------------------
	if constraints is not None:
		cl_pairs = constraints.get("cannot_link_pairs", None)
		if isinstance(cl_pairs, list) and len(cl_pairs) > 0:
			# Precompute SINGLE_TRACKLET node -> base_tracklet_id mapping (if available).
			node_base2: Dict[str, str] = {}
			if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
				single2 = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
				if "base_tracklet_id" in single2.columns:
					single2["base_tracklet_id"] = single2["base_tracklet_id"].astype(str)
					for _, r in single2.iterrows():
						node_base2[str(r["node_id"])] = str(r["base_tracklet_id"])
				# Disable only CONTINUE edges crossing the pair.
				for pair in cl_pairs:
					if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
						continue
					a = str(pair[0])
					b = str(pair[1])
					for _, e in edges.iterrows():
						etype = str(e.get("edge_type"))
						if etype != "EdgeType.CONTINUE":
							continue
						u = str(e["u"])
						v = str(e["v"])
						bu = node_base2.get(u, None)
						bv = node_base2.get(v, None)
						if bu is None or bv is None:
							continue
						if (bu == a and bv == b) or (bu == b and bv == a):
							eid = str(e["edge_id"])
							if eid in var_by_edge:
								model.Add(var_by_edge[eid] == 0)

		# ------------------------------------------------------------
		# Precompute CONTINUE edges by base-tracklet pair (SINGLE↔SINGLE only).
		# Used for safe, conditional group-derived tightening without relying on CONTINUE(d,n).
		# ------------------------------------------------------------
		node_base_single: Dict[str, str] = {}
		if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
			single_tmp = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			if len(single_tmp) > 0 and "base_tracklet_id" in single_tmp.columns:
				single_tmp["node_id"] = single_tmp["node_id"].astype(str)
				single_tmp["base_tracklet_id"] = single_tmp["base_tracklet_id"].astype(str)
				for _, r in single_tmp.iterrows():
					node_base_single[str(r["node_id"])] = str(r["base_tracklet_id"])

		def _pair_key(a: str, b: str) -> Tuple[str, str]:
			return (a, b) if a <= b else (b, a)

		continue_edges_by_pair: Dict[Tuple[str, str], List[str]] = {}
		if len(node_base_single) > 0:
			for _, e in edges.iterrows():
				etype = str(e.get("edge_type"))
				if etype != "EdgeType.CONTINUE":
					continue
				u = str(e["u"])
				v = str(e["v"])
				bu = node_base_single.get(u, None)
				bv = node_base_single.get(v, None)
				if bu is None or bv is None:
					continue
				# Skip same-base segment connectivity; handled elsewhere.
				if bu == bv:
					continue
				key = _pair_key(bu, bv)
				eid = str(e["edge_id"])
				if eid in var_by_edge:
					continue_edges_by_pair.setdefault(key, []).append(eid)

		def _continue_eids(a: str, b: str) -> List[str]:
			return continue_edges_by_pair.get(_pair_key(a, b), [])

		# Optional: AprilTag-derived must-link pairs can supersede group-derived cannot-links.
		tag_must_link: Set[Tuple[str, str]] = set()
		if constraints is not None:
			tml = constraints.get("tag_must_link_pairs", None)
			if isinstance(tml, list):
				for pair in tml:
					if isinstance(pair, (list, tuple)) and len(pair) == 2:
						tag_must_link.add(_pair_key(str(pair[0]), str(pair[1])))

	# For each GROUP_TRACKLET node, enforce structural overlap semantics
	if "carrier_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		groups = nodes[nodes["node_type"].astype(str) == "NodeType.GROUP_TRACKLET"].copy()
		for _, gn in groups.iterrows():
			gid = str(gn["node_id"])

			used = model.NewBoolVar(f"g_used_{gid}")

			# Total flow through group is either 0 or 2.
			in_vars = [var_by_edge[eid] for eid in in_edges.get(gid, [])]
			model.Add(sum(in_vars) == 2 * used)

			# Determine boundary eligibility
			gs = gn.get("start_frame", None)
			ge = gn.get("end_frame", None)
			try:
				gs_i = int(gs) if gs is not None and not (isinstance(gs, float) and math.isnan(gs)) else None
			except Exception:
				gs_i = None
			try:
				ge_i = int(ge) if ge is not None and not (isinstance(ge, float) and math.isnan(ge)) else None
			except Exception:
				ge_i = None

			is_start_boundary = bool(gs_i is not None and gs_i <= (clip_first_frame + int(group_boundary_window_frames) - 1))
			is_end_boundary = bool(ge_i is not None and ge_i >= (clip_last_frame - int(group_boundary_window_frames) + 1))

			# Helper: pick deterministic carrier-chain CONT edges (exclude reconnect edges)
			cont_in = [eid for eid in in_edges.get(gid, []) if str(eid).startswith("E:CONT:") and edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE"]
			cont_out = [eid for eid in out_edges.get(gid, []) if str(eid).startswith("E:CONT:") and edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE"]

			# There should be at most one chain CONT in/out; if multiple exist, keep the first by edge_id ordering.
			cont_in_eid = cont_in[0] if len(cont_in) > 0 else None
			cont_out_eid = cont_out[0] if len(cont_out) > 0 else None

			# Second participant structural edges
			merge_in = [eid for eid in in_edges.get(gid, []) if edge_type_by_id.get(str(eid)) == "EdgeType.MERGE"]
			split_out = [eid for eid in out_edges.get(gid, []) if edge_type_by_id.get(str(eid)) == "EdgeType.SPLIT"]
			birth_in = f"E:BIRTH:{gid}"
			death_out = f"E:DEATH:{gid}"
			local_start_boundary = (cont_in_eid is None) and (birth_in in var_by_edge)

			# Metadata flags
			disp = gn.get("disappearing_tracklet_id", None)
			new_tid = gn.get("new_tracklet_id", None)
			has_disp = disp is not None and pd.notna(disp) and str(disp) != "none"
			has_new = new_tid is not None and pd.notna(new_tid) and str(new_tid) != "none"

			# ------------------------------------------------------------
			# Safe group-based identity tightening (PM-owned, stable):
			# When a group episode is USED, prevent the carrier identity from being
			# stitched (via CONTINUE) to the other participant tids implied by the group.
			#
			# This does NOT require CONTINUE(d,n) to exist.
			# It only disables CONTINUE edges for (carrier, disappearing) and (carrier, new),
			# gated by group usage. AprilTag-derived must-links (if provided) supersede.
			# ------------------------------------------------------------
			def _norm_tid(x: Any) -> str | None:
				if x is None:
					return None
				if isinstance(x, float) and math.isnan(x):
					return None
				s = str(x)
				if s == "none":
					return None
				return s

			c_tid = _norm_tid(gn.get("carrier_tracklet_id", None))
			d_tid = _norm_tid(gn.get("disappearing_tracklet_id", None))
			n_tid = _norm_tid(gn.get("new_tracklet_id", None))

			# Carrier cannot-link to disappearing/new, conditional on group usage.
			for a, b in ((c_tid, d_tid), (c_tid, n_tid)):
				if a is None or b is None:
					continue
				key = _pair_key(str(a), str(b))
				# If AprilTag evidence says these must-link, do not apply group-derived cannot-link.
				if key in tag_must_link:
					continue
				for eid in _continue_eids(str(a), str(b)):
					model.Add(var_by_edge[eid] == 0).OnlyEnforceIf(used)

			# Carrier participant must traverse via chain CONT when not at boundaries.
			if cont_in_eid is not None:
				model.Add(var_by_edge[cont_in_eid] == used)
			else:
				# Start of carrier chain: carrier enters via BIRTH
				if birth_in in var_by_edge:
					# When used, both participants are present at start-boundary (2 units via BIRTH)
					model.Add(var_by_edge[birth_in] == 2 * used)
				else:
					model.Add(used == 0)

			if cont_out_eid is not None:
				model.Add(var_by_edge[cont_out_eid] == used)
			else:
				# End of carrier chain: carrier exits via DEATH
				if death_out in var_by_edge:
					model.Add(var_by_edge[death_out] == 2 * used)
				else:
					model.Add(used == 0)

			# Second participant must enter via MERGE unless start-boundary substitute applies.
			if has_disp:
				if len(merge_in) == 0:
					model.Add(used == 0)
				else:
					# D1 emits exactly one MERGE edge into this group segment.
					model.Add(var_by_edge[merge_in[0]] == used)
					# Disallow any additional MERGE edges if present.
					for extra in merge_in[1:]:
						model.Add(var_by_edge[extra] == 0)
			else:
				# No disappearing participant: legal if group is at start boundary OR if the carrier's first
				# observation begins already-merged (local_start_boundary: no CONT-in, but BIRTH exists).
				if not (is_start_boundary or local_start_boundary):
					model.Add(used == 0)
				# Disallow MERGE edges in this case.
				for eid in merge_in:
					model.Add(var_by_edge[eid] == 0)

			# Second participant must exit via SPLIT unless end-boundary substitute applies.
			if has_new:
				if len(split_out) == 0:
					model.Add(used == 0)
				else:
					model.Add(var_by_edge[split_out[0]] == used)
					# Split ownership (implication): if group is used, the new tracklet must be considered used.
					tid_new = str(new_tid)
					if tid_new in use_tid:
						model.Add(used <= use_tid[tid_new])
					else:
						use_tid[tid_new] = model.NewBoolVar(f"tid_used_{tid_new}")
						model.Add(used <= use_tid[tid_new])
					for extra in split_out[1:]:
						model.Add(var_by_edge[extra] == 0)
			else:
				# No new participant: only legal if group is at end boundary.
				if not is_end_boundary:
					model.Add(used == 0)
				for eid in split_out:
					model.Add(var_by_edge[eid] == 0)

	# ------------------------------------------------------------
	# Coverage policy (Worker I, manager-locked)
	#
	# Soft coverage with strong penalty:
	#  - use_tid[tid] ∈ {0,1} indicates whether base tracklet tid is included in the explanation.
	#  - For each segment node n in tid: flow_in[n] <= use_tid[tid]
	#  - If use_tid[tid]=1, at least one segment is used: sum_n flow_in[n] >= use_tid[tid]
	#  - Penalize dropped tracklets once per base_tracklet_id: penalty * (1 - use_tid[tid])
	# ------------------------------------------------------------
	tracklet_penalty_scaled: int | None = None
	drop_var_by_tid: Dict[str, cp_model.IntVar] = {}

	if unexplained_tracklet_penalty is not None and float(unexplained_tracklet_penalty) > 0:
		tracklet_penalty_scaled = int(round(float(unexplained_tracklet_penalty) * float(scale)))

		if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
			single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)

			for tid, grp in single.groupby("base_tracklet_id", sort=True):
				tid = str(tid)
				# Ensure var exists even if precomputed block didn't create it (defensive).
				if tid not in use_tid:
					use_tid[tid] = model.NewBoolVar(f"tid_used_{tid}")
				node_list = [str(x) for x in grp["node_id"].astype(str).tolist()]
				# Per-segment: flow_in <= use_tid
				for nid in node_list:
					model.Add(flow_in_by_node[nid] <= use_tid[tid])
				# If used, at least one segment must carry flow.
				model.Add(sum(flow_in_by_node[nid] for nid in node_list) >= use_tid[tid])
				# Drop var and linkage
				drop = model.NewBoolVar(f"tid_drop_{tid}")
				model.Add(drop + use_tid[tid] == 1)
				drop_var_by_tid[tid] = drop

	# Objective
	terms = []
	for _, e in edges.iterrows():
		eid = str(e["edge_id"])
		coef = int(cost_int[eid])
		terms.append(coef * var_by_edge[eid])
	# Add unexplained-tracklet penalties (if enabled)
	if tracklet_penalty_scaled is not None and tracklet_penalty_scaled > 0:
		for tid, drop in drop_var_by_tid.items():
			terms.append(int(tracklet_penalty_scaled) * drop)
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

	n_tracklets_total = int(len(drop_var_by_tid))
	n_tracklets_explained = 0
	n_tracklets_unexplained = 0
	if status in ("OPTIMAL", "FEASIBLE") and n_tracklets_total > 0:
		for tid, drop in drop_var_by_tid.items():
			if int(solver.Value(drop)) == 1:
				n_tracklets_unexplained += 1
			else:
				n_tracklets_explained += 1

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
	*,
	compiled: CompiledInputs,
	layout: ClipOutputLayout,
	manifest: ClipManifest,
	checkpoint: str,
	unexplained_tracklet_penalty: float | None = None,
	group_boundary_window_frames: int = 10,
) -> ILPResult:
	"""POC_1 wrapper: solve + write debug outputs + audit summary."""
	res = solve_structure_ilp_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		constraints=compiled.constraints,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		group_boundary_window_frames=int(group_boundary_window_frames),
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