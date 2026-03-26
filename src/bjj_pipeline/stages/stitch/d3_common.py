"""Shared helpers for Stage D3 solvers and debug artifact writers.

These utilities are solver-agnostic and used by d3_ilp2.py and the D4 emit
pipeline. They were originally defined in d3_ilp.py and extracted during the
consolidation to a single solver (ilp2).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_compile import CompiledInputs


def _debug_dir(layout: ClipOutputLayout) -> Path:
	return layout.clip_root / "_debug"


def _require_columns(df: pd.DataFrame, *, name: str, cols: List[str]) -> None:
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _find_unique_node_id(nodes_df: pd.DataFrame, *, node_type: str) -> str:
	_require_columns(nodes_df, name="d1_graph_nodes", cols=["node_id", "node_type"])
	m = nodes_df[nodes_df["node_type"].astype(str) == node_type]
	if len(m) != 1:
		raise ValueError(f"Expected exactly 1 node with node_type={node_type}, found {len(m)}")
	return str(m.iloc[0]["node_id"])


def _payload_fields_for_logging(payload_json: Any) -> Dict[str, Any]:
	"""Parse a small set of payload_json fields for dev-only diagnostics.

	No solver behavior should depend on this.
	"""
	out: Dict[str, Any] = {
		"payload_json_parse_ok": False,
		"payload_desired_capacity": None,
		"payload_dest_groupish": None,
		"payload_src_groupish": None,
		"payload_promoted_dest": None,
		"payload_promoted_src": None,
		"payload_reconnect": None,
	}
	if not isinstance(payload_json, str) or not payload_json:
		return out
	try:
		payload = json.loads(payload_json)
	except Exception:
		return out
	out["payload_json_parse_ok"] = True
	if not isinstance(payload, dict):
		return out
	try:
		if payload.get("desired_capacity", None) is not None:
			out["payload_desired_capacity"] = int(payload.get("desired_capacity"))
	except Exception:
		out["payload_desired_capacity"] = None
	if "dest_groupish" in payload:
		out["payload_dest_groupish"] = bool(payload.get("dest_groupish"))
	if "src_groupish" in payload:
		out["payload_src_groupish"] = bool(payload.get("src_groupish"))
	if "promoted_dest" in payload:
		out["payload_promoted_dest"] = bool(payload.get("promoted_dest"))
	if "promoted_src" in payload:
		out["payload_promoted_src"] = bool(payload.get("promoted_src"))
	if "reconnect" in payload:
		out["payload_reconnect"] = bool(payload.get("reconnect"))
	return out


def _extract_entity_paths_format_a(
	*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, flow_by_edge_id: Dict[str, int]
) -> List[Dict[str, Any]]:
	"""Decompose selected edge flows into per-entity SOURCE->SINK paths (Format A).

	This is a debug/POC artifact only (not an F0 contract).
	"""
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	nodes["node_id"] = nodes["node_id"].astype(str)
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges["u"] = edges["u"].astype(str)
	edges["v"] = edges["v"].astype(str)

	nodes_ix = nodes.set_index("node_id", drop=False)
	edges_ix = edges.set_index("edge_id", drop=False)

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
		for k in ("start_frame", "end_frame", "base_tracklet_id", "carrier_tracklet_id", "disappearing_tracklet_id", "new_tracklet_id"):
			if k in r.index and pd.notna(r[k]):
				out[k] = int(r[k]) if isinstance(r[k], (int, float)) and str(r[k]).isdigit() else str(r[k])
		return out

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
				raise RuntimeError("Path extraction exceeded step limit; possible cycle in selected edges.")

			choices = out_by_u.get(cur, [])
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

			for nid in (u, v):
				if nid in nodes_ix.index and "base_tracklet_id" in nodes_ix.columns:
					bt = nodes_ix.loc[nid].get("base_tracklet_id")
					if pd.notna(bt):
						bt_s = str(bt)
						if (len(tracklets_in_order) == 0) or (tracklets_in_order[-1] != bt_s):
							tracklets_in_order.append(bt_s)

			steps.append(step)
			cur = v

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
	*, layout: ClipOutputLayout, compiled: CompiledInputs, res: Any, checkpoint: str, manifest: ClipManifest
) -> Path:
	"""Write Format A entity paths to _debug and return output path."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	entities = _extract_entity_paths_format_a(
		nodes_df=compiled.nodes_df, edges_df=compiled.edges_df, flow_by_edge_id=res.flow_by_edge_id
	)

	tag_by_tracklet: Dict[str, Set[str]] = {}
	try:
		groups = compiled.constraints.get("must_link_groups", [])
		if isinstance(groups, list):
			for g in groups:
				if not isinstance(g, dict):
					continue
				anchor = g.get("anchor_key")
				if not (isinstance(anchor, str) and anchor.startswith("tag:")):
					continue
				tids = g.get("tracklet_ids", [])
				if not isinstance(tids, list):
					continue
				for tid in tids:
					if isinstance(tid, str) and tid:
						tag_by_tracklet.setdefault(tid, set()).add(anchor)
	except Exception:
		tag_by_tracklet = {}

	for ent in entities:
		tids = ent.get("tracklets_in_order", [])
		found: Set[str] = set()
		if isinstance(tids, list):
			for tid in tids:
				if isinstance(tid, str) and tid in tag_by_tracklet:
					found |= tag_by_tracklet[tid]
		ent["april_tag_found_in"] = sorted(found) if found else None

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


def _write_solution_ledger_json(
	*,
	layout: ClipOutputLayout,
	compiled: CompiledInputs,
	res: Any,
	checkpoint: str,
	manifest: ClipManifest,
	tag_info: Dict[str, Any] | None = None,
) -> Path:
	"""Write a D3 solution ledger to _debug for full decision transparency."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)
	out = dbg / "d3_solution_ledger.json"

	edges_sel = compiled.edges_df.copy()
	edges_sel["edge_id"] = edges_sel["edge_id"].astype(str)
	edges_sel = edges_sel[edges_sel["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges_sel) > 0:
		edges_sel = edges_sel.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
	else:
		edges_sel = edges_sel.iloc[0:0].copy()

	costs_sel = compiled.costs_df.copy()
	costs_sel["edge_id"] = costs_sel["edge_id"].astype(str)
	costs_sel = costs_sel[costs_sel["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(costs_sel) > 0:
		costs_sel = costs_sel.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	if len(edges_sel) > 0 and len(costs_sel) > 0:
		edges_sel = edges_sel.merge(costs_sel, on="edge_id", how="left", validate="1:1", suffixes=("", "_cost"))

	edges_sel["flow"] = edges_sel["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))

	sum_edge_costs = 0.0
	sum_edge_costs_w_flow = 0.0
	n_edge_instances = 0
	if len(edges_sel) > 0 and "total_cost" in edges_sel.columns:
		for _, r in edges_sel.iterrows():
			try:
				c = float(r.get("total_cost", 0.0))
			except Exception:
				c = 0.0
			f = int(r.get("flow", 0))
			sum_edge_costs += c
			sum_edge_costs_w_flow += c * float(f)
			n_edge_instances += int(f)

	penalty = res.unexplained_tracklet_penalty
	dropped = sorted([str(x) for x in (res.dropped_tracklet_ids or [])])
	explained = sorted([str(x) for x in (res.explained_tracklet_ids or [])])
	sum_penalties = 0.0
	if penalty is not None and penalty > 0 and len(dropped) > 0:
		sum_penalties = float(penalty) * float(len(dropped))

	term_cols = [c for c in edges_sel.columns if isinstance(c, str) and c.startswith("term_")]
	term_cols = sorted(term_cols)
	feature_allow = [
		"dt_frames",
		"dt_s",
		"dist_m",
		"v_req_mps",
		"dist_norm",
		"contact_rel",
		"endpoint_flagged",
	]
	selected_edges: List[Dict[str, Any]] = []
	for _, r in edges_sel.iterrows():
		rec: Dict[str, Any] = {
			"edge_id": str(r.get("edge_id")),
			"flow": int(r.get("flow", 0)),
			"edge_type": str(r.get("edge_type")),
			"src_node_id": str(r.get("u")),
			"dst_node_id": str(r.get("v")),
		}
		cap_raw = r.get("capacity", None)
		cap_eff_df = r.get("capacity_eff", None) if "capacity_eff" in edges_sel.columns else None
		try:
			rec["capacity_raw"] = int(cap_raw) if cap_raw is not None and not pd.isna(cap_raw) else None
		except Exception:
			rec["capacity_raw"] = None
		payload_fields = _payload_fields_for_logging(r.get("payload_json", None))
		desired_cap = payload_fields.get("payload_desired_capacity")
		try:
			rec["payload_desired_capacity"] = int(desired_cap) if desired_cap is not None else None
		except Exception:
			rec["payload_desired_capacity"] = None

		cap_eff_calc: int | None = None
		try:
			if isinstance(rec.get("capacity_raw"), int) and isinstance(rec.get("payload_desired_capacity"), int):
				cap_eff_calc = max(int(rec["capacity_raw"]), int(rec["payload_desired_capacity"]))
		except Exception:
			cap_eff_calc = None
		if cap_eff_calc is None:
			try:
				cap_eff_calc = int(cap_eff_df) if cap_eff_df is not None and not pd.isna(cap_eff_df) else None
			except Exception:
				cap_eff_calc = None
		if cap_eff_calc is None:
			cap_eff_calc = rec.get("capacity_raw")
		rec["capacity_eff"] = cap_eff_calc
		if isinstance(rec.get("capacity_eff"), int):
			rec["expected_var_domain"] = f"0..{rec.get('capacity_eff')}"
		if isinstance(rec.get("payload_desired_capacity"), int) and isinstance(rec.get("capacity_eff"), int):
			rec["desired_gt_eff"] = bool(rec["payload_desired_capacity"] > rec["capacity_eff"])
		try:
			rec["total_cost"] = float(r.get("total_cost"))
		except Exception:
			rec["total_cost"] = None

		terms: Dict[str, Any] = {}
		for c in term_cols:
			val = r.get(c, None)
			try:
				terms[str(c)] = float(val) if val is not None and (not pd.isna(val)) else None
			except Exception:
				terms[str(c)] = None
		rec["terms"] = terms

		feats: Dict[str, Any] = {}
		for c in feature_allow:
			if c not in edges_sel.columns:
				continue
			val = r.get(c, None)
			if val is None or (isinstance(val, float) and math.isnan(val)):
				feats[c] = None
			else:
				if c == "dt_frames":
					try:
						feats[c] = int(val)
					except Exception:
						feats[c] = None
				elif c == "endpoint_flagged":
					feats[c] = bool(val)
				else:
					try:
						feats[c] = float(val)
					except Exception:
						feats[c] = None
		rec["features"] = feats
		selected_edges.append(rec)

	capacity_summary = {
		"num_selected_edges_with_flow_gt_1": int(
			sum(1 for e in selected_edges if int(e.get("flow", 0)) > 1)
		),
		"num_selected_edges_payload_desired_capacity_gt_1": int(
			sum(
				1
				for e in selected_edges
				if isinstance(e.get("payload_desired_capacity"), int)
				and e.get("payload_desired_capacity") > 1
			)
		),
		"num_selected_edges_where_desired_gt_eff": int(
			sum(1 for e in selected_edges if e.get("desired_gt_eff") is True)
		),
		"num_all_edges_payload_desired_capacity_gt_1": int(
			sum(
				1
				for _, rr in compiled.edges_df.iterrows()
				if isinstance(
					_payload_fields_for_logging(rr.get("payload_json")).get("payload_desired_capacity"),
					(int, float),
				)
				and int(
					_payload_fields_for_logging(rr.get("payload_json")).get("payload_desired_capacity")
				)
				> 1
			)
		),
	}

	ledger: Dict[str, Any] = {
		"artifact_type": "d3_solution_ledger",
		"schema_version": 1,
		"checkpoint": checkpoint,
		"created_at_ms": int(time.time() * 1000),
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"objective": {
			"status": res.status,
			"objective_scaled": res.objective_scaled,
			"objective_value": res.objective_value,
			"cost_scale": res.cost_scale,
			"sum_selected_edge_costs": float(sum_edge_costs),
			"sum_selected_edge_costs_weighted_by_flow": float(sum_edge_costs_w_flow),
			"sum_unexplained_penalties": float(sum_penalties),
			"n_selected_edges": int(len(res.selected_edge_ids)),
			"n_selected_edge_instances": int(n_edge_instances),
			"n_tracklets_total": int(res.n_tracklets_total),
			"n_tracklets_explained": int(res.n_tracklets_explained),
			"n_tracklets_unexplained": int(res.n_tracklets_unexplained),
			"unexplained_tracklet_penalty": res.unexplained_tracklet_penalty,
		},
		"rounding": {
			"rounding_n_edges": res.rounding_n_edges,
			"rounding_n_edges_nonzero": res.rounding_n_edges_nonzero,
			"rounding_max_abs_scaled_error": res.rounding_max_abs_scaled_error,
			"rounding_max_abs_cost_error": res.rounding_max_abs_cost_error,
		},
		"dropped_tracklets": [
			{"base_tracklet_id": tid, "penalty": float(penalty) if penalty is not None else 0.0} for tid in dropped
		],
		"explained_tracklets": explained,
		"capacity_summary": capacity_summary,
		"selected_edges": selected_edges,
	}
	if tag_info is not None:
		ledger["tags"] = tag_info

	out.write_text(json.dumps(ledger, sort_keys=True, indent=2), encoding="utf-8")
	return out
