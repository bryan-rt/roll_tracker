"""Stage D3 — compile solver inputs (shared foundation for ILP checkpoints).

POC_0 delivers:
	- Load D1 graph + D2 costs/constraints (manifest-driven)
	- Assert edge-cost coverage is well-defined:
		"d2_edge_costs.parquet has exactly one row per d1_graph_edges.parquet.edge_id
		(complete coverage, no duplicates)."
	- Validate D2 normalized constraints ordering/dedup invariants
	- Deterministically prune disallowed edges (is_allowed == False)
	- Emit a single audit summary event to stage_D/audit.jsonl

Later checkpoints reuse this compilation output as the input to ILP solving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event


@dataclass(frozen=True)
class CompiledInputs:
	nodes_df: pd.DataFrame
	edges_df: pd.DataFrame
	costs_df: pd.DataFrame
	constraints: Dict[str, Any]
	stats: Dict[str, Any]


def _abs_path(layout: ClipOutputLayout, relpath: str) -> Path:
	return layout.clip_root / relpath


def _load_parquet(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing parquet: {path}")
	return pd.read_parquet(path)


def _load_json(path: Path) -> Dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Missing json: {path}")
	return json.loads(path.read_text(encoding="utf-8"))


def _sorted_unique(xs: Iterable[str]) -> List[str]:
	return sorted(set(xs))


def _assert_edge_costs_cover_edges(edges_df: pd.DataFrame, costs_df: pd.DataFrame) -> None:
	"""Assert join integrity between D1 edges and D2 edge costs.

	Required invariant:
		D2 must provide exactly one cost row per D1 edge_id
		(complete coverage, no duplicates).
	"""
	if "edge_id" not in edges_df.columns:
		raise ValueError("d1_graph_edges missing required column: edge_id")
	if "edge_id" not in costs_df.columns:
		raise ValueError("d2_edge_costs missing required column: edge_id")

	edge_ids = edges_df["edge_id"].astype(str)
	cost_edge_ids = costs_df["edge_id"].astype(str)

	dup = cost_edge_ids[cost_edge_ids.duplicated()].tolist()
	missing = sorted(set(edge_ids.tolist()) - set(cost_edge_ids.tolist()))
	extra = sorted(set(cost_edge_ids.tolist()) - set(edge_ids.tolist()))

	if dup or missing or extra:
		raise ValueError(
			"D2 edge costs must map exactly one row per D1 edge_id. "
			f"duplicates={_sorted_unique(dup)[:25]} missing={missing[:25]} extra={extra[:25]}"
		)


def _validate_constraints_canonical(constraints: Dict[str, Any]) -> None:
	"""Validate D2 normalized constraints ordering/dedup invariants.

	We validate only invariants implied by the current D2 writer:
		- must_link_groups sorted by anchor_key
		- each tracklet_ids list sorted unique
		- cannot_link_pairs are canonical [a,b] with a < b
		- cannot_link_pairs sorted unique
	"""
	if not isinstance(constraints, dict):
		raise ValueError("d2_constraints is not a dict")
	for k in ("must_link_groups", "cannot_link_pairs", "stats"):
		if k not in constraints:
			raise ValueError(f"d2_constraints missing required top-level key: {k}")

	groups = constraints.get("must_link_groups")
	pairs = constraints.get("cannot_link_pairs")

	if not isinstance(groups, list):
		raise ValueError("d2_constraints.must_link_groups must be a list")
	if not isinstance(pairs, list):
		raise ValueError("d2_constraints.cannot_link_pairs must be a list")

	# must_link_groups: sorted by anchor_key
	anchor_keys: List[str] = []
	for g in groups:
		if not isinstance(g, dict):
			raise ValueError("must_link_groups contains a non-dict entry")
		if "anchor_key" not in g or "tracklet_ids" not in g:
			raise ValueError("must_link_groups entry missing anchor_key or tracklet_ids")
		ak = str(g["anchor_key"])
		tids = g["tracklet_ids"]
		if not isinstance(tids, list):
			raise ValueError(f"must_link_groups[{ak}].tracklet_ids must be a list")
		tid_strs = [str(x) for x in tids]
		if tid_strs != sorted(tid_strs):
			raise ValueError(f"must_link_groups[{ak}] tracklet_ids not sorted")
		if len(tid_strs) != len(set(tid_strs)):
			raise ValueError(f"must_link_groups[{ak}] tracklet_ids contains duplicates")
		if len(tid_strs) == 0:
			raise ValueError(f"must_link_groups[{ak}] tracklet_ids is empty")
		anchor_keys.append(ak)
	if anchor_keys != sorted(anchor_keys):
		raise ValueError("must_link_groups not sorted by anchor_key")

	# cannot_link_pairs: canonical + sorted
	canonical_pairs: List[Tuple[str, str]] = []
	for p in pairs:
		if not isinstance(p, list) or len(p) != 2:
			raise ValueError("cannot_link_pairs must be a list of [a,b] pairs")
		a, b = str(p[0]), str(p[1])
		if not (a < b):
			raise ValueError(f"cannot_link_pairs contains non-canonical pair: [{a},{b}]")
		canonical_pairs.append((a, b))
	if canonical_pairs != sorted(canonical_pairs):
		raise ValueError("cannot_link_pairs not sorted")
	if len(canonical_pairs) != len(set(canonical_pairs)):
		raise ValueError("cannot_link_pairs contains duplicates")


def _prune_disallowed_edges(
	*, edges_df: pd.DataFrame, costs_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
	"""Drop edges marked disallowed by D2 (deterministic)."""
	if "is_allowed" not in costs_df.columns:
		raise ValueError("d2_edge_costs missing required column: is_allowed")

	# Normalize edge_id to string for joins.
	edges = edges_df.copy()
	costs = costs_df.copy()
	edges["edge_id"] = edges["edge_id"].astype(str)
	costs["edge_id"] = costs["edge_id"].astype(str)

	before = int(len(edges))
	allowed_ids = set(costs.loc[costs["is_allowed"].astype(bool), "edge_id"].tolist())
	edges = edges[edges["edge_id"].isin(allowed_ids)].copy()

	# Deterministic ordering by edge_id
	edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	# Align costs to pruned/ordered edges.
	costs = costs[costs["edge_id"].isin(allowed_ids)].copy()
	costs = costs.set_index("edge_id").loc[edges["edge_id"].tolist()].reset_index()

	after = int(len(edges))
	stats = {"n_edges_before": before, "n_edges_after": after, "n_edges_pruned": before - after}
	return edges, costs, stats


def _count_disallow_reasons(costs_df: pd.DataFrame) -> Dict[str, int]:
	if "disallow_reasons_json" not in costs_df.columns:
		return {}
	counts: Dict[str, int] = {}
	for raw in costs_df["disallow_reasons_json"].dropna().tolist():
		try:
			reasons = json.loads(raw) if isinstance(raw, str) else []
		except Exception:
			reasons = []
		if not isinstance(reasons, list):
			continue
		for r in reasons:
			key = str(r)
			counts[key] = counts.get(key, 0) + 1
	return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _debug_dir(layout: ClipOutputLayout) -> Path:
	return layout.clip_root / "_debug"


def _write_debug_compiled_inputs(
	*, layout: ClipOutputLayout, edges_df: pd.DataFrame, costs_df: pd.DataFrame, constraints: Dict[str, Any], stats: Dict[str, Any]
) -> Dict[str, str]:
	"""Write pruned solver inputs to _debug for transparency.

	Returns a dict of relative paths (relative to clip_root) suitable for audit logging.
	"""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)
	out_edges = dbg / "d3_compiled_edges_pruned.parquet"
	out_costs = dbg / "d3_compiled_costs_pruned.parquet"
	out_constraints = dbg / "d3_constraints_snapshot.json"
	out_stats = dbg / "d3_compile_stats.json"

	edges_df.to_parquet(out_edges, index=False)
	costs_df.to_parquet(out_costs, index=False)
	out_constraints.write_text(json.dumps(constraints, sort_keys=True, indent=2), encoding="utf-8")
	out_stats.write_text(json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8")

	def rel(p: Path) -> str:
		return str(p.relative_to(layout.clip_root))

	return {
		"d3_compiled_edges_pruned_parquet": rel(out_edges),
		"d3_compiled_costs_pruned_parquet": rel(out_costs),
		"d3_constraints_snapshot_json": rel(out_constraints),
		"d3_compile_stats_json": rel(out_stats),
	}


def compile_solver_inputs(
	*,
	config: Dict[str, Any],
	layout: ClipOutputLayout,
	manifest: ClipManifest,
	checkpoint: str,
) -> CompiledInputs:
	"""Load/validate/prune canonical D1/D2 inputs and emit audit summary."""
	# Resolve inputs via manifest (no stage-internal path assumptions)
	d1_nodes_rel = manifest.get_artifact_path(stage="D", key="d1_graph_nodes_parquet")
	d1_edges_rel = manifest.get_artifact_path(stage="D", key="d1_graph_edges_parquet")
	d2_costs_rel = manifest.get_artifact_path(stage="D", key="d2_edge_costs_parquet")
	d2_constraints_rel = manifest.get_artifact_path(stage="D", key="d2_constraints_json")

	d1_nodes = _load_parquet(_abs_path(layout, d1_nodes_rel))
	d1_edges = _load_parquet(_abs_path(layout, d1_edges_rel))
	d2_costs = _load_parquet(_abs_path(layout, d2_costs_rel))
	d2_constraints = _load_json(_abs_path(layout, d2_constraints_rel))

	_assert_edge_costs_cover_edges(d1_edges, d2_costs)
	_validate_constraints_canonical(d2_constraints)

	edges_pruned, costs_pruned, prune_stats = _prune_disallowed_edges(edges_df=d1_edges, costs_df=d2_costs)

	# Node/edge counts
	node_counts: Dict[str, int] = {}
	if "node_type" in d1_nodes.columns:
		for key, value in d1_nodes["node_type"].astype(str).value_counts().items():
			node_counts[str(key)] = int(value)

	edge_counts: Dict[str, int] = {}
	if "edge_type" in edges_pruned.columns:
		for key, value in edges_pruned["edge_type"].astype(str).value_counts().items():
			edge_counts[str(key)] = int(value)

	stats: Dict[str, Any] = {
		"checkpoint": checkpoint,
		"n_nodes": int(len(d1_nodes)),
		"n_edges": int(len(d1_edges)),
		"n_edges_allowed": int(len(edges_pruned)),
		"nodes_by_type": dict(sorted(node_counts.items(), key=lambda kv: kv[0])),
		"edges_by_type": dict(sorted(edge_counts.items(), key=lambda kv: kv[0])),
		"disallow_reason_counts": _count_disallow_reasons(d2_costs),
		**prune_stats,
	}

	debug_outputs = _write_debug_compiled_inputs(
		layout=layout,
		edges_df=edges_pruned,
		costs_df=costs_pruned,
		constraints=d2_constraints,
		stats=stats,
	)

	append_audit_event(
		layout=layout,
		event={
			"event_type": "d3_compile_summary",
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"checkpoint": checkpoint,
			"stats": stats,
			"debug_outputs": debug_outputs,
		},
	)

	return CompiledInputs(
		nodes_df=d1_nodes,
		edges_df=edges_pruned,
		costs_df=costs_pruned,
		constraints=d2_constraints,
		stats=stats,
	)
