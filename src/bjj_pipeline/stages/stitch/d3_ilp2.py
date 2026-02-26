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

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Any, Dict, List

import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event
from bjj_pipeline.stages.stitch.d3_compile import CompiledInputs

# Breadcrumb constants (written into audit + debug ledger copy).
_SOLVER_IMPL: str = "ilp2"
_SOLVER_MODULE: str = __name__
_SOLVER_VERSION: str = "scaffold_delegate_ilp1"


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


def _emit_mcf_tag_inputs(*, debug_dir: Path, constraints: Optional[Dict[str, Any]]) -> Path:
	"""Emit a small JSON describing the tag inputs we will use for MCF work (MCF-0).

	This file is purely diagnostic and should not affect solver behavior.
	It is intentionally tolerant of multiple possible constraint shapes that D2
	may produce (legacy `must_link_groups`, `tag_pings`, or `identity_hints`).
	"""
	import json
	out = debug_dir / "d3_mcf_tag_inputs.json"
	tmp = debug_dir / (out.name + ".tmp")

	payload: Dict[str, Any] = {"discovered": False, "tags": {}, "notes": []}
	if constraints is None:
		payload["notes"].append("compiled.constraints is None")
	else:
		payload["discovered"] = True
		# Common D2 shapes we expect:
		# - must_link_groups: { "tag:1": ["t5", "t16"], ... }
		# - tag_pings: [ {"tag_id":"tag:1","tracklet_id":"t5","frame":240}, ... ]
		# - identity_hints (stage_C passthrough) - we'll try to be generous
		if "must_link_groups" in constraints and isinstance(constraints["must_link_groups"], dict):
			for tagk, tids in constraints["must_link_groups"].items():
				payload["tags"].setdefault(tagk, {})["must_link_tracklets"] = list(tids or [])
		if "tag_pings" in constraints and isinstance(constraints["tag_pings"], list):
			# Normalize ping list per tag
			for p in constraints["tag_pings"]:
				try:
					tid = p.get("tracklet_id") or p.get("node_id") or p.get("bound_tracklet_id")
					tagk = p.get("tag_id") or p.get("tag") or p.get("anchor_key")
					frame = p.get("frame") or p.get("frame_index") or p.get("timestamp")
				except Exception:
					continue
				if tagk is None:
					continue
				tinfo = payload["tags"].setdefault(tagk, {})
				tinfo.setdefault("pings", []).append({"tracklet_id": tid, "frame": frame, "raw": p})
		# Catch a generic stage_C identity_hints list shape if present
		if "identity_hints" in constraints and isinstance(constraints["identity_hints"], list):
			for h in constraints["identity_hints"]:
				try:
					tagk = h.get("anchor_key") or h.get("tag_id") or h.get("tag")
					tid = h.get("tracklet_id") or h.get("node_id")
					frame = h.get("frame") or h.get("frame_index")
				except Exception:
					continue
				if tagk is None:
					continue
				tinfo = payload["tags"].setdefault(tagk, {})
				tinfo.setdefault("hints", []).append({"tracklet_id": tid, "frame": frame, "raw": h})

		# If nothing recognized, attach a note and copy a small sample
		if not payload["tags"]:
			payload["notes"].append("no recognisable tag structures found; including constraints sample")
			# include small sample (first 10 keys)
			try:
				payload["constraints_sample"] = {}
				for i, (k, v) in enumerate((constraints.items() if isinstance(constraints, dict) else [])):
					if i >= 10:
						break
					payload["constraints_sample"][str(k)] = str(type(v).__name__)
			except Exception:
				pass

	# Write atomically
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)
	tmp.replace(out)
	return out


def _write_solver_breadcrumbs_json(*, debug_dir: Path, payload: Dict[str, Any]) -> Path:
	"""Write a tiny json file to make it provable which solver module ran."""
	import json

	out = debug_dir / "d3_solver_breadcrumbs.json"
	tmp = debug_dir / (out.name + ".tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)
	tmp.replace(out)
	return out


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
	# --- MCF per tag (future) ---
	# tag_pings: list[dict] | None = None,
	# tag_miss_penalty: float | None = None,
	# ...
) -> ILPResult:
	"""Core solver for d3_ilp2.

	For scaffolding, we delegate to the existing d3_ilp.solve_structure_ilp_core.
	When we implement MCF-per-tag, this function becomes the new home.
	"""
	from bjj_pipeline.stages.stitch.d3_ilp import solve_structure_ilp_core as _core
	from bjj_pipeline.stages.stitch.d3_ilp import ILPResult as _ILP1Result

	res1: _ILP1Result = _core(
		nodes_df=nodes_df,
		edges_df=edges_df,
		costs_df=costs_df,
		constraints=constraints,
		debug_dir=debug_dir,
		emit_transparency=emit_transparency,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		group_boundary_window_frames=int(group_boundary_window_frames),
	)
	# Re-wrap to avoid type coupling between modules.
	return ILPResult(**res1.__dict__)


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

	# MCF-0: emit tag input diagnostics so we have a stable starting point for MCF.
	# This is non-behavioral: it only inspects compiled.constraints and emits JSON.
	try:
		out_mcf_inputs = _emit_mcf_tag_inputs(debug_dir=dbg, constraints=compiled.constraints)
	except Exception as e:  # defensive: do not fail the run for diagnostics
		out_mcf_inputs = None
		# We purposely do not raise here; log would be nice, but keep parity.

	t0 = time.time()
	res = solve_structure_ilp2_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		constraints=compiled.constraints,
		debug_dir=dbg,
		emit_transparency=True,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		group_boundary_window_frames=int(group_boundary_window_frames),
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
			},
		},
	)

	return res
