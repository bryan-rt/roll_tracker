"""Stage D3 — Solver wiring (ILP) + shared input compilation.

This module is the stable home for Stage D3 as we iterate through POC checkpoints.

Checkpoint philosophy:
	- D3 is invoked via stage_D.run_until == "D3".
	- Behavior within D3 is selected by stage_D.d3_checkpoint (default "POC_0").
	- Code is additive: POC_0 compile/validate is the shared foundation for later ILP work.
"""

from __future__ import annotations

from typing import Any, Dict


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
	"""Lightweight dot-path getter (local to D3).

	We keep a local helper to avoid cross-module coupling during POC.
	"""
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur



def run_d3(*, config: Dict[str, Any], inputs: Dict[str, Any]) -> tuple[Any, Any]:
	"""Run Stage D3 (checkpointed).

	POC_0: compile + validate solver inputs (audit-only).
	POC_1/POC_2_TAGS: run ILP solve; returns solver result.
	
	Return: (compiled_inputs, ilp_result_or_None)
	"""
	layout = inputs["layout"]
	manifest = inputs["manifest"]

	checkpoint = _cfg_get(config, "stages.stage_D.d3_checkpoint", None)
	if checkpoint is None:
		checkpoint = _cfg_get(config, "stage_D.d3_checkpoint", "POC_0")

	# Solver implementation toggle: default to ilp1 for stability.
	# ilp2 is an alternate implementation (e.g., multi-commodity per tag work).
	solver_impl = _cfg_get(
		config,
		"stages.stage_D.d3.solver_impl",
		_cfg_get(config, "stage_D.d3.solver_impl", "ilp1"),
	)

	from bjj_pipeline.stages.stitch.d3_compile import compile_solver_inputs

	_ = compile_solver_inputs(config=config, layout=layout, manifest=manifest, checkpoint=str(checkpoint))

	if str(checkpoint) == "POC_0":
		return (_, None)

	if str(checkpoint) == "POC_1":
		if str(solver_impl) == "ilp2":
			from bjj_pipeline.stages.stitch.d3_ilp2 import solve_structure_ilp2 as solve_structure_ilp
		else:
			from bjj_pipeline.stages.stitch.d3_ilp import solve_structure_ilp

		# D3 — "explain each tracklet or pay a penalty": optional penalty from config.
		# D3 — GROUP_TRACKLET boundary substitute window (frames)
		gbw = _cfg_get(
			config,
			"stages.stage_D.d3.group_boundary_window_frames",
			_default_gbw := _cfg_get(config, "stage_D.d3.group_boundary_window_frames", None),
		)
		gbw_i = int(gbw) if gbw is not None else 10

		penalty = _cfg_get(
			config,
			"stages.stage_D.d3.unexplained_tracklet_penalty",
			_default := _cfg_get(config, "stage_D.d3.unexplained_tracklet_penalty", None),
		)
		unexplained_group_ping_penalty = _cfg_get(
			config,
			"stages.stage_D.d3.unexplained_group_ping_penalty",
			_cfg_get(config, "stage_D.d3.unexplained_group_ping_penalty", 5000.0),
		)
		# Tag fragmentation (time-separated): penalty per new fragment start
		tag_frag_pen = _cfg_get(
			config,
			"stages.stage_D.d3.tag_fragment_start_penalty",
			_default2 := _cfg_get(config, "stage_D.d3.tag_fragment_start_penalty", None),
		)
		tag_frag_pen_f = float(tag_frag_pen) if tag_frag_pen is not None else None
		res = solve_structure_ilp(
			compiled=_,
			layout=layout,
			manifest=manifest,
			checkpoint=str(checkpoint),
			unexplained_tracklet_penalty=float(penalty) if penalty is not None else None,
			group_boundary_window_frames=int(gbw_i),
		)
		return (_, res)

	if str(checkpoint) == "POC_2_TAGS":
		if str(solver_impl) == "ilp2":
			# ilp2 is an alternate solver implementation (planned: full multi-commodity flow per tag).
			# For now it should remain explicitly enabled only when you are ready to A/B test.
			from bjj_pipeline.stages.stitch.d3_ilp2 import solve_structure_ilp2 as solve_structure_ilp_tags
		else:
			from bjj_pipeline.stages.stitch.d3_ilp import solve_structure_ilp_tags

		# D3 — "explain each tracklet or pay a penalty": optional penalty from config.
		# D3 — GROUP_TRACKLET boundary substitute window (frames)
		gbw = _cfg_get(
			config,
			"stages.stage_D.d3.group_boundary_window_frames",
			_default_gbw := _cfg_get(config, "stage_D.d3.group_boundary_window_frames", None),
		)
		gbw_i = int(gbw) if gbw is not None else 10

		penalty = _cfg_get(
			config,
			"stages.stage_D.d3.unexplained_tracklet_penalty",
			_default := _cfg_get(config, "stage_D.d3.unexplained_tracklet_penalty", None),
		)
		unexplained_group_ping_penalty = _cfg_get(
			config,
			"stages.stage_D.d3.unexplained_group_ping_penalty",
			_cfg_get(config, "stage_D.d3.unexplained_group_ping_penalty", 5000.0),
		)
		# Tag fragmentation (time-separated): penalty per new fragment start (legacy absolute)
		tag_frag_pen = _cfg_get(
			config,
			"stages.stage_D.d3.tag_fragment_start_penalty",
			_default2 := _cfg_get(config, "stage_D.d3.tag_fragment_start_penalty", 2500.0),
		)
		tag_frag_pen_f = float(tag_frag_pen)

		# Penalty scaling (professional): reference edge cost quantile/min and multiplier/abs knobs
		ref_q = _cfg_get(
			config,
			"stages.stage_D.d3.penalty_ref_edge_cost_quantile",
			_cfg_get(config, "stage_D.d3.penalty_ref_edge_cost_quantile", None),
		)
		ref_min = _cfg_get(
			config,
			"stages.stage_D.d3.penalty_ref_edge_cost_min",
			_cfg_get(config, "stage_D.d3.penalty_ref_edge_cost_min", None),
		)
		solo_mult = _cfg_get(
			config,
			"stages.stage_D.d3.solo_ping_miss_penalty_mult",
			_cfg_get(config, "stage_D.d3.solo_ping_miss_penalty_mult", None),
		)
		group_mult = _cfg_get(
			config,
			"stages.stage_D.d3.group_ping_miss_penalty_mult",
			_cfg_get(config, "stage_D.d3.group_ping_miss_penalty_mult", None),
		)
		solo_abs = _cfg_get(
			config,
			"stages.stage_D.d3.solo_ping_miss_penalty_abs",
			_cfg_get(config, "stage_D.d3.solo_ping_miss_penalty_abs", None),
		)
		group_abs = _cfg_get(
			config,
			"stages.stage_D.d3.group_ping_miss_penalty_abs",
			_cfg_get(config, "stage_D.d3.group_ping_miss_penalty_abs", None),
		)
		frag_mult = _cfg_get(
			config,
			"stages.stage_D.d3.tag_fragment_start_penalty_mult",
			_cfg_get(config, "stage_D.d3.tag_fragment_start_penalty_mult", None),
		)
		frag_abs = _cfg_get(
			config,
			"stages.stage_D.d3.tag_fragment_start_penalty_abs",
			_cfg_get(config, "stage_D.d3.tag_fragment_start_penalty_abs", None),
		)
		ref_q_f = float(ref_q) if ref_q is not None else None
		ref_min_f = float(ref_min) if ref_min is not None else None
		solo_mult_f = float(solo_mult) if solo_mult is not None else None
		group_mult_f = float(group_mult) if group_mult is not None else None
		solo_abs_f = float(solo_abs) if solo_abs is not None else None
		group_abs_f = float(group_abs) if group_abs is not None else None
		frag_mult_f = float(frag_mult) if frag_mult is not None else None
		frag_abs_f = float(frag_abs) if frag_abs is not None else None
		res = solve_structure_ilp_tags(
			compiled=_,
			layout=layout,
			manifest=manifest,
			checkpoint=str(checkpoint),
			unexplained_tracklet_penalty=float(penalty) if penalty is not None else None,
			group_boundary_window_frames=int(gbw_i),
		)
		return (_, res)

	raise NotImplementedError(
		f"Stage D3 checkpoint not implemented: {checkpoint!r}. "
		"Supported today: ['POC_0', 'POC_1', 'POC_2_TAGS']."
	)

