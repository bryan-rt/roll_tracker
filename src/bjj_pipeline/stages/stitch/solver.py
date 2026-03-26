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
	POC_1/POC_2_TAGS: run ILP solve via d3_ilp2; returns solver result.

	Return: (compiled_inputs, ilp_result_or_None)
	"""
	layout = inputs["layout"]
	manifest = inputs["manifest"]

	checkpoint = _cfg_get(config, "stages.stage_D.d3_checkpoint", None)
	if checkpoint is None:
		checkpoint = _cfg_get(config, "stage_D.d3_checkpoint", "POC_0")

	from bjj_pipeline.stages.stitch.d3_compile import compile_solver_inputs

	_ = compile_solver_inputs(config=config, layout=layout, manifest=manifest, checkpoint=str(checkpoint))

	if str(checkpoint) == "POC_0":
		return (_, None)

	if str(checkpoint) in ("POC_1", "POC_2_TAGS"):
		from bjj_pipeline.stages.stitch.d3_ilp2 import solve_structure_ilp2

		# D3 — GROUP_TRACKLET boundary substitute window (frames)
		gbw = _cfg_get(
			config,
			"stages.stage_D.d3.group_boundary_window_frames",
			_cfg_get(config, "stage_D.d3.group_boundary_window_frames", None),
		)
		gbw_i = int(gbw) if gbw is not None else 10

		# D3 — "explain each tracklet or pay a penalty": optional penalty from config.
		penalty = _cfg_get(
			config,
			"stages.stage_D.d3.unexplained_tracklet_penalty",
			_cfg_get(config, "stage_D.d3.unexplained_tracklet_penalty", None),
		)

		res = solve_structure_ilp2(
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
