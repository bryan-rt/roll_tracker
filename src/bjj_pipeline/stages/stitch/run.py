"""Stage D runner: Multi-step stitching container (D0..D6).

Exposes the stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Stage D is a multi-step container. For POC we implement only D0:
	- writes Stage D bank tables:
	  - stage_D/tracklet_bank_frames.parquet
	  - stage_D/tracklet_bank_summaries.parquet
	  - stage_D/audit.jsonl

Later workers (D1..D6) will extend the dispatcher via stage_D.run_until.
"""

from __future__ import annotations

from typing import Any, Dict

from bjj_pipeline.contracts.f0_manifest import register_stage_D0_defaults, write_manifest


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
	"""Lightweight dot-path getter (local to Stage D)."""
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint (dispatcher).

	Reads stage_D.run_until (default D0) to decide how far to run.
	"""
	layout = inputs["layout"]
	manifest = inputs["manifest"]

	run_until = _cfg_get(config, "stages.stage_D.run_until", None)
	if run_until is None:
		run_until = _cfg_get(config, "stage_D.run_until", "D0")

	# Ensure stage directory exists
	layout.stage_dir("D").mkdir(parents=True, exist_ok=True)

	outputs: Dict[str, Any] = {"run_until": run_until}

	if run_until in ("D0", "D1", "D2", "D6"):
		from bjj_pipeline.stages.stitch.d0_bank import run_d0

		run_d0(config=config, layout=layout, manifest=manifest)
		register_stage_D0_defaults(manifest, layout)
		write_manifest(manifest, layout.clip_manifest_path())

		# D1: construct candidate graph
		if run_until in ("D1", "D2", "D6"):
			from bjj_pipeline.stages.stitch.d1_graph_build import run_d1

			run_d1(cfg=config, layout=layout, manifest=manifest)

		# Optional visual QA (Checkpoint 2.5): write a mat-space footpath PNG.
		try:
			from bjj_pipeline.viz.stage_d_paths import render_stage_d_paths_png

			render_stage_d_paths_png(config=config, inputs=inputs)
		except Exception as e:
			# Visual QA must never fail the pipeline; audit logging occurs inside the helper.
			print(f"[roll-tracker] Stage D visual QA skipped due to error: {e}")

		print(
			f"[roll-tracker] Stage D dispatcher completed run_until={run_until} outputs="
			f"[{layout.rel_to_clip_root(layout.tracklet_bank_frames_parquet())}, "
			f"{layout.rel_to_clip_root(layout.tracklet_bank_summaries_parquet())}]"
		)
		return outputs

	raise ValueError(f"Unsupported stage_D.run_until={run_until!r}")


def main() -> None:
	raise SystemExit(
		"Stage D (stitch) does not yet implement a standalone CLI; "
		"run via `roll-tracker`."
	)
