"""Stage D runner: Global stitching (Min-Cost Flow).

Exposes the stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Stage D implementation is owned by the Stage D workers. This is a contract stub
so orchestration and tests can reliably patch `run()`.
"""

from __future__ import annotations

from typing import Any, Dict


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint (stub).

	Expected outputs (once implemented):
	- stage_D/person_tracks.parquet
	- stage_D/identity_assignments.jsonl
	- stage_D/audit.jsonl
	"""
	raise NotImplementedError(
		"Stage D (stitch) is not implemented yet. "
		"This stub exists to satisfy the orchestration contract."
	)


def main() -> None:
	raise SystemExit(
		"Stage D (stitch) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
