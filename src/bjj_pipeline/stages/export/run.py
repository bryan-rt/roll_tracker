"""Stage F runner: Export + privacy + database persistence.

Exposes the stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Stage F implementation is owned by the Stage F workers. This is a contract stub
so orchestration and tests can reliably patch `run()`.
"""

from __future__ import annotations

from typing import Any, Dict


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint (stub).

	Expected outputs (once implemented):
	- stage_F/export_manifest.jsonl
	- exported mp4 clips under stage_F/exports/
	- stage_F/audit.jsonl
	"""
	raise NotImplementedError(
		"Stage F (export) is not implemented yet. "
		"This stub exists to satisfy the orchestration contract."
	)


def main() -> None:
	raise SystemExit(
		"Stage F (export) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
