"""Stage C runner: AprilTag scanning + identity hints (Slice 2 placeholder).

Writes empty JSONL artifacts so orchestration can run end-to-end:
- stage_C/tag_observations.jsonl (empty file)
- stage_C/identity_hints.jsonl (empty file)
- stage_C/audit.jsonl (minimal line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Write empty JSONL outputs for Stage C."""
	layout: ClipOutputLayout = inputs["layout"]
	layout.ensure_dirs_for_stage("C")

	# Touch empty JSONL files
	Path(layout.identity_hints_jsonl()).write_text("", encoding="utf-8")
	Path(layout.tag_observations_jsonl()).write_text("", encoding="utf-8")

	# Minimal audit entry
	Path(layout.audit_jsonl("C")).write_text(json.dumps({"event":"stage_C_placeholder"})+"\n", encoding="utf-8")

	return {
		"identity_hints_jsonl": layout.rel_to_clip_root(layout.identity_hints_jsonl()),
		"tag_observations_jsonl": layout.rel_to_clip_root(layout.tag_observations_jsonl()),
		"audit_jsonl": layout.rel_to_clip_root(layout.audit_jsonl("C")),
	}


def main() -> None:
	raise SystemExit("Run via roll-tracker CLI; this is a placeholder stage.")
