"""Stage C runner: AprilTag scanning + identity hints (Milestone 1 placeholder).

Writes empty JSONL artifacts so orchestration can run end-to-end:
- stage_C/tag_observations.jsonl (empty file)
- stage_C/identity_hints.jsonl (empty file)
- stage_C/audit.jsonl (header + summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Write *contract-valid* Stage C outputs.

	Milestone 1 behavior:
	- tag_observations.jsonl is empty (0 records)
	- identity_hints.jsonl is empty (0 records)
	- audit.jsonl contains a run header + summary (for evidence/review)
	"""

	layout: ClipOutputLayout = inputs["layout"]
	manifest = inputs["manifest"]
	layout.ensure_dirs_for_stage("C")

	# Touch empty JSONL files
	Path(layout.identity_hints_jsonl()).write_text("", encoding="utf-8")
	Path(layout.tag_observations_jsonl()).write_text("", encoding="utf-8")

	# Audit: header + summary (always present) so we can review artifacts as we build.
	def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
		cur: Any = cfg
		for part in path.split("."):
			if not isinstance(cur, dict) or part not in cur:
				return default
			cur = cur[part]
		return cur

	tag_family_default = _cfg_get(config, "stages.stage_C.tag_family", "36h11")
	sched_cfg = _cfg_get(config, "stages.stage_C.c0_scheduler", {}) or {}
	header = {
		"event": "stage_C_run_header",
		"stage": "C",
		"schema_version": "0",
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"pipeline_version": manifest.pipeline_version,
		"created_at_ms": int(getattr(manifest, "created_at_ms", 0) or 0),
		"mode": "multipass",
		"tag_family_default": str(tag_family_default) if tag_family_default is not None else "36h11",
		"scheduler": {
			"enabled": bool(sched_cfg.get("enabled", True)),
			"k_seek": int(sched_cfg.get("k_seek", 1)),
			"k_verify": int(sched_cfg.get("k_verify", 30)),
			"n_ramp": int(sched_cfg.get("n_ramp", 60)),
		},
		"note": "stage_C_enabled_no_decode_yet",
	}
	summary = {
		"event": "stage_C_run_summary",
		"stage": "C",
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"counters": {
			"total_candidates_seen": 0,
			"total_scheduled_attempts": 0,
			"total_skipped": 0,
			"skips_by_reason": {},
			"total_decode_attempts": 0,
			"total_tag_observations_emitted": 0,
			"total_identity_hints_emitted": 0,
		},
	}
	Path(layout.audit_jsonl("C")).write_text(
		json.dumps(header) + "\n" + json.dumps(summary) + "\n",
		encoding="utf-8",
	)

	return {
		"identity_hints_jsonl": layout.rel_to_clip_root(layout.identity_hints_jsonl()),
		"tag_observations_jsonl": layout.rel_to_clip_root(layout.tag_observations_jsonl()),
		"audit_jsonl": layout.rel_to_clip_root(layout.audit_jsonl("C")),
	}


def main() -> None:
    raise SystemExit("Run via roll-tracker CLI; this is a placeholder stage.")
