"""Stage E audit helpers.

We append JSONL audit events to the canonical Stage E audit file:
	outputs/<clip_id>/stage_E/audit.jsonl

This is intentionally lightweight and deterministic.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


def _now_ms() -> int:
	return int(time.time() * 1000)


def append_audit_event(*, layout: ClipOutputLayout, event: Dict[str, Any]) -> None:
	"""Append a single audit event to Stage E audit.jsonl."""
	path = layout.audit_jsonl("E")
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(event, sort_keys=True) + "\n")
