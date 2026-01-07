"""Stage A runner: Detect + Tracklets (local association).

This module intentionally exposes a stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Slice 2 placeholder: write schema-correct empty artifacts so the pipeline can
run end-to-end before the real detector/tracker is integrated.
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import pandas as pd

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.contracts.f0_validate import (
	validate_detections_df,
	validate_tracklet_tables,
)
from .outputs import empty_df_for_schema_key


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint (placeholder implementation).

	Writes empty schema-correct parquet outputs for Stage A:
	- stage_A/detections.parquet
	- stage_A/tracklet_frames.parquet
	- stage_A/tracklet_summaries.parquet
	Also ensures the stage directory exists and drops a minimal audit line.
	"""
	layout: ClipOutputLayout = inputs["layout"]
	stage_dir = layout.stage_dir("A")
	stage_dir.mkdir(parents=True, exist_ok=True)

	# Build empty dataframes matching canonical schemas
	det_df = empty_df_for_schema_key("detections")
	tf_df = empty_df_for_schema_key("tracklet_frames")
	ts_df = empty_df_for_schema_key("tracklet_summaries")

	# Write parquet outputs
	det_path = layout.detections_parquet()
	tf_path = layout.tracklet_frames_parquet()
	ts_path = layout.tracklet_summaries_parquet()
	det_df.to_parquet(det_path)
	tf_df.to_parquet(tf_path)
	ts_df.to_parquet(ts_path)

	# Validate shape/contracts
	validate_detections_df(pd.read_parquet(det_path))
	validate_tracklet_tables(pd.read_parquet(tf_path), pd.read_parquet(ts_path))

	# Minimal audit to aid debugging; orchestration also appends events
	audit_path = layout.audit_jsonl("A")
	Path(audit_path).write_text("{}\n", encoding="utf-8")

	# Returning is optional; manifest registration is done by the orchestrator
	return {
		"detections_parquet": layout.rel_to_clip_root(det_path),
		"tracklet_frames_parquet": layout.rel_to_clip_root(tf_path),
		"tracklet_summaries_parquet": layout.rel_to_clip_root(ts_path),
		"audit_jsonl": layout.rel_to_clip_root(audit_path),
	}


def main() -> None:
	"""Optional stage-local CLI.

	Prefer running via `roll-tracker run ...` unless debugging Stage A in isolation.
	"""
	raise SystemExit(
		"Stage A (detect_track) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
