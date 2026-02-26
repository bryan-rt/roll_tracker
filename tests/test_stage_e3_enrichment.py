from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_validate import validate_match_sessions_records
from bjj_pipeline.stages.matches.run import run


@dataclass
class _Layout:
	root: Path

	def person_spans_parquet(self) -> Path:
		return self.root / "stage_D" / "person_spans.parquet"

	def person_tracks_parquet(self) -> Path:
		return self.root / "stage_D" / "person_tracks.parquet"

	def identity_assignments_jsonl(self) -> Path:
		return self.root / "stage_D" / "identity_assignments.jsonl"

	def match_sessions_jsonl(self) -> Path:
		return self.root / "stage_E" / "match_sessions.jsonl"

	def audit_jsonl(self, stage: str) -> Path:
		assert stage == "E"
		return self.root / "stage_E" / "audit.jsonl"


def test_stage_e3_attaches_april_tag_ids_or_null(tmp_path: Path) -> None:
	layout = _Layout(tmp_path)

	stage_d = tmp_path / "stage_D"
	stage_d.mkdir(parents=True, exist_ok=True)

	# One cap2 seed for p01/p02
	spans = pd.DataFrame(
		[
			{"person_id": "p01", "node_id": "G:0-10", "start_frame": 0, "end_frame": 10, "effective_cap": 2},
			{"person_id": "p02", "node_id": "G:0-10", "start_frame": 0, "end_frame": 10, "effective_cap": 2},
		]
	)
	spans.to_parquet(layout.person_spans_parquet(), index=False)

	tracks = pd.DataFrame(
		[
			{"person_id": "p01", "frame_index": 0, "timestamp_ms": 0, "x_m": 0.0, "y_m": 0.0},
			{"person_id": "p01", "frame_index": 10, "timestamp_ms": 1000, "x_m": 0.0, "y_m": 0.0},
		]
	)
	tracks.to_parquet(layout.person_tracks_parquet(), index=False)

	# Identity assignments: p01 tagged, p02 missing -> should be null for b
	ia_lines = [
		{
			"schema_version": "0.3.0",
			"artifact_type": "identity_assignment",
			"clip_id": "clip_test",
			"camera_id": "cam_test",
			"pipeline_version": "test",
			"created_at_ms": 0,
			"person_id": "p01",
			"tag_id": "1",
			"assignment_confidence": 0.9,
			"evidence": {"anchor_key": "tag:1"},
		}
	]
	layout.identity_assignments_jsonl().write_text(
		"\n".join(json.dumps(x) for x in ia_lines) + "\n",
		encoding="utf-8",
	)

	manifest = ClipManifest(
		clip_id="clip_test",
		camera_id="cam_test",
		input_video_path="dummy.mp4",
		fps=10.0,
		frame_count=50,
		duration_ms=5000,
		pipeline_version="test",
		created_at_ms=0,
	)

	run(
		config={"stages": {"stage_E": {"seed_confidence": 0.5, "max_gap_frames": 30}}},
		inputs={"layout": layout, "manifest": manifest},
	)

	lines = [
		json.loads(l)
		for l in layout.match_sessions_jsonl().read_text(encoding="utf-8").splitlines()
		if l.strip()
	]
	assert len(lines) == 1
	rec = lines[0]

	ev = rec["evidence"]
	assert ev["april_tag_id_a"] == "1"
	assert ev["april_tag_id_b"] is None
	assert ev["april_anchor_key_a"] == "tag:1"
	assert ev["april_anchor_key_b"] is None

	validate_match_sessions_records(lines, expected_clip_id="clip_test")
