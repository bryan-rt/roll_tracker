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


def test_stage_e1_emits_unmerged_seed_match_sessions(tmp_path: Path) -> None:
	layout = _Layout(tmp_path)

	# Stage D inputs
	stage_d = tmp_path / "stage_D"
	stage_d.mkdir(parents=True, exist_ok=True)

	spans = pd.DataFrame(
		[
			{
				"person_id": "p01",
				"node_id": "N:GROUPISH:1",
				"start_frame": 10,
				"end_frame": 50,
				"effective_cap": 2,
			},
			{
				"person_id": "p02",
				"node_id": "N:GROUPISH:1",
				"start_frame": 20,
				"end_frame": 60,
				"effective_cap": 2,
			},
		]
	)
	spans.to_parquet(layout.person_spans_parquet(), index=False)

	tracks = pd.DataFrame(
		[
			{"person_id": "p01", "frame_index": 20, "timestamp_ms": 2000, "x_m": 0.0, "y_m": 0.0},
			{"person_id": "p01", "frame_index": 50, "timestamp_ms": 5000, "x_m": 0.0, "y_m": 0.0},
		]
	)
	tracks.to_parquet(layout.person_tracks_parquet(), index=False)

	# Identity assignments not used yet, but Stage E checks existence.
	layout.identity_assignments_jsonl().write_text("", encoding="utf-8")

	manifest = ClipManifest(
		clip_id="clip_test",
		camera_id="cam_test",
		input_video_path="dummy.mp4",
		fps=10.0,
		frame_count=100,
		duration_ms=10000,
		pipeline_version="test",
		created_at_ms=0,
	)

	run(
		config={"stages": {"stage_E": {"seed_confidence": 0.77, "max_gap_frames": 30}}},
		inputs={"layout": layout, "manifest": manifest},
	)

	out_path = layout.match_sessions_jsonl()
	assert out_path.exists()

	lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
	assert len(lines) == 1
	rec = lines[0]

	assert rec["artifact_type"] == "match_session"
	assert rec["clip_id"] == "clip_test"
	assert rec["camera_id"] == "cam_test"
	assert rec["person_id_a"] == "p01"
	assert rec["person_id_b"] == "p02"
	assert rec["start_frame"] == 20
	assert rec["end_frame"] == 50
	assert rec["start_ts_ms"] == 2000
	assert rec["end_ts_ms"] == 5000
	assert rec["method"] == "distance_hysteresis_v1"
	assert abs(float(rec["confidence"]) - 0.77) < 1e-9
	assert rec["evidence"]["merge_max_gap_frames"] == 30
	assert rec["evidence"]["seed_node_ids"] == ["N:GROUPISH:1"]
	assert len(rec["evidence"]["seed_match_ids"]) == 1
	assert rec["evidence"]["seed_match_ids"][0].startswith("mseed_")

	validate_match_sessions_records(lines, expected_clip_id="clip_test")
