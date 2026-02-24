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


def test_stage_e2_merges_seeds_within_gap(tmp_path: Path) -> None:
	layout = _Layout(tmp_path)

	stage_d = tmp_path / "stage_D"
	stage_d.mkdir(parents=True, exist_ok=True)

	# Same pair, two cap2 nodes with a gap of 9 frames -> should merge with max_gap_frames=30
	# Third node far away -> should remain separate.
	spans = pd.DataFrame(
		[
			{"person_id": "p01", "node_id": "G:0-10", "start_frame": 0, "end_frame": 10, "effective_cap": 2},
			{"person_id": "p02", "node_id": "G:0-10", "start_frame": 0, "end_frame": 10, "effective_cap": 2},
			{"person_id": "p01", "node_id": "G:20-30", "start_frame": 20, "end_frame": 30, "effective_cap": 2},
			{"person_id": "p02", "node_id": "G:20-30", "start_frame": 20, "end_frame": 30, "effective_cap": 2},
			{"person_id": "p01", "node_id": "G:100-110", "start_frame": 100, "end_frame": 110, "effective_cap": 2},
			{"person_id": "p02", "node_id": "G:100-110", "start_frame": 100, "end_frame": 110, "effective_cap": 2},
		]
	)
	spans.to_parquet(layout.person_spans_parquet(), index=False)

	# Provide timestamps for the merged boundaries
	tracks = pd.DataFrame(
		[
			{"person_id": "p01", "frame_index": 0, "timestamp_ms": 0, "x_m": 0.0, "y_m": 0.0},
			{"person_id": "p01", "frame_index": 30, "timestamp_ms": 3000, "x_m": 0.0, "y_m": 0.0},
			{"person_id": "p01", "frame_index": 100, "timestamp_ms": 10000, "x_m": 0.0, "y_m": 0.0},
			{"person_id": "p01", "frame_index": 110, "timestamp_ms": 11000, "x_m": 0.0, "y_m": 0.0},
		]
	)
	tracks.to_parquet(layout.person_tracks_parquet(), index=False)

	layout.identity_assignments_jsonl().write_text("", encoding="utf-8")

	manifest = ClipManifest(
		clip_id="clip_test",
		camera_id="cam_test",
		input_video_path="dummy.mp4",
		fps=10.0,
		frame_count=200,
		duration_ms=20000,
		pipeline_version="test",
		created_at_ms=0,
	)

	run(
		config={"stages": {"stage_E": {"seed_confidence": 0.5, "max_gap_frames": 30}}},
		inputs={"layout": layout, "manifest": manifest},
	)

	out_path = layout.match_sessions_jsonl()
	lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]

	# Expect 2 sessions: merged [0,30] and separate [100,110]
	assert len(lines) == 2
	lines_sorted = sorted(lines, key=lambda r: (r["start_frame"], r["end_frame"]))

	a = lines_sorted[0]
	b = lines_sorted[1]

	assert a["start_frame"] == 0
	assert a["end_frame"] == 30
	assert a["start_ts_ms"] == 0
	assert a["end_ts_ms"] == 3000
	assert a["evidence"]["merge_max_gap_frames"] == 30
	assert set(a["evidence"]["seed_node_ids"]) == {"G:0-10", "G:20-30"}
	assert len(a["evidence"]["seed_match_ids"]) == 2

	assert b["start_frame"] == 100
	assert b["end_frame"] == 110
	assert b["start_ts_ms"] == 10000
	assert b["end_ts_ms"] == 11000
	assert b["evidence"]["seed_node_ids"] == ["G:100-110"]
	assert len(b["evidence"]["seed_match_ids"]) == 1

	validate_match_sessions_records(lines, expected_clip_id="clip_test")
