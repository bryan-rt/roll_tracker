from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from bjj_pipeline.stages.stitch.d1_graph_build import run_d1
from bjj_pipeline.stages.stitch.graph import EdgeType, NodeType


@dataclass
class _Layout:
	root: Path

	def tracklet_bank_frames_parquet(self) -> Path:
		return self.root / "stage_D" / "tracklet_bank_frames.parquet"

	def tracklet_bank_summaries_parquet(self) -> Path:
		return self.root / "stage_D" / "tracklet_bank_summaries.parquet"

	def audit_jsonl(self, stage: str) -> Path:
		assert stage == "D"
		return self.root / "stage_D" / "audit.jsonl"


def _write_parquets(tmp_path: Path, tf: pd.DataFrame, ts: pd.DataFrame) -> _Layout:
	stage_d = tmp_path / "stage_D"
	stage_d.mkdir(parents=True, exist_ok=True)
	tf.to_parquet(stage_d / "tracklet_bank_frames.parquet", index=False)
	ts.to_parquet(stage_d / "tracklet_bank_summaries.parquet", index=False)
	return _Layout(tmp_path)


def test_d1_merge_split_creates_group(tmp_path: Path):
	# Two tracklets end near each other and close in space; one disappears; new starts near carrier.
	# Tracklets: A (disappear), S (survivor/carrier), N (new)
	ts = pd.DataFrame(
		[
			{"tracklet_id": "A", "start_frame": 0, "end_frame": 10, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
			{"tracklet_id": "S", "start_frame": 0, "end_frame": 50, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
			{"tracklet_id": "N", "start_frame": 20, "end_frame": 40, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
		]
	)
	tf = pd.DataFrame(
		[
			# endpoints for A and S near end
			{"tracklet_id": "A", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.0, "y_m_repaired": 1.0, "x_m": 1.0, "y_m": 1.0},
			{"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.1, "y_m_repaired": 1.0, "x_m": 1.1, "y_m": 1.0},
			# carrier position at split frame 20
			{"tracklet_id": "S", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.2, "y_m_repaired": 1.0, "x_m": 1.2, "y_m": 1.0},
			# N starts near carrier at frame 20
			{"tracklet_id": "N", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.25, "y_m_repaired": 1.0, "x_m": 1.25, "y_m": 1.0},
		]
	)
	layout = _write_parquets(tmp_path, tf, ts)
	cfg: Dict[str, Any] = {"stages": {"stage_D": {"d1": {"enable_group_nodes": True, "split_search_horizon_frames": 50}}}}
	manifest = {"fps": 30.0, "frame_count": 60, "duration_ms": 2000}
	graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)

	group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
	assert len(group_nodes) == 1

	# Expect merge and split edges
	edge_types = [e.type for e in graph.edges.values()]
	assert EdgeType.MERGE in edge_types
	assert EdgeType.SPLIT in edge_types


def test_d1_start_merged_then_split(tmp_path: Path):
	ts = pd.DataFrame(
		[
			{"tracklet_id": "S", "start_frame": 0, "end_frame": 80, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
			{"tracklet_id": "N", "start_frame": 10, "end_frame": 50, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
		]
	)
	tf = pd.DataFrame(
		[
			{"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 2.0, "y_m_repaired": 2.0, "x_m": 2.0, "y_m": 2.0},
			{"tracklet_id": "N", "frame_index": 10, "on_mat": True, "x_m_repaired": 2.1, "y_m_repaired": 2.0, "x_m": 2.1, "y_m": 2.0},
		]
	)
	layout = _write_parquets(tmp_path, tf, ts)
	cfg: Dict[str, Any] = {"stages": {"stage_D": {"d1": {"enable_group_nodes": True}}}}
	manifest = {"fps": 30.0, "frame_count": 100, "duration_ms": 4000}
	graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
	group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
	assert len(group_nodes) >= 1
	# group-at-start should have a cap=2 birth
	birth_edges = [e for e in graph.edges.values() if e.type == EdgeType.BIRTH and e.v.startswith("G:")]
	assert any(e.capacity == 2 for e in birth_edges)


def test_d1_merge_open_ended_to_clip_end(tmp_path: Path):
	ts = pd.DataFrame(
		[
			{"tracklet_id": "A", "start_frame": 0, "end_frame": 10, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
			{"tracklet_id": "S", "start_frame": 0, "end_frame": 200, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
		]
	)
	tf = pd.DataFrame(
		[
			{"tracklet_id": "A", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.0, "y_m_repaired": 1.0, "x_m": 1.0, "y_m": 1.0},
			{"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.1, "y_m_repaired": 1.0, "x_m": 1.1, "y_m": 1.0},
		]
	)
	layout = _write_parquets(tmp_path, tf, ts)
	cfg: Dict[str, Any] = {"stages": {"stage_D": {"d1": {"enable_group_nodes": True}}}}
	manifest = {"fps": 30.0, "frame_count": 240, "duration_ms": 8000}
	graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
	group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
	assert len(group_nodes) == 1
	death_edges = [e for e in graph.edges.values() if e.type == EdgeType.DEATH and e.u.startswith("G:")]
	assert any(e.capacity == 2 for e in death_edges)


def test_d1_fallback_to_raw_coords_when_repaired_missing(tmp_path: Path):
	ts = pd.DataFrame(
		[
			{"tracklet_id": "S", "start_frame": 0, "end_frame": 20, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
			{"tracklet_id": "N", "start_frame": 10, "end_frame": 15, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
		]
	)
	tf = pd.DataFrame(
		[
			{"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": None, "y_m_repaired": None, "x_m": 3.0, "y_m": 3.0},
			{"tracklet_id": "N", "frame_index": 10, "on_mat": True, "x_m_repaired": None, "y_m_repaired": None, "x_m": 3.1, "y_m": 3.0},
		]
	)
	layout = _write_parquets(tmp_path, tf, ts)
	cfg: Dict[str, Any] = {"stages": {"stage_D": {"d1": {"enable_group_nodes": True}}}}
	manifest = {"fps": 30.0, "frame_count": 30, "duration_ms": 1000}
	graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
	group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
	assert len(group_nodes) >= 1
