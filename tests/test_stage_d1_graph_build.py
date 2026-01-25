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

    @property
    def clip_root(self) -> Path:
        # Match ClipOutputLayout: root directory for the clip output.
        return self.root

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




def _base_cfg(**overrides: Any) -> Dict[str, Any]:
    d1 = {
        "enable_group_nodes": True,
        "enable_lifespan_segmentation": True,
        "write_debug_graph_artifacts": False,
        "merge_dist_m": 0.50,
        "split_dist_m": 0.50,
        "split_search_horizon_frames": 200,
        "min_group_duration_frames": 1,
        "min_split_separation_frames": 0,
        "carrier_coord_window_frames": 2,
        "merge_trigger_max_age_frames": 9999,
    }
    d1.update(overrides)
    return {"stages": {"stage_D": {"d1": d1}}}


def test_d1_lifespan_segmentation_merge_inside_carrier(tmp_path: Path):
    # A disappears at frame 10 while S continues; N appears later near S.
    ts = pd.DataFrame(
        [
            {"tracklet_id": "A", "start_frame": 0, "end_frame": 10, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
            {"tracklet_id": "S", "start_frame": 0, "end_frame": 50, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
            {"tracklet_id": "N", "start_frame": 20, "end_frame": 40, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
        ]
    )
    tf = pd.DataFrame(
        [
            {"tracklet_id": "A", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.0, "y_m_repaired": 1.0, "x_m": 1.0, "y_m": 1.0},
            {"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.1, "y_m_repaired": 1.0, "x_m": 1.1, "y_m": 1.0},
            {"tracklet_id": "S", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.2, "y_m_repaired": 1.0, "x_m": 1.2, "y_m": 1.0},
            {"tracklet_id": "N", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.25, "y_m_repaired": 1.0, "x_m": 1.25, "y_m": 1.0},
        ]
    )
    layout = _write_parquets(tmp_path, tf, ts)
    manifest = {"fps": 30.0, "frame_count": 60, "duration_ms": 2000}
    graph = run_d1(cfg=_base_cfg(), layout=layout, manifest=manifest)

    group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
    assert len(group_nodes) >= 1
    assert any((n.start_frame == 11 and n.end_frame == 19) for n in group_nodes)
    edge_types = {e.type for e in graph.edges.values()}
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
    cfg: Dict[str, Any] = _base_cfg(split_search_horizon_frames=50)
    manifest = {"fps": 30.0, "frame_count": 100, "duration_ms": 4000}
    graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
    group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
    assert len(group_nodes) >= 1
    # start-merged group should span [0, split_frame-1]
    assert any((n.start_frame == 0 and n.end_frame == 9) for n in group_nodes)
    # group-at-start should have a cap=2 birth
    birth_edges = [e for e in graph.edges.values() if e.type == EdgeType.BIRTH and e.v.startswith("G:")]
    assert any(e.capacity == 2 for e in birth_edges)


def test_d1_merge_open_ended_clamped_to_carrier_end(tmp_path: Path):
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
    cfg: Dict[str, Any] = _base_cfg(split_search_horizon_frames=30)
    manifest = {"fps": 30.0, "frame_count": 240, "duration_ms": 8000}
    graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
    group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
    assert len(group_nodes) == 1
    # open-ended groups must be clamped to the carrier lifespan (not manifest.frame_count)
    assert group_nodes[0].start_frame == 11
    assert group_nodes[0].end_frame == 200
    def test_d1_segments_within_base_tracklet_bounds(tmp_path: Path):
        ts = pd.DataFrame(
            [
                {"tracklet_id": "A", "start_frame": 0, "end_frame": 10, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
                {"tracklet_id": "S", "start_frame": 0, "end_frame": 50, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
            ]
        )
        tf = pd.DataFrame(
            [
                {"tracklet_id": "A", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.0, "y_m_repaired": 1.0, "x_m": 1.0, "y_m": 1.0},
                {"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.1, "y_m_repaired": 1.0, "x_m": 1.1, "y_m": 1.0},
            ]
        )
        layout = _write_parquets(tmp_path, tf, ts)
        manifest = {"fps": 30.0, "frame_count": 240, "duration_ms": 8000}
        run_d1(cfg=_base_cfg(write_debug_graph_artifacts=True), layout=layout, manifest=manifest)

        segs = pd.read_parquet(tmp_path / "_debug" / "d1_segments.parquet")
        # Every segment must lie within its base tracklet's [start_frame, end_frame] from summaries.
        bounds = {r["tracklet_id"]: (int(r["start_frame"]), int(r["end_frame"])) for r in ts.to_dict(orient="records")}
        for r in segs.to_dict(orient="records"):
            tid = str(r["base_tracklet_id"])
            sf = int(r["start_frame"])
            ef = int(r["end_frame"])
            bs, be = bounds[tid]
            assert bs <= sf <= ef <= be, (tid, (sf, ef), (bs, be))
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
    cfg: Dict[str, Any] = _base_cfg(split_search_horizon_frames=50)
    manifest = {"fps": 30.0, "frame_count": 30, "duration_ms": 1000}
    graph = run_d1(cfg=cfg, layout=layout, manifest=manifest)
    group_nodes = [n for n in graph.nodes.values() if n.type == NodeType.GROUP_TRACKLET]
    assert len(group_nodes) >= 1


def test_d1_debug_artifacts_include_segments(tmp_path: Path):
    ts = pd.DataFrame(
        [
            {"tracklet_id": "A", "start_frame": 0, "end_frame": 10, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
            {"tracklet_id": "S", "start_frame": 0, "end_frame": 50, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
            {"tracklet_id": "N", "start_frame": 20, "end_frame": 40, "must_link_anchor_key": None, "cannot_link_anchor_keys_json": "[]"},
        ]
    )
    tf = pd.DataFrame(
        [
            {"tracklet_id": "A", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.0, "y_m_repaired": 1.0, "x_m": 1.0, "y_m": 1.0},
            {"tracklet_id": "S", "frame_index": 10, "on_mat": True, "x_m_repaired": 1.1, "y_m_repaired": 1.0, "x_m": 1.1, "y_m": 1.0},
            {"tracklet_id": "S", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.2, "y_m_repaired": 1.0, "x_m": 1.2, "y_m": 1.0},
            {"tracklet_id": "N", "frame_index": 20, "on_mat": True, "x_m_repaired": 1.25, "y_m_repaired": 1.0, "x_m": 1.25, "y_m": 1.0},
        ]
    )
    layout = _write_parquets(tmp_path, tf, ts)
    manifest = {"fps": 30.0, "frame_count": 60, "duration_ms": 2000}
    run_d1(cfg=_base_cfg(write_debug_graph_artifacts=True), layout=layout, manifest=manifest)

    debug_dir = tmp_path / "_debug"
    assert (debug_dir / "d1_graph_nodes.parquet").exists()
    assert (debug_dir / "d1_graph_edges.parquet").exists()
    assert (debug_dir / "d1_group_spans.parquet").exists()
    assert (debug_dir / "d1_suppressed_continue_edges.parquet").exists()
    assert (debug_dir / "d1_segments.parquet").exists()
    assert (debug_dir / "d1_merge_triggers.parquet").exists()
    assert (debug_dir / "d1_split_triggers.parquet").exists()
    assert (debug_dir / "d1_suppressed_split_triggers.parquet").exists()
    assert (debug_dir / "d1_suppressed_group_spans.parquet").exists()
