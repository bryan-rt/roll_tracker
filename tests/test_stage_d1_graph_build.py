from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from bjj_pipeline.stages.stitch.d1_graph_build import (
    _consolidate_parallel_triggers,
    run_d1,
)
from bjj_pipeline.stages.stitch.graph import EdgeType, NodeType



@dataclass
class _Layout:
    root: Path

    def d1_graph_nodes_parquet(self) -> Path:
        return self.root / "stage_D" / "d1_graph_nodes.parquet"

    def d1_graph_edges_parquet(self) -> Path:
        return self.root / "stage_D" / "d1_graph_edges.parquet"

    def d1_segments_parquet(self) -> Path:
        return self.root / "stage_D" / "d1_segments.parquet"

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

    stage_d_dir = tmp_path / "stage_D"
    assert (stage_d_dir / "d1_graph_nodes.parquet").exists()
    assert (stage_d_dir / "d1_graph_edges.parquet").exists()
    assert (stage_d_dir / "d1_segments.parquet").exists()


# ---------------------------------------------------------------------------
# CP5: _consolidate_parallel_triggers unit tests
# ---------------------------------------------------------------------------

def _fake_ts(tid: str, n_frames: int) -> pd.Series:
    return pd.Series({
        "tracklet_id": tid,
        "n_frames": n_frames,
        "start_frame": 0,
        "end_frame": 999,
    })


def test_consolidate_4way_all_tiebreak_levels():
    """4-way carrier competition exercising all three tiebreak levels.

    Event 1 (disappear=t99, frame=500): 4 carriers.
      - t40 wins on dist (0.1 vs 0.3).
    Event 2 (disappear=t88, frame=600): 3 carriers with same dist.
      - t20 wins: dist tied → n_frames (t20=200 ties t30=200, beats t10=100)
        → lexicographic ("t20" < "t30").
    """
    ts_by_tid = {
        "t10": _fake_ts("t10", 100),
        "t20": _fake_ts("t20", 200),
        "t30": _fake_ts("t30", 200),
        "t40": _fake_ts("t40", 50),
    }

    triggers = [
        # Event 1: 4-way, t40 wins on dist
        {"carrier": "t10", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.3},
        {"carrier": "t20", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.3},
        {"carrier": "t30", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.3},
        {"carrier": "t40", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.1},
        # Event 2: 3-way, all dist=0.2, t20 wins on n_frames then lexicographic
        {"carrier": "t10", "disappear": "t88", "merge_frame": 600, "merge_end": 599, "merge_dist_m": 0.2},
        {"carrier": "t20", "disappear": "t88", "merge_frame": 600, "merge_end": 599, "merge_dist_m": 0.2},
        {"carrier": "t30", "disappear": "t88", "merge_frame": 600, "merge_end": 599, "merge_dist_m": 0.2},
    ]

    kept, records = _consolidate_parallel_triggers(
        triggers=triggers,
        event_key="disappear",
        frame_key="merge_frame",
        dist_key="merge_dist_m",
        ts_by_tid=ts_by_tid,
    )

    # Exactly 2 kept (one per event)
    assert len(kept) == 2

    # Event 1: t40 wins on lowest dist
    event1_winner = [t for t in kept if t["disappear"] == "t99"]
    assert len(event1_winner) == 1
    assert event1_winner[0]["carrier"] == "t40"

    # Event 2: t20 wins (dist tie → n_frames tie t20=t30 → lexicographic "t20" < "t30")
    event2_winner = [t for t in kept if t["disappear"] == "t88"]
    assert len(event2_winner) == 1
    assert event2_winner[0]["carrier"] == "t20", (
        "t20 must win event 2 despite losing event 1 — per-event independence"
    )

    # Original trigger dicts preserved (merge_end key survives)
    assert event1_winner[0]["merge_end"] == 499
    assert event2_winner[0]["merge_end"] == 599

    # Consolidation records
    assert len(records) == 2
    rec1 = [r for r in records if r["event_key_value"] == "t99"][0]
    rec2 = [r for r in records if r["event_key_value"] == "t88"][0]
    assert rec1["n_discarded"] == 3
    assert rec1["chosen_carrier"] == "t40"
    assert rec2["n_discarded"] == 2
    assert rec2["chosen_carrier"] == "t20"

    # Discarded records include dist and n_frames
    for d in rec1["discarded"]:
        assert "dist" in d and "n_frames" in d
    for d in rec2["discarded"]:
        assert "dist" in d and "n_frames" in d


def test_consolidate_no_parallel_triggers():
    """No consolidation needed — each event has a unique carrier."""
    ts_by_tid = {
        "t1": _fake_ts("t1", 100),
        "t2": _fake_ts("t2", 200),
    }

    triggers = [
        {"carrier": "t1", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.2},
        {"carrier": "t2", "disappear": "t88", "merge_frame": 600, "merge_end": 599, "merge_dist_m": 0.3},
    ]

    kept, records = _consolidate_parallel_triggers(
        triggers=triggers,
        event_key="disappear",
        frame_key="merge_frame",
        dist_key="merge_dist_m",
        ts_by_tid=ts_by_tid,
    )

    assert len(kept) == 2
    assert len(records) == 0


def test_consolidate_empty_triggers():
    """Empty input returns empty output."""
    kept, records = _consolidate_parallel_triggers(
        triggers=[],
        event_key="disappear",
        frame_key="merge_frame",
        dist_key="merge_dist_m",
        ts_by_tid={},
    )
    assert kept == []
    assert records == []


def test_consolidate_nan_dist_fallback():
    """NaN dist falls back to inf — carrier with valid dist always wins."""
    ts_by_tid = {
        "t1": _fake_ts("t1", 100),
        "t2": _fake_ts("t2", 200),
    }

    triggers = [
        {"carrier": "t1", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": float("nan")},
        {"carrier": "t2", "disappear": "t99", "merge_frame": 500, "merge_end": 499, "merge_dist_m": 0.5},
    ]

    kept, records = _consolidate_parallel_triggers(
        triggers=triggers,
        event_key="disappear",
        frame_key="merge_frame",
        dist_key="merge_dist_m",
        ts_by_tid=ts_by_tid,
    )

    assert len(kept) == 1
    assert kept[0]["carrier"] == "t2"
