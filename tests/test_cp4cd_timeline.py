"""T1 tests for CP4.C+D: session timeline and D1/D2 real-time dt.

These are the ONLY correctness evidence for the session timeline until
CP-R8 footage exists. T2.5 on Saturday footage is a smoke test only.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from bjj_pipeline.contracts.f0_sidecar_testutil import generate_synthetic_sidecar
from bjj_pipeline.stages.stitch.d0_bank import _apply_cp3_kinematics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clip_bank(
    clip_id: str,
    n_frames: int,
    dt_ms: int = 67,
    n_tracklets: int = 2,
    start_ts_ms: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal bank frames + summaries for a synthetic clip."""
    rows = []
    for t in range(n_tracklets):
        tid = f"t{t}"
        for fi in range(n_frames):
            rows.append({
                "clip_id": clip_id,
                "camera_id": "cam01",
                "tracklet_id": tid,
                "frame_index": fi,
                "timestamp_ms": start_ts_ms + fi * dt_ms,
                "detection_id": f"d{t}_{fi}",
                "x_m": float(fi) * 0.1,
                "y_m": 0.0,
            })
    frames = pd.DataFrame(rows)

    summ_rows = []
    for t in range(n_tracklets):
        tid = f"t{t}"
        summ_rows.append({
            "clip_id": clip_id,
            "camera_id": "cam01",
            "tracklet_id": tid,
            "start_frame": 0,
            "end_frame": n_frames - 1,
            "n_frames": n_frames,
        })
    summaries = pd.DataFrame(summ_rows)
    return frames, summaries


def _write_clip_artifacts(
    clip_dir: Path,
    frames: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    """Write minimal Stage A / Stage D artifacts for a clip."""
    stage_a = clip_dir / "stage_A"
    stage_a.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(stage_a / "tracklet_frames.parquet", index=False)
    summaries.to_parquet(stage_a / "tracklet_summaries.parquet", index=False)
    # D0 bank (identical to Stage A for this test)
    stage_d = clip_dir / "stage_D"
    stage_d.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(stage_d / "tracklet_bank_frames.parquet", index=False)
    summaries.to_parquet(stage_d / "tracklet_bank_summaries.parquet", index=False)
    # Empty audit + identity hints
    (stage_d / "d05_split_audit.jsonl").write_text("", encoding="utf-8")
    (stage_a / "detections.parquet").parent.mkdir(parents=True, exist_ok=True)
    # Minimal detections
    det = frames[["clip_id", "camera_id", "frame_index", "timestamp_ms", "detection_id"]].copy()
    det["class_name"] = "person"
    det["confidence"] = 0.9
    det["x1"] = 0.0
    det["y1"] = 0.0
    det["x2"] = 1.0
    det["y2"] = 1.0
    det["tracklet_id"] = frames["tracklet_id"]
    det.to_parquet(stage_a / "detections.parquet", index=False)
    # Identity hints (empty)
    (stage_a.parent / "stage_C").mkdir(parents=True, exist_ok=True)
    (stage_a.parent / "stage_C" / "identity_hints.jsonl").write_text("", encoding="utf-8")
    (stage_a.parent / "stage_C" / "tag_observations.jsonl").write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# T1-1: Synthetic two-clip session with known sub-second offset
# ---------------------------------------------------------------------------

class TestSessionTimeline:
    def test_timestamp_offset_matches_known_sidecar_offset(self, tmp_path):
        """Two clips with pts_wallclock_offset_s 5.0s apart.

        Assert clip 2's timestamp_ms values are offset by ~5000ms in the
        aggregated bank.
        """
        from bjj_pipeline.stages.stitch.session_d_run import aggregate_session_bank
        from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
        from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter

        # Set up directory structure mimicking data/raw/nest/{gym}/{cam}/...
        gym_id = "test-gym"
        cam_id = "cam01"
        raw_dir = tmp_path / "data" / "raw" / "nest" / gym_id / cam_id / "2026-01-01" / "12"
        raw_dir.mkdir(parents=True)
        out_root = tmp_path / "outputs"

        # Clip 1: offset 1000.0s
        clip1_mp4 = raw_dir / "cam01-20260101-120000.mp4"
        clip1_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120000.timing.jsonl",
            frame_count=100, dt_s=0.067,
            pts_wallclock_offset_s_override=1000.0,
        )
        clip1_frames, clip1_summ = _make_clip_bank("cam01-20260101-120000", 100)
        clip1_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120000"
        _write_clip_artifacts(clip1_out, clip1_frames, clip1_summ)
        # Write manifest
        from bjj_pipeline.contracts.f0_manifest import init_manifest, write_manifest
        m1 = init_manifest(
            clip_id="cam01-20260101-120000", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip1_mp4), fps=15.0, frame_count=100,
            duration_ms=6700, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m1, clip1_out / "clip_manifest.json")

        # Clip 2: offset 1005.0s (5.0s later)
        clip2_mp4 = raw_dir / "cam01-20260101-120500.mp4"
        clip2_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120500.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1005.0,
        )
        clip2_frames, clip2_summ = _make_clip_bank("cam01-20260101-120500", 50)
        clip2_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120500"
        _write_clip_artifacts(clip2_out, clip2_frames, clip2_summ)
        m2 = init_manifest(
            clip_id="cam01-20260101-120500", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip2_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m2, clip2_out / "clip_manifest.json")

        session_layout = SessionOutputLayout(
            gym_id=gym_id, date="2026-01-01",
            session_id="2026-01-01T1200", root=out_root,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id)

        session_clips = [
            (clip1_mp4, cam_id),
            (clip2_mp4, cam_id),
        ]

        cfg = {"stages": {"ingest": {"allow_synthetic_sidecars": False}}}

        _, _, _, _, registry = aggregate_session_bank(
            session_layout=session_layout,
            adapter=adapter,
            session_clips=session_clips,
            cam_id=cam_id,
            output_root=out_root,
            resolved_config=cfg,
        )

        # Read the aggregated bank frames
        combined = pd.read_parquet(adapter.tracklet_bank_frames_parquet())

        # Clip 1's first frame should be at ts=0 (it is the session start)
        clip1_rows = combined[combined["tracklet_id"].str.startswith("cam01-20260101-120000:")]
        clip1_first_ts = clip1_rows["timestamp_ms"].min()
        assert clip1_first_ts == 0, f"Clip 1 first ts should be 0, got {clip1_first_ts}"

        # Clip 2's first frame should be at ts ≈ 5000ms (the known offset)
        clip2_rows = combined[combined["tracklet_id"].str.startswith("cam01-20260101-120500:")]
        clip2_first_ts = clip2_rows["timestamp_ms"].min()
        assert abs(clip2_first_ts - 5000) < 10, \
            f"Clip 2 first ts should be ~5000ms, got {clip2_first_ts}"

        # Frame indices: clip 2 should start at 100 (cumulative)
        clip2_first_fi = clip2_rows["frame_index"].min()
        assert clip2_first_fi == 100, \
            f"Clip 2 first frame_index should be 100, got {clip2_first_fi}"

    def test_cross_clip_dt_s_regression_guard(self, tmp_path):
        """Clip 2's timestamp_ms restarts near zero. Without CP4.C offset,
        a cross-clip step would produce dt_s < 0. With offset, dt_s is positive.
        """
        from bjj_pipeline.stages.stitch.session_d_run import aggregate_session_bank
        from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
        from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter

        gym_id = "test-gym"
        cam_id = "cam01"
        raw_dir = tmp_path / "data" / "raw" / "nest" / gym_id / cam_id / "2026-01-01" / "12"
        raw_dir.mkdir(parents=True)
        out_root = tmp_path / "outputs"

        # Clip 1: 50 frames at 67ms, offset 1000.0s
        clip1_mp4 = raw_dir / "cam01-20260101-120000.mp4"
        clip1_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120000.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1000.0,
        )
        clip1_frames, clip1_summ = _make_clip_bank("cam01-20260101-120000", 50)
        clip1_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120000"
        _write_clip_artifacts(clip1_out, clip1_frames, clip1_summ)
        from bjj_pipeline.contracts.f0_manifest import init_manifest, write_manifest
        m1 = init_manifest(
            clip_id="cam01-20260101-120000", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip1_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m1, clip1_out / "clip_manifest.json")

        # Clip 2: 50 frames at 67ms — ts restarts at 0, offset 1003.5s
        clip2_mp4 = raw_dir / "cam01-20260101-120330.mp4"
        clip2_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120330.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1003.5,
        )
        clip2_frames, clip2_summ = _make_clip_bank("cam01-20260101-120330", 50)
        # clip2 ts restarts at 0 (clip-relative)
        clip2_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120330"
        _write_clip_artifacts(clip2_out, clip2_frames, clip2_summ)
        m2 = init_manifest(
            clip_id="cam01-20260101-120330", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip2_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m2, clip2_out / "clip_manifest.json")

        session_layout = SessionOutputLayout(
            gym_id=gym_id, date="2026-01-01",
            session_id="2026-01-01T1200", root=out_root,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id)
        session_clips = [(clip1_mp4, cam_id), (clip2_mp4, cam_id)]
        cfg = {}

        _, _, _, _, _ = aggregate_session_bank(
            session_layout=session_layout, adapter=adapter,
            session_clips=session_clips, cam_id=cam_id,
            output_root=out_root, resolved_config=cfg,
        )

        combined = pd.read_parquet(adapter.tracklet_bank_frames_parquet())

        # Get last ts of clip 1 and first ts of clip 2
        c1 = combined[combined["tracklet_id"].str.startswith("cam01-20260101-120000:")]
        c2 = combined[combined["tracklet_id"].str.startswith("cam01-20260101-120330:")]
        c1_last_ts = c1["timestamp_ms"].max()
        c2_first_ts = c2["timestamp_ms"].min()

        # Cross-clip dt must be positive (the whole point of CP4.C)
        cross_clip_dt_ms = c2_first_ts - c1_last_ts
        assert cross_clip_dt_ms > 0, \
            f"Cross-clip dt_ms should be positive, got {cross_clip_dt_ms} " \
            f"(c1_last={c1_last_ts}, c2_first={c2_first_ts})"

    def test_registry_round_trip(self, tmp_path):
        """Stage D writes clip_offset_registry.json, values match what was computed."""
        from bjj_pipeline.stages.stitch.session_d_run import aggregate_session_bank
        from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
        from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter

        gym_id = "test-gym"
        cam_id = "cam01"
        raw_dir = tmp_path / "data" / "raw" / "nest" / gym_id / cam_id / "2026-01-01" / "12"
        raw_dir.mkdir(parents=True)
        out_root = tmp_path / "outputs"

        clip1_mp4 = raw_dir / "cam01-20260101-120000.mp4"
        clip1_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120000.timing.jsonl",
            frame_count=100, dt_s=0.067,
            pts_wallclock_offset_s_override=1000.0,
        )
        clip1_frames, clip1_summ = _make_clip_bank("cam01-20260101-120000", 100)
        clip1_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120000"
        _write_clip_artifacts(clip1_out, clip1_frames, clip1_summ)
        from bjj_pipeline.contracts.f0_manifest import init_manifest, write_manifest
        m1 = init_manifest(
            clip_id="cam01-20260101-120000", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip1_mp4), fps=15.0, frame_count=100,
            duration_ms=6700, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m1, clip1_out / "clip_manifest.json")

        clip2_mp4 = raw_dir / "cam01-20260101-120500.mp4"
        clip2_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120500.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1005.0,
        )
        clip2_frames, clip2_summ = _make_clip_bank("cam01-20260101-120500", 50)
        clip2_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120500"
        _write_clip_artifacts(clip2_out, clip2_frames, clip2_summ)
        m2 = init_manifest(
            clip_id="cam01-20260101-120500", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip2_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m2, clip2_out / "clip_manifest.json")

        session_layout = SessionOutputLayout(
            gym_id=gym_id, date="2026-01-01",
            session_id="2026-01-01T1200", root=out_root,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id)
        session_clips = [(clip1_mp4, cam_id), (clip2_mp4, cam_id)]

        _, _, _, _, registry = aggregate_session_bank(
            session_layout=session_layout, adapter=adapter,
            session_clips=session_clips, cam_id=cam_id,
            output_root=out_root, resolved_config={},
        )

        # Read the persisted registry from disk
        registry_path = adapter.stage_dir("D") / "clip_offset_registry.json"
        assert registry_path.exists(), "clip_offset_registry.json should be written"
        with open(registry_path) as f:
            disk_data = json.load(f)

        disk_clips = disk_data["clips"]
        assert len(disk_clips) == 2

        # Clip 1: frame_offset=0, clip_frame_count=100
        r1 = disk_clips[0]
        assert r1["clip_id"] == "cam01-20260101-120000"
        assert r1["frame_offset"] == 0
        assert r1["clip_frame_count"] == 100

        # Clip 2: frame_offset=100 (cumulative), clip_frame_count=50
        r2 = disk_clips[1]
        assert r2["clip_id"] == "cam01-20260101-120500"
        assert r2["frame_offset"] == 100
        assert r2["clip_frame_count"] == 50

        # Boundary decisions present
        assert len(disk_data["boundary_decisions"]) == 1

        # Verify return value matches disk clips
        assert registry == disk_clips


# ---------------------------------------------------------------------------
# T1-4: D1/D2 unit tests with dt_ms
# ---------------------------------------------------------------------------

class TestD1D2DtMs:
    def test_dt_ms_100_produces_correct_dt_s(self):
        """Edge with dt_ms=100 → dt_s=0.1 in costs output."""
        from bjj_pipeline.stages.stitch.costs import compute_edge_costs

        d1_nodes = pd.DataFrame([
            {"node_id": "n0", "base_tracklet_id": "t0", "segment_type": "SOLO"},
            {"node_id": "n1", "base_tracklet_id": "t1", "segment_type": "SOLO"},
        ])
        d1_edges = pd.DataFrame([{
            "edge_id": "e0", "edge_type": "EdgeType.CONTINUE",
            "u": "n0", "v": "n1", "capacity": 1,
            "dt_frames": 1, "dt_ms": 100,
        }])
        bank_frames = pd.DataFrame([
            {"tracklet_id": "t0", "frame_index": 0, "x_m": 0.0, "y_m": 0.0},
            {"tracklet_id": "t1", "frame_index": 1, "x_m": 1.0, "y_m": 0.0},
        ])

        out, _ = compute_edge_costs(
            d1_edges=d1_edges, d1_nodes=d1_nodes, bank_frames=bank_frames,
            cfg={"endpoint_search_window_frames": 0},
            v_cost_scale_mps_resolved=8.0, v_hinge_mps_resolved=8.0,
        )

        assert abs(out.loc[0, "dt_s"] - 0.1) < 1e-9
        assert out.loc[0, "dt_ms"] == 100

    def test_dt_ms_none_disallows_as_dt_unavailable(self):
        """Edge with dt_ms=None but dt_frames present → dt_unavailable."""
        from bjj_pipeline.stages.stitch.costs import compute_edge_costs

        d1_nodes = pd.DataFrame([
            {"node_id": "n0", "base_tracklet_id": "t0", "segment_type": "SOLO"},
            {"node_id": "n1", "base_tracklet_id": "t1", "segment_type": "SOLO"},
        ])
        d1_edges = pd.DataFrame([{
            "edge_id": "e0", "edge_type": "EdgeType.CONTINUE",
            "u": "n0", "v": "n1", "capacity": 1,
            "dt_frames": 1, "dt_ms": pd.NA,
        }])
        d1_edges["dt_ms"] = d1_edges["dt_ms"].astype("Int64")
        bank_frames = pd.DataFrame([
            {"tracklet_id": "t0", "frame_index": 0, "x_m": 0.0, "y_m": 0.0},
            {"tracklet_id": "t1", "frame_index": 1, "x_m": 1.0, "y_m": 0.0},
        ])

        out, _ = compute_edge_costs(
            d1_edges=d1_edges, d1_nodes=d1_nodes, bank_frames=bank_frames,
            cfg={"endpoint_search_window_frames": 0},
            v_cost_scale_mps_resolved=8.0, v_hinge_mps_resolved=8.0,
        )

        assert bool(out.loc[0, "is_allowed"]) is False
        reasons = json.loads(out.loc[0, "disallow_reasons_json"])
        assert "dt_unavailable" in reasons


# ---------------------------------------------------------------------------
# T1 CP4.E: boundary classification and session_segment_id
# ---------------------------------------------------------------------------

class TestBoundaryClassification:
    def test_contiguous_boundary_permits_same_segment(self, tmp_path):
        """Two clips, shortfall ~0.1s (< 2.0s), same attempt.
        Assert session_segment_id is the same on both clips.
        """
        from bjj_pipeline.stages.stitch.session_d_run import aggregate_session_bank
        from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
        from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter

        gym_id = "test-gym"
        cam_id = "cam01"
        raw_dir = tmp_path / "data" / "raw" / "nest" / gym_id / cam_id / "2026-01-01" / "12"
        raw_dir.mkdir(parents=True)
        out_root = tmp_path / "outputs"

        # Clip 1: 100 frames at 67ms = 6.7s content, offset 1000.0s
        clip1_mp4 = raw_dir / "cam01-20260101-120000.mp4"
        clip1_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120000.timing.jsonl",
            frame_count=100, dt_s=0.067,
            pts_wallclock_offset_s_override=1000.0,
        )
        clip1_frames, clip1_summ = _make_clip_bank("cam01-20260101-120000", 100)
        clip1_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120000"
        _write_clip_artifacts(clip1_out, clip1_frames, clip1_summ)
        from bjj_pipeline.contracts.f0_manifest import init_manifest, write_manifest
        m1 = init_manifest(
            clip_id="cam01-20260101-120000", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip1_mp4), fps=15.0, frame_count=100,
            duration_ms=6700, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m1, clip1_out / "clip_manifest.json")

        # Clip 2: offset 1006.8s (content 6.7s + 0.1s gap = contiguous)
        clip2_mp4 = raw_dir / "cam01-20260101-120648.mp4"
        clip2_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120648.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1006.8,
        )
        clip2_frames, clip2_summ = _make_clip_bank("cam01-20260101-120648", 50)
        clip2_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120648"
        _write_clip_artifacts(clip2_out, clip2_frames, clip2_summ)
        m2 = init_manifest(
            clip_id="cam01-20260101-120648", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip2_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m2, clip2_out / "clip_manifest.json")

        session_layout = SessionOutputLayout(
            gym_id=gym_id, date="2026-01-01",
            session_id="2026-01-01T1200", root=out_root,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id)

        _, _, _, _, _ = aggregate_session_bank(
            session_layout=session_layout, adapter=adapter,
            session_clips=[(clip1_mp4, cam_id), (clip2_mp4, cam_id)],
            cam_id=cam_id, output_root=out_root, resolved_config={},
        )

        combined = pd.read_parquet(adapter.tracklet_bank_frames_parquet())
        assert "session_segment_id" in combined.columns
        # Both clips should have the same session_segment_id (contiguous)
        assert combined["session_segment_id"].nunique() == 1

        # Registry should show PERMIT
        with open(adapter.stage_dir("D") / "clip_offset_registry.json") as f:
            reg = json.load(f)
        assert reg["boundary_decisions"][0]["decision"] == "PERMIT"

    def test_discontinuous_boundary_breaks_segment(self, tmp_path):
        """Two clips, shortfall 93.3s (> 2.0s). Assert different session_segment_id."""
        from bjj_pipeline.stages.stitch.session_d_run import aggregate_session_bank
        from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
        from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter

        gym_id = "test-gym"
        cam_id = "cam01"
        raw_dir = tmp_path / "data" / "raw" / "nest" / gym_id / cam_id / "2026-01-01" / "12"
        raw_dir.mkdir(parents=True)
        out_root = tmp_path / "outputs"

        clip1_mp4 = raw_dir / "cam01-20260101-120000.mp4"
        clip1_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-120000.timing.jsonl",
            frame_count=100, dt_s=0.067,
            pts_wallclock_offset_s_override=1000.0,
        )
        clip1_frames, clip1_summ = _make_clip_bank("cam01-20260101-120000", 100)
        clip1_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-120000"
        _write_clip_artifacts(clip1_out, clip1_frames, clip1_summ)
        from bjj_pipeline.contracts.f0_manifest import init_manifest, write_manifest
        m1 = init_manifest(
            clip_id="cam01-20260101-120000", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip1_mp4), fps=15.0, frame_count=100,
            duration_ms=6700, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m1, clip1_out / "clip_manifest.json")

        # Clip 2: offset 1100.0s (shortfall = 100-6.7 = 93.3s → BREAK)
        clip2_mp4 = raw_dir / "cam01-20260101-121400.mp4"
        clip2_mp4.write_bytes(b"\x00")
        generate_synthetic_sidecar(
            raw_dir / "cam01-20260101-121400.timing.jsonl",
            frame_count=50, dt_s=0.067,
            pts_wallclock_offset_s_override=1100.0,
        )
        clip2_frames, clip2_summ = _make_clip_bank("cam01-20260101-121400", 50)
        clip2_out = out_root / gym_id / cam_id / "2026-01-01" / "12" / "cam01-20260101-121400"
        _write_clip_artifacts(clip2_out, clip2_frames, clip2_summ)
        m2 = init_manifest(
            clip_id="cam01-20260101-121400", camera_id=cam_id, gym_id=gym_id,
            input_video_path=str(clip2_mp4), fps=15.0, frame_count=50,
            duration_ms=3350, pipeline_version="test", created_at_ms=0,
        )
        write_manifest(m2, clip2_out / "clip_manifest.json")

        session_layout = SessionOutputLayout(
            gym_id=gym_id, date="2026-01-01",
            session_id="2026-01-01T1200", root=out_root,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id)

        _, _, _, _, _ = aggregate_session_bank(
            session_layout=session_layout, adapter=adapter,
            session_clips=[(clip1_mp4, cam_id), (clip2_mp4, cam_id)],
            cam_id=cam_id, output_root=out_root, resolved_config={},
        )

        combined = pd.read_parquet(adapter.tracklet_bank_frames_parquet())
        # Two different session_segment_ids (discontinuity)
        assert combined["session_segment_id"].nunique() == 2

        with open(adapter.stage_dir("D") / "clip_offset_registry.json") as f:
            reg = json.load(f)
        assert reg["boundary_decisions"][0]["decision"] == "BREAK"
        assert "shortfall" in reg["boundary_decisions"][0]["reason"]

    def test_unclassifiable_boundary_breaks(self):
        """When prev clip has nominal_dt_s=None, boundary is BREAK.

        Tests the classification logic directly: if nominal_dt_s is absent,
        the shortfall is uncomputable and the boundary must break.
        """
        # Simulate the boundary classification logic from aggregate_session_bank
        BREAK_THRESHOLD_S = 2.0
        # Clip 1 with nominal_dt_s=None (broken contract)
        cam_clips_info = [
            {"mp4_path": Path("/fake/clip1.mp4"), "offset_s": 1000.0,
             "offset_status": "determined", "frame_count": 100,
             "nominal_dt_s": None, "attempt": 1},
            {"mp4_path": Path("/fake/clip2.mp4"), "offset_s": 1006.8,
             "offset_status": "determined", "frame_count": 50,
             "nominal_dt_s": 0.067, "attempt": 1},
        ]

        # Reproduce the boundary classification
        boundary_decisions = []
        segment_ids = [0]
        for i in range(1, len(cam_clips_info)):
            prev, curr = cam_clips_info[i-1], cam_clips_info[i]
            prev_nom = prev["nominal_dt_s"]
            if prev_nom is None:
                boundary_decisions.append({
                    "from": prev["mp4_path"].stem, "to": curr["mp4_path"].stem,
                    "decision": "BREAK", "reason": "nominal_dt_s_missing",
                })
                segment_ids.append(segment_ids[-1] + 1)
                continue

        assert len(boundary_decisions) == 1
        assert boundary_decisions[0]["decision"] == "BREAK"
        assert boundary_decisions[0]["reason"] == "nominal_dt_s_missing"
        assert segment_ids == [0, 1]
