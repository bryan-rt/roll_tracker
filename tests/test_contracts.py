"""
Role: unit tests for F0 data contracts (authoritative).

These tests enforce:
- Parquet schema shape + dtype-family compliance
- FK/traceability invariants between detections, tracklets, person tracks
- identity_hints / identity_assignments JSONL shape constraints
- clip_manifest write/read + artifact registration

These tests are intentionally "small and synthetic" to run fast in CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from bjj_pipeline.contracts import f0_parquet as pq
from bjj_pipeline.contracts import f0_validate as v
from bjj_pipeline.contracts.f0_manifest import init_manifest, load_manifest, write_manifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


def _now_ms() -> int:
	return int(time.time() * 1000)


def _make_min_detections_df() -> pd.DataFrame:
	# Two frames, one detection each
	return pd.DataFrame(
		[
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"frame_index": 0,
				"timestamp_ms": 0,
				"detection_id": "det0",
				"class_name": "person",
				"confidence": 0.9,
				"x1": 10.0,
				"y1": 20.0,
				"x2": 110.0,
				"y2": 220.0,
			},
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"frame_index": 1,
				"timestamp_ms": 33,
				"detection_id": "det1",
				"class_name": "person",
				"confidence": 0.95,
				"x1": 12.0,
				"y1": 22.0,
				"x2": 112.0,
				"y2": 222.0,
			},
		]
	)


def _make_min_tracklet_frames_df() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"tracklet_id": "trkA",
				"frame_index": 0,
				"timestamp_ms": 0,
				"detection_id": "det0",
			},
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"tracklet_id": "trkA",
				"frame_index": 1,
				"timestamp_ms": 33,
				"detection_id": "det1",
			},
		]
	)


def _make_min_tracklet_summaries_df() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"tracklet_id": "trkA",
				"start_frame": 0,
				"end_frame": 1,
				"n_frames": 2,
				# optional mean bbox
				"mean_x1": 11.0,
				"mean_y1": 21.0,
				"mean_x2": 111.0,
				"mean_y2": 221.0,
			}
		]
	)


def _make_min_d2_edge_costs_df() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"edge_id": "e0",
				"edge_type": "EdgeType.CONTINUE",
				"src_node_id": "n0",
				"dst_node_id": "n1",
				"is_allowed": True,
				"disallow_reasons_json": "[]",
				"dt_frames": 3,
				"dt_s": 0.1,
				"dist_m": 0.2,
				"v_req_mps": 2.0,
				"dist_norm": 0.5,
				"contact_rel": 0.9,
				"endpoint_flagged": False,
				"term_env": 0.01,
				"term_time": 0.01,
				"term_vreq": 0.0,
				"term_missing_geom": 0.0,
				"term_flags": 0.0,
				"term_group_coherence": 0.0,
				"term_birth_prior": 0.0,
				"term_death_prior": 0.0,
				"term_merge_prior": 0.0,
				"term_split_prior": 0.0,
				"total_cost": 0.02,
			}
		]
	)


def test_parquet_schema_ok_d2_edge_costs() -> None:
	df = _make_min_d2_edge_costs_df()
	pq.validate_df_schema_by_key(df, "d2_edge_costs")
	v.validate_d2_edge_costs_df(df)


def _make_min_person_tracks_df() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"person_id": "p0",
				"frame_index": 0,
				"timestamp_ms": 0,
				"detection_id": "det0",
				"tracklet_id": "trkA",
				"x1": 10.0,
				"y1": 20.0,
				"x2": 110.0,
				"y2": 220.0,
				"x_m": 0.1,
				"y_m": 0.2,
				"stitch_edge_type": "same_tracklet",
			},
			{
				"clip_id": "cam03-20260103-124000",
				"camera_id": "cam03",
				"person_id": "p0",
				"frame_index": 1,
				"timestamp_ms": 33,
				"detection_id": "det1",
				"tracklet_id": "trkA",
				"x1": 12.0,
				"y1": 22.0,
				"x2": 112.0,
				"y2": 222.0,
				"x_m": 0.11,
				"y_m": 0.21,
				"stitch_edge_type": "same_tracklet",
			},
		]
	)


def test_parquet_schema_ok_detections():
	df = _make_min_detections_df()
	pq.validate_df_schema_by_key(df, "detections")  # should not raise


def test_parquet_schema_rejects_unexpected_column():
	df = _make_min_detections_df()
	df["unexpected"] = 123
	with pytest.raises(pq.ParquetSchemaError):
		pq.validate_df_schema_by_key(df, "detections")


def test_detections_mask_ref_must_be_relative():
	df = _make_min_detections_df()
	# relative path is ok
	df.loc[0, "mask_ref"] = "stage_A/masks/frame_000000_det_d1.npz"
	v.validate_detections_df(df)

	# absolute path is rejected (non-portable)
	df.loc[0, "mask_ref"] = "/tmp/frame_000000_det_d1.npz"
	with pytest.raises(v.ValidationError):
		v.validate_detections_df(df)


def test_detections_invariant_rejects_bad_bbox():
	df = _make_min_detections_df()
	df.loc[0, "x2"] = 5.0  # x2 < x1 invalid
	with pytest.raises(v.ValidationError):
		v.validate_detections_df(df)


def test_tracklet_fk_to_detections_ok():
	det = _make_min_detections_df()
	tf = _make_min_tracklet_frames_df()
	ts = _make_min_tracklet_summaries_df()

	v.validate_detections_df(det)
	v.validate_tracklet_tables(tf, ts)
	v.validate_tracklet_frames_fk_to_detections(tf, det)  # should not raise


def test_tracklet_fk_to_detections_missing_detection_fails():
	det = _make_min_detections_df()
	tf = _make_min_tracklet_frames_df()
	ts = _make_min_tracklet_summaries_df()

	# break FK
	tf.loc[0, "detection_id"] = "det_DOES_NOT_EXIST"

	with pytest.raises(v.ValidationError):
		v.validate_tracklet_frames_fk_to_detections(tf, det)


def test_tracklet_frames_requires_tracklet_in_summaries():
	tf = _make_min_tracklet_frames_df()
	ts = _make_min_tracklet_summaries_df()

	# break cross-table requirement by changing summary tracklet_id
	ts.loc[0, "tracklet_id"] = "trkB"

	with pytest.raises(v.ValidationError):
		v.validate_tracklet_tables(tf, ts)


def test_person_tracks_traceability_ok():
	det = _make_min_detections_df()
	tf = _make_min_tracklet_frames_df()
	ts = _make_min_tracklet_summaries_df()
	pt = _make_min_person_tracks_df()

	# schema + invariants
	v.validate_detections_df(det)
	v.validate_tracklet_tables(tf, ts)
	v.validate_tracklet_frames_fk_to_detections(tf, det)
	v.validate_person_tracks_df(pt)
	v.validate_person_tracks_traceability(pt, det, tf)  # should not raise


def test_person_tracks_traceability_missing_tracklet_fails():
	det = _make_min_detections_df()
	tf = _make_min_tracklet_frames_df()
	pt = _make_min_person_tracks_df()

	# break tracklet id reference
	pt.loc[0, "tracklet_id"] = "trk_DOES_NOT_EXIST"

	with pytest.raises(v.ValidationError):
		v.validate_person_tracks_traceability(pt, det, tf)


def test_identity_hints_shape_ok():
	records = [
		{
			"schema_version": "0.1.0",
			"artifact_type": "identity_hint",
			"clip_id": "cam03-20260103-124000",
			"camera_id": "cam03",
			"pipeline_version": "dev",
			"created_at_ms": _now_ms(),
			"tracklet_id": "trkA",
			"anchor_key": "tag:23",
			"constraint": "must_link",
			"confidence": 0.9,
			"evidence": {"votes": 12, "frames_seen": [10, 11, 12]},
		}
	]
	v.validate_identity_hints_records(records, expected_clip_id="cam03-20260103-124000")


def test_identity_hints_rejects_non_namespaced_anchor_key():
	records = [
		{
			"schema_version": "0.1.0",
			"artifact_type": "identity_hint",
			"clip_id": "cam03-20260103-124000",
			"camera_id": "cam03",
			"pipeline_version": "dev",
			"created_at_ms": _now_ms(),
			"tracklet_id": "trkA",
			"anchor_key": "23",  # invalid (no namespace)
			"constraint": "must_link",
			"confidence": 0.9,
			"evidence": {"votes": 12},
		}
	]
	with pytest.raises(v.ValidationError):
		v.validate_identity_hints_records(records)


def test_manifest_round_trip_and_artifact_registration(tmp_path: Path):
	clip_id = "cam03-20260103-124000"
	camera_id = "cam03"

	m = init_manifest(
		clip_id=clip_id,
		camera_id=camera_id,
		input_video_path="data/raw/nest/cam03/2026-01-03/12/cam03-20260103-124000.mp4",
		fps=30.0,
		frame_count=75,
		duration_ms=2500,
		pipeline_version="dev",
		created_at_ms=_now_ms(),
	)

	layout = ClipOutputLayout(clip_id=clip_id, root=tmp_path / "outputs")

	# Register a canonical artifact path via layout (clip-relative)
	layout.ensure_dirs_for_stage("A")
	rel = layout.rel_to_clip_root(layout.detections_parquet())
	m.register_artifact(stage="A", key="detections_parquet", relpath=rel, content_type="application/parquet")

	# Round trip write/read
	manifest_path = layout.clip_manifest_path()
	write_manifest(m, manifest_path)
	m2 = load_manifest(manifest_path)

	assert m2.clip_id == clip_id
	assert m2.get_artifact_path(stage="A", key="detections_parquet") == rel
