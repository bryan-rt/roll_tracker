from __future__ import annotations

from typing import Any, Dict, Optional

from bjj_pipeline.stages.tags.identity_registry import C2IdentityRegistry


def _mk_obs(
	*,
	tracklet_id: str,
	detection_id: str,
	frame_index: int,
	tag_id: str,
	confidence: float,
	roi_method: Optional[str] = "bbox_pad_frac",
) -> Dict[str, Any]:
	return {
		"schema_version": "0",
		"artifact_type": "tag_observation",
		"clip_id": "clip1",
		"camera_id": "cam01",
		"pipeline_version": "0",
		"created_at_ms": 0,
		"frame_index": int(frame_index),
		"timestamp_ms": int(frame_index * 33),
		"detection_id": str(detection_id),
		"tracklet_id": str(tracklet_id),
		"tag_id": str(tag_id),
		"tag_family": "36h11",
		"confidence": float(confidence),
		"roi_method": str(roi_method) if roi_method is not None else None,
	}


def _mk_registry(cfg: Optional[Dict[str, Any]] = None) -> C2IdentityRegistry:
	return C2IdentityRegistry(
		cfg=cfg or {},
		clip_id="clip1",
		camera_id="cam01",
		pipeline_version="0",
		created_at_ms=0,
	)


def test_single_ping_strong_emits_must_link_tentative() -> None:
	r = _mk_registry({"min_conf": 0.70, "min_weight": 0.70, "min_conf_strong_single": 0.90, "min_margin_abs": 0.15})
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d1", frame_index=10, tag_id="7", confidence=0.95))
	hints, events = r.finalize(tracklet_spans=None)
	must = [h for h in hints if h["constraint"] == "must_link"]
	assert len(must) == 1
	h = must[0]
	assert h["tracklet_id"] == "t1"
	assert h["anchor_key"] == "tag:7"
	assert h["evidence"]["strength"] == "tentative"


def test_single_ping_not_strong_single_no_must_link() -> None:
	r = _mk_registry({"min_conf": 0.70, "min_weight": 0.70, "min_conf_strong_single": 0.90, "min_margin_abs": 0.15})
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d1", frame_index=10, tag_id="7", confidence=0.80))
	hints, events = r.finalize(tracklet_spans=None)
	assert [h for h in hints if h["constraint"] == "must_link"] == []


def test_two_pings_emits_strong_must_link() -> None:
	r = _mk_registry({"min_conf": 0.70, "min_weight": 0.70, "min_conf_strong_single": 0.90, "min_margin_abs": 0.15})
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d1", frame_index=10, tag_id="7", confidence=0.80))
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d2", frame_index=20, tag_id="7", confidence=0.80))
	hints, _ = r.finalize(tracklet_spans=None)
	must = [h for h in hints if h["constraint"] == "must_link"]
	assert len(must) == 1
	assert must[0]["evidence"]["strength"] == "strong"


def test_type_a_conflict_suppresses_must_link() -> None:
	# Two single strong pings of different tags, close in weight => Type A conflict => no must_link.
	r = _mk_registry({"min_conf": 0.70, "min_weight": 0.70, "min_conf_strong_single": 0.90, "min_margin_abs": 0.15})
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d1", frame_index=10, tag_id="7", confidence=0.95))
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d2", frame_index=11, tag_id="8", confidence=0.92))
	hints, events = r.finalize(tracklet_spans=None)
	assert [h for h in hints if h["constraint"] == "must_link"] == []
	assert any(e.get("event") == "c2_conflict_two_tags_one_tracklet" for e in events)


def test_type_b_conflict_emits_symmetric_cannot_links_even_if_must_link_suppressed() -> None:
	# t1 has Type A conflict (no must_link), but still supports tag 7 strongly.
	# t2 supports tag 7 strongly and overlaps => cannot_link both directions.
	r = _mk_registry({"min_conf": 0.70, "min_weight": 0.70, "min_conf_strong_single": 0.90, "min_margin_abs": 0.15})

	# Tracklet t1: two strong singles close => suppress must_link, but tag 7 supported.
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d1", frame_index=10, tag_id="7", confidence=0.95))
	r.ingest_tag_observation(_mk_obs(tracklet_id="t1", detection_id="d2", frame_index=11, tag_id="8", confidence=0.92))

	# Tracklet t2: strong for tag 7
	r.ingest_tag_observation(_mk_obs(tracklet_id="t2", detection_id="d3", frame_index=11, tag_id="7", confidence=0.93))

	# Spans overlap (t1:10-11, t2:11-11)
	hints, events = r.finalize(tracklet_spans={"t1": (10, 11), "t2": (11, 11)})
	cl = [h for h in hints if h["constraint"] == "cannot_link"]
	# expect symmetric
	assert {(h["tracklet_id"], h["anchor_key"]) for h in cl} == {("t1", "tracklet:t2"), ("t2", "tracklet:t1")}
