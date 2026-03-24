"""Stage E runner: Match session detection (proximity hysteresis + cap2 seeds).

Exposes the stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Two-layer engagement detection:
  E0: plumbing + input validation
  E1: cap2 GROUP seed extraction
  E2: proximity hysteresis state machine
  E3: union seeds + proximity, apply buffer
  E4: buzzer soft gate (optional)
  E5: minimum duration filter + emit
  E6: identity enrichment (April tag labels)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bjj_pipeline.contracts.f0_models import SCHEMA_VERSION_DEFAULT
from bjj_pipeline.contracts.f0_validate import validate_match_sessions_records
from bjj_pipeline.stages.matches.audit import _now_ms, append_audit_event
from bjj_pipeline.stages.matches.buzzer import apply_buzzer_soft_gate, load_audio_events
from bjj_pipeline.stages.matches.hysteresis import EngagementInterval, run_proximity_hysteresis
from bjj_pipeline.stages.matches.merge import merge_seeds_by_pair, union_engagement_intervals
from bjj_pipeline.stages.matches.pairing import compute_pair_distances
from bjj_pipeline.stages.matches.seeds import SeedInterval, extract_cap2_seeds
from bjj_pipeline.stages.orchestration.pipeline import PipelineError


def _match_id_for_seed(seed: SeedInterval) -> str:
	raw = f"{seed.person_id_a}|{seed.person_id_b}|{seed.start_frame}|{seed.end_frame}|{seed.node_id}"
	digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
	return f"mseed_{digest}"


def _match_id_for_merged(
	*, person_id_a: str, person_id_b: str, start_frame: int, end_frame: int, seed_match_ids: list[str]
) -> str:
	raw = f"{person_id_a}|{person_id_b}|{start_frame}|{end_frame}|{','.join(seed_match_ids)}"
	digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
	return f"mmerge_{digest}"


def _match_id_for_engagement(
	*, person_id_a: str, person_id_b: str, start_frame: int, end_frame: int,
	evidence_sources: Tuple[str, ...],
) -> str:
	raw = f"{person_id_a}|{person_id_b}|{start_frame}|{end_frame}|{','.join(evidence_sources)}"
	digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
	return f"mengage_{digest}"


def _build_frame_to_ts_map(person_tracks_df: Optional[pd.DataFrame]) -> Dict[int, int]:
	if person_tracks_df is None or person_tracks_df.empty:
		return {}
	if "frame_index" not in person_tracks_df.columns or "timestamp_ms" not in person_tracks_df.columns:
		return {}

	# timestamps should be identical across persons for the same frame; take first
	tmp = person_tracks_df[["frame_index", "timestamp_ms"]].dropna()
	tmp = tmp.sort_values(["frame_index", "timestamp_ms"])
	return {int(r.frame_index): int(r.timestamp_ms) for r in tmp.drop_duplicates("frame_index").itertuples(index=False)}


def _stage_e_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
	# Preferred: repo-wide structure from configs/default.yaml
	stages_blk = config.get("stages")
	if isinstance(stages_blk, dict):
		e_blk = stages_blk.get("stage_E")
		if isinstance(e_blk, dict):
			return e_blk

	# Backward-compatible fallback (older overlays/tests)
	blk = config.get("stage_E")
	return blk if isinstance(blk, dict) else {}


def _seed_confidence(cfg: Dict[str, Any]) -> float:
	v = cfg.get("seed_confidence", 0.70)
	try:
		return float(v)
	except Exception:
		return 0.70


def _max_gap_frames(cfg: Dict[str, Any]) -> int:
	v = cfg.get("max_gap_frames", 30)
	try:
		return int(v)
	except Exception:
		return 30


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
	v = cfg.get(key, default)
	try:
		return float(v)
	except Exception:
		return default


def _cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
	v = cfg.get(key, default)
	try:
		return int(v)
	except Exception:
		return default


def _load_identity_assignments(
	*, path: Path, expected_clip_id: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
	"""
	Load stage_D/identity_assignments.jsonl and choose the best assignment per person_id.

	Rule:
	- choose max assignment_confidence
	- tie-break by tag_id lexicographically
	- log duplicate cases (returned in stats)
	"""
	best: Dict[str, Dict[str, Any]] = {}
	all_counts: Dict[str, int] = {}
	duplicate_details: List[Dict[str, Any]] = []

	if not path.exists():
		return best, {
			"exists": False,
			"n_lines": 0,
			"n_records": 0,
			"n_best": 0,
			"n_persons_with_duplicates": 0,
			"duplicate_samples": [],
		}

	n_lines = 0
	n_records = 0

	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			n_lines += 1
			try:
				rec = json.loads(line)
			except Exception:
				# ignore malformed lines; Stage D should be clean, but we won't crash E3
				continue

			if not isinstance(rec, dict):
				continue

			# Accept only the identity_assignment artifact type
			if rec.get("artifact_type") != "identity_assignment":
				continue

			clip_id = rec.get("clip_id")
			if isinstance(clip_id, str) and clip_id != expected_clip_id:
				# Different clip in same file (shouldn't happen) -> ignore
				continue

			person_id = rec.get("person_id")
			tag_id = rec.get("tag_id")
			conf = rec.get("assignment_confidence")

			if not isinstance(person_id, str) or not isinstance(tag_id, str):
				continue
			try:
				conf_f = float(conf)
			except Exception:
				conf_f = 0.0

			n_records += 1
			all_counts[person_id] = all_counts.get(person_id, 0) + 1

			# choose best by (confidence desc, tag_id asc)
			key = (conf_f, tag_id)

			prev = best.get(person_id)
			if prev is None:
				rec2 = dict(rec)
				rec2["_cmp"] = key
				best[person_id] = rec2
			else:
				prev_key = prev.get("_cmp", (0.0, ""))
				if key[0] > prev_key[0] or (key[0] == prev_key[0] and key[1] < prev_key[1]):
					duplicate_details.append(
						{
							"person_id": person_id,
							"kept": {"tag_id": tag_id, "assignment_confidence": conf_f},
							"dropped": {
								"tag_id": prev.get("tag_id"),
								"assignment_confidence": float(prev_key[0]),
							},
						}
					)
					rec2 = dict(rec)
					rec2["_cmp"] = key
					best[person_id] = rec2
				else:
					duplicate_details.append(
						{
							"person_id": person_id,
							"kept": {"tag_id": prev.get("tag_id"), "assignment_confidence": float(prev_key[0])},
							"dropped": {"tag_id": tag_id, "assignment_confidence": conf_f},
						}
					)

	# clean internal field
	for v in best.values():
		v.pop("_cmp", None)

	n_persons_with_dupes = sum(1 for _, c in all_counts.items() if c > 1)

	# keep audit payload bounded
	dup_samples = duplicate_details[:10]

	return best, {
		"exists": True,
		"n_lines": int(n_lines),
		"n_records": int(n_records),
		"n_best": int(len(best)),
		"n_persons_with_duplicates": int(n_persons_with_dupes),
		"duplicate_samples": dup_samples,
	}


def _apply_identity_enrichment(
	*,
	match_records: List[Dict[str, Any]],
	best_by_person_id: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
	"""
	Mutates match_records in-place by adding explicit null fields:
	  evidence.april_tag_id_a / evidence.april_tag_id_b
	Also attaches confidence + anchor_key if present in identity assignment evidence.
	"""
	labeled_a = 0
	labeled_b = 0
	total = 0

	for rec in match_records:
		if not isinstance(rec, dict):
			continue
		if rec.get("artifact_type") != "match_session":
			continue

		total += 1
		pa = rec.get("person_id_a")
		pb = rec.get("person_id_b")

		ev = rec.get("evidence")
		if not isinstance(ev, dict):
			ev = {}
			rec["evidence"] = ev

		# Defaults: explicit nulls
		ev["april_tag_id_a"] = None
		ev["april_tag_id_b"] = None
		ev["april_assignment_confidence_a"] = None
		ev["april_assignment_confidence_b"] = None
		ev["april_anchor_key_a"] = None
		ev["april_anchor_key_b"] = None

		if isinstance(pa, str) and pa in best_by_person_id:
			a = best_by_person_id[pa]
			ev["april_tag_id_a"] = a.get("tag_id")
			ev["april_assignment_confidence_a"] = a.get("assignment_confidence")
			a_ev = a.get("evidence")
			if isinstance(a_ev, dict):
				ev["april_anchor_key_a"] = a_ev.get("anchor_key")
			labeled_a += 1

		if isinstance(pb, str) and pb in best_by_person_id:
			b = best_by_person_id[pb]
			ev["april_tag_id_b"] = b.get("tag_id")
			ev["april_assignment_confidence_b"] = b.get("assignment_confidence")
			b_ev = b.get("evidence")
			if isinstance(b_ev, dict):
				ev["april_anchor_key_b"] = b_ev.get("anchor_key")
			labeled_b += 1

	return {
		"n_sessions": int(total),
		"n_labeled_a": int(labeled_a),
		"n_labeled_b": int(labeled_b),
	}


def _promote_april_fields(records: List[Dict[str, Any]]) -> None:
	"""Move april tag fields from evidence sub-dict to top-level.

	_apply_identity_enrichment writes into evidence.*; the v2 schema
	expects top-level april_* fields. This shim promotes them.
	"""
	for rec in records:
		ev = rec.get("evidence")
		if not isinstance(ev, dict):
			continue

		for suffix in ("a", "b"):
			for field_base in ("april_tag_id", "april_assignment_confidence", "april_anchor_key"):
				key = f"{field_base}_{suffix}"
				rec[key] = ev.pop(key, None)


def _derive_frame_bounds(
	spans_df: Optional[pd.DataFrame],
	tracks_df: Optional[pd.DataFrame],
) -> Tuple[int, int]:
	"""Derive session frame bounds from actual data ranges.

	Returns (session_start_frame, session_end_frame) from the union of
	available data. Uses min of all start frames and max of all end frames.
	"""
	starts: List[int] = []
	ends: List[int] = []

	if spans_df is not None and not spans_df.empty:
		if "start_frame" in spans_df.columns:
			val = spans_df["start_frame"].dropna()
			if not val.empty:
				starts.append(int(val.min()))
		if "end_frame" in spans_df.columns:
			val = spans_df["end_frame"].dropna()
			if not val.empty:
				ends.append(int(val.max()))

	if tracks_df is not None and not tracks_df.empty:
		if "frame_index" in tracks_df.columns:
			val = tracks_df["frame_index"].dropna()
			if not val.empty:
				starts.append(int(val.min()))
				ends.append(int(val.max()))

	if not starts or not ends:
		return (0, 0)

	return (min(starts), max(ends))


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint.

	E0: plumbing + input validation
	E1: cap2 GROUP seed extraction
	E2: proximity hysteresis
	E3: union seeds + proximity, apply buffer
	E4: buzzer soft gate (optional)
	E5: minimum duration filter + emit
	E6: identity enrichment (April tag labels)
	"""
	layout = inputs["layout"]
	manifest = inputs["manifest"]

	stage_cfg = _stage_e_cfg(config)

	# -------------------
	# E0: plumbing + input validation
	# -------------------
	person_spans_path = layout.person_spans_parquet()
	person_tracks_path = layout.person_tracks_parquet()
	identity_path = layout.identity_assignments_jsonl()

	# Audio events path: session-level (wired for future Stage A.5)
	audio_events_path = layout.stage_dir("E") / "audio_events.jsonl"

	has_spans = person_spans_path.exists()
	has_tracks = person_tracks_path.exists()

	missing_inputs = []
	if not has_spans:
		missing_inputs.append(f"person_spans: {person_spans_path}")
	if not has_tracks:
		missing_inputs.append(f"person_tracks: {person_tracks_path}")

	append_audit_event(
		layout=layout,
		event={
			"artifact_type": "stage_e_started",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"mode": "proximity_hysteresis_v2",
			"inputs": {
				"person_spans_parquet": str(person_spans_path),
				"person_tracks_parquet": str(person_tracks_path),
				"identity_assignments_jsonl": str(identity_path),
				"audio_events_jsonl": str(audio_events_path),
			},
			"missing_inputs": missing_inputs,
			"has_spans": has_spans,
			"has_tracks": has_tracks,
		},
	)

	if not has_spans and not has_tracks:
		append_audit_event(
			layout=layout,
			event={
				"artifact_type": "stage_e_error",
				"created_at_ms": _now_ms(),
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"error": "no_valid_frames",
				"detail": "both person_spans and person_tracks are missing",
			},
		)
		raise PipelineError("Stage E: no valid frames — both person_spans and person_tracks are missing")

	# Load available data
	spans_df: Optional[pd.DataFrame] = None
	tracks_df: Optional[pd.DataFrame] = None

	if has_spans:
		spans_df = pd.read_parquet(person_spans_path)
		if spans_df.empty:
			spans_df = None
			logger.warning("Stage E: person_spans.parquet exists but is empty")

	if has_tracks:
		tracks_df = pd.read_parquet(person_tracks_path)
		if tracks_df.empty:
			tracks_df = None
			logger.warning("Stage E: person_tracks.parquet exists but is empty")

	# Derive frame bounds from actual data
	session_start_frame, session_end_frame = _derive_frame_bounds(spans_df, tracks_df)

	# Timestamp map for record emission
	frame_to_ts = _build_frame_to_ts_map(tracks_df)

	def _ts_for_frame(frame_index: int) -> int:
		if frame_index in frame_to_ts:
			return int(frame_to_ts[frame_index])
		# fallback: derive from fps
		return int(round((frame_index / float(manifest.fps)) * 1000.0))

	# Config values
	confidence = _seed_confidence(stage_cfg)
	max_gap = _max_gap_frames(stage_cfg)
	engage_dist_m = _cfg_float(stage_cfg, "engage_dist_m", 0.75)
	disengage_dist_m = _cfg_float(stage_cfg, "disengage_dist_m", 2.0)
	engage_min_frames = _cfg_int(stage_cfg, "engage_min_frames", 15)
	hysteresis_frames = _cfg_int(stage_cfg, "hysteresis_frames", 450)
	min_clip_duration_frames = _cfg_int(stage_cfg, "min_clip_duration_frames", 150)
	clip_buffer_frames = _cfg_int(stage_cfg, "clip_buffer_frames", 45)
	buzzer_boundary_window_frames = _cfg_int(stage_cfg, "buzzer_boundary_window_frames", 90)

	# -------------------
	# E1: cap2 GROUP seed extraction
	# -------------------
	seeds: List[SeedInterval] = []
	if spans_df is not None:
		seeds = extract_cap2_seeds(spans_df)
	logger.info("Stage E1: {} cap2 seeds extracted", len(seeds))

	# -------------------
	# E2: proximity hysteresis
	# -------------------
	proximity_intervals: List[EngagementInterval] = []
	pair_dist_df = pd.DataFrame(columns=["frame_index", "person_id_a", "person_id_b", "dist_m"])

	if tracks_df is not None:
		pair_dist_df = compute_pair_distances(tracks_df, fps=manifest.fps)
		if not pair_dist_df.empty:
			proximity_intervals = run_proximity_hysteresis(
				pair_dist_df,
				session_start_frame=session_start_frame,
				session_end_frame=session_end_frame,
				engage_dist_m=engage_dist_m,
				disengage_dist_m=disengage_dist_m,
				engage_min_frames=engage_min_frames,
				hysteresis_frames=hysteresis_frames,
				min_clip_duration_frames=min_clip_duration_frames,
			)
	logger.info("Stage E2: {} proximity intervals", len(proximity_intervals))

	# -------------------
	# E3: union seeds + proximity, apply buffer
	# -------------------
	merged = union_engagement_intervals(
		proximity_intervals=proximity_intervals,
		seed_intervals=seeds,
		max_gap_frames=max_gap,
		session_start_frame=session_start_frame,
		session_end_frame=session_end_frame,
		clip_buffer_frames=clip_buffer_frames,
	)
	logger.info("Stage E3: {} merged intervals after union + buffer", len(merged))

	# -------------------
	# E4: buzzer soft gate (optional)
	# -------------------
	audio_events = load_audio_events(audio_events_path)
	if audio_events:
		merged = apply_buzzer_soft_gate(
			merged, audio_events,
			fps=manifest.fps,
			buzzer_boundary_window_frames=buzzer_boundary_window_frames,
			pair_distances_df=pair_dist_df,
			disengage_dist_m=disengage_dist_m,
		)
		logger.info("Stage E4: buzzer gate applied ({} audio events)", len(audio_events))
	else:
		logger.debug("Stage E4: no audio events — buzzer gate skipped")

	# -------------------
	# E5: minimum duration filter + emit
	# -------------------
	final = [m for m in merged if (m.end_frame - m.start_frame) >= min_clip_duration_frames]

	# Collect diagnostic stats for audit
	n_pairs_evaluated = len(pair_dist_df[["person_id_a", "person_id_b"]].drop_duplicates()) if not pair_dist_df.empty else 0
	max_proximity_reached_m: Optional[float] = None
	if not pair_dist_df.empty:
		max_proximity_reached_m = float(pair_dist_df["dist_m"].min())

	if not final:
		# Write empty match_sessions.jsonl
		out_path = layout.match_sessions_jsonl()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with out_path.open("w", encoding="utf-8") as f:
			pass  # empty file

		# Determine reason
		if not seeds and not proximity_intervals:
			reason = "no_engagement_found"
		elif merged and not final:
			reason = "all_below_min_duration"
		else:
			reason = "no_engagement_found"

		append_audit_event(
			layout=layout,
			event={
				"artifact_type": "session_no_matches",
				"created_at_ms": _now_ms(),
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"n_pairs_evaluated": int(n_pairs_evaluated),
				"max_proximity_reached_m": max_proximity_reached_m,
				"n_cap2_seeds": len(seeds),
				"n_proximity_intervals_before_filter": len(proximity_intervals),
				"n_merged_before_filter": len(merged),
				"reason": reason,
			},
		)
		logger.info("Stage E5: no match sessions — reason={}", reason)
		return {}

	# Build seed match ID map for evidence tracking
	seed_match_id_by_seed: Dict[Tuple[str, str, int, int, str], str] = {}
	for s in seeds:
		key = (s.person_id_a, s.person_id_b, int(s.start_frame), int(s.end_frame), s.node_id)
		seed_match_id_by_seed[key] = _match_id_for_seed(s)

	# Also run the existing seed merge to get seed_node_ids and seed_match_ids per pair
	merged_seeds = merge_seeds_by_pair(
		seeds=seeds,
		seed_match_id_by_seed=seed_match_id_by_seed,
		max_gap_frames=max_gap,
	) if seeds else []

	# Build lookup: (person_id_a, person_id_b) -> merged seed evidence
	seed_evidence_by_pair: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
	for ms in merged_seeds:
		k = (ms.person_id_a, ms.person_id_b)
		existing = seed_evidence_by_pair.get(k, {"node_ids": [], "match_ids": []})
		existing["node_ids"].extend(ms.seed_node_ids)
		existing["match_ids"].extend(ms.seed_match_ids)
		seed_evidence_by_pair[k] = existing

	created_at = _now_ms()

	records: List[Dict[str, Any]] = []
	for eng in final:
		pair_key = (eng.person_id_a, eng.person_id_b)
		seed_ev = seed_evidence_by_pair.get(pair_key, {"node_ids": [], "match_ids": []})

		match_id = _match_id_for_engagement(
			person_id_a=eng.person_id_a,
			person_id_b=eng.person_id_b,
			start_frame=eng.start_frame,
			end_frame=eng.end_frame,
			evidence_sources=eng.evidence_sources,
		)

		records.append({
			"schema_version": SCHEMA_VERSION_DEFAULT,
			"artifact_type": "match_session",
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"pipeline_version": manifest.pipeline_version,
			"created_at_ms": created_at,
			"match_id": match_id,
			"person_id_a": eng.person_id_a,
			"person_id_b": eng.person_id_b,
			"start_frame": int(eng.start_frame),
			"end_frame": int(eng.end_frame),
			"start_ts_ms": _ts_for_frame(int(eng.start_frame)),
			"end_ts_ms": _ts_for_frame(int(eng.end_frame)),
			"partial_start": eng.partial_start,
			"partial_end": eng.partial_end,
			"method": "proximity_hysteresis_v2",
			"confidence": float(confidence),
			"evidence": {
				"sources": list(eng.evidence_sources),
				"engage_dist_m": engage_dist_m,
				"disengage_dist_m": disengage_dist_m,
				"hysteresis_frames": hysteresis_frames,
				"clip_buffer_frames": clip_buffer_frames,
				"buzzer_adjusted": "buzzer" in eng.evidence_sources,
				"seed_node_ids": sorted(set(seed_ev["node_ids"])),
				"seed_match_ids": sorted(set(seed_ev["match_ids"])),
			},
			# Top-level april tag fields — populated by E6 enrichment
			"april_tag_id_a": None,
			"april_tag_id_b": None,
			"april_assignment_confidence_a": None,
			"april_assignment_confidence_b": None,
			"april_anchor_key_a": None,
			"april_anchor_key_b": None,
		})

	# -------------------
	# E6: identity enrichment (April tag labels)
	# -------------------
	identity_path = layout.identity_assignments_jsonl()
	best_by_person_id, ia_stats = _load_identity_assignments(
		path=identity_path,
		expected_clip_id=manifest.clip_id,
	)
	enrich_stats = _apply_identity_enrichment(
		match_records=records,
		best_by_person_id=best_by_person_id,
	)

	# Promote april fields from evidence sub-dict to top-level
	_promote_april_fields(records)

	append_audit_event(
		layout=layout,
		event={
			"artifact_type": "e6_identity_enrichment_summary",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"identity_assignments_path": str(identity_path),
			"identity_assignments_stats": ia_stats,
			"enrichment_stats": enrich_stats,
		},
	)

	validate_match_sessions_records(records, expected_clip_id=manifest.clip_id)

	out_path = layout.match_sessions_jsonl()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		for r in records:
			f.write(json.dumps(r, sort_keys=True))
			f.write("\n")

	append_audit_event(
		layout=layout,
		event={
			"artifact_type": "e5_summary",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"session_start_frame": session_start_frame,
			"session_end_frame": session_end_frame,
			"n_cap2_seeds": len(seeds),
			"n_proximity_intervals": len(proximity_intervals),
			"n_merged_after_union": len(merged),
			"n_final_sessions": len(final),
			"n_pairs_evaluated": int(n_pairs_evaluated),
			"max_proximity_reached_m": max_proximity_reached_m,
			"n_buzzer_events": len(audio_events),
			"min_clip_duration_frames": min_clip_duration_frames,
			"match_sessions_written": str(out_path),
			"timestamp_source": "person_tracks_parquet" if frame_to_ts else "fps_fallback",
		},
	)

	logger.info("Stage E: wrote {} match sessions → {}", len(final), out_path.name)
	return {}


def main() -> None:
	raise SystemExit(
		"Stage E (matches) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
