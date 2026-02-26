"""Stage E runner: Match session detection (hysteresis).

Exposes the stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Stage E implementation is owned by the Stage E workers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bjj_pipeline.contracts.f0_models import SCHEMA_VERSION_DEFAULT
from bjj_pipeline.contracts.f0_validate import validate_match_sessions_records
from bjj_pipeline.stages.matches.audit import _now_ms, append_audit_event
from bjj_pipeline.stages.matches.merge import merge_seeds_by_pair
from bjj_pipeline.stages.matches.seeds import SeedInterval, extract_cap2_seeds


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


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint.

	E0: plumbing + audit
	E1: seed extraction (cap2 nodes)
	E2: merge seeds per unordered pair + emit match_sessions.jsonl
	E3: enrich with April tag labels from stage_D/identity_assignments.jsonl
	"""
	layout = inputs["layout"]
	manifest = inputs["manifest"]

	stage_cfg = _stage_e_cfg(config)

	# -------------------
	# E0: plumbing + audit
	# -------------------
	person_spans_path = layout.person_spans_parquet()
	person_tracks_path = layout.person_tracks_parquet()
	identity_path = layout.identity_assignments_jsonl()

	missing_required = []
	if not person_spans_path.exists():
		missing_required.append(str(person_spans_path))

	missing_optional = []
	if not person_tracks_path.exists():
		missing_optional.append(str(person_tracks_path))
	if not identity_path.exists():
		missing_optional.append(str(identity_path))

	append_audit_event(
		layout=layout,
		event={
			"artifact_type": "stage_e_started",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"mode": "E2_merge_seeds_emit_match_sessions",
			"inputs": {
				"person_spans_parquet": str(person_spans_path),
				"person_tracks_parquet": str(person_tracks_path),
				"identity_assignments_jsonl": str(identity_path),
			},
			"missing_required": missing_required,
			"missing_optional": missing_optional,
		},
	)

	if missing_required:
		append_audit_event(
			layout=layout,
			event={
				"artifact_type": "stage_e_error",
				"created_at_ms": _now_ms(),
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"error": "missing_required_inputs",
				"missing": missing_required,
			},
		)
		raise FileNotFoundError(f"Stage E missing required inputs: {missing_required}")

	# -------------------
	# E1: seed extraction
	# -------------------
	spans_df = pd.read_parquet(person_spans_path)
	seeds = extract_cap2_seeds(spans_df)

	seed_match_id_by_seed = {}
	for s in seeds:
		key = (s.person_id_a, s.person_id_b, int(s.start_frame), int(s.end_frame), s.node_id)
		seed_match_id_by_seed[key] = _match_id_for_seed(s)

	frame_to_ts: Dict[int, int] = {}
	if person_tracks_path.exists():
		pt_df = pd.read_parquet(person_tracks_path)
		frame_to_ts = _build_frame_to_ts_map(pt_df)

	def _ts_for_frame(frame_index: int) -> int:
		if frame_index in frame_to_ts:
			return int(frame_to_ts[frame_index])
		# fallback: derive from fps
		return int(round((frame_index / float(manifest.fps)) * 1000.0))

	created_at = _now_ms()
	confidence = _seed_confidence(stage_cfg)
	max_gap_frames = _max_gap_frames(stage_cfg)

	merged = merge_seeds_by_pair(
		seeds=seeds,
		seed_match_id_by_seed=seed_match_id_by_seed,
		max_gap_frames=max_gap_frames,
	)

	records = []
	for m in merged:
		records.append(
			{
				"schema_version": SCHEMA_VERSION_DEFAULT,
				"artifact_type": "match_session",
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"pipeline_version": manifest.pipeline_version,
				"created_at_ms": created_at,
				"match_id": _match_id_for_merged(
					person_id_a=m.person_id_a,
					person_id_b=m.person_id_b,
					start_frame=int(m.start_frame),
					end_frame=int(m.end_frame),
					seed_match_ids=m.seed_match_ids,
				),
				"person_id_a": m.person_id_a,
				"person_id_b": m.person_id_b,
				"start_frame": int(m.start_frame),
				"end_frame": int(m.end_frame),
				"start_ts_ms": _ts_for_frame(int(m.start_frame)),
				"end_ts_ms": _ts_for_frame(int(m.end_frame)),
				# F0 constraint: method literal
				"method": "distance_hysteresis_v1",
				"confidence": float(confidence),
				"evidence": {
					"seed_source": "effective_cap_2_nodes",
					"pair_rule": "strict_unordered_pair",
					"note": "merged_from_seeds",
					"merge_max_gap_frames": int(max_gap_frames),
					"seed_node_ids": m.seed_node_ids,
					"seed_match_ids": m.seed_match_ids,
				},
			}
		)

	# -------------------
	# E3: enrichment (April tag labels)
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

	append_audit_event(
		layout=layout,
		event={
			"artifact_type": "e3_identity_enrichment_summary",
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
			"artifact_type": "e2_merge_summary",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"n_spans_total": int(len(spans_df)),
			"n_spans_cap2": int((spans_df["effective_cap"] == 2).sum()),
			"n_seeds": int(len(seeds)),
			"n_merged_sessions": int(len(merged)),
			"merge_max_gap_frames": int(max_gap_frames),
			"match_sessions_written": str(out_path),
			"timestamp_source": "person_tracks_parquet" if frame_to_ts else "fps_fallback",
		},
	)

	return {}


def main() -> None:
	raise SystemExit(
		"Stage E (matches) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
