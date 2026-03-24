"""Stage D2 — Normalize identity constraints (solver-agnostic).

Inputs:
  - stage_C/identity_hints.jsonl

Outputs:
  - dict suitable for JSON serialization (written by d2_run.py)

Semantics (POC locked):
  - must_link: tracklet_id -> anchor_key "tag:<id>" (hard)
  - cannot_link: tracklet_id -> anchor_key "tracklet:<other_tracklet_id>" (hard, symmetric)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
	if not path.exists():
		return []
	lines = path.read_text(encoding="utf-8").splitlines()
	out: List[Dict[str, Any]] = []
	for line in lines:
		line = line.strip()
		if not line:
			continue
		out.append(json.loads(line))
	return out


def normalize_identity_constraints(identity_hints_path: Path) -> Dict[str, Any]:
	events = _read_jsonl(identity_hints_path)

	# Evidence counters (additive: stored in stats)
	n_events_read = len(events)
	n_events_used = 0
	n_events_skipped_non_identity_hint = 0
	n_events_missing_artifact_type = 0
	schema_versions_seen: Set[str] = set()

	must: Dict[str, Set[str]] = {}
	cannot_pairs: Set[Tuple[str, str]] = set()
	# Time-local tag pings: (tracklet_id, anchor_key, frame_index) -> best_confidence
	tag_pings_best: Dict[Tuple[str, str, int], Optional[float]] = {}
	n_tag_pings_raw = 0
	n_tag_pings_missing_frame_index = 0

	for ev in events:
		# Optional hygiene: filter by artifact_type if provided
		if "schema_version" in ev:
			schema_versions_seen.add(str(ev.get("schema_version")))
		if "artifact_type" in ev:
			if ev.get("artifact_type") != "identity_hint":
				n_events_skipped_non_identity_hint += 1
				continue
		else:
			# Backward compatible: accept legacy events with no artifact_type,
			# but record that the field is missing.
			n_events_missing_artifact_type += 1

		constraint = ev.get("constraint", None)
		tracklet_id = ev.get("tracklet_id", None)
		anchor_key = ev.get("anchor_key", None)
		if constraint not in ("must_link", "cannot_link"):
			continue
		if not isinstance(tracklet_id, str) or not tracklet_id:
			continue
		if not isinstance(anchor_key, str) or not anchor_key:
			continue

		n_events_used += 1

		if constraint == "must_link":
			must.setdefault(anchor_key, set()).add(tracklet_id)
			# If this is a tag ping with time evidence, surface it explicitly.
			if anchor_key.startswith("tag:"):
				n_tag_pings_raw += 1
				evidence = ev.get("evidence") if isinstance(ev.get("evidence"), dict) else None
				fi = (evidence.get("frame_index") or evidence.get("first_seen_frame")) if isinstance(evidence, dict) else None
				if fi is None:
					n_tag_pings_missing_frame_index += 1
				else:
					try:
						fi_i = int(fi)
					except Exception:
						fi_i = None
					if fi_i is None:
						n_tag_pings_missing_frame_index += 1
					else:
						key = (tracklet_id, anchor_key, fi_i)
						conf = ev.get("confidence", None)
						try:
							conf_f = float(conf) if conf is not None else None
						except Exception:
							conf_f = None
						prev = tag_pings_best.get(key, None)
						# Keep max confidence when duplicates exist.
						if prev is None or (conf_f is not None and (prev is None or conf_f > prev)):
							tag_pings_best[key] = conf_f

		if constraint == "cannot_link":
			# anchor_key encodes the other tracklet id: "tracklet:<id>"
			if not anchor_key.startswith("tracklet:"):
				continue
			other = anchor_key.split("tracklet:", 1)[1]
			if not other:
				continue
			a, b = sorted([tracklet_id, other])
			if a != b:
				cannot_pairs.add((a, b))

	must_groups = []
	for anchor_key in sorted(must.keys()):
		tids = sorted(must[anchor_key])
		must_groups.append({"anchor_key": anchor_key, "tracklet_ids": tids})

	cannot_list = [[a, b] for (a, b) in sorted(cannot_pairs)]

	# Deterministic, deduped, sorted list of tag pings.
	tag_pings: List[Dict[str, Any]] = []
	for (tid, ak, fi), conf in sorted(tag_pings_best.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
		rec: Dict[str, Any] = {"tracklet_id": tid, "anchor_key": ak, "frame_index": int(fi)}
		if conf is not None:
			rec["confidence"] = float(conf)
		tag_pings.append(rec)

	spec = {
		"must_link_groups": must_groups,
		"cannot_link_pairs": cannot_list,
		"tag_pings": tag_pings,
		"stats": {
			"n_must_link_groups": len(must_groups),
			"n_must_link_tracklets": int(sum(len(g["tracklet_ids"]) for g in must_groups)),
			"n_cannot_link_pairs": len(cannot_list),
			"n_events_read": int(n_events_read),
			"n_events_used": int(n_events_used),
			"n_events_skipped_non_identity_hint": int(n_events_skipped_non_identity_hint),
			"n_events_missing_artifact_type": int(n_events_missing_artifact_type),
			"n_tag_pings_emitted": int(len(tag_pings)),
			"n_tag_pings_missing_frame_index": int(n_tag_pings_missing_frame_index),
			"n_tag_pings_deduped": int(max(0, n_tag_pings_raw - len(tag_pings_best))),
			"schema_versions_seen": sorted(schema_versions_seen),
		},
	}
	return spec
