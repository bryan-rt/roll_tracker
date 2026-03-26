"""CP17: Cross-camera tag corroboration evidence for two-pass ILP.

This module builds cross-camera evidence from Pass 1 identity assignments
and prepares it for injection into D2 constraints before Pass 2 re-solve.

Separate from cross_camera_merge.py: merge is post-hoc linking;
evidence is pre-solve priors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger


def build_cross_camera_tag_evidence(
	*,
	cam_ids: List[str],
	adapter_map: Dict[str, Any],
) -> Dict[str, Any]:
	"""Analyze Pass 1 identity_assignments across cameras to find corroborated tags.

	A tag is "corroborated" if it appears in identity_assignments for 2+ cameras.

	Returns:
		{
			"corroborated_tags": {
				"tag:1": {
					"observed_on_cameras": ["FP7oJQ", "J_EDEw"],
					"n_cameras": 2,
					"per_camera_person_ids": {"FP7oJQ": [...], "J_EDEw": [...]},
					"per_camera_observation_count": {"FP7oJQ": 4, "J_EDEw": 7}
				}
			},
			"evidence_source": "pass1_identity_assignments",
			"n_corroborated_tags": 1,
			"n_total_tags_observed": 1
		}
	"""
	# Collect all (cam_id, tag_id, person_id) tuples from identity assignments
	tag_index: Dict[str, Dict[str, List[str]]] = {}  # tag_key -> {cam_id -> [person_ids]}

	for cam_id in cam_ids:
		adapter = adapter_map.get(cam_id)
		if adapter is None:
			continue
		ia_path = adapter.identity_assignments_jsonl()
		if not ia_path.exists():
			continue
		text = ia_path.read_text(encoding="utf-8").strip()
		if not text:
			continue
		for line in text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
			except Exception:
				continue
			tag_id = rec.get("tag_id")
			person_id = rec.get("person_id")
			if tag_id is None or person_id is None:
				continue
			tag_key = f"tag:{tag_id}"
			if tag_key not in tag_index:
				tag_index[tag_key] = {}
			if cam_id not in tag_index[tag_key]:
				tag_index[tag_key][cam_id] = []
			tag_index[tag_key][cam_id].append(str(person_id))

	# Build corroborated tags (seen on 2+ cameras)
	corroborated: Dict[str, Dict[str, Any]] = {}
	for tag_key, cam_map in sorted(tag_index.items()):
		if len(cam_map) < 2:
			continue
		corroborated[tag_key] = {
			"observed_on_cameras": sorted(cam_map.keys()),
			"n_cameras": len(cam_map),
			"per_camera_person_ids": {
				cam: sorted(set(pids)) for cam, pids in sorted(cam_map.items())
			},
			"per_camera_observation_count": {
				cam: len(pids) for cam, pids in sorted(cam_map.items())
			},
		}

	return {
		"corroborated_tags": corroborated,
		"evidence_source": "pass1_identity_assignments",
		"n_corroborated_tags": len(corroborated),
		"n_total_tags_observed": len(tag_index),
	}
