"""CP17: Cross-camera corroboration evidence for two-pass ILP.

This module builds cross-camera evidence from Pass 1 outputs
and prepares it for injection into D2 constraints before Pass 2 re-solve.

Evidence channels:
  - Tag evidence (Tier 1): Same tag_id on 2+ cameras → hard corroboration.
  - Coordinate evidence (Tier 2): Spatial proximity of person tracks across
    cameras → soft corroboration via world-coordinate agreement.

Separate from cross_camera_merge.py: merge is post-hoc linking;
evidence is pre-solve priors.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# CP17 Tier 2: Coordinate evidence
# ---------------------------------------------------------------------------

def _load_person_tracks(adapter: Any) -> Optional[pd.DataFrame]:
	"""Load person_tracks parquet from D4 output. Returns None on failure."""
	path = adapter.person_tracks_parquet()
	if not path.exists():
		logger.debug("person_tracks not found: {}", path)
		return None
	try:
		df = pd.read_parquet(path, columns=["person_id", "frame_index", "x_m", "y_m"])
	except Exception as e:
		logger.warning("Failed to read person_tracks {}: {}", path, e)
		return None
	if df.empty:
		return None
	return df


def _load_person_tag_map(adapter: Any) -> Dict[str, str]:
	"""Build person_id → tag_key mapping from identity_assignments JSONL."""
	ia_path = adapter.identity_assignments_jsonl()
	if not ia_path.exists():
		return {}
	mapping: Dict[str, str] = {}
	text = ia_path.read_text(encoding="utf-8").strip()
	if not text:
		return {}
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
		if tag_id is not None and person_id is not None:
			mapping[str(person_id)] = f"tag:{tag_id}"
	return mapping


def _compute_pairwise_proximity(
	df_a: pd.DataFrame,
	df_b: pd.DataFrame,
	person_a: str,
	person_b: str,
	*,
	window_frames: int,
	tolerance_frames: int,
	proximity_threshold_m: float,
) -> Optional[Dict[str, Any]]:
	"""Compute spatial agreement between two person tracks from different cameras.

	Returns metrics dict if enough overlapping data exists, else None.
	"""
	pa = df_a[df_a["person_id"] == person_a].sort_values("frame_index")
	pb = df_b[df_b["person_id"] == person_b].sort_values("frame_index")

	if pa.empty or pb.empty:
		return None

	# Frame range overlap (with tolerance)
	range_start = max(pa["frame_index"].min(), pb["frame_index"].min()) - tolerance_frames
	range_end = min(pa["frame_index"].max(), pb["frame_index"].max()) + tolerance_frames
	if range_start >= range_end:
		return None

	# Filter to overlapping range
	pa = pa[(pa["frame_index"] >= range_start) & (pa["frame_index"] <= range_end)]
	pb = pb[(pb["frame_index"] >= range_start) & (pb["frame_index"] <= range_end)]
	if pa.empty or pb.empty:
		return None

	# Index by frame for fast lookup
	pa_indexed = pa.set_index("frame_index")[["x_m", "y_m"]]
	pb_indexed = pb.set_index("frame_index")[["x_m", "y_m"]]

	# Find matched frames (within tolerance) via merge_asof
	pa_sorted = pa_indexed.reset_index().sort_values("frame_index")
	pb_sorted = pb_indexed.reset_index().sort_values("frame_index")
	matched = pd.merge_asof(
		pa_sorted, pb_sorted,
		on="frame_index",
		tolerance=tolerance_frames,
		direction="nearest",
		suffixes=("_a", "_b"),
	)
	matched = matched.dropna(subset=["x_m_a", "y_m_a", "x_m_b", "y_m_b"])
	if len(matched) < 3:
		return None

	# Compute per-frame Euclidean distance
	matched["dist_m"] = np.sqrt(
		(matched["x_m_a"] - matched["x_m_b"]) ** 2
		+ (matched["y_m_a"] - matched["y_m_b"]) ** 2
	)

	# Rolling window proximity analysis
	n_windows = 0
	n_proximate = 0
	all_distances: List[float] = []

	frame_min = int(matched["frame_index"].min())
	frame_max = int(matched["frame_index"].max())
	window_start = frame_min

	while window_start <= frame_max:
		window_end = window_start + window_frames
		window_data = matched[
			(matched["frame_index"] >= window_start)
			& (matched["frame_index"] < window_end)
		]
		if len(window_data) >= 1:
			median_dist = float(window_data["dist_m"].median())
			all_distances.append(median_dist)
			n_windows += 1
			if median_dist < proximity_threshold_m:
				n_proximate += 1
		window_start += window_frames // 2  # 50% overlap for smoother estimate

	if n_windows == 0:
		return None

	spatial_agreement_ratio = n_proximate / n_windows
	mean_distance_m = float(np.mean(all_distances))

	return {
		"spatial_agreement_ratio": round(spatial_agreement_ratio, 4),
		"n_overlapping_windows": n_windows,
		"n_proximate_windows": n_proximate,
		"mean_distance_m": round(mean_distance_m, 4),
		"n_matched_frames": len(matched),
	}


def build_cross_camera_coordinate_evidence(
	*,
	cam_ids: List[str],
	adapter_map: Dict[str, Any],
	config: Dict[str, Any],
	fps: float,
) -> Dict[str, Any]:
	"""Build coordinate-based cross-camera corroboration from D4 person tracks.

	Compares world-coordinate positions of persons across camera pairs.
	When two persons on different cameras consistently occupy the same
	physical location, this provides evidence they are the same athlete.

	Args:
		cam_ids: Camera IDs in the session.
		adapter_map: cam_id → SessionStageLayoutAdapter with person_tracks_parquet().
		config: Full pipeline config dict.
		fps: Authoritative frames-per-second from session manifest.

	Returns:
		Dict with coordinate_corroborated_tags, coordinate_conflicts, and metadata.
	"""
	cc_cfg = config.get("cross_camera", {}).get("coordinate_evidence", {})
	temporal_window_s = float(cc_cfg.get("temporal_window_s", 2.5))
	temporal_tolerance_s = float(cc_cfg.get("temporal_tolerance_s", 2.0))
	proximity_threshold_m = float(cc_cfg.get("proximity_threshold_m", 0.5))
	agreement_ratio_threshold = float(cc_cfg.get("agreement_ratio_threshold", 0.6))

	window_frames = max(1, int(temporal_window_s * fps))
	tolerance_frames = max(1, int(temporal_tolerance_s * fps))

	# Load person tracks and tag maps per camera
	tracks: Dict[str, pd.DataFrame] = {}
	tag_maps: Dict[str, Dict[str, str]] = {}  # cam_id -> {person_id -> tag_key}
	for cam_id in cam_ids:
		adapter = adapter_map.get(cam_id)
		if adapter is None:
			continue
		df = _load_person_tracks(adapter)
		if df is not None:
			tracks[cam_id] = df
			logger.debug(
				"Loaded {} person_tracks rows for {} ({} persons)",
				len(df), cam_id, df["person_id"].nunique(),
			)
		tag_maps[cam_id] = _load_person_tag_map(adapter)

	if len(tracks) < 2:
		logger.info("Coordinate evidence: <2 cameras with person_tracks, skipping")
		return {
			"coordinate_corroborated_tags": {},
			"coordinate_conflicts": [],
			"n_coordinate_corroborated_tags": 0,
			"n_coordinate_conflicts": 0,
			"evidence_source": "pass1_person_tracks_coordinate",
		}

	# Pairwise camera comparison
	corroborated_tags: Dict[str, Dict[str, Any]] = {}
	conflicts: List[Dict[str, Any]] = []

	for cam_a, cam_b in combinations(sorted(tracks.keys()), 2):
		df_a = tracks[cam_a]
		df_b = tracks[cam_b]
		persons_a = df_a["person_id"].unique()
		persons_b = df_b["person_id"].unique()

		logger.debug(
			"Coordinate evidence: comparing {} ({} persons) vs {} ({} persons)",
			cam_a, len(persons_a), cam_b, len(persons_b),
		)

		for pa in persons_a:
			for pb in persons_b:
				metrics = _compute_pairwise_proximity(
					df_a, df_b, pa, pb,
					window_frames=window_frames,
					tolerance_frames=tolerance_frames,
					proximity_threshold_m=proximity_threshold_m,
				)
				if metrics is None:
					continue
				if metrics["spatial_agreement_ratio"] < agreement_ratio_threshold:
					continue

				# This pair is coordinate-corroborated
				tag_a = tag_maps.get(cam_a, {}).get(pa)
				tag_b = tag_maps.get(cam_b, {}).get(pb)

				logger.info(
					"Coordinate corroboration: {}:{} ~ {}:{} "
					"(agreement={:.2f}, dist={:.3f}m, tag_a={}, tag_b={})",
					cam_a, pa, cam_b, pb,
					metrics["spatial_agreement_ratio"],
					metrics["mean_distance_m"],
					tag_a, tag_b,
				)

				# Conflict detection: both have tags but they differ
				if tag_a is not None and tag_b is not None and tag_a != tag_b:
					conflicts.append({
						"camera_a": cam_a,
						"person_a": pa,
						"tag_a": tag_a,
						"camera_b": cam_b,
						"person_b": pb,
						"tag_b": tag_b,
						"spatial_agreement_ratio": metrics["spatial_agreement_ratio"],
						"mean_distance_m": metrics["mean_distance_m"],
						"conflict_type": "tag_mismatch_with_spatial_agreement",
					})

				# Tag linkage: propagate tag from one side to the other
				for tag_key in (tag_a, tag_b):
					if tag_key is None:
						continue
					coord_entry = corroborated_tags.get(tag_key)
					if coord_entry is None:
						corroborated_tags[tag_key] = {
							"coordinate_evidence": {
								"camera_pairs": [(cam_a, cam_b)],
								"spatial_agreement_ratio": metrics["spatial_agreement_ratio"],
								"n_overlapping_windows": metrics["n_overlapping_windows"],
								"mean_distance_m": metrics["mean_distance_m"],
								"linked_persons": {cam_a: pa, cam_b: pb},
							}
						}
					else:
						# Append camera pair if new
						existing_pairs = coord_entry["coordinate_evidence"]["camera_pairs"]
						pair = (cam_a, cam_b)
						if pair not in existing_pairs:
							existing_pairs.append(pair)
						# Keep best (highest agreement) metrics
						if metrics["spatial_agreement_ratio"] > coord_entry["coordinate_evidence"]["spatial_agreement_ratio"]:
							coord_entry["coordinate_evidence"]["spatial_agreement_ratio"] = metrics["spatial_agreement_ratio"]
							coord_entry["coordinate_evidence"]["mean_distance_m"] = metrics["mean_distance_m"]
							coord_entry["coordinate_evidence"]["n_overlapping_windows"] = metrics["n_overlapping_windows"]

	logger.info(
		"Coordinate evidence complete: {} corroborated tags, {} conflicts",
		len(corroborated_tags), len(conflicts),
	)

	return {
		"coordinate_corroborated_tags": corroborated_tags,
		"coordinate_conflicts": conflicts,
		"n_coordinate_corroborated_tags": len(corroborated_tags),
		"n_coordinate_conflicts": len(conflicts),
		"evidence_source": "pass1_person_tracks_coordinate",
	}
