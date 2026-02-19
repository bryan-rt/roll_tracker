from __future__ import annotations
from bjj_pipeline.contracts.f0_parquet import validate_df_schema_by_key
"""Stage D1 4 Candidate graph construction.

D1 constructs a solver-agnostic graph over tracklets, including explicit
GROUP_TRACKLET nodes (capacity=2) and merge/split edges to represent
22122 events.

Authoritative spatial source: Stage D0 bank frames (repaired coords with
per-frame fallback to raw coords).
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bjj_pipeline.stages.stitch.graph import (
	EdgeType,
	GraphEdge,
	GraphNode,
	NodeType,
	TrackletGraph,
)


def _now_ms() -> int:
	return int(time.time() * 1000)


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur


def _write_audit_event(audit_path: Path, event: Dict[str, Any]) -> None:
	audit_path.parent.mkdir(parents=True, exist_ok=True)
	with audit_path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(event, sort_keys=True) + "\n")


def _is_finite(x: Any) -> bool:
	try:
		return x is not None and math.isfinite(float(x))
	except Exception:
		return False


def _effective_xy_row(row: pd.Series) -> Optional[Tuple[float, float, bool]]:
	"""Return (x_eff, y_eff, used_raw_fallback) or None if invalid."""
	x_rep = row.get("x_m_repaired", None)
	y_rep = row.get("y_m_repaired", None)
	x_raw = row.get("x_m", None)
	y_raw = row.get("y_m", None)
	use_raw = (not _is_finite(x_rep)) or (not _is_finite(y_rep))
	if use_raw:
		x = x_raw
		y = y_raw
	else:
		x = x_rep
		y = y_rep
	if not (_is_finite(x) and _is_finite(y)):
		return None
	return (float(x), float(y), bool(use_raw))


def _dist_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _parse_json_list(s: Any) -> List[str]:
	if s is None:
		return []
	if isinstance(s, list):
		return [str(x) for x in s]
	if not isinstance(s, str):
		return []
	ss = s.strip()
	if not ss:
		return []
	try:
		obj = json.loads(ss)
		if isinstance(obj, list):
			return [str(x) for x in obj]
	except Exception:
		return []
	return []


def _get_manifest_fields(manifest: Any) -> Tuple[Optional[float], Optional[int], Optional[int]]:
	"""Extract fps, frame_count, duration_ms from manifest (dict or ClipManifest)."""
	fps = None
	frame_count = None
	duration_ms = None
	for key in ("fps", "frame_count", "duration_ms"):
		if hasattr(manifest, key):
			val = getattr(manifest, key)
		elif isinstance(manifest, dict):
			val = manifest.get(key)
		else:
			val = None
		if key == "fps":
			fps = float(val) if val is not None else None
		elif key == "frame_count":
			frame_count = int(val) if val is not None else None
		elif key == "duration_ms":
			duration_ms = int(val) if val is not None else None
	return fps, frame_count, duration_ms


def _get_manifest_video_path(manifest: Any) -> Optional[str]:
	"""Extract input_video_path from manifest (dict or ClipManifest)."""
	if hasattr(manifest, "input_video_path"):
		val = getattr(manifest, "input_video_path")
	elif isinstance(manifest, dict):
		val = manifest.get("input_video_path")
	else:
		val = None
	if val is None:
		return None
	try:
		return str(val)
	except Exception:
		return None


def _probe_video_wh(video_path: str) -> Optional[Tuple[int, int]]:
	"""Probe (width,height) from a video file using OpenCV. Returns None on failure."""
	try:
		import cv2  # type: ignore
	except Exception:
		return None

	cap = None
	try:
		cap = cv2.VideoCapture(str(video_path))
		if not cap or (hasattr(cap, "isOpened") and not cap.isOpened()):
			return None

		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		if w <= 0 or h <= 0:
			ok, frame = cap.read()
			if ok and frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
				h, w = int(frame.shape[0]), int(frame.shape[1])

		if w <= 0 or h <= 0:
			return None
		return (w, h)
	except Exception:
		return None
	finally:
		if cap is not None:
			try:
				cap.release()
			except Exception:
				pass


def run_d1(*, cfg: Dict[str, Any], layout: Any, manifest: Any) -> TrackletGraph:
	"""Build the candidate graph for D1.

	Reads D0 bank artifacts from `layout`:
	  - stage_D/tracklet_bank_frames.parquet
	  - stage_D/tracklet_bank_summaries.parquet

	Writes audit events into stage_D/audit.jsonl.
	"""
	# ---- config ----
	d1_cfg = _cfg_get(cfg, "stages.stage_D.d1", {}) or {}
	write_debug_graph_artifacts = bool(d1_cfg.get("write_debug_graph_artifacts", False))
	enable_lifespan_segmentation = bool(d1_cfg.get("enable_lifespan_segmentation", True))
	min_group_duration_frames = int(d1_cfg.get("min_group_duration_frames", 10))
	min_split_separation_frames = int(d1_cfg.get("min_split_separation_frames", 10))
	carrier_coord_window_frames = int(d1_cfg.get("carrier_coord_window_frames", 8))
	merge_trigger_max_age_frames = int(d1_cfg.get("merge_trigger_max_age_frames", 60))
	enable_group_nodes = bool(d1_cfg.get("enable_group_nodes", True))
	max_continue_gap_frames = int(d1_cfg.get("max_continue_gap_frames", 90))
	start_window_frames = int(d1_cfg.get("start_window_frames", 10))
	end_window_frames = int(d1_cfg.get("end_window_frames", 10))
	merge_dist_m = float(d1_cfg.get("merge_dist_m", 0.45))
	merge_end_sync_frames = int(d1_cfg.get("merge_end_sync_frames", 3))
	merge_disappear_gap_frames = int(d1_cfg.get("merge_disappear_gap_frames", 6))
	split_dist_m = float(d1_cfg.get("split_dist_m", 0.60))
	split_search_horizon_frames = int(d1_cfg.get("split_search_horizon_frames", 120))
	suppress_start_merged_if_entrance_like = bool(d1_cfg.get("suppress_start_merged_if_entrance_like", True))
	reconnect_enabled = bool(d1_cfg.get("reconnect_enabled", False))
	reconnect_max_gap_frames = int(d1_cfg.get("reconnect_max_gap_frames", 250))
	reconnect_boundary_on_mat_required = bool(d1_cfg.get("reconnect_boundary_on_mat_required", True))
	reconnect_boundary_slack_frames = int(d1_cfg.get("reconnect_boundary_slack_frames", 2))
	reconnect_solo_only = bool(d1_cfg.get("reconnect_solo_only", True))
	reconnect_allow_group_source = bool(d1_cfg.get("reconnect_allow_group_source", True))

	# Promotion rule (group-capacity continuation through occlusion)
	# If a CONT_RECONNECT candidate comes from a GROUP node into a SOLO node and there is no
	# strong evidence of a second nearby SOLO birth, promote the destination node to GROUP.
	promote_group_reconnect_enabled = bool(d1_cfg.get("promote_group_reconnect_enabled", False))
	promote_group_reconnect_nearby_start_window_frames = int(
		d1_cfg.get("promote_group_reconnect_nearby_start_window_frames", 30)
	)
	promote_group_reconnect_nearby_dist_m = float(
		d1_cfg.get("promote_group_reconnect_nearby_dist_m", merge_dist_m)
	)

	# Hard gate: if a new tracklet is born at the image border (based on u_px/v_px),
	# do not treat it as a split-out of an existing carrier. This prevents false
	# start_merged GROUP spans from entrance coincidences (Case 2).
	split_border_gate_enabled = bool(d1_cfg.get("split_border_gate_enabled", True))
	split_border_margin_px = int(d1_cfg.get("split_border_margin_px", 40))

	d0_kin_cfg = _cfg_get(cfg, "stages.stage_D.d0.kinematics", {}) or {}
	v_max_mps = float(d0_kin_cfg.get("v_max_mps", 8.0))

	fps, frame_count, duration_ms = _get_manifest_fields(manifest)
	last_frame = (frame_count - 1) if frame_count is not None else None

	# ---- paths ----
	frames_path = Path(layout.tracklet_bank_frames_parquet())
	summ_path = Path(layout.tracklet_bank_summaries_parquet())
	audit_path = Path(layout.audit_jsonl("D"))
	# Dev-only debug artifacts live under outputs/<clip_id>/_debug/.
	# Prefer layout.clip_root when available; otherwise infer from stage_D parquet path.
	clip_root = Path(getattr(layout, "clip_root", frames_path.parent.parent))
	debug_dir = clip_root / "_debug"

	# ---- load ----
	tf = pd.read_parquet(frames_path)
	ts = pd.read_parquet(summ_path)

	# required columns
	for col in ("tracklet_id", "frame_index"):
		if col not in tf.columns:
			raise ValueError(f"D1 requires {col} in bank frames: {frames_path}")
	for col in ("x_m_repaired", "y_m_repaired", "x_m", "y_m"):
		if col not in tf.columns:
			raise ValueError(f"D1 requires {col} in bank frames: {frames_path}")
	for col in ("tracklet_id", "start_frame", "end_frame"):
		if col not in ts.columns:
			raise ValueError(f"D1 requires {col} in bank summaries: {summ_path}")

	on_mat_missing = "on_mat" not in tf.columns
	if on_mat_missing:
		tf = tf.copy()
		tf["on_mat"] = True

	# normalize types
	tf = tf.copy()
	tf["tracklet_id"] = tf["tracklet_id"].astype(str)
	tf["frame_index"] = tf["frame_index"].astype(int)
	ts = ts.copy()
	ts["tracklet_id"] = ts["tracklet_id"].astype(str)
	ts["start_frame"] = ts["start_frame"].astype(int)
	ts["end_frame"] = ts["end_frame"].astype(int)

	# Identity evidence (AprilTag pings) is time-local and is NOT consumed at D1
	# in this checkpoint. D1 only constructs the segmentation graph. D2/D3 are
	# responsible for binding and enforcing identity evidence.

	# pre-index summaries for fast lookup
	ts_by_tid = {r["tracklet_id"]: r for _, r in ts.iterrows()}

	# Precompute observed pixel-space bounds for border gating.
	# We do not have video frame dimensions in the D0 bank artifacts, so we use
	# the observed u_px/v_px extent across the clip as a deterministic proxy.
	# This is sufficient for the intended entrance suppression: entrants born at
	# the extreme observed left/right/top/bottom are treated as border-born.
	border_uv_bounds = None  # (u_min, u_max, v_min, v_max)
	start_uv_by_tid: Dict[str, Tuple[float, float]] = {}
	border_gate_mode: str = "disabled_missing_uv"
	frame_wh: Optional[Tuple[int, int]] = None
	entrance_uv_series_by_tid: Dict[str, List[Tuple[float, float]]] = {}
	if split_border_gate_enabled:
		# If pixel coordinates are unavailable (e.g., unit tests with minimal fixtures),
		# silently disable the border gate rather than failing hard.
		if "u_px" not in tf.columns or "v_px" not in tf.columns:
			split_border_gate_enabled = False
		else:
			uv = tf[["u_px", "v_px"]].dropna()
			if len(uv) == 0:
				# No usable coordinates; disable gate.
				split_border_gate_enabled = False
				border_gate_mode = "disabled_empty_uv"
			else:
				u_min = float(uv["u_px"].min())
				u_max = float(uv["u_px"].max())
				v_min = float(uv["v_px"].min())
				v_max = float(uv["v_px"].max())
				border_uv_bounds = (u_min, u_max, v_min, v_max)

				# Map each tracklet_id -> (u_px, v_px) at its start_frame when available.
				start_frames = ts[["tracklet_id", "start_frame"]].copy()
				start_frames["tracklet_id"] = start_frames["tracklet_id"].astype(str)
				start_frames["start_frame"] = start_frames["start_frame"].astype(int)
				tf_uv0 = tf[["tracklet_id", "frame_index", "u_px", "v_px"]].copy()
				tf_uv0["tracklet_id"] = tf_uv0["tracklet_id"].astype(str)
				tf_uv0["frame_index"] = tf_uv0["frame_index"].astype(int)
				merged = tf_uv0.merge(
					start_frames,
					left_on=["tracklet_id", "frame_index"],
					right_on=["tracklet_id", "start_frame"],
					how="inner",
				)
				if len(merged) > 0:
					merged = merged.sort_values(["tracklet_id"], kind="mergesort").drop_duplicates("tracklet_id")
					for _, r in merged.iterrows():
						tid = str(r["tracklet_id"])
						upx = r.get("u_px", None)
						vpx = r.get("v_px", None)
						if upx is None or vpx is None or (pd.isna(upx) or pd.isna(vpx)):
							continue
						start_uv_by_tid[tid] = (float(upx), float(vpx))

				# Enriched entrance diagnostics: true video bounds when available + early-series samples.
				video_path = _get_manifest_video_path(manifest)
				frame_wh = _probe_video_wh(video_path) if video_path else None
				if frame_wh is not None:
					border_gate_mode = "video_bounds"
				else:
					border_gate_mode = "observed_bounds_fallback"

				# Build per-tid early (u_px, v_px) samples for entrance classification.
				k_frames = int(d1_cfg.get("split_entrance_k_frames", 10))
				k_frames = max(1, k_frames)
				tf_uv = tf[["tracklet_id", "frame_index", "u_px", "v_px"]].copy()
				tf_uv["tracklet_id"] = tf_uv["tracklet_id"].astype(str)
				tf_uv["frame_index"] = tf_uv["frame_index"].astype(int)
				tf_uv = tf_uv.sort_values(["tracklet_id", "frame_index"], kind="mergesort")
				for tid, row in ts_by_tid.items():
					sf = int(row["start_frame"])
					gdf = tf_uv[tf_uv["tracklet_id"] == tid]
					if len(gdf) == 0:
						continue
					window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] < (sf + k_frames))]
					if len(window) == 0:
						continue
					samples: List[Tuple[float, float]] = []
					for _, r in window.iterrows():
						upx = r.get("u_px", None)
						vpx = r.get("v_px", None)
						if upx is None or vpx is None or (pd.isna(upx) or pd.isna(vpx)):
							continue
						samples.append((float(upx), float(vpx)))
						if len(samples) >= k_frames:
							break
					if samples:
						entrance_uv_series_by_tid[tid] = samples

	def _entrance_border_distance_px(u: float, v: float) -> Optional[float]:
		"""Distance (px) to nearest image border (true bounds preferred)."""
		if frame_wh is not None:
			w, h = frame_wh
			u_c = min(max(u, 0.0), float(max(0, w - 1)))
			v_c = min(max(v, 0.0), float(max(0, h - 1)))
			return float(min(u_c, float(w - 1) - u_c, v_c, float(h - 1) - v_c))
		if bool(d1_cfg.get("split_entrance_allow_observed_fallback", True)) and border_uv_bounds is not None:
			u_min, u_max, v_min, v_max = border_uv_bounds
			return float(min(u - u_min, u_max - u, v - v_min, v_max - v))
		return None

	def _entrance_diagnostics(tid: str) -> Optional[Dict[str, Any]]:
		if not split_border_gate_enabled:
			return None
		samples = entrance_uv_series_by_tid.get(str(tid))
		if not samples:
			return None
		ds: List[float] = []
		for (u, v) in samples:
			d = _entrance_border_distance_px(u, v)
			if d is None or not math.isfinite(float(d)):
				continue
			ds.append(float(d))
		if not ds:
			return None
		return {
			"n_samples": int(len(ds)),
			"d0_px": float(ds[0]),
			"d_last_px": float(ds[-1]),
			"mode": str(border_gate_mode),
			"frame_wh": tuple(frame_wh) if frame_wh is not None else None,
			"margin_px": int(split_border_margin_px),
			"min_inward_px": int(d1_cfg.get("split_entrance_min_inward_px", 20)),
			"min_samples": int(d1_cfg.get("split_entrance_min_samples", 3)),
			"k_frames": int(d1_cfg.get("split_entrance_k_frames", 10)),
		}

	def _is_entrance_like_tid(tid: str) -> bool:
		"""True if tid appears to enter from off-screen: starts near border and moves inward."""
		if not split_border_gate_enabled:
			return False
		diag = _entrance_diagnostics(str(tid))
		if diag is None:
			return False
		min_samples = int(diag.get("min_samples", 3))
		if int(diag.get("n_samples", 0)) < max(1, min_samples):
			return False
		d0 = float(diag.get("d0_px", 1e9))
		d_last = float(diag.get("d_last_px", 1e9))

		margin_px = float(diag.get("margin_px", split_border_margin_px))
		min_inward = float(diag.get("min_inward_px", 20))

		gate_logic = str(diag.get("gate_logic", d1_cfg.get("split_entrance_gate_logic", "or"))).lower().strip()
		inward_margin_px = float(
			diag.get("inward_margin_px", d1_cfg.get("split_entrance_inward_margin_px", int(margin_px + 30)))
		)

		border_spawn = bool(d0 <= margin_px)
		moved_inward = bool((d0 <= inward_margin_px) and ((d_last - d0) >= min_inward))

		if gate_logic == "and":
			return bool(border_spawn and moved_inward)
		# default: "or"
		return bool(border_spawn or moved_inward)

	# frames grouped
	tf = tf.sort_values(["tracklet_id", "frame_index"], kind="mergesort")
	frames_by_tid: Dict[str, pd.DataFrame] = {tid: g for tid, g in tf.groupby("tracklet_id", sort=False)}

	# ---- endpoint helpers ----
	used_raw_frames = 0
	used_repaired_frames = 0
	used_raw_start = 0
	used_raw_end = 0
	missing_endpoint_coords = 0

	def endpoint_start(tid: str) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames, used_raw_start, missing_endpoint_coords
		row = ts_by_tid.get(tid)
		if row is None:
			missing_endpoint_coords += 1
			return None
		sf = int(row["start_frame"])
		ef = sf + start_window_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			missing_endpoint_coords += 1
			return None
		window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] <= ef)]
		for _, fr in window.iterrows():
			if not bool(fr.get("on_mat", True)):
				continue
			res = _effective_xy_row(fr)
			if res is None:
				continue
			x, y, used_raw = res
			if used_raw:
				used_raw_frames += 1
				used_raw_start += 1
			else:
				used_repaired_frames += 1
			return ((x, y), used_raw)
		missing_endpoint_coords += 1
		return None

	def endpoint_end(tid: str) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames, used_raw_end, missing_endpoint_coords
		row = ts_by_tid.get(tid)
		if row is None:
			missing_endpoint_coords += 1
			return None
		ef = int(row["end_frame"])
		sf = ef - end_window_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			missing_endpoint_coords += 1
			return None
		window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] <= ef)]
		for _, fr in window.iloc[::-1].iterrows():
			if not bool(fr.get("on_mat", True)):
				continue
			res = _effective_xy_row(fr)
			if res is None:
				continue
			x, y, used_raw = res
			if used_raw:
				used_raw_frames += 1
				used_raw_end += 1
			else:
				used_repaired_frames += 1
			return ((x, y), used_raw)
		missing_endpoint_coords += 1
		return None

	def _boundary_on_mat_ok(tid: str, which: str) -> bool:
		"""Return True if boundary window contains at least one on-mat frame.

		Missing on_mat is treated as off-mat for this check (teleportation guardrail).
		"""
		row = ts_by_tid.get(tid)
		if row is None:
			return False
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			return False
		try:
			if which == "end":
				frame = int(row["end_frame"])
				sf = frame - reconnect_boundary_slack_frames
				ef = frame
			else:
				frame = int(row["start_frame"])
				sf = frame
				ef = frame + reconnect_boundary_slack_frames
		except Exception:
			return False
		window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] <= ef)]
		for _, fr in window.iterrows():
			# IMPORTANT: missing on_mat => treat as NOT on-mat.
			val = fr.get("on_mat", None)
			if val is True:
				return True
		return False

	def carrier_pos_at_frame(tid: str, frame_idx: int) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			return None
		row = gdf[gdf["frame_index"] == frame_idx]
		if len(row) == 0:
			return None
		fr = row.iloc[0]
		if not bool(fr.get("on_mat", True)):
			return None
		res = _effective_xy_row(fr)
		if res is None:
			return None
		x, y, used_raw = res
		if used_raw:
			used_raw_frames += 1
		else:
			used_repaired_frames += 1
		return ((x, y), used_raw)

	def _cannot_link_for(tid: str) -> List[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return []
		return _parse_json_list(row.get("cannot_link_anchor_keys_json", None))

	def _must_link_key(tid: str) -> Optional[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return None
		val = row.get("must_link_anchor_key", None)
		return str(val) if val not in (None, "", "null") else None

	# =====================================================================
	# NEW PATH: full-lifespan segmentation (SOLO/GROUP segments)
	#
	# This path does NOT remove your existing endpoint-based logic. Instead,
	# it runs first when enabled, returning early. If disabled, the original
	# D1 behavior is preserved below.
	# =====================================================================
	if enable_group_nodes and enable_lifespan_segmentation and last_frame is not None:
		# Precompute endpoints for disappear/new using existing helpers.
		endpoints_end: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
		endpoints_start: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
		for tid in sorted(ts_by_tid.keys()):
			endpoints_end[tid] = endpoint_end(tid)
			endpoints_start[tid] = endpoint_start(tid)

		def _carrier_pos_near_frame(tid: str, target_frame: int, window: int) -> Optional[Tuple[Tuple[float, float], bool]]:
			# Deterministic nearest-frame lookup (exact -> +/-1 -> +/-2 ...).
			for off in range(0, window + 1):
				if off == 0:
					res = carrier_pos_at_frame(tid, target_frame)
					if res is not None:
						return res
					continue
				for f in (target_frame - off, target_frame + off):
					res = carrier_pos_at_frame(tid, int(f))
					if res is not None:
						return res
			return None

		# ---- merge triggers: disappear tid ends while carrier continues, close in space ----
		merge_triggers: List[Dict[str, Any]] = []
		for disappear, drow in ts_by_tid.items():
			d_end = int(drow["end_frame"])
			disp_ep = endpoints_end.get(disappear)
			if disp_ep is None:
				continue
			disp_xy = disp_ep[0]
			for carrier, crow in ts_by_tid.items():
				if carrier == disappear:
					continue
				cs = int(crow["start_frame"])
				ce = int(crow["end_frame"])
				if not (cs <= d_end <= ce):
					continue
				cpos = _carrier_pos_near_frame(carrier, d_end, carrier_coord_window_frames)
				if cpos is None:
					continue
				dist = _dist_m(disp_xy, cpos[0])
				if dist > merge_dist_m:
					continue
				merge_triggers.append(
					{
						"carrier": carrier,
						"disappear": disappear,
						"merge_frame": d_end + 1,  # group begins immediately after disappearance
						"merge_end": d_end,
						"merge_dist_m": float(dist),
					}
				)

		# Optional timing suppression (do not infer “ancient” merges).
		if merge_trigger_max_age_frames > 0:
			merge_triggers = [
				m
				for m in merge_triggers
				if (int(m["merge_frame"]) - int(ts_by_tid[m["disappear"]]["end_frame"])) <= merge_trigger_max_age_frames
			]

		# ---- split triggers: new tid starts while carrier exists, close in space ----
		split_triggers_raw: List[Dict[str, Any]] = []
		suppressed_split_triggers_entrance: List[Dict[str, Any]] = []
		for new_tid, nrow in ts_by_tid.items():
			n_start = int(nrow["start_frame"])
			new_ep = endpoints_start.get(new_tid)
			if new_ep is None:
				continue
			new_xy = new_ep[0]
			# Entrance-like gate: tracklets that appear to ENTER the image from off-screen
			# should not induce split triggers (prevents false group spans from entrance coincidences).
			if _is_entrance_like_tid(str(new_tid)):
				diag = _entrance_diagnostics(str(new_tid)) or {}
				suppressed_split_triggers_entrance.append(
					{
						"new": str(new_tid),
						"split_frame": int(n_start),
						"reason": "suppressed_entrance_like",
						"entrance": diag,
					}
				)
				continue
			for carrier, crow in ts_by_tid.items():
				if carrier == new_tid:
					continue
				cs = int(crow["start_frame"])
				ce = int(crow["end_frame"])
				if not (cs <= n_start <= ce):
					continue
				cpos = _carrier_pos_near_frame(carrier, n_start, carrier_coord_window_frames)
				if cpos is None:
					continue
				dist = _dist_m(new_xy, cpos[0])
				if dist > split_dist_m:
					continue
				split_triggers_raw.append(
					{
						"carrier": carrier,
						"new": new_tid,
						"split_frame": n_start,
						"split_dist_m": float(dist),
					}
				)

		# Suppress split triggers too close in time on the same carrier.
		suppressed_split_triggers: List[Dict[str, Any]] = []
		split_triggers: List[Dict[str, Any]] = []
		for carrier in sorted({t["carrier"] for t in split_triggers_raw}):
			cands = sorted(
				[t for t in split_triggers_raw if t["carrier"] == carrier],
				key=lambda x: (int(x["split_frame"]), float(x["split_dist_m"]), str(x["new"])),
			)
			kept: List[Dict[str, Any]] = []
			for t in cands:
				if not kept:
					kept.append(t)
					continue
				prev = kept[-1]
				if int(t["split_frame"]) - int(prev["split_frame"]) < min_split_separation_frames:
					# keep the better (lower dist), deterministic tie-breaker by new id
					if (float(t["split_dist_m"]), str(t["new"])) < (float(prev["split_dist_m"]), str(prev["new"])):
						worse = prev
						kept[-1] = t
					else:
						worse = t
					worse = dict(worse)
					worse["reason"] = "suppressed_close_split"
					suppressed_split_triggers.append(worse)
				else:
					kept.append(t)
			split_triggers.extend(kept)

		# ---- pair merges to splits to form GROUP spans per carrier ----
		group_spans: List[Dict[str, Any]] = []
		suppressed_group_spans: List[Dict[str, Any]] = []
		reconnect_edges_debug: List[Dict[str, Any]] = []
		suppressed_start_merged_entrance: List[Dict[str, Any]] = []

		for carrier in sorted({m["carrier"] for m in merge_triggers} | {s["carrier"] for s in split_triggers}):
			# Carrier lifespan bounds are authoritative (from tracklet_bank_summaries).
			crow = ts_by_tid.get(carrier)
			if crow is None:
				continue
			cs = int(crow["start_frame"])
			ce = int(crow["end_frame"])

			merges = sorted([m for m in merge_triggers if m["carrier"] == carrier], key=lambda x: int(x["merge_frame"]))
			splits = sorted([s for s in split_triggers if s["carrier"] == carrier], key=lambda x: int(x["split_frame"]))

			# clip-start merged: split occurs early and no merge evidence before it
			if splits:
				first_split = splits[0]
				if int(first_split["split_frame"]) <= split_search_horizon_frames:
					has_prior_merge = any(int(m["merge_frame"]) <= int(first_split["split_frame"]) for m in merges)
					if not has_prior_merge:
							suppress_start = False
							new_tid = str(first_split["new"])
							if suppress_start_merged_if_entrance_like and _is_entrance_like_tid(new_tid):
								diag = _entrance_diagnostics(new_tid) or {}
								suppress_start = True
								suppressed_start_merged_entrance.append(
									{
										"carrier": str(carrier),
										"new": new_tid,
										"split_frame": int(first_split["split_frame"]),
										"reason": "suppressed_start_merged_entrance_like_new",
										"entrance": diag,
									}
								)
							if not suppress_start:
								gs_raw = 0
								ge_raw = int(first_split["split_frame"]) - 1
								gs = max(gs_raw, cs)
								ge = min(ge_raw, ce)
								if ge < gs:
									suppressed_group_spans.append(
										{
											"kind": "start_merged",
											"carrier": carrier,
											"disappear": None,
											"new": first_split["new"],
											"group_start_raw": gs_raw,
											"group_end_raw": ge_raw,
											"group_start": gs,
											"group_end": ge,
											"merge_end": None,
											"split_start": int(first_split["split_frame"]),
											"merge_dist_m": None,
											"split_dist_m": float(first_split["split_dist_m"]),
											"reason": "invalid_after_clamp",
										}
									)
								elif (ge - gs + 1) >= min_group_duration_frames:
									group_spans.append(
										{
											"kind": "start_merged",
											"carrier": carrier,
											"disappear": None,
											"new": first_split["new"],
											"group_start_raw": gs_raw,
											"group_end_raw": ge_raw,
											"group_start": gs,
											"group_end": ge,
											"merge_end": None,
											"split_start": int(first_split["split_frame"]),
											"merge_dist_m": None,
											"split_dist_m": float(first_split["split_dist_m"]),
										}
									)
								else:
									suppressed_group_spans.append(
										{
											"kind": "start_merged",
											"carrier": carrier,
											"disappear": None,
											"new": first_split["new"],
											"group_start_raw": gs_raw,
											"group_end_raw": ge_raw,
											"group_start": gs,
											"group_end": ge,
											"merge_end": None,
											"split_start": int(first_split["split_frame"]),
											"merge_dist_m": None,
											"split_dist_m": float(first_split["split_dist_m"]),
											"reason": "too_short",
										}
									)

			used_split_idx: set = set()
			cursor_end: Optional[int] = None
			for m in merges:
				mf = int(m["merge_frame"])
				if cursor_end is not None and mf <= cursor_end:
					suppressed_group_spans.append({**m, "reason": "overlaps_existing_group"})
					continue

				chosen: Optional[Tuple[int, Dict[str, Any]]] = None
				for i, s in enumerate(splits):
					if i in used_split_idx:
						continue
					sf = int(s["split_frame"])
					if sf <= mf:
						continue
					if (sf - mf) > split_search_horizon_frames:
						break
					chosen = (i, s)
					break

				if chosen is None:
					# open-ended merged until the end of the carrier lifespan (never beyond it)
					gs_raw = mf
					ge_raw = int(last_frame)
					gs = max(gs_raw, cs)
					ge = min(ge_raw, ce)
					if ge < gs:
						suppressed_group_spans.append(
							{
								**m,
								"kind": "merge_open_end",
								"group_start_raw": gs_raw,
								"group_end_raw": ge_raw,
								"group_start": gs,
								"group_end": ge,
								"reason": "invalid_after_clamp",
							}
						)
						continue
					if (ge - gs + 1) >= min_group_duration_frames:
						group_spans.append(
							{
								"kind": "merge_open_end",
								"carrier": carrier,
								"disappear": m["disappear"],
								"new": None,
								"group_start_raw": gs_raw,
								"group_end_raw": ge_raw,
								"group_start": gs,
								"group_end": ge,
								"merge_end": int(m["merge_end"]),
								"split_start": None,
								"merge_dist_m": float(m["merge_dist_m"]),
								"split_dist_m": None,
							}
						)
						cursor_end = ge
					else:
						suppressed_group_spans.append({**m, "reason": "too_short"})
					continue

				i, s = chosen
				used_split_idx.add(i)
				gs_raw = mf
				ge_raw = int(s["split_frame"]) - 1
				gs = max(gs_raw, cs)
				ge = min(ge_raw, ce)
				if ge < gs:
					suppressed_group_spans.append(
						{
							**m,
							**s,
							"kind": "merge_split",
							"group_start_raw": gs_raw,
							"group_end_raw": ge_raw,
							"group_start": gs,
							"group_end": ge,
							"reason": "invalid_after_clamp",
						}
					)
					continue
				if (ge - gs + 1) < min_group_duration_frames:
					suppressed_group_spans.append({**m, **s, "reason": "too_short"})
					continue
				group_spans.append(
					{
						"kind": "merge_split",
						"carrier": carrier,
						"disappear": m["disappear"],
						"new": s["new"],
						"group_start_raw": gs_raw,
						"group_end_raw": ge_raw,
						"group_start": gs,
						"group_end": ge,
						"merge_end": int(m["merge_end"]),
						"split_start": int(s["split_frame"]),
						"merge_dist_m": float(m["merge_dist_m"]),
						"split_dist_m": float(s["split_dist_m"]),
					}
				)
				cursor_end = ge

		# ---- segment each base tracklet into SOLO segments + GROUP segments ----
		segments_by_base: Dict[str, List[Dict[str, Any]]] = {}
		all_segments: List[Dict[str, Any]] = []

		for tid, row in sorted(ts_by_tid.items(), key=lambda kv: (int(kv[1]["start_frame"]), str(kv[0]))):
			ts0 = int(row["start_frame"])
			te0 = int(row["end_frame"])
			spans = sorted([g for g in group_spans if g["carrier"] == tid], key=lambda x: (int(x["group_start"]), int(x["group_end"])))

			out: List[Dict[str, Any]] = []
			cursor = ts0
			k = 0
			for gspan in spans:
				gs = int(gspan["group_start"])
				ge = int(gspan["group_end"])
				if cursor < gs:
					out.append(
						{
							"segment_type": "SOLO",
							"base_tracklet_id": tid,
							"start_frame": cursor,
							"end_frame": gs - 1,
							"k": k,
							"payload": {},
						}
					)
					k += 1
				out.append(
					{
						"segment_type": "GROUP",
						"base_tracklet_id": tid,
						"start_frame": gs,
						"end_frame": ge,
						"k": k,
						"payload": dict(gspan),
					}
				)
				k += 1
				cursor = ge + 1
			if cursor <= te0:
				out.append(
					{
						"segment_type": "SOLO",
						"base_tracklet_id": tid,
						"start_frame": cursor,
						"end_frame": te0,
						"k": k,
						"payload": {},
					}
				)

			# Invariant: every emitted segment must lie within the base tracklet lifespan.
			for seg in out:
				sf = int(seg["start_frame"])
				ef = int(seg["end_frame"])
				if ef < sf or sf < ts0 or ef > te0:
					raise ValueError(
						f"D1 segmentation produced out-of-bounds segment for {tid}: {sf}-{ef} not within {ts0}-{te0}"
					)

			# Assign stable node_ids with backwards-compat where possible:
			# - If only one SOLO segment == full lifespan => node_id "T:<tid>".
			# - Otherwise SOLO segments => "T:<tid>:s<k>:<start>-<end>"
			# - GROUP segments => "G:<start>-<end>:carrier=<tid>:d=<...>:n=<...>"
			if len(out) == 1 and out[0]["segment_type"] == "SOLO" and int(out[0]["start_frame"]) == ts0 and int(out[0]["end_frame"]) == te0:
				out[0]["node_id"] = f"T:{tid}"
				out[0]["capacity"] = 1
			else:
				for seg in out:
					if seg["segment_type"] == "SOLO":
						seg["node_id"] = f"T:{tid}:s{int(seg['k'])}:{int(seg['start_frame'])}-{int(seg['end_frame'])}"
						seg["capacity"] = 1
					else:
						gs = int(seg["start_frame"])
						ge = int(seg["end_frame"])
						p = seg["payload"]
						disp = p.get("disappear", None) or "none"
						new = p.get("new", None) or "none"
						seg["node_id"] = f"G:{gs}-{ge}:carrier={tid}:d={disp}:n={new}"
						seg["capacity"] = 2

			segments_by_base[tid] = out
			all_segments.extend(out)

		# ---- Build the segmented graph ----
		g = TrackletGraph()
		g.add_node(GraphNode(node_id="SOURCE", type=NodeType.SOURCE, capacity=0, start_frame=None, end_frame=None, payload={}))
		g.add_node(GraphNode(node_id="SINK", type=NodeType.SINK, capacity=0, start_frame=None, end_frame=None, payload={}))

		for tid, chain in segments_by_base.items():
			for seg in chain:
				node_id = str(seg["node_id"])
				seg_type = str(seg["segment_type"])
				start_f = int(seg["start_frame"])
				end_f = int(seg["end_frame"])
				cap = int(seg["capacity"])

				# D3 binds pings to nodes using (member_tracklet_ids, start_frame, end_frame).
				payload = {
					"tracklet_id": tid,
					"base_tracklet_id": tid,
					"segment_type": seg_type,
					"member_tracklet_ids": [tid],
				}
				if seg_type == "GROUP":
					p = seg["payload"]
					members: List[str] = []
					for cand in [tid, p.get("disappear", None), p.get("new", None)]:
						if cand is None:
							continue
						cand_s = str(cand)
						if cand_s and cand_s != "none" and cand_s not in members:
							members.append(cand_s)
					# Deterministic ordering for member ids.
					payload["member_tracklet_ids"] = sorted(members)
					payload.update(
						{
							"carrier_tracklet_id": tid,
							"disappearing_tracklet_id": p.get("disappear", None),
							"new_tracklet_id": p.get("new", None),
							"kind": p.get("kind", None),
							# Optional geometry/time metadata for downstream pricing (D2).
							# These are derived during D1 group-span construction and are
							# safe to surface via the node payload without changing the
							# public parquet schema.
							"merge_end": p.get("merge_end", None),
							"split_start": p.get("split_start", None),
							"merge_dist_m": p.get("merge_dist_m", None),
							"split_dist_m": p.get("split_dist_m", None),
						}
					)

				node_type = NodeType.GROUP_TRACKLET if seg_type == "GROUP" else NodeType.SINGLE_TRACKLET
				g.add_node(GraphNode(node_id=node_id, type=node_type, capacity=cap, start_frame=start_f, end_frame=end_f, payload=payload))

			# births/deaths for the chain: if first/last are GROUP, use cap=2 like original semantics
			first = chain[0]
			last = chain[-1]
			g.add_edge(GraphEdge(edge_id=f"E:BIRTH:{first['node_id']}", u="SOURCE", v=str(first["node_id"]), type=EdgeType.BIRTH, capacity=int(first["capacity"]), payload={}))
			g.add_edge(GraphEdge(edge_id=f"E:DEATH:{last['node_id']}", u=str(last["node_id"]), v="SINK", type=EdgeType.DEATH, capacity=int(last["capacity"]), payload={}))

			# temporal edges between consecutive segments of same base tracklet
			for a, b in zip(chain, chain[1:]):
				u = str(a["node_id"])
				v = str(b["node_id"])
				g.add_edge(
					GraphEdge(
						edge_id=f"E:CONT:{u}->{v}",
						u=u,
						v=v,
						type=EdgeType.CONTINUE,
						capacity=1,
						payload={"dt_frames": int(b["start_frame"]) - int(a["end_frame"])},
					)
				)

		# ---- Occlusion reconnect edges between different base tracklets ----
		if reconnect_enabled and fps is not None:
			def _count_nearby_births(
				*,
				dest_tid: str,
				dest_start_frame: int,
				dest_start_xy_m: tuple[float, float],
			) -> int:
				"""Count other tracklets that start near dest in time+space (evidence of 2-person re-entry)."""
				cnt = 0
				for other_tid, other_ts in ts_by_tid.items():
					if other_tid == dest_tid:
						continue
					other_start_frame = int(other_ts["start_frame"])
					if (
						abs(other_start_frame - dest_start_frame)
						> promote_group_reconnect_nearby_start_window_frames
					):
						continue
					other_start = endpoints_start.get(other_tid)
					if other_start is None:
						continue
					(other_xy_m, _other_used_raw) = other_start
					if other_xy_m is None:
						continue
					if _dist_m(other_xy_m, dest_start_xy_m) <= promote_group_reconnect_nearby_dist_m:
						cnt += 1
				return cnt

			# Soft Option B: annotate reconnect candidates when a coherent MERGE/SPLIT chain exists.
			# We only shadow when both disappearing and new tracklet ids are present (coherent span).
			shadow_witness_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
			for gs in group_spans:
				try:
					disp = gs.get("disappear", None)
					new = gs.get("new", None)
					merge_end = gs.get("merge_end", None)
					split_start = gs.get("split_start", None)
					if disp is None or new is None or merge_end is None or split_start is None:
						continue
					carrier = str(gs.get("carrier"))
					gs0 = int(gs.get("group_start"))
					ge0 = int(gs.get("group_end"))
					group_node_id = f"G:{gs0}-{ge0}:carrier={carrier}:d={str(disp)}:n={str(new)}"
					key = (str(disp), str(new))
					w = {
						"group_node_id": group_node_id,
						"carrier_tracklet_id": carrier,
						"disappearing_tracklet_id": str(disp),
						"new_tracklet_id": str(new),
						"group_start": int(gs0),
						"group_end": int(ge0),
						"merge_end": int(merge_end),
						"split_start": int(split_start),
					}
					# Deterministic choice if multiple spans map to same (disp,new): pick earliest group_start, then group_end.
					if key not in shadow_witness_by_pair:
						shadow_witness_by_pair[key] = w
					else:
						prev = shadow_witness_by_pair[key]
						if (w["group_start"], w["group_end"]) < (int(prev.get("group_start", 1 << 30)), int(prev.get("group_end", 1 << 30))):
							shadow_witness_by_pair[key] = w
				except Exception:
					# Shadowing must never block reconnect generation.
					continue
			base_ids = sorted(ts_by_tid.keys())

			# -----------------------------------------------------------------
			# Reconnect-only "promotion" (groupish capacity overlay)
			#
			# IMPORTANT:
			# - We do NOT mutate node type/capacity.
			# - We do NOT mutate SOURCE/SINK birth/death capacities.
			# - We only allow capacity=2 on CONT_RECONNECT edges when a SOLO node
			#   is acting as a continuation of a GROUP chain through occlusion.
			# - We compute promotions via a fixed-point loop so it can propagate
			#   (e.g. G->t11 makes t11 groupish, then t11->t14 can also become groupish).
			#
			# Firewall:
			# Do not promote a destination base tracklet that participates in any
			# explicit group span (carrier/disappear/new) — those semantics are already
			# represented by real GROUP nodes and MERGE/SPLIT edges.
			# -----------------------------------------------------------------
			tracklets_in_group_spans: set[str] = set()
			for gs in group_spans:
				try:
					for k in ("carrier", "disappear", "new"):
						v = gs.get(k, None)
						if v is None:
							continue
						vs = str(v)
						if vs and vs != "none":
							tracklets_in_group_spans.add(vs)
				except Exception:
					continue

			# Each candidate describes a possible CONT_RECONNECT edge between two base tracklets.
			# We build these first, then apply the promotion loop, then materialize edges once.
			reconnect_candidates: List[Dict[str, Any]] = []

			def _pick_dest_segment_for_tm(*, tm: str, require_solo: bool = True) -> Optional[Dict[str, Any]]:
				chain_m = segments_by_base.get(tm)
				if not chain_m:
					return None
				if require_solo:
					for seg in chain_m:
						if str(seg.get("segment_type", "")) == "SOLO":
							return seg
					return None
				# fallback: first segment (caller must decide if allowed)
				return chain_m[0]

			for tn in base_ids:
				row_n = ts_by_tid.get(tn)
				if row_n is None:
					continue
				tn_end = int(row_n["end_frame"])
				for tm in base_ids:
					if tm == tn:
						continue
					row_m = ts_by_tid.get(tm)
					if row_m is None:
						continue
					tm_start = int(row_m["start_frame"])
					if tm_start <= tn_end:
						continue
					gap_frames = tm_start - tn_end
					if gap_frames <= 0:
						continue
					# Hard cap reconnect gap (separate from split horizon).
					if gap_frames > reconnect_max_gap_frames:
						continue

					# cannot-link pruning based on anchor keys (mirror legacy CONT logic)
					mi = _must_link_key(tn)
					mj = _must_link_key(tm)
					if mj is not None and mj in set(_cannot_link_for(tn)):
						continue
					if mi is not None and mi in set(_cannot_link_for(tm)):
						continue

					# Endpoint positions
					p_end = endpoints_end.get(tn)
					p_start = endpoints_start.get(tm)
					if p_end is None or p_start is None:
						continue
					if reconnect_boundary_on_mat_required:
						if not _boundary_on_mat_ok(tn, "end"):
							continue
						if not _boundary_on_mat_ok(tm, "start"):
							continue
					dist = _dist_m(p_end[0], p_start[0])
					dt_s = float(gap_frames) / float(fps)
					if dt_s <= 0:
						continue
					speed_mps = float(dist) / max(1e-6, dt_s)
					if speed_mps > float(v_max_mps):
						continue

					chain_n = segments_by_base.get(tn)
					if not chain_n:
						continue

					u_seg = chain_n[-1]
					src_seg_type = str(u_seg.get("segment_type", ""))
					src_is_solo = (src_seg_type == "SOLO")
					src_is_group = (src_seg_type == "GROUP")

					if reconnect_solo_only:
						if not (src_is_solo or (src_is_group and reconnect_allow_group_source)):
							continue

					# Destination selection:
					# Prefer SOLO destination segments when available.
					dest_seg = _pick_dest_segment_for_tm(tm=tm, require_solo=True)
					if dest_seg is None:
						# If tm has no SOLO segments, allow GROUP destination only for GROUP->GROUP when enabled.
						if src_is_group and reconnect_allow_group_source:
							dest_seg = _pick_dest_segment_for_tm(tm=tm, require_solo=False)
							if dest_seg is None:
								continue
							if str(dest_seg.get("segment_type", "")) != "GROUP":
								continue
						else:
							continue

					u_node = str(u_seg["node_id"])
					v_node = str(dest_seg["node_id"])
					dest_seg_type = str(dest_seg.get("segment_type", ""))

					reconnect_candidates.append(
						{
							"tn": str(tn),
							"tm": str(tm),
							"tn_end": int(tn_end),
							"tm_start": int(tm_start),
							"gap_frames": int(gap_frames),
							"dt_s": float(dt_s),
							"dist_m": float(dist),
							"speed_mps": float(speed_mps),
							"u_node": u_node,
							"v_node": v_node,
							"src_seg_type": src_seg_type,
							"dest_seg_type": dest_seg_type,
						}
					)

			# Deterministic iteration order for both promotion + edge materialization.
			reconnect_candidates.sort(
				key=lambda c: (
					int(c["tn_end"]),
					int(c["tm_start"]),
					str(c["tn"]),
					str(c["tm"]),
					str(c["u_node"]),
					str(c["v_node"]),
				),
			)

			# Fixed-point loop: compute which SOLO nodes should be treated as "groupish"
			# (capacity=2) for reconnect edges ONLY.
			promoted_groupish_nodes: set[str] = set()
			if promote_group_reconnect_enabled and reconnect_candidates:
				changed = True
				max_iters = int(d1_cfg.get("promote_group_reconnect_max_iters", 25))
				it = 0
				while changed and it < max_iters:
					changed = False
					it += 1
					for cand in reconnect_candidates:
						# Only promote SOLO destinations (never mutate true GROUP nodes).
						if str(cand.get("dest_seg_type")) != "SOLO":
							continue
						tm = str(cand["tm"])
						v_node = str(cand["v_node"])

						# Firewall: do not promote destinations whose base tracklet participates in group spans.
						if tm in tracklets_in_group_spans:
							continue

						# Source must be groupish (true GROUP segment OR previously promoted).
						u_node = str(cand["u_node"])
						src_groupish = bool(str(cand.get("src_seg_type")) == "GROUP" or (u_node in promoted_groupish_nodes))
						if not src_groupish:
							continue

						# Require reconnect into the *start* of tm (within slack)
						# and into the chosen dest segment start (within slack).
						try:
							tm_start = int(cand["tm_start"])
							dest_start_frame = int(ts_by_tid[tm]["start_frame"])
						except Exception:
							continue
						if abs(dest_start_frame - tm_start) > reconnect_boundary_slack_frames:
							continue

						# If the chosen dest segment isn't anchored at tm_start, skip promotion.
						# (We only want "group continuation through occlusion", not mid-life promotions.)
						try:
							dest_seg_chain = segments_by_base.get(tm, [])
							dest_seg_obj = None
							for seg in dest_seg_chain:
								if str(seg.get("node_id")) == v_node:
									dest_seg_obj = seg
									break
							if dest_seg_obj is None:
								continue
							if abs(int(dest_seg_obj["start_frame"]) - tm_start) > reconnect_boundary_slack_frames:
								continue
						except Exception:
							continue

						# Nearby birth evidence: if there is another SOLO start near tm, treat as 2-person re-entry.
						start_rec = endpoints_start.get(tm)
						if start_rec is None:
							continue
						(tm_start_xy_m, _tm_used_raw) = start_rec
						if tm_start_xy_m is None:
							continue
						nearby = _count_nearby_births(
												dest_tid=tm,
												dest_start_frame=tm_start,
												dest_start_xy_m=tm_start_xy_m,
											)
						if nearby != 0:
							continue

						if v_node not in promoted_groupish_nodes:
							promoted_groupish_nodes.add(v_node)
							changed = True

			# Materialize CONT_RECONNECT edges with capacity determined by:
			# - true GROUP segment endpoints => capacity 2
			# - promoted groupish SOLO nodes => capacity 2 (reconnect-only overlay)
			#
			# NOTE (POC_2_TAGS gate):
			# Current POC requires ALL CONTINUE edges (including reconnect variants)
			# to have capacity=1. We therefore encode "desired capacity" only in payload.
			# Downstream (D3) may optionally interpret desired_capacity once the gate is lifted.
			for cand in reconnect_candidates:
				tn = str(cand["tn"])
				tm = str(cand["tm"])
				u_node = str(cand["u_node"])
				v_node = str(cand["v_node"])

				src_seg_type = str(cand.get("src_seg_type", ""))
				dest_seg_type = str(cand.get("dest_seg_type", ""))

				src_groupish = bool(src_seg_type == "GROUP" or (u_node in promoted_groupish_nodes))
				dest_groupish = bool(dest_seg_type == "GROUP" or (v_node in promoted_groupish_nodes))

				cap_u = 2 if src_groupish else 1
				cap_v = 2 if dest_groupish else 1
				desired_cap = min(cap_u, cap_v)
				emit_cap = 1  # hard-gated by POC_2_TAGS: CONTINUE edges must be capacity=1

				edge_id = f"E:CONT_RECONNECT:{u_node}->{v_node}"
				payload = {
					"dt_frames": int(cand["gap_frames"]),
					"reconnect": True,
					"dt_s": float(cand["dt_s"]),
					"dist_m": float(cand["dist_m"]),
					"speed_mps": float(cand["speed_mps"]),
					"desired_capacity": int(desired_cap),
					# Debug/traceability:
					"src_groupish": bool(src_groupish),
					"dest_groupish": bool(dest_groupish),
					"promoted_src": bool(u_node in promoted_groupish_nodes),
					"promoted_dest": bool(v_node in promoted_groupish_nodes),
				}
				w = shadow_witness_by_pair.get((str(tn), str(tm)))
				if w is not None:
					payload["shadowed_by_group_chain"] = True
					payload["shadow_witness"] = w
				else:
					payload["shadowed_by_group_chain"] = False

				g.add_edge(
					GraphEdge(
						edge_id=edge_id,
						u=u_node,
						v=v_node,
						type=EdgeType.CONTINUE,
						capacity=int(emit_cap),
						payload=payload,
					)
				)
				reconnect_edges_debug.append(
					{
						"from_tracklet_id": tn,
						"to_tracklet_id": tm,
						"gap_frames": int(cand["gap_frames"]),
						"dt_s": float(cand["dt_s"]),
						"dist_m": float(cand["dist_m"]),
						"speed_mps": float(cand["speed_mps"]),
							"capacity": int(emit_cap),
							"desired_capacity": int(desired_cap),
						"src_groupish": bool(src_groupish),
						"dest_groupish": bool(dest_groupish),
						"promoted_src": bool(u_node in promoted_groupish_nodes),
						"promoted_dest": bool(v_node in promoted_groupish_nodes),
						"shadowed_by_group_chain": bool(payload.get("shadowed_by_group_chain", False)),
						"shadow_group_node_id": (payload.get("shadow_witness", {}) or {}).get("group_node_id", None),
					}
				)

		# MERGE edges: disappearing identity flows into the GROUP segment.
		for gspan in group_spans:
			disp = gspan.get("disappear", None)
			merge_end = gspan.get("merge_end", None)
			if disp is None or merge_end is None:
				continue
			carrier = str(gspan["carrier"])
			gs = int(gspan["group_start"])
			ge = int(gspan["group_end"])
			group_id = f"G:{gs}-{ge}:carrier={carrier}:d={(disp or 'none')}:n={(gspan.get('new') or 'none')}"

			# Find disappearing segment containing merge_end; fallback to legacy node if unsplit.
			u_node: Optional[str] = None
			for seg in segments_by_base.get(str(disp), []):
				if int(seg["start_frame"]) <= int(merge_end) <= int(seg["end_frame"]):
					u_node = str(seg["node_id"])
					break
			if u_node is None:
				u_node = f"T:{disp}"
			g.add_edge(GraphEdge(edge_id=f"E:MERGE:{u_node}->{group_id}", u=u_node, v=group_id, type=EdgeType.MERGE, capacity=1, payload={"merge_end": int(merge_end)}))

		# SPLIT edges: GROUP segment to new identity segment.
		for gspan in group_spans:
			new_tid = gspan.get("new", None)
			split_start = gspan.get("split_start", None)
			if new_tid is None or split_start is None:
				continue
			carrier = str(gspan["carrier"])
			gs = int(gspan["group_start"])
			ge = int(gspan["group_end"])
			group_id = f"G:{gs}-{ge}:carrier={carrier}:d={(gspan.get('disappear') or 'none')}:n={(new_tid or 'none')}"

			v_node: Optional[str] = None
			for seg in segments_by_base.get(str(new_tid), []):
				if int(seg["start_frame"]) <= int(split_start) <= int(seg["end_frame"]):
					v_node = str(seg["node_id"])
					break
			if v_node is None:
				v_node = f"T:{new_tid}"
			g.add_edge(GraphEdge(edge_id=f"E:SPLIT:{group_id}->{v_node}", u=group_id, v=v_node, type=EdgeType.SPLIT, capacity=1, payload={"split_start": int(split_start)}))

		# Validate + write debug artifacts (adds segments + triggers).
		g.validate()

		# Canonical graph artifacts (always written)
		nodes_rows: List[Dict[str, Any]] = []
		for n in g.sorted_nodes():
			nodes_rows.append(
				{
					"node_id": n.node_id,
					"node_type": str(n.type),
					"capacity": int(n.capacity),
					"start_frame": n.start_frame,
					"end_frame": n.end_frame,
					# Solver-agnostic join keys / features for D2 pricing (redundant with payload_json).
					"base_tracklet_id": n.payload.get("base_tracklet_id"),
					"segment_type": n.payload.get("segment_type"),
					"carrier_tracklet_id": n.payload.get("carrier_tracklet_id"),
					"disappearing_tracklet_id": n.payload.get("disappearing_tracklet_id"),
					"new_tracklet_id": n.payload.get("new_tracklet_id"),
					"must_link_anchor_key": n.payload.get("must_link_anchor_key"),
					"must_link_confidence": n.payload.get("must_link_confidence"),
					"cannot_link_anchor_keys_json": json.dumps(
						n.payload.get("cannot_link_anchor_keys", []), sort_keys=True
					),
					# Full structured payload (lossless; forwards compatible).
					"payload_json": json.dumps(n.payload, sort_keys=True),
				}
			)
		edges_rows: List[Dict[str, Any]] = []
		for e in g.sorted_edges():
			# Extract dt_frames, merge_end, split_start from payload if present, else None
			dt_frames = e.payload.get("dt_frames")
			merge_end = e.payload.get("merge_end")
			split_start = e.payload.get("split_start")
			edges_rows.append(
				{
					"edge_id": e.edge_id,
					"edge_type": str(e.type),
					"u": e.u,
					"v": e.v,
					"capacity": int(e.capacity),
					"dt_frames": dt_frames if dt_frames is not None else None,
					"merge_end": merge_end if merge_end is not None else None,
					"split_start": split_start if split_start is not None else None,
					"payload_json": json.dumps(e.payload, sort_keys=True),
				}
			)
		segments_df = (pd.DataFrame(all_segments) if all_segments else pd.DataFrame([]))
		if not segments_df.empty:
			# Add payload_json column as required by schema
			segments_df["payload_json"] = segments_df["payload"].apply(lambda p: json.dumps(p, sort_keys=True))
			segments_df = segments_df.drop(columns=["payload"])  # Remove payload dict column to match schema

		nodes_df = pd.DataFrame(nodes_rows)
		# Ensure integer columns are int, not float
		for col in ("start_frame", "end_frame"):
			if col in nodes_df.columns:
				nodes_df[col] = nodes_df[col].astype('Int64' if nodes_df[col].isnull().any() else int)
		# Coerce must_link_confidence to float if present
		if "must_link_confidence" in nodes_df.columns:
			nodes_df["must_link_confidence"] = pd.to_numeric(nodes_df["must_link_confidence"], errors="coerce")
		# Diagnostics for debugging
		if nodes_df.empty:
			print("[D1 DEBUG] nodes_df is empty before writing d1_graph_nodes.parquet")
		else:
			print("[D1 DEBUG] nodes_df columns and dtypes before write:\n", nodes_df.dtypes)
		edges_df = pd.DataFrame(edges_rows)
		# Coerce dt_frames, merge_end, split_start to Int64 (nullable int) if present
		for col in ("dt_frames", "merge_end", "split_start"):
			if col in edges_df.columns:
				edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce").astype('Int64')
		# Diagnostics for debugging
		if edges_df.empty:
			print("[D1 DEBUG] edges_df is empty before writing d1_graph_edges.parquet")
		else:
			print("[D1 DEBUG] edges_df columns and dtypes before write:\n", edges_df.dtypes)
		validate_df_schema_by_key(nodes_df, "d1_graph_nodes")
		validate_df_schema_by_key(edges_df, "d1_graph_edges")
		# empty is allowed early / for short clips
		if len(segments_df) > 0:
			validate_df_schema_by_key(segments_df, "d1_segments")

		nodes_out = layout.d1_graph_nodes_parquet()
		edges_out = layout.d1_graph_edges_parquet()
		segs_out = layout.d1_segments_parquet()
		nodes_out.parent.mkdir(parents=True, exist_ok=True)
		nodes_df.to_parquet(nodes_out, index=False)
		edges_df.to_parquet(edges_out, index=False)
		segments_df.to_parquet(segs_out, index=False)

		debug_outputs: Dict[str, str] = {
			"d1_graph_nodes_parquet": str(nodes_out),
			"d1_graph_edges_parquet": str(edges_out),
			"d1_segments_parquet": str(segs_out),
		}
		if write_debug_graph_artifacts:
			debug_dir.mkdir(parents=True, exist_ok=True)
			groups_path = debug_dir / "d1_group_spans.parquet"
			suppressed_path = debug_dir / "d1_suppressed_continue_edges.parquet"
			pd.DataFrame(group_spans).to_parquet(groups_path, index=False)
			# segmentation path doesn’t use suppressed-continue, but keep file for compat
			pd.DataFrame([]).to_parquet(suppressed_path, index=False)
			(pd.DataFrame(merge_triggers) if merge_triggers else pd.DataFrame([])).to_parquet(debug_dir / "d1_merge_triggers.parquet", index=False)
			(pd.DataFrame(split_triggers) if split_triggers else pd.DataFrame([])).to_parquet(debug_dir / "d1_split_triggers.parquet", index=False)
			(pd.DataFrame(suppressed_split_triggers) if suppressed_split_triggers else pd.DataFrame([])).to_parquet(debug_dir / "d1_suppressed_split_triggers.parquet", index=False)
			(pd.DataFrame(suppressed_split_triggers_entrance) if suppressed_split_triggers_entrance else pd.DataFrame([])).to_parquet(
				debug_dir / "d1_suppressed_split_triggers_entrance.parquet",
				index=False,
			)
			(pd.DataFrame(suppressed_start_merged_entrance) if suppressed_start_merged_entrance else pd.DataFrame([])).to_parquet(
				debug_dir / "d1_suppressed_start_merged_entrance.parquet",
				index=False,
			)
			(pd.DataFrame(suppressed_group_spans) if suppressed_group_spans else pd.DataFrame([])).to_parquet(debug_dir / "d1_suppressed_group_spans.parquet", index=False)
			(pd.DataFrame(reconnect_edges_debug) if reconnect_edges_debug else pd.DataFrame([])).to_parquet(debug_dir / "d1_reconnect_edges.parquet", index=False)

			debug_outputs.update(
				{
					"d1_group_spans_parquet": str(groups_path),
					"d1_suppressed_continue_edges_parquet": str(suppressed_path),
					"d1_merge_triggers_parquet": str(debug_dir / "d1_merge_triggers.parquet"),
					"d1_split_triggers_parquet": str(debug_dir / "d1_split_triggers.parquet"),
					"d1_suppressed_split_triggers_parquet": str(debug_dir / "d1_suppressed_split_triggers.parquet"),
					"d1_suppressed_split_triggers_entrance_parquet": str(
						debug_dir / "d1_suppressed_split_triggers_entrance.parquet"
					),
					"d1_suppressed_start_merged_entrance_parquet": str(debug_dir / "d1_suppressed_start_merged_entrance.parquet"),
					"d1_suppressed_group_spans_parquet": str(debug_dir / "d1_suppressed_group_spans.parquet"),
					"d1_reconnect_edges_parquet": str(debug_dir / "d1_reconnect_edges.parquet"),
				}
			)


		_write_audit_event(
			audit_path,
			{
				"event": "d1_graph_built",
				"event_type": "d1_graph_built",
				"stage": "D1",
				"ts_ms": _now_ms(),
				"manifest": {"fps": fps, "frame_count": frame_count, "duration_ms": duration_ms},
				"video": {
					"input_video_path": _get_manifest_video_path(manifest),
					"frame_wh": tuple(frame_wh) if frame_wh is not None else None,
				},
				"border_gate": {
					"enabled": bool(split_border_gate_enabled),
					"mode": str(border_gate_mode),
					"margin_px": int(split_border_margin_px),
					"entrance_k_frames": int(d1_cfg.get("split_entrance_k_frames", 10)),
					"entrance_min_samples": int(d1_cfg.get("split_entrance_min_samples", 3)),
					"entrance_min_inward_px": int(d1_cfg.get("split_entrance_min_inward_px", 20)),
					"suppress_start_merged_if_entrance_like": bool(suppress_start_merged_if_entrance_like),
					"allow_observed_fallback": bool(d1_cfg.get("split_entrance_allow_observed_fallback", True)),
				},
				"graph": {
					"n_nodes": len(g.nodes),
					"n_edges": len(g.edges),
					"n_group_nodes": len([n for n in g.nodes.values() if n.type == NodeType.GROUP_TRACKLET]),
					"n_segments_total": len(all_segments),
					"n_group_spans_total": len(group_spans),
				},
				"coords": {
					"primary": ["x_m_repaired", "y_m_repaired"],
					"fallback": ["x_m", "y_m"],
					"frames_used_repaired": used_repaired_frames,
					"frames_used_raw_fallback": used_raw_frames,
					"on_mat_missing": bool(on_mat_missing),
				},
				"debug_outputs": debug_outputs,
			},
		)

		return g

	# ---- build base graph ----
	g = TrackletGraph()
	g.add_node(GraphNode(node_id="SOURCE", type=NodeType.SOURCE, capacity=0, start_frame=None, end_frame=None, payload={}))
	g.add_node(GraphNode(node_id="SINK", type=NodeType.SINK, capacity=0, start_frame=None, end_frame=None, payload={}))

	# nodes and birth/death
	for _, r in ts.sort_values(["start_frame", "tracklet_id"], kind="mergesort").iterrows():
		tid = str(r["tracklet_id"])
		node_id = f"T:{tid}"
		payload = {
			"tracklet_id": tid,
			"must_link_anchor_key": r.get("must_link_anchor_key", None),
			"cannot_link_anchor_keys": _parse_json_list(r.get("cannot_link_anchor_keys_json", None)),
		}
		g.add_node(
			GraphNode(
				node_id=node_id,
				type=NodeType.SINGLE_TRACKLET,
				capacity=1,
				start_frame=int(r["start_frame"]),
				end_frame=int(r["end_frame"]),
				payload=payload,
			)
		)
		g.add_edge(GraphEdge(edge_id=f"E:BIRTH:{node_id}", u="SOURCE", v=node_id, type=EdgeType.BIRTH, capacity=1, payload={}))
		g.add_edge(GraphEdge(edge_id=f"E:DEATH:{node_id}", u=node_id, v="SINK", type=EdgeType.DEATH, capacity=1, payload={}))

	# helper for cannot-link pruning using anchor keys
	def cannot_link_for(tid: str) -> List[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return []
		return _parse_json_list(row.get("cannot_link_anchor_keys_json", None))

	def must_link_key(tid: str) -> Optional[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return None
		val = row.get("must_link_anchor_key", None)
		return str(val) if val not in (None, "", "null") else None

	# ---- CONTINUE edges (single->single) ----
	continue_pruned_cannot = 0
	singles = ts.sort_values(["end_frame", "tracklet_id"], kind="mergesort")[["tracklet_id", "start_frame", "end_frame"]]
	singles_list = [(str(r.tracklet_id), int(r.start_frame), int(r.end_frame)) for r in singles.itertuples(index=False)]

	# Pre-sort potential successors by start_frame
	succ = ts.sort_values(["start_frame", "tracklet_id"], kind="mergesort")[["tracklet_id", "start_frame", "end_frame"]]
	succ_list = [(str(r.tracklet_id), int(r.start_frame), int(r.end_frame)) for r in succ.itertuples(index=False)]

	for tid_i, _si, ei in singles_list:
		for tid_j, sj, _ej in succ_list:
			if sj <= ei:
				continue
			dt = sj - ei
			if dt > max_continue_gap_frames:
				break

			# cannot-link pruning based on anchor keys
			mi = must_link_key(tid_i)
			mj = must_link_key(tid_j)
			if mi is not None and mj is not None:
				# different anchors are fine; cannot-link lists apply
				pass
			if mj is not None and mj in set(cannot_link_for(tid_i)):
				continue_pruned_cannot += 1
				continue
			if mi is not None and mi in set(cannot_link_for(tid_j)):
				continue_pruned_cannot += 1
				continue

			u = f"T:{tid_i}"
			v = f"T:{tid_j}"
			g.add_edge(
				GraphEdge(
					edge_id=f"E:CONT:{u}->{v}",
					u=u,
					v=v,
					type=EdgeType.CONTINUE,
					capacity=1,
					payload={"dt_frames": dt},
				)
			)

	# ---- GROUP inference ----
	group_nodes_created = 0
	group_open_ended = 0
	group_from_start = 0
	merge_split_groups = 0
	suppressed_continue_edges = 0
	suppressed_continue_edges_rows: List[Dict[str, Any]] = []

	# Precompute endpoints for efficiency (deterministic)
	endpoints_end: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
	endpoints_start: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
	for tid, _, _ in singles_list:
		endpoints_end[tid] = endpoint_end(tid)
		endpoints_start[tid] = endpoint_start(tid)

	# candidate merges: pairs close in time and space at end
	candidates: List[Tuple[int, str, str, float]] = []  # (disappear_end, disappear_tid, carrier_tid, dist_m)

	def _carrier_pos_near_frame(tid: str, target_frame: int, max_offset: int) -> Optional[Tuple[Tuple[float, float], bool]]:
		# Prefer exact frame; then nearest offsets.
		for off in range(0, max_offset + 1):
			for f in ((target_frame - off), (target_frame + off)) if off > 0 else (target_frame,):
				res = carrier_pos_at_frame(tid, int(f))
				if res is not None:
					return res
		return None

	for tid_d, sd, ed in singles_list:
		disp_end = int(ed)
		p_disp = endpoints_end.get(tid_d)
		if p_disp is None:
			continue
		for tid_s, ss, es in singles_list:
			if tid_s == tid_d:
				continue
			# carrier must exist at the disappearance frame
			if int(ss) > disp_end:
				continue
			if int(es) < disp_end:
				continue
			p_car = _carrier_pos_near_frame(tid_s, disp_end, merge_end_sync_frames)
			if p_car is None:
				continue
			d = _dist_m(p_disp[0], p_car[0])
			if d > merge_dist_m:
				continue
			candidates.append((disp_end, tid_d, tid_s, float(d)))

	# Deterministic greedy selection: one carrier per disappearing tracklet.
	candidates.sort(key=lambda t: (t[0], t[1], t[3], t[2]))

	used_in_merge: Dict[str, int] = {}
	groups: List[Dict[str, Any]] = []

	def _tracklet_span(tid: str) -> Tuple[int, int]:
		row = ts_by_tid[tid]
		return int(row["start_frame"]), int(row["end_frame"])

	for disp_end, disappear, survivor, _dist in candidates:
		if used_in_merge.get(disappear) == disp_end or used_in_merge.get(survivor) == disp_end:
			continue
		used_in_merge[disappear] = disp_end
		used_in_merge[survivor] = disp_end

		group_start = disp_end + 1
		_surv_start, surv_end = _tracklet_span(survivor)
		if surv_end < group_start:
			continue

		best_split: Optional[Tuple[int, float, str]] = None  # (start_frame, dist, tid)
		for tid_n, sn, _en in succ_list:
			if tid_n == survivor:
				continue
			if sn < group_start:
				continue
			if sn > group_start + split_search_horizon_frames:
				break
			psn = endpoints_start.get(tid_n)
			if psn is None:
				continue
			pcar = carrier_pos_at_frame(survivor, sn)
			if pcar is None:
				pcar = endpoints_end.get(survivor)
			if pcar is None:
				continue
			d = _dist_m(psn[0], pcar[0])
			if d > split_dist_m:
				continue
			cand = (sn, d, tid_n)
			if best_split is None or cand < best_split:
				best_split = cand

		if best_split is not None:
			split_start, split_dist, tid_new = best_split
			group_end = split_start - 1
			if group_end < group_start:
				continue
			groups.append(
				{
					"kind": "merge_split",
					"carrier": survivor,
					"disappear": disappear,
					"new": tid_new,
					"group_start": group_start,
					"group_end": group_end,
					"split_start": split_start,
					"split_dist_m": split_dist,
					"merge_end": disp_end,
				}
			)
			merge_split_groups += 1
		else:
			if last_frame is not None:
				group_end = last_frame
				if group_end >= group_start:
					groups.append(
						{
							"kind": "merge_open_end",
							"carrier": survivor,
							"disappear": disappear,
							"new": None,
							"group_start": group_start,
							"group_end": group_end,
							"merge_end": disp_end,
						}
					)
					group_open_ended += 1

	# Clip-start merged inference: split-like event without explicit merge
	if enable_group_nodes:
		for tid_s, ss, _es in succ_list:
			if ss > 2:
				break
			best_split2: Optional[Tuple[int, float, str]] = None
			for tid_n, sn, _en in succ_list:
				if tid_n == tid_s:
					continue
				if sn <= ss:
					continue
				if sn > ss + split_search_horizon_frames:
					break
				psn = endpoints_start.get(tid_n)
				if psn is None:
					continue
				# Require direct evidence the carrier exists at the split start frame.
				pcar = carrier_pos_at_frame(tid_s, sn)
				if pcar is None:
					continue
				d = _dist_m(psn[0], pcar[0])
				if d > split_dist_m:
					continue
				cand = (sn, d, tid_n)
				if best_split2 is None or cand < best_split2:
					best_split2 = cand
			if best_split2 is None:
				continue
			sn, dist2, tid_n = best_split2
			group_start = 0
			group_end = sn - 1
			if group_end < group_start:
				continue
			# Avoid duplicating a (carrier,new) group span already inferred via merge/split.
			already_pair = any(
				(gr.get("carrier") == tid_s and gr.get("new") == tid_n and int(gr.get("group_end", -1)) == group_end)
				for gr in groups
			)
			if already_pair:
				continue
			already = any(
				(gr["group_start"] == 0 and gr["group_end"] == group_end and gr.get("carrier") == tid_s) for gr in groups
			)
			if already:
				continue
			groups.append(
				{
					"kind": "start_split",
					"carrier": tid_s,
					"disappear": None,
					"new": tid_n,
					"group_start": group_start,
					"group_end": group_end,
					"split_start": sn,
					"split_dist_m": dist2,
				}
			)
			group_from_start += 1
			break

	# ---- materialize groups into graph ----
	if enable_group_nodes:
		for gr in sorted(groups, key=lambda x: (x["group_start"], x["group_end"], str(x.get("carrier")), str(x.get("new")))):
			gs = int(gr["group_start"])
			ge = int(gr["group_end"])
			if ge < gs:
				continue
			carrier = str(gr["carrier"])
			disappear = gr.get("disappear", None)
			new_tid = gr.get("new", None)
			gn_id = f"G:{gs}-{ge}:carrier={carrier}:d={disappear or 'none'}:n={new_tid or 'none'}"
			g.add_node(
				GraphNode(
					node_id=gn_id,
					type=NodeType.GROUP_TRACKLET,
					capacity=2,
					start_frame=gs,
					end_frame=ge,
					payload={
						"carrier_tracklet_id": carrier,
						"disappearing_tracklet_id": disappear,
						"new_tracklet_id": new_tid,
						"kind": gr["kind"],
					},
				)
			)
			group_nodes_created += 1

			# group-at-start: birth cap=2
			if gs == 0:
				g.add_edge(GraphEdge(edge_id=f"E:BIRTH:{gn_id}", u="SOURCE", v=gn_id, type=EdgeType.BIRTH, capacity=2, payload={}))

			# group-at-end: death cap=2
			if last_frame is not None and ge == last_frame:
				g.add_edge(GraphEdge(edge_id=f"E:DEATH:{gn_id}", u=gn_id, v="SINK", type=EdgeType.DEATH, capacity=2, payload={}))

			# merge edges (two singles into group) when we have disappear+carrier
			if disappear is not None:
				u1 = f"T:{disappear}"
				u2 = f"T:{carrier}"
				g.add_edge(
					GraphEdge(
						edge_id=f"E:MERGE:{u1}->{gn_id}",
						u=u1,
						v=gn_id,
						type=EdgeType.MERGE,
						capacity=1,
						payload={"merge_end": int(gr.get("merge_end", -1))},
					)
				)
				g.add_edge(
					GraphEdge(
						edge_id=f"E:MERGE:{u2}->{gn_id}",
						u=u2,
						v=gn_id,
						type=EdgeType.MERGE,
						capacity=1,
						payload={"merge_end": int(gr.get("merge_end", -1))},
					)
				)

			# split edges: group -> carrier and group -> new
			if new_tid is not None:
				v1 = f"T:{carrier}"
				v2 = f"T:{new_tid}"
				g.add_edge(
					GraphEdge(
						edge_id=f"E:SPLIT:{gn_id}->{v1}",
						u=gn_id,
						v=v1,
						type=EdgeType.SPLIT,
						capacity=1,
						payload={"split_start": int(gr.get("split_start", -1))},
					)
				)
				g.add_edge(
					GraphEdge(
						edge_id=f"E:SPLIT:{gn_id}->{v2}",
						u=gn_id,
						v=v2,
						type=EdgeType.SPLIT,
						capacity=1,
						payload={"split_start": int(gr.get("split_start", -1))},
					)
				)

	# ---- suppress conflicting CONTINUE edges around group spans ----
	if enable_group_nodes and group_nodes_created > 0:
		group_intervals: List[Tuple[int, int, str]] = []
		for node in g.nodes.values():
			if node.type == NodeType.GROUP_TRACKLET:
				group_intervals.append((int(node.start_frame), int(node.end_frame), node.node_id))
		group_intervals.sort()

		def _in_any_group(frame_idx: int) -> bool:
			for a, b, _ in group_intervals:
				if a <= frame_idx <= b:
					return True
			return False

		edges_to_remove: List[str] = []
		edges_remove_detail: List[Tuple[str, str, str, int, int]] = []
		for e in g.edges.values():
			if e.type != EdgeType.CONTINUE:
				continue
			u_tid = e.u.split("T:", 1)[-1] if e.u.startswith("T:") else None
			v_tid = e.v.split("T:", 1)[-1] if e.v.startswith("T:") else None
			if u_tid is None or v_tid is None:
				continue
			ei = int(ts_by_tid[u_tid]["end_frame"])
			sj = int(ts_by_tid[v_tid]["start_frame"])
			if _in_any_group(ei + 1) or _in_any_group(sj - 1):
				edges_to_remove.append(e.edge_id)
				edges_remove_detail.append((e.edge_id, e.u, e.v, ei, sj))
		for eid in edges_to_remove:
			g.edges.pop(eid, None)
		suppressed_continue_edges = len(edges_to_remove)
		# Preserve suppressed edges as a debug-only artifact for candidate visibility.
		for edge_id, u, v, u_end_frame, v_start_frame in edges_remove_detail:
			suppressed_continue_edges_rows.append(
				{
					"edge_id": edge_id,
					"u": u,
					"v": v,
					"u_end_frame": int(u_end_frame),
					"v_start_frame": int(v_start_frame),
					"reason": "suppressed_by_group_interval",
				}
			)
	# ---- debug artifacts (dev-only; not part of public contracts) ----
	debug_outputs: Dict[str, str] = {}
	if write_debug_graph_artifacts:
		debug_dir.mkdir(parents=True, exist_ok=True)
		nodes_path = debug_dir / "d1_graph_nodes.parquet"
		edges_path = debug_dir / "d1_graph_edges.parquet"
		groups_path = debug_dir / "d1_group_spans.parquet"
		suppressed_path = debug_dir / "d1_suppressed_continue_edges.parquet"

		nodes_rows: List[Dict[str, Any]] = []
		for n in g.sorted_nodes():
			nodes_rows.append(
				{
					"node_id": n.node_id,
					"node_type": str(n.type),
					"capacity": int(n.capacity),
					"start_frame": n.start_frame,
					"end_frame": n.end_frame,
					# Solver-agnostic join keys / features for D2 pricing (redundant with payload_json).
					"base_tracklet_id": n.payload.get("base_tracklet_id"),
					"segment_type": n.payload.get("segment_type"),
					"carrier_tracklet_id": n.payload.get("carrier_tracklet_id"),
					"disappearing_tracklet_id": n.payload.get("disappearing_tracklet_id"),
					"new_tracklet_id": n.payload.get("new_tracklet_id"),
					"must_link_anchor_key": n.payload.get("must_link_anchor_key"),
					"must_link_confidence": n.payload.get("must_link_confidence"),
					"cannot_link_anchor_keys_json": json.dumps(
						n.payload.get("cannot_link_anchor_keys", []), sort_keys=True
					),
					# Full structured payload (lossless; forwards compatible).
					"payload_json": json.dumps(n.payload, sort_keys=True),
				}
			)
		pd.DataFrame(nodes_rows).to_parquet(nodes_path, index=False)

		edges_rows: List[Dict[str, Any]] = []
		for e in g.sorted_edges():
			edges_rows.append(
				{
					"edge_id": e.edge_id,
					"edge_type": str(e.type),
					"u": e.u,
					"v": e.v,
					"capacity": int(e.capacity),
					"dt_frames": e.payload.get("dt_frames"),
					"merge_end": e.payload.get("merge_end"),
					"split_start": e.payload.get("split_start"),
					"payload_json": json.dumps(e.payload, sort_keys=True),
				}
			)
		pd.DataFrame(edges_rows).to_parquet(edges_path, index=False)

		pd.DataFrame(groups).to_parquet(groups_path, index=False)
		pd.DataFrame(suppressed_continue_edges_rows).to_parquet(suppressed_path, index=False)

		# clip-relative debug paths for audit traceability
		debug_outputs = {
			"d1_graph_nodes_parquet": str(nodes_path.relative_to(clip_root)),
			"d1_graph_edges_parquet": str(edges_path.relative_to(clip_root)),
			"d1_group_spans_parquet": str(groups_path.relative_to(clip_root)),
			"d1_suppressed_continue_edges_parquet": str(suppressed_path.relative_to(clip_root)),
		}

	# ---- final validate ----
	g.validate()

	# ---- audit ----
	audit_evt = {
		"ts_ms": _now_ms(),
		"stage": "D1",
		"event": "d1_graph_built",
		"inputs": {
			"bank_frames": str(frames_path),
			"bank_summaries": str(summ_path),
		},
		"manifest": {
			"fps": fps,
			"frame_count": frame_count,
			"duration_ms": duration_ms,
		},
		"coords": {
			"primary": ["x_m_repaired", "y_m_repaired"],
			"fallback": ["x_m", "y_m"],
			"on_mat_missing": on_mat_missing,
			"frames_used_repaired": used_repaired_frames,
			"frames_used_raw_fallback": used_raw_frames,
			"start_endpoints_used_raw": used_raw_start,
			"end_endpoints_used_raw": used_raw_end,
		},
		"graph": {
			"n_nodes": len(g.nodes),
			"n_edges": len(g.edges),
			"n_group_nodes": group_nodes_created,
			"n_groups_from_start": group_from_start,
			"n_groups_open_ended": group_open_ended,
			"n_groups_merge_split": merge_split_groups,
			"continue_pruned_cannot": continue_pruned_cannot,
			"suppressed_continue_edges": suppressed_continue_edges,
			"missing_endpoint_coords": missing_endpoint_coords,
		},
		"debug_outputs": debug_outputs if write_debug_graph_artifacts else None,
	}
	_write_audit_event(audit_path, audit_evt)
	return g
