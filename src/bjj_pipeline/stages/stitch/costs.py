"""Stage D2 — Edge cost terms + normalization (solver-agnostic).

D2 prices the *canonical* D1 candidate edges:
    stage_D/d1_graph_edges.parquet

It does not invent edges and does not run a solver. It emits:
    - stage_D/d2_edge_costs.parquet (one row per D1 edge, with term breakdowns)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DISALLOWED_EDGE_COST: float = 1e6
EPS: float = 1e-6


def _is_nullish(val: Any) -> bool:
	"""Return True for None/NaN/pandas NA-style values.

	This is intentionally permissive so we don't accidentally call
	int()/float() on extension dtypes like pandas.NA.
	"""
	if val is None:
		return True
	try:
		# Handles pandas.NA, numpy.nan, and friends.
		if pd.isna(val):
			return True
	except Exception:
		# pd.isna may not like arbitrary containers; treat as non-null here.
		pass
	if isinstance(val, float) and math.isnan(val):
		return True
	return False


def _hinge2(z: float) -> float:
	return max(0.0, z - 1.0) ** 2


def _effective_xy(row: pd.Series) -> Tuple[float | None, float | None]:
	"""Return effective world coords using repaired-or-original fallback."""
	x = row.get("x_m_repaired", None)
	y = row.get("y_m_repaired", None)
	if _is_nullish(x):
		x = row.get("x_m", None)
	if _is_nullish(y):
		y = row.get("y_m", None)
	if _is_nullish(x) or _is_nullish(y):
		return None, None
	try:
		xf = float(x)
		yf = float(y)
	except Exception:
		return None, None
	if math.isnan(xf) or math.isnan(yf):
		return None, None
	return xf, yf


def _parse_edge_type(edge_type: str) -> str:
	# Stored as string like "EdgeType.CONTINUE"; normalize to suffix.
	if not isinstance(edge_type, str):
		return str(edge_type)
	if "." in edge_type:
		return edge_type.split(".")[-1]
	return edge_type


def compute_edge_costs(
	*,
	d1_edges: pd.DataFrame,
	d1_nodes: pd.DataFrame,
	bank_frames: pd.DataFrame,
	fps: float,
	cfg: Dict[str, Any],
	v_cost_scale_mps_resolved: float,
	v_hinge_mps_resolved: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
	"""Compute per-edge costs for every D1 edge (one output row per edge_id)."""
	# Deterministic ordering
	edges = d1_edges.copy()
	edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	nodes = d1_nodes.set_index("node_id", drop=False)

	# Fast lookup for bank endpoints
	bank = bank_frames.copy()
	bank_key = bank[["tracklet_id", "frame_index"]].astype({"tracklet_id": "string", "frame_index": "int64"})
	bank_index = pd.MultiIndex.from_frame(bank_key, names=["tracklet_id", "frame_index"])
	bank.index = bank_index
	# Precompute available frames per tracklet for bounded nearest-frame lookup.
	# This avoids brittle exact-frame requirements at SOLO/GROUP boundaries.
	_tmp = bank_key.copy()
	frames_by_tid: Dict[str, np.ndarray] = (
		_tmp.groupby("tracklet_id")["frame_index"]
		.apply(lambda s: np.sort(s.unique()))
		.to_dict()
	)

	# ---------------------------------------------------------------------
	# Smart birth/death pricing support
	#
	# We compute four booleans per tracklet:
	#   - near_clip_start / near_clip_end
	#   - entrance_like / exit_like
	# and then apply additive penalties to births/deaths that are neither
	# near the clip boundary nor have border evidence.
	#
	# This mirrors the heuristics developed in D1/D3, but is re-computed here in D2
	# so pricing can evolve without perturbing graph topology.
	# ---------------------------------------------------------------------

	bank_df = bank.reset_index(drop=True)
	clip_start_frame = int(bank_df["frame_index"].min())
	clip_end_frame = int(bank_df["frame_index"].max())

	# Per-tracklet temporal extent (used for near_clip_start/end and for entrance/exit windows).
	tid_start_end = bank_df.groupby("tracklet_id")["frame_index"].agg(["min", "max"]).to_dict("index")

	def _node_id_to_tid(node_id: str):
		# Solo nodes: T:<tid> or T:<tid>:s0:...
		if node_id.startswith("T:"):
			rest = node_id[2:]
			return rest.split(":", 1)[0]
		# Group nodes: G:...:carrier=<tid>:...
		if node_id.startswith("G:") and "carrier=" in node_id:
			try:
				after = node_id.split("carrier=", 1)[1]
				return after.split(":", 1)[0]
			except Exception:
				return None
		return None

	have_uv_px = ("u_px" in bank_df.columns) and ("v_px" in bank_df.columns)
	if have_uv_px:
		u_min = float(bank_df["u_px"].min())
		u_max = float(bank_df["u_px"].max())
		v_min = float(bank_df["v_px"].min())
		v_max = float(bank_df["v_px"].max())
	else:
		u_min = u_max = v_min = v_max = float("nan")

	def _border_dist_px(u_px: float, v_px: float) -> float:
		# Distance to the nearest border using observed bounds.
		# Smaller is "closer to border".
		if not have_uv_px:
			return float("inf")
		return min(u_px - u_min, u_max - u_px, v_px - v_min, v_max - v_px)

	border_gate_enabled = bool(cfg.get("border_gate_enabled", True))
	border_margin_px = float(cfg.get("border_margin_px", 40))
	entrance_k_frames = int(cfg.get("entrance_k_frames", 10))
	exit_k_frames = int(cfg.get("exit_k_frames", 10))
	entrance_min_inward_px = float(cfg.get("entrance_min_inward_px", 20.0))
	exit_min_outward_px = float(cfg.get("exit_min_outward_px", 20.0))
	entrance_min_samples = int(cfg.get("entrance_min_samples", 3))
	exit_min_samples = int(cfg.get("exit_min_samples", 3))
	near_clip_start_frames = int(cfg.get("near_clip_start_frames", 15))
	near_clip_end_frames = int(cfg.get("near_clip_end_frames", 15))

	def _near_clip_start_tid(tid: str) -> bool:
		if tid not in tid_start_end:
			return False
		start_f = int(tid_start_end[tid]["min"])
		return (start_f - clip_start_frame) <= near_clip_start_frames

	def _near_clip_end_tid(tid: str) -> bool:
		if tid not in tid_start_end:
			return False
		end_f = int(tid_start_end[tid]["max"])
		return (clip_end_frame - end_f) <= near_clip_end_frames

	def _is_entrance_like_tid(tid: str) -> bool:
		if (not border_gate_enabled) or (not have_uv_px):
			return False
		if tid not in tid_start_end:
			return False
		start_f = int(tid_start_end[tid]["min"])
		win_end = start_f + max(entrance_k_frames, 0)
		df = bank_df[(bank_df["tracklet_id"] == tid) & (bank_df["frame_index"] >= start_f) & (bank_df["frame_index"] <= win_end)]
		if len(df) < entrance_min_samples:
			return False
		df = df.sort_values("frame_index")
		d0 = _border_dist_px(float(df.iloc[0]["u_px"]), float(df.iloc[0]["v_px"]))
		d1 = _border_dist_px(float(df.iloc[-1]["u_px"]), float(df.iloc[-1]["v_px"]))
		near_border = d0 <= border_margin_px
		moved_inward = (d1 - d0) >= entrance_min_inward_px
		return bool(near_border or moved_inward)

	def _is_exit_like_tid(tid: str) -> bool:
		if (not border_gate_enabled) or (not have_uv_px):
			return False
		if tid not in tid_start_end:
			return False
		end_f = int(tid_start_end[tid]["max"])
		win_start = end_f - max(exit_k_frames, 0)
		df = bank_df[(bank_df["tracklet_id"] == tid) & (bank_df["frame_index"] >= win_start) & (bank_df["frame_index"] <= end_f)]
		if len(df) < exit_min_samples:
			return False
		df = df.sort_values("frame_index")
		d0 = _border_dist_px(float(df.iloc[0]["u_px"]), float(df.iloc[0]["v_px"]))
		d1 = _border_dist_px(float(df.iloc[-1]["u_px"]), float(df.iloc[-1]["v_px"]))
		near_border = d1 <= border_margin_px
		moved_outward = (d0 - d1) >= exit_min_outward_px
		return bool(near_border or moved_outward)

	rows: List[Dict[str, Any]] = []

	dt_max_s = float(cfg.get("dt_max_s", 1.0))
	missing_geom_policy = cfg.get("missing_geom_policy", "disallow")
	if missing_geom_policy != "disallow":
		raise ValueError(f"D2 only supports missing_geom_policy='disallow' for POC (got {missing_geom_policy!r})")

	# Endpoint lookup policy: for CONTINUE edges we need endpoint geometry at node boundaries.
	# Exact boundary frames can be missing in tracklet_bank_frames (e.g., at SOLO/GROUP boundaries),
	# so we allow a bounded nearest-frame fallback within +/- endpoint_search_window_frames.
	if "endpoint_search_window_frames" not in cfg:
		raise ValueError(
			"D2 requires endpoint_search_window_frames in cfg (resolved in d2_run.py; do not default silently in costs.py)"
		)
	endpoint_search_window_frames = int(cfg["endpoint_search_window_frames"])
	if endpoint_search_window_frames < 0:
		raise ValueError(f"endpoint_search_window_frames must be >=0 (got {endpoint_search_window_frames})")

	endpoint_exact_hits = 0
	endpoint_fallback_hits = 0
	endpoint_misses = 0

	# weights
	w_time = float(cfg.get("w_time", 0.1))
	w_vreq = float(cfg.get("w_vreq", 1.0))
	base_env_cost = float(cfg.get("base_env_cost", 0.01))

	use_flags = bool(cfg.get("use_flags", True))
	w_flags = float(cfg.get("w_flags", 0.25))

	use_contact_rel = bool(cfg.get("use_contact_rel", True))
	contact_conf_floor = float(cfg.get("contact_conf_floor", 0.25))
	contact_rel_alpha = float(cfg.get("contact_rel_alpha", 0.35))

	def _pos_float(x: Any) -> float | None:
		"""Parse a non-negative float, returning None for nullish/invalid."""
		if _is_nullish(x):
			return None
		try:
			v = float(x)
		except Exception:
			return None
		if math.isnan(v):
			return None
		return max(0.0, v)

	# Positive-only merge/split coherence costs (preferred).
	coherent_merge_cost = _pos_float(cfg.get("coherent_merge_cost", None))
	incoherent_merge_cost = _pos_float(cfg.get("incoherent_merge_cost", None))
	coherent_split_cost = _pos_float(cfg.get("coherent_split_cost", None))
	incoherent_split_cost = _pos_float(cfg.get("incoherent_split_cost", None))

	# Back-compat: if new keys are absent, derive from legacy bonus/penalty keys.
	if coherent_merge_cost is None or incoherent_merge_cost is None:
		legacy_bonus = _pos_float(cfg.get("bonus_group_coherent", 0.0))
		legacy_penalty = _pos_float(cfg.get("penalty_group_incoherent", 0.0))
		# legacy bonus was previously applied as a negative reward; convert to a small positive cost
		# (default 0.0 if unspecified).
		try:
			legacy_bonus_raw = float(cfg.get("bonus_group_coherent", 0.0))
		except Exception:
			legacy_bonus_raw = 0.0
		coherent_merge_cost = max(0.0, -legacy_bonus_raw) if coherent_merge_cost is None else coherent_merge_cost
		incoherent_merge_cost = float(legacy_penalty or 0.0) if incoherent_merge_cost is None else incoherent_merge_cost

	# Split defaults mirror merge if unspecified.
	if coherent_split_cost is None:
		coherent_split_cost = float(coherent_merge_cost)
	if incoherent_split_cost is None:
		incoherent_split_cost = float(incoherent_merge_cost)

	birth_cost = float(cfg.get("birth_cost", 2.0))
	death_cost = float(cfg.get("death_cost", 2.0))
	birth_non_entrance_add = float(cfg.get("birth_non_entrance_add_cost", 8.0))
	death_non_exit_add = float(cfg.get("death_non_exit_add_cost", 8.0))
	merge_prior = float(cfg.get("merge_prior", 0.1))
	split_prior = float(cfg.get("split_prior", 0.1))
	reconnect_extra_env_cost = float(cfg.get("reconnect_extra_env_cost", 0.0))
	reconnect_w_z = float(cfg.get("reconnect_w_z", 1.0))
	reconnect_w_time = float(cfg.get("reconnect_w_time", 0.0))
	# Soft Option B: reconnect edges shadowed by a coherent MERGE/SPLIT chain.
	shadowed_reconnect_policy = str(cfg.get("shadowed_reconnect_policy", "disallow"))
	if shadowed_reconnect_policy not in ("disallow", "penalize", "allow"):
		raise ValueError(
			f"shadowed_reconnect_policy must be one of disallow|penalize|allow (got {shadowed_reconnect_policy!r})"
		)
	shadowed_reconnect_penalty = float(cfg.get("shadowed_reconnect_penalty", 10.0))
	if shadowed_reconnect_penalty < 0:
		raise ValueError(f"shadowed_reconnect_penalty must be >=0 (got {shadowed_reconnect_penalty!r})")
	reconnect_v_max_mps = float(cfg.get("reconnect_v_max_mps", v_hinge_mps_resolved))
	reconnect_w_speed = float(cfg.get("reconnect_w_speed", 1.0))
	reconnect_speed_power = float(cfg.get("reconnect_speed_power", 2.0))
	reconnect_dt_ref_s = float(cfg.get("reconnect_dt_ref_s", 0.75))
	reconnect_dt_power = float(cfg.get("reconnect_dt_power", 2.2))

	def _node_payload(node_id: str) -> Dict[str, Any]:
		"""Return parsed payload_json for a D1 node, or {} on failure.

		This helper is intentionally tolerant of missing columns so tests can
		use minimal D1 node tables that omit payload_json.
		"""
		try:
			n = nodes.loc[node_id]
		except KeyError:
			return {}
		raw = n.get("payload_json", None)
		try:
			return json.loads(str(raw)) if not _is_nullish(raw) else {}
		except Exception:
			return {}

	for _, e in edges.iterrows():
		edge_id = str(e["edge_id"])
		edge_type_raw = str(e["edge_type"])
		et = _parse_edge_type(edge_type_raw)
		u = str(e["u"])
		v = str(e["v"])
		payload_raw = e.get("payload_json", None)
		try:
			payload = json.loads(str(payload_raw)) if not _is_nullish(payload_raw) else {}
		except Exception:
			payload = {}
		is_reconnect = bool(payload.get("reconnect", False))
		is_shadowed_reconnect = bool(payload.get("shadowed_by_group_chain", False))

		is_allowed = True
		reasons: List[str] = []

		# Always-present term columns (0.0 default)
		term_env = base_env_cost
		term_time = 0.0
		term_vreq = 0.0
		term_missing_geom = 0.0
		term_flags = 0.0
		term_group_coherence = 0.0
		term_birth_prior = 0.0
		term_death_prior = 0.0
		term_merge_prior = 0.0
		term_split_prior = 0.0

		# scalar features (nullable by type)
		dt_frames = e.get("dt_frames", None)
		dt_s = None
		dist_m = None
		v_req_mps = None
		dist_norm = None
		contact_rel = None
		endpoint_flagged = False

		# dt gating (only if dt present); allow larger dt for reconnect edges
		if not _is_nullish(dt_frames):
			try:
				dt_frames_i = int(dt_frames)
				dt_s = float(dt_frames_i) / float(fps)
				if dt_s > dt_max_s and not is_reconnect:
					is_allowed = False
					reasons.append("dt_too_large")
			except Exception:
				# malformed dt -> disallow
				is_allowed = False
				reasons.append("dt_invalid")

		# endpoint lookups for kinematic features (only when needed)
		def _lookup_bank_row_near(*, tid: str, frame_i_req: int) -> Tuple[pd.Series | None, int | None, bool]:
			"""Lookup bank row for (tid, frame), with bounded nearest-frame fallback.

			Tie-breaks deterministically: prefer smallest |delta|; on ties prefer earlier frame.
			"""
			# Exact hit
			try:
				b = bank.loc[(tid, frame_i_req)]
				return b, frame_i_req, False
			except KeyError:
				pass

			if endpoint_search_window_frames == 0:
				return None, None, False

			frames = frames_by_tid.get(tid, None)
			if frames is None or len(frames) == 0:
				return None, None, False

			lo = frame_i_req - endpoint_search_window_frames
			hi = frame_i_req + endpoint_search_window_frames

			# Use searchsorted to find nearest candidates.
			j = int(np.searchsorted(frames, frame_i_req))
			candidates: List[int] = []
			if 0 <= j < len(frames):
				candidates.append(int(frames[j]))
			if 0 <= j - 1 < len(frames):
				candidates.append(int(frames[j - 1]))

			best_frame: int | None = None
			best_abs: int | None = None
			for fi in candidates:
				if fi < lo or fi > hi:
					continue
				absd = abs(fi - frame_i_req)
				if best_abs is None or absd < best_abs or (absd == best_abs and (best_frame is None or fi < best_frame)):
					best_abs = absd
					best_frame = fi

			if best_frame is None:
				return None, None, False

			try:
				b = bank.loc[(tid, int(best_frame))]
			except KeyError:
				# Should not happen if frames_by_tid is built from bank_key, but stay safe.
				return None, None, False
			return b, int(best_frame), True

		def _endpoint_features(node_id: str, which: str) -> Tuple[Tuple[float | None, float | None], float | None, bool]:
			nonlocal endpoint_exact_hits, endpoint_fallback_hits, endpoint_misses
			n = nodes.loc[node_id]  # raises KeyError if missing: hard fail (schema/graph bug)
			tid = n.get("base_tracklet_id", None)
			if _is_nullish(tid):
				return (None, None), None, False
			tid = str(tid)
			frame_key = "end_frame" if which == "u_end" else "start_frame"
			frame = n.get(frame_key, None)
			if _is_nullish(frame):
				return (None, None), None, False
			try:
				frame_i = int(frame)
			except Exception:
				return (None, None), None, False
			b, frame_i_used, used_fallback = _lookup_bank_row_near(tid=tid, frame_i_req=frame_i)
			if b is None:
				endpoint_misses += 1
				return (None, None), None, False
			if used_fallback:
				endpoint_fallback_hits += 1
			else:
				endpoint_exact_hits += 1
			x_eff, y_eff = _effective_xy(b)
			conf = b.get("contact_conf", None)
			conf_f = None
			if not _is_nullish(conf):
				try:
					conf_f = float(conf)
				except Exception:
					conf_f = None
			flagged = False
			if use_flags:
				flagged = bool(b.get("speed_is_implausible", False) or b.get("accel_is_implausible", False))
			return (x_eff, y_eff), conf_f, flagged

		# CONTINUE edges: kinematic cost (including reconnect edges marked in payload)
		if et == "CONTINUE":
			(uxy, uconf, uflag) = _endpoint_features(u, "u_end")
			(vxy, vconf, vflag) = _endpoint_features(v, "v_start")
			endpoint_flagged = bool(uflag or vflag)

			if uxy[0] is None or vxy[0] is None:
				is_allowed = False
				reasons.append("missing_geom")
				term_missing_geom = DISALLOWED_EDGE_COST
			else:
				dx = float(vxy[0]) - float(uxy[0])
				dy = float(vxy[1]) - float(uxy[1])
				dist_m = float(math.hypot(dx, dy))
				if dt_s is None or dt_s <= EPS:
					is_allowed = False
					reasons.append("dt_missing_or_zero")
					term_missing_geom = DISALLOWED_EDGE_COST
				else:
					v_req_mps = float(dist_m) / max(EPS, float(dt_s))
					dist_norm = float(dist_m) / max(EPS, float(v_cost_scale_mps_resolved) * float(dt_s))

					if is_reconnect:
						# Soft Option B: if this reconnect is shadowed by a coherent MERGE/SPLIT chain,
						# either disallow it outright or penalize it so it only wins when necessary.
						if is_shadowed_reconnect:
							if shadowed_reconnect_policy == "disallow":
								is_allowed = False
								reasons.append("shadowed_by_group_chain")
								# Mark as disallowed (handled by the unified not-allowed total_cost branch).
							elif shadowed_reconnect_policy == "penalize":
								term_env = float(term_env) + float(shadowed_reconnect_penalty)
							# "allow" => no-op

						# Reconnect edges (occlusion only): monotone, convex dt penalty + speed hinge.
						# Prefer D1 payload geometry to stay consistent with D1 endpoint selection.
						dt_s_eff = dt_s
						dist_eff = dist_m
						v_req_eff = v_req_mps
						try:
							p_dt_s = payload.get("dt_s", None)
							p_dist_m = payload.get("dist_m", None)
							p_v = payload.get("speed_mps", None)
							if not _is_nullish(p_dt_s):
								dt_s_eff = float(p_dt_s)
							if not _is_nullish(p_dist_m):
								dist_eff = float(p_dist_m)
							if not _is_nullish(p_v):
								v_req_eff = float(p_v)
						except Exception:
							# Fall back to bank-derived features.
							pass
						if dt_s_eff is None or dt_s_eff <= EPS or dist_eff is None:
							is_allowed = False
							reasons.append("reconnect_missing_or_invalid_geom")
							term_missing_geom = DISALLOWED_EDGE_COST
						else:
							# Compute v_req if not provided.
							if v_req_eff is None:
								v_req_eff = float(dist_eff) / max(EPS, float(dt_s_eff))
							if float(v_req_eff) > float(reconnect_v_max_mps):
								is_allowed = False
								reasons.append("reconnect_speed_too_high")
								term_missing_geom = DISALLOWED_EDGE_COST
							else:
								term_env = base_env_cost + reconnect_extra_env_cost
								# Convex time penalty: strictly increasing in dt, dominated for long gaps.
								dt_norm = float(dt_s_eff) / max(EPS, float(reconnect_dt_ref_s))
								term_time = reconnect_w_time * (dt_norm ** float(reconnect_dt_power))
								# Speed hinge around reconnect_v_max_mps (clear threshold semantics).
								zv = float(v_req_eff) / max(EPS, float(reconnect_v_max_mps))
								term_vreq = reconnect_w_speed * (max(0.0, zv - 1.0) ** float(reconnect_speed_power))
								# Keep scalar features consistent with the chosen geometry.
								dt_s = float(dt_s_eff)
								dist_m = float(dist_eff)
								v_req_mps = float(v_req_eff)
								dist_norm = float(dist_m) / max(EPS, float(reconnect_v_max_mps) * float(dt_s))
					else:
						term_time = w_time * math.log1p(float(dt_s))
						z = float(v_req_mps) / max(EPS, float(v_hinge_mps_resolved))
						term_vreq = w_vreq * _hinge2(z)

					# contact reliability scaling (gentle; applies to both normal and reconnect)
					if use_contact_rel:
						rel = None
						if uconf is not None and vconf is not None:
							rel = min(float(uconf), float(vconf))
						elif uconf is not None:
							rel = float(uconf)
						elif vconf is not None:
							rel = float(vconf)
						if rel is not None:
							contact_rel = max(0.0, min(1.0, rel))
							rel_eff = max(contact_rel, contact_conf_floor)
							scale = 1.0 + contact_rel_alpha * (1.0 - rel_eff)
							term_vreq *= scale

					if endpoint_flagged and use_flags:
						term_flags = w_flags

		# MERGE / SPLIT coherence (structural; dt typically null)
		if et in ("MERGE", "SPLIT"):
			# Coherence depends on touching GROUP nodes and labeled participants.
			def _is_group(node_id: str) -> bool:
				seg = nodes.loc[node_id].get("segment_type", None)
				return isinstance(seg, str) and seg.upper() == "GROUP"

			def _coherence_case() -> str:
				u_is_group = _is_group(u)
				v_is_group = _is_group(v)
				# Expect merge: SOLO->GROUP; split: GROUP->SOLO
				if et == "MERGE" and v_is_group:
					grp = nodes.loc[v]
					disappearing = grp.get("disappearing_tracklet_id", None)
					src_tid = nodes.loc[u].get("base_tracklet_id", None)
					if disappearing is not None and src_tid is not None and str(src_tid) == str(disappearing):
						return "coherent"
					return "incoherent"
				if et == "SPLIT" and u_is_group:
					grp = nodes.loc[u]
					new_tid = grp.get("new_tracklet_id", None)
					carrier_tid = grp.get("carrier_tracklet_id", None)
					dst_tid = nodes.loc[v].get("base_tracklet_id", None)
					if dst_tid is None:
						return "incoherent"
					dst_tid_s = str(dst_tid)
					if (new_tid is not None and dst_tid_s == str(new_tid)) or (
						carrier_tid is not None and dst_tid_s == str(carrier_tid)
					):
						return "coherent"
					return "incoherent"
				# touches group but not in expected direction
				if u_is_group or v_is_group:
					return "incoherent"
				return "na"

			cc = _coherence_case()
			if cc == "coherent":
				# Positive-only: coherent structure is slightly costly (never rewarded).
				if et == "MERGE":
					term_group_coherence = float(coherent_merge_cost)
				else:
					term_group_coherence = float(coherent_split_cost)
			elif cc == "incoherent":
				# Positive-only: incoherent structure is expensive.
				if et == "MERGE":
					term_group_coherence = float(incoherent_merge_cost)
				else:
					term_group_coherence = float(incoherent_split_cost)
			else:
				term_group_coherence = 0.0

			if et == "MERGE":
				term_merge_prior = merge_prior
			else:
				term_split_prior = split_prior

			# Geometry-aware MERGE/SPLIT cost using D1 thresholds as scale.
			# We reuse w_vreq so that distance penalties live on a similar
			# scale to CONTINUE kinematic costs, without adding new knobs.
			try:
				alpha_ms = 1.0
				w_ms_geom = alpha_ms * w_vreq
				if w_ms_geom > 0.0:
					if et == "MERGE":
						grp_payload = _node_payload(v)
						md = grp_payload.get("merge_dist_m", None)
						thr = cfg.get("merge_dist_m", None)
					elif et == "SPLIT":
						grp_payload = _node_payload(u)
						md = grp_payload.get("split_dist_m", None)
						thr = cfg.get("split_dist_m", None)
					else:
						md = None
						thr = None
					if not _is_nullish(md) and not _is_nullish(thr):
						try:
							dm = float(md)
							T = max(0.1, float(thr))
							# Expose the distance feature for diagnostics.
							if dist_m is None:
								dist_m = dm
							z = dm / max(EPS, T)
							term_vreq += w_ms_geom * (z ** 2)
						except Exception:
							pass
			except Exception:
				# Geometry shaping must never compromise core MERGE/SPLIT
				# coherence logic; fail closed by ignoring the extra term.
				pass

		# BIRTH / DEATH priors
		if et == "BIRTH":
			term_birth_prior = birth_cost
			# Smart birth pricing: penalize births that are neither near the clip start
			# nor have entrance-like border evidence.
			dst_tid = _node_id_to_tid(str(v))
			if dst_tid is not None:
				near_start = _near_clip_start_tid(dst_tid)
				entrance_like = _is_entrance_like_tid(dst_tid)
				if not (near_start or entrance_like):
					term_birth_prior += birth_non_entrance_add
		if et == "DEATH":
			term_death_prior = death_cost
			# Smart death pricing: penalize deaths that are neither near the clip end
			# nor have exit-like border evidence.
			src_tid = _node_id_to_tid(str(u))
			if src_tid is not None:
				near_end = _near_clip_end_tid(src_tid)
				exit_like = _is_exit_like_tid(src_tid)
				if not (near_end or exit_like):
					term_death_prior += death_non_exit_add

		# if disallowed, set a finite sentinel total_cost for safety
		if not is_allowed:
			total_cost = DISALLOWED_EDGE_COST
		else:
			total_cost = (
				term_env
				+ term_time
				+ term_vreq
				+ term_missing_geom
				+ term_flags
				+ term_group_coherence
				+ term_birth_prior
				+ term_death_prior
				+ term_merge_prior
				+ term_split_prior
			)

		if _is_nullish(dt_frames):
			dt_frames_out = None
		else:
			try:
				dt_frames_out = int(dt_frames)
			except Exception:
				dt_frames_out = None

		# Canonicalize disallow reasons for determinism.
		# NOTE: json.dumps(sort_keys=True) does not affect list ordering.
		reasons_canon = sorted(set(reasons))

		rows.append(
			{
				"edge_id": edge_id,
				"edge_type": edge_type_raw,
				"src_node_id": u,
				"dst_node_id": v,
				"is_allowed": bool(is_allowed),
				"disallow_reasons_json": json.dumps(reasons_canon),
				"dt_frames": dt_frames_out,
				"dt_s": dt_s,
				"dist_m": dist_m,
				"v_req_mps": v_req_mps,
				"dist_norm": dist_norm,
				"contact_rel": contact_rel,
				"endpoint_flagged": bool(endpoint_flagged),
				"term_env": float(term_env),
				"term_time": float(term_time),
				"term_vreq": float(term_vreq),
				"term_missing_geom": float(term_missing_geom),
				"term_flags": float(term_flags),
				"term_group_coherence": float(term_group_coherence),
				"term_birth_prior": float(term_birth_prior),
				"term_death_prior": float(term_death_prior),
				"term_merge_prior": float(term_merge_prior),
				"term_split_prior": float(term_split_prior),
				"total_cost": float(total_cost),
			}
		)

	out_df = pd.DataFrame(rows)
	if "dt_frames" in out_df.columns:
		# Use nullable Int64 so schema family "int" is preserved while allowing NA.
		out_df["dt_frames"] = out_df["dt_frames"].astype("Int64")
	endpoint_stats: Dict[str, Any] = {
		"endpoint_search_window_frames": int(endpoint_search_window_frames),
		"endpoint_exact_hits": int(endpoint_exact_hits),
		"endpoint_fallback_hits": int(endpoint_fallback_hits),
		"endpoint_misses": int(endpoint_misses),
		"endpoint_requests_total": int(endpoint_exact_hits + endpoint_fallback_hits + endpoint_misses),
	}
	return out_df, endpoint_stats

