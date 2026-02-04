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
) -> pd.DataFrame:
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

	rows: List[Dict[str, Any]] = []

	dt_max_s = float(cfg.get("dt_max_s", 1.0))
	missing_geom_policy = cfg.get("missing_geom_policy", "disallow")
	if missing_geom_policy != "disallow":
		raise ValueError(f"D2 only supports missing_geom_policy='disallow' for POC (got {missing_geom_policy!r})")

	# weights
	w_time = float(cfg.get("w_time", 0.1))
	w_vreq = float(cfg.get("w_vreq", 1.0))
	base_env_cost = float(cfg.get("base_env_cost", 0.01))

	use_flags = bool(cfg.get("use_flags", True))
	w_flags = float(cfg.get("w_flags", 0.25))

	use_contact_rel = bool(cfg.get("use_contact_rel", True))
	contact_conf_floor = float(cfg.get("contact_conf_floor", 0.25))
	contact_rel_alpha = float(cfg.get("contact_rel_alpha", 0.35))

	bonus_group_coherent = float(cfg.get("bonus_group_coherent", 0.5))
	penalty_group_incoherent = float(cfg.get("penalty_group_incoherent", 0.5))

	birth_cost = float(cfg.get("birth_cost", 2.0))
	death_cost = float(cfg.get("death_cost", 2.0))
	merge_prior = float(cfg.get("merge_prior", 0.1))
	split_prior = float(cfg.get("split_prior", 0.1))

	for _, e in edges.iterrows():
		edge_id = str(e["edge_id"])
		edge_type_raw = str(e["edge_type"])
		et = _parse_edge_type(edge_type_raw)
		u = str(e["u"])
		v = str(e["v"])

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

		# dt gating (only if dt present)
		if not _is_nullish(dt_frames):
			try:
				dt_frames_i = int(dt_frames)
				dt_s = float(dt_frames_i) / float(fps)
				if dt_s > dt_max_s:
					is_allowed = False
					reasons.append("dt_too_large")
			except Exception:
				# malformed dt -> disallow
				is_allowed = False
				reasons.append("dt_invalid")

		# endpoint lookups for kinematic features (only when needed)
		def _endpoint_features(node_id: str, which: str) -> Tuple[Tuple[float | None, float | None], float | None, bool]:
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
			try:
				b = bank.loc[(tid, frame_i)]
			except KeyError:
				return (None, None), None, False
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

		# CONTINUE edges: kinematic cost
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

					term_time = w_time * math.log1p(float(dt_s))
					z = float(v_req_mps) / max(EPS, float(v_hinge_mps_resolved))
					term_vreq = w_vreq * _hinge2(z)

					# contact reliability scaling (gentle)
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
				term_group_coherence = -bonus_group_coherent
			elif cc == "incoherent":
				term_group_coherence = penalty_group_incoherent

			if et == "MERGE":
				term_merge_prior = merge_prior
			else:
				term_split_prior = split_prior

		# BIRTH / DEATH priors
		if et == "BIRTH":
			term_birth_prior = birth_cost
		if et == "DEATH":
			term_death_prior = death_cost

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
	return out_df

