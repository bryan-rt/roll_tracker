"""Stage D0 — Tracklet bank creation (master tables for downstream D stages).

Responsibilities in Checkpoint 1:
	- Create per-frame and per-tracklet bank tables under stage_D/
	- Preserve Stage C identity_hints.jsonl records losslessly (do NOT collapse pings)
	- Write a minimal deterministic stage_D/audit.jsonl

No geometry repair is performed in this checkpoint; bank tables are pass-through
from Stage A with identity hints preserved losslessly for downstream stages.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _now_ms() -> int:
	return int(time.time() * 1000)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
	if not path.exists():
		return []
	lines = path.read_text(encoding="utf-8").splitlines()
	return [json.loads(line) for line in lines if line.strip()]


def _write_audit_event(audit_path: Path, event: Dict[str, Any]) -> None:
	audit_path.parent.mkdir(parents=True, exist_ok=True)
	with audit_path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass(frozen=True)
class HintAgg:
	"""Lossless identity hint aggregation (no time-collapsing).

	We intentionally do NOT compute tracklet-global labels from time-local pings.
	Downstream stages (D1/D2/D3) are responsible for binding/propagating
	identity evidence at the node/time level.
	
	We keep the historical columns for backward compatibility, but they are left
	unset (None).
	"""
	identity_hints_json: Optional[str]
	must_link_anchor_key: Optional[str]
	must_link_confidence: Optional[float]
	cannot_link_anchor_keys_json: Optional[str]


def _aggregate_identity_hints(records: List[Dict[str, Any]]) -> Dict[str, HintAgg]:
	"""Aggregate identity_hints.jsonl records into tracklet-level bank fields.

	Deterministic policy:
	  - identity_hints_json: JSON list of all records for the tracklet, stably sorted.
	 
	IMPORTANT:
	  - We do NOT compute a "best" must_link label or cannot_link list here.
	    Collapsing time-local hints into tracklet-global labels undermines the goal
	    of using sparse pings to constrain ILP chains.
	"""
	by_tid: Dict[str, List[Dict[str, Any]]] = {}
	for r in records:
		tid = str(r.get("tracklet_id", ""))
		if not tid:
			continue
		by_tid.setdefault(tid, []).append(r)

	out: Dict[str, HintAgg] = {}
	for tid, recs in by_tid.items():
		# stable sort for the full record list
		def _k(rr: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
			constraint = str(rr.get("constraint", ""))
			conf = rr.get("confidence", None)
			try:
				conf_f = float(conf) if conf is not None else float("nan")
			except Exception:
				conf_f = float("nan")
			anchor_key = str(rr.get("anchor_key", ""))
			# include evidence as stable json string for tie-break
			ev = rr.get("evidence", None)
			ev_s = json.dumps(ev, sort_keys=True) if isinstance(ev, (dict, list)) else str(ev)
			return (constraint, -conf_f if conf_f == conf_f else 0.0, anchor_key, ev_s)

		recs_sorted = sorted(recs, key=_k)
		identity_hints_json = json.dumps(recs_sorted, sort_keys=True)

		# Historical summary columns are intentionally left unset in this checkpoint.
		must_best_key: Optional[str] = None
		must_best_conf: Optional[float] = None
		cannot_json: Optional[str] = None

		out[tid] = HintAgg(
			identity_hints_json=identity_hints_json,
			must_link_anchor_key=None,
			must_link_confidence=None,
			cannot_link_anchor_keys_json=None,
		)
	return out


def _rolling_median(s: pd.Series, window: int) -> pd.Series:
	w = max(int(window), 1)
	return s.rolling(window=w, min_periods=1).median()


def _compute_occ_ratios(tf: pd.DataFrame, *, onset_window: int) -> pd.DataFrame:
	"""Compute per-frame occlusion helpers used by D0.

	We compute a few absolute bbox delta helpers (dy2/dy1/dh) that can gate onset.
	Primary evidence + span logic is handled in `_detect_occlusion_linker2`.

	IMPORTANT: preserve the original index so we can write results back into the
	full tracklet_frames table using `.loc[sub2.index, ...]`.
	"""
	out = tf.copy()
	# Preserve original index while ensuring deterministic order
	out = out.sort_values(["frame_index"], kind="mergesort")

	# absolute bbox deltas (pixel space)
	out["_occ_dy2_px"] = out["bbox_bottom"].diff().abs()
	out["_occ_dy1_px"] = out["bbox_top"].diff().abs()
	out["_occ_dh_px"] = out["bbox_h"].diff().abs()

	# initialize expected output columns
	out["occ_r_bottom"] = 0.0
	out["occ_r_height"] = 0.0
	out["occ_span_active"] = False
	return out


def _find_spans(rb: np.ndarray, rh: np.ndarray, dy2: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
	"""Deprecated helper (kept for backwards compat).

	Span detection for D0 is now implemented with linker_2 semantics in
	`_detect_occlusion_linker2`.
	"""
	min_b = float(cfg.get("min_bottom_frac", 0.15))
	min_h = float(cfg.get("min_height_frac", 0.10))
	rec_b = float(cfg.get("recover_bottom_frac", 0.10))
	rec_h = float(cfg.get("recover_height_frac", 0.08))
	onset_min_frames = int(cfg.get("onset_min_frames", 1))
	recover_min_frames = int(cfg.get("recover_min_frames", 3))
	merge_gap = int(cfg.get("merge_gap_frames", 2))
	min_window = int(cfg.get("min_window_frames", 2))
	max_span = cfg.get("max_span_frames", None)

	onset = (rb >= min_b) & (rh >= min_h)
	still = (rb >= min_b) | (rh >= min_h)
	recover = (rb <= rec_b) & (rh <= rec_h)

	spans: List[Tuple[int, int]] = []
	i = 0
	n = len(rb)
	while i < n:
		if not onset[i]:
			i += 1
			continue
		j = i
		while j < n and onset[j]:
			j += 1
		if (j - i) < onset_min_frames:
			i = j
			continue
		start = i
		k = j
		rec_run = 0
		while k < n:
			if still[k]:
				rec_run = 0
				k += 1
				continue
			if recover[k]:
				rec_run += 1
			else:
				rec_run = 0
			k += 1
			if rec_run >= recover_min_frames:
				break
		end = min(n - 1, k - 1)
		if max_span is not None:
			end = min(end, start + int(max_span) - 1)
		if (end - start + 1) >= min_window:
			spans.append((start, end))
		i = end + 1

	if merge_gap > 0 and spans:
		merged: List[Tuple[int, int]] = []
		cs, ce = spans[0]
		for s, e in spans[1:]:
			if s - ce - 1 <= merge_gap:
				ce = e
			else:
				merged.append((cs, ce))
				cs, ce = s, e
		merged.append((cs, ce))
		spans = merged

	return spans


def _detect_occlusion_linker2(
	sub: pd.DataFrame,
	occ_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
	"""Compute linker_2-style occlusion evidence + spans for one tracklet.

	Returns:
	- sub2 with occ_r_bottom, occ_r_height, occ_span_active
	- spans as (start_idx, end_idx) in sub2 positional indices (inclusive)

	Notes:
	- baselines are trailing medians over onset_window
	- once a span starts, baselines are frozen until recovery
	- recovery tail frames are INCLUDED in the span (recall-first)
	"""
	ow = int(occ_cfg.get("onset_window", 5) or 5)
	min_b = float(occ_cfg.get("min_bottom_frac", 0.15))
	min_h = float(occ_cfg.get("min_height_frac", 0.10))
	omf = int(occ_cfg.get("onset_min_frames", 1) or 1)
	omf = max(1, min(omf, ow))

	rec_b = float(occ_cfg.get("recover_bottom_frac", 0.10))
	rec_h = float(occ_cfg.get("recover_height_frac", 0.08))
	rmin = int(occ_cfg.get("recover_min_frames", 3) or 3)
	rmin = max(1, rmin)

	merge_gap = int(occ_cfg.get("merge_gap_frames", 2) or 0)
	min_window = int(occ_cfg.get("min_window_frames", 2) or 1)
	max_span = occ_cfg.get("max_span_frames", None)

	# Default dy2 gate matches config model default; disables only if explicitly set.
	dy2_px_min = float(occ_cfg.get("dy2_px_min", 3.0) or 0.0)
	gate_onset = bool(occ_cfg.get("gate_onset_with_dy2", True))

	sub2 = _compute_occ_ratios(sub, onset_window=ow)

	# ensure float arrays for scan
	y2 = sub2["bbox_bottom"].astype("float64").to_numpy()
	h = sub2["bbox_h"].astype("float64").to_numpy()
	dy2 = sub2["_occ_dy2_px"].fillna(0.0).astype("float64").to_numpy()

	n = len(sub2)
	rb_out = np.zeros(n, dtype="float64")
	rh_out = np.zeros(n, dtype="float64")
	active = np.zeros(n, dtype=bool)

	baseline_bottom_q: deque = deque(maxlen=ow)
	baseline_height_q: deque = deque(maxlen=ow)
	recent_flags: deque = deque(maxlen=ow)

	in_occ = False
	start_i: Optional[int] = None
	occ_b0: Optional[float] = None
	occ_h0: Optional[float] = None
	recover_streak = 0
	last_occ_i: Optional[int] = None

	def _median(q: deque) -> Optional[float]:
		if not q:
			return None
		xs = sorted(q)
		m = xs[len(xs) // 2]
		if len(xs) % 2 == 0:
			m = 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
		return float(m)

	def _current_baseline() -> Tuple[Optional[float], Optional[float]]:
		b0 = _median(baseline_bottom_q)
		h0 = _median(baseline_height_q)
		if b0 is None or h0 is None:
			return None, None
		if h0 <= 0.0:
			return b0, None
		return b0, h0

	spans: List[Tuple[int, int]] = []

	for i in range(n):
		b = float(y2[i]) if np.isfinite(y2[i]) else None
		hh = float(h[i]) if np.isfinite(h[i]) else None
		b0, h0 = _current_baseline()

		# invalid bbox/height handling
		if b is None or hh is None or hh <= 0.0 or b0 is None or h0 is None or h0 <= 0.0:
			if (not in_occ) and (b is not None) and (hh is not None) and (hh > 0.0):
				baseline_bottom_q.append(b)
				baseline_height_q.append(hh)
			elif in_occ:
				recover_streak = 0
				last_occ_i = i
				active[i] = True
			continue

		# choose baseline: frozen during occlusion
		if in_occ and (occ_b0 is not None) and (occ_h0 is not None):
			base_b, base_h = occ_b0, occ_h0
		else:
			base_b, base_h = b0, h0

		if base_h is None or base_h <= 0.0:
			if in_occ:
				recover_streak = 0
				last_occ_i = i
				active[i] = True
			continue

		rb = (base_b - b) / base_h
		rh0 = (base_h - hh) / base_h
		if rb < 0.0:
			rb = 0.0
		if rh0 < 0.0:
			rh0 = 0.0
		rb_out[i] = float(rb)
		rh_out[i] = float(rh0)

		if not in_occ:
			is_candidate = (rb >= min_b and rh0 >= min_h)
			if gate_onset:
				is_candidate = bool(is_candidate and (float(dy2[i]) >= dy2_px_min))
			recent_flags.append(bool(is_candidate))
			if len(recent_flags) == ow and sum(1 for x in recent_flags if x) >= omf:
				first_off = None
				for k in range(ow):
					if recent_flags[k]:
						first_off = k
						break
				if first_off is None:
					continue
				start_i = max(0, (i - ow + 1) + int(first_off))
				in_occ = True
				recover_streak = 0
				occ_b0, occ_h0 = b0, h0
				last_occ_i = i
				active[i] = True
			else:
				baseline_bottom_q.append(b)
				baseline_height_q.append(hh)
			continue

		# in occlusion
		still_occ = (rb >= min_b) or (rh0 >= min_h)
		recovered = (rb <= rec_b) and (rh0 <= rec_h)

		if still_occ:
			last_occ_i = i
			recover_streak = 0
			active[i] = True
			continue

		if recovered:
			recover_streak += 1
			# recall-first: keep recovery frames active AND include them in span tail
			last_occ_i = i
			active[i] = True
			if recover_streak >= rmin:
				if start_i is not None and last_occ_i is not None:
					spans.append((start_i, last_occ_i))
				in_occ = False
				start_i = None
				occ_b0 = None
				occ_h0 = None
				recover_streak = 0
				recent_flags.clear()
			continue

		# ambiguous frame => treat as still occluded
		last_occ_i = i
		recover_streak = 0
		active[i] = True

	if in_occ and start_i is not None and last_occ_i is not None:
		spans.append((start_i, last_occ_i))

	# enforce min_window and optional hard cap
	filtered: List[Tuple[int, int]] = []
	for s, e in spans:
		if (e - s + 1) < min_window:
			continue
		if max_span is not None and int(max_span) > 0:
			e = min(e, s + int(max_span) - 1)
		filtered.append((int(s), int(e)))
	spans = filtered

	if merge_gap > 0 and spans:
		merged: List[Tuple[int, int]] = []
		cs, ce = spans[0]
		for s, e in spans[1:]:
			if s - ce - 1 <= merge_gap:
				ce = max(ce, e)
			else:
				merged.append((cs, ce))
				cs, ce = s, e
		merged.append((cs, ce))
		spans = merged

	# materialize framewise outputs
	for s, e in spans:
		active[s : e + 1] = True

	sub2["occ_r_bottom"] = rb_out
	sub2["occ_r_height"] = rh_out
	sub2["occ_span_active"] = active
	return sub2, spans


def _compute_nn_and_density(
	all_pos: pd.DataFrame,
	*,
	tid: str,
	frame_index: int,
	x: float,
	y: float,
	context_radius_m: Optional[float],
) -> Tuple[Optional[float], Optional[int]]:
	at = all_pos[(all_pos["frame_index"] == frame_index) & (all_pos["tracklet_id"] != tid)]
	at = at.dropna(subset=["x", "y"])
	if at.empty:
		return None, (0 if context_radius_m is not None else None)
	dx = (at["x"].astype("float64") - float(x)).to_numpy()
	dy = (at["y"].astype("float64") - float(y)).to_numpy()
	d = np.sqrt(dx * dx + dy * dy)
	nn = float(np.min(d)) if d.size else None
	if context_radius_m is None:
		return nn, None
	cnt = int(np.sum(d <= float(context_radius_m)))
	return nn, cnt


def _get_d0_cfg(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
	# config is the fully merged dict (post overlays)
	stage_d: Dict[str, Any] = {}
	if isinstance(config.get("stages"), dict) and isinstance(config["stages"].get("stage_D"), dict):
		stage_d = config["stages"]["stage_D"]
	elif isinstance(config.get("stage_D"), dict):
		# runtime_config compatibility
		stage_d = config["stage_D"]

	d0 = stage_d.get("d0", {}) if isinstance(stage_d, dict) else {}
	occ = d0.get("occlusion_repair", {}) if isinstance(d0, dict) else {}
	ctx = d0.get("global_context", {}) if isinstance(d0, dict) else {}
	kin = d0.get("kinematics", {}) if isinstance(d0, dict) else {}
	return (
		(occ if isinstance(occ, dict) else {}),
		(ctx if isinstance(ctx, dict) else {}),
		(kin if isinstance(kin, dict) else {}),
	)


def _apply_cp3_kinematics(
	tf: pd.DataFrame, *, fps: float, kin_cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
	"""Checkpoint 3 (CP3): recompute dt-aware velocities from *effective* world coords and flag implausible kinematics.

	- Uses repaired coords where CP2 marked `is_repaired`; otherwise uses original x_m/y_m.
	- Does NOT clamp or suppress; emits velocities + flags only.
	- Tracklet-local; no cross-tracklet logic.
	"""
	if not (fps > 0.0):
		raise ValueError(f"D0 CP3 requires fps > 0, got {fps!r}")

	enabled = bool(kin_cfg.get("enabled", True))
	if not enabled:
		return tf, {"enabled": False}

	v_max = float(kin_cfg.get("v_max_mps", 8.0))
	a_max = float(kin_cfg.get("a_max_mps2", 12.0))

	# Ensure columns exist with deterministic defaults.
	for c in ["vx_mps_k", "vy_mps_k", "speed_mps_k", "accel_mps2_k"]:
		if c not in tf.columns:
			tf[c] = np.nan
			tf[c] = tf[c].astype("float64")
	for c in ["speed_is_implausible", "accel_is_implausible"]:
		if c not in tf.columns:
			tf[c] = False
			tf[c] = tf[c].astype("bool")

	# Fast path: if no rows, return early.
	if len(tf) == 0:
		return tf, {
			"enabled": True,
			"n_rows": 0,
			"n_tracklets": 0,
			"n_speed_flagged": 0,
			"n_accel_flagged": 0,
			"n_bad_df_steps": 0,
			"max_speed_mps_k": float("nan"),
			"max_accel_mps2_k": float("nan"),
		}

	# Compute per-tracklet in deterministic order.
	# We compute and write values into tf via iloc positions.
	if "tracklet_id" not in tf.columns or "frame_index" not in tf.columns:
		raise ValueError("D0 CP3 requires tf to include tracklet_id and frame_index")

	# Sorting index (stable, deterministic).
	sort_cols = ["tracklet_id", "frame_index"]
	if "detection_id" in tf.columns:
		sort_cols.append("detection_id")
	tf = tf.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

	# Required CP2 columns (fallback to originals if missing, but CP2 should have created them).
	if "is_repaired" not in tf.columns:
		tf["is_repaired"] = False
		tf["is_repaired"] = tf["is_repaired"].astype("bool")

	# Pre-allocate counters.
	n_bad_df_steps = 0

	# Group indices per tracklet.
	# We avoid pandas groupby-apply to keep types stable and runtime reasonable.
	tids = tf["tracklet_id"].astype(str).to_numpy()
	frames = tf["frame_index"].to_numpy()

	# Effective coords per-row.
	# Note: treat non-finite as missing; this will propagate to NaNs in velocities.
	x_m = tf.get("x_m")
	y_m = tf.get("y_m")
	if x_m is None or y_m is None:
		raise ValueError("D0 CP3 requires tf to include x_m and y_m (world coords)")
	x_rep = tf.get("x_m_repaired") if "x_m_repaired" in tf.columns else None
	y_rep = tf.get("y_m_repaired") if "y_m_repaired" in tf.columns else None
	is_rep = tf["is_repaired"].to_numpy(dtype=bool)

	x_eff = (x_rep.to_numpy() if x_rep is not None else x_m.to_numpy()).astype("float64").copy()
	y_eff = (y_rep.to_numpy() if y_rep is not None else y_m.to_numpy()).astype("float64").copy()
	# Where not repaired, use original coords if present.
	x_orig = x_m.to_numpy(dtype="float64")
	y_orig = y_m.to_numpy(dtype="float64")
	x_eff[~is_rep] = x_orig[~is_rep]
	y_eff[~is_rep] = y_orig[~is_rep]

	vx = tf["vx_mps_k"].to_numpy(dtype="float64", copy=True)
	vy = tf["vy_mps_k"].to_numpy(dtype="float64", copy=True)
	speed = tf["speed_mps_k"].to_numpy(dtype="float64", copy=True)
	accel = tf["accel_mps2_k"].to_numpy(dtype="float64", copy=True)

	# Initialize outputs to NaN/False (in case tf came with junk values).
	vx[:] = np.nan
	vy[:] = np.nan
	speed[:] = np.nan
	accel[:] = np.nan
	speed_flag = np.zeros(len(tf), dtype=bool)
	accel_flag = np.zeros(len(tf), dtype=bool)

	# Walk contiguous blocks of the same tracklet_id.
	start = 0
	N = len(tf)
	while start < N:
		tid = tids[start]
		end = start + 1
		while end < N and tids[end] == tid:
			end += 1

		# Compute stepwise velocities for this block [start, end).
		for i in range(start + 1, end):
			df = int(frames[i] - frames[i - 1])
			if df <= 0:
				n_bad_df_steps += 1
				continue
			dt_s = df / fps
			# Require finite endpoints.
			if not (
				np.isfinite(x_eff[i])
				and np.isfinite(y_eff[i])
				and np.isfinite(x_eff[i - 1])
				and np.isfinite(y_eff[i - 1])
			):
				continue
			dx = float(x_eff[i] - x_eff[i - 1])
			dy = float(y_eff[i] - y_eff[i - 1])
			vx_i = dx / dt_s
			vy_i = dy / dt_s
			vx[i] = vx_i
			vy[i] = vy_i
			speed_i = float(np.hypot(vx_i, vy_i))
			speed[i] = speed_i
			if speed_i > v_max:
				speed_flag[i] = True
			# Accel requires previous speed.
			if i - 1 >= start and np.isfinite(speed[i - 1]):
				acc_i = abs(speed_i - float(speed[i - 1])) / dt_s
				accel[i] = float(acc_i)
				if acc_i > a_max:
					accel_flag[i] = True

		start = end

	# Write back.
	tf["vx_mps_k"] = vx
	tf["vy_mps_k"] = vy
	tf["speed_mps_k"] = speed
	tf["accel_mps2_k"] = accel
	tf["speed_is_implausible"] = speed_flag.astype("bool")
	tf["accel_is_implausible"] = accel_flag.astype("bool")

	# Summary
	return tf, {
		"enabled": True,
		"v_max_mps": float(v_max),
		"a_max_mps2": float(a_max),
		"n_rows": int(N),
		"n_tracklets": int(len(pd.unique(tf["tracklet_id"]))),
		"n_speed_flagged": int(np.sum(speed_flag)),
		"n_accel_flagged": int(np.sum(accel_flag)),
		"n_bad_df_steps": int(n_bad_df_steps),
		"max_speed_mps_k": float(np.nanmax(speed)) if np.any(np.isfinite(speed)) else float("nan"),
		"max_accel_mps2_k": float(np.nanmax(accel)) if np.any(np.isfinite(accel)) else float("nan"),
	}


def run_d0(*, config: Dict[str, Any], layout: Any, manifest: Any) -> None:
	"""Write stage_D bank tables (Checkpoint 1) + occlusion repair evidence (Checkpoint 2)."""
	# Read Stage A base tables
	tf_path = Path(layout.tracklet_frames_parquet())
	ts_path = Path(layout.tracklet_summaries_parquet())
	if not tf_path.exists() or not ts_path.exists():
		raise FileNotFoundError("Stage D0 requires Stage A tracklet_frames.parquet and tracklet_summaries.parquet")

	tf = pd.read_parquet(tf_path)
	ts = pd.read_parquet(ts_path)

	# Ensure x_m_repaired and y_m_repaired exist on tf regardless of whether tf is
	# empty. D1 requires both columns unconditionally. Without this, empty-tracklet
	# clips (zero Stage A detections) skip the occlusion block and the columns are
	# never added, causing D1 to raise ValueError.
	if "x_m_repaired" not in tf.columns:
		tf["x_m_repaired"] = pd.to_numeric(
			tf["x_m"] if "x_m" in tf.columns else pd.Series(dtype="float64"),
			errors="coerce"
		).astype("float64")
	if "y_m_repaired" not in tf.columns:
		tf["y_m_repaired"] = pd.to_numeric(
			tf["y_m"] if "y_m" in tf.columns else pd.Series(dtype="float64"),
			errors="coerce"
		).astype("float64")

	kin_summary: Dict[str, Any] = {"enabled": False}
	occ_cfg, ctx_cfg, kin_cfg = _get_d0_cfg(config)
	enable_norm = bool(occ_cfg.get("enable_normalized", True))
	onset_window = int(occ_cfg.get("onset_window", 5) or 5)
	context_radius_m = ctx_cfg.get("context_radius_m", None)
	candidate_radius_m = ctx_cfg.get("candidate_radius_m", None)
	context_radius_f: Optional[float] = float(context_radius_m) if context_radius_m is not None else None
	candidate_radius_f: Optional[float] = float(candidate_radius_m) if candidate_radius_m is not None else None

	# Bank frames: pass-through + deterministic sort
	if not tf.empty:
		sort_cols = [c for c in ["tracklet_id", "frame_index", "detection_id"] if c in tf.columns]
		if sort_cols:
			tf = tf.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

	# Bank summaries: pass-through + deterministic sort + join identity hints
	if not ts.empty:
		if "tracklet_id" in ts.columns:
			ts = ts.sort_values(["tracklet_id"], kind="mergesort").reset_index(drop=True)

	# Join identity hints (tracklet-level)
	ih_path = Path(layout.identity_hints_jsonl())
	ih_records = _read_jsonl(ih_path)
	agg = _aggregate_identity_hints(ih_records) if ih_records else {}

	# Ensure hint columns exist with dtype-stable defaults (avoid row-wise apply upcasting)
	if "identity_hints_json" not in ts.columns:
		ts["identity_hints_json"] = pd.Series([None] * len(ts), dtype="object")
	if "must_link_anchor_key" not in ts.columns:
		ts["must_link_anchor_key"] = pd.Series([None] * len(ts), dtype="object")
	if "must_link_confidence" not in ts.columns:
		# float column: missing values are NaN, not None
		ts["must_link_confidence"] = pd.Series([float("nan")] * len(ts), dtype="float64")
	if "cannot_link_anchor_keys_json" not in ts.columns:
		ts["cannot_link_anchor_keys_json"] = pd.Series([None] * len(ts), dtype="object")

	if agg and (not ts.empty) and ("tracklet_id" in ts.columns):
		# Build a tiny dataframe for a deterministic left-join on tracklet_id
		rows: List[Dict[str, Any]] = []
		for tid, h in agg.items():
			rows.append(
				{
					"tracklet_id": tid,
					"identity_hints_json": h.identity_hints_json,
					"must_link_anchor_key": h.must_link_anchor_key,
					"must_link_confidence": float(h.must_link_confidence)
					if h.must_link_confidence is not None
					else float("nan"),
					"cannot_link_anchor_keys_json": h.cannot_link_anchor_keys_json,
				}
			)
		hints_df = pd.DataFrame(rows)
		if not hints_df.empty:
			# Enforce expected dtypes
			hints_df["tracklet_id"] = hints_df["tracklet_id"].astype(str)
			hints_df["must_link_confidence"] = hints_df["must_link_confidence"].astype("float64")
			ts = ts.merge(hints_df, on="tracklet_id", how="left", suffixes=("", "_hint"))
			# Coalesce into the target columns
			for col in [
				"identity_hints_json",
				"must_link_anchor_key",
				"must_link_confidence",
				"cannot_link_anchor_keys_json",
			]:
				hc = f"{col}_hint"
				if hc in ts.columns:
					# Prefer hint values when present
					if col == "must_link_confidence":
						ts[col] = ts[hc].combine_first(ts[col]).astype("float64")
					else:
						ts[col] = ts[hc].combine_first(ts[col])
					ts = ts.drop(columns=[hc])

	# Nit: enforce float family consistently at write time
	# (prevents downstream sample/JSON renderings from showing ints)
	ts["must_link_confidence"] = ts["must_link_confidence"].astype("float64")

	span_events: List[Dict[str, Any]] = []
	bbox_missing_after_join = 0

	# ----------------------------
	# Checkpoint 2: occlusion spans + repaired x/y + evidence
	# ----------------------------
	if not tf.empty:
		# Add frame-level columns (dtype-stable defaults)
		if "x_m_repaired" not in tf.columns:
			tf["x_m_repaired"] = pd.to_numeric(tf.get("x_m", np.nan), errors="coerce").astype("float64")
		if "y_m_repaired" not in tf.columns:
			tf["y_m_repaired"] = pd.to_numeric(tf.get("y_m", np.nan), errors="coerce").astype("float64")
		if "is_repaired" not in tf.columns:
			tf["is_repaired"] = pd.Series([False] * len(tf), dtype="bool")
		if "repair_method" not in tf.columns:
			tf["repair_method"] = pd.Series([None] * len(tf), dtype="object")
		if "repair_span_id" not in tf.columns:
			tf["repair_span_id"] = pd.Series([pd.NA] * len(tf), dtype="Int64")
		if "occ_span_active" not in tf.columns:
			tf["occ_span_active"] = pd.Series([False] * len(tf), dtype="bool")
		if "occ_r_bottom" not in tf.columns:
			tf["occ_r_bottom"] = pd.Series([0.0] * len(tf), dtype="float64")
		if "occ_r_height" not in tf.columns:
			tf["occ_r_height"] = pd.Series([0.0] * len(tf), dtype="float64")

		# Add summary-level evidence columns (dtype-stable defaults)
		if "n_occlusion_spans" not in ts.columns:
			ts["n_occlusion_spans"] = pd.Series([0] * len(ts), dtype="int64")
		if "n_repaired_frames" not in ts.columns:
			ts["n_repaired_frames"] = pd.Series([0] * len(ts), dtype="int64")
		if "min_nn_dist_m_at_anchors" not in ts.columns:
			ts["min_nn_dist_m_at_anchors"] = pd.Series([float("nan")] * len(ts), dtype="float64")
		if "mean_nn_dist_m_at_anchors" not in ts.columns:
			ts["mean_nn_dist_m_at_anchors"] = pd.Series([float("nan")] * len(ts), dtype="float64")
		if "min_tracks_within_r_at_anchors" not in ts.columns:
			ts["min_tracks_within_r_at_anchors"] = pd.Series([pd.NA] * len(ts), dtype="Int64")
		if "mean_tracks_within_r_at_anchors" not in ts.columns:
			ts["mean_tracks_within_r_at_anchors"] = pd.Series([float("nan")] * len(ts), dtype="float64")
		if "n_spans_with_plausible_other_candidate" not in ts.columns:
			ts["n_spans_with_plausible_other_candidate"] = pd.Series([0] * len(ts), dtype="int64")

		# Join bbox y1/y2 from detections for occlusion signal
		det_path = Path(layout.detections_parquet())
		have_bbox = False
		if det_path.exists() and "detection_id" in tf.columns:
			det = pd.read_parquet(det_path)
			keep = [c for c in ["detection_id", "y1", "y2"] if c in det.columns]
			det = det[keep].copy()
			det["detection_id"] = det["detection_id"].astype(str)
			tf["detection_id"] = tf["detection_id"].astype(str)
			tf = tf.merge(det, on="detection_id", how="left")
			tf["bbox_top"] = pd.to_numeric(tf["y1"], errors="coerce").astype("float64")
			tf["bbox_bottom"] = pd.to_numeric(tf["y2"], errors="coerce").astype("float64")
			tf["bbox_h"] = (tf["bbox_bottom"] - tf["bbox_top"]).astype("float64")
			bbox_missing_after_join = int(tf["bbox_bottom"].isna().sum())
			have_bbox = True

		# Scene positions for context (use repaired where available)
		tf["x_ctx"] = pd.to_numeric(tf["x_m_repaired"].combine_first(tf.get("x_m")), errors="coerce").astype("float64")
		tf["y_ctx"] = pd.to_numeric(tf["y_m_repaired"].combine_first(tf.get("y_m")), errors="coerce").astype("float64")
		all_pos = tf[["tracklet_id", "frame_index", "x_ctx", "y_ctx"]].rename(columns={"x_ctx": "x", "y_ctx": "y"})

		n_spans_by_tid: Dict[str, int] = {}
		n_plausible_by_tid: Dict[str, int] = {}
		nn_vals_by_tid: Dict[str, List[float]] = {}
		cnt_vals_by_tid: Dict[str, List[int]] = {}

		if enable_norm and have_bbox and ("tracklet_id" in tf.columns):
			tracklets = sorted(tf["tracklet_id"].dropna().astype(str).unique().tolist())
			for tid in tracklets:
				sub = tf[tf["tracklet_id"].astype(str) == tid].copy()
				sub = (
					sub.sort_values(["frame_index", "detection_id"], kind="mergesort")
					if "detection_id" in sub.columns
					else sub.sort_values(["frame_index"], kind="mergesort")
				)
				if sub.empty or sub["bbox_bottom"].isna().all() or sub["bbox_h"].isna().all():
					continue

				sub2, spans = _detect_occlusion_linker2(sub, occ_cfg)
				spans_abs = [(int(s), int(e)) for (s, e) in spans]

				# initialize repaired as pass-through
				sub2["x_m_repaired"] = pd.to_numeric(sub2.get("x_m"), errors="coerce").astype("float64")
				sub2["y_m_repaired"] = pd.to_numeric(sub2.get("y_m"), errors="coerce").astype("float64")

				plausible_count = 0
				for span_id, (a, b) in enumerate(spans_abs):
					n_spans_by_tid[tid] = n_spans_by_tid.get(tid, 0) + 1
					pre_i = a - 1
					post_i = b + 1
					event: Dict[str, Any] = {
						"event_type": "d0_occlusion_span",
						"timestamp": _now_ms(),
						"clip_id": getattr(manifest, "clip_id", None),
						"camera_id": getattr(manifest, "camera_id", None),
						"tracklet_id": tid,
						"repair_span_id": int(span_id),
						"span_start_frame": int(sub2.iloc[a]["frame_index"]),
						"span_end_frame": int(sub2.iloc[b]["frame_index"]),
						"n_frames": int(b - a + 1),
						"occ_detector": "linker2",
						"occ_evidence": {
							"max_r_bottom": float(np.max(sub2["occ_r_bottom"].to_numpy()[a : b + 1])) if b >= a else 0.0,
							"max_r_height": float(np.max(sub2["occ_r_height"].to_numpy()[a : b + 1])) if b >= a else 0.0,
							"max_dy2_px": float(np.max(sub2["_occ_dy2_px"].fillna(0.0).to_numpy()[a : b + 1])) if b >= a else 0.0,
						},
						"repaired": False,
						"skip_reason": None,
					}
					if pre_i < 0 or post_i >= len(sub2):
						event["skip_reason"] = "missing_anchors_out_of_bounds"
						span_events.append(event)
						continue
					x0 = sub2.iloc[pre_i]["x_m_repaired"]
					y0 = sub2.iloc[pre_i]["y_m_repaired"]
					x1 = sub2.iloc[post_i]["x_m_repaired"]
					y1 = sub2.iloc[post_i]["y_m_repaired"]
					if pd.isna(x0) or pd.isna(y0) or pd.isna(x1) or pd.isna(y1):
						event["skip_reason"] = "missing_anchor_world_nan"
						span_events.append(event)
						continue

					f0 = int(sub2.iloc[pre_i]["frame_index"])
					f1 = int(sub2.iloc[post_i]["frame_index"])
					if f1 == f0:
						event["skip_reason"] = "invalid_anchor_frames_equal"
						span_events.append(event)
						continue

					event["anchors"] = {
						"pre": {"frame_index": f0, "x_m": float(x0), "y_m": float(y0)},
						"post": {"frame_index": f1, "x_m": float(x1), "y_m": float(y1)},
						"anchor_separation_m": float(
							sqrt((float(x1) - float(x0)) ** 2 + (float(y1) - float(y0)) ** 2)
						),
					}

					# interpolate repaired coords inside span using frame_index alpha
					for i2 in range(a, b + 1):
						fi = int(sub2.iloc[i2]["frame_index"])
						alpha = (fi - f0) / (f1 - f0)
						sub2.iat[i2, sub2.columns.get_loc("x_m_repaired")] = (1.0 - alpha) * float(x0) + alpha * float(x1)
						sub2.iat[i2, sub2.columns.get_loc("y_m_repaired")] = (1.0 - alpha) * float(y0) + alpha * float(y1)
						sub2.iat[i2, sub2.columns.get_loc("is_repaired")] = True
						sub2.iat[i2, sub2.columns.get_loc("repair_method")] = "interp_partial_occlusion"
						sub2.iat[i2, sub2.columns.get_loc("repair_span_id")] = int(span_id)

					event["repaired"] = True
					event["skip_reason"] = None

					# global context at anchors (evidence only)
					if (context_radius_f is None) and (candidate_radius_f is None):
						event["global_context"] = {"disabled": True}
					else:
						nn_pre, cnt_pre = _compute_nn_and_density(
							all_pos,
							tid=tid,
							frame_index=f0,
							x=float(x0),
							y=float(y0),
							context_radius_m=context_radius_f,
						)
						nn_post, cnt_post = _compute_nn_and_density(
							all_pos,
							tid=tid,
							frame_index=f1,
							x=float(x1),
							y=float(y1),
							context_radius_m=context_radius_f,
						)
						if nn_pre is not None:
							nn_vals_by_tid.setdefault(tid, []).append(float(nn_pre))
						if nn_post is not None:
							nn_vals_by_tid.setdefault(tid, []).append(float(nn_post))
						if cnt_pre is not None:
							cnt_vals_by_tid.setdefault(tid, []).append(int(cnt_pre))
						if cnt_post is not None:
							cnt_vals_by_tid.setdefault(tid, []).append(int(cnt_post))

						plausible = None
						if candidate_radius_f is not None and nn_post is not None:
							plausible = bool(nn_post <= candidate_radius_f)
							if plausible:
								plausible_count += 1
						event["global_context"] = {
							"disabled": False,
							"context_radius_m": context_radius_f,
							"candidate_radius_m": candidate_radius_f,
							"nn_dist_m_pre": nn_pre,
							"nn_dist_m_post": nn_post,
							"tracks_within_r_pre": cnt_pre,
							"tracks_within_r_post": cnt_post,
							"plausible_other_candidate_near_post": plausible,
						}

					span_events.append(event)

				if plausible_count:
					n_plausible_by_tid[tid] = n_plausible_by_tid.get(tid, 0) + int(plausible_count)

				# Write back sub2 values into tf for this tid
				tf.loc[sub2.index, "occ_r_bottom"] = sub2["occ_r_bottom"].astype("float64")
				tf.loc[sub2.index, "occ_r_height"] = sub2["occ_r_height"].astype("float64")
				tf.loc[sub2.index, "occ_span_active"] = sub2["occ_span_active"].astype("bool")
				tf.loc[sub2.index, "x_m_repaired"] = sub2["x_m_repaired"].astype("float64")
				tf.loc[sub2.index, "y_m_repaired"] = sub2["y_m_repaired"].astype("float64")
				tf.loc[sub2.index, "is_repaired"] = (
					sub2["is_repaired"].astype("bool")
					if "is_repaired" in sub2.columns
					else tf.loc[sub2.index, "is_repaired"]
				)
				tf.loc[sub2.index, "repair_method"] = (
					sub2["repair_method"] if "repair_method" in sub2.columns else tf.loc[sub2.index, "repair_method"]
				)
				if "repair_span_id" in sub2.columns:
					tf.loc[sub2.index, "repair_span_id"] = pd.to_numeric(sub2["repair_span_id"], errors="coerce").astype(
						"Int64"
					)

		# summary aggregates
		if not ts.empty and "tracklet_id" in ts.columns:
			repaired = tf[tf["is_repaired"] == True]  # noqa: E712
			n_rep = repaired.groupby(tf["tracklet_id"].astype(str)).size().to_dict() if not repaired.empty else {}
			ts["n_occlusion_spans"] = (
				ts["tracklet_id"].astype(str).map(lambda t: int(n_spans_by_tid.get(t, 0))).astype("int64")
			)
			ts["n_repaired_frames"] = (
				ts["tracklet_id"].astype(str).map(lambda t: int(n_rep.get(t, 0))).astype("int64")
			)
			ts["n_spans_with_plausible_other_candidate"] = (
				ts["tracklet_id"].astype(str).map(lambda t: int(n_plausible_by_tid.get(t, 0))).astype("int64")
			)

			def _min_or_nan(vals: List[float]) -> float:
				return float(np.min(vals)) if vals else float("nan")

			def _mean_or_nan(vals: List[float]) -> float:
				return float(np.mean(vals)) if vals else float("nan")

			ts["min_nn_dist_m_at_anchors"] = (
				ts["tracklet_id"].astype(str).map(lambda t: _min_or_nan(nn_vals_by_tid.get(t, []))).astype("float64")
			)
			ts["mean_nn_dist_m_at_anchors"] = (
				ts["tracklet_id"].astype(str).map(lambda t: _mean_or_nan(nn_vals_by_tid.get(t, []))).astype("float64")
			)

			# counts within radius are only valid if context_radius_m is configured; otherwise keep NA/NaN
			if context_radius_f is not None:
				ts["min_tracks_within_r_at_anchors"] = (
					ts["tracklet_id"].astype(str)
					.map(
						lambda t: (
							int(np.min(cnt_vals_by_tid.get(t, []))) if cnt_vals_by_tid.get(t, []) else pd.NA
						)
					)
					.astype("Int64")
				)
				ts["mean_tracks_within_r_at_anchors"] = (
					ts["tracklet_id"].astype(str)
					.map(
						lambda t: (
							float(np.mean(cnt_vals_by_tid.get(t, []))) if cnt_vals_by_tid.get(t, []) else float("nan")
						)
					)
					.astype("float64")
				)

		# drop helper columns from tf before writing (schema strict)
		# CP3: dt-aware kinematics (velocity/accel) from effective world coords; flag-only (no clamping).
		fps = float(getattr(manifest, "fps", 0.0) or 0.0)
		if not (fps > 0.0):
			raise ValueError(f"D0 CP3 requires manifest.fps > 0, got {fps!r}")
		tf, kin_summary = _apply_cp3_kinematics(tf, fps=fps, kin_cfg=kin_cfg)

		# drop helper columns from tf before writing (schema strict)
		for c in ["y1", "y2", "bbox_top", "bbox_bottom", "bbox_h", "x_ctx", "y_ctx", "_occ_dy2_px"]:
			if c in tf.columns:
				tf = tf.drop(columns=[c])

	# Write outputs
	out_frames = Path(layout.tracklet_bank_frames_parquet())
	out_summ = Path(layout.tracklet_bank_summaries_parquet())
	out_frames.parent.mkdir(parents=True, exist_ok=True)

	# Coerce dt_s to float64 if present — Stage A may produce object dtype
	# when values are mixed None/float, which fails D1 schema validation.
	if "dt_s" in tf.columns:
		tf["dt_s"] = pd.to_numeric(tf["dt_s"], errors="coerce").astype("float64")

	tf.to_parquet(out_frames, index=False)
	ts.to_parquet(out_summ, index=False)

	# Minimal audit
	audit_path = Path(layout.audit_jsonl("D"))
	# Span-level events in deterministic order (tracklet_id asc, span_id asc)
	for ev in sorted(span_events, key=lambda e: (str(e.get("tracklet_id", "")), int(e.get("repair_span_id", -1)))):
		_write_audit_event(audit_path, ev)
	_write_audit_event(
		audit_path,
		{
			"event_type": "d0_occlusion_summary",
			"timestamp": _now_ms(),
			"clip_id": getattr(manifest, "clip_id", None),
			"camera_id": getattr(manifest, "camera_id", None),
			"totals": {
				"spans_detected": int(sum(n_spans_by_tid.values())) if "n_spans_by_tid" in locals() else 0,
				"bbox_missing_rows_after_join": int(bbox_missing_after_join) if "bbox_missing_after_join" in locals() else 0,
				"context_enabled": bool((context_radius_f is not None) or (candidate_radius_f is not None)),
			},
		},
	)
	_write_audit_event(
		audit_path,
		{
			"event_type": "d0_kinematics_summary",
			"timestamp": _now_ms(),
			"clip_id": getattr(manifest, "clip_id", None),
			"camera_id": getattr(manifest, "camera_id", None),
			"totals": kin_summary,
		},
	)
	_write_audit_event(
		audit_path,
		{
			"event": "stage_D0_bank_written",
			"event_type": "stage_D0_bank_written",
			"timestamp": _now_ms(),
			"clip_id": getattr(manifest, "clip_id", None),
			"camera_id": getattr(manifest, "camera_id", None),
			"outputs": {
				"tracklet_bank_frames_parquet": layout.rel_to_clip_root(out_frames),
				"tracklet_bank_summaries_parquet": layout.rel_to_clip_root(out_summ),
			},
			"counts": {
				"bank_frames_rows": int(len(tf)),
				"bank_summaries_rows": int(len(ts)),
				"identity_hints_records": int(len(ih_records)),
			},
		},
	)
