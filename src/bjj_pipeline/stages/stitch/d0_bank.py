"""Stage D0 — Tracklet bank creation (master tables for downstream D stages).

Responsibilities in Checkpoint 1:
	- Create per-frame and per-tracklet bank tables under stage_D/
	- Join Stage C identity_hints.jsonl into the TRACKLET-LEVEL bank (summaries)
	- Write a minimal deterministic stage_D/audit.jsonl

No geometry repair is performed in this checkpoint; bank tables are pass-through
from Stage A plus identity-hint aggregation.
"""

from __future__ import annotations

import json
import time
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
	identity_hints_json: Optional[str]
	must_link_anchor_key: Optional[str]
	must_link_confidence: Optional[float]
	cannot_link_anchor_keys_json: Optional[str]


def _aggregate_identity_hints(records: List[Dict[str, Any]]) -> Dict[str, HintAgg]:
	"""Aggregate identity_hints.jsonl records into tracklet-level summary fields.

	Deterministic policy:
	  - identity_hints_json: JSON list of all records for the tracklet, sorted by:
	      (constraint, -confidence, anchor_key, tag_id/evidence stable json)
	  - must_link_*: choose best must_link by (-confidence, anchor_key)
	  - cannot_link_anchor_keys_json: sorted unique anchor_keys for cannot_link
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

		# best must_link
		must = [r for r in recs if str(r.get("constraint", "")) == "must_link"]
		must_best_key: Optional[str] = None
		must_best_conf: Optional[float] = None
		if must:
			def _mk(rr: Dict[str, Any]) -> Tuple[float, str]:
				conf = rr.get("confidence", 0.0)
				try:
					cf = float(conf)
				except Exception:
					cf = 0.0
				return (-cf, str(rr.get("anchor_key", "")))

			best = sorted(must, key=_mk)[0]
			must_best_key = str(best.get("anchor_key", "")) or None
			try:
				must_best_conf = float(best.get("confidence", 0.0))
			except Exception:
				must_best_conf = None

		# cannot_link anchors
		cannot = [r for r in recs if str(r.get("constraint", "")) == "cannot_link"]
		cannot_keys = sorted({str(r.get("anchor_key", "")) for r in cannot if str(r.get("anchor_key", ""))})
		cannot_json = json.dumps(cannot_keys, sort_keys=True) if cannot_keys else None

		out[tid] = HintAgg(
			identity_hints_json=identity_hints_json,
			must_link_anchor_key=must_best_key,
			must_link_confidence=must_best_conf,
			cannot_link_anchor_keys_json=cannot_json,
		)
	return out


def _rolling_median(s: pd.Series, window: int) -> pd.Series:
	w = max(int(window), 1)
	return s.rolling(window=w, min_periods=1).median()


def _compute_occ_ratios(df: pd.DataFrame, *, onset_window: int) -> pd.DataFrame:
	"""
	Compute normalized occlusion ratios using bbox_bottom and bbox_h:
	  r_bottom = (base_bottom - bottom) / base_height
	  r_height = (base_height - height) / base_height
	"""
	out = df.copy()
	bottom = out["bbox_bottom"].astype("float64")
	height = out["bbox_h"].astype("float64")

	base_bottom = _rolling_median(bottom, onset_window)
	base_height = _rolling_median(height, onset_window)

	bh = base_height.to_numpy(copy=True)
	bh[bh == 0] = np.nan

	rb = (base_bottom.to_numpy() - bottom.to_numpy()) / bh
	rh = (base_height.to_numpy() - height.to_numpy()) / bh
	rb = np.nan_to_num(rb, nan=0.0, posinf=0.0, neginf=0.0).astype("float64")
	rh = np.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0).astype("float64")

	out["occ_r_bottom"] = rb
	out["occ_r_height"] = rh
	return out


def _find_spans(rb: np.ndarray, rh: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
	"""
	Find occlusion spans (index-space, inclusive).
	Spans start when rb>=min_bottom and rh>=min_height (onset),
	and end when rb<=recover_bottom and rh<=recover_height for recover_min_frames.
	"""
	min_bottom = float(cfg.get("min_bottom_frac", 0.15))
	min_height = float(cfg.get("min_height_frac", 0.10))
	onset_min = int(cfg.get("onset_min_frames", 1))

	rec_bottom = float(cfg.get("recover_bottom_frac", 0.10))
	rec_height = float(cfg.get("recover_height_frac", 0.08))
	rec_min = int(cfg.get("recover_min_frames", 3))

	merge_gap = int(cfg.get("merge_gap_frames", 2))
	min_win = int(cfg.get("min_window_frames", 2))
	max_span = cfg.get("max_span_frames", None)
	max_span_i = int(max_span) if max_span is not None else None

	n = int(len(rb))
	if n == 0:
		return []

	on = (rb >= min_bottom) & (rh >= min_height)
	off = (rb <= rec_bottom) & (rh <= rec_height)

	spans: List[Tuple[int, int]] = []
	i = 0
	while i < n:
		if not on[i]:
			i += 1
			continue

		j = i
		while j < n and on[j]:
			j += 1
		if (j - i) < onset_min:
			i = j
			continue

		end = j - 1
		k = j
		rec_run = 0
		while k < n:
			if off[k]:
				rec_run += 1
			else:
				rec_run = 0
			if rec_run >= rec_min:
				end = k - rec_min
				break
			k += 1
		else:
			end = n - 1

		spans.append((i, end))
		i = end + 1

	# merge close spans
	if not spans:
		return []
	merged: List[Tuple[int, int]] = [spans[0]]
	for a, b in spans[1:]:
		pa, pb = merged[-1]
		if a - pb - 1 <= merge_gap:
			merged[-1] = (pa, max(pb, b))
		else:
			merged.append((a, b))

	out: List[Tuple[int, int]] = []
	for a, b in merged:
		if (b - a + 1) < min_win:
			continue
		if max_span_i is not None and (b - a + 1) > max_span_i:
			b = a + max_span_i - 1
		out.append((a, b))
	return out


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


def _get_d0_cfg(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
	return (occ if isinstance(occ, dict) else {}), (ctx if isinstance(ctx, dict) else {})


def run_d0(*, config: Dict[str, Any], layout: Any, manifest: Any) -> None:
	"""Write stage_D bank tables (Checkpoint 1) + occlusion repair evidence (Checkpoint 2)."""
	# Read Stage A base tables
	tf_path = Path(layout.tracklet_frames_parquet())
	ts_path = Path(layout.tracklet_summaries_parquet())
	if not tf_path.exists() or not ts_path.exists():
		raise FileNotFoundError("Stage D0 requires Stage A tracklet_frames.parquet and tracklet_summaries.parquet")

	tf = pd.read_parquet(tf_path)
	ts = pd.read_parquet(ts_path)

	occ_cfg, ctx_cfg = _get_d0_cfg(config)
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

				sub2 = _compute_occ_ratios(sub, onset_window=onset_window)
				rb = sub2["occ_r_bottom"].to_numpy(dtype="float64", copy=False)
				rh = sub2["occ_r_height"].to_numpy(dtype="float64", copy=False)
				spans = _find_spans(rb, rh, occ_cfg)

				# write back ratios + active flags
				active = np.zeros(len(sub2), dtype=bool)
				for a, b in spans:
					active[a : b + 1] = True
				sub2["occ_span_active"] = active

				# initialize repaired as pass-through
				sub2["x_m_repaired"] = pd.to_numeric(sub2.get("x_m"), errors="coerce").astype("float64")
				sub2["y_m_repaired"] = pd.to_numeric(sub2.get("y_m"), errors="coerce").astype("float64")

				plausible_count = 0
				for span_id, (a, b) in enumerate(spans):
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
		for c in ["y1", "y2", "bbox_top", "bbox_bottom", "bbox_h", "x_ctx", "y_ctx"]:
			if c in tf.columns:
				tf = tf.drop(columns=[c])

	# Write outputs
	out_frames = Path(layout.tracklet_bank_frames_parquet())
	out_summ = Path(layout.tracklet_bank_summaries_parquet())
	out_frames.parent.mkdir(parents=True, exist_ok=True)

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
