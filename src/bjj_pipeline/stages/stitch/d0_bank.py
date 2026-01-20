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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def run_d0(*, config: Dict[str, Any], layout: Any, manifest: Any) -> None:
	"""Write stage_D bank tables (Checkpoint 1)."""
	# Read Stage A base tables
	tf_path = Path(layout.tracklet_frames_parquet())
	ts_path = Path(layout.tracklet_summaries_parquet())
	if not tf_path.exists() or not ts_path.exists():
		raise FileNotFoundError("Stage D0 requires Stage A tracklet_frames.parquet and tracklet_summaries.parquet")

	tf = pd.read_parquet(tf_path)
	ts = pd.read_parquet(ts_path)

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

	# Write outputs
	out_frames = Path(layout.tracklet_bank_frames_parquet())
	out_summ = Path(layout.tracklet_bank_summaries_parquet())
	out_frames.parent.mkdir(parents=True, exist_ok=True)

	tf.to_parquet(out_frames, index=False)
	ts.to_parquet(out_summ, index=False)

	# Minimal audit
	audit_path = Path(layout.audit_jsonl("D"))
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
