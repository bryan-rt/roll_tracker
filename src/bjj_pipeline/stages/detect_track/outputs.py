"""Stage A (detect_track) — output helpers.

Slice 2 goal: stop using a hard NotImplemented stub so the orchestrator can
produce schema-correct, validator-passing artifacts even before the real
detector/tracker is wired in.

The actual compute logic will evolve, but the output schema and canonical
paths must remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from bjj_pipeline.contracts import f0_parquet as pq
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout

_FAMILY_TO_DTYPE: Dict[str, str] = {
	"string": "object",  # accept object/StringDtype
	"int": "Int64",  # nullable integer
	"float": "float64",
	"bool": "boolean",
}


def empty_df_for_schema_key(key: str) -> pd.DataFrame:
	"""Create an empty pandas DataFrame that passes f0_parquet schema checks."""
	specs = pq.PARQUET_SCHEMAS[key]
	data = {spec.name: pd.Series([], dtype=_FAMILY_TO_DTYPE[spec.family]) for spec in specs}
	return pd.DataFrame(data)


def _coerce_df_to_schema(df: pd.DataFrame, schema_key: str) -> pd.DataFrame:
	"""Ensure df has exactly the schema columns (order preserved) and dtypes."""
	specs = pq.PARQUET_SCHEMAS[schema_key]
	cols = [s.name for s in specs]

	# add missing columns
	for s in specs:
		if s.name not in df.columns:
			df[s.name] = pd.NA

	# drop extras
	df = df[cols].copy()

	# dtype coercion (best-effort)
	for s in specs:
		family = s.family
		dtype = _FAMILY_TO_DTYPE.get(family, "object")
		try:
			df[s.name] = df[s.name].astype(dtype)
		except Exception:
			# keep as-is; f0_validate will catch severe mismatches
			pass
	return df


def _write_jsonl(path: Path, events: Iterable[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for e in events:
			f.write(json.dumps(e, sort_keys=True))
			f.write("\n")


@dataclass
class StageAWriteResult:
	detections_ref: str
	tracklet_frames_ref: str
	tracklet_summaries_ref: str
	audit_ref: str


class StageAWriter:
	"""Accumulates Stage A rows and writes canonical artifacts.

	This writer enforces deterministic ordering and clip-relative references.
	It is designed to be used by both multipass and multiplex execution.
	"""

	def __init__(self, *, layout: ClipOutputLayout, clip_id: str, camera_id: str) -> None:
		self.layout = layout
		self.clip_id = clip_id
		self.camera_id = camera_id

		self._det_rows: List[Dict[str, Any]] = []
		self._tf_rows: List[Dict[str, Any]] = []
		self._audit_events: List[Dict[str, Any]] = []

	# -------------------------
	# Audit
	# -------------------------
	def audit(self, event_type: str, payload: Dict[str, Any]) -> None:
		self._audit_events.append(
			{
				"stage": "A",
				"event_type": event_type,
				"clip_id": self.clip_id,
				"camera_id": self.camera_id,
				**payload,
			}
		)

	def flush_audit_now(self) -> None:
		path = Path(self.layout.audit_jsonl("A"))
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("a", encoding="utf-8") as f:
			for e in self._audit_events:
				f.write(json.dumps(e, sort_keys=True) + "\n")
		self._audit_events.clear()

	# -------------------------
	# Masks
	# -------------------------
	def write_mask_npz(self, *, frame_index: int, detection_id: str, mask: np.ndarray) -> str:
		"""Write a canonical per-detection mask .npz and return clip-relative ref."""
		path = self.layout.stage_A_mask_npz_path(frame_index, detection_id)
		path.parent.mkdir(parents=True, exist_ok=True)

		# canonical: uint8 {0,1} with key "mask"
		m = mask.astype(np.uint8)
		np.savez_compressed(path, mask=m)
		return self.layout.rel_to_clip_root(path)

	# -------------------------
	# Append rows
	# -------------------------
	def append_detection_row(
		self,
		*,
		frame_index: int,
		timestamp_ms: int,
		detection_id: str,
		class_name: str,
		confidence: float,
		x1: float,
		y1: float,
		x2: float,
		y2: float,
		tracklet_id: Optional[str] = None,
		mask_ref: Optional[str] = None,
		mask_source: Optional[str] = None,
		mask_quality: Optional[float] = None,
		debug_json: Optional[str] = None,
	) -> None:
		self._det_rows.append(
			{
				"clip_id": self.clip_id,
				"camera_id": self.camera_id,
				"frame_index": frame_index,
				"timestamp_ms": timestamp_ms,
				"detection_id": detection_id,
				"tracklet_id": tracklet_id,
				"class_name": class_name,
				"confidence": confidence,
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"mask_ref": mask_ref,
				"mask_source": mask_source,
				"mask_quality": mask_quality,
				"debug_json": debug_json,
			}
		)

	def append_tracklet_frame_row(
		self,
		*,
		frame_index: int,
		timestamp_ms: int,
		tracklet_id: str,
		detection_id: str,
		x1: float,
		y1: float,
		x2: float,
		y2: float,
		local_track_conf: Optional[float] = None,
		u_px: Optional[float] = None,
		v_px: Optional[float] = None,
		x_m: Optional[float] = None,
		y_m: Optional[float] = None,
		vx_m: Optional[float] = None,
		vy_m: Optional[float] = None,
		on_mat: Optional[bool] = None,
		contact_conf: Optional[float] = None,
		contact_method: Optional[str] = None,
		debug_json: Optional[str] = None,
	) -> None:
		self._tf_rows.append(
			{
				"clip_id": self.clip_id,
				"camera_id": self.camera_id,
				"frame_index": frame_index,
				"timestamp_ms": timestamp_ms,
				"tracklet_id": tracklet_id,
				"detection_id": detection_id,
				"local_track_conf": local_track_conf,
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"u_px": u_px,
				"v_px": v_px,
				"x_m": x_m,
				"y_m": y_m,
				"vx_m": vx_m,
				"vy_m": vy_m,
				"on_mat": on_mat,
				"contact_conf": contact_conf,
				"contact_method": contact_method,
				"debug_json": debug_json,
			}
		)

	# -------------------------
	# Finalize
	# -------------------------
	def finalize_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		det_df = pd.DataFrame(self._det_rows) if self._det_rows else empty_df_for_schema_key("detections")
		tf_df = (
			pd.DataFrame(self._tf_rows) if self._tf_rows else empty_df_for_schema_key("tracklet_frames")
		)

		# deterministic ordering
		if not det_df.empty and "frame_index" in det_df.columns and "detection_id" in det_df.columns:
			det_df = det_df.sort_values(["frame_index", "detection_id"], kind="mergesort")
		if not tf_df.empty and "tracklet_id" in tf_df.columns and "frame_index" in tf_df.columns:
			tf_df = tf_df.sort_values(["tracklet_id", "frame_index", "detection_id"], kind="mergesort")

		det_df = _coerce_df_to_schema(det_df, "detections")
		tf_df = _coerce_df_to_schema(tf_df, "tracklet_frames")

		ts_df = self._build_tracklet_summaries(tf_df)
		ts_df = _coerce_df_to_schema(ts_df, "tracklet_summaries")

		self._assert_invariants(det_df, tf_df)
		return det_df, tf_df, ts_df

	def _build_tracklet_summaries(self, tf_df: pd.DataFrame) -> pd.DataFrame:
		if tf_df.empty:
			return empty_df_for_schema_key("tracklet_summaries")
		g = tf_df.groupby("tracklet_id", dropna=False, sort=False)

		rows: List[Dict[str, Any]] = []
		for tid, sub in g:
			try:
				start_frame = int(sub["frame_index"].min())
				end_frame = int(sub["frame_index"].max())
				n_frames = int(sub.shape[0])
			except Exception:
				start_frame = None
				end_frame = None
				n_frames = int(sub.shape[0])

			rows.append(
				{
					"clip_id": self.clip_id,
					"camera_id": self.camera_id,
					"tracklet_id": tid,
					"start_frame": start_frame,
					"end_frame": end_frame,
					"n_frames": n_frames,
				}
			)
		out = pd.DataFrame(rows)
		# deterministic order
		out = out.sort_values(["tracklet_id"], kind="mergesort")
		return out

	def _assert_invariants(self, det_df: pd.DataFrame, tf_df: pd.DataFrame) -> None:
		# Invariant: every tracklet frame references an existing detection
		if tf_df.empty:
			return
		det_ids = set(det_df["detection_id"].dropna().astype(str).tolist()) if not det_df.empty else set()
		tf_ids = set(tf_df["detection_id"].dropna().astype(str).tolist())
		missing = sorted([d for d in tf_ids if d not in det_ids])
		if missing:
			raise ValueError(f"StageA invariant violated: tracklet_frames references missing detection_id(s): {missing[:5]}")

		# Invariant: mask_ref (if present) is clip-relative and file exists
		if "mask_ref" in det_df.columns and not det_df.empty:
			for mr in det_df["mask_ref"].dropna().astype(str).tolist():
				if mr.startswith("/") or ":" in mr[:4]:
					raise ValueError(f"mask_ref must be clip-relative (got: {mr})")
				abs_path = self.layout.clip_root / Path(mr)
				if not abs_path.exists():
					raise ValueError(f"mask_ref file does not exist: {mr}")

	# -------------------------
	# Write
	# -------------------------
	def write_all(self) -> StageAWriteResult:
		det_df, tf_df, ts_df = self.finalize_tables()

		# Create stage dir
		self.layout.stage_dir("A").mkdir(parents=True, exist_ok=True)

		det_path = self.layout.detections_parquet()
		tf_path = self.layout.tracklet_frames_parquet()
		ts_path = self.layout.tracklet_summaries_parquet()
		audit_path = self.layout.audit_jsonl("A")

		det_df.to_parquet(det_path, index=False)
		tf_df.to_parquet(tf_path, index=False)
		ts_df.to_parquet(ts_path, index=False)
		_write_jsonl(audit_path, self._audit_events)

		return StageAWriteResult(
			detections_ref=self.layout.rel_to_clip_root(det_path),
			tracklet_frames_ref=self.layout.rel_to_clip_root(tf_path),
			tracklet_summaries_ref=self.layout.rel_to_clip_root(ts_path),
			audit_ref=self.layout.rel_to_clip_root(audit_path),
		)
