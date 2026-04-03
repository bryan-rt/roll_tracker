"""Stage A (detect_track) — output helpers.

Slice 2 goal: stop using a hard NotImplemented stub so the orchestrator can
produce schema-correct, validator-passing artifacts even before the real
detector/tracker is wired in.

The actual compute logic will evolve, but the output schema and canonical
paths must remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
	"float": "Float64",
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


def _homography_id_for_camera(camera_id: str) -> Optional[str]:
	"""Best-effort stable identifier for the homography used by Stage A.

	We use the canonical per-camera homography.json path and hash its bytes.
	"""
	path = Path("configs") / "cameras" / camera_id / "homography.json"
	if not path.exists():
		return None
	try:
		digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
	except Exception:
		return None
	return f"{path.as_posix()}@sha256:{digest}"


COCO_KEYPOINT_NAMES = [
	"nose", "left_eye", "right_eye", "left_ear", "right_ear",
	"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
	"left_wrist", "right_wrist", "left_hip", "right_hip",
	"left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _kp_column_names() -> List[str]:
	"""Generate keypoint column names: kp_{name}_x, kp_{name}_y, kp_{name}_conf."""
	cols: List[str] = []
	for name in COCO_KEYPOINT_NAMES:
		cols.extend([f"kp_{name}_x", f"kp_{name}_y", f"kp_{name}_conf"])
	return cols


@dataclass
class StageAWriteResult:
	detections_ref: str
	tracklet_frames_ref: str
	tracklet_summaries_ref: str
	contact_points_ref: str
	audit_ref: str
	keypoints_ref: Optional[str] = None
	color_histograms_ref: Optional[str] = None
	tracklet_histogram_summaries_ref: Optional[str] = None


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
		self._kp_rows: List[Dict[str, Any]] = []
		self._hist_rows: List[Dict[str, Any]] = []
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
	# CP20: Keypoints + Histograms
	# -------------------------
	def append_keypoint_row(
		self,
		*,
		frame_index: int,
		track_id: str,
		keypoints: Optional[np.ndarray] = None,
		is_isolated: Optional[bool] = None,
	) -> None:
		"""Append a keypoint row for one detection. keypoints shape (17, 3) or None."""
		row: Dict[str, Any] = {
			"frame_index": frame_index,
			"track_id": track_id,
			"is_isolated": is_isolated,
		}
		kp_cols = _kp_column_names()
		if keypoints is not None and keypoints.shape[0] >= 17:
			for i, name in enumerate(COCO_KEYPOINT_NAMES):
				row[f"kp_{name}_x"] = float(keypoints[i, 0])
				row[f"kp_{name}_y"] = float(keypoints[i, 1])
				row[f"kp_{name}_conf"] = float(keypoints[i, 2])
		else:
			for col in kp_cols:
				row[col] = np.nan
		self._kp_rows.append(row)

	def append_histogram_row(
		self,
		*,
		frame_index: int,
		track_id: str,
		is_isolated: bool,
		histogram: Optional[np.ndarray] = None,
		crop_method: str = "not_isolated",
	) -> None:
		"""Append a histogram row for one detection. histogram is 144-element float32 or None."""
		from .histogram import HIST_SIZE

		row: Dict[str, Any] = {
			"frame_index": frame_index,
			"track_id": track_id,
			"is_isolated": is_isolated,
			"crop_method": crop_method,
		}
		if histogram is not None:
			for i in range(HIST_SIZE):
				row[f"hist_{i}"] = float(histogram[i])
		else:
			for i in range(HIST_SIZE):
				row[f"hist_{i}"] = np.nan
		self._hist_rows.append(row)

	# -------------------------
	# Finalize
	# -------------------------
	def get_tracklet_spans(self) -> Dict[str, Tuple[int, int]]:
		"""Return deterministic tracklet spans without writing artifacts.

		This is safe to call during multiplex runs before Stage A artifacts are
		written. It does not mutate internal state and only inspects the in-memory
		tracklet frame rows appended so far.
		
		Returns:
			Dict mapping tracklet_id -> (start_frame, end_frame) inclusive.
		"""
		if not self._tf_rows:
			return {}
		spans: Dict[str, Tuple[int, int]] = {}
		for r in self._tf_rows:
			try:
				tid = str(r.get("tracklet_id"))
				fi = int(r.get("frame_index"))
			except Exception:
				continue
			if not tid:
				continue
			lo_hi = spans.get(tid)
			if lo_hi is None:
				spans[tid] = (fi, fi)
			else:
				lo, hi = lo_hi
				if fi < lo:
					lo = fi
				if fi > hi:
					hi = fi
				spans[tid] = (lo, hi)
		# deterministic iteration for callers
		return {k: spans[k] for k in sorted(spans.keys())}

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
		cp_path = self.layout.stage_A_contact_points_parquet()
		audit_path = self.layout.audit_jsonl("A")

		# Derive baseline contact_points directly from tracklet_frames (F0G)
		cp_cols = [
			"clip_id",
			"camera_id",
			"frame_index",
			"timestamp_ms",
			"detection_id",
			"tracklet_id",
			"u_px",
			"v_px",
			"x_m",
			"y_m",
			"on_mat",
			"contact_conf",
			"contact_method",
		]
		if tf_df.empty:
			cp_df = empty_df_for_schema_key("contact_points")[cp_cols].copy()
		else:
			# Keep only Stage A canonical columns and enforce deterministic ordering.
			cp_df = tf_df[cp_cols].copy()
			cp_df = cp_df.sort_values(["frame_index", "detection_id"], kind="mergesort")

		# Back-compat: populate Stage B legacy columns for convenience.
		# These do not affect Stage A validation and help downstream users inspect the table.
		cp_df["method"] = cp_df.get("contact_method")
		if not det_df.empty:
			# Prefer mask_quality as "confidence of the mask"; fall back to detector confidence.
			if "mask_quality" in det_df.columns:
				q = det_df.set_index("detection_id")["mask_quality"]
				cp_df["confidence"] = cp_df["detection_id"].map(q)
			elif "confidence" in det_df.columns:
				c = det_df.set_index("detection_id")["confidence"]
				cp_df["confidence"] = cp_df["detection_id"].map(c)
			else:
				cp_df["confidence"] = cp_df.get("contact_conf")
		else:
			cp_df["confidence"] = cp_df.get("contact_conf")

		hid = _homography_id_for_camera(self.camera_id)
		cp_df["homography_id"] = hid if hid is not None else pd.NA

		if not tf_df.empty:
			# Audit simple health metrics
			try:
				on = cp_df["on_mat"]
				on_true = int(on.fillna(False).sum())
				on_false = int((on == False).sum())  # noqa: E712
				on_null = int(on.isna().sum())
			except Exception:
				on_true = on_false = on_null = 0

			null_x = int(cp_df["x_m"].isna().sum()) if "x_m" in cp_df.columns else 0
			null_y = int(cp_df["y_m"].isna().sum()) if "y_m" in cp_df.columns else 0

			method_counts = {}
			if "contact_method" in cp_df.columns:
				try:
					method_counts = cp_df["contact_method"].fillna("null").value_counts().to_dict()
				except Exception:
					method_counts = {}

			self.audit(
				"contact_points_stats",
				{
					"n_rows": int(len(cp_df)),
					"n_null_x_m": null_x,
					"n_null_y_m": null_y,
					"on_mat_true": on_true,
					"on_mat_false": on_false,
					"on_mat_null": on_null,
					"contact_method_counts": method_counts,
				},
			)

		det_df.to_parquet(det_path, index=False)
		tf_df.to_parquet(tf_path, index=False)
		ts_df.to_parquet(ts_path, index=False)
		cp_df = _coerce_df_to_schema(cp_df, "contact_points")
		cp_df.to_parquet(cp_path, index=False)
		_write_jsonl(audit_path, self._audit_events)

		# CP20: Write keypoints sidecar
		keypoints_ref = None
		if self._kp_rows:
			kp_df = pd.DataFrame(self._kp_rows)
			if not kp_df.empty:
				kp_df = kp_df.sort_values(["frame_index", "track_id"], kind="mergesort")
			kp_path = self.layout.keypoints_parquet()
			kp_df.to_parquet(kp_path, index=False)
			keypoints_ref = self.layout.rel_to_clip_root(kp_path)

		# CP20: Write histogram sidecars
		color_histograms_ref = None
		tracklet_histogram_summaries_ref = None
		if self._hist_rows:
			hist_df = pd.DataFrame(self._hist_rows)
			if not hist_df.empty:
				hist_df = hist_df.sort_values(["frame_index", "track_id"], kind="mergesort")
			hist_path = self.layout.color_histograms_parquet()
			hist_df.to_parquet(hist_path, index=False)
			color_histograms_ref = self.layout.rel_to_clip_root(hist_path)

			# Per-tracklet summary: average isolated-frame histograms
			tracklet_histogram_summaries_ref = self._write_tracklet_histogram_summaries(hist_df)

		return StageAWriteResult(
			detections_ref=self.layout.rel_to_clip_root(det_path),
			tracklet_frames_ref=self.layout.rel_to_clip_root(tf_path),
			tracklet_summaries_ref=self.layout.rel_to_clip_root(ts_path),
			contact_points_ref=self.layout.rel_to_clip_root(cp_path),
			audit_ref=self.layout.rel_to_clip_root(audit_path),
			keypoints_ref=keypoints_ref,
			color_histograms_ref=color_histograms_ref,
			tracklet_histogram_summaries_ref=tracklet_histogram_summaries_ref,
		)

	def _write_tracklet_histogram_summaries(self, hist_df: pd.DataFrame) -> Optional[str]:
		"""Compute per-tracklet average histograms from isolated frames and write sidecar."""
		from .histogram import HIST_SIZE

		hist_cols = [f"hist_{i}" for i in range(HIST_SIZE)]

		# Filter to isolated frames with valid histograms
		isolated = hist_df[hist_df["is_isolated"] == True].copy()  # noqa: E712
		if isolated.empty:
			return None

		# Drop rows where histogram is all NaN
		has_hist = isolated[hist_cols].notna().any(axis=1)
		isolated = isolated[has_hist]
		if isolated.empty:
			return None

		rows: List[Dict[str, Any]] = []
		for tid, grp in isolated.groupby("track_id", sort=True):
			hist_vals = grp[hist_cols].values.astype(np.float32)
			avg = hist_vals.mean(axis=0)
			total = avg.sum()
			if total > 0:
				avg /= total

			method_counts = grp["crop_method"].value_counts().to_dict()

			row: Dict[str, Any] = {
				"tracklet_id": tid,
				"camera_id": self.camera_id,
				"clip_id": self.clip_id,
				"n_isolated_frames": int(len(grp)),
				"crop_method_distribution_json": json.dumps(method_counts, sort_keys=True),
			}
			for i in range(HIST_SIZE):
				row[f"hist_{i}"] = float(avg[i])
			rows.append(row)

		if not rows:
			return None

		summary_df = pd.DataFrame(rows)
		summary_path = self.layout.tracklet_histogram_summaries_parquet()
		summary_df.to_parquet(summary_path, index=False)
		return self.layout.rel_to_clip_root(summary_path)
