"""Stage A runner: Detect + Tracklets (local association).

This module intentionally exposes a stable orchestration contract:

	run(config: dict, inputs: dict) -> dict

Slice 3+: real implementation (optional) using:
	- Ultralytics YOLO (detection + optional segmentation)
	- BoxMOT BoT-SORT (tracklet generator)
	- StageAProcessor (per-frame deterministic engine)

IMPORTANT: Unit tests should not require ultralytics/boxmot/cv2.
Therefore this runner supports a safe "stub" mode that writes empty,
schema-correct outputs (validator passing) unless explicitly enabled.
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.contracts.f0_projection import CameraProjection, load_calibration_from_payload
from bjj_pipeline.contracts.f0_validate import (
	validate_detections_df,
	validate_stage_A_contact_points_df,
	validate_tracklet_tables,
)
from .outputs import StageAWriter


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
	"""Get nested config value from dict-like or object-like config.

	path uses dot-notation, e.g. "stage_A.mode" or "detector.model_path".
	"""
	cur: Any = cfg
	for key in path.split("."):
		if cur is None:
			return default
		# dict-like
		if isinstance(cur, dict):
			cur = cur.get(key, None)
			continue
		# pydantic-like or plain object
		if hasattr(cur, key):
			cur = getattr(cur, key)
			continue
		return default
	return default if cur is None else cur


def _load_json(path: Path) -> Dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def _load_mat_blueprint(cfg: Any) -> Dict[str, Any]:
	# Prefer explicit config value
	p = _cfg_get(cfg, "mat_blueprint_path", None)
	if p:
		pp = Path(p)
		if pp.exists():
			return _load_json(pp)

	# Default repo location
	pp = Path("configs") / "mat_blueprint.json"
	if pp.exists():
		return _load_json(pp)

	# Last resort: empty (processor will treat on_mat as None/False via helper)
	return {}


def _load_homography_matrix(cfg: Any, camera_id: str) -> CameraProjection:
	"""Load homography + optional lens calibration for camera.

	NOTE: This isolation-path loader does NOT apply direction correction
	(unlike the multiplex_runner version). Known pre-existing issue — not
	addressed in CP16a.
	"""
	p = _cfg_get(cfg, "homography_path", None)
	if p:
		pp = Path(p)
		if pp.exists():
			j = _load_json(pp)
			H = np.asarray(j.get("H", j.get("homography", j.get("matrix"))), dtype=np.float64).reshape((3, 3))
			cm, dc = load_calibration_from_payload(j)
			return CameraProjection(H=H, camera_matrix=cm, dist_coefficients=dc)

	# Typical camera config locations
	cam_dir = Path("configs") / "cameras" / camera_id
	candidates = [
		cam_dir / "homography.json",
		cam_dir / "homography_pipeline.json",
		cam_dir / "homography_from_npy.json",
	]
	for pp in candidates:
		if pp.exists():
			j = _load_json(pp)
			H = np.asarray(j.get("H", j.get("homography", j.get("matrix"))), dtype=np.float64).reshape((3, 3))
			cm, dc = load_calibration_from_payload(j)
			return CameraProjection(H=H, camera_matrix=cm, dist_coefficients=dc)

	raise FileNotFoundError(
		f"Homography not found for camera_id={camera_id}. "
		f"Tried config.homography_path and {candidates}."
	)


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint.

	Always-real mode:
	  - run YOLO + BoT-SORT via StageAProcessor
	"""
	layout: ClipOutputLayout = inputs["layout"]

	camera_id = str(inputs.get("camera_id", getattr(layout, "camera_id", "cam")))
	clip_id = str(inputs.get("clip_id", getattr(layout, "clip_id", layout.clip_root.name)))

	# Ensure stage dir exists
	layout.stage_dir("A").mkdir(parents=True, exist_ok=True)

	# -----------------------------
	# REAL MODE
	# -----------------------------
	# Lazy imports so tests don't require these deps
	try:
		from .processor import StageAProcessor
		from .detector import UltralyticsYoloDetector
		from .tracker import BotSortTracker
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"Stage A is configured as always-real, but required dependencies could not be imported. "
			"Ensure ultralytics + boxmot (and any video deps) are installed."
		) from e

	# Frame iteration: prefer shared FrameIterator if provided by orchestration (Z3),
	# otherwise open video directly (multipass).
	frame_iter = inputs.get("frame_iterator", None)
	if frame_iter is None:
		try:
			from bjj_pipeline.core.frame_iterator import FrameIterator
		except Exception as e:  # pragma: no cover
			raise RuntimeError("FrameIterator not available; cannot run Stage A in real mode.") from e

		clip_path = inputs.get("clip_path", inputs.get("clip", None))
		if clip_path is None:
			raise KeyError("Stage A real mode requires inputs['clip_path'] (or inputs['clip']).")
		frame_iter = FrameIterator(Path(str(clip_path)))

	# Load homography + calibration + blueprint
	proj = _load_homography_matrix(config, camera_id)
	mat_blueprint = inputs.get("mat_blueprint", None) or _load_mat_blueprint(config)

	# Writer
	writer = StageAWriter(layout=layout, clip_id=clip_id, camera_id=camera_id)

	# Prefer structured config: stages.stage_A.detector.*
	model_path = str(
		_cfg_get(
			config,
			"stages.stage_A.detector.model_path",
			_cfg_get(config, "detector.model_path", _cfg_get(config, "models.yolo_det", "models/yolov8n.pt")),
		)
	)
	seg_model_path = _cfg_get(
		config,
		"stages.stage_A.detector.seg_model_path",
		_cfg_get(config, "detector.seg_model_path", _cfg_get(config, "models.yolo_seg", None)),
	)
	use_seg = bool(_cfg_get(config, "stages.stage_A.detector.use_seg", _cfg_get(config, "detector.use_seg", True)))
	conf = float(_cfg_get(config, "stages.stage_A.detector.conf", _cfg_get(config, "detector.conf", 0.25)))
	imgsz = _cfg_get(config, "stages.stage_A.detector.imgsz", _cfg_get(config, "detector.imgsz", None))
	device = _cfg_get(config, "stages.stage_A.detector.device", _cfg_get(config, "detector.device", None))

	detector = UltralyticsYoloDetector(
		model_path=model_path,
		seg_model_path=str(seg_model_path) if seg_model_path is not None else None,
		use_seg=use_seg,
		conf=conf,
		imgsz=int(imgsz) if imgsz is not None else None,
		device=str(device) if device is not None else None,
	)

	# Tracker config
	with_reid = bool(_cfg_get(config, "stages.stage_A.tracker.with_reid", _cfg_get(config, "tracker.with_reid", False)))
	tracker_mode = str(_cfg_get(config, "stages.stage_A.tracker.mode", _cfg_get(config, "tracker.mode", "botsort"))).lower()
	if tracker_mode != "botsort":
		raise ValueError(f"Unsupported tracker.mode={tracker_mode!r}. Only 'botsort' is supported in Stage A.")
	params = _cfg_get(config, "stages.stage_A.tracker.params", _cfg_get(config, "tracker.params", {}))
	if not isinstance(params, dict):
		params = dict(params)

	tracker = BotSortTracker(with_reid=with_reid, params=params)

	processor = StageAProcessor(
		config=config,
		homography=proj.H,
		camera_matrix=proj.camera_matrix,
		dist_coefficients=proj.dist_coefficients,
		mat_blueprint=mat_blueprint,
		writer=writer,
		detector=detector,
		tracker=tracker,
	)

	# Iterate frames with optional batching
	yolo_batch_size = int(_cfg_get(
		config, "stages.stage_A.detector.yolo_batch_size",
		_cfg_get(config, "detector.yolo_batch_size", 1),
	))
	n = 0
	batch: list[tuple] = []  # (frame_bgr, frame_index, timestamp_ms)

	def _unpack_frame(frame):
		if isinstance(frame, tuple) and len(frame) >= 3:
			return frame[0], int(frame[1]), int(frame[2])
		return frame.image_bgr, int(frame.frame_index), int(frame.timestamp_ms)

	def _flush_batch(batch):
		nonlocal n
		if not batch:
			return
		if yolo_batch_size > 1:
			# Batch YOLO inference, then feed per-frame results to processor
			batch_dets = detector.infer_batch(
				clip_id=clip_id,
				camera_id=camera_id,
				frames=batch,
			)
			for (fb, fi, ts), dets in zip(batch, batch_dets):
				processor.process_frame(frame_bgr=fb, frame_index=fi, timestamp_ms=ts,
				                       precomputed_dets=dets)
				n += 1
		else:
			for fb, fi, ts in batch:
				processor.process_frame(frame_bgr=fb, frame_index=fi, timestamp_ms=ts)
				n += 1

	for frame in frame_iter:
		batch.append(_unpack_frame(frame))
		if len(batch) >= yolo_batch_size:
			_flush_batch(batch)
			batch = []

	# Flush remaining frames
	_flush_batch(batch)

	writer.audit("stage_a_completed", {"n_frames": n})
	res = writer.write_all()

	# Validate shape/contracts (parquet schemas)
	det_path = layout.detections_parquet()
	tf_path = layout.tracklet_frames_parquet()
	ts_path = layout.tracklet_summaries_parquet()
	cp_path = layout.stage_A_contact_points_parquet()
	validate_detections_df(pd.read_parquet(det_path))
	validate_tracklet_tables(pd.read_parquet(tf_path), pd.read_parquet(ts_path))
	validate_stage_A_contact_points_df(pd.read_parquet(cp_path))

	return {
		"detections_parquet": res.detections_ref,
		"tracklet_frames_parquet": res.tracklet_frames_ref,
		"tracklet_summaries_parquet": res.tracklet_summaries_ref,
		"contact_points_parquet": res.contact_points_ref,
		"audit_jsonl": res.audit_ref,
	}


def main() -> None:
	"""Optional stage-local CLI.

	Prefer running via `roll-tracker run ...` unless debugging Stage A in isolation.
	"""
	raise SystemExit(
		"Stage A (detect_track) does not yet implement a standalone CLI; "
		"run via `roll-tracker` or implement main() when the stage is ready."
	)
