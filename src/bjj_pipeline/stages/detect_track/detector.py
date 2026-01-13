"""Stage A (detect_track) — person detector adapter.

This module is intentionally structured so that:
  - unit tests do not require ultralytics installed
  - runtime can use Ultralytics YOLO for detection and optional segmentation
  - detection_id assignment is deterministic within a run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from bjj_pipeline.stages.detect_track.types import Detection, MaskSource


class DetectorBackend(Protocol):
	def infer(
		self,
		*,
		clip_id: str,
		camera_id: str,
		frame_index: int,
		timestamp_ms: int,
		frame_bgr: np.ndarray,
	) -> list[Detection]:
		...


def _sorted_det_key(x1: float, y1: float, x2: float, y2: float, conf: float) -> Tuple[float, float, float, float, float]:
	# sort stable, deterministic: left-to-right, top-to-bottom, then higher conf first
	return (float(x1), float(y1), float(x2), float(y2), float(-conf))


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
	"""Convert any reasonable mask representation to uint8 {0,1}."""
	if mask.dtype == np.bool_:
		return mask.astype(np.uint8)
	if np.issubdtype(mask.dtype, np.floating):
		return (mask > 0.5).astype(np.uint8)
	# ints
	return (mask > 0).astype(np.uint8)


class UltralyticsYoloDetector(DetectorBackend):
	"""Ultralytics YOLO detector with optional segmentation support.

	Note: ultralytics import is deferred so that tests don't require the dependency.
	"""

	def __init__(
		self,
		*,
		model_path: str,
		seg_model_path: Optional[str],
		use_seg: bool,
		conf: float,
		imgsz: Optional[int],
		device: Optional[str],
	) -> None:
		self.model_path = model_path
		self.seg_model_path = seg_model_path
		self.use_seg = use_seg
		self.conf = float(conf)
		self.imgsz = imgsz
		self.device = device

		self._model = None
		self._seg_model = None

	def _lazy_load_models(self) -> None:
		if self._model is None:
			try:
				from ultralytics import YOLO  # type: ignore
			except Exception as e:  # pragma: no cover
				raise RuntimeError(
					"Ultralytics is not installed. Install 'ultralytics' to use the YOLO detector backend."
				) from e
			self._model = YOLO(self.model_path)

		if self.use_seg and self._seg_model is None:
			if not self.seg_model_path:
				# seg requested but no weights path; stay None and allow fallback later
				return
			p = Path(self.seg_model_path)
			if not p.exists():
				# seg requested but file missing; stay None and allow fallback later
				return
			try:
				from ultralytics import YOLO  # type: ignore
			except Exception as e:  # pragma: no cover
				raise RuntimeError(
					"Ultralytics is not installed. Install 'ultralytics' to use YOLO segmentation."
				) from e
			self._seg_model = YOLO(str(p))

	def infer(
		self,
		*,
		clip_id: str,
		camera_id: str,
		frame_index: int,
		timestamp_ms: int,
		frame_bgr: np.ndarray,
	) -> list[Detection]:
		# Defensive guard: callers must pass a real frame.
		# This makes failures actionable (instead of a deep Ultralytics error).
		if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
			# This is the single most important fact when debugging "NoneType" reports.
			# Keep it in the exception string so it ends up in stage_A/audit.jsonl.
			raise TypeError(
				f"Unsupported 'img_numpy' input format '{type(frame_bgr)}', valid format is np.ndarray "
				f"(clip_id={clip_id}, camera_id={camera_id}, frame_index={frame_index}, timestamp_ms={timestamp_ms}, "
				f"detector_file={__file__}, model_path={self.model_path}, use_seg={self.use_seg})"
			)

		self._lazy_load_models()

		# Use segmentation model if available; else detection model
		model = self._seg_model if (self.use_seg and self._seg_model is not None) else self._model
		if model is None:  # pragma: no cover
			raise RuntimeError("YOLO model failed to load")

		kwargs = {"conf": self.conf}
		if self.imgsz is not None:
			kwargs["imgsz"] = int(self.imgsz)
		if self.device is not None:
			kwargs["device"] = self.device

		# Ultralytics expects images in BGR (OpenCV) or RGB; it handles numpy arrays.
		results = model.predict(source=frame_bgr, verbose=False, **kwargs)
		if not results:
			return []

		r0 = results[0]
		boxes = getattr(r0, "boxes", None)
		if boxes is None or boxes.xyxy is None:
			return []

		xyxy = boxes.xyxy.cpu().numpy()
		confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones((xyxy.shape[0],))
		clses = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],))

		# Ultralytics uses COCO class indices by default; class 0 is 'person'
		# We hard-filter to class 0 for this pipeline.
		keep = (clses.astype(int) == 0)
		xyxy = xyxy[keep]
		confs = confs[keep]

		masks_full = None
		if model is self._seg_model:
			m = getattr(r0, "masks", None)
			if m is not None and getattr(m, "data", None) is not None:
				# shape: (N, H, W) in float {0..1}
				masks_full = m.data.cpu().numpy()
				masks_full = masks_full[keep] if masks_full.shape[0] == keep.shape[0] else masks_full

		dets: list[Tuple[float, float, float, float, float, Optional[np.ndarray]]] = []
		for i in range(xyxy.shape[0]):
			x1, y1, x2, y2 = (float(x) for x in xyxy[i].tolist())
			c = float(confs[i])
			mask = None
			if masks_full is not None and i < masks_full.shape[0]:
				mask = _as_uint8_mask(masks_full[i])
			dets.append((x1, y1, x2, y2, c, mask))

		# deterministic ordering before assigning IDs
		dets.sort(key=lambda t: _sorted_det_key(t[0], t[1], t[2], t[3], t[4]))

		out: list[Detection] = []
		for k, (x1, y1, x2, y2, c, mask) in enumerate(dets):
			detection_id = f"d{frame_index:06d}_{k}"
			if mask is not None:
				ms: Optional[MaskSource] = "yolo_seg"
			else:
				ms = None
			out.append(
				Detection(
					clip_id=clip_id,
					camera_id=camera_id,
					frame_index=frame_index,
					timestamp_ms=timestamp_ms,
					detection_id=detection_id,
					class_name="person",
					confidence=c,
					x1=x1,
					y1=y1,
					x2=x2,
					y2=y2,
					mask=mask,
					mask_source=ms,
					mask_quality=None,
				)
			)
		return out
