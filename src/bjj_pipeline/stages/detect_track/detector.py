"""Stage A (detect_track) — person detector adapter.

This module is intentionally structured so that:
  - unit tests do not require ultralytics installed
  - runtime can use Ultralytics YOLO for detection and optional segmentation
  - detection_id assignment is deterministic within a run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from bjj_pipeline.stages.detect_track.types import Detection, MaskSource

logger = logging.getLogger(__name__)


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

def _resize_mask_to_frame(mask_u8: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
	"""Resize a mask to full-frame resolution using nearest-neighbor.

	Ultralytics segmentation masks often come back at the model inference resolution (e.g. 384x640),
	while our geometry / gating code expects masks aligned to the original frame (e.g. 720x1280).
	"""
	if mask_u8.shape[0] == frame_h and mask_u8.shape[1] == frame_w:
		return mask_u8

	# Prefer OpenCV when available (fast + deterministic).
	try:
		import cv2  # type: ignore
		resized = cv2.resize(mask_u8, (int(frame_w), int(frame_h)), interpolation=cv2.INTER_NEAREST)
		return (resized > 0).astype(np.uint8)
	except Exception:
		# Pure-numpy nearest-neighbor fallback
		h0, w0 = mask_u8.shape[:2]
		# Map output pixel centers back to input indices
		ys = (np.linspace(0, h0 - 1, frame_h)).astype(np.int32)
		xs = (np.linspace(0, w0 - 1, frame_w)).astype(np.int32)
		return mask_u8[ys[:, None], xs[None, :]].astype(np.uint8)


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

		# Auto-detect device: MPS (Apple Silicon) > CUDA > CPU
		if device in (None, "null", "auto"):
			import torch
			if torch.backends.mps.is_available():
				resolved_device = "mps"
			elif torch.cuda.is_available():
				resolved_device = "cuda"
			else:
				resolved_device = "cpu"
			print(f"[Stage A] Using device: {resolved_device} (auto-detected)", flush=True)
			# MPS validation
			if resolved_device == "mps":
				try:
					_test = torch.zeros(1, 3, 64, 64, device="mps")
					del _test
				except Exception as e:
					print(f"[Stage A] MPS validation failed ({e}), falling back to CPU", flush=True)
					resolved_device = "cpu"
			self.device = resolved_device
		else:
			self.device = device
			print(f"[Stage A] Using device: {device} (configured)", flush=True)

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
			# "Always try" segmentation: attempt explicit seg_model_path first, then a derived
			# "<model_stem>-seg.pt" path (common Ultralytics naming), then gracefully fall back.
			candidates: list[Path] = []
			if self.seg_model_path:
				candidates.append(Path(self.seg_model_path))
			# Derive from detection model weights, e.g. yolov8n.pt -> yolov8n-seg.pt
			try:
				mp = Path(self.model_path)
				if mp.suffix.lower() == ".pt":
					candidates.append(mp.with_name(mp.stem + "-seg.pt"))
			except Exception:
				pass

			seg_path = None
			for c in candidates:
				try:
					if c and c.exists():
						seg_path = c
						break
				except Exception:
					continue

			if seg_path is None:
				# No seg weights found; keep _seg_model=None and let infer() fall back to bbox.
				logger.warning(
					"YOLO segmentation requested but seg weights not found; falling back to bbox-only detection. "
					"model_path=%s seg_model_path=%s",
					self.model_path,
					self.seg_model_path,
				)
				return

			try:
				from ultralytics import YOLO  # type: ignore
			except Exception as e:  # pragma: no cover
				raise RuntimeError(
					"Ultralytics is not installed. Install 'ultralytics' to use YOLO segmentation."
				) from e
			self._seg_model = YOLO(str(seg_path))

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

		frame_h = int(frame_bgr.shape[0])
		frame_w = int(frame_bgr.shape[1])

		# Use segmentation model if available; else detection model
		using_seg_model = bool(self.use_seg and self._seg_model is not None)
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

		# Evidence-grade debug: prove whether masks are present on this result.
		# (This will tell us if we're truly running a seg model and getting masks back.)
		try:
			m = getattr(r0, "masks", None)
			has_masks = bool(m is not None and getattr(m, "data", None) is not None)
			mask_shape = None
			if has_masks:
				mask_shape = tuple(getattr(m.data, "shape", ()))
			logger.info(
				"YOLO infer: clip=%s cam=%s frame=%d seg_requested=%s seg_model_loaded=%s using_seg_model=%s has_masks=%s masks_shape=%s model_path=%s seg_model_path=%s",
				clip_id,
				camera_id,
				int(frame_index),
				bool(self.use_seg),
				bool(self._seg_model is not None),
				bool(using_seg_model),
				bool(has_masks),
				mask_shape,
				str(self.model_path),
				str(self.seg_model_path),
			)
		except Exception:
			pass
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
				mask = _resize_mask_to_frame(mask, frame_h, frame_w)
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

	def infer_batch(
		self,
		*,
		clip_id: str,
		camera_id: str,
		frames: list[tuple[np.ndarray, int, int]],
	) -> list[list['Detection']]:
		"""Run YOLO predict on a batch of frames. Returns list of per-frame detection lists.

		frames: list of (frame_bgr, frame_index, timestamp_ms) tuples.
		"""
		if not frames:
			return []

		self._lazy_load_models()

		using_seg_model = bool(self.use_seg and self._seg_model is not None)
		model = self._seg_model if (self.use_seg and self._seg_model is not None) else self._model
		if model is None:
			raise RuntimeError("YOLO model failed to load")

		kwargs = {"conf": self.conf}
		if self.imgsz is not None:
			kwargs["imgsz"] = int(self.imgsz)
		if self.device is not None:
			kwargs["device"] = self.device

		# Batch predict: pass list of BGR arrays
		frame_bgrs = [f[0] for f in frames]
		results = model.predict(source=frame_bgrs, verbose=False, **kwargs)
		if not results:
			return [[] for _ in frames]

		all_dets: list[list[Detection]] = []
		for i, (frame_bgr, frame_index, timestamp_ms) in enumerate(frames):
			if i >= len(results):
				all_dets.append([])
				continue
			r0 = results[i]
			frame_h = int(frame_bgr.shape[0])
			frame_w = int(frame_bgr.shape[1])

			boxes = getattr(r0, "boxes", None)
			if boxes is None or len(boxes) == 0:
				all_dets.append([])
				continue

			masks_obj = getattr(r0, "masks", None) if using_seg_model else None
			out: list[Detection] = []
			for j in range(len(boxes)):
				cls_id = int(boxes.cls[j].item()) if hasattr(boxes.cls[j], "item") else int(boxes.cls[j])
				if cls_id != 0:
					continue
				c = float(boxes.conf[j].item()) if hasattr(boxes.conf[j], "item") else float(boxes.conf[j])
				xyxy = boxes.xyxy[j].cpu().numpy() if hasattr(boxes.xyxy[j], "cpu") else boxes.xyxy[j]
				x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
				detection_id = f"d{frame_index:06d}_{j}"
				mask = None
				ms = "none"
				if masks_obj is not None and hasattr(masks_obj, "data") and j < len(masks_obj.data):
					try:
						raw = masks_obj.data[j].cpu().numpy()
						import cv2
						mask = cv2.resize(raw, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
						ms = "yolo_seg"
					except Exception:
						mask = None
						ms = "seg_error"
				out.append(Detection(
					clip_id=clip_id, camera_id=camera_id,
					frame_index=frame_index, timestamp_ms=timestamp_ms,
					detection_id=detection_id, class_name="person",
					confidence=c, x1=x1, y1=y1, x2=x2, y2=y2,
					mask=mask, mask_source=ms, mask_quality=None,
				))
			all_dets.append(out)
		return all_dets
