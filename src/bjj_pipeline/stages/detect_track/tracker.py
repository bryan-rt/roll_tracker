"""Stage A (detect_track) — tracker adapter.

BoT-SORT is used as a *tracklet generator* (high precision, allowed to break).

This module is structured so that:
  - unit tests do not require boxmot installed (imports deferred)
  - runtime can use BoxMOT BoT-SORT where available
  - stable, deterministic tracklet_id strings are produced (e.g. "t12")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np

from bjj_pipeline.stages.detect_track.types import Detection, TrackedDetection


class TrackerBackend(Protocol):
	def update(self, *, frame_index: int, detections: list[Detection], frame_bgr: Optional[np.ndarray] = None) -> list[TrackedDetection]:
		...


def _tid(track_id: int) -> str:
	return f"t{int(track_id)}"


class BotSortTracker(TrackerBackend):
	"""BoT-SORT adapter via BoxMOT (if installed).

	Notes:
	  - We keep this adapter minimal: StageAProcessor decides which bbox to pass
		(mask-tight vs raw) and whether to include an image for appearance cues.
	  - If boxmot is not installed, this will raise at runtime when constructed.
	"""

	def __init__(self, *, with_reid: bool, params: Optional[Dict[str, Any]] = None) -> None:
		self.with_reid = bool(with_reid)
		# Allow None for convenience in local debugging; treat as empty dict.
		self.params = dict(params or {})

		self._tracker = None
		self._lazy_init()

	def _lazy_init(self) -> None:
		if self._tracker is not None:
			return
		try:
			# BoxMOT API can vary by version; we keep this encapsulated here.
			# Most versions support:
			#   from boxmot import BotSort
			#   tracker = BotSort(reid_weights=..., device=..., ...)
			from boxmot import BotSort  # type: ignore
		except Exception as e:  # pragma: no cover
			raise RuntimeError(
				"boxmot is not installed. Install 'boxmot' to use tracker.mode='botsort'."
			) from e

		# Params pass-through with sensible defaults for BoxMOT v16+.
		# We include defaults so construction doesn't fail when params are omitted.
		cfg = dict(self.params)
		cfg["with_reid"] = self.with_reid
		# Normalize required args for boxmot>=16 (required even when with_reid=False).
		cfg["device"] = str(cfg.get("device") or "cpu")
		cfg["half"] = bool(cfg.get("half", False))
		# Even if with_reid=False, some versions still require the argument to exist.
		cfg["reid_weights"] = str(cfg.get("reid_weights") or "")

		try:
			self._tracker = BotSort(**cfg)
		except TypeError as e:
			try:
				import boxmot  # type: ignore
				boxmot_ver = getattr(boxmot, "__version__", "unknown")
			except Exception:
				boxmot_ver = "unknown"
			raise TypeError(
				f"Failed to construct BoxMOT BotSort (boxmot=={boxmot_ver}). "
				f"Original error: {e}. "
				f"Provided params keys={sorted(self.params.keys())}, expanded cfg keys={sorted(cfg.keys())}. "
				f"Common fix: include tracker params like 'reid_weights', 'device', 'half'."
			) from e

	def update(
		self,
		*,
		frame_index: int,
		detections: list[Detection],
		frame_bgr: Optional[np.ndarray] = None,
	) -> list[TrackedDetection]:
		"""Update tracker and return association results for this frame.

		Contract:
		  - input detections MUST have deterministic detection_id values already assigned
		  - output must map each detection to exactly one tracklet_id (where possible)
		"""
		if self._tracker is None:  # pragma: no cover
			self._lazy_init()

		if not detections:
			return []

		# BoxMOT expects dets as ndarray [x1,y1,x2,y2,conf,class]
		# For this pipeline, class is always 0 ("person").
		det_arr = np.zeros((len(detections), 6), dtype=np.float32)
		for i, d in enumerate(detections):
			det_arr[i, 0] = float(d.x1)
			det_arr[i, 1] = float(d.y1)
			det_arr[i, 2] = float(d.x2)
			det_arr[i, 3] = float(d.y2)
			det_arr[i, 4] = float(d.confidence)
			det_arr[i, 5] = 0.0

		# BoxMOT/BoT-SORT requires img (np.ndarray) even when ReID is disabled.
		# It uses img for internal steps (e.g., ECC) and asserts on None.
		if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
			raise TypeError(
				f"BotSortTracker.update requires frame_bgr np.ndarray (got {type(frame_bgr)})"
			)

		# Filter degenerate boxes (zero width or height) — YOLO edge-case outputs
		# that cause divide-by-zero in BoT-SORT IoU → Kalman NaN → crash
		if det_arr.shape[0] > 0:
			valid = (det_arr[:, 2] > det_arr[:, 0]) & (det_arr[:, 3] > det_arr[:, 1])
			det_arr = det_arr[valid]

		# Update tracker; common signature: update(dets, img, embs=None)
		tracks = self._tracker.update(det_arr, frame_bgr)  # type: ignore[attr-defined]

		if tracks is None:
			return []

		tracks = np.asarray(tracks)
		if tracks.ndim == 1 and tracks.shape[0] == 0:
			return []  # BoT-SORT returns (0,) when no active tracks — treat as empty
		if tracks.ndim != 2 or tracks.shape[1] < 5:
			raise RuntimeError(f"Unexpected BoxMOT tracker output shape: {tracks.shape}")

		# We need to map tracker output back to our detection_ids.
		# Many trackers do not return a direct det index. For this POC adapter,
		# we do a greedy IoU match between tracker boxes and input det boxes.
		# This is stable because both lists are deterministic-ordered.
		in_boxes = np.asarray([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
		out_boxes = tracks[:, 0:4].astype(np.float32)
		out_ids = tracks[:, 4].astype(np.int64)

		matches = _greedy_iou_match(in_boxes, out_boxes)

		out: list[TrackedDetection] = []
		for in_i, out_j in matches.items():
			d = detections[in_i]
			tid = _tid(int(out_ids[out_j]))
			out.append(
				TrackedDetection(
					detection_id=d.detection_id,
					tracklet_id=tid,
					local_track_conf=None,
					x1=float(d.x1),
					y1=float(d.y1),
					x2=float(d.x2),
					y2=float(d.y2),
				)
			)
		return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
	ax1, ay1, ax2, ay2 = (float(x) for x in a.tolist())
	bx1, by1, bx2, by2 = (float(x) for x in b.tolist())
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter
	if union <= 0.0:
		return 0.0
	return float(inter / union)


def _greedy_iou_match(in_boxes: np.ndarray, out_boxes: np.ndarray) -> Dict[int, int]:
	"""Greedy stable IoU match: each input det matched to at most 1 output track."""
	pairs: list[tuple[float, int, int]] = []
	for i in range(in_boxes.shape[0]):
		for j in range(out_boxes.shape[0]):
			pairs.append((_iou(in_boxes[i], out_boxes[j]), i, j))
	# sort by IoU desc, stable tie-break by indices
	pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

	used_in: set[int] = set()
	used_out: set[int] = set()
	matches: Dict[int, int] = {}
	for iou, i, j in pairs:
		if i in used_in or j in used_out:
			continue
		if iou <= 0.0:
			continue
		matches[i] = j
		used_in.add(i)
		used_out.add(j)
	return matches
