"""AprilTag/ArUco decoding adapter.

Milestone 5: Provide a deterministic helper to decode AprilTags inside a bbox ROI.

This module is intentionally small and pure (no file IO) so it can be called from the
multiplex_AC frame loop.

Tag family policy:
- Default family is "36h11" (manager-locked for POC) unless explicitly overridden by config.

Implementation notes:
- Uses OpenCV's ArUco/AprilTag dictionaries via cv2.aruco.
- If cv2.aruco is not available (opencv-contrib missing), returns an error code and no detections.
- Returns detections in deterministic order (sorted by tag_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TagDetection:
	tag_id: int
	tag_family: str
	corners_px: List[Tuple[float, float]]  # full-frame pixel coords (x,y) for 4 corners


def _get_cv2_aruco():
	try:
		import cv2  # type: ignore
	except Exception:
		return None, None
	aruco = getattr(cv2, "aruco", None)
	return cv2, aruco


def _dictionary_for_family(aruco, tag_family: str):
	fam = str(tag_family)
	# OpenCV predefined dictionaries (AprilTag families)
	mapping = {
		"16h5": getattr(aruco, "DICT_APRILTAG_16h5", None),
		"25h9": getattr(aruco, "DICT_APRILTAG_25h9", None),
		"36h10": getattr(aruco, "DICT_APRILTAG_36h10", None),
		"36h11": getattr(aruco, "DICT_APRILTAG_36h11", None),
	}
	code = mapping.get(fam)
	if code is None:
		return None
	try:
		return aruco.getPredefinedDictionary(code)
	except Exception:
		# Older OpenCV versions may not expose getPredefinedDictionary
		try:
			return aruco.Dictionary_get(code)
		except Exception:
			return None


def _make_detector_params(aruco) -> Any:
	# OpenCV API differences across versions
	if hasattr(aruco, "DetectorParameters"):
		return aruco.DetectorParameters()
	if hasattr(aruco, "DetectorParameters_create"):
		return aruco.DetectorParameters_create()
	return None


def decode_apriltags_in_roi(
	*,
	frame_bgr: np.ndarray,
	roi_xyxy: Sequence[int],
	tag_family: str = "36h11",
	detector_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Attempt to decode AprilTags inside an ROI.

	Args:
		frame_bgr: Full frame in BGR.
		roi_xyxy: ROI in full-frame pixel coords [x1,y1,x2,y2] (inclusive/exclusive not strict; we clamp).
		tag_family: AprilTag family string (e.g., "36h11").
		detector_params: Optional cv2.aruco.DetectorParameters overrides (best-effort).

	Returns:
		Dict with:
		  - detections: List[TagDetection]
		  - error: Optional[str]
		  - error_detail: Optional[str]
	"""

	cv2, aruco = _get_cv2_aruco()
	if cv2 is None:
		return {"detections": [], "error": "missing_cv2", "error_detail": "cv2 import failed"}
	if aruco is None:
		return {"detections": [], "error": "missing_cv2_aruco", "error_detail": "opencv-contrib not installed"}

	if frame_bgr is None or not hasattr(frame_bgr, "shape"):
		return {"detections": [], "error": "invalid_frame", "error_detail": "frame_bgr is None or invalid"}

	h, w = frame_bgr.shape[:2]
	try:
		x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
	except Exception:
		return {"detections": [], "error": "invalid_roi", "error_detail": f"roi_xyxy={roi_xyxy!r}"}

	# Clamp ROI
	x1 = max(0, min(w - 1, x1))
	y1 = max(0, min(h - 1, y1))
	x2 = max(0, min(w, x2))
	y2 = max(0, min(h, y2))

	if x2 <= x1 or y2 <= y1:
		return {"detections": [], "error": "empty_roi", "error_detail": f"clamped roi=[{x1},{y1},{x2},{y2}]"}

	roi = frame_bgr[y1:y2, x1:x2]
	if roi.size == 0:
		return {"detections": [], "error": "empty_roi", "error_detail": "roi crop empty"}

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

	dictionary = _dictionary_for_family(aruco, tag_family)
	if dictionary is None:
		return {"detections": [], "error": "unsupported_tag_family", "error_detail": str(tag_family)}

	params = _make_detector_params(aruco)
	if params is None:
		return {"detections": [], "error": "unsupported_aruco_api", "error_detail": "missing DetectorParameters"}

	# Best-effort apply parameter overrides for supported fields
	if isinstance(detector_params, dict):
		for k, v in detector_params.items():
			if hasattr(params, k):
				try:
					setattr(params, k, v)
				except Exception:
					pass

	try:
		if hasattr(aruco, "ArucoDetector"):
			detector = aruco.ArucoDetector(dictionary, params)
			corners, ids, _rejected = detector.detectMarkers(gray)
		else:
			corners, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=params)
	except Exception as e:
		return {"detections": [], "error": "detect_failed", "error_detail": f"{type(e).__name__}: {e}"}

	if ids is None or len(ids) == 0:
		return {"detections": [], "error": None, "error_detail": None}

	dets: List[TagDetection] = []
	try:
		ids_list = [int(i[0]) if hasattr(i, "__len__") else int(i) for i in ids]
	except Exception:
		ids_list = [int(i) for i in np.array(ids).reshape(-1).tolist()]

	for idx, tag_id in enumerate(ids_list):
		if idx >= len(corners):
			continue
		c = corners[idx]
		# c can be shape (1,4,2) or (4,2)
		arr = np.array(c, dtype=float)
		arr = arr.reshape(-1, 2)
		if arr.shape[0] < 4:
			continue
		pts = []
		for j in range(4):
			px, py = float(arr[j, 0]) + float(x1), float(arr[j, 1]) + float(y1)
			pts.append((px, py))
		dets.append(TagDetection(tag_id=int(tag_id), tag_family=str(tag_family), corners_px=pts))

	dets = sorted(dets, key=lambda d: int(d.tag_id))

	return {"detections": dets, "error": None, "error_detail": None}
