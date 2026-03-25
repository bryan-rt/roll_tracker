"""Stage A (detect_track) — deterministic quality/geometry helpers.

This module intentionally contains *pure functions* (no IO, no global state).
It is used by StageAProcessor to compute:
  - mask gating/quality
  - contact point estimation
  - homography projection to mat-space (x_m, y_m)
  - on-mat classification
  - simple kinematics + physics plausibility scoring (audit-only initially)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _clip_int(v: int, lo: int, hi: int) -> int:
	return max(lo, min(hi, v))


def bbox_fallback_mask(
	frame_shape_hw: Tuple[int, int],
	bbox_xyxy: Tuple[float, float, float, float],
) -> np.ndarray:
	"""Deterministic bbox mask fallback: filled rectangle on a full-frame mask.

	Returns uint8 mask with values {0,1} in full-frame coordinates.
	"""
	h, w = frame_shape_hw
	x1, y1, x2, y2 = bbox_xyxy
	x1i = _clip_int(int(np.floor(x1)), 0, w - 1)
	y1i = _clip_int(int(np.floor(y1)), 0, h - 1)
	x2i = _clip_int(int(np.ceil(x2)), 0, w)
	y2i = _clip_int(int(np.ceil(y2)), 0, h)

	mask = np.zeros((h, w), dtype=np.uint8)
	if x2i <= x1i or y2i <= y1i:
		return mask
	mask[y1i:y2i, x1i:x2i] = 1
	return mask


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
	"""Compute a tight xyxy bbox from a full-frame mask.

	Returns None if mask is empty.
	"""
	if mask.size == 0:
		return None
	ys, xs = np.where(mask.astype(bool))
	if ys.size == 0:
		return None
	x1 = float(xs.min())
	y1 = float(ys.min())
	x2 = float(xs.max() + 1)
	y2 = float(ys.max() + 1)
	return (x1, y1, x2, y2)


def _bbox_area(b: Tuple[float, float, float, float]) -> float:
	x1, y1, x2, y2 = b
	return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	union = _bbox_area(a) + _bbox_area(b) - inter
	if union <= 0.0:
		return 0.0
	return float(inter / union)


def compute_mask_quality(
	mask: np.ndarray,
	bbox_xyxy: Tuple[float, float, float, float],
	*,
	min_area_frac: float = 0.10,
	max_area_frac: float = 1.10,
) -> float:
	"""Compute a deterministic scalar quality score in [0,1].

	Heuristic (POC-friendly, monotonic):
	  - empty mask => 0
	  - area sanity: reward mask area close to bbox area and within [min,max] multiples
	  - overlap sanity: reward IoU between bbox and mask-derived bbox
	"""
	if mask.size == 0:
		return 0.0
	m = mask.astype(bool)
	area = float(m.sum())
	if area <= 0.0:
		return 0.0

	mb = bbox_from_mask(mask)
	if mb is None:
		return 0.0

	bbox_area = _bbox_area(bbox_xyxy)
	if bbox_area <= 0.0:
		return 0.0

	area_ratio = area / bbox_area
	# map ratio into [0,1] with a plateau in the "reasonable" range
	if area_ratio < min_area_frac:
		area_score = float(area_ratio / max(1e-9, min_area_frac))
	elif area_ratio > max_area_frac:
		# decay as ratio exceeds max; clamp to 0 at 2*max
		over = area_ratio - max_area_frac
		denom = max(1e-9, max_area_frac)
		area_score = float(max(0.0, 1.0 - (over / denom)))
	else:
		area_score = 1.0

	iou = _iou_xyxy(bbox_xyxy, mb)
	# blend: IoU is strong signal; area sanity protects against degenerate masks
	q = 0.25 * area_score + 0.75 * iou
	return float(max(0.0, min(1.0, q)))


def mask_passes_gate(
	*,
	det_conf: float,
	mask_quality: float,
	gate_cfg: Dict[str, Any],
) -> bool:
	det_min = float(gate_cfg.get("det_conf_min", 0.0))
	q_min = float(gate_cfg.get("mask_quality_min", 0.0))
	return (det_conf >= det_min) and (mask_quality >= q_min)


def contact_point_from_bbox(
	bbox_xyxy: Tuple[float, float, float, float],
	*,
	det_conf: float = 1.0,
) -> Tuple[float, float, str, float]:
	"""Bottom-center of bbox."""
	x1, y1, x2, y2 = bbox_xyxy
	u = 0.5 * (x1 + x2)
	v = float(y2)
	conf = float(max(0.0, min(1.0, det_conf)))
	return (u, v, "bbox", conf)


def contact_point_from_mask(
	mask: np.ndarray,
	bbox_xyxy: Tuple[float, float, float, float],
	*,
	det_conf: float = 1.0,
	mask_quality: float = 1.0,
) -> Tuple[float, float, str, float]:
	"""Estimate a ground-contact point from a binary mask.

	Deterministic definition:
	  - take the bottom-most mask row (max y)
	  - choose median x among mask pixels at that row
	  - fallback to bbox bottom-center if mask is empty
	"""
	m = mask.astype(bool)
	ys, xs = np.where(m)
	if ys.size == 0:
		return contact_point_from_bbox(bbox_xyxy, det_conf=det_conf)
	y_max = int(ys.max())
	xs_at = xs[ys == y_max]
	if xs_at.size == 0:
		return contact_point_from_bbox(bbox_xyxy, det_conf=det_conf)
	u = float(np.median(xs_at))
	v = float(y_max)
	conf = float(max(0.0, min(1.0, 0.5 * det_conf + 0.5 * mask_quality)))
	return (u, v, "yolo_mask", conf)


def project_uv_to_xy(H: np.ndarray, u_px: float, v_px: float) -> Tuple[float, float]:
	"""Apply 3x3 homography H to pixel point (u,v) -> (x,y) in world/mat space.

	.. deprecated:: CP16a
		Use ``bjj_pipeline.contracts.f0_projection.project_to_world()`` instead.
		That utility adds optional lens undistortion before homography.
	"""
	p = np.array([u_px, v_px, 1.0], dtype=np.float64)
	q = H @ p
	if float(q[2]) == 0.0:
		return (float("nan"), float("nan"))
	x = float(q[0] / q[2])
	y = float(q[1] / q[2])
	return (x, y)


def _point_in_poly(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
	"""Ray casting point-in-polygon (deterministic)."""
	n = len(poly)
	if n < 3:
		return False
	inside = False
	x0, y0 = poly[-1]
	for x1, y1 in poly:
		# check if edge crosses horizontal ray at y
		cond = (y1 > y) != (y0 > y)
		if cond:
			x_int = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
			if x < x_int:
				inside = not inside
		x0, y0 = x1, y1
	return inside


def point_in_mat(x_m: float, y_m: float, blueprint: Any) -> bool:
	"""Determine whether (x_m,y_m) lies on the mat region defined by blueprint.

	Supports:
	  - rectangle(s): {"mats":[{"x_min":..,"x_max":..,"y_min":..,"y_max":..}, ...]}
	  - rectangle list (repo default): [{"x":..,"y":..,"width":..,"height":.., ...}, ...]
	  - polygon(s): {"polygons":[[(x,y),...], ...]}  (or list of dicts with "points")
	"""
	if not np.isfinite(x_m) or not np.isfinite(y_m):
		return False

	# Normalize blueprint into a dict-like structure when the repo provides a list-of-rects.
	# configs/mat_blueprint.json is currently a list of dicts with x,y,width,height (see viz/mat_view.py).
	if isinstance(blueprint, list):
		mats_list = []
		for r in blueprint:
			if not isinstance(r, dict):
				continue
			# Prefer explicit bounds if present
			if all(k in r for k in ("x_min", "x_max", "y_min", "y_max")):
				try:
					mats_list.append(
						{
							"x_min": float(r["x_min"]),
							"x_max": float(r["x_max"]),
							"y_min": float(r["y_min"]),
							"y_max": float(r["y_max"]),
						}
					)
				except Exception:
					continue
				continue

			# Common rectangle format: x,y,width,height (with some configs using w/h keys)
			x = r.get("x", r.get("x_min", None))
			y = r.get("y", r.get("y_min", None))
			w = r.get("width", r.get("w", None))
			h = r.get("height", r.get("h", None))
			if x is None or y is None or w is None or h is None:
				continue
			try:
				x0 = float(x)
				y0 = float(y)
				w0 = float(w)
				h0 = float(h)
			except Exception:
				continue
			mats_list.append(
				{
					"x_min": x0,
					"x_max": x0 + w0,
					"y_min": y0,
					"y_max": y0 + h0,
				}
			)
		blueprint = {"mats": mats_list}

	# If blueprint is not dict-like even after normalization, treat as "unknown -> False"
	if not isinstance(blueprint, dict):
		return False

	mats = blueprint.get("mats")
	if isinstance(mats, list):
		for m in mats:
			try:
				xmin = float(m.get("x_min"))
				xmax = float(m.get("x_max"))
				ymin = float(m.get("y_min"))
				ymax = float(m.get("y_max"))
			except Exception:
				continue
			if xmin <= x_m <= xmax and ymin <= y_m <= ymax:
				return True

	polys = blueprint.get("polygons")
	if isinstance(polys, list):
		for p in polys:
			pts = None
			if isinstance(p, dict) and "points" in p:
				pts = p["points"]
			elif isinstance(p, (list, tuple)):
				pts = p
			if not isinstance(pts, (list, tuple)):
				continue
			try:
				poly = [(float(a), float(b)) for (a, b) in pts]
			except Exception:
				continue
			if _point_in_poly(x_m, y_m, poly):
				return True

	return False


def compute_velocity(
	prev_xy: Tuple[float, float],
	prev_t_ms: int,
	xy: Tuple[float, float],
	t_ms: int,
) -> Tuple[float, float, float]:
	"""Compute (vx, vy, speed) in m/s given consecutive positions and timestamps."""
	x0, y0 = prev_xy
	x1, y1 = xy
	dt = (t_ms - prev_t_ms) / 1000.0
	if dt <= 0.0 or (not np.isfinite(dt)):
		return (float("nan"), float("nan"), float("nan"))
	vx = (x1 - x0) / dt
	vy = (y1 - y0) / dt
	speed = float(np.hypot(vx, vy))
	return (float(vx), float(vy), speed)


def is_physics_warning(speed_mps: float, max_speed_mps: float) -> bool:
	"""True if the speed exceeds threshold (audit-only initially)."""
	if not np.isfinite(speed_mps):
		return True
	return speed_mps > max_speed_mps
