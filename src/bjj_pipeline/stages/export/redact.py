"""Privacy/redaction planning helpers for Stage F exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2  # type: ignore
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RedactionTarget:
	frame_index: int
	person_id: str
	method: str  # "mask" | "bbox"
	bbox_xyxy: tuple[int, int, int, int]
	mask_ref: str | None


@dataclass(frozen=True)
class RedactionPlan:
	enabled: bool
	mode: str
	export_id: str
	focus_person_ids: tuple[str, str]
	export_start_frame: int
	export_end_frame: int
	n_targets: int
	n_frames_with_targets: int
	n_mask_targets: int
	n_bbox_targets: int
	targets: tuple[RedactionTarget, ...]


@dataclass(frozen=True)
class RedactionRenderResult:
	output_video_path: str
	n_frames_written: int
	n_mask_targets_applied: int
	n_bbox_targets_applied: int


class RedactionRenderError(RuntimeError):
	"""Raised when privacy rendering fails."""


def _ensure_odd_kernel(value: int) -> int:
	k = max(3, int(value))
	return k if (k % 2 == 1) else (k + 1)


def _normalize_privacy_mode(value: Any) -> str:
	allowed = {"none", "blur_non_focus_bbox", "blur_non_focus_mask"}
	txt = str(value or "none").strip().lower()
	return txt if txt in allowed else "none"


def _required_track_columns() -> set[str]:
	return {"frame_index", "person_id", "x1", "y1", "x2", "y2"}


def _resolve_mask_ref_for_row(row: pd.Series) -> str | None:
	if "mask_ref" not in row.index:
		return None
	value = row.get("mask_ref")
	if value is None:
		return None
	txt = str(value).strip()
	return txt if txt else None


def _intersects_crop(
	bbox_xyxy: tuple[int, int, int, int],
	crop_xywh: tuple[int, int, int, int],
) -> bool:
	x1, y1, x2, y2 = bbox_xyxy
	crop_x, crop_y, crop_w, crop_h = crop_xywh
	crop_x2 = crop_x + crop_w
	crop_y2 = crop_y + crop_h
	return (x1 < crop_x2) and (x2 > crop_x) and (y1 < crop_y2) and (y2 > crop_y)


def _targets_by_frame(targets: Iterable[RedactionTarget]) -> dict[int, list[RedactionTarget]]:
	out: dict[int, list[RedactionTarget]] = {}
	for target in targets:
		out.setdefault(int(target.frame_index), []).append(target)
	return out


def build_redaction_plan(
	*,
	export_session: Any,
	crop_plan: Any,
	person_tracks_df: pd.DataFrame,
	privacy_mode: str,
	redact_non_focus_people: bool,
	redact_use_masks_when_available: bool,
	redact_fallback_to_bbox: bool,
) -> RedactionPlan:
	mode = _normalize_privacy_mode(privacy_mode)
	enabled = bool(redact_non_focus_people) and mode != "none"
	focus_person_ids = (
		str(export_session.person_id_a),
		str(export_session.person_id_b),
	)

	if not enabled:
		return RedactionPlan(
			enabled=False,
			mode="none",
			export_id=str(export_session.export_id),
			focus_person_ids=focus_person_ids,
			export_start_frame=int(export_session.export_start_frame),
			export_end_frame=int(export_session.export_end_frame),
			n_targets=0,
			n_frames_with_targets=0,
			n_mask_targets=0,
			n_bbox_targets=0,
			targets=tuple(),
		)

	required = _required_track_columns()
	missing = sorted(required - set(person_tracks_df.columns))
	if missing:
		raise ValueError(f"person_tracks.parquet missing required columns for redaction: {missing}")

	export_start = int(export_session.export_start_frame)
	export_end = int(export_session.export_end_frame)
	focus_set = {str(export_session.person_id_a), str(export_session.person_id_b)}
	crop_xywh = (
		int(crop_plan.x),
		int(crop_plan.y),
		int(crop_plan.width),
		int(crop_plan.height),
	)

	interval_df = person_tracks_df.loc[
		(person_tracks_df["frame_index"] >= export_start)
		& (person_tracks_df["frame_index"] <= export_end)
	].copy()

	if interval_df.empty:
		return RedactionPlan(
			enabled=True,
			mode=mode,
			export_id=str(export_session.export_id),
			focus_person_ids=focus_person_ids,
			export_start_frame=export_start,
			export_end_frame=export_end,
			n_targets=0,
			n_frames_with_targets=0,
			n_mask_targets=0,
			n_bbox_targets=0,
			targets=tuple(),
		)

	interval_df = interval_df.loc[~interval_df["person_id"].astype(str).isin(focus_set)].copy()
	if interval_df.empty:
		return RedactionPlan(
			enabled=True,
			mode=mode,
			export_id=str(export_session.export_id),
			focus_person_ids=focus_person_ids,
			export_start_frame=export_start,
			export_end_frame=export_end,
			n_targets=0,
			n_frames_with_targets=0,
			n_mask_targets=0,
			n_bbox_targets=0,
			targets=tuple(),
		)

	interval_df = interval_df.sort_values(["frame_index", "person_id"], kind="mergesort")
	targets: list[RedactionTarget] = []

	for _, row in interval_df.iterrows():
		bbox_xyxy = (
			int(round(float(row["x1"]))),
			int(round(float(row["y1"]))),
			int(round(float(row["x2"]))),
			int(round(float(row["y2"]))),
		)
		if not _intersects_crop(bbox_xyxy, crop_xywh):
			continue

		mask_ref = _resolve_mask_ref_for_row(row)
		method: str | None = None
		if redact_use_masks_when_available and mask_ref is not None:
			method = "mask"
		elif redact_fallback_to_bbox:
			method = "bbox"

		if method is None:
			continue

		targets.append(
			RedactionTarget(
				frame_index=int(row["frame_index"]),
				person_id=str(row["person_id"]),
				method=method,
				bbox_xyxy=bbox_xyxy,
				mask_ref=mask_ref,
			)
		)

	n_mask_targets = sum(1 for t in targets if t.method == "mask")
	n_bbox_targets = sum(1 for t in targets if t.method == "bbox")
	n_frames_with_targets = len({t.frame_index for t in targets})

	return RedactionPlan(
		enabled=True,
		mode=mode,
		export_id=str(export_session.export_id),
		focus_person_ids=focus_person_ids,
		export_start_frame=export_start,
		export_end_frame=export_end,
		n_targets=len(targets),
		n_frames_with_targets=n_frames_with_targets,
		n_mask_targets=n_mask_targets,
		n_bbox_targets=n_bbox_targets,
		targets=tuple(targets),
	)


def _resolve_mask_path(mask_ref: str | None) -> Path | None:
	if mask_ref is None:
		return None
	p = Path(str(mask_ref))
	if p.exists():
		return p
	if not p.is_absolute():
		rp = Path.cwd() / p
		if rp.exists():
			return rp
	return None


def _load_mask_npz(mask_path: Path) -> np.ndarray:
	with np.load(mask_path, allow_pickle=False) as payload:
		if "mask" in payload:
			arr = payload["mask"]
		elif "arr_0" in payload:
			arr = payload["arr_0"]
		else:
			keys = list(payload.keys())
			if not keys:
				raise RedactionRenderError(f"mask npz contains no arrays: {mask_path}")
			arr = payload[keys[0]]
	arr = np.asarray(arr)
	if arr.ndim != 2:
		raise RedactionRenderError(f"expected 2D mask array, got shape={arr.shape} path={mask_path}")
	return arr.astype(bool)


def _project_bbox_to_crop(
	bbox_xyxy: tuple[int, int, int, int],
	crop_xywh: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
	x1, y1, x2, y2 = bbox_xyxy
	crop_x, crop_y, crop_w, crop_h = crop_xywh
	px1 = max(0, x1 - crop_x)
	py1 = max(0, y1 - crop_y)
	px2 = min(crop_w, x2 - crop_x)
	py2 = min(crop_h, y2 - crop_y)
	if px2 <= px1 or py2 <= py1:
		return None
	return (int(px1), int(py1), int(px2), int(py2))


def _resize_mask_to_bbox(mask: np.ndarray, width: int, height: int) -> np.ndarray:
	resized = cv2.resize(
		mask.astype(np.uint8),
		(int(width), int(height)),
		interpolation=cv2.INTER_NEAREST,
	)
	return resized.astype(bool)


def _project_mask_to_crop(
	*,
	local_mask: np.ndarray,
	bbox_xyxy: tuple[int, int, int, int],
	crop_xywh: tuple[int, int, int, int],
	source_frame_shape: tuple[int, int],
) -> np.ndarray | None:
	"""
	Project a mask into crop space.

	# NOTE:
	# Stage A produces YOLO segmentation masks stored in .npz files.
	# In practice we observed mask alignment inconsistencies when projecting
	# them into Stage F crop space. For Privacy v1 we therefore rely on the
	# more robust bbox blur fallback which uses the stitched person_tracks
	# bounding boxes.
	#
	# Mask-based blur remains implemented but is not relied upon for the
	# default redaction path until mask projection is revisited in a future
	# revision.

	Stage A currently writes canonical masks as full-frame uint8 {0,1} arrays.
	For robustness, we also retain support for legacy/local masks that need
	resizing into the target bbox.
	"""
	crop_x, crop_y, crop_w, crop_h = crop_xywh
	frame_h, frame_w = source_frame_shape

	mask = np.asarray(local_mask).astype(bool)

	# Preferred path: canonical full-frame mask from Stage A.
	if mask.ndim == 2 and int(mask.shape[0]) == int(frame_h) and int(mask.shape[1]) == int(frame_w):
		return mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()

	# Fallback path: legacy / local mask associated with bbox region.
	proj = _project_bbox_to_crop(bbox_xyxy, crop_xywh)
	if proj is None:
		return None
	px1, py1, px2, py2 = proj
	width = int(px2 - px1)
	height = int(py2 - py1)
	if width <= 0 or height <= 0:
		return None
	resized = _resize_mask_to_bbox(mask, width, height)
	out = np.zeros((int(crop_h), int(crop_w)), dtype=bool)
	out[py1:py2, px1:px2] = resized
	return out


def _apply_mask_blur(
	crop_frame: np.ndarray,
	crop_mask: np.ndarray,
	blur_kernel_size: int,
) -> np.ndarray:
	if not np.any(crop_mask):
		return crop_frame
	k = _ensure_odd_kernel(blur_kernel_size)
	blurred = cv2.GaussianBlur(crop_frame, (k, k), 0)
	out = crop_frame.copy()
	mask_bool = np.asarray(crop_mask).astype(bool)
	out[mask_bool] = blurred[mask_bool]
	return out


def _apply_bbox_blur(
	crop_frame: np.ndarray,
	crop_bbox_xyxy: tuple[int, int, int, int],
	blur_kernel_size: int,
) -> np.ndarray:
	x1, y1, x2, y2 = crop_bbox_xyxy
	if x2 <= x1 or y2 <= y1:
		return crop_frame
	k = _ensure_odd_kernel(blur_kernel_size)
	out = crop_frame.copy()
	roi = out[y1:y2, x1:x2]
	if roi.size == 0:
		return out
	out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
	return out


def render_redacted_clip(
	*,
	input_video_path: Path,
	output_video_path: Path,
	crop_plan: Any,
	redaction_plan: RedactionPlan,
	fps: float,
	export_start_frame: int,
	export_end_frame: int,
	blur_kernel_size: int = 31,
) -> RedactionRenderResult:
	if fps <= 0.0:
		raise RedactionRenderError("fps must be positive for redacted rendering")

	cap = cv2.VideoCapture(str(input_video_path))
	if not cap.isOpened():
		raise RedactionRenderError(f"unable to open input video: {input_video_path}")

	try:
		output_video_path.parent.mkdir(parents=True, exist_ok=True)
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(
			str(output_video_path),
			fourcc,
			float(fps),
			(int(crop_plan.width), int(crop_plan.height)),
		)
		if not writer.isOpened():
			raise RedactionRenderError(f"unable to open video writer: {output_video_path}")

		targets_by_frame = _targets_by_frame(redaction_plan.targets)
		crop_xywh = (
			int(crop_plan.x),
			int(crop_plan.y),
			int(crop_plan.width),
			int(crop_plan.height),
		)

		cap.set(cv2.CAP_PROP_POS_FRAMES, int(export_start_frame))
		n_frames_written = 0
		n_mask_targets_applied = 0
		n_bbox_targets_applied = 0

		for frame_index in range(int(export_start_frame), int(export_end_frame) + 1):
			ok, frame = cap.read()
			if not ok or frame is None:
				break
			frame_h, frame_w = frame.shape[:2]

			cx, cy, cw, ch = crop_xywh
			crop_frame = frame[cy : cy + ch, cx : cx + cw].copy()
			frame_targets = targets_by_frame.get(int(frame_index), [])

			for target in frame_targets:
				if target.method == "mask":
					try:
						mask_path = _resolve_mask_path(target.mask_ref)
						if mask_path is None:
							raise RedactionRenderError(f"unresolvable mask_ref={target.mask_ref}")
						local_mask = _load_mask_npz(mask_path)
						crop_mask = _project_mask_to_crop(
							local_mask=local_mask,
							bbox_xyxy=target.bbox_xyxy,
							crop_xywh=crop_xywh,
							source_frame_shape=(int(frame_h), int(frame_w)),
						)
						if crop_mask is None:
							raise RedactionRenderError("mask projection returned None")
						crop_frame = _apply_mask_blur(
							crop_frame=crop_frame,
							crop_mask=crop_mask,
							blur_kernel_size=blur_kernel_size,
						)
						n_mask_targets_applied += 1
					except Exception:
						proj_bbox = _project_bbox_to_crop(target.bbox_xyxy, crop_xywh)
						if proj_bbox is not None:
							crop_frame = _apply_bbox_blur(
								crop_frame=crop_frame,
								crop_bbox_xyxy=proj_bbox,
								blur_kernel_size=blur_kernel_size,
							)
							n_bbox_targets_applied += 1
				elif target.method == "bbox":
					proj_bbox = _project_bbox_to_crop(target.bbox_xyxy, crop_xywh)
					if proj_bbox is not None:
						crop_frame = _apply_bbox_blur(
							crop_frame=crop_frame,
							crop_bbox_xyxy=proj_bbox,
							blur_kernel_size=blur_kernel_size,
						)
						n_bbox_targets_applied += 1

			writer.write(crop_frame)
			n_frames_written += 1

		writer.release()
		if not output_video_path.exists():
			raise RedactionRenderError(f"redacted output file was not created: {output_video_path}")
		return RedactionRenderResult(
			output_video_path=str(output_video_path),
			n_frames_written=int(n_frames_written),
			n_mask_targets_applied=int(n_mask_targets_applied),
			n_bbox_targets_applied=int(n_bbox_targets_applied),
		)
	finally:
		cap.release()


def summarize_redaction_plan(plan: RedactionPlan) -> Dict[str, Any]:
	return {
		"enabled": bool(plan.enabled),
		"mode": str(plan.mode),
		"focus_person_ids": list(plan.focus_person_ids),
		"n_targets": int(plan.n_targets),
		"n_frames_with_targets": int(plan.n_frames_with_targets),
		"n_mask_targets": int(plan.n_mask_targets),
		"n_bbox_targets": int(plan.n_bbox_targets),
	}
