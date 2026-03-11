"""Privacy/redaction planning helpers for Stage F exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
