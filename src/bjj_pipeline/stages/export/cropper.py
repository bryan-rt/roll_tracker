"""Crop planning utilities for Stage F match clip exports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class FixedRoiCropPlan:
	mode: str
	x: int
	y: int
	width: int
	height: int
	start_frame: int
	end_frame: int
	person_id_a: str
	person_id_b: str
	n_track_rows: int
	n_pair_frames: int
	envelope_method: str
	padding_px: int


class CropPlanError(RuntimeError):
	"""Raised when a match crop cannot be planned deterministically."""


def _compute_pair_boxes(
	tracks_df: pd.DataFrame,
	person_id_a: str,
	person_id_b: str,
	start_frame: int,
	end_frame: int,
) -> pd.DataFrame:
	mask = (
		tracks_df["person_id"].isin([person_id_a, person_id_b])
		& (tracks_df["frame_index"] >= int(start_frame))
		& (tracks_df["frame_index"] <= int(end_frame))
	)
	rows = tracks_df.loc[mask, ["frame_index", "person_id", "x1", "y1", "x2", "y2"]].copy()
	if rows.empty:
		raise CropPlanError(
			f"no person_tracks rows for pair=({person_id_a},{person_id_b}) frames=[{start_frame},{end_frame}]"
		)

	rows = rows.sort_values(["frame_index", "person_id"], kind="mergesort")

	out_rows = []
	for frame_index, frame_df in rows.groupby("frame_index", sort=True):
		x1 = float(frame_df["x1"].min())
		y1 = float(frame_df["y1"].min())
		x2 = float(frame_df["x2"].max())
		y2 = float(frame_df["y2"].max())
		n_people_present = int(frame_df["person_id"].nunique())
		out_rows.append(
			{
				"frame_index": int(frame_index),
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"n_people_present": n_people_present,
			}
		)

	pair_boxes = pd.DataFrame(out_rows)
	if pair_boxes.empty:
		raise CropPlanError(
			f"no usable pair boxes for pair=({person_id_a},{person_id_b}) frames=[{start_frame},{end_frame}]"
		)
	return pair_boxes


def _expand_to_min_size(
	x1: float,
	y1: float,
	x2: float,
	y2: float,
	*,
	min_crop_width: int,
	min_crop_height: int,
) -> Tuple[float, float, float, float]:
	width = x2 - x1
	height = y2 - y1
	if width < float(min_crop_width):
		cx = 0.5 * (x1 + x2)
		half_w = 0.5 * float(min_crop_width)
		x1 = cx - half_w
		x2 = cx + half_w
	if height < float(min_crop_height):
		cy = 0.5 * (y1 + y2)
		half_h = 0.5 * float(min_crop_height)
		y1 = cy - half_h
		y2 = cy + half_h
	return x1, y1, x2, y2


def _clamp_box(
	x1: float,
	y1: float,
	x2: float,
	y2: float,
	*,
	frame_width: int,
	frame_height: int,
) -> Tuple[int, int, int, int]:
	fw = int(frame_width)
	fh = int(frame_height)
	x1i = int(math.floor(x1))
	y1i = int(math.floor(y1))
	x2i = int(math.ceil(x2))
	y2i = int(math.ceil(y2))

	if x1i < 0:
		x2i += -x1i
		x1i = 0
	if y1i < 0:
		y2i += -y1i
		y1i = 0
	if x2i > fw:
		shift = x2i - fw
		x1i -= shift
		x2i = fw
	if y2i > fh:
		shift = y2i - fh
		y1i -= shift
		y2i = fh

	x1i = max(0, x1i)
	y1i = max(0, y1i)
	x2i = min(fw, x2i)
	y2i = min(fh, y2i)

	width = x2i - x1i
	height = y2i - y1i
	if width <= 0 or height <= 0:
		raise CropPlanError(
			f"invalid clamped crop rectangle x1={x1i} y1={y1i} x2={x2i} y2={y2i}"
		)
	return x1i, y1i, width, height


def plan_crop_fixed_roi(
	*,
	tracks_df: pd.DataFrame,
	person_id_a: str,
	person_id_b: str,
	start_frame: int,
	end_frame: int,
	frame_width: int,
	frame_height: int,
	padding_px: int = 80,
	low_quantile: float = 0.05,
	high_quantile: float = 0.95,
	min_crop_width: int = 160,
	min_crop_height: int = 160,
) -> FixedRoiCropPlan:
	required_cols = {"person_id", "frame_index", "x1", "y1", "x2", "y2"}
	missing = sorted(required_cols - set(tracks_df.columns))
	if missing:
		raise CropPlanError(f"person_tracks missing required columns: {missing}")

	pair_boxes = _compute_pair_boxes(
		tracks_df=tracks_df,
		person_id_a=person_id_a,
		person_id_b=person_id_b,
		start_frame=start_frame,
		end_frame=end_frame,
	)

	env_x1 = float(pair_boxes["x1"].quantile(low_quantile, interpolation="linear"))
	env_y1 = float(pair_boxes["y1"].quantile(low_quantile, interpolation="linear"))
	env_x2 = float(pair_boxes["x2"].quantile(high_quantile, interpolation="linear"))
	env_y2 = float(pair_boxes["y2"].quantile(high_quantile, interpolation="linear"))

	x1 = env_x1 - float(padding_px)
	y1 = env_y1 - float(padding_px)
	x2 = env_x2 + float(padding_px)
	y2 = env_y2 + float(padding_px)

	x1, y1, x2, y2 = _expand_to_min_size(
		x1,
		y1,
		x2,
		y2,
		min_crop_width=min_crop_width,
		min_crop_height=min_crop_height,
	)
	crop_x, crop_y, crop_w, crop_h = _clamp_box(
		x1,
		y1,
		x2,
		y2,
		frame_width=frame_width,
		frame_height=frame_height,
	)

	return FixedRoiCropPlan(
		mode="fixed_roi",
		x=crop_x,
		y=crop_y,
		width=crop_w,
		height=crop_h,
		start_frame=int(start_frame),
		end_frame=int(end_frame),
		person_id_a=str(person_id_a),
		person_id_b=str(person_id_b),
		n_track_rows=int(
			(
				tracks_df["person_id"].isin([person_id_a, person_id_b])
				& (tracks_df["frame_index"] >= int(start_frame))
				& (tracks_df["frame_index"] <= int(end_frame))
			).sum()
		),
		n_pair_frames=int(len(pair_boxes)),
		envelope_method=f"quantile_{low_quantile:.2f}_{high_quantile:.2f}",
		padding_px=int(padding_px),
	)
