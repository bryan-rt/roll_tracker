"""
Role: unit test for Stage D0 Checkpoint 3 (CP3) kinematics computation.

CP3 is flag-only:
- compute dt-aware velocity + accel from effective world coords (repaired-or-original)
- flag implausible speeds/accels, but do NOT clamp/suppress

CP4.B (site #5): dt_s derived from timestamp_ms instead of df/fps. int-ms precision
is lossless on post-R13a footage (frame_index_join_1/findings.md §10 precision finding).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bjj_pipeline.stages.stitch.d0_bank import _apply_cp3_kinematics


def test_d0_cp3_kinematics_flags_and_effective_coords_selection() -> None:
	"""Original CP3 test, updated for CP4.B (timestamp_ms instead of fps).

	Fixture: frame spacing = 1, timestamp_ms spacing = 100ms (10 fps equivalent).
	At 100ms spacing, dt_s = 0.1s — same as the old df/fps = 1/10 path.
	Expected values unchanged.
	"""
	# Three tracklets:
	# - t1: has a single bad x_m at frame 2, but CP2 repaired it (should NOT be flagged)
	# - t2: has a teleport step (should trigger speed flag)
	# - t3: speeds are plausible, but acceleration spike should trigger accel flag (not speed flag)
	tf = pd.DataFrame(
		[
			# t1 (frame 2 is repaired)
			{"tracklet_id": "t1", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t1", "frame_index": 1, "timestamp_ms": 100, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},
			{
				"tracklet_id": "t1",
				"frame_index": 2,
				"timestamp_ms": 200,
				"detection_id": "d2",
				"x_m": 100.0,
				"y_m": 0.0,
				"x_m_repaired": 0.2,
				"y_m_repaired": 0.0,
				"is_repaired": True,
			},
			{"tracklet_id": "t1", "frame_index": 3, "timestamp_ms": 300, "detection_id": "d3", "x_m": 0.3, "y_m": 0.0, "is_repaired": False},
			# t2 (teleport not repaired)
			{"tracklet_id": "t2", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t2", "frame_index": 1, "timestamp_ms": 100, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t2", "frame_index": 2, "timestamp_ms": 200, "detection_id": "d2", "x_m": 10.0, "y_m": 0.0, "is_repaired": False},
			# t3 (accel spike but speed plausible)
			{"tracklet_id": "t3", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t3", "frame_index": 1, "timestamp_ms": 100, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},  # speed 1 m/s
			{"tracklet_id": "t3", "frame_index": 2, "timestamp_ms": 200, "detection_id": "d2", "x_m": 0.6, "y_m": 0.0, "is_repaired": False},  # speed 5 m/s, accel 40 m/s^2
		]
	)

	out, summary = _apply_cp3_kinematics(
		tf,
		kin_cfg={"enabled": True, "v_max_mps": 8.0, "a_max_mps2": 12.0},
	)

	# Sanity: summary present and enabled
	assert summary["enabled"] is True
	assert summary["n_tracklets"] == 3

	# t1: repaired frame should eliminate the bad coordinate jump
	t1_f2 = out[(out.tracklet_id == "t1") & (out.frame_index == 2)].iloc[0]
	assert math.isfinite(t1_f2.speed_mps_k)
	assert abs(t1_f2.speed_mps_k - 1.0) < 1e-6
	assert bool(t1_f2.speed_is_implausible) is False

	# t2: teleport step should trigger speed flag
	t2_f2 = out[(out.tracklet_id == "t2") & (out.frame_index == 2)].iloc[0]
	assert math.isfinite(t2_f2.speed_mps_k)
	assert abs(t2_f2.speed_mps_k - 99.0) < 1e-6  # dx=9.9, dt=0.1
	assert bool(t2_f2.speed_is_implausible) is True

	# t3: accel spike should trigger accel flag while speed remains plausible
	t3_f2 = out[(out.tracklet_id == "t3") & (out.frame_index == 2)].iloc[0]
	assert math.isfinite(t3_f2.speed_mps_k)
	assert abs(t3_f2.speed_mps_k - 5.0) < 1e-6
	assert bool(t3_f2.speed_is_implausible) is False
	assert math.isfinite(t3_f2.accel_mps2_k)
	assert abs(t3_f2.accel_mps2_k - 40.0) < 1e-6
	assert bool(t3_f2.accel_is_implausible) is True

	# First row of each tracklet has NaN velocity and flags False
	first_rows = (
		out.sort_values(["tracklet_id", "frame_index", "detection_id"])
		.groupby("tracklet_id", as_index=False)
		.head(1)
	)
	assert np.all(~np.isfinite(first_rows["speed_mps_k"].to_numpy()))
	assert np.all(first_rows["speed_is_implausible"].to_numpy() == False)
	assert np.all(first_rows["accel_is_implausible"].to_numpy() == False)


def test_uniform_timestamp_equivalence() -> None:
	"""dt_s from timestamp_ms matches the old df/fps path on uniform data.

	Fixture: timestamp_ms spacing = 50ms (20 fps equivalent). Chosen because
	50ms / 1000.0 = 0.05 and 1 / 20.0 = 0.05 are both exactly representable,
	so comparison is exact (atol=0).
	"""
	tf = pd.DataFrame([
		{"tracklet_id": "t1", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 1, "timestamp_ms": 50, "detection_id": "d1", "x_m": 0.5, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 2, "timestamp_ms": 100, "detection_id": "d2", "x_m": 1.0, "y_m": 0.0},
		{"tracklet_id": "t2", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0},
		{"tracklet_id": "t2", "frame_index": 1, "timestamp_ms": 50, "detection_id": "d1", "x_m": 0.0, "y_m": 0.3},
	])

	out, summary = _apply_cp3_kinematics(tf, kin_cfg={"enabled": True})

	# t1 frame 1: dx=0.5m in 0.05s = 10.0 m/s
	t1_f1 = out[(out.tracklet_id == "t1") & (out.frame_index == 1)].iloc[0]
	assert t1_f1.speed_mps_k == 10.0  # exact

	# t1 frame 2: dx=0.5m in 0.05s = 10.0 m/s
	t1_f2 = out[(out.tracklet_id == "t1") & (out.frame_index == 2)].iloc[0]
	assert t1_f2.speed_mps_k == 10.0  # exact

	# t2 frame 1: dy=0.3m in 0.05s = 6.0 m/s (np.hypot float precision)
	t2_f1 = out[(out.tracklet_id == "t2") & (out.frame_index == 1)].iloc[0]
	assert abs(t2_f1.speed_mps_k - 6.0) < 1e-12

	assert summary["n_bad_dt_steps"] == 0


def test_nonuniform_timestamp_velocity() -> None:
	"""Variable timestamp_ms spacing produces correct velocities.

	Two steps: 50ms then 100ms. dx=0.5m each. Expected:
	  step 1: speed = 0.5 / 0.050 = 10.0 m/s
	  step 2: speed = 0.5 / 0.100 = 5.0 m/s
	  accel at step 2: |5.0 - 10.0| / 0.100 = 50.0 m/s²
	"""
	tf = pd.DataFrame([
		{"tracklet_id": "t1", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 1, "timestamp_ms": 50, "detection_id": "d1", "x_m": 0.5, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 2, "timestamp_ms": 150, "detection_id": "d2", "x_m": 1.0, "y_m": 0.0},
	])

	out, summary = _apply_cp3_kinematics(tf, kin_cfg={"enabled": True})

	f1 = out[out.frame_index == 1].iloc[0]
	f2 = out[out.frame_index == 2].iloc[0]

	assert f1.speed_mps_k == 10.0  # 0.5m / 0.05s
	assert f2.speed_mps_k == 5.0   # 0.5m / 0.10s
	assert abs(f2.accel_mps2_k - 50.0) < 1e-9  # |5-10| / 0.10

	assert summary["n_bad_dt_steps"] == 0


def test_zero_dt_no_raise() -> None:
	"""Two frames with same timestamp_ms: no raise, counter increments.

	Frame 0: ts=0, frame 1: ts=100, frame 2: ts=100 (duplicate-PTS).
	The step from frame 1→2 has dt_ms=0. The dt_ms <= 0 guard should
	continue past it without computing speed — so speed at frame 2
	retains its NaN initialisation (speed array is filled with np.nan
	before the loop, d0_bank.py:551). The counter n_bad_dt_steps
	increments.
	"""
	tf = pd.DataFrame([
		{"tracklet_id": "t1", "frame_index": 0, "timestamp_ms": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 1, "timestamp_ms": 100, "detection_id": "d1", "x_m": 0.5, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 2, "timestamp_ms": 100, "detection_id": "d2", "x_m": 1.0, "y_m": 0.0},
		{"tracklet_id": "t1", "frame_index": 3, "timestamp_ms": 200, "detection_id": "d3", "x_m": 1.5, "y_m": 0.0},
	])

	# Must not raise
	out, summary = _apply_cp3_kinematics(tf, kin_cfg={"enabled": True})

	# Zero-dt step counted
	assert summary["n_bad_dt_steps"] == 1

	# Frame 2: speed was not computed (guard continued before write)
	f2_speed = out[out.frame_index == 2].iloc[0].speed_mps_k
	assert not np.isfinite(f2_speed), f"Expected non-finite speed at frame 2, got {f2_speed}"

	# Frame 1 and 3: speed IS computed normally
	f1_speed = out[out.frame_index == 1].iloc[0].speed_mps_k
	f3_speed = out[out.frame_index == 3].iloc[0].speed_mps_k
	assert f1_speed == 5.0   # 0.5m / 0.1s
	assert f3_speed == 5.0   # 0.5m / 0.1s

	# No inf anywhere
	assert not np.any(np.isinf(out["speed_mps_k"].to_numpy()))
	assert not np.any(np.isinf(out["accel_mps2_k"].to_numpy()))
