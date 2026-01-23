"""
Role: unit test for Stage D0 Checkpoint 3 (CP3) kinematics computation.

CP3 is flag-only:
- compute dt-aware velocity + accel from effective world coords (repaired-or-original)
- flag implausible speeds/accels, but do NOT clamp/suppress
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bjj_pipeline.stages.stitch.d0_bank import _apply_cp3_kinematics


def test_d0_cp3_kinematics_flags_and_effective_coords_selection() -> None:
	fps = 10.0  # dt_s = df / fps

	# Three tracklets:
	# - t1: has a single bad x_m at frame 2, but CP2 repaired it (should NOT be flagged)
	# - t2: has a teleport step (should trigger speed flag)
	# - t3: speeds are plausible, but acceleration spike should trigger accel flag (not speed flag)
	tf = pd.DataFrame(
		[
			# t1 (frame 2 is repaired)
			{"tracklet_id": "t1", "frame_index": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t1", "frame_index": 1, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},
			{
				"tracklet_id": "t1",
				"frame_index": 2,
				"detection_id": "d2",
				"x_m": 100.0,
				"y_m": 0.0,
				"x_m_repaired": 0.2,
				"y_m_repaired": 0.0,
				"is_repaired": True,
			},
			{"tracklet_id": "t1", "frame_index": 3, "detection_id": "d3", "x_m": 0.3, "y_m": 0.0, "is_repaired": False},
			# t2 (teleport not repaired)
			{"tracklet_id": "t2", "frame_index": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t2", "frame_index": 1, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t2", "frame_index": 2, "detection_id": "d2", "x_m": 10.0, "y_m": 0.0, "is_repaired": False},
			# t3 (accel spike but speed plausible)
			{"tracklet_id": "t3", "frame_index": 0, "detection_id": "d0", "x_m": 0.0, "y_m": 0.0, "is_repaired": False},
			{"tracklet_id": "t3", "frame_index": 1, "detection_id": "d1", "x_m": 0.1, "y_m": 0.0, "is_repaired": False},  # speed 1 m/s
			{"tracklet_id": "t3", "frame_index": 2, "detection_id": "d2", "x_m": 0.6, "y_m": 0.0, "is_repaired": False},  # speed 5 m/s, accel 40 m/s^2
		]
	)

	out, summary = _apply_cp3_kinematics(
		tf,
		fps=fps,
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
