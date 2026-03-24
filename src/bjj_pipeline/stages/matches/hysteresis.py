"""Role: hysteresis state machine for robust engagement start/stop detection.

Implements proximity-based engagement detection with a four-state machine:
DISENGAGED → ENGAGING → ENGAGED → GRACE → DISENGAGED, with configurable
thresholds and grace periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class EngagementInterval:
    person_id_a: str
    person_id_b: str
    start_frame: int
    end_frame: int
    evidence_sources: Tuple[str, ...]
    partial_start: bool
    partial_end: bool


class _State(Enum):
    DISENGAGED = auto()
    ENGAGING = auto()
    ENGAGED = auto()
    GRACE = auto()


def run_proximity_hysteresis(
    pair_distances_df: pd.DataFrame,
    *,
    session_start_frame: int,
    session_end_frame: int,
    engage_dist_m: float,
    disengage_dist_m: float,
    engage_min_frames: int,
    hysteresis_frames: int,
    min_clip_duration_frames: int,
) -> List[EngagementInterval]:
    """Run the engagement state machine per unordered pair.

    State transitions:
    - DISENGAGED → ENGAGING: dist < engage_dist_m
    - ENGAGING → ENGAGED: sustained for engage_min_frames consecutive frames
    - ENGAGING → DISENGAGED: broken before engage_min_frames
    - ENGAGED → GRACE: dist > disengage_dist_m
    - GRACE → ENGAGED: dist drops back below disengage_dist_m
    - GRACE → DISENGAGED: dist > disengage_dist_m for >= hysteresis_frames → emit

    Partial handling:
    - If pair within engage_dist_m at their first frame AND that frame ==
      session_start_frame, start in ENGAGED with partial_start=True.
    - If still ENGAGED/GRACE at session_end_frame, emit with partial_end=True.

    Filter: drop intervals shorter than min_clip_duration_frames.
    """
    if pair_distances_df is None or pair_distances_df.empty:
        return []

    results: List[EngagementInterval] = []

    # Group by pair
    grouped = pair_distances_df.groupby(["person_id_a", "person_id_b"], sort=True)

    for (pid_a, pid_b), grp in grouped:
        pair_frames = grp.sort_values("frame_index")
        frames = pair_frames["frame_index"].values
        dists = pair_frames["dist_m"].values

        if len(frames) == 0:
            continue

        intervals = _run_pair_fsm(
            person_id_a=str(pid_a),
            person_id_b=str(pid_b),
            frames=frames,
            dists=dists,
            session_start_frame=session_start_frame,
            session_end_frame=session_end_frame,
            engage_dist_m=engage_dist_m,
            disengage_dist_m=disengage_dist_m,
            engage_min_frames=engage_min_frames,
            hysteresis_frames=hysteresis_frames,
            min_clip_duration_frames=min_clip_duration_frames,
        )
        results.extend(intervals)

    results.sort(key=lambda i: (i.person_id_a, i.person_id_b, i.start_frame, i.end_frame))
    return results


def _run_pair_fsm(
    *,
    person_id_a: str,
    person_id_b: str,
    frames,
    dists,
    session_start_frame: int,
    session_end_frame: int,
    engage_dist_m: float,
    disengage_dist_m: float,
    engage_min_frames: int,
    hysteresis_frames: int,
    min_clip_duration_frames: int,
) -> List[EngagementInterval]:
    """FSM for a single pair."""
    intervals: List[EngagementInterval] = []

    state = _State.DISENGAGED
    engage_start = 0
    engaging_counter = 0
    grace_counter = 0
    partial_start = False

    first_frame = int(frames[0])
    first_dist = float(dists[0])

    # Partial start: if first observation is at session_start_frame and close
    if first_frame == session_start_frame and first_dist < engage_dist_m:
        state = _State.ENGAGED
        engage_start = first_frame
        partial_start = True
    elif first_dist < engage_dist_m:
        state = _State.ENGAGING
        engage_start = first_frame
        engaging_counter = 1
        if engaging_counter >= engage_min_frames:
            state = _State.ENGAGED

    for idx in range(1 if state != _State.DISENGAGED or first_dist >= engage_dist_m else 1, len(frames)):
        frame = int(frames[idx])
        dist = float(dists[idx])

        if state == _State.DISENGAGED:
            if dist < engage_dist_m:
                state = _State.ENGAGING
                engage_start = frame
                engaging_counter = 1
                if engaging_counter >= engage_min_frames:
                    state = _State.ENGAGED

        elif state == _State.ENGAGING:
            if dist < engage_dist_m:
                engaging_counter += 1
                if engaging_counter >= engage_min_frames:
                    state = _State.ENGAGED
            else:
                state = _State.DISENGAGED
                engaging_counter = 0

        elif state == _State.ENGAGED:
            if dist > disengage_dist_m:
                state = _State.GRACE
                grace_counter = 1

        elif state == _State.GRACE:
            if dist <= disengage_dist_m:
                state = _State.ENGAGED
                grace_counter = 0
            else:
                grace_counter += 1
                if grace_counter >= hysteresis_frames:
                    # Emit interval — end_frame is the frame where grace started
                    # (hysteresis_frames ago from current)
                    end_frame = frame - hysteresis_frames
                    if end_frame < engage_start:
                        end_frame = engage_start
                    _maybe_emit(
                        intervals,
                        person_id_a=person_id_a,
                        person_id_b=person_id_b,
                        start_frame=engage_start,
                        end_frame=end_frame,
                        partial_start=partial_start,
                        partial_end=False,
                        min_clip_duration_frames=min_clip_duration_frames,
                    )
                    state = _State.DISENGAGED
                    grace_counter = 0
                    partial_start = False

    # End of data: emit if still engaged or in grace
    if state in (_State.ENGAGED, _State.GRACE):
        last_frame = int(frames[-1])
        is_partial_end = last_frame == session_end_frame
        _maybe_emit(
            intervals,
            person_id_a=person_id_a,
            person_id_b=person_id_b,
            start_frame=engage_start,
            end_frame=last_frame,
            partial_start=partial_start,
            partial_end=is_partial_end,
            min_clip_duration_frames=min_clip_duration_frames,
        )

    return intervals


def _maybe_emit(
    intervals: List[EngagementInterval],
    *,
    person_id_a: str,
    person_id_b: str,
    start_frame: int,
    end_frame: int,
    partial_start: bool,
    partial_end: bool,
    min_clip_duration_frames: int,
) -> None:
    duration = end_frame - start_frame
    if duration < min_clip_duration_frames:
        return
    intervals.append(
        EngagementInterval(
            person_id_a=person_id_a,
            person_id_b=person_id_b,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            evidence_sources=("proximity",),
            partial_start=partial_start,
            partial_end=partial_end,
        )
    )
