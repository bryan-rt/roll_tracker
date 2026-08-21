"""Variable-dt BotSort subclass.

Replaces two constant-dt sites in stock BotSort:

1. The Kalman filter's baked dt=1.0 state-transition matrix — replaced with a
   per-frame dt ratio (dt_s / nominal_dt_s) via VariableDtKalmanFilterXYWH.
   Both the per-instance KF (self.kalman_filter) and the class-level
   STrack.shared_kalman are replaced.  Without replacing shared_kalman,
   STrack.multi_predict (the actual prediction path) is completely untouched.

2. The frame-count-based track lifetime comparison in _update_track_states —
   replaced with a wall-time comparison using accumulated per-frame dt_s.

Depends on boxmot internals (verified against boxmot==16.0.8, V1-V8):
  - BotSort.__init__ sets self.kalman_filter (V3)
  - STrack.shared_kalman class attribute (V4, V5)
  - STrack.multi_predict uses shared_kalman, not per-instance (V5)
  - BotSort._update_track_states with frame_count/end_frame comparison (V6)
  - BotSort.frame_count incremented inside update() (basetracker.py)
  - BaseTrack.end_frame = frame_id (last-seen frame_count)

Single-instance-per-process assumption (inherited from stock boxmot):
  STrack.shared_kalman and BaseTrack._count are class-level / process-global.
  A second concurrent tracker would silently corrupt both prediction and track
  IDs.  This is load-bearing in stock boxmot (BaseTrack.clear_count() in
  BotSort.__init__).  Multi-camera parallelization within one process must
  revisit this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack

from bjj_pipeline.tracking.variable_dt_kalman import VariableDtKalmanFilterXYWH


class VariableDtBotSort(BotSort):
    """BotSort with per-frame variable dt and wall-time track lifetime."""

    def __init__(
        self,
        *,
        max_lost_seconds: float = 2.0,
        nominal_dt_s: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._nominal_dt_s = nominal_dt_s
        self._max_lost_seconds = max_lost_seconds

        # Replace BOTH KF sites.
        self._vdt_kf = VariableDtKalmanFilterXYWH()
        self.kalman_filter = self._vdt_kf
        STrack.shared_kalman = self._vdt_kf

        assert isinstance(STrack.shared_kalman, VariableDtKalmanFilterXYWH), (
            f"STrack.shared_kalman assignment failed: expected "
            f"VariableDtKalmanFilterXYWH, got {type(STrack.shared_kalman).__name__}"
        )

        # Wall-time accumulator and per-frame timestamp map.
        self._elapsed_time_s: float = 0.0
        # Maps frame_count (1-based, post-increment) -> elapsed wall time.
        self._frame_timestamps: Dict[int, float] = {}

    def set_dt(self, dt_s: float, *, is_first_frame: bool = False) -> None:
        """Set the dt for the upcoming update() call.

        Called by BotSortTracker adapter before each update().

        Args:
            dt_s: Real inter-frame interval in seconds from the sidecar.
                  On frame 0 this is nominal_dt_s (sentinel — the KF won't
                  be reached because strack_pool is empty, but the guard
                  requires a non-None value).
            is_first_frame: True on frame 0.  Resets elapsed time to 0.
        """
        # Ratio formulation: velocity stays in px/nominal-frame.
        self._vdt_kf.current_dt = dt_s / self._nominal_dt_s

        if is_first_frame:
            self._elapsed_time_s = 0.0
        else:
            self._elapsed_time_s += dt_s

    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        # Record timestamp BEFORE super increments frame_count.
        # super().update() does frame_count += 1 at line 116, so the frame
        # that is about to be processed will have frame_count = current + 1.
        next_frame_count = self.frame_count + 1
        self._frame_timestamps[next_frame_count] = self._elapsed_time_s

        result = super().update(dets, img, embs)

        # Verify multi_predict used our subclassed filter (V5 trap detection).
        # On frame 1 (frame_count==1 after increment), strack_pool is empty
        # so multi_predict returns early without entering the KF — expected.
        if self.frame_count > 1:
            assert getattr(self._vdt_kf, "_multi_predict_entered", False), (
                "multi_predict was never entered on the variable-dt KF — "
                "prediction is using stock shared_kalman, not the subclass"
            )
        self._vdt_kf._multi_predict_entered = False

        return result

    def _update_track_states(self, removed_stracks: list) -> None:
        """Replace frame-count lifetime with wall-time lifetime.

        Stock: ``frame_count - track.end_frame > max_time_lost``
        Ours:  ``elapsed[frame_count] - elapsed[end_frame] > max_lost_seconds``
        """
        current_time = self._frame_timestamps.get(self.frame_count)
        if current_time is None:
            raise RuntimeError(
                f"_update_track_states: no timestamp for frame_count="
                f"{self.frame_count}. _frame_timestamps has "
                f"{len(self._frame_timestamps)} entries "
                f"(keys {min(self._frame_timestamps, default='?')}"
                f"..{max(self._frame_timestamps, default='?')}). "
                f"This indicates set_dt() was not called before update()."
            )

        for track in self.lost_stracks:
            last_seen_time = self._frame_timestamps.get(track.end_frame)
            if last_seen_time is None:
                raise RuntimeError(
                    f"_update_track_states: no timestamp for "
                    f"track.end_frame={track.end_frame} (track.id={track.id}). "
                    f"Timestamp keys span "
                    f"{min(self._frame_timestamps, default='?')}"
                    f"..{max(self._frame_timestamps, default='?')}. "
                    f"This should never happen — every frame_count value is "
                    f"recorded before update()."
                )
            time_lost = current_time - last_seen_time
            if time_lost > self._max_lost_seconds:
                track.mark_removed()
                removed_stracks.append(track)
