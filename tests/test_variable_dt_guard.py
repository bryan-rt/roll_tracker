"""T1: variable-dt tracker guard accepts dt_s=0.0 and rejects dt_s<0.

dt_s=0.0 represents a duplicate-PTS frame (MUXER-PTS-1). The Kalman filter
treats ratio 0.0 as a position no-op by design (CLAUDE.md:196). The guard
at tracker.py must allow it through. Negative dt is genuinely invalid
(non-monotonic time) and must still raise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bjj_pipeline.stages.detect_track.tracker import BotSortTracker, Detection


def _make_sidecar(dt_by_frame: dict[int, float], nominal: float = 0.067):
    """Build a mock SidecarData returning dt_s per frame_index."""
    sidecar = MagicMock()
    sidecar.nominal_dt_s = nominal
    sidecar.dt_s = MagicMock(side_effect=lambda fi: dt_by_frame.get(fi, nominal))
    return sidecar


def _make_tracker(sidecar) -> BotSortTracker:
    return BotSortTracker(
        with_reid=False,
        params={},
        variable_dt=True,
        sidecar_data=sidecar,
        max_lost_seconds=2.0,
    )


def _one_detection() -> list[Detection]:
    return [Detection(
        clip_id="test", camera_id="test", frame_index=0, timestamp_ms=0,
        detection_id="d0", class_name="person", confidence=0.9,
        x1=10.0, y1=10.0, x2=50.0, y2=50.0,
    )]


def _frame(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestVariableDtGuard:

    def test_dt_zero_accepted(self):
        """dt_s=0.0 at frame 2 must not raise — duplicate PTS is valid."""
        sidecar = _make_sidecar({2: 0.0})
        tracker = _make_tracker(sidecar)
        img = _frame()
        dets = _one_detection()

        # Frame 0 (uses nominal_dt_s, not dt_s lookup)
        tracker.update(frame_index=0, detections=dets, frame_bgr=img)
        # Frame 1 (dt=0.067)
        tracker.update(frame_index=1, detections=dets, frame_bgr=img)
        # Frame 2 (dt=0.0) — the MUXER-PTS-1 duplicate
        tracker.update(frame_index=2, detections=dets, frame_bgr=img)

    def test_dt_negative_raises(self):
        """dt_s < 0 must raise ValueError — non-monotonic time is invalid."""
        sidecar = _make_sidecar({1: -0.001})
        tracker = _make_tracker(sidecar)
        img = _frame()
        dets = _one_detection()

        tracker.update(frame_index=0, detections=dets, frame_bgr=img)
        with pytest.raises(ValueError, match="non-negative"):
            tracker.update(frame_index=1, detections=dets, frame_bgr=img)

    def test_dt_none_raises(self):
        """dt_s = None must raise ValueError."""
        sidecar = _make_sidecar({1: None})
        tracker = _make_tracker(sidecar)
        img = _frame()
        dets = _one_detection()

        tracker.update(frame_index=0, detections=dets, frame_bgr=img)
        with pytest.raises(ValueError):
            tracker.update(frame_index=1, detections=dets, frame_bgr=img)
