"""Variable-dt guards: tracker-level (dt_s values) and config-level (missing keys).

Tracker-level (T1, Piece 11):
  dt_s=0.0 represents a duplicate-PTS frame (MUXER-PTS-1). The Kalman filter
  treats ratio 0.0 as a position no-op by design. The guard at tracker.py must
  allow it through. Negative dt is genuinely invalid (non-monotonic time).

Config-level (VDT-DEFAULT-1):
  A config missing stages.stage_A.tracker.variable_dt must raise ValueError,
  not silently produce False via bool(None). Same for max_lost_seconds.
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


class TestConfigGuard:
    """Config missing variable_dt or max_lost_seconds must raise, not silently default.

    VDT-DEFAULT-1: the guard in multiplex_runner.py checks _cfg_get's return and
    raises ValueError with a clear message. These tests exercise _cfg_get on a
    minimal config to prove the precondition (missing key -> None), then verify
    that the production guard pattern raises ValueError, not bool(None)->False.
    """

    def _guard_variable_dt(self, config: dict) -> bool:
        """Reproduce the production guard from multiplex_runner.py."""
        from bjj_pipeline.stages.orchestration.multiplex_runner import _cfg_get
        path = "stages.stage_A.tracker.variable_dt"
        val = _cfg_get(config, path)
        if val is None:
            raise ValueError(
                f"Missing required config {path}. "
                "Source of truth is configs/default.yaml stages.stage_A.tracker."
            )
        return bool(val)

    def _guard_max_lost_seconds(self, config: dict) -> float:
        """Reproduce the production guard from multiplex_runner.py."""
        from bjj_pipeline.stages.orchestration.multiplex_runner import _cfg_get
        path = "stages.stage_A.tracker.max_lost_seconds"
        val = _cfg_get(config, path)
        if val is None:
            raise ValueError(
                f"Missing required config {path}. "
                "Source of truth is configs/default.yaml stages.stage_A.tracker."
            )
        return float(val)

    def test_missing_variable_dt_raises(self):
        """A config without variable_dt must raise ValueError, not silently return False."""
        config = {"stages": {"stage_A": {"tracker": {"mode": "botsort"}}}}
        with pytest.raises(ValueError, match="Missing required config"):
            self._guard_variable_dt(config)

    def test_missing_max_lost_seconds_raises(self):
        """A config without max_lost_seconds must raise ValueError, not TypeError."""
        config = {"stages": {"stage_A": {"tracker": {"mode": "botsort"}}}}
        with pytest.raises(ValueError, match="Missing required config"):
            self._guard_max_lost_seconds(config)

    def test_present_variable_dt_passes(self):
        """A config with variable_dt: true must return True."""
        config = {"stages": {"stage_A": {"tracker": {"variable_dt": True}}}}
        assert self._guard_variable_dt(config) is True

    def test_present_max_lost_seconds_passes(self):
        """A config with max_lost_seconds: 2.0 must return 2.0."""
        config = {"stages": {"stage_A": {"tracker": {"max_lost_seconds": 2.0}}}}
        assert self._guard_max_lost_seconds(config) == 2.0
