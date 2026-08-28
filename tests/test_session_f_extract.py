"""Tests for session-level Stage F extract — PIECE6-FIX-1 regression guard.

Two tests:
1. test_extract_passes_correct_kwargs: mocks export_clip to verify the call
   uses start_sec/duration_sec (not the removed fps/start_frame/end_frame).
   Catches the TypeError regression directly.

2. test_extract_clip_b_seek_values: builds a two-clip registry with known
   timestamps and ts_offset_ms, verifies that a match landing in clip B
   produces the correct clip-local seek time (session_ts - ts_offset_ms),
   not the session-relative timestamp. The distinction matters: clip B's
   session timestamp is ~543s, but the correct start_sec is small.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bjj_pipeline.stages.export.session_f_run import (
    SourceClipInfo,
    _extract_session_clip,
    _segment_seek_times,
)
from bjj_pipeline.stages.export.ffmpeg import ExportClipError, ExportResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clip(
    clip_id: str = "clip_A",
    frame_offset: int = 0,
    ts_offset_ms: int = 0,
    duration_frames: int = 1800,
) -> SourceClipInfo:
    return SourceClipInfo(
        clip_id=clip_id,
        mp4_path=Path(f"/fake/{clip_id}.mp4"),
        cam_id="FP7oJQ",
        frame_offset=frame_offset,
        ts_offset_ms=ts_offset_ms,
        duration_frames=duration_frames,
    )


def _make_crop_plan():
    from bjj_pipeline.stages.export.cropper import FixedRoiCropPlan
    return FixedRoiCropPlan(
        mode="fixed_roi",
        x=100, y=100, width=200, height=200,
        start_frame=100, end_frame=200,
        person_id_a="p0001", person_id_b="p0002",
        n_track_rows=50, n_pair_frames=10,
        envelope_method="test",
        padding_px=10,
    )


# ---------------------------------------------------------------------------
# Test 1: kwarg regression guard (TypeError on pre-fix code)
# ---------------------------------------------------------------------------

@patch("bjj_pipeline.stages.export.session_f_run.export_clip")
def test_extract_passes_correct_kwargs(mock_export_clip, tmp_path):
    """export_clip must be called with start_sec/duration_sec, not fps/start_frame/end_frame.

    Pre-fix: raises TypeError('export_clip() got an unexpected keyword argument \'fps\'').
    Post-fix: passes with mocked export_clip.
    """
    out_path = tmp_path / "out.mp4"
    mock_export_clip.return_value = ExportResult(
        output_video_path=out_path,
        ffmpeg_cmd="ffmpeg ...",
        return_code=0,
    )

    clip_a = _make_clip()
    crop_plan = _make_crop_plan()

    # Match spanning frames 100-200 in session-relative space (all within clip A)
    # 67ms per frame at 15fps
    frame_to_ts_ms = {i: i * 67 for i in range(1800)}

    _extract_session_clip(
        source_clips=[clip_a],
        match_start_frame=100,
        match_end_frame=200,
        output_path=out_path,
        frame_to_ts_ms=frame_to_ts_ms,
        crop_plan=crop_plan,
    )

    mock_export_clip.assert_called_once()
    call_kwargs = mock_export_clip.call_args.kwargs
    assert "start_sec" in call_kwargs, "export_clip must receive start_sec"
    assert "duration_sec" in call_kwargs, "export_clip must receive duration_sec"
    assert "fps" not in call_kwargs, "export_clip must NOT receive fps (removed in Piece 6)"
    assert "start_frame" not in call_kwargs, "export_clip must NOT receive start_frame"
    assert "end_frame" not in call_kwargs, "export_clip must NOT receive end_frame"


# ---------------------------------------------------------------------------
# Test 2: value correctness — clip B offset derivation
# ---------------------------------------------------------------------------

def test_segment_seek_times_clip_b():
    """Verify clip-local seek time subtracts ts_offset_ms, not session-relative.

    Two-clip registry using the real Saturday values:
      Clip A: frame_offset=0,    ts_offset_ms=0
      Clip B: frame_offset=1800, ts_offset_ms=543280

    A match at session-relative frame 1850 has session timestamp ~543280 + 50*67 = 546630ms.
    The clip-local start_sec must be (546630 - 543280) / 1000 = 3.35s, NOT 546.63s.
    If ts_offset_ms subtraction were dropped, the test fails by >500s.
    """
    clip_b = _make_clip(
        clip_id="clip_B",
        frame_offset=1800,
        ts_offset_ms=543280,
        duration_frames=1709,
    )

    # Build frame_to_ts_ms with known session-relative timestamps.
    # Clip B's local frame 0 = session frame 1800, session ts = 543280 (= ts_offset_ms + 0)
    # Clip B's local frame 50 = session frame 1850, session ts = 543280 + 50*67 = 546630
    # Clip B's local frame 150 = session frame 1950, session ts = 543280 + 150*67 = 553330
    frame_to_ts_ms = {}
    for local_f in range(1709):
        session_f = 1800 + local_f
        frame_to_ts_ms[session_f] = 543280 + local_f * 67

    # Match at session frames [1850, 1950] = clip-local [50, 150]
    start_sec, duration_sec = _segment_seek_times(
        clip_b,
        local_start=50,   # clip-local
        local_end=150,    # clip-local
        frame_to_ts_ms=frame_to_ts_ms,
    )

    # Expected clip-local start: (546630 - 543280) / 1000 = 3.35s
    expected_start = (543280 + 50 * 67 - 543280) / 1000.0  # = 3.35
    assert abs(start_sec - expected_start) < 0.001, (
        f"start_sec={start_sec} but expected {expected_start} "
        f"(clip-local, not session-relative)"
    )
    # Must NOT be the session-relative value
    session_start_sec = (543280 + 50 * 67) / 1000.0  # = 546.63
    assert abs(start_sec - session_start_sec) > 500.0, (
        f"start_sec={start_sec} is close to session-relative {session_start_sec} — "
        f"ts_offset_ms subtraction is missing"
    )

    # Expected duration: (553330 - 546630) / 1000 = 6.7s
    expected_duration = (150 - 50) * 67 / 1000.0  # = 6.7
    assert abs(duration_sec - expected_duration) < 0.001, (
        f"duration_sec={duration_sec} but expected {expected_duration}"
    )


def test_segment_seek_times_clip_a_offset_zero():
    """Clip A with ts_offset_ms=0: start_sec equals session-relative (no-op subtraction)."""
    clip_a = _make_clip(clip_id="clip_A", frame_offset=0, ts_offset_ms=0)
    frame_to_ts_ms = {i: i * 67 for i in range(1800)}

    start_sec, duration_sec = _segment_seek_times(
        clip_a, local_start=100, local_end=200, frame_to_ts_ms=frame_to_ts_ms,
    )

    assert abs(start_sec - 100 * 67 / 1000.0) < 0.001
    assert abs(duration_sec - 100 * 67 / 1000.0) < 0.001


def test_segment_seek_times_missing_frame_raises():
    """Missing session-relative frame must raise, not return a wrong value."""
    clip_b = _make_clip(clip_id="clip_B", frame_offset=1800, ts_offset_ms=543280)
    frame_to_ts_ms = {}  # empty

    with pytest.raises(ExportClipError, match="frame_to_ts_ms missing"):
        _segment_seek_times(clip_b, local_start=50, local_end=150, frame_to_ts_ms=frame_to_ts_ms)
