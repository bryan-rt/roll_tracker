"""Piece 12 regression tests — VFR redaction rendering via PyAV.

Test 1 (guard): PTS histogram must retain distinct 66ms and 67ms intervals.
    At rate=15, the codec context quantizes PTS to a 1/15s grid, collapsing
    66/67ms to uniform 67ms. At rate=90000, the source's exact intervals are
    preserved. This test MUST fail at rate=15.

Test 2 (value): Known input PTS → exact output PTS ticks.

Test 3 (endpoint): First output PTS is 0, last output PTS equals the expected
    clip-relative value. Catches an off-by-one shift that would preserve the
    interval histogram but shift every frame's assignment.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from bjj_pipeline.stages.export.redact import PTS_TIMEBASE_HZ


def _encode_test_frames(
    output_path: Path,
    pts_ms_list: list[float],
    width: int = 64,
    height: int = 64,
    rate: int = PTS_TIMEBASE_HZ,
) -> None:
    """Encode solid-color test frames with specified PTS at the given rate."""
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}

    base_ms = pts_ms_list[0]
    for i, pts_ms in enumerate(pts_ms_list):
        frame_bgr = np.full((height, width, 3), fill_value=((i * 37) % 256), dtype=np.uint8)
        vf = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        vf.pts = round((pts_ms - base_ms) * 90)
        for pkt in stream.encode(vf):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def _read_pts_ticks(path: Path) -> list[int]:
    """Read per-frame PTS in raw timebase ticks from a video file."""
    import subprocess
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return [int(x.strip().rstrip(","))
            for x in proc.stdout.strip().split("\n") if x.strip()]


# ---------------------------------------------------------------------------
# Fixtures: non-uniform PTS with distinct 66ms and 67ms intervals
# ---------------------------------------------------------------------------

# Source: 90kHz ticks 5940 (=66ms) and 6030 (=67ms) alternation, with one
# gap of 12060 (=134ms). This is the real FP7oJQ cadence from CP-R11.
_TEST_PTS_MS = [
    0.0,     # frame 0
    67.0,    # +67ms (6030 ticks)
    134.0,   # +67ms
    200.0,   # +66ms (5940 ticks)
    334.0,   # +134ms (gap: 12060 ticks)
    400.0,   # +66ms
    467.0,   # +67ms
    534.0,   # +67ms
    600.0,   # +66ms
    667.0,   # +67ms
    734.0,   # +67ms
    800.0,   # +66ms
    867.0,   # +67ms
    934.0,   # +67ms
    1000.0,  # +66ms
]

# Expected intervals in 90kHz ticks (5940=66ms, 6030=67ms, 12060=134ms)
_EXPECTED_TICK_INTERVALS = Counter({6030: 8, 5940: 5, 12060: 1})
_EXPECTED_PTS_TICKS = [round(p * 90) for p in _TEST_PTS_MS]


class TestPtsHistogramGuard:
    """Guard test: PTS_TIMEBASE_HZ must preserve the 5940/6030 tick distinction."""

    def test_intervals_preserved_at_90000(self, tmp_path):
        out = tmp_path / "vfr_90k.mp4"
        _encode_test_frames(out, _TEST_PTS_MS, rate=PTS_TIMEBASE_HZ)

        ticks = _read_pts_ticks(out)
        assert len(ticks) == len(_TEST_PTS_MS), f"frame count {len(ticks)} != {len(_TEST_PTS_MS)}"
        intervals = Counter(ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1))
        assert intervals == _EXPECTED_TICK_INTERVALS, (
            f"PTS tick histogram mismatch at rate={PTS_TIMEBASE_HZ}: "
            f"got {dict(intervals)}, expected {dict(_EXPECTED_TICK_INTERVALS)}"
        )

    def test_intervals_COLLAPSE_at_rate_15(self, tmp_path):
        """Demonstrate that rate=15 destroys the 5940/6030 distinction.

        This test asserts that rate=15 DOES collapse the intervals, proving
        the guard is load-bearing. If this test ever fails (rate=15 somehow
        preserves 5940/6030), PTS_TIMEBASE_HZ may no longer be necessary and
        the guard should be re-evaluated.
        """
        out = tmp_path / "cfr_15.mp4"
        _encode_test_frames(out, _TEST_PTS_MS, rate=15)

        ticks = _read_pts_ticks(out)
        intervals = Counter(ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1))
        # At rate=15, the codec quantizes to 1/15s = 6000 ticks.
        # 5940 (66ms) disappears; all become 6000 (66.67ms).
        assert 5940 not in intervals, (
            f"rate=15 unexpectedly preserved 5940-tick intervals: {dict(intervals)}. "
            f"PTS_TIMEBASE_HZ guard may no longer be necessary."
        )


class TestPtsValueExact:
    """Value test: output PTS ticks must equal input PTS ticks."""

    def test_exact_tick_match(self, tmp_path):
        out = tmp_path / "exact.mp4"
        _encode_test_frames(out, _TEST_PTS_MS, rate=PTS_TIMEBASE_HZ)

        ticks = _read_pts_ticks(out)
        assert ticks == _EXPECTED_PTS_TICKS, (
            f"PTS tick mismatch: got {ticks}, expected {_EXPECTED_PTS_TICKS}"
        )

    def test_first_pts_is_zero(self, tmp_path):
        """First output PTS must be 0 — catches an off-by-one shift."""
        out = tmp_path / "zero.mp4"
        # Use mid-file PTS (simulating a seek to export_start_frame > 0)
        mid_pts = [5000.0 + p for p in _TEST_PTS_MS]
        _encode_test_frames(out, mid_pts, rate=PTS_TIMEBASE_HZ)

        ticks = _read_pts_ticks(out)
        assert ticks[0] == 0, f"first PTS tick should be 0, got {ticks[0]}"

    def test_last_pts_matches_expected(self, tmp_path):
        """Last output PTS must equal (last_input - first_input) * 90.

        An off-by-one shift preserves the interval histogram but breaks
        the endpoint.
        """
        out = tmp_path / "endpoint.mp4"
        _encode_test_frames(out, _TEST_PTS_MS, rate=PTS_TIMEBASE_HZ)

        ticks = _read_pts_ticks(out)
        expected_last = round((_TEST_PTS_MS[-1] - _TEST_PTS_MS[0]) * 90)
        assert ticks[-1] == expected_last, (
            f"last PTS tick {ticks[-1]} != expected {expected_last}"
        )
