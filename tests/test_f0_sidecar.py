"""Tests for f0_sidecar reader and synthetic generator."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from bjj_pipeline.contracts.f0_sidecar import (
    SidecarData,
    SidecarError,
    SidecarSchemaError,
    SidecarValidityError,
    load_sidecar,
    parse_sidecar,
)
from bjj_pipeline.contracts.f0_sidecar_testutil import generate_synthetic_sidecar


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Valid passthrough, unimodal
# ---------------------------------------------------------------------------

class TestValidPassthrough:
    def test_loads_and_basic_fields(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 300, dt_s=0.067)
        data = parse_sidecar(sc)
        assert data.sidecar_schema == 5
        assert data.timing_mode == "passthrough"
        assert data.source_pts is True
        assert data.row_source == "mp4"
        assert data.frame_count == 300
        assert data.has_source_pts is True
        assert data.has_showinfo is True
        assert data.is_passthrough is True

    def test_nominal_fps(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 300, dt_s=0.067)
        data = parse_sidecar(sc)
        assert abs(data.nominal_fps - 1.0 / 0.067) < 0.1

    def test_dt_s_frame_0_is_none(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10, dt_s=0.067)
        data = parse_sidecar(sc)
        assert data.dt_s(0) is None

    def test_dt_s_frame_1(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10, dt_s=0.067)
        data = parse_sidecar(sc)
        assert abs(data.dt_s(1) - 0.067) < 1e-6

    def test_pts_time_s_cumsum_consistent(self, tmp_dir):
        """pts_time_s and dt_s agree: pts[i] = sum(dt[1:i])."""
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 20, dt_s=0.067, gaps=[5, 10]
        )
        data = parse_sidecar(sc)
        cumsum = 0.0
        for i in range(data.frame_count):
            assert abs(data.pts_time_s(i) - cumsum) < 1e-5, f"frame {i}"
            if i + 1 < data.frame_count:
                dt = data.dt_s(i + 1)
                cumsum += dt if dt is not None else 0


# ---------------------------------------------------------------------------
# Bimodal
# ---------------------------------------------------------------------------

class TestBimodal:
    def test_is_bimodal_true(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 100, dt_s=0.067, is_bimodal=True
        )
        data = parse_sidecar(sc)
        assert data.is_bimodal is True
        # short_mode_* in raw_meta
        assert "short_mode_fraction" in data.raw_meta
        assert "short_mode_fps" in data.raw_meta
        assert "short_mode_dt_s" in data.raw_meta
        assert "long_mode_dt_s" in data.raw_meta

    def test_is_bimodal_advisory_caveat(self, tmp_dir):
        """is_bimodal is advisory — it structurally cannot fire when the majority
        mode is the short one (contract §5 known limitation)."""
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 100, dt_s=0.067, is_bimodal=False
        )
        data = parse_sidecar(sc)
        assert data.is_bimodal is False


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

class TestGaps:
    def test_gap_frames_have_doubled_dt(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 30, dt_s=0.067, gaps=[12, 24]
        )
        data = parse_sidecar(sc)
        assert abs(data.dt_s(12) - 0.134) < 1e-6
        assert abs(data.dt_s(24) - 0.134) < 1e-6
        assert abs(data.dt_s(11) - 0.067) < 1e-6


# ---------------------------------------------------------------------------
# source_pts=false
# ---------------------------------------------------------------------------

class TestSourcePtsFalse:
    def test_strict_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, source_pts=False, timing_mode="cfr_grid"
        )
        mp4 = tmp_dir / "test.mp4"
        with pytest.raises(SidecarValidityError, match="source_pts"):
            load_sidecar(mp4)

    def test_permissive_succeeds(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, source_pts=False, timing_mode="cfr_grid"
        )
        data = parse_sidecar(sc)
        assert data.has_source_pts is False
        assert data.nominal_dt_s is None

    def test_dt_s_raises_validity_error(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, source_pts=False, timing_mode="cfr_grid"
        )
        data = parse_sidecar(sc)
        with pytest.raises(SidecarValidityError, match="timing_mode"):
            data.dt_s(1)

    def test_nominal_fps_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, source_pts=False, timing_mode="cfr_grid"
        )
        data = parse_sidecar(sc)
        with pytest.raises(SidecarValidityError, match="nominal_dt_s"):
            _ = data.nominal_fps


# ---------------------------------------------------------------------------
# CFR grid — strict
# ---------------------------------------------------------------------------

class TestCfrGrid:
    def test_strict_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, timing_mode="cfr_grid"
        )
        mp4 = tmp_dir / "test.mp4"
        with pytest.raises(SidecarValidityError, match="timing_mode"):
            load_sidecar(mp4)


# ---------------------------------------------------------------------------
# Missing sidecar
# ---------------------------------------------------------------------------

class TestMissingSidecar:
    def test_load_raises(self, tmp_dir):
        mp4 = tmp_dir / "nonexistent.mp4"
        with pytest.raises(SidecarError):
            load_sidecar(mp4)

    def test_parse_raises(self, tmp_dir):
        with pytest.raises(SidecarError):
            parse_sidecar(tmp_dir / "nonexistent.timing.jsonl")


# ---------------------------------------------------------------------------
# Schema 4
# ---------------------------------------------------------------------------

class TestSchema4:
    def test_strict_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, schema=4
        )
        mp4 = tmp_dir / "test.mp4"
        with pytest.raises(SidecarSchemaError, match="schema 4"):
            load_sidecar(mp4)

    def test_permissive_succeeds(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, schema=4
        )
        data = parse_sidecar(sc)
        assert data.sidecar_schema == 4


# ---------------------------------------------------------------------------
# mp4_regenerated
# ---------------------------------------------------------------------------

class TestMp4Regenerated:
    def test_has_showinfo_false(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, row_source="mp4_regenerated"
        )
        data = parse_sidecar(sc)
        assert data.has_showinfo is False
        assert data.showinfo_residual is None
        assert data.showinfo_frame_count is None

    def test_host_arrival_absent(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, row_source="mp4_regenerated"
        )
        data = parse_sidecar(sc)
        assert data.host_arrival_s(0) is None
        assert data.host_arrival_s(5) is None


# ---------------------------------------------------------------------------
# Per-row host_arrival_s miss
# ---------------------------------------------------------------------------

class TestPerRowHostMiss:
    def test_mixed_presence(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 20, dt_s=0.067,
            host_arrival_base=1786988300.0,
            host_arrival_missing=[5, 10, 15],
        )
        data = parse_sidecar(sc)
        assert data.host_arrival_s(4) is not None
        assert data.host_arrival_s(5) is None
        assert data.host_arrival_s(6) is not None
        assert data.host_arrival_s(10) is None
        assert data.host_arrival_s(15) is None


# ---------------------------------------------------------------------------
# Drift fields absent (n_drift_windows < 4)
# ---------------------------------------------------------------------------

class TestDriftAbsent:
    def test_drift_absent_below_4_windows(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, n_drift_windows=2
        )
        data = parse_sidecar(sc)
        assert data.drift_rate_s_per_s is None
        assert data.drift_ppm is None
        assert data.n_drift_windows == 2

    def test_drift_present_above_4_windows(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "test.timing.jsonl", 10, n_drift_windows=6
        )
        data = parse_sidecar(sc)
        assert data.drift_rate_s_per_s is not None
        assert data.n_drift_windows == 6


# ---------------------------------------------------------------------------
# Out-of-range frame_index
# ---------------------------------------------------------------------------

class TestOutOfRange:
    def test_pts_time_s_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10)
        data = parse_sidecar(sc)
        with pytest.raises(IndexError):
            data.pts_time_s(999)

    def test_dt_s_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10)
        data = parse_sidecar(sc)
        with pytest.raises(IndexError):
            data.dt_s(999)

    def test_negative_index_raises(self, tmp_dir):
        sc = generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10)
        data = parse_sidecar(sc)
        with pytest.raises(IndexError):
            data.pts_time_s(-1)


# ---------------------------------------------------------------------------
# Contiguity violations
# ---------------------------------------------------------------------------

class TestContiguity:
    def test_hole_raises(self, tmp_dir):
        """frame_index with a gap (0, 1, 3) raises on parse."""
        path = tmp_dir / "hole.timing.jsonl"
        meta = {"_meta": True, "sidecar_schema": 5, "timing_mode": "passthrough",
                "source_pts": True, "pts_origin": "segment_relative", "fps_method": "trimmed_mean",
                "row_source": "mp4", "segment_start_epoch": 0, "attempt": 1,
                "input_frame_count": 3, "output_frame_count": 3, "mismatch": False,
                "measured_fps_mean": 15.0, "pts_timebase": 90000,
                "pts_tick_delta_median": 6000, "pts_tick_delta_mean": 6000,
                "pts_delta_trim_kept": 2, "pts_delta_trim_total": 2,
                "pts_mean_delta_ms": 66.7, "pts_stdev_delta_ms": 0.0}
        with open(path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            f.write(json.dumps({"frame_index": 0, "pts_time_s": 0.0, "dt_s": None}) + "\n")
            f.write(json.dumps({"frame_index": 1, "pts_time_s": 0.067, "dt_s": 0.067}) + "\n")
            # Skip frame_index 2, emit 3
            f.write(json.dumps({"frame_index": 3, "pts_time_s": 0.200, "dt_s": 0.066}) + "\n")
        with pytest.raises(SidecarError, match="contiguity"):
            parse_sidecar(path)

    def test_duplicate_raises(self, tmp_dir):
        """Duplicate frame_index (0, 1, 1) raises on parse."""
        path = tmp_dir / "dup.timing.jsonl"
        meta = {"_meta": True, "sidecar_schema": 5, "timing_mode": "passthrough",
                "source_pts": True, "pts_origin": "segment_relative", "fps_method": "trimmed_mean",
                "row_source": "mp4", "segment_start_epoch": 0, "attempt": 1,
                "input_frame_count": 3, "output_frame_count": 3, "mismatch": False,
                "measured_fps_mean": 15.0, "pts_timebase": 90000,
                "pts_tick_delta_median": 6000, "pts_tick_delta_mean": 6000,
                "pts_delta_trim_kept": 2, "pts_delta_trim_total": 2,
                "pts_mean_delta_ms": 66.7, "pts_stdev_delta_ms": 0.0}
        with open(path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            f.write(json.dumps({"frame_index": 0, "pts_time_s": 0.0, "dt_s": None}) + "\n")
            f.write(json.dumps({"frame_index": 1, "pts_time_s": 0.067, "dt_s": 0.067}) + "\n")
            # Duplicate frame_index 1
            f.write(json.dumps({"frame_index": 1, "pts_time_s": 0.134, "dt_s": 0.067}) + "\n")
        with pytest.raises(SidecarError, match="contiguity"):
            parse_sidecar(path)


# ---------------------------------------------------------------------------
# Row count != output_frame_count
# ---------------------------------------------------------------------------

class TestRowCountMismatch:
    def test_raises_on_mismatch(self, tmp_dir):
        path = tmp_dir / "bad_count.timing.jsonl"
        meta = {"_meta": True, "sidecar_schema": 5, "timing_mode": "passthrough",
                "source_pts": True, "pts_origin": "segment_relative", "fps_method": "trimmed_mean",
                "row_source": "mp4", "segment_start_epoch": 0, "attempt": 1,
                "input_frame_count": 5, "output_frame_count": 5, "mismatch": False,
                "measured_fps_mean": 15.0, "pts_timebase": 90000,
                "pts_tick_delta_median": 6000, "pts_tick_delta_mean": 6000,
                "pts_delta_trim_kept": 4, "pts_delta_trim_total": 4,
                "pts_mean_delta_ms": 66.7, "pts_stdev_delta_ms": 0.0}
        with open(path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            # Only 3 rows, but output_frame_count says 5
            for i in range(3):
                f.write(json.dumps({"frame_index": i, "pts_time_s": i * 0.067,
                                    "dt_s": None if i == 0 else 0.067}) + "\n")
        with pytest.raises(SidecarError, match="Row count.*output_frame_count"):
            parse_sidecar(path)


# ---------------------------------------------------------------------------
# Round-trip: generator -> write -> parse -> verify
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip(self, tmp_dir):
        sc = generate_synthetic_sidecar(
            tmp_dir / "rt.timing.jsonl", 50, dt_s=0.067,
            gaps=[10, 20, 30], host_arrival_base=1786988300.0,
        )
        data = parse_sidecar(sc)
        assert data.frame_count == 50
        assert data.dt_s(0) is None
        assert abs(data.dt_s(1) - 0.067) < 1e-6
        assert abs(data.dt_s(10) - 0.134) < 1e-6  # gap
        assert data.host_arrival_s(0) is not None
        assert data.pts_time_s(0) == 0.0

    def test_load_sidecar_strict(self, tmp_dir):
        """load_sidecar succeeds on valid schema-5 passthrough."""
        generate_synthetic_sidecar(tmp_dir / "test.timing.jsonl", 10, dt_s=0.067)
        mp4 = tmp_dir / "test.mp4"
        data = load_sidecar(mp4)
        assert data.sidecar_schema == 5
        assert data.frame_count == 10
