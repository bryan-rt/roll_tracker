"""Tests for the sidecar ingest gate (CP4.A).

Tests the _validate_sidecar_ingest helper and the frame_iterator
timestamp changes. Gate tests use synthetic sidecars in temp dirs.
The FrameIterator fixture test requires the local GT corpus
(data/raw/nest/_eval_gt/) — skipped if not present.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bjj_pipeline.stages.orchestration.pipeline import (
    PipelineError,
    _validate_sidecar_ingest,
)
from bjj_pipeline.contracts.f0_paths import resolve_sidecar_path
from bjj_pipeline.contracts.f0_sidecar_testutil import generate_synthetic_sidecar
from bjj_pipeline.core.frame_iterator import FrameIterator, FramePacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mp4_stub(d: Path, name: str = "clip.mp4") -> Path:
    """Create an empty file as an mp4 stand-in. load_sidecar does not read it."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.touch()
    return p


# ---------------------------------------------------------------------------
# Ingest gate — valid sidecar passes
# ---------------------------------------------------------------------------

class TestIngestGateValid:
    def test_real_sidecar_passes(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100,
        )
        _, prov, path = _validate_sidecar_ingest(
            ingest_path=mp4,
            clip_id="clip",
            resolved_config={},
            out_root=tmp_path,
        )
        assert prov == "real"
        assert "clip.timing.jsonl" in path

    def test_synthetic_override_passes(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path / "raw")
        (tmp_path / "raw").mkdir(exist_ok=True)
        mp4 = _make_mp4_stub(tmp_path / "raw", "clip.mp4")
        # No real sidecar next to mp4
        # Synthetic sidecar under out_root/_synthetic_sidecars/
        syn_dir = tmp_path / "out" / "_synthetic_sidecars"
        syn_dir.mkdir(parents=True)
        generate_synthetic_sidecar(
            syn_dir / "clip.timing.jsonl", frame_count=100,
        )
        _, prov, path = _validate_sidecar_ingest(
            ingest_path=mp4,
            clip_id="clip",
            resolved_config={"stages": {"ingest": {"allow_synthetic_sidecars": True}}},
            out_root=tmp_path / "out",
        )
        assert prov == "synthetic"
        assert "_synthetic_sidecars" in path

    def test_synthetic_disabled_by_default(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path / "raw")
        (tmp_path / "raw").mkdir(exist_ok=True)
        mp4 = _make_mp4_stub(tmp_path / "raw", "clip.mp4")
        # Synthetic sidecar exists but config is default (false)
        syn_dir = tmp_path / "out" / "_synthetic_sidecars"
        syn_dir.mkdir(parents=True)
        generate_synthetic_sidecar(
            syn_dir / "clip.timing.jsonl", frame_count=100,
        )
        with pytest.raises(PipelineError, match="Sidecar ingest gate failed"):
            _validate_sidecar_ingest(
                ingest_path=mp4,
                clip_id="clip",
                resolved_config={},
                out_root=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# Ingest gate — invalid sidecars fail
# ---------------------------------------------------------------------------

class TestIngestGateInvalid:
    def test_missing_sidecar(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        with pytest.raises(PipelineError, match="Sidecar ingest gate failed"):
            _validate_sidecar_ingest(
                ingest_path=mp4,
                clip_id="clip",
                resolved_config={},
                out_root=tmp_path,
            )

    def test_schema_4_rejected(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100, schema=4,
        )
        with pytest.raises(PipelineError, match="Sidecar ingest gate failed"):
            _validate_sidecar_ingest(
                ingest_path=mp4,
                clip_id="clip",
                resolved_config={},
                out_root=tmp_path,
            )

    def test_source_pts_false_rejected(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100, source_pts=False,
        )
        with pytest.raises(PipelineError, match="Sidecar ingest gate failed"):
            _validate_sidecar_ingest(
                ingest_path=mp4,
                clip_id="clip",
                resolved_config={},
                out_root=tmp_path,
            )

    def test_cfr_grid_rejected(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100,
            timing_mode="cfr_grid",
        )
        with pytest.raises(PipelineError, match="Sidecar ingest gate failed"):
            _validate_sidecar_ingest(
                ingest_path=mp4,
                clip_id="clip",
                resolved_config={},
                out_root=tmp_path,
            )


# ---------------------------------------------------------------------------
# Resolver unit tests
# ---------------------------------------------------------------------------

class TestResolveSidecarPath:
    def test_sibling_when_synthetic_disabled(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        path, prov = resolve_sidecar_path(mp4, {}, tmp_path)
        assert prov == "real"
        assert path == tmp_path / "clip.timing.jsonl"

    def test_synthetic_when_enabled_and_present(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path / "raw")
        syn_dir = tmp_path / "out" / "_synthetic_sidecars"
        syn_dir.mkdir(parents=True)
        generate_synthetic_sidecar(syn_dir / "clip.timing.jsonl", frame_count=10)
        cfg = {"stages": {"ingest": {"allow_synthetic_sidecars": True}}}
        path, prov = resolve_sidecar_path(mp4, cfg, tmp_path / "out")
        assert prov == "synthetic"
        assert path == syn_dir / "clip.timing.jsonl"

    def test_sibling_fallback_when_enabled_but_absent(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        cfg = {"stages": {"ingest": {"allow_synthetic_sidecars": True}}}
        path, prov = resolve_sidecar_path(mp4, cfg, tmp_path / "out")
        assert prov == "real"
        assert path == tmp_path / "clip.timing.jsonl"

    def test_provenance_correct_for_each_case(self, tmp_path):
        """Provenance string is exactly 'real' or 'synthetic', never anything else."""
        mp4 = _make_mp4_stub(tmp_path)
        _, prov_real = resolve_sidecar_path(mp4, {}, tmp_path)
        assert prov_real == "real"

        syn_dir = tmp_path / "_synthetic_sidecars"
        syn_dir.mkdir()
        generate_synthetic_sidecar(syn_dir / "clip.timing.jsonl", frame_count=10)
        cfg = {"stages": {"ingest": {"allow_synthetic_sidecars": True}}}
        _, prov_syn = resolve_sidecar_path(mp4, cfg, tmp_path)
        assert prov_syn == "synthetic"


# ---------------------------------------------------------------------------
# Audit fields round-trip (Step 3 — dropped fields restored)
# ---------------------------------------------------------------------------

class TestAuditFields:
    def test_sidecar_data_returned_with_all_fields(self, tmp_path):
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100,
        )
        sc_data, prov, path = _validate_sidecar_ingest(
            ingest_path=mp4,
            clip_id="clip",
            resolved_config={},
            out_root=tmp_path,
        )
        assert sc_data.attempt is not None
        assert sc_data.nominal_dt_s is not None
        assert sc_data.pts_wallclock_offset_s is not None
        assert sc_data.showinfo_offset_status is not None


# ---------------------------------------------------------------------------
# Provenance stamp round-trip
# ---------------------------------------------------------------------------

class TestProvenanceStamp:
    def test_provenance_in_audit_round_trip(self, tmp_path):
        """Verify provenance stamp is readable from disk in a separate context."""
        mp4 = _make_mp4_stub(tmp_path)
        generate_synthetic_sidecar(
            tmp_path / "clip.timing.jsonl", frame_count=100,
        )
        sc_data, prov, resolved_path = _validate_sidecar_ingest(
            ingest_path=mp4,
            clip_id="clip",
            resolved_config={},
            out_root=tmp_path,
        )
        # Simulate the audit event that run_pipeline would write
        audit_path = tmp_path / "audit.jsonl"
        event = {
            "event": "sidecar_validated",
            "clip_id": "clip",
            "sidecar_provenance": prov,
            "sidecar_path": resolved_path,
        }
        audit_path.write_text(json.dumps(event) + "\n")

        # Read back from disk (simulates a separate process)
        loaded = json.loads(audit_path.read_text().strip())
        assert loaded["sidecar_provenance"] == "real"
        assert loaded["sidecar_path"] == resolved_path
        assert loaded["event"] == "sidecar_validated"


# ---------------------------------------------------------------------------
# FrameIterator — frame-0 exempt, frame>0 fails on bad POS_MSEC
# ---------------------------------------------------------------------------

_LEGACY_CLIP = Path("data/raw/nest/_eval_gt/PPDmUg/2026-03-18/20/PPDmUg-20260318-training.mp4")


class TestFrameIterator:
    @pytest.mark.skipif(
        not _LEGACY_CLIP.exists(),
        reason="Legacy GT corpus not present (data/raw/nest/_eval_gt/)",
    )
    def test_frame_0_no_raise(self):
        """Frame 0 must yield ts_ms=0 without raising.

        Requires local GT corpus: PPDmUg-20260318-training.mp4 (5MB).
        """
        it = FrameIterator(_LEGACY_CLIP)
        packets = []
        for pkt in it:
            packets.append(pkt)
            if pkt.frame_index >= 2:
                break
        assert len(packets) >= 2
        assert packets[0].frame_index == 0
        assert packets[0].timestamp_ms == 0
        assert packets[1].frame_index == 1
        assert packets[1].timestamp_ms > 0

    def test_bad_pos_msec_raises_at_frame_gt0(self):
        """If POS_MSEC returns 0.0 at frame_index > 0, RuntimeError is raised."""
        import cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,  # bad: 0.0 at all frames
        }.get(prop, 0.0)
        # First read: ok frame, second read: ok frame, third: stop
        frame = MagicMock()
        mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]

        it = FrameIterator(Path("/fake/clip.mp4"))
        it._cap = mock_cap
        it._fps = 15.0

        frames = []
        with pytest.raises(RuntimeError, match="CAP_PROP_POS_MSEC=0.0 at frame_index=1"):
            for pkt in it:
                frames.append(pkt)

        # Frame 0 should have succeeded (exempt)
        assert len(frames) == 1
        assert frames[0].frame_index == 0
        assert frames[0].timestamp_ms == 0
