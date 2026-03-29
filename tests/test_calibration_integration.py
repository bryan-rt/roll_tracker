"""Integration tests for CP18 calibration pipeline (v2).

Loads real calibration data from outputs/calibration_test/ and verifies
that Layer 1 calibration produces corrections that improve inside-mat %.

v2 additions: footpath-only fallback tests, mat line result integration.
"""

from pathlib import Path

import pandas as pd
import pytest

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import classify_tracklets
from calibration_pipeline.mat_walk import calibrate_single_camera, CalibrationResult
from calibration_pipeline.mat_line_detection import MatLineResult, DetectedMatLine


BLUEPRINT_PATH = Path("configs/mat_blueprint.json")
OUTPUTS_DIR = Path("outputs/calibration_test")

CAMERAS = {
    "FP7oJQ": "FP7oJQ-20260326211406-20260326211906",
    "J_EDEw": "J_EDEw-20260326211213-20260326211713",
    "PPDmUg": "PPDmUg-20260326211000-20260326211430",
}


def _load_bank_frames(camera_id: str, clip_id: str) -> pd.DataFrame:
    """Load D0 tracklet_bank_frames for a single clip."""
    path = OUTPUTS_DIR / camera_id / "2026-03-26" / "20" / clip_id / "stage_D" / "tracklet_bank_frames.parquet"
    if not path.exists():
        pytest.skip(f"Calibration data not available: {path}")
    return pd.read_parquet(path)


@pytest.fixture
def blueprint():
    if not BLUEPRINT_PATH.exists():
        pytest.skip("Blueprint not available")
    return MatBlueprint.from_json(BLUEPRINT_PATH)


class TestLayer1Integration:
    """Run Layer 1 on real calibration data and verify improvement."""

    @pytest.mark.parametrize("camera_id,clip_id", [
        ("FP7oJQ", CAMERAS["FP7oJQ"]),
        ("J_EDEw", CAMERAS["J_EDEw"]),
        ("PPDmUg", CAMERAS["PPDmUg"]),
    ])
    def test_calibration_does_not_crash(self, blueprint, camera_id, clip_id):
        """Calibration runs without error on real data (footpath-only)."""
        df = _load_bank_frames(camera_id, clip_id)
        features = classify_tracklets(df, blueprint, fps=30.0)
        result = calibrate_single_camera(features, blueprint, camera_id)
        assert isinstance(result, CalibrationResult)
        assert result.confidence in ("high", "medium", "low", "inconclusive")

    @pytest.mark.parametrize("camera_id,clip_id", [
        ("FP7oJQ", CAMERAS["FP7oJQ"]),
        ("J_EDEw", CAMERAS["J_EDEw"]),
        ("PPDmUg", CAMERAS["PPDmUg"]),
    ])
    def test_inside_mat_does_not_decrease(self, blueprint, camera_id, clip_id):
        """If a correction is applied, inside-mat % should not decrease."""
        df = _load_bank_frames(camera_id, clip_id)
        features = classify_tracklets(df, blueprint, fps=30.0)
        result = calibrate_single_camera(features, blueprint, camera_id)
        if result.correction_matrix is not None:
            assert result.inside_mat_fraction_after >= result.inside_mat_fraction_before

    @pytest.mark.parametrize("camera_id,clip_id", [
        ("FP7oJQ", CAMERAS["FP7oJQ"]),
        ("J_EDEw", CAMERAS["J_EDEw"]),
        ("PPDmUg", CAMERAS["PPDmUg"]),
    ])
    def test_signal_type_populated(self, blueprint, camera_id, clip_id):
        """Result should have signal_type set when correction applied."""
        df = _load_bank_frames(camera_id, clip_id)
        features = classify_tracklets(df, blueprint, fps=30.0)
        result = calibrate_single_camera(features, blueprint, camera_id)
        if result.correction_matrix is not None:
            assert result.signal_type in (
                "mat_lines+footpath", "mat_lines_only", "footpath_only"
            )


class TestFootpathOnlyFallback:
    """Verify footpath-only mode works when no mat_line_result provided."""

    @pytest.mark.parametrize("camera_id,clip_id", [
        ("FP7oJQ", CAMERAS["FP7oJQ"]),
    ])
    def test_footpath_only_no_mat_lines(self, blueprint, camera_id, clip_id):
        """With no mat_line_result, should use footpath_only signal."""
        df = _load_bank_frames(camera_id, clip_id)
        features = classify_tracklets(df, blueprint, fps=30.0)
        result = calibrate_single_camera(
            features, blueprint, camera_id, mat_line_result=None
        )
        if result.correction_matrix is not None:
            assert result.signal_type == "footpath_only"
            assert result.n_matched_lines == 0


class TestEdgeCases:
    def test_empty_features(self, blueprint):
        """Empty feature list produces inconclusive result."""
        result = calibrate_single_camera([], blueprint, "empty_cam")
        assert result.confidence == "inconclusive"
        assert result.correction_matrix is None

    def test_all_lingering(self, blueprint):
        """All lingering tracklets (no cleaning) produces inconclusive."""
        df = pd.DataFrame({
            "tracklet_id": ["t1"] * 100,
            "frame_index": list(range(100)),
            "x_m": [52.0] * 100,
            "y_m": [46.0] * 100,
            "camera_id": ["test"] * 100,
            "clip_id": ["test"] * 100,
        })
        features = classify_tracklets(df, blueprint, fps=30.0)
        result = calibrate_single_camera(features, blueprint, "test_cam")
        assert result.confidence == "inconclusive"

    def test_empty_mat_line_result(self, blueprint):
        """Empty MatLineResult should fall through to footpath."""
        empty_mlr = MatLineResult(
            n_frames_analyzed=5,
            n_lines_detected=0,
            n_lines_matched=0,
            matched_lines=[],
            projected_blueprint_edges_px=[],
        )
        result = calibrate_single_camera(
            [], blueprint, "test_cam", mat_line_result=empty_mlr
        )
        assert result.confidence == "inconclusive"
