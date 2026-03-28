"""Integration tests for CP18 calibration pipeline.

Loads real calibration data from outputs/calibration_test/ and verifies
that Layer 1 calibration produces corrections that improve inside-mat %.
"""

from pathlib import Path

import pandas as pd
import pytest

from calibration_pipeline.blueprint_geometry import MatBlueprint
from calibration_pipeline.tracklet_classifier import classify_tracklets
from calibration_pipeline.mat_walk import calibrate_single_camera, CalibrationResult


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
        """Calibration runs without error on real data."""
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


class TestEdgeCases:
    def test_empty_features(self, blueprint):
        """Empty feature list produces inconclusive result."""
        result = calibrate_single_camera([], blueprint, "empty_cam")
        assert result.confidence == "inconclusive"
        assert result.correction_matrix is None

    def test_all_lingering(self, blueprint):
        """All lingering tracklets (no cleaning) produces inconclusive."""
        # Stationary tracklet — will be classified as lingering
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
