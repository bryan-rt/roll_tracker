"""Drift detection — daily baseline comparison for camera drift.

CP18 will implement this module.

Inputs
------
- Empty-mat baseline snapshot (taken after successful calibration)
- Current empty-mat frame (from nest_recorder secondary pipeline window)
- Per-camera lens calibration (K + distortion)
- Per-camera homography

Outputs
-------
- Drift score (float): degree of deviation from baseline
- Alert status: whether drift exceeds threshold requiring recalibration
- Drift score written to Supabase for monitoring

Algorithm (planned)
-------------------
1. Capture baseline snapshot immediately after successful calibration
2. Daily: capture current empty-mat frame at scheduled time
3. Apply lens undistortion to both baseline and current frame
4. Edge detection (Canny or similar) on both frames
5. Compute structural similarity or edge alignment score
6. If drift score exceeds hard threshold AND recalibration quality is poor:
   notify gym owner via Supabase Realtime → push notification
7. Scheduled recalibration via nest_recorder secondary pipeline window
"""


def run(**kwargs):  # type: ignore[no-untyped-def]
    """Run drift detection. Not yet implemented — see CP18."""
    raise NotImplementedError(
        "Drift detection is planned for CP18. "
        "See module docstring for intended algorithm."
    )
