"""Mat walk calibration — grid pattern detection from tagged person walk.

CP18 will implement this module.

Inputs
------
- Video of a single known tagged person walking a grid pattern on the mat
- Per-camera lens calibration (K + distortion from lens_calibration.py)
- AprilTag detection of the walker's tag for identity confirmation

Outputs
-------
- Labeled world coordinate correspondences at grid intersections
- Per-camera refined homography computed from undistorted correspondences
- Inter-camera affine alignment (shared correspondences across cameras)

Algorithm (planned)
-------------------
1. Detect the tagged walker in each frame via YOLO + AprilTag
2. Extract foot contact points at grid intersections (known world coords)
3. Collect correspondences across the mat surface
4. Compute refined homography from dense correspondences (vs 4-corner overlay)
5. For inter-camera sync: identify shared timestamps where walker appears in
   overlapping camera views, use least-squares affine solve
"""


def run(**kwargs):  # type: ignore[no-untyped-def]
    """Run mat walk calibration. Not yet implemented — see CP18."""
    raise NotImplementedError(
        "Mat walk calibration is planned for CP18. "
        "See module docstring for intended algorithm."
    )
