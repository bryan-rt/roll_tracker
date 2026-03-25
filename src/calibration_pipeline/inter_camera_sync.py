"""Inter-camera sync — cross-camera affine alignment via mat walk.

CP18 will implement this module.

Inputs
------
- Per-camera refined homographies (from mat_walk or overlay calibration)
- Labeled world coordinate correspondences from mat walk (shared across cameras)
- Per-camera lens calibration (K + distortion)

Outputs
-------
- Global affine transform aligning all cameras to a shared world coordinate system
- Residual alignment error per camera pair

Algorithm (planned)
-------------------
1. Collect labeled correspondences from mat walk across all cameras
2. For each camera pair with shared world points:
   a. Project world points through each camera's homography to get predicted
      pixel positions
   b. Compute least-squares affine transform between camera coordinate systems
3. Chain affine transforms to build global alignment
4. Write refined homographies to Supabase and local config

Three correction layers with different update frequencies:
  (1) Lens calibration — one-time per camera, essentially permanent
  (2) Per-camera homography — nightly recalibration attempt
  (3) Inter-camera affine alignment — derived from mat walk
"""


def run(**kwargs):  # type: ignore[no-untyped-def]
    """Run inter-camera sync. Not yet implemented — see CP18."""
    raise NotImplementedError(
        "Inter-camera sync is planned for CP18. "
        "See module docstring for intended algorithm."
    )
