"""F0 projection utility — canonical pixel-to-world coordinate transform.

This is the **only** permitted projection path in src/bjj_pipeline/.
All pixel → world coordinate transforms must go through ``project_to_world()``.

CP16a: identity fallback (no undistortion) when camera_matrix / dist_coefficients
are None.  CP16b will supply real lens calibration values.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from loguru import logger


class CameraProjection(NamedTuple):
    """Pre-loaded projection parameters for a single camera."""

    H: np.ndarray  # 3x3, image-pixels → world/mat (direction-corrected)
    camera_matrix: Optional[np.ndarray]  # 3x3 intrinsic K, or None
    dist_coefficients: Optional[np.ndarray]  # (4,) [k1,k2,p1,p2], or None


_cv2_warned: bool = False


def project_to_world(
    pixel_xy: Tuple[float, float],
    H: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coefficients: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Project a single pixel coordinate to world/mat space.

    Parameters
    ----------
    pixel_xy : (u, v) in image-pixel coordinates.
    H : 3x3 homography matrix (image-pixels → world/mat). Must already be
        direction-corrected (img→mat, not mat→img).
    camera_matrix : 3x3 intrinsic camera matrix K.  If provided together with
        *dist_coefficients*, ``cv2.undistortPoints`` is applied before the
        homography.  If ``None``, undistortion is skipped (identity fallback).
    dist_coefficients : distortion vector ``[k1, k2, p1, p2]``.  ``None``
        means no undistortion.

    Returns
    -------
    (x_m, y_m) in world/mat coordinates, or ``(nan, nan)`` on degenerate
    homography (division by zero).
    """
    u, v = pixel_xy

    # --- Step 1: optional lens undistortion ---
    if camera_matrix is not None and dist_coefficients is not None:
        try:
            import cv2  # lazy import — hot path skips when K is None

            pts = np.array([[[u, v]]], dtype=np.float64)
            K = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
            D = np.asarray(dist_coefficients, dtype=np.float64).ravel()
            # P=K re-projects undistorted points back to pixel space so
            # the downstream homography (which expects pixel coords) works.
            undistorted = cv2.undistortPoints(pts, K, D, P=K)
            u = float(undistorted[0, 0, 0])
            v = float(undistorted[0, 0, 1])
        except ImportError:
            global _cv2_warned  # noqa: PLW0603
            if not _cv2_warned:
                logger.warning(
                    "cv2 not available — skipping lens undistortion. "
                    "Install opencv-python to enable."
                )
                _cv2_warned = True

    # --- Step 2: homography projection ---
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    w = float(q[2])
    if w == 0.0:
        return (float("nan"), float("nan"))
    x_m = float(q[0] / w)
    y_m = float(q[1] / w)
    return (x_m, y_m)


def load_calibration_from_payload(
    payload: dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract camera_matrix and dist_coefficients from a homography.json payload.

    Returns (camera_matrix, dist_coefficients) — both ``None`` when absent or
    explicitly null.  This is a pure helper for the stage loaders.
    """
    cm_raw = payload.get("camera_matrix")
    dc_raw = payload.get("dist_coefficients")

    camera_matrix: Optional[np.ndarray] = None
    dist_coefficients: Optional[np.ndarray] = None

    if cm_raw is not None:
        camera_matrix = np.asarray(cm_raw, dtype=np.float64).reshape((3, 3))
    if dc_raw is not None:
        dist_coefficients = np.asarray(dc_raw, dtype=np.float64).ravel()

    return camera_matrix, dist_coefficients
