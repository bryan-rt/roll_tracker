"""Variable-dt Kalman filter for BoT-SORT bounding-box tracking.

Subclasses boxmot's KalmanFilterXYWH to rebuild the state-transition matrix
per step from a dt *ratio* (dt_s / nominal_dt_s) rather than the baked dt=1.0.

The ratio formulation keeps velocity in pixels-per-nominal-frame, so all
process-noise and initial-covariance constants in KalmanFilterXYWH remain
correctly calibrated without rescaling.  A ratio of 1.0 reproduces stock
behavior exactly; ~2.0 represents a gap (missed frame); ~0.5 represents a
30fps block inside a 15fps stream.

Depends on boxmot internals (verified against boxmot==16.0.8, V1-V8):
  - BaseKalmanFilter._motion_mat (8x8 ndarray, off-diag positions [i, ndim+i])
  - BaseKalmanFilter.ndim (int, =4 for XYWH)
  - predict(mean, covariance) signature
  - multi_predict(mean, covariance) signature

Process-noise dt-scaling (sqrt(dt) for continuous-time white noise) is a
recorded follow-up — under the ratio formulation it is genuinely second-order.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH


class VariableDtKalmanFilterXYWH(KalmanFilterXYWH):
    """KalmanFilterXYWH with per-step dt ratio instead of baked dt=1.0."""

    def __init__(self) -> None:
        super().__init__()
        # Must be set before every predict/multi_predict call.
        # None = unset sentinel — raises RuntimeError if reached.
        self.current_dt: float | None = None
        # Instrumentation: proves multi_predict entered the subclass (V5 trap).
        self._multi_predict_entered: bool = False

    def _rebuild_motion_mat(self, dt: float) -> None:
        """Overwrite the velocity-to-position entries with the given dt ratio.

        Mutates self._motion_mat in-place (4 assignments, no allocation).
        """
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_dt is None:
            raise RuntimeError(
                "VariableDtKalmanFilterXYWH.predict: current_dt not set. "
                "The caller must call set_dt() before each tracker update."
            )
        self._rebuild_motion_mat(self.current_dt)
        return super().predict(mean, covariance)

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_dt is None:
            raise RuntimeError(
                "VariableDtKalmanFilterXYWH.multi_predict: current_dt not set. "
                "The caller must call set_dt() before each tracker update."
            )
        self._rebuild_motion_mat(self.current_dt)
        self._multi_predict_entered = True
        return super().multi_predict(mean, covariance)
