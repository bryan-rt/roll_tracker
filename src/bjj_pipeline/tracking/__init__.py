"""Variable-dt tracker subclasses for BoT-SORT.

Replaces boxmot's constant dt=1.0 Kalman filter with a per-frame dt ratio
derived from the schema-5 timing sidecar.  See CLAUDE.md for the boxmot
internals dependency list and version-bump verification protocol (V1-V8).
"""

from bjj_pipeline.tracking.variable_dt_kalman import VariableDtKalmanFilterXYWH
from bjj_pipeline.tracking.variable_dt_botsort import VariableDtBotSort

__all__ = ["VariableDtKalmanFilterXYWH", "VariableDtBotSort"]
