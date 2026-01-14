"""Stage A (detect_track) — typed structs used by detector/tracker/processor.

These are intentionally lightweight dataclasses so that:
  - multiplex mode can pass in-memory structs around cleanly
  - multipass mode can deterministically serialize to parquet via writer helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


MaskSource = Literal["yolo_seg", "bbox_fallback", "sam"]


@dataclass
class Detection:
    clip_id: str
    camera_id: str
    frame_index: int
    timestamp_ms: int

    detection_id: str  # deterministic: d{frame_index:06d}_{k}
    class_name: str
    confidence: float

    # xyxy bbox in pixel coords
    x1: float
    y1: float
    x2: float
    y2: float

    # Full-frame mask (HxW) in {0,1} uint8 or bool. Optional.
    mask: Optional[np.ndarray] = None
    mask_source: Optional[MaskSource] = None
    mask_quality: Optional[float] = None


@dataclass
class TrackedDetection:
    detection_id: str
    tracklet_id: str
    local_track_conf: Optional[float]

    # bbox used for association in this frame (may be mask-tight if enabled)
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class TrackState:
  """Per-tracklet state for velocity + physics."""
  tracklet_id: str
  last_frame: int
  last_x_m: float
  last_y_m: float
  last_timestamp_ms: int


@dataclass
class OverlayItem:
    """Per-frame visualization payload for debug videos.

    These are debug-only and are not persisted to parquet directly.
    """
    tracklet_id: str
    detection_id: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    mask: Optional[np.ndarray] = None  # uint8 {0,1} in full-frame resolution
    mask_source: Optional[MaskSource] = None
    u_px: Optional[float] = None
    v_px: Optional[float] = None
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    on_mat: Optional[bool] = None
