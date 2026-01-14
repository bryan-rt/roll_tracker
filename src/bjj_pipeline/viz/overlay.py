from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from bjj_pipeline.stages.detect_track.types import OverlayItem


def overlay_on_frame(frame_bgr: np.ndarray, overlay_items: Sequence[OverlayItem], alpha: float = 0.35) -> None:
    """Draw masks, bboxes, and labels onto a BGR frame in-place.

    Safe no-op if OpenCV is unavailable.
    """
    if cv2 is None:
        return

    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])

    for ov in overlay_items:
        # mask overlay (semi-transparent)
        if ov.mask is not None and isinstance(ov.mask, np.ndarray) and ov.mask.ndim == 2:
            m = ov.mask
            if m.shape[0] == h and m.shape[1] == w:
                colored = np.zeros_like(frame_bgr)
                colored[:, :, 1] = 180  # green channel
                sel = m.astype(bool)
                frame_bgr[sel] = (frame_bgr[sel] * (1.0 - alpha) + colored[sel] * alpha).astype(frame_bgr.dtype)

        # bbox + label
        x1, y1, x2, y2 = int(round(ov.x1)), int(round(ov.y1)), int(round(ov.x2)), int(round(ov.y2))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{ov.tracklet_id} conf={ov.confidence:.2f}"
        if ov.mask_source:
            label += f" {ov.mask_source}"
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
