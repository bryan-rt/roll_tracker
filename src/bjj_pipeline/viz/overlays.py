from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_text_top_left(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()
    # black shadow then white text for readability
    org = (10, 25)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out
