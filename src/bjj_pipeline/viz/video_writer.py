from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoWriter:
    path: Path
    fps: float
    frame_size: Tuple[int, int]  # (width, height)
    fourcc: str = "mp4v"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._vw = cv2.VideoWriter(str(self.path), cc, float(self.fps), self.frame_size)
        if not self._vw.isOpened():
            raise RuntimeError(f"failed to open VideoWriter: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._vw.write(frame_bgr)

    def close(self) -> None:
        self._vw.release()
