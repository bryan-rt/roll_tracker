from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ms: int
    image_bgr: np.ndarray


class FrameIterator:
    """Single-pass iterator over video frames.

    Invariants:
    - frame_index is 0-based and increments by 1 for each yielded frame.
    - timestamp_ms is deterministic for the same input and iterator settings.
    """

    def __init__(self, ingest_path: Path):
        self.ingest_path = Path(ingest_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.ingest_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.ingest_path}")
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 0 else 0.0
        return cap

    @property
    def fps(self) -> float:
        if self._fps is None:
            # lazily open to read fps
            cap = self._open()
            # do not consume frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return float(self._fps or 0.0)

    def __iter__(self) -> Iterator[FramePacket]:
        cap = self._cap or self._open()
        frame_index = 0
        fps = self._fps or 0.0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # Prefer container timestamp if available; fall back to fps-derived timebase.
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                ts_ms = int(round(pos_msec))
            elif fps and fps > 0:
                ts_ms = int(round(1000.0 * frame_index / fps))
            else:
                ts_ms = 0
            yield FramePacket(frame_index=frame_index, timestamp_ms=ts_ms, image_bgr=frame)
            frame_index += 1
