from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Tuple, Optional, Sequence, Dict, List

import cv2
import numpy as np

from bjj_pipeline.viz.video_writer import VideoWriter
from bjj_pipeline.viz.mat_view import render_mat_canvas
from bjj_pipeline.stages.detect_track.types import OverlayItem
from bjj_pipeline.viz.overlay import overlay_on_frame


@dataclass(frozen=True)
class MuxVisualizer:
    annotated_path: Path
    mat_view_path: Path
    fps: float
    frame_size: Tuple[int, int]
    mat_size: Tuple[int, int] = (640, 640)
    mat_blueprint: Any = None
    _trail_len: int = 18
    _trails: Dict[str, List[Tuple[float, float, int]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.annotated_path.parent.mkdir(parents=True, exist_ok=True)
        self.mat_view_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> Tuple[VideoWriter, VideoWriter]:
        ann = VideoWriter(self.annotated_path, fps=self.fps, frame_size=self.frame_size)
        mat = VideoWriter(self.mat_view_path, fps=self.fps, frame_size=self.mat_size)
        return ann, mat

    def render_frame(self, frame_bgr: np.ndarray, idx: int, overlays: Optional[Sequence[OverlayItem]] = None) -> Tuple[np.ndarray, np.ndarray]:
        out = frame_bgr.copy()
        # draw overlays using shared helper
        if overlays:
            overlay_on_frame(out, overlays, alpha=0.35)

        cv2.putText(out, f"frame={idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Build mat-view payload: age trails and add newest points
        points: List[Tuple[float, float, str, Optional[bool]]] = []
        for tid, trail in list(self._trails.items()):
            aged: List[Tuple[float, float, int]] = []
            for (x, y, age) in trail:
                age2 = int(age) + 1
                if age2 <= self._trail_len:
                    aged.append((float(x), float(y), age2))
            self._trails[tid] = aged
            if not aged:
                del self._trails[tid]

        if overlays:
            for ov in overlays:
                if ov.x_m is None or ov.y_m is None:
                    continue
                tid = str(ov.tracklet_id)
                x = float(ov.x_m)
                y = float(ov.y_m)
                points.append((x, y, tid, ov.on_mat))
                cur = self._trails.get(tid, [])
                cur.insert(0, (x, y, 0))
                if len(cur) > self._trail_len:
                    cur = cur[: self._trail_len]
                self._trails[tid] = cur

        mat = render_mat_canvas(
            blueprint=self.mat_blueprint,
            width=self.mat_size[0],
            height=self.mat_size[1],
            points=points if points else None,
            trails=self._trails if self._trails else None,
            frame_index=idx,
            title=None,
        )
        cv2.putText(mat, f"frame={idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return out, mat


def load_mat_blueprint(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
