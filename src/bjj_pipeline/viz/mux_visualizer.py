from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

from bjj_pipeline.viz.video_writer import VideoWriter
from bjj_pipeline.viz.mat_view import render_mat_canvas


@dataclass(frozen=True)
class MuxVisualizer:
    annotated_path: Path
    mat_view_path: Path
    fps: float
    frame_size: Tuple[int, int]
    mat_size: Tuple[int, int] = (640, 640)
    mat_blueprint: Any = None

    def __post_init__(self) -> None:
        self.annotated_path.parent.mkdir(parents=True, exist_ok=True)
        self.mat_view_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> Tuple[VideoWriter, VideoWriter]:
        ann = VideoWriter(self.annotated_path, fps=self.fps, frame_size=self.frame_size)
        mat = VideoWriter(self.mat_view_path, fps=self.fps, frame_size=self.mat_size)
        return ann, mat

    def render_frame(self, frame_bgr: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        out = frame_bgr.copy()
        cv2.putText(out, f"frame={idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        mat = render_mat_canvas(blueprint=self.mat_blueprint, width=self.mat_size[0], height=self.mat_size[1])
        cv2.putText(mat, f"frame={idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return out, mat


def load_mat_blueprint(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
