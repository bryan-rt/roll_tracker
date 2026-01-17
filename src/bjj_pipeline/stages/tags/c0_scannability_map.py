from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ScannabilityMap:
    grid: np.ndarray  # shape (H, W), float32 in [0,1]
    image_w: int
    image_h: int

    def sample(self, cx: float, cy: float) -> float:
        cx = max(0.0, min(float(self.image_w - 1), float(cx)))
        cy = max(0.0, min(float(self.image_h - 1), float(cy)))

        gh, gw = self.grid.shape
        gx = int(round(cx / (self.image_w - 1) * (gw - 1)))
        gy = int(round(cy / (self.image_h - 1) * (gh - 1)))
        gx = max(0, min(gw - 1, gx))
        gy = max(0, min(gh - 1, gy))

        return float(max(0.0, min(1.0, float(self.grid[gy, gx]))))


def load_scannability_map(path: Path) -> Optional[ScannabilityMap]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        grid = np.asarray(data["grid"], dtype=np.float32).reshape((data["grid_h"], data["grid_w"]))
        return ScannabilityMap(
            grid=grid,
            image_w=int(data["image_w"]),
            image_h=int(data["image_h"]),
        )
    except Exception:
        return None
