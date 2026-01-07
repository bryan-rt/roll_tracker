from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np
import cv2


def _iter_rects(blueprint: Any) -> Iterable[Tuple[float, float, float, float, str]]:
    """Yield (x, y, w, h, label) from the mat blueprint JSON.

    The repo's configs/mat_blueprint.json is currently a list of dicts with:
      - x, y, width, height
      - optional: name/label/id
    """
    if not isinstance(blueprint, list):
        return
    for item in blueprint:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            w = float(item.get("width", 0.0))
            h = float(item.get("height", 0.0))
        except Exception:
            continue
        label = str(item.get("name") or item.get("label") or item.get("id") or "")
        yield x, y, w, h, label


def render_mat_canvas(
    *,
    blueprint: Any,
    width: int = 640,
    height: int = 640,
    margin_px: int = 24,
) -> np.ndarray:
    """Render a 2D mat blueprint into a fixed-size image.

    This is a visualization helper; it does not assume units (meters vs inches).
    It just fits the blueprint bounding box into the canvas.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = 255

    rects = list(_iter_rects(blueprint))
    if not rects:
        # Fallback: blank canvas with border
        cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
        return img

    xs = [x for x, _, w, _, _ in rects] + [x + w for x, _, w, _, _ in rects]
    ys = [y for _, y, _, h, _ in rects] + [y + h for _, y, _, h, _ in rects]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    usable_w = max(width - 2 * margin_px, 1)
    usable_h = max(height - 2 * margin_px, 1)
    scale = min(usable_w / span_x, usable_h / span_y)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int(margin_px + (x - min_x) * scale)
        py = int(margin_px + (y - min_y) * scale)
        return px, py

    # Draw rects
    for x, y, w, h, label in rects:
        p1 = to_px(x, y)
        p2 = to_px(x + w, y + h)
        cv2.rectangle(img, p1, p2, (0, 0, 0), 2)
        if label:
            cv2.putText(
                img,
                label,
                (p1[0] + 6, p1[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # Outer border
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 1)
    return img
