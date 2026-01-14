from __future__ import annotations

from typing import Any, Iterable, Tuple, Dict, List, Optional

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
    points: Optional[List[Tuple[float, float, str, Optional[bool]]]] = None,
    trails: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
    frame_index: Optional[int] = None,
    title: Optional[str] = None,
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
    # Optional: draw current points
    if points:
        for (x_m, y_m, tid, on_mat) in points:
            u, v = to_px(float(x_m), float(y_m))
            col = (0, 180, 0) if bool(on_mat) else (0, 0, 180)
            cv2.circle(img, (u, v), 4, col, -1)
            cv2.putText(img, str(tid), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Optional: draw trails
    if trails:
        for tid, trail in trails.items():
            for (x_m, y_m, age) in trail:
                u, v = to_px(float(x_m), float(y_m))
                alpha = max(0.15, 1.0 - (float(age) / 18.0))
                col = (int(255 * alpha), int(255 * alpha), int(0))
                cv2.circle(img, (u, v), 3, col, -1)
    # Optional title/frame index
    if title:
        cv2.putText(img, str(title), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    elif frame_index is not None:
        cv2.putText(img, f"frame={int(frame_index)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img
