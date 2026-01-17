from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from .c0_scannability_map import ScannabilityMap


@dataclass
class GateResult:
    scannable: bool
    reason: str
    roi_xyxy: Tuple[int, int, int, int]
    roi_w: int
    roi_h: int
    roi_area: int
    blur_var: Optional[float]
    contrast_std: Optional[float]
    prior: Optional[float]
    on_mat: Optional[bool]


def pad_and_clip_bbox(x1, y1, x2, y2, *, pad_frac, frame_w, frame_h):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pw = int(round(w * pad_frac))
    ph = int(round(h * pad_frac))
    nx1 = max(0, x1 - pw)
    ny1 = max(0, y1 - ph)
    nx2 = min(frame_w - 1, x2 + pw)
    ny2 = min(frame_h - 1, y2 + ph)
    if nx2 <= nx1:
        nx2 = min(frame_w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(frame_h - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _blur_var(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _contrast_std(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())


def evaluate_scannability(
    *,
    frame_bgr,
    x1,
    y1,
    x2,
    y2,
    det_conf,
    on_mat,
    gating_cfg: Dict,
    prior_map: Optional[ScannabilityMap],
):
    if not gating_cfg.get("enabled", True):
        return GateResult(
            True,
            "ok",
            (x1, y1, x2, y2),
            x2 - x1,
            y2 - y1,
            (x2 - x1) * (y2 - y1),
            None,
            None,
            None,
            on_mat,
        )

    if float(det_conf) < float(gating_cfg.get("min_det_conf", 0.0)):
        return GateResult(
            False,
            "skip_det_conf_low",
            (x1, y1, x2, y2),
            x2 - x1,
            y2 - y1,
            (x2 - x1) * (y2 - y1),
            None,
            None,
            None,
            on_mat,
        )

    fh, fw = frame_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = pad_and_clip_bbox(
        x1,
        y1,
        x2,
        y2,
        pad_frac=gating_cfg.get("bbox_pad_frac", 0.15),
        frame_w=fw,
        frame_h=fh,
    )

    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    area = rw * rh

    if (
        rw < int(gating_cfg.get("min_roi_side_px", 0))
        or rh < int(gating_cfg.get("min_roi_side_px", 0))
        or area < int(gating_cfg.get("min_roi_area_px", 0))
    ):
        return GateResult(
            False,
            "skip_roi_too_small",
            (rx1, ry1, rx2, ry2),
            rw,
            rh,
            area,
            None,
            None,
            None,
            on_mat,
        )

    if gating_cfg.get("require_on_mat", False) and on_mat is False:
        return GateResult(
            False,
            "skip_off_mat",
            (rx1, ry1, rx2, ry2),
            rw,
            rh,
            area,
            None,
            None,
            None,
            on_mat,
        )

    prior = None
    if gating_cfg.get("use_scannability_prior", False) and prior_map is not None:
        prior = float(prior_map.sample((rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5))
        if prior < float(gating_cfg.get("prior_min", 0.0)):
            return GateResult(
                False,
                "skip_low_prior",
                (rx1, ry1, rx2, ry2),
                rw,
                rh,
                area,
                None,
                None,
                prior,
                on_mat,
            )

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    blur = _blur_var(roi)
    if blur < float(gating_cfg.get("blur", {}).get("min_var", 0.0)):
        return GateResult(
            False,
            "skip_blurry",
            (rx1, ry1, rx2, ry2),
            rw,
            rh,
            area,
            blur,
            None,
            prior,
            on_mat,
        )

    contrast = _contrast_std(roi)
    if contrast < float(gating_cfg.get("contrast", {}).get("min_std", 0.0)):
        return GateResult(
            False,
            "skip_low_contrast",
            (rx1, ry1, rx2, ry2),
            rw,
            rh,
            area,
            blur,
            contrast,
            prior,
            on_mat,
        )

    return GateResult(
        True,
        "ok",
        (rx1, ry1, rx2, ry2),
        rw,
        rh,
        area,
        blur,
        contrast,
        prior,
        on_mat,
    )
