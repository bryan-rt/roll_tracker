import cv2  # type: ignore
import numpy as np

from bjj_pipeline.stages.tags.c0_gating import evaluate_scannability


def _sharp_high_contrast():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
    return img


def test_scannable_sharp_high_contrast():
    frame = _sharp_high_contrast()
    res = evaluate_scannability(
        frame_bgr=frame,
        x1=10,
        y1=10,
        x2=90,
        y2=90,
        det_conf=0.9,
        on_mat=True,
        gating_cfg={
            "enabled": True,
            "bbox_pad_frac": 0.15,
            "min_det_conf": 0.1,
            "min_roi_side_px": 20,
            "min_roi_area_px": 400,
            "blur": {"min_var": 10.0},
            "contrast": {"min_std": 5.0},
        },
        prior_map=None,
    )
    assert res.scannable is True


def test_skip_blurry():
    frame = cv2.GaussianBlur(_sharp_high_contrast(), (31, 31), 0)
    res = evaluate_scannability(
        frame_bgr=frame,
        x1=10,
        y1=10,
        x2=90,
        y2=90,
        det_conf=0.9,
        on_mat=True,
        gating_cfg={
            "enabled": True,
            "bbox_pad_frac": 0.15,
            "min_det_conf": 0.1,
            "min_roi_side_px": 20,
            "min_roi_area_px": 400,
            "blur": {"min_var": 100.0},
            "contrast": {"min_std": 5.0},
        },
        prior_map=None,
    )
    assert res.scannable is False
    assert res.reason == "skip_blurry"
