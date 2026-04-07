"""Background subtraction — build per-camera background models and detect foreground."""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
from loguru import logger

_DEFAULT_BG_DIR = Path("data/background_models")


class BBox(NamedTuple):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


def build_background_model(
    cam_id: str,
    clip_paths: List[str | Path],
    sample_rate: int = 1,
    output_dir: Path | None = None,
) -> np.ndarray:
    """Build a per-pixel median background model from empty-gym footage.

    Parameters
    ----------
    cam_id : Camera identifier.
    clip_paths : Paths to empty-gym video clips (multiple times of day for
        lighting variation).
    sample_rate : Sample every Nth frame (1 = every frame).
    output_dir : Where to save the .npy model. Defaults to data/background_models/.

    Returns
    -------
    Background model as uint8 grayscale ndarray (H, W).
    """
    output_dir = output_dir or _DEFAULT_BG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: List[np.ndarray] = []
    for clip_path in clip_paths:
        clip_path = Path(clip_path)
        if not clip_path.exists():
            logger.warning(f"Clip not found, skipping: {clip_path}")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.warning(f"Cannot open clip: {clip_path}")
            continue

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            idx += 1
        cap.release()
        logger.info(f"Sampled {len(frames)} frames from {clip_path.name}")

    if not frames:
        raise ValueError(f"No frames collected for camera {cam_id}")

    # Stack and compute per-pixel median
    stack = np.stack(frames, axis=0)
    bg_model = np.median(stack, axis=0).astype(np.uint8)

    out_path = output_dir / f"{cam_id}_bg.npy"
    np.save(str(out_path), bg_model)
    logger.info(f"Background model saved: {out_path} ({bg_model.shape})")
    return bg_model


def load_background_model(cam_id: str, bg_dir: Path | None = None) -> np.ndarray:
    """Load a previously saved background model."""
    bg_dir = bg_dir or _DEFAULT_BG_DIR
    path = bg_dir / f"{cam_id}_bg.npy"
    if not path.exists():
        raise FileNotFoundError(f"No background model for camera {cam_id}: {path}")
    return np.load(str(path))


def detect_foreground(
    bg_model: np.ndarray,
    frame: np.ndarray,
    threshold: int = 30,
    min_area: int = 500,
) -> List[BBox]:
    """Detect foreground objects via frame differencing against background model.

    Parameters
    ----------
    bg_model : Grayscale background model (H, W) uint8.
    frame : Current frame (BGR or grayscale).
    threshold : Absolute difference threshold for binary mask.
    min_area : Minimum contour area in pixels to keep.

    Returns
    -------
    List of bounding boxes for detected foreground regions.
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Absolute difference
    diff = cv2.absdiff(gray, bg_model)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    # Binary threshold
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Morphological open (remove small noise) then close (fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[BBox] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append(BBox(x1=x, y1=y, x2=x + w, y2=y + h))

    return bboxes
