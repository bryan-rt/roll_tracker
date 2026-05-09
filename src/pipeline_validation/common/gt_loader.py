"""CVAT zip extraction and YOLO label parsing.

Extracts GT annotation zips to temp directories, parses 6-field YOLO
track-detection format (class cx cy w h track_id), remaps class 1->0.

CORRECTNESS CONTRACT (non-negotiable):
    The GT loader ONLY loads labels for frames defined by the model
    manifest's annotated_range x split. Specifically:
    - Iterate frames in annotated_range (start, stop, stride) intersected
      with the requested split (train or val).
    - Load GT only from those exact frame indices.
    - NEVER load GT from a frame outside annotated_range, even if a
      non-empty label file exists for it in the zip. CVAT auto-interpolates
      annotations on non-hand-labeled frames; these are NOT trusted GT.
    - annotated_range (from the manifest) is authoritative. Zip contents
      are advisory only.
"""
from __future__ import annotations

import contextlib
import logging
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

from pipeline_validation.common.manifest import (
    enumerate_annotated_frames,
    enumerate_split_frames,
)
from pipeline_validation.common.schemas import ExportEntry, GTBox

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def extract_zip(zip_path: Path):
    """Extract zip to a temp directory. Yields the temp path."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        yield Path(tmp)


def frame_index_from_filename(path: Path) -> int:
    """Extract frame index from label filename like frame_000100.txt."""
    m = re.match(r"frame_(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse frame index from {path.name}")
    return int(m.group(1))


def parse_label_file(
    path: Path, resolution: tuple[int, int]
) -> list[GTBox]:
    """Parse a 6-field YOLO label file, remap class 1->0, denormalize."""
    if not path.exists():
        return []
    content = path.read_text().strip()
    if not content:
        return []

    img_w, img_h = resolution
    boxes = []
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        track_id = int(parts[5])

        # Remap class 1 -> 0
        if cls_id == 1:
            cls_id = 0

        # Denormalize to pixel-space x1y1x2y2
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        boxes.append(GTBox(
            class_id=cls_id,
            cx=cx, cy=cy, w=w, h=h,
            track_id=track_id,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return boxes


def load_gt_for_split(
    zip_path: Path,
    export: ExportEntry,
    split: Literal["train", "val"],
) -> dict[int, list[GTBox]]:
    """Load GT boxes for the specified split, strictly filtered by annotated_range.

    Returns dict mapping frame_index -> list of GTBox.
    Logs once per call how many non-empty labels were dropped outside
    annotated_range.
    """
    split_frames = set(enumerate_split_frames(export, split))
    annotated_frames = set(enumerate_annotated_frames(export))
    resolution = (export.resolution[0], export.resolution[1])

    gt: dict[int, list[GTBox]] = {}
    dropped_non_empty = 0

    with extract_zip(zip_path) as tmp_dir:
        # Find all label files
        label_files = sorted(tmp_dir.rglob("*.txt"))
        label_files = [f for f in label_files if f.stem.startswith("frame_")]

        for lf in label_files:
            fidx = frame_index_from_filename(lf)

            if fidx in split_frames:
                gt[fidx] = parse_label_file(lf, resolution)
            elif fidx not in annotated_frames:
                # Outside annotated_range entirely
                content = lf.read_text().strip()
                if content:
                    dropped_non_empty += 1

    if dropped_non_empty > 0:
        logger.info(
            "%s/%s: Dropped %d non-empty label files outside annotated_range",
            export.camera_id, split, dropped_non_empty,
        )

    # Ensure every split frame has an entry (even if label was missing/empty)
    for fidx in split_frames:
        if fidx not in gt:
            gt[fidx] = []

    return gt
