"""Pydantic models for manifest and per-frame match results.

Match decisions are persisted at IoU >= 0.5. TB-EVAL-2 ID-switch cause
classification uses the iou column directly: matched rows with iou < 0.7
are sloppy_box candidates.
"""
from __future__ import annotations

from pydantic import BaseModel


class FrameRange(BaseModel):
    start: int
    stop: int       # inclusive last frame
    stride: int
    count: int


class SplitInfo(BaseModel):
    train: FrameRange
    val: FrameRange | None = None


class ExportEntry(BaseModel):
    export: str                          # zip filename
    source_video: str                    # video basename
    source_video_path: str | None = None # full relative path (e.g. PPDmUg)
    pipeline_output_clip_id: str | None = None  # override clip_id for pipeline outputs
    camera_id: str
    resolution: list[int]                # [w, h]
    annotated_range: FrameRange
    splits: SplitInfo


class ModelManifest(BaseModel):
    model_id: str
    weights_path: str
    pipeline_gym_id: str | None = None   # gym_id for pipeline output paths
    base_model: str
    trained_at: str
    training_config: dict
    training_data: list[ExportEntry]
    notes: str = ""


class GTBox(BaseModel):
    """Ground truth bounding box from CVAT annotation."""
    class_id: int       # always 0 after remap
    cx: float           # normalized [0,1]
    cy: float
    w: float
    h: float
    track_id: int
    x1: float = 0.0     # pixel-space, filled by denormalize
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


class GTTrackSequence(BaseModel):
    """Per-GT-track temporal sequence of assigned person_ids."""
    gt_track_id: int
    camera_id: str
    split: str
    frames: list[dict]  # [{frame_index, person_id, match_status, iou}, ...]
