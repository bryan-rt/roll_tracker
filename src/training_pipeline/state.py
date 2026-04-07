"""Persistent state tracker for training pipeline rounds and CVAT tasks."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_STATE_PATH = Path("data/training_data/pipeline_state.json")


class RoundInfo(BaseModel):
    """Record of a single training round."""

    model_config = ConfigDict(extra="allow")

    round: int
    model: str
    base_model: str
    total_frames: int
    training_date: str  # ISO date
    val_session: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    freeze: int = 10
    epochs: int = 100
    notes: str = ""


class CvatTaskInfo(BaseModel):
    """Tracking info for a CVAT annotation task."""

    task_id: int
    status: str = "created"  # created | in_progress | completed
    session_id: str = ""
    cam_id: str = ""
    video_path: str = ""  # source video path for frame extraction at download time


class PipelineState(BaseModel):
    """Top-level persistent state for the training pipeline."""

    model_config = ConfigDict(extra="allow")

    current_round: int = 0
    rounds: List[RoundInfo] = Field(default_factory=list)
    sessions_annotated: List[str] = Field(default_factory=list)
    sessions_validation: List[str] = Field(default_factory=list)
    total_annotated_frames: int = 0
    cvat_tasks: Dict[str, CvatTaskInfo] = Field(default_factory=dict)


def load_state(path: Optional[Path] = None) -> PipelineState:
    """Load pipeline state from JSON, returning empty state if file doesn't exist."""
    path = path or _DEFAULT_STATE_PATH
    if path.exists():
        raw = json.loads(path.read_text())
        return PipelineState(**raw)
    return PipelineState()


def save_state(state: PipelineState, path: Optional[Path] = None) -> None:
    """Atomically write pipeline state to JSON (write-then-rename)."""
    path = path or _DEFAULT_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = state.model_dump(mode="json")
    # Atomic write: tmp file in same directory, then rename
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_round(state: PipelineState, info: RoundInfo) -> None:
    """Append a completed training round to state."""
    state.rounds.append(info)
    state.current_round = info.round


def record_session(
    state: PipelineState,
    session_key: str,
    *,
    is_validation: bool = False,
    frame_count: int = 0,
) -> None:
    """Record an annotated session in state."""
    if is_validation:
        if session_key not in state.sessions_validation:
            state.sessions_validation.append(session_key)
    else:
        if session_key not in state.sessions_annotated:
            state.sessions_annotated.append(session_key)
    state.total_annotated_frames += frame_count
