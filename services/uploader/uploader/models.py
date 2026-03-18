from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ManifestRecord:
    export_id: str
    camera_id: str
    input_video_path: str
    local_output_path: str
    storage_bucket: str
    storage_object_path: str
    clip_row: dict[str, Any]
    log_events: list[dict[str, Any]]
    gym_id: str | None = None
    collision_hints: dict[str, Any] | None = None
