"""Manifest/uploader-contract helpers for Stage F clip exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class StorageTarget:
    bucket: str
    object_path: str
    file_name: str


def _safe_date_from_clip_id(clip_id: str) -> str:
    """
    Best-effort deterministic date extraction from clip ids like:
      cam03-20260103-124000_0-30s
    Returns YYYY-MM-DD, or 'unknown-date' if parsing fails.
    """
    txt = str(clip_id)
    parts = txt.split("-")
    if len(parts) >= 2:
        date_part = parts[1]
        if len(date_part) == 8 and date_part.isdigit():
            return f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"
    return "unknown-date"


def derive_video_slug(*, clip_id: str) -> str:
    return str(clip_id)


def derive_storage_target(
    *,
    gym_id: str,
    camera_id: str,
    clip_id: str,
    export_id: str,
    storage_bucket: str,
) -> StorageTarget:
    video_slug = derive_video_slug(clip_id=clip_id)
    recorded_date = _safe_date_from_clip_id(clip_id)
    file_name = f"{export_id}.mp4"
    object_path = (
        f"gym/{gym_id}/camera/{camera_id}/date/{recorded_date}/"
        f"video/{video_slug}/clips/{file_name}"
    )
    return StorageTarget(
        bucket=str(storage_bucket),
        object_path=object_path,
        file_name=file_name,
    )


def compute_clip_seconds(
    *,
    fps: float,
    export_start_frame: int,
    export_end_frame: int,
) -> Dict[str, float]:
    fps_f = float(fps)
    if fps_f <= 0.0:
        raise ValueError("fps must be positive to compute clip seconds")
    start_seconds = float(export_start_frame) / fps_f
    end_seconds = float(export_end_frame + 1) / fps_f
    duration_seconds = max(0.0, end_seconds - start_seconds)
    return {
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": duration_seconds,
    }


def get_file_stats(local_output_path: Path) -> Dict[str, Any]:
    stats = local_output_path.stat()
    return {
        "file_size_bytes": int(stats.st_size),
    }


def build_supabase_clip_contract(
    *,
    export_session: Any,
    clip_id: str,
    camera_id: str,
    local_output_path: Path,
    storage_target: StorageTarget,
    clip_type: str,
    initial_status: str,
    fighter_a_tag_id: str | None,
    fighter_b_tag_id: str | None,
    seconds_payload: Dict[str, float],
    pipeline_version: str,
    crop_mode: str,
    hash_sha256: str | None,
    file_size_bytes: int | None,
) -> Dict[str, Any]:
    return {
        "video_id": None,
        "match_id": str(export_session.source_match_ids[0]),
        "clip_type": str(clip_type),
        "file_path": str(storage_target.object_path),
        "storage_bucket": str(storage_target.bucket),
        "storage_object_path": str(storage_target.object_path),
        "start_seconds": float(seconds_payload["start_seconds"]),
        "end_seconds": float(seconds_payload["end_seconds"]),
        "duration_seconds": float(seconds_payload["duration_seconds"]),
        "camera_id": str(camera_id),
        "status": str(initial_status),
        "fighter_a_tag_id": fighter_a_tag_id,
        "fighter_b_tag_id": fighter_b_tag_id,
        "metadata": {
            "clip_id": str(clip_id),
            "export_id": str(export_session.export_id),
            "local_output_path": str(local_output_path),
            "source_match_ids": list(export_session.source_match_ids),
            "source_match_count": int(len(export_session.source_match_ids)),
            "resolved_pair_key": list(export_session.resolved_pair_key),
            "source_person_ids": list(export_session.source_person_ids),
            "match_start_frame": int(export_session.match_start_frame),
            "match_end_frame": int(export_session.match_end_frame),
            "export_start_frame": int(export_session.export_start_frame),
            "export_end_frame": int(export_session.export_end_frame),
            "pipeline_version": str(pipeline_version),
            "crop_mode": str(crop_mode),
            "hash_sha256": hash_sha256,
            "file_size_bytes": file_size_bytes,
        },
    }


def build_supabase_log_contracts(
    *,
    export_session: Any,
    clip_id: str,
    camera_id: str,
    storage_target: StorageTarget,
    clip_row_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "clip_id": None,
            "video_id": None,
            "event_type": "clip_exported_local",
            "event_level": "info",
            "message": "Clip exported locally and ready for upload.",
            "details": {
                "clip_id": str(clip_id),
                "camera_id": str(camera_id),
                "export_id": str(export_session.export_id),
                "source_match_ids": list(export_session.source_match_ids),
                "storage_bucket": str(storage_target.bucket),
                "storage_object_path": str(storage_target.object_path),
                "clip_row_preview": clip_row_payload,
            },
        }
    ]
