from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .database import Database
from .manifest import load_manifest
from .storage import StorageClient


def _make_log_event_payload(
    event: dict[str, Any],
    *,
    video_id: str | None,
    clip_id: str | None,
) -> dict[str, Any]:
    payload = dict(event)
    payload["video_id"] = video_id
    payload["clip_id"] = clip_id
    return payload


def run_upload(manifest_path: str, config: Config) -> None:
    records = load_manifest(manifest_path)
    if not records:
        print("No manifest rows found.")
        return

    storage = StorageClient(config.supabase_url, config.service_role_key)
    db = Database(config.postgres_url)

    try:
        for rec in records:
            print(f"[uploader] processing export_id={rec.export_id}")

            local_path = Path(rec.local_output_path)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local export file missing for export_id={rec.export_id}: {local_path}"
                )

            video_id = db.find_video(rec.input_video_path)
            if video_id is None:
                print(
                    f"[uploader] no video row for source_path={rec.input_video_path}; creating one"
                )
                video_id = db.create_video(
                    camera_id=rec.camera_id,
                    source_path=rec.input_video_path,
                )

            existing_clip_id = db.find_clip(
                rec.storage_bucket,
                rec.storage_object_path,
            )

            if existing_clip_id is None:
                print(
                    "[uploader] uploading "
                    f"{local_path} -> {rec.storage_bucket}/{rec.storage_object_path}"
                )
                storage.upload(
                    rec.storage_bucket,
                    rec.storage_object_path,
                    local_path,
                )

                clip_row = dict(rec.clip_row)
                clip_row["video_id"] = video_id

                clip_id = db.insert_clip(clip_row)
                print(
                    f"[uploader] inserted clip row clip_id={clip_id} export_id={rec.export_id}"
                )
            else:
                clip_id = existing_clip_id
                print(
                    f"[uploader] clip already exists clip_id={clip_id}; skipping clip insert"
                )

            for event in rec.log_events:
                payload = _make_log_event_payload(
                    event,
                    video_id=video_id,
                    clip_id=clip_id,
                )
                event_id = db.insert_log_event(payload)
                print(
                    f"[uploader] inserted log_event id={event_id} for export_id={rec.export_id}"
                )
    finally:
        db.close()
