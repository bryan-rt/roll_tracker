from __future__ import annotations

import os 

from .database import Database
from .storage import StorageClient
from .manifest import load_manifest

def run_upload(manifest_path: str, cfg) -> None:
    db = Database(cfg.db_url)
    storage = StorageClient(cfg.supabase_url, cfg.service_role_key)

    try:
        for rec in load_manifest(manifest_path):
            export_id = rec.export_id
            print(f"[uploader] processing export_id={export_id}")

            local_path = rec.local_output_path
            clip_row = dict(rec.clip_row)
            bucket = rec.storage_bucket
            object_path = rec.storage_object_path

            existing = db.find_clip(bucket, object_path)
            if existing:
                print(f"[uploader] clip already exists, skipping {export_id}")
                continue

            video_id = db.find_video(rec.input_video_path)
            if not video_id:
                print(
                    f"[uploader] no video row for source_path={rec.input_video_path}; creating one"
                )
                video_id = db.create_video(
                    camera_id=rec.camera_id,
                    source_path=rec.input_video_path,
                )

            clip_row["video_id"] = video_id

            # Phase C: resolve tag IDs → profile IDs via active gym check-ins
            gym_id = db.get_video_gym_id(video_id)
            if gym_id is None:
                print(
                    f"[uploader] warning: no gym_id for video_id={video_id}, "
                    "skipping profile resolution"
                )
            for side in ("a", "b"):
                tag_key = f"fighter_{side}_tag_id"
                profile_key = f"fighter_{side}_profile_id"
                tag_val = clip_row.get(tag_key)
                if tag_val is not None and tag_val != "" and gym_id is not None:
                    profile_id = db.resolve_profile_by_tag_and_gym(
                        int(tag_val), gym_id
                    )
                    clip_row[profile_key] = profile_id
                    print(
                        f"[uploader] {tag_key}={tag_val} -> {profile_key}={profile_id}"
                    )
                else:
                    clip_row[profile_key] = None

            print(f"[uploader] uploading {local_path} -> {bucket}/{object_path}")
            storage.upload(
                bucket,
                object_path,
                local_path,
            )

            clip_id = db.insert_clip(clip_row)
            print(f"[uploader] inserted clip row clip_id={clip_id} export_id={export_id}")

            for event in rec.log_events:
                payload = dict(event)
                payload["clip_id"] = clip_id
                payload["video_id"] = video_id
                log_id = db.insert_log_event(payload)
                print(
                    f"[uploader] inserted log_event id={log_id} for export_id={export_id}"
                )

            if cfg.delete_local and os.path.exists(local_path):
                os.remove(local_path)
                print(f"[uploader] removed local clip {local_path}")
    finally:
        db.close()
