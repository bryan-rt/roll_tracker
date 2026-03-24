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
                    gym_id=rec.gym_id,
                )
            else:
                # Backfill gym_id on pre-existing video rows
                if rec.gym_id is not None:
                    db.update_video_gym_id(video_id, rec.gym_id)

            clip_row["video_id"] = video_id

            # Session-level: resolve source_video_ids from source_video_paths
            source_video_paths = clip_row.pop("source_video_paths", None)
            if source_video_paths and isinstance(source_video_paths, list):
                resolved_video_ids = []
                for src_path in source_video_paths:
                    vid_id = db.find_video(src_path)
                    if not vid_id:
                        vid_id = db.create_video(
                            camera_id=rec.camera_id,
                            source_path=src_path,
                            gym_id=rec.gym_id,
                        )
                    elif rec.gym_id is not None:
                        db.update_video_gym_id(vid_id, rec.gym_id)
                    resolved_video_ids.append(vid_id)
                clip_row["source_video_ids"] = resolved_video_ids
                # Set video_id to first source for backward compat with existing FK
                if resolved_video_ids and clip_row.get("video_id") is None:
                    clip_row["video_id"] = resolved_video_ids[0]
                print(
                    f"[uploader] resolved {len(resolved_video_ids)} source_video_ids "
                    f"for export_id={export_id}"
                )

            # Phase C: resolve tag IDs → profile IDs via active gym check-ins
            gym_id = rec.gym_id
            if gym_id is None:
                print(
                    f"[uploader] warning: no gym_id in manifest for export_id={export_id}, "
                    "skipping profile resolution"
                )

            # Collision detection
            tag_a = clip_row.get("fighter_a_tag_id")
            tag_b = clip_row.get("fighter_b_tag_id")
            collision = False
            collision_signal = None

            # Signal A: within-clip same tag
            if rec.collision_hints and rec.collision_hints.get("same_tag_collision"):
                collision = True
                collision_signal = "signal_a"

            # Signal B: multiple active check-ins for either tag at this gym
            if gym_id and not collision:
                if tag_a is not None and tag_a != "":
                    if db.count_active_checkins_for_tag(int(tag_a), gym_id) > 1:
                        collision = True
                        collision_signal = "signal_b"
                if tag_b is not None and tag_b != "" and not collision:
                    if db.count_active_checkins_for_tag(int(tag_b), gym_id) > 1:
                        collision = True
                        collision_signal = "signal_b"

            if collision:
                clip_row["status"] = "collision_flagged"
                clip_row["fighter_a_profile_id"] = None
                clip_row["fighter_b_profile_id"] = None
                print(
                    f"[uploader] collision_flagged: tag_a={tag_a} tag_b={tag_b} "
                    f"gym_id={gym_id} export_id={export_id} signal={collision_signal}"
                )
            else:
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
