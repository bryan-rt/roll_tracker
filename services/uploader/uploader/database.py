from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class Database:
    def __init__(self, url: str) -> None:
        self.conn = psycopg.connect(url)

    def close(self) -> None:
        self.conn.close()

    def find_video(self, source_path: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                select id
                from public.videos
                where source_path = %s
                limit 1
                """,
                (source_path,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def create_video(
        self,
        camera_id: str,
        source_path: str,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into public.videos (
                    camera_id,
                    source_path,
                    source_type,
                    recorded_at,
                    status,
                    metadata
                )
                values (%s, %s, %s, now(), %s, %s)
                returning id
                """,
                (
                    camera_id,
                    source_path,
                    "pipeline",
                    "ready",
                    Jsonb({"created_by": "services.uploader"}),
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to create video row.")
            self.conn.commit()
            return str(row[0])

    def find_clip(self, bucket: str, path: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                select id
                from public.clips
                where storage_bucket = %s
                  and storage_object_path = %s
                limit 1
                """,
                (bucket, path),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def insert_clip(self, clip_row: dict[str, Any]) -> str:
        keys = list(clip_row.keys())
        columns_sql = ", ".join(keys)
        placeholders_sql = ", ".join(["%s"] * len(keys))
        values = []
        for k in keys:
            value = clip_row[k]
            if k == "metadata" and isinstance(value, dict):
                value = Jsonb(value)
            values.append(value)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                insert into public.clips ({columns_sql})
                values ({placeholders_sql})
                returning id
                """,
                values,
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert clip row.")
            self.conn.commit()
            return str(row[0])

    def insert_log_event(self, event: dict[str, Any]) -> int:
        keys = list(event.keys())
        columns_sql = ", ".join(keys)
        placeholders_sql = ", ".join(["%s"] * len(keys))
        values = []
        for k in keys:
            value = event[k]
            if k == "details" and isinstance(value, dict):
                value = Jsonb(value)
            values.append(value)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                insert into public.log_events ({columns_sql})
                values ({placeholders_sql})
                returning id
                """,
                values,
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert log event.")
            self.conn.commit()
            return int(row[0])
