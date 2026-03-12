from __future__ import annotations

from pathlib import Path

from supabase import Client
from supabase import create_client


class StorageClient:
    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)

    def upload(self, bucket: str, object_path: str, local_file: str | Path) -> None:
        local_path = Path(local_file)
        if not local_path.exists():
            raise FileNotFoundError(f"Upload source file not found: {local_path}")

        with local_path.open("rb") as f:
            self.client.storage.from_(bucket).upload(
                object_path,
                f,
                {"content-type": "video/mp4", "upsert": False},
            )
