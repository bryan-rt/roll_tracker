from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    supabase_url: str
    service_role_key: str
    db_url: str
    storage_bucket: str
    scan_root: str
    poll_seconds: int
    delete_local: bool


def load_config() -> Config:
    return Config(
        supabase_url=os.environ["SUPABASE_URL"],
        service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        db_url=os.environ["SUPABASE_DB_URL"],
        storage_bucket=os.environ["SUPABASE_STORAGE_BUCKET"],
        scan_root=os.environ.get("UPLOADER_SCAN_ROOT", "outputs"),
        poll_seconds=int(os.environ.get("UPLOADER_POLL_SECONDS", "60")),
        delete_local=os.environ.get("UPLOADER_DELETE_LOCAL", "true").lower() == "true",
    )
