from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    supabase_url: str
    service_role_key: str
    postgres_url: str
    storage_bucket_default: str = "match-clips"


def load_config() -> Config:
    return Config(
        supabase_url=os.environ["SUPABASE_URL"],
        service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        postgres_url=os.environ["SUPABASE_DB_URL"],
    )
