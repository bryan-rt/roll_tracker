"""Processor service configuration via environment variables."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    SCAN_ROOT: Path = Path("data/raw/nest")
    OUTPUT_ROOT: Path = Path("outputs")
    POLL_INTERVAL_SECONDS: int = 30
    GYM_ID: Optional[str] = None
    MAX_CLIP_AGE_HOURS: int = 6  # clips older than this are ignored; 0 = no limit
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # Phase 1/2 parallelism
    MAX_WORKERS: int = 3  # one per camera, Phase 1 (A+C) only
    PARALLEL_DEVICE: str = "cpu"  # Phase 1 workers use CPU
    SEQUENTIAL_DEVICE: str = "auto"  # Phase 2 (D+E+F) uses MPS if available

    # Debug
    VISUALIZE: bool = False  # write debug videos (_debug/annotated.mp4, mat_view.mp4)
