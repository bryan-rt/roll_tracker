"""Training pipeline configuration — Pydantic v2 model with YAML persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict

_DEFAULT_CONFIG_PATH = Path("src/training_pipeline/pipeline_config.yaml")


class TrainingPipelineConfig(BaseModel):
    """All settings for the training pipeline with sensible defaults."""

    model_config = ConfigDict(extra="forbid")

    # CVAT connection
    cvat_url: str = "https://app.cvat.ai"
    cvat_username: str = ""
    cvat_password: str = ""
    cvat_project_name: str = "BJJ Training Data"

    # Paths
    training_data_dir: Path = Path("data/training_data")
    background_models_dir: Path = Path("data/background_models")
    cvat_tasks_dir: Path = Path("data/cvat_tasks")
    training_runs_dir: Path = Path("models/training_runs")

    # Processing
    frame_sample_rate: int = 3  # every Nth frame (3 = 10fps from 30fps)
    bg_subtraction_threshold: int = 30
    bg_min_contour_area: int = 500
    cross_camera_iou_threshold: float = 0.3

    # Training defaults (overridable per round)
    base_model: str = "models/yolo26n-pose.pt"
    imgsz: int = 640
    device: str = "mps"
    batch_size: int = 8


def load_config(path: Optional[Path] = None) -> TrainingPipelineConfig:
    """Load config from YAML, creating defaults if the file doesn't exist."""
    path = path or _DEFAULT_CONFIG_PATH
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        return TrainingPipelineConfig(**raw)
    # First run — write defaults
    cfg = TrainingPipelineConfig()
    save_config(cfg, path)
    return cfg


def save_config(cfg: TrainingPipelineConfig, path: Optional[Path] = None) -> None:
    """Persist config to YAML."""
    path = path or _DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # mode="json" converts Path objects to strings automatically
    data = cfg.model_dump(mode="json")
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
