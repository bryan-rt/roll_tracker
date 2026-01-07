from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic import ConfigDict
from pydantic import field_validator, model_validator


Numeric = Union[int, float]


class PathsConfig(BaseModel):
    """Path-related configuration.

    Only defines subdirectory names inside outputs/<clip_id>.
    """

    model_config = ConfigDict(extra="forbid")

    cache_dir_name: str = Field(default="_cache", description="Name for per-clip cache directory")
    debug_dir_name: Optional[str] = Field(default="_debug", description="Optional debug subdirectory name")
    use_relative_paths: bool = Field(default=True, description="Prefer relative paths in runtime")


class ComputeConfig(BaseModel):
    """Compute/runtime configuration: devices, workers and generic toggles."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field(default="cpu", description="Execution device: cpu/cuda")
    num_workers: int = Field(default=0, ge=0, description="Worker processes/threads count")
    batch_size: int = Field(default=1, ge=1, description="Generic batch size where applicable")
    use_sam: bool = Field(default=False, description="Enable SAM where supported")
    use_reid: bool = Field(default=False, description="Enable person ReID where supported")


class CameraConfig(BaseModel):
    """Camera-specific parameters and calibration."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    roi: Optional[List[Numeric]] = Field(default=None, description="Optional ROI specification")
    homography: Optional[List[List[Numeric]]] = Field(
        default=None,
        description="Optional 3x3 homography matrix (row-major)",
    )
    meters_per_pixel: Optional[float] = Field(default=None, gt=0, description="Metric scale for pixel distance")
    fps: Optional[float] = Field(default=None, gt=0, description="Frames per second for the clip")
    mat_width: Optional[int] = Field(default=None, gt=0, description="Optional mat width in pixels")
    mat_height: Optional[int] = Field(default=None, gt=0, description="Optional mat height in pixels")

    @field_validator("homography")
    @classmethod
    def _validate_homography(cls, v: Optional[List[List[Numeric]]]) -> Optional[List[List[float]]]:
        if v is None:
            return v
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("homography must be 3x3")
        out: List[List[float]] = []
        for row in v:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("homography must be 3x3")
            try:
                out.append([float(x) for x in row])
            except Exception:
                raise ValueError("homography must contain numeric values")
        return out


class StageAConfig(BaseModel):
    """Stage A thresholds/toggles."""

    model_config = ConfigDict(extra="forbid")

    start_dist_m: Optional[float] = Field(default=None, description="Hysteresis start distance (meters)")
    end_dist_m: Optional[float] = Field(default=None, description="Hysteresis end distance (meters)")
    frame_stride: Optional[int] = Field(default=None, ge=0, description="Frame stride; 0/None means no skipping")

    @model_validator(mode="after")
    def _validate_hysteresis(self) -> "StageAConfig":
        if self.start_dist_m is not None and self.end_dist_m is not None:
            if self.end_dist_m < 0:
                raise ValueError("end_dist_m must be >= 0")
            if not (self.start_dist_m > self.end_dist_m):
                raise ValueError("start_dist_m must be strictly greater than end_dist_m")
        return self


class StageBConfig(BaseModel):
    """Stage B configuration (masks and related toggles)."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)


class StageCConfig(BaseModel):
    """Stage C configuration (tags extraction)."""

    model_config = ConfigDict(extra="forbid")

    sample_frames_for_tags: Optional[int] = Field(default=None, description="0 means all frames")

    @field_validator("sample_frames_for_tags")
    @classmethod
    def _validate_sample_frames(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("sample_frames_for_tags must be >= 0")
        return v


class StageDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)


class StageEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)


class StageFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)


class PipelineConfig(BaseModel):
    """Root pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    camera: CameraConfig

    stage_A: Optional[StageAConfig] = None
    stage_B: Optional[StageBConfig] = None
    stage_C: Optional[StageCConfig] = None
    stage_D: Optional[StageDConfig] = None
    stage_E: Optional[StageEConfig] = None
    stage_F: Optional[StageFConfig] = None

    def as_dict(self) -> dict:
        """Return a plain dict suitable for audit logging."""
        return self.model_dump(exclude_none=True)
