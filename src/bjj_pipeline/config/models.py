from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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
    """Stage A configuration (Detection + Tracklets).

    Stage A owns:
      - YOLO person detection (bboxes + confidence)
      - Optional YOLO segmentation masks (file-backed, canonical)
      - MOT association to produce tracklets (BoT-SORT)
      - First-pass contact point + homography projection to (x_m, y_m)

    Stage A must NOT perform homography calibration/validation (handled by preflight).
    """

    model_config = ConfigDict(extra="forbid")

    # Accept a mode key for compatibility with orchestration configs.
    # Not used by the processor directly.
    mode: Optional[str] = Field(default=None, description="Execution mode (compat)")

    class DetectorConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        model_path: str = Field(default="models/yolov8n.pt", description="YOLO detection weights")
        seg_model_path: Optional[str] = Field(
            default=None,
            description="Optional YOLO segmentation weights (if use_seg is true and the file exists)",
        )
        use_seg: bool = Field(default=False, description="Attempt to use YOLO segmentation masks")
        conf: float = Field(default=0.25, ge=0.0, le=1.0, description="Detection confidence threshold")
        imgsz: Optional[int] = Field(default=None, gt=0, description="Optional inference image size")
        device: Optional[str] = Field(default=None, description="Optional detector device override")

    class MaskGateConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        det_conf_min: float = Field(default=0.25, ge=0.0, le=1.0)
        mask_quality_min: float = Field(default=0.4, ge=0.0, le=1.0)
        min_area_frac: float = Field(default=0.10, ge=0.0, le=10.0)
        max_area_frac: float = Field(default=1.10, gt=0.0, le=10.0)

        @model_validator(mode="after")
        def _validate_area_frac(self) -> "StageAConfig.MaskGateConfig":
            if self.min_area_frac >= self.max_area_frac:
                raise ValueError("min_area_frac must be < max_area_frac")
            return self

    class MasksConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        gate: "StageAConfig.MaskGateConfig" = Field(default_factory=lambda: StageAConfig.MaskGateConfig())

    class PhysicsConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        audit_only: bool = Field(default=True, description="If true, physics is logged but does not gate tracking")
        max_speed_mps: float = Field(default=8.0, gt=0.0, description="Speed threshold for physics warnings")

    class TrackerConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        mode: str = Field(default="botsort", description="Tracker backend mode")
        with_reid: bool = Field(default=False, description="Enable appearance/ReID matching where supported")
        use_mask_bbox: bool = Field(default=True, description="Use mask-tight bbox for tracker association")
        params: dict = Field(default_factory=dict, description="Backend-specific tracker parameters")
        physics: "StageAConfig.PhysicsConfig" = Field(default_factory=lambda: StageAConfig.PhysicsConfig())

        @field_validator("mode")
        @classmethod
        def _validate_mode(cls, v: str) -> str:
            if v != "botsort":
                raise ValueError("Only tracker.mode='botsort' is supported in this build")
            return v

    # Optional stride for POC perf (0/None means no skipping)
    frame_stride: Optional[int] = Field(default=None, ge=0, description="Frame stride; 0/None means no skipping")

    detector: DetectorConfig = Field(default_factory=lambda: StageAConfig.DetectorConfig())
    masks: MasksConfig = Field(default_factory=lambda: StageAConfig.MasksConfig())
    tracker: TrackerConfig = Field(default_factory=lambda: StageAConfig.TrackerConfig())


class StageBConfig(BaseModel):
    """Stage B configuration (masks and related toggles)."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)


class StageCConfig(BaseModel):
    """Stage C configuration (tags extraction)."""

    model_config = ConfigDict(extra="forbid")

    sample_frames_for_tags: Optional[int] = Field(default=None, description="0 means all frames")

    # Tag decoder settings (used by Stage C + multiplex AC validation/audit)
    tag_family: Optional[str] = Field(default=None, description="Expected AprilTag family (e.g. 36h11)")
    c0_scheduler: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cadence + gating configuration for Stage C C0 scheduler",
    )

    @field_validator("sample_frames_for_tags")
    @classmethod
    def _validate_sample_frames(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("sample_frames_for_tags must be >= 0")
        return v


class OcclusionRepairConfig(BaseModel):

    """D0 occlusion-span detection + interpolation repair configuration.

    Occlusion evidence is computed from bbox deltas (see Stage D0 implementation), and
    span onset is gated by a hard pixel threshold on dy2 (bottom movement).
    """

    model_config = ConfigDict(extra="forbid")

    enable_normalized: bool = Field(default=True)
    # thresholds are normalized by rolling median baseline height
    min_bottom_frac: float = Field(default=0.15, ge=0.0)
    min_height_frac: float = Field(default=0.10, ge=0.0)
    # hard pixel gate for dy2 (bbox bottom delta); suppresses normal jitter
    dy2_px_min: float = Field(default=3.0, ge=0.0)
    gate_onset_with_dy2: bool = Field(default=True, description="If true, require dy2 >= dy2_px_min to start an occlusion span")
    onset_window: int = Field(default=5, ge=1)
    onset_min_frames: int = Field(default=1, ge=1)
    recover_bottom_frac: float = Field(default=0.10, ge=0.0)
    recover_height_frac: float = Field(default=0.08, ge=0.0)
    recover_min_frames: int = Field(default=3, ge=1)
    merge_gap_frames: int = Field(default=2, ge=0)
    min_window_frames: int = Field(default=2, ge=1)
    # null => no hard cap; audit span lengths and tune later
    max_span_frames: Optional[int] = Field(default=None, ge=1, description="Optional cap for safety")


class GlobalContextConfig(BaseModel):

    """Evidence-only global context for downstream stages.

    These do not gate repair in D0; they are emitted as features.
    """

    model_config = ConfigDict(extra="forbid")

    # Evidence-only; does NOT gate repairs.
    # If null, context metrics are omitted.
    context_radius_m: Optional[float] = Field(default=None, gt=0)
    candidate_radius_m: Optional[float] = Field(default=None, gt=0)


class KinematicsConfig(BaseModel):

    """D0 Checkpoint 3 (CP3): flag-only kinematics derived from effective world coordinates.

    These thresholds are used only for emitting boolean flags; D0 does not clamp or suppress values.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    v_max_mps: float = Field(default=8.0, gt=0)
    a_max_mps2: float = Field(default=12.0, gt=0)


class StageD0Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occlusion_repair: OcclusionRepairConfig = Field(default_factory=OcclusionRepairConfig)
    global_context: GlobalContextConfig = Field(default_factory=GlobalContextConfig)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)


class StageDQAConfig(BaseModel):
    """Stage D visual QA configuration (Checkpoint 2.5)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="If true, write Stage D footpath PNG under _debug/")
    # Where to load the mat blueprint from (repo-relative).
    mat_blueprint_path: str = Field(default="configs/mat_blueprint.json")
    # Output filename inside outputs/<clip_id>/_debug/
    output_name: str = Field(default="stage_D_paths.png")
    # Canvas sizing
    canvas_size_px: int = Field(default=640, gt=0)
    margin_px: int = Field(default=24, ge=0)
    # Grouping: auto chooses person_tracks if present else tracklet bank frames.
    group_by: str = Field(default="auto", description="auto|person|tracklet")
    # If bank contains repaired columns, overlay repaired spans as dotted lines.
    prefer_repaired: bool = Field(default=True)

    @field_validator("group_by")
    @classmethod
    def _validate_group_by(cls, v: str) -> str:
        allowed = {"auto", "person", "tracklet"}
        if v not in allowed:
            raise ValueError(f"stage_D.qa.group_by must be one of {sorted(allowed)} (got {v!r})")
        return v


class StageD1Config(BaseModel):

    """Stage D1 configuration (graph build only; no costs, no solving)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    write_debug_graph_artifacts: bool = Field(
        default=False,
        description="If true, write dev-only D1 graph candidate artifacts under outputs/<clip_id>/_debug/",
    )

    enable_lifespan_segmentation: bool = Field(
        default=True,
        description="If true, segment full carrier lifespans into SOLO/GROUP segments (intra-lifespan merges).",
    )
    min_group_duration_frames: int = Field(default=10, ge=0)
    min_split_separation_frames: int = Field(default=10, ge=0)
    carrier_coord_window_frames: int = Field(default=8, ge=0)
    merge_trigger_max_age_frames: int = Field(default=60, ge=0)

    # maximum allowed gap (in frames) for single→single continuation edges
    max_continue_gap_frames: int = Field(default=90, ge=0)
    # endpoint extraction windows (frames)
    start_window_frames: int = Field(default=10, ge=0)
    end_window_frames: int = Field(default=10, ge=0)
    # group-tracklet inference
    enable_group_nodes: bool = Field(default=True)
    merge_dist_m: float = Field(default=0.45, gt=0)
    merge_end_sync_frames: int = Field(default=3, ge=0)
    merge_disappear_gap_frames: int = Field(default=6, ge=0)
    split_dist_m: float = Field(default=0.60, gt=0)
    split_search_horizon_frames: int = Field(default=120, ge=0)


class StageDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)
    run_until: str = Field(default="D0", description="Stage D dispatcher target: D0|D1|D2|D6")
    d0: Optional["StageD0Config"] = Field(default=None, description="Stage D0 cleanup configuration")
    qa: Optional["StageDQAConfig"] = Field(default=None, description="Stage D visual QA configuration")
    d1: Optional["StageD1Config"] = Field(default=None, description="Stage D1 graph build configuration")

    @field_validator("run_until")
    @classmethod
    def _validate_run_until(cls, v: str) -> str:
        allowed = {"D0", "D1", "D2", "D6"}
        if v not in allowed:
            raise ValueError(f"stage_D.run_until must be one of {sorted(allowed)} (got {v!r})")
        return v


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
