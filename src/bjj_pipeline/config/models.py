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
    # Hard gate: new tracklets born at the image border cannot induce split triggers.
    split_border_gate_enabled: bool = Field(default=True)
    split_border_margin_px: int = Field(default=40, ge=0)
    # Entrance-like suppression (refines naive border-born check).
    # A tracklet is treated as an "entrance" only if it starts near the true image border
    # and moves inward over its first K frames.
    split_entrance_k_frames: int = Field(default=10, ge=0)
    split_entrance_min_samples: int = Field(default=3, ge=0)
    split_entrance_min_inward_px: int = Field(default=20, ge=0)
    split_entrance_allow_observed_fallback: bool = Field(default=True)
    suppress_start_merged_if_entrance_like: bool = Field(default=True)
    # Optional occlusion reconnect proposals (between non-overlapping lifespans)
    reconnect_enabled: bool = Field(
        default=False,
        description=(
            "If true, propose CONT edges between base tracklets whose lifespans do not overlap "
            "but are temporally and kinematically plausible reconnects."
        ),
    )

    reconnect_max_gap_frames: int = Field(
        default=120,
        ge=0,
        description="Maximum allowed gap (in frames) for occlusion reconnect edge proposals (separate from split_search_horizon_frames).",
    )
    reconnect_boundary_on_mat_required: bool = Field(
        default=True,
        description=(
            "If true, require each reconnect endpoint to have at least one on-mat frame within a small boundary window "
            "around the tracklet end/start. Missing on_mat is treated as off-mat."
        ),
    )
    reconnect_boundary_slack_frames: int = Field(
        default=2,
        ge=0,
        description="Boundary window slack (frames) for reconnect endpoint on-mat requirement.",
    )
    reconnect_solo_only: bool = Field(
        default=True,
        description="If true, only propose reconnect edges from a SOLO end-segment to a SOLO start-segment (never across GROUP boundaries).",
    )
    # Optional promotion of reconnect destinations from SOLO to GROUP when evidence suggests
    # a 2-person re-entry is continuing through occlusion.
    promote_group_reconnect_enabled: bool = Field(
        default=False,
        description=(
            "If true, allow GROUP->SOLO reconnects to promote the SOLO destination segment to a GROUP node "
            "when there is no nearby second SOLO birth (continuing group-capacity through occlusion)."
        ),
    )
    promote_group_reconnect_nearby_start_window_frames: int = Field(
        default=30,
        ge=0,
        description="Time window (frames) for counting nearby SOLO births around a reconnect destination start.",
    )
    promote_group_reconnect_nearby_dist_m: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Max spatial distance (meters) to consider another SOLO birth 'nearby' for promotion; "
            "defaults to merge_dist_m when unset."
        ),
    )


class StageD2CostsConfig(BaseModel):
    """Stage D2 configuration (costs + constraints; solver-agnostic).

    Note: v_cost_scale_mps and v_hinge_mps default to Stage D0 kinematics v_max_mps
    when unset, to avoid redundant speed knobs.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)

    # hard policies (locked for POC)
    missing_geom_policy: str = Field(default="disallow", description="Only 'disallow' is supported for POC")
    dt_max_s: float = Field(default=1.0, gt=0)

    # Endpoint lookup window for CONTINUE edges: if the exact boundary frame is missing in
    # tracklet_bank_frames (e.g., at SOLO/GROUP boundaries), D2 may search within +/- this
    # many frames for the nearest available row for the carrier tracklet.
    # None => resolved by the D2 runner (typically from stage_D.d1.carrier_coord_window_frames).
    endpoint_search_window_frames: Optional[int] = Field(default=None, ge=0)

    # motion normalization (non-redundant; derived by default)
    v_cost_scale_mps: Optional[float] = Field(default=None, gt=0)
    v_hinge_mps: Optional[float] = Field(default=None, gt=0)

    w_time: float = Field(default=0.1, ge=0)
    w_vreq: float = Field(default=1.0, ge=0)
    base_env_cost: float = Field(default=0.01, ge=0)

    # soft flags (from D0 CP3)
    use_flags: bool = Field(default=True)
    w_flags: float = Field(default=0.25, ge=0)

    # contact reliability weighting
    use_contact_rel: bool = Field(default=True)
    contact_conf_floor: float = Field(default=0.25, ge=0, le=1.0)
    contact_rel_alpha: float = Field(default=0.35, ge=0)

    # merge/split coherence
    # merge/split coherence (positive-only system; preferred)
    coherent_merge_cost: float = Field(default=0.05, ge=0)
    incoherent_merge_cost: float = Field(default=1.5, ge=0)
    # If unset, split defaults mirror merge.
    coherent_split_cost: Optional[float] = Field(default=None, ge=0)
    incoherent_split_cost: Optional[float] = Field(default=None, ge=0)

    # Deprecated (backward compatibility): older configs used a negative reward for coherent structure.
    # If coherent_* keys are present, these are ignored by D2 costs logic.
    bonus_group_coherent: float = Field(default=0.5, ge=0)
    penalty_group_incoherent: float = Field(default=0.5, ge=0)

    # edge priors (D5 will later refine birth/death policies)
    birth_cost: float = Field(default=2.0, ge=0)
    death_cost: float = Field(default=2.0, ge=0)
    merge_prior: float = Field(default=0.1, ge=0)
    split_prior: float = Field(default=0.1, ge=0)
    # Reconnect edge shaping (tn -> tm after occlusion); applied only to edges
    # explicitly marked as reconnects in D1 payloads.
    reconnect_extra_env_cost: float = Field(
        default=0.0,
        ge=0,
        description="Additional environment cost added only to reconnect edges.",
    )
    reconnect_w_z: float = Field(
        default=1.0,
        ge=0,
        description="Weight for elliptical distance term on reconnect edges.",
    )
    reconnect_w_time: float = Field(
        default=0.35,
        ge=0,
        description="Optional additional time penalty weight for reconnect edges (seconds).",
    )

    # Soft Option B: when a reconnect edge is shadowed by a coherent MERGE/SPLIT chain
    # (annotated in D1 payload), choose how D2 handles it.
    shadowed_reconnect_policy: str = Field(
        default="disallow",
        description="Policy for reconnect edges shadowed by a coherent group chain: disallow|penalize|allow",
    )
    shadowed_reconnect_penalty: float = Field(
        default=10.0,
        ge=0,
        description="Additive penalty applied to shadowed reconnect edges when policy=penalize.",
    )

    # New reconnect shaping (teleportation guardrails). These keys are additive and do not
    # remove the legacy reconnect_w_z / reconnect_w_time knobs for back-compat.
    reconnect_v_max_mps: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum allowed required speed (m/s) for reconnect edges. If unset, defaults to stage_D.d0.kinematics.v_max_mps "
            "via D2 runner resolution."
        ),
    )
    reconnect_w_speed: float = Field(
        default=1.0,
        ge=0,
        description="Weight for reconnect speed hinge term (applies to reconnect edges only).",
    )
    reconnect_speed_power: float = Field(
        default=2.0,
        gt=0,
        description="Exponent for reconnect speed hinge (>=1 recommended).",
    )
    reconnect_dt_ref_s: float = Field(
        default=0.75,
        gt=0,
        description="Reference dt (seconds) for reconnect convex time penalty normalization.",
    )
    reconnect_dt_power: float = Field(
        default=2.2,
        gt=0,
        description="Exponent for reconnect convex time penalty (must be >1 for accelerating penalty).",
    )

    @field_validator("missing_geom_policy")
    @classmethod
    def _validate_missing_geom_policy(cls, v: str) -> str:
        allowed = {"disallow"}
        if v not in allowed:
            raise ValueError(f"stage_D.d2_costs.missing_geom_policy must be one of {sorted(allowed)} (got {v!r})")
        return v


class StageD3Config(BaseModel):
    """Stage D3 (solver) configuration."""

    model_config = ConfigDict(extra="forbid")

    # "Explain each tracklet OR pay a penalty":
    # We define a boolean per base_tracklet_id present in D1 SINGLE_TRACKLET nodes.
    # If the solution uses zero flow through all nodes belonging to that base_tracklet_id,
    # we pay this penalty (in cost units) once for that base_tracklet_id.
    unexplained_tracklet_penalty: float = Field(default=5.0, ge=0.0)

    # Tag fragmentation (time-separated): penalty for starting a new fragment of the same
    # AprilTag across disconnected time windows. Larger values prefer continuity when possible
    # but still allow multiple disjoint fragments to avoid infeasibility.
    tag_fragment_start_penalty: float = Field(default=2500.0, ge=0.0)

    # POC_2_TAGS (Option B): penalty for leaving a ping-bound GROUP_TRACKLET unexplained.
    # When >0, forced GROUP pings are not hard-must-use; instead we add this penalty
    # for each required group ping that is not carried by any used GROUP node.
    unexplained_group_ping_penalty: float = Field(default=5000.0, ge=0.0)

    # Penalty reference: compute a per-clip reference edge cost (default p95 of allowed edge total_cost)
    # and scale soft penalties as multipliers of that reference. This keeps behavior stable across clips.
    penalty_ref_edge_cost_quantile: float = Field(default=0.95, ge=0.0, le=1.0)
    penalty_ref_edge_cost_min: float = Field(default=0.01, ge=0.0)

    # Ping miss penalties (soft): prefer using ping-bound nodes, but allow missing when required to avoid infeasibility.
    # These are expressed as multipliers of the reference edge cost (unless the *_abs override is provided).
    solo_ping_miss_penalty_mult: float = Field(default=50.0, ge=0.0)
    group_ping_miss_penalty_mult: float = Field(default=60.0, ge=0.0)
    # Optional absolute overrides (cost units). When set, they take precedence over multipliers.
    solo_ping_miss_penalty_abs: float | None = Field(default=None, ge=0.0)
    group_ping_miss_penalty_abs: float | None = Field(default=None, ge=0.0)

    # Fragment start penalty scaling: prefer continuity of the same tag across time.
    # When tag_fragment_start_penalty_abs is set it takes precedence; otherwise use
    # tag_fragment_start_penalty_mult * ref_edge_cost.
    tag_fragment_start_penalty_mult: float = Field(default=20.0, ge=0.0)
    tag_fragment_start_penalty_abs: float | None = Field(default=None, ge=0.0)

    # Frames near the start/end of the clip considered "boundary" for group gating logic
    # in D3. This is read in stages/stitch/solver.py and consumed by d3_ilp.py.
    group_boundary_window_frames: int = Field(default=10, ge=0)


class StageDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = Field(default=None)
    run_until: str = Field(default="D0", description="Stage D dispatcher target: D0|D1|D2|D3|D6")
    d3_checkpoint: str = Field(
        default="POC_0",
        description="Stage D3 internal checkpoint selector (POC_0..POC_4). Used only when run_until == D3.",
    )
    d0: Optional["StageD0Config"] = Field(default=None, description="Stage D0 cleanup configuration")
    qa: Optional["StageDQAConfig"] = Field(default=None, description="Stage D visual QA configuration")
    d1: Optional["StageD1Config"] = Field(default=None, description="Stage D1 graph build configuration")
    d2_costs: Optional["StageD2CostsConfig"] = Field(default=None, description="Stage D2 costs + constraints configuration")
    d3: "StageD3Config" = Field(default_factory=StageD3Config, description="Stage D3 solver configuration")

    @field_validator("run_until")
    @classmethod
    def _validate_run_until(cls, v: str) -> str:
        allowed = {"D0", "D1", "D2", "D3", "D6"}
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
