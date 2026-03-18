"""
F0 — Core Contracts & Artifact Schemas (authoritative)

Rules:
- JSONL artifacts MUST include ArtifactBase fields: schema_version, artifact_type, clip_id, camera_id,
  pipeline_version, created_at_ms.
- Parquet artifacts do not repeat base metadata per-row; instead it lives in clip_manifest.json.
- All IDs are strings; paths are relative to outputs/<clip_id>/ unless otherwise stated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ----------------------------
# Global constants / literals
# ----------------------------

# NOTE: Schema version must be bumped whenever any canonical artifact schema
# (including parquet column sets) changes.
# 0.3.0 bump: allow Stage A to optionally persist lightweight masks (YOLO-seg)
# and to persist per-frame geometry fields in parquet (u_px/v_px/x_m/y_m/...)
# while keeping backwards compatibility (columns are optional).
SCHEMA_VERSION_DEFAULT = "0.4.0"

StageLetter = Literal["A", "B", "C", "D", "E", "F"]
Severity = Literal["debug", "info", "warn", "error"]

RoiMethod = Literal["mask_roi", "bbox_roi", "full_frame"]

MaskFormat = Literal["npz_path", "png_path", "rle"]

ContactMethod = Literal["mask_lowest_point", "keypoints", "heuristic"]

MatchMethod = Literal["distance_hysteresis_v1"]

CropMode = Literal["dynamic_pair_crop", "fixed_roi"]

IdentityConstraint = Literal["must_link", "cannot_link"]


# ----------------------------
# Small reusable primitives
# ----------------------------

class BBoxXYXY(BaseModel):
    """Axis-aligned pixel bbox in xyxy format (float)."""
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def _x2_ge_x1(cls, v: float, info):
        x1 = info.data.get("x1")
        if x1 is not None and v < x1:
            raise ValueError("x2 must be >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_ge_y1(cls, v: float, info):
        y1 = info.data.get("y1")
        if y1 is not None and v < y1:
            raise ValueError("y2 must be >= y1")
        return v


class CornerPoints4(BaseModel):
    """Four 2D corners in pixel coords, ordered as provided by detector."""
    model_config = ConfigDict(extra="forbid")

    # list of 4 points, each point is [x, y]
    corners: List[Tuple[float, float]] = Field(..., min_length=4, max_length=4)


# ----------------------------
# JSONL base envelope
# ----------------------------

class ArtifactBase(BaseModel):
    """
    Required base fields on every JSONL record type.

    NOTE: Parquet rows do not carry these; they are captured in ClipManifest instead.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION_DEFAULT)
    artifact_type: str  # fixed literal in subclasses (still stored as str for JSONL ergonomics)
    clip_id: str
    camera_id: str
    pipeline_version: str  # git SHA or semver
    created_at_ms: int     # wall-clock timestamp when record was written (ms since epoch)


# ----------------------------
# Stage A — Detections + Tracklets
# ----------------------------

class DetectionRow(BaseModel):
    """
    Row schema for detections.parquet (dense).
    Stored in Parquet, not JSONL.
    """
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    camera_id: str
    frame_index: int
    timestamp_ms: int
    detection_id: str

    class_name: str
    confidence: float
    bbox_xyxy: BBoxXYXY

    # written by Stage A (tracklet generator) optionally
    tracklet_id: Optional[str] = None

    # optional metadata/debug
    source: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class TrackletFrameRow(BaseModel):
    """
    Row schema for tracklet_frames.parquet (dense).
    """
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    camera_id: str
    tracklet_id: str
    frame_index: int
    timestamp_ms: int
    detection_id: str

    local_track_conf: Optional[float] = None

    # --- Geometry (Stage A-owned)
    # These fields are populated in Stage A during the online loop so that
    # downstream stages can operate in mat-space without re-decoding video.
    # Stage B may optionally produce refined geometry, but Stage A remains the
    # canonical source for dense per-frame contact & projection signals.
    contact_u_px: Optional[float] = None
    contact_v_px: Optional[float] = None
    contact_x_m: Optional[float] = None
    contact_y_m: Optional[float] = None
    on_mat: Optional[bool] = None

    # Metadata about how the contact point was produced.
    contact_method: Optional[str] = None  # e.g., "bbox_bottom", "yolo_mask", "sam_mask"
    contact_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TrackletSummaryRow(BaseModel):
    """
    Row schema for tracklet_summaries.parquet (dense).
    """
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    camera_id: str
    tracklet_id: str
    start_frame: int
    end_frame: int
    n_frames: int

    mean_bbox_xyxy: Optional[BBoxXYXY] = None
    quality_score: Optional[float] = None
    reason_codes: Optional[List[str]] = None


# ----------------------------
# Stage B — Masks + Contact Points + Homography projection
# ----------------------------

class MaskRef(BaseModel):
    """
    Reference to a stored mask. Canonical: file-backed path (npz/png) relative to outputs/<clip_id>/.
    RLE is allowed as a portability fallback.
    """
    model_config = ConfigDict(extra="forbid")

    frame_index: int
    detection_id: str

    format: MaskFormat
    ref: str  # relative path OR RLE payload depending on format

    # optional debug/metadata
    roi_bbox_xyxy: Optional[BBoxXYXY] = None
    mask_iou_est: Optional[float] = None


class ContactPointRow(BaseModel):
    """
    Row schema for contact_points.parquet (dense).
    Contact point in image pixels and projected ground-plane meters.
    """
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    camera_id: str
    frame_index: int
    timestamp_ms: int
    detection_id: str

    u_px: float
    v_px: float

    x_m: float
    y_m: float

    method: ContactMethod
    confidence: float

    homography_id: Optional[str] = None


# ----------------------------
# Stage C — AprilTag observations + Identity hints
# ----------------------------

class TagObservation(ArtifactBase):
    """
    JSONL record: one AprilTag observation at a specific frame (ideally within mask ROI).
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["tag_observation"] = "tag_observation"

    frame_index: int
    timestamp_ms: int

    detection_id: str
    tracklet_id: Optional[str] = None

    tag_id: str
    tag_family: str
    confidence: float

    roi_method: RoiMethod

    # optional quality fields
    hamming: Optional[int] = None
    decision_margin: Optional[float] = None
    tag_corners_px: Optional[CornerPoints4] = None


class IdentityHint(ArtifactBase):
    """
    JSONL record: hard constraint hint keyed to tracklet_id, consumed by Stage D stitching.

    anchor_key is canonical string: f"tag:{tag_id}".
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["identity_hint"] = "identity_hint"

    tracklet_id: str
    anchor_key: str  # e.g., "tag:23"
    constraint: IdentityConstraint  # must_link or cannot_link
    confidence: float

    evidence: Dict[str, Any]  # votes, frames_seen, conflict notes, etc.

    @field_validator("anchor_key")
    @classmethod
    def _anchor_key_format(cls, v: str):
        if ":" not in v:
            raise ValueError("anchor_key must be namespaced like 'tag:<id>'")
        return v


# ----------------------------
# Stage D — Stitched person tracks + Identity assignments (post-pass)
# ----------------------------

class PersonTrackPointRow(BaseModel):
    """
    Row schema for person_tracks.parquet (dense).
    Must be traceable back to (tracklet_id, detection_id).
    """
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    camera_id: str
    person_id: str

    frame_index: int
    timestamp_ms: int

    detection_id: str
    tracklet_id: str

    bbox_xyxy: BBoxXYXY
    x_m: float
    y_m: float

    # optional
    mask_ref: Optional[str] = None  # relative path to mask file if stored
    reid_sim: Optional[float] = None
    stitch_edge_type: Optional[str] = None  # e.g., "same_tracklet", "mcf_link"


class IdentityAssignment(ArtifactBase):
    """
    JSONL record: final identity assignment keyed to stitched person_id.
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["identity_assignment"] = "identity_assignment"

    person_id: str
    tag_id: str
    assignment_confidence: float

    evidence: Dict[str, Any]  # tracklets involved, frame counts, vote stats, etc.
    conflicts: Optional[List[Dict[str, Any]]] = None


# ----------------------------
# Stage E — Match sessions
# ----------------------------

class MatchSession(ArtifactBase):
    """
    JSONL record: one detected match session between two stitched people.
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["match_session"] = "match_session"

    match_id: str
    person_id_a: str
    person_id_b: str

    start_frame: int
    end_frame: int

    start_ts_ms: int
    end_ts_ms: int

    method: MatchMethod
    confidence: float
    evidence: Dict[str, Any]

    notes: Optional[str] = None


# ----------------------------
# Stage F — Export manifest
# ----------------------------

class ExportManifest(ArtifactBase):
    """
    JSONL record: describes an exported clip artifact and how it was produced.
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["export_manifest"] = "export_manifest"

    gym_id: Optional[str] = None  # from ClipManifest; used by uploader for tag→profile resolution

    export_id: str
    match_id: str

    output_video_path: str  # relative path under outputs/<clip_id>/
    crop_mode: CropMode
    privacy: Dict[str, Any]  # redaction on/off, method, params
    inputs: Dict[str, Any]   # source clip path, person ids, frame span, etc.

    ffmpeg_cmd: Optional[str] = None
    hash_sha256: Optional[str] = None
    collision_hints: Optional[Dict[str, Any]] = None


# ----------------------------
# Audit events (all stages)
# ----------------------------

class AuditEvent(ArtifactBase):
    """
    JSONL record: one audit log event for determinism + debugging.
    """
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["audit_event"] = "audit_event"

    event_id: str
    stage: StageLetter
    severity: Severity
    timestamp_ms: int  # event time (can match frame time or wall time; decide per-stage)
    message: str
    context: Dict[str, Any]

    metrics: Optional[Dict[str, Any]] = None
    links: Optional[List[str]] = None  # relative paths to debug images/graphs/etc.


# ----------------------------
# JSONL helpers (contract-friendly)
# ----------------------------

def jsonl_serialize(record: ArtifactBase) -> str:
    """
    Deterministic JSON serialization for JSONL.
    - Uses model_dump with sorted keys for stable diffs.
    """
    import json
    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def jsonl_parse_line(line: str) -> Dict[str, Any]:
    import json
    return json.loads(line)
