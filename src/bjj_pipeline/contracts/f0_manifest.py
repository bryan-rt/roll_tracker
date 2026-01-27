
# Place this after all class and helper function definitions

def register_stage_D1_defaults(manifest: 'ClipManifest', layout: 'ClipOutputLayout') -> None:
    """
    Convenience: register canonical Stage D1 artifacts (graph tables).

    These artifacts are solver-agnostic inputs for D2 (cost modeling) and D3 (MCF/ILP solving).
    Call after writing the files.
    """
    manifest.register_artifact(
        stage="D",
        key="d1_graph_nodes_parquet",
        relpath=layout.rel_to_clip_root(layout.d1_graph_nodes_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="D",
        key="d1_graph_edges_parquet",
        relpath=layout.rel_to_clip_root(layout.d1_graph_edges_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="D",
        key="d1_segments_parquet",
        relpath=layout.rel_to_clip_root(layout.d1_segments_parquet()),
        content_type="application/parquet",
    )

# ...existing code...

# Place this after all class and helper function definitions
"""
F0 — Clip manifest model + helpers (authoritative)

A clip manifest is the single source of truth linking:
- input video metadata
- produced artifacts (paths under outputs/<clip_id>/...)
- optional checksums for determinism and caching

Stages should register artifacts as they are produced.
"""



from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .f0_models import SCHEMA_VERSION_DEFAULT
from .f0_paths import ClipOutputLayout, StageLetter


class ArtifactRef(BaseModel):
    """
    Reference to a produced artifact on disk.
    path is RELATIVE to outputs/<clip_id>/.
    """
    model_config = ConfigDict(extra="forbid")

    path: str
    content_type: Optional[str] = None  # e.g., "application/parquet", "application/jsonl"
    sha256: Optional[str] = None


class StageArtifacts(BaseModel):
    """
    Per-stage artifact registry. Keys are stable and must match F0 conventions.
    """
    model_config = ConfigDict(extra="forbid")

    # A dict allows stage-specific additions while keeping stable top-level structure.
    items: Dict[str, ArtifactRef] = Field(default_factory=dict)


class ClipManifest(BaseModel):
    """
    Canonical per-clip manifest.

    Stored at: outputs/<clip_id>/clip_manifest.json
    All registered artifact paths are relative to outputs/<clip_id>/.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION_DEFAULT)

    clip_id: str
    camera_id: str

    input_video_path: str  # repo-relative or absolute; stages should not assume absolute
    fps: float
    frame_count: int
    duration_ms: int

    pipeline_version: str
    created_at_ms: int

    # Registry by stage letter
    stages: Dict[StageLetter, StageArtifacts] = Field(default_factory=dict)

    # Orchestration-level or non-stage artifacts (optional)
    misc_artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)

    # Optional: arbitrary provenance (e.g. git commit, docker image, camera config hash)
    provenance: Optional[Dict[str, Any]] = None

    def register_artifact(
        self,
        *,
        stage: StageLetter,
        key: str,
        relpath: str,
        content_type: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> None:
        if stage not in self.stages:
            self.stages[stage] = StageArtifacts()
        self.stages[stage].items[key] = ArtifactRef(path=relpath, content_type=content_type, sha256=sha256)

    def register_misc_artifact(
        self,
        *,
        key: str,
        relpath: str,
        content_type: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> None:
        """Register a non-stage artifact (e.g., orchestration audit)."""
        self.misc_artifacts[key] = ArtifactRef(path=relpath, content_type=content_type, sha256=sha256)

    def get_misc_artifact_path(self, *, key: str) -> str:
        if key not in self.misc_artifacts:
            raise KeyError(f"Misc artifact not registered: key={key}")
        return self.misc_artifacts[key].path

    def get_artifact_path(self, *, stage: StageLetter, key: str) -> str:
        if stage not in self.stages or key not in self.stages[stage].items:
            raise KeyError(f"Artifact not registered: stage={stage} key={key}")
        return self.stages[stage].items[key].path


# ----------------------------
# IO helpers
# ----------------------------

def write_manifest(manifest: ClipManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2, by_alias=False), encoding="utf-8")


def load_manifest(path: Path) -> ClipManifest:
    return ClipManifest.model_validate_json(path.read_text(encoding="utf-8"))


def init_manifest(
    *,
    clip_id: str,
    camera_id: str,
    input_video_path: str,
    fps: float,
    frame_count: int,
    duration_ms: int,
    pipeline_version: str,
    created_at_ms: int,
    provenance: Optional[Dict[str, Any]] = None,
) -> ClipManifest:
    return ClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        input_video_path=input_video_path,
        fps=fps,
        frame_count=frame_count,
        duration_ms=duration_ms,
        pipeline_version=pipeline_version,
        created_at_ms=created_at_ms,
        provenance=provenance,
    )


def register_stage_A_defaults(manifest: ClipManifest, layout: ClipOutputLayout) -> None:
    """
    Convenience: register canonical stage A artifacts, if present.
    Call after writing the files.
    """
    manifest.register_artifact(
        stage="A",
        key="detections_parquet",
        relpath=layout.rel_to_clip_root(layout.detections_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="A",
        key="tracklet_frames_parquet",
        relpath=layout.rel_to_clip_root(layout.tracklet_frames_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="A",
        key="tracklet_summaries_parquet",
        relpath=layout.rel_to_clip_root(layout.tracklet_summaries_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="A",
        key="contact_points_parquet",
        relpath=layout.rel_to_clip_root(layout.stage_A_contact_points_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="A",
        key="audit_jsonl",
        relpath=layout.rel_to_clip_root(layout.audit_jsonl("A")),
        content_type="application/jsonl",
    )


def register_stage_B_defaults(manifest: ClipManifest, layout: ClipOutputLayout) -> None:
    """
    Convenience: register canonical stage B artifacts, if present.
    Call after writing the files.
    """
    manifest.register_artifact(
        stage="B",
        key="contact_points_parquet",
        relpath=layout.rel_to_clip_root(layout.contact_points_parquet()),
        content_type="application/parquet",
    )
    # Back-compat: if the legacy path exists, register it under a separate key.
    legacy = layout.stage_B_contact_points_parquet_legacy()
    if legacy.exists():
        manifest.register_artifact(
            stage="B",
            key="contact_points_parquet_legacy",
            relpath=layout.rel_to_clip_root(legacy),
            content_type="application/parquet",
        )
    manifest.register_artifact(
        stage="B",
        key="masks_dir",
        relpath=layout.rel_to_clip_root(layout.masks_dir()),
        content_type="inode/directory",
    )
    manifest.register_artifact(
        stage="B",
        key="audit_jsonl",
        relpath=layout.rel_to_clip_root(layout.audit_jsonl("B")),
        content_type="application/jsonl",
    )


def register_stage_D0_defaults(manifest: ClipManifest, layout: ClipOutputLayout) -> None:
    """
    Convenience: register canonical Stage D0 artifacts (bank tables + audit).
    Call after writing the files.
    """
    manifest.register_artifact(
        stage="D",
        key="tracklet_bank_frames_parquet",
        relpath=layout.rel_to_clip_root(layout.tracklet_bank_frames_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="D",
        key="tracklet_bank_summaries_parquet",
        relpath=layout.rel_to_clip_root(layout.tracklet_bank_summaries_parquet()),
        content_type="application/parquet",
    )
    manifest.register_artifact(
        stage="D",
        key="audit_jsonl",
        relpath=layout.rel_to_clip_root(layout.audit_jsonl("D")),
        content_type="application/jsonl",
    )
