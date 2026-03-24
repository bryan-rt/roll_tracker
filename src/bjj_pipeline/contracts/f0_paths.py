"""
F0 — Output layout + path builders (authoritative)

All stage outputs live under:
  outputs/<clip_id>/stage_<X>/...

All artifact references stored in manifests / JSONL should be RELATIVE paths
(from outputs/<clip_id>/).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

StageLetter = Literal["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class ClipOutputLayout:
    def tracklet_bank_summaries_parquet(self) -> Path:
        """Canonical Stage D tracklet bank summaries artifact path."""
        return self.stage_dir("D") / "tracklet_bank_summaries.parquet"
    """
    Canonical layout for a single clip's outputs.

    root points to the *repo* outputs directory by default: Path("outputs").
    clip_root = root / clip_id
    """
    clip_id: str
    root: Path = Path("outputs")

    @property
    def clip_root(self) -> Path:
        return self.root / self.clip_id

    def stage_dir(self, stage: StageLetter) -> Path:
        return self.clip_root / f"stage_{stage}"

    # ---- Stage A ----
    def detections_parquet(self) -> Path:
        return self.stage_dir("A") / "detections.parquet"

    def tracklet_frames_parquet(self) -> Path:
        return self.stage_dir("A") / "tracklet_frames.parquet"

    def tracklet_summaries_parquet(self) -> Path:
        return self.stage_dir("A") / "tracklet_summaries.parquet"

    def stage_A_contact_points_parquet(self) -> Path:
        """Canonical Stage A baseline contact points (derived from tracklet_frames).

        Required for Stage A completion (F0G).
        """
        return self.stage_dir("A") / "contact_points.parquet"

    def stage_A_masks_dir(self) -> Path:
        """Directory for Stage A lightweight masks (e.g., YOLO-seg).

        These masks are optional. When written, each detection row may reference
        a mask blob via detections.mask_ref (clip-relative path).
        """
        return self.stage_dir("A") / "masks"

    def stage_A_mask_npz_path(self, frame_index: int, detection_id: str) -> Path:
        # canonical file name
        return self.stage_A_masks_dir() / f"frame_{frame_index:06d}_det_{detection_id}.npz"

    def audit_jsonl(self, stage: StageLetter) -> Path:
        return self.stage_dir(stage) / "audit.jsonl"

    # ---- Stage B ----
    def stage_B_contact_points_refined_parquet(self) -> Path:
        """Canonical Stage B refinement contact points (subset overrides)."""
        return self.stage_dir("B") / "contact_points_refined.parquet"

    def stage_B_contact_points_parquet_legacy(self) -> Path:
        """Legacy Stage B contact points path (pre-refined rename)."""
        return self.stage_dir("B") / "contact_points.parquet"

    def contact_points_parquet(self) -> Path:
        """Stage B contact points artifact path.

        Canonical as of Jan 2026: stage_B/contact_points_refined.parquet.
        """
        return self.stage_B_contact_points_refined_parquet()

    def masks_dir(self) -> Path:
        return self.stage_dir("B") / "masks"

    def masks_png_dir(self) -> Path:
        return self.stage_dir("B") / "masks_png"

    def mask_npz_path(self, frame_index: int, detection_id: str) -> Path:
        # canonical file name
        return self.masks_dir() / f"frame_{frame_index:06d}_det_{detection_id}.npz"

    def mask_png_path(self, frame_index: int, detection_id: str) -> Path:
        return self.masks_png_dir() / f"frame_{frame_index:06d}_det_{detection_id}.png"

    # ---- Stage C ----
    def tag_observations_jsonl(self) -> Path:
        return self.stage_dir("C") / "tag_observations.jsonl"

    def identity_hints_jsonl(self) -> Path:
        return self.stage_dir("C") / "identity_hints.jsonl"

    # ---- Stage D ----
    def person_tracks_parquet(self) -> Path:
        return self.stage_dir("D") / "person_tracks.parquet"

    def person_spans_parquet(self) -> Path:
        """Optional Stage D4 helper artifact: per-person per-node span table.

        Not part of the strict Stage D completion contract today, but useful for
        Stage E (match detection) to understand effective-capacity spans.
        """
        return self.stage_dir("D") / "person_spans.parquet"

    def identity_assignments_jsonl(self) -> Path:
        # Produced after stitching (post-pass) but stored under stage_D outputs canonically
        return self.stage_dir("D") / "identity_assignments.jsonl"

    def tracklet_bank_frames_parquet(self) -> Path:
        return self.stage_dir("D") / "tracklet_bank_frames.parquet"


    # ---- Stage D1 (graph construction) ----
    # Canonical (solver-agnostic) graph artifacts for downstream D2 (costing) and D3 (solving).
    # We keep a "d1_" prefix for clarity/provenance.
    def d1_graph_nodes_parquet(self) -> Path:
        return self.stage_dir("D") / "d1_graph_nodes.parquet"

    def d1_graph_edges_parquet(self) -> Path:
        return self.stage_dir("D") / "d1_graph_edges.parquet"

    def d1_segments_parquet(self) -> Path:
        return self.stage_dir("D") / "d1_segments.parquet"

    # ---- Stage D2 (costs + constraints; solver-agnostic) ----
    def d2_edge_costs_parquet(self) -> Path:
        """Canonical Stage D2 per-edge cost table (one row per D1 edge)."""
        return self.stage_dir("D") / "d2_edge_costs.parquet"

    def d2_constraints_json(self) -> Path:
        """Canonical Stage D2 normalized identity constraint spec (solver-agnostic)."""
        return self.stage_dir("D") / "d2_constraints.json"

    # ---- Stage E ----
    def match_sessions_jsonl(self) -> Path:
        return self.stage_dir("E") / "match_sessions.jsonl"

    # ---- Stage F ----
    def export_manifest_jsonl(self) -> Path:
        return self.stage_dir("F") / "export_manifest.jsonl"

    def exports_dir(self) -> Path:
        return self.stage_dir("F") / "exports"

    # ---- Manifest ----
    def clip_manifest_path(self) -> Path:
        return self.clip_root / "clip_manifest.json"

    # ---- Relative path helpers (relative to outputs/<clip_id>/) ----
    def rel_to_clip_root(self, path: Path) -> str:
        """
        Convert an absolute or repo-relative path to a clip-relative string path.
        Example:
          outputs/<clip_id>/stage_A/detections.parquet  ->  stage_A/detections.parquet
        """
        p = path
        try:
            p = p.relative_to(self.clip_root)
        except ValueError:
            # If caller passes a relative path already (e.g. stage_A/foo), accept it.
            p = Path(path)
        return p.as_posix()

    def ensure_dirs_for_stage(self, stage: StageLetter) -> None:
        self.stage_dir(stage).mkdir(parents=True, exist_ok=True)

    def ensure_mask_dirs(self) -> None:
        self.masks_dir().mkdir(parents=True, exist_ok=True)
        # masks_png is optional; create only if used
        self.masks_png_dir().mkdir(parents=True, exist_ok=True)

    def ensure_stage_A_mask_dirs(self) -> None:
        # Stage A masks are optional; create only if used.
        self.stage_A_masks_dir().mkdir(parents=True, exist_ok=True)

    def ensure_exports_dir(self) -> None:
        self.exports_dir().mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SessionOutputLayout:
    """
    Canonical layout for session-level outputs (multi-clip, multi-camera).

    A session groups clips from the same gym class schedule window.
    root points to the *repo* outputs directory by default: Path("outputs").
    session_root = root / gym_id / "sessions" / date / session_id
    """
    gym_id: str
    date: str          # YYYY-MM-DD in gym's local timezone
    session_id: str    # e.g. "2026-03-18T2000" — date + start time, no colon
    root: Path = Path("outputs")

    @property
    def session_root(self) -> Path:
        return self.root / self.gym_id / "sessions" / self.date / self.session_id

    def stage_dir(self, stage: StageLetter) -> Path:
        return self.session_root / f"stage_{stage}"

    # ---- Stage D (session-level) ----
    def session_tracklet_bank_frames_parquet(self, cam_id: str) -> Path:
        return self.stage_dir("D") / f"tracklet_bank_frames_{cam_id}.parquet"

    def session_person_tracks_parquet(self, cam_id: str) -> Path:
        return self.stage_dir("D") / f"person_tracks_{cam_id}.parquet"

    def session_identity_merge_jsonl(self) -> Path:
        return self.stage_dir("D") / "identity_merge.jsonl"

    def session_cross_camera_identities_jsonl(self) -> Path:
        return self.stage_dir("D") / "cross_camera_identities.jsonl"

    # ---- Stage E (session-level) ----
    def session_match_sessions_jsonl(self) -> Path:
        return self.stage_dir("E") / "match_sessions.jsonl"

    # ---- Stage F (session-level) ----
    def session_export_manifest_jsonl(self) -> Path:
        return self.stage_dir("F") / "export_manifest.jsonl"

    # ---- Audit ----
    def session_audit_jsonl(self, stage: StageLetter) -> Path:
        return self.stage_dir(stage) / "audit.jsonl"

    # ---- Sentinels ----
    def phase1_complete_sentinel(self, cam_id: str) -> Path:
        return self.session_root / f".phase1_complete_{cam_id}"

    def session_ready_sentinel(self) -> Path:
        return self.session_root / ".session_ready"

    def tag_required_sentinel(self) -> Path:
        return self.session_root / ".tag_required"

    def processing_sentinel(self) -> Path:
        return self.session_root / ".processing"

    def uploaded_sentinel(self) -> Path:
        return self.session_root / ".uploaded"

    # ---- Utility ----
    def ensure_session_root(self) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)

    def ensure_dirs_for_stage(self, stage: StageLetter) -> None:
        self.stage_dir(stage).mkdir(parents=True, exist_ok=True)
