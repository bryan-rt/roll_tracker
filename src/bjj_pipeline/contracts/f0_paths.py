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
    def contact_points_parquet(self) -> Path:
        return self.stage_dir("B") / "contact_points.parquet"

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

    def identity_assignments_jsonl(self) -> Path:
        # Produced after stitching (post-pass) but stored under stage_D outputs canonically
        return self.stage_dir("D") / "identity_assignments.jsonl"

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
