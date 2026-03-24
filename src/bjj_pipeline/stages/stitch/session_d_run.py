"""Session-level Stage D runner: aggregate per-clip D0 banks → run D1→D4.

Does NOT call run_d0 — aggregates existing per-clip D0 outputs into a
combined bank, then runs the existing D1→D4 ILP pipeline via a layout
adapter that redirects all path lookups to session-level directories.

CP14c: session-level stitching for cross-clip identity resolution.
CP14e: frame index offset fix — per-clip 0-based frame indices are offset
       by wall-clock time relative to session start before aggregation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bjj_pipeline.contracts.f0_manifest import load_manifest
from bjj_pipeline.contracts.f0_paths import (
    ClipOutputLayout,
    SessionOutputLayout,
    StageLetter,
)
from bjj_pipeline.stages.orchestration.pipeline import (
    PipelineError,
    compute_output_root,
    validate_ingest_path,
)


# ---------------------------------------------------------------------------
# SessionManifest — lightweight stand-in for ClipManifest
# ---------------------------------------------------------------------------

@dataclass
class SessionManifest:
    """Manifest-like object for session-level D1→D4 runs.

    Provides the same attribute interface that D1–D4 runners read from
    ClipManifest, plus get_artifact_path() backed by a mutable registry.
    """
    clip_id: str            # session_id used as clip_id
    camera_id: str
    gym_id: str
    fps: float
    frame_count: int
    duration_ms: int
    pipeline_version: str = ""
    input_video_path: str = ""   # empty — no single source video
    _artifact_registry: Dict[str, str] = field(default_factory=dict, repr=False)

    def get_artifact_path(self, *, stage: str, key: str) -> str:
        """Resolve an artifact path from the registry (relative to session_root)."""
        lookup = f"{stage}:{key}"
        if lookup not in self._artifact_registry:
            raise FileNotFoundError(
                f"SessionManifest: artifact {lookup!r} not registered"
            )
        return self._artifact_registry[lookup]

    def register_artifact(self, *, stage: str, key: str, relpath: str) -> None:
        """Register an artifact path for later lookup by D3 compile."""
        self._artifact_registry[f"{stage}:{key}"] = relpath


# ---------------------------------------------------------------------------
# SessionStageLayoutAdapter — duck-typed layout for D1→D4, E, F
# ---------------------------------------------------------------------------

class SessionStageLayoutAdapter:
    """Wraps SessionOutputLayout to expose the same method interface that
    run_d1, run_d2, run_d3, run_d4, Stage E, and Stage F call on
    ClipOutputLayout.

    All write paths point to session_layout.stage_dir(stage).
    All read paths point to aggregated session bank files.
    """

    def __init__(self, session_layout: SessionOutputLayout, cam_id: str) -> None:
        self._sl = session_layout
        self._cam_id = cam_id
        self._stage_d = session_layout.stage_dir("D")
        self._stage_d.mkdir(parents=True, exist_ok=True)

    # ---- clip_root (property) ----
    @property
    def clip_root(self) -> Path:
        return self._sl.session_root

    # ---- stage_dir ----
    def stage_dir(self, stage: StageLetter) -> Path:
        d = self._sl.stage_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- Aggregated bank read paths ----
    def tracklet_bank_frames_parquet(self) -> Path:
        return self._sl.session_tracklet_bank_frames_parquet(self._cam_id)

    def tracklet_bank_summaries_parquet(self) -> Path:
        return self._stage_d / f"tracklet_bank_summaries_{self._cam_id}.parquet"

    # Stage A aliases — in session context, D0 is skipped so D1 reads the
    # aggregated bank directly via these aliases.
    def tracklet_frames_parquet(self) -> Path:
        return self.tracklet_bank_frames_parquet()

    def tracklet_summaries_parquet(self) -> Path:
        return self.tracklet_bank_summaries_parquet()

    def identity_hints_jsonl(self) -> Path:
        return self._stage_d / f"identity_hints_{self._cam_id}.jsonl"

    def detections_parquet(self) -> Path:
        return self._stage_d / f"detections_{self._cam_id}.parquet"

    # ---- D1 write paths ----
    def d1_graph_nodes_parquet(self) -> Path:
        return self._stage_d / "d1_graph_nodes.parquet"

    def d1_graph_edges_parquet(self) -> Path:
        return self._stage_d / "d1_graph_edges.parquet"

    def d1_segments_parquet(self) -> Path:
        return self._stage_d / "d1_segments.parquet"

    # ---- D2 write paths ----
    def d2_edge_costs_parquet(self) -> Path:
        return self._stage_d / "d2_edge_costs.parquet"

    def d2_constraints_json(self) -> Path:
        return self._stage_d / "d2_constraints.json"

    # ---- D4 write paths ----
    def person_tracks_parquet(self) -> Path:
        return self._sl.session_person_tracks_parquet(self._cam_id)

    def person_spans_parquet(self) -> Path:
        return self._stage_d / f"person_spans_{self._cam_id}.parquet"

    def identity_assignments_jsonl(self) -> Path:
        return self._stage_d / f"identity_assignments_{self._cam_id}.jsonl"

    # ---- Stage E write paths ----
    def match_sessions_jsonl(self) -> Path:
        return self._sl.session_match_sessions_jsonl()

    # ---- Stage F write paths ----
    def export_manifest_jsonl(self) -> Path:
        return self._sl.session_export_manifest_jsonl()

    def exports_dir(self) -> Path:
        return self._sl.stage_dir("F") / "exports"

    def ensure_exports_dir(self) -> None:
        self.exports_dir().mkdir(parents=True, exist_ok=True)

    def ensure_dirs_for_stage(self, stage: StageLetter) -> None:
        self.stage_dir(stage)  # stage_dir already creates on demand

    # ---- Audit ----
    def audit_jsonl(self, stage: StageLetter) -> Path:
        return self._sl.session_audit_jsonl(stage)

    # ---- Manifest path (needed by run.py but not used in session context) ----
    def clip_manifest_path(self) -> Path:
        return self._sl.session_root / "session_manifest.json"

    # ---- Path helpers ----
    def rel_to_clip_root(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._sl.session_root))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Frame offset helpers (CP14e)
# ---------------------------------------------------------------------------

_CLIP_TS_RE = re.compile(r"-(\d{8})-(\d{6})\.")


def parse_clip_timestamp(mp4_path: Path) -> Optional[datetime]:
    """Parse clip start time from MP4 filename.

    Filename format: {cam_id}-{YYYYMMDD}-{HHMMSS}.mp4
    e.g. FP7oJQ-20260318-200014.mp4 → 2026-03-18 20:00:14

    Returns None on any parse failure.
    """
    m = _CLIP_TS_RE.search(mp4_path.name)
    if not m:
        return None
    try:
        date_str, time_str = m.group(1), m.group(2)
        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except (ValueError, IndexError):
        return None


def derive_clip_frame_offset(
    mp4_path: Path, session_start_dt: datetime, fps: float
) -> int:
    """Compute frame offset for a clip relative to session start.

    Returns int: round((clip_start_dt - session_start_dt).total_seconds() * fps)
    Returns 0 on any parse failure (conservative fallback).
    """
    clip_dt = parse_clip_timestamp(mp4_path)
    if clip_dt is None or fps <= 0:
        return 0
    delta_sec = (clip_dt - session_start_dt).total_seconds()
    if delta_sec < 0:
        return 0
    return round(delta_sec * fps)


# ---------------------------------------------------------------------------
# Bank aggregation
# ---------------------------------------------------------------------------

def _get_clip_layout(mp4_path: Path, cam_id: str, output_root: Path) -> Optional[ClipOutputLayout]:
    """Derive ClipOutputLayout for a per-clip MP4."""
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        scoped_root = compute_output_root(info, base_root=output_root)
        return ClipOutputLayout(clip_id=mp4_path.stem, root=scoped_root)
    except PipelineError:
        return None


def aggregate_session_bank(
    *,
    session_layout: SessionOutputLayout,
    adapter: SessionStageLayoutAdapter,
    session_clips: List[Tuple[Path, str]],
    cam_id: str,
    output_root: Path,
    fps: float,
) -> Tuple[Path, Path, Path, Path]:
    """Aggregate per-clip D0 bank outputs for a single camera into session-level
    combined bank files.

    Returns (frames_path, summaries_path, detections_path, hints_path).

    Tracklet ID namespacing: every tracklet_id is prefixed with
    "{clip_id}:{original_tracklet_id}" before concatenation.

    Frame index offsetting (CP14e): per-clip 0-based frame indices are offset
    by the clip's wall-clock offset relative to the earliest clip in the session.
    This ensures D1 sees correct temporal ordering across clips.
    """
    # --- Pre-scan: parse clip timestamps, find session start ---
    cam_clips: List[Tuple[Path, Optional[datetime]]] = []
    for mp4_path, clip_cam_id in session_clips:
        if clip_cam_id != cam_id:
            continue
        dt = parse_clip_timestamp(mp4_path)
        cam_clips.append((mp4_path, dt))

    valid_dts = [dt for _, dt in cam_clips if dt is not None]
    session_start_dt = min(valid_dts) if valid_dts else None

    all_frames: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    all_detections: List[pd.DataFrame] = []
    all_hints: List[dict] = []

    for mp4_path, clip_dt in cam_clips:
        layout = _get_clip_layout(mp4_path, cam_id, output_root)
        if layout is None:
            logger.warning("session_bank: cannot derive layout for {}", mp4_path.name)
            continue

        clip_id_prefix = mp4_path.stem

        # Compute frame offset for this clip
        frame_offset = 0
        if session_start_dt is not None:
            frame_offset = derive_clip_frame_offset(mp4_path, session_start_dt, fps)
        if frame_offset > 0:
            logger.info(
                "session_bank: {} offset={} frames ({:.1f}s)",
                clip_id_prefix, frame_offset, frame_offset / fps if fps > 0 else 0,
            )

        # --- Bank frames ---
        frames_path = layout.tracklet_bank_frames_parquet()
        if not frames_path.exists():
            logger.warning("session_bank: missing bank frames for {}", clip_id_prefix)
            continue

        frames_df = pd.read_parquet(frames_path)
        if "tracklet_id" in frames_df.columns:
            frames_df["tracklet_id"] = clip_id_prefix + ":" + frames_df["tracklet_id"].astype(str)
        # Apply frame offset (CP14e)
        if frame_offset > 0 and "frame_index" in frames_df.columns:
            frames_df["frame_index"] = frames_df["frame_index"] + frame_offset
        all_frames.append(frames_df)

        # --- Bank summaries ---
        summ_path = layout.tracklet_bank_summaries_parquet()
        if summ_path.exists():
            summ_df = pd.read_parquet(summ_path)
            if "tracklet_id" in summ_df.columns:
                summ_df["tracklet_id"] = clip_id_prefix + ":" + summ_df["tracklet_id"].astype(str)
            # Apply frame offset to summary frame ranges (CP14e)
            if frame_offset > 0:
                if "start_frame" in summ_df.columns:
                    summ_df["start_frame"] = summ_df["start_frame"] + frame_offset
                if "end_frame" in summ_df.columns:
                    summ_df["end_frame"] = summ_df["end_frame"] + frame_offset
            all_summaries.append(summ_df)
        else:
            logger.warning("session_bank: missing bank summaries for {}", clip_id_prefix)

        # --- Detections (Stage A) ---
        det_path = layout.detections_parquet()
        if det_path.exists():
            det_df = pd.read_parquet(det_path)
            # Apply frame offset to detections (CP14e)
            if frame_offset > 0 and "frame_index" in det_df.columns:
                det_df["frame_index"] = det_df["frame_index"] + frame_offset
            all_detections.append(det_df)
        else:
            logger.warning("session_bank: missing detections for {}", clip_id_prefix)

        # --- Identity hints ---
        # CP15: hints are frame-offset to match D1 node frame ranges.
        # D3 binds tag pings to nodes by frame range — both must use the
        # same session-level offset frame space.
        hints_path = layout.identity_hints_jsonl()
        if hints_path.exists():
            with open(hints_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        hint = json.loads(line)
                        if "tracklet_id" in hint:
                            hint["tracklet_id"] = f"{clip_id_prefix}:{hint['tracklet_id']}"
                        # Apply frame offset to evidence frame indices (CP15)
                        if frame_offset > 0 and isinstance(hint.get("evidence"), dict):
                            hint = dict(hint)
                            evidence = dict(hint["evidence"])
                            for fkey in ("first_seen_frame", "frame_index"):
                                if fkey in evidence and evidence[fkey] is not None:
                                    try:
                                        evidence[fkey] = int(evidence[fkey]) + frame_offset
                                    except (TypeError, ValueError):
                                        pass
                            hint["evidence"] = evidence
                        all_hints.append(hint)
                    except json.JSONDecodeError:
                        pass

    if not all_frames:
        raise PipelineError("no valid frames for session bank aggregation")

    # --- Write aggregated outputs ---
    session_layout.ensure_dirs_for_stage("D")

    # Frames
    combined_frames = pd.concat(all_frames, ignore_index=True)
    frames_out = adapter.tracklet_bank_frames_parquet()
    combined_frames.to_parquet(frames_out, index=False)
    logger.info("session_bank: wrote {} frames → {}", len(combined_frames), frames_out.name)

    # Summaries
    summaries_out = adapter.tracklet_bank_summaries_parquet()
    if all_summaries:
        combined_summaries = pd.concat(all_summaries, ignore_index=True)
        combined_summaries.to_parquet(summaries_out, index=False)
    else:
        pd.DataFrame().to_parquet(summaries_out, index=False)

    # Detections
    detections_out = adapter.detections_parquet()
    if all_detections:
        combined_detections = pd.concat(all_detections, ignore_index=True)
        combined_detections.to_parquet(detections_out, index=False)
    else:
        pd.DataFrame().to_parquet(detections_out, index=False)

    # Hints
    hints_out = adapter.identity_hints_jsonl()
    with open(hints_out, "w", encoding="utf-8") as f:
        for hint in all_hints:
            f.write(json.dumps(hint) + "\n")

    return frames_out, summaries_out, detections_out, hints_out


# ---------------------------------------------------------------------------
# Config helpers (mirrored from run.py / solver.py)
# ---------------------------------------------------------------------------

def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Lightweight dot-path getter."""
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Session-level Stage D runner
# ---------------------------------------------------------------------------

def run_session_d(
    *,
    config: Dict[str, Any],
    session_layout: SessionOutputLayout,
    session_clips: List[Tuple[Path, str]],
    cam_id: str,
    output_root: Path,
) -> Optional[SessionManifest]:
    """Run session-level Stage D (D1→D4) for one camera.

    1. Scan per-clip manifests for fps (needed before aggregation)
    2. aggregate_session_bank() — build combined bank with frame offsets
    3. Build SessionManifest
    4. Run D1 → register artifacts → D2 → register artifacts → D3 → D4

    Returns SessionManifest (always, when fps is available) so the processor
    can pass it to Stage E and F. Returns None only if no valid manifest
    with fps>0 was found.
    """
    adapter = SessionStageLayoutAdapter(session_layout, cam_id)

    # --- Step 1: Scan manifests for fps (must happen before aggregation) ---
    fps: Optional[float] = None
    pipeline_version: str = ""
    total_frame_count: int = 0
    total_duration_ms: int = 0
    gym_id = session_layout.gym_id

    for mp4_path, clip_cam_id in session_clips:
        if clip_cam_id != cam_id:
            continue
        layout = _get_clip_layout(mp4_path, cam_id, output_root)
        if layout is None:
            continue
        manifest_path = layout.clip_manifest_path()
        if not manifest_path.exists():
            continue
        try:
            cm = load_manifest(manifest_path)
            if fps is None and cm.fps > 0:
                fps = cm.fps
            if not pipeline_version and cm.pipeline_version:
                pipeline_version = cm.pipeline_version
            total_frame_count += cm.frame_count
            total_duration_ms += cm.duration_ms
        except Exception as exc:
            logger.warning("session_d: failed to load manifest {}: {}", manifest_path, exc)
            continue

    if fps is None or fps <= 0:
        raise PipelineError(
            f"session_d: no valid ClipManifest with fps>0 found for session "
            f"{session_layout.session_id} cam={cam_id}"
        )

    # --- Step 2: Aggregate with frame offsets ---
    frames_out, summaries_out, detections_out, hints_out = aggregate_session_bank(
        session_layout=session_layout,
        adapter=adapter,
        session_clips=session_clips,
        cam_id=cam_id,
        output_root=output_root,
        fps=fps,
    )

    # --- Step 3: Build SessionManifest ---
    session_manifest = SessionManifest(
        clip_id=session_layout.session_id,
        camera_id=cam_id,
        gym_id=gym_id,
        fps=fps,
        frame_count=total_frame_count,
        duration_ms=total_duration_ms,
        pipeline_version=pipeline_version,
        input_video_path="",
    )

    # --- Step 4: Build inputs dict ---
    inputs: Dict[str, Any] = {
        "layout": adapter,
        "manifest": session_manifest,
    }

    # --- Step 5: D1 ---
    from bjj_pipeline.stages.stitch.d1_graph_build import run_d1

    logger.info("session_d: running D1 for session={} cam={}", session_layout.session_id, cam_id)
    run_d1(cfg=config, layout=adapter, manifest=session_manifest)

    # Register D1 artifacts for D3 compile's manifest.get_artifact_path() calls
    session_manifest.register_artifact(
        stage="D", key="d1_graph_nodes_parquet",
        relpath=adapter.rel_to_clip_root(adapter.d1_graph_nodes_parquet()),
    )
    session_manifest.register_artifact(
        stage="D", key="d1_graph_edges_parquet",
        relpath=adapter.rel_to_clip_root(adapter.d1_graph_edges_parquet()),
    )

    # --- Step 6: D2 ---
    from bjj_pipeline.stages.stitch.d2_run import run_d2

    logger.info("session_d: running D2 for session={} cam={}", session_layout.session_id, cam_id)
    run_d2(config=config, inputs=inputs)

    # Register D2 artifacts for D3 compile
    session_manifest.register_artifact(
        stage="D", key="d2_edge_costs_parquet",
        relpath=adapter.rel_to_clip_root(adapter.d2_edge_costs_parquet()),
    )
    session_manifest.register_artifact(
        stage="D", key="d2_constraints_json",
        relpath=adapter.rel_to_clip_root(adapter.d2_constraints_json()),
    )

    # --- Step 7: D3 — use existing run_d3 dispatch (handles compile + solver) ---
    from bjj_pipeline.stages.stitch.solver import run_d3

    logger.info("session_d: running D3 for session={} cam={}", session_layout.session_id, cam_id)
    compiled, ilp_res = run_d3(config=config, inputs=inputs)

    # --- Step 8: D4 ---
    if ilp_res is None:
        checkpoint = _cfg_get(config, "stages.stage_D.d3_checkpoint", None)
        if checkpoint is None:
            checkpoint = _cfg_get(config, "stage_D.d3_checkpoint", "POC_0")
        logger.warning(
            "session_d: D3 checkpoint={} did not produce ILP result — skipping D4",
            checkpoint,
        )
        # Return manifest even if D4 skipped — Stage E handles missing inputs
        return session_manifest

    from bjj_pipeline.stages.stitch.d4_emit import run_d4_emit

    checkpoint = _cfg_get(config, "stages.stage_D.d3_checkpoint", None)
    if checkpoint is None:
        checkpoint = _cfg_get(config, "stage_D.d3_checkpoint", "POC_0")

    logger.info("session_d: running D4 for session={} cam={}", session_layout.session_id, cam_id)
    run_d4_emit(
        config=config,
        inputs=inputs,
        compiled=compiled,
        res=ilp_res,
        checkpoint=str(checkpoint),
    )

    logger.info(
        "session_d: completed D1→D4 for session={} cam={} person_tracks={}",
        session_layout.session_id,
        cam_id,
        adapter.person_tracks_parquet(),
    )

    return session_manifest
