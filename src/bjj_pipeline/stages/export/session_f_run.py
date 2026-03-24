"""Session-level Stage F runner: export match clips from session-level data.

Operates on session-level match_sessions.jsonl and person_tracks.parquet
from Stage D4 (via CP14c adapter). Extracts exact frame ranges from each
source MP4, concatenates multi-source segments via ffmpeg concat demuxer.

CP14e: session-level clip export with multi-source video support.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bjj_pipeline.contracts.f0_models import SCHEMA_VERSION_DEFAULT
from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
from bjj_pipeline.contracts.f0_validate import validate_export_manifest_records
from bjj_pipeline.stages.export.cropper import CropPlanError, FixedRoiCropPlan, plan_crop_fixed_roi
from bjj_pipeline.stages.export.ffmpeg import ExportClipError, export_clip, probe_video_metadata
from bjj_pipeline.stages.export.manifest import (
    build_supabase_log_contracts,
    compute_clip_seconds,
    derive_storage_target,
    get_file_stats,
)
from bjj_pipeline.stages.stitch.session_d_run import (
    SessionStageLayoutAdapter,
    derive_clip_frame_offset,
    parse_clip_timestamp,
    _get_clip_layout,
)
from bjj_pipeline.stages.orchestration.pipeline import PipelineError


# ---------------------------------------------------------------------------
# Source clip registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceClipInfo:
    clip_id: str
    mp4_path: Path
    cam_id: str
    frame_offset: int
    fps: float
    duration_frames: Optional[int] = None


def _build_source_registry(
    session_clips: List[Tuple[Path, str]],
    cam_id: str,
    fps: float,
) -> List[SourceClipInfo]:
    """Build ordered registry of source clips for one camera."""
    cam_clips = [(mp4, cid) for mp4, cid in session_clips if cid == cam_id]
    if not cam_clips:
        return []

    # Find session start from earliest clip timestamp
    timestamps = []
    for mp4, _ in cam_clips:
        dt = parse_clip_timestamp(mp4)
        if dt is not None:
            timestamps.append((mp4, dt))

    if not timestamps:
        # Fallback: no parseable timestamps, use single clip with offset 0
        return [
            SourceClipInfo(
                clip_id=mp4.stem, mp4_path=mp4, cam_id=cid,
                frame_offset=0, fps=fps,
            )
            for mp4, cid in cam_clips
        ]

    session_start_dt = min(dt for _, dt in timestamps)

    registry: List[SourceClipInfo] = []
    for mp4, cid in cam_clips:
        offset = derive_clip_frame_offset(mp4, session_start_dt, fps)
        # Probe duration
        dur_frames = None
        try:
            meta = probe_video_metadata(mp4)
            if meta.duration_sec is not None and fps > 0:
                dur_frames = int(round(meta.duration_sec * fps))
        except Exception:
            pass
        registry.append(SourceClipInfo(
            clip_id=mp4.stem, mp4_path=mp4, cam_id=cid,
            frame_offset=offset, fps=fps, duration_frames=dur_frames,
        ))

    registry.sort(key=lambda s: s.frame_offset)
    return registry


# ---------------------------------------------------------------------------
# Multi-source extraction
# ---------------------------------------------------------------------------

def _extract_session_clip(
    *,
    source_clips: List[SourceClipInfo],
    match_start_frame: int,
    match_end_frame: int,
    output_path: Path,
    fps: float,
    crop_plan: FixedRoiCropPlan,
) -> str:
    """Extract frame ranges from source files and concatenate.

    For each source clip whose frame range overlaps [match_start, match_end]:
    1. Compute local frame range
    2. Extract segment via ffmpeg with crop
    3. Concatenate using ffmpeg concat demuxer

    Returns ffmpeg command string for audit.
    """
    # Find overlapping clips
    segments: List[Tuple[SourceClipInfo, int, int]] = []
    for clip in source_clips:
        # Clip covers frames [clip.frame_offset, clip.frame_offset + duration)
        clip_end = clip.frame_offset + (clip.duration_frames or 999999)

        # Check overlap with match range
        if clip.frame_offset > match_end_frame:
            continue
        if clip_end < match_start_frame:
            continue

        local_start = max(0, match_start_frame - clip.frame_offset)
        local_end = match_end_frame - clip.frame_offset
        if clip.duration_frames is not None:
            local_end = min(local_end, clip.duration_frames - 1)

        if local_end <= local_start:
            continue

        segments.append((clip, local_start, local_end))

    if not segments:
        raise ExportClipError(
            f"No source clips overlap frame range [{match_start_frame}, {match_end_frame}]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Single segment: extract directly to output
    if len(segments) == 1:
        clip, local_start, local_end = segments[0]
        result = export_clip(
            input_video_path=clip.mp4_path,
            output_video_path=output_path,
            crop_plan=crop_plan,
            fps=fps,
            start_frame=local_start,
            end_frame=local_end,
        )
        return result.ffmpeg_cmd

    # Multiple segments: extract to temp files, then concatenate
    with tempfile.TemporaryDirectory(prefix="session_f_") as tmpdir:
        tmp = Path(tmpdir)
        seg_paths: List[Path] = []
        cmds: List[str] = []

        for i, (clip, local_start, local_end) in enumerate(segments):
            seg_path = tmp / f"seg_{i:03d}.mp4"
            result = export_clip(
                input_video_path=clip.mp4_path,
                output_video_path=seg_path,
                crop_plan=crop_plan,
                fps=fps,
                start_frame=local_start,
                end_frame=local_end,
            )
            seg_paths.append(seg_path)
            cmds.append(result.ffmpeg_cmd)

        # Write concat list
        concat_list = tmp / "concat.txt"
        with concat_list.open("w") as f:
            for sp in seg_paths:
                f.write(f"file '{sp}'\n")

        # Concatenate
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.run(concat_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip()[-1200:]
            raise ExportClipError(
                f"ffmpeg concat failed: returncode={proc.returncode} stderr={stderr_tail}"
            )
        if not output_path.exists():
            raise ExportClipError(f"ffmpeg concat completed but output missing: {output_path}")

        cmds.append(subprocess.list2cmdline(concat_cmd))
        return " && ".join(cmds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_id_hash(*, person_id_a: str, person_id_b: str, start_frame: int, end_frame: int) -> str:
    raw = f"{person_id_a}|{person_id_b}|{start_frame}|{end_frame}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"sexport_{digest}"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _load_match_sessions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("artifact_type") == "match_session":
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: (int(r.get("start_frame", 0)), str(r.get("match_id", ""))))
    return records


def _stage_f_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    stages_blk = config.get("stages")
    if isinstance(stages_blk, dict):
        f_blk = stages_blk.get("stage_F")
        if isinstance(f_blk, dict):
            return f_blk
    blk = config.get("stage_F")
    return blk if isinstance(blk, dict) else {}


def _cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


# ---------------------------------------------------------------------------
# Session-level Stage F runner
# ---------------------------------------------------------------------------

def run_session_f(
    *,
    config: dict,
    session_layout: SessionOutputLayout,
    session_clips: List[Tuple[Path, str]],
    cam_id: str,
    output_root: Path,
) -> dict:
    """Run session-level Stage F for one camera.

    Reads:
      - session_layout.session_match_sessions_jsonl()
      - session_layout.session_person_tracks_parquet(cam_id)
      - source MP4 files from session_clips

    Writes:
      - session_layout.session_export_manifest_jsonl()
      - exported MP4 clips under session_layout.stage_dir("F") / "exports/"

    Returns dict with n_matches, n_exports, n_skipped, status.
    """
    stage_cfg = _stage_f_cfg(config)
    padding_px = _cfg_int(stage_cfg, "padding_px", 80)
    low_quantile = _cfg_float(stage_cfg, "low_quantile", 0.05)
    high_quantile = _cfg_float(stage_cfg, "high_quantile", 0.95)
    min_crop_width = _cfg_int(stage_cfg, "min_crop_width", 160)
    min_crop_height = _cfg_int(stage_cfg, "min_crop_height", 160)
    gym_id = session_layout.gym_id
    storage_bucket = str(stage_cfg.get("storage_bucket", "match-clips"))
    initial_status = str(stage_cfg.get("initial_status", "exported_local"))

    session_id = session_layout.session_id
    exports_dir = session_layout.stage_dir("F") / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = session_layout.session_audit_jsonl("F")
    export_manifest_path = session_layout.session_export_manifest_jsonl()

    # Clear previous run artifacts
    if export_manifest_path.exists():
        export_manifest_path.unlink()
    if audit_path.exists():
        audit_path.unlink()

    _append_jsonl(audit_path, {
        "artifact_type": "session_f_started",
        "created_at_ms": _now_ms(),
        "session_id": session_id,
        "cam_id": cam_id,
    })

    # --- Load match sessions ---
    match_sessions_path = session_layout.session_match_sessions_jsonl()
    matches = _load_match_sessions(match_sessions_path)

    if not matches:
        # Write no_matches manifest so already-processed guard fires
        export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with export_manifest_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "schema_version": SCHEMA_VERSION_DEFAULT,
                "artifact_type": "export_manifest",
                "status": "no_matches",
                "clip_id": session_id,
                "camera_id": cam_id,
                "pipeline_version": "session",
                "created_at_ms": _now_ms(),
            }) + "\n")
        _append_jsonl(audit_path, {
            "artifact_type": "session_f_no_matches",
            "created_at_ms": _now_ms(),
            "session_id": session_id,
        })
        logger.info("session_f: no match sessions for session={} cam={}", session_id, cam_id)
        return {"status": "no_matches", "n_matches": 0, "n_exports": 0, "n_skipped": 0}

    # --- Load person tracks ---
    person_tracks_path = session_layout.session_person_tracks_parquet(cam_id)
    if not person_tracks_path.exists():
        raise PipelineError(f"session_f: missing person_tracks: {person_tracks_path}")

    person_tracks_df = pd.read_parquet(person_tracks_path)
    required_cols = {"person_id", "frame_index", "x1", "y1", "x2", "y2"}
    missing_cols = sorted(required_cols - set(person_tracks_df.columns))
    if missing_cols:
        raise PipelineError(f"session_f: person_tracks missing columns: {missing_cols}")
    person_tracks_df = person_tracks_df.sort_values(
        ["frame_index", "person_id"], kind="mergesort"
    ).reset_index(drop=True)

    # --- Build source clip registry ---
    # Probe first clip for video metadata (all same camera = same resolution)
    first_clip = next((mp4 for mp4, cid in session_clips if cid == cam_id), None)
    if first_clip is None:
        raise PipelineError(f"session_f: no source clips for cam_id={cam_id}")

    video_meta = probe_video_metadata(first_clip)
    fps = float(video_meta.fps) if video_meta.fps > 0 else 30.0

    source_registry = _build_source_registry(session_clips, cam_id, fps)
    if not source_registry:
        raise PipelineError(f"session_f: empty source registry for cam_id={cam_id}")

    source_video_paths = [str(s.mp4_path) for s in source_registry]

    logger.info(
        "session_f: {} matches, {} source clips for session={} cam={}",
        len(matches), len(source_registry), session_id, cam_id,
    )

    # --- Process each match session ---
    export_records: List[Dict[str, Any]] = []
    skipped = 0
    created_at = _now_ms()

    for match in matches:
        match_id = match.get("match_id", "unknown")
        person_id_a = str(match.get("person_id_a", ""))
        person_id_b = str(match.get("person_id_b", ""))
        start_frame = int(match.get("start_frame", 0))
        end_frame = int(match.get("end_frame", 0))

        export_id = _match_id_hash(
            person_id_a=person_id_a, person_id_b=person_id_b,
            start_frame=start_frame, end_frame=end_frame,
        )

        try:
            # Crop plan
            crop_plan = plan_crop_fixed_roi(
                tracks_df=person_tracks_df,
                person_id_a=person_id_a,
                person_id_b=person_id_b,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_width=int(video_meta.width),
                frame_height=int(video_meta.height),
                padding_px=padding_px,
                low_quantile=low_quantile,
                high_quantile=high_quantile,
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
            )

            # Extract clip
            output_abs = exports_dir / f"{export_id}.mp4"
            ffmpeg_cmd = _extract_session_clip(
                source_clips=source_registry,
                match_start_frame=start_frame,
                match_end_frame=end_frame,
                output_path=output_abs,
                fps=fps,
                crop_plan=crop_plan,
            )

            # Compute metadata
            file_hash = _sha256_file(output_abs)
            file_stats = get_file_stats(output_abs)
            file_size_bytes = file_stats.get("file_size_bytes")
            seconds_payload = compute_clip_seconds(
                fps=fps,
                export_start_frame=start_frame,
                export_end_frame=end_frame,
            )
            storage_target = derive_storage_target(
                gym_id=gym_id or "unknown",
                camera_id=cam_id,
                clip_id=session_id,
                export_id=export_id,
                storage_bucket=storage_bucket,
            )

            # Build clip row for uploader
            april_tag_a = match.get("april_tag_id_a")
            april_tag_b = match.get("april_tag_id_b")
            clip_row = {
                "match_id": str(match_id),
                "file_path": str(storage_target.object_path),
                "storage_bucket": str(storage_target.bucket),
                "storage_object_path": str(storage_target.object_path),
                "start_seconds": float(seconds_payload["start_seconds"]),
                "end_seconds": float(seconds_payload["end_seconds"]),
                "duration_seconds": float(seconds_payload["duration_seconds"]),
                "camera_id": str(cam_id),
                "status": str(initial_status),
                "fighter_a_tag_id": april_tag_a,
                "fighter_b_tag_id": april_tag_b,
                # source_video_paths: popped by uploader, used to resolve source_video_ids
                "source_video_paths": source_video_paths,
                "metadata": {
                    "session_id": session_id,
                    "clip_id": session_id,
                    "export_id": export_id,
                    "local_output_path": str(output_abs),
                    "source_match_ids": [match_id],
                    "source_match_count": 1,
                    "match_start_frame": start_frame,
                    "match_end_frame": end_frame,
                    "export_start_frame": start_frame,
                    "export_end_frame": end_frame,
                    "pipeline_version": "session",
                    "crop_mode": "fixed_roi",
                    "hash_sha256": file_hash,
                    "file_size_bytes": file_size_bytes,
                    "source_clip_count": len(source_registry),
                },
            }

            # Use a simple ExportSession-like duck type for log contracts
            class _LogProxy:
                def __init__(self):
                    self.export_id = export_id
                    self.source_match_ids = (str(match_id),)

            log_events = build_supabase_log_contracts(
                export_session=_LogProxy(),
                clip_id=session_id,
                camera_id=cam_id,
                storage_target=storage_target,
                clip_row_payload=clip_row,
            )

            record = {
                "schema_version": SCHEMA_VERSION_DEFAULT,
                "artifact_type": "export_manifest",
                "clip_id": session_id,
                "camera_id": cam_id,
                "gym_id": gym_id,
                "pipeline_version": "session",
                "created_at_ms": created_at,
                "export_id": export_id,
                "match_id": match_id,
                "output_video_path": str(output_abs),
                "crop_mode": "fixed_roi",
                "privacy": {"redaction_enabled": False, "method": None},
                "inputs": {
                    "input_video_path": str(source_registry[0].mp4_path),
                    "source_video_paths": source_video_paths,
                    "person_id_a": person_id_a,
                    "person_id_b": person_id_b,
                    "match_start_frame": start_frame,
                    "match_end_frame": end_frame,
                    "export_start_frame": start_frame,
                    "export_end_frame": end_frame,
                    "crop_rect_xywh": [crop_plan.x, crop_plan.y, crop_plan.width, crop_plan.height],
                    "start_seconds": float(seconds_payload["start_seconds"]),
                    "end_seconds": float(seconds_payload["end_seconds"]),
                    "duration_seconds": float(seconds_payload["duration_seconds"]),
                    "file_size_bytes": file_size_bytes,
                    "storage_bucket": str(storage_target.bucket),
                    "storage_object_path": str(storage_target.object_path),
                    "uploader_contract": {
                        "storage": {
                            "bucket": str(storage_target.bucket),
                            "object_path": str(storage_target.object_path),
                            "file_name": str(storage_target.file_name),
                        },
                        "clip_row": clip_row,
                        "log_events": log_events,
                    },
                },
                "ffmpeg_cmd": ffmpeg_cmd,
                "hash_sha256": file_hash,
                "collision_hints": (
                    {"same_tag_collision": True, "tag_id": int(april_tag_a)}
                    if (april_tag_a is not None and april_tag_a == april_tag_b)
                    else None
                ),
            }
            export_records.append(record)

            _append_jsonl(audit_path, {
                "artifact_type": "session_clip_exported",
                "created_at_ms": _now_ms(),
                "session_id": session_id,
                "export_id": export_id,
                "match_id": match_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "source_clip_count": len(source_registry),
                "output_path": str(output_abs),
            })

        except (CropPlanError, ExportClipError, FileNotFoundError, ValueError) as e:
            skipped += 1
            _append_jsonl(audit_path, {
                "artifact_type": "session_clip_skipped",
                "created_at_ms": _now_ms(),
                "session_id": session_id,
                "export_id": export_id,
                "match_id": match_id,
                "reason": str(e),
            })
            logger.warning("session_f: skipped match {} — {}", match_id, e)
            continue

    # --- Write export manifest ---
    validate_export_manifest_records(export_records, expected_clip_id=session_id)

    export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with export_manifest_path.open("w", encoding="utf-8") as f:
        for rec in export_records:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

    _append_jsonl(audit_path, {
        "artifact_type": "session_f_summary",
        "created_at_ms": _now_ms(),
        "session_id": session_id,
        "cam_id": cam_id,
        "n_matches": len(matches),
        "n_exports": len(export_records),
        "n_skipped": skipped,
    })

    logger.info(
        "session_f: completed session={} cam={} exports={} skipped={}",
        session_id, cam_id, len(export_records), skipped,
    )

    return {
        "status": "completed",
        "n_matches": len(matches),
        "n_exports": len(export_records),
        "n_skipped": skipped,
    }
