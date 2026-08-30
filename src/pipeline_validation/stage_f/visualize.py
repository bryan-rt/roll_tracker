"""Match preview visualization with diagnostic layers.

Renders a per-clip mp4 with four visual layers:
1. All Stage A detections (grey, thin) — what the detection model saw
2. Stage D-accepted detections (colored by person_id, thick) — what survived stitching
3. Stage E match envelopes (orange, dashed) — what Stage F would crop
4. Tag observation icons (yellow) — AprilTag-evidence-bearing tracklets

Grey-only boxes (layer 1 without layer 2 overlay) are the dropped-tracklet
diagnostic: Stage A detected, Stage D rejected.
"""
from __future__ import annotations

import colorsys
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import yaml

from bjj_pipeline.stages.export.cropper import (
    CropPlanError,
    FixedRoiCropPlan,
    plan_crop_fixed_roi,
)
from pipeline_validation.common.manifest import load_manifest
from pipeline_validation.common.schemas import ExportEntry, ModelManifest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = REPO_ROOT / "outputs" / "_eval" / "stage_f"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class DetRow(NamedTuple):
    detection_id: str
    tracklet_id: str
    x1: int
    y1: int
    x2: int
    y2: int


class EnvelopePlan(NamedTuple):
    start_frame: int
    end_frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    person_id_a: str
    person_id_b: str


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_pipeline_paths(
    manifest: ModelManifest, export: ExportEntry
) -> dict[str, Path]:
    gym_id = manifest.pipeline_gym_id or "_eval_gt"
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id
    matches = list(OUTPUTS_DIR.glob(f"{gym_id}/{cam}/**/{clip_id}"))
    if not matches:
        raise FileNotFoundError(f"No pipeline output for {cam}/{clip_id}")
    clip_dir = matches[0]
    return {
        "clip_dir": clip_dir,
        "detections": clip_dir / "stage_A" / "detections.parquet",
        "person_tracks": clip_dir / "stage_D" / "person_tracks.parquet",
        "identity_hints": clip_dir / "stage_C" / "identity_hints.jsonl",
        "match_sessions": clip_dir / "stage_E" / "match_sessions.jsonl",
    }


def _find_source_video(export: ExportEntry, clip_dir: Path) -> Path | None:
    if export.source_video_path:
        p = REPO_ROOT / export.source_video_path
        if p.exists():
            return p
    cm_path = clip_dir / "clip_manifest.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cm = json.load(f)
        p = REPO_ROOT / cm.get("input_video_path", "")
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Stage F config
# ---------------------------------------------------------------------------

def _load_stage_f_config() -> dict:
    """Read Stage F crop parameters from production config."""
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    stage_f = cfg.get("stages", {}).get("stage_F", cfg.get("stage_F", {}))
    return {
        "padding_px": int(stage_f.get("padding_px", 80)),
        "low_quantile": float(stage_f.get("low_quantile", 0.05)),
        "high_quantile": float(stage_f.get("high_quantile", 0.95)),
        "min_crop_width": int(stage_f.get("min_crop_width", 160)),
        "min_crop_height": int(stage_f.get("min_crop_height", 160)),
    }


# ---------------------------------------------------------------------------
# Data pre-loading
# ---------------------------------------------------------------------------

def _preload_detections(det_df: pd.DataFrame) -> dict[int, list[DetRow]]:
    """Group detections by frame_index for fast per-frame access."""
    by_frame: dict[int, list[DetRow]] = defaultdict(list)
    for _, r in det_df.iterrows():
        by_frame[int(r.frame_index)].append(DetRow(
            detection_id=str(r.detection_id),
            tracklet_id=str(r.tracklet_id) if pd.notna(r.tracklet_id) else "",
            x1=int(r.x1), y1=int(r.y1), x2=int(r.x2), y2=int(r.y2),
        ))
    return dict(by_frame)


def _build_person_colors(person_ids: list[str]) -> dict[str, tuple[int, int, int]]:
    """HSV palette with evenly spaced hues."""
    n = len(person_ids)
    colors = {}
    for i, pid in enumerate(sorted(person_ids)):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[pid] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def _load_tag_observations(hints_path: Path) -> dict[str, list[dict]]:
    """Build tracklet_id -> list of tag observations."""
    tags: dict[str, list[dict]] = defaultdict(list)
    if not hints_path.exists():
        return dict(tags)
    content = hints_path.read_text().strip()
    if not content:
        return dict(tags)

    # Also check tag_observations.jsonl (sibling file)
    tag_obs_path = hints_path.parent / "tag_observations.jsonl"
    if tag_obs_path.exists():
        for line in tag_obs_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            obs = json.loads(line)
            tid = obs.get("tracklet_id")
            if tid:
                tags[tid].append(obs)

    # Fallback: identity_hints also carry tracklet_id + tag info
    for line in content.split("\n"):
        if not line.strip():
            continue
        hint = json.loads(line)
        tid = hint.get("tracklet_id")
        anchor = hint.get("anchor_key", "")
        if tid and anchor.startswith("tag:") and tid not in tags:
            tags[tid].append({
                "tag_id": anchor.replace("tag:", ""),
                "tracklet_id": tid,
            })

    return dict(tags)


def _compute_envelope_plans(
    match_sessions_path: Path,
    person_tracks_df: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    stage_f_cfg: dict,
) -> tuple[list[EnvelopePlan], int]:
    """Compute crop envelopes for each match session using plan_crop_fixed_roi.

    Returns (plans, n_failed) where n_failed is count of sessions that raised
    CropPlanError.
    """
    if not match_sessions_path.exists():
        return [], 0

    content = match_sessions_path.read_text().strip()
    if not content:
        return [], 0

    sessions = [json.loads(line) for line in content.split("\n") if line.strip()]
    plans = []
    n_failed = 0

    for s in sessions:
        try:
            plan = plan_crop_fixed_roi(
                tracks_df=person_tracks_df,
                person_id_a=s["person_id_a"],
                person_id_b=s["person_id_b"],
                start_frame=s["start_frame"],
                end_frame=s["end_frame"],
                frame_width=frame_width,
                frame_height=frame_height,
                padding_px=stage_f_cfg.get("padding_px", 80),
                low_quantile=stage_f_cfg.get("low_quantile", 0.05),
                high_quantile=stage_f_cfg.get("high_quantile", 0.95),
                min_crop_width=stage_f_cfg.get("min_crop_width", 160),
                min_crop_height=stage_f_cfg.get("min_crop_height", 160),
            )
            plans.append(EnvelopePlan(
                start_frame=s["start_frame"],
                end_frame=s["end_frame"],
                x1=plan.x,
                y1=plan.y,
                x2=plan.x + plan.width,
                y2=plan.y + plan.height,
                person_id_a=s["person_id_a"],
                person_id_b=s["person_id_b"],
            ))
        except (CropPlanError, Exception) as e:
            logger.warning("Envelope failed for %s vs %s (frames %d-%d): %s",
                           s["person_id_a"], s["person_id_b"],
                           s["start_frame"], s["end_frame"], e)
            n_failed += 1

    return plans, n_failed


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed rectangle by iterating dash segments along each edge."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        dx = ex - sx
        dy = ey - sy
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        step = dash_len + gap_len
        for offset in range(0, length, step):
            seg_start = offset
            seg_end = min(offset + dash_len, length)
            if dx != 0:
                s = (sx + int(dx * seg_start / length), sy)
                e = (sx + int(dx * seg_end / length), sy)
            else:
                s = (sx, sy + int(dy * seg_start / length))
                e = (sx, sy + int(dy * seg_end / length))
            cv2.line(img, s, e, color, thickness)


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

def render_clip(
    manifest: ModelManifest,
    export: ExportEntry,
) -> dict | None:
    """Render match preview mp4 for one clip. Returns render_manifest dict."""
    cam = export.camera_id
    model_id = manifest.model_id

    paths = _resolve_pipeline_paths(manifest, export)
    source_video = _find_source_video(export, paths["clip_dir"])
    if source_video is None:
        logger.error("No source video for %s", cam)
        return None

    # Load data
    det_df = pd.read_parquet(paths["detections"])
    pt_df = pd.read_parquet(paths["person_tracks"])
    det_to_person: dict[str, str] = dict(zip(pt_df.detection_id, pt_df.person_id))
    tracklet_to_tags = _load_tag_observations(paths["identity_hints"])
    stage_f_cfg = _load_stage_f_config()

    # Per-frame timestamp lookup from the parquet (TIMING-PRINCIPLE-1: read time,
    # don't convert). Stage A writes timestamp_ms for every detection; we take the
    # first value per frame (all detections on the same frame share the same PTS).
    frame_to_ts_ms: dict[int, int] = {}
    if "timestamp_ms" in det_df.columns:
        for fi, ts in zip(det_df["frame_index"], det_df["timestamp_ms"]):
            fi_int = int(fi)
            if fi_int not in frame_to_ts_ms and pd.notna(ts):
                frame_to_ts_ms[fi_int] = int(ts)

    # Pre-group detections by frame
    det_by_frame = _preload_detections(det_df)

    # Person color palette
    person_ids = sorted(pt_df.person_id.unique())
    person_colors = _build_person_colors(person_ids)

    # Open source video
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        logger.error("Cannot open %s", source_video)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preview playback rate from sidecar nominal_dt_s. This is deliberately CFR
    # via cv2.VideoWriter — acceptable for a diagnostic preview instrument, not
    # athlete-facing output. See Piece 12 for the VFR fix on the export path.
    try:
        from bjj_pipeline.contracts.f0_sidecar import load_sidecar
        sidecar = load_sidecar(source_video)
        preview_fps = 1.0 / sidecar.nominal_dt_s
    except Exception:
        preview_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    # Compute envelopes
    envelope_plans, n_envelope_failed = _compute_envelope_plans(
        paths["match_sessions"], pt_df, frame_width, frame_height, stage_f_cfg
    )

    # Count match sessions
    n_sessions = 0
    if paths["match_sessions"].exists():
        content = paths["match_sessions"].read_text().strip()
        if content:
            n_sessions = len(content.split("\n"))

    # Output path
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    out_dir = EVAL_DIR / model_id / cam
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "match_preview.mp4"

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, preview_fps, (frame_width, frame_height))
    if not writer.isOpened():
        logger.error("Cannot open video writer for %s", out_path)
        cap.release()
        return None

    # Rendering stats
    total_det_count = 0
    total_kept_count = 0
    t_start = time.time()

    logger.info("Rendering %s: %d frames, %d person_ids, %d envelopes, %d tagged tracklets",
                cam, total_frames, len(person_ids), len(envelope_plans), len(tracklet_to_tags))

    for fi in range(total_frames):
        ret, img = cap.read()
        if not ret:
            break

        frame_dets = det_by_frame.get(fi, [])
        n_det = len(frame_dets)
        n_kept = 0

        # Layer 1: All Stage A detections (grey, 1px)
        for d in frame_dets:
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), (128, 128, 128), 1)

        # Layer 2: Stage D-accepted detections (colored, 2px)
        for d in frame_dets:
            pid = det_to_person.get(d.detection_id)
            if pid is not None:
                n_kept += 1
                color = person_colors[pid]
                cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(img, pid, (d.x1, d.y1 - 4), FONT, 0.35, color, 1)

        # Layer 3: Stage E match envelopes (orange dashed, 2px)
        active_matches = 0
        for ep in envelope_plans:
            if ep.start_frame <= fi <= ep.end_frame:
                active_matches += 1
                _draw_dashed_rect(img, (ep.x1, ep.y1), (ep.x2, ep.y2), (0, 165, 255), 2)
                cv2.putText(img, f"match: {ep.person_id_a} vs {ep.person_id_b}",
                            (ep.x1, ep.y1 - 4), FONT, 0.35, (0, 165, 255), 1)

        # Layer 4: Tag observation icons (yellow, 16x16)
        for d in frame_dets:
            tags = tracklet_to_tags.get(d.tracklet_id, [])
            if tags:
                tag_id = tags[0].get("tag_id", "?")
                tx = d.x2 - 18
                ty = d.y1 + 2
                cv2.rectangle(img, (tx, ty), (tx + 16, ty + 16), (0, 255, 255), -1)
                cv2.putText(img, f"T:{tag_id}", (tx + 1, ty + 12), FONT, 0.3, (0, 0, 0), 1)

        # Frame metadata overlay
        n_dropped = n_det - n_kept
        ts_ms = frame_to_ts_ms.get(fi)
        ts_label = f"t={ts_ms}ms" if ts_ms is not None else "t=N/A"
        lines = [
            f"{cam} | frame {fi}/{total_frames} | {ts_label}",
            f"detections: {n_det} | accepted: {n_kept} | dropped: {n_dropped}",
            f"active matches: {active_matches}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 18 + i * 16), FONT, 0.4, (255, 255, 255), 1)

        writer.write(img)
        total_det_count += n_det
        total_kept_count += n_kept

    cap.release()
    writer.release()

    render_time = time.time() - t_start
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    logger.info("%s: rendered in %.1fs, %.1f MB", cam, render_time, file_size_mb)

    # Render manifest
    render_manifest = {
        "camera_id": cam,
        "model_id": model_id,
        "clip_id": clip_id,
        "total_frames": total_frames,
        "unique_person_ids": len(person_ids),
        "total_detections": total_det_count,
        "kept_detections": total_kept_count,
        "dropped_detections": total_det_count - total_kept_count,
        "match_sessions": n_sessions,
        "match_sessions_envelope_rendered": len(envelope_plans),
        "match_sessions_envelope_failed": n_envelope_failed,
        "tagged_tracklets": len(tracklet_to_tags),
        "file_size_mb": round(file_size_mb, 1),
        "render_time_seconds": round(render_time, 1),
    }

    (out_dir / "render_manifest.json").write_text(json.dumps(render_manifest, indent=2))
    return render_manifest


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_all(manifest_path: Path) -> None:
    """Render match preview mp4 for all cameras in the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest = load_manifest(manifest_path)

    # Filter to exports with val splits (train-only entries skip evaluation)
    eval_exports = [e for e in manifest.training_data if e.splits.val is not None]

    for export in eval_exports:
        try:
            result = render_clip(manifest, export)
            if result:
                logger.info("Done: %s (%.1f MB)", export.camera_id, result["file_size_mb"])
        except Exception as e:
            logger.error("Failed %s: %s", export.camera_id, e, exc_info=True)
