"""Stage A topology census — greedy per-GT matching + classification.

Classifies each GT-person-frame into: tight_match, pair_box, split, miss.
Produces gt_signal_trace_stage_a.parquet and topology_summary.json.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from pipeline_validation.common.gt_loader import load_gt_for_split
from pipeline_validation.common.manifest import (
    enumerate_annotated_frames,
    load_manifest,
)
from pipeline_validation.common.schemas import ExportEntry, ModelManifest
from pipeline_validation.signal_trace.greedy_matcher import greedy_match

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
EVAL_DIR = OUTPUTS_DIR / "_eval" / "signal_trace"


def _load_gt_all_annotated(
    zip_path: Path, export: ExportEntry,
) -> dict[int, list]:
    """Load GT boxes for ALL annotated frames (train + val merged)."""
    gt_train = load_gt_for_split(zip_path, export, "train")
    if export.splits.val is not None:
        gt_val = load_gt_for_split(zip_path, export, "val")
        gt_train.update(gt_val)
    return gt_train


def _resolve_detections_path(
    manifest: ModelManifest, export: ExportEntry, gym_id: str,
) -> Path | None:
    """Find detections.parquet for a camera's clip."""
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id
    pattern = f"{gym_id}/{cam}/**/{clip_id}/stage_A/detections.parquet"
    matches = list(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


def run_census(
    manifest: ModelManifest,
    export: ExportEntry,
    gym_id: str,
    iou_threshold: float = 0.3,
) -> tuple[pd.DataFrame, dict]:
    """Run Stage A topology census for one camera.

    Returns (trace_df, summary_dict).
    """
    cam = export.camera_id
    zip_path = TRAINING_DATA_DIR / export.export

    # Load GT
    gt_by_frame = _load_gt_all_annotated(zip_path, export)
    annotated_frames = sorted(enumerate_annotated_frames(export))
    logger.info("%s: %d annotated frames, GT loaded for %d",
                cam, len(annotated_frames), len(gt_by_frame))

    # Load detections
    det_path = _resolve_detections_path(manifest, export, gym_id)
    if det_path is None:
        raise FileNotFoundError(
            f"detections.parquet not found for {cam} under gym_id={gym_id}"
        )
    det_df = pd.read_parquet(det_path)
    logger.info("%s: %d detections loaded", cam, len(det_df))

    # Process each annotated frame
    records: list[dict] = []

    for fi in annotated_frames:
        gt_boxes_raw = gt_by_frame.get(fi, [])
        if not gt_boxes_raw:
            continue

        # GT boxes as (x1, y1, x2, y2) tuples
        gt_tuples = [(b.x1, b.y1, b.x2, b.y2) for b in gt_boxes_raw]
        gt_track_ids = [b.track_id for b in gt_boxes_raw]

        # Detection boxes at this frame
        frame_dets = det_df[det_df.frame_index == fi]
        det_tuples = list(zip(
            frame_dets.x1.values,
            frame_dets.y1.values,
            frame_dets.x2.values,
            frame_dets.y2.values,
        ))
        det_ids = frame_dets.detection_id.values.tolist()
        det_tracklet_ids = frame_dets.tracklet_id.values.tolist()

        # Greedy match
        matches = greedy_match(gt_tuples, det_tuples, iou_threshold)

        # Build gt_idx -> list of (det_idx, iou) for split detection
        gt_to_dets: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for gi, di, iou in matches:
            gt_to_dets[gi].append((di, iou))

        # Count how many GT people matched each detection
        det_to_gt_count: dict[int, int] = defaultdict(int)
        # Map gt_idx -> best det_idx (highest IoU)
        gt_best_det: dict[int, tuple[int, float]] = {}
        for gi, di, iou in matches:
            det_to_gt_count[di] += 1
            if gi not in gt_best_det or iou > gt_best_det[gi][1]:
                gt_best_det[gi] = (di, iou)

        # Classify each GT person
        for gi, gt_box in enumerate(gt_boxes_raw):
            if gi not in gt_best_det:
                # Miss
                records.append({
                    "gt_track_id": gt_track_ids[gi],
                    "frame_index": fi,
                    "classification": "miss",
                    "detection_id": None,
                    "tracklet_id": None,
                    "iou": None,
                    "n_gt_sharing_detection": 0,
                    "gt_x1": gt_box.x1, "gt_y1": gt_box.y1,
                    "gt_x2": gt_box.x2, "gt_y2": gt_box.y2,
                    "det_x1": None, "det_y1": None,
                    "det_x2": None, "det_y2": None,
                })
                continue

            best_di, best_iou = gt_best_det[gi]
            n_sharing = det_to_gt_count[best_di]
            n_dets_matched = len(gt_to_dets[gi])

            if n_dets_matched >= 2:
                classification = "split"
            elif n_sharing >= 2:
                classification = "pair_box"
            else:
                classification = "tight_match"

            det_row_idx = list(frame_dets.index)[best_di]
            records.append({
                "gt_track_id": gt_track_ids[gi],
                "frame_index": fi,
                "classification": classification,
                "detection_id": det_ids[best_di],
                "tracklet_id": det_tracklet_ids[best_di],
                "iou": round(best_iou, 4),
                "n_gt_sharing_detection": n_sharing,
                "gt_x1": gt_box.x1, "gt_y1": gt_box.y1,
                "gt_x2": gt_box.x2, "gt_y2": gt_box.y2,
                "det_x1": float(frame_dets.at[det_row_idx, "x1"]),
                "det_y1": float(frame_dets.at[det_row_idx, "y1"]),
                "det_x2": float(frame_dets.at[det_row_idx, "x2"]),
                "det_y2": float(frame_dets.at[det_row_idx, "y2"]),
            })

    trace_df = pd.DataFrame(records)

    # Build summary
    total = len(trace_df)
    counts = trace_df.classification.value_counts().to_dict()
    summary = {
        "camera_id": cam,
        "total_gt_person_frames": total,
        "tight_match": {
            "count": counts.get("tight_match", 0),
            "pct": round(counts.get("tight_match", 0) / total, 4) if total else 0,
        },
        "pair_box": {
            "count": counts.get("pair_box", 0),
            "pct": round(counts.get("pair_box", 0) / total, 4) if total else 0,
        },
        "split": {
            "count": counts.get("split", 0),
            "pct": round(counts.get("split", 0) / total, 4) if total else 0,
        },
        "miss": {
            "count": counts.get("miss", 0),
            "pct": round(counts.get("miss", 0) / total, 4) if total else 0,
        },
        "n_unique_gt_tracks": int(trace_df.gt_track_id.nunique()),
        "n_unique_detections_shared": int(
            trace_df[trace_df.n_gt_sharing_detection >= 2]
            .detection_id.nunique()
        ) if total else 0,
    }

    # Per-GT-track breakdown
    per_track: dict[str, dict[str, int]] = {}
    for tid, grp in trace_df.groupby("gt_track_id"):
        tc = grp.classification.value_counts().to_dict()
        per_track[f"gt_track_{tid}"] = {
            "tight": tc.get("tight_match", 0),
            "pair_box": tc.get("pair_box", 0),
            "split": tc.get("split", 0),
            "miss": tc.get("miss", 0),
        }
    summary["per_gt_track_summary"] = per_track

    return trace_df, summary


def write_census_artifacts(
    model_id: str,
    camera_id: str,
    trace_df: pd.DataFrame,
    summary: dict,
) -> Path:
    """Write parquet, JSON summary, and markdown aggregate."""
    out_dir = EVAL_DIR / model_id / camera_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet
    trace_df.to_parquet(out_dir / "gt_signal_trace_stage_a.parquet", index=False)

    # JSON summary
    with open(out_dir / "topology_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown
    total = summary["total_gt_person_frames"]
    md_lines = [
        f"# Signal Trace Stage A Census: {model_id} / {camera_id}",
        "",
        f"Total GT-person-frames: {total}",
        "",
        "| Classification | Count | Pct |",
        "|---|---:|---:|",
    ]
    for cls in ("tight_match", "pair_box", "split", "miss"):
        c = summary[cls]
        md_lines.append(f"| {cls} | {c['count']} | {c['pct']:.1%} |")

    md_lines.extend([
        "",
        f"Unique GT tracks: {summary['n_unique_gt_tracks']}",
        f"Unique detections shared by 2+ GT: {summary['n_unique_detections_shared']}",
        "",
        "## Per-GT-Track Breakdown",
        "",
        "| GT Track | tight | pair_box | split | miss | total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for tid, tc in sorted(summary["per_gt_track_summary"].items()):
        row_total = tc["tight"] + tc["pair_box"] + tc["split"] + tc["miss"]
        md_lines.append(
            f"| {tid} | {tc['tight']} | {tc['pair_box']} | "
            f"{tc['split']} | {tc['miss']} | {row_total} |"
        )

    (out_dir / "_aggregate.md").write_text("\n".join(md_lines) + "\n")
    return out_dir
