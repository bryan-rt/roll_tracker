"""Stage A detection recall/precision evaluation.

Compares Stage A detections.parquet against CVAT GT annotations using
Hungarian IoU matching. Reports per-camera and aggregate metrics, split
by in-distribution (train) and held-out (val) frames.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from pipeline_validation.common.gt_loader import load_gt_for_split
from pipeline_validation.common.manifest import (
    enumerate_split_frames,
    load_manifest,
)
from pipeline_validation.common.matching import hungarian_match, iou_matrix
from pipeline_validation.common.schemas import ExportEntry, GTBox, ModelManifest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = REPO_ROOT / "outputs" / "_eval" / "stage_a"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredBox:
    detection_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class FrameResult:
    frame_index: int
    gt_boxes: list[GTBox]
    pred_boxes: list[PredBox]
    matches_05: list[tuple[int, int]]  # (gt_idx, pred_idx) at IoU >= 0.5
    matches_07: list[tuple[int, int]]
    matches_09: list[tuple[int, int]]
    iou_mat: np.ndarray  # raw IoU matrix for merge/split analysis


@dataclass
class SplitMetrics:
    split: str
    n_frames: int
    # Per-threshold metrics
    recall: dict[str, float] = field(default_factory=dict)     # "@0.5", "@0.7", "@0.9"
    precision: dict[str, float] = field(default_factory=dict)
    mean_iou: dict[str, float] = field(default_factory=dict)
    # IoU histogram bins at @0.5
    iou_histogram: dict[str, int] = field(default_factory=dict)
    # Box count analysis
    box_count_analysis: list[dict] = field(default_factory=list)
    merge_rate: float = 0.0
    split_rate: float = 0.0
    # Frame coverage
    zero_det_frames: int = 0
    zero_det_fraction: float = 0.0
    # Bootstrap CIs (val only)
    bootstrap_ci: dict[str, dict] | None = None
    # Total counts for aggregate
    total_gt: int = 0
    total_pred: int = 0
    total_matched_05: int = 0


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def _resolve_gym_id(gym_id_override: str | None) -> str:
    """Find the gym_id directory under outputs/. Error if ambiguous."""
    if gym_id_override:
        return gym_id_override

    candidates = [
        d.name for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and len(d.name) > 10  # UUID-like
    ]
    if len(candidates) == 0:
        raise FileNotFoundError("No gym_id directories found under outputs/")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple gym_id directories found: {candidates}. "
            f"Use --gym-id to specify which one."
        )
    return candidates[0]


def _find_parquet_path(
    export: ExportEntry, gym_id: str
) -> Path | None:
    """Locate detections.parquet for a clip matching this export's source video."""
    clip_id = export.source_video.replace(".mp4", "")
    cam = export.camera_id
    # Search under outputs/{gym_id}/{cam}/**/clip_id/stage_A/detections.parquet
    pattern = f"{gym_id}/{cam}/**/{clip_id}/stage_A/detections.parquet"
    matches = list(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


def _find_source_video(
    export: ExportEntry, parquet_path: Path | None
) -> Path | None:
    """Find the source video for failure gallery rendering."""
    # Option 1: manifest has explicit source_video_path
    if export.source_video_path:
        p = REPO_ROOT / export.source_video_path
        if p.exists():
            return p

    # Option 2: derive from clip_manifest.json alongside parquet
    if parquet_path:
        clip_dir = parquet_path.parent.parent
        manifest_file = clip_dir / "clip_manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                cm = json.load(f)
            video_path = REPO_ROOT / cm.get("input_video_path", "")
            if video_path.exists():
                return video_path

    return None


def load_preds_from_parquet(
    parquet_path: Path, frame_indices: set[int]
) -> dict[int, list[PredBox]]:
    """Load predictions from detections.parquet for specified frames."""
    df = pd.read_parquet(parquet_path)
    df = df[df.frame_index.isin(frame_indices)]

    preds: dict[int, list[PredBox]] = {fi: [] for fi in frame_indices}
    for _, row in df.iterrows():
        fi = int(row.frame_index)
        preds[fi].append(PredBox(
            detection_id=str(row.detection_id),
            x1=float(row.x1), y1=float(row.y1),
            x2=float(row.x2), y2=float(row.y2),
            confidence=float(row.confidence),
        ))
    return preds


def load_preds_from_model(
    model_path: Path,
    video_path: Path,
    frame_indices: set[int],
    conf: float = 0.45,
) -> dict[int, list[PredBox]]:
    """Run direct inference on specified frames of the video."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    preds: dict[int, list[PredBox]] = {fi: [] for fi in frame_indices}
    sorted_frames = sorted(frame_indices)

    for fi in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame %d from %s", fi, video_path)
            continue

        results = model(frame, conf=conf, verbose=False)
        r = results[0]
        for i, (xyxy, c) in enumerate(zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        )):
            preds[fi].append(PredBox(
                detection_id=f"d{fi:06d}_{i}",
                x1=float(xyxy[0]), y1=float(xyxy[1]),
                x2=float(xyxy[2]), y2=float(xyxy[3]),
                confidence=float(c),
            ))

    cap.release()
    return preds


# ---------------------------------------------------------------------------
# Matching and metrics
# ---------------------------------------------------------------------------

def _match_frame(
    gt_boxes: list[GTBox],
    pred_boxes: list[PredBox],
) -> FrameResult:
    """Run IoU + Hungarian matching at thresholds {0.5, 0.7, 0.9}."""
    gt_arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in gt_boxes]) if gt_boxes else np.zeros((0, 4))
    pred_arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in pred_boxes]) if pred_boxes else np.zeros((0, 4))

    iou_mat = iou_matrix(gt_arr, pred_arr)

    return FrameResult(
        frame_index=0,  # set by caller
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        matches_05=hungarian_match(iou_mat, 0.5),
        matches_07=hungarian_match(iou_mat, 0.7),
        matches_09=hungarian_match(iou_mat, 0.9),
        iou_mat=iou_mat,
    )


def _compute_merge_split(frame_results: list[FrameResult]) -> tuple[float, float]:
    """Compute merge and split rates across frames.

    Merge: fraction of multi-person frames where >=1 pred box overlaps >=2 GT (IoU>=0.3).
    Split: fraction of frames where >=1 GT box is overlapped by >=2 pred (IoU>=0.3).
    """
    multi_person_frames = 0
    merge_frames = 0
    total_frames = 0
    split_frames = 0

    for fr in frame_results:
        total_frames += 1
        n_gt = len(fr.gt_boxes)
        n_pred = len(fr.pred_boxes)

        if n_gt >= 2:
            multi_person_frames += 1

        if fr.iou_mat.size == 0:
            continue

        # Merge: any pred column has IoU >= 0.3 with >= 2 GT rows
        if n_gt >= 2 and n_pred >= 1:
            pred_overlap_counts = np.sum(fr.iou_mat >= 0.3, axis=0)  # per pred
            if np.any(pred_overlap_counts >= 2):
                merge_frames += 1

        # Split: any GT row has IoU >= 0.3 with >= 2 pred columns
        if n_pred >= 2 and n_gt >= 1:
            gt_overlap_counts = np.sum(fr.iou_mat >= 0.3, axis=1)  # per GT
            if np.any(gt_overlap_counts >= 2):
                split_frames += 1

    merge_rate = merge_frames / multi_person_frames if multi_person_frames > 0 else 0.0
    split_rate = split_frames / total_frames if total_frames > 0 else 0.0

    return merge_rate, split_rate


def _compute_split_metrics(
    frame_results: list[FrameResult],
    split_name: str,
    do_bootstrap: bool = False,
) -> SplitMetrics:
    """Compute aggregate metrics for a split from per-frame match results."""
    metrics = SplitMetrics(split=split_name, n_frames=len(frame_results))

    total_gt = sum(len(fr.gt_boxes) for fr in frame_results)
    total_pred = sum(len(fr.pred_boxes) for fr in frame_results)
    metrics.total_gt = total_gt
    metrics.total_pred = total_pred

    # Per-threshold metrics
    for threshold, key, matches_attr in [
        (0.5, "@0.5", "matches_05"),
        (0.7, "@0.7", "matches_07"),
        (0.9, "@0.9", "matches_09"),
    ]:
        total_matched = sum(len(getattr(fr, matches_attr)) for fr in frame_results)
        if key == "@0.5":
            metrics.total_matched_05 = total_matched
        metrics.recall[key] = total_matched / total_gt if total_gt > 0 else 0.0
        metrics.precision[key] = total_matched / total_pred if total_pred > 0 else 0.0

        # Mean IoU on matched pairs
        ious = []
        for fr in frame_results:
            for gi, pi in getattr(fr, matches_attr):
                ious.append(fr.iou_mat[gi, pi])
        metrics.mean_iou[key] = float(np.mean(ious)) if ious else 0.0

    # IoU histogram at @0.5
    matched_ious = []
    for fr in frame_results:
        for gi, pi in fr.matches_05:
            matched_ious.append(fr.iou_mat[gi, pi])

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        label = f"[{lo:.1f}-{hi:.1f})"
        metrics.iou_histogram[label] = sum(1 for v in matched_ious if lo <= v < hi)

    # Box count analysis
    gt_count_buckets = {0: [], 1: [], 2: [], "3+": []}
    for fr in frame_results:
        n_gt = len(fr.gt_boxes)
        n_pred = len(fr.pred_boxes)
        bucket = n_gt if n_gt <= 2 else "3+"
        gt_count_buckets[bucket].append(n_pred)

    for bucket in [0, 1, 2, "3+"]:
        preds = gt_count_buckets[bucket]
        metrics.box_count_analysis.append({
            "gt_count": str(bucket),
            "n_frames": len(preds),
            "mean_pred": float(np.mean(preds)) if preds else 0.0,
        })

    # Merge and split rates
    metrics.merge_rate, metrics.split_rate = _compute_merge_split(frame_results)

    # Frame coverage
    metrics.zero_det_frames = sum(1 for fr in frame_results if len(fr.pred_boxes) == 0)
    metrics.zero_det_fraction = metrics.zero_det_frames / len(frame_results) if frame_results else 0.0

    # Bootstrap CIs (val only)
    if do_bootstrap and frame_results:
        metrics.bootstrap_ci = _bootstrap_ci(frame_results)

    return metrics


def _bootstrap_ci(
    frame_results: list[FrameResult],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict[str, dict]:
    """Bootstrap 95% CIs on recall@0.5, precision@0.5, mean_iou, frame-level resampling."""
    rng = np.random.RandomState(42)
    n = len(frame_results)

    recall_samples = []
    precision_samples = []
    iou_samples = []

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        sampled = [frame_results[i] for i in indices]

        total_gt = sum(len(fr.gt_boxes) for fr in sampled)
        total_pred = sum(len(fr.pred_boxes) for fr in sampled)
        total_matched = sum(len(fr.matches_05) for fr in sampled)

        recall_samples.append(total_matched / total_gt if total_gt > 0 else 0.0)
        precision_samples.append(total_matched / total_pred if total_pred > 0 else 0.0)

        ious = []
        for fr in sampled:
            for gi, pi in fr.matches_05:
                ious.append(fr.iou_mat[gi, pi])
        iou_samples.append(float(np.mean(ious)) if ious else 0.0)

    alpha = (1 - ci) / 2
    lo_pct = alpha * 100
    hi_pct = (1 - alpha) * 100

    return {
        "recall@0.5": {
            "mean": float(np.mean(recall_samples)),
            "ci_lo": float(np.percentile(recall_samples, lo_pct)),
            "ci_hi": float(np.percentile(recall_samples, hi_pct)),
        },
        "precision@0.5": {
            "mean": float(np.mean(precision_samples)),
            "ci_lo": float(np.percentile(precision_samples, lo_pct)),
            "ci_hi": float(np.percentile(precision_samples, hi_pct)),
        },
        "mean_iou": {
            "mean": float(np.mean(iou_samples)),
            "ci_lo": float(np.percentile(iou_samples, lo_pct)),
            "ci_hi": float(np.percentile(iou_samples, hi_pct)),
        },
    }


# ---------------------------------------------------------------------------
# Per-frame match persistence
# ---------------------------------------------------------------------------

def _build_match_records(
    frame_results: list[FrameResult],
    model_id: str,
    camera_id: str,
    split_name: str,
) -> list[dict]:
    """Build flat match records for parquet persistence."""
    records = []
    for fr in frame_results:
        matched_gt = set()
        matched_pred = set()

        for gi, pi in fr.matches_05:
            matched_gt.add(gi)
            matched_pred.add(pi)
            gt = fr.gt_boxes[gi]
            pred = fr.pred_boxes[pi]
            records.append({
                "model_id": model_id,
                "camera_id": camera_id,
                "split": split_name,
                "frame_index": fr.frame_index,
                "gt_track_id": gt.track_id,
                "gt_x1": gt.x1, "gt_y1": gt.y1, "gt_x2": gt.x2, "gt_y2": gt.y2,
                "pred_detection_id": pred.detection_id,
                "pred_x1": pred.x1, "pred_y1": pred.y1,
                "pred_x2": pred.x2, "pred_y2": pred.y2,
                "iou": float(fr.iou_mat[gi, pi]),
                "match_status": "matched",
            })

        for gi, gt in enumerate(fr.gt_boxes):
            if gi not in matched_gt:
                records.append({
                    "model_id": model_id,
                    "camera_id": camera_id,
                    "split": split_name,
                    "frame_index": fr.frame_index,
                    "gt_track_id": gt.track_id,
                    "gt_x1": gt.x1, "gt_y1": gt.y1, "gt_x2": gt.x2, "gt_y2": gt.y2,
                    "pred_detection_id": None,
                    "pred_x1": None, "pred_y1": None,
                    "pred_x2": None, "pred_y2": None,
                    "iou": 0.0,
                    "match_status": "unmatched_gt",
                })

        for pi, pred in enumerate(fr.pred_boxes):
            if pi not in matched_pred:
                records.append({
                    "model_id": model_id,
                    "camera_id": camera_id,
                    "split": split_name,
                    "frame_index": fr.frame_index,
                    "gt_track_id": None,
                    "gt_x1": None, "gt_y1": None, "gt_x2": None, "gt_y2": None,
                    "pred_detection_id": pred.detection_id,
                    "pred_x1": pred.x1, "pred_y1": pred.y1,
                    "pred_x2": pred.x2, "pred_y2": pred.y2,
                    "iou": 0.0,
                    "match_status": "unmatched_pred",
                })

    return records


# ---------------------------------------------------------------------------
# Failure gallery
# ---------------------------------------------------------------------------

def _render_failure_gallery(
    frame_results: list[FrameResult],
    source_video: Path | None,
    out_dir: Path,
    max_per_category: int = 10,
) -> None:
    """Render failure gallery images."""
    if source_video is None or not source_video.exists():
        logger.warning("No source video for failure gallery, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Rank frames by failure type
    fp_frames = sorted(frame_results, key=lambda fr: sum(
        1 for pi in range(len(fr.pred_boxes))
        if pi not in {p for _, p in fr.matches_05}
    ), reverse=True)

    fn_frames = sorted(frame_results, key=lambda fr: sum(
        1 for gi in range(len(fr.gt_boxes))
        if gi not in {g for g, _ in fr.matches_05}
    ), reverse=True)

    # Merge frames: pred overlaps >= 2 GT at IoU >= 0.3
    merge_frames = []
    for fr in frame_results:
        if fr.iou_mat.size == 0 or len(fr.gt_boxes) < 2:
            continue
        pred_overlap = np.sum(fr.iou_mat >= 0.3, axis=0)
        merge_count = int(np.sum(pred_overlap >= 2))
        if merge_count > 0:
            merge_frames.append((merge_count, fr))
    merge_frames.sort(key=lambda x: x[0], reverse=True)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        logger.warning("Cannot open %s for failure gallery", source_video)
        return

    def _draw_frame(fr: FrameResult, category: str) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_index)
        ret, img = cap.read()
        if not ret:
            return None

        matched_gt = {g for g, _ in fr.matches_05}
        matched_pred = {p for _, p in fr.matches_05}

        if category == "false_positive":
            for gi, gt in enumerate(fr.gt_boxes):
                cv2.rectangle(img, (int(gt.x1), int(gt.y1)), (int(gt.x2), int(gt.y2)), (0, 255, 0), 2)
            for pi, pred in enumerate(fr.pred_boxes):
                if pi not in matched_pred:
                    cv2.rectangle(img, (int(pred.x1), int(pred.y1)), (int(pred.x2), int(pred.y2)), (0, 0, 255), 2)

        elif category == "missed_detection":
            for pi, pred in enumerate(fr.pred_boxes):
                cv2.rectangle(img, (int(pred.x1), int(pred.y1)), (int(pred.x2), int(pred.y2)), (255, 255, 255), 1)
            for gi, gt in enumerate(fr.gt_boxes):
                if gi not in matched_gt:
                    cv2.rectangle(img, (int(gt.x1), int(gt.y1)), (int(gt.x2), int(gt.y2)), (0, 255, 255), 2)

        elif category == "merge":
            for gi, gt in enumerate(fr.gt_boxes):
                cv2.rectangle(img, (int(gt.x1), int(gt.y1)), (int(gt.x2), int(gt.y2)), (0, 255, 0), 2)
            for pi, pred in enumerate(fr.pred_boxes):
                if fr.iou_mat.shape[1] > pi:
                    overlap_count = int(np.sum(fr.iou_mat[:, pi] >= 0.3))
                    if overlap_count >= 2:
                        cv2.rectangle(img, (int(pred.x1), int(pred.y1)), (int(pred.x2), int(pred.y2)), (0, 165, 255), 2)

        return img

    # Render each category
    for i, fr in enumerate(fp_frames[:max_per_category]):
        n_fp = sum(1 for pi in range(len(fr.pred_boxes)) if pi not in {p for _, p in fr.matches_05})
        if n_fp == 0:
            break
        img = _draw_frame(fr, "false_positive")
        if img is not None:
            cv2.imwrite(str(out_dir / f"false_positive_{i:02d}_f{fr.frame_index}.jpg"), img)

    for i, fr in enumerate(fn_frames[:max_per_category]):
        n_fn = sum(1 for gi in range(len(fr.gt_boxes)) if gi not in {g for g, _ in fr.matches_05})
        if n_fn == 0:
            break
        img = _draw_frame(fr, "missed_detection")
        if img is not None:
            cv2.imwrite(str(out_dir / f"missed_detection_{i:02d}_f{fr.frame_index}.jpg"), img)

    for i, (_, fr) in enumerate(merge_frames[:max_per_category]):
        img = _draw_frame(fr, "merge")
        if img is not None:
            cv2.imwrite(str(out_dir / f"merge_{i:02d}_f{fr.frame_index}.jpg"), img)

    cap.release()
    logger.info("Failure gallery: %s", out_dir)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _format_split_report(metrics: SplitMetrics, header: str) -> str:
    """Format a split's metrics as markdown."""
    lines = [f"## {header}\n"]

    # Main metrics table
    lines.append("| Metric | @0.5 | @0.7 | @0.9 |")
    lines.append("|--------|------|------|------|")
    for name, data in [("Recall", metrics.recall), ("Precision", metrics.precision), ("Mean IoU", metrics.mean_iou)]:
        lines.append(f"| {name} | {data.get('@0.5', 0):.3f} | {data.get('@0.7', 0):.3f} | {data.get('@0.9', 0):.3f} |")

    lines.append(f"\nTotal GT boxes: {metrics.total_gt} | Total predictions: {metrics.total_pred} | Matched @0.5: {metrics.total_matched_05}")

    # IoU histogram
    lines.append("\n### IoU Histogram (matched pairs @0.5)\n")
    for label, count in metrics.iou_histogram.items():
        lines.append(f"  {label}: {count}")

    # Box count analysis
    lines.append("\n### Box Count Analysis\n")
    for bca in metrics.box_count_analysis:
        lines.append(f"  GT={bca['gt_count']}: {bca['n_frames']} frames, mean pred={bca['mean_pred']:.1f}")
    lines.append(f"  Merge rate: {metrics.merge_rate:.1%} of multi-person frames had >=1 pred box overlapping >=2 GT boxes (IoU >= 0.3)")
    lines.append(f"  Split rate: {metrics.split_rate:.1%} of frames had >=1 GT box overlapped by >=2 pred boxes (IoU >= 0.3)")

    # Frame coverage
    lines.append(f"\n### Frame Coverage\n")
    lines.append(f"  Zero-detection frames: {metrics.zero_det_frames} / {metrics.n_frames} ({metrics.zero_det_fraction:.1%})")

    # Bootstrap CIs
    if metrics.bootstrap_ci:
        lines.append("\n### Bootstrap 95% CIs (N=1000, frame-level)\n")
        for metric_name, ci in metrics.bootstrap_ci.items():
            lines.append(f"  {metric_name}: {ci['mean']:.3f} [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")

    return "\n".join(lines)


def write_clip_report(
    camera_id: str,
    model_id: str,
    source_type: str,
    train_metrics: SplitMetrics,
    val_metrics: SplitMetrics,
    out_dir: Path,
) -> None:
    """Write per-clip report as .md and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# Stage A Detection Evaluation: {camera_id}\n",
        f"Model: {model_id} | Source: {source_type}",
        f"Train frames: {train_metrics.n_frames} | Val frames: {val_metrics.n_frames}\n",
        _format_split_report(train_metrics, "In-Distribution (train split)"),
        "",
        _format_split_report(val_metrics, "Held-Out (val split)"),
    ]

    md_text = "\n".join(md_lines)
    (out_dir / "report.md").write_text(md_text)

    json_data = {
        "camera_id": camera_id,
        "model_id": model_id,
        "source_type": source_type,
        "train": _metrics_to_dict(train_metrics),
        "val": _metrics_to_dict(val_metrics),
    }
    (out_dir / "report.json").write_text(json.dumps(json_data, indent=2))


def _metrics_to_dict(m: SplitMetrics) -> dict:
    return {
        "split": m.split,
        "n_frames": m.n_frames,
        "recall": m.recall,
        "precision": m.precision,
        "mean_iou": m.mean_iou,
        "iou_histogram": m.iou_histogram,
        "box_count_analysis": m.box_count_analysis,
        "merge_rate": m.merge_rate,
        "split_rate": m.split_rate,
        "zero_det_frames": m.zero_det_frames,
        "zero_det_fraction": m.zero_det_fraction,
        "total_gt": m.total_gt,
        "total_pred": m.total_pred,
        "total_matched_05": m.total_matched_05,
        "bootstrap_ci": m.bootstrap_ci,
    }


# ---------------------------------------------------------------------------
# Cross-validation gate
# ---------------------------------------------------------------------------

def _check_parquet_staleness(
    parquet_path: Path,
    manifest: ModelManifest,
) -> bool:
    """Check if the parquet was generated before the model was trained.

    Returns True if stale (parquet predates model).
    """
    clip_dir = parquet_path.parent.parent
    cm_path = clip_dir / "clip_manifest.json"
    if not cm_path.exists():
        return False

    with open(cm_path) as f:
        cm = json.load(f)

    pipeline_ts = cm.get("created_at_ms", 0) / 1000
    model_path = REPO_ROOT / manifest.weights_path
    if model_path.exists():
        model_ts = model_path.stat().st_mtime
        return pipeline_ts < model_ts
    return False


def _cross_validate_fp7(
    manifest: ModelManifest,
    export: ExportEntry,
    gym_id: str,
) -> dict:
    """Run FP7oJQ via both parquet and direct inference; compare metrics.

    If the parquet was generated with a different (older) model, the gate
    is skipped and marked as 'stale_parquet'. All cameras should use
    direct inference in this case.
    """
    zip_path = TRAINING_DATA_DIR / export.export
    parquet_path = _find_parquet_path(export, gym_id)
    source_video = _find_source_video(export, parquet_path)

    if parquet_path is None or source_video is None:
        return {
            "passed": True,
            "skipped": True,
            "reason": "FP7oJQ parquet or source video not found; using direct inference only",
        }

    # Check if parquet was generated with a different model
    if _check_parquet_staleness(parquet_path, manifest):
        logger.warning(
            "FP7oJQ parquet predates model %s — generated with older model. "
            "Gate comparison not meaningful. Using direct inference for all cameras.",
            manifest.model_id,
        )
        return {
            "passed": True,
            "skipped": True,
            "reason": (
                "Parquet was generated before the current model was trained "
                "(pipeline ran with older model). Gate comparison skipped. "
                "All cameras evaluated via direct inference."
            ),
        }

    # Use val split for the gate (smaller, faster)
    val_frames = set(enumerate_split_frames(export, "val"))
    gt_val = load_gt_for_split(zip_path, export, "val")

    # Parquet path
    preds_parquet = load_preds_from_parquet(parquet_path, val_frames)
    fr_parquet = _match_all_frames(gt_val, preds_parquet)
    m_parquet = _compute_split_metrics(fr_parquet, "val")

    # Direct inference path
    model_path = REPO_ROOT / manifest.weights_path
    preds_direct = load_preds_from_model(model_path, source_video, val_frames)
    fr_direct = _match_all_frames(gt_val, preds_direct)
    m_direct = _compute_split_metrics(fr_direct, "val")

    gate = {
        "skipped": False,
        "recall_parquet": m_parquet.recall["@0.5"],
        "recall_direct": m_direct.recall["@0.5"],
        "precision_parquet": m_parquet.precision["@0.5"],
        "precision_direct": m_direct.precision["@0.5"],
        "mean_iou_parquet": m_parquet.mean_iou["@0.5"],
        "mean_iou_direct": m_direct.mean_iou["@0.5"],
    }

    gate["recall_diff"] = abs(gate["recall_parquet"] - gate["recall_direct"])
    gate["precision_diff"] = abs(gate["precision_parquet"] - gate["precision_direct"])
    gate["mean_iou_diff"] = abs(gate["mean_iou_parquet"] - gate["mean_iou_direct"])
    gate["passed"] = all(
        gate[f"{m}_diff"] <= 0.01
        for m in ("recall", "precision", "mean_iou")
    )

    return gate


def _match_all_frames(
    gt: dict[int, list[GTBox]],
    preds: dict[int, list[PredBox]],
) -> list[FrameResult]:
    """Match GT vs pred for every frame."""
    results = []
    for fi in sorted(gt.keys()):
        gt_boxes = gt[fi]
        pred_boxes = preds.get(fi, [])
        fr = _match_frame(gt_boxes, pred_boxes)
        fr.frame_index = fi
        results.append(fr)
    return results


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_all(
    manifest_path: Path,
    run_model: bool = False,
    gym_id: str | None = None,
) -> None:
    """Run evaluation for all cameras in the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest = load_manifest(manifest_path)
    model_id = manifest.model_id
    resolved_gym_id = _resolve_gym_id(gym_id)

    eval_base = EVAL_DIR / model_id

    # FP7oJQ cross-validation gate
    fp7_export = next(e for e in manifest.training_data if e.camera_id == "FP7oJQ")
    logger.info("Running FP7oJQ cross-validation gate...")
    gate = _cross_validate_fp7(manifest, fp7_export, resolved_gym_id)

    force_direct = False
    if gate.get("skipped"):
        logger.info("Cross-validation gate skipped: %s", gate.get("reason", ""))
        force_direct = True  # parquet is stale, use direct inference for all
    elif not gate["passed"]:
        print("\n=== CROSS-VALIDATION GATE FAILED ===")
        print(f"  Recall diff:    {gate['recall_diff']:.4f}")
        print(f"  Precision diff: {gate['precision_diff']:.4f}")
        print(f"  Mean IoU diff:  {gate['mean_iou_diff']:.4f}")
        print("PPDmUg numbers are NOT trustworthy. Investigation needed.")
    else:
        logger.info("Cross-validation gate PASSED (all diffs <= 0.01)")

    # Run per-camera evaluation
    all_train_metrics = []
    all_val_metrics = []

    for export in manifest.training_data:
        cam = export.camera_id
        logger.info("Evaluating %s...", cam)

        zip_path = TRAINING_DATA_DIR / export.export

        # Determine prediction source
        use_direct = run_model or force_direct
        parquet_path = _find_parquet_path(export, resolved_gym_id)
        source_video = _find_source_video(export, parquet_path)

        if export.source_video_path:
            sv_from_manifest = REPO_ROOT / export.source_video_path
            if sv_from_manifest.exists():
                source_video = sv_from_manifest

        if parquet_path is None or use_direct:
            use_direct = True
            if source_video is None or not source_video.exists():
                logger.error("No source video for %s, skipping", cam)
                continue

        source_type = "direct_inference" if use_direct else "parquet"

        # Load predictions
        all_frames = set(enumerate_split_frames(export, "train")) | set(enumerate_split_frames(export, "val"))
        if use_direct:
            if source_video is None or not source_video.exists():
                logger.error("Source video not found for %s direct inference", cam)
                continue
            model_path = REPO_ROOT / manifest.weights_path
            preds = load_preds_from_model(model_path, source_video, all_frames)
        else:
            preds = load_preds_from_parquet(parquet_path, all_frames)

        # Evaluate each split
        clip_dir = eval_base / cam
        all_records = []

        for split_name in ("train", "val"):
            gt = load_gt_for_split(zip_path, export, split_name)
            split_frames = set(enumerate_split_frames(export, split_name))
            split_preds = {fi: preds.get(fi, []) for fi in split_frames}

            frame_results = _match_all_frames(gt, split_preds)
            do_bootstrap = (split_name == "val")
            metrics = _compute_split_metrics(frame_results, split_name, do_bootstrap)

            if split_name == "train":
                all_train_metrics.append((cam, metrics))
            else:
                all_val_metrics.append((cam, metrics))

            all_records.extend(_build_match_records(
                frame_results, model_id, cam, split_name,
            ))

            # Failure gallery (combine train+val)
            if split_name == "val":
                all_fr = _match_all_frames(
                    {**load_gt_for_split(zip_path, export, "train"),
                     **load_gt_for_split(zip_path, export, "val")},
                    preds,
                )
                _render_failure_gallery(
                    all_fr,
                    source_video,
                    clip_dir / "failures",
                )

        # Write per-clip report
        train_m = next(m for c, m in all_train_metrics if c == cam)
        val_m = next(m for c, m in all_val_metrics if c == cam)
        write_clip_report(cam, model_id, source_type, train_m, val_m, clip_dir)

        # Write per_frame_matches.parquet
        if all_records:
            matches_df = pd.DataFrame(all_records)
            matches_path = clip_dir / "per_frame_matches.parquet"
            matches_df.to_parquet(matches_path, index=False)
            logger.info("Wrote %d match records to %s", len(all_records), matches_path)

    # Write aggregate report
    _write_aggregate_report(model_id, all_train_metrics, all_val_metrics, gate, eval_base)


def _write_aggregate_report(
    model_id: str,
    all_train: list[tuple[str, SplitMetrics]],
    all_val: list[tuple[str, SplitMetrics]],
    gate: dict,
    out_dir: Path,
) -> None:
    """Write cross-camera aggregate report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Stage A Aggregate Report: {model_id}\n",
        "## Cross-Validation Gate (FP7oJQ parquet vs direct inference)\n",
    ]

    if gate.get("skipped"):
        lines.append(f"Gate **SKIPPED**: {gate.get('reason', 'unknown')}\n")
    else:
        lines.extend([
            "| Metric | Parquet | Direct | Diff |",
            "|--------|---------|--------|------|",
            f"| Recall@0.5 | {gate['recall_parquet']:.3f} | {gate['recall_direct']:.3f} | {gate['recall_diff']:.4f} |",
            f"| Precision@0.5 | {gate['precision_parquet']:.3f} | {gate['precision_direct']:.3f} | {gate['precision_diff']:.4f} |",
            f"| Mean IoU | {gate['mean_iou_parquet']:.3f} | {gate['mean_iou_direct']:.3f} | {gate['mean_iou_diff']:.4f} |",
            f"\nGate: **{'PASSED' if gate['passed'] else 'FAILED'}**\n",
        ])

    # Per-camera val summary
    lines.append("## Per-Camera Val (Held-Out) Summary\n")
    lines.append("| Camera | Recall@0.5 | Precision@0.5 | Mean IoU | N frames | 95% CI Recall |")
    lines.append("|--------|-----------|--------------|----------|----------|---------------|")
    for cam, m in all_val:
        ci_str = ""
        if m.bootstrap_ci:
            ci = m.bootstrap_ci["recall@0.5"]
            ci_str = f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
        lines.append(
            f"| {cam} | {m.recall.get('@0.5', 0):.3f} | {m.precision.get('@0.5', 0):.3f} "
            f"| {m.mean_iou.get('@0.5', 0):.3f} | {m.n_frames} | {ci_str} |"
        )

    # Aggregate val
    total_gt = sum(m.total_gt for _, m in all_val)
    total_pred = sum(m.total_pred for _, m in all_val)
    total_matched = sum(m.total_matched_05 for _, m in all_val)
    agg_recall = total_matched / total_gt if total_gt > 0 else 0
    agg_precision = total_matched / total_pred if total_pred > 0 else 0
    lines.append(
        f"\n**Aggregate val:** Recall@0.5={agg_recall:.3f}, "
        f"Precision@0.5={agg_precision:.3f} "
        f"({total_matched}/{total_gt} GT matched, {total_pred} total pred)\n"
    )

    # Per-camera train summary
    lines.append("## Per-Camera Train (In-Distribution) Summary\n")
    lines.append("| Camera | Recall@0.5 | Precision@0.5 | Mean IoU | N frames |")
    lines.append("|--------|-----------|--------------|----------|----------|")
    for cam, m in all_train:
        lines.append(
            f"| {cam} | {m.recall.get('@0.5', 0):.3f} | {m.precision.get('@0.5', 0):.3f} "
            f"| {m.mean_iou.get('@0.5', 0):.3f} | {m.n_frames} |"
        )

    md_text = "\n".join(lines)
    (out_dir / "_aggregate.md").write_text(md_text)

    json_data = {
        "model_id": model_id,
        "cross_validation_gate": gate,
        "val": {cam: _metrics_to_dict(m) for cam, m in all_val},
        "train": {cam: _metrics_to_dict(m) for cam, m in all_train},
    }
    (out_dir / "_aggregate.json").write_text(json.dumps(json_data, indent=2))
