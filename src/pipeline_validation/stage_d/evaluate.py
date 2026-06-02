"""Stage D identity stitching evaluation.

Self-contained evaluation — does not depend on TB-EVAL-1 outputs. Loads GT
annotations directly from CVAT zips, re-matches against pipeline detections
via Hungarian matching, and chains detection_id -> person_id via
person_tracks.parquet.

Evaluates whether Stage D global person IDs correctly group detections
that share a GT track_id across frames. Classifies identity errors by
cause: detection_failure, tracklet_dropped, sloppy_box, true_switch.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
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
from pipeline_validation.common.schemas import (
    ExportEntry,
    GTBox,
    GTTrackSequence,
    ModelManifest,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = REPO_ROOT / "outputs" / "_eval" / "stage_d"

ENGINEERING_IMPLICATIONS = {
    "detection_failure": (
        "Detection model is the primary bottleneck. Additional training "
        "data or better model expected to improve identity stitching."
    ),
    "tracklet_dropped": (
        "Stage D's tracklet acceptance criteria are rejecting valid "
        "identities. Investigate Stage D filtering thresholds."
    ),
    "sloppy_box": (
        "Detection precision is the bottleneck. Tighter bounding boxes "
        "(better training, possibly higher imgsz) expected to reduce "
        "identity fragmentation."
    ),
    "true_switch": (
        "Stage D stitching logic is the primary bottleneck. Tier 3 "
        "(HSV histogram) evidence may be insufficient for the observed "
        "scene density."
    ),
    "mixed": (
        "Mixed failure modes. Address detection quality and stitching "
        "logic together."
    ),
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_pipeline_paths(
    manifest: ModelManifest, export: ExportEntry
) -> dict[str, Path]:
    """Resolve paths to pipeline outputs for this export."""
    gym_id = manifest.pipeline_gym_id or "_eval_gt"
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id

    # Search for clip output directory
    pattern = f"{gym_id}/{cam}/**/{clip_id}"
    matches = list(OUTPUTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pipeline output found for {cam}/{clip_id} under {gym_id}"
        )
    clip_dir = matches[0]

    return {
        "clip_dir": clip_dir,
        "detections": clip_dir / "stage_A" / "detections.parquet",
        "person_tracks": clip_dir / "stage_D" / "person_tracks.parquet",
        "identity_hints": clip_dir / "stage_C" / "identity_hints.jsonl",
    }


def _find_source_video(
    export: ExportEntry, clip_dir: Path
) -> Path | None:
    """Find source video for visual rendering."""
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
# Data loading
# ---------------------------------------------------------------------------

def _load_det_to_person(person_tracks_path: Path) -> dict[str, str]:
    """Build detection_id -> person_id mapping from person_tracks."""
    pt = pd.read_parquet(person_tracks_path)
    return dict(zip(pt.detection_id, pt.person_id))


def _load_tier1_evidence(hints_path: Path) -> list[dict]:
    """Load identity_hints.jsonl entries."""
    if not hints_path.exists():
        return []
    content = hints_path.read_text().strip()
    if not content:
        return []
    return [json.loads(line) for line in content.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# GT-to-pipeline matching (self-contained, no TB-EVAL-1 dependency)
# ---------------------------------------------------------------------------

def _match_gt_to_pipeline(
    gt_boxes: list[GTBox],
    det_df_frame: pd.DataFrame,
    det_to_person: dict[str, str],
) -> list[dict]:
    """Hungarian-match GT against pipeline detections for one frame.

    Returns list of per-GT-box results with:
    gt_track_id, person_id, iou, match_status, detection_id
    """
    results = []

    gt_arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in gt_boxes]) if gt_boxes else np.zeros((0, 4))
    pred_arr = det_df_frame[["x1", "y1", "x2", "y2"]].values if len(det_df_frame) > 0 else np.zeros((0, 4))
    det_ids = det_df_frame["detection_id"].tolist() if len(det_df_frame) > 0 else []

    iou_mat = iou_matrix(gt_arr, pred_arr)
    matches = hungarian_match(iou_mat, 0.5)
    matched_gt = {g for g, _ in matches}

    for gi, pi in matches:
        det_id = det_ids[pi]
        person_id = det_to_person.get(det_id)
        results.append({
            "gt_track_id": gt_boxes[gi].track_id,
            "person_id": person_id,
            "iou": float(iou_mat[gi, pi]),
            "match_status": "matched",
            "detection_id": det_id,
        })

    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt:
            results.append({
                "gt_track_id": gt.track_id,
                "person_id": None,
                "iou": 0.0,
                "match_status": "unmatched_gt",
                "detection_id": None,
            })

    return results


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def _build_gt_track_sequences(
    frame_results: dict[int, list[dict]],
    split_name: str,
    camera_id: str,
) -> list[GTTrackSequence]:
    """Collect per-GT-track temporal sequences from per-frame match results."""
    tracks: dict[int, list[dict]] = defaultdict(list)

    for fi in sorted(frame_results.keys()):
        for r in frame_results[fi]:
            tracks[r["gt_track_id"]].append({
                "frame_index": fi,
                "person_id": r["person_id"],
                "match_status": r["match_status"],
                "iou": r["iou"],
            })

    return [
        GTTrackSequence(
            gt_track_id=tid,
            camera_id=camera_id,
            split=split_name,
            frames=frames,
        )
        for tid, frames in sorted(tracks.items())
    ]


# ---------------------------------------------------------------------------
# Identity mapping (most-frequent vote)
# ---------------------------------------------------------------------------

def _compute_identity_mapping(
    sequences: list[GTTrackSequence],
) -> dict[int, dict]:
    """Most-frequent vote per GT track. Tie-break earliest frame."""
    mapping = {}
    for seq in sequences:
        person_ids = []
        earliest: dict[str, int] = {}
        for f in seq.frames:
            pid = f["person_id"]
            if pid is not None:
                person_ids.append(pid)
                if pid not in earliest:
                    earliest[pid] = f["frame_index"]

        if not person_ids:
            mapping[seq.gt_track_id] = {
                "canonical_person_id": None,
                "purity": 0.0,
                "frames_matched": 0,
                "frames_total": len(seq.frames),
            }
            continue

        counts = Counter(person_ids)
        # Tie-break by earliest frame
        canonical = min(
            counts.keys(),
            key=lambda pid: (-counts[pid], earliest.get(pid, 0)),
        )
        purity = counts[canonical] / len(person_ids)

        mapping[seq.gt_track_id] = {
            "canonical_person_id": canonical,
            "purity": purity,
            "frames_matched": len(person_ids),
            "frames_total": len(seq.frames),
        }

    return mapping


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _compute_aggregate_metrics(
    sequences: list[GTTrackSequence],
    identity_mapping: dict[int, dict],
    all_person_ids_in_tracks: set[str],
) -> dict:
    """Compute identity recall, precision, coverage, purity."""
    gt_count = len(sequences)

    # Identity recall: fraction of GT tracks with non-null canonical
    recalled = sum(
        1 for m in identity_mapping.values()
        if m["canonical_person_id"] is not None
    )
    identity_recall = recalled / gt_count if gt_count > 0 else 0.0

    # Canonical person_ids and merger detection
    canonical_pids: dict[str, list[int]] = defaultdict(list)
    for tid, m in identity_mapping.items():
        pid = m["canonical_person_id"]
        if pid is not None:
            canonical_pids[pid].append(tid)

    unique_canonical = sum(1 for tracks in canonical_pids.values() if len(tracks) == 1)
    merger_errors = sum(1 for tracks in canonical_pids.values() if len(tracks) >= 2)
    identity_precision = (
        unique_canonical / (unique_canonical + merger_errors)
        if (unique_canonical + merger_errors) > 0 else 0.0
    )

    # Unmatched person_ids (no matched GT frames at all)
    matched_pids = set(canonical_pids.keys())
    unmatched_person_ids = all_person_ids_in_tracks - matched_pids

    # Mean coverage and purity
    coverages = []
    purities = []
    for m in identity_mapping.values():
        coverage = m["frames_matched"] / m["frames_total"] if m["frames_total"] > 0 else 0.0
        coverages.append(coverage)
        if m["canonical_person_id"] is not None:
            purities.append(m["purity"])

    return {
        "identity_recall": identity_recall,
        "identity_precision": identity_precision,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "gt_track_count": gt_count,
        "person_id_count": len(all_person_ids_in_tracks),
        "unique_canonical_count": unique_canonical,
        "merger_error_count": merger_errors,
        "merger_details": {
            pid: tracks for pid, tracks in canonical_pids.items() if len(tracks) >= 2
        },
        "unmatched_person_id_count": len(unmatched_person_ids),
        "fragmentation_flags": [
            {
                "gt_track_id": tid,
                "canonical_person_id": m["canonical_person_id"],
                "purity": m["purity"],
            }
            for tid, m in identity_mapping.items()
            if m["canonical_person_id"] is not None and m["purity"] < 0.9
        ],
    }


# ---------------------------------------------------------------------------
# Switch classification
# ---------------------------------------------------------------------------

def _classify_switches(
    sequences: list[GTTrackSequence],
) -> list[dict]:
    """Walk each sequence, detect transitions, classify cause.

    Four cause categories:
    - detection_failure: Stage A produced no matched detection
    - tracklet_dropped: Stage A matched but Stage D dropped the tracklet
    - sloppy_box: IoU < 0.7 on bracketing frames
    - true_switch: IoU >= 0.7, Stage D mis-stitched
    """
    events = []

    for seq in sequences:
        frames = seq.frames
        if len(frames) < 2:
            continue

        i = 0
        while i < len(frames):
            cur = frames[i]
            cur_pid = cur["person_id"]

            # Detect gap start
            if cur_pid is None:
                gap_start = i
                # Find gap end
                gap_end = i
                while gap_end < len(frames) and frames[gap_end]["person_id"] is None:
                    gap_end += 1

                # Classify gap cause
                gap_frames = frames[gap_start:gap_end]
                all_unmatched = all(f["match_status"] == "unmatched_gt" for f in gap_frames)
                any_matched_no_person = any(
                    f["match_status"] == "matched" and f["person_id"] is None
                    for f in gap_frames
                )

                if all_unmatched:
                    cause = "detection_failure"
                elif any_matched_no_person:
                    cause = "tracklet_dropped"
                else:
                    cause = "detection_failure"

                pre_pid = frames[gap_start - 1]["person_id"] if gap_start > 0 else None
                post_pid = frames[gap_end]["person_id"] if gap_end < len(frames) else None

                # Record gap event
                events.append({
                    "gt_track_id": seq.gt_track_id,
                    "split": seq.split,
                    "event_type": "gap",
                    "frame_before": frames[gap_start - 1]["frame_index"] if gap_start > 0 else None,
                    "frame_after": frames[gap_end]["frame_index"] if gap_end < len(frames) else None,
                    "person_id_before": pre_pid,
                    "person_id_after": post_pid,
                    "iou_before": frames[gap_start - 1]["iou"] if gap_start > 0 else 0.0,
                    "iou_after": frames[gap_end]["iou"] if gap_end < len(frames) else 0.0,
                    "cause": cause,
                    "gap_length": gap_end - gap_start,
                })

                # Check for gap-then-switch
                if (pre_pid is not None and post_pid is not None
                        and pre_pid != post_pid):
                    iou_before = frames[gap_start - 1]["iou"] if gap_start > 0 else 0.0
                    iou_after = frames[gap_end]["iou"] if gap_end < len(frames) else 0.0
                    if iou_before < 0.7 or iou_after < 0.7:
                        switch_cause = "sloppy_box"
                    else:
                        switch_cause = "true_switch"

                    events.append({
                        "gt_track_id": seq.gt_track_id,
                        "split": seq.split,
                        "event_type": "gap_then_switch",
                        "frame_before": frames[gap_start - 1]["frame_index"],
                        "frame_after": frames[gap_end]["frame_index"],
                        "person_id_before": pre_pid,
                        "person_id_after": post_pid,
                        "iou_before": iou_before,
                        "iou_after": iou_after,
                        "cause": switch_cause,
                    })

                i = gap_end
                continue

            # Detect direct switch (no gap)
            if i > 0:
                prev = frames[i - 1]
                prev_pid = prev["person_id"]
                if prev_pid is not None and cur_pid is not None and prev_pid != cur_pid:
                    if prev["iou"] < 0.7 or cur["iou"] < 0.7:
                        cause = "sloppy_box"
                    else:
                        cause = "true_switch"

                    events.append({
                        "gt_track_id": seq.gt_track_id,
                        "split": seq.split,
                        "event_type": "switch",
                        "frame_before": prev["frame_index"],
                        "frame_after": cur["frame_index"],
                        "person_id_before": prev_pid,
                        "person_id_after": cur_pid,
                        "iou_before": prev["iou"],
                        "iou_after": cur["iou"],
                        "cause": cause,
                    })

            i += 1

    return events


# ---------------------------------------------------------------------------
# Visual strip
# ---------------------------------------------------------------------------

def _render_lowest_purity_strip(
    sequences: list[GTTrackSequence],
    identity_mapping: dict[int, dict],
    source_video: Path | None,
    det_to_person: dict[str, str],
    out_path: Path,
) -> None:
    """Render 4-frame strip of the lowest-purity GT track."""
    if source_video is None or not source_video.exists():
        logger.warning("No source video for purity strip, skipping")
        return

    # Find lowest-purity track with at least some matched frames
    candidates = [
        (tid, m) for tid, m in identity_mapping.items()
        if m["canonical_person_id"] is not None and m["frames_matched"] >= 4
    ]
    if not candidates:
        return

    worst_tid, worst_m = min(candidates, key=lambda x: x[1]["purity"])
    worst_seq = next((s for s in sequences if s.gt_track_id == worst_tid), None)
    if not worst_seq:
        return

    # Pick 4 equally-spaced frames from the matched frames
    matched_frames = [f for f in worst_seq.frames if f["person_id"] is not None]
    if len(matched_frames) < 4:
        sample_frames = matched_frames
    else:
        step = len(matched_frames) / 4
        sample_frames = [matched_frames[int(i * step)] for i in range(4)]

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return

    panels = []
    for sf in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf["frame_index"])
        ret, img = cap.read()
        if not ret:
            continue

        # Draw GT box in green with track_id label
        gt_seq_frame = next(
            (f for f in worst_seq.frames if f["frame_index"] == sf["frame_index"]),
            None,
        )
        if gt_seq_frame:
            pid = sf["person_id"]
            label = f"GT:{worst_tid} -> {pid}" if pid else f"GT:{worst_tid} -> None"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"f{sf['frame_index']} IoU={sf['iou']:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize for strip
        h, w = img.shape[:2]
        target_h = 360
        scale = target_h / h
        img = cv2.resize(img, (int(w * scale), target_h))
        panels.append(img)

    cap.release()

    if panels:
        strip = np.hstack(panels)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), strip)
        logger.info("Purity strip: %s (GT track %d, purity %.2f)",
                     out_path, worst_tid, worst_m["purity"])


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _failure_mode_summary(all_events: list[dict]) -> tuple[str, dict[str, int]]:
    """Generate failure mode summary text and cause counts."""
    cause_counts: dict[str, int] = Counter()
    for e in all_events:
        cause_counts[e["cause"]] += 1

    total = sum(cause_counts.values())
    if total == 0:
        return "No identity errors detected.\n", cause_counts

    lines = ["## Failure Mode Summary\n"]
    for cause in ["detection_failure", "tracklet_dropped", "sloppy_box", "true_switch"]:
        n = cause_counts.get(cause, 0)
        pct = n / total * 100 if total > 0 else 0
        labels = {
            "detection_failure": "detection_failure (Stage A missed the person entirely)",
            "tracklet_dropped": "tracklet_dropped (Stage A matched, Stage D rejected the tracklet)",
            "sloppy_box": "sloppy_box (Stage A found person but boxes too loose for stitching)",
            "true_switch": "true_switch (Stage A clean, Stage D mis-stitched)",
        }
        lines.append(f"- {pct:.0f}% of identity errors are {labels[cause]}")

    # Engineering implication
    dominant = max(cause_counts, key=cause_counts.get) if cause_counts else "mixed"
    dominant_pct = cause_counts.get(dominant, 0) / total * 100 if total > 0 else 0
    if dominant_pct <= 50:
        implication = ENGINEERING_IMPLICATIONS["mixed"]
    else:
        implication = ENGINEERING_IMPLICATIONS.get(dominant, ENGINEERING_IMPLICATIONS["mixed"])
    lines.append(f"\nEngineering implication: {implication}\n")

    return "\n".join(lines), cause_counts


def _format_split_section(
    split_name: str,
    metrics: dict,
    events: list[dict],
    identity_mapping: dict[int, dict],
    sequences: list[GTTrackSequence],
) -> str:
    """Format a split's metrics as markdown."""
    header = "In-Distribution (train split)" if split_name == "train" else "Held-Out (val split)"
    lines = [f"## {header}\n"]

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Identity recall | {metrics['identity_recall']:.3f} |")
    lines.append(f"| Identity precision | {metrics['identity_precision']:.3f} |")
    lines.append(f"| Mean coverage | {metrics['mean_coverage']:.3f} |")
    lines.append(f"| Mean purity | {metrics['mean_purity']:.3f} |")
    lines.append(f"| GT track count | {metrics['gt_track_count']} |")
    lines.append(f"| Person ID count | {metrics['person_id_count']} |")
    lines.append(f"| Unmatched person IDs | {metrics['unmatched_person_id_count']} |")

    # Cause-classified switch summary
    split_events = [e for e in events if e["split"] == split_name]
    cause_counts = Counter(e["cause"] for e in split_events)
    total_events = sum(cause_counts.values())

    lines.append("\n### Cause-classified switch summary\n")
    lines.append("| Cause | Count | % of switches |")
    lines.append("|-------|-------|---------------|")
    for cause in ["detection_failure", "tracklet_dropped", "sloppy_box", "true_switch"]:
        n = cause_counts.get(cause, 0)
        pct = n / total_events * 100 if total_events > 0 else 0
        lines.append(f"| {cause} | {n} | {pct:.0f}% |")

    # Fragmentation flags
    frags = metrics.get("fragmentation_flags", [])
    if frags:
        lines.append("\n### Fragmentation flags (purity < 0.9)\n")
        lines.append("| gt_track_id | canonical_person_id | purity |")
        lines.append("|-------------|---------------------|--------|")
        for f in frags:
            lines.append(f"| {f['gt_track_id']} | {f['canonical_person_id']} | {f['purity']:.3f} |")

    # Merger flags
    mergers = metrics.get("merger_details", {})
    if mergers:
        lines.append("\n### Merger flags (one person_id, multiple GT tracks)\n")
        lines.append("| person_id | GT tracks merged | track count |")
        lines.append("|-----------|-----------------|-------------|")
        for pid, tracks in mergers.items():
            lines.append(f"| {pid} | {tracks} | {len(tracks)} |")

    # Per-GT-track summary
    lines.append("\n### Per-GT-track summary\n")
    lines.append("| gt_track_id | canonical_person_id | coverage | purity | switches | dominant_cause |")
    lines.append("|-------------|---------------------|----------|--------|----------|----------------|")

    for seq in sorted(sequences, key=lambda s: -identity_mapping[s.gt_track_id]["frames_total"]):
        m = identity_mapping[seq.gt_track_id]
        coverage = m["frames_matched"] / m["frames_total"] if m["frames_total"] > 0 else 0
        track_events = [e for e in split_events if e["gt_track_id"] == seq.gt_track_id]
        track_causes = Counter(e["cause"] for e in track_events)
        n_switches = len(track_events)
        dom_cause = track_causes.most_common(1)[0][0] if track_causes else "-"
        lines.append(
            f"| {seq.gt_track_id} | {m['canonical_person_id']} | {coverage:.3f} "
            f"| {m['purity']:.3f} | {n_switches} | {dom_cause} |"
        )

    return "\n".join(lines)


def _write_clip_report(
    camera_id: str,
    model_id: str,
    tier1_text: str,
    failure_summary: str,
    train_section: str,
    val_section: str,
    all_metrics: dict,
    out_dir: Path,
) -> None:
    """Write per-clip report as .md and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# Stage D Identity Evaluation: {camera_id}\n",
        f"Model: {model_id}",
        "Self-contained evaluation -- does not depend on TB-EVAL-1 outputs.\n",
        tier1_text,
        failure_summary,
        train_section,
        "",
        val_section,
    ]

    (out_dir / "report.md").write_text("\n".join(md_lines))
    (out_dir / "report.json").write_text(json.dumps(all_metrics, indent=2))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_all(manifest_path: Path) -> None:
    """Run Stage D evaluation for all cameras in the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest = load_manifest(manifest_path)
    model_id = manifest.model_id
    eval_base = EVAL_DIR / model_id

    all_camera_results = []

    # Filter to exports with val splits (train-only entries skip evaluation)
    eval_exports = [e for e in manifest.training_data if e.splits.val is not None]

    for export in eval_exports:
        cam = export.camera_id
        logger.info("Evaluating Stage D for %s...", cam)

        zip_path = TRAINING_DATA_DIR / export.export

        # Resolve pipeline paths
        try:
            paths = _resolve_pipeline_paths(manifest, export)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", cam, e)
            continue

        det_df = pd.read_parquet(paths["detections"])
        det_to_person = _load_det_to_person(paths["person_tracks"])
        pt_df = pd.read_parquet(paths["person_tracks"])
        tier1_hints = _load_tier1_evidence(paths["identity_hints"])

        # Tier 1 anchor text
        if tier1_hints:
            tier1_lines = ["## Tier 1 Evidence Anchor\n"]
            for hint in tier1_hints:
                tid = hint.get("tracklet_id", "?")
                tag = hint.get("anchor_key", "?")
                first_frame = hint.get("evidence", {}).get("first_seen_frame", "?")
                tkl_rows = pt_df[pt_df.tracklet_id == tid]
                if len(tkl_rows) > 0:
                    pid = tkl_rows.person_id.iloc[0]
                    status = f"Stage D assigned to person_id **{pid}**"
                else:
                    status = "**Stage D dropped this tracklet** (not in person_tracks)"
                tier1_lines.append(
                    f"- Tracklet {tid} has {tag} (frame {first_frame}). {status}."
                )
            tier1_text = "\n".join(tier1_lines) + "\n"
        else:
            tier1_text = "## Tier 1 Evidence Anchor\n\nNo tag evidence for this camera.\n"

        # Evaluate each split
        clip_dir = eval_base / cam
        all_sequences = []
        all_events = []
        split_metrics = {}
        split_sections = {}

        # Collect all person_ids visible at annotated frames per split
        for split_name in ("train", "val"):
            gt = load_gt_for_split(zip_path, export, split_name)
            split_frames = sorted(enumerate_split_frames(export, split_name))

            # Match GT against pipeline detections per frame
            frame_results: dict[int, list[dict]] = {}
            for fi in split_frames:
                gt_boxes = gt.get(fi, [])
                det_frame = det_df[det_df.frame_index == fi]
                frame_results[fi] = _match_gt_to_pipeline(gt_boxes, det_frame, det_to_person)

            sequences = _build_gt_track_sequences(frame_results, split_name, cam)
            identity_map = _compute_identity_mapping(sequences)

            # All person_ids that appear in pipeline at annotated frames
            all_pids_at_frames = set()
            for fi in split_frames:
                frame_pids = pt_df[pt_df.frame_index == fi].person_id.unique()
                all_pids_at_frames.update(frame_pids)

            metrics = _compute_aggregate_metrics(sequences, identity_map, all_pids_at_frames)
            events = _classify_switches(sequences)

            all_sequences.extend(sequences)
            all_events.extend(events)
            split_metrics[split_name] = metrics
            split_sections[split_name] = _format_split_section(
                split_name, metrics, events, identity_map, sequences
            )

        # Combined identity mapping for persistence
        all_gt = load_gt_for_split(zip_path, export, "train")
        all_gt.update(load_gt_for_split(zip_path, export, "val"))
        all_frame_results: dict[int, list[dict]] = {}
        for fi in sorted(all_gt.keys()):
            gt_boxes = all_gt.get(fi, [])
            det_frame = det_df[det_df.frame_index == fi]
            all_frame_results[fi] = _match_gt_to_pipeline(gt_boxes, det_frame, det_to_person)
        combined_seqs = _build_gt_track_sequences(all_frame_results, "all", cam)
        combined_map = _compute_identity_mapping(combined_seqs)

        # Failure mode summary (across both splits)
        failure_summary, cause_counts = _failure_mode_summary(all_events)

        # Write outputs
        clip_dir.mkdir(parents=True, exist_ok=True)

        # gt_track_sequences.jsonl
        with open(clip_dir / "gt_track_sequences.jsonl", "w") as f:
            for seq in all_sequences:
                f.write(seq.model_dump_json() + "\n")

        # identity_mapping.json
        mapping_out = {
            f"gt_track_{tid}": m for tid, m in combined_map.items()
        }
        (clip_dir / "identity_mapping.json").write_text(json.dumps(mapping_out, indent=2))

        # id_switches.jsonl
        with open(clip_dir / "id_switches.jsonl", "w") as f:
            for e in all_events:
                f.write(json.dumps(e) + "\n")

        # Report
        _write_clip_report(
            cam, model_id, tier1_text, failure_summary,
            split_sections["train"], split_sections["val"],
            {"train": split_metrics["train"], "val": split_metrics["val"],
             "cause_counts": dict(cause_counts)},
            clip_dir,
        )

        # Visual strip
        source_video = _find_source_video(export, paths["clip_dir"])
        _render_lowest_purity_strip(
            combined_seqs, combined_map, source_video, det_to_person,
            clip_dir / "lowest_purity_strip.jpg",
        )

        logger.info(
            "%s: identity_recall=%.3f/%.3f (train/val), %d events",
            cam,
            split_metrics["train"]["identity_recall"],
            split_metrics["val"]["identity_recall"],
            len(all_events),
        )
        all_camera_results.append((cam, split_metrics, cause_counts))

    # Aggregate report
    _write_aggregate_report(model_id, all_camera_results, eval_base)

    # Step 4: GT person trace (CP6)
    from pipeline_validation.gt_person_trace import (
        compute_gt_person_trace,
        write_camera_summary,
        write_trace_artifacts,
    )

    eval_root = EVAL_DIR.parent  # outputs/_eval/
    trace_results = []
    for export in eval_exports:
        cam = export.camera_id
        try:
            paths = _resolve_pipeline_paths(manifest, export)
            clip_dir = paths["clip_dir"]
        except FileNotFoundError:
            clip_dir = None
        try:
            tr = compute_gt_person_trace(
                eval_dir=eval_root,
                model_id=model_id,
                camera_id=cam,
                pipeline_clip_dir=clip_dir,
            )
            write_trace_artifacts(tr, eval_root)
            trace_results.append(tr)
            logger.info(
                "%s trace: mode=%s, %d frames, %d warnings",
                cam, tr.mode, len(tr.trace_df), len(tr.warnings),
            )
            if tr.mode == "lite":
                logger.warning(
                    "%s ran in LITE MODE -- 3 Stage D failure modes collapsed "
                    "into stage_d_no_person. Pipeline artifacts at %s not "
                    "available.",
                    cam, clip_dir,
                )
        except Exception as exc:
            logger.warning("GT trace failed for %s: %s", cam, exc)

    if trace_results:
        write_camera_summary(trace_results, eval_root, model_id)


def _write_aggregate_report(
    model_id: str,
    all_results: list[tuple[str, dict, dict]],
    out_dir: Path,
) -> None:
    """Write cross-camera aggregate report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Stage D Aggregate Report: {model_id}\n"]

    # Cross-camera failure mode summary
    lines.append("## Cross-Camera Failure Mode Summary\n")
    lines.append("| Camera | detection_failure | tracklet_dropped | sloppy_box | true_switch | total |")
    lines.append("|--------|-------------------|------------------|------------|-------------|-------|")
    for cam, _, causes in all_results:
        total = sum(causes.values())
        lines.append(
            f"| {cam} | {causes.get('detection_failure', 0)} "
            f"| {causes.get('tracklet_dropped', 0)} "
            f"| {causes.get('sloppy_box', 0)} "
            f"| {causes.get('true_switch', 0)} | {total} |"
        )

    # Val summary
    lines.append("\n## Per-Camera Val (Held-Out) Summary\n")
    lines.append("| Camera | ID Recall | ID Precision | Mean Coverage | Mean Purity | GT tracks | Person IDs |")
    lines.append("|--------|-----------|-------------|--------------|-------------|-----------|------------|")
    for cam, metrics, _ in all_results:
        m = metrics["val"]
        lines.append(
            f"| {cam} | {m['identity_recall']:.3f} | {m['identity_precision']:.3f} "
            f"| {m['mean_coverage']:.3f} | {m['mean_purity']:.3f} "
            f"| {m['gt_track_count']} | {m['person_id_count']} |"
        )

    # Train summary
    lines.append("\n## Per-Camera Train (In-Distribution) Summary\n")
    lines.append("| Camera | ID Recall | ID Precision | Mean Coverage | Mean Purity | GT tracks | Person IDs |")
    lines.append("|--------|-----------|-------------|--------------|-------------|-----------|------------|")
    for cam, metrics, _ in all_results:
        m = metrics["train"]
        lines.append(
            f"| {cam} | {m['identity_recall']:.3f} | {m['identity_precision']:.3f} "
            f"| {m['mean_coverage']:.3f} | {m['mean_purity']:.3f} "
            f"| {m['gt_track_count']} | {m['person_id_count']} |"
        )

    md_text = "\n".join(lines)
    (out_dir / "_aggregate.md").write_text(md_text)

    json_data = {
        "model_id": model_id,
        "cameras": {
            cam: {"train": metrics["train"], "val": metrics["val"],
                   "cause_counts": dict(causes)}
            for cam, metrics, causes in all_results
        },
    }
    (out_dir / "_aggregate.json").write_text(json.dumps(json_data, indent=2, default=str))
