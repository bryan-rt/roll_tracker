"""D-stage signal preservation trace (CP-TRACE-2).

Joins CP-TRACE-1's per-GT-person-frame Stage A trace through to Stage D's
person_tracks.parquet. For each GT person at each annotated frame, determines
what person_id(s) Stage D assigned and classifies signal preservation.

Key design: (tracklet_id, frame_index) is NOT unique in person_tracks — GROUP
segments produce 2 rows (one per person_id). The join returns ALL person_ids
as a set, and classification checks whether the GT track's dominant_person_id
is in that set.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from pipeline_validation.common.schemas import ExportEntry, ModelManifest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "_eval" / "signal_trace"


def _resolve_stage_d_path(
    manifest: ModelManifest, export: ExportEntry, gym_id: str,
) -> Path | None:
    """Find stage_D directory for a camera's clip."""
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id
    pattern = f"{gym_id}/{cam}/**/{clip_id}/stage_D"
    matches = list(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


def _build_segment_lookup(
    seg_df: pd.DataFrame,
) -> dict[str, list[tuple[int, int, str, str, str]]]:
    """Pre-index d1_segments by base_tracklet_id for interval lookup.

    Returns: {tracklet_id: [(start, end, seg_type, node_id, payload_json), ...]}
    """
    lookup: dict[str, list[tuple[int, int, str, str, str]]] = defaultdict(list)
    for _, row in seg_df.iterrows():
        lookup[row.base_tracklet_id].append((
            int(row.start_frame),
            int(row.end_frame),
            row.segment_type,
            row.node_id,
            row.payload_json if pd.notna(row.payload_json) else "{}",
        ))
    return dict(lookup)


def _lookup_node_type(
    lookup: dict[str, list], tracklet_id: str, frame_index: int,
) -> str | None:
    """Interval lookup: returns 'SOLO' or 'GROUP', or None if not found."""
    segments = lookup.get(tracklet_id)
    if not segments:
        return None
    for start, end, seg_type, _, _ in segments:
        if start <= frame_index <= end:
            return seg_type
    return None


def _build_split_resolution(
    stage_d_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, tuple[int, int]]]:
    """Build split-product resolution from d05_split_audit.jsonl + bank summaries.

    Returns:
        split_map: {original_tid: [product_tids]} for tracklets that were split.
        tid_frame_range: {tid: (start_frame, end_frame)} for ALL bank tracklets.
    """
    # Load split audit
    split_map: dict[str, set[str]] = defaultdict(set)
    audit_path = stage_d_dir / "d05_split_audit.jsonl"
    if audit_path.exists():
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("artifact_type") == "d05_split_event":
                    split_map[ev["original_tracklet_id"]].add(ev["new_tracklet_id"])

    # Load bank summaries for frame ranges
    bank = pd.read_parquet(stage_d_dir / "tracklet_bank_summaries.parquet")
    tid_frame_range: dict[str, tuple[int, int]] = {}
    for _, row in bank.iterrows():
        tid_frame_range[row.tracklet_id] = (int(row.start_frame), int(row.end_frame))

    return {k: sorted(v) for k, v in split_map.items()}, tid_frame_range


def _resolve_tracklet_id(
    tid: str,
    frame_index: int,
    split_map: dict[str, list[str]],
    tid_frame_range: dict[str, tuple[int, int]],
) -> str:
    """Resolve original tracklet_id to the split product covering this frame.

    If tid was not split, returns tid unchanged.
    If tid was split and a product covers this frame, returns the product tid.
    If no product covers the frame, returns tid (will result in no_id downstream).
    """
    products = split_map.get(tid)
    if not products:
        return tid

    # Check if the original still covers this frame
    orig_range = tid_frame_range.get(tid)
    if orig_range and orig_range[0] <= frame_index <= orig_range[1]:
        return tid

    # Check split products
    for prod in products:
        prod_range = tid_frame_range.get(prod)
        if prod_range and prod_range[0] <= frame_index <= prod_range[1]:
            return prod

    return tid  # no product covers this frame


def _compute_dominant_person_ids(
    trace_df: pd.DataFrame,
    pt_df: pd.DataFrame,
    split_map: dict[str, list[str]],
    tid_frame_range: dict[str, tuple[int, int]],
) -> dict[int, str]:
    """Majority-vote per GT track: gt_track_id -> dominant person_id.

    For each GT track, collects all person_ids assigned to its matched
    tracklets across all frames, counts occurrences, returns the most
    frequent. Ties broken lexicographically.

    Resolves D0.5 split products before the person_tracks lookup.
    """
    # Build (tracklet_id, frame_index) -> list of person_ids from person_tracks
    pt_grouped = pt_df.groupby(["tracklet_id", "frame_index"])["person_id"].apply(list)
    pt_lookup = pt_grouped.to_dict()

    # Per GT track: count all person_id occurrences
    gt_counters: dict[int, Counter] = defaultdict(Counter)
    for _, row in trace_df.iterrows():
        tid = row.tracklet_id
        fi = row.frame_index
        gt_tid = row.gt_track_id
        if pd.isna(tid):
            continue
        resolved_tid = _resolve_tracklet_id(tid, fi, split_map, tid_frame_range)
        pids = pt_lookup.get((resolved_tid, fi), [])
        for pid in pids:
            gt_counters[gt_tid][pid] += 1

    # Majority vote with lexicographic tiebreak
    result: dict[int, str] = {}
    for gt_tid, counter in gt_counters.items():
        if not counter:
            continue
        max_count = max(counter.values())
        candidates = sorted(pid for pid, c in counter.items() if c == max_count)
        result[gt_tid] = candidates[0]

    return result


def run_d_trace(
    manifest: ModelManifest,
    export: ExportEntry,
    gym_id: str,
    stage_a_trace_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Run D-stage trace for one camera.

    Returns (d_trace_df, preservation_summary).
    """
    cam = export.camera_id

    # Load CP-TRACE-1 Stage A trace
    trace_df = pd.read_parquet(stage_a_trace_path)
    logger.info("%s: loaded %d Stage A trace rows", cam, len(trace_df))

    # Load Stage D artifacts
    stage_d_dir = _resolve_stage_d_path(manifest, export, gym_id)
    if stage_d_dir is None:
        raise FileNotFoundError(f"stage_D directory not found for {cam} under {gym_id}")

    pt_df = pd.read_parquet(stage_d_dir / "person_tracks.parquet")
    seg_df = pd.read_parquet(stage_d_dir / "d1_segments.parquet")
    logger.info("%s: %d person_track rows, %d segments", cam, len(pt_df), len(seg_df))

    # Build lookups
    seg_lookup = _build_segment_lookup(seg_df)
    split_map, tid_frame_range = _build_split_resolution(stage_d_dir)
    logger.info("%s: %d tracklets with split products", cam, len(split_map))

    # Build (tracklet_id, frame_index) -> list of person_ids
    pt_grouped = pt_df.groupby(["tracklet_id", "frame_index"])["person_id"].apply(list)
    pt_lookup = pt_grouped.to_dict()

    # Compute dominant person_ids (with split resolution)
    dominant_map = _compute_dominant_person_ids(
        trace_df, pt_df, split_map, tid_frame_range,
    )

    # Extend trace with D-stage columns
    records: list[dict] = []
    for _, row in trace_df.iterrows():
        tid = row.tracklet_id
        fi = row.frame_index
        gt_tid = row.gt_track_id
        dominant_pid = dominant_map.get(gt_tid)

        if pd.isna(tid):
            # miss from Stage A
            records.append({
                "person_ids_json": "[]",
                "n_person_ids": 0,
                "dominant_person_id": dominant_pid,
                "d_classification": "no_detection",
                "node_type": None,
            })
            continue

        resolved_tid = _resolve_tracklet_id(tid, fi, split_map, tid_frame_range)
        pids = pt_lookup.get((resolved_tid, fi), [])
        node_type = _lookup_node_type(seg_lookup, resolved_tid, fi)

        if not pids:
            d_class = "no_id"
        elif dominant_pid is not None and dominant_pid in pids:
            d_class = "correct_id"
        elif pids:
            d_class = "wrong_id"
        else:
            d_class = "no_id"

        records.append({
            "person_ids_json": json.dumps(sorted(pids)),
            "n_person_ids": len(pids),
            "dominant_person_id": dominant_pid,
            "d_classification": d_class,
            "node_type": node_type,
        })

    new_cols = pd.DataFrame(records)
    d_trace_df = pd.concat([trace_df.reset_index(drop=True), new_cols], axis=1)

    # Build summary
    total = len(d_trace_df)
    d_counts = d_trace_df.d_classification.value_counts().to_dict()
    n_pid_counts = d_trace_df.n_person_ids.value_counts().sort_index().to_dict()

    summary: dict = {
        "camera_id": cam,
        "total_gt_person_frames": total,
        "correct_id": {
            "count": d_counts.get("correct_id", 0),
            "pct": round(d_counts.get("correct_id", 0) / total, 4) if total else 0,
        },
        "wrong_id": {
            "count": d_counts.get("wrong_id", 0),
            "pct": round(d_counts.get("wrong_id", 0) / total, 4) if total else 0,
        },
        "no_id": {
            "count": d_counts.get("no_id", 0),
            "pct": round(d_counts.get("no_id", 0) / total, 4) if total else 0,
        },
        "no_detection": {
            "count": d_counts.get("no_detection", 0),
            "pct": round(d_counts.get("no_detection", 0) / total, 4) if total else 0,
        },
        "n_person_ids_distribution": {str(k): int(v) for k, v in n_pid_counts.items()},
        "n_unique_gt_tracks": int(d_trace_df.gt_track_id.nunique()),
        "n_unique_person_ids_assigned": int(
            d_trace_df[d_trace_df.n_person_ids > 0]
            .person_ids_json.apply(lambda x: json.loads(x))
            .explode().nunique()
        ) if d_counts.get("correct_id", 0) + d_counts.get("wrong_id", 0) > 0 else 0,
    }

    # Identity collisions: GT tracks sharing the same dominant_person_id
    dom_to_gts: dict[str, list[int]] = defaultdict(list)
    for gt_tid, pid in dominant_map.items():
        dom_to_gts[pid].append(gt_tid)
    collisions = [
        {
            "dominant_person_id": pid,
            "gt_tracks": [f"gt_track_{t}" for t in sorted(gts)],
            "note": "two GT people collapsed to same person_id",
        }
        for pid, gts in sorted(dom_to_gts.items())
        if len(gts) >= 2
    ]
    summary["identity_collisions"] = collisions

    # Per-GT-track breakdown
    per_track: dict[str, dict] = {}
    for gt_tid, grp in d_trace_df.groupby("gt_track_id"):
        dc = grp.d_classification.value_counts().to_dict()
        dom_pid = dominant_map.get(gt_tid)
        correct = dc.get("correct_id", 0)
        total_t = len(grp)
        per_track[f"gt_track_{gt_tid}"] = {
            "dominant_person_id": dom_pid,
            "correct": correct,
            "wrong": dc.get("wrong_id", 0),
            "no_id": dc.get("no_id", 0),
            "no_det": dc.get("no_detection", 0),
            "purity": round(correct / total_t, 4) if total_t else 0,
        }
    summary["per_gt_track"] = per_track

    return d_trace_df, summary


def write_d_trace_artifacts(
    model_id: str,
    camera_id: str,
    d_trace_df: pd.DataFrame,
    summary: dict,
) -> Path:
    """Write D-trace parquet, JSON summary, and markdown."""
    out_dir = EVAL_DIR / model_id / camera_id
    out_dir.mkdir(parents=True, exist_ok=True)

    d_trace_df.to_parquet(out_dir / "gt_signal_trace_d.parquet", index=False)

    with open(out_dir / "signal_preservation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown
    total = summary["total_gt_person_frames"]
    md = [
        f"# Signal Trace D-Stage: {model_id} / {camera_id}",
        "",
        f"Total GT-person-frames: {total}",
        "",
        "## Signal Preservation",
        "",
        "| Classification | Count | Pct |",
        "|---|---:|---:|",
    ]
    for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
        c = summary[cls]
        md.append(f"| {cls} | {c['count']} | {c['pct']:.1%} |")

    md.extend([
        "",
        f"Unique GT tracks: {summary['n_unique_gt_tracks']}",
        f"Unique person_ids assigned: {summary['n_unique_person_ids_assigned']}",
        "",
        "## n_person_ids Distribution",
        "",
        "| n_person_ids | Count |",
        "|---:|---:|",
    ])
    for k, v in sorted(summary["n_person_ids_distribution"].items()):
        md.append(f"| {k} | {v} |")

    if summary["identity_collisions"]:
        md.extend(["", "## Identity Collisions", ""])
        for col in summary["identity_collisions"]:
            md.append(f"- **{col['dominant_person_id']}**: "
                       f"{', '.join(col['gt_tracks'])}")
    else:
        md.extend(["", "## Identity Collisions", "", "None detected."])

    md.extend(["", "## Per-GT-Track Breakdown", "",
               "| GT Track | dominant | correct | wrong | no_id | no_det | purity |",
               "|---|---|---:|---:|---:|---:|---:|"])
    for tid, tc in sorted(summary["per_gt_track"].items()):
        md.append(
            f"| {tid} | {tc['dominant_person_id']} | {tc['correct']} | "
            f"{tc['wrong']} | {tc['no_id']} | {tc['no_det']} | "
            f"{tc['purity']:.1%} |"
        )

    (out_dir / "_aggregate_d.md").write_text("\n".join(md) + "\n")
    return out_dir
