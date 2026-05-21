#!/usr/bin/env python3
"""CP7-pre-5: GT-as-Stage-A ceiling run (FP7oJQ).

Injects ground-truth per-person bboxes AS Stage A output, runs D->E unmodified,
then scores with gt_person_trace to isolate whether D->F produces correct
identities given perfect detection input.

Scope: FP7oJQ only, frames 0-300 (dense, stride 1).
Tracklet identity: GT track_id threaded as tracklet_id (pure ceiling arm).
Stage C: empty (no tags) -- purest structural read.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data/training_data/training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip"
HOMOGRAPHY_PATH = REPO_ROOT / "configs/cameras/FP7oJQ/homography.json"
PRODUCTION_DET_PATH = (
    REPO_ROOT / "outputs/_eval_gt/FP7oJQ/2026-03-18/20"
    "/FP7oJQ-20260318-200014/stage_A/detections.parquet"
)
INGEST_CLIP = (
    REPO_ROOT / "data/raw/nest/_eval_gt/FP7oJQ/2026-03-18/20"
    "/FP7oJQ-20260318-200014.mp4"
)

CLIP_ID = "FP7oJQ-20260318-200014"
CAMERA_ID = "FP7oJQ"
MODEL_ID = "gt-ceiling"
RESOLUTION = (1920, 1080)
FRAME_RANGE = range(0, 301)  # 0-300 inclusive

SANDBOX_OUT_ROOT = REPO_ROOT / "outputs/_gt_ceiling"
EVAL_DIR = REPO_ROOT / "outputs/_eval"


def sandbox_clip_dir() -> Path:
    return SANDBOX_OUT_ROOT / "_eval_gt/FP7oJQ/2026-03-18/20" / CLIP_ID


# ---------------------------------------------------------------------------
# GT label loading
# ---------------------------------------------------------------------------

def load_gt_labels() -> dict[int, list[dict]]:
    """Load GT labels from zip for frames 0-300. Returns {frame_idx: [box_dicts]}."""
    img_w, img_h = RESOLUTION
    gt: dict[int, list[dict]] = {}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.endswith(".txt") and "frame_" in n
        )
        for name in names:
            # Extract frame index from filename
            stem = Path(name).stem  # e.g. "frame_000042"
            fidx = int(stem.split("_")[-1])
            if fidx not in FRAME_RANGE:
                continue

            content = zf.read(name).decode().strip()
            if not content:
                gt[fidx] = []
                continue

            boxes = []
            for line in content.split("\n"):
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
                track_id = int(parts[5])

                if cls_id == 1:
                    cls_id = 0

                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h

                boxes.append({
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
            gt[fidx] = boxes

    # Ensure every frame in range has an entry
    for fidx in FRAME_RANGE:
        if fidx not in gt:
            gt[fidx] = []

    return gt


# ---------------------------------------------------------------------------
# Homography + projection
# ---------------------------------------------------------------------------

def load_homography() -> np.ndarray:
    with open(HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.asarray(data["H"], dtype=np.float64).reshape((3, 3))


def project_to_world(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Project pixel (u, v) to world coords via homography.
    FP7oJQ has no lens calibration or correction matrix -- H only.
    """
    q = H @ np.array([u, v, 1.0], dtype=np.float64)
    return float(q[0] / q[2]), float(q[1] / q[2])


# ---------------------------------------------------------------------------
# Timestamp mapping from production
# ---------------------------------------------------------------------------

def load_timestamp_map() -> dict[int, int]:
    """Load frame_index -> timestamp_ms from production Stage A detections."""
    det = pd.read_parquet(PRODUCTION_DET_PATH, columns=["frame_index", "timestamp_ms"])
    ts = det.groupby("frame_index")["timestamp_ms"].first()
    return {int(k): int(v) for k, v in ts.items()}


# ---------------------------------------------------------------------------
# Synthetic Stage A artifact generation
# ---------------------------------------------------------------------------

def _sorted_det_key(box: dict) -> tuple:
    """Match production sort: left-to-right, top-to-bottom."""
    return (box["x1"], box["y1"], box["x2"], box["y2"])


def generate_stage_a_artifacts(
    gt: dict[int, list[dict]],
    H: np.ndarray,
    ts_map: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate detections, tracklet_frames, tracklet_summaries from GT."""

    det_rows = []
    tf_rows = []
    track_frames: dict[int, list[dict]] = defaultdict(list)

    for fidx in sorted(gt.keys()):
        boxes = sorted(gt[fidx], key=_sorted_det_key)
        ts_ms = ts_map.get(fidx, int(fidx * 33.333))

        for k, box in enumerate(boxes):
            det_id = f"d{fidx:06d}_{k}"
            tid = f"t{box['track_id']}"

            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Contact point: bottom-center of bbox
            u_px = (x1 + x2) / 2.0
            v_px = y2
            x_m, y_m = project_to_world(u_px, v_px, H)

            det_rows.append({
                "clip_id": CLIP_ID,
                "camera_id": CAMERA_ID,
                "frame_index": fidx,
                "timestamp_ms": ts_ms,
                "detection_id": det_id,
                "class_name": "person",
                "confidence": 1.0,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "tracklet_id": tid,
                "mask_ref": None,
                "mask_source": None,
                "mask_quality": None,
                "source": None,
                "debug_json": None,
            })

            tf_rows.append({
                "clip_id": CLIP_ID,
                "camera_id": CAMERA_ID,
                "tracklet_id": tid,
                "frame_index": fidx,
                "timestamp_ms": ts_ms,
                "detection_id": det_id,
                "local_track_conf": 1.0,
                "u_px": u_px,
                "v_px": v_px,
                "x_m": x_m,
                "y_m": y_m,
                "vx_m": 0.0,
                "vy_m": 0.0,
                "on_mat": True,
                "contact_conf": 1.0,
                "contact_method": "bbox",
            })

            track_frames[box["track_id"]].append({
                "frame_index": fidx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    # Build detections DataFrame with production dtypes
    det_df = pd.DataFrame(det_rows)
    det_df["frame_index"] = det_df["frame_index"].astype("Int64")
    det_df["timestamp_ms"] = det_df["timestamp_ms"].astype("Int64")
    det_df["confidence"] = det_df["confidence"].astype("Float64")
    for col in ("x1", "y1", "x2", "y2"):
        det_df[col] = det_df[col].astype("Float64")

    # Build tracklet_frames with production dtypes
    tf_df = pd.DataFrame(tf_rows)
    tf_df["frame_index"] = tf_df["frame_index"].astype("Int64")
    tf_df["timestamp_ms"] = tf_df["timestamp_ms"].astype("Int64")
    for col in ("local_track_conf", "u_px", "v_px", "x_m", "y_m",
                "vx_m", "vy_m", "contact_conf"):
        tf_df[col] = tf_df[col].astype("Float64")
    tf_df["on_mat"] = tf_df["on_mat"].astype("boolean")

    # Build tracklet_summaries
    summary_rows = []
    for track_id, frames in sorted(track_frames.items()):
        fids = [f["frame_index"] for f in frames]
        summary_rows.append({
            "clip_id": CLIP_ID,
            "camera_id": CAMERA_ID,
            "tracklet_id": f"t{track_id}",
            "start_frame": min(fids),
            "end_frame": max(fids),
            "n_frames": len(fids),
            "mean_x1": np.mean([f["x1"] for f in frames]),
            "mean_y1": np.mean([f["y1"] for f in frames]),
            "mean_x2": np.mean([f["x2"] for f in frames]),
            "mean_y2": np.mean([f["y2"] for f in frames]),
            "quality_score": 1.0,
            "reason_codes_json": "[]",
        })

    ts_df = pd.DataFrame(summary_rows)
    ts_df["start_frame"] = ts_df["start_frame"].astype("Int64")
    ts_df["end_frame"] = ts_df["end_frame"].astype("Int64")
    ts_df["n_frames"] = ts_df["n_frames"].astype("Int64")
    for col in ("mean_x1", "mean_y1", "mean_x2", "mean_y2", "quality_score"):
        ts_df[col] = ts_df[col].astype("Float64")

    return det_df, tf_df, ts_df


# ---------------------------------------------------------------------------
# Write artifacts
# ---------------------------------------------------------------------------

def write_stage_a(det_df: pd.DataFrame, tf_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    stage_a = sandbox_clip_dir() / "stage_A"
    stage_a.mkdir(parents=True, exist_ok=True)

    det_df.to_parquet(stage_a / "detections.parquet", index=False)
    tf_df.to_parquet(stage_a / "tracklet_frames.parquet", index=False)
    ts_df.to_parquet(stage_a / "tracklet_summaries.parquet", index=False)

    logger.info(
        "Wrote Stage A: %d detections, %d tracklet_frames, %d summaries",
        len(det_df), len(tf_df), len(ts_df),
    )


def write_stage_c() -> None:
    stage_c = sandbox_clip_dir() / "stage_C"
    stage_c.mkdir(parents=True, exist_ok=True)
    (stage_c / "identity_hints.jsonl").write_text("")
    logger.info("Wrote empty Stage C identity_hints.jsonl")


# ---------------------------------------------------------------------------
# Run D->E via pipeline CLI
# ---------------------------------------------------------------------------

def run_pipeline_d_to_e() -> bool:
    cmd = [
        sys.executable, "-m", "bjj_pipeline.stages.orchestration.cli", "run",
        "--clip", str(INGEST_CLIP),
        "--camera", CAMERA_ID,
        "--from-stage", "D",
        "--to-stage", "E",
        "--force",
        "--out", str(SANDBOX_OUT_ROOT),
    ]
    logger.info("Running pipeline D->E: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Pipeline failed (rc=%d):\nSTDOUT:\n%s\nSTDERR:\n%s",
                      result.returncode, result.stdout[-2000:], result.stderr[-2000:])
        return False
    logger.info("Pipeline D->E completed successfully")
    return True


# ---------------------------------------------------------------------------
# Generate per_frame_matches (GT matches itself -- presence gate only)
# ---------------------------------------------------------------------------

def generate_per_frame_matches(
    det_df: pd.DataFrame,
    gt: dict[int, list[dict]],
) -> pd.DataFrame:
    """Build per_frame_matches where each GT box matches its injected detection.

    This is NOT circular: per_frame_matches is the presence gate only.
    Identity determination flows through D4 person_tracks + identity_mapping.
    """
    records = []
    for fidx in sorted(gt.keys()):
        boxes = sorted(gt[fidx], key=_sorted_det_key)
        for k, box in enumerate(boxes):
            det_id = f"d{fidx:06d}_{k}"
            # Determine split (matches production manifest)
            split = "train" if fidx <= 249 else "val"

            records.append({
                "model_id": MODEL_ID,
                "camera_id": CAMERA_ID,
                "split": split,
                "frame_index": fidx,
                "gt_track_id": float(box["track_id"]),
                "gt_x1": box["x1"],
                "gt_y1": box["y1"],
                "gt_x2": box["x2"],
                "gt_y2": box["y2"],
                "pred_detection_id": det_id,
                "pred_x1": box["x1"],
                "pred_y1": box["y1"],
                "pred_x2": box["x2"],
                "pred_y2": box["y2"],
                "iou": 1.0,
                "match_status": "matched",
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generate identity_mapping from D4 output (run-local, plurality rule)
# ---------------------------------------------------------------------------

def generate_identity_mapping(
    det_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Derive identity_mapping from THIS run's D4 person_tracks.

    Returns (mapping_dict, collapse_audit).
    mapping_dict: {gt_track_id_int: {canonical_person_id, purity, ...}}
    collapse_audit: {many_to_one: [...], one_to_many: [...]}
    """
    pt_path = sandbox_clip_dir() / "stage_D" / "person_tracks.parquet"
    if not pt_path.exists():
        raise FileNotFoundError(f"D4 person_tracks not found: {pt_path}")

    pt_df = pd.read_parquet(pt_path)

    # Build detection_id -> person_id mapping from D4
    det_to_person = dict(zip(pt_df["detection_id"], pt_df["person_id"]))

    # Build tracklet_id -> gt_track_id from our injected detections
    # tracklet_id = "t{track_id}", so gt_track_id = int(tracklet_id[1:])
    tid_to_gt = {}
    for _, row in det_df.iterrows():
        tid = row["tracklet_id"]
        if tid and tid not in tid_to_gt:
            tid_to_gt[tid] = int(tid[1:])  # "t14" -> 14

    # For each GT track, collect person_ids across its frames
    gt_track_persons: dict[int, list[tuple[int, str | None]]] = defaultdict(list)

    for _, row in det_df.iterrows():
        det_id = row["detection_id"]
        tid = row["tracklet_id"]
        gt_tid = tid_to_gt.get(tid)
        if gt_tid is None:
            continue
        pid = det_to_person.get(det_id)
        gt_track_persons[gt_tid].append((int(row["frame_index"]), pid))

    # Compute identity_mapping: plurality rule with earliest-frame tiebreak
    mapping: dict[int, dict] = {}
    for gt_tid, frame_pids in sorted(gt_track_persons.items()):
        person_ids = [pid for _, pid in frame_pids if pid is not None]
        if not person_ids:
            mapping[gt_tid] = {
                "canonical_person_id": None,
                "purity": 0.0,
                "frames_matched": 0,
                "frames_total": len(frame_pids),
            }
            continue

        counts = Counter(person_ids)
        earliest: dict[str, int] = {}
        for fidx, pid in frame_pids:
            if pid is not None and pid not in earliest:
                earliest[pid] = fidx

        canonical = min(
            counts.keys(),
            key=lambda pid: (-counts[pid], earliest.get(pid, 0)),
        )
        purity = counts[canonical] / len(person_ids)

        mapping[gt_tid] = {
            "canonical_person_id": canonical,
            "purity": purity,
            "frames_matched": len(person_ids),
            "frames_total": len(frame_pids),
        }

    # Identity-collapse audit
    # Many-to-one: person_id absorbing >= 2 GT tracks
    pid_to_gt_tracks: dict[str, list[int]] = defaultdict(list)
    for gt_tid, m in mapping.items():
        cpid = m["canonical_person_id"]
        if cpid is not None:
            pid_to_gt_tracks[cpid].append(gt_tid)

    many_to_one = [
        {"person_id": pid, "gt_tracks": sorted(tids), "count": len(tids)}
        for pid, tids in sorted(pid_to_gt_tracks.items())
        if len(tids) >= 2
    ]

    # One-to-many: GT track split across >= 2 person_ids
    one_to_many = []
    for gt_tid, frame_pids in sorted(gt_track_persons.items()):
        person_ids = [pid for _, pid in frame_pids if pid is not None]
        unique_pids = set(person_ids)
        if len(unique_pids) >= 2:
            pid_counts = Counter(person_ids)
            one_to_many.append({
                "gt_track_id": gt_tid,
                "person_ids": {pid: cnt for pid, cnt in pid_counts.most_common()},
                "count": len(unique_pids),
            })

    collapse_audit = {
        "many_to_one": many_to_one,
        "one_to_many": one_to_many,
    }

    return mapping, collapse_audit


# ---------------------------------------------------------------------------
# Write eval artifacts + run gt_person_trace
# ---------------------------------------------------------------------------

def write_eval_artifacts(
    pfm_df: pd.DataFrame,
    identity_mapping: dict[int, dict],
) -> None:
    """Write per_frame_matches and identity_mapping to eval dirs."""
    # per_frame_matches
    pfm_dir = EVAL_DIR / "stage_a" / MODEL_ID / CAMERA_ID
    pfm_dir.mkdir(parents=True, exist_ok=True)
    pfm_df.to_parquet(pfm_dir / "per_frame_matches.parquet", index=False)
    logger.info("Wrote per_frame_matches: %d rows", len(pfm_df))

    # identity_mapping
    idm_dir = EVAL_DIR / "stage_d" / MODEL_ID / CAMERA_ID
    idm_dir.mkdir(parents=True, exist_ok=True)
    mapping_out = {
        f"gt_track_{tid}": m for tid, m in identity_mapping.items()
    }
    (idm_dir / "identity_mapping.json").write_text(
        json.dumps(mapping_out, indent=2)
    )
    logger.info("Wrote identity_mapping: %d GT tracks", len(identity_mapping))


def run_gt_person_trace() -> pd.DataFrame:
    """Run gt_person_trace and return trace DataFrame."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pipeline_validation.gt_person_trace import compute_gt_person_trace

    result = compute_gt_person_trace(
        eval_dir=EVAL_DIR,
        model_id=MODEL_ID,
        camera_id=CAMERA_ID,
        pipeline_clip_dir=sandbox_clip_dir(),
    )

    # Write outputs
    out_dir = EVAL_DIR / "stage_d" / MODEL_ID / CAMERA_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trace_df.to_parquet(out_dir / "gt_person_trace.parquet", index=False)
    result.person_summary_df.to_parquet(out_dir / "gt_person_summary.parquet", index=False)
    (out_dir / "gt_camera_summary.json").write_text(
        json.dumps(result.camera_summary, indent=2)
    )

    if result.warnings:
        logger.warning("Trace warnings: %s", result.warnings)

    return result.trace_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def check_solver_status() -> str:
    """Check D3 solution ledger for OPTIMAL status."""
    ledger_path = sandbox_clip_dir() / "_debug" / "d3_solution_ledger.json"
    if not ledger_path.exists():
        ledger_path = sandbox_clip_dir() / "stage_D" / "d3_solution_ledger.json"
    if not ledger_path.exists():
        return "LEDGER_NOT_FOUND"
    with open(ledger_path) as f:
        ledger = json.load(f)
    # Status lives inside objective dict
    obj = ledger.get("objective", {})
    if isinstance(obj, dict) and "status" in obj:
        return obj["status"]
    return ledger.get("status", ledger.get("solver_status", "UNKNOWN"))


def load_production_baseline() -> dict | None:
    """Load CP5 production baseline trace for comparison."""
    baseline_path = (
        EVAL_DIR / "stage_d" / "bjj-detect-all-cameras" / "FP7oJQ"
        / "gt_person_trace.parquet"
    )
    if not baseline_path.exists():
        return None
    df = pd.read_parquet(baseline_path)
    total = len(df)
    counts = df["failure_mode"].value_counts()
    return {
        mode: {"n": int(counts.get(mode, 0)),
               "pct": float(counts.get(mode, 0)) / total if total > 0 else 0.0}
        for mode in [
            "present", "stage_a_no_detection", "stage_a_untracked",
            "d3_dropped", "d4_unassigned", "present_misattributed",
            "missing_canonical",
        ]
    }


def generate_report(
    trace_df: pd.DataFrame,
    collapse_audit: dict,
    identity_mapping: dict[int, dict],
    gt: dict[int, list[dict]],
    solver_status: str,
) -> str:
    """Generate the results document."""
    total = len(trace_df)
    counts = trace_df["failure_mode"].value_counts()

    modes = [
        "present", "stage_a_no_detection", "stage_a_untracked",
        "d3_dropped", "d4_unassigned", "present_misattributed",
        "missing_canonical",
    ]

    # Conservation check per GT track
    conservation_ok = True
    conservation_details = []
    for gt_pid, grp in trace_df.groupby("gt_person_id"):
        n_total = len(grp)
        n_modes = int(grp["failure_mode"].value_counts().sum())
        ok = n_total == n_modes
        if not ok:
            conservation_ok = False
        conservation_details.append(
            f"  gt_track_{gt_pid}: {n_total} frames, {n_modes} mode entries -> {'OK' if ok else 'FAIL'}"
        )

    # Production baseline comparison
    baseline = load_production_baseline()

    # Per-GT-track summary
    per_track_lines = []
    for gt_pid, grp in trace_df.groupby("gt_person_id"):
        n = len(grp)
        mc = grp["failure_mode"].value_counts()
        canonical = identity_mapping.get(gt_pid, {}).get("canonical_person_id", "None")
        purity = identity_mapping.get(gt_pid, {}).get("purity", 0.0)
        per_track_lines.append(
            f"| gt_track_{gt_pid} | {canonical} | {purity:.3f} | {n} | "
            f"{int(mc.get('present', 0))} | {int(mc.get('present_misattributed', 0))} | "
            f"{int(mc.get('d3_dropped', 0))} | {int(mc.get('d4_unassigned', 0))} |"
        )

    # Count GT boxes across all frames
    total_gt_boxes = sum(len(boxes) for boxes in gt.values())

    lines = [
        "# CP7-pre-5: GT-as-Stage-A Ceiling Run (FP7oJQ)",
        "",
        "## Setup",
        "",
        f"- **Scope:** FP7oJQ, frames 0-300 (dense, stride 1), {len(FRAME_RANGE)} frames",
        f"- **GT tracks:** 14 (track_ids 14-27)",
        f"- **Total GT boxes:** {total_gt_boxes}",
        f"- **Tracklet identity:** GT track_id threaded as tracklet_id (pure ceiling arm)",
        f"- **Stage C:** Empty (no tags)",
        f"- **Solver status:** {solver_status}",
        f"- **Reference point:** bottom-center of bbox, H-only projection (no lens cal)",
        "",
        "## Headline: Six-Mode Failure Breakdown",
        "",
        "| Mode | N | % |",
        "|------|---|---|",
    ]
    for mode in modes:
        n = int(counts.get(mode, 0))
        pct = n / total * 100 if total > 0 else 0.0
        lines.append(f"| {mode} | {n} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{total}** | **100.0%** |")

    if baseline:
        lines.extend([
            "",
            "### Comparison with CP5 Production Baseline (FP7oJQ)",
            "",
            "| Mode | GT-Ceiling % | Production % | Delta |",
            "|------|-------------|-------------|-------|",
        ])
        for mode in modes:
            gt_n = int(counts.get(mode, 0))
            gt_pct = gt_n / total * 100 if total > 0 else 0.0
            prod_pct = baseline[mode]["pct"] * 100
            delta = gt_pct - prod_pct
            lines.append(f"| {mode} | {gt_pct:.1f}% | {prod_pct:.1f}% | {delta:+.1f}% |")

    lines.extend([
        "",
        "Re-baseline against a CP5-state full-mode snapshot. When reading the post-CP7",
        "six-mode shift, treat any movement in a metric that was stable across CP0-CP5",
        "as expected-until-explained, then run the conservation check before trusting",
        "the magnitude.",
        "",
        "## Identity-Collapse Audit",
        "",
        "### Many-to-One (person_id absorbing >= 2 GT tracks)",
        "",
    ])

    if collapse_audit["many_to_one"]:
        lines.append("| Person ID | GT Tracks | Count |")
        lines.append("|-----------|-----------|-------|")
        for item in collapse_audit["many_to_one"]:
            gt_tracks_str = ", ".join(f"gt_track_{t}" for t in item["gt_tracks"])
            lines.append(f"| {item['person_id']} | {gt_tracks_str} | {item['count']} |")
    else:
        lines.append("**None** -- every person_id maps to exactly one GT track.")

    lines.extend([
        "",
        "### One-to-Many (GT track split across >= 2 person_ids)",
        "",
    ])

    if collapse_audit["one_to_many"]:
        lines.append("| GT Track | Person IDs (frames) | Count |")
        lines.append("|----------|-------------------|-------|")
        for item in collapse_audit["one_to_many"]:
            pids_str = ", ".join(
                f"{pid}({cnt})" for pid, cnt in item["person_ids"].items()
            )
            lines.append(f"| gt_track_{item['gt_track_id']} | {pids_str} | {item['count']} |")
    else:
        lines.append("**None** -- every GT track maps to exactly one person_id.")

    lines.extend([
        "",
        "### Summary",
        "",
        f"- Many-to-one cases: {len(collapse_audit['many_to_one'])}",
        f"- One-to-many cases: {len(collapse_audit['one_to_many'])}",
    ])

    if not collapse_audit["many_to_one"] and not collapse_audit["one_to_many"]:
        lines.append("- **D->F is sound on perfect input: zero identity collapse.**")
    else:
        lines.append(
            "- **D->F has structural identity issues even on perfect input.**"
        )

    lines.extend([
        "",
        "## Per-GT-Track Detail",
        "",
        "| GT Track | Canonical PID | Purity | Frames | Present | Misattrib | D3 Drop | D4 Unasgn |",
        "|----------|--------------|--------|--------|---------|-----------|---------|-----------|",
    ])
    lines.extend(per_track_lines)

    lines.extend([
        "",
        "## Conservation Check",
        "",
        f"**Status:** {'PASS' if conservation_ok else 'FAIL'}",
        "",
        "Six modes sum to total GT frames per track:",
        "",
    ])
    lines.extend(conservation_details)

    # RIDER: GT person count at production misattribution locations
    lines.extend([
        "",
        "## RIDER: GT Person Count at Pair-Box Locations",
        "",
        "For the production pair-boxes that caused misattribution in CP7-pre-3,",
        "the GT-injected run shows whether each location has ONE GT person or TWO.",
        "",
        "Since GT tracklet_ids are unique per GT person and threaded 1:1, every",
        "detection in the GT-ceiling run corresponds to exactly one GT person.",
        "If D1 creates GROUP nodes merging two GT tracklets, that means two real",
        "GT people were spatially close enough to trigger a merge event -- confirming",
        "the second person was REAL, not a duplicate detection.",
        "",
    ])

    # Check D1 for GROUP nodes
    d1_nodes_path = sandbox_clip_dir() / "stage_D" / "d1_graph_nodes.parquet"
    if d1_nodes_path.exists():
        d1_nodes = pd.read_parquet(d1_nodes_path)
        group_nodes = d1_nodes[d1_nodes["node_type"].str.contains("GROUP", na=False)]
        lines.append(f"D1 GROUP nodes in GT-ceiling run: {len(group_nodes)}")
        if len(group_nodes) > 0:
            lines.append("")
            lines.append("| Node ID | Type | Carrier | Disappearing | New | Start | End |")
            lines.append("|---------|------|---------|-------------|-----|-------|-----|")
            for _, row in group_nodes.iterrows():
                lines.append(
                    f"| {row['node_id']} | {row['node_type']} | "
                    f"{row.get('carrier_tracklet_id', '')} | "
                    f"{row.get('disappearing_tracklet_id', '')} | "
                    f"{row.get('new_tracklet_id', '')} | "
                    f"{int(row['start_frame'])} | {int(row['end_frame'])} |"
                )
            lines.extend([
                "",
                "Each GROUP node involves two distinct GT people (by construction --",
                "GT tracklets are identity-pure). These are REAL spatial overlaps,",
                "not duplicate detections.",
            ])
        else:
            lines.extend([
                "",
                "No GROUP nodes -- GT tracks never triggered merge/split events.",
                "In the production run, GROUP events at these locations were caused",
                "by tracker-assigned tracklet fragmentation, not by real person overlap.",
            ])

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    misattrib_pct = float(counts.get("present_misattributed", 0)) / total * 100 if total > 0 else 0.0
    if misattrib_pct < 1.0:
        lines.extend([
            f"**present_misattributed = {misattrib_pct:.1f}%** (near-zero).",
            "",
            "D->F produces correct identities given perfect detection input.",
            "Detection is confirmed as the lever for CP7. The 59-66% misattribution",
            "in the production run is caused by detection under-segmentation and",
            "tracker fragmentation, not by Stage D routing bugs.",
        ])
    else:
        lines.extend([
            f"**present_misattributed = {misattrib_pct:.1f}%** (nonzero).",
            "",
            "D->F has a routing bug independent of detection quality.",
            "Quantify which stage: check D1 topology (GROUP node count),",
            "D3 routing (dropped tracklets), D4 emission (person_id assignment).",
        ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== CP7-pre-5: GT-as-Stage-A Ceiling Run (FP7oJQ) ===")

    # Step 1: Load GT labels
    logger.info("Loading GT labels from zip...")
    gt = load_gt_labels()
    total_boxes = sum(len(boxes) for boxes in gt.values())
    track_ids = set()
    for boxes in gt.values():
        for box in boxes:
            track_ids.add(box["track_id"])
    logger.info(
        "Loaded %d frames, %d total boxes, %d unique tracks: %s",
        len(gt), total_boxes, len(track_ids), sorted(track_ids),
    )

    # Step 2: Load homography + timestamp map
    logger.info("Loading homography and timestamp map...")
    H = load_homography()
    ts_map = load_timestamp_map()

    # Step 3-4: Generate synthetic Stage A artifacts
    logger.info("Generating synthetic Stage A artifacts...")
    det_df, tf_df, ts_df = generate_stage_a_artifacts(gt, H, ts_map)

    # Step 5-6: Write to sandbox
    write_stage_a(det_df, tf_df, ts_df)
    write_stage_c()

    # Step 7: Run D->E
    logger.info("Running pipeline D->E...")
    ok = run_pipeline_d_to_e()
    if not ok:
        logger.error("Pipeline D->E failed. Aborting.")
        sys.exit(1)

    # Check solver status
    solver_status = check_solver_status()
    logger.info("Solver status: %s", solver_status)

    # Step 8: Generate per_frame_matches
    logger.info("Generating per_frame_matches...")
    pfm_df = generate_per_frame_matches(det_df, gt)

    # Step 9: Generate identity_mapping
    logger.info("Generating identity_mapping from D4 output...")
    identity_mapping, collapse_audit = generate_identity_mapping(det_df)

    logger.info("Identity mapping:")
    for gt_tid, m in sorted(identity_mapping.items()):
        logger.info(
            "  gt_track_%d -> %s (purity=%.3f, %d/%d frames)",
            gt_tid, m["canonical_person_id"], m["purity"],
            m["frames_matched"], m["frames_total"],
        )

    logger.info("Collapse audit:")
    logger.info("  Many-to-one: %d cases", len(collapse_audit["many_to_one"]))
    for item in collapse_audit["many_to_one"]:
        logger.info("    %s absorbs %s", item["person_id"], item["gt_tracks"])
    logger.info("  One-to-many: %d cases", len(collapse_audit["one_to_many"]))
    for item in collapse_audit["one_to_many"]:
        logger.info("    gt_track_%d -> %s", item["gt_track_id"], item["person_ids"])

    # Step 10: Write eval artifacts + run trace
    write_eval_artifacts(pfm_df, identity_mapping)
    logger.info("Running gt_person_trace...")
    trace_df = run_gt_person_trace()

    # Step 11: Report
    logger.info("Generating report...")
    report = generate_report(trace_df, collapse_audit, identity_mapping, gt, solver_status)

    doc_path = REPO_ROOT / "docs" / "cp7_pre5_gt_ceiling_run.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    logger.info("Report written to %s", doc_path)

    # Print headline
    total = len(trace_df)
    counts = trace_df["failure_mode"].value_counts()
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    for mode in [
        "present", "stage_a_no_detection", "stage_a_untracked",
        "d3_dropped", "d4_unassigned", "present_misattributed",
        "missing_canonical",
    ]:
        n = int(counts.get(mode, 0))
        pct = n / total * 100 if total > 0 else 0.0
        print(f"  {mode:30s} {n:5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s} {total:5d}  (100.0%)")
    print()
    print(f"Solver: {solver_status}")
    print(f"Many-to-one collapse: {len(collapse_audit['many_to_one'])} cases")
    print(f"One-to-many collapse: {len(collapse_audit['one_to_many'])} cases")
    print("=" * 60)


if __name__ == "__main__":
    main()
