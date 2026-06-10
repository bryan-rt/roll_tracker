"""CP-SPLIT-VALIDATE: GT-validate D0.5 splits and characterize post-V Tier-3 explosion.

Measurement only. No production code changes.

Phases:
  0 — Reconstruct pre-V (144) and post-V (864) split audits
  1 — GT-validate all splits (correct / spurious / undecidable)
  2 — Characterize spurious T3 splits (motion-correlated shadow vs real swap)
  3 — Threshold-sweep counterfactual
  4 — k-distribution for impure tracklets
  5 — Clean-point change-point feasibility probe

Usage:
    PYTHONPATH=src python tools/cp_split_validate.py
    PYTHONPATH=src python tools/cp_split_validate.py --skip-reconstruct  # reuse existing audits
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline_validation.common.manifest import (
    enumerate_annotated_frames,
    load_manifest as load_model_manifest,
)
from pipeline_validation.signal_trace.greedy_matcher import greedy_match
from pipeline_validation.signal_trace.stage_a_census import _load_gt_all_annotated

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_split_validate"
WORK_DIR = OUTPUTS_DIR / "_eval_gt_oracle" / "split_validate"

CAM_ID = "J_EDEw"
GYM_ID = "_eval_gt"
DENSE_MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"

VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID1_CLIP_ID
VID2_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID2_CLIP_ID

IOU_THRESHOLD = 0.3
SEED = 42

# Threshold sweep values
THRESHOLD_SWEEP = [0.15, 0.18, 0.22, 0.25]


# ---------------------------------------------------------------------------
# Phase 0: Reconstruct pre/post split audits
# ---------------------------------------------------------------------------

def _run_d0_d05(clip_id: str) -> Dict[str, int]:
    """Run D0→D0.5 for a clip and return tier counts from the audit."""
    import yaml
    from bjj_pipeline.contracts.f0_manifest import load_manifest as load_clip_manifest
    from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
    from bjj_pipeline.stages.stitch.d0_bank import run_d0
    from bjj_pipeline.stages.stitch.d05_split import run_d05_split

    base = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / clip_id
    layout = ClipOutputLayout(clip_id=clip_id, root=base.parent)
    manifest = load_clip_manifest(base / "clip_manifest.json")

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())

    # Run D0
    layout.ensure_dirs_for_stage("D")
    run_d0(config=cfg, layout=layout, manifest=manifest)

    # Truncate audit before D0.5
    audit_path = layout.stage_dir("D") / "d05_split_audit.jsonl"
    audit_path.write_text("", encoding="utf-8")

    # Run D0.5
    run_d05_split(config=cfg, layout=layout, manifest=manifest)

    # Parse audit
    tiers: Dict[str, int] = {}
    splits = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("artifact_type") == "d05_split_event":
            t = ev.get("tier", "unknown")
            tiers[t] = tiers.get(t, 0) + 1
            splits.append(ev)

    return tiers, splits


def _extract_histograms_for_clip(clip_id: str) -> None:
    """Re-extract histograms for a clip using the CURRENT histogram.py."""
    from bjj_pipeline.stages.detect_track.histogram import (
        HIST_SIZE,
        extract_histogram,
        compute_tracklet_histogram_summary,
    )
    from bjj_pipeline.stages.detect_track.isolation import compute_isolation_flags

    base = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / clip_id
    det = pd.read_parquet(base / "stage_A" / "detections.parquet")
    clip_dir = REPO_ROOT / "data" / "raw" / "nest" / "c8a592a4-2bca-400a-80e1-fec0e5cbea77" / CAM_ID / "2026-03-18" / "20"
    video_path = clip_dir / f"{clip_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))

    hist_rows = []
    per_tracklet: Dict[str, Dict] = {}

    for fi, frame_dets in det.groupby("frame_index"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue

        bboxes = frame_dets[["x1", "y1", "x2", "y2"]].values.tolist()
        iso_flags = compute_isolation_flags(
            bboxes=bboxes, keypoints_list=[None] * len(bboxes),
            config={"require_keypoints": False},
        )

        for det_idx, (_, row) in enumerate(frame_dets.iterrows()):
            is_iso = iso_flags[det_idx] if det_idx < len(iso_flags) else False
            bbox = (row.x1, row.y1, row.x2, row.y2)
            hist, method = extract_histogram(
                frame_bgr=frame, bbox=bbox, keypoints=None, is_isolated=is_iso,
            )

            hr = {
                "frame_index": int(fi),
                "track_id": row.tracklet_id,
                "is_isolated": bool(is_iso),
                "crop_method": method,
            }
            if hist is not None:
                for i in range(HIST_SIZE):
                    hr[f"hist_{i}"] = float(hist[i])
            else:
                for i in range(HIST_SIZE):
                    hr[f"hist_{i}"] = float("nan")
            hist_rows.append(hr)

            tid = row.tracklet_id
            if tid not in per_tracklet:
                per_tracklet[tid] = {"histograms": [], "crop_methods": []}
            if hist is not None and is_iso:
                per_tracklet[tid]["histograms"].append(hist)
                per_tracklet[tid]["crop_methods"].append(method)

    cap.release()

    hist_df = pd.DataFrame(hist_rows)
    hist_df.to_parquet(base / "stage_A" / "color_histograms.parquet", index=False)

    summary_rows = []
    for tid, data in sorted(per_tracklet.items()):
        avg, n, method_dist = compute_tracklet_histogram_summary(
            data["histograms"], data["crop_methods"]
        )
        sr = {"tracklet_id": tid, "camera_id": CAM_ID, "clip_id": clip_id,
              "n_isolated_frames": n,
              "crop_method_distribution_json": json.dumps(method_dist, sort_keys=True)}
        from bjj_pipeline.stages.detect_track.histogram import HIST_SIZE as HS
        if avg is not None:
            for i in range(HS):
                sr[f"hist_{i}"] = float(avg[i])
        else:
            for i in range(HS):
                sr[f"hist_{i}"] = float("nan")
        summary_rows.append(sr)
    pd.DataFrame(summary_rows).to_parquet(
        base / "stage_A" / "tracklet_histogram_summaries.parquet", index=False
    )
    logger.info("  Extracted {} histograms ({}-dim) for {}", len(hist_rows), HIST_SIZE, clip_id)


def reconstruct_split_audits() -> Dict[str, Any]:
    """Phase 0: Reconstruct pre-V (144) and post-V (864) split audits."""
    logger.info("=== Phase 0: Reconstructing split audits ===")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    hist_path = REPO_ROOT / "src" / "bjj_pipeline" / "stages" / "detect_track" / "histogram.py"
    new_hist = hist_path.read_text(encoding="utf-8")

    for label, setup_fn in [
        ("pre_v", lambda: _revert_to_144()),
        ("post_v", lambda: _restore_864(new_hist)),
    ]:
        logger.info("--- {} ---", label)
        setup_fn()

        # Re-extract histograms + run D0.5 for both clips
        clip_results = {}
        for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
            logger.info("  Extracting histograms for {}...", clip_id)
            _extract_histograms_for_clip(clip_id)
            logger.info("  Running D0→D0.5 for {}...", clip_id)
            tiers, splits = _run_d0_d05(clip_id)
            clip_results[clip_id] = {"tiers": tiers, "splits": splits}
            logger.info("  {}: {}", clip_id, tiers)

        # Save audit
        audit_path = WORK_DIR / f"{label}_splits.json"
        audit_path.write_text(json.dumps(clip_results, indent=2, default=str), encoding="utf-8")
        results[label] = clip_results

    # Ensure we end with 864-dim (production state)
    _restore_864(new_hist)
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        _extract_histograms_for_clip(clip_id)

    return results


def _revert_to_144():
    """Temporarily revert histogram.py to H+S only (144-dim)."""
    hist_path = REPO_ROOT / "src" / "bjj_pipeline" / "stages" / "detect_track" / "histogram.py"
    content = hist_path.read_text(encoding="utf-8")
    # Replace HIST_V_BINS and HIST_SIZE
    content = content.replace("HIST_V_BINS = 6\n", "HIST_V_BINS = 1\n")
    content = content.replace(
        "HIST_SIZE = HIST_H_BINS * HIST_S_BINS * HIST_V_BINS  # 864",
        "HIST_SIZE = HIST_H_BINS * HIST_S_BINS  # 144",
    )
    # Replace calcHist to use only H+S
    content = content.replace(
        "        [hsv], [0, 1, 2], None,\n"
        "        [HIST_H_BINS, HIST_S_BINS, HIST_V_BINS],\n"
        "        [0, 180, 0, 256, 0, 256],",
        "        [hsv], [0, 1], None,\n"
        "        [HIST_H_BINS, HIST_S_BINS],\n"
        "        [0, 180, 0, 256],",
    )
    hist_path.write_text(content, encoding="utf-8")
    # Force reimport
    import importlib
    import bjj_pipeline.stages.detect_track.histogram as hmod
    importlib.reload(hmod)
    logger.info("  Reverted to 144-dim (HIST_SIZE={})", hmod.HIST_SIZE)


def _restore_864(original_content: str):
    """Restore histogram.py to 864-dim."""
    hist_path = REPO_ROOT / "src" / "bjj_pipeline" / "stages" / "detect_track" / "histogram.py"
    hist_path.write_text(original_content, encoding="utf-8")
    import importlib
    import bjj_pipeline.stages.detect_track.histogram as hmod
    importlib.reload(hmod)
    logger.info("  Restored to 864-dim (HIST_SIZE={})", hmod.HIST_SIZE)


# ---------------------------------------------------------------------------
# Phase 1: GT-validate splits
# ---------------------------------------------------------------------------

def _build_gt_map(clip_id: str) -> Dict[str, Dict[int, int]]:
    """Build per-frame GT mapping: {tracklet_id: {frame_index: gt_track_id}}."""
    clip_dir = VID1_DIR if clip_id == VID1_CLIP_ID else VID2_DIR
    manifest = load_model_manifest(DENSE_MANIFEST_PATH)
    for exp in manifest.training_data:
        if exp.camera_id != CAM_ID:
            continue
        src = exp.source_video.replace(".mp4", "")
        if clip_id == VID1_CLIP_ID and "200015" not in src:
            continue
        if clip_id == VID2_CLIP_ID and "200246" not in src:
            continue

        zip_path = REPO_ROOT / "data" / "training_data" / exp.export
        gt = _load_gt_all_annotated(zip_path, exp)
        ann = sorted(enumerate_annotated_frames(exp))
        det = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
        det_by_f = {fi: g for fi, g in det.groupby("frame_index")}

        # Per-frame per-tracklet GT assignment
        tid_frame_gt: Dict[str, Dict[int, int]] = defaultdict(dict)

        for fi in ann:
            boxes = gt.get(fi, [])
            fd = det_by_f.get(fi, pd.DataFrame())
            if not boxes or fd.empty:
                continue
            ga = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes])
            da = np.array([[r.x1, r.y1, r.x2, r.y2] for _, r in fd.iterrows()])
            for gi, di, iou in greedy_match(ga, da, iou_threshold=IOU_THRESHOLD):
                tid = fd.iloc[di].tracklet_id
                tid_frame_gt[tid][fi] = boxes[gi].track_id

        return dict(tid_frame_gt)
    raise ValueError(f"No export for {clip_id}")


def _validate_split(
    split: Dict,
    tid_frame_gt: Dict[str, Dict[int, int]],
    window: int = 30,
) -> str:
    """Validate one split against GT. Returns 'correct', 'spurious', or 'undecidable'."""
    tid = split["original_tracklet_id"]
    split_frame = split["split_frame"]

    frame_gt = tid_frame_gt.get(tid, {})
    if not frame_gt:
        return "undecidable"

    # Collect GT assignments in windows before and after split
    before_gts = []
    after_gts = []
    for fi, gt_id in frame_gt.items():
        if split_frame - window <= fi < split_frame:
            before_gts.append(gt_id)
        elif split_frame <= fi < split_frame + window:
            after_gts.append(gt_id)

    if len(before_gts) < 3 or len(after_gts) < 3:
        return "undecidable"

    # Majority GT on each side
    before_majority = Counter(before_gts).most_common(1)[0][0]
    after_majority = Counter(after_gts).most_common(1)[0][0]

    if before_majority != after_majority:
        return "correct"
    else:
        return "spurious"


def validate_splits(split_data: Dict) -> Dict[str, Any]:
    """Phase 1: Validate all splits against GT."""
    logger.info("=== Phase 1: GT-validate splits ===")
    results = {}

    for label in ["pre_v", "post_v"]:
        label_results = {}
        for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
            splits = split_data[label][clip_id]["splits"]
            tid_frame_gt = _build_gt_map(clip_id)

            per_tier: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for s in splits:
                verdict = _validate_split(s, tid_frame_gt)
                per_tier[s["tier"]][verdict] += 1

            clip_result = {}
            for tier in ["tier1_speed_cap", "tier2_kinematic_spike", "tier3_histogram"]:
                counts = per_tier.get(tier, {})
                total = sum(counts.values())
                clip_result[tier] = {
                    "correct": counts.get("correct", 0),
                    "spurious": counts.get("spurious", 0),
                    "undecidable": counts.get("undecidable", 0),
                    "total": total,
                    "precision": round(
                        counts.get("correct", 0) / (counts.get("correct", 0) + counts.get("spurious", 0))
                        if (counts.get("correct", 0) + counts.get("spurious", 0)) > 0 else 0, 3
                    ),
                }

            label_results[clip_id] = clip_result
            logger.info("  {} {}: {}", label, clip_id,
                        {t: f"c={d['correct']}/s={d['spurious']}/u={d['undecidable']}"
                         for t, d in clip_result.items() if d['total'] > 0})

        results[label] = label_results

    # Compute NEW T3 headline (post_v minus pre_v)
    new_t3 = {"correct": 0, "spurious": 0, "undecidable": 0}
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        pre_c = results["pre_v"][clip_id]["tier3_histogram"]["correct"]
        post_c = results["post_v"][clip_id]["tier3_histogram"]["correct"]
        pre_s = results["pre_v"][clip_id]["tier3_histogram"]["spurious"]
        post_s = results["post_v"][clip_id]["tier3_histogram"]["spurious"]
        pre_u = results["pre_v"][clip_id]["tier3_histogram"]["undecidable"]
        post_u = results["post_v"][clip_id]["tier3_histogram"]["undecidable"]
        new_t3["correct"] += max(0, post_c - pre_c)
        new_t3["spurious"] += max(0, post_s - pre_s)
        new_t3["undecidable"] += max(0, post_u - pre_u)

    new_total = sum(new_t3.values())
    results["new_t3_headline"] = {
        **new_t3,
        "total": new_total,
        "correct_frac": round(new_t3["correct"] / new_total, 3) if new_total else 0,
        "spurious_frac": round(new_t3["spurious"] / new_total, 3) if new_total else 0,
    }
    logger.info("NEW T3 headline: {}", results["new_t3_headline"])

    return results


# ---------------------------------------------------------------------------
# Phase 2: Characterize spurious T3 splits
# ---------------------------------------------------------------------------

def characterize_spurious_t3(split_data: Dict, validation: Dict) -> Dict[str, Any]:
    """Phase 2: Classify spurious T3 splits by motion-correlated pattern."""
    logger.info("=== Phase 2: Characterize spurious T3 splits ===")
    results = {}

    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        post_splits = split_data["post_v"][clip_id]["splits"]
        tid_frame_gt = _build_gt_map(clip_id)

        spurious_t3 = []
        for s in post_splits:
            if s["tier"] != "tier3_histogram":
                continue
            verdict = _validate_split(s, tid_frame_gt)
            if verdict != "spurious":
                continue
            spurious_t3.append(s)

        # Classify each spurious split
        shapes = {"motion_shadow_pose": 0, "single_point_blip": 0, "sustained_same_person": 0}

        for s in spurious_t3:
            pre_frames = s.get("pre_segment_frames", 0)
            post_frames = s.get("post_segment_frames", 0)
            bhatt = s.get("bhattacharyya_dist", 0)
            speed = s.get("speed_at_frame", 0)

            # All T3 cleared 2x speed gate → all are motion-correlated
            # Classify by segment duration pattern
            if min(pre_frames, post_frames) < 10:
                shapes["single_point_blip"] += 1
            elif bhatt < 0.20:
                # Barely above threshold, likely transient shadow/pose
                shapes["motion_shadow_pose"] += 1
            else:
                shapes["sustained_same_person"] += 1

        results[clip_id] = {
            "n_spurious_t3": len(spurious_t3),
            "shapes": shapes,
            "note": (
                "All T3 splits cleared the 2x-speed kinematic gate → all are "
                "MOTION-correlated color jumps, not stationary noise. "
                "Shape classification: single_point_blip = one side <10 frames; "
                "motion_shadow_pose = barely above 0.15 threshold (transient); "
                "sustained_same_person = persistent color change on same GT person."
            ),
        }

        logger.info("  {}: {} spurious T3 → {}", clip_id, len(spurious_t3), shapes)

    return results


# ---------------------------------------------------------------------------
# Phase 3: Threshold-sweep counterfactual
# ---------------------------------------------------------------------------

def threshold_sweep(split_data: Dict) -> Dict[str, Any]:
    """Phase 3: How many NEW T3 splits survive at different thresholds?"""
    logger.info("=== Phase 3: Threshold sweep ===")

    # Validate all post-V T3 splits
    all_validated = []
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        tid_frame_gt = _build_gt_map(clip_id)
        for s in split_data["post_v"][clip_id]["splits"]:
            if s["tier"] != "tier3_histogram":
                continue
            verdict = _validate_split(s, tid_frame_gt)
            all_validated.append({
                "clip_id": clip_id,
                "bhattacharyya_dist": s.get("bhattacharyya_dist", 0),
                "verdict": verdict,
            })

    results = {}
    for thresh in THRESHOLD_SWEEP:
        surviving = [v for v in all_validated if v["bhattacharyya_dist"] >= thresh]
        correct = sum(1 for v in surviving if v["verdict"] == "correct")
        spurious = sum(1 for v in surviving if v["verdict"] == "spurious")
        undecidable = sum(1 for v in surviving if v["verdict"] == "undecidable")
        total = len(surviving)
        decidable = correct + spurious

        results[str(thresh)] = {
            "threshold": thresh,
            "surviving": total,
            "correct": correct,
            "spurious": spurious,
            "undecidable": undecidable,
            "precision": round(correct / decidable, 3) if decidable else 0,
        }
        logger.info(
            "  thresh={:.2f}: {} surviving (c={}, s={}, u={}), precision={:.3f}",
            thresh, total, correct, spurious, undecidable,
            correct / decidable if decidable else 0,
        )

    return results


# ---------------------------------------------------------------------------
# Phase 4: k-distribution
# ---------------------------------------------------------------------------

def k_distribution(split_data: Dict) -> Dict[str, Any]:
    """Phase 4: How many GT identities do impure tracklets touch?"""
    logger.info("=== Phase 4: k-distribution ===")
    results = {}

    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        tid_frame_gt = _build_gt_map(clip_id)

        k_counts: Dict[int, int] = Counter()
        for tid, frame_gt in tid_frame_gt.items():
            unique_gts = set(frame_gt.values())
            k = len(unique_gts)
            k_counts[k] += 1

        results[clip_id] = {
            "k_distribution": {str(k): c for k, c in sorted(k_counts.items())},
            "n_tracklets": len(tid_frame_gt),
            "n_impure": sum(c for k, c in k_counts.items() if k >= 2),
            "n_pure": k_counts.get(1, 0),
            "max_k": max(k_counts.keys()) if k_counts else 0,
        }
        logger.info("  {}: k_dist={}, impure={}/{}", clip_id,
                     dict(sorted(k_counts.items())),
                     results[clip_id]["n_impure"],
                     results[clip_id]["n_tracklets"])

    return results


# ---------------------------------------------------------------------------
# Phase 5: Clean-point change-point feasibility
# ---------------------------------------------------------------------------

def changepoint_feasibility(split_data: Dict) -> Dict[str, Any]:
    """Phase 5: Do clean isolated HSV points show segmentable structure?"""
    logger.info("=== Phase 5: Change-point feasibility probe ===")
    results = {}

    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        clip_dir = VID1_DIR if clip_id == VID1_CLIP_ID else VID2_DIR
        tid_frame_gt = _build_gt_map(clip_id)

        hist_df = pd.read_parquet(clip_dir / "stage_A" / "color_histograms.parquet")
        hist_cols = [c for c in hist_df.columns if c.startswith("hist_")]
        isolated = hist_df[hist_df.is_isolated == True].copy()

        n_probed = 0
        n_pure_clean = 0  # pure tracklets with consistent clean-point signal
        n_impure_segmentable = 0  # impure tracklets with visible multi-state structure
        n_impure_not_segmentable = 0

        for tid, frame_gt in tid_frame_gt.items():
            tid_iso = isolated[isolated.track_id == tid].sort_values("frame_index")
            if len(tid_iso) < 10:
                continue

            unique_gts = set(frame_gt.values())
            k = len(unique_gts)
            n_probed += 1

            # Get time-ordered clean histograms
            hists = tid_iso[hist_cols].values.astype(np.float32)
            frames = tid_iso["frame_index"].values

            # Compute consecutive-frame Bhattacharyya distances
            dists = []
            for i in range(1, len(hists)):
                bc = float(np.sum(np.sqrt(np.maximum(hists[i] * hists[i - 1], 0.0))))
                dists.append(1.0 - bc)

            if not dists:
                continue

            dists_arr = np.array(dists)
            mean_dist = float(dists_arr.mean())
            max_dist = float(dists_arr.max())
            n_above_015 = int(np.sum(dists_arr > 0.15))

            if k == 1:
                # Pure tracklet: clean signal should show single-state (low variance)
                if max_dist < 0.25 and n_above_015 < len(dists) * 0.1:
                    n_pure_clean += 1
            else:
                # Impure: should show multi-state structure (high jumps at GT transitions)
                # Check if the highest-distance frame aligns with a GT transition
                if len(dists) > 0:
                    peak_idx = int(np.argmax(dists_arr))
                    peak_frame = int(frames[peak_idx + 1])

                    # Check if GT identity changes near the peak
                    window = 15
                    before_gts = [frame_gt[fi] for fi in frame_gt if peak_frame - window <= fi < peak_frame]
                    after_gts = [frame_gt[fi] for fi in frame_gt if peak_frame <= fi < peak_frame + window]

                    if before_gts and after_gts:
                        before_maj = Counter(before_gts).most_common(1)[0][0]
                        after_maj = Counter(after_gts).most_common(1)[0][0]
                        if before_maj != after_maj:
                            n_impure_segmentable += 1
                        else:
                            n_impure_not_segmentable += 1
                    else:
                        n_impure_not_segmentable += 1

        results[clip_id] = {
            "n_probed": n_probed,
            "n_pure_clean_signal": n_pure_clean,
            "n_pure_total": sum(1 for tid, fg in tid_frame_gt.items() if len(set(fg.values())) == 1
                                and len(isolated[isolated.track_id == tid]) >= 10),
            "n_impure_segmentable": n_impure_segmentable,
            "n_impure_not_segmentable": n_impure_not_segmentable,
            "n_impure_total": sum(1 for tid, fg in tid_frame_gt.items() if len(set(fg.values())) >= 2
                                  and len(isolated[isolated.track_id == tid]) >= 10),
            "verdict": (
                "Clean-point signal carries segmentable structure"
                if n_impure_segmentable > n_impure_not_segmentable
                else "Mixed — some impure tracklets not segmentable from clean points alone"
            ),
        }
        logger.info(
            "  {}: probed={}, pure_clean={}, impure_seg={}, impure_not={}",
            clip_id, n_probed, n_pure_clean, n_impure_segmentable, n_impure_not_segmentable,
        )

    return results


# ---------------------------------------------------------------------------
# Evidence + report
# ---------------------------------------------------------------------------

def write_evidence(
    phase0: Dict, phase1: Dict, phase2: Dict, phase3: Dict, phase4: Dict, phase5: Dict,
) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, data: Any) -> None:
        (EVIDENCE_DIR / name).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    _dump("phase0_split_audits.json", {
        label: {clip: {"tiers": d["tiers"]} for clip, d in clips.items()}
        for label, clips in phase0.items()
    })
    _dump("phase1_gt_validation.json", phase1)
    _dump("phase2_spurious_characterization.json", phase2)
    _dump("phase3_threshold_sweep.json", phase3)
    _dump("phase4_k_distribution.json", phase4)
    _dump("phase5_changepoint_feasibility.json", phase5)

    # Report
    lines = [
        "# CP-SPLIT-VALIDATE: GT-Validate D0.5 Splits",
        "",
        "## Phase 0: Reconstructed Split Counts",
        "",
        "| Clip | Pre-V T1 | Pre-V T2 | Pre-V T3 | Post-V T1 | Post-V T2 | Post-V T3 |",
        "|------|----------|----------|----------|-----------|-----------|-----------|",
    ]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        pre = phase0["pre_v"][clip_id]["tiers"]
        post = phase0["post_v"][clip_id]["tiers"]
        lines.append(
            f"| {clip_id} | {pre.get('tier1_speed_cap', 0)} | {pre.get('tier2_kinematic_spike', 0)} "
            f"| {pre.get('tier3_histogram', 0)} | {post.get('tier1_speed_cap', 0)} "
            f"| {post.get('tier2_kinematic_spike', 0)} | {post.get('tier3_histogram', 0)} |"
        )

    lines += ["", "## Phase 1: GT Validation", ""]
    for label in ["pre_v", "post_v"]:
        lines.append(f"### {label}")
        lines.append("| Clip | Tier | Correct | Spurious | Undecidable | Precision |")
        lines.append("|------|------|---------|----------|-------------|-----------|")
        for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
            for tier in ["tier1_speed_cap", "tier2_kinematic_spike", "tier3_histogram"]:
                d = phase1[label][clip_id].get(tier, {})
                if d.get("total", 0) > 0:
                    lines.append(
                        f"| {clip_id} | {tier} | {d['correct']} | {d['spurious']} "
                        f"| {d['undecidable']} | {d['precision']} |"
                    )
        lines.append("")

    hl = phase1.get("new_t3_headline", {})
    lines += [
        f"**NEW T3 headline:** {hl.get('correct', 0)} correct, {hl.get('spurious', 0)} spurious, "
        f"{hl.get('undecidable', 0)} undecidable. "
        f"Correct fraction: {hl.get('correct_frac', 0)}, Spurious fraction: {hl.get('spurious_frac', 0)}.",
        "",
        "## Phase 2: Spurious T3 Characterization", "",
    ]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        d = phase2.get(clip_id, {})
        lines.append(f"**{clip_id}:** {d.get('n_spurious_t3', 0)} spurious → {d.get('shapes', {})}")
    lines.append("")

    lines += ["## Phase 3: Threshold Sweep", "",
              "| Threshold | Surviving | Correct | Spurious | Precision |",
              "|-----------|-----------|---------|----------|-----------|"]
    for thresh_str, d in phase3.items():
        lines.append(
            f"| {d['threshold']} | {d['surviving']} | {d['correct']} "
            f"| {d['spurious']} | {d['precision']} |"
        )

    lines += ["", "## Phase 4: k-Distribution", ""]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        d = phase4.get(clip_id, {})
        lines.append(f"**{clip_id}:** k_dist={d.get('k_distribution', {})}, "
                      f"impure={d.get('n_impure', 0)}/{d.get('n_tracklets', 0)}")
    lines.append("")

    lines += ["## Phase 5: Change-Point Feasibility", ""]
    for clip_id in [VID1_CLIP_ID, VID2_CLIP_ID]:
        d = phase5.get(clip_id, {})
        lines.append(f"**{clip_id}:** {d.get('verdict', 'N/A')}")
        lines.append(f"  pure_clean={d.get('n_pure_clean_signal', 0)}/{d.get('n_pure_total', 0)}, "
                      f"impure_seg={d.get('n_impure_segmentable', 0)}/{d.get('n_impure_total', 0)}")
    lines.append("")

    (EVIDENCE_DIR / "split_validate_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evidence written to {}", EVIDENCE_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reconstruct", action="store_true",
                        help="Skip Phase 0 (reuse existing audits)")
    args = parser.parse_args()

    if args.skip_reconstruct and (WORK_DIR / "pre_v_splits.json").exists():
        logger.info("Loading cached split audits...")
        phase0 = {
            "pre_v": json.loads((WORK_DIR / "pre_v_splits.json").read_text()),
            "post_v": json.loads((WORK_DIR / "post_v_splits.json").read_text()),
        }
    else:
        phase0 = reconstruct_split_audits()

    phase1 = validate_splits(phase0)
    phase2 = characterize_spurious_t3(phase0, phase1)
    phase3 = threshold_sweep(phase0)
    phase4 = k_distribution(phase0)
    phase5 = changepoint_feasibility(phase0)

    write_evidence(phase0, phase1, phase2, phase3, phase4, phase5)
    logger.info("CP-SPLIT-VALIDATE complete.")


if __name__ == "__main__":
    main()
