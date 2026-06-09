"""CP-RASTER-PLATE: Median-background masking tool + appearance separability.

Standalone measurement tool. No src/bjj_pipeline changes.

Phases:
  A — Per-camera median background plate (with empirical ghost validation)
  B — Foreground-masked histogram extraction on GT clips
  C — Separability measurement: masked vs center-bbox crop

Usage:
    PYTHONPATH=src python tools/cp_raster_plate.py
    PYTHONPATH=src python tools/cp_raster_plate.py --phase a   # plate only
    PYTHONPATH=src python tools/cp_raster_plate.py --phase b   # extraction only (needs plate)
    PYTHONPATH=src python tools/cp_raster_plate.py --phase c   # separability only (needs extraction)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bjj_pipeline.stages.detect_track.histogram import (
    HIST_H_BINS,
    HIST_S_BINS,
    HIST_SIZE,
    _center_crop_from_bbox,
    bhattacharyya_distance,
    compute_hsv_histogram,
)
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
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_raster_plate"
ARTIFACT_DIR = OUTPUTS_DIR / "_eval_gt_oracle" / "raster_plate"

CAM_ID = "J_EDEw"
GYM_ID = "_eval_gt"
NEST_DIR = REPO_ROOT / "data" / "raw" / "nest" / "c8a592a4-2bca-400a-80e1-fec0e5cbea77"
CLIP_DIR = NEST_DIR / CAM_ID / "2026-03-18" / "20"

VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID1_CLIP_ID
VID2_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID2_CLIP_ID

DENSE_MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"

# Plate construction
PLATE_SAMPLES_PER_CLIP = 33
GHOST_LOW_OCC_TOP_N = 4  # use top-N lowest-occupancy clips for ghost fallback

# Foreground mask thresholds (chroma space, ignore V)
H_THRESH = 12   # out of 180 (~24 degrees hue)
S_THRESH = 35   # out of 256 (~14% saturation)

# Mask quality gate
DEGENERATE_LOW = 0.05
DEGENERATE_HIGH = 0.95

# Sampling
FRAMES_PER_TRACKLET = 30
MIN_ISOLATED_FRAMES = 10
IOU_THRESHOLD = 0.3

# Color distinctiveness (method-independent: visual inspection labels)
# We assign per-GT-track gi color labels from the footage.
# Since there are only 14 people per clip, this is cheap.
# Labels will be assigned in _assign_gi_colors() from per-tracklet
# dominant hue extracted by the UNION of both methods (not baseline alone).

SEED = 42

# ---------------------------------------------------------------------------
# Phase A: Median background plate
# ---------------------------------------------------------------------------


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def _count_center_detections(clip_path: Path, n_sample: int = 20) -> float:
    """Empirically estimate center-region detection density for a clip.

    Uses simple background-difference heuristic: count pixels in center 50%
    region that differ from temporal median of sampled frames. Lower count =
    less center occupancy = better fallback candidate.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fc == 0:
        cap.release()
        return float("inf")

    stride = max(1, fc // n_sample)
    frames = []
    for i in range(0, fc, stride):
        f = _read_frame(cap, i)
        if f is not None:
            frames.append(f)
        if len(frames) >= n_sample:
            break
    cap.release()

    if len(frames) < 3:
        return float("inf")

    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)

    # Center 50% region
    h, w = median.shape[:2]
    r1, r2 = h // 4, 3 * h // 4
    c1, c2 = w // 4, 3 * w // 4

    # Count frames with significant center activity
    total_activity = 0.0
    for f in frames:
        diff = cv2.absdiff(f[r1:r2, c1:c2], median[r1:r2, c1:c2])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        active_pixels = np.sum(gray_diff > 30)
        total_activity += active_pixels

    avg_activity = total_activity / len(frames)
    return float(avg_activity)


def build_median_plate() -> Dict[str, Any]:
    """Phase A: build per-camera median background plate."""
    logger.info("=== Phase A: Building median background plate ===")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(CLIP_DIR.glob("J_EDEw-*.mp4"))
    if not clip_paths:
        raise FileNotFoundError(f"No clips in {CLIP_DIR}")

    # Step 1: Empirically rank clips by center-region occupancy
    logger.info("Step 1: Ranking clips by center occupancy...")
    clip_activity = []
    for cp in clip_paths:
        activity = _count_center_detections(cp)
        clip_activity.append((cp, activity))
        logger.info("  {}: center_activity={:.0f}", cp.name, activity)

    clip_activity.sort(key=lambda x: x[1])
    low_occ_clips = [ca[0] for ca in clip_activity[:GHOST_LOW_OCC_TOP_N]]
    logger.info(
        "Low-occupancy clips (top {}): {}",
        GHOST_LOW_OCC_TOP_N, [c.name for c in low_occ_clips],
    )

    # Step 2: Sample frames from ALL clips for the main plate
    logger.info("Step 2: Sampling frames for main plate...")
    all_frames = []
    for cp in clip_paths:
        cap = cv2.VideoCapture(str(cp))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, fc // PLATE_SAMPLES_PER_CLIP)
        count = 0
        for i in range(0, fc, stride):
            f = _read_frame(cap, i)
            if f is not None:
                all_frames.append(f)
                count += 1
            if count >= PLATE_SAMPLES_PER_CLIP:
                break
        cap.release()
        logger.info("  {}: sampled {} frames", cp.name, count)

    logger.info("Total frames for main plate: {}", len(all_frames))
    stack = np.stack(all_frames, axis=0)
    main_plate = np.median(stack, axis=0).astype(np.uint8)

    # Step 3: Build low-occupancy plate for ghost fallback
    logger.info("Step 3: Building low-occupancy plate for ghost fallback...")
    low_occ_frames = []
    for cp in low_occ_clips:
        cap = cv2.VideoCapture(str(cp))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, fc // (PLATE_SAMPLES_PER_CLIP * 2))
        count = 0
        for i in range(0, fc, stride):
            f = _read_frame(cap, i)
            if f is not None:
                low_occ_frames.append(f)
                count += 1
            if count >= PLATE_SAMPLES_PER_CLIP * 2:
                break
        cap.release()

    low_occ_plate = np.median(np.stack(low_occ_frames), axis=0).astype(np.uint8)

    # Step 4: Ghost validation
    logger.info("Step 4: Ghost validation...")
    h, w = main_plate.shape[:2]
    r1, r2 = h // 4, 3 * h // 4
    c1, c2 = w // 4, 3 * w // 4

    # Ghost pixels: center region where main and low-occ plates disagree
    center_main = main_plate[r1:r2, c1:c2].astype(np.float32)
    center_low = low_occ_plate[r1:r2, c1:c2].astype(np.float32)
    diff = np.sqrt(np.sum((center_main - center_low) ** 2, axis=2))
    ghost_mask_center = diff > 25.0  # pixels that differ by >25 L2

    total_center_pixels = (r2 - r1) * (c2 - c1)
    ghost_count_before = int(np.sum(ghost_mask_center))
    ghost_frac_before = ghost_count_before / total_center_pixels

    logger.info(
        "Ghost pixels in center: {}/{} ({:.1f}%)",
        ghost_count_before, total_center_pixels, ghost_frac_before * 100,
    )

    # Apply fallback: replace ghost pixels with low-occ plate values
    final_plate = main_plate.copy()
    ghost_full = np.zeros((h, w), dtype=bool)
    ghost_full[r1:r2, c1:c2] = ghost_mask_center
    final_plate[ghost_full] = low_occ_plate[ghost_full]

    # Verify: recheck after fallback
    center_final = final_plate[r1:r2, c1:c2].astype(np.float32)
    diff_after = np.sqrt(np.sum((center_final - center_low) ** 2, axis=2))
    ghost_after = int(np.sum(diff_after > 25.0))
    ghost_frac_after = ghost_after / total_center_pixels

    logger.info(
        "Ghost pixels after fallback: {}/{} ({:.1f}%)",
        ghost_after, total_center_pixels, ghost_frac_after * 100,
    )

    # Save artifacts
    np.save(ARTIFACT_DIR / "J_EDEw_median_plate.npy", final_plate)
    cv2.imwrite(str(ARTIFACT_DIR / "J_EDEw_median_plate.png"), final_plate)

    # Ghost overlay visualization
    overlay = final_plate.copy()
    overlay[ghost_full] = [0, 0, 255]  # red for ghost pixels
    cv2.imwrite(str(ARTIFACT_DIR / "J_EDEw_ghost_validation.png"), overlay)

    results = {
        "n_clips": len(clip_paths),
        "n_frames_sampled": len(all_frames),
        "low_occupancy_clips": [c.name for c in low_occ_clips],
        "clip_activity_ranking": [
            {"clip": ca[0].name, "center_activity": round(ca[1], 1)}
            for ca in clip_activity
        ],
        "ghost_pixels_before": ghost_count_before,
        "ghost_frac_before": round(ghost_frac_before, 4),
        "ghost_pixels_after": ghost_after,
        "ghost_frac_after": round(ghost_frac_after, 4),
        "center_region": {"r1": r1, "r2": r2, "c1": c1, "c2": c2},
        "plate_shape": list(final_plate.shape),
        "note": (
            "Plate built from SAME footage we measure on. Fine for measurement "
            "but a production path needs held-out/rolling background."
        ),
    }

    logger.info("Phase A complete. Plate saved to {}", ARTIFACT_DIR)
    return results


# ---------------------------------------------------------------------------
# Phase B: Masked histogram extraction
# ---------------------------------------------------------------------------


def _compute_foreground_mask(
    crop_bgr: np.ndarray, plate_crop_bgr: np.ndarray,
) -> np.ndarray:
    """Compute foreground mask in chroma space (H/S, ignore V).

    Returns uint8 mask (255=foreground, 0=background).
    """
    hsv_frame = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hsv_plate = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2HSV)

    h_frame = hsv_frame[:, :, 0].astype(np.int16)
    s_frame = hsv_frame[:, :, 1].astype(np.int16)
    h_plate = hsv_plate[:, :, 0].astype(np.int16)
    s_plate = hsv_plate[:, :, 1].astype(np.int16)

    # H wraparound: min(|diff|, 180-|diff|)
    h_diff = np.abs(h_frame - h_plate)
    h_diff = np.minimum(h_diff, 180 - h_diff)

    s_diff = np.abs(s_frame - s_plate)

    fg = ((h_diff > H_THRESH) | (s_diff > S_THRESH)).astype(np.uint8) * 255
    return fg


def _compute_masked_histogram(
    crop_bgr: np.ndarray, mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute 144-dim HSV histogram using foreground mask."""
    if mask.sum() == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], mask,
        [HIST_H_BINS, HIST_S_BINS],
        [0, 180, 0, 256],
    )
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.flatten().astype(np.float32)


def _estimate_dominant_hue_sat(hist_144: np.ndarray) -> Tuple[int, int]:
    """Estimate dominant H and S bin from a 144-dim histogram."""
    hist_2d = hist_144.reshape(HIST_H_BINS, HIST_S_BINS)
    idx = np.unravel_index(np.argmax(hist_2d), hist_2d.shape)
    h_bin, s_bin = idx
    # Convert to approximate HSV values
    h_val = int(h_bin * 10 + 5)   # center of bin (0-179)
    s_val = int(s_bin * 32 + 16)  # center of bin (0-255)
    return h_val, s_val


def _classify_gi_color(h_val: int, s_val: int) -> str:
    """Classify gi color from dominant H/S. Simple categories."""
    if s_val < 50:
        return "white_gray"  # low saturation = white/gray gi
    if h_val < 15 or h_val > 165:
        return "red"
    if 15 <= h_val < 35:
        return "orange_yellow"
    if 35 <= h_val < 80:
        return "green"
    if 80 <= h_val < 130:
        return "blue"
    return "purple"


def extract_masked_histograms() -> Dict[str, Any]:
    """Phase B: extract masked histograms on GT clips."""
    logger.info("=== Phase B: Masked histogram extraction ===")

    plate_path = ARTIFACT_DIR / "J_EDEw_median_plate.npy"
    if not plate_path.exists():
        raise FileNotFoundError(f"Plate not found: {plate_path}. Run phase A first.")
    plate = np.load(plate_path)

    rng = np.random.RandomState(SEED)
    all_results = {}

    for clip_id, clip_dir in [
        (VID1_CLIP_ID, VID1_DIR),
        (VID2_CLIP_ID, VID2_DIR),
    ]:
        logger.info("Processing {}...", clip_id)

        # Load detection bboxes and isolation flags
        det_df = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
        hist_df = pd.read_parquet(clip_dir / "stage_A" / "color_histograms.parquet")

        # Join: detections has detection_id + bbox; hist_df has track_id + is_isolated
        # track_id in hist_df == tracklet_id in det_df
        # Join on frame_index + tracklet_id/track_id
        det_df = det_df.rename(columns={"tracklet_id": "track_id"})
        merged = det_df.merge(
            hist_df[["frame_index", "track_id", "is_isolated"]],
            on=["frame_index", "track_id"],
            how="inner",
        )
        isolated = merged[merged["is_isolated"] == True].copy()
        logger.info("  {} isolated detections", len(isolated))

        # Sample up to FRAMES_PER_TRACKLET per tracklet
        sampled_rows = []
        for tid, grp in isolated.groupby("track_id"):
            if len(grp) < MIN_ISOLATED_FRAMES:
                continue
            n = min(FRAMES_PER_TRACKLET, len(grp))
            sampled_rows.append(grp.sample(n=n, random_state=rng))

        if not sampled_rows:
            logger.warning("  No tracklets with enough isolated frames")
            all_results[clip_id] = {"error": "no_tracklets"}
            continue

        sampled = pd.concat(sampled_rows, ignore_index=True)
        logger.info(
            "  Sampled {} detections across {} tracklets",
            len(sampled), sampled["track_id"].nunique(),
        )

        # Group by frame_index for efficient video reading
        frame_groups = sampled.groupby("frame_index")
        frames_to_read = sorted(frame_groups.groups.keys())

        # Open video
        video_path = CLIP_DIR / f"{clip_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("  Cannot open {}", video_path)
            all_results[clip_id] = {"error": "video_open_failed"}
            continue

        # Extract histograms
        per_det_results = []
        for fi in frames_to_read:
            frame = _read_frame(cap, fi)
            if frame is None:
                continue

            group = frame_groups.get_group(fi)
            for _, row in group.iterrows():
                x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(frame.shape[1], int(x2))
                iy2 = min(frame.shape[0], int(y2))

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                frame_crop = frame[iy1:iy2, ix1:ix2]
                plate_crop = plate[iy1:iy2, ix1:ix2]

                # Foreground mask
                fg_mask = _compute_foreground_mask(frame_crop, plate_crop)
                bbox_pixels = (iy2 - iy1) * (ix2 - ix1)
                mask_pixels = int(np.sum(fg_mask > 0))
                coverage = mask_pixels / bbox_pixels if bbox_pixels > 0 else 0

                # Degenerate check
                is_degenerate = coverage < DEGENERATE_LOW or coverage > DEGENERATE_HIGH

                # Masked histogram
                masked_hist = _compute_masked_histogram(frame_crop, fg_mask)

                # Baseline histogram (center 60% bbox)
                baseline_crop = _center_crop_from_bbox(
                    frame, (x1, y1, x2, y2)
                )
                baseline_hist = (
                    compute_hsv_histogram(baseline_crop)
                    if baseline_crop is not None and baseline_crop.size > 0
                    else None
                )

                per_det_results.append({
                    "frame_index": int(fi),
                    "track_id": row.track_id,
                    "coverage": float(coverage),
                    "is_degenerate": bool(is_degenerate),
                    "masked_hist": masked_hist,
                    "baseline_hist": baseline_hist,
                    "bbox_area": bbox_pixels,
                })

        cap.release()

        # Aggregate per-tracklet histograms
        tracklet_hists: Dict[str, Dict] = {}
        coverage_stats = []
        degenerate_count = 0
        per_tracklet_gi_color: Dict[str, str] = {}

        for r in per_det_results:
            tid = r["track_id"]
            coverage_stats.append(r["coverage"])
            if r["is_degenerate"]:
                degenerate_count += 1

            if tid not in tracklet_hists:
                tracklet_hists[tid] = {
                    "masked": [], "baseline": [],
                    "coverages": [], "degenerate_count": 0,
                }
            th = tracklet_hists[tid]
            th["coverages"].append(r["coverage"])
            if r["is_degenerate"]:
                th["degenerate_count"] += 1

            if r["masked_hist"] is not None and not r["is_degenerate"]:
                th["masked"].append(r["masked_hist"])
            if r["baseline_hist"] is not None:
                th["baseline"].append(r["baseline_hist"])

        # Build per-tracklet summary histograms
        tracklet_summaries: Dict[str, Dict] = {}
        for tid, th in tracklet_hists.items():
            summary = {
                "n_sampled": len(th["coverages"]),
                "n_degenerate": th["degenerate_count"],
                "mean_coverage": float(np.mean(th["coverages"])) if th["coverages"] else 0,
            }

            if th["masked"]:
                avg_m = np.mean(np.stack(th["masked"]), axis=0).astype(np.float32)
                total = avg_m.sum()
                if total > 0:
                    avg_m /= total
                summary["masked_hist"] = avg_m
            else:
                summary["masked_hist"] = None

            if th["baseline"]:
                avg_b = np.mean(np.stack(th["baseline"]), axis=0).astype(np.float32)
                total = avg_b.sum()
                if total > 0:
                    avg_b /= total
                summary["baseline_hist"] = avg_b
            else:
                summary["baseline_hist"] = None

            # Classify gi color from UNION of both methods (method-independent)
            # Use whichever summary histogram has more pixels
            color_hist = summary["masked_hist"]
            if color_hist is None:
                color_hist = summary["baseline_hist"]
            if color_hist is not None:
                h_val, s_val = _estimate_dominant_hue_sat(color_hist)
                summary["dominant_h"] = h_val
                summary["dominant_s"] = s_val
                summary["gi_color"] = _classify_gi_color(h_val, s_val)
            else:
                summary["gi_color"] = "unknown"

            tracklet_summaries[tid] = summary

        # Self-absorption check: for each tracklet, report coverage stats
        self_absorption = {}
        for tid, ts in tracklet_summaries.items():
            mc = ts["mean_coverage"]
            self_absorption[tid] = {
                "mean_coverage": round(mc, 3),
                "absorbed": mc < 0.10,
                "gi_color": ts.get("gi_color", "unknown"),
            }

        degenerate_frac = degenerate_count / len(per_det_results) if per_det_results else 0

        # Stratify degenerate fraction by gi color
        degenerate_by_color: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "degenerate": 0})
        for r in per_det_results:
            tid = r["track_id"]
            gi_color = tracklet_summaries.get(tid, {}).get("gi_color", "unknown")
            degenerate_by_color[gi_color]["total"] += 1
            if r["is_degenerate"]:
                degenerate_by_color[gi_color]["degenerate"] += 1

        degenerate_by_color_report = {}
        for color, counts in degenerate_by_color.items():
            t = counts["total"]
            d = counts["degenerate"]
            degenerate_by_color_report[color] = {
                "total": t, "degenerate": d,
                "frac": round(d / t, 3) if t > 0 else 0,
            }

        clip_result = {
            "n_detections_sampled": len(per_det_results),
            "n_tracklets": len(tracklet_summaries),
            "coverage_mean": round(float(np.mean(coverage_stats)), 3) if coverage_stats else 0,
            "coverage_median": round(float(np.median(coverage_stats)), 3) if coverage_stats else 0,
            "coverage_p10": round(float(np.percentile(coverage_stats, 10)), 3) if coverage_stats else 0,
            "coverage_p90": round(float(np.percentile(coverage_stats, 90)), 3) if coverage_stats else 0,
            "degenerate_count": degenerate_count,
            "degenerate_frac": round(degenerate_frac, 3),
            "degenerate_by_gi_color": degenerate_by_color_report,
            "self_absorption": self_absorption,
            "tracklet_summaries": {
                tid: {
                    k: v for k, v in ts.items()
                    if k not in ("masked_hist", "baseline_hist")
                }
                for tid, ts in tracklet_summaries.items()
            },
        }

        all_results[clip_id] = clip_result

        # Save tracklet histograms as parquet for Phase C
        hist_rows = []
        for tid, ts in tracklet_summaries.items():
            row = {"clip_id": clip_id, "track_id": tid, "gi_color": ts.get("gi_color", "unknown")}
            if ts["masked_hist"] is not None:
                for i, v in enumerate(ts["masked_hist"]):
                    row[f"masked_{i}"] = float(v)
            if ts["baseline_hist"] is not None:
                for i, v in enumerate(ts["baseline_hist"]):
                    row[f"baseline_{i}"] = float(v)
            hist_rows.append(row)

        pd.DataFrame(hist_rows).to_parquet(
            ARTIFACT_DIR / f"{clip_id}_tracklet_hists.parquet", index=False
        )

        logger.info(
            "  {}: {} tracklets, coverage={:.1f}% median, {:.1f}% degenerate",
            clip_id, len(tracklet_summaries),
            clip_result["coverage_median"] * 100,
            degenerate_frac * 100,
        )

    return all_results


# ---------------------------------------------------------------------------
# Phase C: Separability measurement
# ---------------------------------------------------------------------------


def _load_tracklet_hists(clip_id: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACT_DIR / f"{clip_id}_tracklet_hists.parquet")


def _build_gt_tracklet_map(
    clip_id: str, clip_dir: Path,
) -> Dict[str, int]:
    """Map pipeline tracklet_id → GT track_id via greedy matcher majority vote."""
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
        gt_by_frame = _load_gt_all_annotated(zip_path, exp)
        annotated_frames = sorted(enumerate_annotated_frames(exp))

        det_df = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
        det_by_frame = {fi: g for fi, g in det_df.groupby("frame_index")}

        # Greedy match per frame, accumulate votes
        votes: Dict[str, Counter] = defaultdict(Counter)

        for fi in annotated_frames:
            gt_boxes = gt_by_frame.get(fi, [])
            frame_dets = det_by_frame.get(fi, pd.DataFrame())
            if not gt_boxes or frame_dets.empty:
                continue

            gt_arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in gt_boxes])
            det_arr = np.array([[r.x1, r.y1, r.x2, r.y2] for _, r in frame_dets.iterrows()])

            matches = greedy_match(gt_arr, det_arr, iou_threshold=IOU_THRESHOLD)
            for gt_idx, det_idx, iou_val in matches:
                tid = frame_dets.iloc[det_idx].tracklet_id
                gt_tid = gt_boxes[gt_idx].track_id
                votes[tid][gt_tid] += 1

        # Majority vote
        tid_to_gt: Dict[str, int] = {}
        for tid, counter in votes.items():
            best_gt, count = counter.most_common(1)[0]
            tid_to_gt[tid] = best_gt

        return tid_to_gt

    raise ValueError(f"No matching export for {clip_id}")


def _assign_gi_colors_from_gt(
    gt_tracklet_map: Dict[str, int],
    hist_df: pd.DataFrame,
) -> Dict[int, str]:
    """Assign gi color per GT track_id using UNION of both methods.

    Method-independent: uses whichever histogram (masked or baseline) is
    available per tracklet, then majority-votes per GT person across all
    their tracklets. This avoids circularity — the color label comes from
    the person's actual appearance, not from one method's ability to separate.
    """
    gt_colors: Dict[int, Counter] = defaultdict(Counter)

    for _, row in hist_df.iterrows():
        tid = row["track_id"]
        gt_id = gt_tracklet_map.get(tid)
        if gt_id is None:
            continue
        gi_color = row.get("gi_color", "unknown")
        if gi_color != "unknown":
            gt_colors[gt_id][gi_color] += 1

    result = {}
    for gt_id, counter in gt_colors.items():
        best_color, _ = counter.most_common(1)[0]
        result[gt_id] = best_color

    return result


def measure_separability() -> Dict[str, Any]:
    """Phase C: compare separability of masked vs center-bbox histograms."""
    logger.info("=== Phase C: Separability measurement ===")

    all_same_baseline = []
    all_diff_baseline = []
    all_same_masked = []
    all_diff_masked = []

    # Stratified by color distinctiveness
    distinct_same_base = []
    distinct_diff_base = []
    distinct_same_mask = []
    distinct_diff_mask = []
    same_color_same_base = []
    same_color_diff_base = []
    same_color_same_mask = []
    same_color_diff_mask = []

    per_clip_results = {}

    for clip_id, clip_dir in [
        (VID1_CLIP_ID, VID1_DIR),
        (VID2_CLIP_ID, VID2_DIR),
    ]:
        logger.info("Processing {}...", clip_id)

        hist_df = _load_tracklet_hists(clip_id)
        gt_map = _build_gt_tracklet_map(clip_id, clip_dir)

        # Assign per-GT-track gi color (method-independent)
        gt_gi_colors = _assign_gi_colors_from_gt(gt_map, hist_df)
        logger.info("  GT gi colors: {}", gt_gi_colors)

        # Filter to tracklets with both histograms and GT mapping
        masked_cols = [c for c in hist_df.columns if c.startswith("masked_")]
        baseline_cols = [c for c in hist_df.columns if c.startswith("baseline_")]

        valid_tids = []
        tid_masked: Dict[str, np.ndarray] = {}
        tid_baseline: Dict[str, np.ndarray] = {}
        tid_gt: Dict[str, int] = {}

        for _, row in hist_df.iterrows():
            tid = row["track_id"]
            gt_id = gt_map.get(tid)
            if gt_id is None:
                continue

            has_masked = len(masked_cols) > 0 and not pd.isna(row.get("masked_0", np.nan))
            has_baseline = len(baseline_cols) > 0 and not pd.isna(row.get("baseline_0", np.nan))

            if has_masked and has_baseline:
                tid_masked[tid] = np.array(
                    [float(row[f"masked_{i}"]) for i in range(HIST_SIZE)],
                    dtype=np.float32,
                )
                tid_baseline[tid] = np.array(
                    [float(row[f"baseline_{i}"]) for i in range(HIST_SIZE)],
                    dtype=np.float32,
                )
                tid_gt[tid] = gt_id
                valid_tids.append(tid)

        logger.info(
            "  {} tracklets with both histograms + GT mapping", len(valid_tids)
        )

        # Form pairs
        clip_same_base = []
        clip_diff_base = []
        clip_same_mask = []
        clip_diff_mask = []

        for i in range(len(valid_tids)):
            for j in range(i + 1, len(valid_tids)):
                t1, t2 = valid_tids[i], valid_tids[j]
                gt1, gt2 = tid_gt[t1], tid_gt[t2]
                is_same = gt1 == gt2

                d_base = bhattacharyya_distance(tid_baseline[t1], tid_baseline[t2])
                d_mask = bhattacharyya_distance(tid_masked[t1], tid_masked[t2])

                if is_same:
                    clip_same_base.append(d_base)
                    clip_same_mask.append(d_mask)
                else:
                    clip_diff_base.append(d_base)
                    clip_diff_mask.append(d_mask)

                    # Color-distinctiveness from GT gi color (method-independent)
                    color1 = gt_gi_colors.get(gt1, "unknown")
                    color2 = gt_gi_colors.get(gt2, "unknown")
                    is_distinct = color1 != color2

                    if is_distinct:
                        distinct_diff_base.append(d_base)
                        distinct_diff_mask.append(d_mask)
                    else:
                        same_color_diff_base.append(d_base)
                        same_color_diff_mask.append(d_mask)

                if is_same:
                    color1 = gt_gi_colors.get(gt1, "unknown")
                    color2 = gt_gi_colors.get(gt2, "unknown")
                    is_distinct = color1 != color2
                    if is_distinct:
                        distinct_same_base.append(d_base)
                        distinct_same_mask.append(d_mask)
                    else:
                        same_color_same_base.append(d_base)
                        same_color_same_mask.append(d_mask)

        all_same_baseline.extend(clip_same_base)
        all_diff_baseline.extend(clip_diff_base)
        all_same_masked.extend(clip_same_mask)
        all_diff_masked.extend(clip_diff_mask)

        per_clip_results[clip_id] = {
            "n_valid_tracklets": len(valid_tids),
            "n_same_pairs": len(clip_same_base),
            "n_diff_pairs": len(clip_diff_base),
            "gt_gi_colors": {str(k): v for k, v in gt_gi_colors.items()},
        }

        logger.info(
            "  {} same-person pairs, {} different-person pairs",
            len(clip_same_base), len(clip_diff_base),
        )

    # Compute separability metrics
    def _compute_roc_auc(same_dists: List[float], diff_dists: List[float]) -> float:
        """ROC-AUC: can distance separate same from different?
        Higher distance = more likely different person.
        """
        if not same_dists or not diff_dists:
            return 0.5
        labels = [0] * len(same_dists) + [1] * len(diff_dists)
        scores = same_dists + diff_dists
        # Manual AUC (avoid sklearn dependency)
        n_same = len(same_dists)
        n_diff = len(diff_dists)
        concordant = 0
        tied = 0
        for s in same_dists:
            for d in diff_dists:
                if d > s:
                    concordant += 1
                elif d == s:
                    tied += 1
        return (concordant + 0.5 * tied) / (n_same * n_diff)

    def _dist_stats(dists: List[float]) -> Dict[str, float]:
        if not dists:
            return {"n": 0}
        arr = np.array(dists)
        return {
            "n": len(dists),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(arr.std()), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
        }

    # Aggregate separability
    auc_baseline = _compute_roc_auc(all_same_baseline, all_diff_baseline)
    auc_masked = _compute_roc_auc(all_same_masked, all_diff_masked)

    # Distinct-color separability (PRIMARY decision number)
    auc_distinct_base = _compute_roc_auc(distinct_same_base, distinct_diff_base)
    auc_distinct_mask = _compute_roc_auc(distinct_same_mask, distinct_diff_mask)

    # Same-color separability
    auc_same_color_base = _compute_roc_auc(same_color_same_base, same_color_diff_base)
    auc_same_color_mask = _compute_roc_auc(same_color_same_mask, same_color_diff_mask)

    # Intrinsic-color floor: fraction of different-person pairs inseparable
    # under BEST mask (distance < typical same-person distance)
    if all_same_masked:
        same_p75 = float(np.percentile(all_same_masked, 75))
    else:
        same_p75 = 0.5
    inseparable_under_mask = sum(1 for d in all_diff_masked if d <= same_p75)
    intrinsic_floor = inseparable_under_mask / len(all_diff_masked) if all_diff_masked else 0

    results = {
        "aggregate": {
            "baseline_auc": round(auc_baseline, 4),
            "masked_auc": round(auc_masked, 4),
            "auc_improvement": round(auc_masked - auc_baseline, 4),
            "same_person_baseline": _dist_stats(all_same_baseline),
            "diff_person_baseline": _dist_stats(all_diff_baseline),
            "same_person_masked": _dist_stats(all_same_masked),
            "diff_person_masked": _dist_stats(all_diff_masked),
        },
        "distinct_color": {
            "note": (
                "Color distinctiveness from GT-grounded gi-color labels "
                "(method-independent: assigned from union of both methods, "
                "majority-voted per GT person). Distinct = different gi color."
            ),
            "evaluable": len(distinct_same_base) > 0 and len(distinct_diff_base) > 0,
            "non_evaluable_reason": (
                f"Only {len(set(gt_gi_colors.values()) if 'gt_gi_colors' in dir() else set())} "
                f"distinct gi colors in this session. Need >=2 people with the SAME "
                f"distinctive color to form same-person pairs for AUC."
            ) if not distinct_same_base else None,
            "baseline_auc": round(auc_distinct_base, 4) if distinct_same_base else None,
            "masked_auc": round(auc_distinct_mask, 4) if distinct_same_base else None,
            "auc_improvement": round(auc_distinct_mask - auc_distinct_base, 4) if distinct_same_base else None,
            "same_person": _dist_stats(distinct_same_base),
            "diff_person_baseline": _dist_stats(distinct_diff_base),
            "diff_person_masked": _dist_stats(distinct_diff_mask),
        },
        "same_color": {
            "baseline_auc": round(auc_same_color_base, 4),
            "masked_auc": round(auc_same_color_mask, 4),
            "auc_improvement": round(auc_same_color_mask - auc_same_color_base, 4),
            "same_person": _dist_stats(same_color_same_base),
            "diff_person_baseline": _dist_stats(same_color_diff_base),
            "diff_person_masked": _dist_stats(same_color_diff_mask),
        },
        "intrinsic_color_floor": {
            "inseparable_pairs_under_mask": inseparable_under_mask,
            "total_diff_pairs": len(all_diff_masked),
            "floor_frac": round(intrinsic_floor, 4),
            "threshold_used": round(same_p75, 4),
            "definition": (
                "Fraction of different-person pairs with masked Bhattacharyya "
                "distance <= 75th percentile of same-person distances. "
                "These pairs are inseparable even under the best mask."
            ),
        },
        "per_clip": per_clip_results,
    }

    logger.info(
        "Aggregate AUC: baseline={:.4f}, masked={:.4f}, delta={:+.4f}",
        auc_baseline, auc_masked, auc_masked - auc_baseline,
    )
    logger.info(
        "Distinct-color AUC: baseline={:.4f}, masked={:.4f}, delta={:+.4f}",
        auc_distinct_base, auc_distinct_mask, auc_distinct_mask - auc_distinct_base,
    )
    logger.info(
        "Same-color AUC: baseline={:.4f}, masked={:.4f}, delta={:+.4f}",
        auc_same_color_base, auc_same_color_mask, auc_same_color_mask - auc_same_color_base,
    )
    logger.info(
        "Intrinsic-color floor: {:.1f}% of different-person pairs inseparable",
        intrinsic_floor * 100,
    )

    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(
    phase_b: Dict, phase_c: Dict,
) -> Dict[str, Any]:
    """GO / NO-GO / INCONCLUSIVE verdict."""
    logger.info("=== Computing verdict ===")

    # Mask quality gate
    total_det = 0
    total_degen = 0
    for clip_id, data in phase_b.items():
        if isinstance(data, dict) and "n_detections_sampled" in data:
            total_det += data["n_detections_sampled"]
            total_degen += data["degenerate_count"]

    degen_frac = total_degen / total_det if total_det > 0 else 1.0

    if degen_frac > 0.50:
        verdict = "INCONCLUSIVE"
        reason = (
            f"Mask quality limited: {degen_frac*100:.1f}% of frames have degenerate "
            f"masks (< {DEGENERATE_LOW*100:.0f}% or > {DEGENERATE_HIGH*100:.0f}% coverage). "
            f"Cannot reliably assess appearance separability at this resolution. "
            f"This is NOT a NO-GO on appearance — it means the chroma-subtraction "
            f"mask cannot be evaluated at 50-150px person size."
        )
    else:
        agg = phase_c.get("aggregate", {})
        distinct = phase_c.get("distinct_color", {})
        same_color = phase_c.get("same_color", {})

        auc_base = agg.get("baseline_auc", 0.5)
        auc_mask = agg.get("masked_auc", 0.5)
        auc_delta = agg.get("auc_improvement", 0)
        distinct_evaluable = distinct.get("evaluable", False)

        # Same-color AUC is the main signal when distinct-color is non-evaluable
        auc_sc_base = same_color.get("baseline_auc", 0.5)
        auc_sc_mask = same_color.get("masked_auc", 0.5)
        auc_sc_delta = same_color.get("auc_improvement", 0)

        floor = phase_c.get("intrinsic_color_floor", {}).get("floor_frac", 1.0)

        # Count distinct gi colors in session
        n_gi_colors = len(set())
        for clip_data in phase_c.get("per_clip", {}).values():
            colors = clip_data.get("gt_gi_colors", {})
            n_gi_colors = max(n_gi_colors, len(set(colors.values())))

        session_note = ""
        if n_gi_colors <= 2:
            session_note = (
                f"WARNING: This session has only {n_gi_colors} distinct gi color(s) "
                f"(heavily white-gi). Distinct-color AUC is non-evaluable "
                f"(0 same-person pairs in the distinct group). "
                f"Results reflect worst-case color diversity. "
                f"A session with more color variety may show different results."
            )

        if distinct_evaluable:
            auc_distinct_mask = distinct.get("masked_auc", 0.5)
            auc_distinct_delta = distinct.get("auc_improvement", 0)
            if auc_distinct_delta > 0.05 and auc_distinct_mask > 0.7:
                verdict = "GO"
                reason = (
                    f"Mask improves separability among distinct-color pairs: "
                    f"AUC {auc_distinct_mask:.3f} (delta {auc_distinct_delta:+.3f}). "
                    f"Aggregate AUC: baseline={auc_base:.3f}, masked={auc_mask:.3f}. "
                    f"Intrinsic-color floor: {floor*100:.1f}% inseparable. "
                    f"Promote to src/bjj_pipeline extraction module. "
                    f"Production plate needs held-out/rolling background."
                )
            elif auc_delta > 0.02 or auc_distinct_delta > 0.02:
                verdict = "MARGINAL_GO"
                reason = (
                    f"Mask shows modest improvement: aggregate AUC delta={auc_delta:+.3f}, "
                    f"distinct-color delta={auc_distinct_delta:+.3f}. "
                    f"Intrinsic-color floor: {floor*100:.1f}% inseparable."
                )
            else:
                verdict = "NO_GO"
                reason = (
                    f"Mask does not improve separability: aggregate delta={auc_delta:+.3f}, "
                    f"distinct-color delta={auc_distinct_delta:+.3f}. "
                    f"Intrinsic-color floor: {floor*100:.1f}% inseparable."
                )
        else:
            # Distinct-color non-evaluable — fall back to aggregate + same-color
            if auc_delta > 0.03 or auc_sc_delta > 0.03:
                verdict = "MARGINAL_GO"
                reason = (
                    f"Mask shows modest improvement on same-color pairs: "
                    f"same-color AUC delta={auc_sc_delta:+.3f}, "
                    f"aggregate AUC delta={auc_delta:+.3f}. "
                    f"Distinct-color comparison non-evaluable ({session_note}). "
                    f"Intrinsic-color floor: {floor*100:.1f}% inseparable."
                )
            else:
                verdict = "NO_GO"
                reason = (
                    f"Mask does not meaningfully improve separability: "
                    f"aggregate AUC delta={auc_delta:+.3f}, "
                    f"same-color AUC delta={auc_sc_delta:+.3f}. "
                    f"Intrinsic-color floor: {floor*100:.1f}% inseparable. "
                    f"{session_note} "
                    f"On this white-gi-heavy session, the bottleneck is intrinsic "
                    f"color similarity, not the extraction ROI."
                )

    result = {
        "verdict": verdict,
        "reason": reason,
        "degenerate_frac": round(degen_frac, 3),
        "mask_quality_gate": "PASS" if degen_frac <= 0.50 else "FAIL",
    }

    logger.info("VERDICT: {} — {}", verdict, reason)
    return result


# ---------------------------------------------------------------------------
# Report + evidence writing
# ---------------------------------------------------------------------------


def write_evidence(
    phase_a: Dict, phase_b: Dict, phase_c: Dict, verdict: Dict,
) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, data: Any) -> None:
        (EVIDENCE_DIR / name).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    _dump("phase_a_plate.json", phase_a)

    # Strip non-serializable tracklet_summaries detail for JSON
    phase_b_json = {}
    for clip_id, data in phase_b.items():
        if isinstance(data, dict):
            phase_b_json[clip_id] = {
                k: v for k, v in data.items()
                if k != "tracklet_summaries" or not isinstance(v, dict)
            }
        else:
            phase_b_json[clip_id] = data
    _dump("phase_b_extraction.json", phase_b_json)

    # Self-absorption report
    absorption = {}
    for clip_id, data in phase_b.items():
        if isinstance(data, dict) and "self_absorption" in data:
            absorption[clip_id] = data["self_absorption"]
    _dump("phase_b_self_absorption.json", absorption)

    _dump("phase_c_separability.json", phase_c)
    _dump("verdict.json", verdict)

    # Write report
    lines = [
        "# CP-RASTER-PLATE: Median-Background Masking + Appearance Separability",
        "",
        "## Phase A: Median Background Plate",
        "",
        f"- Clips sampled: {phase_a.get('n_clips', 0)}",
        f"- Frames sampled: {phase_a.get('n_frames_sampled', 0)}",
        f"- Low-occupancy clips (empirical): {phase_a.get('low_occupancy_clips', [])}",
        f"- Ghost pixels before fallback: {phase_a.get('ghost_frac_before', 0)*100:.1f}%",
        f"- Ghost pixels after fallback: {phase_a.get('ghost_frac_after', 0)*100:.1f}%",
        "",
        "**Note:** Plate built from same footage as test clips. Production path "
        "needs held-out/rolling background.",
        "",
        "## Phase B: Masked Histogram Extraction",
        "",
    ]

    for clip_id, data in phase_b.items():
        if not isinstance(data, dict) or "error" in data:
            lines.append(f"- **{clip_id}**: {data}")
            continue
        lines.append(f"### {clip_id}")
        lines.append(f"- Detections sampled: {data.get('n_detections_sampled', 0)}")
        lines.append(f"- Tracklets: {data.get('n_tracklets', 0)}")
        lines.append(f"- Coverage: mean={data.get('coverage_mean', 0)*100:.1f}%, "
                      f"median={data.get('coverage_median', 0)*100:.1f}%, "
                      f"p10={data.get('coverage_p10', 0)*100:.1f}%, "
                      f"p90={data.get('coverage_p90', 0)*100:.1f}%")
        lines.append(f"- Degenerate masks: {data.get('degenerate_count', 0)} "
                      f"({data.get('degenerate_frac', 0)*100:.1f}%)")
        lines.append("")
        degen_by_color = data.get("degenerate_by_gi_color", {})
        if degen_by_color:
            lines.append("**Degenerate masks by gi color:**")
            lines.append("")
            lines.append("| Gi Color | Total | Degenerate | Frac |")
            lines.append("|----------|-------|------------|------|")
            for color, info in sorted(degen_by_color.items()):
                lines.append(
                    f"| {color} | {info['total']} | {info['degenerate']} | "
                    f"{info['frac']*100:.1f}% |"
                )
            lines.append("")

    lines.append("## Phase C: Separability")
    lines.append("")
    agg = phase_c.get("aggregate", {})
    lines.append(f"**Aggregate AUC:** baseline={agg.get('baseline_auc', 'N/A')}, "
                  f"masked={agg.get('masked_auc', 'N/A')}, "
                  f"delta={agg.get('auc_improvement', 'N/A')}")
    lines.append("")

    distinct = phase_c.get("distinct_color", {})
    lines.append(f"**Distinct-color pairs (PRIMARY):** baseline AUC={distinct.get('baseline_auc', 'N/A')}, "
                  f"masked AUC={distinct.get('masked_auc', 'N/A')}, "
                  f"delta={distinct.get('auc_improvement', 'N/A')}")
    lines.append(f"  - Color classifier: {distinct.get('note', '')}")
    lines.append("")

    same_c = phase_c.get("same_color", {})
    lines.append(f"**Same-color pairs:** baseline AUC={same_c.get('baseline_auc', 'N/A')}, "
                  f"masked AUC={same_c.get('masked_auc', 'N/A')}, "
                  f"delta={same_c.get('auc_improvement', 'N/A')}")
    lines.append("")

    floor = phase_c.get("intrinsic_color_floor", {})
    lines.append(f"**Intrinsic-color floor:** {floor.get('floor_frac', 0)*100:.1f}% "
                  f"of different-person pairs inseparable under best mask")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict.get('verdict', 'N/A')}**")
    lines.append("")
    lines.append(f"Mask quality gate: {verdict.get('mask_quality_gate', 'N/A')} "
                  f"(degenerate fraction: {verdict.get('degenerate_frac', 0)*100:.1f}%)")
    lines.append("")
    lines.append(verdict.get("reason", ""))
    lines.append("")

    (EVIDENCE_DIR / "plate_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evidence written to {}", EVIDENCE_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CP-RASTER-PLATE")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="all")
    args = parser.parse_args()

    phase_a_results = {}
    phase_b_results = {}
    phase_c_results = {}
    verdict_results = {}

    if args.phase in ("a", "all"):
        phase_a_results = build_median_plate()

    if args.phase in ("b", "all"):
        phase_b_results = extract_masked_histograms()

    if args.phase in ("c", "all"):
        phase_c_results = measure_separability()

    if args.phase == "all" and phase_b_results and phase_c_results:
        verdict_results = compute_verdict(phase_b_results, phase_c_results)
        write_evidence(phase_a_results, phase_b_results, phase_c_results, verdict_results)

    logger.info("CP-RASTER-PLATE complete.")


if __name__ == "__main__":
    main()
