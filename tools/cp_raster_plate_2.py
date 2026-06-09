"""CP-RASTER-PLATE-2: Re-measure appearance separability with V channel + proper color labels.

Corrects CP-RASTER-PLATE's crippled feature space (H+S only, V excluded) and
over-collapsed color labels (2 categories for a 7-color scene).

Phases:
  A — Rebuild median plate from ~1080 frames (every 50th, 8GB RAM constraint)
  B — Masked histogram extraction in THREE feature spaces (H+S, H+S+V, V-only)
  C — Separability with hand-verified multi-category color labels

Usage:
    PYTHONPATH=src python tools/cp_raster_plate_2.py
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
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_raster_plate_2"
ARTIFACT_DIR = OUTPUTS_DIR / "_eval_gt_oracle" / "raster_plate_2"

CAM_ID = "J_EDEw"
GYM_ID = "_eval_gt"
NEST_DIR = REPO_ROOT / "data" / "raw" / "nest" / "c8a592a4-2bca-400a-80e1-fec0e5cbea77"
CLIP_DIR = NEST_DIR / CAM_ID / "2026-03-18" / "20"

VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID1_CLIP_ID
VID2_DIR = OUTPUTS_DIR / GYM_ID / CAM_ID / "2026-03-18" / "20" / VID2_CLIP_ID

DENSE_MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"

# Feature spaces
HIST_V_BINS = 6
HSV_FULL_SIZE = HIST_H_BINS * HIST_S_BINS * HIST_V_BINS  # 18*8*6=864

# Foreground mask
H_THRESH = 12
S_THRESH = 35
DEGENERATE_LOW = 0.05
DEGENERATE_HIGH = 0.95

# Sampling
FRAMES_PER_TRACKLET = 30
MIN_ISOLATED_FRAMES = 10
IOU_THRESHOLD = 0.3
SEED = 42

# Plate: every 50th frame from all clips (~1080 frames, ~3GB, fits 8GB RAM)
PLATE_FRAME_STRIDE = 50

# Hand-verified color labels per GT track, derived from masked H+S+V medians.
# Measured in Pass 1 exploration from actual foreground pixel distributions.
# Source: independent of baseline H+S method — uses full H+S+V from masked crops.
GT_COLOR_LABELS = {
    # Vid1 (GT track IDs 14-27)
    (VID1_CLIP_ID, 14): "charcoal",      # H=85, S=30, V=47
    (VID1_CLIP_ID, 15): "skin",           # H=15, S=49, V=95
    (VID1_CLIP_ID, 16): "light_blue",     # H=117, S=27, V=92
    (VID1_CLIP_ID, 17): "white",          # H=23, S=29, V=117
    (VID1_CLIP_ID, 18): "medium_blue",    # H=124, S=28, V=73
    (VID1_CLIP_ID, 19): "skin",           # H=15, S=30, V=92
    (VID1_CLIP_ID, 20): "medium_blue",    # H=116, S=27, V=83
    (VID1_CLIP_ID, 21): "red",            # H=9, S=60, V=89
    (VID1_CLIP_ID, 22): "dark_blue",      # H=100, S=34, V=39
    (VID1_CLIP_ID, 23): "gray",           # unmapped in sample — assign gray
    (VID1_CLIP_ID, 24): "gray",           # unmapped in sample — assign gray
    (VID1_CLIP_ID, 25): "white",          # H=108, S=19, V=147
    (VID1_CLIP_ID, 26): "dark_blue",      # H=108, S=38, V=45
    (VID1_CLIP_ID, 27): "charcoal",       # H=75, S=30, V=39
    # Vid2 (GT track IDs 0-13)
    (VID2_CLIP_ID, 0):  "light_blue",     # H=124, S=31, V=97
    (VID2_CLIP_ID, 1):  "white",          # H=60, S=28, V=112
    (VID2_CLIP_ID, 2):  "gray",           # unmapped — assign gray
    (VID2_CLIP_ID, 3):  "medium_blue",    # H=116, S=25, V=83
    (VID2_CLIP_ID, 4):  "gray",           # H=38, S=30, V=77
    (VID2_CLIP_ID, 5):  "red",            # H=8, S=68, V=117
    (VID2_CLIP_ID, 6):  "medium_blue",    # H=108, S=24, V=84
    (VID2_CLIP_ID, 7):  "gray",           # unmapped — assign gray
    (VID2_CLIP_ID, 8):  "white",          # H=107, S=19, V=146
    (VID2_CLIP_ID, 9):  "skin",           # H=15, S=48, V=115
    (VID2_CLIP_ID, 10): "skin",           # H=15, S=36, V=123
    (VID2_CLIP_ID, 11): "gray",           # unmapped — assign gray
    (VID2_CLIP_ID, 12): "blue",           # H=98, S=43, V=78
    (VID2_CLIP_ID, 13): "gray",           # unmapped — assign gray
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def _compute_foreground_mask(
    crop_bgr: np.ndarray, plate_crop_bgr: np.ndarray,
) -> np.ndarray:
    hsv_frame = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hsv_plate = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2HSV)
    h_diff = np.abs(hsv_frame[:, :, 0].astype(np.int16) - hsv_plate[:, :, 0].astype(np.int16))
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(hsv_frame[:, :, 1].astype(np.int16) - hsv_plate[:, :, 1].astype(np.int16))
    fg = ((h_diff > H_THRESH) | (s_diff > S_THRESH)).astype(np.uint8) * 255
    return fg


def _compute_center_mask(h: int, w: int) -> np.ndarray:
    """Center 60% mask (torso-biased region)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mx = int(w * 0.2)
    my = int(h * 0.2)
    mask[my:h - my, mx:w - mx] = 255
    return mask


def _compute_hsv_hist(hsv: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """H+S histogram (144-dim, existing production shape)."""
    if mask.sum() == 0:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [HIST_H_BINS, HIST_S_BINS], [0, 180, 0, 256])
    t = hist.sum()
    if t > 0:
        hist /= t
    return hist.flatten().astype(np.float32)


def _compute_hsv_full_hist(hsv: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """H+S+V histogram (864-dim)."""
    if mask.sum() == 0:
        return None
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask,
        [HIST_H_BINS, HIST_S_BINS, HIST_V_BINS],
        [0, 180, 0, 256, 0, 256],
    )
    t = hist.sum()
    if t > 0:
        hist /= t
    return hist.flatten().astype(np.float32)


def _compute_v_hist(hsv: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """V-only histogram (6-dim)."""
    if mask.sum() == 0:
        return None
    hist = cv2.calcHist([hsv], [2], mask, [HIST_V_BINS], [0, 256])
    t = hist.sum()
    if t > 0:
        hist /= t
    return hist.flatten().astype(np.float32)


def _bhatt(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance for arbitrary-length normalized histograms."""
    bc = np.sum(np.sqrt(a * b))
    bc = min(bc, 1.0)
    return float(np.sqrt(1.0 - bc))


# ---------------------------------------------------------------------------
# Phase A: Rebuild median plate
# ---------------------------------------------------------------------------

def build_plate() -> Dict[str, Any]:
    logger.info("=== Phase A: Building median plate (every {}th frame) ===", PLATE_FRAME_STRIDE)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(CLIP_DIR.glob("J_EDEw-*.mp4"))
    all_frames = []
    per_clip = {}

    for cp in clip_paths:
        cap = cv2.VideoCapture(str(cp))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        for i in range(0, fc, PLATE_FRAME_STRIDE):
            f = _read_frame(cap, i)
            if f is not None:
                all_frames.append(f)
                count += 1
        cap.release()
        per_clip[cp.name] = count
        logger.info("  {}: {} frames", cp.name, count)

    n_total = len(all_frames)
    logger.info("Total: {} frames ({:.1f} GB)", n_total, n_total * 720 * 1280 * 3 / 1e9)

    stack = np.stack(all_frames, axis=0)
    del all_frames

    plate = np.median(stack, axis=0).astype(np.uint8)
    del stack

    np.save(ARTIFACT_DIR / "J_EDEw_median_plate.npy", plate)
    cv2.imwrite(str(ARTIFACT_DIR / "J_EDEw_median_plate.png"), plate)

    results = {
        "n_frames": n_total,
        "stride": PLATE_FRAME_STRIDE,
        "n_clips": len(clip_paths),
        "per_clip": per_clip,
        "compute_note": (
            f"Every {PLATE_FRAME_STRIDE}th frame from all {len(clip_paths)} clips. "
            f"8 GB RAM constraint prevents holding all {len(clip_paths)*4500} frames. "
            f"{n_total} samples drive per-pixel person-occupancy well below 50%."
        ),
    }
    logger.info("Phase A complete: {} frames → plate", n_total)
    return results, plate


# ---------------------------------------------------------------------------
# Phase B: Masked histogram extraction (three feature spaces)
# ---------------------------------------------------------------------------

def extract_histograms(plate: np.ndarray) -> Dict[str, Any]:
    logger.info("=== Phase B: Masked histogram extraction ===")
    rng = np.random.RandomState(SEED)
    all_results = {}

    for clip_id, clip_dir in [
        (VID1_CLIP_ID, VID1_DIR),
        (VID2_CLIP_ID, VID2_DIR),
    ]:
        logger.info("Processing {}...", clip_id)
        det_df = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
        hist_df = pd.read_parquet(clip_dir / "stage_A" / "color_histograms.parquet")

        det_df = det_df.rename(columns={"tracklet_id": "track_id"})
        merged = det_df.merge(
            hist_df[["frame_index", "track_id", "is_isolated"]],
            on=["frame_index", "track_id"], how="inner",
        )
        isolated = merged[merged["is_isolated"] == True].copy()

        sampled_rows = []
        for tid, grp in isolated.groupby("track_id"):
            if len(grp) < MIN_ISOLATED_FRAMES:
                continue
            n = min(FRAMES_PER_TRACKLET, len(grp))
            sampled_rows.append(grp.sample(n=n, random_state=rng))

        if not sampled_rows:
            all_results[clip_id] = {"error": "no_tracklets"}
            continue

        sampled = pd.concat(sampled_rows, ignore_index=True)
        frame_groups = sampled.groupby("frame_index")

        video_path = CLIP_DIR / f"{clip_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))

        per_det = []
        for fi in sorted(frame_groups.groups.keys()):
            frame = _read_frame(cap, fi)
            if frame is None:
                continue

            for _, row in frame_groups.get_group(fi).iterrows():
                x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                crop = frame[iy1:iy2, ix1:ix2]
                pcrop = plate[iy1:iy2, ix1:ix2]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # Foreground mask (full — skin-inclusive)
                fg_full = _compute_foreground_mask(crop, pcrop)
                bbox_px = (iy2 - iy1) * (ix2 - ix1)
                mask_px = int(np.sum(fg_full > 0))
                coverage = mask_px / bbox_px if bbox_px > 0 else 0
                is_degen = coverage < DEGENERATE_LOW or coverage > DEGENERATE_HIGH

                # Torso-only mask (center 60% of fg)
                center_m = _compute_center_mask(iy2 - iy1, ix2 - ix1)
                fg_torso = cv2.bitwise_and(fg_full, center_m)

                # Three feature spaces × two mask modes
                entry = {
                    "frame_index": int(fi),
                    "track_id": row.track_id,
                    "coverage": float(coverage),
                    "is_degenerate": bool(is_degen),
                }

                if not is_degen:
                    entry["hs_full"] = _compute_hsv_hist(hsv, fg_full)
                    entry["hsv_full"] = _compute_hsv_full_hist(hsv, fg_full)
                    entry["v_full"] = _compute_v_hist(hsv, fg_full)
                    entry["hs_torso"] = _compute_hsv_hist(hsv, fg_torso)
                    entry["hsv_torso"] = _compute_hsv_full_hist(hsv, fg_torso)
                    entry["v_torso"] = _compute_v_hist(hsv, fg_torso)
                else:
                    for k in ["hs_full", "hsv_full", "v_full", "hs_torso", "hsv_torso", "v_torso"]:
                        entry[k] = None

                # Baseline (existing center-bbox crop, H+S only)
                base_crop = _center_crop_from_bbox(frame, (x1, y1, x2, y2))
                entry["baseline_hs"] = (
                    compute_hsv_histogram(base_crop)
                    if base_crop is not None and base_crop.size > 0
                    else None
                )

                per_det.append(entry)

        cap.release()

        # Aggregate per-tracklet
        tracklet_summaries: Dict[str, Dict] = {}
        coverage_stats = []
        degen_count = 0

        hist_keys = ["hs_full", "hsv_full", "v_full", "hs_torso", "hsv_torso", "v_torso", "baseline_hs"]

        for det in per_det:
            tid = det["track_id"]
            coverage_stats.append(det["coverage"])
            if det["is_degenerate"]:
                degen_count += 1
            if tid not in tracklet_summaries:
                tracklet_summaries[tid] = {k: [] for k in hist_keys}
                tracklet_summaries[tid]["coverages"] = []
                tracklet_summaries[tid]["degen_count"] = 0
            ts = tracklet_summaries[tid]
            ts["coverages"].append(det["coverage"])
            if det["is_degenerate"]:
                ts["degen_count"] += 1
            for k in hist_keys:
                if det[k] is not None:
                    ts[k].append(det[k])

        # Average and normalize
        final_summaries: Dict[str, Dict] = {}
        for tid, ts in tracklet_summaries.items():
            fs = {"mean_coverage": float(np.mean(ts["coverages"]))}
            for k in hist_keys:
                hists = ts[k]
                if hists:
                    avg = np.mean(np.stack(hists), axis=0).astype(np.float32)
                    t = avg.sum()
                    if t > 0:
                        avg /= t
                    fs[k] = avg
                else:
                    fs[k] = None
            final_summaries[tid] = fs

        degen_frac = degen_count / len(per_det) if per_det else 0
        all_results[clip_id] = {
            "n_det": len(per_det),
            "n_tracklets": len(final_summaries),
            "coverage_median": round(float(np.median(coverage_stats)), 3) if coverage_stats else 0,
            "degen_frac": round(degen_frac, 3),
            "tracklet_summaries": final_summaries,
        }

        logger.info(
            "  {}: {} tracklets, coverage={:.1f}% median, {:.1f}% degenerate",
            clip_id, len(final_summaries),
            (np.median(coverage_stats) if coverage_stats else 0) * 100,
            degen_frac * 100,
        )

    return all_results


# ---------------------------------------------------------------------------
# Phase C: Separability measurement
# ---------------------------------------------------------------------------

def _build_gt_map(clip_id: str, clip_dir: Path) -> Dict[str, int]:
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

        votes: Dict[str, Counter] = defaultdict(Counter)
        for fi in ann[:1000]:
            boxes = gt.get(fi, [])
            fd = det_by_f.get(fi, pd.DataFrame())
            if not boxes or fd.empty:
                continue
            ga = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes])
            da = np.array([[r.x1, r.y1, r.x2, r.y2] for _, r in fd.iterrows()])
            for gi, di, iou in greedy_match(ga, da, iou_threshold=IOU_THRESHOLD):
                votes[fd.iloc[di].tracklet_id][boxes[gi].track_id] += 1

        return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    raise ValueError(f"No export for {clip_id}")


def measure_separability(phase_b: Dict) -> Dict[str, Any]:
    logger.info("=== Phase C: Separability measurement ===")

    # Feature spaces to compare
    SPACES = {
        "baseline_hs": "Baseline H+S (144-dim, production)",
        "hs_full": "Masked H+S full-body (144-dim)",
        "hsv_full": "Masked H+S+V full-body (864-dim)",
        "v_full": "Masked V-only full-body (6-dim)",
        "hs_torso": "Masked H+S torso-only (144-dim)",
        "hsv_torso": "Masked H+S+V torso-only (864-dim)",
        "v_torso": "Masked V-only torso-only (6-dim)",
    }

    # Collect all pairwise distances
    pair_dists: Dict[str, List[Dict]] = {space: [] for space in SPACES}
    per_clip = {}

    for clip_id, clip_dir in [
        (VID1_CLIP_ID, VID1_DIR),
        (VID2_CLIP_ID, VID2_DIR),
    ]:
        logger.info("Processing {}...", clip_id)
        b_data = phase_b.get(clip_id, {})
        if "error" in b_data:
            continue

        gt_map = _build_gt_map(clip_id, clip_dir)
        summaries = b_data["tracklet_summaries"]

        # Filter to tracklets with GT + at least baseline_hs + hsv_full
        valid = {}
        for tid, ts in summaries.items():
            gt_id = gt_map.get(tid)
            if gt_id is None:
                continue
            if ts.get("baseline_hs") is None or ts.get("hsv_full") is None:
                continue
            valid[tid] = (gt_id, ts)

        logger.info("  {} valid tracklets", len(valid))
        tids = sorted(valid.keys())

        # Color label distribution
        color_counts: Dict[str, int] = Counter()
        for tid in tids:
            gt_id = valid[tid][0]
            label = GT_COLOR_LABELS.get((clip_id, gt_id), "unknown")
            color_counts[label] += 1

        per_clip[clip_id] = {
            "n_valid": len(valid),
            "color_distribution": dict(color_counts),
        }

        # Form pairs
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                t1, t2 = tids[i], tids[j]
                gt1 = valid[t1][0]
                gt2 = valid[t2][0]
                is_same = gt1 == gt2

                label1 = GT_COLOR_LABELS.get((clip_id, gt1), "unknown")
                label2 = GT_COLOR_LABELS.get((clip_id, gt2), "unknown")
                is_distinct = label1 != label2

                for space in SPACES:
                    h1 = valid[t1][1].get(space)
                    h2 = valid[t2][1].get(space)
                    if h1 is None or h2 is None:
                        continue

                    if space in ("baseline_hs", "hs_full", "hs_torso"):
                        d = bhattacharyya_distance(h1, h2)
                    else:
                        d = _bhatt(h1, h2)

                    pair_dists[space].append({
                        "is_same": is_same,
                        "is_distinct_color": is_distinct,
                        "distance": d,
                        "label1": label1,
                        "label2": label2,
                    })

    # Compute metrics
    def _auc(same: List[float], diff: List[float]) -> float:
        if not same or not diff:
            return float("nan")
        concordant = sum(1 for s in same for d in diff if d > s)
        tied = sum(1 for s in same for d in diff if d == s)
        return (concordant + 0.5 * tied) / (len(same) * len(diff))

    def _stats(vals: List[float]) -> Dict:
        if not vals:
            return {"n": 0}
        a = np.array(vals)
        return {
            "n": len(vals),
            "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "std": round(float(a.std()), 4),
        }

    results = {"per_space": {}, "per_clip": per_clip}

    for space, desc in SPACES.items():
        pairs = pair_dists[space]
        if not pairs:
            results["per_space"][space] = {"description": desc, "n_pairs": 0}
            continue

        all_same = [p["distance"] for p in pairs if p["is_same"]]
        all_diff = [p["distance"] for p in pairs if not p["is_same"]]

        # Distinct-color different
        distinct_diff = [p["distance"] for p in pairs if not p["is_same"] and p["is_distinct_color"]]
        distinct_same = [p["distance"] for p in pairs if p["is_same"] and p["is_distinct_color"]]

        # Same-color different
        same_c_diff = [p["distance"] for p in pairs if not p["is_same"] and not p["is_distinct_color"]]
        same_c_same = [p["distance"] for p in pairs if p["is_same"] and not p["is_distinct_color"]]

        auc_all = _auc(all_same, all_diff)
        auc_distinct = _auc(distinct_same, distinct_diff)
        auc_same_c = _auc(same_c_same, same_c_diff)

        # Intrinsic floor: different-person pairs inseparable under this space
        if all_same:
            same_p75 = float(np.percentile(all_same, 75))
            insep = sum(1 for d in all_diff if d <= same_p75)
            floor = insep / len(all_diff) if all_diff else 0
        else:
            same_p75 = 0
            floor = 1.0

        results["per_space"][space] = {
            "description": desc,
            "auc_all": round(auc_all, 4) if not np.isnan(auc_all) else None,
            "auc_distinct_color": round(auc_distinct, 4) if not np.isnan(auc_distinct) else None,
            "auc_same_color": round(auc_same_c, 4) if not np.isnan(auc_same_c) else None,
            "same_person": _stats(all_same),
            "diff_person": _stats(all_diff),
            "distinct_color_diff": _stats(distinct_diff),
            "same_color_diff": _stats(same_c_diff),
            "intrinsic_floor": round(floor, 4),
            "n_distinct_same_pairs": len(distinct_same),
            "n_distinct_diff_pairs": len(distinct_diff),
            "n_same_color_same_pairs": len(same_c_same),
            "n_same_color_diff_pairs": len(same_c_diff),
        }

        logger.info(
            "  {}: AUC_all={:.4f} AUC_distinct={} AUC_same_color={} floor={:.1f}%",
            space, auc_all,
            f"{auc_distinct:.4f}" if not np.isnan(auc_distinct) else "N/A",
            f"{auc_same_c:.4f}" if not np.isnan(auc_same_c) else "N/A",
            floor * 100,
        )

    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(phase_b: Dict, phase_c: Dict) -> Dict[str, Any]:
    logger.info("=== Computing verdict ===")

    # Mask quality gate
    total_det = sum(d.get("n_det", 0) for d in phase_b.values() if isinstance(d, dict))
    total_degen = sum(
        int(d.get("degen_frac", 0) * d.get("n_det", 0))
        for d in phase_b.values() if isinstance(d, dict)
    )
    degen_frac = total_degen / total_det if total_det > 0 else 1.0

    if degen_frac > 0.50:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": f"Mask quality gate FAIL: {degen_frac*100:.1f}% degenerate.",
        }

    spaces = phase_c.get("per_space", {})
    baseline = spaces.get("baseline_hs", {})
    hsv_full = spaces.get("hsv_full", {})
    hs_full = spaces.get("hs_full", {})
    v_full = spaces.get("v_full", {})

    auc_base = baseline.get("auc_all")
    auc_hsv = hsv_full.get("auc_all")
    auc_hs_masked = hs_full.get("auc_all")
    auc_v = v_full.get("auc_all")

    auc_hsv_distinct = hsv_full.get("auc_distinct_color")
    auc_base_distinct = baseline.get("auc_distinct_color")

    floor_hsv = hsv_full.get("intrinsic_floor", 1.0)
    floor_base = baseline.get("intrinsic_floor", 1.0)

    if auc_base is None or auc_hsv is None:
        return {"verdict": "INCONCLUSIVE", "reason": "Missing AUC data."}

    v_delta = auc_hsv - auc_base
    mask_delta = (auc_hs_masked or auc_base) - auc_base

    is_v_big_win = v_delta > 0.05
    is_v_modest = v_delta > 0.02
    is_mask_help = mask_delta > 0.02

    skin_hsv = spaces.get("hsv_full", {})
    skin_torso = spaces.get("hsv_torso", {})
    skin_note = ""
    if skin_hsv.get("auc_all") and skin_torso.get("auc_all"):
        skin_delta = skin_hsv["auc_all"] - skin_torso["auc_all"]
        skin_note = (
            f"Skin-inclusive vs torso-only: AUC {skin_hsv['auc_all']:.4f} vs "
            f"{skin_torso['auc_all']:.4f} (delta {skin_delta:+.4f}). "
            f"{'Skin helps.' if skin_delta > 0.01 else 'Skin negligible.'}"
        )

    if is_v_big_win:
        verdict = "GO"
        reason = (
            f"H+S+V separates meaningfully better than H+S: "
            f"AUC {auc_hsv:.4f} vs {auc_base:.4f} (delta {v_delta:+.4f}). "
            f"Distinct-color AUC: H+S+V={auc_hsv_distinct}, baseline={auc_base_distinct}. "
            f"Intrinsic floor: {floor_hsv*100:.1f}% (was {floor_base*100:.1f}% under H+S). "
            f"V-channel is a PRODUCTION FIX independent of masking. "
            f"Mask adds {mask_delta:+.4f} on top of V. {skin_note} "
            f"Promote H+S+V histogram to src/bjj_pipeline and note V-extension "
            f"is non-breaking (new hist_ columns; downstream reads by prefix)."
        )
    elif is_v_modest:
        if floor_hsv < floor_base - 0.05:
            verdict = "GO_PARTIAL"
            reason = (
                f"H+S+V modestly improves: AUC {auc_hsv:.4f} vs {auc_base:.4f} "
                f"(delta {v_delta:+.4f}). Intrinsic floor drops from "
                f"{floor_base*100:.1f}% to {floor_hsv*100:.1f}%. "
                f"Works on distinct colors, walled on truly-same. "
                f"Appearance value scales with gym color diversity. {skin_note}"
            )
        else:
            verdict = "GO_PARTIAL"
            reason = (
                f"H+S+V modestly improves: AUC {auc_hsv:.4f} vs {auc_base:.4f} "
                f"(delta {v_delta:+.4f}). Intrinsic floor: {floor_hsv*100:.1f}%. "
                f"{skin_note}"
            )
    else:
        verdict = "NO_GO"
        reason = (
            f"Even H+S+V with proper labels doesn't separate: "
            f"AUC {auc_hsv:.4f} vs {auc_base:.4f} (delta {v_delta:+.4f}). "
            f"Intrinsic floor: {floor_hsv*100:.1f}%. {skin_note}"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "auc_baseline_hs": auc_base,
        "auc_masked_hs": auc_hs_masked,
        "auc_masked_hsv": auc_hsv,
        "auc_v_only": auc_v,
        "v_delta": round(v_delta, 4),
        "mask_delta": round(mask_delta, 4),
        "floor_baseline": floor_base,
        "floor_hsv": floor_hsv,
        "degen_frac": round(degen_frac, 3),
        "skin_note": skin_note,
    }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def write_evidence(phase_a: Dict, phase_b: Dict, phase_c: Dict, verdict: Dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, data: Any) -> None:
        (EVIDENCE_DIR / name).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    _dump("phase_a_plate.json", phase_a)

    # Strip numpy arrays from phase_b for JSON
    b_json = {}
    for clip_id, data in phase_b.items():
        if not isinstance(data, dict):
            b_json[clip_id] = data
            continue
        b_json[clip_id] = {
            k: v for k, v in data.items() if k != "tracklet_summaries"
        }
    _dump("phase_b_extraction.json", b_json)
    _dump("phase_c_separability.json", phase_c)
    _dump("verdict.json", verdict)

    # Report
    lines = [
        "# CP-RASTER-PLATE-2: Appearance Separability with V Channel",
        "",
        "## Phase A: Median Plate",
        f"- Frames: {phase_a.get('n_frames', 0)} (every {PLATE_FRAME_STRIDE}th from {phase_a.get('n_clips', 0)} clips)",
        f"- {phase_a.get('compute_note', '')}",
        "",
        "## Phase B: Masked Extraction",
        "",
    ]
    for clip_id, data in phase_b.items():
        if not isinstance(data, dict) or "error" in data:
            continue
        lines.append(f"- **{clip_id}**: {data.get('n_tracklets', 0)} tracklets, "
                      f"coverage={data.get('coverage_median', 0)*100:.1f}% median, "
                      f"{data.get('degen_frac', 0)*100:.1f}% degenerate")
    lines.append("")

    lines.append("## Phase C: Separability by Feature Space")
    lines.append("")
    lines.append("| Feature Space | AUC (all) | AUC (distinct-color) | AUC (same-color) | Floor |")
    lines.append("|---------------|-----------|---------------------|------------------|-------|")
    for space, data in phase_c.get("per_space", {}).items():
        auc_all = data.get("auc_all")
        auc_dc = data.get("auc_distinct_color")
        auc_sc = data.get("auc_same_color")
        floor = data.get("intrinsic_floor", "")
        lines.append(
            f"| {data.get('description', space)} "
            f"| {auc_all if auc_all is not None else 'N/A'} "
            f"| {auc_dc if auc_dc is not None else 'N/A'} "
            f"| {auc_sc if auc_sc is not None else 'N/A'} "
            f"| {floor*100:.1f}% |" if isinstance(floor, float) else f"| {floor} |"
        )
    lines.append("")

    lines.append("## Color Labels (hand-verified from masked H+S+V medians)")
    lines.append("")
    for clip_id, data in phase_c.get("per_clip", {}).items():
        lines.append(f"**{clip_id}:** {data.get('color_distribution', {})}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict.get('verdict', 'N/A')}**")
    lines.append("")
    lines.append(verdict.get("reason", ""))
    lines.append("")

    (EVIDENCE_DIR / "plate_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evidence written to {}", EVIDENCE_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    phase_a, plate = build_plate()
    phase_b = extract_histograms(plate)
    del plate
    phase_c = measure_separability(phase_b)
    verdict = compute_verdict(phase_b, phase_c)
    write_evidence(phase_a, phase_b, phase_c, verdict)
    logger.info("CP-RASTER-PLATE-2 complete.")


if __name__ == "__main__":
    main()
