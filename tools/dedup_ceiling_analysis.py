#!/usr/bin/env python3
"""DEDUP-CEILING-1: What would perfect deduplication actually buy?

GT-labelled ceiling analysis. Merges concurrent-overlapping tracklets on the
same GT person (rule (c): >=50% overlap fraction), re-runs D0-D4 + Stage E,
re-runs GT matching and evaluation against scratch artifacts.

Usage:
    PYTHONPATH=src python tools/dedup_ceiling_analysis.py
"""

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLIP_ID = "FP7oJQ-20260822-132650"
CAMERA_ID = "FP7oJQ"
GYM_ID = "00000000-0000-0000-0000-000000000003"

PROD_CLIP_DIR = (
    REPO_ROOT / "outputs" / GYM_ID / "FP7oJQ" / "2026-08-22" / "13" / CLIP_ID
)
EVAL_DIR = REPO_ROOT / "outputs" / "_eval"
PFM_PATH = (
    EVAL_DIR / "stage_a" / "gt-eval-fp7oJQ-132650" / "FP7oJQ"
    / "per_frame_matches.parquet"
)
GT_ZIP = (
    REPO_ROOT / "data" / "training_data"
    / "ground_truth_FP7oJQ_20260822_132650.zip"
)

SCRATCH_ROOT = REPO_ROOT / "outputs" / "_dedup_ceiling"
SCRATCH_CLIP = SCRATCH_ROOT / CLIP_ID
SCRATCH_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "dedup_ceiling_1"

MIN_OVERLAP_FRACTION = 0.50  # Rule (c)


def build_merge_set(det: pd.DataFrame, pfm: pd.DataFrame):
    """Identify concurrent-overlap merge pairs using rule (c).

    Returns dict: absorbed_tid -> canonical_tid, and diagnostics.
    """
    det_tid = det[["frame_index", "detection_id", "tracklet_id"]].copy()
    pfm_matched = pfm[pfm["match_status"] == "matched"].copy()
    pfm_with_tid = pfm_matched.merge(
        det_tid,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
    )
    pfm_with_tid = pfm_with_tid.dropna(subset=["tracklet_id"])

    tid_gt_dom = pfm_with_tid.groupby("tracklet_id")["gt_track_id"].agg(
        lambda x: int(x.value_counts().index[0])
    )

    tid_frames = {}
    for tid, grp in det.groupby("tracklet_id"):
        tid_frames[tid] = set(grp["frame_index"].values)

    gt_tids = defaultdict(list)
    for tid, gt in tid_gt_dom.items():
        gt_tids[gt].append(tid)

    # For each concurrent pair, shorter absorbed by longer
    import itertools

    dup_to_canon = {}
    pair_overlaps = {}
    for gt, tids in gt_tids.items():
        if len(tids) < 2:
            continue
        for a, b in itertools.combinations(tids, 2):
            overlap = tid_frames[a] & tid_frames[b]
            if not overlap:
                continue
            if len(tid_frames[a]) >= len(tid_frames[b]):
                canon, dup = a, b
            else:
                canon, dup = b, a

            if dup in dup_to_canon:
                existing = dup_to_canon[dup]
                if len(tid_frames[canon]) > len(tid_frames[existing]):
                    dup_to_canon[dup] = canon
            else:
                dup_to_canon[dup] = canon
            pair_overlaps[(canon, dup)] = len(overlap)

    # Apply rule (c): >=50% of absorbed tracklet concurrent with canonical
    filtered = {}
    excluded = {}
    for dup, canon in dup_to_canon.items():
        overlap = tid_frames[canon] & tid_frames[dup]
        ratio = len(overlap) / len(tid_frames[dup])
        if ratio >= MIN_OVERLAP_FRACTION:
            filtered[dup] = canon
        else:
            excluded[dup] = {
                "canonical": canon,
                "total_frames": len(tid_frames[dup]),
                "overlap_frames": len(overlap),
                "ratio": ratio,
                "gt": tid_gt_dom[dup],
            }

    # Build merge groups for reporting
    merge_groups = defaultdict(list)
    for dup, canon in filtered.items():
        overlap = tid_frames[canon] & tid_frames[dup]
        merge_groups[canon].append({
            "tid": dup,
            "total_frames": len(tid_frames[dup]),
            "overlap_frames": len(overlap),
            "ratio": len(overlap) / len(tid_frames[dup]),
            "gt": tid_gt_dom[dup],
        })

    return {
        "dup_to_canon": filtered,
        "excluded": excluded,
        "merge_groups": merge_groups,
        "tid_gt_dom": dict(tid_gt_dom),
        "tid_frames": tid_frames,
    }


def apply_merges(scratch_clip: Path, merge_info: dict):
    """Apply tracklet merges to scratch Stage A artifacts.

    Returns diagnostics about dropped detections.
    """
    dup_to_canon = merge_info["dup_to_canon"]
    if not dup_to_canon:
        print("  No merges to apply.")
        return {"dropped_detections": 0, "gt_matched_dropped": 0}

    stage_a = scratch_clip / "stage_A"

    # --- detections.parquet ---
    det = pd.read_parquet(stage_a / "detections.parquet")
    original_det_count = len(det)

    # Remap tracklet_ids
    det["tracklet_id"] = det["tracklet_id"].map(
        lambda t: dup_to_canon.get(t, t)
    )

    # On overlap frames (same tracklet_id, same frame_index), keep higher confidence
    det = det.sort_values(
        ["tracklet_id", "frame_index", "confidence"],
        ascending=[True, True, False],
    )
    # Mark duplicates: keep first (highest confidence)
    dup_mask = det.duplicated(subset=["tracklet_id", "frame_index"], keep="first")
    dropped_det_ids = set(det[dup_mask]["detection_id"].values)
    det = det[~dup_mask].reset_index(drop=True)

    det.to_parquet(stage_a / "detections.parquet", index=False)
    print(f"  detections.parquet: {original_det_count} -> {len(det)} "
          f"({len(dropped_det_ids)} dropped)")

    # --- Check which dropped detections were GT-matched ---
    pfm = pd.read_parquet(PFM_PATH)
    gt_matched_dets = set(
        pfm[pfm["match_status"] == "matched"]["pred_detection_id"].values
    )
    gt_matched_dropped = dropped_det_ids & gt_matched_dets
    print(f"  Of {len(dropped_det_ids)} dropped: {len(gt_matched_dropped)} were GT-matched")

    # --- tracklet_frames.parquet ---
    tf = pd.read_parquet(stage_a / "tracklet_frames.parquet")
    original_tf_count = len(tf)
    tf["tracklet_id"] = tf["tracklet_id"].map(lambda t: dup_to_canon.get(t, t))

    # Drop duplicate (tracklet_id, frame_index) rows — keep by detection_id not in dropped set
    tf = tf[~tf["detection_id"].isin(dropped_det_ids)].reset_index(drop=True)
    tf.to_parquet(stage_a / "tracklet_frames.parquet", index=False)
    print(f"  tracklet_frames.parquet: {original_tf_count} -> {len(tf)}")

    # --- tracklet_summaries.parquet ---
    ts = pd.read_parquet(stage_a / "tracklet_summaries.parquet")
    # Drop absorbed tracklets
    absorbed_tids = set(dup_to_canon.keys())
    ts = ts[~ts["tracklet_id"].isin(absorbed_tids)].copy()

    # Recompute canonical summaries from merged tracklet_frames
    for canon_tid in set(dup_to_canon.values()):
        canon_tf = tf[tf["tracklet_id"] == canon_tid]
        if canon_tf.empty:
            continue
        idx = ts.index[ts["tracklet_id"] == canon_tid]
        if len(idx) == 0:
            continue
        i = idx[0]
        ts.loc[i, "first_frame"] = int(canon_tf["frame_index"].min())
        ts.loc[i, "last_frame"] = int(canon_tf["frame_index"].max())
        ts.loc[i, "n_frames"] = len(canon_tf)

    ts.to_parquet(stage_a / "tracklet_summaries.parquet", index=False)
    print(f"  tracklet_summaries.parquet: absorbed {len(absorbed_tids)} tracklets")

    # --- color_histograms.parquet ---
    ch_path = stage_a / "color_histograms.parquet"
    if ch_path.exists():
        ch = pd.read_parquet(ch_path)
        # Column is 'track_id', not 'tracklet_id'
        if "track_id" in ch.columns:
            ch["track_id"] = ch["track_id"].map(lambda t: dup_to_canon.get(str(t), str(t)))
        # Drop rows on overlap frames for absorbed tracklets (by frame_index + track_id dup)
        ch = ch.sort_values(["track_id", "frame_index"]).reset_index(drop=True)
        ch = ch.drop_duplicates(subset=["track_id", "frame_index"], keep="first")
        ch.to_parquet(ch_path, index=False)
        print(f"  color_histograms.parquet: remapped ({len(ch)} rows)")

    # --- tracklet_histogram_summaries.parquet ---
    ths_path = stage_a / "tracklet_histogram_summaries.parquet"
    if ths_path.exists():
        ths = pd.read_parquet(ths_path)
        # Drop absorbed, keep canonical (slightly wrong avg but acceptable for ceiling)
        ths = ths[~ths["tracklet_id"].isin(absorbed_tids)].reset_index(drop=True)
        ths.to_parquet(ths_path, index=False)
        print(f"  tracklet_histogram_summaries.parquet: dropped absorbed")

    # --- identity_hints.jsonl --- clear it
    ih_path = scratch_clip / "stage_C" / "identity_hints.jsonl"
    ih_path.parent.mkdir(parents=True, exist_ok=True)
    ih_path.write_text("")
    # Also clear tag_observations if present
    to_path = scratch_clip / "stage_C" / "tag_observations.jsonl"
    if to_path.exists():
        to_path.write_text("")
    print("  identity_hints.jsonl: cleared (Tier 1 evidence removed)")

    return {
        "dropped_detections": len(dropped_det_ids),
        "dropped_det_ids": dropped_det_ids,
        "gt_matched_dropped": len(gt_matched_dropped),
        "gt_matched_dropped_ids": gt_matched_dropped,
    }


def run_stage_d(scratch_clip: Path):
    """Run D0->D4 on scratch artifacts."""
    from bjj_pipeline.contracts.f0_manifest import ClipManifest, write_manifest
    from bjj_pipeline.contracts.f0_paths import ClipOutputLayout

    # Load production manifest for metadata
    prod_manifest_path = PROD_CLIP_DIR / "clip_manifest.json"
    prod_manifest = ClipManifest.model_validate_json(
        prod_manifest_path.read_text(encoding="utf-8")
    )

    manifest = ClipManifest(
        clip_id=CLIP_ID,
        camera_id=CAMERA_ID,
        gym_id="_dedup_ceiling",
        input_video_path=prod_manifest.input_video_path,
        fps=prod_manifest.fps,
        frame_count=prod_manifest.frame_count,
        duration_ms=prod_manifest.duration_ms,
        pipeline_version="dedup_ceiling",
        created_at_ms=int(time.time() * 1000),
    )

    manifest_path = scratch_clip / "clip_manifest.json"
    write_manifest(manifest, manifest_path)

    layout = ClipOutputLayout(clip_id=CLIP_ID, root=SCRATCH_ROOT)

    config_path = REPO_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault("stages", {}).setdefault("stage_D", {})["run_until"] = "D4"

    print("\n[dedup-ceiling] Running D0->D4...")
    t0 = time.monotonic()

    from bjj_pipeline.stages.stitch.run import run as stage_d_run

    stage_d_run(config=config, inputs={"layout": layout, "manifest": manifest})

    wall = time.monotonic() - t0
    print(f"  D0->D4 completed in {wall:.1f}s")

    return layout, manifest, config


def run_stage_e(layout, manifest, config):
    """Run Stage E on scratch artifacts."""
    print("\n[dedup-ceiling] Running Stage E...")
    t0 = time.monotonic()

    from bjj_pipeline.stages.matches.run import run as stage_e_run

    stage_e_run(config=config, inputs={"layout": layout, "manifest": manifest})

    wall = time.monotonic() - t0
    print(f"  Stage E completed in {wall:.1f}s")


def run_gt_matching(scratch_clip: Path):
    """Re-run GT matching against scratch detections.parquet.

    Uses the frozen CP-EVAL-1 instrument: Hungarian IoU >= 0.5.
    """
    print("\n[dedup-ceiling] Re-running GT matching...")

    from pipeline_validation.common.gt_loader import load_gt_for_split
    from pipeline_validation.common.manifest import load_manifest
    from pipeline_validation.stage_a.evaluate import (
        PredBox,
        _match_all_frames,
        _build_match_records,
        load_preds_from_parquet,
    )

    manifest_path = REPO_ROOT / "configs" / "models" / "gt-eval-fp7oJQ-132650.yaml"
    manifest = load_manifest(manifest_path)
    export = manifest.training_data[0]
    zip_path = REPO_ROOT / "data" / "training_data" / export.export

    # Load GT for val split
    gt = load_gt_for_split(zip_path, export, "val")
    print(f"  GT: {len(gt)} frames, {sum(len(v) for v in gt.values())} boxes")

    # Load scratch detections as PredBox dicts
    preds = load_preds_from_parquet(
        scratch_clip / "stage_A" / "detections.parquet",
        set(gt.keys()),
    )

    # Run Hungarian matching at IoU 0.5 (frozen instrument)
    frame_results = _match_all_frames(gt, preds)
    records = _build_match_records(
        frame_results, "gt-eval-fp7oJQ-132650", CAMERA_ID, "val",
    )

    pfm = pd.DataFrame(records)

    # Save
    out_dir = SCRATCH_EVIDENCE / "gt_matching"
    out_dir.mkdir(parents=True, exist_ok=True)
    pfm.to_parquet(out_dir / "per_frame_matches.parquet", index=False)
    print(f"  Saved {len(pfm)} rows to {out_dir / 'per_frame_matches.parquet'}")

    # Report: how many matched detections changed vs production?
    prod_pfm = pd.read_parquet(PFM_PATH)
    prod_matched = set(
        prod_pfm[prod_pfm["match_status"] == "matched"]["pred_detection_id"].dropna()
    )
    new_matched = set(
        pfm[pfm["match_status"] == "matched"]["pred_detection_id"].dropna()
    )
    lost = prod_matched - new_matched
    gained = new_matched - prod_matched
    print(f"  vs production: {len(lost)} matches lost, {len(gained)} gained, "
          f"{len(prod_matched & new_matched)} unchanged")

    return pfm


def compute_correct_id(pfm: pd.DataFrame, person_tracks: pd.DataFrame, det: pd.DataFrame):
    """Compute strict correct_id from per_frame_matches + person_tracks.

    Returns (strict_correct, total_gt_frames, per_gt_breakdown).
    """
    # Join pfm -> detections -> person_tracks to get person_id for each GT match
    pfm_matched = pfm[pfm["match_status"] == "matched"].copy()

    # Get tracklet_id from detections
    det_lookup = det[["frame_index", "detection_id", "tracklet_id"]].copy()
    pfm_with_tid = pfm_matched.merge(
        det_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
    )

    # Get person_id from person_tracks
    pt_lookup = person_tracks[["frame_index", "detection_id", "person_id"]].copy()
    pfm_with_pid = pfm_with_tid.merge(
        pt_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
        suffixes=("", "_pt"),
    )

    # Build canonical mapping: for each GT track, which person_id is dominant?
    gt_pid_counts = (
        pfm_with_pid.dropna(subset=["person_id"])
        .groupby(["gt_track_id", "person_id"])
        .size()
        .reset_index(name="count")
    )

    canonical = {}
    for gt_id in pfm["gt_track_id"].unique():
        sub = gt_pid_counts[gt_pid_counts["gt_track_id"] == gt_id]
        if sub.empty:
            continue
        best = sub.loc[sub["count"].idxmax()]
        canonical[int(gt_id)] = best["person_id"]

    # Score each GT frame
    total = 0
    correct = 0
    per_gt = defaultdict(lambda: {"total": 0, "correct": 0, "person_ids": set()})

    for _, row in pfm.iterrows():
        if pd.isna(row["gt_track_id"]):
            continue
        gt_id = int(row["gt_track_id"])
        total += 1
        per_gt[gt_id]["total"] += 1

        if row["match_status"] != "matched":
            continue

        # Find person_id for this detection
        sub = pfm_with_pid[
            (pfm_with_pid["frame_index"] == row["frame_index"])
            & (pfm_with_pid["gt_track_id"] == row["gt_track_id"])
        ]
        if sub.empty or pd.isna(sub.iloc[0].get("person_id")):
            continue

        pid = sub.iloc[0]["person_id"]
        per_gt[gt_id]["person_ids"].add(pid)

        if gt_id in canonical and pid == canonical[gt_id]:
            correct += 1
            per_gt[gt_id]["correct"] += 1

    return correct, total, dict(per_gt), canonical


def compute_partner_tolerant(
    pfm, person_tracks, det, canonical, config, total_frames=1764
):
    """Compute partner-tolerant correct_id.

    Uses pre-computed GT engagements from GT-VERIFY-2 (derived from ALL GT boxes,
    not just detected ones). This avoids losing co-detection frames due to dedup.
    """
    # Load GT engagements from GT-VERIFY-2
    gt_verify_path = REPO_ROOT / "docs" / "evidence" / "gt_diag_1" / "gt_verify_2_analysis.json"
    with open(gt_verify_path) as f:
        gt_verify = json.load(f)

    # Build per-frame engagement lookup from precomputed intervals
    engaged_at_frame = {}
    for e in gt_verify["gt_engagements"]:
        a_id = int(e["person_id_a"].replace("gt", ""))
        b_id = int(e["person_id_b"].replace("gt", ""))
        pair = frozenset({a_id, b_id})
        for f_idx in range(e["start_frame"], e["end_frame"] + 1):
            engaged_at_frame.setdefault(f_idx, set()).add(pair)

    # Build person_id -> dominant GT
    pfm_matched = pfm[pfm["match_status"] == "matched"].copy()
    det_lookup = det[["frame_index", "detection_id", "tracklet_id"]].copy()
    pt_lookup = person_tracks[["frame_index", "detection_id", "person_id"]].copy()

    pfm_with_tid = pfm_matched.merge(
        det_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
    )
    pfm_with_pid = pfm_with_tid.merge(
        pt_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
        suffixes=("", "_pt"),
    )

    pid_gt = {}
    for pid in pfm_with_pid["person_id"].dropna().unique():
        sub = pfm_with_pid[pfm_with_pid["person_id"] == pid]
        gt_counts = sub["gt_track_id"].value_counts()
        pid_gt[pid] = int(gt_counts.index[0])

    # Score
    tolerant_correct = 0
    for _, row in pfm.iterrows():
        if pd.isna(row["gt_track_id"]):
            continue
        gt_id = int(row["gt_track_id"])
        fi = int(row["frame_index"])

        if row["match_status"] != "matched":
            continue

        sub = pfm_with_pid[
            (pfm_with_pid["frame_index"] == fi)
            & (pfm_with_pid["gt_track_id"] == gt_id)
        ]
        if sub.empty or pd.isna(sub.iloc[0].get("person_id")):
            continue

        pid = sub.iloc[0]["person_id"]
        if gt_id in canonical and pid == canonical[gt_id]:
            tolerant_correct += 1
        elif pid in pid_gt:
            assigned_gt = pid_gt[pid]
            pair = frozenset({gt_id, assigned_gt})
            if pair in engaged_at_frame.get(fi, set()):
                tolerant_correct += 1

    return tolerant_correct


def evaluate_sessions(scratch_clip: Path, pfm: pd.DataFrame, config: dict):
    """Evaluate Stage E sessions with three-bucket classification."""
    sessions_path = scratch_clip / "stage_E" / "match_sessions.jsonl"
    if not sessions_path.exists():
        print("  No match_sessions.jsonl found")
        return {"total": 0, "correct": 0, "contaminated": 0, "phantom": 0, "sessions": []}

    sessions = []
    with open(sessions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    if not sessions:
        return {"total": 0, "correct": 0, "contaminated": 0, "phantom": 0, "sessions": []}

    # Load GT engagements from GT-VERIFY-2
    gt_verify_path = REPO_ROOT / "docs" / "evidence" / "gt_diag_1" / "gt_verify_2_analysis.json"
    with open(gt_verify_path) as f:
        gt_verify = json.load(f)

    gt_engaged = set()
    for e in gt_verify["gt_engagements"]:
        a_id = int(e["person_id_a"].replace("gt", ""))
        b_id = int(e["person_id_b"].replace("gt", ""))
        gt_engaged.add(frozenset({a_id, b_id}))

    # Load person_tracks and detections for GT composition
    pfm_matched = pfm[pfm["match_status"] == "matched"].copy()
    pt = pd.read_parquet(scratch_clip / "stage_D" / "person_tracks.parquet")
    det = pd.read_parquet(scratch_clip / "stage_A" / "detections.parquet")

    det_lookup = det[["frame_index", "detection_id", "tracklet_id"]].copy()
    pfm_with_tid = pfm_matched.merge(
        det_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
    )
    pt_lookup = pt[["frame_index", "detection_id", "person_id"]].copy()
    pfm_with_pid = pfm_with_tid.merge(
        pt_lookup,
        left_on=["frame_index", "pred_detection_id"],
        right_on=["frame_index", "detection_id"],
        how="left",
        suffixes=("", "_pt"),
    )

    # Classify each session
    results = []
    counts = {"CORRECT_ENGAGED": 0, "CONTAMINATED": 0, "PHANTOM": 0}

    for sess in sessions:
        sf = sess.get("start_frame", sess.get("start_frame_index", 0))
        ef = sess.get("end_frame", sess.get("end_frame_index", 0))
        pid_a = sess.get("person_id_a", sess.get("person_a"))
        pid_b = sess.get("person_id_b", sess.get("person_b"))

        # GT composition of each person_id in this session's frame range
        in_range = pfm_with_pid[
            (pfm_with_pid["frame_index"] >= sf)
            & (pfm_with_pid["frame_index"] <= ef)
        ]

        comp_a = in_range[in_range["person_id"] == pid_a]["gt_track_id"].dropna().value_counts()
        comp_b = in_range[in_range["person_id"] == pid_b]["gt_track_id"].dropna().value_counts()

        if comp_a.empty or comp_b.empty:
            category = "PHANTOM"
        else:
            dom_a = int(comp_a.index[0])
            dom_b = int(comp_b.index[0])
            dominant_pair = frozenset({dom_a, dom_b})

            # Check all GT pairs present
            all_gts_a = set(int(x) for x in comp_a.index)
            all_gts_b = set(int(x) for x in comp_b.index)
            gt_pairs = {
                frozenset({ga, gb})
                for ga in all_gts_a
                for gb in all_gts_b
                if ga != gb
            }

            if dominant_pair in gt_engaged:
                category = "CORRECT_ENGAGED"
            elif gt_pairs & gt_engaged:
                category = "CONTAMINATED"
            else:
                category = "PHANTOM"

        counts[category] = counts.get(category, 0) + 1
        results.append({
            "person_id_a": pid_a,
            "person_id_b": pid_b,
            "start_frame": sf,
            "end_frame": ef,
            "category": category,
        })

    return {
        "total": len(sessions),
        "correct": counts.get("CORRECT_ENGAGED", 0),
        "contaminated": counts.get("CONTAMINATED", 0),
        "phantom": counts.get("PHANTOM", 0),
        "sessions": results,
    }


def main():
    print("=" * 70)
    print("DEDUP-CEILING-1: Perfect deduplication ceiling analysis")
    print(f"Clip: {CLIP_ID}")
    print(f"Rule: (c) >= {MIN_OVERLAP_FRACTION:.0%} overlap fraction")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Build merge set
    # ---------------------------------------------------------------
    print("\n--- Step 1: Build merge set ---")
    det = pd.read_parquet(PROD_CLIP_DIR / "stage_A" / "detections.parquet")
    pfm = pd.read_parquet(PFM_PATH)

    merge_info = build_merge_set(det, pfm)
    dup_to_canon = merge_info["dup_to_canon"]

    print(f"  Tracklets to merge: {len(dup_to_canon)}")
    print(f"  Merge groups: {len(merge_info['merge_groups'])}")

    if merge_info["excluded"]:
        print(f"  Excluded by rule (c): {len(merge_info['excluded'])}")
        for tid, info in merge_info["excluded"].items():
            print(f"    {tid} ({info['total_frames']}f): {info['overlap_frames']}f overlap "
                  f"({info['ratio']:.1%}) with {info['canonical']} — GT {info['gt']}")

    # ---------------------------------------------------------------
    # Step 2: Prepare scratch directory
    # ---------------------------------------------------------------
    print("\n--- Step 2: Prepare scratch directory ---")
    if SCRATCH_CLIP.exists():
        shutil.rmtree(SCRATCH_CLIP)
    SCRATCH_CLIP.mkdir(parents=True)

    stage_a_scratch = SCRATCH_CLIP / "stage_A"
    stage_a_scratch.mkdir()

    # Copy Stage A artifacts
    stage_a_prod = PROD_CLIP_DIR / "stage_A"
    for f in [
        "detections.parquet",
        "tracklet_frames.parquet",
        "tracklet_summaries.parquet",
        "color_histograms.parquet",
        "tracklet_histogram_summaries.parquet",
        "contact_points.parquet",
    ]:
        src = stage_a_prod / f
        if src.exists():
            shutil.copy2(src, stage_a_scratch / f)

    # Copy sidecar-related files needed by D0
    # Copy audit.jsonl if exists
    audit_src = stage_a_prod / "audit.jsonl"
    if audit_src.exists():
        shutil.copy2(audit_src, stage_a_scratch / "audit.jsonl")

    # Copy identity_hints (will be cleared by apply_merges)
    stage_c_prod = PROD_CLIP_DIR / "stage_C"
    if stage_c_prod.exists():
        stage_c_scratch = SCRATCH_CLIP / "stage_C"
        stage_c_scratch.mkdir(parents=True, exist_ok=True)
        for f in stage_c_prod.iterdir():
            if f.is_file():
                shutil.copy2(f, stage_c_scratch / f.name)

    print(f"  Copied Stage A artifacts to {stage_a_scratch}")

    # ---------------------------------------------------------------
    # Step 3: Apply merges
    # ---------------------------------------------------------------
    print("\n--- Step 3: Apply merges ---")
    drop_info = apply_merges(SCRATCH_CLIP, merge_info)

    # ---------------------------------------------------------------
    # Step 4: Run D0->D4
    # ---------------------------------------------------------------
    print("\n--- Step 4: Run D0->D4 ---")
    layout, manifest, config = run_stage_d(SCRATCH_CLIP)

    # ---------------------------------------------------------------
    # Step 5: Run Stage E
    # ---------------------------------------------------------------
    print("\n--- Step 5: Run Stage E ---")
    run_stage_e(layout, manifest, config)

    # ---------------------------------------------------------------
    # Step 6: Re-run GT matching
    # ---------------------------------------------------------------
    print("\n--- Step 6: Re-run GT matching ---")
    scratch_pfm = run_gt_matching(SCRATCH_CLIP)

    # If GT matching failed, fall back to production pfm with caveats
    if scratch_pfm is None:
        print("  FALLBACK: using production per_frame_matches")
        scratch_pfm = pfm

    # ---------------------------------------------------------------
    # Step 7: Compute metrics
    # ---------------------------------------------------------------
    print("\n--- Step 7: Compute metrics ---")

    # Load scratch outputs
    scratch_pt = pd.read_parquet(SCRATCH_CLIP / "stage_D" / "person_tracks.parquet")
    scratch_det = pd.read_parquet(SCRATCH_CLIP / "stage_A" / "detections.parquet")

    # 1. Person count
    n_persons = scratch_pt["person_id"].nunique()
    person_ids = sorted(scratch_pt["person_id"].unique())
    print(f"\n  Person count: {n_persons} (baseline: 17, GT: 8)")

    # 2. correct_id strict
    strict_correct, total_gt, per_gt, canonical = compute_correct_id(
        scratch_pfm, scratch_pt, scratch_det
    )
    strict_pct = strict_correct / total_gt * 100 if total_gt > 0 else 0
    print(f"  correct_id strict: {strict_pct:.1f}% ({strict_correct}/{total_gt}) "
          f"(baseline: 34.3%)")

    # 3. correct_id partner-tolerant
    tolerant_correct = compute_partner_tolerant(
        scratch_pfm, scratch_pt, scratch_det, canonical, config
    )
    tolerant_pct = tolerant_correct / total_gt * 100 if total_gt > 0 else 0
    print(f"  correct_id partner-tolerant: {tolerant_pct:.1f}% ({tolerant_correct}/{total_gt}) "
          f"(baseline: 37.4%)")

    # 4. Stage E sessions
    sess_result = evaluate_sessions(SCRATCH_CLIP, scratch_pfm, config)
    print(f"\n  Stage E sessions: {sess_result['total']} (baseline: 23, GT: 3)")
    print(f"    CORRECT_ENGAGED: {sess_result['correct']} (baseline: 6)")
    print(f"    CONTAMINATED: {sess_result['contaminated']} (baseline: 13)")
    print(f"    PHANTOM: {sess_result['phantom']} (baseline: 4)")

    # 5. Per-GT-track residual
    print(f"\n  Per-GT-track person_id count (residual):")
    for gt_id in sorted(per_gt.keys()):
        info = per_gt[gt_id]
        n_pids = len(info["person_ids"])
        correct_pct = info["correct"] / info["total"] * 100 if info["total"] > 0 else 0
        canon_pid = canonical.get(gt_id, "none")
        print(f"    GT {gt_id}: {n_pids} person_ids, "
              f"{correct_pct:.1f}% correct ({info['correct']}/{info['total']}), "
              f"canonical={canon_pid}")

    # 6. Dropped detection analysis
    print(f"\n  Dropped detection analysis:")
    print(f"    Total dropped: {drop_info['dropped_detections']}")
    print(f"    GT-matched dropped: {drop_info['gt_matched_dropped']} "
          f"({drop_info['gt_matched_dropped']}/{drop_info['dropped_detections']})")

    # ---------------------------------------------------------------
    # Step 8: Write findings
    # ---------------------------------------------------------------
    print("\n--- Step 8: Write findings ---")
    SCRATCH_EVIDENCE.mkdir(parents=True, exist_ok=True)

    findings = {
        "clip_id": CLIP_ID,
        "camera_id": CAMERA_ID,
        "rule": f"(c) >= {MIN_OVERLAP_FRACTION:.0%} overlap fraction",
        "merge_set": {
            "tracklets_before": 66,
            "tracklets_removed": len(dup_to_canon),
            "tracklets_after": 66 - len(dup_to_canon),
            "merge_groups": len(merge_info["merge_groups"]),
            "excluded_by_rule_c": len(merge_info["excluded"]),
            "excluded_details": {
                tid: {
                    "canonical": info["canonical"],
                    "total_frames": info["total_frames"],
                    "overlap_frames": info["overlap_frames"],
                    "ratio": round(info["ratio"], 3),
                    "gt": info["gt"],
                }
                for tid, info in merge_info["excluded"].items()
            },
        },
        "dropped_detections": {
            "total": drop_info["dropped_detections"],
            "gt_matched": drop_info["gt_matched_dropped"],
        },
        "metrics": {
            "person_count": {"dedup": n_persons, "baseline": 17, "gt": 8},
            "correct_id_strict": {
                "dedup": round(strict_pct, 1),
                "baseline": 34.3,
                "delta": round(strict_pct - 34.3, 1),
            },
            "correct_id_tolerant": {
                "dedup": round(tolerant_pct, 1),
                "baseline": 37.4,
                "delta": round(tolerant_pct - 37.4, 1),
            },
            "sessions": {
                "dedup_total": sess_result["total"],
                "dedup_correct": sess_result["correct"],
                "dedup_contaminated": sess_result["contaminated"],
                "dedup_phantom": sess_result["phantom"],
                "baseline_total": 23,
                "baseline_correct": 6,
                "baseline_contaminated": 13,
                "baseline_phantom": 4,
                "gt_target": 3,
            },
        },
        "per_gt_residual": {
            str(gt_id): {
                "n_person_ids": len(info["person_ids"]),
                "person_ids": sorted(info["person_ids"]),
                "correct_pct": round(
                    info["correct"] / info["total"] * 100 if info["total"] > 0 else 0, 1
                ),
                "correct_frames": info["correct"],
                "total_frames": info["total"],
                "canonical_pid": canonical.get(gt_id, None),
            }
            for gt_id, info in sorted(per_gt.items())
        },
        "identity_hints_cleared": True,
        "ceiling_caveat": (
            "GT-labelled ceiling — strictly better than any real deduplicator. "
            "DEDUP-MEASURE-1 found physically-motivated features do not separate "
            "(sep_m 0.616, motion_corr 0.704, containment 0.537, IoU 0.504). "
            "One clip; no base-rate claim."
        ),
    }

    # Custom serializer for numpy/pandas types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(SCRATCH_EVIDENCE / "findings.json", "w") as f:
        json.dump(findings, f, indent=2, cls=NumpyEncoder)
    print(f"  Written findings.json to {SCRATCH_EVIDENCE}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEDUP-CEILING-1 SUMMARY")
    print("=" * 70)
    print(f"\nMerge: 66 -> {66 - len(dup_to_canon)} tracklets "
          f"({len(dup_to_canon)} removed, {len(merge_info['excluded'])} excluded by rule (c))")
    print(f"Detections dropped: {drop_info['dropped_detections']} "
          f"({drop_info['gt_matched_dropped']} GT-matched)")
    print()
    print(f"{'Metric':<30} {'Baseline':>10} {'Dedup':>10} {'Delta':>10} {'GT':>10}")
    print("-" * 70)
    print(f"{'Person count':<30} {'17':>10} {n_persons:>10} {n_persons-17:>+10} {'8':>10}")
    print(f"{'correct_id strict':<30} {'34.3%':>10} {strict_pct:>9.1f}% {strict_pct-34.3:>+9.1f}% {'-':>10}")
    print(f"{'correct_id tolerant':<30} {'37.4%':>10} {tolerant_pct:>9.1f}% {tolerant_pct-37.4:>+9.1f}% {'-':>10}")
    print(f"{'Sessions total':<30} {'23':>10} {sess_result['total']:>10} {sess_result['total']-23:>+10} {'3':>10}")
    print(f"{'  CORRECT_ENGAGED':<30} {'6':>10} {sess_result['correct']:>10} {sess_result['correct']-6:>+10} {'-':>10}")
    print(f"{'  CONTAMINATED':<30} {'13':>10} {sess_result['contaminated']:>10} {sess_result['contaminated']-13:>+10} {'-':>10}")
    print(f"{'  PHANTOM':<30} {'4':>10} {sess_result['phantom']:>10} {sess_result['phantom']-4:>+10} {'-':>10}")


if __name__ == "__main__":
    main()
