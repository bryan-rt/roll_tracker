#!/usr/bin/env python3
"""Replay cached detections through BotSort with custom params.

Produces remapped Stage A artifacts (tracklet_frames, tracklet_summaries,
detections, color_histograms) and remapped identity_hints, ready for
Stage D re-run.

Output layout:
    outputs/_sweep/runs/<run-id>/<clip_id>/
        stage_A/
            tracklet_frames.parquet
            tracklet_summaries.parquet
            detections.parquet
            color_histograms.parquet
        stage_C/
            identity_hints.jsonl
            tag_observations.jsonl  (symlink from baseline)
        run_metadata.json

Usage:
    python tools/sweep/replay_tracker.py --clip-id J_EDEw-20260318-200015 \
        --run-id baseline --params '{}'
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bjj_pipeline.stages.detect_track.tracker import BotSortTracker
from bjj_pipeline.stages.detect_track.types import Detection

VIDEO_BASE = REPO_ROOT / "data" / "raw" / "nest" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
CACHE_BASE = REPO_ROOT / "outputs" / "_sweep" / "detection_cache"
BASELINE_BASE = REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_tracker_replay(clip_id: str, camera_id: str, params: dict) -> pd.DataFrame:
    """Run BotSort on cached detections, return (detection_id -> tracklet_id) mapping."""
    det_df = pd.read_parquet(CACHE_BASE / clip_id / "detections.parquet")
    video_path = VIDEO_BASE / f"{clip_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    merged_params = dict(params)
    merged_params.setdefault("device", "cpu")
    merged_params.setdefault("half", False)
    merged_params.setdefault("reid_weights", "")

    tracker = BotSortTracker(with_reid=False, params=merged_params)

    # Group detections by frame_index
    grouped: dict[int, list] = {}
    for _, row in det_df.iterrows():
        fi = int(row["frame_index"])
        if fi not in grouped:
            grouped[fi] = []
        grouped[fi].append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}, stopping")
            break

        rows = grouped.get(frame_idx)
        if not rows:
            # Production behavior: tracker.update short-circuits on empty detections
            # (returns [] without calling BoT-SORT's internal update). We replicate this.
            continue

        dets = []
        for row in rows:
            dets.append(Detection(
                clip_id=clip_id,
                camera_id=camera_id,
                frame_index=frame_idx,
                timestamp_ms=int(row.get("timestamp_ms", 0)),
                detection_id=str(row["detection_id"]),
                class_name="person",
                confidence=float(row["confidence"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
            ))

        tracked = tracker.update(
            frame_index=frame_idx,
            detections=dets,
            frame_bgr=frame_bgr,
        )

        for td in tracked:
            results.append({
                "detection_id": td.detection_id,
                "tracklet_id": td.tracklet_id,
            })

    cap.release()

    return pd.DataFrame(results) if results else pd.DataFrame(columns=["detection_id", "tracklet_id"])


def _remap_tracklet_frames(
    replay_map: pd.DataFrame, clip_id: str
) -> pd.DataFrame:
    """Inner-join baseline tracklet_frames with replay map on detection_id."""
    baseline = pd.read_parquet(BASELINE_BASE / clip_id / "stage_A" / "tracklet_frames.parquet")
    baseline = baseline.drop(columns=["tracklet_id"])
    # Inner join: only detections that got a tracklet assignment
    joined = baseline.merge(replay_map, on="detection_id", how="inner")
    joined = joined.sort_values(
        ["tracklet_id", "frame_index", "detection_id"], kind="mergesort"
    ).reset_index(drop=True)
    return joined


def _build_tracklet_summaries(
    tf: pd.DataFrame, clip_id: str, camera_id: str
) -> pd.DataFrame:
    """Reaggregate tracklet summaries from remapped tracklet_frames."""
    if tf.empty:
        return pd.DataFrame(columns=[
            "clip_id", "camera_id", "tracklet_id",
            "start_frame", "end_frame", "n_frames",
            "mean_x1", "mean_y1", "mean_x2", "mean_y2",
            "quality_score", "reason_codes_json",
        ])

    g = tf.groupby("tracklet_id", sort=True)
    rows = []
    for tid, sub in g:
        rows.append({
            "clip_id": clip_id,
            "camera_id": camera_id,
            "tracklet_id": tid,
            "start_frame": int(sub["frame_index"].min()),
            "end_frame": int(sub["frame_index"].max()),
            "n_frames": int(len(sub)),
            "mean_x1": pd.NA,
            "mean_y1": pd.NA,
            "mean_x2": pd.NA,
            "mean_y2": pd.NA,
            "quality_score": pd.NA,
            "reason_codes_json": None,
        })
    return pd.DataFrame(rows)


def _remap_detections(
    replay_map: pd.DataFrame, clip_id: str
) -> pd.DataFrame:
    """Left-join baseline detections with replay map on detection_id."""
    baseline = pd.read_parquet(BASELINE_BASE / clip_id / "stage_A" / "detections.parquet")
    baseline = baseline.drop(columns=["tracklet_id"])
    # Left join: ALL detections preserved, untracked get tracklet_id=None
    joined = baseline.merge(replay_map, on="detection_id", how="left")
    return joined


def _remap_color_histograms(
    replay_map: pd.DataFrame, clip_id: str
) -> pd.DataFrame:
    """Remap track_id in color_histograms via detection_id join.

    Join path: histograms(frame_index, track_id) -> baseline detections
    (frame_index, tracklet_id) -> detection_id -> replay_map -> new tracklet_id.
    """
    hist = pd.read_parquet(BASELINE_BASE / clip_id / "stage_A" / "color_histograms.parquet")
    n_baseline = len(hist)

    # Get (frame_index, tracklet_id) -> detection_id from baseline detections
    det = pd.read_parquet(
        BASELINE_BASE / clip_id / "stage_A" / "detections.parquet",
        columns=["frame_index", "tracklet_id", "detection_id"],
    )

    # Join histograms to detections on (frame_index, track_id == tracklet_id)
    hist_with_det = hist.merge(
        det,
        left_on=["frame_index", "track_id"],
        right_on=["frame_index", "tracklet_id"],
        how="inner",
    )

    if len(hist_with_det) != n_baseline:
        raise RuntimeError(
            f"color_histograms join failed: {len(hist_with_det)} matched vs "
            f"{n_baseline} baseline rows ({n_baseline - len(hist_with_det)} orphans). "
            f"This indicates a (frame_index, track_id) mismatch between "
            f"color_histograms.parquet and detections.parquet."
        )

    # Now join to replay map on detection_id to get new tracklet_id
    hist_remapped = hist_with_det.merge(
        replay_map.rename(columns={"tracklet_id": "new_tracklet_id"}),
        on="detection_id",
        how="left",
    )

    # Replace track_id with new tracklet_id (None for untracked)
    hist_remapped["track_id"] = hist_remapped["new_tracklet_id"]

    # Drop helper columns, keep original schema
    original_cols = list(hist.columns)
    hist_remapped = hist_remapped[original_cols]

    if len(hist_remapped) != n_baseline:
        raise RuntimeError(
            f"color_histograms remap row count mismatch: "
            f"{len(hist_remapped)} vs {n_baseline} baseline"
        )

    return hist_remapped


def _remap_identity_hints(
    replay_map: pd.DataFrame, clip_id: str
) -> tuple[list[dict], bool]:
    """Remap identity_hints.jsonl tracklet_ids using replay map.

    Returns (hints_list, tag_hint_dropped).
    """
    hints_path = BASELINE_BASE / clip_id / "stage_C" / "identity_hints.jsonl"
    if not hints_path.exists():
        return [], False

    lines = hints_path.read_text().strip().splitlines()
    if not lines:
        return [], False

    det = pd.read_parquet(
        BASELINE_BASE / clip_id / "stage_A" / "detections.parquet",
        columns=["frame_index", "tracklet_id", "detection_id"],
    )

    replay_dict = dict(zip(replay_map["detection_id"], replay_map["tracklet_id"]))
    tag_hint_dropped = False
    remapped_hints = []

    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        old_tid = rec.get("tracklet_id")
        first_frame = rec.get("evidence", {}).get("first_seen_frame")

        if old_tid is None or first_frame is None:
            remapped_hints.append(rec)
            continue

        # Find the detection_id for this tracklet at the first_seen_frame
        match = det[
            (det["tracklet_id"] == old_tid) & (det["frame_index"] == first_frame)
        ]
        if match.empty:
            logger.warning(
                f"identity_hint: no detection found for tracklet={old_tid} "
                f"frame={first_frame}, dropping hint"
            )
            tag_hint_dropped = True
            continue

        det_id = str(match.iloc[0]["detection_id"])
        new_tid = replay_dict.get(det_id)

        if new_tid is None:
            logger.warning(
                f"identity_hint: detection {det_id} at frame {first_frame} "
                f"is untracked in new run, dropping hint"
            )
            tag_hint_dropped = True
            continue

        rec["tracklet_id"] = new_tid
        remapped_hints.append(rec)

    return remapped_hints, tag_hint_dropped


def replay(clip_id: str, camera_id: str, params: dict, run_id: str, output_dir: Path) -> dict:
    clip_dir = output_dir / clip_id
    stage_a_dir = clip_dir / "stage_A"
    stage_c_dir = clip_dir / "stage_C"
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    stage_c_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Replaying {clip_id} with params={params}, run_id={run_id}")
    t0 = time.monotonic()

    # Step 1: Run tracker replay
    replay_map = _run_tracker_replay(clip_id, camera_id, params)
    t_tracker = time.monotonic() - t0

    n_tracked = len(replay_map)
    det_df = pd.read_parquet(CACHE_BASE / clip_id / "detections.parquet", columns=["detection_id"])
    n_total_dets = len(det_df)
    n_untracked = n_total_dets - n_tracked
    logger.info(f"Tracker replay: {n_tracked}/{n_total_dets} tracked ({n_untracked} untracked)")

    # Step 2: Remap tracklet_frames (inner join)
    tf = _remap_tracklet_frames(replay_map, clip_id)
    tf.to_parquet(stage_a_dir / "tracklet_frames.parquet", index=False)

    # Step 3: Reaggregate tracklet_summaries
    ts = _build_tracklet_summaries(tf, clip_id, camera_id)
    ts.to_parquet(stage_a_dir / "tracklet_summaries.parquet", index=False)

    # Step 4: Remap detections (left join)
    det_remapped = _remap_detections(replay_map, clip_id)
    det_remapped.to_parquet(stage_a_dir / "detections.parquet", index=False)

    # Step 5: Remap color_histograms (with hard-fail assertion)
    hist_remapped = _remap_color_histograms(replay_map, clip_id)
    hist_remapped.to_parquet(stage_a_dir / "color_histograms.parquet", index=False)

    # Step 6: Remap identity_hints
    hints, tag_hint_dropped = _remap_identity_hints(replay_map, clip_id)
    hints_path = stage_c_dir / "identity_hints.jsonl"
    with open(hints_path, "w") as f:
        for h in hints:
            f.write(json.dumps(h, sort_keys=True) + "\n")

    # Step 7: Symlink tag_observations.jsonl from baseline
    tag_obs_src = BASELINE_BASE / clip_id / "stage_C" / "tag_observations.jsonl"
    tag_obs_dst = stage_c_dir / "tag_observations.jsonl"
    if tag_obs_src.exists() and not tag_obs_dst.exists():
        # Copy instead of symlink to avoid Path.resolve() issues
        import shutil
        shutil.copy2(tag_obs_src, tag_obs_dst)

    wall_time = time.monotonic() - t0

    # Compute diagnostic stats
    n_tracklets = replay_map["tracklet_id"].nunique() if len(replay_map) > 0 else 0
    if n_tracklets > 0:
        tlen = replay_map.groupby("tracklet_id")["detection_id"].count()
        mean_len = float(tlen.mean())
        short_30 = float((tlen < 30).sum() / len(tlen))
        short_10 = float((tlen < 10).sum() / len(tlen))
    else:
        mean_len = 0.0
        short_30 = 0.0
        short_10 = 0.0

    metadata = {
        "run_id": run_id,
        "clip_id": clip_id,
        "camera_id": camera_id,
        "params": params,
        "git_sha": get_git_sha(),
        "wall_time_seconds": round(wall_time, 1),
        "tracker_time_seconds": round(t_tracker, 1),
        "n_total_detections": n_total_dets,
        "n_tracked_detections": n_tracked,
        "n_untracked_detections": n_untracked,
        "n_tracklets": n_tracklets,
        "mean_tracklet_length": round(mean_len, 1),
        "short_tracklet_ratio_lt30": round(short_30, 3),
        "short_tracklet_ratio_lt10": round(short_10, 3),
        "n_identity_hints": len(hints),
        "tag_hint_dropped": tag_hint_dropped,
    }

    meta_path = clip_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Done: {n_tracklets} tracklets, mean_len={mean_len:.1f}, "
        f"short<30={short_30:.1%}, short<10={short_10:.1%}, "
        f"untracked={n_untracked}, tag_hint_dropped={tag_hint_dropped}, "
        f"wall={wall_time:.1f}s"
    )

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Replay detections through BotSort")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", default="J_EDEw")
    parser.add_argument("--params", default="{}", help="JSON string of tracker params")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    params = json.loads(args.params)
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "outputs" / "_sweep" / "runs" / args.run_id
    )

    replay(args.clip_id, args.camera_id, params, args.run_id, output_dir)


if __name__ == "__main__":
    main()
