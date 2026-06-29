#!/usr/bin/env python3
"""Replay cached detections through BotSort with custom params.

Loads detection cache, reads source video frames, and replays through
BotSortTracker to produce tracklet assignments. Does NOT run Stage D.

Usage:
    python tools/sweep/replay_tracker.py --clip-id J_EDEw-20260318-200015 \
        --run-id baseline_vid1 --params '{}'
"""

import argparse
import json
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

# Source video path resolution for the two known J_EDEw clips
VIDEO_BASE = REPO_ROOT / "data" / "raw" / "nest" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"
CACHE_BASE = REPO_ROOT / "outputs" / "_sweep" / "detection_cache"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_detections(clip_id: str) -> pd.DataFrame:
    pq_path = CACHE_BASE / clip_id / "detections.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(
            f"Detection cache not found: {pq_path}\n"
            f"Run cache_detections.py first."
        )
    return pd.read_parquet(pq_path)


def replay(clip_id: str, camera_id: str, params: dict, run_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    det_df = load_detections(clip_id)
    video_path = VIDEO_BASE / f"{clip_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    # Build merged params (user params + required boxmot args)
    merged_params = dict(params)
    merged_params.setdefault("device", "cpu")
    merged_params.setdefault("half", False)
    merged_params.setdefault("reid_weights", "")

    logger.info(f"Replaying {clip_id} with params={merged_params}, run_id={run_id}")

    tracker = BotSortTracker(with_reid=False, params=merged_params)

    # Group detections by frame_index
    grouped = {}
    for _, row in det_df.iterrows():
        fi = int(row["frame_index"])
        if fi not in grouped:
            grouped[fi] = []
        grouped[fi].append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps} fps")

    results = []
    t0 = time.monotonic()

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}, stopping")
            break

        rows = grouped.get(frame_idx)
        if not rows:
            # No detections this frame — call tracker with empty list
            # (matches production: tracker.update returns [] immediately)
            tracker.update(frame_index=frame_idx, detections=[], frame_bgr=frame_bgr)
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
                "frame_index": frame_idx,
                "tracklet_id": td.tracklet_id,
                "detection_id": td.detection_id,
                "x1": td.x1,
                "y1": td.y1,
                "x2": td.x2,
                "y2": td.y2,
            })

    cap.release()
    wall_time = time.monotonic() - t0

    # Write tracklets
    track_df = pd.DataFrame(results)
    track_path = output_dir / "tracklets.parquet"
    track_df.to_parquet(track_path, index=False)

    # Compute summary stats
    n_tracklets = track_df["tracklet_id"].nunique() if len(track_df) > 0 else 0
    if n_tracklets > 0:
        tlen = track_df.groupby("tracklet_id")["frame_index"].count()
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
        "merged_params": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                         for k, v in merged_params.items()},
        "git_sha": get_git_sha(),
        "wall_time_seconds": round(wall_time, 1),
        "video_fps": fps,
        "video_total_frames": total_frames,
        "n_detection_rows": len(det_df),
        "n_tracked_rows": len(track_df),
        "n_tracklets": n_tracklets,
        "mean_tracklet_length": round(mean_len, 1),
        "short_tracklet_ratio_lt30": round(short_30, 3),
        "short_tracklet_ratio_lt10": round(short_10, 3),
    }

    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Done: {n_tracklets} tracklets, mean_len={mean_len:.1f}, "
        f"short<30={short_30:.1%}, short<10={short_10:.1%}, "
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
