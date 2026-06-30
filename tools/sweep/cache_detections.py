#!/usr/bin/env python3
"""Cache Stage A detections for tracker param sweep.

Copies detections.parquet from the eval_gt pipeline outputs into
outputs/_sweep/detection_cache/<clip_id>/ with a content hash and
per-frame bbox count summary.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Known J_EDEw clips in the eval harness
EVAL_GT_BASE = REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20"

DEFAULT_CLIPS = [
    "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246",
]


def cache_clip(clip_id: str, camera_id: str) -> dict:
    source = EVAL_GT_BASE / clip_id / "stage_A" / "detections.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Missing: {source}")

    dest_dir = REPO_ROOT / "outputs" / "_sweep" / "detection_cache" / clip_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "detections.parquet"

    shutil.copy2(source, dest)

    content_hash = hashlib.sha256(dest.read_bytes()).hexdigest()

    df = pd.read_parquet(dest, columns=["frame_index"])
    counts = df["frame_index"].value_counts()

    # Probe clip FPS
    import cv2
    video_path = EVAL_GT_BASE / clip_id / ".." / ".." / ".." / ".." / ".." / ".." / "data" / "raw" / "nest" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20" / f"{clip_id}.mp4"
    # Use the known video path directly
    video_path = REPO_ROOT / "data" / "raw" / "nest" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20" / f"{clip_id}.mp4"
    clip_fps = 30.0  # default
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        clip_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    summary = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "source_path": str(source),
        "content_hash_sha256": content_hash,
        "n_frames": int(counts.shape[0]),
        "n_detections": int(len(df)),
        "clip_fps": clip_fps,
        "bbox_per_frame": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "p50": int(np.percentile(counts.values, 50)),
            "p95": int(np.percentile(counts.values, 95)),
            "mean": round(float(counts.mean()), 1),
        },
    }

    summary_path = dest_dir / "cache_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cache Stage A detections for sweep")
    parser.add_argument("--clip-id", nargs="*", default=None)
    parser.add_argument("--camera-id", default="J_EDEw")
    args = parser.parse_args()

    clips = args.clip_id or DEFAULT_CLIPS

    for clip_id in clips:
        print(f"Caching {clip_id}...")
        summary = cache_clip(clip_id, args.camera_id)
        print(f"  {summary['n_detections']:,} detections across {summary['n_frames']:,} frames")
        print(f"  Bboxes/frame: min={summary['bbox_per_frame']['min']}, "
              f"max={summary['bbox_per_frame']['max']}, "
              f"p50={summary['bbox_per_frame']['p50']}, "
              f"p95={summary['bbox_per_frame']['p95']}")
        print(f"  Hash: {summary['content_hash_sha256'][:16]}...")
        print(f"  Written: outputs/_sweep/detection_cache/{clip_id}/")


if __name__ == "__main__":
    main()
