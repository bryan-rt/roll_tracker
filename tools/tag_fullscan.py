"""Full-frame AprilTag scan — theoretical ceiling experiment.

Iterates every frame of a video and runs decode_apriltags_in_roi with
ROI = full frame. Answers: how many frames have a physically visible tag
if we remove all pipeline restrictions (bbox gating, cadence)?

Usage:
    PYTHONPATH=src python tools/tag_fullscan.py \
      --video path/to/video1.mp4 \
      --video path/to/video2.mp4

Output:
    outputs/_experiments/tag_fullscan/
      {clip_id}/tag_observations.jsonl
      experiment_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from bjj_pipeline.stages.tags.apriltag_runner import decode_apriltags_in_roi

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "_experiments" / "tag_fullscan"


def derive_clip_id(video_path: Path) -> str:
    return video_path.stem


def derive_camera_id(clip_id: str) -> str:
    parts = clip_id.split("-")
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def scan_video(
    video_path: Path,
    output_dir: Path,
    tag_family: str = "36h11",
    max_frames: int | None = None,
) -> dict:
    """Scan every frame of a video for AprilTags using full-frame ROI."""
    clip_id = derive_clip_id(video_path)
    camera_id = derive_camera_id(clip_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {"error": f"Cannot open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_to_scan = total_video_frames
    if max_frames is not None:
        frames_to_scan = min(frames_to_scan, max_frames)

    print(f"  Video: {video_path.name}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Total frames: {total_video_frames}")
    print(f"  Scanning: {frames_to_scan} frames, tag_family={tag_family}")

    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    obs_path = clip_dir / "tag_observations.jsonl"

    full_roi = [0, 0, w, h]
    total_observations = 0
    frames_with_tags = 0
    tag_ids_seen: set[str] = set()
    all_observations: list[dict] = []

    t0 = time.time()

    with open(obs_path, "w") as f:
        fi = 0
        while fi < frames_to_scan:
            ret, frame = cap.read()
            if not ret:
                break

            result = decode_apriltags_in_roi(
                frame_bgr=frame,
                roi_xyxy=full_roi,
                tag_family=tag_family,
            )

            dets = result.get("detections", []) or []
            if dets:
                frames_with_tags += 1
                for t in dets:
                    tag_id_str = str(t.tag_id)
                    tag_ids_seen.add(tag_id_str)
                    timestamp_ms = int(fi * 1000.0 / fps)
                    rec = {
                        "schema_version": "0",
                        "artifact_type": "tag_observation",
                        "clip_id": clip_id,
                        "camera_id": camera_id,
                        "frame_index": fi,
                        "timestamp_ms": timestamp_ms,
                        "tag_id": tag_id_str,
                        "tag_family": tag_family,
                        "confidence": 1.0,
                        "roi_method": "full_frame",
                        "roi_xyxy": full_roi,
                        "tag_corners_px": [
                            [float(px), float(py)]
                            for (px, py) in (t.corners_px or [])
                        ],
                    }
                    f.write(json.dumps(rec) + "\n")
                    all_observations.append(rec)
                    total_observations += 1

            fi += 1
            if fi % 500 == 0:
                elapsed = time.time() - t0
                fps_actual = fi / elapsed if elapsed > 0 else 0
                print(f"    Frame {fi}/{frames_to_scan} "
                      f"({fi/frames_to_scan*100:.0f}%) "
                      f"| {fps_actual:.0f} fps "
                      f"| tags so far: {total_observations}")

    cap.release()
    elapsed = time.time() - t0

    detection_rate = frames_with_tags / fi if fi > 0 else 0.0
    print(f"  Done: {fi} frames in {elapsed:.1f}s "
          f"({fi/elapsed:.0f} fps)")
    print(f"  Tag observations: {total_observations} "
          f"in {frames_with_tags} frames "
          f"({detection_rate:.4%} detection rate)")
    print(f"  Tag IDs seen: {sorted(tag_ids_seen) or 'none'}")

    return {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "video_path": str(video_path),
        "resolution": [w, h],
        "total_frames_scanned": fi,
        "total_observations": total_observations,
        "frames_with_tags": frames_with_tags,
        "detection_rate": detection_rate,
        "tag_ids_seen": sorted(tag_ids_seen),
        "elapsed_seconds": round(elapsed, 1),
        "tag_family": tag_family,
        "observations": all_observations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full-frame AprilTag scan (theoretical ceiling experiment)"
    )
    parser.add_argument(
        "--video", type=Path, action="append", required=True,
        help="Video file(s) to scan (can specify multiple)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--tag-family", type=str, default="36h11",
        help="AprilTag family (default: 36h11)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to scan per video (for quick tests)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load original pipeline observations for comparison
    original_obs = {}
    pipeline_dirs = {
        "J_EDEw-20260318-200015": REPO_ROOT / "outputs" / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20" / "J_EDEw-20260318-200015",
        "J_EDEw-20260318-200246": REPO_ROOT / "outputs" / "c8a592a4-2bca-400a-80e1-fec0e5cbea77" / "J_EDEw" / "2026-03-18" / "20" / "J_EDEw-20260318-200246",
    }
    for clip_id, pdir in pipeline_dirs.items():
        tag_path = pdir / "stage_C" / "tag_observations.jsonl"
        if tag_path.exists():
            count = sum(1 for line in tag_path.read_text().splitlines() if line.strip())
            original_obs[clip_id] = count

    results = []
    for video_path in args.video:
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            sys.exit(1)
        print(f"\n=== Scanning: {video_path.name} ===")
        result = scan_video(video_path, args.output_dir, args.tag_family, args.max_frames)
        results.append(result)

    # Write experiment summary
    summary = {
        "experiment": "CP-TAG-2 full-frame tag scan",
        "tag_family": args.tag_family,
        "videos": [],
    }
    for r in results:
        clip_id = r["clip_id"]
        orig_count = original_obs.get(clip_id, 0)
        video_summary = {
            "clip_id": clip_id,
            "camera_id": r["camera_id"],
            "total_frames_scanned": r["total_frames_scanned"],
            "fullscan_observations": r["total_observations"],
            "fullscan_frames_with_tags": r["frames_with_tags"],
            "fullscan_detection_rate": r["detection_rate"],
            "fullscan_tag_ids": r["tag_ids_seen"],
            "original_pipeline_observations": orig_count,
            "multiplier": (r["total_observations"] / orig_count
                           if orig_count > 0 else None),
        }
        summary["videos"].append(video_summary)
        # Remove observations list from per-video result (already in jsonl)
        r.pop("observations", None)

    summary_path = args.output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Written to: {summary_path}")
    for vs in summary["videos"]:
        print(f"  {vs['clip_id']}: "
              f"{vs['fullscan_observations']} full-scan obs "
              f"(vs {vs['original_pipeline_observations']} pipeline) "
              f"in {vs['fullscan_frames_with_tags']} frames "
              f"({vs['fullscan_detection_rate']:.4%})")


if __name__ == "__main__":
    main()
