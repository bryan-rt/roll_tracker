"""CP22d: CoreML parallelization benchmark.

Tests whether multi-instance threading improves CoreML inference throughput
on Apple Silicon. The Neural Engine may or may not benefit from pipelined
requests from concurrent model instances.

Configurations tested:
  - Baseline: single model, sequential predict
  - Workers-2: 2 model instances on 2 threads
  - Workers-4: 4 model instances on 4 threads

Batch predict is excluded — CoreML crashes with list-of-frames input
(IndexError in ultralytics stream_inference).

Usage:
    python tools/coreml_parallel_benchmark.py
    python tools/coreml_parallel_benchmark.py --frames 200
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

TEST_CLIP = "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014.mp4"
MODEL_PATH = "models/yolo26n-pose.mlpackage"
CONF_THRESHOLD = 0.25
DEFAULT_FRAMES = 200


def _read_frames(clip_path: str, n_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _benchmark_sequential(model_path: str, frames: List[np.ndarray]) -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warmup
    for f in frames[:5]:
        model.predict(source=f, verbose=False, conf=CONF_THRESHOLD)

    times = []
    total_dets = 0
    t_start = time.perf_counter()
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(source=f, verbose=False, conf=CONF_THRESHOLD)
        times.append(time.perf_counter() - t0)
        total_dets += len(r[0].boxes) if r and r[0].boxes is not None else 0
    total_wall = time.perf_counter() - t_start

    return {
        "config": "baseline",
        "n_workers": 1,
        "n_frames": len(frames),
        "total_wall_s": round(total_wall, 3),
        "fps": round(len(frames) / total_wall, 2),
        "mean_ms": round(np.mean(times) * 1000, 2),
        "median_ms": round(np.median(times) * 1000, 2),
        "p95_ms": round(np.percentile(times, 95) * 1000, 2),
        "total_detections": total_dets,
    }


def _worker_fn(
    model_path: str,
    frame_indices: List[int],
    frames: List[np.ndarray],
) -> List[Dict[str, Any]]:
    """Worker thread: load own model instance, process assigned frames."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warmup with first frame
    if frame_indices:
        model.predict(source=frames[frame_indices[0]], verbose=False, conf=CONF_THRESHOLD)

    results = []
    for idx in frame_indices:
        t0 = time.perf_counter()
        r = model.predict(source=frames[idx], verbose=False, conf=CONF_THRESHOLD)
        elapsed = time.perf_counter() - t0
        n_dets = len(r[0].boxes) if r and r[0].boxes is not None else 0
        results.append({"frame_idx": idx, "elapsed_ms": round(elapsed * 1000, 2), "n_dets": n_dets})
    return results


def _benchmark_threaded(
    model_path: str,
    frames: List[np.ndarray],
    n_workers: int,
) -> Dict[str, Any]:
    # Distribute frames round-robin across workers
    worker_assignments: List[List[int]] = [[] for _ in range(n_workers)]
    for i in range(len(frames)):
        worker_assignments[i % n_workers].append(i)

    t_start = time.perf_counter()
    all_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_worker_fn, model_path, assignment, frames)
            for assignment in worker_assignments
        ]
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    total_wall = time.perf_counter() - t_start

    # Sort by frame index to verify we got all frames
    all_results.sort(key=lambda r: r["frame_idx"])
    times = [r["elapsed_ms"] for r in all_results]
    total_dets = sum(r["n_dets"] for r in all_results)

    return {
        "config": f"workers-{n_workers}",
        "n_workers": n_workers,
        "n_frames": len(frames),
        "total_wall_s": round(total_wall, 3),
        "fps": round(len(frames) / total_wall, 2),
        "mean_ms": round(np.mean(times), 2),
        "median_ms": round(np.median(times), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "total_detections": total_dets,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CP22d: CoreML parallelization benchmark")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: CoreML model not found: {MODEL_PATH}")
        print("Run: python tools/coreml_benchmark.py first to export the .mlpackage")
        raise SystemExit(1)

    if not Path(TEST_CLIP).exists():
        print(f"ERROR: Test clip not found: {TEST_CLIP}")
        raise SystemExit(1)

    print(f"Reading {args.frames} frames from {TEST_CLIP}...")
    frames = _read_frames(TEST_CLIP, args.frames)
    print(f"  Loaded {len(frames)} frames into memory.\n")

    results: Dict[str, Any] = {
        "model": MODEL_PATH,
        "test_clip": TEST_CLIP,
        "n_frames": len(frames),
        "note": "Batch predict excluded — CoreML crashes with list-of-frames input in ultralytics.",
        "configs": [],
    }

    # Baseline
    print("=" * 60)
    print("Baseline: single model, sequential predict")
    print("=" * 60)
    baseline = _benchmark_sequential(MODEL_PATH, frames)
    results["configs"].append(baseline)
    print(f"  {baseline['fps']:.1f} fps, {baseline['mean_ms']:.1f}ms mean, {baseline['total_wall_s']:.1f}s total")

    # Workers-2
    print(f"\n{'=' * 60}")
    print("Workers-2: 2 model instances, 2 threads")
    print("=" * 60)
    w2 = _benchmark_threaded(MODEL_PATH, frames, 2)
    results["configs"].append(w2)
    speedup_2 = w2["fps"] / baseline["fps"] if baseline["fps"] > 0 else 0
    print(f"  {w2['fps']:.1f} fps, {w2['mean_ms']:.1f}ms mean, {w2['total_wall_s']:.1f}s total, speedup={speedup_2:.2f}x")

    # Workers-4
    print(f"\n{'=' * 60}")
    print("Workers-4: 4 model instances, 4 threads")
    print("=" * 60)
    w4 = _benchmark_threaded(MODEL_PATH, frames, 4)
    results["configs"].append(w4)
    speedup_4 = w4["fps"] / baseline["fps"] if baseline["fps"] > 0 else 0
    print(f"  {w4['fps']:.1f} fps, {w4['mean_ms']:.1f}ms mean, {w4['total_wall_s']:.1f}s total, speedup={speedup_4:.2f}x")

    # Summary
    results["summary"] = {
        "baseline_fps": baseline["fps"],
        "workers_2_speedup": round(speedup_2, 2),
        "workers_4_speedup": round(speedup_4, 2),
        "recommendation": (
            f"workers-2 ({speedup_2:.2f}x)" if speedup_2 > 1.2
            else "sequential (threading does not help)"
        ),
    }

    out_dir = Path("outputs/_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coreml_parallel_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline:   {baseline['fps']:.1f} fps")
    print(f"  Workers-2:  {w2['fps']:.1f} fps ({speedup_2:.2f}x)")
    print(f"  Workers-4:  {w4['fps']:.1f} fps ({speedup_4:.2f}x)")
    print(f"  Recommendation: {results['summary']['recommendation']}")


if __name__ == "__main__":
    main()
