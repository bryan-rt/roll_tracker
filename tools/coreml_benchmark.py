"""CP22: CoreML vs MPS inference benchmark for pose models.

Exports YOLO pose models to CoreML (.mlpackage) and benchmarks inference
throughput against MPS. Results written to outputs/_benchmarks/coreml_benchmark.json.

Usage:
    python tools/coreml_benchmark.py
    python tools/coreml_benchmark.py --frames 200
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

TEST_CLIP = "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014.mp4"
DEFAULT_FRAMES = 200
CONF_THRESHOLD = 0.25


def _read_frames(clip_path: str, n_frames: int) -> List[np.ndarray]:
    """Pre-read frames into memory so I/O doesn't affect timing."""
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _benchmark_mps(model_path: str, frames: List[np.ndarray]) -> Dict[str, Any]:
    """Benchmark MPS inference, return per-frame times."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warmup
    for f in frames[:5]:
        model.predict(source=f, verbose=False, conf=CONF_THRESHOLD, device="mps")

    times = []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(source=f, verbose=False, conf=CONF_THRESHOLD, device="mps")
        times.append(time.perf_counter() - t0)

    # Verify keypoint output shape
    r0 = r[0] if r else None
    kp_shape = None
    if r0 is not None and hasattr(r0, "keypoints") and r0.keypoints is not None:
        kp_shape = list(r0.keypoints.data.shape)

    return {
        "device": "mps",
        "n_frames": len(frames),
        "per_frame_ms": [round(t * 1000, 2) for t in times],
        "mean_ms": round(np.mean(times) * 1000, 2),
        "median_ms": round(np.median(times) * 1000, 2),
        "p95_ms": round(np.percentile(times, 95) * 1000, 2),
        "keypoint_shape": kp_shape,
    }


def _try_coreml_export(model_path: str) -> Dict[str, Any]:
    """Attempt CoreML export. Returns dict with success status and path or error."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    try:
        exported_path = model.export(format="coreml")
        return {"success": True, "path": str(exported_path), "error": None}
    except Exception as e:
        return {"success": False, "path": None, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def _benchmark_coreml(mlpackage_path: str, frames: List[np.ndarray]) -> Dict[str, Any]:
    """Benchmark CoreML inference via ultralytics."""
    from ultralytics import YOLO

    model = YOLO(mlpackage_path)
    # Warmup
    for f in frames[:5]:
        model.predict(source=f, verbose=False, conf=CONF_THRESHOLD)

    times = []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(source=f, verbose=False, conf=CONF_THRESHOLD)
        times.append(time.perf_counter() - t0)

    r0 = r[0] if r else None
    kp_shape = None
    if r0 is not None and hasattr(r0, "keypoints") and r0.keypoints is not None:
        kp_shape = list(r0.keypoints.data.shape)

    return {
        "device": "coreml",
        "n_frames": len(frames),
        "per_frame_ms": [round(t * 1000, 2) for t in times],
        "mean_ms": round(np.mean(times) * 1000, 2),
        "median_ms": round(np.median(times) * 1000, 2),
        "p95_ms": round(np.percentile(times, 95) * 1000, 2),
        "keypoint_shape": kp_shape,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CP22: CoreML vs MPS benchmark")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()

    if not Path(TEST_CLIP).exists():
        print(f"ERROR: Test clip not found: {TEST_CLIP}")
        raise SystemExit(1)

    print(f"Reading {args.frames} frames from {TEST_CLIP}...")
    frames = _read_frames(TEST_CLIP, args.frames)
    print(f"  Loaded {len(frames)} frames into memory.")

    results: Dict[str, Any] = {
        "test_clip": TEST_CLIP,
        "n_frames": len(frames),
        "models": {},
    }

    models_to_test = [
        ("yolov8n-pose", "models/yolov8n-pose.pt"),
    ]
    # Include yolo26n-pose if available
    if Path("models/yolo26n-pose.pt").exists():
        models_to_test.append(("yolo26n-pose", "models/yolo26n-pose.pt"))

    for model_name, model_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_path})")
        print(f"{'='*60}")

        model_result: Dict[str, Any] = {}

        # MPS benchmark
        print(f"  Benchmarking MPS ({len(frames)} frames)...")
        mps_result = _benchmark_mps(model_path, frames)
        model_result["mps"] = mps_result
        print(f"  MPS: {mps_result['mean_ms']:.1f}ms mean, {mps_result['median_ms']:.1f}ms median, kp_shape={mps_result['keypoint_shape']}")

        # CoreML export + benchmark
        print(f"  Attempting CoreML export...")
        export_result = _try_coreml_export(model_path)
        model_result["coreml_export"] = export_result

        if export_result["success"]:
            print(f"  Export succeeded: {export_result['path']}")
            print(f"  Benchmarking CoreML ({len(frames)} frames)...")
            coreml_result = _benchmark_coreml(export_result["path"], frames)
            model_result["coreml"] = coreml_result
            print(f"  CoreML: {coreml_result['mean_ms']:.1f}ms mean, {coreml_result['median_ms']:.1f}ms median, kp_shape={coreml_result['keypoint_shape']}")

            speedup = mps_result["mean_ms"] / coreml_result["mean_ms"] if coreml_result["mean_ms"] > 0 else 0
            model_result["speedup_vs_mps"] = round(speedup, 2)
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print(f"  CoreML export FAILED: {export_result['error']}")
            model_result["coreml"] = None
            model_result["speedup_vs_mps"] = None

        results["models"][model_name] = model_result

    # Write results
    out_dir = Path("outputs/_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coreml_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
