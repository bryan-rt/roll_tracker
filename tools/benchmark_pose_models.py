"""CP20 A1: Benchmark YOLO pose models against detect baseline.

Runs multiple YOLO models on test clips (one per camera) and compares:
  - Inference speed (ms/frame)
  - Detection count parity vs baseline
  - Confidence distributions
  - Keypoint confidence (pose models only)
  - BoT-SORT tracklet outcomes

Usage:
    python tools/benchmark_pose_models.py
    python tools/benchmark_pose_models.py --max-frames 100
    python tools/benchmark_pose_models.py --models yolov8n-pose yolov8s-pose
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# ─── Configuration ───────────────────────────────────────────────────────

MODELS_TO_BENCHMARK = [
    "models/yolov8n.pt",       # current baseline (detect only)
    "models/yolov8n-pose.pt",  # v8 nano + pose
    "models/yolov8s-pose.pt",  # v8 small + pose
    "models/yolo11n-pose.pt",  # v11 nano + pose
    "models/yolo11s-pose.pt",  # v11 small + pose
]

TEST_CLIPS = {
    "FP7oJQ": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014.mp4",
    "J_EDEw": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/J_EDEw/2026-03-18/20/J_EDEw-20260318-200015.mp4",
    "PPDmUg": "data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/PPDmUg/2026-03-18/20/PPDmUg-20260318-200019.mp4",
}

COCO_TORSO_INDICES = [5, 6, 11, 12]  # L/R shoulder, L/R hip
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DEFAULT_MAX_FRAMES = 200


# ─── Data structures ────────────────────────────────────────────────────

@dataclass
class FrameResult:
    n_detections: int
    confidences: List[float]
    inference_ms: float
    # Pose-specific (empty for detect-only)
    torso_kp_confs: List[List[float]] = field(default_factory=list)  # per-det list of torso confs
    all_kp_confs: List[List[float]] = field(default_factory=list)


@dataclass
class ClipResult:
    camera_id: str
    model_name: str
    n_frames: int
    frame_results: List[FrameResult]
    tracklet_count: int
    tracker_error: Optional[str] = None

    @property
    def is_pose_model(self) -> bool:
        return "pose" in self.model_name

    def inference_ms_stats(self) -> Dict[str, float]:
        times = [r.inference_ms for r in self.frame_results]
        if not times:
            return {"mean": 0, "p50": 0, "p95": 0}
        arr = np.array(times)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    def detections_per_frame_mean(self) -> float:
        counts = [r.n_detections for r in self.frame_results]
        return float(np.mean(counts)) if counts else 0.0

    def confidence_stats(self) -> Dict[str, float]:
        all_confs = []
        for r in self.frame_results:
            all_confs.extend(r.confidences)
        if not all_confs:
            return {"mean": 0, "p50": 0}
        arr = np.array(all_confs)
        return {"mean": float(np.mean(arr)), "p50": float(np.median(arr))}

    def torso_kp_stats(self) -> Dict[str, Any]:
        """Torso keypoint confidence stats (pose models only)."""
        if not self.is_pose_model:
            return {"mean_conf": None, "pct_gte4_above_03": None}

        all_torso_confs = []
        n_frames_with_dets = 0
        n_frames_gte4 = 0

        for r in self.frame_results:
            if not r.torso_kp_confs:
                continue
            frame_has_good_det = False
            for det_torso in r.torso_kp_confs:
                all_torso_confs.extend(det_torso)
                n_above = sum(1 for c in det_torso if c > 0.3)
                if n_above >= 4:
                    frame_has_good_det = True
            n_frames_with_dets += 1
            if frame_has_good_det:
                n_frames_gte4 += 1

        mean_conf = float(np.mean(all_torso_confs)) if all_torso_confs else 0.0
        pct = (n_frames_gte4 / n_frames_with_dets * 100) if n_frames_with_dets > 0 else 0.0

        return {"mean_conf": mean_conf, "pct_gte4_above_03": pct}


# ─── Model runner ────────────────────────────────────────────────────────

def run_model_on_clip(
    model_path: str,
    clip_path: str,
    camera_id: str,
    max_frames: int,
    device: str,
) -> ClipResult:
    from ultralytics import YOLO

    model_name = Path(model_path).stem
    is_pose = "pose" in model_name

    console.print(f"  [dim]{model_name}[/dim] on [dim]{camera_id}[/dim]...", end=" ")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {clip_path}")

    frame_results: List[FrameResult] = []
    all_boxes_for_tracker: List[tuple] = []  # (frame_idx, frame_bgr, dets_array)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Timed inference
        t0 = time.perf_counter()
        results = model.predict(source=frame_bgr, verbose=False, conf=0.25, device=device)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        r0 = results[0] if results else None
        boxes = getattr(r0, "boxes", None) if r0 is not None else None

        n_dets = 0
        confidences: List[float] = []
        torso_kp_confs: List[List[float]] = []
        all_kp_confs: List[List[float]] = []
        dets_for_tracker = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy()

            # Filter to person (class 0)
            keep = clses.astype(int) == 0
            xyxy = xyxy[keep]
            confs = confs[keep]

            n_dets = len(xyxy)
            confidences = confs.tolist()

            # Tracker input: Nx6 (x1,y1,x2,y2,conf,cls)
            if n_dets > 0:
                cls_col = np.zeros((n_dets, 1))
                tracker_arr = np.hstack([xyxy, confs.reshape(-1, 1), cls_col])
                dets_for_tracker = tracker_arr

            # Extract keypoints (pose models)
            if is_pose:
                kps_obj = getattr(r0, "keypoints", None)
                if kps_obj is not None and hasattr(kps_obj, "data"):
                    kps_data = kps_obj.data.cpu().numpy()  # (N_all, 17, 3)
                    kps_data = kps_data[keep] if kps_data.shape[0] == keep.shape[0] else kps_data
                    for i in range(min(n_dets, kps_data.shape[0])):
                        kp = kps_data[i]  # (17, 3)
                        torso = [float(kp[j, 2]) for j in COCO_TORSO_INDICES]
                        torso_kp_confs.append(torso)
                        all_kp_confs.append([float(kp[j, 2]) for j in range(17)])

        all_boxes_for_tracker.append((frame_idx, frame_bgr, dets_for_tracker))
        frame_results.append(FrameResult(
            n_detections=n_dets,
            confidences=confidences,
            inference_ms=inference_ms,
            torso_kp_confs=torso_kp_confs,
            all_kp_confs=all_kp_confs,
        ))
        frame_idx += 1

    cap.release()

    # Run BoT-SORT tracking
    tracklet_count = 0
    tracker_error = None
    try:
        from boxmot import BotSort
        tracker = BotSort(
            reid_weights="",
            device=torch.device("cpu"),
            half=False,
            with_reid=False,
        )

        seen_ids = set()
        for fi, frame_bgr, dets in all_boxes_for_tracker:
            if isinstance(dets, np.ndarray) and len(dets) > 0:
                tracks = tracker.update(dets, frame_bgr)
                if len(tracks) > 0:
                    for t in tracks:
                        seen_ids.add(int(t[4]))
            else:
                tracker.update(np.empty((0, 6)), frame_bgr)

        tracklet_count = len(seen_ids)
    except Exception as e:
        tracker_error = str(e)

    result = ClipResult(
        camera_id=camera_id,
        model_name=model_name,
        n_frames=frame_idx,
        frame_results=frame_results,
        tracklet_count=tracklet_count,
        tracker_error=tracker_error,
    )

    ms = result.inference_ms_stats()
    console.print(f"[green]{frame_idx}f[/green] {ms['mean']:.1f}ms/f {result.detections_per_frame_mean():.1f}det/f")
    return result


# ─── Aggregate and display ───────────────────────────────────────────────

@dataclass
class ModelSummary:
    model_name: str
    model_size_mb: float
    clip_results: List[ClipResult]

    def avg_inference_ms(self) -> Dict[str, float]:
        all_means = [r.inference_ms_stats()["mean"] for r in self.clip_results]
        all_p50 = [r.inference_ms_stats()["p50"] for r in self.clip_results]
        all_p95 = [r.inference_ms_stats()["p95"] for r in self.clip_results]
        return {
            "mean": float(np.mean(all_means)),
            "p50": float(np.mean(all_p50)),
            "p95": float(np.mean(all_p95)),
        }

    def avg_dets_per_frame(self) -> float:
        return float(np.mean([r.detections_per_frame_mean() for r in self.clip_results]))

    def avg_confidence(self) -> Dict[str, float]:
        means = [r.confidence_stats()["mean"] for r in self.clip_results]
        p50s = [r.confidence_stats()["p50"] for r in self.clip_results]
        return {"mean": float(np.mean(means)), "p50": float(np.mean(p50s))}

    def avg_torso_kp(self) -> Dict[str, Any]:
        stats = [r.torso_kp_stats() for r in self.clip_results]
        confs = [s["mean_conf"] for s in stats if s["mean_conf"] is not None]
        pcts = [s["pct_gte4_above_03"] for s in stats if s["pct_gte4_above_03"] is not None]
        if not confs:
            return {"mean_conf": None, "pct_gte4_above_03": None}
        return {
            "mean_conf": float(np.mean(confs)),
            "pct_gte4_above_03": float(np.mean(pcts)),
        }

    def avg_tracklet_count(self) -> float:
        return float(np.mean([r.tracklet_count for r in self.clip_results]))

    def any_tracker_errors(self) -> bool:
        return any(r.tracker_error is not None for r in self.clip_results)

    def extrapolated_36clip_minutes(self, frames_per_clip: float) -> float:
        """Extrapolate total time for 36 clips based on avg ms/frame."""
        ms_per_frame = self.avg_inference_ms()["mean"]
        total_ms = ms_per_frame * frames_per_clip * 36
        return total_ms / 60_000


def display_results(summaries: List[ModelSummary], baseline_name: str, frames_per_clip: float) -> None:
    baseline = next((s for s in summaries if s.model_name == baseline_name), None)
    baseline_dets = baseline.avg_dets_per_frame() if baseline else 1.0
    baseline_tracks = baseline.avg_tracklet_count() if baseline else 1.0

    table = Table(title="CP20 A1: Pose Model Benchmark", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Size\n(MB)", justify="right")
    table.add_column("Inference ms/frame\nmean / p50 / p95", justify="right")
    table.add_column("Dets/frame\n(vs baseline)", justify="right")
    table.add_column("Det conf\nmean / p50", justify="right")
    table.add_column("Torso KP conf\nmean / %≥4@0.3", justify="right")
    table.add_column("Tracklets\n(vs baseline)", justify="right")
    table.add_column("36-clip\n(min)", justify="right")

    for s in summaries:
        ms = s.avg_inference_ms()
        dets = s.avg_dets_per_frame()
        conf = s.avg_confidence()
        torso = s.avg_torso_kp()
        tracks = s.avg_tracklet_count()
        ext = s.extrapolated_36clip_minutes(frames_per_clip)

        det_delta = ((dets - baseline_dets) / baseline_dets * 100) if baseline_dets > 0 else 0
        track_delta = ((tracks - baseline_tracks) / baseline_tracks * 100) if baseline_tracks > 0 else 0

        ms_str = f"{ms['mean']:.1f} / {ms['p50']:.1f} / {ms['p95']:.1f}"
        det_str = f"{dets:.1f} ({det_delta:+.0f}%)"
        conf_str = f"{conf['mean']:.3f} / {conf['p50']:.3f}"

        if torso["mean_conf"] is not None:
            torso_str = f"{torso['mean_conf']:.3f} / {torso['pct_gte4_above_03']:.0f}%"
        else:
            torso_str = "N/A"

        track_str = f"{tracks:.0f} ({track_delta:+.0f}%)"
        if s.any_tracker_errors():
            track_str += " [red]ERR[/red]"

        ext_str = f"{ext:.1f}"

        style = "bold green" if s.model_name == baseline_name else None
        table.add_row(s.model_name, f"{s.model_size_mb:.1f}", ms_str, det_str,
                      conf_str, torso_str, track_str, ext_str, style=style)

    console.print(table)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CP20 A1: Benchmark pose models")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Max frames per clip (default {DEFAULT_MAX_FRAMES})")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model stems to benchmark (e.g. yolov8n-pose yolo11n-pose)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    console.print(f"Device: [bold]{device}[/bold]")

    # Filter models
    models = MODELS_TO_BENCHMARK
    if args.models:
        stems = set(args.models)
        models = [m for m in models if Path(m).stem in stems]

    # Validate clips exist
    for cam_id, clip_path in TEST_CLIPS.items():
        if not Path(clip_path).exists():
            console.print(f"[red]Missing clip for {cam_id}: {clip_path}[/red]")
            raise SystemExit(1)

    # Validate models exist
    for m in models:
        if not Path(m).exists():
            console.print(f"[red]Missing model: {m}[/red]")
            raise SystemExit(1)

    console.print(f"Benchmarking {len(models)} models × {len(TEST_CLIPS)} clips × {args.max_frames} frames")
    console.print()

    summaries: List[ModelSummary] = []

    for model_path in models:
        model_name = Path(model_path).stem
        model_size_mb = Path(model_path).stat().st_size / 1e6
        console.print(f"[bold]{model_name}[/bold] ({model_size_mb:.1f} MB)")

        clip_results: List[ClipResult] = []
        for cam_id, clip_path in TEST_CLIPS.items():
            result = run_model_on_clip(
                model_path=model_path,
                clip_path=clip_path,
                camera_id=cam_id,
                max_frames=args.max_frames,
                device=device,
            )
            clip_results.append(result)

        summaries.append(ModelSummary(
            model_name=model_name,
            model_size_mb=model_size_mb,
            clip_results=clip_results,
        ))
        console.print()

    # Estimate frames per full clip (from first clip's total frame count)
    cap = cv2.VideoCapture(list(TEST_CLIPS.values())[0])
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    frames_per_clip = total_frames if total_frames > 0 else 4500  # ~2.5min @ 30fps

    display_results(summaries, baseline_name="yolov8n", frames_per_clip=frames_per_clip)

    # Save JSON
    out_dir = Path("outputs/_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pose_model_benchmark.json"

    json_data = {
        "device": device,
        "max_frames": args.max_frames,
        "frames_per_clip_estimate": frames_per_clip,
        "models": [],
    }
    for s in summaries:
        json_data["models"].append({
            "model_name": s.model_name,
            "model_size_mb": s.model_size_mb,
            "avg_inference_ms": s.avg_inference_ms(),
            "avg_dets_per_frame": s.avg_dets_per_frame(),
            "avg_confidence": s.avg_confidence(),
            "avg_torso_kp": s.avg_torso_kp(),
            "avg_tracklet_count": s.avg_tracklet_count(),
            "any_tracker_errors": s.any_tracker_errors(),
            "extrapolated_36clip_min": s.extrapolated_36clip_minutes(frames_per_clip),
            "per_clip": [
                {
                    "camera_id": r.camera_id,
                    "n_frames": r.n_frames,
                    "inference_ms": r.inference_ms_stats(),
                    "dets_per_frame": r.detections_per_frame_mean(),
                    "confidence": r.confidence_stats(),
                    "torso_kp": r.torso_kp_stats(),
                    "tracklet_count": r.tracklet_count,
                    "tracker_error": r.tracker_error,
                }
                for r in s.clip_results
            ],
        })

    out_path.write_text(json.dumps(json_data, indent=2))
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
