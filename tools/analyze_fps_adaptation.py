#!/usr/bin/env python3
"""CP-R1b: Analyze fps adaptation across recorder segments.

Walks all segments for FP7oJQ and PPDmUg, extracts sidecar metadata,
measures brightness and person count from sampled frames, and computes
per-frame dt across transition segments.

Usage:
  python tools/analyze_fps_adaptation.py \
    --recordings data/raw/nest/00000000-0000-0000-0000-000000000003 \
    --output-dir docs/evidence/recorder_fps_adaptation_1 \
    --model models/bjj-detect-all-cameras-v2.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Phase 2 segments requiring fine-grained dt + brightness analysis
PHASE2_SEGMENTS = {
    "FP7oJQ-20260804-163102",  # transition 15->30
    "FP7oJQ-20260804-162900",  # pre-transition control
    "FP7oJQ-20260804-163300",  # post-transition control
    "PPDmUg-20260804-163041",  # 17.6fps outlier
    "PPDmUg-20260804-163240",  # ramp segment 2
    "PPDmUg-20260804-163439",  # ramp segment 3
    "PPDmUg-20260804-163640",  # ramp segment 4
    "PPDmUg-20260804-162841",  # pre-ramp control
}


def parse_segment_filename(name: str) -> dict:
    """Parse CAMID-YYYYMMDD-HHMMSS from filename."""
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        cam = parts[0]
        date_str = parts[1]
        time_str = parts[2]
        return {"camera": cam, "date": date_str, "time": time_str}
    return {"camera": "", "date": "", "time": ""}


def classify_run(hour: str, filename: str) -> str:
    """Label each segment's run context."""
    if hour != "15":
        return f"smoke_{hour}"
    parsed = parse_segment_filename(filename)
    t = parsed["time"]
    # Aborted run: segments before 153236 (started ~19:08, died ~19:10)
    if t < "153236":
        return "aborted"
    return "capture"


def extract_sidecar_meta(sidecar_path: Path) -> dict:
    """Read _meta line from a timing sidecar."""
    with open(sidecar_path) as f:
        first_line = f.readline().strip()
    try:
        meta = json.loads(first_line)
        if meta.get("_meta"):
            return meta
    except json.JSONDecodeError:
        pass
    return {}


def extract_per_frame_dt(sidecar_path: Path) -> list[dict]:
    """Read per-frame pts_time_s from sidecar, compute dt."""
    rows = []
    pts_values = []
    with open(sidecar_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_meta"):
                continue
            fi = row.get("frame_index")
            pts = row.get("pts_time_s")
            if fi is not None and pts is not None:
                pts_values.append((fi, pts))

    pts_values.sort(key=lambda x: x[0])
    for i, (fi, pts) in enumerate(pts_values):
        dt_ms = None
        if i > 0:
            dt_ms = round((pts - pts_values[i - 1][1]) * 1000, 4)
        rows.append({"frame_index": fi, "pts_time_s": pts, "dt_ms": dt_ms})
    return rows


def get_ffprobe_info(mp4_path: Path) -> dict:
    """Get resolution and bitrate via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,bit_rate",
                "-of", "csv=p=0", str(mp4_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            return {
                "width": int(parts[0]),
                "height": int(parts[1]),
                "bitrate_bps": int(parts[2]) if parts[2] != "N/A" else None,
            }
    except Exception:
        pass
    return {"width": None, "height": None, "bitrate_bps": None}


def measure_brightness_and_detections(
    mp4_path: Path,
    stride: int,
    model=None,
) -> dict:
    """Sequential decode, sample every stride-th frame for brightness + detection."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {
            "brightness_mean": None, "brightness_std": None,
            "people_mean_025": None, "people_max_025": None,
            "people_mean_045": None, "people_max_045": None,
            "n_samples": 0, "per_frame": [],
        }

    brightness_vals = []
    counts_025 = []
    counts_045 = []
    per_frame = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bri = float(np.mean(gray))
            brightness_vals.append(bri)

            c025 = 0
            c045 = 0
            if model is not None:
                results = model(frame, verbose=False, conf=0.25)
                for r in results:
                    if r.boxes is not None:
                        confs = r.boxes.conf.cpu().numpy()
                        c025 = int(len(confs))
                        c045 = int(np.sum(confs >= 0.45))
            counts_025.append(c025)
            counts_045.append(c045)
            per_frame.append({
                "frame_index": frame_idx,
                "brightness": round(bri, 2),
                "people_025": c025,
                "people_045": c045,
            })
        frame_idx += 1

    cap.release()

    n = len(brightness_vals)
    return {
        "brightness_mean": round(float(np.mean(brightness_vals)), 2) if n else None,
        "brightness_std": round(float(np.std(brightness_vals)), 2) if n else None,
        "people_mean_025": round(float(np.mean(counts_025)), 2) if n else None,
        "people_max_025": int(max(counts_025)) if n else None,
        "people_mean_045": round(float(np.mean(counts_045)), 2) if n else None,
        "people_max_045": int(max(counts_045)) if n else None,
        "n_samples": n,
        "per_frame": per_frame,
    }


def discover_segments(recordings_root: Path) -> list[dict]:
    """Find all mp4+sidecar pairs across hours."""
    segments = []
    for cam in ["FP7oJQ", "PPDmUg"]:
        cam_dir = recordings_root / cam / "2026-08-04"
        if not cam_dir.exists():
            continue
        for hour_dir in sorted(cam_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            hour = hour_dir.name
            for mp4 in sorted(hour_dir.glob("*.mp4")):
                sidecar = mp4.with_suffix(".timing.jsonl")
                if not sidecar.exists():
                    continue
                stem = mp4.stem
                seg_id = stem  # e.g. FP7oJQ-20260804-163102
                segments.append({
                    "mp4_path": mp4,
                    "sidecar_path": sidecar,
                    "segment_filename": mp4.name,
                    "segment_id": seg_id,
                    "camera": cam,
                    "hour": hour,
                    "run_label": classify_run(hour, mp4.name),
                    "is_phase2": seg_id in PHASE2_SEGMENTS,
                })
    return segments


def main():
    parser = argparse.ArgumentParser(description="CP-R1b: fps adaptation analysis")
    parser.add_argument("--recordings", required=True, help="Path to gym recordings root")
    parser.add_argument("--output-dir", required=True, help="Evidence output directory")
    parser.add_argument("--model", default=None, help="Path to detection model .pt")
    parser.add_argument("--skip-detection", action="store_true", help="Skip detection (brightness only)")
    args = parser.parse_args()

    recordings_root = Path(args.recordings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transition_dt").mkdir(exist_ok=True)

    # Load detection model if provided
    det_model = None
    if args.model and not args.skip_detection:
        from ultralytics import YOLO
        det_model = YOLO(args.model)
        print(f"Loaded detection model: {args.model}")

    segments = discover_segments(recordings_root)
    print(f"Discovered {len(segments)} segments across 2 cameras")

    t0 = time.time()

    # Phase 1+5: sidecar metadata + ffprobe
    print("\n=== Phase 1+5: Sidecar metadata + ffprobe ===")
    for seg in segments:
        meta = extract_sidecar_meta(seg["sidecar_path"])
        seg["timing_mode"] = meta.get("timing_mode", "")
        seg["measured_fps"] = meta.get("measured_fps")
        seg["input_frame_count"] = meta.get("input_frame_count")
        seg["output_frame_count"] = meta.get("output_frame_count")
        seg["mismatch"] = meta.get("mismatch")
        seg["attempt"] = meta.get("attempt")
        seg["sidecar_schema"] = meta.get("sidecar_schema")

        # Exclude cfr_grid from analysis
        seg["excluded_cfr_grid"] = seg["timing_mode"] == "cfr_grid"

        # Duration estimate
        fps = seg["measured_fps"]
        fc = seg["input_frame_count"]
        if fps and fps > 0.1 and fc:
            seg["seg_seconds"] = round(fc / fps, 1)
        else:
            seg["seg_seconds"] = None

        # Wall start from filename
        parsed = parse_segment_filename(seg["segment_filename"])
        seg["wall_start_utc"] = f"{parsed['date']}-{parsed['time']}"

        # ffprobe
        probe = get_ffprobe_info(seg["mp4_path"])
        seg.update(probe)

    print(f"  Metadata extracted for {len(segments)} segments")

    # Phase 2: per-frame dt for transition segments
    print("\n=== Phase 2: Per-frame dt for transition segments ===")
    for seg in segments:
        if not seg["is_phase2"]:
            continue
        print(f"  Computing dt: {seg['segment_id']}")
        dt_rows = extract_per_frame_dt(seg["sidecar_path"])
        # Write dt CSV
        dt_path = output_dir / "transition_dt" / f"{seg['segment_id']}_dt.csv"
        with open(dt_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame_index", "pts_time_s", "dt_ms"])
            w.writeheader()
            w.writerows(dt_rows)

        # Compute step/ramp characterization
        valid_dts = [r["dt_ms"] for r in dt_rows if r["dt_ms"] is not None]
        if valid_dts:
            seg["dt_median_ms"] = round(float(np.median(valid_dts)), 3)
            seg["dt_min_ms"] = round(float(np.min(valid_dts)), 3)
            seg["dt_max_ms"] = round(float(np.max(valid_dts)), 3)
            seg["dt_std_ms"] = round(float(np.std(valid_dts)), 3)
            # Check for bimodality: count frames near 33ms vs 67ms
            arr = np.array(valid_dts)
            near_33 = int(np.sum((arr > 25) & (arr < 42)))
            near_67 = int(np.sum((arr > 58) & (arr < 75)))
            seg["n_near_33ms"] = near_33
            seg["n_near_67ms"] = near_67
            seg["n_dt_total"] = len(valid_dts)

    # Phase 3+4: brightness + detection (sequential decode)
    print("\n=== Phase 3+4: Brightness + detection (sequential decode) ===")
    phase2_per_frame_data = {}
    for i, seg in enumerate(segments):
        if seg["excluded_cfr_grid"]:
            seg["brightness_mean"] = None
            seg["brightness_std"] = None
            seg["people_mean_025"] = None
            seg["people_max_025"] = None
            seg["people_mean_045"] = None
            seg["people_max_045"] = None
            seg["n_samples"] = 0
            continue

        # C1: sequential decode. C2: stride 15 for phase2, 150 otherwise.
        # For short segments (20s smoke tests), use stride 30.
        if seg["is_phase2"]:
            stride = 15
        elif seg.get("seg_seconds") and seg["seg_seconds"] < 30:
            stride = 30
        else:
            stride = 150

        label = f"[{i+1}/{len(segments)}] {seg['segment_id']} (stride={stride})"
        print(f"  {label}", end="", flush=True)

        result = measure_brightness_and_detections(seg["mp4_path"], stride, det_model)
        seg["brightness_mean"] = result["brightness_mean"]
        seg["brightness_std"] = result["brightness_std"]
        seg["people_mean_025"] = result["people_mean_025"]
        seg["people_max_025"] = result["people_max_025"]
        seg["people_mean_045"] = result["people_mean_045"]
        seg["people_max_045"] = result["people_max_045"]
        seg["n_samples"] = result["n_samples"]

        if seg["is_phase2"] and result["per_frame"]:
            phase2_per_frame_data[seg["segment_id"]] = result["per_frame"]

        print(f" -> bri={result['brightness_mean']}, people@0.25={result['people_mean_025']}, "
              f"people@0.45={result['people_mean_045']}, n={result['n_samples']}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Write per-frame brightness for phase2 segments (for dt-brightness alignment)
    for seg_id, pf_data in phase2_per_frame_data.items():
        pf_path = output_dir / "transition_dt" / f"{seg_id}_brightness.csv"
        with open(pf_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame_index", "brightness", "people_025", "people_045"])
            w.writeheader()
            w.writerows(pf_data)

    # Write main survey CSV
    csv_path = output_dir / "segment_survey.csv"
    fieldnames = [
        "camera", "hour", "run_label", "segment_filename", "wall_start_utc",
        "timing_mode", "excluded_cfr_grid", "sidecar_schema",
        "measured_fps", "input_frame_count", "output_frame_count", "mismatch",
        "attempt", "seg_seconds", "is_phase2",
        "width", "height", "bitrate_bps",
        "brightness_mean", "brightness_std",
        "people_mean_025", "people_max_025", "people_mean_045", "people_max_045",
        "n_samples",
    ]
    # Add phase2-only dt fields
    dt_fields = ["dt_median_ms", "dt_min_ms", "dt_max_ms", "dt_std_ms",
                 "n_near_33ms", "n_near_67ms", "n_dt_total"]
    fieldnames.extend(dt_fields)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for seg in segments:
            row = {k: seg.get(k) for k in fieldnames}
            # Clean up paths
            row.pop("mp4_path", None)
            row.pop("sidecar_path", None)
            w.writerow(row)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {len(phase2_per_frame_data)} phase2 per-frame files")
    print(f"Wrote {sum(1 for s in segments if s['is_phase2'])} dt CSVs")


if __name__ == "__main__":
    main()
