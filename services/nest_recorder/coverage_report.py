#!/usr/bin/env python3
"""CP-R7/R10: Per-camera recording coverage metric.

Reads an output directory (one camera, one hour) and computes coverage against
an intended recording window.

Usage:
  python services/nest_recorder/coverage_report.py \
    data/raw/nest/.../FP7oJQ/2026-08-04/15 \
    --window 3900 --start-epoch 1785871950

  # Machine-readable:
  python services/nest_recorder/coverage_report.py ... --json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_segment_epoch(path: Path, tz_offset_hours: float) -> int | None:
    """Extract start epoch from segment filename like FP7oJQ-20260804-153236.mp4.

    Segment filenames use the container's local time (typically EDT = UTC-4).
    tz_offset_hours is the offset from UTC (e.g. -4 for EDT).
    """
    m = re.search(r"(\d{8})-(\d{6})", path.stem)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    tz = timezone(timedelta(hours=tz_offset_hours))
    dt = datetime(
        int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]),
        int(hms[:2]), int(hms[2:4]), int(hms[4:6]),
        tzinfo=tz,
    )
    return int(dt.timestamp())


def ffprobe_duration(path: Path) -> float:
    """Get duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    val = result.stdout.strip()
    if not val or val == "N/A":
        return 0.0
    return float(val)


def count_attempts(cam_dir: Path, window_start: int, window_end: int) -> int | None:
    """Count attempts from attempt_log.jsonl within the window."""
    log_path = cam_dir / "attempt_log.jsonl"
    if not log_path.exists():
        return None
    count = 0
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            epoch = float(entry.get("ffmpeg_start_epoch", 0))
            if window_start <= epoch <= window_end:
                count += 1
        except (json.JSONDecodeError, ValueError):
            continue
    return count


def compute_coverage(
    cam_dir: Path,
    window_seconds: float,
    start_epoch: int,
    gap_threshold: float,
    tz_offset_hours: float = -4.0,
) -> dict:
    """Compute coverage metrics for one camera directory."""
    window_start = start_epoch
    window_end = start_epoch + int(window_seconds)

    # Discover and filter segments
    mp4s = sorted(cam_dir.glob("*.mp4"))
    segments = []
    for mp4 in mp4s:
        epoch = parse_segment_epoch(mp4, tz_offset_hours)
        if epoch is None:
            continue
        if epoch < window_start:
            continue
        dur = ffprobe_duration(mp4)
        if dur <= 0:
            continue
        segments.append({"path": mp4.name, "start_epoch": epoch, "duration_s": dur})

    segments.sort(key=lambda s: s["start_epoch"])

    # Directory structure: .../gym_id/cam_id/YYYY-MM-DD/HH/
    # Walk up to find the camera ID (not a date or hour)
    cam_id = cam_dir.name
    for parent in [cam_dir.parent, cam_dir.parent.parent]:
        if not re.match(r"^\d{1,4}(-\d{2}){0,2}$", parent.name):
            cam_id = parent.name
            break

    # Attempt count
    attempts = count_attempts(cam_dir, window_start, window_end)

    if not segments:
        return {
            "cam_id": cam_id,
            "window_start": window_start,
            "window_end": window_end,
            "window_s": window_seconds,
            "total_recorded_s": 0.0,
            "coverage_pct": 0.0,
            "n_segments": 0,
            "n_attempts": attempts,
            "gaps": [{"name": "entire_window", "start_epoch": window_start,
                       "duration_s": window_seconds}],
            "n_gaps": 1,
            "longest_gap_s": window_seconds,
            "longest_run_s": 0.0,
            "total_gap_s": window_seconds,
            "lead_in_s": window_seconds,
            "tail_s": 0.0,
        }

    total_recorded = sum(s["duration_s"] for s in segments)
    coverage_pct = (total_recorded / window_seconds) * 100 if window_seconds > 0 else 0

    # Build gap list
    gaps = []

    # Lead-in
    lead_in = segments[0]["start_epoch"] - window_start
    if lead_in > gap_threshold:
        gaps.append({
            "name": "lead_in",
            "start_epoch": window_start,
            "duration_s": round(lead_in, 1),
        })

    # Inter-segment gaps
    gap_idx = 0
    for i in range(len(segments) - 1):
        seg_end = segments[i]["start_epoch"] + segments[i]["duration_s"]
        next_start = segments[i + 1]["start_epoch"]
        delta = next_start - seg_end
        if delta > gap_threshold:
            gap_idx += 1
            gaps.append({
                "name": f"gap_{gap_idx}",
                "start_epoch": round(seg_end),
                "duration_s": round(delta, 1),
            })

    # Tail
    last_end = segments[-1]["start_epoch"] + segments[-1]["duration_s"]
    tail = window_end - last_end
    if tail > gap_threshold:
        gaps.append({
            "name": "tail",
            "start_epoch": round(last_end),
            "duration_s": round(tail, 1),
        })

    total_gap = sum(g["duration_s"] for g in gaps)
    longest_gap = max((g["duration_s"] for g in gaps), default=0.0)

    # Longest continuous run: sequence of segments with inter-segment delta <= threshold
    longest_run = segments[0]["duration_s"] if segments else 0.0
    current_run_start = segments[0]["start_epoch"]
    current_run_end = segments[0]["start_epoch"] + segments[0]["duration_s"]
    for i in range(1, len(segments)):
        seg_end_prev = segments[i - 1]["start_epoch"] + segments[i - 1]["duration_s"]
        delta = segments[i]["start_epoch"] - seg_end_prev
        if delta <= gap_threshold:
            current_run_end = segments[i]["start_epoch"] + segments[i]["duration_s"]
        else:
            run_len = current_run_end - current_run_start
            if run_len > longest_run:
                longest_run = run_len
            current_run_start = segments[i]["start_epoch"]
            current_run_end = segments[i]["start_epoch"] + segments[i]["duration_s"]
    # Final run
    run_len = current_run_end - current_run_start
    if run_len > longest_run:
        longest_run = run_len

    return {
        "cam_id": cam_id,
        "window_start": window_start,
        "window_end": window_end,
        "window_s": window_seconds,
        "total_recorded_s": round(total_recorded, 1),
        "coverage_pct": round(coverage_pct, 1),
        "n_segments": len(segments),
        "n_attempts": attempts,
        "gaps": gaps,
        "n_gaps": len(gaps),
        "longest_gap_s": round(longest_gap, 1),
        "longest_run_s": round(longest_run, 1),
        "total_gap_s": round(total_gap, 1),
        "lead_in_s": round(lead_in, 1) if lead_in > 0 else 0.0,
        "tail_s": round(tail, 1) if tail > 0 else 0.0,
    }


def epoch_to_utc(epoch: int | float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def print_report(result: dict) -> None:
    print(f"\n=== Coverage Report: {result['cam_id']} ===")
    print(f"Window:     {result['window_s']:.0f}s "
          f"({epoch_to_utc(result['window_start'])} → "
          f"{epoch_to_utc(result['window_end'])} UTC)")
    print(f"Recorded:   {result['total_recorded_s']:.1f}s")
    print(f"Coverage:   {result['coverage_pct']:.1f}%")
    print(f"Segments:   {result['n_segments']}")
    attempts_str = str(result['n_attempts']) if result['n_attempts'] is not None else "unknown"
    print(f"Attempts:   {attempts_str}")
    print(f"Lead-in:    {result['lead_in_s']:.1f}s")
    print(f"Tail:       {result['tail_s']:.1f}s")
    print()

    if result["gaps"]:
        print(f"Gaps ({result['n_gaps']}):")
        for g in result["gaps"]:
            start_str = epoch_to_utc(g["start_epoch"])
            end_epoch = g["start_epoch"] + g["duration_s"]
            end_str = epoch_to_utc(end_epoch)
            print(f"  {g['name']:12s} {g['duration_s']:8.1f}s  ({start_str} → {end_str})")
        print()

    print(f"Total gap:    {result['total_gap_s']:.1f}s")
    print(f"Longest run:  {result['longest_run_s']:.1f}s")
    print(f"Longest gap:  {result['longest_gap_s']:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Per-camera recording coverage metric (CP-R7/R10)")
    parser.add_argument("cam_dir", type=Path,
                        help="Camera output directory (contains *.mp4)")
    parser.add_argument("--window", type=float, required=True,
                        help="Intended recording window in seconds")
    parser.add_argument("--start-epoch", type=int, required=True,
                        help="Unix epoch of window start")
    parser.add_argument("--gap-threshold", type=float, default=2.0,
                        help="Minimum gap duration in seconds (default: 2.0)")
    parser.add_argument("--tz-offset", type=float, default=-4.0,
                        help="Container timezone offset from UTC in hours (default: -4 = EDT)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of human-readable report")

    args = parser.parse_args()

    if not args.cam_dir.is_dir():
        print(f"Error: {args.cam_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = compute_coverage(
        args.cam_dir, args.window, args.start_epoch, args.gap_threshold,
        args.tz_offset,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
