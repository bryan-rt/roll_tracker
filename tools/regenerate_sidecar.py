#!/usr/bin/env python3
"""Regenerate a timing sidecar from an mp4 file.

Produces a schema 5 sidecar with row_source: "mp4_regenerated".
Frame rows and tick statistics from the mp4's PTS.
Showinfo-dependent fields (host_arrival_s, drift) are ABSENT (omission = invalid).

Usage:
    python tools/regenerate_sidecar.py <mp4_path> [--output <sidecar_path>]

Refuses pre-CP-R13a footage (container timebase != 1/90000) to avoid
silently producing degraded timing from requantized 1/15360 PTS.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _get_stream_info(mp4_path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=time_base,nb_frames",
         "-of", "json", str(mp4_path)],
        capture_output=True, text=True, timeout=30,
    )
    payload = json.loads(proc.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    return stream


def _get_frame_pts(mp4_path: Path) -> List[int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts",
         "-of", "csv=p=0", str(mp4_path)],
        capture_output=True, text=True, timeout=120,
    )
    pts = []
    for line in proc.stdout.strip().split("\n"):
        line = line.strip().rstrip(",")
        if line:
            pts.append(int(line))
    return pts


def _parse_epoch_from_filename(mp4_path: Path) -> int:
    m = re.search(r"(\d{8})-(\d{6})", mp4_path.stem)
    if not m:
        return 0
    try:
        from datetime import datetime
        dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
        return int(dt.timestamp())
    except Exception:
        return 0


def regenerate(mp4_path: Path, output_path: Optional[Path] = None) -> Path:
    if not mp4_path.exists():
        print(f"ERROR: mp4 not found: {mp4_path}", file=sys.stderr)
        sys.exit(1)

    # Check container timebase
    stream = _get_stream_info(mp4_path)
    time_base = stream.get("time_base", "")
    if time_base != "1/90000":
        print(
            f"ERROR: container timebase is {time_base}, not 1/90000. "
            f"This mp4 predates CP-R13a and carries requantized 1/15360 PTS. "
            f"Regenerating would produce degraded timing. Refusing.",
            file=sys.stderr,
        )
        sys.exit(1)

    timebase = 90000
    nb_frames = int(stream.get("nb_frames", 0))

    # Get frame PTS
    pts_list = _get_frame_pts(mp4_path)
    if not pts_list:
        print(f"ERROR: no frames found in {mp4_path}", file=sys.stderr)
        sys.exit(1)

    nm = len(pts_list)
    if nb_frames > 0 and nm != nb_frames:
        print(f"WARNING: ffprobe nb_frames={nb_frames} but got {nm} frame PTS", file=sys.stderr)

    # Base subtraction
    base = pts_list[0]

    # Tick deltas
    deltas = [pts_list[i] - pts_list[i - 1] for i in range(1, nm)]

    # Sort deltas for statistics
    sorted_d = sorted(deltas)
    nd = len(sorted_d)

    # Median
    if nd > 0:
        if nd % 2 == 1:
            median_tick = sorted_d[nd // 2]
        else:
            median_tick = (sorted_d[nd // 2 - 1] + sorted_d[nd // 2]) / 2.0
    else:
        median_tick = 0

    # Trimmed mean
    lo = median_tick * 0.5
    hi = median_tick * 1.5
    trim_vals = [d for d in sorted_d if lo <= d <= hi]
    trim_n = len(trim_vals)
    trimmed_mean_tick = sum(trim_vals) / trim_n if trim_n > 0 else median_tick

    # Mean
    mean_tick = sum(sorted_d) / nd if nd > 0 else 0

    # FPS
    measured_fps = timebase / trimmed_mean_tick if trimmed_mean_tick > 0 else 0
    measured_fps_median = timebase / median_tick if median_tick > 0 else 0
    last_s = (pts_list[-1] - base) / timebase if nm > 1 else 0
    measured_fps_mean = (nm - 1) / last_s if last_s > 0 else 0

    # Delta stats (ms)
    delta_ms = [d * 1000.0 / timebase for d in sorted_d]
    mean_d = sum(delta_ms) / nd if nd > 0 else 0
    sum_d2 = sum(x * x for x in delta_ms)
    stdev_d = math.sqrt(sum_d2 / nd - mean_d * mean_d) if nd > 0 else 0

    # Bimodal detection
    discard_below = sum(1 for d in sorted_d if d < lo)
    discard_above = sum(1 for d in sorted_d if d > hi)
    total_discard = discard_below + discard_above
    is_bimodal = False
    sm_frac = sm_fps = sm_dt = lm_dt = 0
    if total_discard >= 3 and discard_below > 0.3 * total_discard:
        is_bimodal = True
        short_thresh = median_tick * 0.75
        short_d = [d for d in sorted_d if d < short_thresh]
        long_d = [d for d in sorted_d if d >= short_thresh]
        sm_frac = len(short_d) / nd if nd > 0 else 0
        sm_tick = sum(short_d) / len(short_d) if short_d else 0
        lm_tick = sum(long_d) / len(long_d) if long_d else 0
        sm_fps = timebase / sm_tick if sm_tick > 0 else 0
        sm_dt = sm_tick / timebase if sm_tick > 0 else 0
        lm_dt = lm_tick / timebase if lm_tick > 0 else 0

    nominal_dt = median_tick / timebase if median_tick > 0 else 0

    epoch = _parse_epoch_from_filename(mp4_path)

    # Build _meta
    meta = {
        "_meta": True,
        "sidecar_schema": 5,
        "timing_mode": "passthrough",
        "source_pts": True,
        "pts_origin": "segment_relative",
        "fps_method": "trimmed_mean",
        "row_source": "mp4_regenerated",
        "segment_start_epoch": epoch,
        "attempt": 0,
        "input_frame_count": nm,
        "output_frame_count": nm,
        "nominal_dt_s": round(nominal_dt, 6),
        "measured_fps": round(measured_fps, 4),
        "measured_fps_median": round(measured_fps_median, 4),
        "measured_fps_mean": round(measured_fps_mean, 4),
        "pts_timebase": timebase,
        "pts_tick_delta_median": round(median_tick, 1),
        "pts_tick_delta_mean": round(mean_tick, 1),
        "pts_delta_trim_kept": trim_n,
        "pts_delta_trim_total": nd,
        "mismatch": False,
        "is_bimodal": is_bimodal,
        "pts_mean_delta_ms": round(mean_d, 4),
        "pts_stdev_delta_ms": round(stdev_d, 4),
    }
    if is_bimodal:
        meta["short_mode_fraction"] = round(sm_frac, 4)
        meta["short_mode_fps"] = round(sm_fps, 4)
        meta["short_mode_dt_s"] = round(sm_dt, 6)
        meta["long_mode_dt_s"] = round(lm_dt, 6)
    # Showinfo-dependent fields ABSENT (not emitted):
    # host_arrival_s, pts_wallclock_offset_s, offset_method,
    # drift_rate_s_per_s, drift_flat, drift_ppm, n_drift_windows,
    # showinfo_frame_count, showinfo_residual

    # Output path
    if output_path is None:
        output_path = mp4_path.with_suffix("").with_suffix(".timing.jsonl")
        # Handle .mp4 -> .timing.jsonl
        if mp4_path.suffix == ".mp4":
            output_path = mp4_path.parent / (mp4_path.stem + ".timing.jsonl")

    # Write sidecar
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        for i in range(nm):
            pts_s = (pts_list[i] - base) / timebase
            row = {"frame_index": i, "pts_time_s": round(pts_s, 6)}
            if i == 0:
                row["dt_s"] = None
            else:
                dt_s = (pts_list[i] - pts_list[i - 1]) / timebase
                row["dt_s"] = round(dt_s, 6)
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Assertion
    with open(output_path) as f:
        line_count = sum(1 for _ in f) - 1  # subtract _meta
    if line_count != nm:
        print(f"FATAL: sidecar row count ({line_count}) != frame count ({nm})", file=sys.stderr)
        sys.exit(1)

    print(f"Regenerated: {output_path} ({nm} frames, timebase 1/{timebase})", file=sys.stderr)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate timing sidecar from mp4")
    parser.add_argument("mp4", type=Path, help="Path to mp4 file")
    parser.add_argument("--output", type=Path, default=None, help="Output sidecar path")
    args = parser.parse_args()
    regenerate(args.mp4, args.output)


if __name__ == "__main__":
    main()
