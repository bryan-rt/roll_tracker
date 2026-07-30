#!/usr/bin/env python3
"""RECORDER-TIMING-1: Analyze per-frame timing from recorder test captures.

Subcommands:
  analyze   — Extract per-frame timing from a clip, flag anomalies, check Stage A compat
  compare   — Compare frame counts between VFR and CFR clips
  dupfix    — RECORDER-DUPFIX-1: Resolve duplicate-frame contradiction at decode level

Usage:
  python tools/analyze_recorder_timing.py analyze \
    --clip <path.mp4> --mode {vfr,cfr,cfr-sidecar} \
    [--stderr <ffmpeg.stderr>] [--output-dir <dir>]

  python tools/analyze_recorder_timing.py compare \
    --vfr <vfr.mp4> --cfr <cfr.mp4> [--output-dir <dir>]

  python tools/analyze_recorder_timing.py dupfix \
    --segments <mp4> [<mp4> ...] \
    --stderr <stderr> [<stderr> ...] \
    --output-dir docs/evidence/recorder_dupfix_1/
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frame timing extraction
# ---------------------------------------------------------------------------

def extract_pts_ffprobe(clip_path: Path) -> pd.DataFrame:
    """Extract per-frame PTS via ffprobe -show_frames."""
    cmd = [
        "ffprobe", "-hide_banner", "-select_streams", "v:0",
        "-show_entries",
        "frame=pts_time,pkt_pts_time,pkt_dts_time,pkt_duration_time,"
        "best_effort_timestamp_time",
        "-of", "json", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"ffprobe failed: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)
    rows = []
    for i, f in enumerate(data.get("frames", [])):
        # Prefer best_effort_timestamp_time > pts_time > pkt_pts_time > pkt_dts_time
        pts = (f.get("best_effort_timestamp_time")
               or f.get("pts_time")
               or f.get("pkt_pts_time")
               or f.get("pkt_dts_time"))
        dts = f.get("pkt_dts_time")
        dur = f.get("pkt_duration_time")
        rows.append({
            "frame_index": i,
            "pts_time_s": float(pts) if pts else None,
            "dts_time_s": float(dts) if dts else None,
            "duration_s": float(dur) if dur else None,
        })
    return pd.DataFrame(rows)


def extract_stts(clip_path: Path) -> list[dict]:
    """Extract stts (sample-to-time) atom entries via ffprobe."""
    cmd = [
        "ffprobe", "-hide_banner", "-select_streams", "v:0",
        "-show_entries", "stream=time_base,nb_frames,r_frame_rate,avg_frame_rate",
        "-of", "json", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stream_info = {}
    if result.returncode == 0:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            stream_info = streams[0]

    # Get raw stts via mp4 box dump — use ffprobe -show_entries format_tags
    # Simpler: count unique durations from the frame data
    return stream_info


def parse_showinfo_stderr(stderr_path: Path) -> pd.DataFrame:
    """Parse showinfo filter output from ffmpeg stderr.

    Lines look like:
    [Parsed_showinfo_0 @ 0x...] n:   0 pts:      0 pts_time:0        ...
    [Parsed_showinfo_0 @ 0x...] n:   1 pts:    512 pts_time:0.033333  ...
    """
    pattern = re.compile(
        r"\[Parsed_showinfo_\d+ @ 0x[0-9a-f]+\]\s+"
        r"n:\s*(\d+)\s+"
        r"pts:\s*(-?\d+)\s+"
        r"pts_time:\s*([0-9.eE+-]+)"
    )
    rows = []
    with open(stderr_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "frame_index": int(m.group(1)),
                    "showinfo_pts": int(m.group(2)),
                    "showinfo_pts_time_s": float(m.group(3)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame, pts_col: str = "pts_time_s") -> pd.DataFrame:
    """Add inter-frame delta column."""
    df = df.copy()
    pts = pd.to_numeric(df[pts_col], errors="coerce").values
    deltas = np.diff(pts)
    df["delta_ms"] = np.concatenate([[np.nan], deltas * 1000.0])
    return df


def flag_anomalies(df: pd.DataFrame, delta_col: str = "delta_ms",
                   low_factor: float = 0.5, high_factor: float = 2.0) -> pd.DataFrame:
    """Flag frames with anomalous inter-frame deltas."""
    df = df.copy()
    valid = df[delta_col].dropna()
    if valid.empty:
        df["anomaly"] = False
        return df
    median_delta = valid.median()
    lo = median_delta * low_factor
    hi = median_delta * high_factor
    df["anomaly"] = df[delta_col].apply(
        lambda d: False if pd.isna(d) else (d < lo or d > hi)
    )
    return df


def summarize_timing(df: pd.DataFrame, delta_col: str = "delta_ms",
                     label: str = "") -> dict:
    """Produce a timing summary dict."""
    valid = df[delta_col].dropna()
    anomalies = df[df.get("anomaly", False) == True]
    unique_deltas_6dp = set(round(d, 3) for d in valid) if not valid.empty else set()

    summary = {
        "label": label,
        "frame_count": len(df),
        "min_delta_ms": round(float(valid.min()), 6) if not valid.empty else None,
        "max_delta_ms": round(float(valid.max()), 6) if not valid.empty else None,
        "mean_delta_ms": round(float(valid.mean()), 6) if not valid.empty else None,
        "median_delta_ms": round(float(valid.median()), 6) if not valid.empty else None,
        "stdev_delta_ms": round(float(valid.std()), 6) if not valid.empty else None,
        "unique_deltas_count": len(unique_deltas_6dp),
        "anomaly_count": len(anomalies),
        "anomaly_pct": round(100.0 * len(anomalies) / len(valid), 2) if not valid.empty else 0,
    }

    # Anomaly windows (contiguous runs of anomalous frames)
    if not anomalies.empty:
        anom_indices = anomalies["frame_index"].tolist()
        windows = []
        start = anom_indices[0]
        prev = start
        for idx in anom_indices[1:]:
            if idx > prev + 3:  # gap of >3 frames = new window
                windows.append({"start": start, "end": prev,
                                "frames": prev - start + 1})
                start = idx
            prev = idx
        windows.append({"start": start, "end": prev,
                        "frames": prev - start + 1})
        summary["anomaly_windows"] = windows
    else:
        summary["anomaly_windows"] = []

    return summary


def check_stage_a_compat(clip_path: Path) -> dict:
    """Verify the clip is compatible with Stage A's frame_index contract."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {clip_path}"}

    prop_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prop_fps = cap.get(cv2.CAP_PROP_FPS)

    actual_count = 0
    decode_errors = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.size == 0:
            decode_errors += 1
        actual_count += 1
    cap.release()

    contiguous = True  # frame_index is a simple counter in FrameIterator
    # so it's always contiguous by construction — the question is whether
    # cap.read() returns all frames without skipping

    return {
        "clip": str(clip_path),
        "cap_prop_frame_count": prop_frame_count,
        "actual_iterated_count": actual_count,
        "frame_count_match": prop_frame_count == actual_count,
        "cap_prop_fps": round(prop_fps, 4) if prop_fps else None,
        "decode_errors": decode_errors,
        "contiguous_index": contiguous,
        "stage_a_compatible": (
            actual_count > 0
            and decode_errors == 0
            and prop_frame_count == actual_count
        ),
    }


def timing_varies(summary: dict, threshold_unique: int = 5) -> bool:
    """Does the timing actually vary (core validation)?

    Returns True if there are more than threshold_unique distinct delta values
    (at 3-decimal-place / microsecond precision). CFR clips have 1-2 unique
    deltas (floating-point noise). VFR/sidecar should have many more.
    """
    return summary.get("unique_deltas_count", 0) > threshold_unique


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    clip_path = Path(args.clip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Analyzing {clip_path.name} (mode={args.mode}) ===\n")

    # 1. Extract container PTS
    print("Extracting per-frame PTS via ffprobe...")
    df_pts = extract_pts_ffprobe(clip_path)
    df_pts = compute_deltas(df_pts, "pts_time_s")
    df_pts = flag_anomalies(df_pts)
    summary_pts = summarize_timing(df_pts, label=f"container_pts ({args.mode})")

    print(f"  Frames: {summary_pts['frame_count']}")
    print(f"  Delta range: {summary_pts['min_delta_ms']:.3f} – {summary_pts['max_delta_ms']:.3f} ms")
    print(f"  Mean±stdev: {summary_pts['mean_delta_ms']:.3f} ± {summary_pts['stdev_delta_ms']:.6f} ms")
    print(f"  Unique deltas (0.001ms): {summary_pts['unique_deltas_count']}")
    print(f"  Anomalies: {summary_pts['anomaly_count']} ({summary_pts['anomaly_pct']:.1f}%)")

    varies = timing_varies(summary_pts)
    print(f"\n  >>> TIMING VARIES: {'YES' if varies else 'NO (uniform — timing NOT preserved)'}")
    if summary_pts["anomaly_windows"]:
        print(f"  Anomaly windows:")
        for w in summary_pts["anomaly_windows"][:20]:
            print(f"    frames {w['start']}–{w['end']} ({w['frames']} frames)")
    print()

    # 2. Showinfo sidecar (REENCODE=2 only)
    summary_showinfo = None
    df_showinfo = None
    if args.mode == "cfr-sidecar":
        if not args.stderr:
            print("ERROR: --stderr required for cfr-sidecar mode", file=sys.stderr)
            sys.exit(1)
        stderr_path = Path(args.stderr)
        print(f"Parsing showinfo from {stderr_path.name}...")
        df_showinfo = parse_showinfo_stderr(stderr_path)
        if df_showinfo.empty:
            print("  WARNING: No showinfo lines found in stderr!")
            print("  Check that REENCODE=2 was used and -vf showinfo was active.")
        else:
            df_showinfo = compute_deltas(df_showinfo, "showinfo_pts_time_s")
            df_showinfo = flag_anomalies(df_showinfo, "delta_ms")
            summary_showinfo = summarize_timing(
                df_showinfo, label="showinfo_sidecar")

            print(f"  Showinfo frames: {summary_showinfo['frame_count']}")
            print(f"  Delta range: {summary_showinfo['min_delta_ms']:.3f} – {summary_showinfo['max_delta_ms']:.3f} ms")
            print(f"  Mean±stdev: {summary_showinfo['mean_delta_ms']:.3f} ± {summary_showinfo['stdev_delta_ms']:.6f} ms")
            print(f"  Unique deltas (0.001ms): {summary_showinfo['unique_deltas_count']}")
            print(f"  Anomalies: {summary_showinfo['anomaly_count']} ({summary_showinfo['anomaly_pct']:.1f}%)")

            si_varies = timing_varies(summary_showinfo)
            print(f"\n  >>> SHOWINFO TIMING VARIES: {'YES' if si_varies else 'NO (uniform — sidecar NOT useful)'}")

            # Cross-check: showinfo frame count vs container frame count
            if summary_showinfo["frame_count"] != summary_pts["frame_count"]:
                print(f"  WARNING: showinfo frames ({summary_showinfo['frame_count']}) != "
                      f"container frames ({summary_pts['frame_count']})")
            else:
                print(f"  Showinfo frame count matches container: {summary_showinfo['frame_count']}")
        print()

    # 3. Stream info
    stream_info = extract_stts(clip_path)
    if stream_info:
        print(f"Stream info:")
        for k, v in stream_info.items():
            print(f"  {k}: {v}")
        print()

    # 4. Stage A compatibility check
    print("Running Stage A compatibility check (cv2.VideoCapture iteration)...")
    compat = check_stage_a_compat(clip_path)
    print(f"  CAP_PROP_FRAME_COUNT: {compat['cap_prop_frame_count']}")
    print(f"  Actual iterated count: {compat['actual_iterated_count']}")
    print(f"  Frame count match: {compat['frame_count_match']}")
    print(f"  CAP_PROP_FPS: {compat['cap_prop_fps']}")
    print(f"  Decode errors: {compat['decode_errors']}")
    print(f"  Stage A compatible: {'YES' if compat['stage_a_compatible'] else 'NO'}")
    print()

    # 5. Save outputs
    df_pts.to_parquet(output_dir / "frame_timing_container.parquet", index=False)

    results = {
        "clip": str(clip_path),
        "mode": args.mode,
        "container_pts": summary_pts,
        "stream_info": stream_info,
        "stage_a_compat": compat,
        "timing_varies": varies,
    }
    if summary_showinfo is not None:
        results["showinfo_sidecar"] = summary_showinfo
        results["showinfo_timing_varies"] = timing_varies(summary_showinfo)
        df_showinfo.to_parquet(output_dir / "frame_timing_showinfo.parquet",
                               index=False)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Outputs saved to {output_dir}/")

    # 6. Verdict
    print("\n" + "=" * 60)
    if args.mode == "vfr":
        if varies:
            print("VERDICT: VFR timing PRESERVED — container PTS varies at "
                  "sub-frame granularity.")
            if not compat["stage_a_compatible"]:
                print("WARNING: Stage A compatibility issue detected!")
        else:
            print("VERDICT: VFR timing NOT preserved — container PTS is uniform.")
            print("Stream copy may have preserved camera's original uniform PTS")
            print("rather than injecting wall-clock variation. Path 1 FAILED.")
    elif args.mode == "cfr-sidecar":
        si_ok = summary_showinfo and timing_varies(summary_showinfo)
        if si_ok:
            print("VERDICT: CFR+sidecar timing PRESERVED — showinfo PTS varies")
            print("(captured before CFR re-timestamping).")
        elif summary_showinfo:
            print("VERDICT: CFR+sidecar timing NOT preserved — showinfo PTS is")
            print("uniform. The -vf showinfo may be seeing post-CFR timing.")
            print("Path 2 FAILED.")
        else:
            print("VERDICT: No showinfo data found. Check REENCODE=2 config.")
    elif args.mode == "cfr":
        if varies:
            print("UNEXPECTED: CFR baseline shows timing variation!")
        else:
            print("CONFIRMED: CFR baseline has uniform timing (expected).")
    print("=" * 60)


def cmd_compare(args: argparse.Namespace) -> None:
    vfr_path = Path(args.vfr)
    cfr_path = Path(args.cfr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Frame Count Comparison ===\n")

    vfr_compat = check_stage_a_compat(vfr_path)
    cfr_compat = check_stage_a_compat(cfr_path)

    vfr_count = vfr_compat["actual_iterated_count"]
    cfr_count = cfr_compat["actual_iterated_count"]
    delta = cfr_count - vfr_count

    print(f"VFR clip: {vfr_path.name}")
    print(f"  Frames: {vfr_count}  FPS: {vfr_compat['cap_prop_fps']}")
    print(f"  Stage A compatible: {'YES' if vfr_compat['stage_a_compatible'] else 'NO'}")
    print()
    print(f"CFR clip: {cfr_path.name}")
    print(f"  Frames: {cfr_count}  FPS: {cfr_compat['cap_prop_fps']}")
    print(f"  Stage A compatible: {'YES' if cfr_compat['stage_a_compatible'] else 'NO'}")
    print()
    print(f"Delta (CFR - VFR): {delta:+d} frames")
    if cfr_count > 0:
        print(f"  = {100.0 * abs(delta) / cfr_count:.2f}% of CFR frame count")
    print()

    if delta > 0:
        print(f"CFR has {delta} MORE frames than VFR.")
        print("These are likely duplicate frames inserted during camera pauses.")
    elif delta < 0:
        print(f"VFR has {abs(delta)} MORE frames than CFR.")
        print("CFR likely dropped frames during camera speed-up bursts.")
    else:
        print("Frame counts are IDENTICAL.")
        print("Either no lag events occurred, or VFR is not truly variable-rate.")
    print()

    # Duration comparison via ffprobe
    for label, path in [("VFR", vfr_path), ("CFR", cfr_path)]:
        cmd = [
            "ffprobe", "-hide_banner", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "json", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams and "duration" in streams[0]:
                dur = float(streams[0]["duration"])
                count = vfr_count if label == "VFR" else cfr_count
                eff_fps = count / dur if dur > 0 else 0
                print(f"{label} duration: {dur:.3f}s  effective fps: {eff_fps:.3f}")

    print()
    print("NOTE: VFR and CFR clips were captured in DIFFERENT time windows.")
    print("Frame count delta reflects BOTH timing differences AND different")
    print("lag events during each capture. Directional signal only.")

    comparison = {
        "vfr": {"clip": str(vfr_path), **vfr_compat},
        "cfr": {"clip": str(cfr_path), **cfr_compat},
        "frame_count_delta_cfr_minus_vfr": delta,
    }
    with open(output_dir / "frame_count_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/frame_count_comparison.json")

    # Dependencies reminder
    print("\n" + "=" * 60)
    print("DEPENDENCIES if adopting VFR for future captures:")
    print("  1. derive_clip_frame_offset (session_d_run.py) must be rewritten")
    print("     to align by wall-clock PTS, not frame_index * fps")
    print("  2. VFR frame indices are NOT comparable to existing CFR/GT frames")
    print("     — all GT annotations would need re-capture under VFR")
    print("  3. Cross-camera offset computation needs wall-clock alignment")
    print("=" * 60)


# ---------------------------------------------------------------------------
# DUPFIX helpers
# ---------------------------------------------------------------------------

def extract_framehash(clip_path: Path) -> tuple[int, int, list[int]]:
    """Decode all video frames and compute per-frame MD5 hashes.

    Returns (total_frames, adjacent_dup_count, dup_frame_indices).
    Uses -map 0:v:0 to exclude audio (C3).
    """
    cmd = [
        "ffmpeg", "-i", str(clip_path),
        "-map", "0:v:0",
        "-f", "framehash", "-hash", "md5", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"framehash failed: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    hashes = []
    stream_indices = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: stream_index, packet_dts, packet_pts, packet_duration, packet_size, hash
        parts = line.split(",")
        if len(parts) >= 6:
            stream_indices.add(parts[0].strip())
            hashes.append(parts[-1].strip())

    if len(stream_indices) > 1:
        print(f"ERROR: framehash found multiple stream indices: {stream_indices}. "
              f"Expected exactly 1 (video only).", file=sys.stderr)
        sys.exit(1)

    adj_dups = 0
    dup_indices = []
    for i in range(1, len(hashes)):
        if hashes[i] == hashes[i - 1]:
            adj_dups += 1
            dup_indices.append(i)

    return len(hashes), adj_dups, dup_indices


def run_mpdecimate_fresh(clip_path: Path, strict: bool = True
                         ) -> tuple[int, int, int]:
    """Run mpdecimate on clip.

    Args:
        strict: If True, use hi=1:lo=1:frac=1 (pixel-identical only).
                If False, use default thresholds (near-identical).

    Returns (input_frames, output_frames, dropped_frames).
    Dropped = input - output (mpdecimate removes frames from the stream).
    """
    vf = "mpdecimate=hi=1:lo=1:frac=1" if strict else "mpdecimate"
    cmd = [
        "ffmpeg", "-i", str(clip_path),
        "-map", "0:v:0",
        "-vf", vf,
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Get input frame count from nb_frames
    cmd_nb = [
        "ffprobe", "-hide_banner", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(clip_path),
    ]
    nb_result = subprocess.run(cmd_nb, capture_output=True, text=True, timeout=300)
    input_frames = int(nb_result.stdout.strip()) if nb_result.stdout.strip() else 0

    # Get output frame count from the final progress line
    output_frames = 0
    for line in result.stderr.splitlines():
        m = re.search(r"frame=\s*(\d+)", line)
        if m:
            output_frames = int(m.group(1))

    dropped = input_frames - output_frames
    return input_frames, output_frames, dropped


def split_showinfo_by_segment(stderr_path: Path) -> dict[str, list[str]]:
    """Split showinfo lines by Opening boundaries, mirroring extract_timing_sidecars().

    Returns {segment_filename: [showinfo_lines]}.
    """
    with open(stderr_path) as f:
        all_lines = f.readlines()

    # Find Opening boundaries
    opening_re = re.compile(r"Opening '(.+?\.mp4)' for writing")
    boundaries = []  # (line_index, filename)
    for i, line in enumerate(all_lines):
        m = opening_re.search(line)
        if m:
            boundaries.append((i, Path(m.group(1)).name))

    if not boundaries:
        return {}

    showinfo_re = re.compile(r"Parsed_showinfo.*pts_time:")
    result = {}
    for bi, (start_idx, filename) in enumerate(boundaries):
        end_idx = boundaries[bi + 1][0] if bi + 1 < len(boundaries) else len(all_lines)
        si_lines = [l for l in all_lines[start_idx:end_idx] if showinfo_re.search(l)]
        result[filename] = si_lines

    return result


def read_sidecar_meta(timing_jsonl_path: Path) -> dict | None:
    """Read _meta line from existing .timing.jsonl sidecar."""
    if not timing_jsonl_path.exists():
        return None
    with open(timing_jsonl_path) as f:
        first = f.readline().strip()
        if first:
            meta = json.loads(first)
            if meta.get("_meta"):
                return meta
    return None


def count_sidecar_input_n_patterns(timing_jsonl_path: Path) -> dict:
    """Count adjacent-identical input_n (dups) and forward jumps in sidecar.

    Returns {total_frames, dups, dup_pct, jumps, jump_pct}.
    """
    if not timing_jsonl_path.exists():
        return {"error": f"sidecar not found: {timing_jsonl_path}"}

    input_ns = []
    with open(timing_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                continue
            input_ns.append(row.get("input_n", -1))

    if len(input_ns) < 2:
        return {"total_frames": len(input_ns), "dups": 0, "dup_pct": 0.0,
                "jumps": 0, "jump_pct": 0.0}

    dups = 0
    jumps = 0
    for i in range(1, len(input_ns)):
        diff = input_ns[i] - input_ns[i - 1]
        if diff == 0:
            dups += 1
        elif diff > 1:
            jumps += 1

    total = len(input_ns)
    return {
        "total_frames": total,
        "dups": dups,
        "dup_pct": round(100.0 * dups / total, 2) if total > 0 else 0.0,
        "jumps": jumps,
        "jump_pct": round(100.0 * jumps / total, 2) if total > 0 else 0.0,
    }


def compute_pts_gaps(showinfo_lines: list[str],
                     nominal_interval_ms: float | None = None
                     ) -> dict:
    """Compute per-frame PTS gap analysis from raw showinfo lines.

    Args:
        showinfo_lines: Lines from ffmpeg stderr containing Parsed_showinfo.
        nominal_interval_ms: Expected inter-frame interval. If None, uses median.

    Returns dict with pts_nominal_interval_ms, pts_gap_count,
    pts_implied_missing_frames, pts_gap_histogram.
    """
    pts_re = re.compile(r"pts_time:\s*([0-9.eE+-]+)")
    pts_vals = []
    for line in showinfo_lines:
        m = pts_re.search(line)
        if m:
            pts_vals.append(float(m.group(1)))

    if len(pts_vals) < 2:
        return {
            "pts_nominal_interval_ms": None,
            "pts_gap_count": 0,
            "pts_implied_missing_frames": 0,
            "pts_gap_histogram": {"2x": 0, "3x": 0, "4x": 0, "5x_plus": 0},
        }

    deltas_ms = [
        (pts_vals[j] - pts_vals[j - 1]) * 1000.0
        for j in range(1, len(pts_vals))
    ]
    arr = np.array(deltas_ms)
    nominal = nominal_interval_ms if nominal_interval_ms else float(np.median(arr))

    threshold = 1.5 * nominal
    gap_count = 0
    implied_missing = 0
    histogram = {"2x": 0, "3x": 0, "4x": 0, "5x_plus": 0}

    for d in deltas_ms:
        if d > threshold:
            gap_count += 1
            ratio = d / nominal
            missing = round(ratio) - 1
            implied_missing += missing
            if ratio < 2.5:
                histogram["2x"] += 1
            elif ratio < 3.5:
                histogram["3x"] += 1
            elif ratio < 4.5:
                histogram["4x"] += 1
            else:
                histogram["5x_plus"] += 1

    return {
        "pts_nominal_interval_ms": round(nominal, 4),
        "pts_gap_count": gap_count,
        "pts_implied_missing_frames": implied_missing,
        "pts_gap_histogram": histogram,
    }


def get_nb_read_frames(clip_path: Path) -> int:
    """Get true decoded frame count via ffprobe -count_frames."""
    cmd = [
        "ffprobe", "-hide_banner", "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    val = result.stdout.strip()
    if val and val != "N/A":
        return int(val)
    return -1


def get_encoder_tag(clip_path: Path) -> str:
    """Get encoder format tag from mp4."""
    cmd = [
        "ffprobe", "-hide_banner",
        "-show_entries", "format_tags=encoder",
        "-of", "csv=p=0", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout.strip() or "unknown"


def infer_recorder_mode(clip_path: Path) -> str:
    """Infer recorder mode from filename date.

    Pre-July 2026 = arrival-PTS (old recorder), July+ = source-PTS.
    """
    name = clip_path.stem
    m = re.search(r"(\d{8})-(\d{6})", name)
    if m:
        date_str = m.group(1)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        if year < 2026 or (year == 2026 and month < 7):
            return "arrival-PTS"
    return "source-PTS"


# ---------------------------------------------------------------------------
# DUPFIX subcommand
# ---------------------------------------------------------------------------

def cmd_dupfix(args: argparse.Namespace) -> None:
    segments = [Path(s) for s in args.segments]
    stderr_files = [Path(s) for s in args.stderr_files]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(segments) != len(stderr_files):
        print(f"ERROR: {len(segments)} segments but {len(stderr_files)} stderr files. "
              f"Must be 1:1.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("RECORDER-DUPFIX-1: Resolve duplicate-frame contradiction at decode level")
    print("=" * 70)
    print()

    # --- Instrument Validation ---
    # Find the arrival-PTS control clip (first one with arrival-PTS mode)
    control_idx = None
    for i, seg in enumerate(segments):
        if infer_recorder_mode(seg) == "arrival-PTS":
            control_idx = i
            break

    if control_idx is not None:
        control_path = segments[control_idx]
        print("=" * 70)
        print("INSTRUMENT VALIDATION — Known-Bad Control")
        print(f"Clip: {control_path.name}")
        print(f"Expected: arrival-PTS era, mpdecimate ~255 dups (~5.6%)")
        print("=" * 70)
        print()

        # Step 1: framehash
        print("Running framehash (video-only, MD5)...")
        fh_total, fh_dups, _ = extract_framehash(control_path)
        print(f"  framehash_total: {fh_total}")
        print(f"  framehash_adjacent_dups: {fh_dups}")
        print(f"  framehash_dup_pct: {100.0 * fh_dups / fh_total:.2f}%" if fh_total > 0 else "")

        # Gate check 1: total matches historical 4530 (same decode)
        historical_total = 4530
        if fh_total != historical_total:
            print(f"\n  WARNING: framehash_total ({fh_total}) != historical "
                  f"total ({historical_total}).")
            print(f"  Not the same decode as RELIABILITY-1. Investigating but not gating.")
        else:
            print(f"  framehash_total matches historical {historical_total}.")
        print()

        # Step 2: fresh mpdecimate (strict = pixel-identical only)
        print("Running mpdecimate fresh (strict: hi=1:lo=1:frac=1)...")
        mpd_input, mpd_output, mpd_dups = run_mpdecimate_fresh(control_path, strict=True)
        print(f"  mpdecimate input:  {mpd_input}")
        print(f"  mpdecimate output: {mpd_output}")
        print(f"  mpdecimate strict dups (input-output): {mpd_dups}")
        print()

        # Also run default for comparison
        print("Running mpdecimate fresh (default thresholds)...")
        mpd_d_input, mpd_d_output, mpd_d_dups = run_mpdecimate_fresh(control_path, strict=False)
        print(f"  mpdecimate default dups: {mpd_d_dups}")
        print(f"  (Default thresholds catch near-identical; strict catches pixel-identical)")
        print(f"  RELIABILITY-1 reported 255 dups — likely default or intermediate thresholds.")
        print()

        # Gate: framehash should agree with strict mpdecimate (both pixel-identical)
        # Allow small difference (1-2 frames) due to hash vs pixel-diff methods
        diff = abs(fh_dups - mpd_dups)
        if fh_dups == 0:
            print(f"  GATE FAIL: framehash found 0 dups on a known-bad clip.")
            print(f"  Instrument is broken. Stopping.")
            sys.exit(1)
        elif diff <= 5:
            print(f"  GATE PASS: framehash ({fh_dups}) ≈ strict mpdecimate ({mpd_dups}), "
                  f"diff={diff}.")
            print(f"  Both methods agree on pixel-identical duplicates.")
        else:
            print(f"  GATE WARNING: framehash ({fh_dups}) vs strict mpdecimate ({mpd_dups}), "
                  f"diff={diff}.")
            print(f"  Methods disagree — investigate threshold sensitivity.")
        print()

    # --- Falsifiable Prediction (C4) ---
    print("=" * 70)
    print("FALSIFIABLE PREDICTION (recorded before measurement)")
    print("=" * 70)
    print()
    print("If Count 2 (nb_read_frames) for FP7oJQ-20260728-062531.mp4 returns ~1867")
    print("rather than 1830, the sidecar's output_count is wrong at source and a large")
    print("share of the reported mismatch dissolves without any CFR-padding argument")
    print("being needed.")
    print()
    print("Existing evidence from RELIABILITY-1:")
    print("  mpdecimate Frames: 1867, Dups: 37, nb_frames: 1830")
    print("  1867 - 37 = 1830 (matches nb_frames exactly)")
    print("  showinfo_count: 1857 (from sidecar)")
    print("  Three counts: showinfo 1857, decode ~1867, nb_frames 1830")
    print()

    # --- Per-Segment Analysis ---
    print("=" * 70)
    print("PER-SEGMENT ANALYSIS")
    print("=" * 70)
    print()

    results = []

    for i, (seg_path, stderr_path) in enumerate(zip(segments, stderr_files)):
        seg_name = seg_path.name
        cam_id = seg_name.split("-")[0]
        recorder_mode = infer_recorder_mode(seg_path)
        is_arrival = recorder_mode == "arrival-PTS"

        print(f"--- Segment {i+1}/{len(segments)}: {seg_name} ---")
        print(f"    Camera: {cam_id}  Mode: {recorder_mode}")
        print()

        # Count 1: nb_frames (container metadata)
        cmd_nb = [
            "ffprobe", "-hide_banner", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0", str(seg_path),
        ]
        nb_frames_result = subprocess.run(cmd_nb, capture_output=True, text=True, timeout=30)
        nb_frames_str = nb_frames_result.stdout.strip()
        nb_frames = int(nb_frames_str) if nb_frames_str and nb_frames_str != "N/A" else -1
        print(f"    Count 1 (nb_frames):       {nb_frames}")

        # Count 2: nb_read_frames (true decoded)
        print(f"    Count 2 (nb_read_frames):  ", end="", flush=True)
        nb_read = get_nb_read_frames(seg_path)
        print(nb_read)

        # Count 3: cv2 iterated count
        compat = check_stage_a_compat(seg_path)
        cv2_count = compat["actual_iterated_count"]
        cv2_fps = compat["cap_prop_fps"]
        print(f"    Count 3 (cv2 iterated):    {cv2_count}")

        # Count 4: framehash adjacent dups
        print(f"    Count 4 (framehash):       ", end="", flush=True)
        fh_total, fh_dups, fh_dup_indices = extract_framehash(seg_path)
        fh_pct = round(100.0 * fh_dups / fh_total, 2) if fh_total > 0 else 0.0
        print(f"{fh_total} total, {fh_dups} adj dups ({fh_pct}%)")

        # Count 5: showinfo count (attributed to this segment)
        seg_showinfo = split_showinfo_by_segment(stderr_path)
        si_count = 0
        seg_key = seg_name
        if seg_key in seg_showinfo:
            si_count = len(seg_showinfo[seg_key])
        else:
            # Try matching without exact filename
            for k, v in seg_showinfo.items():
                if seg_name.replace(".mp4", "") in k:
                    si_count = len(v)
                    seg_key = k
                    break
            if si_count == 0:
                # Single-segment stderr: count all showinfo lines
                total_si = sum(len(v) for v in seg_showinfo.values())
                if len(seg_showinfo) == 1:
                    si_count = list(seg_showinfo.values())[0].__len__()
                else:
                    # Count total from file directly
                    with open(stderr_path) as f:
                        si_count = sum(1 for l in f if "Parsed_showinfo" in l and "pts_time:" in l)
        print(f"    Count 5 (showinfo):        {si_count}")

        # Sidecar data
        sidecar_path = seg_path.with_suffix(".timing.jsonl")
        sidecar_meta = read_sidecar_meta(sidecar_path)
        sidecar_input_count = sidecar_meta.get("input_frame_count", -1) if sidecar_meta else -1
        sidecar_output_count = sidecar_meta.get("output_frame_count", -1) if sidecar_meta else -1
        sidecar_measured_fps = sidecar_meta.get("measured_fps", None) if sidecar_meta else None

        # C1: input_n patterns from sidecar
        input_n_patterns = count_sidecar_input_n_patterns(sidecar_path)
        sidecar_input_n_dups = input_n_patterns.get("dups", -1)
        sidecar_input_n_dup_pct = input_n_patterns.get("dup_pct", 0.0)
        sidecar_input_n_jumps = input_n_patterns.get("jumps", -1)
        sidecar_input_n_jump_pct = input_n_patterns.get("jump_pct", 0.0)

        print(f"    Sidecar input_count:       {sidecar_input_count}")
        print(f"    Sidecar output_count:      {sidecar_output_count}")
        print(f"    Sidecar input_n dups:      {sidecar_input_n_dups} ({sidecar_input_n_dup_pct}%)")
        print(f"    Sidecar input_n jumps:     {sidecar_input_n_jumps} ({sidecar_input_n_jump_pct}%)")

        # Rates
        encoder = get_encoder_tag(seg_path)
        stream_info = extract_stts(seg_path)
        r_frame_rate = stream_info.get("r_frame_rate", "?")
        avg_frame_rate = stream_info.get("avg_frame_rate", "?")

        # True capture fps from showinfo PTS deltas (C5: null for arrival-PTS)
        true_capture_fps = None
        input_pts_stdev_ms = None
        if not is_arrival and seg_key in seg_showinfo and len(seg_showinfo[seg_key]) > 1:
            # Parse PTS from this segment's showinfo lines only
            showinfo_re_pts = re.compile(r"pts_time:\s*([0-9.eE+-]+)")
            pts_vals = []
            for line in seg_showinfo[seg_key]:
                m = showinfo_re_pts.search(line)
                if m:
                    pts_vals.append(float(m.group(1)))
            if len(pts_vals) > 1:
                deltas_ms = [
                    (pts_vals[j] - pts_vals[j - 1]) * 1000.0
                    for j in range(1, len(pts_vals))
                ]
                arr = np.array(deltas_ms)
                median_delta_s = float(np.median(arr)) / 1000.0
                if median_delta_s > 0:
                    true_capture_fps = round(1.0 / median_delta_s, 4)
                input_pts_stdev_ms = round(float(np.std(arr)), 4)

        # Output PTS stdev
        output_pts_stdev_ms = None
        df_pts = extract_pts_ffprobe(seg_path)
        if not df_pts.empty:
            df_pts = compute_deltas(df_pts, "pts_time_s")
            out_valid = df_pts["delta_ms"].dropna()
            if not out_valid.empty:
                output_pts_stdev_ms = round(float(out_valid.std()), 4)

        print(f"    Encoder:                   {encoder}")
        print(f"    r_frame_rate:              {r_frame_rate}")
        print(f"    avg_frame_rate:            {avg_frame_rate}")
        print(f"    cv2 CAP_PROP_FPS:          {cv2_fps}")
        print(f"    True capture fps:          {true_capture_fps}" +
              (" (N/A for arrival-PTS)" if is_arrival else ""))
        print(f"    Sidecar measured_fps:      {sidecar_measured_fps}")
        print(f"    Input PTS stdev (ms):      {input_pts_stdev_ms}")
        print(f"    Output PTS stdev (ms):     {output_pts_stdev_ms}")

        # C6: Per-segment frame ledger
        ledger_residual = None
        if si_count > 0 and nb_read > 0:
            ledger_residual = si_count - (nb_read - fh_dups)
        print(f"    Ledger: showinfo - (nb_read - fh_dups) = "
              f"{si_count} - ({nb_read} - {fh_dups}) = {ledger_residual}")
        print()

        row = {
            "segment": seg_name,
            "camera": cam_id,
            "date": seg_name.split("-")[1][:8] if "-" in seg_name else "unknown",
            "recorder_mode": recorder_mode,
            "encoder": encoder,
            "nb_frames": nb_frames,
            "nb_read_frames": nb_read,
            "cv2_iterated_count": cv2_count,
            "framehash_total": fh_total,
            "framehash_adjacent_dups": fh_dups,
            "framehash_dup_pct": fh_pct,
            "showinfo_count": si_count,
            "sidecar_input_count": sidecar_input_count,
            "sidecar_output_count": sidecar_output_count,
            "sidecar_input_n_dups": sidecar_input_n_dups,
            "sidecar_input_n_dup_pct": sidecar_input_n_dup_pct,
            "sidecar_input_n_jumps": sidecar_input_n_jumps,
            "sidecar_input_n_jump_pct": sidecar_input_n_jump_pct,
            "true_capture_fps": true_capture_fps,
            "sidecar_measured_fps": sidecar_measured_fps,
            "container_r_frame_rate": r_frame_rate,
            "container_avg_frame_rate": avg_frame_rate,
            "cv2_cap_prop_fps": cv2_fps,
            "input_pts_stdev_ms": input_pts_stdev_ms,
            "output_pts_stdev_ms": output_pts_stdev_ms,
            "ledger_residual": ledger_residual,
        }
        results.append(row)

    # --- Cross-Segment Showinfo Reconciliation ---
    print("=" * 70)
    print("CROSS-SEGMENT SHOWINFO RECONCILIATION")
    print("=" * 70)
    print()

    # Group segments by stderr file for multi-segment checks
    stderr_groups: dict[str, list[dict]] = {}
    for row, stderr_path in zip(results, stderr_files):
        key = str(stderr_path)
        stderr_groups.setdefault(key, []).append(row)

    reconciliation = []
    for stderr_key, group in stderr_groups.items():
        sp = Path(stderr_key)
        with open(sp) as f:
            total_showinfo = sum(1 for l in f if "Parsed_showinfo" in l and "pts_time:" in l)
        sum_per_seg = sum(r["showinfo_count"] for r in group)

        # Leading-edge loss: showinfo lines before first Opening marker
        leading_edge_loss = total_showinfo - sum_per_seg

        print(f"  stderr: {sp.name}")
        print(f"    total showinfo in file:    {total_showinfo}")
        print(f"    sum of per-segment counts: {sum_per_seg}")
        print(f"    leading-edge loss:         {leading_edge_loss}")
        print(f"    (This check can only detect lines lost before the first Opening marker;")
        print(f"     boundary misattribution between segments is invisible to this sum.)")
        print()

        reconciliation.append({
            "stderr_file": sp.name,
            "total_showinfo_lines": total_showinfo,
            "sum_per_segment_showinfo": sum_per_seg,
            "leading_edge_loss": leading_edge_loss,
        })

    # --- Save results ---
    output = {
        "per_segment": results,
        "cross_segment_reconciliation": reconciliation,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/results.json")
    print(f"\nWrite findings.md manually from these results.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RECORDER-TIMING-1: Analyze per-frame timing from recorder captures")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze timing of a single clip")
    p_analyze.add_argument("--clip", required=True, help="Path to mp4 clip")
    p_analyze.add_argument("--mode", required=True,
                           choices=["vfr", "cfr", "cfr-sidecar"],
                           help="Recording mode used")
    p_analyze.add_argument("--stderr", help="Path to ffmpeg.stderr (for cfr-sidecar)")
    p_analyze.add_argument("--output-dir",
                           default="docs/evidence/recorder_timing_1/",
                           help="Output directory")

    # compare
    p_compare = sub.add_parser("compare", help="Compare VFR vs CFR frame counts")
    p_compare.add_argument("--vfr", required=True, help="VFR clip path")
    p_compare.add_argument("--cfr", required=True, help="CFR clip path")
    p_compare.add_argument("--output-dir",
                           default="docs/evidence/recorder_timing_1/",
                           help="Output directory")

    # dupfix
    p_dupfix = sub.add_parser("dupfix",
        help="RECORDER-DUPFIX-1: Resolve duplicate-frame contradiction")
    p_dupfix.add_argument("--segments", nargs="+", required=True,
                          help="Paths to mp4 segments to analyze")
    p_dupfix.add_argument("--stderr-files", nargs="+", required=True,
                          help="Paths to ffmpeg stderr files (1:1 with segments)")
    p_dupfix.add_argument("--output-dir",
                          default="docs/evidence/recorder_dupfix_1/",
                          help="Output directory")

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "dupfix":
        cmd_dupfix(args)


if __name__ == "__main__":
    main()
