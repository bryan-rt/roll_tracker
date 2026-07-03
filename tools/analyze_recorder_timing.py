#!/usr/bin/env python3
"""RECORDER-TIMING-1: Analyze per-frame timing from recorder test captures.

Subcommands:
  analyze   — Extract per-frame timing from a clip, flag anomalies, check Stage A compat
  compare   — Compare frame counts between VFR and CFR clips

Usage:
  python tools/analyze_recorder_timing.py analyze \
    --clip <path.mp4> --mode {vfr,cfr,cfr-sidecar} \
    [--stderr <ffmpeg.stderr>] [--output-dir <dir>]

  python tools/analyze_recorder_timing.py compare \
    --vfr <vfr.mp4> --cfr <cfr.mp4> [--output-dir <dir>]
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

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
