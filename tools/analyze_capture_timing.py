#!/usr/bin/env python3
"""CAPTURE-TIME-2: Analyze multi-camera timing diagnostic session.

Usage:
  python tools/analyze_capture_timing.py analyze-session <session_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_sidecar(path: Path) -> tuple[dict, list[dict]]:
    """Load a .timing.jsonl sidecar. Returns (meta, rows)."""
    with open(path) as f:
        meta = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    return meta, rows


def analyze_session(session_dir: Path) -> None:
    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.exists():
        print(f"No session_manifest.json in {session_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"=== Session {manifest['session_ts']} ===")
    print(f"Cameras: {len(manifest['cameras'])}\n")

    offsets = {}
    fps_values = {}
    drift_rates = {}

    for cam in manifest["cameras"]:
        label = cam["label"]
        cam_dir = session_dir / label
        timing = cam.get("timing", {})

        print(f"--- {label} (cam_id={cam['cam_id']}) ---")

        # Timing anchors
        if timing:
            offset = timing.get("pts_wallclock_offset_s", 0)
            drift = timing.get("drift_rate_s_per_s", 0)
            drift_ppm = timing.get("drift_ppm", 0)
            drift_flat = timing.get("drift_flat", True)
            mfps = timing.get("measured_fps", 0)
            stdev = timing.get("pts_stdev_ms", 0)
            inp = timing.get("input_frames", 0)
            out = timing.get("output_frames", 0)
            mismatch = timing.get("mismatch", False)

            offsets[label] = offset
            fps_values[label] = mfps
            drift_rates[label] = drift

            print(f"  Measured FPS:         {mfps:.4f}")
            print(f"  PTS stdev:            {stdev:.4f} ms")
            print(f"  Frames (in/out):      {inp}/{out} {'⚠ MISMATCH' if mismatch else '✓'}")
            print(f"  PTS→wallclock offset: {offset:.6f}s")
            print(f"  Drift rate:           {drift_ppm:.3f} ppm {'(FLAT ✓)' if drift_flat else '(TRENDING ⚠)'}")
            if not drift_flat:
                print(f"    → {abs(drift)*1000:.4f} ms/s drift — alignment degrades by "
                      f"{abs(drift)*300*1000:.1f}ms over 5 min")

            # Windowed offsets
            wins = timing.get("windowed_offsets", [])
            if wins:
                win_offsets = [w["offset"] for w in wins]
                print(f"  Windowed offsets ({len(wins)} windows):")
                print(f"    range: {min(win_offsets):.6f} – {max(win_offsets):.6f}s")
                print(f"    spread: {(max(win_offsets) - min(win_offsets))*1000:.2f}ms")
        else:
            print("  No timing anchors")

        # RTCP
        rtcp = cam.get("rtcp", {})
        tcp_r = rtcp.get("tcp", "?")
        udp_r = rtcp.get("udp", "?")
        print(f"  RTCP TCP:  {tcp_r} mentions")
        print(f"  RTCP UDP:  {udp_r}")

        # Load sidecar if available
        sidecars = list(cam_dir.glob("*.timing.jsonl")) if cam_dir.exists() else []
        if sidecars:
            meta, rows = load_sidecar(sidecars[0])
            pts = np.array([r["pts_time_s"] for r in rows])
            deltas = np.diff(pts) * 1000
            med = np.median(deltas)
            # Uniformity check: >90% of deltas within ±15% of median
            lo, hi = med * 0.85, med * 1.15
            pct_uniform = 100 * np.sum((deltas > lo) & (deltas < hi)) / len(deltas) if len(deltas) > 0 else 0
            pct_burst = 100 * np.sum(deltas < 1) / len(deltas) if len(deltas) > 0 else 0
            nominal_fps = round(1000 / med) if med > 0 else 0
            print(f"  PTS deltas:           {med:.2f}ms median (~{nominal_fps}fps), "
                  f"{pct_uniform:.1f}% uniform, {pct_burst:.1f}% burst (<1ms)")
            source_pts_ok = pct_uniform > 80
            print(f"  Source PTS uniform:   {'YES ✓' if source_pts_ok else 'NO ⚠ (still bursty — source PTS not working)'}")

        print()

    # Cross-camera analysis
    if len(offsets) >= 2:
        print("=" * 60)
        print("CROSS-CAMERA ALIGNMENT (Tier-2: source PTS + shared host clock)")
        print("=" * 60)

        labels = sorted(offsets.keys())
        offset_vals = [offsets[l] for l in labels]
        offset_spread = max(offset_vals) - min(offset_vals)

        print(f"\nPTS→wallclock offsets:")
        for l in labels:
            print(f"  {l}: {offsets[l]:.6f}s")

        print(f"\nOffset spread (max - min): {offset_spread*1000:.2f}ms")
        print(f"  NOTE: offset spread includes stream-start stagger (cameras started at")
        print(f"  different times). This is NOT alignment uncertainty — it's expected.")
        print(f"  Alignment works by mapping each camera's source PTS to absolute time")
        print(f"  via its own offset. Per-camera offset uncertainty (windowed spread)")
        print(f"  bounds the achievable alignment precision per camera.")

        # Measured FPS comparison
        print(f"\nMeasured FPS per camera:")
        for l in labels:
            print(f"  {l}: {fps_values[l]:.4f} fps")
        fps_spread = max(fps_values.values()) - min(fps_values.values())
        print(f"  FPS spread: {fps_spread:.4f} fps")
        if fps_spread > 0.01:
            # Time to drift by 1 frame at this rate difference
            drift_1frame = 1.0 / (fps_spread * 30) if fps_spread > 0 else float("inf")
            print(f"  → cameras drift by 1 frame every {drift_1frame:.0f}s")
            print(f"  → over 5 min: {fps_spread * 300:.2f} frames of drift")
        else:
            print(f"  → negligible cross-camera fps drift")

        # Drift rates
        print(f"\nClock drift rates:")
        for l in labels:
            print(f"  {l}: {drift_rates[l]*1e6:.3f} ppm")

    # RTCP verdict
    print(f"\n{'=' * 60}")
    print("RTCP VERDICT")
    print("=" * 60)
    any_rtcp = False
    for cam in manifest["cameras"]:
        rtcp = cam.get("rtcp", {})
        tcp_v = rtcp.get("tcp", "0")
        udp_v = rtcp.get("udp", "0")
        try:
            tcp_n = int(tcp_v)
        except (ValueError, TypeError):
            tcp_n = 0
        if tcp_n > 0:
            any_rtcp = True
            print(f"  {cam['label']}: RTCP found over TCP ({tcp_n} mentions)")
        if udp_v == "connection_refused":
            print(f"  {cam['label']}: UDP transport refused by relay (RTCP-over-separate-port unavailable)")
        elif isinstance(udp_v, str) and udp_v.startswith("connected"):
            n = udp_v.split("_")[1] if "_" in udp_v else "0"
            if int(n) > 0:
                any_rtcp = True
                print(f"  {cam['label']}: RTCP found over UDP ({n} mentions)")

    if not any_rtcp:
        print("  No RTCP sender reports found under any transport.")
        print("  Absolute camera clock NOT available from stream.")
        print("  Cross-camera sync relies on Tier-2: source PTS + host-clock anchors.")


def main():
    parser = argparse.ArgumentParser(description="CAPTURE-TIME-2: Analyze timing session")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze-session", help="Analyze a timing diagnostic session")
    p.add_argument("session_dir", type=Path, help="Path to timing session directory")

    args = parser.parse_args()
    if args.command == "analyze-session":
        analyze_session(args.session_dir)


if __name__ == "__main__":
    main()
