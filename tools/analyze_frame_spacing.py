#!/usr/bin/env python3
"""CP-R11: Definitive frame-spacing characterization.

Pure sidecar analysis — no video decoding. Reads all source-PTS .timing.jsonl
files and produces per-segment, per-attempt, and per-camera frame-spacing
statistics with run-length structure, gap periodicity, and grid-vs-effective
rate correlation.

Usage:
  python tools/analyze_frame_spacing.py \
    --recordings data/raw/nest/00000000-0000-0000-0000-000000000003 \
    --output-dir docs/evidence/frame_spacing_1
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sidecar reading
# ---------------------------------------------------------------------------

def read_sidecar(path: Path) -> tuple[dict, list[float]]:
    """Read a timing sidecar. Returns (meta, list_of_pts_time_s)."""
    meta = {}
    pts_values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_meta"):
                meta = row
                continue
            pts = row.get("pts_time_s")
            if pts is not None:
                pts_values.append(pts)
    return meta, pts_values


def pts_to_ticks(pts_values: list[float], timebase: int) -> list[int]:
    """Convert pts_time_s values to integer ticks and compute deltas."""
    ticks = [round(p * timebase) for p in pts_values]
    deltas = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    return deltas


# ---------------------------------------------------------------------------
# Histogram and mode detection
# ---------------------------------------------------------------------------

def tick_histogram(deltas: list[int]) -> dict[int, int]:
    """Count occurrences of each distinct tick delta."""
    return dict(collections.Counter(deltas))


def cluster_ticks(hist: dict[int, int]) -> list[tuple[int, int]]:
    """Cluster nearby tick values (handling 2970/3060 and 5940/6030 alternation).

    Merges ticks within 15% of each other into clusters. Returns sorted list of
    (weighted_centre, total_count) from lowest to highest centre.
    """
    if not hist:
        return []

    sorted_ticks = sorted(hist.keys())
    clusters: list[list[tuple[int, int]]] = []  # list of [(tick, count), ...]

    for tick in sorted_ticks:
        count = hist[tick]
        merged = False
        for cluster in clusters:
            centre = sum(t * c for t, c in cluster) / sum(c for _, c in cluster)
            if abs(tick - centre) <= centre * 0.15:
                cluster.append((tick, count))
                merged = True
                break
        if not merged:
            clusters.append([(tick, count)])

    result = []
    for cluster in clusters:
        total_count = sum(c for _, c in cluster)
        weighted_centre = round(sum(t * c for t, c in cluster) / total_count)
        result.append((weighted_centre, total_count))

    return sorted(result, key=lambda x: x[0])


def derive_mode_centres(hist: dict[int, int], total: int) -> tuple[int | None, int | None, bool]:
    """Derive fast and slow mode centres from histogram.

    Returns (fast_centre, slow_centre, is_2to1).
    'fast' = ~3000 ticks (~33ms, 30fps cadence)
    'slow' = ~6000 ticks (~67ms, 15fps cadence)

    Logic:
    1. Find the primary cluster (most intervals).
    2. Check if any other cluster is at ~0.5x the primary (genuine second cadence mode).
    3. Clusters at ~2x the primary are GAPS, not a second mode.
    4. Only report fast_centre when a true half-rate cluster exists.
    """
    clusters = cluster_ticks(hist)
    if not clusters:
        return None, None, False

    # Filter to material clusters (>0.5% of total)
    material = [(centre, count) for centre, count in clusters if count > total * 0.005]
    if not material:
        material = clusters[:1]

    # Primary = most frequent cluster
    material.sort(key=lambda x: -x[1])
    primary_centre, primary_count = material[0]

    # Look for a half-rate cluster (~0.5x primary = fast mode)
    half_centre = None
    for centre, count in material[1:]:
        ratio = centre / primary_centre if primary_centre else 0
        if 0.4 < ratio < 0.6:  # roughly half
            half_centre = centre
            break

    if half_centre is not None:
        # Bimodal: half_centre is fast, primary is slow
        fast = half_centre
        slow = primary_centre
        is_2to1 = True
        return fast, slow, is_2to1

    # No half-rate cluster. Check for a double-rate cluster (~2x primary) that
    # might be a genuine second cadence rather than gaps.
    # Distinction: genuine second cadence has >15% of intervals; gaps are <10%.
    double_centre = None
    double_count = 0
    for centre, count in material[1:]:
        ratio = centre / primary_centre if primary_centre else 0
        if 1.7 < ratio < 2.3:
            double_centre = centre
            double_count = count
            break

    if double_centre is not None and double_count > total * 0.15:
        # Genuine bimodal: primary is fast, double is slow
        if primary_centre < double_centre:
            return primary_centre, double_centre, True
        else:
            return double_centre, primary_centre, True

    # Single cadence mode — check if primary is fast (~3000) or slow (~6000)
    if primary_centre < 4500:
        return primary_centre, None, False
    else:
        return None, primary_centre, False


# ---------------------------------------------------------------------------
# Interval classification
# ---------------------------------------------------------------------------

def classify_intervals(
    deltas: list[int],
    fast_centre: int | None,
    slow_centre: int | None,
) -> list[str]:
    """Classify each tick delta as fast/slow/gap/other.

    - fast: ~3000 ticks (~33ms, 30fps cadence)
    - slow: ~6000 ticks (~67ms, 15fps cadence)
    - gap: exceeds 1.5x the LOCAL cadence (whichever mode was most recently active)
    - other: none of the above

    Gap detection uses the preceding run's cadence, not the segment median.
    This prevents the C1 bug where the median-based threshold misclassifies
    normal slow intervals as gaps on majority-fast segments.
    """
    def in_fast(t: int) -> bool:
        if fast_centre is None:
            return False
        return abs(t - fast_centre) <= fast_centre * 0.15

    def in_slow(t: int) -> bool:
        if slow_centre is None:
            return False
        return abs(t - slow_centre) <= slow_centre * 0.15

    classes = []
    # Start with the higher cadence as the local baseline (conservative for gap detection)
    current_mode_tick = slow_centre or fast_centre or 6000

    for t in deltas:
        if in_fast(t):
            classes.append("fast")
            current_mode_tick = fast_centre
        elif in_slow(t):
            classes.append("slow")
            current_mode_tick = slow_centre
        elif t > 1.5 * current_mode_tick:
            classes.append("gap")
        else:
            classes.append("other")

    return classes


# ---------------------------------------------------------------------------
# Run-length encoding
# ---------------------------------------------------------------------------

def run_length_encode(classes: list[str]) -> list[tuple[str, int]]:
    """RLE of classification sequence. Returns [(class, length), ...]."""
    if not classes:
        return []
    runs = []
    current = classes[0]
    length = 1
    for c in classes[1:]:
        if c == current:
            length += 1
        else:
            runs.append((current, length))
            current = c
            length = 1
    runs.append((current, length))
    return runs


def run_stats(runs: list[tuple[str, int]], cls: str) -> dict:
    """Compute run-length statistics for a given class."""
    lengths = [l for c, l in runs if c == cls]
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "max": 0, "min": 0}
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "max": max(lengths),
        "min": min(lengths),
    }


def boundary_truncation_fraction(
    runs: list[tuple[str, int]],
    total_intervals: int,
) -> float:
    """Fraction of runs that terminate at a segment boundary."""
    if len(runs) <= 2:
        return 1.0  # first and last runs always touch a boundary
    # First and last runs are boundary-touching by definition
    boundary_runs = 2
    return boundary_runs / len(runs) if runs else 0


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def gap_spacing(classes: list[str]) -> list[int]:
    """Return list of inter-gap spacings (frames between consecutive gaps)."""
    gap_positions = [i for i, c in enumerate(classes) if c == "gap"]
    if len(gap_positions) < 2:
        return []
    return [gap_positions[i] - gap_positions[i - 1] for i in range(1, len(gap_positions))]


def gap_spacing_histogram(spacings: list[int]) -> dict[int, int]:
    """Histogram of inter-gap spacings."""
    return dict(collections.Counter(spacings))


# ---------------------------------------------------------------------------
# Grid-vs-effective rate (A1 decisive test)
# ---------------------------------------------------------------------------

def grid_effective_analysis(
    deltas: list[int],
    classes: list[str],
    fast_centre: int | None,
    slow_centre: int | None,
    timebase: int,
) -> dict:
    """Compute grid rate, effective rate, predicted and observed skip period."""
    n_frames = len(deltas) + 1
    total_ticks = sum(deltas)
    total_span_s = total_ticks / timebase

    # Modal tick = most common non-gap interval
    non_gap_deltas = [d for d, c in zip(deltas, classes) if c != "gap"]
    if not non_gap_deltas:
        return {}

    modal_tick = statistics.mode(non_gap_deltas) if non_gap_deltas else (slow_centre or fast_centre or 6000)

    grid_rate = timebase / modal_tick
    effective_rate = (n_frames - 1) / total_span_s if total_span_s > 0 else 0

    n_gaps = sum(1 for c in classes if c == "gap")
    spacings = gap_spacing(classes)
    observed_skip = statistics.mode(spacings) if spacings else None

    predicted_skip = None
    ratio = grid_rate / effective_rate if effective_rate > 0 else 1
    if ratio > 1.001:  # measurable shortfall
        predicted_skip = round(1 / (ratio - 1), 2)

    return {
        "modal_tick": modal_tick,
        "grid_rate": round(grid_rate, 4),
        "effective_rate": round(effective_rate, 4),
        "rate_ratio": round(ratio, 6),
        "predicted_skip": predicted_skip,
        "observed_skip": observed_skip,
        "n_gaps": n_gaps,
        "n_frames": n_frames,
        "segment_length": n_frames - 1,  # in intervals
    }


# ---------------------------------------------------------------------------
# Sliding-window fast-mode proportion (A4)
# ---------------------------------------------------------------------------

def sliding_window_proportion(
    classes: list[str],
    window: int = 100,
    stride: int = 1,
) -> list[tuple[int, float]]:
    """Sliding-window fast-mode fraction. Returns [(centre_index, fraction), ...]."""
    results = []
    for start in range(0, len(classes) - window + 1, stride):
        chunk = classes[start : start + window]
        n_fast = sum(1 for c in chunk if c == "fast")
        frac = n_fast / window
        centre = start + window // 2
        results.append((centre, round(frac, 4)))
    return results


# ---------------------------------------------------------------------------
# Compact run-length sequence
# ---------------------------------------------------------------------------

def compact_rle_string(runs: list[tuple[str, int]], max_runs: int = 30) -> str:
    """Compact string like 'L370 S34 L12 S209 ...'"""
    abbrev = {"fast": "F", "slow": "S", "gap": "G", "other": "O"}
    parts = [f"{abbrev.get(c, '?')}{l}" for c, l in runs[:max_runs]]
    suffix = f" ...+{len(runs) - max_runs}" if len(runs) > max_runs else ""
    return " ".join(parts) + suffix


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

def discover_segments(recordings_root: Path) -> list[dict]:
    """Find all passthrough .timing.jsonl sidecars."""
    segments = []
    for cam in ["FP7oJQ", "PPDmUg"]:
        cam_dir = recordings_root / cam
        if not cam_dir.exists():
            continue
        for sidecar_path in sorted(cam_dir.rglob("*.timing.jsonl")):
            meta, pts_values = read_sidecar(sidecar_path)
            if meta.get("timing_mode") != "passthrough":
                continue
            stem = sidecar_path.stem.replace(".timing", "")
            parts = stem.split("-")
            camera = parts[0] if parts else cam
            date_str = parts[1] if len(parts) > 1 else ""
            time_str = parts[2] if len(parts) > 2 else ""

            segments.append({
                "path": sidecar_path,
                "camera": camera,
                "date": date_str,
                "time": time_str,
                "segment_id": f"{camera}-{date_str}-{time_str}",
                "meta": meta,
                "pts_values": pts_values,
            })
    return segments


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_segment(seg: dict) -> dict:
    """Run all measurements on a single segment."""
    meta = seg["meta"]
    pts_values = seg["pts_values"]
    timebase = meta.get("pts_timebase", 90000)
    schema = meta.get("sidecar_schema", 1)

    deltas = pts_to_ticks(pts_values, timebase)
    n_frames = len(pts_values)
    n_intervals = len(deltas)

    if n_intervals == 0:
        return {
            "segment_id": seg["segment_id"], "camera": seg["camera"],
            "date": seg["date"], "time": seg["time"], "schema": schema,
            "attempt": meta.get("attempt", ""), "n_frames": n_frames,
            "n_intervals": 0, "total_span_s": 0,
            "fast_centre": None, "slow_centre": None, "is_2to1": False,
            "n_fast": 0, "n_slow": 0, "n_gap": 0, "n_other": 0,
            "fast_frac": 0, "fast_runs_count": 0, "fast_runs_mean": 0,
            "fast_runs_median": 0, "fast_runs_max": 0,
            "slow_runs_count": 0, "slow_runs_mean": 0, "slow_runs_median": 0,
            "slow_runs_max": 0, "n_mode_switches": 0, "boundary_trunc_frac": 0,
            "rle_compact": "", "gap_total": 0, "gap_spacing_mode": None,
            "gap_spacing_mean": None, "gap_spacing_stdev": None,
            "gap_spacing_hist_top3": "", "longest_gap_free_frames": 0,
            "longest_gap_free_s": 0, "top_ticks": "",
            "modal_tick": None, "grid_rate": None, "effective_rate": None,
            "rate_ratio": None, "predicted_skip": None, "observed_skip": None,
            "v4_is_bimodal": meta.get("is_bimodal"),
            "v4_nominal_dt_s": meta.get("nominal_dt_s"),
            "v4_short_mode_fraction": meta.get("short_mode_fraction"),
        }

    # Tick histogram
    hist = tick_histogram(deltas)
    fast_centre, slow_centre, is_2to1 = derive_mode_centres(hist, n_intervals)

    # Classification
    classes = classify_intervals(deltas, fast_centre, slow_centre)
    n_fast = sum(1 for c in classes if c == "fast")
    n_slow = sum(1 for c in classes if c == "slow")
    n_gap = sum(1 for c in classes if c == "gap")
    n_other = sum(1 for c in classes if c == "other")

    # Run-length structure
    runs = run_length_encode(classes)
    fast_runs = run_stats(runs, "fast")
    slow_runs = run_stats(runs, "slow")
    n_mode_switches = sum(1 for i in range(1, len(runs))
                         if runs[i][0] in ("fast", "slow") and runs[i-1][0] in ("fast", "slow")
                         and runs[i][0] != runs[i-1][0])

    # Boundary truncation
    boundary_frac = boundary_truncation_fraction(runs, n_intervals)

    # Gap analysis
    spacings = gap_spacing(classes)

    # Grid-vs-effective
    grid_eff = grid_effective_analysis(deltas, classes, fast_centre, slow_centre, timebase)

    # Compact RLE (always emit for segments with gaps too, not just mode switches)
    rle_str = compact_rle_string(runs) if (n_mode_switches >= 1 or n_gap >= 1) else ""

    # Longest gap-free run (fast or slow, contiguous)
    longest_gap_free = 0
    current_gap_free = 0
    for c in classes:
        if c in ("fast", "slow", "other"):
            current_gap_free += 1
            longest_gap_free = max(longest_gap_free, current_gap_free)
        else:
            current_gap_free = 0

    # Determine dominant cadence
    if n_fast > n_slow:
        dominant_tick = fast_centre
    elif n_slow > 0:
        dominant_tick = slow_centre
    else:
        dominant_tick = None

    # Top tick values for the histogram report
    top_ticks = sorted(hist.items(), key=lambda x: -x[1])
    top_ticks_str = "; ".join(f"{t}:{c}" for t, c in top_ticks[:6])

    # Fast-mode fraction (fraction of intervals at ~30fps cadence)
    fast_frac = n_fast / n_intervals if n_intervals else 0

    total_ticks = sum(deltas)
    total_span_s = total_ticks / timebase

    return {
        "segment_id": seg["segment_id"],
        "camera": seg["camera"],
        "date": seg["date"],
        "time": seg["time"],
        "schema": schema,
        "attempt": meta.get("attempt", ""),
        "n_frames": n_frames,
        "n_intervals": n_intervals,
        "total_span_s": round(total_span_s, 3),
        "fast_centre": fast_centre,
        "slow_centre": slow_centre,
        "is_2to1": is_2to1,
        "n_fast": n_fast,
        "n_slow": n_slow,
        "n_gap": n_gap,
        "n_other": n_other,
        "fast_frac": round(fast_frac, 4),
        "fast_runs_count": fast_runs["count"],
        "fast_runs_mean": fast_runs["mean"],
        "fast_runs_median": fast_runs["median"],
        "fast_runs_max": fast_runs["max"],
        "slow_runs_count": slow_runs["count"],
        "slow_runs_mean": slow_runs["mean"],
        "slow_runs_median": slow_runs["median"],
        "slow_runs_max": slow_runs["max"],
        "n_mode_switches": n_mode_switches,
        "boundary_trunc_frac": round(boundary_frac, 4),
        "rle_compact": rle_str,
        "gap_total": n_gap,
        "gap_spacing_mode": statistics.mode(spacings) if spacings else None,
        "gap_spacing_mean": round(statistics.mean(spacings), 2) if spacings else None,
        "gap_spacing_stdev": round(statistics.stdev(spacings), 2) if len(spacings) > 1 else None,
        "gap_spacing_hist_top3": "; ".join(f"{s}:{c}" for s, c in
                                           sorted(gap_spacing_histogram(spacings).items(),
                                                  key=lambda x: -x[1])[:3]) if spacings else "",
        "longest_gap_free_frames": longest_gap_free,
        "longest_gap_free_s": round(longest_gap_free * (dominant_tick / timebase if dominant_tick else 0.067), 2),
        "top_ticks": top_ticks_str,
        # Grid-vs-effective
        "modal_tick": grid_eff.get("modal_tick"),
        "grid_rate": grid_eff.get("grid_rate"),
        "effective_rate": grid_eff.get("effective_rate"),
        "rate_ratio": grid_eff.get("rate_ratio"),
        "predicted_skip": grid_eff.get("predicted_skip"),
        "observed_skip": grid_eff.get("observed_skip"),
        # V4 contract fields
        "v4_is_bimodal": meta.get("is_bimodal"),
        "v4_nominal_dt_s": meta.get("nominal_dt_s"),
        "v4_short_mode_fraction": meta.get("short_mode_fraction"),
    }


def analyze_attempt(attempt_segments: list[dict], attempt_key: tuple) -> dict:
    """Concatenate segments within an attempt, recompute run-length stats."""
    camera, date, attempt_num = attempt_key
    all_deltas = []
    all_pts = []
    timebase = 90000

    for seg in attempt_segments:
        meta = seg["meta"]
        timebase = meta.get("pts_timebase", 90000)
        pts_values = seg["pts_values"]
        deltas = pts_to_ticks(pts_values, timebase)
        if all_pts:
            # Cross-segment gap: compute delta from last pts of previous to first pts of current
            # We use segment_start_epoch difference as the real elapsed time
            prev_meta = attempt_segments[attempt_segments.index(seg) - 1]["meta"]
            prev_pts = attempt_segments[attempt_segments.index(seg) - 1]["pts_values"]
            if prev_pts and pts_values:
                prev_epoch = prev_meta.get("segment_start_epoch", 0)
                curr_epoch = meta.get("segment_start_epoch", 0)
                prev_span = prev_pts[-1]
                curr_start = pts_values[0]  # should be ~0
                # Real gap in ticks
                gap_s = (curr_epoch - prev_epoch) - prev_span + curr_start
                gap_ticks = round(gap_s * timebase)
                if gap_ticks > 0:
                    all_deltas.append(gap_ticks)

        all_deltas.extend(deltas)
        all_pts.extend(pts_values)

    n_intervals = len(all_deltas)
    if n_intervals == 0:
        return {
            "camera": camera, "date": date, "attempt": attempt_num,
            "n_segments": len(attempt_segments), "n_intervals": 0,
        }

    hist = tick_histogram(all_deltas)
    fast_centre, slow_centre, is_2to1 = derive_mode_centres(hist, n_intervals)
    classes = classify_intervals(all_deltas, fast_centre, slow_centre)
    runs = run_length_encode(classes)

    n_fast = sum(1 for c in classes if c == "fast")
    n_slow = sum(1 for c in classes if c == "slow")
    n_gap = sum(1 for c in classes if c == "gap")

    fast_runs = run_stats(runs, "fast")
    slow_runs = run_stats(runs, "slow")
    n_mode_switches = sum(1 for i in range(1, len(runs))
                         if runs[i][0] in ("fast", "slow") and runs[i-1][0] in ("fast", "slow")
                         and runs[i][0] != runs[i-1][0])

    # Count segment-boundary-truncated runs
    seg_boundary_indices = []
    idx = 0
    for seg in attempt_segments[:-1]:
        seg_meta = seg["meta"]
        seg_pts = seg["pts_values"]
        seg_deltas = pts_to_ticks(seg_pts, seg_meta.get("pts_timebase", 90000))
        idx += len(seg_deltas)
        idx += 1  # cross-segment gap interval
        seg_boundary_indices.append(idx)

    # Count runs that span a segment boundary
    boundary_split_count = 0
    run_start = 0
    for cls_val, length in runs:
        run_end = run_start + length
        for bi in seg_boundary_indices:
            if run_start < bi < run_end:
                boundary_split_count += 1
                break
        run_start = run_end

    fast_frac = n_fast / n_intervals if n_intervals else 0

    # Longest gap-free run
    longest_gap_free = 0
    current_gap_free = 0
    for c in classes:
        if c in ("fast", "slow", "other"):
            current_gap_free += 1
            longest_gap_free = max(longest_gap_free, current_gap_free)
        else:
            current_gap_free = 0

    return {
        "camera": camera,
        "date": date,
        "attempt": attempt_num,
        "n_segments": len(attempt_segments),
        "n_intervals": n_intervals,
        "fast_centre": fast_centre,
        "slow_centre": slow_centre,
        "n_fast": n_fast,
        "n_slow": n_slow,
        "n_gap": n_gap,
        "fast_frac": round(fast_frac, 4),
        "fast_runs_count": fast_runs["count"],
        "fast_runs_mean": fast_runs["mean"],
        "fast_runs_max": fast_runs["max"],
        "slow_runs_count": slow_runs["count"],
        "slow_runs_mean": slow_runs["mean"],
        "slow_runs_max": slow_runs["max"],
        "n_mode_switches": n_mode_switches,
        "boundary_split_runs": boundary_split_count,
        "longest_gap_free_frames": longest_gap_free,
        "rle_compact": compact_rle_string(runs, max_runs=40),
    }


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    """Write list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CP-R11: Frame-spacing characterization")
    parser.add_argument("--recordings", required=True, help="Root dir with camera subdirs")
    parser.add_argument("--output-dir", required=True, help="Output directory for findings")
    args = parser.parse_args()

    recordings_root = Path(args.recordings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering segments under {recordings_root} ...")
    segments = discover_segments(recordings_root)
    print(f"Found {len(segments)} passthrough segments")

    # --- Per-segment analysis ---
    print("Analyzing segments ...")
    seg_results = []
    seg_data = {}  # segment_id -> (deltas, classes, fast_centre, slow_centre)
    for seg in segments:
        result = analyze_segment(seg)
        seg_results.append(result)

        # Cache classified data for detailed analyses
        meta = seg["meta"]
        timebase = meta.get("pts_timebase", 90000)
        deltas = pts_to_ticks(seg["pts_values"], timebase)
        if deltas:
            hist = tick_histogram(deltas)
            sc, lc, _ = derive_mode_centres(hist, len(deltas))
            classes = classify_intervals(deltas, sc, lc)
            seg_data[seg["segment_id"]] = (deltas, classes, sc, lc)

    # Write segment summary
    if seg_results:
        write_csv(output_dir / "segment_summary.csv", seg_results)
        print(f"Wrote segment_summary.csv ({len(seg_results)} rows)")

    # --- Per-attempt analysis (A2) ---
    print("Analyzing attempts ...")
    attempts: dict[tuple, list[dict]] = collections.defaultdict(list)
    for seg in segments:
        key = (seg["camera"], seg["date"], seg["meta"].get("attempt", 0))
        attempts[key].append(seg)

    attempt_results = []
    for key in sorted(attempts.keys()):
        att_segs = sorted(attempts[key], key=lambda s: s["time"])
        result = analyze_attempt(att_segs, key)
        attempt_results.append(result)

    if attempt_results:
        write_csv(output_dir / "attempt_summary.csv", attempt_results)
        print(f"Wrote attempt_summary.csv ({len(attempt_results)} rows)")

    # --- Grid-vs-effective correlation (A1) ---
    print("Computing grid-vs-effective correlation ...")
    grid_rows = []
    for r in seg_results:
        if r["camera"] != "FP7oJQ":
            continue
        # Include ALL segments (C3)
        pred = r.get("predicted_skip")
        obs = r.get("observed_skip")
        n_gaps = r.get("gap_total", 0)
        seg_len = r.get("n_intervals", 0)

        category = "no_skip_predicted"
        if pred is not None:
            if pred > seg_len:
                category = "no_skip_predicted"
            else:
                category = "skip_predicted"

        grid_rows.append({
            "segment_id": r["segment_id"],
            "n_intervals": r["n_intervals"],
            "modal_tick": r.get("modal_tick"),
            "grid_rate": r.get("grid_rate"),
            "effective_rate": r.get("effective_rate"),
            "rate_ratio": r.get("rate_ratio"),
            "predicted_skip": pred,
            "observed_skip": obs,
            "n_gaps": n_gaps,
            "category": category,
        })

    # Also include PPDmUg for completeness (C3 — all segments)
    for r in seg_results:
        if r["camera"] != "PPDmUg":
            continue
        pred = r.get("predicted_skip")
        obs = r.get("observed_skip")
        n_gaps = r.get("gap_total", 0)
        seg_len = r.get("n_intervals", 0)

        category = "no_skip_predicted"
        if pred is not None and pred <= seg_len:
            category = "skip_predicted"

        grid_rows.append({
            "segment_id": r["segment_id"],
            "n_intervals": r["n_intervals"],
            "modal_tick": r.get("modal_tick"),
            "grid_rate": r.get("grid_rate"),
            "effective_rate": r.get("effective_rate"),
            "rate_ratio": r.get("rate_ratio"),
            "predicted_skip": pred,
            "observed_skip": obs,
            "n_gaps": n_gaps,
            "category": category,
        })

    if grid_rows:
        write_csv(output_dir / "grid_correlation.csv", grid_rows)
        print(f"Wrote grid_correlation.csv ({len(grid_rows)} rows)")

    # Compute correlation for FP7oJQ segments with both predicted and observed
    fp7_pairs = [(r["predicted_skip"], r["observed_skip"])
                 for r in grid_rows
                 if r["segment_id"].startswith("FP7oJQ")
                 and r["predicted_skip"] is not None
                 and r["observed_skip"] is not None]
    if len(fp7_pairs) >= 3:
        preds, obs = zip(*fp7_pairs)
        # Pearson correlation
        mean_p = statistics.mean(preds)
        mean_o = statistics.mean(obs)
        num = sum((p - mean_p) * (o - mean_o) for p, o in zip(preds, obs))
        den_p = math.sqrt(sum((p - mean_p) ** 2 for p in preds))
        den_o = math.sqrt(sum((o - mean_o) ** 2 for o in obs))
        corr = num / (den_p * den_o) if den_p * den_o > 0 else 0
        print(f"FP7oJQ predicted-vs-observed skip correlation: r={corr:.4f} (n={len(fp7_pairs)})")
    else:
        corr = None
        print(f"FP7oJQ: insufficient paired data for correlation (n={len(fp7_pairs)})")

    # --- A4: FP7oJQ-163102 sliding window reproduction ---
    detail_seg_id = "FP7oJQ-20260804-163102"
    if detail_seg_id in seg_data:
        print(f"A4: Reproducing sliding-window analysis on {detail_seg_id} ...")
        deltas, classes, sc, lc = seg_data[detail_seg_id]
        window_results = sliding_window_proportion(classes, window=100, stride=10)
        detail_rows = [{"centre_index": c, "fast_frac": f} for c, f in window_results]
        write_csv(output_dir / "fp7_163102_detail.csv", detail_rows)
        print(f"Wrote fp7_163102_detail.csv ({len(detail_rows)} rows)")

        # Also write the full per-interval classification for this segment
        rle = run_length_encode(classes)
        rle_str = compact_rle_string(rle, max_runs=100)
        print(f"  163102 RLE ({len(rle)} runs): {rle_str}")
        print(f"  163102 mode switches (fast<->slow): "
              f"{sum(1 for i in range(1, len(rle)) if rle[i][0] in ('fast','slow') and rle[i-1][0] in ('fast','slow') and rle[i][0] != rle[i-1][0])}")
    else:
        print(f"WARNING: {detail_seg_id} not found in data")

    # --- A3: V4 contract check ---
    v4_segments = [r for r in seg_results if r["schema"] == 4]
    if v4_segments:
        print(f"\nA3: V4 contract check ({len(v4_segments)} segments)")
        for r in v4_segments:
            v4_bim = r.get("v4_is_bimodal")
            v4_nom = r.get("v4_nominal_dt_s")
            v4_smf = r.get("v4_short_mode_fraction")
            actual_fast_frac = r["fast_frac"]
            actual_switches = r["n_mode_switches"]
            slow_tick = r.get("slow_centre")
            fast_tick = r.get("fast_centre")
            slow_dt = slow_tick / 90000 if slow_tick else None
            fast_dt = fast_tick / 90000 if fast_tick else None
            print(f"  {r['segment_id']}: is_bimodal={v4_bim} nominal_dt={v4_nom} "
                  f"v4_smf={v4_smf} | "
                  f"fast_frac={actual_fast_frac:.3f} switches={actual_switches} "
                  f"fast_dt={fast_dt}s slow_dt={slow_dt}s "
                  f"gaps={r['gap_total']}")

    # --- Per-camera summary (H4) ---
    print("\n=== Per-Camera Summary ===")
    for cam in ["FP7oJQ", "PPDmUg"]:
        cam_results = [r for r in seg_results if r["camera"] == cam]
        if not cam_results:
            continue
        total_gaps = sum(r["gap_total"] for r in cam_results)
        total_intervals = sum(r["n_intervals"] for r in cam_results)
        total_frames = sum(r["n_frames"] for r in cam_results)
        gap_free_segs = sum(1 for r in cam_results if r["gap_total"] == 0)
        max_gap_free = max(r["longest_gap_free_frames"] for r in cam_results)
        max_gap_free_s = max(r["longest_gap_free_s"] for r in cam_results)
        gap_rate = total_gaps / total_intervals * 1000 if total_intervals else 0

        print(f"\n{cam}: {len(cam_results)} segments, {total_frames} frames, {total_intervals} intervals")
        print(f"  Total gaps: {total_gaps} ({gap_rate:.2f} per 1000 intervals)")
        print(f"  Gap-free segments: {gap_free_segs}/{len(cam_results)} ({100*gap_free_segs/len(cam_results):.1f}%)")
        print(f"  Longest gap-free run: {max_gap_free} frames ({max_gap_free_s:.1f}s)")
        print(f"  All segments gap-free: {'YES' if gap_free_segs == len(cam_results) else 'NO'}")

        # Mode centre summary
        fast_centres = [r["fast_centre"] for r in cam_results if r["fast_centre"] is not None]
        slow_centres = [r["slow_centre"] for r in cam_results if r["slow_centre"] is not None]
        if fast_centres:
            print(f"  Fast mode centres: {min(fast_centres)}-{max(fast_centres)} ticks "
                  f"({min(fast_centres)/90:.1f}-{max(fast_centres)/90:.1f}ms, n={len(fast_centres)} segments)")
        if slow_centres:
            print(f"  Slow mode centres: {min(slow_centres)}-{max(slow_centres)} ticks "
                  f"({min(slow_centres)/90:.1f}-{max(slow_centres)/90:.1f}ms, n={len(slow_centres)} segments)")

    # --- Save tick histograms as JSON ---
    hist_dir = output_dir / "tick_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for seg_id, (deltas, classes, sc, lc) in seg_data.items():
        hist = tick_histogram(deltas)
        with open(hist_dir / f"{seg_id}.json", "w") as f:
            json.dump({
                "segment_id": seg_id,
                "fast_centre": sc,
                "slow_centre": lc,
                "histogram": {str(k): v for k, v in sorted(hist.items())},
                "n_intervals": len(deltas),
            }, f, indent=2)

    print(f"\nWrote {len(seg_data)} tick histograms to {hist_dir}/")
    print(f"\nDone. All outputs in {output_dir}/")

    # Return summary data for findings generation
    return {
        "seg_results": seg_results,
        "attempt_results": attempt_results,
        "grid_rows": grid_rows,
        "corr": corr,
        "v4_segments": v4_segments,
        "seg_data": seg_data,
    }


if __name__ == "__main__":
    main()
