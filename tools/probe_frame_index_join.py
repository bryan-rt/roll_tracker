#!/usr/bin/env python3
"""Piece 0 probe: measure frame_index join between sidecar and decoded mp4.

Imports the production FrameIterator unmodified. Does not modify any pipeline code.

Usage:
    PYTHONPATH=src python tools/probe_frame_index_join.py \
        --root data/raw/nest/00000000-0000-0000-0000-000000000003 \
        --output docs/evidence/frame_index_join_1/probe_results.json

Measures per segment:
    (a) sidecar frame-row count
    (b) output_frame_count from _meta
    (c) decoded frame count from FrameIterator
    (d) input_frame_count from _meta
    (e) ffprobe -count_frames (nb_read_frames)
    frame_index contiguity and pts_time_s monotonicity
    positional alignment (POS_MSEC vs pts_time_s) with cross-correlation anchor
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_sidecar(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load sidecar, return (_meta dict, list of frame rows)."""
    meta = None
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                meta = rec
            else:
                rows.append(rec)
    if meta is None:
        raise ValueError(f"No _meta line in {path}")
    return meta, rows


def _ffprobe_count_frames(mp4_path: Path) -> Optional[int]:
    """Run ffprobe -count_frames and return nb_read_frames."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,nb_frames",
                "-of", "json",
                str(mp4_path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        payload = json.loads(proc.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        nb_read = stream.get("nb_read_frames")
        return int(nb_read) if nb_read is not None else None
    except Exception:
        return None


def _check_contiguity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check frame_index is exactly {0..N-1}, monotonic, no holes, no repeats."""
    indices = [r["frame_index"] for r in rows]
    n = len(indices)
    if n == 0:
        return {"contiguous": True, "monotonic": True, "n": 0}
    expected = list(range(n))
    contiguous = (indices == expected)
    monotonic = all(indices[i] < indices[i + 1] for i in range(n - 1))
    has_holes = len(set(indices)) < n or (n > 0 and max(indices) != n - 1)
    has_repeats = len(set(indices)) < len(indices)
    return {
        "contiguous": contiguous,
        "monotonic": monotonic,
        "has_holes": has_holes,
        "has_repeats": has_repeats,
        "min_index": min(indices) if indices else None,
        "max_index": max(indices) if indices else None,
        "n": n,
    }


def _check_pts_monotonicity(rows: List[Dict[str, Any]], nominal_dt_s: Optional[float]) -> Dict[str, Any]:
    """Check pts_time_s is strictly increasing and deltas consistent with nominal_dt_s."""
    pts_vals = [r.get("pts_time_s") for r in rows if r.get("pts_time_s") is not None]
    n = len(pts_vals)
    if n < 2:
        return {"strictly_increasing": True, "n": n}

    strictly_increasing = all(pts_vals[i] < pts_vals[i + 1] for i in range(n - 1))
    deltas = [pts_vals[i + 1] - pts_vals[i] for i in range(n - 1)]

    result: Dict[str, Any] = {
        "strictly_increasing": strictly_increasing,
        "n": n,
        "delta_min_s": round(min(deltas), 6),
        "delta_max_s": round(max(deltas), 6),
        "delta_mean_s": round(float(np.mean(deltas)), 6),
    }

    if nominal_dt_s and nominal_dt_s > 0:
        # Check that every delta is an integer multiple of nominal_dt_s (within 20%)
        ratios = [d / nominal_dt_s for d in deltas]
        rounded_ratios = [round(r) for r in ratios]
        fractional_errors = [abs(r - rr) for r, rr in zip(ratios, rounded_ratios)]
        max_frac_error = max(fractional_errors) if fractional_errors else 0.0
        result["max_ratio_fractional_error"] = round(max_frac_error, 4)
        result["all_integer_multiples"] = max_frac_error < 0.2
        # Check for discontinuities (deltas that are NOT near-integer multiples)
        discontinuities = [
            {"index": i, "delta_s": round(deltas[i], 6), "ratio": round(ratios[i], 3)}
            for i in range(len(deltas))
            if abs(ratios[i] - round(ratios[i])) >= 0.2
        ]
        if discontinuities:
            result["discontinuities"] = discontinuities[:5]  # cap output

    return result


def _decode_with_timestamps(mp4_path: Path) -> List[Tuple[int, float]]:
    """Decode with production FrameIterator, return [(frame_index, pos_msec), ...]."""
    import cv2
    from bjj_pipeline.core.frame_iterator import FrameIterator

    it = FrameIterator(mp4_path)
    results: List[Tuple[int, float]] = []
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")

    frame_index = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        results.append((frame_index, float(pos_msec)))
        frame_index += 1

    cap.release()

    # Also get the FrameIterator count to confirm it matches
    it2 = FrameIterator(mp4_path)
    fi_count = sum(1 for _ in it2)
    if fi_count != frame_index:
        print(f"  WARNING: cv2 loop={frame_index} vs FrameIterator={fi_count} on {mp4_path.name}", file=sys.stderr)

    return results


def _cross_correlate_alignment(
    sidecar_pts_ms: List[float],
    decoded_pos_ms: List[float],
    max_offset: int = 15,
) -> Dict[str, Any]:
    """Try aligning sidecar pts sequence against decoded POS_MSEC at offsets k=0..max_offset.

    Returns best k, its mean absolute error, and the error at k=0.
    """
    n_sc = len(sidecar_pts_ms)
    n_dec = len(decoded_pos_ms)
    if n_sc == 0 or n_dec == 0:
        return {"best_k": 0, "best_k_mae_ms": None, "k0_mae_ms": None}

    results = []
    for k in range(min(max_offset + 1, n_dec)):
        # Align sidecar[i] with decoded[i + k]
        overlap = min(n_sc, n_dec - k)
        if overlap < 5:
            break
        errors = [
            abs(sidecar_pts_ms[i] - decoded_pos_ms[i + k])
            for i in range(overlap)
        ]
        mae = float(np.mean(errors))
        max_err = float(np.max(errors))
        results.append({"k": k, "overlap": overlap, "mae_ms": round(mae, 3), "max_err_ms": round(max_err, 3)})

    if not results:
        return {"best_k": 0, "best_k_mae_ms": None, "k0_mae_ms": None}

    best = min(results, key=lambda r: r["mae_ms"])
    k0 = next((r for r in results if r["k"] == 0), None)

    return {
        "best_k": best["k"],
        "best_k_mae_ms": best["mae_ms"],
        "best_k_max_err_ms": best["max_err_ms"],
        "best_k_overlap": best["overlap"],
        "k0_mae_ms": k0["mae_ms"] if k0 else None,
        "k0_max_err_ms": k0["max_err_ms"] if k0 else None,
        "all_offsets": results[:max_offset + 1],
    }


def _analyze_pos_msec_pattern(decoded_pos_ms: List[float]) -> Dict[str, Any]:
    """Analyze whether POS_MSEC follows a uniform grid or shows real gaps."""
    if len(decoded_pos_ms) < 3:
        return {"pattern": "too_short"}

    deltas = [decoded_pos_ms[i + 1] - decoded_pos_ms[i] for i in range(len(decoded_pos_ms) - 1)]
    median_delta = float(np.median(deltas))
    # Count how many deltas are >1.5x the median (gap candidates)
    gap_deltas = [d for d in deltas if d > 1.5 * median_delta]
    # Count how many deltas are within 5% of median (uniform)
    uniform_deltas = [d for d in deltas if abs(d - median_delta) / max(median_delta, 0.001) < 0.05]

    unique_deltas = sorted(set(round(d, 1) for d in deltas))

    return {
        "median_delta_ms": round(median_delta, 3),
        "n_gap_deltas": len(gap_deltas),
        "n_uniform_deltas": len(uniform_deltas),
        "n_total_deltas": len(deltas),
        "uniform_fraction": round(len(uniform_deltas) / len(deltas), 4) if deltas else 0,
        "unique_delta_values": unique_deltas[:20],  # cap
        "gap_fraction": round(len(gap_deltas) / len(deltas), 4) if deltas else 0,
    }


def _compute_gap_density(rows: List[Dict[str, Any]], nominal_dt_s: Optional[float]) -> Optional[float]:
    """Fraction of dt_s values > 1.5 * nominal_dt_s."""
    if not nominal_dt_s or nominal_dt_s <= 0:
        return None
    dt_vals = [r["dt_s"] for r in rows if r.get("dt_s") is not None]
    if not dt_vals:
        return None
    gaps = [d for d in dt_vals if d > 1.5 * nominal_dt_s]
    return round(len(gaps) / len(dt_vals), 4)


def _parse_camera(path: Path) -> str:
    """Extract camera ID from path."""
    parts = path.parts
    for p in parts:
        if p in ("FP7oJQ", "PPDmUg", "J_EDEw"):
            return p
    return "unknown"


def _parse_attempt_key(path: Path, meta: Dict[str, Any]) -> str:
    """Build an attempt grouping key from path + _meta.attempt."""
    camera = _parse_camera(path)
    # Extract date/hour from path
    parts = path.parts
    date_part = ""
    hour_part = ""
    for i, p in enumerate(parts):
        if len(p) == 10 and p[4] == "-" and p[7] == "-":  # YYYY-MM-DD
            date_part = p
            if i + 1 < len(parts) and len(parts[i + 1]) <= 2:
                hour_part = parts[i + 1]
    attempt = meta.get("attempt", "?")
    return f"{camera}/{date_part}/{hour_part}/att{attempt}"


def probe_segment(sidecar_path: Path, do_alignment: bool = False) -> Dict[str, Any]:
    """Probe one segment. Returns result dict."""
    mp4_path = sidecar_path.with_suffix("").with_suffix(".mp4")
    # Handle .timing.jsonl -> .mp4
    name = sidecar_path.name
    if name.endswith(".timing.jsonl"):
        mp4_name = name.replace(".timing.jsonl", ".mp4")
        mp4_path = sidecar_path.parent / mp4_name

    if not mp4_path.exists():
        return {"error": f"mp4 not found: {mp4_path}", "segment": name}

    meta, rows = _load_sidecar(sidecar_path)
    camera = _parse_camera(sidecar_path)
    attempt_key = _parse_attempt_key(sidecar_path, meta)

    # (a) sidecar frame-row count
    a_count = len(rows)

    # (b) output_frame_count from _meta
    b_count = meta.get("output_frame_count")

    # (d) input_frame_count from _meta
    d_count = meta.get("input_frame_count")

    # (c) decoded frame count from FrameIterator
    from bjj_pipeline.core.frame_iterator import FrameIterator
    it = FrameIterator(mp4_path)
    c_count = sum(1 for _ in it)

    # (e) ffprobe -count_frames
    e_count = _ffprobe_count_frames(mp4_path)

    # Contiguity
    contiguity = _check_contiguity(rows)

    # PTS monotonicity
    nominal_dt_s = meta.get("nominal_dt_s")
    pts_mono = _check_pts_monotonicity(rows, nominal_dt_s)

    # Gap density
    gap_density = _compute_gap_density(rows, nominal_dt_s)

    # Residual (CP-R5 convention: output - input)
    residual = (b_count - d_count) if (b_count is not None and d_count is not None) else None
    residual_sign = None
    if residual is not None:
        residual_sign = "positive" if residual > 0 else ("negative" if residual < 0 else "zero")

    # Join predicate
    a_eq_c = (a_count == c_count)
    b_eq_c = (b_count == c_count) if b_count is not None else None
    b_eq_e = (b_count == e_count) if (b_count is not None and e_count is not None) else None
    a_eq_d = (a_count == d_count) if d_count is not None else None

    result: Dict[str, Any] = {
        "segment": name.replace(".timing.jsonl", ""),
        "camera": camera,
        "attempt_key": attempt_key,
        "a_sidecar_rows": a_count,
        "b_output_frame_count": b_count,
        "c_decoded_frames": c_count,
        "d_input_frame_count": d_count,
        "e_nb_read_frames": e_count,
        "residual": residual,
        "residual_sign": residual_sign,
        "a_eq_c": a_eq_c,
        "a_eq_d": a_eq_d,
        "b_eq_c": b_eq_c,
        "b_eq_e": b_eq_e,
        "contiguity": contiguity,
        "pts_monotonicity": pts_mono,
        "gap_density": gap_density,
        "meta_context": {
            "timing_mode": meta.get("timing_mode"),
            "source_pts": meta.get("source_pts"),
            "mismatch": meta.get("mismatch"),
            "is_bimodal": meta.get("is_bimodal"),
            "nominal_dt_s": nominal_dt_s,
            "attempt": meta.get("attempt"),
            "sidecar_schema": meta.get("sidecar_schema"),
        },
    }

    # Alignment analysis
    if do_alignment:
        sidecar_pts_ms = [r["pts_time_s"] * 1000.0 for r in rows if "pts_time_s" in r]
        decoded_data = _decode_with_timestamps(mp4_path)
        decoded_pos_ms = [d[1] for d in decoded_data]

        xcorr = _cross_correlate_alignment(sidecar_pts_ms, decoded_pos_ms)
        pos_msec_pattern = _analyze_pos_msec_pattern(decoded_pos_ms)

        # Also compare sidecar dt_s gaps against POS_MSEC deltas at best-k alignment
        best_k = xcorr["best_k"]
        if best_k is not None and len(sidecar_pts_ms) > 0 and len(decoded_pos_ms) > best_k:
            overlap = min(len(sidecar_pts_ms), len(decoded_pos_ms) - best_k)
            # Sample comparison at key frames
            sample_indices = sorted(set(
                [0, 1, 2, 5, 10] +
                list(range(0, overlap, max(1, overlap // 10))) +
                [overlap - 1]
            ))
            sample_indices = [i for i in sample_indices if i < overlap]
            alignment_samples = []
            for i in sample_indices:
                sc_ms = sidecar_pts_ms[i]
                dec_ms = decoded_pos_ms[i + best_k]
                delta = round(sc_ms - dec_ms, 3)
                alignment_samples.append({
                    "sidecar_i": i,
                    "decoded_i": i + best_k,
                    "sidecar_pts_ms": round(sc_ms, 3),
                    "decoded_pos_ms": round(dec_ms, 3),
                    "delta_ms": delta,
                })
            xcorr["alignment_samples"] = alignment_samples

        result["alignment"] = xcorr
        result["pos_msec_pattern"] = pos_msec_pattern

    return result


def main():
    parser = argparse.ArgumentParser(description="Probe frame_index join between sidecar and decoded mp4")
    parser.add_argument("--root", type=Path, required=True, help="Root directory to scan for sidecars")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--align-all", action="store_true", help="Run alignment analysis on ALL segments (slow)")
    args = parser.parse_args()

    # Discover all passthrough+source_pts sidecars
    sidecars = sorted(args.root.rglob("*.timing.jsonl"))
    print(f"Found {len(sidecars)} sidecar files", file=sys.stderr)

    valid_sidecars: List[Path] = []
    for sc in sidecars:
        try:
            with open(sc) as f:
                meta = json.loads(f.readline())
            if meta.get("timing_mode") == "passthrough" and meta.get("source_pts") is True:
                valid_sidecars.append(sc)
        except Exception:
            pass

    print(f"Valid passthrough+source_pts: {len(valid_sidecars)}", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    disagreeing: List[str] = []

    for i, sc in enumerate(valid_sidecars):
        seg_name = sc.name.replace(".timing.jsonl", "")
        print(f"  [{i+1}/{len(valid_sidecars)}] {seg_name}...", end="", file=sys.stderr, flush=True)
        t0 = time.time()

        # First pass: counts only (fast)
        r = probe_segment(sc, do_alignment=False)
        elapsed = time.time() - t0

        if not r.get("a_eq_c", True):
            disagreeing.append(seg_name)

        results.append(r)
        status = "OK" if r.get("a_eq_c") else f"MISMATCH a={r['a_sidecar_rows']} c={r['c_decoded_frames']}"
        print(f" {status} ({elapsed:.1f}s)", file=sys.stderr)

    # Second pass: alignment on disagreeing segments + control sample
    control_agree = [r["segment"] for r in results if r.get("a_eq_c")][:5]
    align_targets = set(disagreeing) | set(control_agree)
    if args.align_all:
        align_targets = set(r["segment"] for r in results)

    print(f"\nAlignment analysis on {len(align_targets)} segments...", file=sys.stderr)
    for i, r in enumerate(results):
        if r["segment"] not in align_targets:
            continue
        sc = next(s for s in valid_sidecars if r["segment"] in s.name)
        print(f"  Aligning {r['segment']}...", end="", file=sys.stderr, flush=True)
        t0 = time.time()
        aligned = probe_segment(sc, do_alignment=True)
        r["alignment"] = aligned.get("alignment")
        r["pos_msec_pattern"] = aligned.get("pos_msec_pattern")
        print(f" done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Summary
    n_total = len(results)
    n_agree = sum(1 for r in results if r.get("a_eq_c"))
    n_disagree = sum(1 for r in results if not r.get("a_eq_c"))

    fp7_results = [r for r in results if r["camera"] == "FP7oJQ"]
    ppd_results = [r for r in results if r["camera"] == "PPDmUg"]

    summary = {
        "total_segments": n_total,
        "a_eq_c_true": n_agree,
        "a_eq_c_false": n_disagree,
        "by_camera": {
            "FP7oJQ": {
                "total": len(fp7_results),
                "agree": sum(1 for r in fp7_results if r.get("a_eq_c")),
                "disagree": sum(1 for r in fp7_results if not r.get("a_eq_c")),
                "residual_positive": sum(1 for r in fp7_results if r.get("residual_sign") == "positive"),
                "residual_negative": sum(1 for r in fp7_results if r.get("residual_sign") == "negative"),
                "residual_zero": sum(1 for r in fp7_results if r.get("residual_sign") == "zero"),
            },
            "PPDmUg": {
                "total": len(ppd_results),
                "agree": sum(1 for r in ppd_results if r.get("a_eq_c")),
                "disagree": sum(1 for r in ppd_results if not r.get("a_eq_c")),
                "residual_positive": sum(1 for r in ppd_results if r.get("residual_sign") == "positive"),
                "residual_negative": sum(1 for r in ppd_results if r.get("residual_sign") == "negative"),
                "residual_zero": sum(1 for r in ppd_results if r.get("residual_sign") == "zero"),
            },
        },
    }

    # Attempt-level conservation
    attempts: Dict[str, List[Dict]] = {}
    for r in results:
        key = r.get("attempt_key", "unknown")
        attempts.setdefault(key, []).append(r)

    attempt_summaries = []
    for key in sorted(attempts.keys()):
        segs = attempts[key]
        residuals = [r["residual"] for r in segs if r["residual"] is not None]
        positives = [r for r in residuals if r > 0]
        negatives = [r for r in residuals if r < 0]
        attempt_summaries.append({
            "attempt_key": key,
            "n_segments": len(segs),
            "residuals": residuals,
            "total_residual": sum(residuals) if residuals else None,
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "n_zero": len([r for r in residuals if r == 0]),
            "sum_positive": sum(positives) if positives else 0,
            "sum_negative": sum(negatives) if negatives else 0,
        })

    output = {
        "summary": summary,
        "attempt_conservation": attempt_summaries,
        "segments": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2, default=str)


if __name__ == "__main__":
    main()
