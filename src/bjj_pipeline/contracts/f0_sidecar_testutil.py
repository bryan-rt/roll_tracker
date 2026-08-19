"""Synthetic sidecar generator for testing.

Produces schema-5-conformant JSONL files that the f0_sidecar reader accepts.
Emits JSONL to disk so tests exercise the full parse path including gate logic.

pts_time_s and dt_s are mutually consistent: pts_time_s[i] = cumsum(dt_s[0:i]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def generate_synthetic_sidecar(
    path: Path,
    frame_count: int,
    dt_s: float = 0.067,
    *,
    source_pts: bool = True,
    timing_mode: str = "passthrough",
    row_source: str = "mp4",
    is_bimodal: bool = False,
    gaps: Optional[List[int]] = None,
    schema: int = 5,
    host_arrival_base: Optional[float] = None,
    host_arrival_missing: Optional[List[int]] = None,
    n_drift_windows: int = 6,
    segment_start_epoch: int = 1786988297,
) -> Path:
    """Write a synthetic sidecar JSONL file.

    Args:
        path: Output path for the .timing.jsonl file.
        frame_count: Number of frame rows to emit.
        dt_s: Nominal inter-frame interval in seconds.
        source_pts: Whether source PTS is available.
        timing_mode: "passthrough" or "cfr_grid".
        row_source: "mp4", "mp4_regenerated", or "showinfo_grid".
        is_bimodal: Emit bimodal _meta fields.
        gaps: Frame indices where dt doubles (simulates gaps).
        schema: Sidecar schema version.
        host_arrival_base: Base epoch for host_arrival_s. None = no host_arrival.
        host_arrival_missing: Frame indices where host_arrival_s is absent (join miss).
        n_drift_windows: Drift window count.
        segment_start_epoch: Epoch for segment_start_epoch.

    Returns:
        The path written to.
    """
    gaps_set = set(gaps or [])
    host_missing_set = set(host_arrival_missing or [])

    # Build frame-level data (pts_time_s and dt_s consistent via cumsum)
    pts_values: List[float] = []
    dt_values: List[Optional[float]] = []
    host_values: List[Optional[float]] = []

    pts = 0.0
    for i in range(frame_count):
        pts_values.append(round(pts, 6))

        if i == 0:
            dt_values.append(None)
        else:
            interval = dt_s * 2 if i in gaps_set else dt_s
            dt_values.append(round(interval, 6))

        # host_arrival_s
        if source_pts and row_source != "mp4_regenerated":
            if host_arrival_base is not None and i not in host_missing_set:
                host_values.append(round(host_arrival_base + pts, 6))
            elif host_arrival_base is not None:
                host_values.append(None)  # join miss
            else:
                host_values.append(None)
        else:
            host_values.append(None)

        # Advance pts for next frame
        if i + 1 < frame_count:
            interval = dt_s * 2 if (i + 1) in gaps_set else dt_s
            pts += interval

    # Compute tick-level statistics
    timebase = 90000
    tick_deltas = []
    for d in dt_values[1:]:
        if d is not None:
            tick_deltas.append(round(d * timebase))

    if tick_deltas:
        sorted_td = sorted(tick_deltas)
        nd = len(sorted_td)
        median_tick = sorted_td[nd // 2] if nd % 2 == 1 else (sorted_td[nd // 2 - 1] + sorted_td[nd // 2]) / 2.0
        mean_tick = sum(tick_deltas) / nd
        lo = median_tick * 0.5
        hi = median_tick * 1.5
        trim_vals = [d for d in sorted_td if lo <= d <= hi]
        trim_n = len(trim_vals)
        trimmed_mean = sum(trim_vals) / trim_n if trim_n else median_tick
        delta_ms = [d * 1000.0 / timebase for d in sorted_td]
        mean_d_ms = sum(delta_ms) / nd
        sum_d2 = sum(x * x for x in delta_ms)
        stdev_d_ms = (sum_d2 / nd - mean_d_ms * mean_d_ms) ** 0.5
    else:
        median_tick = mean_tick = 0
        trim_n = 0
        trimmed_mean = 0
        mean_d_ms = stdev_d_ms = 0

    nominal_dt = median_tick / timebase if median_tick > 0 else dt_s
    measured_fps_val = timebase / trimmed_mean if trimmed_mean > 0 else 0
    measured_fps_median_val = timebase / median_tick if median_tick > 0 else 0
    last_pts = pts_values[-1] if pts_values else 0
    measured_fps_mean_val = (frame_count - 1) / last_pts if last_pts > 0 else 0

    # Build _meta
    meta: dict = {
        "_meta": True,
        "sidecar_schema": schema,
        "timing_mode": timing_mode,
        "source_pts": source_pts,
        "pts_origin": "segment_relative",
        "fps_method": "trimmed_mean",
        "row_source": row_source,
        "segment_start_epoch": segment_start_epoch,
        "attempt": 1,
        "input_frame_count": frame_count,
        "output_frame_count": frame_count,
        "measured_fps_mean": round(measured_fps_mean_val, 4),
        "pts_timebase": timebase,
        "pts_tick_delta_median": round(median_tick, 1),
        "pts_tick_delta_mean": round(mean_tick, 1),
        "pts_delta_trim_kept": trim_n,
        "pts_delta_trim_total": len(tick_deltas),
        "mismatch": False,
        "pts_mean_delta_ms": round(mean_d_ms, 4),
        "pts_stdev_delta_ms": round(stdev_d_ms, 4),
    }

    if source_pts:
        meta["nominal_dt_s"] = round(nominal_dt, 6)
        meta["measured_fps"] = round(measured_fps_val, 4)
        meta["measured_fps_median"] = round(measured_fps_median_val, 4)
        meta["is_bimodal"] = is_bimodal
        if is_bimodal:
            meta["short_mode_fraction"] = 0.15
            meta["short_mode_fps"] = 30.0
            meta["short_mode_dt_s"] = 0.033333
            meta["long_mode_dt_s"] = round(dt_s, 6)
        meta["pts_wallclock_offset_s"] = float(segment_start_epoch)
        meta["offset_method"] = "lower_envelope"
        if n_drift_windows >= 4:
            meta["drift_rate_s_per_s"] = -0.000000603
            meta["drift_ppm"] = -0.603
        meta["drift_flat"] = n_drift_windows < 4 or True
        meta["n_drift_windows"] = n_drift_windows

    if row_source != "mp4_regenerated":
        gap_count = len(gaps_set)
        meta["showinfo_frame_count"] = frame_count + gap_count  # simulate surplus
        meta["showinfo_residual"] = -gap_count
        meta["showinfo_pts_offset"] = 0
        meta["showinfo_matched_count"] = frame_count
        meta["showinfo_unmatched_mp4_count"] = 0
        meta["showinfo_surplus_count"] = gap_count
        meta["showinfo_offset_status"] = "determined"

    # For schema 4 compat: add input_n to frame rows
    emit_input_n = schema < 5

    # Write JSONL
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        for i in range(frame_count):
            row: dict = {
                "frame_index": i,
                "pts_time_s": pts_values[i],
            }
            if timing_mode == "passthrough" and source_pts:
                row["dt_s"] = dt_values[i]
            if host_values[i] is not None:
                row["host_arrival_s"] = host_values[i]
            if emit_input_n:
                row["input_n"] = i
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    return path
