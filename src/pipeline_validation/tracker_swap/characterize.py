"""Swap pattern characterization (CP-SWAP-2).

Four analyses that characterize the actual swap patterns in GT-confirmed
swap events from CP-SWAP-1, producing structured data for splitter design.

GT assignment uses IoU >= 0.3 (same as CP-SWAP-1, not the frozen 0.5).
Stride-1 (FP7oJQ) vs stride-10 (J_EDEw, PPDmUg) differences are noted
throughout — the ±3 GT-frame cascade window maps to ±3 raw frames on
stride-1 and ±30 raw frames on stride-10. The sustained persistence
threshold of 10 GT-annotated frames means 10 raw frames on stride-1 and
100 raw frames on stride-10.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_validation.common.matching import iou_matrix

logger = logging.getLogger(__name__)

GT_ASSIGN_IOU_THRESHOLD = 0.3
SUSTAINED_THRESHOLD = 10  # GT-annotated frames
CASCADE_WINDOW_GT_FRAMES = 3
PROXIMITY_RADIUS_M = 0.5
SPIKE_WINDOW_HALF = 10
SPIKE_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SwapContext:
    camera_id: str
    stride: int
    swap_events: list[dict]
    frame_features: pd.DataFrame
    bank_frames: pd.DataFrame | None
    detections: pd.DataFrame


@dataclass
class CharacterizationResult:
    camera_id: str
    stride: int
    topology: list[dict]
    persistence: list[dict]
    persistence_segments: dict  # tracklet_id -> list of segments
    spatial_context: list[dict]
    spike_profiles: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_frame_gt_map(
    frame_features: pd.DataFrame,
) -> dict[int, dict[str, int]]:
    """Build frame -> {tracklet_id: gt_person_id} for valid assignments."""
    result: dict[int, dict[str, int]] = {}
    valid = frame_features[frame_features["gt_person_id"] != -1]
    for _, row in valid.iterrows():
        fidx = int(row["frame_index"])
        if fidx not in result:
            result[fidx] = {}
        result[fidx][row["tracklet_id"]] = int(row["gt_person_id"])
    return result


def _get_world_pos(
    bank_frames: pd.DataFrame | None,
    tracklet_id: str,
    frame_index: int,
) -> tuple[float, float] | None:
    """Get world position for a tracklet at a frame, preferring repaired coords."""
    if bank_frames is None:
        return None
    mask = (bank_frames["tracklet_id"] == tracklet_id) & (
        bank_frames["frame_index"] == frame_index
    )
    rows = bank_frames.loc[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for xc, yc in [("x_m_repaired", "y_m_repaired"), ("x_m", "y_m")]:
        if xc in row.index and yc in row.index:
            x, y = row[xc], row[yc]
            if pd.notna(x) and pd.notna(y):
                return (float(x), float(y))
    return None


def _get_pixel_center(
    detections: pd.DataFrame,
    tracklet_id: str,
    frame_index: int,
) -> tuple[float, float] | None:
    """Get bbox center in pixel space."""
    mask = (detections["tracklet_id"] == tracklet_id) & (
        detections["frame_index"] == frame_index
    )
    rows = detections.loc[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    cx = (float(row["x1"]) + float(row["x2"])) / 2
    cy = (float(row["y1"]) + float(row["y2"])) / 2
    return (cx, cy)


def _world_positions_at_frame(
    bank_frames: pd.DataFrame | None,
    frame_index: int,
    exclude_tracklet: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Get world positions of all tracklets at a frame."""
    if bank_frames is None:
        return {}
    mask = bank_frames["frame_index"] == frame_index
    if exclude_tracklet:
        mask = mask & (bank_frames["tracklet_id"] != exclude_tracklet)
    rows = bank_frames.loc[mask]
    result: dict[str, tuple[float, float]] = {}
    for _, row in rows.iterrows():
        for xc, yc in [("x_m_repaired", "y_m_repaired"), ("x_m", "y_m")]:
            if xc in row.index and yc in row.index:
                x, y = row[xc], row[yc]
                if pd.notna(x) and pd.notna(y):
                    result[row["tracklet_id"]] = (float(x), float(y))
                    break
    return result


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


# ---------------------------------------------------------------------------
# Analysis 1 — Swap Topology
# ---------------------------------------------------------------------------

def analyze_topology(ctx: SwapContext) -> list[dict]:
    """Classify each swap event by its topological pattern."""
    frame_gt_map = _build_frame_gt_map(ctx.frame_features)
    results: list[dict] = []

    # Pre-index: for each frame, collect all swap events
    swap_by_frame: dict[int, list[dict]] = {}
    for ev in ctx.swap_events:
        fb = ev["frame_before"]
        fa = ev["frame_after"]
        for f in [fb, fa]:
            swap_by_frame.setdefault(f, []).append(ev)

    for ev in ctx.swap_events:
        tid = ev["tracklet_id"]
        fb = ev["frame_before"]
        fa = ev["frame_after"]
        gp_before = ev["gt_person_before"]
        gp_after = ev["gt_person_after"]

        gt_at_before = frame_gt_map.get(fb, {})
        gt_at_after = frame_gt_map.get(fa, {})

        # Check for reciprocal swap
        reciprocal_tid = None
        for other_tid, other_gp in gt_at_before.items():
            if other_tid == tid:
                continue
            if other_gp == gp_after:
                # This tracklet was on gp_after before the swap
                other_gp_after = gt_at_after.get(other_tid)
                if other_gp_after == gp_before:
                    reciprocal_tid = other_tid
                    break

        # Check for cascade: count tracklets that change GT assignment
        # within ±CASCADE_WINDOW_GT_FRAMES of the swap
        gt_frames_sorted = sorted(frame_gt_map.keys())
        try:
            fb_pos = gt_frames_sorted.index(fb)
        except ValueError:
            fb_pos = min(range(len(gt_frames_sorted)),
                        key=lambda i: abs(gt_frames_sorted[i] - fb),
                        default=0)

        window_start = max(0, fb_pos - CASCADE_WINDOW_GT_FRAMES)
        window_end = min(len(gt_frames_sorted) - 1, fb_pos + CASCADE_WINDOW_GT_FRAMES)
        window_frames = gt_frames_sorted[window_start:window_end + 1]

        # Find tracklets that change GT person within the window
        tracklet_changes: set[str] = set()
        for wf_idx in range(len(window_frames) - 1):
            f1 = window_frames[wf_idx]
            f2 = window_frames[wf_idx + 1]
            gt1 = frame_gt_map.get(f1, {})
            gt2 = frame_gt_map.get(f2, {})
            for t in set(gt1.keys()) & set(gt2.keys()):
                if gt1[t] != gt2[t]:
                    tracklet_changes.add(t)

        n_rearranged = len(tracklet_changes)

        # Classify
        if n_rearranged >= 3:
            topo_class = "cascade"
        elif reciprocal_tid is not None:
            topo_class = "exchange"
        else:
            # Check if gt_person_after was covered by another tracklet at frame_before
            other_on_target = any(
                gp == gp_after
                for t, gp in gt_at_before.items()
                if t != tid
            )
            if other_on_target:
                topo_class = "hop_into_occupied"
            else:
                topo_class = "hop_into_unoccupied"

        results.append({
            "tracklet_id": tid,
            "frame_before": fb,
            "frame_after": fa,
            "gt_person_before": gp_before,
            "gt_person_after": gp_after,
            "topology_class": topo_class,
            "reciprocal_tracklet_id": reciprocal_tid,
            "n_tracklets_rearranged": n_rearranged,
        })

    return results


# ---------------------------------------------------------------------------
# Analysis 2 — Swap Duration and Persistence
# ---------------------------------------------------------------------------

def analyze_persistence(ctx: SwapContext) -> tuple[list[dict], dict]:
    """Classify swap persistence and extract GT assignment segments."""
    swap_tracklets = {ev["tracklet_id"] for ev in ctx.swap_events}
    segments_by_tracklet: dict[str, list[dict]] = {}
    persistence_results: list[dict] = []

    for tid in sorted(swap_tracklets):
        tf = ctx.frame_features[ctx.frame_features["tracklet_id"] == tid].copy()
        tf = tf.sort_values("frame_index")
        # Only valid GT assignments
        valid = tf[tf["gt_person_id"] != -1]
        if valid.empty:
            continue

        # Build segments of consecutive same-GT-person
        segments: list[dict] = []
        prev_gp = None
        seg_start = None
        seg_frames = 0
        for _, row in valid.iterrows():
            gp = int(row["gt_person_id"])
            fi = int(row["frame_index"])
            if gp != prev_gp:
                if prev_gp is not None:
                    segments.append({
                        "gt_person": prev_gp,
                        "start_frame": seg_start,
                        "end_frame": prev_fi,
                        "n_gt_frames": seg_frames,
                    })
                prev_gp = gp
                seg_start = fi
                seg_frames = 1
            else:
                seg_frames += 1
            prev_fi = fi
        if prev_gp is not None:
            segments.append({
                "gt_person": prev_gp,
                "start_frame": seg_start,
                "end_frame": prev_fi,
                "n_gt_frames": seg_frames,
            })

        segments_by_tracklet[tid] = segments

        # Classify each transition between segments
        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            next_seg = segments[i + 1] if i + 1 < len(segments) else None

            dwell = curr_seg["n_gt_frames"]

            if dwell >= SUSTAINED_THRESHOLD:
                p_class = "sustained"
            elif next_seg is not None and next_seg["gt_person"] == prev_seg["gt_person"]:
                p_class = "transient_return"
            elif next_seg is not None and next_seg["gt_person"] != prev_seg["gt_person"]:
                p_class = "transient_onward"
            else:
                # End of tracklet with short segment
                if dwell >= SUSTAINED_THRESHOLD:
                    p_class = "sustained"
                else:
                    p_class = "sustained"  # end-of-tracklet, treat as sustained

            persistence_results.append({
                "tracklet_id": tid,
                "frame_before": prev_seg["end_frame"],
                "frame_after": curr_seg["start_frame"],
                "gt_person_before": prev_seg["gt_person"],
                "gt_person_after": curr_seg["gt_person"],
                "persistence_class": p_class,
                "dwell_gt_frames": dwell,
                "dwell_raw_frames": (curr_seg["end_frame"] - curr_seg["start_frame"]),
            })

    return persistence_results, segments_by_tracklet


# ---------------------------------------------------------------------------
# Analysis 3 — Spatial Context at Swap Time
# ---------------------------------------------------------------------------

def analyze_spatial_context(ctx: SwapContext) -> list[dict]:
    """Measure spatial environment at each swap event."""
    results: list[dict] = []
    has_world = ctx.bank_frames is not None and not ctx.bank_frames.empty

    for ev in ctx.swap_events:
        tid = ev["tracklet_id"]
        fb = ev["frame_before"]
        fa = ev["frame_after"]

        rec: dict = {
            "tracklet_id": tid,
            "frame_before": fb,
            "frame_after": fa,
        }

        # Pixel space displacement
        px_before = _get_pixel_center(ctx.detections, tid, fb)
        px_after = _get_pixel_center(ctx.detections, tid, fa)
        if px_before and px_after:
            rec["swap_distance_px"] = _euclidean(px_before, px_after)
        else:
            rec["swap_distance_px"] = None

        # World space displacement and proximity
        if has_world:
            pos_before = _get_world_pos(ctx.bank_frames, tid, fb)
            pos_after = _get_world_pos(ctx.bank_frames, tid, fa)

            if pos_before and pos_after:
                rec["swap_distance_m"] = _euclidean(pos_before, pos_after)
            else:
                rec["swap_distance_m"] = None

            # Other tracklets near source at frame_before
            others_before = _world_positions_at_frame(
                ctx.bank_frames, fb, exclude_tracklet=tid
            )
            if pos_before:
                near_source = sum(
                    1 for p in others_before.values()
                    if _euclidean(p, pos_before) < PROXIMITY_RADIUS_M
                )
                rec["n_other_tracklets_near_source"] = near_source
                if others_before:
                    rec["nearest_other_tracklet_dist_m_source"] = min(
                        _euclidean(p, pos_before) for p in others_before.values()
                    )
                else:
                    rec["nearest_other_tracklet_dist_m_source"] = None
            else:
                rec["n_other_tracklets_near_source"] = None
                rec["nearest_other_tracklet_dist_m_source"] = None

            # Other tracklets near dest at frame_after
            others_after = _world_positions_at_frame(
                ctx.bank_frames, fa, exclude_tracklet=tid
            )
            if pos_after:
                near_dest = sum(
                    1 for p in others_after.values()
                    if _euclidean(p, pos_after) < PROXIMITY_RADIUS_M
                )
                rec["n_other_tracklets_near_dest"] = near_dest
                if others_after:
                    rec["nearest_other_tracklet_dist_m"] = min(
                        _euclidean(p, pos_after) for p in others_after.values()
                    )
                else:
                    rec["nearest_other_tracklet_dist_m"] = None
            else:
                rec["n_other_tracklets_near_dest"] = None
                rec["nearest_other_tracklet_dist_m"] = None

            # Source persistence: any detection within 0.5m of source in N+1..N+5
            source_persists = False
            if pos_before:
                for check_f in range(fa + 1, fa + 6):
                    others = _world_positions_at_frame(ctx.bank_frames, check_f)
                    if any(
                        _euclidean(p, pos_before) < PROXIMITY_RADIUS_M
                        for p in others.values()
                    ):
                        source_persists = True
                        break
            rec["source_persists"] = source_persists

            # Dest pre-occupancy: any other tracklet within 0.5m of dest in N-5..N-1
            dest_was_occupied = False
            if pos_after:
                for check_f in range(fb - 4, fb + 1):
                    others = _world_positions_at_frame(
                        ctx.bank_frames, check_f, exclude_tracklet=tid
                    )
                    if any(
                        _euclidean(p, pos_after) < PROXIMITY_RADIUS_M
                        for p in others.values()
                    ):
                        dest_was_occupied = True
                        break
            rec["dest_was_occupied"] = dest_was_occupied
        else:
            logger.warning(
                "%s: bank_frames unavailable, spatial context limited to pixel space",
                ctx.camera_id,
            )
            for key in [
                "swap_distance_m", "n_other_tracklets_near_source",
                "n_other_tracklets_near_dest", "nearest_other_tracklet_dist_m",
                "nearest_other_tracklet_dist_m_source",
                "source_persists", "dest_was_occupied",
            ]:
                rec[key] = None

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Analysis 4 — Kinematic Spike Profile
# ---------------------------------------------------------------------------

def analyze_spike_profiles(ctx: SwapContext) -> list[dict]:
    """Characterize kinematic spike shape at each swap event."""
    results: list[dict] = []
    if ctx.bank_frames is None or ctx.bank_frames.empty:
        logger.warning("%s: bank_frames unavailable, skipping spike analysis", ctx.camera_id)
        return results

    # Pre-index bank frames for fast lookup
    bf_indexed = ctx.bank_frames.set_index(["tracklet_id", "frame_index"])

    for ev in ctx.swap_events:
        tid = ev["tracklet_id"]
        fb = ev["frame_before"]
        fa = ev["frame_after"]
        swap_frame = fa  # frame where new assignment begins

        rec: dict = {
            "tracklet_id": tid,
            "frame_before": fb,
            "frame_after": fa,
            "stride": ctx.stride,
        }

        # Get tracklet's bank frame data
        try:
            tid_data = bf_indexed.loc[tid]
        except KeyError:
            results.append(rec)
            continue

        if ctx.stride == 1:
            # Full window profile
            window_start = swap_frame - SPIKE_WINDOW_HALF
            window_end = swap_frame + SPIKE_WINDOW_HALF
            window_frames = list(range(window_start, window_end + 1))

            speed_profile = []
            accel_profile = []
            for wf in window_frames:
                if wf in tid_data.index:
                    row = tid_data.loc[wf]
                    s = row.get("speed_mps_k", np.nan)
                    a = row.get("accel_mps2_k", np.nan)
                    speed_profile.append(
                        float(s) if pd.notna(s) else None
                    )
                    accel_profile.append(
                        float(a) if pd.notna(a) else None
                    )
                else:
                    speed_profile.append(None)
                    accel_profile.append(None)

            rec["speed_profile"] = speed_profile
            rec["accel_profile"] = accel_profile
            rec["window_start"] = window_start
            rec["window_end"] = window_end

            # Classify spike shape
            valid_speeds = [s for s in speed_profile if s is not None]
            if not valid_speeds:
                rec["spike_class"] = "no_spike"
            else:
                # Tracklet median speed (full tracklet, not just window)
                all_speeds = tid_data["speed_mps_k"].dropna()
                if all_speeds.empty:
                    rec["spike_class"] = "no_spike"
                else:
                    median_speed = float(all_speeds.median())
                    threshold = SPIKE_MULTIPLIER * median_speed if median_speed > 0 else float("inf")

                    max_speed = max(valid_speeds)
                    if max_speed < threshold:
                        rec["spike_class"] = "no_spike"
                    else:
                        # Count elevated frames and crossings
                        elevated = [
                            s is not None and s >= threshold for s in speed_profile
                        ]
                        n_elevated = sum(elevated)

                        # Count crossings of threshold
                        crossings = 0
                        prev_high = False
                        for e in elevated:
                            if e and not prev_high:
                                crossings += 1
                            prev_high = e

                        if crossings >= 3:
                            rec["spike_class"] = "oscillating"
                        elif n_elevated <= 2:
                            rec["spike_class"] = "impulse"
                        else:
                            # Check if it stays elevated
                            # Find longest consecutive elevated run
                            max_run = 0
                            cur_run = 0
                            for e in elevated:
                                if e:
                                    cur_run += 1
                                    max_run = max(max_run, cur_run)
                                else:
                                    cur_run = 0
                            if max_run >= 5:
                                rec["spike_class"] = "ramp_sustain"
                            else:
                                rec["spike_class"] = "impulse"

                    rec["median_speed"] = median_speed
                    rec["max_speed_in_window"] = max_speed
        else:
            # Stride-10: gap summary
            gap_frames = list(range(fb, fa + 1))
            gap_speeds = []
            for gf in gap_frames:
                if gf in tid_data.index:
                    s = tid_data.loc[gf].get("speed_mps_k", np.nan)
                    if pd.notna(s):
                        gap_speeds.append((gf, float(s)))

            if gap_speeds:
                max_entry = max(gap_speeds, key=lambda x: x[1])
                rec["max_speed_in_gap"] = max_entry[1]
                rec["frame_of_max_speed"] = max_entry[0]

                # Start and end speeds
                rec["gap_start_speed"] = gap_speeds[0][1]
                rec["gap_end_speed"] = gap_speeds[-1][1]

                # Compare to tracklet median
                all_speeds = tid_data["speed_mps_k"].dropna()
                if not all_speeds.empty:
                    median_speed = float(all_speeds.median())
                    rec["median_speed"] = median_speed
                    rec["max_exceeds_2x_median"] = (
                        max_entry[1] >= SPIKE_MULTIPLIER * median_speed
                        if median_speed > 0 else False
                    )
                else:
                    rec["median_speed"] = None
                    rec["max_exceeds_2x_median"] = None
            else:
                rec["max_speed_in_gap"] = None
                rec["frame_of_max_speed"] = None

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_characterization(ctx: SwapContext) -> CharacterizationResult:
    """Run all four characterization analyses for one camera."""
    logger.info("Analysis 1: swap topology for %s", ctx.camera_id)
    topology = analyze_topology(ctx)

    logger.info("Analysis 2: swap persistence for %s", ctx.camera_id)
    persistence, segments = analyze_persistence(ctx)

    logger.info("Analysis 3: spatial context for %s", ctx.camera_id)
    spatial = analyze_spatial_context(ctx)

    logger.info("Analysis 4: spike profiles for %s (%s)",
                ctx.camera_id, "stride-1" if ctx.stride == 1 else f"stride-{ctx.stride}")
    spikes = analyze_spike_profiles(ctx)

    return CharacterizationResult(
        camera_id=ctx.camera_id,
        stride=ctx.stride,
        topology=topology,
        persistence=persistence,
        persistence_segments=segments,
        spatial_context=spatial,
        spike_profiles=spikes,
    )
