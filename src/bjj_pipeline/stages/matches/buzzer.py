"""Role: buzzer soft gate for engagement interval boundary adjustment.

Optionally adjusts interval end boundaries using audio landmark events
(sustained tones / buzzers). Entirely skipped when no audio_events.jsonl exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bjj_pipeline.stages.matches.hysteresis import EngagementInterval


def load_audio_events(audio_events_path: Path) -> List[Dict]:
    """Load audio_events.jsonl for a session.

    Returns [] if file missing or unreadable. Never raises.
    Filters to event_class == "sustained_tone" only.
    """
    if not audio_events_path.exists():
        return []

    events: List[Dict] = []
    try:
        with audio_events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if rec.get("event_class") == "sustained_tone":
                    events.append(rec)
    except Exception as exc:
        logger.warning("buzzer: failed to read {}: {}", audio_events_path, exc)
        return []

    return events


def _lookup_pair_distance(
    pair_distances_df: pd.DataFrame,
    person_id_a: str,
    person_id_b: str,
    frame_index: int,
) -> Optional[float]:
    """Look up distance for a specific pair at a specific frame. Returns None if not found."""
    if pair_distances_df is None or pair_distances_df.empty:
        return None

    mask = (
        (pair_distances_df["person_id_a"] == person_id_a)
        & (pair_distances_df["person_id_b"] == person_id_b)
        & (pair_distances_df["frame_index"] == frame_index)
    )
    matches = pair_distances_df.loc[mask, "dist_m"]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def apply_buzzer_soft_gate(
    intervals: List[EngagementInterval],
    audio_events: List[Dict],
    *,
    fps: float,
    buzzer_boundary_window_frames: int,
    pair_distances_df: pd.DataFrame,
    disengage_dist_m: float,
) -> List[EngagementInterval]:
    """Optionally adjust interval end boundaries using buzzer events.

    For each interval end:
    - Find any sustained_tone event within buzzer_boundary_window_frames of end_frame
    - If found AND pair distance at that frame exceeds disengage_dist_m:
      → adjust end_frame to the buzzer's frame_index
      → add "buzzer" to evidence_sources
    - Otherwise: no change

    If audio_events is empty: return intervals unchanged.
    Never raises. Returns same-length list.
    """
    if not audio_events:
        return intervals

    # Extract frame indices from audio events
    buzzer_frames: List[int] = []
    for ev in audio_events:
        fi = ev.get("frame_index")
        if fi is not None:
            try:
                buzzer_frames.append(int(fi))
            except (ValueError, TypeError):
                continue

    if not buzzer_frames:
        return intervals

    buzzer_frames_sorted = sorted(buzzer_frames)

    result: List[EngagementInterval] = []
    for interval in intervals:
        adjusted = _try_adjust_end(
            interval,
            buzzer_frames=buzzer_frames_sorted,
            buzzer_boundary_window_frames=buzzer_boundary_window_frames,
            pair_distances_df=pair_distances_df,
            disengage_dist_m=disengage_dist_m,
        )
        result.append(adjusted)

    return result


def _try_adjust_end(
    interval: EngagementInterval,
    *,
    buzzer_frames: List[int],
    buzzer_boundary_window_frames: int,
    pair_distances_df: pd.DataFrame,
    disengage_dist_m: float,
) -> EngagementInterval:
    """Try to snap interval end to a nearby buzzer event."""
    end = interval.end_frame

    # Find buzzer frames within window of end_frame
    candidates: List[int] = []
    for bf in buzzer_frames:
        if abs(bf - end) <= buzzer_boundary_window_frames:
            candidates.append(bf)

    if not candidates:
        return interval

    # Pick closest buzzer frame to end
    best_buzzer = min(candidates, key=lambda bf: abs(bf - end))

    # Check pair distance at buzzer frame
    dist = _lookup_pair_distance(
        pair_distances_df,
        interval.person_id_a,
        interval.person_id_b,
        best_buzzer,
    )

    # Only adjust if distance exceeds disengage threshold (confirming separation)
    if dist is None or dist <= disengage_dist_m:
        return interval

    # Adjust end_frame and add buzzer to evidence sources
    new_sources = tuple(sorted(set(interval.evidence_sources) | {"buzzer"}))
    return EngagementInterval(
        person_id_a=interval.person_id_a,
        person_id_b=interval.person_id_b,
        start_frame=interval.start_frame,
        end_frame=best_buzzer,
        evidence_sources=new_sources,
        partial_start=interval.partial_start,
        partial_end=interval.partial_end,
    )
