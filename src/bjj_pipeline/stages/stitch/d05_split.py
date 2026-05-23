"""D0.5: Post-D0 tracklet splitter (CP-SPLIT-1).

Detects swap boundaries within tracklets using GT-free signals (kinematic
spikes, histogram discontinuity) and splits affected tracklets so D1
onwards sees purer tracklets.

Placement: after D0 (world coordinates + kinematics), before D1 (graph).
Modifies D0 output tables in-place (tracklet_bank_frames.parquet,
tracklet_bank_summaries.parquet). Does NOT modify Stage A artifacts.

Tiered detection:
  Tier 1 — Hard speed cap (teleportation)
  Tier 2 — Kinematic spike with isolation (impulse pattern)
  Tier 3 — Histogram discontinuity with kinematic corroboration
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    speed_cap_mps: Optional[float] = Field(
        default=None,
        description=(
            "Hard speed cap (m/s). If None, uses 6x d0.kinematics.v_max_mps "
            "(default 48 m/s). This is a teleportation detector, not a plausibility flag."
        ),
    )
    spike_speed_ratio: float = Field(default=5.0, gt=0)
    spike_min_speed_mps: float = Field(
        default=5.0, ge=0,
        description="Absolute speed floor for Tier 2 — spike ratio only fires above this.",
    )
    spike_isolation_ratio: float = Field(default=3.0, gt=0)
    spike_max_duration_frames: int = Field(default=2, ge=1)
    histogram_bhattacharyya_threshold: float = Field(default=0.15, ge=0)
    histogram_require_kinematic_corroboration: bool = Field(default=True)
    histogram_kinematic_corroboration_ratio: float = Field(default=2.0, gt=0)
    min_dwell_frames: int = Field(default=5, ge=1)
    write_audit: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitCandidate:
    frame_index: int
    tier: str  # "tier1_speed_cap", "tier2_kinematic_spike", "tier3_histogram"
    speed_at_frame: float
    isolation_ratio: Optional[float]
    bhattacharyya_dist: Optional[float]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_config(config: Dict[str, Any]) -> Tuple[SplitConfig, float]:
    """Resolve SplitConfig and effective speed_cap from pipeline config.

    Returns (split_config, effective_speed_cap_mps).
    """
    stage_d: Dict[str, Any] = {}
    if isinstance(config.get("stages"), dict) and isinstance(
        config["stages"].get("stage_D"), dict
    ):
        stage_d = config["stages"]["stage_D"]
    elif isinstance(config.get("stage_D"), dict):
        stage_d = config["stage_D"]

    raw = stage_d.get("d05_split", None)
    if raw is None or not isinstance(raw, dict):
        cfg = SplitConfig()
    else:
        cfg = SplitConfig(**raw)

    # Resolve speed_cap: explicit value, or 6x v_max_mps (teleportation, not plausibility)
    if cfg.speed_cap_mps is not None:
        speed_cap = cfg.speed_cap_mps
    else:
        d0 = stage_d.get("d0", {}) if isinstance(stage_d, dict) else {}
        kin = d0.get("kinematics", {}) if isinstance(d0, dict) else {}
        speed_cap = 6.0 * float(kin.get("v_max_mps", 8.0))  # default 48 m/s

    return cfg, speed_cap


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------

def detect_swap_boundaries(
    tracklet_frames: pd.DataFrame,
    tracklet_histograms: Optional[pd.DataFrame],
    cfg: SplitConfig,
    speed_cap_mps: float,
) -> List[SplitCandidate]:
    """Detect candidate swap boundaries within a single tracklet.

    tracklet_frames: rows from tracklet_bank_frames for one tracklet,
        sorted by frame_index. Must have 'speed_mps_k' column.
    tracklet_histograms: rows from color_histograms for this tracklet
        (joined on track_id == tracklet_id), or None.
    """
    if len(tracklet_frames) < 2:
        return []

    speeds = tracklet_frames["speed_mps_k"].values.astype(float)
    frames = tracklet_frames["frame_index"].values.astype(int)

    # Tracklet median speed (excluding NaN)
    valid_speeds = speeds[np.isfinite(speeds)]
    median_speed = float(np.median(valid_speeds)) if len(valid_speeds) > 0 else 0.0

    # Pre-build histogram lookup if available
    hist_by_frame: Dict[int, np.ndarray] = {}
    if tracklet_histograms is not None and not tracklet_histograms.empty:
        hist_cols = [c for c in tracklet_histograms.columns if c.startswith("hist_")]
        if hist_cols:
            for _, row in tracklet_histograms.iterrows():
                fi = int(row["frame_index"])
                hist_by_frame[fi] = row[hist_cols].values.astype(float)

    candidates: List[SplitCandidate] = []
    seen_frames: set = set()  # avoid duplicate candidates at same frame

    for i in range(1, len(frames)):
        fi = int(frames[i])
        speed = float(speeds[i])

        if not np.isfinite(speed):
            continue

        # --- Tier 1: Hard speed cap ---
        if speed > speed_cap_mps:
            if fi not in seen_frames:
                candidates.append(SplitCandidate(
                    frame_index=fi,
                    tier="tier1_speed_cap",
                    speed_at_frame=speed,
                    isolation_ratio=None,
                    bhattacharyya_dist=None,
                ))
                seen_frames.add(fi)
            continue  # Tier 1 fires unconditionally, skip lower tiers

        # --- Tier 2: Kinematic spike with isolation ---
        # Absolute floor: spike ratio only fires when speed >= spike_min_speed_mps.
        # This avoids triggering on near-zero-median tracklets where any movement
        # looks like a 100x spike.
        if speed >= cfg.spike_min_speed_mps and median_speed > 0:
            spike_ratio = speed / median_speed
        else:
            spike_ratio = 0.0

        if spike_ratio >= cfg.spike_speed_ratio:
            # Compute isolation ratio: speed[N] / max(speed[N-2], speed[N+2])
            neighbors = []
            for offset in [-2, 2]:
                ni = i + offset
                if 0 <= ni < len(speeds) and np.isfinite(speeds[ni]):
                    neighbors.append(float(speeds[ni]))
            if not neighbors:
                # Fall back to ±1
                for offset in [-1, 1]:
                    ni = i + offset
                    if 0 <= ni < len(speeds) and np.isfinite(speeds[ni]):
                        neighbors.append(float(speeds[ni]))

            max_neighbor = max(neighbors) if neighbors else 1e-6
            isolation = speed / max(max_neighbor, 1e-6)

            if isolation >= cfg.spike_isolation_ratio:
                # Check duration: count consecutive elevated frames from i
                elevated_count = 0
                threshold_2x = 2.0 * median_speed if median_speed > 0 else float("inf")
                for j in range(i, len(speeds)):
                    if np.isfinite(speeds[j]) and speeds[j] >= threshold_2x:
                        elevated_count += 1
                    else:
                        break

                if elevated_count <= cfg.spike_max_duration_frames:
                    if fi not in seen_frames:
                        candidates.append(SplitCandidate(
                            frame_index=fi,
                            tier="tier2_kinematic_spike",
                            speed_at_frame=speed,
                            isolation_ratio=isolation,
                            bhattacharyya_dist=None,
                        ))
                        seen_frames.add(fi)
                    continue

        # --- Tier 3: Histogram discontinuity ---
        if fi in hist_by_frame:
            # Find most recent prior frame with histogram in this tracklet
            prev_hist_frame = None
            for pi in range(i - 1, -1, -1):
                pf = int(frames[pi])
                if pf in hist_by_frame:
                    prev_hist_frame = pf
                    break

            if prev_hist_frame is not None:
                h_curr = hist_by_frame[fi]
                h_prev = hist_by_frame[prev_hist_frame]
                # Bhattacharyya distance: 1 - sum(sqrt(h1 * h2))
                bc = float(np.sum(np.sqrt(np.maximum(h_curr * h_prev, 0.0))))
                bhatt = 1.0 - bc

                if bhatt > cfg.histogram_bhattacharyya_threshold:
                    if cfg.histogram_require_kinematic_corroboration:
                        corroboration_ratio = speed / max(median_speed, 1e-6)
                        if corroboration_ratio < cfg.histogram_kinematic_corroboration_ratio:
                            continue  # No kinematic corroboration, skip
                    if fi not in seen_frames:
                        candidates.append(SplitCandidate(
                            frame_index=fi,
                            tier="tier3_histogram",
                            speed_at_frame=speed,
                            isolation_ratio=None,
                            bhattacharyya_dist=bhatt,
                        ))
                        seen_frames.add(fi)

    return candidates


# ---------------------------------------------------------------------------
# Dwell filtering
# ---------------------------------------------------------------------------

def apply_dwell_filter(
    candidates: List[SplitCandidate],
    tracklet_start_frame: int,
    tracklet_end_frame: int,
    min_dwell: int,
) -> List[SplitCandidate]:
    """Remove candidates that would create segments shorter than min_dwell."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c.frame_index)
    filtered: List[SplitCandidate] = []

    # Build boundary list: [tracklet_start, c1, c2, ..., tracklet_end+1]
    boundaries = [tracklet_start_frame] + [c.frame_index for c in sorted_cands] + [tracklet_end_frame + 1]

    for idx, cand in enumerate(sorted_cands):
        pre_start = boundaries[idx]
        pre_end = cand.frame_index  # exclusive (pre-segment is [pre_start, cand.frame_index))
        post_start = cand.frame_index
        post_end = boundaries[idx + 2]  # exclusive

        pre_length = pre_end - pre_start
        post_length = post_end - post_start

        if pre_length >= min_dwell and post_length >= min_dwell:
            filtered.append(cand)

    return filtered


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def apply_splits(
    bank_frames: pd.DataFrame,
    all_split_points: List[Tuple[str, SplitCandidate]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Apply splits to bank_frames, returning modified DataFrame and audit events.

    split_points: list of (tracklet_id, SplitCandidate) pairs.
    """
    if not all_split_points:
        return bank_frames, []

    # Group split points by tracklet
    splits_by_tid: Dict[str, List[SplitCandidate]] = {}
    for tid, cand in all_split_points:
        splits_by_tid.setdefault(tid, []).append(cand)

    # Sort each tracklet's splits by frame
    for tid in splits_by_tid:
        splits_by_tid[tid].sort(key=lambda c: c.frame_index)

    audit_events: List[Dict[str, Any]] = []
    df = bank_frames.copy()

    for tid, cands in sorted(splits_by_tid.items()):
        mask = df["tracklet_id"] == tid
        tid_indices = df.index[mask]
        if tid_indices.empty:
            continue

        tid_frames = df.loc[tid_indices, "frame_index"].values

        for seq, cand in enumerate(cands, start=1):
            new_tid = f"{tid}_s{seq}"
            # All frames >= split frame that still have the current tid get the new tid
            # (previous splits in this tracklet already changed those frames' tids)
            current_tid = tid if seq == 1 else f"{tid}_s{seq - 1}"
            update_mask = (df["tracklet_id"] == current_tid) & (
                df["frame_index"] >= cand.frame_index
            )
            update_indices = df.index[update_mask]

            if update_indices.empty:
                continue

            # Compute segment sizes for audit
            keep_mask = (df["tracklet_id"] == current_tid) & (
                df["frame_index"] < cand.frame_index
            )
            pre_frames = int(keep_mask.sum())
            post_frames = int(len(update_indices))

            df.loc[update_indices, "tracklet_id"] = new_tid

            audit_events.append({
                "artifact_type": "d05_split_event",
                "original_tracklet_id": tid,
                "new_tracklet_id": new_tid,
                "split_frame": int(cand.frame_index),
                "tier": cand.tier,
                "speed_at_frame": round(cand.speed_at_frame, 4),
                "isolation_ratio": (
                    round(cand.isolation_ratio, 4)
                    if cand.isolation_ratio is not None else None
                ),
                "bhattacharyya_dist": (
                    round(cand.bhattacharyya_dist, 6)
                    if cand.bhattacharyya_dist is not None else None
                ),
                "pre_segment_frames": pre_frames,
                "post_segment_frames": post_frames,
            })

    return df, audit_events


# ---------------------------------------------------------------------------
# Summary rebuild
# ---------------------------------------------------------------------------

def _hint_frame(record: Dict[str, Any]) -> Optional[int]:
    """Extract first_seen_frame from an identity hint record."""
    ev = record.get("evidence", {})
    if isinstance(ev, dict):
        fsf = ev.get("first_seen_frame")
        if fsf is not None:
            try:
                return int(fsf)
            except (ValueError, TypeError):
                pass
    return None


def rebuild_summaries(
    bank_frames: pd.DataFrame,
    original_summaries: pd.DataFrame,
    identity_hints_records: List[Dict[str, Any]],
    split_parent_map: Dict[str, str],
) -> pd.DataFrame:
    """Rebuild tracklet_bank_summaries after splitting.

    split_parent_map: maps new_tracklet_id -> original_tracklet_id
        (only for split tracklets, unsplit tracklets are not in the map).
    """
    # Unsplit tracklets: keep original summary rows unchanged
    split_tids = set(split_parent_map.keys())
    # Also identify original tids that were split (their summary needs replacing)
    split_originals = set(split_parent_map.values())

    unsplit = original_summaries[
        ~original_summaries["tracklet_id"].isin(split_originals)
    ].copy()

    # Build new rows for split segments
    new_rows: List[Dict[str, Any]] = []

    # Pre-index original summaries by tracklet_id for parent data
    orig_by_tid = original_summaries.set_index("tracklet_id")

    # Group identity hints by original tracklet_id
    hints_by_orig_tid: Dict[str, List[Dict[str, Any]]] = {}
    for rec in identity_hints_records:
        tid = str(rec.get("tracklet_id", ""))
        if tid:
            hints_by_orig_tid.setdefault(tid, []).append(rec)

    # Get frame ranges for each new tracklet from bank_frames
    split_tracklets = bank_frames[
        bank_frames["tracklet_id"].isin(split_tids | split_originals)
    ]
    grouped = split_tracklets.groupby("tracklet_id")

    for new_tid, grp in grouped:
        # Determine parent tid
        if new_tid in split_parent_map:
            parent_tid = split_parent_map[new_tid]
        elif new_tid in split_originals:
            parent_tid = new_tid
        else:
            continue

        # Get parent summary for inherited fields
        if parent_tid in orig_by_tid.index:
            parent = orig_by_tid.loc[parent_tid]
        else:
            continue

        start_frame = int(grp["frame_index"].min())
        end_frame = int(grp["frame_index"].max())
        n_frames = len(grp)

        row: Dict[str, Any] = {
            "clip_id": parent.get("clip_id", None),
            "camera_id": parent.get("camera_id", None),
            "tracklet_id": str(new_tid),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            # Inherited from parent (approximate but unused by D1)
            "mean_x1": parent.get("mean_x1", np.nan),
            "mean_y1": parent.get("mean_y1", np.nan),
            "mean_x2": parent.get("mean_x2", np.nan),
            "mean_y2": parent.get("mean_y2", np.nan),
            "quality_score": parent.get("quality_score", np.nan),
            "reason_codes_json": parent.get("reason_codes_json", None),
            "n_occlusion_spans": parent.get("n_occlusion_spans", 0),
            "n_repaired_frames": parent.get("n_repaired_frames", 0),
            "min_nn_dist_m_at_anchors": parent.get("min_nn_dist_m_at_anchors", np.nan),
            "mean_nn_dist_m_at_anchors": parent.get("mean_nn_dist_m_at_anchors", np.nan),
            "min_tracks_within_r_at_anchors": parent.get("min_tracks_within_r_at_anchors", np.nan),
            "mean_tracks_within_r_at_anchors": parent.get("mean_tracks_within_r_at_anchors", np.nan),
            "n_spans_with_plausible_other_candidate": parent.get("n_spans_with_plausible_other_candidate", 0),
        }

        # Re-derive identity hints for this segment's frame range
        # Hints with first_seen_frame >= split_frame go to the post-split segment
        # (consistent with apply_splits where frame_index >= split_frame → new segment)
        parent_hints = hints_by_orig_tid.get(parent_tid, [])
        segment_hints = []
        for h in parent_hints:
            fsf = _hint_frame(h)
            if fsf is not None:
                if start_frame <= fsf <= end_frame:
                    segment_hints.append(h)
            else:
                # No frame info — assign to first segment only (original tid)
                if new_tid == parent_tid:
                    segment_hints.append(h)

        if segment_hints:
            row["identity_hints_json"] = json.dumps(
                sorted(segment_hints, key=lambda r: json.dumps(r, sort_keys=True)),
                sort_keys=True,
            )
            row["must_link_anchor_key"] = None
            row["must_link_confidence"] = float("nan")
            row["cannot_link_anchor_keys_json"] = None
        else:
            row["identity_hints_json"] = None
            row["must_link_anchor_key"] = None
            row["must_link_confidence"] = float("nan")
            row["cannot_link_anchor_keys_json"] = None

        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        result = pd.concat([unsplit, new_df], ignore_index=True)
    else:
        result = unsplit

    # Ensure consistent dtypes
    result["must_link_confidence"] = result["must_link_confidence"].astype("float64")
    result = result.sort_values("tracklet_id", kind="mergesort").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_audit(
    audit_path: Path,
    cfg: SplitConfig,
    speed_cap: float,
    n_tracklets_input: int,
    n_tracklets_output: int,
    candidates_before: int,
    candidates_after: int,
    tier_counts: Dict[str, int],
    split_events: List[Dict[str, Any]],
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "artifact_type": "d05_split_summary",
        "timestamp": _now_ms(),
        "n_tracklets_input": n_tracklets_input,
        "n_tracklets_with_candidates": len(
            {e["original_tracklet_id"] for e in split_events}
        ) if split_events else 0,
        "n_candidates_before_dwell_filter": candidates_before,
        "n_candidates_after_dwell_filter": candidates_after,
        "n_splits_applied": len(split_events),
        "n_tracklets_output": n_tracklets_output,
        "n_new_tracklets_created": n_tracklets_output - n_tracklets_input,
        "tier_counts": tier_counts,
        "config": {
            "enabled": cfg.enabled,
            "speed_cap_mps": speed_cap,
            "spike_speed_ratio": cfg.spike_speed_ratio,
            "spike_min_speed_mps": cfg.spike_min_speed_mps,
            "spike_isolation_ratio": cfg.spike_isolation_ratio,
            "spike_max_duration_frames": cfg.spike_max_duration_frames,
            "histogram_bhattacharyya_threshold": cfg.histogram_bhattacharyya_threshold,
            "histogram_require_kinematic_corroboration": cfg.histogram_require_kinematic_corroboration,
            "histogram_kinematic_corroboration_ratio": cfg.histogram_kinematic_corroboration_ratio,
            "min_dwell_frames": cfg.min_dwell_frames,
        },
    }

    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
        for ev in split_events:
            ev["timestamp"] = _now_ms()
            f.write(json.dumps(ev, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_d05_split(
    *,
    config: Dict[str, Any],
    layout: Any,
    manifest: Any,
) -> None:
    """D0.5: Post-D0 tracklet splitting.

    Reads: stage_D/tracklet_bank_frames.parquet (D0 output)
           stage_D/tracklet_bank_summaries.parquet (D0 output)
           stage_A/color_histograms.parquet (optional)
           stage_C/identity_hints.jsonl (for hint re-derivation)
    Writes: stage_D/tracklet_bank_frames.parquet (overwrite)
            stage_D/tracklet_bank_summaries.parquet (overwrite)
            stage_D/d05_split_audit.jsonl
    """
    cfg, speed_cap = _resolve_config(config)
    audit_path = Path(layout.stage_dir("D")) / "d05_split_audit.jsonl"

    if not cfg.enabled:
        logger.info("D0.5 splitter disabled by config")
        if cfg.write_audit:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "artifact_type": "d05_split_skipped",
                    "timestamp": _now_ms(),
                    "reason": "enabled=false",
                }, sort_keys=True) + "\n")
        return

    # Load artifacts
    bf_path = Path(layout.tracklet_bank_frames_parquet())
    bs_path = Path(layout.tracklet_bank_summaries_parquet())
    if not bf_path.exists() or not bs_path.exists():
        logger.warning("D0.5: bank tables not found, skipping")
        return

    bank_frames = pd.read_parquet(bf_path)
    bank_summaries = pd.read_parquet(bs_path)
    n_tracklets_input = bank_frames["tracklet_id"].nunique()

    # Load histograms (optional)
    hist_path = Path(layout.color_histograms_parquet())
    histograms: Optional[pd.DataFrame] = None
    if hist_path.exists():
        histograms = pd.read_parquet(hist_path)
        # Rename track_id -> tracklet_id for consistent join
        if "track_id" in histograms.columns and "tracklet_id" not in histograms.columns:
            histograms = histograms.rename(columns={"track_id": "tracklet_id"})

    # Load identity hints for summary rebuild
    ih_path = Path(layout.identity_hints_jsonl())
    ih_records: List[Dict[str, Any]] = []
    if ih_path.exists():
        text = ih_path.read_text(encoding="utf-8")
        ih_records = [json.loads(line) for line in text.splitlines() if line.strip()]

    logger.info(
        "D0.5 splitter: %d tracklets, speed_cap=%.1f m/s",
        n_tracklets_input, speed_cap,
    )

    # Detect swap boundaries per tracklet
    all_candidates: List[Tuple[str, SplitCandidate]] = []
    total_before_dwell = 0

    # Sort for determinism
    sorted_tids = sorted(bank_frames["tracklet_id"].unique())

    for tid in sorted_tids:
        tid_mask = bank_frames["tracklet_id"] == tid
        tf = bank_frames.loc[tid_mask].sort_values("frame_index")

        if len(tf) < 2:
            continue

        # Get histograms for this tracklet
        tid_hist = None
        if histograms is not None:
            h_mask = histograms["tracklet_id"] == tid
            tid_hist = histograms.loc[h_mask]
            if tid_hist.empty:
                tid_hist = None

        candidates = detect_swap_boundaries(tf, tid_hist, cfg, speed_cap)
        total_before_dwell += len(candidates)

        if candidates:
            start_frame = int(tf["frame_index"].min())
            end_frame = int(tf["frame_index"].max())
            filtered = apply_dwell_filter(
                candidates, start_frame, end_frame, cfg.min_dwell_frames
            )
            for c in filtered:
                all_candidates.append((str(tid), c))

    logger.info(
        "D0.5: %d candidates before dwell filter, %d after",
        total_before_dwell, len(all_candidates),
    )

    if not all_candidates:
        logger.info("D0.5: no splits to apply")
        if cfg.write_audit:
            _write_audit(
                audit_path, cfg, speed_cap,
                n_tracklets_input=n_tracklets_input,
                n_tracklets_output=n_tracklets_input,
                candidates_before=total_before_dwell,
                candidates_after=0,
                tier_counts={},
                split_events=[],
            )
        return

    # Apply splits
    modified_frames, split_events = apply_splits(bank_frames, all_candidates)
    n_tracklets_output = modified_frames["tracklet_id"].nunique()

    # Build parent map for summary rebuild
    split_parent_map: Dict[str, str] = {}
    for ev in split_events:
        split_parent_map[ev["new_tracklet_id"]] = ev["original_tracklet_id"]

    # Rebuild summaries
    modified_summaries = rebuild_summaries(
        modified_frames, bank_summaries, ih_records, split_parent_map,
    )

    # Overwrite bank tables
    modified_frames.to_parquet(bf_path, index=False)
    modified_summaries.to_parquet(bs_path, index=False)

    logger.info(
        "D0.5: split %d tracklets -> %d new segments (%d -> %d total)",
        len({ev["original_tracklet_id"] for ev in split_events}),
        len(split_events),
        n_tracklets_input,
        n_tracklets_output,
    )

    # Audit
    if cfg.write_audit:
        tier_counts: Dict[str, int] = {}
        for ev in split_events:
            tier_counts[ev["tier"]] = tier_counts.get(ev["tier"], 0) + 1
        _write_audit(
            audit_path, cfg, speed_cap,
            n_tracklets_input=n_tracklets_input,
            n_tracklets_output=n_tracklets_output,
            candidates_before=total_before_dwell,
            candidates_after=len(all_candidates),
            tier_counts=tier_counts,
            split_events=split_events,
        )
