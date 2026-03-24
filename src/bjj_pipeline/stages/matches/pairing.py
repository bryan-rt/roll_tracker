"""Role: pairing logic using world-space proximity.

Computes per-frame pairwise Euclidean distances between all person_id pairs
in world coordinates (meters).
"""
from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

_SCHEMA_COLS = ["frame_index", "person_id_a", "person_id_b", "dist_m"]


def _empty_pair_distances() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCHEMA_COLS).astype(
        {"frame_index": "int64", "person_id_a": "str", "person_id_b": "str", "dist_m": "float64"}
    )


def compute_pair_distances(
    person_tracks_df: pd.DataFrame,
    *,
    fps: float,
) -> pd.DataFrame:
    """Compute per-frame pairwise distances between all person_id pairs.

    Coordinate priority: x_m_repaired/y_m_repaired if present and non-null,
    fallback to x_m/y_m.

    Returns DataFrame with columns: frame_index, person_id_a, person_id_b, dist_m
    One row per frame per unordered pair (a < b lexicographically).
    Only emits rows where both persons have valid coords at that frame.
    """
    if person_tracks_df is None or person_tracks_df.empty:
        return _empty_pair_distances()

    if "person_id" not in person_tracks_df.columns or "frame_index" not in person_tracks_df.columns:
        return _empty_pair_distances()

    df = person_tracks_df.copy()

    # Resolve effective x/y with repaired-first fallback
    if "x_m_repaired" in df.columns:
        df["_x"] = df["x_m_repaired"].fillna(df.get("x_m", np.nan))
    elif "x_m" in df.columns:
        df["_x"] = df["x_m"]
    else:
        return _empty_pair_distances()

    if "y_m_repaired" in df.columns:
        df["_y"] = df["y_m_repaired"].fillna(df.get("y_m", np.nan))
    elif "y_m" in df.columns:
        df["_y"] = df["y_m"]
    else:
        return _empty_pair_distances()

    # Drop rows with invalid coords
    df = df.dropna(subset=["_x", "_y"])
    if df.empty:
        return _empty_pair_distances()

    persons = sorted(df["person_id"].unique())
    if len(persons) < 2:
        return _empty_pair_distances()

    # Build per-frame lookup: frame_index → {person_id: (x, y)}
    df["frame_index"] = df["frame_index"].astype(int)
    df["person_id"] = df["person_id"].astype(str)

    rows: List[dict] = []
    for frame_idx, grp in df.groupby("frame_index", sort=True):
        # persons present at this frame
        coords = {}
        for _, r in grp.iterrows():
            pid = r["person_id"]
            coords[pid] = (float(r["_x"]), float(r["_y"]))

        pids = sorted(coords.keys())
        if len(pids) < 2:
            continue

        for pa, pb in combinations(pids, 2):
            xa, ya = coords[pa]
            xb, yb = coords[pb]
            dist = float(np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2))
            rows.append({
                "frame_index": int(frame_idx),
                "person_id_a": pa,
                "person_id_b": pb,
                "dist_m": dist,
            })

    if not rows:
        return _empty_pair_distances()

    result = pd.DataFrame(rows, columns=_SCHEMA_COLS)
    result = result.sort_values(["frame_index", "person_id_a", "person_id_b"]).reset_index(drop=True)
    return result
