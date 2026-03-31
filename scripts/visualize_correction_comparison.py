#!/usr/bin/env python3
"""Diagnostic: before/after position visualization for CP18 correction comparison.

Produces per-camera scatter plots of Stage A tracklet positions overlaid on
the mat blueprint, plus a difference view highlighting positions that changed
on-mat status between baseline and corrected runs.

Usage:
    python scripts/visualize_correction_comparison.py \
        --baseline-dir outputs/c8a592a4-2bca-400a-80e1-fec0e5cbea77 \
        --corrected-dir outputs_corrected_h/c8a592a4-2bca-400a-80e1-fec0e5cbea77 \
        --blueprint configs/mat_blueprint.json \
        --cameras FP7oJQ J_EDEw PPDmUg \
        --output-dir calibration_results/diagnostics
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Blueprint helpers
# ---------------------------------------------------------------------------

def load_blueprint(path: Path) -> list[dict]:
    """Load mat blueprint JSON → list of {x_min, y_min, x_max, y_max}."""
    panels = json.loads(path.read_text(encoding="utf-8"))
    rects = []
    for p in panels:
        rects.append({
            "x_min": p["x"],
            "y_min": p["y"],
            "x_max": p["x"] + p["width"],
            "y_max": p["y"] + p["height"],
        })
    return rects


def point_in_blueprint(x: np.ndarray, y: np.ndarray, rects: list[dict]) -> np.ndarray:
    """Vectorised on-mat check: True if (x, y) inside any rectangle."""
    mask = np.zeros(len(x), dtype=bool)
    for r in rects:
        mask |= (x >= r["x_min"]) & (x <= r["x_max"]) & (y >= r["y_min"]) & (y <= r["y_max"])
    return mask


def blueprint_bounds(rects: list[dict], pad: float = 5.0):
    """Return (x_lo, x_hi, y_lo, y_hi) with padding (in feet)."""
    # Convert feet padding to metres (~0.3048 m/ft)
    pad_m = pad * 0.3048
    x_lo = min(r["x_min"] for r in rects) - pad_m
    x_hi = max(r["x_max"] for r in rects) + pad_m
    y_lo = min(r["y_min"] for r in rects) - pad_m
    y_hi = max(r["y_max"] for r in rects) + pad_m
    return x_lo, x_hi, y_lo, y_hi


def draw_blueprint(ax, rects: list[dict]):
    """Draw mat panels as gray filled rectangles with black edges."""
    for r in rects:
        rect = plt.Rectangle(
            (r["x_min"], r["y_min"]),
            r["x_max"] - r["x_min"],
            r["y_max"] - r["y_min"],
            facecolor="#d0d0d0",
            edgecolor="black",
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracklet_frames(run_dir: Path, cam: str) -> pd.DataFrame:
    """Concatenate all per-clip Stage A tracklet_frames for a camera."""
    files = sorted(run_dir.rglob(f"{cam}/**/stage_A/tracklet_frames.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f, columns=["x_m", "y_m", "on_mat"]) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_person_tracks(run_dir: Path, cam: str) -> pd.DataFrame:
    """Load session-level Stage D person_tracks for a camera."""
    session_root = run_dir / "sessions"
    if not session_root.exists():
        return pd.DataFrame()
    files = sorted(session_root.rglob(f"stage_D/person_tracks_{cam}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f, columns=["person_id", "tracklet_id"]) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot 1: Side-by-side before/after
# ---------------------------------------------------------------------------

def plot_side_by_side(
    cam: str,
    df_base: pd.DataFrame,
    df_corr: pd.DataFrame,
    rects: list[dict],
    output_dir: Path,
):
    bounds = blueprint_bounds(rects)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    for ax, df, label in [(ax_l, df_base, "Baseline"), (ax_r, df_corr, "Corrected")]:
        draw_blueprint(ax, rects)
        on = df["on_mat"].fillna(False)
        off = ~on
        ax.scatter(df.loc[on, "x_m"], df.loc[on, "y_m"], c="green", s=0.3, alpha=0.15, zorder=2, rasterized=True)
        ax.scatter(df.loc[off, "x_m"], df.loc[off, "y_m"], c="red", s=0.3, alpha=0.3, zorder=3, rasterized=True)
        pct = on.mean() * 100
        ax.set_title(f"{label} — {cam}\n{len(df):,} positions, {pct:.1f}% on-mat", fontsize=11)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x_m")
        ax.set_ylabel("y_m")

    fig.tight_layout()
    out = output_dir / f"{cam}_position_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Difference view
# ---------------------------------------------------------------------------

def plot_diff(
    cam: str,
    df_base: pd.DataFrame,
    df_corr: pd.DataFrame,
    rects: list[dict],
    output_dir: Path,
):
    # Align by index — both come from the same clips in the same order.
    # But row counts can differ if Stage A produced different tracklet counts.
    # Use the on_mat flag recomputed from blueprint for consistency.
    on_b = point_in_blueprint(df_base["x_m"].values, df_base["y_m"].values, rects)
    on_c = point_in_blueprint(df_corr["x_m"].values, df_corr["y_m"].values, rects)

    # If row counts match, do element-wise diff; otherwise use corrected positions
    # with corrected on_mat vs baseline on_mat recomputed on same positions.
    if len(df_base) == len(df_corr):
        x, y = df_corr["x_m"].values, df_corr["y_m"].values
        stayed_on = on_b & on_c
        moved_off = on_b & ~on_c    # regression
        moved_on = ~on_b & on_c     # improvement
        stayed_off = ~on_b & ~on_c
    else:
        # Row counts differ — show corrected positions only, classify by their on_mat
        print(f"  Warning: row counts differ ({len(df_base)} vs {len(df_corr)}), showing corrected only")
        x, y = df_corr["x_m"].values, df_corr["y_m"].values
        stayed_on = on_c
        moved_off = np.zeros(len(df_corr), dtype=bool)
        moved_on = np.zeros(len(df_corr), dtype=bool)
        stayed_off = ~on_c

    bounds = blueprint_bounds(rects)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    draw_blueprint(ax, rects)

    # Draw in order: least interesting first
    layers = [
        (stayed_on, "green", "Stayed on-mat", 0.08),
        (stayed_off, "royalblue", "Stayed off-mat", 0.4),
        (moved_on, "orange", "Moved ON-mat (improvement)", 0.6),
        (moved_off, "red", "Moved OFF-mat (regression)", 0.6),
    ]
    handles = []
    for mask, color, label, alpha in layers:
        n = mask.sum()
        pct = n / len(x) * 100 if len(x) else 0
        full_label = f"{label}: {n:,} ({pct:.1f}%)"
        if n > 0:
            ax.scatter(x[mask], y[mask], c=color, s=0.5, alpha=alpha, zorder=2 + len(handles), rasterized=True)
        handles.append(mpatches.Patch(color=color, label=full_label))

    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"Position Diff — {cam}\n({len(x):,} positions)", fontsize=12)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")

    fig.tight_layout()
    out = output_dir / f"{cam}_position_diff.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    return {
        "stayed_on": int(stayed_on.sum()),
        "stayed_off": int(stayed_off.sum()),
        "moved_off": int(moved_off.sum()),
        "moved_on": int(moved_on.sum()),
        "total": len(x),
    }


# ---------------------------------------------------------------------------
# Stage D identity collapse
# ---------------------------------------------------------------------------

def identity_collapse_stats(run_dir: Path, cam: str) -> dict:
    df = load_person_tracks(run_dir, cam)
    if df.empty:
        return {}
    n_persons = df["person_id"].nunique()
    n_tracklets = df["tracklet_id"].nunique()
    ratio = n_tracklets / max(1, n_persons)
    max_per = df.groupby("person_id")["tracklet_id"].nunique().max()
    return {
        "persons": n_persons,
        "tracklets": n_tracklets,
        "ratio": ratio,
        "max_per": int(max_per),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CP18 correction comparison visualisation")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, default=Path("configs/mat_blueprint.json"))
    parser.add_argument("--cameras", nargs="+", default=["FP7oJQ", "J_EDEw", "PPDmUg"])
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_results/diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rects = load_blueprint(args.blueprint)

    print("=" * 60)
    print("CP18 Correction Comparison")
    print("=" * 60)

    # --- Position comparison ---
    all_diff_stats = {}
    for cam in args.cameras:
        print(f"\n--- {cam} ---")
        df_base = load_tracklet_frames(args.baseline_dir, cam)
        df_corr = load_tracklet_frames(args.corrected_dir, cam)

        if df_base.empty or df_corr.empty:
            print(f"  Skipping {cam}: missing data (base={len(df_base)}, corr={len(df_corr)})")
            continue

        on_b = df_base["on_mat"].fillna(False).mean() * 100
        on_c = df_corr["on_mat"].fillna(False).mean() * 100
        print(f"  Baseline:  {len(df_base):,} positions, {on_b:.1f}% on-mat")
        print(f"  Corrected: {len(df_corr):,} positions, {on_c:.1f}% on-mat")

        plot_side_by_side(cam, df_base, df_corr, rects, args.output_dir)
        diff = plot_diff(cam, df_base, df_corr, rects, args.output_dir)
        all_diff_stats[cam] = diff

        if diff["total"] > 0:
            t = diff["total"]
            print(f"  Stayed on-mat:  {diff['stayed_on']:,} ({diff['stayed_on']/t*100:.1f}%)")
            print(f"  Moved OFF-mat:  {diff['moved_off']:,} ({diff['moved_off']/t*100:.1f}%)  <-- regression")
            print(f"  Moved ON-mat:   {diff['moved_on']:,} ({diff['moved_on']/t*100:.1f}%)  <-- improvement")
            print(f"  Stayed off-mat: {diff['stayed_off']:,} ({diff['stayed_off']/t*100:.1f}%)")

    # --- Stage D identity collapse ---
    print("\n" + "=" * 60)
    print("Stage D Stitching Comparison")
    print("=" * 60)
    print(f"{'Camera':<8} {'Baseline':<28} {'Corrected':<28}")
    print(f"{'':8} {'persons/tracklets (ratio)':<28} {'persons/tracklets (ratio)':<28}")
    print("-" * 64)
    for cam in args.cameras:
        b = identity_collapse_stats(args.baseline_dir, cam)
        c = identity_collapse_stats(args.corrected_dir, cam)
        b_str = f"{b['persons']}/{b['tracklets']} ({b['ratio']:.1f}/p, max {b['max_per']})" if b else "N/A"
        c_str = f"{c['persons']}/{c['tracklets']} ({c['ratio']:.1f}/p, max {c['max_per']})" if c else "N/A"
        print(f"{cam:<8} {b_str:<28} {c_str:<28}")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
