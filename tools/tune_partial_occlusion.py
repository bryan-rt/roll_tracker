from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe for servers
    import matplotlib.pyplot as plt
except ImportError:  # optional dependency
    matplotlib = None
    plt = None


def load_json_tolerant(path: Path):
    txt = path.read_text(encoding="utf-8")
    # fix the common "}{\n{" missing comma pattern
    txt2 = re.sub(r"}\s*\n\s*{", "},\n{", txt)
    return json.loads(txt2)


@dataclass(frozen=True)
class Span:
    camera_id: str
    clip_id: str
    tracklet_id: str
    start_frame: int
    end_frame: int


def spans_to_frame_set(spans: List[Span]) -> Dict[Tuple[str, str], set]:
    """
    Returns dict keyed by (clip_id, tracklet_id) -> set(frame_index)
    """
    out: Dict[Tuple[str, str], set] = {}
    for s in spans:
        k = (s.clip_id, s.tracklet_id)
        out.setdefault(k, set()).update(range(int(s.start_frame), int(s.end_frame) + 1))
    return out


def compute_bbox_signals(det: pd.DataFrame) -> pd.DataFrame:
    # Expect: tracklet_id, frame_index, y1, y2
    det = det.copy()
    # Ensure float for robust median/div
    det["y1"] = det["y1"].astype(float)
    det["y2"] = det["y2"].astype(float)
    det["h"] = (det["y2"] - det["y1"]).astype(float)
    det = det.sort_values(["tracklet_id", "frame_index"], kind="mergesort").reset_index(drop=True)

    # framewise deltas
    det["dy2"] = det.groupby("tracklet_id")["y2"].diff().abs()
    det["dy1"] = det.groupby("tracklet_id")["y1"].diff().abs()
    det["dh"]  = det.groupby("tracklet_id")["h"].diff().abs()

    # ratio features (normalize by prev height to reduce scale effects)
    hprev = det.groupby("tracklet_id")["h"].shift(1).replace(0, np.nan)
    det["r_bottom"] = ((det["dy2"] - det["dy1"]).clip(lower=0.0) / hprev).fillna(0.0)
    det["r_height"] = (det["dh"] / hprev).fillna(0.0)

    return det


def detect_spans_linker2(
    det_sig: pd.DataFrame,
    *,
    onset_window: int,
    min_bottom_frac: float,
    min_height_frac: float,
    onset_min_frames: int,
    recover_bottom_frac: float,
    recover_height_frac: float,
    recover_min_frames: int,
    min_window_frames: int,
    dy2_px_min: float,
    gate_onset_with_dy2: bool = True,
    max_span_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Return det_sig with pred_span_active + (rb_base,rh_base) using linker_2 semantics.

        This mirrors roll_it_back/linker_2 copy/stages/occlusion.py::_find_occlusion_windows:
            - baseline queues are trailing medians over onset_window
            - baseline queues are trailing medians over onset_window
            - onset requires (r_bottom>=min_bottom_frac AND r_height>=min_height_frac)
                sustained as: recent_flags.count(True) >= onset_min_frames in a full onset_window
            - optional onset gate: dy2 >= dy2_px_min (helps reject bbox jitter)
            - once a span starts, baseline is frozen (occ_b0/occ_h0) until recovery
            - continuation uses OR; recovery uses AND-for-N frames

        Recovery semantics: recovery frames increment the recovery streak. In this tuning tool,
        we keep recovery frames active and extend the span tail (recall-first).
    """
    df = det_sig.copy()
    df = df.sort_values(["tracklet_id", "frame_index"], kind="mergesort").reset_index(drop=True)
    df["pred_span_active"] = False
    df["rb_base"] = 0.0
    df["rh_base"] = 0.0

    ow = int(onset_window) if onset_window and int(onset_window) > 0 else 1
    omf = int(onset_min_frames) if onset_min_frames and int(onset_min_frames) > 0 else 1
    omf = min(omf, ow)
    rmin = int(recover_min_frames) if recover_min_frames and int(recover_min_frames) > 0 else 1

    def _scan(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        y2 = g["y2"].astype(float).to_numpy()
        h = g["h"].astype(float).to_numpy()
        dy2 = g["dy2"].fillna(0.0).astype(float).to_numpy()

        n = len(g)
        active = np.zeros(n, dtype=bool)
        rb_out = np.zeros(n, dtype=float)
        rh_out = np.zeros(n, dtype=float)

        baseline_bottom_q: deque = deque(maxlen=ow)
        baseline_height_q: deque = deque(maxlen=ow)
        recent_flags: deque = deque(maxlen=ow)

        in_occ = False
        start_i: Optional[int] = None
        occ_b0: Optional[float] = None
        occ_h0: Optional[float] = None
        recover_streak = 0
        last_occ_i: Optional[int] = None

        def _median(q: deque) -> Optional[float]:
            if not q:
                return None
            xs = sorted(q)
            m = xs[len(xs) // 2]
            # for even lengths, match Python statistics.median behavior (avg of mid two)
            if len(xs) % 2 == 0:
                m = 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
            return float(m)

        def _current_baseline() -> Tuple[Optional[float], Optional[float]]:
            b0 = _median(baseline_bottom_q)
            h0 = _median(baseline_height_q)
            if b0 is None or h0 is None:
                return None, None
            if h0 <= 0.0:
                return b0, None
            return b0, h0

        for i in range(n):
            b = float(y2[i]) if np.isfinite(y2[i]) else None
            hh = float(h[i]) if np.isfinite(h[i]) else None
            b0, h0 = _current_baseline()

            # mimic linker_2: invalid bbox/height handling
            if b is None or hh is None or hh <= 0.0 or b0 is None or h0 is None:
                if (not in_occ) and (b is not None) and (hh is not None) and (hh > 0.0):
                    baseline_bottom_q.append(b)
                    baseline_height_q.append(hh)
                elif in_occ:
                    recover_streak = 0
                    last_occ_i = i
                    active[i] = True
                continue

            if in_occ and occ_b0 is not None and occ_h0 is not None:
                base_b = occ_b0
                base_h = occ_h0
            else:
                base_b = b0
                base_h = h0

            if base_h <= 0.0:
                if in_occ:
                    recover_streak = 0
                    last_occ_i = i
                    active[i] = True
                continue

            r_bottom = (base_b - b) / base_h
            r_height = (base_h - hh) / base_h
            if r_bottom < 0.0:
                r_bottom = 0.0
            if r_height < 0.0:
                r_height = 0.0

            rb_out[i] = float(r_bottom)
            rh_out[i] = float(r_height)

            if not in_occ:
                is_candidate = (r_bottom >= float(min_bottom_frac) and r_height >= float(min_height_frac))
                if gate_onset_with_dy2:
                    is_candidate = bool(is_candidate and (float(dy2[i]) >= float(dy2_px_min)))
                recent_flags.append(bool(is_candidate))
                if len(recent_flags) == ow and sum(1 for x in recent_flags if x) >= omf:
                    first_candidate_offset = None
                    for k in range(ow):
                        if recent_flags[k]:
                            first_candidate_offset = k
                            break
                    if first_candidate_offset is None:
                        continue
                    # i is the current index; window covers [i-ow+1, i]
                    # first_candidate_offset is offset within that window.
                    start_i = max(0, (i - ow + 1) + int(first_candidate_offset))
                    in_occ = True
                    recover_streak = 0
                    occ_b0, occ_h0 = b0, h0
                    last_occ_i = i
                    active[i] = True
                else:
                    baseline_bottom_q.append(b)
                    baseline_height_q.append(hh)
                continue

            # in occ
            still_occ = (r_bottom >= float(min_bottom_frac) or r_height >= float(min_height_frac))
            recovered = (r_bottom <= float(recover_bottom_frac) and r_height <= float(recover_height_frac))

            if still_occ:
                last_occ_i = i
                recover_streak = 0
                active[i] = True
                continue

            if recovered:
                recover_streak += 1
                # IMPORTANT (recall-first): keep recovery frames active so we don't end early.
                # Also advance last_occ_i so the span includes the recovery tail.
                last_occ_i = i
                active[i] = True
                if recover_streak >= rmin:
                    # close window (end at last_occ_i, which includes recovery tail)
                    if start_i is not None and last_occ_i is not None:
                        if (last_occ_i - start_i + 1) >= int(min_window_frames):
                            active[start_i : last_occ_i + 1] = True
                    # reset state
                    in_occ = False
                    start_i = None
                    occ_b0 = None
                    occ_h0 = None
                    recover_streak = 0
                    recent_flags.clear()
                continue

            # ambiguous frame: treat as still part of occlusion, reset recovery streak
            last_occ_i = i
            recover_streak = 0
            active[i] = True

        # finalize open window
        if in_occ and start_i is not None and last_occ_i is not None:
            if (last_occ_i - start_i + 1) >= int(min_window_frames):
                active[start_i : last_occ_i + 1] = True

        # optional hard cap on span length (tool-side)
        if max_span_frames is not None and int(max_span_frames) > 0 and active.any():
            mlen = int(max_span_frames)
            idx = np.where(active)[0]
            s = idx[0]
            e = idx[0]
            for t in idx[1:]:
                if t == e + 1:
                    e = t
                else:
                    if (e - s + 1) > mlen:
                        active[s + mlen : e + 1] = False
                    s = t
                    e = t
            if (e - s + 1) > mlen:
                active[s + mlen : e + 1] = False

        g["pred_span_active"] = active
        g["rb_base"] = rb_out
        g["rh_base"] = rh_out
        return g

    df = df.groupby("tracklet_id", sort=False, group_keys=False).apply(_scan)
    return df


def compute_metric_kinematics(bank: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    # Expect: tracklet_id, frame_index, x_m, y_m  (or x_m_repaired/y_m_repaired optionally)
    df = bank.copy()
    df = df.sort_values(["tracklet_id", "frame_index"], kind="mergesort").reset_index(drop=True)

    # pick the “raw” metric coords for diagnosis; you can also switch to repaired later
    xm = df["x_m"].astype(float)
    ym = df["y_m"].astype(float)

    dt = 1.0 / float(fps)
    dx = df.groupby("tracklet_id")[xm.name].diff()
    dy = df.groupby("tracklet_id")[ym.name].diff()
    speed = np.sqrt(dx * dx + dy * dy) / dt
    df["speed_mps"] = speed

    # accel from speed diff
    df["accel_mps2"] = df.groupby("tracklet_id")["speed_mps"].diff() / dt

    return df


def framewise_eval(
    det_sig: pd.DataFrame,
    kin: pd.DataFrame,
    labeled_frames: Dict[Tuple[str, str], set],
    *,
    bottom_ratio_thresh: float,
    height_ratio_thresh: float,
    dy2_px_min: float,
    onset_window: int = 5,
    onset_min_frames: int = 1,
    recover_bottom_frac: float = 0.10,
    recover_height_frac: float = 0.08,
    recover_min_frames: int = 3,
    min_window_frames: int = 2,
    max_span_frames: Optional[int] = None,
    clip_id: Optional[str] = None,
) -> pd.DataFrame:
    # join det signals + kinematics on (tracklet_id, frame_index)
    key = ["tracklet_id", "frame_index"]
    # include baseline evidence + span prediction (linker_2 semantics)
    det_pred = detect_spans_linker2(
        det_sig,
        onset_window=onset_window,
        min_bottom_frac=bottom_ratio_thresh,
        min_height_frac=height_ratio_thresh,
        onset_min_frames=onset_min_frames,
        recover_bottom_frac=recover_bottom_frac,
        recover_height_frac=recover_height_frac,
        recover_min_frames=recover_min_frames,
        min_window_frames=min_window_frames,
        dy2_px_min=dy2_px_min,
        gate_onset_with_dy2=True,
        max_span_frames=max_span_frames,
    )

    cols_det = key + [
        "dy2",
        "dy1",
        "dh",
        "r_bottom",
        "r_height",
        "h",
        "rb_base",
        "rh_base",
        "pred_span_active",
    ]
    cols_kin = key + ["speed_mps", "accel_mps2"]

    det_has_clip = "clip_id" in det_sig.columns
    kin_has_clip = "clip_id" in kin.columns
    if det_has_clip:
        cols_det = ["clip_id"] + cols_det
    if kin_has_clip:
        cols_kin = ["clip_id"] + cols_kin

    if det_has_clip and kin_has_clip:
        m = det_pred[cols_det].merge(kin[cols_kin], on=["clip_id"] + key, how="left")
    else:
        m = det_pred[cols_det].merge(kin[cols_kin], on=key, how="left")
        if "clip_id" not in m.columns:
            if clip_id is None:
                # try to infer from det_sig if present
                if det_has_clip and not det_sig.empty:
                    clip_id = str(det_sig["clip_id"].iloc[0])
            if clip_id is None:
                raise KeyError("clip_id is required for labeling but was not provided and not present in inputs")
            m["clip_id"] = str(clip_id)

    # Keep legacy cand_frame for reference, but primary prediction is span-active frames.
    m["cand_frame"] = (
        ((m["r_bottom"] >= bottom_ratio_thresh) | (m["r_height"] >= height_ratio_thresh))
        & (m["dy2"].fillna(0.0) >= dy2_px_min)
    )

    # label using (clip_id, tracklet_id) with per-tracklet vectorized membership.
    m["is_label"] = False
    active_clip = str(clip_id) if clip_id is not None else str(m["clip_id"].iloc[0])
    by_tid: Dict[str, set] = {}
    for (cid, tid), frames in labeled_frames.items():
        if str(cid) != active_clip:
            continue
        by_tid[str(tid)] = set(frames)
    if by_tid:
        for tid, frames in by_tid.items():
            if not frames:
                continue
            mask = m["tracklet_id"].astype(str) == tid
            if mask.any():
                m.loc[mask, "is_label"] = m.loc[mask, "frame_index"].astype(int).isin(frames)

    return m


def pr(m: pd.DataFrame, pred_col="cand_frame", label_col="is_label"):
    tp = int(((m[pred_col]) & (m[label_col])).sum())
    fp = int(((m[pred_col]) & (~m[label_col])).sum())
    fn = int(((~m[pred_col]) & (m[label_col])).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return tp, fp, fn, prec, rec


def _boolish(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _plot_paths(
    bank: pd.DataFrame,
    *,
    out_png: Path,
    group_by: str = "tracklet",
    prefer_repaired: bool = True,
    stride: int = 1,
    min_points: int = 5,
) -> None:
    """Plot x_m/y_m footpaths per tracklet_id. Overlay repaired coords if present.

    - Raw: solid line
    - Repaired: dashed line (only if x_m_repaired/y_m_repaired exist)
    """
    if plt is None:
        raise ImportError("matplotlib is required for --plot_paths_png (pip install matplotlib)")

    if group_by != "tracklet":
        raise ValueError("Only group_by='tracklet' is supported in this tool build.")

    required = {"tracklet_id", "frame_index", "x_m", "y_m"}
    missing = [c for c in required if c not in bank.columns]
    if missing:
        raise KeyError(f"bank table missing required columns for plotting: {missing}")

    df = bank.copy()
    df = df.sort_values(["tracklet_id", "frame_index"], kind="mergesort").reset_index(drop=True)

    has_repaired = ("x_m_repaired" in df.columns) and ("y_m_repaired" in df.columns)
    has_is_repaired = "is_repaired" in df.columns
    _ = has_is_repaired  # reserved for future styling

    # Optional downsample for speed/visual clarity
    s = max(1, int(stride))
    if s > 1:
        df = df.iloc[::s].reset_index(drop=True)

    # Ensure numeric
    for c in ["x_m", "y_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if has_repaired:
        df["x_m_repaired"] = pd.to_numeric(df["x_m_repaired"], errors="coerce")
        df["y_m_repaired"] = pd.to_numeric(df["y_m_repaired"], errors="coerce")

    # Create plot
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    ax.set_title("Footpaths in mat space (raw vs repaired)")

    for tid, g in df.groupby("tracklet_id", sort=False):
        g = g.dropna(subset=["x_m", "y_m"])
        if len(g) < int(min_points):
            continue

        # Raw path
        ax.plot(g["x_m"].to_numpy(), g["y_m"].to_numpy(), label=f"{tid} raw")

        # Repaired overlay (dashed)
        if prefer_repaired and has_repaired:
            gr = g.dropna(subset=["x_m_repaired", "y_m_repaired"])
            if not gr.empty:
                ax.plot(
                    gr["x_m_repaired"].to_numpy(),
                    gr["y_m_repaired"].to_numpy(),
                    linestyle="--",
                    label=f"{tid} repaired",
                )

    # Keep legend manageable
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 24:
        ax.legend(loc="best", fontsize="small")
    else:
        ax.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png.as_posix(), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", type=str, required=True, help="outputs/<clip_id> directory that contains stage_A/, stage_D/")
    ap.add_argument("--labels_json", type=str, required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out_csv", type=str, default="occlusion_tuning_framewise.csv")

    # Optional plotting + merged geometry output
    ap.add_argument("--plot_paths_png", type=str, default=None, help="If set, write a mat-space footpath PNG here.")
    ap.add_argument("--plot_group_by", type=str, default="tracklet", help="Only 'tracklet' supported for now.")
    ap.add_argument("--plot_prefer_repaired", type=str, default="true", help="true|false")
    ap.add_argument("--plot_stride", type=int, default=1, help="Downsample points by stride for plotting.")
    ap.add_argument("--plot_min_points", type=int, default=5, help="Skip paths with fewer points.")
    ap.add_argument("--out_geom_csv", type=str, default=None, help="Optional merged CSV: diagnostics + geometry.")
    args = ap.parse_args()

    clip_dir = Path(args.clip_dir)
    manifest = json.load(open(clip_dir / "clip_manifest.json", "r"))
    clip_id = manifest["clip_id"]

    det = pd.read_parquet(clip_dir / "stage_A" / "detections.parquet")
    bank = pd.read_parquet(clip_dir / "stage_D" / "tracklet_bank_frames.parquet")

    # labels
    raw = load_json_tolerant(Path(args.labels_json))
    spans = [
        Span(
            camera_id=r["camera_id"],
            clip_id=r.get("video", r.get("clip_id")),
            tracklet_id=r["tracklet_id"],
            start_frame=int(r["start_frame"]),
            end_frame=int(r["end_frame"]),
        )
        for r in raw
    ]
    spans = [s for s in spans if s.clip_id == clip_id]
    labeled = spans_to_frame_set(spans)

    # compute signals
    det_sig = compute_bbox_signals(det)
    kin = compute_metric_kinematics(bank, fps=args.fps)

    # add clip_id for labeling logic
    det_sig["clip_id"] = clip_id
    kin["clip_id"] = clip_id

    # sweep some reasonable ranges (NOT guesses for final; just exploration)
    sweeps = []
    for bottom_ratio in [0.04, 0.06, 0.08, 0.10, 0.12]:
        for height_ratio in [0.04, 0.06, 0.08, 0.10]:
            for dy2_min in [2.0, 3.0, 4.0, 6.0]:
                m = framewise_eval(
                    det_sig,
                    kin,
                    labeled,
                    bottom_ratio_thresh=bottom_ratio,
                    height_ratio_thresh=height_ratio,
                    dy2_px_min=dy2_min,
                    onset_window=5,
                    onset_min_frames=1,
                    recover_bottom_frac=0.10,
                    recover_height_frac=0.08,
                    recover_min_frames=3,
                    min_window_frames=2,
                    max_span_frames=None,
                    clip_id=clip_id,
                )
                # Evaluate using span-active prediction (roll_it_back style)
                tp, fp, fn, prec, rec = pr(m, pred_col="pred_span_active", label_col="is_label")
                sweeps.append(
                    {
                        "bottom_ratio": bottom_ratio,
                        "height_ratio": height_ratio,
                        "dy2_min_px": dy2_min,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": prec,
                        "recall": rec,
                    }
                )

    sweeps_df = pd.DataFrame(sweeps).sort_values(["recall", "precision"], ascending=False)
    print(f"\n=== Top 15 configs for clip={clip_id} ===")
    print(sweeps_df.head(15).to_string(index=False))

    # also write a “best config framewise CSV” for inspection
    best = sweeps_df.iloc[0].to_dict()
    mbest = framewise_eval(
        det_sig,
        kin,
        labeled,
        bottom_ratio_thresh=float(best["bottom_ratio"]),
        height_ratio_thresh=float(best["height_ratio"]),
        dy2_px_min=float(best["dy2_min_px"]),
        onset_window=5,
        onset_min_frames=1,
        recover_bottom_frac=0.10,
        recover_height_frac=0.08,
        recover_min_frames=3,
        min_window_frames=2,
        max_span_frames=None,
        clip_id=clip_id,
    )
    mbest.to_csv(args.out_csv, index=False)
    print(f"\nWrote framewise table: {args.out_csv}")

    # Optional merged geometry CSV: join mbest with bank coords (raw + repaired)
    if args.out_geom_csv:
        bank_cols = ["tracklet_id", "frame_index"]
        for c in ["x_m", "y_m", "x_m_repaired", "y_m_repaired", "is_repaired", "repair_method", "repair_span_id"]:
            if c in bank.columns:
                bank_cols.append(c)
        geom = bank[bank_cols].copy()
        geom["frame_index"] = geom["frame_index"].astype(int, errors="ignore")
        merged = mbest.merge(geom, on=["tracklet_id", "frame_index"], how="left")
        out_geom = Path(args.out_geom_csv)
        out_geom.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_geom.as_posix(), index=False)
        print(f"Wrote merged geometry+diagnostics CSV: {out_geom.as_posix()}")

    # Optional plot
    if args.plot_paths_png:
        out_png = Path(args.plot_paths_png)
        _plot_paths(
            bank,
            out_png=out_png,
            group_by=str(args.plot_group_by),
            prefer_repaired=_boolish(args.plot_prefer_repaired),
            stride=int(args.plot_stride),
            min_points=int(args.plot_min_points),
        )
        print(f"Wrote footpath plot PNG: {out_png.as_posix()}")


if __name__ == "__main__":
    main()


# python tools/tune_partial_occlusion.py \
#   --clip_dir outputs/cam01-2tags_1person_0-15s \
#   --labels_json outputs/observed_partial_occlusions.json \
#   --fps 30 \
#   --out_csv cam01_framewise.csv

# python tools/tune_partial_occlusion.py \
#   --clip_dir outputs/cam03-20260103-124000_0-30s \
#   --labels_json outputs/observed_partial_occlusions.json \
#   --fps 30 \
#   --out_csv cam03_framewise.csv

# Example with plot + merged geometry CSV:
# python tools/tune_partial_occlusion.py \
#   --clip_dir outputs/cam03-20260103-124000_0-30s \
#   --labels_json outputs/observed_partial_occlusions.json \
#   --fps 30 \
#   --out_csv cam03_framewise_linker2_recoverytail.csv \
#   --out_geom_csv cam03_geom_framewise.csv \
#   --plot_paths_png cam03_paths.png \
#   --plot_stride 2