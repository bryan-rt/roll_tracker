import argparse
from pathlib import Path

import pandas as pd


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def inspect_clip(clip_root: Path, tracklet_ids: list[str]) -> None:
    stage_d = clip_root / "stage_D"
    debug = clip_root / "_debug"

    print(f"clip_root={clip_root}")

    frames_path = stage_d / "tracklet_bank_frames.parquet"
    nodes_path = stage_d / "d1_graph_nodes.parquet"
    edges_path = stage_d / "d1_graph_edges.parquet"

    merge_trig_path = debug / "d1_merge_triggers.parquet"
    split_trig_path = debug / "d1_split_triggers.parquet"
    split_supp_path = debug / "d1_suppressed_split_triggers.parquet"
    group_spans_path = debug / "d1_group_spans.parquet"
    group_supp_path = debug / "d1_suppressed_group_spans.parquet"

    frames = load_parquet_if_exists(frames_path)
    nodes = load_parquet_if_exists(nodes_path)
    edges = load_parquet_if_exists(edges_path)
    merge_trig = load_parquet_if_exists(merge_trig_path)
    split_trig = load_parquet_if_exists(split_trig_path)
    split_supp = load_parquet_if_exists(split_supp_path)
    group_spans = load_parquet_if_exists(group_spans_path)
    group_supp = load_parquet_if_exists(group_supp_path)

    print("=== Presence check ===")
    for p, df in [
        (frames_path, frames),
        (nodes_path, nodes),
        (edges_path, edges),
        (merge_trig_path, merge_trig),
        (split_trig_path, split_trig),
        (split_supp_path, split_supp),
        (group_spans_path, group_spans),
        (group_supp_path, group_supp),
    ]:
        print(f"{p}: {'OK' if df is not None else 'MISSING'}")

    if frames is not None:
        print("\n=== Tracklet lifespans (from frames) ===")
        cols = list(frames.columns)
        frame_col = None
        for cand in ("frame", "frame_idx", "frame_index"):
            if cand in cols:
                frame_col = cand
                break
        if frame_col is None:
            print("(no frame column in tracklet_bank_frames)")
        else:
            for tid in tracklet_ids:
                if "tracklet_id" in frames.columns:
                    sub = frames[frames["tracklet_id"].astype(str) == tid]
                else:
                    sub = pd.DataFrame()
                if sub.empty:
                    print(f"tracklet {tid}: no frames")
                else:
                    f_min = int(sub[frame_col].min())
                    f_max = int(sub[frame_col].max())
                    print(f"tracklet {tid}: {frame_col} [{f_min}, {f_max}]")

        # Optional: pairwise distance checks between consecutive tracklets in the list
        if frame_col is not None:
            def xy(df_row):
                if "x_m_repaired" in df_row.index and pd.notna(df_row["x_m_repaired"]):
                    return float(df_row["x_m_repaired"]), float(df_row["y_m_repaired"])
                return float(df_row["x_m"]), float(df_row["y_m"])

            import math

            print("\n=== Pairwise distances at disappear/appear (same-frame heuristic) ===")
            # Use node lifespans to choose a shared frame where both tracklets are alive
            if nodes is not None and "start_frame" in nodes.columns and "end_frame" in nodes.columns:
                # Build simple lookup for node lifespans (first non-null span per base_tracklet_id)
                node_life: dict[str, tuple[int, int]] = {}
                for _, r in nodes.iterrows():
                    base_tid = str(r.get("base_tracklet_id", ""))
                    if not base_tid:
                        continue
                    sf = r.get("start_frame")
                    ef = r.get("end_frame")
                    if sf is pd.NA or ef is pd.NA:
                        continue
                    if base_tid not in node_life:
                        node_life[base_tid] = (int(sf), int(ef))

                for a, b in zip(tracklet_ids, tracklet_ids[1:]):
                    if a not in node_life or b not in node_life:
                        continue
                    a_start, a_end = node_life[a]
                    b_start, b_end = node_life[b]

                    # Prefer a "merge-like" frame: end of a, if b is alive then
                    shared_frame: int | None = None
                    if a_end >= b_start and a_end <= b_end:
                        shared_frame = a_end
                    # Otherwise, prefer a "split-like" frame: start of b, if a is alive then
                    elif b_start >= a_start and b_start <= a_end:
                        shared_frame = b_start

                    if shared_frame is None:
                        print(f"{a}<->{b}: no overlapping lifespan window")
                        continue

                    fa = frames[frames["tracklet_id"].astype(str) == a]
                    fb = frames[frames["tracklet_id"].astype(str) == b]
                    if fa.empty or fb.empty:
                        print(f"pair {a}->{b}: no frames available")
                        continue

                    # nearest frame in each tracklet to the shared_frame
                    ra = fa.loc[(fa[frame_col] - shared_frame).abs().idxmin()]
                    rb = fb.loc[(fb[frame_col] - shared_frame).abs().idxmin()]
                    xa, ya = xy(ra)
                    xb, yb = xy(rb)
                    da = math.hypot(xa - xb, ya - yb)
                    print(f"{a} vs {b} @frame={shared_frame}: dist_m={da:.3f}")

    if nodes is not None:
        print("\n=== D1 nodes for tracklets ===")
        for tid in tracklet_ids:
            m = nodes[nodes.get("base_tracklet_id", nodes.get("tracklet_id", "")).astype(str) == tid]
            print(f"\n-- nodes for {tid} --")
            if m.empty:
                print("(none)")
            else:
                cols = [c for c in ["node_id", "node_type", "base_tracklet_id", "start_frame", "end_frame"] if c in m.columns]
                print(m[cols].to_string(index=False))

    if merge_trig is not None:
        print("\n=== Merge triggers (disappearing or carrier in tracklets) ===")
        for tid in tracklet_ids:
            cols = [c for c in ["disappearing_tracklet_id", "carrier_tracklet_id"] if c in merge_trig.columns]
            if not cols:
                break
            m = merge_trig[
                merge_trig["disappearing_tracklet_id"].astype(str).eq(tid)
                | merge_trig["carrier_tracklet_id"].astype(str).eq(tid)
            ]
            print(f"\n-- merge triggers involving {tid} --")
            if m.empty:
                print("(none)")
            else:
                keep = [c for c in ["disappearing_tracklet_id", "carrier_tracklet_id", "merge_frame", "dist_m"] if c in m.columns]
                print(m[keep].sort_values(keep).to_string(index=False))

    if split_trig is not None:
        print("\n=== Split triggers (carrier or new in tracklets) ===")
        cols = list(split_trig.columns)
        has_carrier = "carrier_tracklet_id" in cols
        has_new = "new_tracklet_id" in cols
        for tid in tracklet_ids:
            cond = False
            if has_carrier:
                cond = split_trig["carrier_tracklet_id"].astype(str).eq(tid)
            if has_new:
                cond = cond | split_trig["new_tracklet_id"].astype(str).eq(tid) if isinstance(cond, pd.Series) else split_trig["new_tracklet_id"].astype(str).eq(tid)
            m = split_trig[cond] if isinstance(cond, pd.Series) else pd.DataFrame()
            print(f"\n-- split triggers involving {tid} --")
            if m.empty:
                print("(none)")
            else:
                keep = [c for c in ["carrier_tracklet_id", "new_tracklet_id", "split_frame", "dist_m"] if c in m.columns]
                print(m[keep].sort_values(keep).to_string(index=False))

    if split_supp is not None:
        print("\n=== Suppressed split triggers for tracklets ===")
        cols = list(split_supp.columns)
        has_carrier = "carrier_tracklet_id" in cols
        has_new = "new_tracklet_id" in cols
        for tid in tracklet_ids:
            cond = False
            if has_carrier:
                cond = split_supp["carrier_tracklet_id"].astype(str).eq(tid)
            if has_new:
                cond = cond | split_supp["new_tracklet_id"].astype(str).eq(tid) if isinstance(cond, pd.Series) else split_supp["new_tracklet_id"].astype(str).eq(tid)
            m = split_supp[cond] if isinstance(cond, pd.Series) else pd.DataFrame()
            print(f"\n-- suppressed splits involving {tid} --")
            if m.empty:
                print("(none)")
            else:
                keep = [c for c in ["carrier_tracklet_id", "new_tracklet_id", "split_frame", "dist_m", "reason"] if c in m.columns]
                print(m[keep].sort_values(keep).to_string(index=False))

    if group_spans is not None:
        print("\n=== Group spans per carrier (for carriers in tracklets) ===")
        for tid in tracklet_ids:
            if "carrier_tracklet_id" not in group_spans.columns:
                continue
            m = group_spans[group_spans["carrier_tracklet_id"].astype(str) == tid]
            print(f"\n-- group spans for carrier {tid} --")
            if m.empty:
                print("(none)")
            else:
                keep = [c for c in ["carrier_tracklet_id", "start_frame", "end_frame", "span_type", "disappearing_tracklet_id", "new_tracklet_id"] if c in m.columns]
                print(m[keep].sort_values(["start_frame", "end_frame"]).to_string(index=False))

    if edges is not None:
        print("\n=== D1 MERGE/SPLIT edges involving tracklets ===")
        for tid in tracklet_ids:
            m = edges[
                edges["edge_type"].astype(str).isin(["EdgeType.MERGE", "EdgeType.SPLIT"])
                & (
                    edges["u"].astype(str).str.contains(tid)
                    | edges["v"].astype(str).str.contains(tid)
                )
            ]
            print(f"\n-- MERGE/SPLIT edges touching {tid} --")
            if m.empty:
                print("(none)")
            else:
                keep = ["edge_id", "edge_type", "u", "v", "merge_end", "split_start"]
                keep = [c for c in keep if c in m.columns]
                print(m[keep].sort_values(["edge_type", "edge_id"]).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect D1 merge/split triggers and spans for specific tracklets.")
    parser.add_argument("clip_root", help="Path to outputs/<clip_id> (e.g., outputs/cam03-...)")
    parser.add_argument("tracklet_ids", nargs="+", help="Base tracklet ids to inspect (e.g., t5 t6 t16)")
    args = parser.parse_args()

    clip_root = Path(args.clip_root).resolve()
    inspect_clip(clip_root, args.tracklet_ids)


if __name__ == "__main__":
    main()
