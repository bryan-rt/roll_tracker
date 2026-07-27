#!/usr/bin/env python3
"""CP7-pre-10: Bracketed vs unbracketed split of pair_box misattribution mass.

Measures whether pair-box spans resolve into two separately-tracked people
elsewhere in the clip (bracketed → offline-propagation-recoverable) or never
separate (unbracketed → detection-only-recoverable).

READ-ONLY diagnostic. No pipeline/config changes.

Usage:
    PYTHONPATH=src python tools/cp7_pre10_pairbox_bracketing.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
CLIP_ROOT = Path(
    "outputs/_eval_gt/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014"
)
EVAL_ROOT = Path("outputs/_eval/stage_d/bjj-detect-all-cameras/FP7oJQ")
PRE9_DIR = Path("outputs/_eval/_debug/cp7_pre9_branchb_margin")
OUT_DIR = Path("outputs/_eval/_debug/cp7_pre10_pairbox_bracketing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
BRACKET_RADIUS_M = 1.0  # world-coord radius for bracket resolution check
HORIZON_SWEEP = [30, 90, 300, 4530]  # ~1s, ~3s, ~10s, full clip
GAP_SPLIT_THRESHOLD = 5  # frames; gaps > this split spans
SHORT_SPAN_THRESHOLD = 5  # frames; spans < this reported separately
CLIP_MAX_FRAME = 4529
TRUSTED_GT_MAX = 300  # frames 0-300 have dense trusted GT


def load_artifacts():
    """Load all required artifacts."""
    with open(PRE9_DIR / "containment_results.json") as f:
        pre9_results = json.load(f)

    bf = pd.read_parquet(CLIP_ROOT / "stage_D" / "tracklet_bank_frames.parquet")
    bs = pd.read_parquet(CLIP_ROOT / "stage_D" / "tracklet_bank_summaries.parquet")
    trace = pd.read_parquet(EVAL_ROOT / "gt_person_trace.parquet")

    return pre9_results, bf, bs, trace


def build_fragment_map(bs: pd.DataFrame) -> dict[str, list[tuple[str, int, int]]]:
    """Map base tracklet IDs to their D0.5 split fragments.

    gt_person_trace uses original (pre-split) tracklet IDs from detections.parquet,
    but tracklet_bank_frames uses post-split IDs. This map resolves the mismatch.

    Returns {base_tid: [(fragment_tid, start_frame, end_frame), ...]} sorted by start.
    """
    fragments: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for _, row in bs.iterrows():
        tid = row["tracklet_id"]
        base = tid.split("_s")[0] if "_s" in tid else tid
        fragments[base].append((tid, int(row["start_frame"]), int(row["end_frame"])))

    # Sort each by start_frame
    for base in fragments:
        fragments[base].sort(key=lambda x: x[1])

    return dict(fragments)


def resolve_carrier_fragment(
    base_tid: str,
    frame: int,
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> str | None:
    """Find the split fragment of base_tid that covers the given frame.

    Returns the fragment tracklet_id, or None if no fragment covers that frame.
    """
    for frag_tid, fstart, fend in fragment_map.get(base_tid, []):
        if fstart <= frame <= fend:
            return frag_tid
    return None


def resolve_carrier_for_window(
    base_tid: str,
    window_start: int,
    window_end: int,
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> list[str]:
    """Find all split fragments of base_tid that overlap the given window.

    Returns list of fragment tracklet_ids that have any overlap with [window_start, window_end].
    """
    result = []
    for frag_tid, fstart, fend in fragment_map.get(base_tid, []):
        if fend >= window_start and fstart <= window_end:
            result.append(frag_tid)
    return result


def build_gt_attribution(
    trace: pd.DataFrame,
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> dict[str, int]:
    """Attribute each tracklet (and its D0.5 split fragments) to its majority-vote GT person.

    Uses the frozen CP-EVAL-1 identity mapping: for each tracklet, the GT person
    it is matched to most often across traced frames (present + present_misattributed).

    Split fragments inherit their base tracklet's attribution (the trace only knows
    original tracklet IDs; fragments are a D0.5 refinement of the same physical track).

    Attribution-circularity caveat: this uses pipeline-derived GT matching, which is
    most reliable at separation points (isolated boxes) — the exact points we're
    testing. The lean is benign but should be named.
    """
    explained = trace[
        trace.failure_mode.isin(["present", "present_misattributed"])
        & trace.tracklet_id.notna()
    ]

    attribution: dict[str, int] = {}
    for tid, grp in explained.groupby("tracklet_id"):
        votes = grp["gt_person_id"].value_counts()
        if len(votes) > 0:
            gt_id = int(votes.index[0])
            attribution[str(tid)] = gt_id
            # Also attribute all split fragments of this base tracklet
            for frag_tid, _, _ in fragment_map.get(str(tid), []):
                attribution[frag_tid] = gt_id

    return attribution


def collapse_into_spans(pair_box_rows: list[dict]) -> list[dict]:
    """Collapse pair_box frames into contiguous spans per (carrier, contained_gt)."""
    # Group by (tracklet_id, contained_gt_track_id)
    groups: dict[tuple, list[int]] = defaultdict(list)
    # Also collect matched_gt_id per group
    group_matched_gt: dict[tuple, list[int]] = defaultdict(list)

    for r in pair_box_rows:
        key = (r["tracklet_id"], r["contained_gt_track_id"])
        groups[key].append(r["frame_idx"])
        group_matched_gt[key].append(r["gt_person_id"])

    spans = []
    span_id = 0

    for (carrier_tid, contained_gt_id), frames in groups.items():
        frames = sorted(set(frames))
        matched_gts = group_matched_gt[(carrier_tid, contained_gt_id)]
        # Majority vote for matched_gt_id (should be consistent within a group)
        matched_gt_id = Counter(matched_gts).most_common(1)[0][0]

        # Split into contiguous runs (gap > GAP_SPLIT_THRESHOLD)
        runs = []
        run_start = frames[0]
        prev = frames[0]
        run_frames = [frames[0]]

        for f in frames[1:]:
            if f - prev > GAP_SPLIT_THRESHOLD:
                runs.append((run_start, prev, run_frames))
                run_start = f
                run_frames = [f]
            else:
                run_frames.append(f)
            prev = f
        runs.append((run_start, prev, run_frames))

        for rs, re, rf in runs:
            spans.append({
                "span_id": span_id,
                "carrier_tid": carrier_tid,
                "matched_gt_id": int(matched_gt_id),
                "contained_gt_id": int(contained_gt_id),
                "span_start": rs,
                "span_end": re,
                "n_frames": len(rf),
                "short_span": len(rf) < SHORT_SPAN_THRESHOLD,
            })
            span_id += 1

    return spans


def bracket_test_one_window(
    carrier_base_tid: str,
    matched_gt_id: int,
    contained_gt_id: int,
    window_start: int,
    window_end: int,
    bf: pd.DataFrame,
    bf_indexed: pd.DataFrame,
    gt_attribution: dict[str, int],
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> dict:
    """Test whether a bracket window resolves into two distinct GT persons.

    carrier_base_tid is the original (pre-split) tracklet ID from the trace.
    We resolve it to the post-split fragment(s) that overlap the window.

    Returns {resolved: bool, trusted: bool, stayed_apart: bool, detail: str}.
    """
    result = {
        "resolved": False,
        "trusted": False,
        "stayed_apart": False,
        "carrier_present": False,
        "detail": "",
    }

    if window_start > window_end:
        result["detail"] = "empty_window"
        return result

    # Resolve carrier to split fragment(s) overlapping the window
    carrier_frags = resolve_carrier_for_window(
        carrier_base_tid, window_start, window_end, fragment_map
    )

    if not carrier_frags:
        result["detail"] = "carrier_absent"
        return result

    # Collect carrier positions across all fragments in window
    carrier_frames = []
    for frag_tid in carrier_frags:
        try:
            frag_data = bf_indexed.loc[frag_tid]
            in_window = frag_data[
                (frag_data.index >= window_start) & (frag_data.index <= window_end)
            ]
            if not in_window.empty:
                carrier_frames.append(in_window)
        except KeyError:
            continue

    if not carrier_frames:
        result["detail"] = "carrier_absent"
        return result

    carrier_in_window = pd.concat(carrier_frames)
    result["carrier_present"] = True

    # Carrier mean position in window
    cx = float(carrier_in_window["x_m"].mean())
    cy = float(carrier_in_window["y_m"].mean())

    # Find all tracklets in window near carrier (excluding carrier's own fragments)
    carrier_frag_set = set(carrier_frags)
    window_data = bf[
        (bf.frame_index >= window_start) & (bf.frame_index <= window_end)
    ]
    tid_positions = window_data.groupby("tracklet_id")[["x_m", "y_m"]].mean()
    dists = np.sqrt((tid_positions.x_m - cx) ** 2 + (tid_positions.y_m - cy) ** 2)
    nearby_tids = set(dists[dists < BRACKET_RADIUS_M].index)

    # Attribute each nearby tracklet to a GT person
    # Carrier fragments can attribute to matched_gt_id (the carrier IS the matched person)
    nearby_gt_persons: dict[int, list[str]] = defaultdict(list)  # gt_id -> [tids]
    for tid in nearby_tids:
        gt_id = gt_attribution.get(tid)
        if gt_id is not None:
            nearby_gt_persons[gt_id].append(tid)

    # Resolution test: BOTH matched_gt_id AND contained_gt_id must be present
    # AND the contained_gt_id must be on a NON-carrier tracklet (distinct person)
    has_matched = matched_gt_id in nearby_gt_persons
    contained_non_carrier = [
        t for t in nearby_gt_persons.get(contained_gt_id, [])
        if t not in carrier_frag_set
    ]
    has_contained = len(contained_non_carrier) > 0

    if has_matched and has_contained:
        result["resolved"] = True

        # Trust boundary: is the resolution within trusted GT range?
        result["trusted"] = (window_start <= TRUSTED_GT_MAX)

        # Stayed-apart check: do both GT persons remain separately tracked
        # through to (or near) window end? Check if both have tracklets in
        # the last 30% of the window.
        late_start = window_start + int(0.7 * (window_end - window_start))
        late_data = bf[
            (bf.frame_index >= late_start) & (bf.frame_index <= window_end)
        ]
        late_tids = set(late_data["tracklet_id"].unique())
        late_matched = any(
            gt_attribution.get(t) == matched_gt_id for t in late_tids
            if t in nearby_tids
        )
        late_contained = any(
            gt_attribution.get(t) == contained_gt_id for t in late_tids
            if t in nearby_tids and t not in carrier_frag_set
        )
        result["stayed_apart"] = late_matched and late_contained

        result["detail"] = (
            f"matched={nearby_gt_persons[matched_gt_id]}, "
            f"contained={contained_non_carrier}, "
            f"trusted={result['trusted']}, stayed_apart={result['stayed_apart']}"
        )
    else:
        result["detail"] = (
            f"has_matched={has_matched}, has_contained={has_contained}, "
            f"nearby_gt_persons={dict(nearby_gt_persons)}"
        )

    return result


def run_bracket_sweep(
    spans: list[dict],
    bf: pd.DataFrame,
    gt_attribution: dict[str, int],
    horizons: list[int],
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> dict[int, list[dict]]:
    """Run bracket test across all horizons. Returns {horizon: [results]}."""
    bf_indexed = bf.set_index(["tracklet_id", "frame_index"])[["x_m", "y_m"]]

    all_results: dict[int, list[dict]] = {}

    for horizon in horizons:
        results = []
        for span in spans:
            carrier = span["carrier_tid"]
            matched_gt = span["matched_gt_id"]
            contained_gt = span["contained_gt_id"]
            ss = span["span_start"]
            se = span["span_end"]

            # Pre-window
            pre_start = max(0, ss - horizon)
            pre_end = ss - 1

            # Post-window
            post_start = se + 1
            post_end = min(CLIP_MAX_FRAME, se + horizon)

            pre_result = bracket_test_one_window(
                carrier, matched_gt, contained_gt,
                pre_start, pre_end, bf, bf_indexed, gt_attribution,
                fragment_map,
            )
            post_result = bracket_test_one_window(
                carrier, matched_gt, contained_gt,
                post_start, post_end, bf, bf_indexed, gt_attribution,
                fragment_map,
            )

            # Classification
            pre_ok = pre_result["resolved"]
            post_ok = post_result["resolved"]
            pre_carrier = pre_result["carrier_present"]
            post_carrier = post_result["carrier_present"]

            if not pre_carrier or not post_carrier:
                bracket_class = "indeterminate"
            elif pre_ok and post_ok:
                bracket_class = "bracketed"
            elif pre_ok:
                bracket_class = "half_bracket_pre"
            elif post_ok:
                bracket_class = "half_bracket_post"
            else:
                bracket_class = "unbracketed"

            # Trusted + stayed-apart for bracketed
            trusted = False
            stayed_apart = False
            if bracket_class == "bracketed":
                trusted = pre_result["trusted"] and post_result["trusted"]
                stayed_apart = pre_result["stayed_apart"] and post_result["stayed_apart"]

            results.append({
                **span,
                "horizon": horizon,
                "bracket_class": bracket_class,
                "pre_resolved": pre_ok,
                "post_resolved": post_ok,
                "pre_trusted": pre_result.get("trusted", False),
                "post_trusted": post_result.get("trusted", False),
                "bracket_trusted": trusted,
                "stayed_apart": stayed_apart,
                "pre_detail": pre_result["detail"],
                "post_detail": post_result["detail"],
            })

        all_results[horizon] = results

    return all_results


def check_gap_bridges(
    spans: list[dict],
    bf: pd.DataFrame,
    gt_attribution: dict[str, int],
    fragment_map: dict[str, list[tuple[str, int, int]]],
) -> list[dict]:
    """Check gaps between adjacent spans of same (carrier, contained_gt) for resolution."""
    bf_indexed = bf.set_index(["tracklet_id", "frame_index"])[["x_m", "y_m"]]

    # Group spans by (carrier, contained_gt), sorted by span_start
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in spans:
        groups[(s["carrier_tid"], s["contained_gt_id"])].append(s)

    bridges = []
    for key, group_spans in groups.items():
        group_spans = sorted(group_spans, key=lambda s: s["span_start"])
        for i in range(len(group_spans) - 1):
            s1 = group_spans[i]
            s2 = group_spans[i + 1]
            gap_start = s1["span_end"] + 1
            gap_end = s2["span_start"] - 1
            gap_len = gap_end - gap_start + 1

            if gap_len < 1 or gap_len > 20:
                continue

            # Check if gap contains two-distinct-GT-person resolution
            result = bracket_test_one_window(
                key[0], s1["matched_gt_id"], int(key[1]),
                gap_start, gap_end, bf, bf_indexed, gt_attribution,
                fragment_map,
            )

            bridges.append({
                "carrier_tid": key[0],
                "contained_gt_id": int(key[1]),
                "span1_end": s1["span_end"],
                "span2_start": s2["span_start"],
                "gap_frames": gap_len,
                "resolved": result["resolved"],
                "detail": result["detail"],
            })

    return bridges


def print_table(
    results: list[dict],
    horizon: int,
    total_pair_box_frames: int,
    total_misattr: int,
):
    """Print bracket classification table for one horizon."""
    # Separate short vs normal spans
    normal = [r for r in results if not r["short_span"]]
    short = [r for r in results if r["short_span"]]

    for label, subset in [("Normal spans (>=5 frames)", normal), ("Short spans (<5 frames)", short)]:
        if not subset:
            print(f"\n  {label}: none")
            continue

        total_spans = len(subset)
        total_frames = sum(r["n_frames"] for r in subset)
        bc = Counter(r["bracket_class"] for r in subset)
        fc: dict[str, int] = defaultdict(int)
        for r in subset:
            fc[r["bracket_class"]] += r["n_frames"]

        print(f"\n  {label} ({total_spans} spans, {total_frames} frames):")
        print(f"  {'Class':<22} {'Spans':>6} {'%spans':>7} {'Frames':>7} {'%pb':>6} {'%2259':>6}")
        print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
        for cls in ["bracketed", "half_bracket_pre", "half_bracket_post", "unbracketed", "indeterminate"]:
            ns = bc.get(cls, 0)
            nf = fc.get(cls, 0)
            ps = 100 * ns / total_spans if total_spans else 0
            ppb = 100 * nf / total_pair_box_frames if total_pair_box_frames else 0
            pall = 100 * nf / total_misattr if total_misattr else 0
            print(f"  {cls:<22} {ns:>6} {ps:>6.1f}% {nf:>7} {ppb:>5.1f}% {pall:>5.1f}%")

        # For bracketed: show trusted + stayed_apart breakdown
        bracketed = [r for r in subset if r["bracket_class"] == "bracketed"]
        if bracketed:
            trusted_clean = sum(1 for r in bracketed if r["bracket_trusted"] and r["stayed_apart"])
            trusted_remerge = sum(1 for r in bracketed if r["bracket_trusted"] and not r["stayed_apart"])
            untrusted_clean = sum(1 for r in bracketed if not r["bracket_trusted"] and r["stayed_apart"])
            untrusted_remerge = sum(1 for r in bracketed if not r["bracket_trusted"] and not r["stayed_apart"])
            print(f"\n    Bracketed detail ({len(bracketed)} spans):")
            print(f"      trusted + stayed_apart:    {trusted_clean}")
            print(f"      trusted + re-merged:       {trusted_remerge}")
            print(f"      untrusted + stayed_apart:  {untrusted_clean}")
            print(f"      untrusted + re-merged:     {untrusted_remerge}")

            # Same by frames
            tc_f = sum(r["n_frames"] for r in bracketed if r["bracket_trusted"] and r["stayed_apart"])
            tr_f = sum(r["n_frames"] for r in bracketed if r["bracket_trusted"] and not r["stayed_apart"])
            uc_f = sum(r["n_frames"] for r in bracketed if not r["bracket_trusted"] and r["stayed_apart"])
            ur_f = sum(r["n_frames"] for r in bracketed if not r["bracket_trusted"] and not r["stayed_apart"])
            print(f"    By frames:")
            print(f"      trusted + stayed_apart:    {tc_f}")
            print(f"      trusted + re-merged:       {tr_f}")
            print(f"      untrusted + stayed_apart:  {uc_f}")
            print(f"      untrusted + re-merged:     {ur_f}")


def main():
    print("Loading artifacts...")
    pre9_results, bf, bs, trace = load_artifacts()

    pair_box = [r for r in pre9_results if r["outcome"] == "pair_box"]
    total_pair_box_frames = len(pair_box)
    total_misattr = 2259  # from pre-8/pre-9
    print(f"pair_box frames: {total_pair_box_frames}")

    # ── Build fragment map for D0.5 split resolution ──
    print("Building D0.5 fragment map...")
    fragment_map = build_fragment_map(bs)
    n_split = sum(1 for frags in fragment_map.values() if len(frags) > 1)
    print(f"  Base tracklets with splits: {n_split}/{len(fragment_map)}")

    # ── Step 1: Collapse into spans ──
    print("\nCollapsing into pair-spans...")
    spans = collapse_into_spans(pair_box)
    normal_spans = [s for s in spans if not s["short_span"]]
    short_spans = [s for s in spans if s["short_span"]]
    print(f"Total spans: {len(spans)} (normal: {len(normal_spans)}, short: {len(short_spans)})")
    print(f"Frames in normal spans: {sum(s['n_frames'] for s in normal_spans)}")
    print(f"Frames in short spans: {sum(s['n_frames'] for s in short_spans)}")

    # Save spans
    with open(OUT_DIR / "pair_spans.json", "w") as f:
        json.dump(spans, f, indent=2)

    # ── Step 2: Build GT attribution ──
    print("\nBuilding GT person attribution...")
    gt_attr = build_gt_attribution(trace, fragment_map)
    # Count base vs fragment attributions
    n_base = sum(1 for t in gt_attr if "_s" not in t)
    n_frag = sum(1 for t in gt_attr if "_s" in t)
    print(f"Tracklets with GT attribution: {len(gt_attr)} ({n_base} base, {n_frag} fragments)")

    # ── Step 3: Bracket sweep ──
    print(f"\nRunning bracket sweep across horizons: {HORIZON_SWEEP}...")
    sweep_results = run_bracket_sweep(spans, bf, gt_attr, HORIZON_SWEEP, fragment_map)

    # Save sweep results
    for horizon, results in sweep_results.items():
        with open(OUT_DIR / f"brackets_h{horizon}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ── Step 4: Gap-bridge check ──
    print("\nChecking gap bridges (6-20 frame gaps)...")
    bridges = check_gap_bridges(spans, bf, gt_attr, fragment_map)
    resolved_bridges = [b for b in bridges if b["resolved"]]
    print(f"Gaps checked: {len(bridges)}, resolved: {len(resolved_bridges)}")
    with open(OUT_DIR / "gap_bridges.json", "w") as f:
        json.dump(bridges, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════════
    #  REPORTING
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("PAIR-SPAN SUMMARY")
    print("=" * 72)
    print(f"  Total pair_box frames: {total_pair_box_frames}")
    print(f"  Total spans: {len(spans)}")
    print(f"  Normal spans (>=5 frames): {len(normal_spans)}")
    print(f"  Short spans (<5 frames): {len(short_spans)}")
    print(f"  Unique (carrier, contained_gt) pairs: {len(set((s['carrier_tid'], s['contained_gt_id']) for s in spans))}")

    # ── Per-horizon results ──
    for horizon in HORIZON_SWEEP:
        results = sweep_results[horizon]
        label = f"{horizon} frames" if horizon < 4530 else "full clip"
        print(f"\n{'=' * 72}")
        print(f"HORIZON: {label}")
        print(f"{'=' * 72}")
        print_table(results, horizon, total_pair_box_frames, total_misattr)

    # ── Horizon curve (the key output) ──
    print(f"\n{'=' * 72}")
    print("HORIZON CURVE — BRACKETED SHARE")
    print("=" * 72)
    print(f"\n  By spans (normal only):")
    print(f"  {'Horizon':<15} {'Bracketed':>10} {'Total':>7} {'%':>7}")
    print(f"  {'-'*15} {'-'*10} {'-'*7} {'-'*7}")
    for horizon in HORIZON_SWEEP:
        results = [r for r in sweep_results[horizon] if not r["short_span"]]
        n_brack = sum(1 for r in results if r["bracket_class"] == "bracketed")
        total = len(results)
        pct = 100 * n_brack / total if total else 0
        label = f"{horizon}f (~{horizon/30:.0f}s)" if horizon < 4530 else "full clip"
        print(f"  {label:<15} {n_brack:>10} {total:>7} {pct:>6.1f}%")

    print(f"\n  By frames (normal spans only):")
    print(f"  {'Horizon':<15} {'Bracketed':>10} {'Total':>7} {'%pb':>7} {'%2259':>7}")
    print(f"  {'-'*15} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")
    for horizon in HORIZON_SWEEP:
        results = [r for r in sweep_results[horizon] if not r["short_span"]]
        brack_frames = sum(r["n_frames"] for r in results if r["bracket_class"] == "bracketed")
        total_frames = sum(r["n_frames"] for r in results)
        ppb = 100 * brack_frames / total_pair_box_frames if total_pair_box_frames else 0
        pall = 100 * brack_frames / total_misattr if total_misattr else 0
        label = f"{horizon}f (~{horizon/30:.0f}s)" if horizon < 4530 else "full clip"
        print(f"  {label:<15} {brack_frames:>10} {total_frames:>7} {ppb:>6.1f}% {pall:>6.1f}%")

    # ── Trusted + stayed-apart (defensible number) ──
    print(f"\n{'=' * 72}")
    print("DEFENSIBLE BRACKET SHARE (trusted GT + stayed apart)")
    print("=" * 72)

    print(f"\n  By frames (normal spans, all horizons):")
    print(f"  {'Horizon':<15} {'Clean':>7} {'Remerge':>8} {'Untrust':>8} {'%pb clean':>10}")
    print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")
    for horizon in HORIZON_SWEEP:
        results = [r for r in sweep_results[horizon] if not r["short_span"]]
        bracketed = [r for r in results if r["bracket_class"] == "bracketed"]
        clean_f = sum(r["n_frames"] for r in bracketed if r["bracket_trusted"] and r["stayed_apart"])
        remerge_f = sum(r["n_frames"] for r in bracketed if r["bracket_trusted"] and not r["stayed_apart"])
        untrust_f = sum(r["n_frames"] for r in bracketed if not r["bracket_trusted"])
        ppb = 100 * clean_f / total_pair_box_frames if total_pair_box_frames else 0
        label = f"{horizon}f (~{horizon/30:.0f}s)" if horizon < 4530 else "full clip"
        print(f"  {label:<15} {clean_f:>7} {remerge_f:>8} {untrust_f:>8} {ppb:>9.1f}%")

    # ── Gap bridges ──
    print(f"\n{'=' * 72}")
    print("GAP BRIDGES (6-20 frame gaps between adjacent same-pair spans)")
    print("=" * 72)
    print(f"  Total gaps checked: {len(bridges)}")
    print(f"  Gaps with two-GT-person resolution: {len(resolved_bridges)}")
    if resolved_bridges:
        print(f"  These gaps ARE the bracket — adjacent spans are effectively one")
        print(f"  bracketed engagement. Effect: some 'unbracketed' spans may be")
        print(f"  recoverable when joined across their gap.")
        for b in resolved_bridges[:5]:
            print(f"    carrier={b['carrier_tid']}, gt2={b['contained_gt_id']}, "
                  f"gap={b['gap_frames']}f ({b['span1_end']+1}-{b['span2_start']-1})")

    # ── Final split ──
    print(f"\n{'=' * 72}")
    print("FINAL RECOVERY-PATH SPLIT (headline: 30-frame horizon, normal spans)")
    print("=" * 72)

    h30 = [r for r in sweep_results[30] if not r["short_span"]]
    h30_brack_f = sum(r["n_frames"] for r in h30 if r["bracket_class"] == "bracketed")
    h30_half_f = sum(r["n_frames"] for r in h30 if r["bracket_class"] in ["half_bracket_pre", "half_bracket_post"])
    h30_unbrack_f = sum(r["n_frames"] for r in h30 if r["bracket_class"] == "unbracketed")
    h30_indet_f = sum(r["n_frames"] for r in h30 if r["bracket_class"] == "indeterminate")

    hfull = [r for r in sweep_results[4530] if not r["short_span"]]
    hfull_brack_f = sum(r["n_frames"] for r in hfull if r["bracket_class"] == "bracketed")
    hfull_clean_f = sum(r["n_frames"] for r in hfull
                        if r["bracket_class"] == "bracketed" and r["bracket_trusted"] and r["stayed_apart"])

    print(f"\n  {'Recovery path':<35} {'30f':>6} {'full':>6} {'%pb(30f)':>9} {'%pb(full)':>10}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*9} {'-'*10}")
    print(f"  {'Propagation-recoverable (brack.)':<35} {h30_brack_f:>6} {hfull_brack_f:>6} "
          f"{100*h30_brack_f/total_pair_box_frames:>8.1f}% {100*hfull_brack_f/total_pair_box_frames:>9.1f}%")
    print(f"  {'One-side anchor (half-bracket)':<35} {h30_half_f:>6} "
          f"{sum(r['n_frames'] for r in hfull if r['bracket_class'] in ['half_bracket_pre','half_bracket_post']):>6} "
          f"{100*h30_half_f/total_pair_box_frames:>8.1f}% "
          f"{100*sum(r['n_frames'] for r in hfull if r['bracket_class'] in ['half_bracket_pre','half_bracket_post'])/total_pair_box_frames:>9.1f}%")
    print(f"  {'Detection-only (unbracketed)':<35} {h30_unbrack_f:>6} "
          f"{sum(r['n_frames'] for r in hfull if r['bracket_class'] == 'unbracketed'):>6} "
          f"{100*h30_unbrack_f/total_pair_box_frames:>8.1f}% "
          f"{100*sum(r['n_frames'] for r in hfull if r['bracket_class'] == 'unbracketed')/total_pair_box_frames:>9.1f}%")
    print(f"  {'Indeterminate':<35} {h30_indet_f:>6} "
          f"{sum(r['n_frames'] for r in hfull if r['bracket_class'] == 'indeterminate'):>6}")

    print(f"\n  Defensible (trusted + stayed-apart, full clip): "
          f"{hfull_clean_f} frames = {100*hfull_clean_f/total_pair_box_frames:.1f}% of pair_box "
          f"= {100*hfull_clean_f/total_misattr:.1f}% of all misattr")

    # ── Verdict ──
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print("=" * 72)

    # Assess curve shape
    h30_pct = 100 * h30_brack_f / total_pair_box_frames if total_pair_box_frames else 0
    hfull_pct = 100 * hfull_brack_f / total_pair_box_frames if total_pair_box_frames else 0
    climb = hfull_pct - h30_pct

    print(f"\n  Bracket curve: {h30_pct:.1f}% (30f) -> {hfull_pct:.1f}% (full clip)")
    print(f"  Climb from 30f to full: +{climb:.1f}pp")

    if hfull_pct < 20:
        print(f"\n  VERDICT: Flat-and-low. Even at full-clip horizon, <20% of pair_box")
        print(f"  mass is bracketed. Pairs genuinely never separate into two tracked boxes.")
        print(f"  The GROUP/offline-propagation path is a sidecar (~{hfull_pct:.0f}% of pair_box,")
        print(f"  ~{100*hfull_brack_f/total_misattr:.0f}% of all misattr).")
        print(f"  Detection separation is the primary lever.")
    elif hfull_pct > 50:
        print(f"\n  VERDICT: Climbs steeply. {hfull_pct:.0f}% of pair_box mass is bracketed")
        print(f"  at full-clip horizon. Long engagements DO resolve given time.")
        print(f"  The offline-propagation path is a materially larger lever than ~10%.")
        if hfull_clean_f / total_pair_box_frames > 0.3:
            print(f"  Defensible (trusted + stayed-apart): {100*hfull_clean_f/total_pair_box_frames:.0f}% — real signal.")
        else:
            print(f"  BUT: defensible share (trusted + stayed-apart) is only "
                  f"{100*hfull_clean_f/total_pair_box_frames:.0f}%. Much of the bracket")
            print(f"  evidence is untrusted or re-merged. Proceed with caution.")
    else:
        print(f"\n  VERDICT: Moderate. {hfull_pct:.0f}% of pair_box mass is bracketed at")
        print(f"  full-clip horizon — neither a sidecar nor a dominant lever.")
        print(f"  Both detection separation and offline propagation warrant investment.")

    print(f"\n  Caveat: 'bracketed' = CEILING for offline propagation, not a guarantee")
    print(f"  any implementation achieves it.")
    print(f"\n  Attribution-circularity caveat: bracket detection uses pipeline-derived")
    print(f"  GT attribution (majority-vote from gt_person_trace). This is most reliable")
    print(f"  at separation points (isolated boxes) — the exact points being tested.")
    print(f"  The lean is benign but the numbers are not ground-truth-verified outside 0-300.")

    print(f"\n  STOP: Report at docs/cp7_pre10_pairbox_bracketing.md.")
    print(f"  Detection-vs-propagation prioritization returns to the web session.")


if __name__ == "__main__":
    main()
