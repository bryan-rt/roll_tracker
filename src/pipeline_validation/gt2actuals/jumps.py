"""CP-GT2ACTUALS-4+5: Jump + group-integrity event detection.

Adds inline `jump_type` column to the dense join artifact, marking frames
where a GT person's identity thread breaks. Also classifies D0.5 split
events as correct vs false at the SPLIT-EVENT level (not per-GT-person).

Jump truth is derived from GT->detection association ONLY — never from
pipeline labels changing. A "jump" for a GT person is a frame where that
person's GT-matched identity assignment transitions.

Five jump types:
- tracklet_drift: within one tracklet's life, the GT-matched identity
  changes (tracker let the thread slide across two GT people).
- false_split: D0.5 split where BOTH products map to the SAME GT person
  (split fragmented one person). Classified at split-event level.
- ilp_misstitch: tracklet changed (not a D0.5 split) and carries a
  different person_id. Solver connected wrong tracklets.
- group_boundary_jump: at GROUP entry/exit, the through-line changed.
- group_membership_drift: mid-GROUP, the node's carried-GT-identity SET
  changed from the set at GROUP entry.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tracklet GT purity (correction #2)
# ---------------------------------------------------------------------------

def _compute_tracklet_gt_purity(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """Per-tracklet GT purity: which GT person(s) does each tracklet carry?

    Returns: {tracklet_id: {
        'dominant_gt': int,        # most-frequent GT track_id
        'purity': float,           # fraction of frames with dominant GT
        'gt_counts': Counter,      # gt_track_id -> frame count
        'is_pure': bool,           # purity >= 0.9
    }}
    """
    detected = df[df['tracklet_id'].notna()].copy()
    if detected.empty:
        return {}

    result: dict[str, dict] = {}
    for tid, grp in detected.groupby('tracklet_id'):
        gt_counts = Counter(grp['gt_track_id'].tolist())
        total = sum(gt_counts.values())
        dominant_gt = gt_counts.most_common(1)[0][0]
        purity = gt_counts[dominant_gt] / total if total > 0 else 0.0
        result[str(tid)] = {
            'dominant_gt': dominant_gt,
            'purity': purity,
            'gt_counts': gt_counts,
            'is_pure': purity >= 0.9,
        }
    return result


# ---------------------------------------------------------------------------
# D0.5 split-event classification (correction #3)
# ---------------------------------------------------------------------------

def classify_split_events(
    split_events: list[dict],
    df: pd.DataFrame,
    split_map: dict[str, list[str]],
) -> list[dict]:
    """Classify each D0.5 split event as correct vs false.

    Iterates split events; for each, finds BOTH sibling products and checks
    their GT mappings:
    - correct_split: products map to DIFFERENT GT people (split separated two)
    - false_split: products map to the SAME GT person (split fragmented one)
    - unclassifiable: one or both products have no GT match

    This is at the SPLIT-EVENT level (correction #3), not per-GT-person.
    Should reconcile with CP-SPLIT-VALIDATE's 2.4% correct / 77.5% spurious.
    """
    # Build: tracklet_id -> dominant GT person from the dense join
    detected = df[df['tracklet_id'].notna()].copy()
    # Use raw tracklet_id (Stage A) for mapping, since split products appear
    # in both raw and resolved columns
    tid_gt_map: dict[str, int] = {}
    for tid, grp in detected.groupby('tracklet_id'):
        gt_counts = Counter(grp['gt_track_id'].tolist())
        tid_gt_map[str(tid)] = gt_counts.most_common(1)[0][0]

    # Also map resolved tracklet IDs
    for rtid, grp in detected.groupby('resolved_tracklet_id'):
        if pd.isna(rtid):
            continue
        gt_counts = Counter(grp['gt_track_id'].tolist())
        tid_gt_map[str(rtid)] = gt_counts.most_common(1)[0][0]

    # Build per-original -> list of products (including the remnant original)
    original_products: dict[str, list[str]] = defaultdict(list)
    for ev in split_events:
        orig = ev['original_tracklet_id']
        prod = ev['new_tracklet_id']
        if prod not in original_products[orig]:
            original_products[orig].append(prod)

    results: list[dict] = []
    for ev in split_events:
        orig = ev['original_tracklet_id']
        new_tid = ev['new_tracklet_id']
        split_frame = ev.get('split_frame', -1)
        tier = ev.get('tier', '')

        # Find sibling: the segment BEFORE this split point.
        # For original t1 with products [t1_s1, t1_s2, t1_s3]:
        # split at frame X creates t1_sN; the "sibling" is the segment
        # covering the frames just before X. That's either the original
        # (if this is the first split) or a prior product.
        # Simplified approach: check the original tid's GT mapping for
        # frames BEFORE split_frame vs the product's GT mapping for
        # frames AT/AFTER split_frame.
        pre_rows = detected[
            (detected['tracklet_id'] == orig) &
            (detected['frame_index'] < split_frame)
        ]
        post_rows = detected[
            ((detected['tracklet_id'] == new_tid) |
             (detected['resolved_tracklet_id'] == new_tid)) &
            (detected['frame_index'] >= split_frame)
        ]

        # Also try: rows on the original tracklet at/after split_frame
        # (the "remnant" that wasn't split off)
        remnant_rows = detected[
            (detected['tracklet_id'] == orig) &
            (detected['frame_index'] >= split_frame)
        ]

        # GT on pre-split side
        if len(pre_rows) > 0:
            pre_gt_counts = Counter(pre_rows['gt_track_id'].tolist())
            pre_gt = pre_gt_counts.most_common(1)[0][0]
        else:
            pre_gt = tid_gt_map.get(orig)

        # GT on post-split side (the new product)
        if len(post_rows) > 0:
            post_gt_counts = Counter(post_rows['gt_track_id'].tolist())
            post_gt = post_gt_counts.most_common(1)[0][0]
        elif len(remnant_rows) > 0:
            # Fall back to remnant
            post_gt_counts = Counter(remnant_rows['gt_track_id'].tolist())
            post_gt = post_gt_counts.most_common(1)[0][0]
        else:
            post_gt = tid_gt_map.get(new_tid)

        if pre_gt is not None and post_gt is not None:
            if pre_gt == post_gt:
                classification = 'false_split'
            else:
                classification = 'correct_split'
        else:
            classification = 'unclassifiable'

        results.append({
            'original_tracklet_id': orig,
            'new_tracklet_id': new_tid,
            'split_frame': split_frame,
            'tier': tier,
            'pre_gt': pre_gt,
            'post_gt': post_gt,
            'classification': classification,
        })

    return results


# ---------------------------------------------------------------------------
# Jump detection (per GT person)
# ---------------------------------------------------------------------------

def detect_jumps(
    df: pd.DataFrame,
    split_events: list[dict],
    split_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Add jump_type + jump_from_person_ids columns to the dense join.

    Modifies df in-place and returns it.

    Two axes combined:
    1. Per-GT-person thread: WHERE a jump occurs (person_ids change)
    2. Per-tracklet purity: WHY the thread broke (drift vs misstitch)
    """
    df['jump_type'] = None
    df['jump_from_person_ids'] = None

    # Pre-compute tracklet GT purity (correction #2)
    tracklet_purity = _compute_tracklet_gt_purity(df)

    # Pre-compute D0.5 split boundaries: {(original_tid, split_frame)} for lookup
    split_boundaries: dict[tuple[str, int], dict] = {}
    for ev in split_events:
        key = (ev['original_tracklet_id'], ev.get('split_frame', -1))
        split_boundaries[key] = ev

    # Also index by product: {new_tracklet_id: split_event}
    product_to_split: dict[str, dict] = {}
    for ev in split_events:
        product_to_split[ev['new_tracklet_id']] = ev

    # Classify split events for false_split detection (correction #1)
    split_classifications = classify_split_events(split_events, df, split_map)
    # Index: (original, new_tid) -> classification
    split_class_idx: dict[tuple[str, str], str] = {}
    for sc in split_classifications:
        split_class_idx[(sc['original_tracklet_id'], sc['new_tracklet_id'])] = sc['classification']

    # Build split family for recognizing split-related tracklet changes
    split_family: dict[str, str] = {}  # product_tid -> original_tid
    for orig, prods in split_map.items():
        for p in prods:
            split_family[p] = orig

    # Per GT person: detect jumps
    for gt_tid, grp in df.groupby('gt_track_id'):
        grp_sorted = grp.sort_values('frame_index')
        indices = grp_sorted.index.tolist()

        # Track GROUP entry sets for membership drift
        group_entry_set: set[int] | None = None
        in_group = False

        prev_i = None
        for idx in indices:
            row = df.loc[idx]

            # Skip undetected frames
            if row['state'] in ('miss', 'untracked', 'no_canonical'):
                prev_i = None
                in_group = False
                group_entry_set = None
                continue

            curr_pids = set(json.loads(row['person_ids'])) if row['person_ids'] else set()
            curr_tid = row['tracklet_id']
            curr_is_group = bool(row['d1_is_group'])
            curr_ngs = _parse_gt_set(row['node_gt_set'])

            # --- Group membership drift ---
            if curr_is_group and curr_ngs:
                if not in_group:
                    # GROUP entry
                    group_entry_set = curr_ngs.copy()
                    in_group = True
                else:
                    # Mid-group: check if carried set drifted
                    if group_entry_set is not None and curr_ngs != group_entry_set:
                        df.at[idx, 'jump_type'] = 'group_membership_drift'
                        df.at[idx, 'jump_from_person_ids'] = json.dumps(
                            sorted(group_entry_set)
                        )
                        group_entry_set = curr_ngs.copy()
            else:
                if in_group:
                    in_group = False
                    group_entry_set = None

            # --- Thread break detection (need prev frame) ---
            if prev_i is None:
                prev_i = idx
                continue

            prev_row = df.loc[prev_i]
            if prev_row['state'] in ('miss', 'untracked', 'no_canonical'):
                prev_i = idx
                continue

            prev_pids = set(json.loads(prev_row['person_ids'])) if prev_row['person_ids'] else set()
            prev_tid = prev_row['tracklet_id']
            prev_is_group = bool(prev_row['d1_is_group'])

            # Thread break = person_ids changed
            if curr_pids == prev_pids or not curr_pids or not prev_pids:
                prev_i = idx
                continue

            # Don't overwrite group_membership_drift if already set
            if df.at[idx, 'jump_type'] is not None:
                prev_i = idx
                continue

            # --- Classify the break ---
            tid_changed = (curr_tid != prev_tid) and pd.notna(curr_tid) and pd.notna(prev_tid)

            if tid_changed:
                # Check if this is a D0.5 split boundary
                is_split = False
                split_class = None

                # Current tid might be a split product
                curr_tid_str = str(curr_tid)
                if curr_tid_str in product_to_split:
                    ev = product_to_split[curr_tid_str]
                    split_class = split_class_idx.get(
                        (ev['original_tracklet_id'], curr_tid_str)
                    )
                    is_split = True

                # Or previous tid's original might have a split at this frame
                prev_tid_str = str(prev_tid)
                curr_frame = int(row['frame_index'])
                if not is_split:
                    orig_of_prev = split_family.get(prev_tid_str, prev_tid_str)
                    if (orig_of_prev, curr_frame) in split_boundaries:
                        ev = split_boundaries[(orig_of_prev, curr_frame)]
                        split_class = split_class_idx.get(
                            (ev['original_tracklet_id'], ev['new_tracklet_id'])
                        )
                        is_split = True

                if is_split and split_class == 'false_split':
                    df.at[idx, 'jump_type'] = 'false_split'
                elif is_split and split_class == 'correct_split':
                    # A correct split that still broke the thread — this is
                    # expected (split separated two people; this GT person's
                    # thread moved to the other product). Not a jump error.
                    prev_i = idx
                    continue
                elif not is_split:
                    # Tracklet changed, not a D0.5 split.
                    # Check tracklet purity to distinguish misstitch from drift
                    prev_purity = tracklet_purity.get(prev_tid_str, {})
                    curr_purity_info = tracklet_purity.get(curr_tid_str, {})

                    # If prev tracklet was impure AND this GT person was NOT
                    # the dominant, the drift was in the prev tracklet
                    if (prev_purity and not prev_purity.get('is_pure', True)
                            and prev_purity.get('dominant_gt') != gt_tid):
                        df.at[idx, 'jump_type'] = 'tracklet_drift'
                    elif (curr_purity_info and not curr_purity_info.get('is_pure', True)
                          and curr_purity_info.get('dominant_gt') != gt_tid):
                        df.at[idx, 'jump_type'] = 'tracklet_drift'
                    else:
                        df.at[idx, 'jump_type'] = 'ilp_misstitch'
                else:
                    # Split but unclassifiable
                    df.at[idx, 'jump_type'] = 'ilp_misstitch'
            else:
                # Same tracklet, person_ids changed
                # Check GROUP boundary
                if curr_is_group != prev_is_group:
                    df.at[idx, 'jump_type'] = 'group_boundary_jump'
                else:
                    # Within same tracklet, no group transition — drift
                    df.at[idx, 'jump_type'] = 'tracklet_drift'

            df.at[idx, 'jump_from_person_ids'] = json.dumps(sorted(prev_pids))
            prev_i = idx

    # Summary
    jump_counts = df['jump_type'].value_counts(dropna=True).to_dict()
    logger.info("Jump detection: %s", jump_counts)
    return df


# ---------------------------------------------------------------------------
# D0.5 net-effect accounting (correction #3)
# ---------------------------------------------------------------------------

def compute_d05_net_effect(
    split_classifications: list[dict],
) -> dict:
    """Per-event D0.5 accounting.

    Returns summary with correct/false/unclassifiable counts per tier.
    """
    by_tier: dict[str, dict[str, int]] = defaultdict(lambda: {
        'correct_split': 0, 'false_split': 0, 'unclassifiable': 0, 'total': 0,
    })
    totals = {'correct_split': 0, 'false_split': 0, 'unclassifiable': 0, 'total': 0}

    for sc in split_classifications:
        tier = sc.get('tier', 'unknown')
        cls = sc['classification']
        by_tier[tier][cls] += 1
        by_tier[tier]['total'] += 1
        totals[cls] += 1
        totals['total'] += 1

    net = totals['correct_split'] - totals['false_split']
    return {
        'totals': totals,
        'net_effect': net,
        'net_verdict': 'net_positive' if net > 0 else ('net_negative' if net < 0 else 'neutral'),
        'by_tier': dict(by_tier),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gt_set(val) -> set[int] | None:
    if pd.isna(val) or val is None:
        return None
    try:
        return set(json.loads(val))
    except (json.JSONDecodeError, TypeError):
        return None


def load_split_events(stage_d_dir: Path) -> list[dict]:
    """Load D0.5 split events from audit JSONL."""
    audit_path = stage_d_dir / "d05_split_audit.jsonl"
    if not audit_path.exists():
        return []
    events = []
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get('artifact_type') == 'd05_split_event':
            events.append(ev)
    return events
