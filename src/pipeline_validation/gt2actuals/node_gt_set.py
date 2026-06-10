"""Node -> GT-identity-SET inversion (first-class CP-GT2ACTUALS-2 component).

For each D1 node at each frame, computes the set of GT track IDs whose matched
tracklets participate in that node. This is the primitive the group-integrity
design rests on: a GROUP node with node_gt_set_size >= 2 carries multiple real
people and is genuinely ambiguous.

INVARIANT: The set is {GT-of(t) for t in member_tracklets}, never the tracklet
IDs themselves. Each member tracklet maps through greedy_match to its GT track.

SPLIT-PRODUCT CONSISTENCY: Both sides of the join (D1 member tracklets and
greedy-match tracklets) are resolved through d05 split lineage. This is the
most likely inversion-bug site (split-product mismatch has bitten us twice).
CP-3 validates this inversion specifically.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_node_gt_sets(
    d1_lookup: dict[tuple[str, int], list[dict]],
    frame_gt_tracklets: dict[tuple[int, str], set[int]],
) -> dict[tuple[str, int], set[int]]:
    """Invert D1 node membership to GT-identity sets.

    Args:
        d1_lookup: (tracklet_id, frame_index) -> [node_info] from _build_d1_lookup.
            Each node_info has 'node_id', 'node_type', etc.
        frame_gt_tracklets: (frame_index, resolved_tracklet_id) -> {gt_track_ids}
            from greedy match. Both tracklet IDs here and in d1_lookup MUST be
            resolved through the same d05 split lineage.

    Returns:
        (node_id, frame_index) -> {gt_track_ids} — the GT people whose tracklets
        participate in this node at this frame.
    """
    # Collect: for each (node_id, frame) accumulate GT track IDs
    node_gt: dict[tuple[str, int], set[int]] = defaultdict(set)

    for (tid, fi), node_infos in d1_lookup.items():
        # Look up which GT people this tracklet carries at this frame
        gt_ids = frame_gt_tracklets.get((fi, tid), set())
        if not gt_ids:
            continue
        for info in node_infos:
            nid = info["node_id"]
            node_gt[(nid, fi)].update(gt_ids)

    logger.info(
        "node_gt_set inversion: %d (node, frame) entries, %d with size >= 2",
        len(node_gt),
        sum(1 for s in node_gt.values() if len(s) >= 2),
    )
    return dict(node_gt)
