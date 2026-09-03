# REACH-1: Graph Reachability Analysis — PRODUCTION

## Method

For each of the 8 GT persons on FP7oJQ-20260822-132650, this analysis:
1. Builds the **target path** — the ordered sequence of D1 nodes the GT person
   occupies, with detection gaps removed and contiguous runs collapsed.
2. Classifies each **hop** (consecutive node pair) in the target path.
3. Identifies **shared nodes** — D1 nodes needed by multiple GT people simultaneously.
4. Checks **independent reachability** — can each GT person's path be walked ignoring others?
5. Solves **joint feasibility** — maximum GT people whose paths coexist given capacity.

**Selected edges are inferred from `person_spans.parquet`:** consecutive nodes on the
same person_id path imply the solver selected the connecting edge. This is an inference
from the D4 output, not a recorded edge-selection artifact.

**D1 parameters:** `reconnect_max_gap_frames` = 250, `v_max_mps` = 8.0

## 1. Detection Coverage per GT Person

| GT | Total frames | Detected | Rate | Path nodes | Hops | Gap-separated hops |
|---|---|---|---|---|---|---|
| 0 | 1765 | 1704 | 96.5% | 61 | 60 | 29 (29/60 hops) |
| 1 | 1765 | 277 | 15.7% | 16 | 15 | 9 (9/15 hops) |
| 2 | 1765 | 864 | 48.9% | 104 | 103 | 58 (58/103 hops) |
| 3 | 1765 | 735 | 41.6% | 58 | 57 | 44 (44/57 hops) |
| 4 | 1765 | 807 | 45.7% | 86 | 85 | 81 (81/85 hops) |
| 5 | 1765 | 273 | 15.5% | 28 | 27 | 26 (26/27 hops) |
| 6 | 1765 | 28 | 1.6% | 9 | 8 | 8 (8/8 hops) |
| 7 | 120 | 59 | 49.2% | 3 | 2 | 2 (2/2 hops) |

## 2. Hop Classification

| Outcome | Count | % |
|---|---|---|
| EDGE_EXISTS_SELECTED | 65 | 18.2% |
| EDGE_EXISTS_NOT_SELECTED | 15 | 4.2% |
| CONCURRENT_NODES | 267 | 74.8% |
| EDGE_ABSENT_IN_WINDOW | 5 | 1.4% |
| UNREACHABLE_BY_WINDOW | 5 | 1.4% |
| **Total** | **357** | |

### CONCURRENT_NODES — overlapping nodes, no temporal edge possible

These hops are between D1 nodes whose frame ranges overlap (frame_gap <= 0).
D1 edges represent temporal transitions; they cannot connect simultaneous nodes.
This is the NOEDGE-1 finding: concurrent-node flicker from detection under-segmentation.

| GT | Concurrent hops | Example src→dst |
|---|---|---|
| 0 | 38 | `G:121-362:carrier=t2:d=t21:n=t49` → `G:121-362:carrier=t2:d=t21:n=t49` |
| 1 | 4 | `T:t1:s0:0-8` → `T:t8` |
| 2 | 86 | `G:0-201:carrier=t3:d=none:n=t34` → `G:0-201:carrier=t3:d=none:n=t34` |
| 3 | 39 | `T:t10:s0:35-103` → `T:t10:s0:35-103` |
| 4 | 73 | `T:t49` → `T:t49` |
| 5 | 20 | `G:4-76:carrier=t4:d=t6:n=none` → `G:4-76:carrier=t4:d=t6:n=none` |
| 6 | 5 | `G:4-76:carrier=t4:d=t6:n=none` → `G:4-76:carrier=t4:d=t6:n=none` |
| 7 | 2 | `T:t36` → `T:t36` |

### EDGE_ABSENT_IN_WINDOW — D1 should have generated these

| GT | Hop | Src node | Dst node | Frame gap | Dist (m) | Speed (m/s) |
|---|---|---|---|---|---|---|
| 2 | 88 | `T:t147:s0:1352-1355` | `T:t152` | 26 | 0.654 | 0.38 |
| 2 | 90 | `G:1386-1558:carrier=t147:d=t152:n=t158` | `G:1561-1649:carrier=t147:d=t158:n=t163` | 3 | 0.033 | 0.16 |
| 3 | 26 | `G:631-704:carrier=t67:d=t79:n=none` | `G:750-820:carrier=t90:d=t67_s2:n=t94` | 46 | 0.936 | 0.3 |
| 3 | 38 | `T:t126` | `G:1356-1380:carrier=t147:d=t4_s12:n=t152` | 130 | 0.492 | 0.06 |
| 4 | 70 | `T:t2:s14:1265-1577` | `T:t161` | 11 | 0.38 | 0.51 |

### UNREACHABLE_BY_WINDOW — correctly excluded by D1 limits

| GT | Hop | Frame gap | Gap margin | Speed (m/s) | Speed margin |
|---|---|---|---|---|---|
| 1 | 11 | 253 | +3 | 0.03 | +? |
| 2 | 78 | 254 | +4 | 0.04 | +? |
| 3 | 36 | 289 | +39 | 0.01 | +? |
| 5 | 8 | 802 | +552 | 0.04 | +? |
| 6 | 7 | 1619 | +1369 | 0.02 | +? |

### EDGE_EXISTS_NOT_SELECTED — edge available, solver chose otherwise

| GT | Hop | Src node | Dst node | Cost | Capacity blocked? |
|---|---|---|---|---|---|
| 0 | 31 | `G:731-1041:carrier=t88:d=none:n=t110` | `T:t88:s1:1042-1042` | 0.01648509723196163 | True |
| 1 | 12 | `T:t70` | `T:t71` | 1000000.0 | True |
| 1 | 14 | `T:t73` | `T:t92` | 3.01 | True |
| 2 | 54 | `G:667-702:carrier=t3:d=t82:n=none` | `T:t90:s0:749-749` | 3.01 | True |
| 2 | 86 | `T:t126` | `T:t141` | 1000000.0 | True |
| 2 | 87 | `T:t141` | `T:t147:s0:1352-1355` | 1000000.0 | True |
| 3 | 3 | `G:104-136:carrier=t10:d=t16:n=none` | `T:t30` | 3.26 | True |
| 3 | 9 | `T:t30` | `T:t52` | 3.01 | True |
| 3 | 11 | `T:t52` | `T:t67:s0:510-522` | 3.01 | True |
| 3 | 54 | `T:t163` | `T:t166` | 3.01 | True |
| 4 | 16 | `T:t62` | `T:t81` | 1000000.0 | True |
| 5 | 6 | `G:4-76:carrier=t4:d=t6:n=none` | `T:t23` | 3.26 | True |
| 5 | 13 | `T:t106` | `T:t115` | 3.26 | True |
| 5 | 20 | `T:t120` | `T:t137` | 3.26 | True |
| 6 | 6 | `G:4-76:carrier=t4:d=t6:n=none` | `T:t23` | 3.26 | True |

**Capacity-blocked:** 15 | **Cost-beaten:** 0

## 3. Shared Node Analysis

**Frame-level co-occupancy (structural impossibility):** 0
**Frame-level co-occupancy (GROUP handles it):** 1
**Sequential use (same node, different frames — no contention):** 27

### GROUP nodes correctly serving co-occupied GT people

1 GROUP nodes (capacity >= 2) where two GT people co-occupy at the same frame.
This is correct behavior — GROUP nodes exist to represent two people on one tracklet.

### Sequential use — same node, interleaved frames, no capacity conflict

27 node-pairs where two GT people use the same node at different frames.
With Hungarian matching, two GT people never match the same detection at the same frame.
A capacity-1 SOLO node can serve both people sequentially — one gets correct attribution
per frame, the other gets misattribution. This is not a structural impossibility;
it is the detection under-segmentation problem expressed as misattribution, not as
a graph capacity limit.

| Node | GT A (frames) | GT B (frames) | Capacity | Seg type |
|---|---|---|---|---|
| `G:1386-1558:carrier=t147:d=t152:n=t158` | 2 (1f) | 3 (171f) | 2 | GROUP |
| `G:1561-1649:carrier=t147:d=t158:n=t163` | 2 (1f) | 3 (87f) | 2 | GROUP |
| `G:1649-1716:carrier=t2:d=t161:n=t165` | 0 (66f) | 4 (1f) | 2 | GROUP |
| `G:1736-1763:carrier=t2:d=t165:n=none` | 0 (9f) | 4 (19f) | 2 | GROUP |
| `G:4-76:carrier=t4:d=t6:n=none` | 5 (35f) | 6 (23f) | 2 | GROUP |
| `G:404-488:carrier=t2:d=t51:n=t62` | 0 (82f) | 4 (2f) | 2 | GROUP |
| `G:683-711:carrier=t2:d=t81_s1:n=t86` | 0 (28f) | 4 (1f) | 2 | GROUP |
| `G:750-820:carrier=t90:d=t67_s2:n=t94` | 2 (54f) | 3 (14f) | 2 | GROUP |
| `G:823-921:carrier=t90:d=t94:n=none` | 2 (54f) | 3 (2f) | 2 | GROUP |
| `G:895-1264:carrier=t2:d=t102:n=t135` | 0 (31f) | 4 (285f) | 2 | GROUP |
| `T:t120` | 2 (4f) | 5 (56f) | 1 | SOLO |
| `T:t126` | 2 (8f) | 3 (9f) | 1 | SOLO |
| `T:t135` | 0 (1f) | 4 (213f) | 1 | SOLO |
| `T:t147:s6:1650-1754` | 2 (66f) | 3 (4f) | 1 | SOLO |
| `T:t152` | 2 (2f) | 3 (2f) | 1 | SOLO |
| `T:t163` | 2 (14f) | 3 (42f) | 1 | SOLO |
| `T:t165` | 0 (3f) | 4 (1f) | 1 | SOLO |
| `T:t166` | 2 (2f) | 3 (22f) | 1 | SOLO |
| `T:t23` | 5 (1f) | 6 (2f) | 1 | SOLO |
| `T:t2:s12:712-894` | 0 (3f) | 4 (161f) | 1 | SOLO |
| `T:t2:s14:1265-1577` | 0 (304f) | 4 (1f) | 1 | SOLO |
| `T:t2:s16:1588-1648` | 0 (54f) | 4 (5f) | 1 | SOLO |
| `T:t2:s18:1717-1735` | 0 (16f) | 4 (3f) | 1 | SOLO |
| `T:t2:s5:363-403` | 0 (38f) | 4 (1f) | 1 | SOLO |
| `T:t49` | 0 (1f) | 4 (11f) | 1 | SOLO |
| `T:t62` | 0 (4f) | 4 (30f) | 1 | SOLO |
| `T:t86` | 0 (15f) | 4 (1f) | 1 | SOLO |

## 4a. Independent Reachability (ignoring contention)

| GT | Reachable? | Path nodes | Hops | Selected | Not selected | Concurrent | Absent | Unreachable |
|---|---|---|---|---|---|---|---|---|
| 0 | NO | 61 | 60 | 21 | 1 | 38 | 0 | 0 |
| 1 | NO | 16 | 15 | 8 | 2 | 4 | 0 | 1 |
| 2 | NO | 104 | 103 | 11 | 3 | 86 | 2 | 1 |
| 3 | NO | 58 | 57 | 11 | 4 | 39 | 2 | 1 |
| 4 | NO | 86 | 85 | 10 | 1 | 73 | 1 | 0 |
| 5 | NO | 28 | 27 | 3 | 3 | 20 | 0 | 1 |
| 6 | NO | 9 | 8 | 1 | 1 | 5 | 0 | 1 |
| 7 | NO | 3 | 2 | 0 | 0 | 2 | 0 | 0 |

**Independent reachability: 0 / 8 GT people**

## 4b. Joint Feasibility (respecting node capacities)

**Method:** Exhaustive search over all 2^8 = 256 subsets.
For each subset, verify that every shared node has capacity >= number of GT people needing it simultaneously.

**Result: ALL 8 GT people can coexist.** No capacity contention.

## 5. Aggregate by Owner

| Owner | Category | Count |
|---|---|---|
| Working correctly | EDGE_EXISTS_SELECTED | 65 |
| D2 cost / D3 solve | EDGE_EXISTS_NOT_SELECTED | 15 |
|   of which: capacity-blocked | | 15 |
|   of which: cost-beaten | | 0 |
| Detection (concurrent nodes) | CONCURRENT_NODES | 267 |
| D1 candidate generation | EDGE_ABSENT_IN_WINDOW | 5 |
| D1 parameters / detection | UNREACHABLE_BY_WINDOW | 5 |
| Detection (co-occupied SOLO) | SHARED_NODE structural impossibility | 0 node-pairs |
| Detection (sequential use) | Same node, interleaved frames | 27 node-pairs |

## Summary Verdict

- **Independent reachability (a):** 0 / 8 GT people have a connected path
- **Joint feasibility (b):** 8 / 8 GT people can coexist given capacity

**Finding:** 8 GT people lack connected paths.
The ceiling includes connectivity (edge generation), not just solver decisions.


---

# REACH-1: Graph Reachability Analysis — DEDUP-CEILING

## Method

For each of the 8 GT persons on FP7oJQ-20260822-132650, this analysis:
1. Builds the **target path** — the ordered sequence of D1 nodes the GT person
   occupies, with detection gaps removed and contiguous runs collapsed.
2. Classifies each **hop** (consecutive node pair) in the target path.
3. Identifies **shared nodes** — D1 nodes needed by multiple GT people simultaneously.
4. Checks **independent reachability** — can each GT person's path be walked ignoring others?
5. Solves **joint feasibility** — maximum GT people whose paths coexist given capacity.

**Selected edges are inferred from `person_spans.parquet`:** consecutive nodes on the
same person_id path imply the solver selected the connecting edge. This is an inference
from the D4 output, not a recorded edge-selection artifact.

**D1 parameters:** `reconnect_max_gap_frames` = 250, `v_max_mps` = 8.0

## 1. Detection Coverage per GT Person

| GT | Total frames | Detected | Rate | Path nodes | Hops | Gap-separated hops |
|---|---|---|---|---|---|---|
| 0 | 1764 | 1525 | 86.5% | 15 | 14 | 3 (3/14 hops) |
| 1 | 1764 | 1623 | 92.0% | 4 | 3 | 0 (0/3 hops) |
| 2 | 1764 | 947 | 53.7% | 21 | 20 | 8 (8/20 hops) |
| 3 | 1764 | 1140 | 64.6% | 18 | 17 | 6 (6/17 hops) |
| 4 | 1764 | 552 | 31.3% | 11 | 10 | 5 (5/10 hops) |
| 5 | 1764 | 1169 | 66.3% | 3 | 2 | 2 (2/2 hops) |
| 6 | 1764 | 68 | 3.9% | 3 | 2 | 2 (2/2 hops) |
| 7 | 120 | 72 | 60.0% | 1 | 0 | 0 (0/0 hops) |

## 2. Hop Classification

| Outcome | Count | % |
|---|---|---|
| EDGE_EXISTS_SELECTED | 28 | 41.2% |
| EDGE_EXISTS_NOT_SELECTED | 13 | 19.1% |
| CONCURRENT_NODES | 17 | 25.0% |
| EDGE_ABSENT_IN_WINDOW | 4 | 5.9% |
| UNREACHABLE_BY_WINDOW | 6 | 8.8% |
| **Total** | **68** | |

### CONCURRENT_NODES — overlapping nodes, no temporal edge possible

These hops are between D1 nodes whose frame ranges overlap (frame_gap <= 0).
D1 edges represent temporal transitions; they cannot connect simultaneous nodes.
This is the NOEDGE-1 finding: concurrent-node flicker from detection under-segmentation.

| GT | Concurrent hops | Example src→dst |
|---|---|---|
| 0 | 5 | `T:t2:s1:363-403` → `T:t49` |
| 2 | 4 | `T:t82` → `T:t3:s5:664-666` |
| 3 | 5 | `G:750-820:carrier=t90:d=t67_s2:n=t9` → `T:t93` |
| 4 | 3 | `T:t49` → `T:t2:s1:363-403` |

### EDGE_ABSENT_IN_WINDOW — D1 should have generated these

| GT | Hop | Src node | Dst node | Frame gap | Dist (m) | Speed (m/s) |
|---|---|---|---|---|---|---|
| 2 | 18 | `T:t147:s0:1352-1355` | `T:t152` | 26 | 0.654 | 0.38 |
| 3 | 4 | `T:t67` | `G:750-820:carrier=t90:d=t67_s2:n=t94` | 46 | 0.936 | 0.3 |
| 3 | 12 | `T:t126` | `G:1356-1380:carrier=t147:d=t4_s17:n=t152` | 130 | 0.492 | 0.06 |
| 4 | 5 | `T:t81` | `G:683-719:carrier=t2:d=t81_s1:n=none` | 11 | 0.997 | 1.35 |

### UNREACHABLE_BY_WINDOW — correctly excluded by D1 limits

| GT | Hop | Frame gap | Gap margin | Speed (m/s) | Speed margin |
|---|---|---|---|---|---|
| 0 | 12 | 323 | +73 | 0.0 | +? |
| 2 | 14 | 254 | +4 | 0.04 | +? |
| 3 | 9 | 254 | +4 | 0.04 | +? |
| 4 | 7 | 371 | +121 | 0.03 | +? |
| 5 | 1 | 1137 | +887 | 0.02 | +? |
| 6 | 1 | 1619 | +1369 | 0.02 | +? |

### EDGE_EXISTS_NOT_SELECTED — edge available, solver chose otherwise

| GT | Hop | Src node | Dst node | Cost | Capacity blocked? |
|---|---|---|---|---|---|
| 0 | 13 | `T:t110` | `T:t135` | 3.01 | True |
| 2 | 7 | `G:667-702:carrier=t3:d=t82:n=none` | `T:t90:s0:749-749` | 3.01 | True |
| 2 | 15 | `T:t124` | `T:t126` | 3.01 | True |
| 2 | 16 | `T:t126` | `T:t141` | 1000000.0 | True |
| 2 | 17 | `T:t141` | `T:t147:s0:1352-1355` | 1000000.0 | True |
| 3 | 1 | `G:104-136:carrier=t10:d=t16:n=none` | `T:t30` | 3.26 | True |
| 3 | 2 | `T:t30` | `T:t52` | 3.01 | True |
| 3 | 3 | `T:t52` | `T:t67` | 3.01 | True |
| 3 | 10 | `T:t124` | `T:t126` | 3.01 | True |
| 4 | 4 | `T:t62` | `T:t81` | 1000000.0 | True |
| 4 | 8 | `T:t135` | `T:t161` | 1000000.0 | True |
| 5 | 0 | `T:t4` | `T:t23` | 3.26 | True |
| 6 | 0 | `T:t4` | `T:t23` | 3.26 | True |

**Capacity-blocked:** 13 | **Cost-beaten:** 0

## 3. Shared Node Analysis

**Frame-level co-occupancy (structural impossibility):** 0
**Frame-level co-occupancy (GROUP handles it):** 0
**Sequential use (same node, different frames — no contention):** 14

### Sequential use — same node, interleaved frames, no capacity conflict

14 node-pairs where two GT people use the same node at different frames.
With Hungarian matching, two GT people never match the same detection at the same frame.
A capacity-1 SOLO node can serve both people sequentially — one gets correct attribution
per frame, the other gets misattribution. This is not a structural impossibility;
it is the detection under-segmentation problem expressed as misattribution, not as
a graph capacity limit.

| Node | GT A (frames) | GT B (frames) | Capacity | Seg type |
|---|---|---|---|---|
| `G:1386-1681:carrier=t147:d=t152:n=none` | 2 (31f) | 3 (263f) | 2 | GROUP |
| `G:404-488:carrier=t2:d=t51:n=t62` | 0 (82f) | 4 (2f) | 2 | GROUP |
| `G:683-719:carrier=t2:d=t81_s1:n=none` | 0 (36f) | 4 (1f) | 2 | GROUP |
| `G:750-820:carrier=t90:d=t67_s2:n=t94` | 2 (54f) | 3 (14f) | 2 | GROUP |
| `G:823-921:carrier=t90:d=t94:n=none` | 2 (54f) | 3 (2f) | 2 | GROUP |
| `T:t124` | 2 (1f) | 3 (1f) | 1 | SOLO |
| `T:t126` | 2 (8f) | 3 (9f) | 1 | SOLO |
| `T:t135` | 0 (1f) | 4 (213f) | 1 | SOLO |
| `T:t152` | 2 (2f) | 3 (2f) | 1 | SOLO |
| `T:t23` | 5 (1f) | 6 (2f) | 1 | SOLO |
| `T:t2:s1:363-403` | 0 (38f) | 4 (1f) | 1 | SOLO |
| `T:t4` | 5 (35f) | 6 (25f) | 1 | SOLO |
| `T:t49` | 0 (1f) | 4 (11f) | 1 | SOLO |
| `T:t62` | 0 (4f) | 4 (30f) | 1 | SOLO |

## 4a. Independent Reachability (ignoring contention)

| GT | Reachable? | Path nodes | Hops | Selected | Not selected | Concurrent | Absent | Unreachable |
|---|---|---|---|---|---|---|---|---|
| 0 | NO | 15 | 14 | 7 | 1 | 5 | 0 | 1 |
| 1 | YES | 4 | 3 | 3 | 0 | 0 | 0 | 0 |
| 2 | NO | 21 | 20 | 10 | 4 | 4 | 1 | 1 |
| 3 | NO | 18 | 17 | 5 | 4 | 5 | 2 | 1 |
| 4 | NO | 11 | 10 | 3 | 2 | 3 | 1 | 1 |
| 5 | NO | 3 | 2 | 0 | 1 | 0 | 0 | 1 |
| 6 | NO | 3 | 2 | 0 | 1 | 0 | 0 | 1 |
| 7 | YES | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

**Independent reachability: 2 / 8 GT people**

## 4b. Joint Feasibility (respecting node capacities)

**Method:** Exhaustive search over all 2^8 = 256 subsets.
For each subset, verify that every shared node has capacity >= number of GT people needing it simultaneously.

**Result: ALL 8 GT people can coexist.** No capacity contention.

## 5. Aggregate by Owner

| Owner | Category | Count |
|---|---|---|
| Working correctly | EDGE_EXISTS_SELECTED | 28 |
| D2 cost / D3 solve | EDGE_EXISTS_NOT_SELECTED | 13 |
|   of which: capacity-blocked | | 13 |
|   of which: cost-beaten | | 0 |
| Detection (concurrent nodes) | CONCURRENT_NODES | 17 |
| D1 candidate generation | EDGE_ABSENT_IN_WINDOW | 4 |
| D1 parameters / detection | UNREACHABLE_BY_WINDOW | 6 |
| Detection (co-occupied SOLO) | SHARED_NODE structural impossibility | 0 node-pairs |
| Detection (sequential use) | Same node, interleaved frames | 14 node-pairs |

## Summary Verdict

- **Independent reachability (a):** 2 / 8 GT people have a connected path
- **Joint feasibility (b):** 8 / 8 GT people can coexist given capacity

**Finding:** 6 GT people lack connected paths.
The ceiling includes connectivity (edge generation), not just solver decisions.


---

# Cross-Check: Production vs Dedup-Ceiling

| Metric | Production | Dedup-Ceiling |
|---|---|---|
| Total hops | 357 | 68 |
| EDGE_EXISTS_SELECTED | 65 | 28 |
| EDGE_EXISTS_NOT_SELECTED | 15 | 13 |
| CONCURRENT_NODES | 267 | 17 |
| EDGE_ABSENT_IN_WINDOW | 5 | 4 |
| UNREACHABLE_BY_WINDOW | 5 | 6 |
| Independent reachability | 0/8 | 2/8 |
| Joint feasibility | 8/8 | 8/8 |
| Shared: structural impossibility | 0 | 0 |
| Shared: GROUP handles it | 1 | 0 |
| Shared: sequential (no contention) | 27 | 14 |

---

# Follow-up Verification

## A. Capacity-blocked re-verification (frame-level)

The 15 production not-selected edges (and 13 dedup) were originally classified as
capacity-blocked using the same code path. After the range-envelope bug was found in the
shared-node analysis, the capacity check was re-verified at frame-level: for each not-selected
edge A→B, check the actual D3 flow (from `person_spans`) at the specific transition frame
(node A's end_frame and node B's start_frame) against each node's capacity.

**Result: all 15 confirmed capacity-blocked at frame-level.** Every endpoint node had
`flow >= capacity` at the transition frame. 0 cost-beaten.

The capacity-blocked check in the hop classifier uses `person_spans` flow counts per node (total
person_ids routed through), which is node-lifetime-level rather than per-frame. This is
conservative — if a node has 1 person routed through it across its lifetime and capacity is 1,
it is saturated at every frame. The frame-level re-verification confirms the node-lifetime check
did not inflate: every endpoint was genuinely saturated at the specific frame that matters.

**The "0 cost-beaten" finding holds with the corrected method.**

## B. CONCURRENT_NODES decomposition — flicker vs genuine transitions

267 concurrent hops is the raw count. As NOEDGE-1 established, many of these are GT-matcher
flicker — the matcher alternating between valid overlapping detections frame by frame.

**Decomposition:**

| Category | Hops | % of 267 | Distinct events |
|---|---|---|---|
| Self-loop (A→gap→A, same node) | 191 | 71.5% | 53 |
| Out-and-back (A→B→A) | 46 | 17.2% | ~23 round-trips |
| One-way concurrent transition | 30 | 11.2% | 27 |
| **Total** | **267** | | **80** |

- **191 self-loops (71.5%):** The GT person is on node A, undetected for some frames, then
  re-detected on the same node A. This is a detection gap within a single node's span, not a
  transition between different nodes. The path builder creates two separate runs on the same
  node. These are pure measurement artifacts — the person never left the node.

- **46 out-and-back (17.2%):** A→B→A patterns where the matcher alternates between two
  overlapping nodes. Each round trip is one flicker event, not two transitions. ~23 distinct
  round-trip events.

- **30 one-way (11.2%):** Genuine transitions between concurrent nodes (one direction only
  within the local context). 27 distinct events.

**Deduplicated event count: 80** (53 self-loop + 27 between-node). The 267 hops inflate the
true concurrent-transition picture by **3.3×** (267/80).

**Interpretation:** The evaluation framework (Hungarian matching, IoU 0.5) cannot distinguish
one person's two concurrent detections from two different people's detections. When a GT
person's box overlaps two pipeline detections, the matcher picks one per frame — sometimes A,
sometimes B — creating the flicker pattern. This is a measurement limitation as much as a
pipeline one. The 27 between-node concurrent events are the genuine graph-level expression of
detection under-segmentation for this clip; the other 53 events (self-loops) are detection
gaps within single nodes.

## C. GROUP trigger mechanism

GROUP formation in `d1_graph_build.py` is triggered by tracklet lifecycle events:

1. **Merge trigger** (line 805): tracklet `disappear` ends while tracklet `carrier` continues,
   within `merge_dist_m`. Requires two concurrent tracklets.
2. **Split trigger** (line 853): tracklet `new` starts while tracklet `carrier` exists,
   within `split_dist_m`. Requires two concurrent tracklets.

Both triggers depend on a second concurrent tracklet existing. The dedup merge collapsed
concurrent tracklets, removing the evidence D1 uses to form groups. This explains the
GROUP count drop (47→36, -23%). Of the 6 nodes originally flagged as "shared SOLOs" in
the dedup arm, only 1 (`T:t4`) was GROUP in production and became SOLO after dedup;
the other 5 were already SOLO in both graphs.

## D. Methodological note

The range-envelope bug produced a plausible, mechanistically-explainable false finding
("shared SOLO nodes = detection ceiling") that survived one review. Two hypotheses were
offered for the production-vs-dedup discrepancy (GROUP trigger removal; artifact of dedup
simulation), and both were partially wrong. The actual cause — range-envelope inflation
over sparse frame occupancy — was found only by checking frame-level co-occupancy directly.

The false finding was coherent with the project's prior understanding (under-segmentation is
the dominant lever) and had a clean causal story (one box, two people, capacity 1). This made
it resistant to challenge. The lesson: a finding that confirms expectations and has a plausible
mechanism is not thereby verified. Checking the specific data (frame-level occupancy) rather
than a derived summary (range envelopes) would have caught it immediately.

---

# Corrected Conclusion

The graph **CAN** represent all 8 GT people correctly. 8/8 jointly feasible, 0 structural
impossibility, in both the production and dedup-ceiling arms. GROUP nodes work as designed.

**What prevents correct identity is connectivity, not capacity or cost:**

- **0/8 independently reachable** (production) — every GT person's target path includes
  hops between nodes that no temporal edge can connect (concurrent nodes) or that D1
  did not generate edges for.
- **0 cost-beaten edges** — the solver has never chosen a wrong edge when the correct one
  was available and capacity existed. Every declined edge (15 production, 13 dedup) was
  at a node already saturated by another person's flow.
- **267 concurrent hops → 80 distinct events → 27 between-node concurrent events** —
  the headline number is dominated by matcher flicker (self-loops and out-and-back).
  The genuine graph-level expression of under-segmentation on this clip is 27 events
  where the GT person transitions between two concurrent nodes.
