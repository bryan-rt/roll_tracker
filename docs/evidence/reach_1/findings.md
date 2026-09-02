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

**Shared SOLO (capacity=1) — structural impossibility:** 0
**Shared GROUP (capacity>=2, can coexist):** 1
**Shared GROUP (capacity>=2, over-subscribed):** 0

### GROUP nodes correctly serving multiple GT people

1 GROUP nodes with capacity >= 2 serving exactly 2 GT people.
This is correct behavior — GROUP nodes exist to represent two people on one tracklet.

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
| Detection (shared SOLO nodes) | SHARED_NODE structural impossibility | 0 node-pairs |

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

**Shared SOLO (capacity=1) — structural impossibility:** 6
**Shared GROUP (capacity>=2, can coexist):** 5
**Shared GROUP (capacity>=2, over-subscribed):** 0

### SOLO nodes shared by multiple GT people (DETECTION CEILING)

| Node | GT A | GT B | Capacity | Seg type | D3 routed |
|---|---|---|---|---|---|
| `T:t126` | 2 | 3 | 1 | SOLO | 1 (p0007) |
| `T:t152` | 2 | 3 | 1 | SOLO | 1 (p0011) |
| `T:t23` | 5 | 6 | 1 | SOLO | 1 (p0011) |
| `T:t2:s1:363-403` | 0 | 4 | 1 | SOLO | 1 (p0003) |
| `T:t4` | 5 | 6 | 1 | SOLO | 1 (p0006) |
| `T:t62` | 0 | 4 | 1 | SOLO | 1 (p0005) |

### GROUP nodes correctly serving multiple GT people

5 GROUP nodes with capacity >= 2 serving exactly 2 GT people.
This is correct behavior — GROUP nodes exist to represent two people on one tracklet.

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

**Result: Maximum 5 of 8 GT people can coexist.**
- Best feasible subset: GT [0, 1, 2, 5, 7]
- Excluded: GT [3, 4, 6]

**Blocking nodes (capacity < simultaneous GT demand):**

| Node | Capacity | Seg type | Max simultaneous GT | GT people |
|---|---|---|---|---|
| `T:t2:s1:363-403` | 1 | SOLO | 2 | [0, 4] |
| `T:t62` | 1 | SOLO | 2 | [0, 4] |
| `T:t126` | 1 | SOLO | 2 | [2, 3] |
| `T:t152` | 1 | SOLO | 2 | [2, 3] |
| `T:t4` | 1 | SOLO | 2 | [5, 6] |
| `T:t23` | 1 | SOLO | 2 | [5, 6] |


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
| Detection (shared SOLO nodes) | SHARED_NODE structural impossibility | 6 node-pairs |

## Summary Verdict

- **Independent reachability (a):** 2 / 8 GT people have a connected path
- **Joint feasibility (b):** 5 / 8 GT people can coexist given capacity

**Finding:** 6 GT people lack connected paths.
The ceiling includes connectivity (edge generation), not just solver decisions.
**Finding:** Contention limits joint feasibility to 5.
Under-segmentation propagates into the graph — one detection covering two grapplers
becomes one SOLO node covering two people. No stitching or cost work can fix this.


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
| Joint feasibility | 8/8 | 5/8 |
| Shared SOLO nodes | 0 | 6 |
| Shared GROUP (ok) | 1 | 5 |
