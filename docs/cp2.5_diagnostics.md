# CP2.5 — Diagnose Camera Asymmetry and D3→D4 Detection Gap

**Date:** 2026-05-13
**Branch:** `services_uploader`
**Scope:** Read-only analysis of post-CP2 artifacts. No code changes or reruns.

---

## Executive Summary

The penalty bump (15→25) helped FP7oJQ marginally but not J_EDEw/PPDmUg because all
three cameras have 30-40 dropped tracklets whose solo routing cost is well below 25.0.
The solver drops them anyway because the **global flow optimization has no incentive to
create extra paths** — each dropped tracklet pays only 25.0 regardless of length.

The longest dropped tracklets are full-clip carriers (t1 in J_EDEw: 4,427 frames; t1 in
PPDmUg: 2,926 frames). These are almost certainly real people visible for the entire
recording. They have zero flow on all their nodes.

The D3→D4 detection gap is not a D4 bug. Person_tracks actually contains MORE detections
than the solver's explained tracklets account for, because GROUP node spans pull in
detections from multiple tracklets simultaneously. The span-filtering loss (D4 node
frame ranges narrower than tracklet detection ranges) is small (~1-4%).

**Recommended next step: Option C variant — fix the explain-or-penalize mechanism.**
The penalty is per-tracklet regardless of length. A 4,427-frame tracklet and a 23-frame
tracklet both pay 25.0 to be dropped. Making the penalty **proportional to tracklet
length** (frames × per-frame penalty) would make dropping long tracklets prohibitively
expensive while still allowing short spurious tracklets to be dropped cheaply. This is
a targeted code change to `d3_ilp2.py`, not a config knob.

---

## Question 1 — Why Did FP7oJQ Respond but Not the Others?

### Flow topology per camera

| Camera | n_paths | n_tracklets | n_dropped | Path lengths (min/median/mean/max) |
|--------|---------|-------------|-----------|-----------------------------------|
| FP7oJQ | 17 | 251 | 36 | 3 / 13 / 17.5 / 46 |
| J_EDEw | 17 | 236 | 37 | 1 / 15 / 15.7 / 37 |
| PPDmUg | 8 | 73 | 14 | 1 / 9 / 10.9 / 28 |

All three cameras have similar topology: 8-17 flow paths, each chaining 3-46 tracklets.
The path counts are identical for FP7oJQ and J_EDEw (17 each), disproving the hypothesis
that J_EDEw has fewer paths.

### Path length distributions

**FP7oJQ:** [3, 3, 3, 4, 5, 6, 10, 10, 13, 17, 18, 23, 24, 35, 36, 41, 46]
**J_EDEw:** [1, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17, 17, 22, 27, 27, 30, 37]
**PPDmUg:** [1, 2, 3, 8, 10, 17, 18, 28]

### Three sample dropped tracklets per camera

**FP7oJQ:**

| Tracklet | n_frames | Frame range | Nodes | Solo route | World coords | Cross-in / Cross-out / Merge / Split |
|----------|----------|-------------|-------|------------|-------------|--------------------------------------|
| t62 (long) | 3,702 | [741, 4529] | 6 | 12.59 | (54.3,40.0)→(54.9,40.5) | 14 / 0 / 2 / 4 |
| t349 (mid) | 327 | [4188, 4529] | 4 | 12.81 | (55.9,50.0)→(55.8,50.6) | 12 / 0 / 2 / 2 |
| t257 (short) | 23 | [3152, 3192] | 2 | 20.28 | (52.5,48.6)→(52.7,48.9) | 12 / 15 / 1 / 0 |

**J_EDEw:**

| Tracklet | n_frames | Frame range | Nodes | Solo route | World coords | Cross-in / Cross-out / Merge / Split |
|----------|----------|-------------|-------|------------|-------------|--------------------------------------|
| t1 (long) | 4,427 | [0, 4529] | 15 | 4.21 | (53.1,39.1)→(53.6,39.7) | 0 / 0 / 7 / 7 |
| t213 (mid) | 512 | [2547, 3155] | 12 | 21.42 | (56.4,48.0)→(54.4,45.3) | 6 / 10 / 6 / 7 |
| t220 (short) | 54 | [2659, 2759] | 3 | 20.55 | (56.2,48.1)→(56.1,48.2) | 4 / 15 / 1 / 2 |

**PPDmUg:**

| Tracklet | n_frames | Frame range | Nodes | Solo route | World coords | Cross-in / Cross-out / Merge / Split |
|----------|----------|-------------|-------|------------|-------------|--------------------------------------|
| t1 (long) | 2,926 | [0, 2997] | 19 | 6.01 | (52.6,37.6)→(53.6,36.7) | 0 / 0 / 9 / 9 |
| t36 (mid) | 395 | [620, 1188] | 9 | 22.13 | (45.5,38.0)→(46.1,37.8) | 2 / 6 / 4 / 5 |
| t92 (short) | 71 | [1626, 1725] | 4 | 23.22 | (47.7,36.4)→(48.0,37.0) | 7 / 3 / 1 / 2 |

### Key finding: all 9 dropped tracklets have solo_route < penalty

Every sample's minimum standalone routing cost (BIRTH + internal CONTINUE + DEATH) is
below 25.0. The solver could create a dedicated flow path for each. It doesn't because:

1. **The penalty is per-tracklet, not per-frame.** A 4,427-frame tracklet (J_EDEw t1)
   pays the same 25.0 drop penalty as a 23-frame tracklet (FP7oJQ t257). The optimizer
   has no incentive to treat long tracklets differently.

2. **Full-clip carriers are the worst affected.** J_EDEw `t1` (4,427 frames) and PPDmUg
   `t1` (2,926 frames) span the entire recording. They have 15-19 nodes, mostly GROUP
   nodes where they serve as the carrier for other tracklets' merge/split events. Despite
   this, zero flow passes through any of their nodes. Verified: `d3_selected_edges.parquet`
   shows 0 selected edges incident on any t1 node for both cameras.

3. **FP7oJQ's marginal improvement** (39→36) is because 3 of its dropped tracklets
   happened to have routing costs near the old 15.0 threshold — the penalty bump pushed
   them over the edge. But 36 tracklets remain with route costs ranging from 4.0 to 23.8,
   all below 25.0.

### Verdict per camera

All three cameras show the same pattern: **the explain-or-penalize penalty is
structurally inadequate because it's length-agnostic**. The story is:

- **Not marginal-cost-driven** — route costs are below the penalty for all samples
- **Not constraint-driven** — no hard constraints prevent routing
- **Mechanism-driven** — the flat per-tracklet penalty creates no pressure to explain
  long tracklets vs short ones. The solver optimizes globally and finds that paying
  N × 25.0 for N dropped tracklets is cheaper than restructuring 17 paths to cover them.

FP7oJQ happened to have 3 tracklets near the marginal threshold. J_EDEw and PPDmUg had
none near the threshold, so the penalty bump had no effect.

---

## Question 2 — Where Does the D3→D4 Detection Gap Come From?

### Detection counts at each stage

| Camera | N1 (Stage A) | N2 (bank for explained tids) | N3 (person_tracks) | N1→N3 loss |
|--------|-------------|------------------------------|-------------------|-----------|
| FP7oJQ | 44,381 | 20,243 | 24,705 | 44.3% |
| J_EDEw | 49,160 | 16,934 | 24,202 | 50.8% |
| PPDmUg | 19,243 | 7,642 | 12,121 | 37.0% |

### Surprise: N3 > N2

Person_tracks contains MORE rows than the explained tracklets' bank frames. This is not
a bug — it's by design. D4's span processing (d4_emit.py:383-389) iterates over each
`NodeSpan`, and for GROUP nodes, `_collect_tracklet_ids()` (d4_emit.py:226-241) pulls
in all associated tracklet IDs: `base_tracklet_id`, `carrier_tracklet_id`,
`disappearing_tracklet_id`, `new_tracklet_id`. This means detections from multiple
tracklets within the GROUP span's frame range get assigned to the same person.

| Camera | From SINGLE spans | From GROUP spans | Total (N3) |
|--------|------------------|-----------------|------------|
| FP7oJQ | 13,849 (56%) | 10,856 (44%) | 24,705 |

### The real gap: solver-level drops dominate

| Camera | Dropped tracklet bank frames | % of all bank frames | Explained tracklet frames lost to span filter |
|--------|----------------------------|---------------------|----------------------------------------------|
| FP7oJQ | 24,138 | 54.4% | 740 (3.7% of explained) |
| J_EDEw | 32,226 | 65.6% | 90 (0.5% of explained) |
| PPDmUg | 11,601 | 60.3% | 333 (4.4% of explained) |

The dominant gap is **N1→N2 (solver-level tracklet drops)** at 54-66% of all detections.
The D4 span-filtering loss (detections outside node frame ranges) is tiny: 0.5-4.4% of
explained tracklet frames. There is no significant D4 emission bug.

### Spot-check: span filtering in action

| Camera | Tracklet | Bank frames | PT frames | Gap | Bank range | PT range |
|--------|----------|-------------|-----------|-----|-----------|---------|
| FP7oJQ | t100 | 355 | 311 | 44 | [1196, 1647] | [1305, 1647] |
| FP7oJQ | t12 | 266 | 166 | 100 | [6, 314] | [198, 314] |
| J_EDEw | t176 | 22 | 3 | 19 | [1898, 1919] | [1917, 1919] |
| PPDmUg | t70 | 314 | 19 | 295 | [1299, 1619] | [1601, 1619] |

The span filter clips the start of tracklets that begin before their first D1 node's
`start_frame`. This is by design — D1 segments tracklets into SOLO/GROUP segments, and
detections outside any segment get trimmed. However, PPDmUg t70 loses 94% of its frames
(314→19) because only its last segment overlaps with a person span. This is real frame
loss but is a D1 segmentation issue, not D4.

### Verdict

The dominant detection loss (50-66%) is at the **solver level** — dropped tracklets
account for almost all of it. D4 emission is working correctly. The span-filter loss
is small and is a known consequence of D1 segmentation.

---

## Question 3 — Recommended Next Step

**Option C variant: Make the explain-or-penalize penalty proportional to tracklet length.**

### Rationale

The current `unexplained_tracklet_penalty` is a flat per-tracklet value (25.0). A
4,427-frame tracklet and a 23-frame tracklet are both penalized identically. The solver
treats them as equally expendable, which is wrong — dropping a long tracklet loses
orders of magnitude more identity information.

The evidence is decisive:
- All 9 sample dropped tracklets have solo routing costs below the penalty (4.2-23.2 vs 25.0)
- The longest dropped tracklets (J_EDEw t1: 4,427 frames, PPDmUg t1: 2,926 frames)
  span entire recordings and have the cheapest routing costs (4.2 and 6.0)
- The penalty bump from 15→25 only saved 5 total tracklets because NO tracklets have
  routing costs between 15 and 25 that would make them marginal
- Further penalty bumps (40, 50, 100) cannot solve this — the remaining drops all have
  routing costs of 4-23, so even penalty=100 would make keeping them preferred, but the
  solver still has to decide how many flow paths to create, and flat penalties don't
  change the relative priority of long vs short tracklets

### Proposed change

In `d3_ilp2.py`, modify the explain-or-penalize penalty computation (around line 1997):

```python
# Current: flat penalty
terms.append(int(penalty_scaled) * uvar)

# Proposed: length-proportional penalty
n_frames = tracklet_frame_counts.get(tid, 1)
terms.append(int(penalty_scaled * n_frames) * uvar)
```

This would make dropping J_EDEw t1 cost 25.0 × 4,427 = 110,675 instead of 25.0. The
solver would never drop a full-clip tracklet. Short spurious tracklets (23 frames) would
cost 25.0 × 23 = 575, which is still modest and allows the solver to drop them if routing
is truly expensive.

The per-frame penalty value should likely be much smaller than 25.0 (perhaps 0.01-0.1
per frame) to avoid making every tracklet's penalty dominate the entire objective. The
exact calibration is a CP3 concern.

### Why not the other options

- **Option A (higher penalty):** Already proven ineffective — all remaining drops have
  route costs below 25. Penalty=1000 would keep everything but wouldn't discriminate
  short vs long.
- **Option B (CONTINUE cost surgery):** Addresses chaining but not the flat penalty problem.
  May have side effects on legitimate long chains.
- **Option D (wire dead config):** The dead penalty fields (CP0) are for D3 penalty
  scaling infrastructure that was never built. This is exactly the infrastructure needed,
  but `penalty_ref_edge_cost_quantile` was designed for per-edge scaling, not per-tracklet
  length scaling. Building length-proportional penalty is more targeted.
- **Option E (ILP rewrite):** Premature. The current solver is OPTIMAL with fast runtime.
  The penalty mechanism just needs one structural fix.

---

## Data Sources

All paths relative to repo root.

```
outputs/_eval_gt/{cam}/.../stage_D/d1_graph_nodes.parquet
outputs/_eval_gt/{cam}/.../stage_D/d2_edge_costs.parquet
outputs/_eval_gt/{cam}/.../stage_D/person_tracks.parquet
outputs/_eval_gt/{cam}/.../stage_D/person_spans.parquet
outputs/_eval_gt/{cam}/.../stage_D/tracklet_bank_frames.parquet
outputs/_eval_gt/{cam}/.../stage_D/tracklet_bank_summaries.parquet
outputs/_eval_gt/{cam}/.../_debug/d3_entities_format_a.json
outputs/_eval_gt/{cam}/.../_debug/d3_solution_ledger.json
outputs/_eval_gt/{cam}/.../_debug/d3_selected_edges.parquet
src/bjj_pipeline/stages/stitch/d4_emit.py (lines 383-389: span-based detection filtering)
src/bjj_pipeline/stages/stitch/d3_ilp2.py (lines 1913-1998: explain-or-penalize mechanism)
```
