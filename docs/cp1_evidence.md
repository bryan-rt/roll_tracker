# CP1 — Quantitative Evidence: Reconciling Two Views of Tracklet Drop

**Date:** 2026-05-13
**Branch:** `services_uploader`
**Scope:** Read-only data extraction from existing `_eval_gt/` and `_eval/` artifacts.
**Method:** Python extraction from solution ledgers, bank summaries, D2 edge cost parquets,
and eval framework id_switches.

---

## Verdict

**Category A: "Few long drops" — confirmed with a precise mechanism.**

The two views agree 100%. Every `tracklet_dropped` event in the eval framework corresponds
to a tracklet the ILP solver explicitly chose to drop. The solver drops them because
**the penalty (15.0) is cheaper than keeping them**: interior BIRTH+DEATH alone costs
20.02, so the optimizer saves 5.02 per dropped interior tracklet.

This is not a tuning problem — it's a **structural cost inversion**. The
`unexplained_tracklet_penalty` (15.0) was set below the `birth_non_entrance_add_cost`
(8.0) + `birth_cost` (2.0) + `death_non_exit_add_cost` (8.0) + `death_cost` (2.0) +
env (0.01×2) cost floor. The optimizer is doing exactly what the costs tell it to do:
dropping interior tracklets is the globally optimal choice.

**CP2 scope:** Proceed as planned — raising `unexplained_tracklet_penalty` above 20.02
is the correct and sufficient fix. No scope expansion needed. The minimum viable value
is ~20.1 (just above BIRTH+DEATH floor); a value of ~25-30 would provide comfortable
margin while still allowing the solver to drop genuinely spurious short tracklets.

---

## Question 1 — Distribution of Unexplained Tracklet Lengths

### Per-camera summary

| Camera | Dropped | Explained | Mean len (dropped) | Mean len (explained) | Ratio | Dropped frame share |
|--------|---------|-----------|-------------------|---------------------|-------|-------------------|
| FP7oJQ | 39 | 212 | 632.1 | 93.1 | **6.8x** | 55.5% |
| J_EDEw | 38 | 198 | 868.3 | 81.6 | **10.6x** | 67.1% |
| PPDmUg | 15 | 58 | 777.1 | 130.8 | **5.9x** | 60.6% |

Dropped tracklets are 6-11x longer on average. ~16% of tracklets by count produce
56-67% of all frames. This is the "few long drops" pattern — a small number of dropped
tracklets dominate frame-cost.

### Median lengths

| Camera | Median (dropped) | Median (explained) |
|--------|-----------------|-------------------|
| FP7oJQ | 327 | 19.5 |
| J_EDEw | 521 | 12.5 |
| PPDmUg | 395 | 3.5 |

### Histogram buckets (count: dropped / explained)

**FP7oJQ:**

| Bucket | Dropped | Dropped frames | Explained | Explained frames |
|--------|---------|---------------|-----------|-----------------|
| 1-10 | 0 | 0 | 79 | 360 |
| 11-30 | 1 | 23 | 44 | 795 |
| 31-100 | 8 | 526 | 43 | 2,506 |
| 101-300 | 10 | 1,827 | 30 | 4,766 |
| 300+ | 20 | 22,277 | 16 | 11,301 |

**J_EDEw:**

| Bucket | Dropped | Dropped frames | Explained | Explained frames |
|--------|---------|---------------|-----------|-----------------|
| 1-10 | 0 | 0 | 91 | 353 |
| 11-30 | 0 | 0 | 34 | 647 |
| 31-100 | 4 | 266 | 43 | 2,628 |
| 101-300 | 6 | 995 | 16 | 2,920 |
| 300+ | 28 | 31,733 | 14 | 9,618 |

**PPDmUg:**

| Bucket | Dropped | Dropped frames | Explained | Explained frames |
|--------|---------|---------------|-----------|-----------------|
| 1-10 | 0 | 0 | 36 | 84 |
| 11-30 | 0 | 0 | 8 | 142 |
| 31-100 | 2 | 126 | 5 | 248 |
| 101-300 | 4 | 497 | 4 | 597 |
| 300+ | 9 | 11,033 | 5 | 6,516 |

Key observation: zero dropped tracklets in the 1-10 bucket across all cameras. The
solver never drops very short tracklets (their BIRTH+DEATH costs are already paid
elsewhere in the flow graph; the explain-or-penalize mechanism adds a per-base-tracklet
penalty that only applies to tracklets with no selected edges at all).

---

## Question 2 — Cross-Reference: Eval Framework vs Solver

### Agreement: 100% across all cameras

| Camera | tracklet_dropped events | Frame-cost | Solver agrees | Solver disagrees | No tracklet found |
|--------|------------------------|------------|--------------|-----------------|------------------|
| FP7oJQ | 18 | 2,202 | 18 (100%) | 0 (0%) | 0 (0%) |
| J_EDEw | 66 | 2,752 | 66 (100%) | 0 (0%) | 0 (0%) |
| PPDmUg | 25 | 1,506 | 25 (100%) | 0 (0%) | 0 (0%) |

Every `tracklet_dropped` event in the eval framework corresponds to a gap where at least
one Stage A tracklet was active, and that tracklet appears in the ILP solver's
`dropped_tracklets` list. There are zero cases of Category B (solver kept but
misassigned) or Category C (tracklet filtered before solver).

### Frame-cost share of tracklet_dropped in total identity errors

| Camera | detection_failure | tracklet_dropped | sloppy_box | true_switch | Total frames |
|--------|------------------|-----------------|------------|------------|-------------|
| FP7oJQ | 513 (18.9%) | **2,202 (81.1%)** | 0 (0%) | 0 (0%) | 2,715 |
| J_EDEw | 470 (14.6%) | **2,752 (85.4%)** | 0 (0%) | 0 (0%) | 3,222 |
| PPDmUg | 178 (10.6%) | **1,506 (89.4%)** | 0 (0%) | 0 (0%) | 1,684 |

Note: `sloppy_box` and `true_switch` have zero frame-cost because `gap_length` is
only non-zero for `gap`-type events; switch events record transitions not gaps.

### Method

For each `tracklet_dropped` event (identified by `gt_track_id` and gap frame range),
the script found all Stage A tracklet IDs active during the gap frames (from
`detections.parquet`), then checked membership in the solution ledger's
`dropped_tracklets` list. "Solver agrees" means at least one tracklet covering the gap
appears in `dropped_tracklets`.

---

## Question 3 — Edge Costs vs Penalty Magnitudes

### The cost inversion

The `unexplained_tracklet_penalty` is 15.0. For an interior tracklet (not near clip
boundary, not entrance/exit-like), the minimum cost to keep it is:

```
BIRTH(interior) + DEATH(interior) = (2.0 + 8.0 + 0.01) + (2.0 + 8.0 + 0.01) = 20.02
```

**20.02 > 15.0. Dropping is always cheaper for interior tracklets.**

The optimizer saves 5.02 per dropped interior tracklet, plus avoids any CONTINUE edge
costs in the chain. This is not a marginal case — it's a 33% cost advantage for
dropping.

### Edge cost distributions (allowed edges only)

**FP7oJQ:**

| Edge type | Count | Median | Mean | p25 | p75 | p90 | p95 |
|-----------|-------|--------|------|-----|-----|-----|-----|
| BIRTH | 251 | 10.010 | 9.468 | 10.010 | 10.010 | 10.010 | 10.010 |
| CONTINUE | 3,281 | 3.010 | 2.742 | 3.010 | 3.260 | 3.260 | 3.260 |
| DEATH | 251 | 10.010 | 9.373 | 10.010 | 10.010 | 10.010 | 10.010 |
| MERGE | 221 | 0.202 | 0.336 | 0.163 | 0.372 | 0.799 | 0.989 |
| SPLIT | 204 | 0.210 | 0.322 | 0.170 | 0.338 | 0.708 | 0.841 |

**J_EDEw:**

| Edge type | Count | Median | Mean | p25 | p75 | p90 | p95 |
|-----------|-------|--------|------|-----|-----|-----|-----|
| BIRTH | 236 | 10.010 | 9.298 | 10.010 | 10.010 | 10.010 | 10.010 |
| CONTINUE | 2,921 | 3.010 | 2.696 | 3.010 | 3.260 | 3.260 | 3.260 |
| DEATH | 236 | 10.010 | 9.434 | 10.010 | 10.010 | 10.010 | 10.010 |
| MERGE | 222 | 0.201 | 0.319 | 0.163 | 0.365 | 0.705 | 0.975 |
| SPLIT | 198 | 0.191 | 0.278 | 0.163 | 0.284 | 0.533 | 0.726 |

**PPDmUg:**

| Edge type | Count | Median | Mean | p25 | p75 | p90 | p95 |
|-----------|-------|--------|------|-----|-----|-----|-----|
| BIRTH | 73 | 10.010 | 8.914 | 10.010 | 10.010 | 10.010 | 10.010 |
| CONTINUE | 488 | 3.010 | 2.205 | 0.263 | 3.260 | 3.260 | 3.260 |
| DEATH | 73 | 10.010 | 8.805 | 10.010 | 10.010 | 10.010 | 10.010 |
| MERGE | 76 | 0.207 | 0.296 | 0.161 | 0.368 | 0.504 | 0.810 |
| SPLIT | 79 | 0.187 | 0.335 | 0.165 | 0.345 | 0.943 | 1.084 |

### BIRTH/DEATH: entrance vs interior breakdown

| Camera | BIRTH entrance-like | BIRTH interior | DEATH exit-like | DEATH interior |
|--------|-------------------|---------------|----------------|---------------|
| FP7oJQ | 17 (cost ~2.01) | 234 (cost ~10.01) | 20 (cost ~2.01) | 231 (cost ~10.01) |
| J_EDEw | 21 (cost ~2.01) | 215 (cost ~10.01) | 17 (cost ~2.01) | 219 (cost ~10.01) |
| PPDmUg | 10 (cost ~2.01) | 63 (cost ~10.01) | 11 (cost ~2.01) | 62 (cost ~10.01) |

The vast majority of tracklets are interior (not near clip boundary, not entrance/exit-like).
For these, BIRTH costs 10.01 and DEATH costs 10.01.

### Cost comparison table

| Scenario | Cost | vs Penalty (15.0) |
|----------|------|------------------|
| Drop interior tracklet | 15.0 | — |
| Keep interior tracklet (BIRTH+DEATH only, no CONTINUE) | 20.02 | +5.02 (33% more) |
| Keep interior tracklet (BIRTH + 1 CONTINUE + DEATH) | ~23.03 | +8.03 (54% more) |
| Keep entrance tracklet (BIRTH + DEATH, both entrance/exit) | 4.02 | -10.98 (keeping wins) |
| Keep mixed tracklet (BIRTH entrance + DEATH interior) | 12.02 | -2.98 (keeping wins) |

The solver correctly retains entrance/exit tracklets (cost 4.02 < 15.0) and entrance-to-interior
tracklets (cost 12.02 < 15.0). It only drops tracklets where both BIRTH and DEATH are interior
(cost 20.02 > 15.0).

### Audit quantiles (total_cost across all allowed edge types)

| Camera | p50 | p90 | p99 |
|--------|-----|-----|-----|
| FP7oJQ | 3.01 | 10.01 | 10.01 |
| J_EDEw | 3.01 | 10.01 | 10.01 |
| PPDmUg | 3.01 | 10.01 | 10.01 |

---

## Question 4 — Diagnosis

### Category A confirmed: "Few long drops" with a precise mechanism

**Evidence ratios:**

1. **Two views agree 100%** (Q2): Every eval `tracklet_dropped` event maps to a solver-dropped
   tracklet. Zero Category B or C cases.

2. **Dropped tracklets are 6-11x longer** (Q1): 15-20% of tracklets by count produce 56-67%
   of all frame coverage. The eval framework's high frame-cost share (81-89%) from
   `tracklet_dropped` is explained by a modest count of very long dropped tracklets.

3. **Cost inversion is the root cause** (Q3): For interior tracklets, BIRTH+DEATH = 20.02
   while the drop penalty = 15.0. The solver is doing exactly what the costs tell it to do.
   This is deterministic — every interior tracklet where the only path is BIRTH→CONTINUE→DEATH
   (no MERGE/SPLIT structure to share costs with other paths) will be dropped.

### Why only long tracklets are dropped

Short tracklets (1-30 frames) are almost never dropped (1 case out of 92 across all cameras)
because:
- They tend to share flow paths with longer tracklets through MERGE/SPLIT/CONTINUE structure
- Their BIRTH/DEATH edges may be entrance/exit-like (cost 2.01 instead of 10.01)
- The solver optimizes globally — a short tracklet embedded in a longer person path doesn't
  face the explain-or-penalize penalty independently

Long interior tracklets are dropped because they're more likely to be isolated: no MERGE/SPLIT
edges connect them to other tracklets, and both their BIRTH and DEATH are interior. The only
way to "explain" them is BIRTH→CONTINUE*→DEATH, which costs at minimum 20.02, exceeding the
15.0 drop penalty.

### CP2 decision

**Proceed as planned. Single lever: raise `unexplained_tracklet_penalty`.**

The minimum viable value is 20.1 (just above the interior BIRTH+DEATH floor of 20.02).
A value of 25-30 would provide margin while still allowing the solver to drop genuinely
spurious tracklets where the routing cost is very high.

Note: this fix will increase the number of explained tracklets, which increases total
objective value (more BIRTH+DEATH+CONTINUE costs paid). The solver will keep tracklets
it currently drops, assigning them person IDs. This should significantly reduce the
`tracklet_dropped` frame-cost in the eval framework — the 81-89% dominant failure mode
should shrink substantially.

---

## Data Sources

All paths relative to repo root.

```
# Solution ledgers (dropped/explained tracklet lists)
outputs/_eval_gt/FP7oJQ/2026-03-18/20/FP7oJQ-20260318-200014/_debug/d3_solution_ledger.json
outputs/_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200015/_debug/d3_solution_ledger.json
outputs/_eval_gt/PPDmUg/2026-03-18/20/PPDmUg-20260318-training/_debug/d3_solution_ledger.json

# Tracklet bank summaries (n_frames per tracklet)
outputs/_eval_gt/{cam}/.../stage_D/tracklet_bank_summaries.parquet

# D2 edge costs (per-edge cost breakdown)
outputs/_eval_gt/{cam}/.../stage_D/d2_edge_costs.parquet

# Stage D audit (d2_costs_summary events)
outputs/_eval_gt/{cam}/.../stage_D/audit.jsonl

# Eval framework id_switches (tracklet_dropped events)
outputs/_eval/stage_d/bjj-detect-all-cameras/{cam}/id_switches.jsonl

# Eval framework gt_track_sequences (per-GT-track per-frame matching)
outputs/_eval/stage_d/bjj-detect-all-cameras/{cam}/gt_track_sequences.jsonl

# Stage A detections (tracklet_id per detection)
outputs/_eval_gt/{cam}/.../stage_A/detections.parquet

# Stage D person_tracks (detection_id → person_id)
outputs/_eval_gt/{cam}/.../stage_D/person_tracks.parquet
```

## Extraction Code

```python
# Q1: Tracklet length distributions
import json, pandas as pd, numpy as np
# For each camera: load ledger dropped_tracklets/explained_tracklets,
# join against tracklet_bank_summaries.parquet on tracklet_id → n_frames,
# compute mean/median/total/histogram buckets.

# Q2: Cross-reference eval vs solver
# For each tracklet_dropped event in id_switches.jsonl:
#   1. Determine gap frame range from frame_before/frame_after
#   2. From detections.parquet, find tracklet_ids active during gap frames
#   3. Check if those tracklet_ids are in ledger's dropped_tracklets list

# Q3: Edge cost analysis
# From d2_edge_costs.parquet (is_allowed==True):
#   Group by edge_type, compute quantiles of total_cost.
#   For BIRTH/DEATH, separate by term_birth_prior <= 2.5 (entrance) vs > 2.5 (interior).
#   Compare interior BIRTH+DEATH cost floor (20.02) to penalty (15.0).
```
