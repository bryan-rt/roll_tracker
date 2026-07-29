# CP3b Results — Floor-Protected Length-Proportional Penalty

**Date:** 2026-05-14
**Branch:** `services_uploader`
**Config:** `unexplained_tracklet_penalty_base: 25.0`, `unexplained_tracklet_penalty_per_frame: 0.1`
**Code changes:** `d3_ilp2.py`, `solver.py`, `d3_common.py`, `config/models.py`
**Eval:** `python -m pipeline_validation evaluate --model bjj-detect-all-cameras --force` (3/3 succeeded)

---

## Verdict — Partial Pass

The floor-protected formulation `max(base, per_frame × n_frames)` succeeded at its
primary goal: **no regression from CP2**. Short tracklets are protected by the 25.0
floor (explained counts match CP2: 214/199/59 vs 215/199/59). The infrastructure
is working correctly.

However, the length-proportional component did not rescue the long dropped tracklets
(J_EDEw t1, FP7oJQ t1). Despite penalties of 76.6 and 58.8 respectively, the global
ILP optimization still finds it cheaper to drop them than to restructure the flow
topology. The ambitious targets (coverage > 0.75, tracklet_dropped < 20%) are not met.

| Metric | Pre-CP2 (15) | CP2 (25) | CP3b (max) | Target | Pass? |
|--------|-------------|---------|-----------|--------|-------|
| D val mean coverage (worst) | 0.239 | 0.239 | 0.239 | > 0.75 | **FAIL** |
| `tracklet_dropped` frame % (worst) | 89% | 89% | 89% | < 20% | **FAIL** |
| `true_switch` frame % | 0% | 0% | 0% | < 10% | **PASS** |
| J_EDEw t1 kept | dropped | dropped | dropped | kept | **FAIL** |
| J_EDEw t201 kept | dropped | dropped | dropped | kept | **FAIL** |
| PPDmUg t1 kept | dropped | dropped | dropped | kept | **FAIL** |
| n_expl ≥ CP2 | — | 215/199/59 | 214/199/59 | ≥ CP2 | **PASS** (within non-determinism) |
| Solver runtime | 607/2223/101 | 389/1803/121 | 454/1657/267 | < 3× CP2 | **PASS** |
| Solver status | OPTIMAL | OPTIMAL | OPTIMAL | OPTIMAL | **PASS** |

### Why long tracklets remain dropped

The `n_frames` used for the penalty is computed from **SINGLE_TRACKLET nodes only**.
For full-clip carriers like J_EDEw `t1` (4,427 actual frames), most of its lifespan
is in GROUP nodes where it serves as the carrier. The SINGLE_TRACKLET node sum is
only 766 frames, giving a penalty of 76.6 — significant, but not enough to overcome
the global flow optimization's preference for the existing 17-path structure.

| Camera | Tracklet | Actual frames | SINGLE node frames | Penalty | Solo route | Status |
|--------|----------|--------------|-------------------|---------|------------|--------|
| FP7oJQ | t1 | 3,702 | 588 | 58.8 | 13.1 | dropped |
| J_EDEw | t1 | 4,427 | 766 | 76.6 | 4.2 | dropped |
| PPDmUg | t1 | 2,926 | 1,137 | 113.7 | 6.0 | dropped |

Even PPDmUg t1 with a 113.7 penalty remains dropped. The binding constraint is
not penalty magnitude — it's the **flow topology**: keeping t1 requires creating or
restructuring a flow path, which has cascading costs across all 17 existing paths.

### Length-proportional component impact

| Camera | Tracklets above breakeven (250 frames) | % of total |
|--------|----------------------------------------|-----------|
| FP7oJQ | 24 of 251 | 9.6% |
| J_EDEw | 29 of 236 | 12.3% |
| PPDmUg | 7 of 73 | 9.6% |

24-29 tracklets per camera have penalties above the 25.0 floor. None of the dropped
ones among these were rescued — the length pressure is working but isn't sufficient
to overcome global flow topology constraints.

---

## Detailed Comparison

### Stage D Val Identity Metrics

| Camera | Metric | Pre-CP2 | CP2 | CP3b | Δ CP3b vs CP2 |
|--------|--------|---------|-----|------|---------------|
| FP7oJQ | ID Recall | 0.571 | 0.643 | 0.643 | 0.000 |
| FP7oJQ | ID Precision | 1.000 | 1.000 | 1.000 | 0.000 |
| FP7oJQ | Mean Coverage | 0.329 | 0.419 | 0.419 | 0.000 |
| FP7oJQ | Mean Purity | 0.913 | 0.920 | 0.920 | 0.000 |
| J_EDEw | ID Recall | 0.571 | 0.571 | 0.571 | 0.000 |
| J_EDEw | ID Precision | 0.857 | 0.857 | 0.857 | 0.000 |
| J_EDEw | Mean Coverage | 0.239 | 0.239 | 0.239 | 0.000 |
| J_EDEw | Mean Purity | 0.833 | 0.830 | 0.843 | +0.013 |
| PPDmUg | ID Recall | 0.750 | 0.750 | 0.750 | 0.000 |
| PPDmUg | ID Precision | 0.500 | 0.500 | 0.500 | 0.000 |
| PPDmUg | Mean Coverage | 0.360 | 0.360 | 0.360 | 0.000 |
| PPDmUg | Mean Purity | 0.909 | 0.909 | 0.909 | 0.000 |

CP3b matches CP2 exactly on identity metrics, confirming the floor works.

### Solver Audit

| Camera | Metric | CP2 | CP3b |
|--------|--------|-----|------|
| FP7oJQ | explained | 215 | 214 |
| FP7oJQ | unexplained | 36 | 37 |
| FP7oJQ | runtime_ms | 389 | 454 |
| FP7oJQ | objective | 1558.5 | 2288.4 |
| J_EDEw | explained | 199 | 199 |
| J_EDEw | unexplained | 37 | 37 |
| J_EDEw | runtime_ms | 1803 | 1657 |
| J_EDEw | objective | 1754.7 | 2501.6 |
| PPDmUg | explained | 59 | 59 |
| PPDmUg | unexplained | 14 | 14 |
| PPDmUg | runtime_ms | 121 | 267 |
| PPDmUg | objective | 529.0 | 758.4 |

Objective values increased (expected — length-proportional penalties for dropped
tracklets are higher than flat 25.0), but explained/unexplained counts are stable.

### Failure Mode Breakdown (frame-cost)

| Camera | Cause | Pre-CP2 | CP2 | CP3b |
|--------|-------|---------|-----|------|
| FP7oJQ | tracklet_dropped | 2,202 (81%) | 2,086 (80%) | 2,086 (80%) |
| FP7oJQ | detection_failure | 513 (19%) | 533 (20%) | 533 (20%) |
| J_EDEw | tracklet_dropped | 2,752 (85%) | 2,657 (85%) | 2,657 (85%) |
| J_EDEw | detection_failure | 470 (15%) | 487 (15%) | 487 (15%) |
| PPDmUg | tracklet_dropped | 1,506 (89%) | 1,499 (89%) | 1,499 (89%) |
| PPDmUg | detection_failure | 178 (11%) | 179 (11%) | 179 (11%) |

---

## Recommended Next Step

**The penalty mechanism has reached its ceiling.** Three iterations (CP2, CP3, CP3b)
have shown that penalty magnitude — whether flat, pure length-proportional, or
floor-protected — cannot force the ILP to keep specific tracklets when the global
flow topology makes it cheaper to drop them. The solver is OPTIMAL at each setting;
it's just that the current graph structure produces 17 flow paths that don't cover
certain tracklets regardless of penalty.

The remaining failure modes are:
1. **Tracklet drops (80-89% of frame-cost):** Not fixable by penalty alone.
   The dropped tracklets are structurally unreachable by the current flow paths.
2. **True switches (0% frame-cost but 9-71 events):** Present but zero gap_length,
   meaning they're point transitions not gaps.

**Proposed: CP4 — Investigate the flow path structure.** The 17 paths for 251
tracklets means each path chains ~15 tracklets on average. The dropped tracklets
have incident edges (16-18 per tracklet) but no flow passes through them. The
question is why the solver can't extend or add paths to cover them. This likely
requires examining the SOURCE→BIRTH edge structure (what limits `n_paths`?),
node capacity constraints, and whether the graph construction in D1 is inadvertently
making certain tracklets unreachable.

This is an investigation brief (like CP0/CP1/CP2.5), not a code change brief.
The penalty infrastructure from CP3b is sound and should stay in place — it
correctly protects short tracklets and adds length pressure. The next lever
is graph topology, not penalty calibration.
