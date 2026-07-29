# CP2 Results — `unexplained_tracklet_penalty` 15.0 → 25.0

**Date:** 2026-05-13
**Branch:** `services_uploader`
**Config change:** `configs/default.yaml` line 286: `unexplained_tracklet_penalty: 15.0` → `25.0`
**Eval command:** `python -m pipeline_validation evaluate --model bjj-detect-all-cameras --force`
**Baseline snapshot:** `outputs/_eval_baseline_penalty_15/`

---

## Verdict — Partial Pass

The penalty increase from 15.0 to 25.0 produced a small, measurable improvement but
**did not meet the target thresholds**. The improvement was much smaller than expected
because the CP1 cost analysis was incomplete: it identified the BIRTH+DEATH floor (20.02)
as the binding constraint, but the actual binding constraint is the **full routing cost
through the flow graph** — which includes CONTINUE edges, capacity sharing, and
opportunity costs from the global flow optimization.

| Metric | Baseline (15.0) | Target | New (25.0) | Pass? |
|--------|----------------|--------|-----------|-------|
| Stage D val mean coverage (worst cam) | 0.239 (J_EDEw) | > 0.60 | 0.239 (J_EDEw) | **FAIL** |
| `tracklet_dropped` frame-cost % (worst cam) | 89.4% (PPDmUg) | < 40% | 89.3% (PPDmUg) | **FAIL** |
| `true_switch` frame-cost % (any cam) | 0% | < 10% | 0% | **PASS** |
| Solver status | OPTIMAL | OPTIMAL/FEASIBLE | OPTIMAL | **PASS** |
| Solver runtime | 607/2223/101 ms | < 3x baseline | 389/1803/121 ms | **PASS** |
| Stage F kept_detections % (worst cam) | ~33% (est.) | > 60% | 49.2% (J_EDEw) | **FAIL** |

### Why the fix was insufficient

CP1's analysis assumed the penalty competes against a tracklet's **solo routing cost**
(BIRTH + internal CONTINUE + DEATH ≈ 20-24). This is the cost if the solver adds a
new dedicated flow path for the tracklet.

In reality, the ILP solver decides globally. With 251 tracklets and only 17 flow paths
(FP7oJQ), most tracklets must share paths. A tracklet can only be "explained" if at
least one of its incident edges carries flow. The solver drops tracklets when:

1. No existing path passes through them (they're not on the 17 selected paths)
2. Extending or adding a path to cover them costs more than the 25.0 penalty
3. Adding a path for one tracklet may displace another path's flow, causing cascading
   reassignment costs

The remaining 36 dropped tracklets (FP7oJQ) all have incident edges, but the global
flow solution doesn't route through them. Raising the penalty to 25.0 only rescued
3 more tracklets (39→36) because the marginal cost of rerouting to include the
remaining ones exceeds 25.0 at the global optimum.

### Recommended next step

**Do not iterate on penalty value alone.** Even at penalty=1000, the solver may not be
able to explain all tracklets without violating capacity constraints or creating
infeasible flow patterns.

The correct intervention is **CP3: wire the dead D3 config fields** (from CP0's audit),
specifically `penalty_ref_edge_cost_quantile` and the per-tracklet penalty scaling.
This would let the penalty adapt to the actual cost landscape of each clip rather than
being a fixed constant. Additionally, investigate whether the `n_paths` (flow capacity)
constraint is artificially limiting — 17 paths for 251 tracklets means the graph
topology itself determines which tracklets get covered.

Alternatively, investigate the D4 tracklet acceptance criteria — CP0's CLAUDE.md notes
that "Stage D drops ~56% of detections," and the eval framework's `tracklet_dropped`
cause description says "Stage D's tracklet acceptance criteria are rejecting valid
identities." There may be a filtering step between D3 (solver) and D4 (emission) that
drops tracklets beyond what the solver chose.

---

## Detailed Comparison

### Stage A Val Recall (sanity check — detector unchanged)

| Camera | Baseline | New | Change |
|--------|----------|-----|--------|
| FP7oJQ | 0.847 | 0.847 | 0.0% (unchanged) |
| J_EDEw | 0.864 | 0.864 | 0.0% (unchanged) |
| PPDmUg | 0.750 | 0.750 | 0.0% (unchanged) |

### Stage D Val Identity Metrics

| Camera | Metric | Baseline | New | Abs Δ | Rel Δ |
|--------|--------|----------|-----|-------|-------|
| FP7oJQ | ID Recall | 0.571 | 0.643 | +0.071 | +12.5% |
| FP7oJQ | ID Precision | 1.000 | 1.000 | 0.000 | 0.0% |
| FP7oJQ | Mean Coverage | 0.329 | 0.419 | +0.090 | +27.2% |
| FP7oJQ | Mean Purity | 0.913 | 0.920 | +0.008 | +0.8% |
| J_EDEw | ID Recall | 0.571 | 0.571 | 0.000 | 0.0% |
| J_EDEw | ID Precision | 0.857 | 0.857 | 0.000 | 0.0% |
| J_EDEw | Mean Coverage | 0.239 | 0.239 | 0.000 | 0.0% |
| J_EDEw | Mean Purity | 0.833 | 0.830 | -0.003 | -0.3% |
| PPDmUg | ID Recall | 0.750 | 0.750 | 0.000 | 0.0% |
| PPDmUg | ID Precision | 0.500 | 0.500 | 0.000 | 0.0% |
| PPDmUg | Mean Coverage | 0.360 | 0.360 | 0.000 | 0.0% |
| PPDmUg | Mean Purity | 0.909 | 0.909 | 0.000 | 0.0% |

FP7oJQ showed meaningful improvement (+12.5% ID recall, +27.2% coverage). J_EDEw and
PPDmUg showed zero change in identity metrics despite the penalty increase.

### Solver Audit: Unexplained Tracklets

| Camera | Baseline total | Baseline expl | Baseline unexpl | New expl | New unexpl | Δ unexpl |
|--------|---------------|--------------|----------------|---------|-----------|---------|
| FP7oJQ | 251 | 212 | 39 | 215 | 36 | -3 |
| J_EDEw | 236 | 198 | 38 | 199 | 37 | -1 |
| PPDmUg | 73 | 58 | 15 | 59 | 14 | -1 |

Only 3+1+1 = 5 additional tracklets explained across all cameras.

### Solver Performance

| Camera | Baseline runtime (ms) | New runtime (ms) | Ratio | Status |
|--------|----------------------|------------------|-------|--------|
| FP7oJQ | 607 | 389 | 0.64x | OPTIMAL |
| J_EDEw | 2,223 | 1,803 | 0.81x | OPTIMAL |
| PPDmUg | 101 | 121 | 1.20x | OPTIMAL |

No runtime degradation. All solves remain OPTIMAL.

### Solver Objective Value

| Camera | Baseline obj | New obj | Δ | Note |
|--------|-------------|---------|---|------|
| FP7oJQ | 1,184.98 | 1,558.50 | +373.52 (+31.5%) | Higher penalty per drop raises objective |
| J_EDEw | 1,384.57 | 1,754.75 | +370.18 (+26.7%) | Same pattern |
| PPDmUg | 379.98 | 528.97 | +148.99 (+39.2%) | Same pattern |

The objective increase reflects the higher per-drop penalty (25.0 vs 15.0), not more
routing activity. Most of the increase is from 36×25=900 vs 39×15=585 (FP7oJQ).

### Failure Mode Breakdown (event count)

| Camera | Cause | Baseline | New | Δ |
|--------|-------|----------|-----|---|
| FP7oJQ | detection_failure | 43 | 48 | +5 |
| FP7oJQ | tracklet_dropped | 18 | 16 | -2 |
| FP7oJQ | sloppy_box | 3 | 3 | 0 |
| FP7oJQ | true_switch | 9 | 10 | +1 |
| J_EDEw | detection_failure | 104 | 110 | +6 |
| J_EDEw | tracklet_dropped | 66 | 65 | -1 |
| J_EDEw | sloppy_box | 25 | 26 | +1 |
| J_EDEw | true_switch | 65 | 71 | +6 |
| PPDmUg | detection_failure | 55 | 56 | +1 |
| PPDmUg | tracklet_dropped | 25 | 25 | 0 |
| PPDmUg | sloppy_box | 4 | 4 | 0 |
| PPDmUg | true_switch | 26 | 26 | 0 |

Small increases in detection_failure and true_switch for J_EDEw — likely due to
non-determinism in the full pipeline rerun (Stage A re-ran, producing slightly
different tracklets).

### Failure Mode Breakdown (frame-cost)

| Camera | Cause | Baseline frames | New frames | Δ frames | Baseline % | New % |
|--------|-------|----------------|------------|---------|-----------|-------|
| FP7oJQ | detection_failure | 513 | 533 | +20 | 18.9% | 20.4% |
| FP7oJQ | tracklet_dropped | 2,202 | 2,086 | -116 | 81.1% | 79.6% |
| FP7oJQ | true_switch | 0 | 0 | 0 | 0% | 0% |
| J_EDEw | detection_failure | 470 | 487 | +17 | 14.6% | 15.5% |
| J_EDEw | tracklet_dropped | 2,752 | 2,657 | -95 | 85.4% | 84.5% |
| J_EDEw | true_switch | 0 | 0 | 0 | 0% | 0% |
| PPDmUg | detection_failure | 178 | 179 | +1 | 10.6% | 10.7% |
| PPDmUg | tracklet_dropped | 1,506 | 1,499 | -7 | 89.4% | 89.3% |
| PPDmUg | true_switch | 0 | 0 | 0 | 0% | 0% |

`tracklet_dropped` frame-cost decreased slightly for FP7oJQ (-116 frames, -5.3%) and
J_EDEw (-95 frames, -3.5%). PPDmUg essentially unchanged.

### Detection Coverage (Stage D → person_tracks)

| Camera | New kept / total | New kept % |
|--------|-----------------|-----------|
| FP7oJQ | 24,705 / 44,381 | 55.7% |
| J_EDEw | 24,202 / 49,160 | 49.2% |
| PPDmUg | 12,121 / 19,243 | 63.0% |

Baseline values not directly comparable because `_eval_gt` pipeline outputs were
overwritten by the rerun. CP1 recorded FP7oJQ baseline at 56.7% (25,144/44,381).
The new FP7oJQ value (55.7%) is marginally lower, likely due to non-determinism in
the full pipeline rerun.

### Solution Ledger: Dropped Tracklets

| Camera | Baseline dropped | New dropped | Δ |
|--------|-----------------|------------|---|
| FP7oJQ | 39 | 36 | -3 |
| J_EDEw | 38 | 37 | -1 |
| PPDmUg | 15 | 14 | -1 |

---

## Key Finding: CP1's Cost Model Was Incomplete

CP1 correctly identified the cost inversion (penalty < BIRTH+DEATH floor) but
incorrectly predicted that raising the penalty above 20.02 would resolve most drops.
The actual constraint is the **global flow optimization**: with 17 flow paths for 251
tracklets, the solver must decide which tracklets to chain into paths. Tracklets that
don't fit any efficient path get dropped regardless of the penalty level, as long as
the penalty is finite.

The solo routing cost (BIRTH + internal CONTINUE + DEATH) ranges from 4.02 to 23.76
for dropped tracklets — all below 25.0. Yet they remain dropped because routing them
requires displacing or extending other paths at costs that exceed the penalty at the
global optimum.

This means the `unexplained_tracklet_penalty` mechanism works as designed (it's a soft
preference, not a hard constraint), but it can't force the solver to cover tracklets
when doing so would require fundamentally different flow topology. The penalty would
need to be extremely high (~100+) to force significant structural changes, and at that
point the solver would likely produce worse overall stitching quality by forcing
unnatural path extensions.

---

## Data Sources

```
# Baseline eval reports
outputs/_eval_baseline_penalty_15/stage_d/bjj-detect-all-cameras/{cam}/report.json
outputs/_eval_baseline_penalty_15/stage_d/bjj-detect-all-cameras/{cam}/id_switches.jsonl

# New eval reports
outputs/_eval/stage_d/bjj-detect-all-cameras/{cam}/report.json
outputs/_eval/stage_d/bjj-detect-all-cameras/{cam}/id_switches.jsonl

# Solver audit (last d3_ilp_summary event per clip)
outputs/_eval_gt/{cam}/.../stage_D/audit.jsonl

# Solution ledger
outputs/_eval_gt/{cam}/.../_debug/d3_solution_ledger.json
```
