# CP3 Results — Length-Proportional Penalty (per_frame=0.1) — REGRESSION

**Date:** 2026-05-13
**Branch:** `services_uploader`
**Config change:** `unexplained_tracklet_penalty: 25.0` → `unexplained_tracklet_penalty_per_frame: 0.1`
**Code change:** `d3_ilp2.py`, `solver.py`, `d3_common.py`, `config/models.py`
**Eval command:** `python -m pipeline_validation evaluate --model bjj-detect-all-cameras --force`
**Outcome:** **REGRESSION. Rolled back.**

---

## Verdict — Regression

The pure length-proportional penalty at 0.1 per frame **worsened all metrics** compared
to both the flat penalty=25 (CP2) and the flat penalty=15 (pre-CP2) baselines. The
solver dropped significantly more tracklets (128 vs 36 in FP7oJQ), and identity metrics
degraded across all cameras.

| Metric | Pre-CP2 (15.0) | CP2 (25.0) | CP3 (0.1/frame) | CP3 Pass? |
|--------|---------------|-----------|----------------|----------|
| D val mean coverage (worst cam) | 0.239 | 0.239 | **0.220** (J_EDEw) | **FAIL — regression** |
| `tracklet_dropped` frame % (worst cam) | 89% | 89% | **92%** (FP7oJQ) | **FAIL — regression** |
| `true_switch` frame % | 0% | 0% | 0% | PASS |
| J_EDEw t1 kept? | dropped | dropped | **dropped** | **FAIL** |
| J_EDEw t201 kept? | dropped | dropped | **dropped** | **FAIL** |
| Solver runtime | 607/2223/101 | 389/1803/121 | 2438/1819/102 | PASS (< 3x) |
| Solver status | OPTIMAL | OPTIMAL | OPTIMAL | PASS |

### Root cause: calibration error

The per-frame penalty of 0.1 was calibrated against the interior BIRTH+DEATH floor
(~20.02) at a breakeven of 200 frames. But this made **short tracklets drastically
cheaper to drop**:

| Tracklet length | Flat penalty (25.0) | Per-frame penalty (0.1) | Effect |
|----------------|--------------------|-----------------------|--------|
| 23 frames | 25.0 | 2.3 | 10x cheaper to drop |
| 50 frames | 25.0 | 5.0 | 5x cheaper to drop |
| 100 frames | 25.0 | 10.0 | 2.5x cheaper to drop |
| 200 frames | 25.0 | 20.0 | Breakeven |
| 500 frames | 25.0 | 50.0 | 2x more expensive to drop |
| 4,427 frames | 25.0 | 442.7 | 18x more expensive to drop |

The median tracklet has ~20 frames. At 0.1 per frame, that's a 2.0 penalty — far below
any routing cost. The solver rationally drops most short tracklets.

### Solver comparison

| Camera | Pre-CP2 expl/unexpl | CP2 expl/unexpl | CP3 expl/unexpl |
|--------|--------------------|-----------------|--------------------|
| FP7oJQ | 212/39 | 215/36 | **123/128** |
| J_EDEw | 198/38 | 199/37 | **105/131** |
| PPDmUg | 58/15 | 59/14 | **37/36** |

CP3 dropped 128 tracklets in FP7oJQ (vs 36 in CP2) — a 3.6x increase.

### What the right fix looks like

The pure per-frame penalty is wrong because it only helps long tracklets while
actively hurting short ones. The correct formulation needs **both a floor and a
length-proportional component**: `penalty = max(base_floor, per_frame * n_frames)`.

For example, with `base_floor=25.0` (current CP2 value) and `per_frame=0.1`:
- 23-frame tracklet: max(25.0, 2.3) = 25.0 (same as CP2)
- 200-frame tracklet: max(25.0, 20.0) = 25.0 (same as CP2)
- 500-frame tracklet: max(25.0, 50.0) = 50.0 (2x CP2)
- 4,427-frame tracklet: max(25.0, 442.7) = 442.7 (18x CP2)

This preserves CP2's improvements for short tracklets while adding the length pressure
that CP2.5 identified as necessary for long tracklets.

### Rollback

Config and code changes rolled back to CP2 state (penalty=25.0 flat). The code changes
(length-proportional penalty mechanism) are retained in git history for the next
iteration but reverted in the working tree.

---

## Data Sources

```
outputs/_eval_baseline_penalty_15/  (pre-CP2, penalty=15.0)
outputs/_eval_baseline_cp2_penalty_25/  (post-CP2, penalty=25.0)
outputs/_eval/  (post-CP3, per_frame=0.1 — overwritten by rollback rerun)
```
