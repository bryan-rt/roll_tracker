# CP-GT2ACTUALS-6: Signal-Shape + Stage-Attribution Analysis

**Date:** 2026-06-10
**Authoritative camera:** vid2 (J_EDEw-200246, 99.4% classified)
**Corroboration:** vid1 (J_EDEw-200015, 54% classified)

## 1. Stage-Attribution of Through-Line Damage

### THE HEADLINE: Stage A (tracklet_drift) is the dominant source

| Stage | vid2 | vid2 % | vid1 | vid1 % |
|-------|------|--------|------|--------|
| **Stage A (tracker)** | **379** | **41%** | **619** | **35%** |
| Group handling (D1/solver) | 300 | 33% | 841 | 47% |
| Solver (D3 misstitch) | 234 | 26% | 311 | 18% |
| D0.5 (false_split jump) | 2 | 0% | 4 | 0% |

**Stage A (tracklet_drift) is the #1 source on vid2** at 41%. Group handling
is #2 at 33%. Solver misstitching is #3 at 26%. D0.5 false_split JUMPS are
negligible (0%) — but D0.5's damage is indirect (fragmenting tracklets that
the solver then misstitches, contributing to the solver/group numbers).

**Upstream of solver (Stage A + D0.5): 42% of vid2 jumps.** The solver cannot
fix these — they arrive as already-broken input.

**Important distinction:** D0.5 produces 317 false SPLITS on vid2, but only 2
false_split JUMPS. The gap: most false splits don't change person_ids because
the solver re-stitches the fragments. D0.5's damage is therefore mainly
INDIRECT — it forces the solver to stitch more fragments, increasing the chance
of misstitching.

### Answer to the headline question

**Yes, Stage A (tracklet_drift) is the dominant source of deliverable damage.**
Appearance-in-solver work addresses the solver's 26% (D3 misstitch) and
partially the 33% group handling. It CANNOT address the 41% that is upstream
(tracker sliding across people). The lever has moved a fifth time: **the
dominant fix is detection/tracking quality, not solver/appearance.**

However, appearance-in-solver is still valuable because:
- 26% solver misstitch + 33% group handling = 59% of jumps involve the solver
- Appearance would help the solver stitch the D0.5 fragments correctly (making
  the 317 false splits harmless rather than a fragmentation burden)

## 2. Signal-Shape Distributions

### Velocity (speed_mps_k)

| Population | n | Median | P75 | P95 | Max |
|-----------|---|--------|-----|-----|-----|
| Real swaps (tracklet_drift) | 361 | 0.77 | 3.04 | 19.2 | 119.8 |
| False split window (+-2f) | 6,553 | 0.09 | 0.39 | 3.51 | 45.7 |
| Correct split window | 675 | 0.28 | 0.82 | 5.87 | 52.3 |
| Calm (correct, no jump) | 17,503 | 0.00 | 0.00 | 1.70 | 55.9 |

**Real swaps are 8x faster than false splits at median** (0.77 vs 0.09 m/s).
But the P95 overlap is massive (19.2 vs 3.51) — speed alone cannot discriminate.
False splits are SLOWER than correct splits at median (0.09 vs 0.28), which
contradicts the "false splits are motion-correlated" prior — they're actually
LOW-MOTION artifacts on vid2 (possible V-channel noise on stationary frames).

### is_isolated

| Population | n | Isolated % |
|-----------|---|-----------|
| Real swaps | 379 | 49% |
| False split window | 6,570 | 82% |
| Correct split window | 676 | 80% |
| Calm | 17,511 | 92% |

**False splits have HIGH isolation (82%)**, meaning the color signal IS
available at those frames. This contradicts the assumption that false splits
happen on entangled (non-isolated) frames where color is polluted. The color
signal exists — it's just not discriminative enough at the current threshold.

Real swaps have LOW isolation (49%) — they happen during grappling exchanges
where people are close (non-isolated), so color is unavailable at the very
frames where it would be most useful.

### HSV Histogram Pre/Post Distance at Splits

| Population | n events | Mean Bhatt | Median Bhatt |
|-----------|----------|-----------|-------------|
| False split | 215 | 0.035 | 0.031 |
| Correct split | 35 | 0.040 | 0.031 |

**No separation.** False and correct splits have indistinguishable histogram
distance at the boundary (mean 0.035 vs 0.040). The color shift is NOT
sustained for correct splits vs transient for false splits — both show similar
small distances. This means **HSV alone cannot distinguish real from false
splits**, even with the V channel.

On vid1 (corroboration): false=0.083, correct=0.101 — slightly higher but
still heavily overlapping. Same conclusion.

## 3. Tier-3-Disable Probe

### D0.5 damage breakdown by tier (vid2, authoritative)

| Tier | Correct | False | Net | % of total damage |
|------|---------|-------|-----|-------------------|
| Tier 3 (histogram) | 19 | 241 | -222 | **79%** |
| Tier 2 (kinematic) | 15 | 76 | -61 | 22% |
| Tier 1 (speed cap) | 1 | 0 | +1 | 0% |
| **Total** | **35** | **317** | **-282** | |

**Tier 3 owns 79% of D0.5's net damage** (222 of 282 net-negative events).

### Blast radius of Tier 3 false splits

**66.5% of vid2 GT-person-frames (38,294 / 57,544)** are on tracklets affected
by Tier 3 false splits. These tracklets carry 19,207 wrong_id + 16,598 correct
+ 2,489 no_id frames. Disabling Tier 3 would prevent this fragmentation.

### Counterfactual: without Tier 3

| Metric | Current | Without Tier 3 | Without all D0.5 |
|--------|---------|----------------|------------------|
| Net D0.5 damage (vid2) | -282 | -60 | 0 |
| False splits removed | 0 | 241 | 317 |
| Correct splits lost | 0 | 19 | 35 |

**Disabling Tier 3 removes 79% of D0.5 damage at the cost of 19 correct
splits** (5.4% of all correct splits). This is a strong cost-benefit ratio.

### Pipeline re-run not performed

A definitive correct_id measurement requires a pipeline re-run with Tier 3
disabled. The counterfactual estimate suggests the 33.9% current correct_id
(which includes the -6.6pp D0.5 regression from CP-3.5) would recover a
significant portion toward the 40.5% pre-split baseline, since Tier 3 owns
79% of the damage. **Exact recovery requires the pipeline re-run** (Tier-3
disable would change the solver's input, which changes all downstream person_ids
in ways the artifact can't predict).

## 4. Appearance-in-Solver Assessment

### Do false-split siblings have agreeing color?

| Metric | vid2 | vid1 |
|--------|------|------|
| False splits with boundary frames | 317 | 124 |
| Mean is_isolated at boundary | **85%** | 51% |
| Histogram available at boundary | **100%** | 100% |

**On vid2, the color signal IS available at 85% of false-split boundaries.**
This means appearance-in-solver COULD help: if the solver saw the HSV
histograms, it would see that both sibling fragments have similar color
(Bhattacharyya distance is only 0.035 at false splits) and stitch them
together, making the false split harmless.

**But there's a catch:** the histogram distance is equally low for correct
splits (0.040) — so the solver can't use color alone to distinguish "stitch
these back" from "keep them separate." It would need a SECOND signal
(different GT people should have different color, but at 0.035 vs 0.040
distance, same vs different GT is not separable by color).

### Verdict on appearance-in-solver

Appearance would help the solver re-stitch false-split fragments (color is
available and agrees), but it CANNOT DISCRIMINATE false from correct splits.
The solver would need either:
1. A stronger appearance signal (current HSV doesn't separate same-person
   from different-person at split boundaries)
2. A structural signal (tracklet lifecycle, temporal gap, spatial trajectory)
   in addition to color

## OPTIONS for Web Session

Grounded in the attribution + Tier-3-disable result:

### Option A: Disable Tier 3 (cheap, immediate)
- Removes 79% of D0.5 damage (-222 of -282 net)
- Costs 19 correct splits (5.4%)
- Expected to recover most of the -6.6pp D0.5 regression
- Requires pipeline re-run to confirm exact recovery
- **Risk:** low (Tier 3 is 7.3% correct — overwhelmingly spurious)

### Option B: Improve Stage A tracking (high-impact, expensive)
- Addresses the #1 damage source (41% of jumps)
- Detection pair-separation + tracker improvements
- Upstream of everything — fixes reduce cascading damage
- **Risk:** high effort (detection model retraining, tracker tuning)

### Option C: Appearance in solver (medium-impact, medium effort)
- Addresses 26% solver misstitch + helps absorb D0.5 fragments
- But color doesn't separate same-person from different-person at splits
- Needs a stronger signal or structural complement
- **Risk:** medium (helps but won't solve the discrimination problem alone)

### Option D: D0.5 Tier 3 redesign (medium-impact, medium effort)
- Current Tier 3 (memoryless single-frame Bhattacharyya threshold) is broken
- Signal analysis shows: speed doesn't discriminate, color doesn't discriminate,
  isolation is NOT the problem (82% isolated at false splits)
- Redesign needs temporal/structural approach, not per-frame threshold tuning
- **Risk:** medium (design space unclear — no single signal separates the classes)

### Recommended sequence (no fix chosen — options for discussion):
1. **Option A first** (cheap, recovers most of the regression)
2. **Option B in parallel** (the dominant lever, long-term)
3. **Option C after B shows gains** (solver absorbs remaining damage)
4. **Option D only if A is insufficient** (redesign is hard given no discriminative signal)
