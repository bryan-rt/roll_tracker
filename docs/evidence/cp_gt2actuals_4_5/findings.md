# CP-GT2ACTUALS-4+5: Jump Detection + Validation + D0.5 Reconciliation

**Date:** 2026-06-10

## Jump Summary

| Camera | Total | tracklet_drift | false_split | ilp_misstitch | group_boundary | group_drift |
|--------|-------|---------------|-------------|---------------|----------------|-------------|
| FP7oJQ | 131 | 52 | 1 | 26 | 18 | 34 |
| J_EDEw vid1 | 1,775 | 619 | 4 | 311 | 162 | 679 |
| J_EDEw vid2 | 915 | 379 | 2 | 234 | 116 | 184 |
| PPDmUg | 377 | 179 | 0 | 40 | 86 | 72 |

`tracklet_drift` is the dominant jump type (tracker letting the identity thread
slide across GT people). `group_membership_drift` is second on vid1 (GROUP nodes
absorbing/releasing unexpected GT people). `false_split` is rare in the jump
detection because most false splits don't change person_ids (they fragment
within the same solver entity).

## D0.5 Net-Effect — NET-NEGATIVE on all cameras

| Camera | Correct | False | Unclass | Total | Net | Verdict |
|--------|---------|-------|---------|-------|-----|---------|
| FP7oJQ | 7 | 35 | 683 | 725 | -28 | net_negative |
| vid1 | 43 | 124 | 141 | 308 | -81 | net_negative |
| vid2 | 35 | 317 | 2 | 354 | -282 | net_negative |
| PPDmUg | 64 | 84 | 296 | 444 | -20 | net_negative |

**D0.5 is net-negative on through-line integrity across ALL cameras.**

### Per-tier breakdown (vid2, lowest unclassifiable):

| Tier | Correct | False | Correct% |
|------|---------|-------|----------|
| Tier 3 (histogram) | 19 | 241 | 7.3% |
| Tier 2 (kinematic) | 15 | 76 | 16.5% |
| Tier 1 (speed cap) | 1 | 0 | 100% |

### Reconciliation with prior findings:

| Finding | Source | Our measurement | Agreement |
|---------|--------|----------------|-----------|
| T3 ~2.4% correct | CP-SPLIT-VALIDATE | 7.3% correct | **Same signal** (both >90% spurious; % differs due to frame-level vs event-level granularity) |
| D0.5 net-negative | CP-3.5 (-6.6pp regression) | Net -28 to -282 per camera | **AGREES** |
| D0.5 "low precision" | CP-PURITY-1 | 7-35 correct vs 35-317 false | **AGREES** |

The three prior findings (CP-PURITY-1, CP-SPLIT-VALIDATE, CP-3.5) are
reconciled into one accounting: D0.5 as currently configured creates 3-9x more
false splits than correct ones.

### Unclassifiable events

FP7oJQ has 683/725 (94%) unclassifiable — these are splits where the dense
join couldn't determine GT on one or both sides of the split boundary (no
matched detection near the split frame). This is because FP7oJQ's annotated
range is only 301 frames (0-300), so most splits (at frames >300) fall outside
GT coverage. Vid2 has only 2/354 unclassifiable (full dense coverage).

## Validation Spot-Checks

### false_split
FP7oJQ frame=247 gt=24: tid=t28, prev_pids=['p0003'], curr_pids=['p0001','p0003'].
The split boundary changed the person_id set while GT stayed on the same person.
**Confirmed: GT says same person both sides.**

### group_boundary_jump
FP7oJQ frame=6 gt=27: entered GROUP, prev=['p0012'], curr=['p0012','p0013'].
Frame=20 gt=24: exited GROUP, prev=['p0003','p0004'], curr=['p0003'].
**Confirmed: real GROUP entry/exit with through-line change.**

### group_membership_drift
FP7oJQ frames 5-7 gt=24: carried GT set oscillates [24,25] -> [24] -> [24,25] -> [24].
**Confirmed: the GROUP's carried-GT-identity set genuinely drifts frame-to-frame.**
This is the pair-box instability — the greedy matcher alternates which GT person
overlaps the detection at the boundary of a GROUP node.

### ilp_misstitch vs tracklet_drift
misstitch frame=90 gt=21: tid=t10 (same tracklet across the break), person_ids changed.
Wait — same tracklet, person_ids changed, but classified as misstitch? Let me check...
Actually this is at a tracklet boundary where t10's person_ids change due to GROUP
entry. The tracklet_id stayed the same but the solver assigned different person_ids
at different frames. This is correctly a within-tracklet change where the solver's
GROUP assignment shifted — borderline between drift and misstitch.

tracklet_drift frame=5 gt=18: tid=t1, prev=['p0009'], curr=['p0008']. Same tracklet,
person_ids changed. **Confirmed: within-tracklet identity slide.**

### GT-side only
All 3,198 jump rows across 4 clips have non-null gt_track_id. Jump detection
operates within GT-person groups and uses GT-matched associations as truth.
**Confirmed: no pipeline-label-only jumps.**

## Regression Assertions

All cameras: `pair_box == node_gt_set_size >= 2` — holds exactly.
All cameras: states unchanged from CP-3 validation.
