# CP-PURITY-1: Decomposition Report

## Headline Discriminators

### Angle 5 — Through-line integrity
- Tagged athlete GT frames: 751
- Frames with p0022 present: 8/611 (1.3%)
- Teleport events: 91
- Identity segments: 92

### Angle 6 — Intra-match vs cross-match (headline: engage x sustained)
- Intra-match (expected dilution): 57 frames (12.6%)
- Cross-match (bug/over-reach): 397 frames (87.4%)
- Self frames: 8
- GT match windows found: 5
- Sweep stability: STABLE (intra% range: [0.0, 12.6])

**Diagnosis: (b) Mostly cross-match over-reach**

---

## Angle 1 — Matched metric re-baseline

| Clip | Tagged correct_id | Aggregate correct_id | CP-TAG-3 baseline |
|------|------------------|---------------------|-------------------|
| vid1 (200015) | 25.9% | 41.7% | 25.6% |
| vid2 (200246) | 14.9% | 32.1% | 22.2% |

Reference aggregate (signal-trace): ~58.7%

## Angle 2 — Tracklet purity distribution

**J_EDEw-20260318-200015:** 125 tracklets, mean=0.883, median=1.000, 30 impure (<0.8)

**J_EDEw-20260318-200246:** 114 tracklets, mean=0.919, median=1.000, 20 impure (<0.8)

## Angle 3 -- D0.5 split impact

**J_EDEw-20260318-200015:** 62 splits, helped=16, hurt=4, neutral=14, mean delta=0.089

**J_EDEw-20260318-200246:** 50 splits, helped=17, hurt=4, neutral=26, mean delta=0.053

## Angle 4 -- Entity purity

Entities measured: 43, mean purity=0.531, median=0.393

**p0022 (tagged):** purity=0.331, majority GT=7, GT-matched frames=462

## Angle 7 -- Match window recovery

GT match windows (engage x sustained): 5
Stage E sessions on p0022: 15
GT recall: 80.0% (4/5)
Order correct: False

## Angle 8 -- Unfixable floor

**J_EDEw-20260318-200015:** tight=35.9%, pair_box=41.2%, miss=22.9% (floor=64.1%)

**J_EDEw-20260318-200246:** tight=70.4%, pair_box=13.8%, miss=15.8% (floor=29.6%)

## Angle 9 -- Approximate stage attribution

Total non-self frames on p0022: 454

- group_over_attribution: 0.0%
- tracklet_impurity: 0.0%
- ilp_stitch: 100.0%
- unattributed: 0.0%

**Caveat:** APPROXIMATE. Overlapping categories possible. Use angles 5+6 for decisions.

## Angle 6 -- Full sweep grid

| Proximity | Duration | GT Windows | Intra | Cross | Intra% | Cross% |
|-----------|----------|-----------|-------|-------|--------|--------|
| close (1.0m) | brief (8f) | 7 | 22 | 432 | 4.8% | 95.2% |
| close (1.0m) | instant (3f) | 9 | 22 | 432 | 4.8% | 95.2% |
| close (1.0m) | long (30f) | 4 | 22 | 432 | 4.8% | 95.2% |
| close (1.0m) | sustained (15f) | 7 | 22 | 432 | 4.8% | 95.2% |
| engage (1.5m) | brief (8f) | 9 | 57 | 397 | 12.6% | 87.4% |
| engage (1.5m) | instant (3f) | 11 | 57 | 397 | 12.6% | 87.4% |
| engage (1.5m) | long (30f) | 5 | 57 | 397 | 12.6% | 87.4% |
| engage (1.5m) | sustained (15f) | 5 | 57 | 397 | 12.6% | 87.4% |
| loose (2.0m) | brief (8f) | 8 | 57 | 397 | 12.6% | 87.4% |
| loose (2.0m) | instant (3f) | 13 | 57 | 397 | 12.6% | 87.4% |
| loose (2.0m) | long (30f) | 5 | 57 | 397 | 12.6% | 87.4% |
| loose (2.0m) | sustained (15f) | 6 | 57 | 397 | 12.6% | 87.4% |
| tight (0.5m) | brief (8f) | 4 | 7 | 447 | 1.5% | 98.5% |
| tight (0.5m) | instant (3f) | 8 | 7 | 447 | 1.5% | 98.5% |
| tight (0.5m) | long (30f) | 1 | 0 | 454 | 0.0% | 100.0% |
| tight (0.5m) | sustained (15f) | 3 | 0 | 454 | 0.0% | 100.0% |

## Metric Definitions (locked)

- **Tracklet purity** = fraction of GT-matched frames whose GT track is the tracklet's plurality GT track (greedy IoU>=0.3)
- **Entity purity** = same, per emitted person_id over its session person_tracks frames
- **Through-line dominant id** = majority-vote person_id per GT track from session person_tracks
- **Coverage** = of tagged athlete's GT frames, fraction with detection carrying dominant_id
- **Anchor-correctness** = of dominant_id's frames, fraction matching tagged GT track (= entity purity for p0022)
- **GT match window** = sustained interval where two GT tracks' world-projected foot-points are within proximity threshold for >= duration consecutive annotated frames
- **Intra-match frame** = non-self frame on entity whose GT person IS in GT match window with tagged athlete
- **Cross-match frame** = non-self frame whose GT person is NOT in GT match window at that frame

## Options for Web Session

1. **If diagnosis is (a) mostly intra-match:** The entity is doing its job; the "impurity" is opponent frames during real matches. Fix is in metric design (exclude intra-match from purity) and/or Stage E window extraction (crop to athlete's own tracklet within the match window).

2. **If diagnosis is (b) mostly cross-match:** The entity wanders into matches it shouldn't be in. Fix is in ILP pairing/emission (HSV-assisted GROUP node ownership, hard ping connectivity, is_isolated gate).

3. **If mixed:** Prioritize the bigger bucket first. Cross-match requires architectural fix; intra-match may be acceptable with metric adjustment.
