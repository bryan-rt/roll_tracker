# CP-PURITY-2: Aggregate Reconciliation + Floor Decomposition

## M1: Aggregate Reconciliation

| Basis | correct_id | Notes |
|-------|-----------|-------|
| J_EDEw val-split, clip-level (CP-TAG-4a) | **63.2%** | Apples-to-apples comparison |
| J_EDEw val-split, clip-level (baseline) | **40.5%** | Signal-trace, pre-CP-TAG-4a |
| Delta | **+22.7pp** | |
| Vid1 full-range, clip-level | 40.5% | |
| Vid1 full-range, session-level | 41.7% | |
| Vid2 full-range, clip-level | 30.7% | |
| Vid2 full-range, session-level | 32.1% | |

**Verdict:** IMPROVEMENT: +22.7pp above baseline

**Canonical definition:** Canonical aggregate correct_id: clip-level person_tracks, greedy IoU>=0.3, majority-vote dominant_pid per GT track, val-split frames only. J_EDEw baseline: 40.5% (signal-trace, bjj-detect-all-cameras). Three-camera aggregate (58.7%) is NOT comparable to single-camera numbers.

## M2: Pair-box Floor Split

| Clip | Proximity | Pair-box | Correct-group | Mishandled | Spurious | Neither |
|------|-----------|----------|--------------|------------|----------|--------|
| vid1 | tight (0.5m) | 124 | 28 (22.6%) | 87 (70.2%) | 6 | 3 |
| vid1 | close (1.0m) | 124 | 34 (27.4%) | 90 (72.6%) | 0 | 0 |
| vid1 | engage (1.5m) | 124 | 34 (27.4%) | 90 (72.6%) | 0 | 0 |
| vid2 | tight (0.5m) | 62 | 3 (4.8%) | 30 (48.4%) | 7 | 22 |
| vid2 | close (1.0m) | 62 | 10 (16.1%) | 50 (80.6%) | 0 | 2 |
| vid2 | engage (1.5m) | 62 | 10 (16.1%) | 52 (83.9%) | 0 | 0 |

Mishandled% stability: UNSTABLE (range: [48.4, 83.9])

## M3: Miss Floor Split

| Clip | Miss | Proxy-occluded | Edge-ROI | Detector-fail |
|------|------|---------------|----------|---------------|
| vid1 | 69 | 51 (73.9%) | 0 (0.0%) | 18 (26.1%) |
| vid2 | 71 | 12 (16.9%) | 4 (5.6%) | 55 (77.5%) |

### CVAT Cross-check: cross-check not established

## M4: True Addressable Ceiling

### vid1 (301 GT frames)

| Bucket | Count | % | Owner |
|--------|-------|---|-------|
| 1_addressable_ceiling | 108 | 35.9% | Appearance-stitch (CP21) UPPER BOUND |
| 2_designed_group_ambiguity | 34 | 11.3% | Window delivery |
| 3_group_formation_defect | 90 | 29.9% | D1 graph (structural) |
| 4_pair_box_other | 0 | 0.0% | Mixed/uncategorized |
| 5_miss_accept_occluded | 51 | 16.9% | Accept |
| 6_miss_edge_roi | 0 | 0.0% | Geometry/config |
| 7_miss_detector_fail | 18 | 6.0% | Detection model (CP23) |

### vid2 (450 GT frames)

| Bucket | Count | % | Owner |
|--------|-------|---|-------|
| 1_addressable_ceiling | 317 | 70.4% | Appearance-stitch (CP21) UPPER BOUND |
| 2_designed_group_ambiguity | 10 | 2.2% | Window delivery |
| 3_group_formation_defect | 52 | 11.6% | D1 graph (structural) |
| 4_pair_box_other | 0 | 0.0% | Mixed/uncategorized |
| 5_miss_accept_occluded | 12 | 2.7% | Accept |
| 6_miss_edge_roi | 4 | 0.9% | Geometry/config |
| 7_miss_detector_fail | 55 | 12.2% | Detection model (CP23) |

## Options for Web Session

1. **Appearance-stitch (CP21):** Bucket 1 sets the upper bound. Realized gain depends on how much ILP mis-stitch appearance evidence can prevent.

2. **Detection model (CP23):** Bucket 7 is trainable detector misses — more training data or architecture changes.

3. **Geometry/config:** Bucket 6 — ROI mask application or camera repositioning.

4. **D1 graph:** Bucket 3 — group-formation defects on pair-box frames.

5. **Accept:** Bucket 5 — occluded frames, no fix possible.

