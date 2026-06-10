# CP-GT2ACTUALS-3: Dense Join Validation Findings

**Date:** 2026-06-10

## 1. Vid2 58% no_id — JOIN ARTIFACT (FIXED)

**Root cause:** `_resolve_tracklet_id` uses bank frame ranges (D0.5 split
boundaries) to find which split product covers a frame. But `person_tracks.parquet`
is written by D4 after the solver, which re-stitches products into chains. The
bank product covering frame X may appear under a DIFFERENT product ID in
person_tracks.

**Fix:** Family-aware person_tracks lookup (`_lookup_person_ids_family`). Tries:
1. resolved tracklet (bank-predicted split product)
2. raw tracklet (pre-split)
3. any split-family member

**Results after fix:**

| Clip | correct | wrong_id | no_id | miss |
|------|---------|----------|-------|------|
| vid1 (pre-fix) | 30.6% | 24.0% | 31.0% | 14.4% |
| vid1 (post-fix) | 40.3% | 42.1% | 3.2% | 14.4% |
| vid2 (pre-fix) | 11.7% | 16.9% | 58.5% | 12.9% |
| vid2 (post-fix) | 30.6% | 49.6% | 6.9% | 12.9% |

**Vid2 no_id dropped 58.5% -> 6.9%** (29,686 rows recovered = 88.2% of original).
Vid1 no_id dropped 31.0% -> 3.2%.

**Residual 6.9% vid2 no_id breakdown:**
- 3,604 (91%): `d3_status = 'explained'` — solver accepted tracklet but D4
  doesn't emit person_tracks at this frame (GROUP frame-range gaps, D4 emit
  quirks). Real pipeline behavior, not join artifact.
- 343 (8.6%): `d3_status = None` — tracklet genuinely unknown to Stage D.
- 22 (0.6%): `d3_status = 'dropped'` — solver explicitly dropped.

**Vid1 vs vid2 asymmetry explained:** After fix, the remaining asymmetry
(vid1 3.2% vs vid2 6.9% no_id) is from vid2's longer clip (4,491 vs 3,001
frames) and more aggressive D0.5 splitting (354 vs 308 events), creating more
GROUP frame-range gaps.

**Verdict: FIXED. Join artifact eliminated. Residual is real pipeline behavior.**

## 2. node_gt_set Inversion — VALIDATED

- **GT IDs not tracklet IDs:** All entries are integers (GT track IDs). Confirmed
  on 5 random samples.
- **Pair-box containment:** 505 detection-sharing pairs checked. 0 mismatches.
  Every pair of GT people sharing a detection appears in each other's node_gt_set.
- **Regression assertion:** `node_gt_set_size >= 2` == `pair_box` count (1010 on
  FP7oJQ). Holds exactly.
- **Split-product resolution:** Both sides of the join resolved through D0.5
  lineage (greedy match tracklets + D1 member tracklets).

## 3. canonical_person_id on Fragmented Tracks — CAVEAT CONFIRMED

**13 of 14 GT tracks on J_EDEw vid1 are fragmented** (canonical person_id
represents < 50% of votes). Examples:

| GT track | canonical | dominance | correct rate |
|----------|-----------|-----------|-------------|
| gt=24 | p0009 | 15% | 16% |
| gt=25 | p0021 | 17% | 18% |
| gt=16 | p0005 | 18% | 28% |
| gt=14 | p0003 | 19% | 26% |

**Impact:** correct/wrong_id verdicts on these tracks are LOW-CONFIDENCE.
A GT person with 15% canonical dominance means the "correct" person_id is
only seen 15% of the time — the canonical is barely better than random among
the fragmented person_ids. The correct rate (16%) is close to the dominance
(15%), confirming the canonical is mapping to the right person_id but that
person_id simply doesn't appear often.

**This is expected behavior for heavily-fragmented identity:** the pipeline
assigns 10-30 person_ids to each GT person, and the most-frequent is barely
dominant. The artifact faithfully reports this.

## 4. group_ambiguous Flag — VALIDATED

All three invariants hold on FP7oJQ:
- 784 frames flagged
- 0 flagged frames with `d1_is_group = False` (correct)
- 0 flagged frames with `node_gt_set_size < 2` (correct)
- 0 GROUP+size>=2 frames NOT flagged (complete)

## 5. Double-Detection Analysis

### 5a. Double-detection rate

| Camera | GT-person-frames with >1 candidate | Rate |
|--------|-----------------------------------|------|
| FP7oJQ | 2,126 / 4,214 | 50.5% |
| J_EDEw vid1 | 24,751 / 42,014 | 58.9% |
| J_EDEw vid2 | 15,188 / 57,544 | 26.4% |

Half or more of GT-person-frames on FP7oJQ and J_EDEw vid1 have >1 candidate
detection at IoU >= 0.1. This is expected for grappling footage.

### 5b. Vid2 no_id: non-winning candidate with person_id

Of 3,969 vid2 no_id frames, **444 (11.2%)** have a non-winning candidate
detection carrying a person_id. This is a THIRD hypothesis: the greedy
matcher bound this GT person to a detection whose tracklet has no person_id,
while a nearby detection (non-winning) DOES have a person_id.

**This is a real but minor channel** — 11.2% of no_id, or 0.8% of total frames.
Not the dominant no_id mechanism.

### 5c. Miss frames with candidates

Of 7,405 vid2 miss frames, **2,643 (35.7%)** have at least one candidate
detection at IoU >= 0.1. These are near-misses where a detection exists but
its IoU with the GT box is between 0.1 and 0.3 (below the match threshold).

### 5d. Unmatched candidate person_ids (dilution source)

On vid1, **23,400/42,014 (55.7%)** of ALL GT-person-frames have a non-winning
candidate detection carrying a person_id. This is a fragmentation signal: more
than half the time, a GT person's neighborhood has a second detection with a
different person_id — a double-coverage that the one-to-one join was blind to.
