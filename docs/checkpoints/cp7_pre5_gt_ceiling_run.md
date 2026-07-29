# CP7-pre-5: GT-as-Stage-A Ceiling Run (FP7oJQ)

## Setup

- **Scope:** FP7oJQ, frames 0-300 (dense, stride 1), 301 frames
- **GT tracks:** 14 (track_ids 14-27)
- **Total GT boxes:** 4214
- **Tracklet identity:** GT track_id threaded as tracklet_id (pure ceiling arm)
- **Stage C:** Empty (no tags)
- **Solver status:** OPTIMAL
- **Reference point:** bottom-center of bbox, H-only projection (no lens cal)

## Headline: Six-Mode Failure Breakdown

| Mode | N | % |
|------|---|---|
| present | 4214 | 100.0% |
| stage_a_no_detection | 0 | 0.0% |
| stage_a_untracked | 0 | 0.0% |
| d3_dropped | 0 | 0.0% |
| d4_unassigned | 0 | 0.0% |
| present_misattributed | 0 | 0.0% |
| missing_canonical | 0 | 0.0% |
| **Total** | **4214** | **100.0%** |

### Comparison with CP5 Production Baseline (FP7oJQ)

| Mode | GT-Ceiling % | Production % | Delta |
|------|-------------|-------------|-------|
| present | 100.0% | 6.4% | +93.6% |
| stage_a_no_detection | 0.0% | 12.2% | -12.2% |
| stage_a_untracked | 0.0% | 10.7% | -10.7% |
| d3_dropped | 0.0% | 4.6% | -4.6% |
| d4_unassigned | 0.0% | 0.5% | -0.5% |
| present_misattributed | 0.0% | 65.6% | -65.6% |
| missing_canonical | 0.0% | 0.0% | +0.0% |

Re-baseline against a CP5-state full-mode snapshot. When reading the post-CP7
six-mode shift, treat any movement in a metric that was stable across CP0-CP5
as expected-until-explained, then run the conservation check before trusting
the magnitude.

## Identity-Collapse Audit

### Many-to-One (person_id absorbing >= 2 GT tracks)

**None** -- every person_id maps to exactly one GT track.

### One-to-Many (GT track split across >= 2 person_ids)

**None** -- every GT track maps to exactly one person_id.

### Summary

- Many-to-one cases: 0
- One-to-many cases: 0
- **D->F is sound on perfect input: zero identity collapse.**

## Per-GT-Track Detail

| GT Track | Canonical PID | Purity | Frames | Present | Misattrib | D3 Drop | D4 Unasgn |
|----------|--------------|--------|--------|---------|-----------|---------|-----------|
| gt_track_14 | p0001 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_15 | p0002 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_16 | p0003 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_17 | p0004 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_18 | p0005 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_19 | p0006 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_20 | p0007 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_21 | p0008 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_22 | p0009 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_23 | p0010 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_24 | p0011 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_25 | p0012 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_26 | p0013 | 1.000 | 301 | 301 | 0 | 0 | 0 |
| gt_track_27 | p0014 | 1.000 | 301 | 301 | 0 | 0 | 0 |

## Conservation Check

**Status:** PASS

Six modes sum to total GT frames per track:

  gt_track_14: 301 frames, 301 mode entries -> OK
  gt_track_15: 301 frames, 301 mode entries -> OK
  gt_track_16: 301 frames, 301 mode entries -> OK
  gt_track_17: 301 frames, 301 mode entries -> OK
  gt_track_18: 301 frames, 301 mode entries -> OK
  gt_track_19: 301 frames, 301 mode entries -> OK
  gt_track_20: 301 frames, 301 mode entries -> OK
  gt_track_21: 301 frames, 301 mode entries -> OK
  gt_track_22: 301 frames, 301 mode entries -> OK
  gt_track_23: 301 frames, 301 mode entries -> OK
  gt_track_24: 301 frames, 301 mode entries -> OK
  gt_track_25: 301 frames, 301 mode entries -> OK
  gt_track_26: 301 frames, 301 mode entries -> OK
  gt_track_27: 301 frames, 301 mode entries -> OK

## RIDER: GT Person Count at Pair-Box Locations

For the production pair-boxes that caused misattribution in CP7-pre-3,
the GT-injected run shows whether each location has ONE GT person or TWO.

Since GT tracklet_ids are unique per GT person and threaded 1:1, every
detection in the GT-ceiling run corresponds to exactly one GT person.
If D1 creates GROUP nodes merging two GT tracklets, that means two real
GT people were spatially close enough to trigger a merge event -- confirming
the second person was REAL, not a duplicate detection.

D1 GROUP nodes in GT-ceiling run: 0

No GROUP nodes -- GT tracks never triggered merge/split events.
In the production run, GROUP events at these locations were caused
by tracker-assigned tracklet fragmentation, not by real person overlap.

## Interpretation

**present_misattributed = 0.0%** (near-zero).

D->F produces correct identities given perfect detection input.
Detection is confirmed as the lever for CP7. The 59-66% misattribution
in the production run is caused by detection under-segmentation and
tracker fragmentation, not by Stage D routing bugs.
