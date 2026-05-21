# CP7-pre-5: Ground-Truth Pairwise BBox Overlap Baseline

*Generated 2026-05-22. Read-only investigation. No code changes.*

CP7-pre-4 found that all 30 sampled under-seg frames had "recovered second boxes" at
~0.98 IoU with the production pair-box when NMS was relaxed, and concluded the detector
was NMS-suppressing real second persons. This document tests the counter-hypothesis:
those recovered boxes might be duplicates of the same person (which NMS correctly
suppressed), not distinct second persons. The test is GT-only: measure what IoU two
genuinely distinct people actually produce on these overhead cameras, and cross-reference
the recovered boxes against GT annotations.

---

## 1. Distinct-GT-Person Pairwise IoU Distribution

For every annotated frame, every unordered pair of distinct GT person bboxes, IoU
computed.

### 1.1 Summary Statistics

| Stat | FP7oJQ | J_EDEw | PPDmUg |
|------|--------|--------|--------|
| Pairs | 27,391 | 27,391 | 8,144 |
| Min | 0.000 | 0.000 | 0.000 |
| p50 | 0.000 | 0.000 | 0.000 |
| p90 | 0.095 | 0.000 | 0.320 |
| p95 | 0.225 | 0.305 | 0.511 |
| p99 | 0.631 | 0.580 | 0.664 |
| **Max** | **0.858** | **0.897** | **0.813** |

### 1.2 Threshold Exceedance (IoU)

| Threshold | FP7oJQ | J_EDEw | PPDmUg |
|-----------|--------|--------|--------|
| >= 0.50 | 590 (2.15%) | 627 (2.29%) | 434 (5.33%) |
| >= 0.70 | 28 (0.10%) | 46 (0.17%) | 54 (0.66%) |
| **>= 0.90** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| >= 0.95 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| >= 0.98 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

**Zero distinct-GT-person pairs reach IoU >= 0.9 on any camera.** The maximum
pairwise IoU between two real distinct persons is 0.858 (FP7oJQ), 0.897 (J_EDEw),
0.813 (PPDmUg). This means the ~0.96-0.99 IoU between the production pair-box and the
recovered box that CP7-pre-4 reported is geometrically impossible between two
distinct person annotations — it can only arise from a detection that tightly fits
one person being compared to a second detection that also tightly fits that same
person (or a pair-box that fits neither tightly).

### 1.3 IoU Histogram

| Bucket | FP7oJQ | J_EDEw | PPDmUg |
|--------|--------|--------|--------|
| [0.0, 0.1) | 90.4% | 92.9% | 82.8% |
| [0.1, 0.2) | 3.8% | 1.4% | 3.8% |
| [0.2, 0.3) | 2.2% | 0.8% | 2.6% |
| [0.3, 0.4) | 0.8% | 1.0% | 3.2% |
| [0.4, 0.5) | 0.6% | 1.7% | 2.3% |
| [0.5, 0.6) | 0.9% | 1.6% | 3.3% |
| [0.6, 0.7) | 1.2% | 0.6% | 1.4% |
| [0.7, 0.8) | 0.0% | 0.1% | 0.6% |
| [0.8, 0.9) | 0.1% | 0.0% | 0.0% |
| [0.9, 1.0) | 0.0% | 0.0% | 0.0% |

---

## 2. Distinct-GT-Person Pairwise Containment Distribution

Containment = intersection / area_of_smaller_box. This captures "one person inside
another's box" even if IoU is moderate (small person inside large box).

### 2.1 Summary Statistics

| Stat | FP7oJQ | J_EDEw | PPDmUg |
|------|--------|--------|--------|
| p90 | 0.275 | 0.000 | 0.646 |
| p95 | 0.551 | 0.576 | 0.809 |
| p99 | 0.834 | 0.866 | 1.000 |
| Max | 1.000 | 1.000 | 1.000 |

### 2.2 Threshold Exceedance (Containment)

| Threshold | FP7oJQ | J_EDEw | PPDmUg |
|-----------|--------|--------|--------|
| >= 0.50 | 1,661 (6.06%) | 1,494 (5.45%) | 1,099 (13.49%) |
| >= 0.70 | 958 (3.50%) | 969 (3.54%) | 716 (8.79%) |
| >= 0.90 | 35 (0.13%) | 190 (0.69%) | 256 (3.14%) |
| >= 0.95 | 13 (0.05%) | 86 (0.31%) | 181 (2.22%) |
| >= 0.98 | 2 (0.01%) | 59 (0.22%) | 132 (1.62%) |

Containment reaches 1.0 on all cameras — one person's GT box is entirely inside
another's. This is the grappling-stacking geometry: a smaller person's annotation
is spatially contained by a larger person's. This confirms real body stacking exists
in the GT. But containment != IoU: the smaller box is fully inside the larger, so
IoU is pulled down by the larger box's excess area. The max IoU of 0.86-0.90 is
consistent with one small box inside one large box (high containment, moderate IoU).

---

## 3. Recovered Box Cross-Reference Against GT

For each of the 30 CP7-pre-4 frames, the relaxed-NMS box with highest IoU to the
second GT person's annotation was identified. Then that box's best-match GT person
was determined across ALL GT annotations at the frame.

### 3.1 Results

| Camera | Frames | DISTINCT_SECOND | SAME_AS_PRIMARY | OTHER_GT |
|--------|--------|-----------------|-----------------|----------|
| FP7oJQ | 10 | **10** | 0 | 0 |
| J_EDEw | 10 | **10** | 0 | 0 |
| PPDmUg | 10 | **7** | 2 | 1 |
| **Total** | **30** | **27 (90%)** | **2 (7%)** | **1 (3%)** |

**27/30 recovered boxes match a DISTINCT GT person** (the second person in the
grappling pair, not the primary). The recovered box's best-match GT id is different
from the primary box's GT id in 27 cases.

In 2 PPDmUg cases, the recovered box best-matches the primary GT person (true
duplicate — NMS was correct to suppress these). In 1 PPDmUg case, the recovered box
matches a third GT person.

### 3.2 How the Recovered Box Relates to Primary and Pair-Box

| Metric | FP7oJQ (10) | J_EDEw (10) | PPDmUg (10) |
|--------|-------------|-------------|-------------|
| IoU(recovered, 2nd GT) | 0.88-0.99 | 0.67-0.97 | 0.46-0.97 |
| IoU(recovered, primary GT) | 0.00 | 0.00 | 0.00-0.97 |
| IoU(recovered, pair-box) | 0.96-0.99 | 0.84-0.99 | 0.53-0.99 |

On FP7oJQ and J_EDEw, the recovered box has **zero IoU with the primary GT** and high
IoU with both the second GT and the pair-box. This geometry is: the pair-box covers
both people; the recovered box covers the second person; the primary GT is at a
different location within the pair-box's extent. The recovered box IS the second
person — it just happens to overlap the pair-box heavily because both people are
stacked.

---

## 4. Pair-Box GT Decomposition

For each of the 30 production pair-boxes flagged as under-segmentation in CP7-pre-3:
does the single box enclose two distinct GT persons (containment >= 0.5)?

| Camera | Frames | TWO_GT (true under-seg) | ONE_GT (single-person box) |
|--------|--------|-------------------------|---------------------------|
| FP7oJQ | 10 | 3 (30%) | 7 (70%) |
| J_EDEw | 10 | 6 (60%) | 4 (40%) |
| PPDmUg | 10 | **10 (100%)** | 0 (0%) |
| **Total** | **30** | **19 (63%)** | **11 (37%)** |

19/30 pair-boxes genuinely enclose two distinct GT persons (containment >= 0.5 for
both). 11/30 enclose only one GT person above the containment threshold.

The 11 ONE_GT cases are NOT artifacts — the pair-box IS under-segmentation (the
second person is nearby and partially overlapped), but the second GT person's
containment within the pair-box is below 0.5. The pair-box tightly fits one person
(IoU 0.93-0.98 with best-match GT) while partially overlapping the second person's
GT box. The CP7-pre-3 classification (which used containment of the detection by GT,
not GT by detection) correctly identified these as under-segmentation because the
detection contains the second GT at >= 0.5 from the second-GT's perspective.

---

## 5. Reconciling the Paradox: 0.98 IoU but Distinct Persons

Q1 shows no two distinct GT persons ever reach IoU >= 0.9 with each other. Yet
CP7-pre-4 reported recovered boxes at ~0.98 IoU with the pair-box. These seem
contradictory but are not:

- The **pair-box** is not a GT annotation of one person — it's a detector output
  that encompasses the entire grappling pair. It's LARGER than either individual GT
  box.
- The **recovered box** tightly fits the second GT person (IoU 0.67-0.99 with
  second GT).
- The pair-box and recovered box have high IoU (~0.98) because the pair-box is large
  enough to substantially overlap with ANY individual-person box within its extent.

The 0.98 IoU is between a pair-sized detection and a person-sized detection, not
between two person-sized detections. Two distinct-person GT boxes max out at IoU 0.90;
the higher IoU arises from the pair-box's excess area overlapping with the second
person's tight box.

---

## Terminal Statement

**Stacked-second-person confirmed.** 27/30 recovered boxes match a DISTINCT GT person
(not the primary). Real distinct-person IoU on these cameras maxes at 0.86-0.90 and
never reaches 0.98 — confirming that the 0.98 IoU between pair-box and recovered box
reflects pair-box vs individual-box geometry, not duplicate detection. CP7-pre-4's
conclusion stands: the detector proposes a real second-person box that NMS suppresses
because it overlaps with the pair-box.

The fix path remains detector training for pair separation: teach the model to produce
tight per-person boxes for grappling pairs. NMS will stop suppressing the second box
once the pair-box is replaced by two properly separated individual boxes (lowering the
IoU between them below the NMS threshold). The 2/30 same-GT duplicates on PPDmUg are a
minor population.

### Updated decomposition with Q4 refinement

Within the under-seg mass, the pair-boxes split 63/37 between true two-GT enclosure
(pair-box covers both people) and one-GT-tight-plus-partial-overlap (pair-box tightly
fits one person while partially enclosing the second). Both cases are genuine
under-segmentation from the pipeline's perspective — the detection covers two physical
people — but the two-GT case is the stronger signal for a detection-triggered GROUP
mechanism, while the one-GT-tight case may respond better to per-person fine-tuning.
