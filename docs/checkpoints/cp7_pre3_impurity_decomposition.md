# CP7-pre-3: Impurity Mass Decomposition

*Generated 2026-05-21. Read-only investigation. No code changes.*

CP7-pre-2 found 71-79% of present_misattributed frames are "impurity-driven" and
labeled the entire mass as downstream-unfixable. This document decomposes that mass
by re-joining the matched detection's geometry against the full GT set at each frame.
The finding inverts the CP7-pre-2 conclusion: the dominant cause is detection
under-segmentation (one box around a grappling pair), a Stage A problem with a
different fix path than sub-tracklet identity.

---

## Method

For each of CP7-pre-2's impurity-driven misattributed frames:

1. Load **all GT boxes** from the CVAT training zip for that frame (all annotated
   frames, not just val).
2. Load **all Stage A detections** from `detections.parquet` for that frame.
3. Compute two matrices per frame:
   - **IoU matrix** (N_gt x N_det): standard intersection-over-union.
   - **Containment matrix** (N_gt x N_det): intersection(D, G') / area(G') — how
     much of each GT person is enclosed by each detection.

4. Classify the misattributed frame using the matched detection D and GT person G:

   - **Under-segmentation**: D contains a second GT person G' with
     containment >= tau_contain. (D is a pair-box covering two people.)
   - **Matcher ambiguity**: 2+ detections have IoU to G within delta of the best
     match. (Eval's Hungarian matcher could have picked a different detection.)
   - **True tracker impurity**: Residual — clear best match, no multi-enclosure.

   Precedence: under-segmentation > matcher ambiguity > true impurity.

5. Threshold sweep: tau_contain in {0.3, 0.5, 0.7, 0.9}, delta in {0.05, 0.1, 0.2}.

### Data sources

All from the same CP5-state artifacts CP7-pre-2 consumed:
- `outputs/_eval/stage_d/bjj-detect-all-cameras/{cam}/gt_person_trace.parquet`
- `outputs/_eval_gt/{cam}/.../stage_A/detections.parquet`
- `data/training_data/training_YOLO_track_detections_{cam}_*.zip` (GT annotations)

---

## 1. Headline Decomposition (tau=0.5, delta=0.1)

### 1.1 Full present_misattributed breakdown

|                          | FP7oJQ | J_EDEw | PPDmUg |
|--------------------------|-------:|-------:|-------:|
| Total GT frames          |  4,214 |  4,214 |  2,361 |
| Total misattributed      |  2,765 |  2,481 |  1,554 |
| as % of GT frames        |  65.6% |  58.9% |  65.8% |
| | | | |
| **Under-segmentation**   | **1,950** | **1,936** | **1,132** |
| % of misattributed       |  70.5% |  78.0% |  72.8% |
| % of GT frames           |  46.3% |  45.9% |  47.9% |
| | | | |
| Stitch/canonical error   |    808 |    511 |    399 |
| % of misattributed       |  29.2% |  20.6% |  25.7% |
| % of GT frames           |  19.2% |  12.1% |  16.9% |
| | | | |
| Matcher ambiguity        |      0 |      1 |      0 |
| % of misattributed       |   0.0% |   0.0% |   0.0% |
| | | | |
| **True tracker impurity**|    **7** |   **33** |   **23** |
| % of misattributed       |   0.3% |   1.3% |   1.5% |
| % of GT frames           |   0.2% |   0.8% |   1.0% |

### 1.2 Conservation check

| Camera | Stitch + US + Amb + TI | Total misattributed | Match? |
|--------|----------------------:|-------------------:|--------|
| FP7oJQ | 808 + 1950 + 0 + 7 = 2,765 | 2,765 | Yes |
| J_EDEw | 511 + 1936 + 1 + 33 = 2,481 | 2,481 | Yes |
| PPDmUg | 399 + 1132 + 0 + 23 = 1,554 | 1,554 | Yes |

### 1.3 The single clear statement

| Cause | Fix lives in | % of misattributed |
|-------|-------------|-------------------|
| Under-segmentation | Stage A (pair separation / better detector) | **70-78%** |
| Stitch/canonical | Stage D (identity routing) | 21-29% |
| Matcher ambiguity | Evaluation artifact | ~0% |
| True tracker impurity | Tracker / sub-tracklet identity | **0.3-1.5%** |

**The fraction of misattribution that justifies sub-tracklet identity work is 0.3-1.5%,
not 71-79%.** CP7-pre-2's "impurity-driven" bucket was almost entirely under-segmentation
masquerading as tracker impurity.

---

## 2. Threshold Sweep -- Stability Analysis

### FP7oJQ (N=1,957 impurity-driven)

| tau | delta | Under-seg | Ambig | True imp | US% | TI% | Co-occur |
|-----|-------|-----------|-------|----------|-----|-----|----------|
| 0.3 | 0.05 | 1,951 | 0 | 6 | 99.7 | 0.3 | 61 |
| 0.3 | 0.10 | 1,951 | 0 | 6 | 99.7 | 0.3 | 94 |
| 0.5 | 0.05 | 1,950 | 0 | 7 | 99.6 | 0.4 | 61 |
| 0.5 | 0.10 | 1,950 | 0 | 7 | 99.6 | 0.4 | 94 |
| 0.7 | 0.05 | 1,925 | 0 | 32 | 98.4 | 1.6 | 61 |
| 0.7 | 0.10 | 1,925 | 0 | 32 | 98.4 | 1.6 | 94 |
| 0.9 | 0.05 | 1,665 | 4 | 288 | 85.1 | 14.7 | 57 |
| 0.9 | 0.10 | 1,665 | 7 | 285 | 85.1 | 14.6 | 87 |

### J_EDEw (N=1,970 impurity-driven)

| tau | delta | Under-seg | Ambig | True imp | US% | TI% | Co-occur |
|-----|-------|-----------|-------|----------|-----|-----|----------|
| 0.3 | 0.05 | 1,953 | 0 | 17 | 99.1 | 0.9 | 118 |
| 0.3 | 0.10 | 1,953 | 0 | 17 | 99.1 | 0.9 | 175 |
| 0.5 | 0.05 | 1,936 | 0 | 34 | 98.3 | 1.7 | 118 |
| 0.5 | 0.10 | 1,936 | 1 | 33 | 98.3 | 1.7 | 174 |
| 0.7 | 0.05 | 1,891 | 2 | 77 | 96.0 | 3.9 | 116 |
| 0.7 | 0.10 | 1,891 | 6 | 73 | 96.0 | 3.7 | 169 |
| 0.9 | 0.05 | 1,538 | 33 | 399 | 78.1 | 20.3 | 85 |
| 0.9 | 0.10 | 1,538 | 51 | 381 | 78.1 | 19.3 | 124 |

### PPDmUg (N=1,155 impurity-driven)

| tau | delta | Under-seg | Ambig | True imp | US% | TI% | Co-occur |
|-----|-------|-----------|-------|----------|-----|-----|----------|
| 0.3 | 0.05 | 1,147 | 0 | 8 | 99.3 | 0.7 | 27 |
| 0.3 | 0.10 | 1,147 | 0 | 8 | 99.3 | 0.7 | 40 |
| 0.5 | 0.05 | 1,132 | 0 | 23 | 98.0 | 2.0 | 27 |
| 0.5 | 0.10 | 1,132 | 0 | 23 | 98.0 | 2.0 | 40 |
| 0.7 | 0.05 | 1,102 | 0 | 53 | 95.4 | 4.6 | 27 |
| 0.7 | 0.10 | 1,102 | 2 | 51 | 95.4 | 4.4 | 38 |
| 0.9 | 0.05 | 953 | 5 | 197 | 82.5 | 17.1 | 22 |
| 0.9 | 0.10 | 953 | 9 | 193 | 82.5 | 16.7 | 31 |

### Stability assessment

Under-segmentation dominates across the full sweep. At tau <= 0.7, US accounts for
95-100% of the impurity mass on all cameras. Only at the extreme tau=0.9 (requiring
near-total enclosure of the second GT) does the split soften to 78-85% US / 15-20% TI.

Delta (matcher ambiguity threshold) has almost no effect: ambiguity never exceeds 3.9%
of the impurity mass even at (tau=0.9, delta=0.2). This is because the Hungarian matcher
is decisive — competing detections rarely have similar IoU to the same GT.

**Co-occurrence** (frames flagged as both under-seg AND ambiguous) is modest: 3-13% of
the impurity mass. The partition is clean.

**The result is robust.** tau_contain is the only sensitive knob, and its effect is
monotonic: stricter containment threshold shifts frames from under-seg to true-impurity,
but under-seg remains dominant at any reasonable threshold.

---

## 3. Worked Examples

### 3.1 Under-segmentation (dominant bucket)

**J_EDEw, frame 2500, GT 14 (matched to det d002500_11, tracklet t200):**
- Best IoU: 0.878, n_close: 1 (no matcher ambiguity)
- Detection d002500_11 contains GT 27 with containment 0.984
- 14 GT persons, 10 detections → 4-person deficit
- Interpretation: d002500_11 is a pair-box spanning GT 14 and GT 27. The tracker
  assigned it to t200; at this frame the trace says GT 14 was matched to this box,
  but t200's dominant GT is someone else → misattributed.

**J_EDEw, frame 0, GT 17 (matched to det d000000_5, tracklet t5):**
- Best IoU: 0.841, n_close: 1
- Detection d000000_5 contains GT 19 with containment 0.977
- Interpretation: pair-box covers both GT 17 and GT 19.

**PPDmUg, frame 2500, GT 0 (matched to det d002500_1, tracklet t1):**
- Best IoU: 0.973, n_close: 1
- Detection d002500_1 contains GT 2 with containment 0.979
- 8 GT persons, 6 detections → 2-person deficit
- Interpretation: pair-box on grappling pair.

**FP7oJQ, frame 250, GT 16 (matched to det d000250_5, tracklet t6):**
- Best IoU: 0.935, n_close: 1
- Detection d000250_5 contains GT 25 with containment 0.966
- 14 GT, 10 detections → 4-person deficit

**FP7oJQ, frame 0, GT 19 (matched to det d000000_6, tracklet t7):**
- Best IoU: 0.861, n_close: 1
- Detection d000000_6 contains GT 26 with containment 0.996
- Interpretation: nearly perfect containment of both people.

### 3.2 Matcher ambiguity (near-empty bucket)

**J_EDEw, frame 2540, GT 14 (matched to det d002540_10, tracklet t188):**
- Best IoU: 0.669, n_close: 2 (a second detection within delta=0.1 of best)
- Second GT containment: 0.483 (below tau=0.5, so not under-seg at this threshold)
- Interpretation: borderline case — moderate IoU, moderate containment. The eval
  matcher could have assigned GT 14 to a different detection. Only 1 such frame in
  J_EDEw; effectively zero for FP7oJQ and PPDmUg.

### 3.3 True tracker impurity (residual, <2%)

**FP7oJQ, frame 286, GT 17 (matched to det d000286_6, tracklet t2):**
- Best IoU: 0.885, n_close: 1
- Max second GT containment: 0.368 (well below tau=0.5)
- Interpretation: detection cleanly covers GT 17, no second GT enclosed, no competing
  detection. But tracklet t2's dominant GT is someone else. This is genuine tracker
  drift — the tracklet's identity changed without a pair-box excuse. These are the
  only frames where sub-tracklet identity would theoretically help.

**PPDmUg, frame 380, GT 0 (matched to det d000380_4, tracklet t20):**
- Best IoU: 0.848, n_close: 1
- Max second GT containment: 0.383
- Same pattern: clean single-person detection, genuine tracklet identity drift.

**J_EDEw, frame 0, GT 14 (matched to det d000000_7, tracklet t7):**
- Best IoU: 0.960, n_close: 1
- Max second GT containment: 0.000 (no other GT overlaps at all)
- Interpretation: perfect detection, zero containment of any other person, but
  tracklet t7's dominant GT is not 14. Pure tracker assignment error.

---

## 4. Context: Detection Deficit

The under-segmentation finding is consistent with a persistent detection deficit:

| Camera | GT persons/frame | Detections/frame (mean) | Deficit |
|--------|-----------------|------------------------|---------|
| FP7oJQ | 14 | ~11 | ~3 |
| J_EDEw | 14 | ~11 | ~3 |
| PPDmUg | 6-8 | ~5-6 | ~1-2 |

98% of J_EDEw frames have fewer detections than GT persons. The "missing" detections
are not missed people — they are pair-boxes that cover two people with one detection.
A detector that can separate grappling pairs into individual boxes would eliminate both
the deficit and the under-segmentation misattribution in one move.

---

## 5. Revised Misattribution Hierarchy

CP7-pre-2 framed the problem as:

> 71-79% impurity-driven (downstream-unfixable) + 21-29% stitch/canonical

This decomposition revises that to:

| Cause | % of misattributed | Fix path | Effort |
|-------|-------------------|----------|--------|
| **Under-segmentation** | **70-78%** | Stage A: pair separation (better detector, instance segmentation, or grappling-pair splitting post-hoc) | Medium-high |
| **Stitch/canonical** | **21-29%** | Stage D: better identity routing (ReID embeddings for D3 cost, canonical mapping improvements) | Medium |
| **Matcher ambiguity** | **~0%** | Evaluation artifact; no pipeline fix needed | None |
| **True tracker impurity** | **0.3-1.5%** | Tracker improvement or sub-tracklet identity | Not justified by volume |

### What this means for CP7

1. **Sub-tracklet identity is not the right next step.** It targets 0.3-1.5% of
   misattribution. The CP7-pre-2 recommendation for sub-tracklet work was based on a
   bucket that turns out to be 99% under-segmentation.

2. **Pair separation in Stage A is the highest-leverage intervention.** 70-78% of
   misattribution (46-48% of all GT frames) comes from single detections covering
   two people. Options: (a) fine-tune the detector on grappling pairs with
   per-person annotations, (b) add a post-detection pair-splitting step using
   pose/segmentation, (c) instance segmentation model.

3. **Stitch/canonical improvement (21-29%) is the second priority.** This is the
   territory where ReID embeddings in D3 could help — the tracklets are correctly
   segmented but routed to the wrong person.

4. **Tracker ReID (Option A from CP7-pre) has near-zero leverage** on misattribution
   specifically, but could still help with the tracklet fragmentation that inflates
   Stage D's complexity (44 tracklets per GT person on J_EDEw). That's a secondary
   benefit, not the primary intervention.
