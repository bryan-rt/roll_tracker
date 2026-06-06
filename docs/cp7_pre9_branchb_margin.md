# CP7-pre-9: True Branch-B Margin — Disambiguating the Suspect Buckets

**Date:** 2026-05-25
**Scope:** READ-ONLY diagnostic. No pipeline/config/node changes.
**Camera:** FP7oJQ, clip FP7oJQ-20260318-200014, frames 0-300 (dense GT)
**Model:** bjj-detect-all-cameras
**Depends on:** CP7-pre-8 (axis1 signature), CP7-pre-3 (containment method)

## Executive Summary

CP7-pre-8 reported 84.3% Branch B (concurrent-swap) as the dominant Axis-1 failure,
recommending a new concurrent-swap node class. That headline was asserted by argument,
not measurement. This diagnostic applies CP7-pre-3's containment test to the two suspect
buckets and finds:

**The 84.3% was almost entirely Axis-2 (detection under-segmentation) in disguise.**

- 92.8% of the suspect frames (1,681/1,811) are pair-boxes: one detection covering two
  GT people. A concurrent-swap node cannot recover these — the detector must separate
  the pair first.
- The TRUE Branch-B margin is **9.9%** (223/2,259 scorable frames): 130 single-person
  frames from the suspect buckets + 93 branch_b_swap passthrough.
- Zero concurrent_role (genuine A/B co-causation) frames found. GROUP overlap is
  incidental tiling, not causal.

**Revised recommendation:** A concurrent-swap node class addresses ~10% of misattribution.
The dominant intervention remains Stage A pair separation (74.4% of misattribution),
consistent with CP7-pre-3's original finding. CP-SPLIT-1 did not change the picture —
under-segmentation was hiding inside pre-8's Branch-B labels all along.

---

## Method

### Containment test (reused from CP7-pre-3)

For each misattributed frame's matched detection box D:

1. Load all GT boxes at that frame from `training_YOLO_track_detections_FP7oJQ_clip1_0-3000.zip`
2. For each GT person G' other than the matched GT person:
   - `containment(D, G') = intersection(D, G') / area(G')`
3. If any G' has containment >= tau_contain (headline: 0.5) -> **pair_box** (Axis-2)

**Precedence rule:** pair_box takes precedence over concurrent_role. A pair-box is
unrecoverable by a concurrent-swap node regardless of what roles concurrent tracklets hold.

### Three-way outcome classification

Applied only to the two suspect buckets from CP7-pre-8 (ambiguous_a_b: 765 frames,
branch_b_persistent: 1,046 frames):

| Outcome | Definition | Recoverable by swap node? |
|---------|-----------|--------------------------|
| **pair_box** | Detection contains 2nd GT person (containment >= tau) | No (Axis-2) |
| **concurrent_role** | Not pair_box, AND both misattrib tracklet and canonical-holder are roles in the SAME GROUP node | No (genuine A/B co-causation) |
| **single_person** | Detection cleanly covers one person, no pair-box, no same-node co-causation | **Yes** (genuine Axis-1) |

### Identity mapping

The concurrent tracklet holding canonical_person_id is identified from the frozen
CP-EVAL-1 identity mapping stored in gt_person_trace.parquet (same mapping pre-8 used).
No re-derivation.

### GT coverage

All 301 frames (0-300, stride 1) have GT label files in the zip. Zero indeterminate
frames. The containment test covers every target row.

---

## Per-Bucket Outcomes (tau=0.5)

### ambiguous_a_b (765 frames)

| Outcome | Frames | % |
|---------|--------|---|
| pair_box | 707 | 92.4% |
| single_person | 58 | 7.6% |
| concurrent_role | 0 | 0.0% |
| indeterminate | 0 | 0.0% |

**92.4% of the ambiguous_a_b bucket is Axis-2.** These frames have a GROUP node active
AND a concurrent tracklet holding the canonical id, but the root cause is that the
detection box covers two people. The GROUP node and concurrent tracklet are both
consequences of the pair-box — not independent failure mechanisms.

### branch_b_persistent (1,046 frames)

| Outcome | Frames | % |
|---------|--------|---|
| pair_box | 974 | 93.1% |
| single_person | 72 | 6.9% |
| concurrent_role | 0 | 0.0% |
| indeterminate | 0 | 0.0% |

**93.1% of branch_b_persistent is Axis-2.** The "persistent concurrent identity
confusion" that pre-8 identified is almost entirely pair-box-driven: one detection
covers two GT people, and the nearby concurrent tracklet covers one of them separately.
The misattribution is because the pair-box tracklet is assigned to the wrong person,
but a swap node cannot fix this — the detector must first separate the pair.

---

## Tau Sweep Stability

| tau | ambig pair_box | b_pers pair_box | Total pair_box | % of 1,811 targets |
|-----|----------------|-----------------|----------------|-------------------|
| 0.3 | 740 | 975 | 1,715 | 94.7% |
| **0.5** | **707** | **974** | **1,681** | **92.8%** |
| 0.7 | 701 | 957 | 1,658 | 91.6% |
| 0.9 | 609 | 802 | 1,411 | 77.9% |

Stable across the sweep. Even at the strictest tau=0.9 (requiring near-total enclosure
of the second GT person), 77.9% of targets remain pair_box. The result is robust.

---

## Corrected GT-0-300 Aggregation

| Category | Frames | % of scorable | % of 2,259 | Source |
|----------|--------|---------------|------------|--------|
| **true_branch_b** | **223** | **9.9%** | **9.9%** | single_person (130) + branch_b_swap (93) |
| **axis2_in_disguise** | **1,681** | **74.4%** | **74.4%** | pair_box from both suspect buckets |
| ab_co_causation | 0 | 0.0% | 0.0% | concurrent_role (same GROUP node) |
| pure_branch_a | 157 | 6.9% | 6.9% | passthrough from pre-8 |
| other | 198 | 8.8% | 8.8% | passthrough from pre-8 |
| indeterminate | 0 | --- | 0.0% | --- |

**TRUE BRANCH-B MARGIN: 223/2,259 = 9.9%**

This replaces the asserted 84.3% from CP7-pre-8.

---

## Cross-Reference with CP7-pre-3

| Metric | CP7-pre-3 | CP7-pre-9 |
|--------|-----------|-----------|
| Run state | Pre-CP-SPLIT-1 | Post-CP-SPLIT-1 |
| Total misattributed | 2,765 | 2,259 |
| Under-seg (Axis-2) | 1,950 (70.5%) | 1,681 (74.4%) |
| Scope tested | All misattr frames | ambig_a_b + b_persistent only (1,811) |

**These are different run states, not a disagreement.** CP-SPLIT-1 changed the tracklet
population (251 → 425 tracklets), which reduced total misattribution (2,765 → 2,259)
but did not eliminate the under-segmentation mass. Pre-9 tested only the two suspect
buckets (1,811 frames); the remaining 448 passthrough frames (branch_a=157,
branch_b_swap=93, other=198) were not re-tested and may contain additional under-seg.

The qualitative finding is consistent: **under-segmentation dominates misattribution
at ~70-74% across both run states.** CP-SPLIT-1 improved tracklet purity but did not
address the root cause (pair-box detections). The under-seg mass that pre-8 labeled
as "Branch B" was always Axis-2 — it was just measured with a different instrument
(concurrent-tracklet presence vs detection geometry).

---

## Verdicts

### Risk (1): Is ambiguous_a_b genuine co-causation or incidental?

**VERDICT: Incidental.** Zero concurrent_role frames (strict same-node test). All 765
ambiguous frames are either pair_box (707, 92.4%) or single_person (58, 7.6%). The
GROUP node overlap is a tiling artifact: GROUP spans are wide and cover most of the
timeline, so any misattributed frame has a high chance of falling inside one. But the
GROUP node is not causing the misattribution — the pair-box is.

Pre-8's argument that "ambiguous_a_b is functionally Branch B" was correct in
direction (not Branch A) but wrong in destination: it's functionally Axis-2, not
Branch B.

### Risk (2): Is there material under-seg hiding in branch_b_persistent?

**VERDICT: YES — overwhelmingly.** 974/1,046 (93.1%) of branch_b_persistent frames
are pair_box. The "concurrent tracklet holding the canonical id" that pre-8 used to
label these as Branch B is a CONSEQUENCE of the pair-box: when one detection covers
two people, a second detection nearby often covers one of them individually. The
"concurrent identity confusion" is real but is caused by under-segmentation, not by
a tracklet swap. A concurrent-swap node cannot recover these frames.

---

## Revised Misattribution Hierarchy (post CP-SPLIT-1)

| Cause | Frames | % | Fix path |
|-------|--------|---|----------|
| **Under-segmentation (Axis-2)** | **1,681** | **74.4%** | Stage A: pair separation |
| **True Branch B (Axis-1)** | **223** | **9.9%** | Stage D: concurrent-swap node |
| Pure Branch A | 157 | 6.9% | Stage D: GROUP routing improvement |
| Other / unclassified | 198 | 8.8% | Investigation needed |

### What this means for CP7

1. **A concurrent-swap node class is a ~10% intervention, not an 84% one.** It's worth
   building if cheap, but it is not the primary lever.

2. **Stage A pair separation remains the highest-leverage intervention** at 74.4% of
   misattribution — unchanged from CP7-pre-3's finding across both run states.

3. **Pure Branch A (6.9%) + true Branch B (9.9%) = 16.8% addressable by Stage D work.**
   Combined, these justify identity-routing improvements but not as the primary
   investment.

4. **CP-SPLIT-1 helped** (reduced misattr from 2,765 to 2,259) but did not change the
   fundamental decomposition. The under-seg mass is structural.

---

## Artifacts

| File | Contents |
|------|----------|
| `outputs/_eval/_debug/cp7_pre9_branchb_margin/containment_results.json` | 1,811 per-frame outcomes |
| `outputs/_eval/_debug/cp7_pre9_branchb_margin/tau_sweep.json` | Sweep data for 4 tau values |
| `tools/cp7_pre9_branchb_margin.py` | Throwaway analysis script |
