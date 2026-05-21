# CP7-pre-2: Misattribution Cause Diagnostic

*Generated 2026-05-21. Read-only investigation. No code changes.*

CP5 collapsed d3_dropped; present_misattributed (59-66%) is now the dominant failure
mode. This document determines whether that mass is caused by tracklet impurity
(tracker box drifting between bodies) or D3 stitch errors (pure tracklets wrongly
merged), and whether improved crops rescue the HSV signal. The answer decides CP7's
direction: Option A (tracker ReID) vs Option B (post-hoc D3 embeddings).

---

## 1. Fragmentation vs Impurity Split -- THE Headline

### 1.1 Methodology

For each `present_misattributed` row in the CP5 gt_person_trace, we classify:

- **Impurity-driven:** The tracklet covers 2+ GT persons over its lifetime AND the
  GT person at this frame is NOT the tracklet's dominant GT person. The tracklet is
  physically on the wrong body at this frame. No D3 fix possible.
- **Stitch/canonical error:** Either the tracklet is pure (covers 1 GT person) but D3
  assigned it to wrong canonical, OR the tracklet is impure but the GT person at this
  frame IS the dominant person (tracklet "belongs" to them, canonical mapping is wrong).

### 1.2 Results

| Camera | Misattributed frames | Impurity-driven | Stitch/canonical | Impurity % |
|--------|---------------------|-----------------|------------------|------------|
| FP7oJQ | 2,765 | 1,957 | 808 | **70.8%** |
| J_EDEw | 2,481 | 1,970 | 511 | **79.4%** |
| PPDmUg | 1,554 | 1,155 | 399 | **74.3%** |

**Headline: 71-79% of all misattributed frames are impurity-driven.** The tracklet's
bbox is physically on the wrong GT person at the misattributed frame. This is a
within-tracklet failure, not a D3 routing error.

### 1.3 Fragmentation Metrics

| Camera | Tracklets/GT person (median) | GT persons/tracklet (median) | Purity (median) |
|--------|------------------------------|------------------------------|-----------------|
| FP7oJQ | 13 | 8 | 0.382 |
| J_EDEw | 44 | 4 | 0.400 |
| PPDmUg | 22 | 3 | 0.500 |

Purity = fraction of a tracklet's trace-matched frames that belong to its dominant GT
person. Median purity of 0.38-0.50 means the typical tracklet spends only 38-50% of its
life on its "correct" person.

### 1.4 Interpretation

The impurity dominance is severe. ~75% of misattributed frames cannot be fixed by any
downstream mechanism (D3, ReID at D-time, post-hoc merge) because the tracklet itself
is wrong -- its bbox is tracking the wrong body. Only fixing the tracker (Phase 1) can
address this.

The 21-29% stitch/canonical share is also important: these ARE fixable by better D3
identity, but they represent only ~13-19% of all GT frames (stitch% * misattributed% of
total), a secondary target.

---

## 2. Crop-Quality Gate -- Does a Better Crop Rescue HSV?

### 2.1 Background Contamination

Current center-bbox crops (60%) on J_EDEw have a median foreground fraction of 48.9%
after background subtraction (threshold 30). In other words, ~51% of pixels in the
typical crop are mat/static background. A tighter 40% crop raises this to 52.6%
(marginal improvement -- the person is roughly centered).

| Crop method | Median fg fraction | Range |
|-------------|-------------------|-------|
| 60% center | 0.489 | 0.062 - 0.899 |
| 40% center | 0.526 | 0.012 - 1.000 |

### 2.2 HSV Bhattacharyya Distances (8 Key J_EDEw Tracklets)

28 tracklet pairs, compared across 4 crop methods:

| Method | Min dist | Median dist | Max dist | Pairs <0.1 | Pairs <0.2 | Pairs >0.5 |
|--------|---------|-------------|---------|------------|------------|------------|
| **Stored (inventory)** | 0.042 | 0.136 | 0.437 | 7 | 22 | 0 |
| **Baseline 60%** (recomputed) | 0.049 | 0.146 | 0.445 | 7 | 21 | 0 |
| **Tight 40%** | 0.043 | 0.175 | 0.554 | 3 | 16 | 2 |
| **BG-sub 60%** | 0.037 | 0.191 | 0.520 | 2 | 16 | 1 |
| **BG-sub + tight 40%** | 0.055 | 0.247 | 0.955 | 1 | 11 | 4 |

### 2.3 Does Separability Improve?

**Modestly.** The combined tight + bg-subtracted method raises the median distance from
0.136 to 0.247 (1.8x) and pushes 4 pairs past 0.5 (vs 0 at baseline). The most
distinct pair (t3 vs t94) reaches 0.955.

But the closest pairs remain stubbornly close: t1 vs t2 at 0.055 (was 0.052), t111 vs
t5 at 0.102 (was 0.052 -- some improvement). And 11 of 28 pairs (39%) remain below
0.2 even with the best crop method.

**Verdict: Crop improvement helps at the margin but does not rescue HSV.** The
fundamental problem is that HSV color distributions at 18x8 resolution cannot
distinguish 14 people wearing gi tops, even with background removed. The overhead
angle means all crops are backs/torsos with limited color diversity.

### 2.4 Implication for Learned Descriptors

This result is a **partial caution** for learned ReID models too. If the input crop is
~50% mat pixels and the remaining 50% is back-of-gi at overhead angles, a learned model
needs to extract identity signal from low-information views. Domain gap from standard
ReID datasets (side/front views, street cameras) will be significant. However, learned
models can potentially exploit texture, pattern, and spatial structure that HSV
histograms discard, so the situation is not as hopeless as for HSV.

Background subtraction as a preprocessing step (masking mat pixels before the ReID
model) is worth testing but not guaranteed to help -- it depends on whether the ReID
backbone already learns to ignore static backgrounds.

---

## 3. Would Tracker ReID Reduce Fragmentation?

### 3.1 Fragmentation Event Characteristics

| Camera | Events | Consecutive annotated (gap=1 stride) | ID switches (both tracklets concurrent) |
|--------|--------|--------------------------------------|----------------------------------------|
| FP7oJQ | 1,031 | 83.2% | **98.9%** |
| J_EDEw | 2,288 | 80.9% | **95.5%** |
| PPDmUg | 1,257 | 87.2% | **97.1%** |

**95-99% of fragmentation events are ID SWITCHES, not track breaks.**

Both the old and new tracklets are active at the switch frame. The GT person's
detection "jumps" from one concurrent tracklet to another. Temporal overlap between
the two tracklets is massive: median 315 frames (FP7oJQ), 556 (J_EDEw), 993 (PPDmUg).

Only 1-5% of events are clean track breaks (old tracklet ends, new one starts after a
gap). These are the only events where ReID re-linking across a gap would help.

### 3.2 Sampled Fragmentation Events (5 per camera)

**FP7oJQ** (stride 1, all 5 are ID switches):

| GT | Frames | Tracklets | People detected | Overlap | Classification |
|----|--------|-----------|-----------------|---------|---------------|
| 25 | 155->156 | t4->t7 | 13/14 | 246 fr | Entangled switch |
| 23 | 136->137 | t3->t6 | 12/14 | 236 fr | Entangled switch |
| 16 | 14->15 | t5->t4 | 12/14 | 262 fr | Entangled switch |
| 27 | 137->142 | t22->t8 | 13/14 | 31 fr | Crowded gap |
| 18 | 213->238 | t8->t7 | 12/14 | 254 fr | Crowded gap |

Every event occurs with 12-13 of 14 GT persons detected -- the mat is full. IoU at
switch is high (0.81-0.97), meaning the detection is a good bbox but the tracker
assigned it to the wrong tracklet.

**J_EDEw and PPDmUg** show the same pattern: all sampled events have 7-13 people
detected, all have large temporal overlaps, all are concurrent tracklet switches.

### 3.3 Why ReID at the Tracker Won't Fix This

The fragmentation is **not** caused by the tracker losing a person and starting a new
track (the scenario ReID is designed to fix). It's caused by the tracker's assignment
step picking the wrong existing tracklet for a detection in a crowded scene.

BoT-SORT's assignment uses IoU + optional ReID distance in a cost matrix, then applies
the Hungarian algorithm. In these events:
- Multiple tracklets are nearby (12-13 people on the mat)
- Two or more tracklets have similar IoU with the detection
- The tracker picks wrong → the detection "jumps" between tracklets

ReID embeddings in the tracker would help IF the embeddings can distinguish the correct
tracklet from nearby alternatives. But at the switch frame, the person is typically in
contact with or adjacent to others (that's why the IoU is ambiguous). The crop at that
moment captures entangled bodies -- exactly the scenario where appearance is least
discriminative.

**Estimate of ReID impact on fragmentation:** Of 95-99% ID-switch events, ReID would
need to distinguish entangled bodies in overhead crops. Based on Section 2's finding
that even clean (isolated, background-subtracted) crops have limited separability, ReID
during active grappling is unlikely to resolve more than a small fraction.

The 1-5% track-break events ARE ReID-helpable, but represent a negligible fraction of
the misattribution mass.

---

## Summary and CP7 Direction Recommendation

### The Numbers That Decide

| Fact | Number | Implication |
|------|--------|-------------|
| Misattributed frames that are impurity-driven | **71-79%** | Tracklet bbox is on wrong body. No downstream fix. |
| Fragmentation events that are ID switches | **95-99%** | Tracker is swapping between concurrent tracklets, not losing-and-restarting. |
| HSV improvement from bg-sub + tight crop | Median 0.136 -> 0.247 | Helps marginally, doesn't solve. |
| Track breaks (ReID-helpable by design) | **1-5%** | Negligible fraction of misattribution. |

### What This Means for Option A vs Option B

**Option A (tracker ReID) cannot fix the dominant failure mode.** The 71-79%
impurity-driven misattribution comes from ID switches in crowded scenes where both
tracklets are active and bodies are entangled. Tracker ReID improves association at
gaps (1-5% of events), not assignment during concurrent overlap. Even if ReID made
gap-association perfect, the impact on misattribution would be <5pp.

**Option B (post-hoc D3 embeddings) targets the 21-29% stitch/canonical share** but
also cannot fix the 71-79% impurity-driven mass, since those frames have tracklets
physically on the wrong body.

**Neither A nor B, as scoped, solves the problem.** The root cause is that BoT-SORT's
motion-based assignment treats 14 people on one mat as interchangeable when IoU is
ambiguous. The resulting tracklets are impure by construction.

### Possible Directions for CP7

1. **Reduce effective scene density.** If the tracker saw fewer candidates per
   assignment step, ID switches would decrease. Options: spatial partitioning (assign
   tracklets in mat-region zones), or grappling-pair tracking (track pairs as a unit
   rather than individuals).

2. **Accept impure tracklets, fix identity at sub-tracklet granularity.** Instead of
   one person_id per tracklet, assign person_id per detection (or per short segment).
   This is a fundamental architecture change (D4 currently assigns per-tracklet).

3. **Better tracker backbone.** Replace BoT-SORT with a transformer tracker or one
   designed for crowded overhead scenes. High risk, high effort.

4. **Hybrid approach.** Enable tracker ReID for the 1-5% track breaks (easy win, low
   effort), AND pilot sub-tracklet identity assignment for the impurity mass (high
   effort, high reward).

The evidence strongly favors Option 2 or 4 over pure Option A or B.
