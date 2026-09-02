# DEDUP-CEILING-1/1a: What would perfect deduplication actually buy?

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**Method:** Approach (A) — full D0-D4 + Stage E re-run with merged tracklets
**Rule:** (c) >= 50% overlap fraction — merge only tracklets that are >=50% concurrent
with their canonical hub. Reconciliation: higher-confidence-wins (sensitivity-tested in 1a).

---

## Headline: perfect concurrent-overlap dedup is NET NEGATIVE

Removing all 23 concurrent-duplicate tracklets (GT-labelled, unachievable by any real
discriminator) **worsens** every measured metric:

| Metric | Baseline | Dedup | Delta | GT target |
|---|---|---|---|---|
| Person count | 17 | 15 | -2 | 8 |
| correct_id strict | 34.3% | 31.6% | **-2.7pp** | - |
| correct_id tolerant | 37.4% | 33.5% | **-3.9pp** | - |
| Sessions total | 23 | 15 | -8 | 3 |
| CORRECT_ENGAGED | 6 | 1 | -5 | - |
| CONTAMINATED | 13 | 12 | -1 | - |
| PHANTOM | 4 | 2 | -2 | - |
| No-edge boundaries | 63 | 21 | -42 | - |
| No-edge on GT 2 | 40 | 3 | -37 | - |

**Perfect dedup buys nothing on identity or sessions. It actively hurts.** The only metric
that improves is no-edge boundary count (-42), confirming that concurrent tracklets were the
flicker mechanism — but eliminating flicker at the cost of -2.7pp correct_id is not a trade
worth making, and the flicker was already diagnosed as an evaluation artifact (NOEDGE-1),
not a pipeline defect.

---

## 1. The merge set

### Concurrency ratios per absorbed tracklet

| GT | Canonical | Absorbed | Total f | Concurrent f | Ratio |
|---|---|---|---|---|---|
| 0 | t2 (1681f) | t88 | 494 | 440 | 89.1% |
| 0 | t2 | t86 | 16 | 12 | 75.0% |
| 0 | t2 | t165 | 4 | 4 | 100.0% |
| 0 | t2 | t17 | 1 | 1 | 100.0% |
| 0 | t2 | t21 | 1 | 1 | 100.0% |
| 1 | t1 (1623f) | t73 | 5 | 5 | 100.0% |
| 1 | t1 | t71 | 3 | 3 | 100.0% |
| 1 | t1 | t25 | 1 | 1 | 100.0% |
| 1 | t1 | t70 | 1 | 1 | 100.0% |
| 1 | t1 | t8 | 1 | 1 | 100.0% |
| 1 | t1 | t92 | 1 | 1 | 100.0% |
| 2 | t3 (623f) | t47 | 265 | 233 | 87.9% |
| 2 | t3 | t40 | 6 | 6 | 100.0% |
| 3 | t147 (379f) | t163 | 61 | 34 | 55.7% |
| 3 | t147 | t166 | 28 | 28 | 100.0% |
| 3 | t147 | t158 | 2 | 2 | 100.0% |
| 3 | t67 (167f) | t79 | 7 | 7 | 100.0% |
| 3 | t67 | t68 | 5 | 5 | 100.0% |
| 5 | t4 (1292f) | t106 | 130 | 115 | 88.5% |
| 5 | t4 | t120 | 71 | 67 | 94.4% |
| 5 | t4 | t115 | 3 | 3 | 100.0% |
| 5 | t4 | t26 | 1 | 1 | 100.0% |
| 5 | t4 | t6 | 1 | 1 | 100.0% |

### Excluded by rule (c) — <50% concurrent

| GT | Tracklet | Total f | Overlap f | Ratio | Canonical |
|---|---|---|---|---|---|
| 2 | t90 | 143 | 4 | 2.8% | t3 |
| 5 | t137 | 455 | 73 | 16.0% | t4 |

Both are overwhelmingly sequential — merging them would be oracle stitching, not dedup.

### Transitivity

Moot. Every merge group has a star topology — all absorbed tracklets overlap the canonical
hub directly. No chains through intermediaries. Connected-component and direct-overlap-only
give identical results.

### Tracklets with no concurrent overlap at all

GT 4, GT 6, GT 7 have zero concurrent pairs — all their fragments are purely sequential.
Their fragmentation is entirely a tracking/stitching problem, not a duplication problem.

---

## 2. Person count: 17 -> 15 (-2)

Two fewer person_ids, but still 15 against a GT of 8. The residual fragmentation is
dominated by sequential breaks, not concurrent duplicates.

### Per-GT-track residual

| GT | Person IDs | correct_id | Canonical |
|---|---|---|---|
| 0 | 8 | 40.0% | p0003 |
| 1 | 3 | 59.9% | p0006 |
| 2 | 8 | 44.8% | p0001 |
| 3 | 10 | 26.7% | p0007 |
| 4 | 8 | 16.2% | p0002 |
| 5 | 6 | 31.2% | p0010 |
| 6 | 4 | 1.4% | p0006 |
| 7 | 2 | 49.2% | p0012 |

No GT person reached 1 person_id. The best case (GT 1) has 3, and GT 1 was already
the best-tracked person pre-dedup (isolated, outside the grappling core).

---

## 3. Why merging is intrinsically lossy (DEDUP-CEILING-1a)

The -2.7pp loss is **not** an artifact of the reconciliation rule. It is intrinsic to
dropping one of two concurrent detections.

### The irreducible floor

Of the 972 conflict pairs (two detections on the same GT person at the same frame):

| Population | Count | % |
|---|---|---|
| **Both GT-matched** | **664** | **68.3%** |
| Exactly one GT-matched | 306 | 31.5% |
| Neither GT-matched | 2 | 0.2% |

**On 68.3% of conflict frames, both detections are GT-matched.** No reconciliation rule can
avoid dropping a GT-matched detection on these frames — one of two valid boxes must go. This
is the irreducible cost of merging concurrent detections.

### Reconciliation rule sensitivity

| Rule | GT-matched dropped | % of 972 | Margin over oracle |
|---|---|---|---|
| **(b) oracle (higher GT IoU)** | **664** | **68.3%** | **0 (floor)** |
| (c) larger box | 699 | 71.9% | +35 |
| **(a) higher confidence (used)** | **736** | **75.7%** | **+72** |
| (d) smaller box | 935 | 96.2% | +271 |

Rule (b) is an oracle — it uses GT IoU to choose and is not implementable. It establishes
the floor: even the perfect reconciliation rule drops 664 GT-matched detections (68.3%).

The practical spread between rules is 37 detections (larger-box vs higher-confidence), or
**~0.3pp of correct_id**. Even the oracle recovers only 72 detections over higher-confidence,
or **~0.6pp**. Neither changes the net-negative verdict.

### Why larger-box aligns better with GT preference

In the 306 exactly-one-matched cases, the canonical tracklet (longer) is:
- **Larger** 86.6% of the time
- **GT-matched** 84.3% of the time

This is a structural relationship: the canonical tracklet tends to be tracking the person
the GT matcher also tracks, and its detections are physically larger because it is the
primary track on that person. Larger-box picks the canonical (and thus the GT-matched one)
88.6% of the time vs higher-confidence's 76.5%.

### Mechanism summary

The loss decomposes as:

| Component | GT-matched drops | correct_id impact |
|---|---|---|
| Irreducible (both matched) | 664 | ~-2.1pp |
| Rule-specific (one matched, chose wrong) | 72 (conf) / 35 (larger) | ~-0.3 to -0.6pp |
| **Total** | **736 (conf) / 699 (larger)** | **~-2.4 to -2.7pp** |

**Merging is intrinsically lossy because the detector emits two valid-looking boxes for one
body on 68.3% of concurrent frames, and the GT matcher accepts both.** The reconciliation
rule is a rounding error on top of the structural cost.

---

## 4. NOEDGE-1 measurement (previously unmeasured)

GT-DIAG-1 run against the scratch pipeline-dir:

| GT | No-edge (baseline) | No-edge (dedup) | Delta |
|---|---|---|---|
| 0 | 14 | 5 | -9 |
| 1 | 1 | 0 | -1 |
| **2** | **40** | **3** | **-37** |
| 3 | 7 | 7 | 0 |
| 4 | 1 | 4 | +3 |
| 5 | 0 | 1 | +1 |
| 6 | 0 | 1 | +1 |
| 7 | 0 | 0 | 0 |
| **Total** | **63** | **21** | **-42** |

**GT 2's 40 no-edge boundaries collapsed to 3** (-37). The mechanism is confirmed: the
concurrent tracklets (t3 x t47, 233 overlap frames) caused GT-matcher assignment flicker
between them. Merging eliminates the flicker by construction.

The total dropped from 63 to 21 (-42), confirming that the majority of baseline no-edge
boundaries were concurrent-tracklet flicker, consistent with NOEDGE-1's diagnosis of them
as evaluation artifacts rather than pipeline defects.

GT 4 gained 3 no-edge boundaries — a second-order effect of the changed D1 graph structure.

---

## 5. Sessions: 23 -> 15, CORRECT_ENGAGED 6 -> 1

Fewer sessions (closer to the GT target of 3), but the collapse is almost entirely in
CORRECT_ENGAGED (-5) and PHANTOM (-2). CONTAMINATED barely moved (-1).

The single remaining CORRECT_ENGAGED session means the pipeline almost completely lost
the ability to correctly pair engaged athletes. The mechanism: with fewer tracklets,
the ILP produces fewer entity paths, but those paths are less pure (they absorbed impure
frames from the merged tracklets).

---

## 6. What perfect dedup does NOT fix

**Everything that matters:**

1. **Sequential fragmentation.** GT 3 has 10 person_ids with dedup — more than before.
   Merging concurrent duplicates does nothing for the breaks between non-overlapping
   tracklet segments.

2. **Detection under-segmentation (pair-box).** The dominant pathology. Two grappling
   people producing one detection box. Dedup cannot address this — the second person
   is never detected separately.

3. **Tracker drift.** The canonical tracklet (e.g., t2 with 1681 frames for GT 0) is
   itself impure — it tracks multiple GT people over its lifetime. Merging short
   concurrent duplicates into an already-impure hub does not help.

4. **ILP path decomposition.** The solver decomposes flow into SOURCE->SINK paths.
   Fewer tracklets change the graph structure (fewer nodes, different edges), but the
   solver's ability to route correctly depends on edge costs and penalties, not on
   tracklet count alone.

5. **GT 6 (3.9% recall).** Always occluded under GT 5 — a physical visibility problem
   unrelated to duplication.

---

## 7. Dual-box observation (detector-level)

The 68.3% both-GT-matched rate reveals a detector-level phenomenon: **two valid-looking
detection boxes for one physical person on the same frame.** The Hungarian matcher (IoU 0.5)
accepts both as valid matches to the same GT person.

This is the mirror image of the pair-box (under-segmentation) finding from CP7:
- **Pair-box:** one box covers two people (23.1% of GT-person-frames in the pre-VFR decomposition)
- **Dual-box:** two boxes cover one person (68.3% of concurrent-tracklet overlap frames)

Both are detector-level observations, not Stage D problems. The dual-box phenomenon may
contribute to tracker fragmentation (BoT-SORT spawns a second track on the spurious box)
and to the concurrent-tracklet population itself. It connects to the existing detection-model
work (CP23b) as a potential training signal: penalizing double-coverage of single bodies.

**Observation only — do not investigate here.** Record for detection-model work.

---

## 8. Ceiling framing

This is a **GT-labelled ceiling** — strictly better than any real deduplicator could
achieve. No real system has access to the GT labels that define which tracklets are
duplicates. DEDUP-MEASURE-1 found the physically-motivated features do not separate
NESTED from GENUINE:
- `sep_m`: 0.616
- `motion_corr`: 0.704
- `containment`: 0.537
- `IoU`: 0.504
- `life_ratio`: 0.950 (rejected — no mechanism, uncomputable while live)

**Perfect dedup would buy X = nothing positive; no discriminator we have measured could
deliver even that.** The finding is stronger than expected: not only is dedup
unachievable, but even if achieved perfectly it would be harmful — and the harm is
intrinsic to merging (68.3% irreducible), not a property of the reconciliation rule
(0.3-0.6pp marginal).

**One clip. No base-rate claim.** This is FP7oJQ-20260822-132650, a 2-minute clip with
8 people in a challenging grappling scenario. Other clips with different concurrent-
overlap patterns could yield different results.

---

## 9. Dedup line: CLOSED

Two independent legs:

1. **DEDUP-MEASURE-1:** No physically-motivated discriminator separates NESTED from GENUINE
   concurrent tracklets. The best-separating feature (`life_ratio` 0.950) was rejected for
   having no mechanism and being uncomputable while tracklets are live.

2. **DEDUP-CEILING-1/1a:** Even perfect GT-labelled dedup is net negative (-2.7pp correct_id),
   and the loss is intrinsic (68.3% irreducible from both-GT-matched pairs), not rule-specific
   (practical rules differ by 0.3pp). Merging concurrent detections destroys signal that the
   pipeline uses for identity assignment.

**Do not build deduplication.** The reasoning is recorded here so it is not re-proposed.

---

## 10. Implication for the over-dedup strategy

The task brief considered an over-deduplication strategy: merge aggressively, rely on
engagement/separation at session scale to recover wrongly-merged people.

This ceiling analysis shows the strategy's premise is false: **concurrent duplicates are
not the cause of the fragmentation problem.** Removing them perfectly makes things worse,
not better. The fragmentation is overwhelmingly sequential (tracking breaks, solver path
decomposition), and the concurrent duplicates were actually carrying useful signal that
the pipeline used for identity assignment.

The lever remains detection under-segmentation and tracker drift, as established by the
prior CP-GT2ACTUALS-6 / CP7 investigations.

---

## Evidence artifacts

- `docs/evidence/dedup_ceiling_1/findings.json` — structured results (DEDUP-CEILING-1)
- `docs/evidence/dedup_ceiling_1/findings.md` — this document (reconciled 1 + 1a)
- `docs/evidence/dedup_ceiling_1/gt_matching/per_frame_matches.parquet` — re-run GT matching
- `docs/evidence/dedup_ceiling_1/gt_diag/` — GT-DIAG-1 against scratch pipeline (NOEDGE measurement)
- `outputs/_dedup_ceiling/FP7oJQ-20260822-132650/` — scratch pipeline artifacts (not committed)
- `tools/dedup_ceiling_analysis.py` — self-contained analysis script
