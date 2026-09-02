# DEDUP-CEILING-1: What would perfect deduplication actually buy?

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**Method:** Approach (A) — full D0-D4 + Stage E re-run with merged tracklets
**Rule:** (c) >= 50% overlap fraction — merge only tracklets that are >=50% concurrent
with their canonical hub

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

**Perfect dedup buys nothing. It actively hurts.**

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

## 3. correct_id: 34.3% -> 31.6% (-2.7pp strict), 37.4% -> 33.5% (-3.9pp tolerant)

**Perfect dedup LOWERED correct_id.** The mechanism:

- **972 detections dropped** (higher-confidence-wins on overlap frames). Of these,
  **736 were GT-matched** (75.7%). Each dropped GT-matched detection is a frame where
  the GT person was detected but the detection was discarded by dedup.
- Only 63 new GT matches gained (the Hungarian matcher found better matches on some
  frames after removing competitors).
- Net: **-673 GT-matched detections**. This directly reduces the denominator's matched
  fraction, lowering correct_id.

The dedup is self-defeating: the "duplicate" detections were carrying GT signal. Removing
them removes signal.

---

## 4. Sessions: 23 -> 15, CORRECT_ENGAGED 6 -> 1

Fewer sessions (closer to the GT target of 3), but the collapse is almost entirely in
CORRECT_ENGAGED (-5) and PHANTOM (-2). CONTAMINATED barely moved (-1).

The single remaining CORRECT_ENGAGED session means the pipeline almost completely lost
the ability to correctly pair engaged athletes. The mechanism: with fewer tracklets,
the ILP produces fewer entity paths, but those paths are less pure (they absorbed impure
frames from the merged tracklets).

---

## 5. What perfect dedup does NOT fix

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

## 6. Dropped detection analysis

| Metric | Count |
|---|---|
| Total detections dropped | 972 |
| GT-matched dropped | 736 (75.7%) |
| Non-GT-matched dropped | 236 (24.3%) |
| New GT matches gained | 63 |
| Net GT-matched change | -673 |

The higher-confidence-wins reconciliation rule systematically drops the GT-matched
detection in 75.7% of cases. This is because the "duplicate" detection often has a
BETTER localization of the person (it was the one the Hungarian matcher preferred),
while the canonical tracklet's detection on that same frame is the one tracking a
different person through the pair-box area.

---

## 7. Ceiling framing

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
unachievable, but even if achieved perfectly it would be harmful.

**One clip. No base-rate claim.** This is FP7oJQ-20260822-132650, a 2-minute clip with
8 people in a challenging grappling scenario. Other clips with different concurrent-
overlap patterns could yield different results.

---

## 8. Implication for the over-dedup strategy

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

- `docs/evidence/dedup_ceiling_1/findings.json` — structured results
- `docs/evidence/dedup_ceiling_1/findings.md` — this document
- `docs/evidence/dedup_ceiling_1/gt_matching/per_frame_matches.parquet` — re-run GT matching
- `outputs/_dedup_ceiling/FP7oJQ-20260822-132650/` — scratch pipeline artifacts (not committed)
- `tools/dedup_ceiling_analysis.py` — self-contained analysis script
