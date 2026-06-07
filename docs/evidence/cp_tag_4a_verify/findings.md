# CP-TAG-4a-VERIFY: Measurement Findings

## Item 1: Aggregate Stage D Regression Gate

| Metric | CP-TAG-3 Baseline | CP-TAG-4a Post | Delta |
|--------|-------------------|----------------|-------|
| Aggregate correct_id | 58.7% (6,330/10,789) | 56.9% (6,140/10,789) | **-1.8pp** |

**PASS** (within 2pp threshold). Only J_EDEw changed (FP7oJQ/PPDmUg were not re-run).

## Item 2: GT-Grounded Coverage + Anchor-Correctness

### Vid1 (J_EDEw-200015, gt_track_id=24, tag assigned to p0032)

| Metric | Value |
|--------|-------|
| Coverage (GT frames carrying p0032) | **12/301 = 4.0%** |
| Anchor-correctness (p0032 frames matching GT) | **12/678 = 1.8%** |

**VERDICT: MISROUTE.** p0032 is overwhelmingly composed of other GT people's frames.
Only 1.8% of p0032's 678 frames correspond to the tagged person.

### Vid2 (J_EDEw-200246, gt_track_id=8, tag assigned to p0022)

| Metric | Value |
|--------|-------|
| Coverage (GT frames carrying p0022) | **29/450 = 6.4%** |
| Anchor-correctness (p0022 frames matching GT) | **29/2544 = 1.1%** |

**VERDICT: MISROUTE.** p0022 has 2544 total frames but only 29 match gt_track_id=8.
The entity path carries 98.9% other-person frames through GROUP dilution.

### Session-Level GT Fragmentation (vid2, gt_track_id=8)

The tagged athlete's GT frames are fragmented across 13 person_ids:

| Person_id | GT-matching frames | Note |
|-----------|-------------------|------|
| p0022 | 29 (25.2%) | tag-assigned |
| p0002 | 23 | |
| p0015 | 19 | |
| p0001 | 14 | |
| p0017 | 9 | |
| (8 others) | 1-5 each | |

p0022 is actually the **plurality winner** among person_ids covering the tagged GT
person — it has the most frames (29/115 = 25.2%). But its own entity path has 2544
total frames, so the tagged person is a tiny fraction of the entity.

## Item 3: Fix A vs Fix C Decomposition

### Vid1 (gt_track_id=24)

| Method | Person_id | Coverage | Anchor-corr | Person frames |
|--------|-----------|----------|-------------|---------------|
| Fix A (thread) | p0032 | 4.0% | 1.8% | 678 |
| Overlap (Fix A OFF, same solver) | p0020 | 5.6% | 0.6% | 3041 |
| CP-TAG-3 baseline (p0010, old solver) | p0010 | 6.3% | 5.8% | 327 |

**Decomposition:**
- Fix C (hard-keep changing solver): p0010 (baseline) → p0020 (overlap on new solver).
  p0010 had 5.8% anchor-correctness; p0020 has 0.6%. Fix C's solver change moved the
  overlap pick to a worse person.
- Fix A (thread vs overlap): p0020 (overlap) → p0032 (thread). Both are misrouted;
  thread pick has slightly higher anchor-correctness (1.8% vs 0.6%) but lower coverage
  (4.0% vs 5.6%).
- **Root cause of vid1 regression:** Fix C changed the solver optimum, which changed which
  person_ids exist and which tracklets they cover. The old p0010 (327 frames, 5.8% anchor)
  was a relatively focused entity; the new entities are much larger and diluted.

### Vid2 (gt_track_id=8)

| Method | Person_id | Coverage | Anchor-corr | Person frames |
|--------|-----------|----------|-------------|---------------|
| Fix A (thread) | p0022 | 6.4% | 1.1% | 2544 |
| Overlap (Fix A OFF) | p0022 | 6.4% | 1.1% | 2544 |

**No Fix A effect on vid2** — thread and overlap agree on p0022. The vid2 regression
(22.2% → 19.1%) is entirely from Fix C's solver change affecting the non-tag portion
of the d-trace (the tagged person's correct_id in the trace changed, but the tag
assignment itself is the same person under both methods).

## Recommendation

**CP-TAG-4a misroutes the tag on both clips.** The headline (tag:1 → 1 assignment
spanning the clip boundary) is correct structurally, but the identity it labels is wrong.

**Root cause:** GROUP dilution in D4 entity emission. The solver's tag thread correctly
visits the tagged physical tracklet (t139_s3, t366), but the entity path carrying that
thread is a massive multi-person chain (678-3041 frames per entity). Fix A binds the tag
to that entity's person_id, which is dominated by other people's frames. The entity
decomposition in D4 is the real bottleneck — not the tag thread, not the binding.

**Fix A needs a coverage-aware correction before CP-TAG-4b.** The thread correctly
identifies which entity visits the tagged tracklet, but that entity is too diluted for
the person_id to be meaningful. Options:
1. **Coverage-aware tiebreak:** When thread and overlap disagree, prefer the candidate
   with higher anchor-correctness (p0010 at 5.8% > p0032 at 1.8% for vid1).
2. **Restrict tag identity to tagged-tracklet frames only:** Instead of assigning the
   entity's person_id, emit a tag-specific identity that covers only the tagged
   tracklet's frames within the entity. This sidesteps GROUP dilution entirely.
3. **Defer to CP-TAG-4b + CP21:** Hard connectivity + appearance costs will shrink
   entity paths and improve anchor-correctness. But the current misroute means the
   4a headline is misleading — the "correct" cross-clip link is carrying the wrong
   person's label.

**Recommendation:** Option 1 (coverage-aware tiebreak) is a small surgical fix that
restores the baseline's anchor quality while keeping Fix 0/C/D. Plan it in the web
session before proceeding to CP-TAG-4b. The aggregate regression (-1.8pp) is within
threshold but should be re-checked after the tiebreak fix.
