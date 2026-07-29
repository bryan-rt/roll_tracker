# CP7-pre-10: Bracketed vs Unbracketed Split of the Pair-Box Misattribution Mass

**Date:** 2026-05-25
**Scope:** READ-ONLY diagnostic. No pipeline/config/node changes.
**Camera:** FP7oJQ, clip FP7oJQ-20260318-200014 (4530 frames, 301 GT-annotated)
**Model:** bjj-detect-all-cameras
**Depends on:** CP7-pre-9 (pair_box classification), CP7-pre-8 (bracket method)

## Executive Summary

Zero pair-box spans are bracketed at any horizon — from 1 second to full clip (~2.5 min).
The grappling pairs that produce 74.4% of misattribution never separate into two
individually-tracked boxes anywhere in the session. The GROUP/offline-propagation path
has no two-box anchors to propagate from. Detection separation is unambiguously the
primary lever.

---

## Method

### Pair-span construction

The 1,681 pair_box frames from CP7-pre-9 (each: one detection covering two GT people)
are grouped by `(carrier_tracklet_id, contained_gt_track_id)` and collapsed into
contiguous spans (gaps > 5 frames split into separate spans).

| Metric | Count |
|--------|-------|
| pair_box frames | 1,681 |
| Unique (carrier, contained_gt) pairs | 32 |
| Total spans | 117 |
| Normal spans (>= 5 frames) | 61 (1,560 frames) |
| Short spans (< 5 frames) | 56 (121 frames) |

### Bracket test (reused from CP7-pre-8 Part 2c, amended)

For each span, search for two-box resolution in pre/post windows:

- **Window sizes:** Sweep over {30, 90, 300, 4530} frames (~1s, ~3s, ~10s, full clip)
- **Radius:** BRACKET_RADIUS_M = 1.0 m (world coordinates)
- **Identity-tracked:** Carrier tracklet must be present in both windows. Carrier
  tracklet IDs are resolved to their D0.5 split fragments via a fragment map
  (`t10` → `t10`, `t10_s1`, ..., `t10_s8`) to account for the post-split tracklet
  population in `tracklet_bank_frames`.
- **Distinct-GT-person guard:** Resolution requires tracklets attributed to BOTH the
  matched GT person AND the contained GT person (from pre-9's `contained_gt_track_id`).
  Two fragments of the same person do NOT count. The contained GT person's tracklet
  must be a non-carrier tracklet (carrier fragments are excluded from the contained
  person check).
- **GT attribution:** Majority-vote from frozen CP-EVAL-1 gt_person_trace (same mapping
  as pre-8/pre-9). Split fragments inherit their base tracklet's attribution.
- **Trust boundary:** Each resolution tagged as trusted (window within 0-300) or
  untrusted (pipeline-only attribution beyond GT range).
- **Stayed-apart check:** For bracketed spans, verifies both GT persons remain
  separately tracked through the last 30% of the window (not re-merged).

### Classification

| Class | Definition |
|-------|-----------|
| **bracketed** | Both pre AND post windows resolve into two distinct GT persons |
| **half_bracket_pre** | Only pre-window resolves |
| **half_bracket_post** | Only post-window resolves |
| **unbracketed** | Neither window resolves |
| **indeterminate** | Carrier absent from one or both windows |

---

## Results: Horizon Curve

### By spans (normal only, >= 5 frames)

| Horizon | Bracketed | Half-brkt | Unbracketed | Indeterminate | Total |
|---------|-----------|-----------|-------------|---------------|-------|
| 30f (~1s) | **0** (0.0%) | 0 | 53 (86.9%) | 8 (13.1%) | 61 |
| 90f (~3s) | **0** (0.0%) | 2 (3.3%) | 51 (83.6%) | 8 (13.1%) | 61 |
| 300f (~10s) | **0** (0.0%) | 3 (4.9%) | 50 (82.0%) | 8 (13.1%) | 61 |
| Full clip | **0** (0.0%) | 3 (4.9%) | 50 (82.0%) | 8 (13.1%) | 61 |

### By frames (normal spans only)

| Horizon | Bracketed | Half-brkt | Unbracketed | Indeterminate | %pb brkt | %2259 brkt |
|---------|-----------|-----------|-------------|---------------|----------|------------|
| 30f (~1s) | **0** | 0 | 1,298 | 262 | 0.0% | 0.0% |
| 90f (~3s) | **0** | 29 | 1,269 | 262 | 0.0% | 0.0% |
| 300f (~10s) | **0** | 39 | 1,259 | 262 | 0.0% | 0.0% |
| Full clip | **0** | 39 | 1,259 | 262 | 0.0% | 0.0% |

### Curve shape: flat at zero

The bracketed share is **exactly zero at every horizon**. No climb. No signal.
The curve is not "flat-and-low" — it is flat-and-absent. Even with the full 4,530-frame
clip available as search space, no pair-box span has the specific two-person resolution
(matched GT + contained GT, separately tracked) on both sides.

The half-bracket growth (0 → 3 spans, 0 → 39 frames from 30f to full clip) shows that
some pairs do resolve on one side at wider horizons. But without both-side resolution,
there is no propagation path.

---

## Defensible Bracket Share

| Horizon | Trusted + stayed-apart | Trusted + re-merged | Untrusted | %pb clean |
|---------|----------------------|--------------------|-----------|-----------|
| 30f | 0 | 0 | 0 | 0.0% |
| 90f | 0 | 0 | 0 | 0.0% |
| 300f | 0 | 0 | 0 | 0.0% |
| Full clip | 0 | 0 | 0 | 0.0% |

Moot — no bracketed spans exist to decompose.

---

## Gap Bridges

52 gaps (6-20 frames) between adjacent same-pair spans were checked for two-GT-person
resolution. **Zero resolved.** No hidden bracket events obscured by span splitting.

---

## Indeterminate Analysis

8 normal spans (262 frames, 13.1%) are indeterminate — the carrier tracklet (or its
split fragments) is absent from one or both bracket windows. These are spans where the
carrier tracklet starts at frame 0 (clip boundary, no pre-window possible) or ends
before the span's post-window. They are NOT folded into either the bracketed or
unbracketed count.

Even if all 8 indeterminate spans were hypothetically bracketed, the bracketed share
would be 8/61 = 13.1% of spans (262/1,681 = 15.6% of pair_box frames). This is the
absolute upper bound, and it requires assuming every indeterminate is bracketed — an
implausible best case.

---

## Final Recovery-Path Split

| Recovery path | Frames (30f) | Frames (full) | %pb (full) | %2259 (full) |
|---|---|---|---|---|
| **Propagation-recoverable (bracketed)** | **0** | **0** | **0.0%** | **0.0%** |
| One-side anchor (half-bracket) | 0 | 39 | 2.3% | 1.7% |
| **Detection-only (unbracketed)** | **1,298** | **1,259** | **74.9%** | **55.7%** |
| Indeterminate | 262 | 262 | 15.6% | 11.6% |
| Short spans (all classes) | 121 | 121 | 7.2% | 5.4% |

---

## Cross-Reference: Updated Misattribution Hierarchy

Combining pre-9 + pre-10 results for the full FP7oJQ misattribution picture:

| Cause | Frames | % of 2,259 | Fix path | Lever size |
|-------|--------|------------|----------|------------|
| **Pair-box, unbracketed** | **1,259** | **55.7%** | Detection separation only | Primary |
| **Pair-box, indeterminate** | **262** | **11.6%** | Likely detection (upper bound 15.6% bracketed) | Likely primary |
| **Pair-box, half-bracket** | **39** | **1.7%** | Partial propagation possible | Marginal |
| **Pair-box, short spans** | **121** | **5.4%** | Mixed | Minor |
| True Branch B (Axis-1) | 223 | 9.9% | Concurrent-swap node | Secondary |
| Pure Branch A | 157 | 6.9% | GROUP routing | Minor |
| Other | 198 | 8.8% | Investigation needed | Unknown |

---

## Caveats

1. **Attribution circularity:** The bracket test uses pipeline-derived GT attribution
   (majority-vote from gt_person_trace). This is most reliable at separation points
   (isolated boxes) — precisely the events being tested. The lean is benign: if
   attribution were wrong at a separation point, the separated boxes would not match
   the specific (matched_gt, contained_gt) pair, and the test would correctly return
   unresolved. False positives are unlikely; false negatives are possible but would
   only make the zero-bracket result more conservative.

2. **Single-clip limitation:** This analysis covers one clip (~2.5 min) from one camera.
   Longer sessions or different camera angles might show different pair separation
   patterns. However, the finding is consistent with the physical reality: grappling
   pairs in BJJ remain in contact for extended periods (rounds are typically 5-6 min),
   and the detector's pair-box failure is structural, not transient.

3. **D0.5 fragment resolution:** The original run (pre-fragment-fix) showed 39.3%
   indeterminate due to carrier tracklet ID mismatch between gt_person_trace (original
   IDs) and tracklet_bank_frames (post-split IDs). After building a fragment map to
   resolve `t10` → `t10_s3` etc., indeterminate dropped to 13.1% (clip-boundary cases
   only). The zero-bracket finding is robust to this fix.

---

## Verdict

**The GROUP/offline-propagation path is dead for pair-box recovery.**

Zero pair-box spans are bracketed at any horizon. The grappling pairs that produce
74.4% of misattribution never resolve into two individually-tracked boxes — not within
1 second, not within 10 seconds, not within the entire clip. There are no two-box
anchors from which to propagate identity through the merged span.

**Detection separation is unambiguously the primary lever.** The pair-box mass (74.4%
of misattribution, 55.7% confirmed unbracketed + 11.6% indeterminate + 5.4% short
spans) is recoverable only by producing two separate detection boxes for each grappling
pair — whether via a better detector, instance segmentation, or post-detection pair
splitting.

The true Branch-B share (9.9%, from pre-9) plus pure Branch-A (6.9%) remain as
secondary Stage D targets totaling ~17% of misattribution. The concurrent-swap node
class recommended by pre-8 addresses real failures but at ~10%, not 84%.

**STOP.** Detection-vs-propagation prioritization returns to the web session.

---

## Artifacts

| File | Contents |
|------|----------|
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/pair_spans.json` | 117 collapsed pair-spans |
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/brackets_h30.json` | Bracket results at 30-frame horizon |
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/brackets_h90.json` | Bracket results at 90-frame horizon |
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/brackets_h300.json` | Bracket results at 300-frame horizon |
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/brackets_h4530.json` | Bracket results at full-clip horizon |
| `outputs/_eval/_debug/cp7_pre10_pairbox_bracketing/gap_bridges.json` | 52 gap-bridge checks |
| `tools/cp7_pre10_pairbox_bracketing.py` | Throwaway analysis script |
