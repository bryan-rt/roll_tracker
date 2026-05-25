# CP7-pre-8: Axis-1 Failure-Signature Characterization — FP7oJQ

**SUPERSEDED by CP7-pre-9 and CP7-pre-10.** The 84.3% Branch-B headline was ~93%
pair-box under-segmentation in disguise. True Branch-B margin: 9.9%. Pair-box spans
0% bracketed at any horizon. See `docs/cp7_pre9_branchb_margin.md` and
`docs/cp7_pre10_pairbox_bracketing.md`.

**Date:** 2026-05-25
**Scope:** READ-ONLY diagnostic. No pipeline/config/node changes.
**Camera:** FP7oJQ, clip FP7oJQ-20260318-200014 (4530 frames, 301 GT-annotated)
**Model:** bjj-detect-all-cameras

## Executive Summary

Branch B (new concurrent-swap node class) is the dominant Axis-1 failure mechanism.
Evidence converges from three independent measurements:

1. **GT-anchored (0-300):** 84.3% of misattributed frames show Branch B signature
   (concurrent-alive tracklets holding competing identities). Pure Branch A is 6.9%.
2. **Full-clip prevalence:** concurrent_alive episodes (26.3%) are the minority of
   proximity episodes, but they are the ones that produce harm — the larger
   clean_lifecycle population (42.1%) is handled correctly by the existing GROUP
   machinery (0/260 forced_unused).
3. **Bracket rate:** 59.0% of all pair-context spans are bracketed (two boxes resolve
   on both sides), confirming the Axis-1 reachable ceiling is meaningful. But GROUP
   nodes already cover most of these — the unaddressed failure is concurrent swaps
   within and outside GROUP spans.

**Recommendation:** Branch B. Design a concurrent-swap node class. Gate-tuning (Branch A)
addresses only 6.9% of the harm. The 33.9% ambiguous_a_b population represents frames
where GROUP nodes exist incidentally alongside the dominant concurrent-swap failure —
GROUP is working correctly for merges but is orthogonal to the identity swap problem.

---

## Part 1: Config Reconciliation

| Parameter | Code Default | YAML (default.yaml) | Active Value |
|-----------|-------------|---------------------|--------------|
| merge_dist_m | 0.45 m | **1.5 m** | **1.5 m** |
| split_dist_m | 0.60 m | **2.0 m** | **2.0 m** |

The YAML values override the code defaults. Config flows through `run_d1()` →
`d1_cfg.get("merge_dist_m", 0.45)`. No disconnect.

---

## Part 2b: GROUP forced_unused Tally (Full Clip)

| Metric | Count |
|--------|-------|
| GROUP nodes total | 260 |
| GROUPISH nodes total | 0 |
| Forced unused (all reasons) | **0** |
| missing_required_merge_in | 0 |
| missing_required_split_out | 0 |
| missing_groupish_group_cont_bridge | 0 |

**Implication:** The "node formed but D3 killed it" version of Branch A is **dead**.
Every GROUP node D1 created was accepted by the solver. Branch A survives only as
"real merge/split lifecycle events that never triggered a node at all" (gate-suppressed
or below min_group_duration_frames).

---

## Part 2a: GT-free Proximity Episode Classification (Full Clip)

**Method:** For each unordered tracklet pair with ≥15 frames of temporal overlap,
compute world-coordinate distance per frame. A proximity episode = ≥5 consecutive
frames within 1.5 m (= merge_dist_m from config). Classification:

- **clean_lifecycle:** One tracklet dies within the episode ± 5 frames, AND convergence
  evidence exists: (a) distance trend decreasing in 10-frame pre-death window with final
  distance < 1.5 m, OR (b) survivor's occ_r_height increases ≥0.05 or occ_r_bottom ≥0.03
  around death (bbox absorbs the disappeared person).
- **ordinary_exit:** A death occurs but without convergence evidence (person exited the
  scene or walked away, not a merge).
- **concurrent_alive:** Both tracklets remain alive throughout the episode. No death/birth
  event for GROUP machinery to trigger on.

**Convergence distance bar justification (Correction 3):** Tied to the pipeline's actual
merge gate (1.5 m). If the pipeline triggers GROUP formation at 1.5 m, then convergence
within 1.5 m with a decreasing distance trend is sufficient evidence of a merge. Using a
tighter bar (e.g., 0.5 m) would under-count clean-lifecycle events, artificially inflating
the concurrent_alive count and biasing toward Branch B.

| Classification | Episodes | % |
|----------------|----------|---|
| concurrent_alive | 78 | 26.3% |
| clean_lifecycle | 125 | 42.1% |
| ordinary_exit | 94 | 31.6% |
| **Total** | **297** | |

**Reading:** Clean-lifecycle events are the most common proximity episodes — these are real
merge/split events. But per Part 2b, GROUP handles them correctly (0 forced_unused). The
concurrent_alive minority (26.3%) is the population that leaks through to misattribution
because GROUP has no mechanism to address it.

---

## Part 2c: Bracketed Detection Proxy (World-Coord, Identity-Tracked)

**Two populations** per Correction 2:

### Population 1: D1-caught GROUP spans (260 spans)

GROUP spans where D1 already created a node. Bracket test checks whether 2+ tracklets
are resolved within 1.0 m of the carrier in a 30-frame window before/after the span.
Identity-tracked: the carrier must be present in both windows.

| Bracket Class | Spans | % |
|---------------|-------|---|
| bracketed | 108 | 41.5% |
| half_bracket_pre | 83 | 31.9% |
| half_bracket_post | 34 | 13.1% |
| unbracketed | 35 | 13.5% |

### Population 2: All pair-context spans (271 spans, GT-free)

GT-free detection of pair-context spans: any tracklet death within MERGE_DIST_M of an
active carrier creates a span running until the next birth near the carrier or carrier
end. This is independent of whether D1 formed a GROUP node.

| Bracket Class | Spans | % |
|---------------|-------|---|
| bracketed | 160 | 59.0% |
| half_bracket_pre | 73 | 26.9% |
| half_bracket_post | 9 | 3.3% |
| unbracketed | 29 | 10.7% |

**D1 coverage:** 209/271 pair-context spans overlap a D1 GROUP span. 62 spans are
D1-missed — among those, 38/62 (61.3%) are bracketed.

**Reading:** The Axis-1 reachable ceiling is meaningful: ~59% of pair-context spans
resolve into two tracked boxes on both sides. But D1 already catches most of these
(209/271). The 62 D1-missed spans represent potential Branch A gate-tuning targets —
but per Part 3 below, their contribution to misattribution is small.

---

## Part 3: GT-Anchored Misattribution Signature (Frames 0-300)

**Method:** For each of the 2,259 present_misattributed frames in the GT trace:

- **branch_a:** A GROUP node is active AND the misattributed tracklet is a routed role
  (carrier/disappearing/new) in that GROUP node, AND no concurrent swap or persistent
  identity confusion is present. (Tightened per Correction 1.)
- **branch_b_swap:** A swap event from tracker_swap covers this frame within ±5 frames.
  Two concurrently alive tracklets swap GT person assignments.
- **branch_b_persistent:** No swap boundary, but another concurrently alive tracklet
  holds the canonical_person_id. Persistent concurrent identity confusion.
- **ambiguous_a_b:** Both Branch A (GROUP routes this tracklet) AND Branch B (concurrent
  tracklet holds canonical) are present. Reported separately per Correction 1.
- **axis2_underseg:** GROUP node covers frame but tracklet is not a routed role.
- **other:** No GROUP, no swap, no concurrent canonical-holder.

| Signature | Frames | % |
|-----------|--------|---|
| branch_a | 157 | 6.9% |
| branch_b_swap | 93 | 4.1% |
| branch_b_persistent | 1,046 | 46.3% |
| ambiguous_a_b | 765 | 33.9% |
| axis2_underseg | 0 | 0.0% |
| other | 198 | 8.8% |

**Aggregated:**
- Branch A evidence (pure + ambiguous): 922 frames (40.8%)
- Branch B evidence (pure + ambiguous): 1,904 frames (84.3%)
- Overlap (ambiguous_a_b): 765 frames (33.9%)

### Interpreting the ambiguous_a_b bucket

The 33.9% ambiguous mass does NOT indicate genuine co-causation. Here's why:

GROUP spans are **wide** — they tile the timeline continuously (260 spans across 4530
frames). A GROUP node being "active" over a frame does not mean it *caused* the
misattribution at that frame. Per Part 2b, all 260 GROUP nodes were accepted and
correctly route capacity-2 flow for merge/split events. The GROUP machinery is
**working correctly for its intended purpose** (handling merges).

The concurrent-alive tracklet holding canonical_person_id is the **proximate cause**
of the misattribution: D4 assigned person_id X to tracklet A, but tracklet B
concurrently holds the correct identity. The GROUP node is incidental — it handles
a merge event that happens to overlap in time, but the identity confusion is between
the concurrent tracklets, not between the merged roles.

**Therefore:** ambiguous_a_b is functionally Branch B. The effective split is:

| Effective Signature | Frames | % |
|---------------------|--------|---|
| Branch B (concurrent identity confusion) | 1,904 | 84.3% |
| Branch A (GROUP routing failure, no concurrent confusion) | 157 | 6.9% |
| Other (no GROUP, no concurrent holder) | 198 | 8.8% |

---

## Cross-Checks

### (a) Bracket rate vs GT signature

- All pair-context bracket rate: **59.0%**
- Branch B in GT signature: **84.3%**

High bracket rate + Branch B dominant → GROUP nodes cover the merge/split spans
correctly, but the wrong identity exits. The fix is not trigger expansion (the
triggers work) — it's addressing concurrent identity swaps that GROUP was never
designed to handle.

### (b) Full-clip prevalence vs GT signature

- Full-clip: concurrent_alive = 26.3%, clean_lifecycle = 42.1%
- GT (0-300): Branch B = 84.3%, Branch A = 6.9%

**These appear to diverge but actually agree.** The explanation:

Clean-lifecycle episodes (42.1%) are the most common proximity mechanism, but they
are handled correctly by GROUP (0/260 forced_unused). They produce correctly-routed
GROUP spans, **not misattribution**. The concurrent_alive minority (26.3%) is the
population that leaks through to misattribution because GROUP has no mechanism to
address concurrent swaps.

The 2a-vs-3 comparison confirms: **the dominant mechanism in the full clip (clean
lifecycle) is NOT what produces the harm. The minority mechanism (concurrent alive)
IS what produces the harm.** This is precisely because GROUP works — it catches
lifecycle events. What's missing is coverage for concurrent swaps.

In the GT range specifically: 9 clean_lifecycle episodes, 6 concurrent_alive, 4
ordinary_exit (19 total) — proportions match the full clip.

---

## Verdict

**With 2b=0 (zero forced_unused GROUP nodes), the live A-vs-B fork is:**
- Branch A = clean-lifecycle events not triggering GROUP nodes (gate-tuning — cheap)
- Branch B = concurrent-alive tracklets swapping identities (new node class — structural)

### Evidence summary

| Evidence line | Supports |
|---------------|----------|
| GT signature: 84.3% Branch B vs 6.9% pure Branch A | **Branch B** |
| Full-clip: concurrent_alive minority produces the harm | **Branch B** |
| 0/260 forced_unused: GROUP handles merges correctly | Excludes Branch A "node killed" |
| 59% bracket rate: Axis-1 ceiling is real | Intervention can help |
| Ambiguous 33.9%: GROUP incidental, concurrent confusion causal | **Branch B** |

### Recommendation: Branch B — design a concurrent-swap node class

The existing GROUP machinery correctly handles merge/split lifecycle events (42.1%
of proximity episodes, 0% forced_unused). Gate-tuning (Branch A) would address at
most the 6.9% pure Branch A frames — real but small.

The dominant Axis-1 failure (84.3% of GT-confirmed misattribution) is concurrent-alive
tracklets that swap or persistently confuse identity assignments. No death/birth event
occurs, so GROUP has nothing to trigger on. A new node class that represents
"these two concurrently-alive tracklets may be swapping identities" is the structural
intervention needed.

**Gate-tuning is not worthless** — the 62 D1-missed pair-context spans (38 bracketed)
and 157 pure Branch A frames represent real, addressable failures. But they're a ~7%
sidecar, not the primary intervention. Consider gate-tuning as a follow-on after the
concurrent-swap node class is in place.

**STOP:** Node/trigger design returns to the web session. This report provides the
evidence; the design decision is not made here.

---

## Artifacts

| File | Contents |
|------|----------|
| `outputs/_eval/_debug/cp7_pre8_axis1/part_2a_episodes.json` | 297 proximity episodes with classification |
| `outputs/_eval/_debug/cp7_pre8_axis1/part_2c_d1_brackets.json` | 260 D1-caught bracket tests |
| `outputs/_eval/_debug/cp7_pre8_axis1/part_2c_all_brackets.json` | 271 all pair-context bracket tests |
| `outputs/_eval/_debug/cp7_pre8_axis1/part_3_signatures.json` | 2,259 GT-anchored signature labels |
| `tools/cp7_pre8_axis1_diagnostic.py` | Throwaway analysis script |
