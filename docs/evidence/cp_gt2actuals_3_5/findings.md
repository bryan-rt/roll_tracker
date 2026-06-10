# CP-GT2ACTUALS-3.5: Signal-Trace Family-Lookup Gap Measurement

**Date:** 2026-06-10

## 1. Bug Confirmed in signal_trace

`stage_d_trace.py` has the SAME single-resolution lookup in TWO places:
- `_compute_dominant_person_ids` (line 160-161): canonical derivation
- `run_d_trace` (line 235-236): per-frame classification

Both use `pids = pt_lookup.get((resolved_tid, fi), [])` without the
family-aware fallback that CP-GT2ACTUALS-3 fixed in dense_join.

**Recovery rate (J_EDEw vid1, all annotated frames):**
- Total lookups: 3,615
- Single-resolution empty: 1,318 (36.5%)
- Family-recovered: 1,193 (33.0%)
- 5 of 14 canonical person_ids would change under family-aware lookup

## 2. Locked Canonical Numbers — NOT BIASED (timing protects them)

**Key finding: the locked d-trace (Jun 7 14:27) predates D0.5 splits (Jun 9).**

The locked numbers (40.5% baseline, 63.2% post-TAG-4a) were computed from a
pre-split pipeline state where `_resolve_tracklet_id` was a no-op (no split_map
entries). The bug only manifests when D0.5 splits exist AND the split products
in bank_summaries don't align with person_tracks.

**Timeline:**
- Jun 7 13:05: person_tracks written (with TAG-4a fixes, no D0.5 splits)
- Jun 7 13:42: CP-TAG-4a code committed
- Jun 7 14:27: signal_trace d-trace computed → 40.5% baseline, 63.2% post-TAG-4a
- Jun 9 19:46: D0.5 split audit written (CP-SPLIT-1 / CP-HSV-V pipeline re-run)
- Jun 9+: bank_summaries modified in-place with split products

**The locked 40.5% and 63.2% are CORRECT for the pipeline state they measured.**
They are not biased by the family-lookup gap because there were no splits at
measurement time.

## 3. HOWEVER: Current Pipeline State Produces Different Numbers

The CURRENT pipeline state has D0.5 splits (308 events on J_EDEw vid1). If
signal_trace were re-run today:

| Method | Val-split correct_id | vs locked 40.5% |
|--------|---------------------|-----------------|
| Single-resolution (buggy) | 20.7% | -19.8pp (WRONG) |
| Family-aware (fixed) | 33.9% | -6.6pp |
| Dense-join (family, stride-1) | 33.9% | -6.6pp |

The 40.5% → 33.9% drop is NOT a regression from the family fix. It reflects
the pipeline change: D0.5 splits (CP-SPLIT-1) + CP-HSV-V introduced more
split products, and some of those splits are spurious (CP-SPLIT-VALIDATE
found Tier 3 is 2.4% precision). The spurious splits fragment tracklets,
reducing correct_id.

**The +22.7pp CP-TAG-4a improvement (40.5% → 63.2%) was measured on the
pre-split pipeline and is valid FOR THAT STATE.** The improvement direction
certainly holds on the current state too, but the absolute numbers would
differ. Re-measuring post-TAG-4a on the current pipeline state requires
re-running the pipeline with TAG-4a fixes ON the current D0.5-enabled config.

## 4. Dense-Join vs Signal-Trace Agreement

Under family-aware lookup on the same pipeline artifacts:
- Signal_trace (stride-10, family): 33.9% correct on val-split
- Dense-join (stride-1, family): 33.9% correct on val-split

**They agree exactly.** The dense-join artifact is consistent with signal_trace
when both use the same lookup method. No residual divergence to explain.

## 5. Blast Radius — Decisions Depending on Locked Canonical

The following recorded numbers/decisions cite the locked 40.5% or 63.2%:

| Location | Number | Status |
|----------|--------|--------|
| CLAUDE.md "CP-TAG-4a" row | 40.5% → 63.2% = +22.7pp | Valid for pre-split state |
| CLAUDE.md "Metric-basis discipline" | J_EDEw baseline ~40.5% | Valid for pre-split state |
| CLAUDE.md "CP-PURITY-2" row | +22.7pp reconciliation | Valid for pre-split state |
| memory/MEMORY.md "CP-TAG-4a" | +22.7pp | Valid for pre-split state |
| docs/evidence/cp_purity_2/m1_reconciliation.json | 40.5%/63.2% | Valid for pre-split state |
| docs/evidence/cp_tag_4_post/README.md | initial misroute verdicts | Valid for pre-split state |
| signal_trace outputs (J_EDEw d-trace) | 40.5% correct_id | STALE (pre-split pipeline) |

**IF the locked numbers are re-baselined on the current pipeline state:**
- New baseline would be ~33.9% (not 40.5%)
- Post-TAG-4a would need re-measurement (pipeline re-run required)
- The +22.7pp improvement claim would need rechecking on current state
- All decision-log rows citing 40.5%/63.2% would need "measured pre-split"
  annotation

**Recommendation (for web-session decision, not made here):**
1. The locked numbers are valid for their measurement context. No retroactive
   correction needed unless we want to re-lock on the current pipeline state.
2. signal_trace SHOULD get the family-aware fix (separate gated change) to
   prevent future measurements from hitting the bug when D0.5 splits exist.
3. Any future re-baselining should note the pipeline state (pre/post split)
   in the measurement metadata — the same metric-basis discipline we apply to
   camera set and frame range.
4. The signal_trace d-trace artifacts are STALE (pre-split) and should NOT be
   used for comparisons against current-state measurements. Use the dense-join
   artifact instead, which reflects the current pipeline state.
