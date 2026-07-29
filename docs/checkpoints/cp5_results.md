# CP5: Parallel-Carrier Consolidation in D1 -- Results

*Generated 2026-05-21. Model: bjj-detect-all-cameras. Full 6-mode trace, cp5_pre vs post-CP5.*

CP5 implements **pick-one consolidation**: when multiple carriers trigger for the same
merge/split event, D1 keeps exactly one carrier (deterministic tiebreak: lowest dist ->
longest tracklet -> lexicographic ID) and discards the rest. Non-chosen carriers continue
as SOLO chains. This preserves the D2/D3/D4 contract unchanged -- consolidated GROUP nodes
look identical to single-carrier GROUP nodes. CP4's `docs/checkpoints/cp4_flow_topology.md` body
recommends option (a) (capacity-2 merge node absorbing both carriers); CP5 chose pick-one
instead to avoid D2/D3/D4 contract changes.

---

## Section 1 -- Failure-Mode Shift (the headline)

Full-mode trace comparison, cp5_pre (baseline from CP6 run) vs post-CP5.

### FP7oJQ (4,214 GT-frame pairs)

| Mode | cp5_pre | post_cp5 | Delta |
|------|---------|----------|-------|
| present | 5.1% | 6.4% | +1.3pp |
| stage_a_no_detection | 9.9% | 12.2% | +2.3pp |
| stage_a_untracked | 8.1% | 10.7% | +2.6pp |
| d3_dropped | 24.0% | 4.6% | **-19.4pp** |
| d4_unassigned | 6.5% | 0.5% | -6.0pp |
| present_misattributed | 17.9% | 65.6% | +47.7pp |
| missing_canonical | 28.6% | 0.0% | -28.6pp |

### J_EDEw (4,214 GT-frame pairs)

| Mode | cp5_pre | post_cp5 | Delta |
|------|---------|----------|-------|
| present | 4.7% | 7.4% | +2.7pp |
| stage_a_no_detection | 11.7% | 11.7% | 0.0pp |
| stage_a_untracked | 13.5% | 13.5% | 0.0pp |
| d3_dropped | 49.7% | 7.9% | **-41.8pp** |
| d4_unassigned | 0.7% | 0.7% | 0.0pp |
| present_misattributed | 19.7% | 58.9% | +39.2pp |
| missing_canonical | 0.0% | 0.0% | 0.0pp |

### PPDmUg (2,361 GT-frame pairs)

| Mode | cp5_pre | post_cp5 | Delta |
|------|---------|----------|-------|
| present | 6.1% | 10.6% | +4.5pp |
| stage_a_no_detection | 13.3% | 13.4% | +0.2pp |
| stage_a_untracked | 7.2% | 8.1% | +0.9pp |
| d3_dropped | 39.9% | 0.0% | **-39.9pp** |
| d4_unassigned | 2.0% | 2.1% | +0.0pp |
| present_misattributed | 18.9% | 65.8% | +46.9pp |
| missing_canonical | 12.7% | 0.0% | -12.7pp |

### Interpretation

**d3_dropped collapsed** across all cameras: J_EDEw -41.8pp, PPDmUg -39.9pp (to zero),
FP7oJQ -19.4pp. This is the primary intended effect of parallel-carrier consolidation.

**present rose modestly**: +2.7pp J_EDEw, +4.5pp PPDmUg, +1.3pp FP7oJQ. Below the CP6
conservative ceilings (14%, 16%, undetermined) because:
1. Recovered frames are mostly `present_misattributed` -- the tracklets now route but
   each tracklet covers multiple GT persons, so only a fraction of frames align with the
   canonical person. This is the representation ceiling CP6 predicted.
2. `missing_canonical` collapsed to 0% (FP7oJQ -28.6pp, PPDmUg -12.7pp) -- GT tracks that
   previously had zero matched frames now have matches because more tracklets are routed.
   These newly-assignable GT tracks contribute mostly to misattributed, not present.

**present_misattributed is now the dominant failure mode** at 59-66% across all cameras,
up from 18-20%. This is the expected representation ceiling: tracklets span multiple
physical people, so a single person_id per tracklet is inherently wrong for most GT
persons it covers. Fixing this requires ReID/pose identity, not routing.

**Solver dropped tracklets**: FP7oJQ 37->4, J_EDEw 37->2, PPDmUg 14->0. Solver status:
OPTIMAL for all three cameras.

---

## Section 2 -- Consolidation Audit

### Per-camera consolidation counts

| Camera | Merge groups | Merge discarded | Split groups | Split discarded |
|--------|-------------|----------------|-------------|----------------|
| FP7oJQ | 111 | 136 | 114 | 149 |
| J_EDEw | 115 | 127 | 118 | 129 |
| PPDmUg | 34 | 36 | 37 | 47 |

### Multi-way events (3+ discarded carriers)

| Camera | Merge 3+ | Split 3+ |
|--------|----------|----------|
| FP7oJQ | 1 | 4 |
| J_EDEw | 2 | 0 |
| PPDmUg | 0 | 1 |

Multi-way consolidations are present as expected from CP6's finding of up to 12
simultaneous carriers. Most events are pairwise (2 carriers, 1 discarded), with a
handful of 3+-way events per camera.

### Sample merge consolidations

**FP7oJQ:**
- event=t1, frame=862: chosen=t61 (dist=0.072m, 1355 frames), discarded t63 (dist=0.514m, 1280 frames)
- event=t104, frame=1326: chosen=t102 (dist=0.064m, 111 frames), discarded t70 (dist=0.262m, 896 frames)

**J_EDEw:**
- event=t102, frame=896: chosen=t9 (dist=0.008m, 810 frames), discarded t99 (dist=0.446m, 15 frames)

**PPDmUg:**
- event=t101, frame=1904: chosen=t96 (dist=0.029m, 830 frames), discarded t47 (dist=0.361m, 1467 frames) + t97 (dist=0.750m, 651 frames) -- 3-way consolidation

---

## Section 3 -- Spot-Check Named Tracklets

### J_EDEw

| Tracklet | cp5_pre D3 | post_cp5 D3 | post_cp5 person_id | GT frames | present | misattrib |
|----------|-----------|-------------|--------------------|-----------|---------|-----------|
| t1 | dropped | **dropped** | -- | 293 | 0 | 0 |
| t2 | explained | explained | p0004 | 78 | 0 | 78 |
| t3 | dropped | **explained** | p0005 | 294 | 63 | 231 |
| t5 | dropped | **explained** | p0007/p0016 | 295 | 22 | 273 |
| t108 | explained | explained | p0019/p0022 | 199 | 46 | 153 |
| t111 | dropped | **explained** | p0012/p0024 | 201 | 36 | 165 |
| t201 | dropped | **explained** | p0006/p0013 | 57 | 13 | 44 |

**t1 is the sole remaining dropped long carrier for J_EDEw** (1 of 2 total dropped). The
solver still finds its routing cost prohibitive as a SOLO chain. t3, t5, t111, t201 all
flipped from dropped to explained -- the parallel-carrier displacement was the binding
constraint for these tracklets.

**t201 (tag-anchored, tag:1) is now explained.** This was a Tier 1 identity anchor that
was lost pre-CP5. It's now routed through the solver and assigned person_ids.

### PPDmUg

| Tracklet | cp5_pre D3 | post_cp5 D3 | post_cp5 person_id | GT frames | present | misattrib |
|----------|-----------|-------------|--------------------|-----------|---------|-----------|
| t1 | dropped | **explained** | p0003/p0017 | 293 | 55 | 238 |
| t2 | explained | explained | p0004/p0014 | 293 | 52 | 241 |

Both t1 and t2 are now explained. t1 flipped from dropped to explained. PPDmUg has zero
dropped tracklets post-CP5.

---

## Section 4 -- Verdict Against Expectations

| Metric | cp5_pre | Expected post-CP5 | Result | Pass? |
|--------|---------|-------------------|--------|-------|
| J_EDEw d3_dropped % | 49.7% | substantial drop (target <30%) | 7.9% | **YES** |
| PPDmUg d3_dropped % | 39.9% | substantial drop (target <25%) | 0.0% | **YES** |
| FP7oJQ d3_dropped % | 24.0% | modest drop (fragment-dominated) | 4.6% | **YES** |
| J_EDEw present % | 4.7% | rise toward 14% ceiling | 7.4% | PARTIAL |
| PPDmUg present % | 6.1% | rise toward 16% ceiling | 10.6% | PARTIAL |
| FP7oJQ present % | 5.1% | modest rise | 6.4% | PARTIAL |
| present_misattributed | 18-20% | may rise (acceptable) | 59-66% | EXPECTED |
| Mergers (person_ids with >1 GT track) | 2-4/cam | not significantly worse | 0-2/cam | **YES** |
| Solver status | OPTIMAL | OPTIMAL/FEASIBLE | OPTIMAL (all 3) | **YES** |
| Solver runtime | ~31 min (full eval) | < 3x | ~31 min (full eval) | **YES** |

### Overall verdict: CP5 SUCCEEDED

**d3_dropped collapsed to near-zero** (7.9% J_EDEw, 0.0% PPDmUg, 4.6% FP7oJQ), far
exceeding the <25-30% targets. The parallel-carrier consolidation hypothesis is validated:
removing duplicate GROUP nodes in D1 allows the solver to route previously-orphaned
carrier chains.

**present rose modestly but below the conservative ceiling.** This is expected: the CP6
analysis showed that the `present_misattributed` representation ceiling (~60% of frames)
would absorb most recovered frames. The recovered tracklets DO route, but because each
tracklet covers multiple GT persons, only a fraction of their frames contribute to
`present`. The present% is a lower bound on actual person-tracking quality; it's limited
by the GT-to-tracklet many-to-many relationship, not by routing.

**missing_canonical collapsed to 0% -- CP6's "GT property" claim was wrong.** CP6 Section 5
stated missing_canonical was "a GT annotation property, not a pipeline variable" because it
was stable across all five CP6 snapshots. CP5 disproved this: the four FP7oJQ GT tracks
(17, 20, 21, 27) had matched detections from Stage A, but those detections lived in dropped
tracklets with no person_id. Pre-CP5: frames_matched=0/301 for all four. Post-CP5:
frames_matched=264-301/301, purity 0.92-1.0. Same for PPDmUg gt_track_2 (0/300 -> 293/300).
The rate was stable across CP6 snapshots only because all had similar Stage D routing
(all pre-consolidation). CP6 Section 5 has been corrected.

**Mergers improved** (J_EDEw 4->2, PPDmUg 2->0, FP7oJQ stable at 2). More routed
tracklets did NOT cause merger explosions.

**Solver runtime:** No per-D3-stage timing exists in audit events. Full evaluate runtime
(pipeline A-E rerun + all eval stages) was ~31 min, comparable to cp5_pre. Solver achieved
OPTIMAL on all cameras without timeout. The graph is smaller post-consolidation (fewer
GROUP nodes), so the solver's problem is at least as easy.

**Next step: attack present_misattributed via ReID/identity.** This is now the dominant
failure mode (59-66%) and represents a fundamental representation problem: tracklets span
multiple physical people. Fixing this requires per-detection identity features (ReID
embeddings, pose signatures), not routing improvements. Likely CP7.
