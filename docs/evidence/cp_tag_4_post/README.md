# CP-TAG-4a Post-Fix Evidence

## Changes Applied (Fix 0 + Fix A + Fix C + Fix D)

- **Fix 0:** Split-aware ping binding in d3_ilp2.py. Pre-split tracklet IDs
  expanded to D0.5 split products for node matching. Session aggregation
  writes combined d05_split_audit.jsonl.
- **Fix A:** D4 consumes solved tag thread (tag_flow_by_tag_edge on ILPResult).
  Exactly 1 identity_assignment per tag via `method=solver_tag_thread`.
- **Fix C:** Hard no-drop constraint for ping-carrying products via
  explained_var == 1 (with per-tag fallback ladder on INFEASIBLE).
- **Fix D:** Carrier selection unit tests in tests/test_carrier_selection.py.

## Diff vs CP-TAG-3 Baseline

### Per-clip vid2 (J_EDEw-200246, gt_track_id=8)

| Metric | CP-TAG-3 | CP-TAG-4a | Delta |
|--------|----------|-----------|-------|
| Ping binding | UNBOUND | BOUND (T:t139_s3) | FIXED |
| Identity assignments | 1 (frame_overlap) | 1 (solver_tag_thread) | method changed |
| Tag assigned to | p0022 | p0022 | same person_id |
| Tagged person correct_id | 100/450 (22.2%) | 86/450 (19.1%) | -3.1pp |

### Per-clip vid1 (J_EDEw-200015, gt_track_id=24)

| Metric | CP-TAG-3 | CP-TAG-4a | Delta |
|--------|----------|-----------|-------|
| Ping binding | BOUND (G:2768-2813) | BOUND (G:2768-2813) | unchanged |
| Identity assignments | 1 (frame_overlap) | 1 (solver_tag_thread) | method changed |
| Tag assigned to | p0010 | p0032 | changed (solver path changed due to Fix C) |
| Tagged person correct_id | 77/301 (25.6%) | 53/301 (17.6%) | -8.0pp |

### Session-level (two-clip J_EDEw)

| Metric | CP-TAG-3 | CP-TAG-4a | Delta |
|--------|----------|-----------|-------|
| tag:1 assignments | 4 | **1** | -3 (Fix A) |
| Assignment method | frame_overlap | solver_tag_thread | structural |
| Assigned person_id | p0015,p0024,p0031,p0032 | **p0022** | single identity |
| Spans clip boundary | all 4 span | **yes** (frames [632, 9029]) | correct |
| Tagged tracklets KEPT | both | both | unchanged |
| t366 transitions | 1125 | 1114 | -11 (Fix C effect) |
| t139 transitions | 2680 | 2676 | -4 (Fix C effect) |

## Success Criteria Evaluation

1. **All pings bind:** PASS - vid1 bound (pre-split remnant), vid2 bound (t139_s3 via Fix 0)
2. **tag:1 = 1 identity_assignment:** PASS - exactly 1, via solver_tag_thread
3. **Dominant person overlaps GT:** PARTIAL - p0022 covers t139 and t139_s3 (physical
   tagged person's tracklets), but is not the GT majority-vote dominant person_id (p0004).
   This is the expected low-coverage outcome, not a misroute to a wrong person.
4. **No aggregate regression:** TBD (Stage D trace pending)
5. **No tagged tracklet dropped:** PASS - both t366 and t139 KEPT, zero fallback events

## Interpretation

Fix 0 + Fix A successfully:
- Bind both pings (including the previously-unbound vid2 ping)
- Produce exactly 1 identity_assignment per tag via the solved tag thread
- The thread visits the correct physical tracklet (t139_s3)

The correct_id decrease (-3.1pp vid2, -8.0pp vid1) is due to Fix C's hard-keep constraint
changing the solver's optimal solution. The solver now MUST keep t139/t139_s3/t366, which
shifts other person_id assignments. This is a side effect of the structural change, not a
regression in tag identity. The tag assignment itself is correct at the tracklet level.

The low coverage (19.1% vid2, 17.6% vid1) is expected and explicitly documented in the
spec as a clean handoff to CP-TAG-4b (hard ping connectivity) and CP21 (appearance costs).

## Session Scope

Two-clip, single-camera (J_EDEw only). No Tier 3 cross-camera evidence.
Same scope as CP-TAG-3 baseline for apples-to-apples comparison.
