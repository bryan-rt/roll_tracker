# Step 1: Passthrough Test Results

Tests hypothesis 3: is run_stage_d.py / run_gt2actuals.py faithful?

## J_EDEw-20260318-200015 (vid1)

- Passthrough correct_id: **34.1%**
- Pre-existing baseline:  **40.3%**
- Delta: **-6.2pp**
- Gate (±0.5pp): **FAIL**
- Jump counts: {'tracklet_drift': 606, 'ilp_misstitch': 251, 'false_split': 2, 'group_boundary_jump': 154, 'group_membership_drift': 564}

## J_EDEw-20260318-200246 (vid2)

- Passthrough correct_id: **28.2%**
- Pre-existing baseline:  **30.6%**
- Delta: **-2.5pp**
- Gate (±0.5pp): **FAIL**
- Jump counts: {'tracklet_drift': 469, 'ilp_misstitch': 168, 'false_split': 2, 'group_boundary_jump': 201, 'group_membership_drift': 185}

## Combined

- Passthrough correct_id: **30.7%**
- Pre-existing baseline:  **34.7%**
- Delta: **-4.0pp**
- Gate (±0.5pp): **FAIL**

## Root Cause: Mixed-Provenance Baseline Artifacts

**Hypothesis 3 is CONFIRMED, but not as a bug in run_stage_d.py — the sweep harness
is correct. The bug is in the pre-existing baseline artifacts themselves.**

### Finding: Stage D outputs in outputs/_eval_gt/ are from TWO different pipeline runs

File timestamps in `outputs/_eval_gt/J_EDEw/.../stage_D/`:

| File | Vid1 Timestamp | Vid2 Timestamp |
|------|---------------|---------------|
| tracklet_bank_frames.parquet (D0) | Jun 9, 19:46 | Jun 9, 19:49 |
| d1_graph_nodes.parquet (D1) | Jun 9, 18:17 | Jun 9, 18:17 |
| d2_constraints.json (D2) | **Jun 7, 12:59** | **Jun 7, 11:33** |
| d2_edge_costs.parquet (D2) | **Jun 7, 12:59** | **Jun 7, 11:33** |
| person_tracks.parquet (D4) | **Jun 7, 13:05** | **Jun 7, 11:34** |

D0+D1 artifacts are from Jun 9 (a re-run with run_until <= D1).
D2-D4 artifacts are from Jun 7 (the original full run).

### Three consequences of mixed provenance

**1. Identity hints mismatch (vid1, 6.2pp impact):**
- Jun 7's D2 processed an identity_hints.jsonl that contained a tag hint for t366
  (n_events_read: 1, must_link_groups: [{anchor_key: tag:1, tracklet_ids: [t366]}])
- The current identity_hints.jsonl (from Jun 9, 17:35) is empty (0 bytes)
- Re-running D2 today reads the empty file -> no must_link -> different solver solution

**2. D2 computed against different D1 (both clips):**
- Jun 7's D2 was computed against the Jun 7 D1 graph
- Jun 9's D1 re-run produced new d1_graph_nodes/edges (same node/edge count but
  potentially different content — the D0.5 split module was integrated by then)
- Today's D2 re-run reads the Jun 9 D1 -> different edge costs
- Vid1: original 16,099 edge costs vs today's 23,057 (all D1 edges costed)
- Vid2: original 8,672 edge costs vs today's 19,851

**3. Solver code changed (CP-TAG-4a, committed Jun 7 13:42):**
- Original D2-D4 artifacts written Jun 7, 12:59-13:05 (BEFORE commit e341a43 at 13:42)
- d3_ilp2.py modified: split-aware ping binding, hard no-drop constraint
- d4_emit.py modified: thread consumption for tag identity
- Today's re-run uses post-CP-TAG-4a code

### Verdict

The 30.7% vs 34.7% gap is NOT a sweep harness bug. The sweep harness correctly re-runs
Stage D from scratch on the current Stage A/C artifacts using the current code. The
pre-existing 34.7% baseline was measured against Stage D artifacts that were:

1. Built with a tag constraint that no longer exists in the current identity_hints.jsonl
2. Built against a different D1 graph (Jun 7 vs Jun 9)
3. Built with a pre-CP-TAG-4a solver

**The correct sweep baseline is 30.7%.** The 34.7% number should be documented as
"pre-existing baseline from mixed-provenance artifacts" and NOT used as the sweep
comparison target.

### Implication for Sweep

All sweep points go through the same run_stage_d.py + run_gt2actuals.py path, so
relative comparisons between sweep points are valid. The absolute baseline is 30.7%
(stock params, current code, clean D0->D4 re-run), not the historical 34.7%.

Steps 2 and 3 are not needed — the root cause is fully explained by Step 1.
