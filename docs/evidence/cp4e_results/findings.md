# CP4.E Results — Clip-boundary discontinuity handling

**Date:** 2026-08-26
**Commit:** (this commit)
**Camera:** FP7oJQ
**Pipeline state:** `variable_dt: false`, post-CP4.E, recalibrated H (`f7d76d6`)
**D0.5 extraction:** output dirs deleted; fresh per-clip `d05_split_summary`

---

## 1. Why the roadmap's `attempt`-change rule was insufficient

The Active Decisions Log specified: "`attempt` change = hard break in session aggregation."
`sidecar_contract.md:85` defines `attempt` as *"Retry attempt counter within the recording
window (1-indexed)."* It is window-scoped — the counter resets per window.

All three Saturday segments report `attempt: 1` from three different recording windows. The
rule as written sees no change across a ~7-minute discontinuity and would produce **zero hard
breaks** while 448 cross-clip edges pass.

---

## 2. Discriminator: shortfall + attempt (OR)

**Shortfall** = wall_gap − content_duration, where:
- wall_gap = `pts_wallclock_offset_s[i] - pts_wallclock_offset_s[i-1]`
- content_duration = `output_frame_count[i-1] × nominal_dt_s[i-1]`

Threshold: `max(2.0s, 10 × nominal_dt_s)`. At `nominal_dt_s=0.067`, threshold = 2.0s.

Shortfall is the discriminator. `attempt` is retained as a cheap corroborating signal that
requires no arithmetic and would catch a boundary where `nominal_dt_s` or
`pts_wallclock_offset_s` were unavailable but `attempt` was present. On the 2026-08-22 corpus
shortfall catches all 8 classifiable boundaries; attempt catches 7 of 8, missing boundary 1
(the 422.7s window reset, att 1→1).

### Full-hour boundary analysis (all 10 non-empty segments)

| # | From | To | Shortfall(s) | Att change | Catches | Decision |
|---|------|----|-------------|------------|---------|----------|
| 1 | 130229 | 131129 | 422.7 | NO | shortfall only | BREAK |
| 2 | 131129 | 131332 | 9.5 | YES | both | BREAK |
| 3 | 131332 | 131413 | 38.1 | YES | both | BREAK |
| 4 | 131413 | 131451 | 35.4 | YES | both | BREAK |
| 5 | 131451 | 131534 | 39.8 | YES | both | BREAK |
| 6 | 131534 | 132048 | ~315 | YES | both | BREAK |
| 7 | 132048 | 132259 | 129.3 | YES | both | BREAK |
| 8 | 132259 | 132508 | 126.3 | YES | both | BREAK |
| 9 | 132508 | 132650 | 99.5 | YES | both | BREAK |

Empty segment 131831 (0 frames, no sidecar): excluded by the ingest gate before session
aggregation. Boundaries 5→6 collapse to 131534→132048 (one boundary, shortfall computable).

**Tightest real boundary:** 9.5s (boundary 2), which is 4.7× the 2.0s threshold.

---

## 3. Permit-branch limitation

The full hour contains **no contiguous cut**. Every boundary is a discontinuity (smallest
shortfall 9.5s vs 2.0s threshold). The discriminator's PERMIT branch has no real-data
validation — only T1 synthetic exercises it. A correct CP4.E on this footage drives
cross-clip edges to ~0, which is indistinguishable from an over-aggressive discriminator.
A future capture with a genuine contiguous cut is needed to validate the permit branch.

---

## 4. The decisive number: cross-clip decomposition

| State | Per-clip sum | Session | Cross-clip (session − per-clip) |
|-------|-------------|---------|--------------------------------|
| Baseline | 1904 | 1903 | **−1 (≈0)** |
| CP4.C/D | 1805 | 2253 | **+448** |
| **CP4.E** | **1805** | **1805** | **0** |

Cross-clip reconnect edges: **+448 → 0.** Per-clip sum unchanged (1805, same as CP4.C/D).
Session total = per-clip sum exactly. Both boundaries classified as BREAK.

Session person_count: 125 (baseline) → 116 (CP4.C/D) → **126** (CP4.E). Returned toward
baseline, consistent with the 11 cross-clip merges from CP4.C/D being suppressed. The +1
vs baseline (126 vs 125) is within the incidental-routing-change range seen across all
checkpoints.

Cross-clip persons: 11 (CP4.C/D) → **0** (CP4.E).

---

## 5. Boundary decisions (from `clip_offset_registry.json`)

```
130229 → 131129: BREAK (shortfall) shortfall=422.680s threshold=2.000s
131129 → 132650: BREAK (shortfall) shortfall=806.841s threshold=2.000s
```

Both boundaries broken by shortfall. `attempt` did not change (all att=1). This is boundary 1
from the full-hour analysis — the one attempt alone would have missed.

---

## 6. Clip-level leak check

| Metric | 130229 | 131129 | 132650 | vs CP4.B |
|--------|--------|--------|--------|----------|
| `speed_max` | 40.58 | 31.55 | 53.30 | identical |
| `speed_p99` | 7.15 | 9.49 | 4.02 | identical |
| `speed_p50` | 0.51 | 0.49 | 0.28 | identical |
| D0.5 total | 44 | 35 | 21 | identical |
| d1_recon | 92 | 886 | 827 | ≤1 diff |
| persons | 106 | 10 | 17 | ≤1 diff |

**Leak check: PASS.** CP4.E is session-scoped and did not affect clip-level metrics.

---

## 7. Stage E crash incidence

| Segment | Baseline | CP4.B | CP4.C/D | CP4.E |
|---------|----------|-------|---------|-------|
| 130229 | CRASHED | CRASHED | CRASHED | CRASHED |
| 131129 | OK (9) | CRASHED | OK (10) | OK (10) |
| 132650 | OK (18) | OK (19) | OK (18) | OK (18) |
| Session | CRASHED | CRASHED | CRASHED | CRASHED |

Incidence: 1/3 (same as CP4.C/D). The defect is NOT fixed — it is a latent defect
sensitive to person track composition. Re-measure after CP4.F.

---

## 8. Validation summary

| Tier | Result |
|------|--------|
| T1 — contiguous boundary permits | PASS (shortfall 0.1s, same segment_id) |
| T1 — discontinuous boundary breaks | PASS (shortfall 93.3s, different segment_id) |
| T1 — unclassifiable boundary breaks | PASS (nominal_dt_s=None → BREAK) |
| T2 — regression suite | 192 passed, 10 skipped, 4 pre-existing |
| T2.5 — cross-clip decomposition | +448 → 0 (both boundaries BREAK) |
| T2.5 — clip-level leak check | PASS |
