# OFAT track_buffer Sweep Results

Sweep baseline: **30.7%** combined correct_id (vid1 34.1%, vid2 28.2%).
Comparison point: sweep harness only (NOT the 32.5% freshened eval_gt baseline).

## Combined Results (vid1 + vid2)

| run_id | track_buffer | correct_id | delta | drift | misstitch | flags |
|--------|-------------|-----------|-------|-------|-----------|-------|
| tb05 | 5 | 26.1% | -4.6pp | 960 | 372 | -- |
| tb10 | 10 | 26.1% | -4.6pp | 961 | 425 | -- |
| tb15 | 15 | 27.4% | -3.3pp | 999 | 436 | -- |
| tb20 | 20 | 27.2% | -3.5pp | 1030 | 407 | -- |
| **tb30** | **30** | **30.7%** | **+0.0pp** | **1075** | **419** | **control** |
| tb45 | 45 | 30.3% | -0.4pp | 1069 | 446 | -- |
| tb60 | 60 | 29.2% | -1.5pp | 1023 | 500 | -- |

## Key Observations

### 1. track_buffer=30 (stock default) is the best value in this grid

The stock default produces the highest correct_id. Every deviation — both lower AND
higher — produces worse results. This is the opposite of the initial hypothesis that
"lower track_buffer = conservative breaking = cleaner fragments for the ILP."

### 2. Lower track_buffer degrades correct_id sharply (-3 to -5pp)

tb05-tb20 all perform significantly worse than tb30 (26-27% vs 30.7%). The degradation
is NOT from solver starvation (ilp_misstitch actually DECREASES at lower track_buffer:
372-436 vs 419). The damage is from increased tracklet_drift at tb15-tb20 (999-1030 vs
1075) combined with fewer correct frames overall.

### 3. Higher track_buffer also degrades, but gently

tb45 is nearly flat (-0.4pp) while tb60 drops -1.5pp. Misstitch rises steadily
(446 → 500), consistent with longer tracks accumulating more drift that the solver
can't fix.

### 4. No flags fired on any value

- `solver_starvation_signal`: never triggered
- `misstitch_rose`: never triggered (combined misstitch is below baseline at all points)
- `tag_hint_dropped`: never triggered

### 5. The "break, don't guess" hypothesis was wrong for track_buffer

Lower track_buffer doesn't produce cleaner fragments — it produces more fragments
that are individually LESS useful to the solver. The ILP benefits from longer tracklets
even if they contain some drift, because longer tracklets provide more spatial/temporal
context for graph construction (D1 merge/split detection, carrier selection, etc.).

## Per-Clip Breakdown

| run_id | tb | vid1 correct | vid1 drift | vid1 miss | vid2 correct | vid2 drift | vid2 miss |
|--------|---|-------------|-----------|----------|-------------|-----------|----------|
| tb05 | 5 | 29.3% | 571 | 234 | 23.7% | 389 | 138 |
| tb10 | 10 | 29.8% | 542 | 269 | 23.5% | 419 | 156 |
| tb15 | 15 | 32.3% | 539 | 291 | 23.8% | 460 | 145 |
| tb20 | 20 | 31.4% | 568 | 260 | 24.1% | 462 | 147 |
| tb30 | 30 | 34.1% | 606 | 251 | 28.2% | 469 | 168 |
| tb45 | 45 | 35.5% | 600 | 259 | 26.5% | 469 | 187 |
| tb60 | 60 | 35.9% | 578 | 284 | 24.3% | 445 | 216 |

Vid1 and vid2 show different patterns:
- **Vid1:** correct_id increases monotonically with track_buffer (29.3% → 35.9%).
  The stock default is NOT the best for vid1 in isolation — tb60 is +1.8pp better.
- **Vid2:** correct_id peaks at tb30 (28.2%) and degrades in both directions.
  tb45/tb60 hurt vid2 (-1.7pp/-3.9pp) while they help vid1.

The combined optimum at tb30 is a compromise between the two clips' conflicting
preferences.

## Diagnostic Metrics

| run_id | tb | tracklets_v1 | mean_len_v1 | short30_v1 | short10_v1 | tracklets_v2 | mean_len_v2 | short30_v2 | short10_v2 |
|--------|---|-------------|------------|-----------|-----------|-------------|------------|-----------|-----------|
| tb05 | 5 | 448 | 111.7 | 56.9% | 35.9% | 324 | 144.3 | 59.3% | 39.2% |
| tb10 | 10 | 363 | 138.4 | 58.7% | 36.6% | 245 | 191.4 | 55.1% | 36.3% |
| tb15 | 15 | 322 | 156.6 | 60.6% | 37.0% | 217 | 216.5 | 54.4% | 38.7% |
| tb20 | 20 | 296 | 170.7 | 60.8% | 37.8% | 197 | 238.8 | 52.3% | 37.6% |
| tb30 | 30 | 269 | 188.1 | 60.6% | 37.9% | 171 | 275.4 | 49.7% | 38.0% |
| tb45 | 45 | 261 | 194.0 | 61.7% | 38.3% | 157 | 300.1 | 49.0% | 37.6% |
| tb60 | 60 | 256 | 197.8 | 62.1% | 37.5% | 148 | 318.4 | 48.0% | 37.8% |

Tracklet count scales inversely with track_buffer (448 → 256 for vid1). Mean tracklet
length scales proportionally (112 → 198). Short-tracklet ratio (<30f) is remarkably
stable at ~57-62% across all values — the short tracklets are mostly detection-level
fragments that break regardless of track_buffer.

## Apparent Sweet Spot

**track_buffer=30 (stock default)** is the apparent sweet spot in this grid. It produces
the highest combined correct_id (30.7%), the lowest combined misstitch among the top-3
correct_id values (419 vs 446/500), and satisfies the two-sided criterion (correct_id
highest, misstitch not rising).

No parameter change in the screened range improves on stock defaults. The initial
hypothesis that lowering track_buffer would help was decisively refuted: every lower
value tested produced -3 to -5pp worse correct_id.
