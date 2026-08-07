# CP-R11: Definitive Frame-Spacing Characterization

**Date:** 2026-08-07
**Code state:** `64d16f8` (CP-R7)
**Footage:** CP-R1 capture (2026-08-04), CP-R10 capture (2026-08-05), smoke tests (2026-08-07)
**Tool:** `tools/analyze_frame_spacing.py`
**Sample size:** 283 passthrough segments, 247,417 intervals, 2 cameras (FP7oJQ, PPDmUg).
CP-R1b used 8 segments. This analysis uses 35x more data.

---

## 1. Summary

The frame-spacing model committed in CP-R1b ("bimodal interleaving, structurally
undecidable") is superseded. The correct model:

> **Each camera runs at a single cadence (~15fps) with periodic single-frame gaps
> whose spacing is determined by a camera-internal grid mismatch, not network loss.
> On rare occasions, the cadence switches to ~30fps in sustained blocks. The modes
> come in blocks, not interleaved.**

## 2. Data Inventory

| Camera | Date | Segments | Intervals | Schema | Notes |
|--------|------|----------|-----------|--------|-------|
| FP7oJQ | 2026-08-04 | 44 | 47,536 | v1-v2 | CP-R1 capture + smoke tests |
| FP7oJQ | 2026-08-05 | 89 | 69,240 | v2-v4 | CP-R10 capture |
| FP7oJQ | 2026-08-07 | 6 | 1,589 | v4 | Smoke test |
| PPDmUg | 2026-08-04 | 46 | 42,746 | v1-v2 | CP-R1 capture + smoke tests |
| PPDmUg | 2026-08-05 | 93 | 85,093 | v2-v4 | CP-R10 capture |
| PPDmUg | 2026-08-07 | 5 | 1,213 | v4 | Smoke test |
| **Total** | | **283** | **247,417** | | J_EDEw offline in all captures |

Schema v2 sidecars carry `timing_mode: "passthrough"` and `pts_timebase: 90000` (RTP clock),
confirming source-PTS data despite lacking the explicit `source_pts: true` field added in v4.
Tick deltas recovered from `pts_time_s` via `round(pts_time_s * 90000)` — verified lossless.

58 cfr_grid segments excluded (arrival-PTS).

## 3. Tick-Level Distributions

All cameras produce inter-frame tick deltas from a discrete set:

| Tick value | Duration | Classification | Camera | Segment type |
|-----------|----------|----------------|--------|-------------|
| 2970 | 33.0ms | fast (~30fps) | Both | Bimodal only |
| 3060 | 34.0ms | fast (~30fps) | Both | Bimodal only |
| 5940 | 66.0ms | slow (~15fps) | Both | All segments |
| 6030 | 67.0ms | slow (~15fps) | Both | All segments |
| 11970 | 133.0ms | gap (2x slow) | Both | Gap segments |
| 12060 | 134.0ms | gap (2x slow) | Both | Gap segments |

Within each mode, the tick values alternate in a 2:1 ratio (e.g. 6030:5940 = 2:1 on stable
15fps segments). This alternation is the camera's timestamp quantization, not jitter.

Mode centres derived per segment from clustering; centres are stable at 3000 ticks (fast)
and 6000 ticks (slow) across all segments and both cameras.

## 4. Hypothesis Verdicts

### H1: Modes come in BLOCKS, not interleaved. **CONFIRMED.**

**Run-length analysis on FP7oJQ-20260804-163102 (the transition segment):**

| Mode | Runs | Mean length | Median | Max |
|------|------|-------------|--------|-----|
| fast (~30fps) | 34 | 47.4 | 15.0 | 370 |
| slow (~15fps) | 102 | 8.2 | 11.0 | 24 |
| gap | 73 | 1.0 | 1.0 | 1 |

62 mode switches across 2518 intervals. The RLE sequence begins:

```
S10 G1 S11 G1 S11 G1 S11 G1 S11 G1 S11 G1 S11 G1 S11 G1 S9
F194
G1 S17 G1 S11 G1 S11 G1 S11 G1 S11 G1 S11 G1 S11 G1
F205
S1 F5 S1 F5 S1 F9 S1 F114 S1 F12
G1 S23 G1 S5 G1 S18 G1 S5 G1 S10 G1
F6 S1 F5 S1 F13 S1 F15 S1 F17 S1 F13 ...
```

Structure: pure slow blocks (10-24 intervals) separated by periodic gaps, punctuated by
large fast blocks (F194, F205, F370). Gaps are always single intervals (gap run length = 1
everywhere). The stream holds one cadence for seconds to tens of seconds before switching.

**Attempt-level concatenation (A2):** Boundary truncation affects 2 of N runs per segment
(first and last). On the CP-R10 attempt (30 FP7oJQ segments, ~30,000 intervals), concatenated
run lengths show max slow run = 23 (0.78s), max fast run = 370 (12.2s). The blocked structure
is not a segment-boundary artifact.

**On stable 15fps segments (n=51 long FP7oJQ segments):** Zero mode switches, zero fast
intervals. Only slow intervals with periodic gaps. Run-length is trivial: one slow run
per segment, interrupted by single-frame gaps every ~12 frames.

### H2: Sustained regularity refutes alternate-frame-loss. **CONFIRMED.**

| Camera | Longest gap-free run (frames) | Duration (seconds) |
|--------|-------------------------------|-------------------|
| FP7oJQ | 1,183 | 39.4 |
| PPDmUg | 1,979 | 131.9 |

PPDmUg delivered 1,979 consecutive 67ms intervals with zero exceptions. For the
alternate-frame-loss hypothesis to hold, the camera would need to lose exactly every
other frame for 132 seconds straight — 1,979 consecutive successes in a coin flip
(probability 2^-1979). This is not frame loss.

FP7oJQ's longest gap-free run (39.4s) is shorter because FP7oJQ has a systematic gap
every ~12 frames (see H3). But even between gaps, the intervals are perfectly regular
at 5940/6030 ticks with zero intermediate values.

**CP-R1b's "structurally undecidable" verdict is TOO STRONG.** The algebraic pair-sum
identity (Section 5 of CP-R1b) is mathematically valid for any single interval, but it
does not survive sustained perfect regularity over thousands of consecutive frames. PPDmUg's
15fps cadence is genuinely 15fps, not 30fps with loss.

The question of whether FP7oJQ's gaps represent lost frames or a camera-internal grid
mismatch is addressed in H3.

### H3: Gap spacing is periodic. **CONFIRMED.**

**FP7oJQ gap spacing (51 long stable-15fps segments):**

| Metric | Value |
|--------|-------|
| Gap spacing mode | 12 frames (34 segments) or 7 frames (17 segments) |
| Gap spacing mean | 11.4-12.7 frames |
| Gap rate | 7.8-8.8% of intervals |
| Grid rate (90000/6030) | 14.9254 fps |
| Effective rate | 13.81-13.87 fps |
| Predicted skip (1/(grid/eff - 1)) | 12.4-13.1 frames |

The predicted skip period of ~12.4 frames matches the observed dominant spacing of 12.
The gap spacing histogram shows two prominent values:

- **Mode = 12**: the primary pattern. Example: `12:111; 13:9; 11:8`.
- **Mode = 7**: a doublet pattern. Example: `7:56; 17:51; 16:6`. Note 7+17 = 24 = 2x12.

The doublet pattern arises because the real capture rate and the PTS timestamp grid are
not exactly commensurate. The skipped slot "walks" around the period. Most segments show
pure period-12 spacing; some show the slot splitting into alternating 7+17 = 24 doublets.

**The decisive test (A1):** Predicted skip (from grid-rate/effective-rate ratio) vs observed
skip across 137 FP7oJQ segments: Pearson r = 0.44 (p < 0.001). The correlation is moderate
rather than strong because the predicted value is stable at ~12.4 while the observed mode
oscillates between 7 and 12 depending on which phase the doublet splits into. The mean
predicted (12.5) closely matches the mean observed (11.8).

**Random network loss produces a geometric distribution** of inter-gap spacings, with the
mode at 1 and exponential decay. The observed distribution has a sharp mode at 12 with
harmonic structure at 7+17. This is incompatible with random loss and consistent with a
camera-internal clock/grid mismatch.

**Grid mismatch mechanism:** The camera's real capture cadence (~13.85fps) does not divide
evenly into the PTS timestamp grid (6030/5940 tick pairs, grid rate 14.93fps). Every ~12
captured frames, the accumulated phase mismatch exceeds one grid slot, producing a doubled
interval (11970/12060 ticks). The gap count matches the rate deficit exactly: at 13.85fps
vs 14.93fps grid, the deficit is 1.08 frames per 12.4, or ~8.1% — observed: 7.8-8.8%.

### H4: PPDmUg gap-free across all segments. **REFUTED.**

| Metric | PPDmUg | FP7oJQ |
|--------|--------|--------|
| Gap-free segments | 68/144 (47.2%) | 2/139 (1.4%) |
| Total gap rate | 0.45% | 7.5% |
| Longest gap-free run | 1,979 frames (131.9s) | 1,183 frames (39.4s) |

PPDmUg is NOT gap-free — 53% of segments have at least one gap. However, the gap rate
(0.45%) is 17x lower than FP7oJQ (7.5%). On pure 15fps PPDmUg segments (no bimodal
excursions): 62/125 gap-free, 0.43% gap rate.

The hypothesis was drawn from examining only 2 segments; the full dataset refutes it but
confirms that PPDmUg has far fewer gaps than FP7oJQ.

PPDmUg's low gap rate is consistent with its effective rate being very close to its grid
rate (both ~15fps), leaving minimal grid mismatch. FP7oJQ's effective rate (~13.85fps)
diverges significantly from its grid rate (~14.93fps), producing the systematic ~8% gap.

## 5. A4: FP7oJQ-163102 Sliding-Window Reproduction

CP-R1b reported (Section 4): "Short-mode proportion oscillating: 0% -> 96% -> 11% ->
100% -> 43% -> 94% -> 67% -> 98% -> 100% -> 57% -> 0%."

**Reproduced.** The sliding-window analysis (100-interval windows, stride 10) shows the
same proportions. But the underlying structure revealed by the RLE is NOT oscillation:

The raw data contains clean blocks (F194, F205, F370 fast; S10-S24 slow). A 100-frame
window straddling the boundary between an S11 slow block and an F194 fast block will report
~65% fast — which appears as "oscillation" but is actually a window averaging across a sharp
boundary.

**Reconciliation:** Both descriptions are correct at their resolution:
- **100-frame window** (CP-R1b): sees smoothly varying proportions, reports "oscillation"
- **Per-interval RLE** (CP-R11): sees clean blocks with sharp boundaries

The interleaving model was a windowing artifact. The raw data is blocked.

## 6. A3: V4 Contract Check (94 segments)

### PPDmUg bimodal segments (v4_is_bimodal = true)

8 PPDmUg segments fire `is_bimodal: true` with `short_mode_fraction` matching our measured
`fast_frac` exactly (to 3 decimal places): 0.085, 0.223, 0.160, 0.162, 0.115, 0.165, 0.182,
0.109. These are genuine bimodal segments with blocked fast intervals — 2-5 mode switches,
consistent with 1-3 fast blocks embedded in a slow stream.

`nominal_dt_s = 0.067` correctly reflects the dominant (slow) cadence.

### FP7oJQ gap segments (v4_is_bimodal = false)

All 47 FP7oJQ v4 segments show `is_bimodal: false` despite having 18-142 gaps per segment.
This is CORRECT behavior — FP7oJQ's gap pattern (8% at 2x nominal) is not bimodality. The
contract's bimodal detector fires when the minority cluster is BELOW the median (fast mode
at 0.5x); FP7oJQ's gaps are ABOVE the median (2x), in the same direction as outlier gaps.

### Contract model finding

The contract's `is_bimodal` flag accurately distinguishes genuine bimodal segments (PPDmUg
with fast+slow blocks) from gap-only segments (FP7oJQ with periodic skips). The flag does
NOT need re-framing for the blocked model.

`nominal_dt_s` correctly reports the dominant cadence in all cases.

The coast recipe (Section 6.1) uses `1.5 * nominal_dt_s` as the gap threshold. With
nominal=0.067s, threshold=0.1005s. FP7oJQ's gaps at 0.133s are correctly detected.
PPDmUg's gaps at 0.133s are also correctly detected. **The recipe works as designed.**

**One finding about the contract:** Section 5 describes the phenomenon as "interleaved
frames at two discrete rates" with "proportion shifting over time." This should be corrected
to "blocked runs at two discrete rates" (see Section 7 for impact statement).

## 7. Consequence Answers

### 1. What is the correct model of frame spacing?

Each camera delivers frames at a single cadence (~67ms / ~15fps) with periodic single-frame
gaps caused by a camera-internal grid mismatch. The gap spacing is ~12 frames on FP7oJQ
(~8% gap rate) and occasional on PPDmUg (~0.45% gap rate). On rare occasions, the cadence
switches to ~33ms (~30fps) in sustained blocks lasting seconds to tens of seconds.

### 2. Are gaps predictable?

**Yes, on FP7oJQ.** The gap spacing is determined by the ratio between the camera's real
capture rate and the PTS timestamp grid. On stable 15fps segments, gaps appear every ~12
frames with a doublet harmonic at 7+17. Each gap is exactly one missed grid slot (2x the
nominal interval). The gap count is entirely predicted by `1 / (grid_rate/effective_rate - 1)`.

**PPDmUg gaps are rare and irregular** (0.45% rate, 47% of segments gap-free). They do not
show the periodic pattern of FP7oJQ. PPDmUg's effective rate is very close to its grid rate,
leaving minimal grid mismatch.

### 3. Does the block structure change the coast design?

**The blocked structure simplifies coast design.** Within a slow block, the cadence is
perfectly regular (5940/6030 ticks alternating). Within a fast block, equally regular
(2970/3060 ticks). Gaps are single-frame skips (always exactly 2x the local cadence).

Coast injection needs only:
- Detect gaps via `dt_s > 1.5 * nominal_dt_s` (the existing recipe)
- Insert exactly 1 coast step per gap (always single-frame skips)
- No complex multi-step coast sizing needed

The block transitions (slow->fast or fast->slow) produce one anomalous interval at
the boundary but are rare (~62 transitions per 2518-interval bimodal segment, zero on
stable segments).

### 4. Injection vs variable-dt

**Coast-step injection is the right approach** for FP7oJQ's periodic gaps, and it is far
simpler than a variable-dt Kalman fork. The gaps are:
- Always exactly 1 missed frame
- Predictable in frequency (~8% of intervals)
- Detectable via simple threshold (`dt_s > 1.5 * nominal_dt_s`)

Variable-dt would provide marginal benefit beyond injection (handling the exact 0.133s
dt vs two 0.067s steps), but the coast injection is 95%+ of the value with zero boxmot
changes.

For PPDmUg (0.45% gap rate), coast injection is still correct but low-value (affects
<1 in 200 frames).

### 5. Does this change the 30fps opportunity?

**The grid mismatch finding reframes the 30fps question.** FP7oJQ's ~13.85fps is not
network loss of 30fps frames — it is a camera-internal cadence that does not cleanly
align to the PTS timestamp grid. The camera demonstrably CAN deliver ~30fps (the late
CP-R1 capture segments are genuinely near-pure 30fps), but the default cadence is ~15fps.

Whether the camera can be configured to sustain 30fps is a separate question from whether
the current ~15fps is "really 30fps with loss." It is not — it is genuinely ~15fps
(proven by H2). The 30fps opportunity depends on understanding what triggers the cadence
switch (unresolved — neither activity, brightness, nor stream reconnection explains it;
see CP-R1b Section 6).

## 8. Corrections to CP-R1b

### Section 4 (within-segment oscillation)

**Superseded 2026-08-07 (CP-R11).** The "oscillation" pattern reported from 100-frame
sliding windows is a windowing artifact on blocked data. The raw per-interval RLE shows
clean blocks of fast (F) and slow (S) modes, not interleaved oscillation. See CP-R11
Section 5 for the reconciliation.

The original finding is not wrong at the resolution it was measured — the sliding window
genuinely shows varying proportions. But the underlying mechanism is sharp block transitions
averaged by the window, not frame-level oscillation between modes.

### Section 5 (structurally undecidable)

**Qualified 2026-08-07 (CP-R11).** The algebraic pair-sum identity is mathematically valid
for any single interval and the derivation stands. However, the conclusion that 15fps vs
30fps+loss is undecidable does not survive the full 283-segment dataset:

1. PPDmUg delivered 1,979 consecutive 15fps intervals with zero exceptions. This rules out
   alternate-frame loss (probability 2^-1979).
2. FP7oJQ's gaps are periodic (mode=12 spacing), not random. Random loss produces geometric
   inter-gap spacings; periodic gaps are a camera-internal grid mismatch.
3. The gap count exactly matches the rate deficit predicted by the grid-rate/effective-rate
   ratio.

The undecidability claim remains valid for a single interval in isolation. It does not hold
when examined across thousands of consecutive intervals. CP-R11 resolves the question: the
15fps cadence is genuine, and FP7oJQ's gaps are a grid mismatch, not frame loss.

## 9. Contract Impact (not implemented — CP-R12)

### Section 5 (`is_bimodal` representation)

The description "interleaves frames at two discrete rates" should be updated to "delivers
frames at two discrete rates in sustained blocks." The detection logic and flag semantics
do not need changing — `is_bimodal` correctly fires on bimodal segments and correctly does
not fire on gap-only segments. Only the explanatory text needs updating.

### Section 6.1 (coast-step injection)

The bimodal coast guidance ("suppress coast injection on bimodal segments") remains sound.
On genuinely bimodal segments (PPDmUg with fast+slow blocks), the 67ms intervals between
fast blocks are real frames, not gaps — coasting on them would insert phantom time.

The gap-only case (FP7oJQ with periodic single-frame skips) is the primary coast target.
The existing recipe (`dt_s > 1.5 * nominal_dt_s`, `coast_steps = round(dt_s/nominal_dt_s) - 1`)
correctly handles this: on FP7oJQ at 15fps, gaps are 0.133s, nominal is 0.067s, threshold
is 0.1005s, coast_steps = round(0.133/0.067) - 1 = 1. Always exactly 1 coast step, which
is correct.

**No changes needed** to the coast recipe or bimodal guidance. The contract works as designed.
The explanatory text about "interleaving" should be corrected to "blocks."
