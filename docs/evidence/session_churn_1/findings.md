# CP-R10: Session Churn Investigation — Findings

## Hypothesis

Host display sleep causes RTSP session invalidation. The CP-R1 capture (2026-08-04)
ran without caffeinate, and the host screen slept repeatedly. `capture.sh` (CP-R9) wraps
the recording in `caffeinate -dim`, which should prevent this.

## Baseline: CP-R1 (2026-08-04, no caffeinate)

Session scoped via `--start-epoch 1785871952` (19:32:32 UTC), `--window 3900`.

| Metric | FP7oJQ | PPDmUg | J_EDEw |
|---|---|---|---|
| Coverage | 85.6% | 86.6% | 0.0% |
| Segments | 31 | 31 | 0 |
| Attempts | 14 | 14 | 19 |
| Total gap | 564.0s | 541.4s | 3900.0s |
| Longest gap | 174.8s | 162.6s | 3900.0s |
| Longest run | 1953.0s | 1980.7s | 0.0s |
| Gap count | 7 | 7 | 1 |

**Gap correlation:** FP7oJQ and PPDmUg gaps occur at nearly identical times (gap_1 at
19:36, gap_2 at 19:39, gap_3 at 19:44, etc.), consistent with a host-level cause
(display sleep) rather than per-camera Nest-side invalidation.

J_EDEw: 0 segments / 19 attempts. Camera offline — not churn-related.

### Inter-segment delta calibration

All 30 inter-segment deltas from FP7oJQ CP-R1 sorted:

- Within-attempt boundaries: -2.27s to +1.07s (24 values). Negative = segment-muxer
  overlap (next segment starts before previous ends). Normal.
- Between-attempt gaps: 9.94s to 174.77s (6 values). Real gaps.
- Clean separation at 2.0s threshold (max boundary 1.07, min gap 9.94).

Gap threshold set to 2.0s. Distribution justifies this choice.

## Test: CP-R10 (2026-08-05, caffeinate active)

### Control conditions

| Condition | Pre-capture | Post-capture |
|---|---|---|
| Display sleep | 60 min (prevented by caffeinate) | 60 min (prevented by caffeinate) |
| System sleep | Prevented by caffeinate | Prevented by caffeinate |
| Power source | AC, 18W, battery full | AC, 18W |
| caffeinate PID | 47447, 48207 | 70313, 71000 (rotated — smoke test caffeinate expired, capture caffeinate active) |
| Lid | Open | Open |
| Machine touched | No | No |

### pmset sleep/wake audit

`pmset -g log` during capture window (19:32–20:42 UTC / 15:32–16:42 EDT):

- **Zero sleep events.** No display sleep, no system sleep.
- caffeinate PID 47447 held PreventUserIdleSystemSleep + PreventUserIdleDisplaySleep
  for 01:09:29 (full capture duration).
- caffeinate ClientDied at 16:42:05 EDT (capture end).
- **Confirmation:** caffeinate engaged and prevented all sleep for the full window.

### Results

Session scoped via `--start-epoch 1785958356` (19:32:36 UTC), `--window 3900`.

| Metric | FP7oJQ | PPDmUg | J_EDEw |
|---|---|---|---|
| Coverage | 82.8% | 95.9% | 0.0% |
| Segments | 29 | 33 | 0 |
| Attempts | 4 | 6 | 19 |
| Total gap | 696.6s | 175.3s | 3900.0s |
| Longest gap | 291.7s | 97.0s | 3900.0s |
| Longest run | 2761.3s | 3152.7s | 0.0s |
| Gap count | 7 | 3 | 1 |

### Comparison

| Metric | FP7oJQ Δ | PPDmUg Δ |
|---|---|---|
| Coverage | 85.6% → 82.8% (-2.8pp) | 86.6% → 95.9% (+9.3pp) |
| Attempts | 14 → 4 (-71%) | 14 → 6 (-57%) |
| Total gap | 564.0s → 696.6s (+23%) | 541.4s → 175.3s (-67.6%) |
| Longest run | 1953.0s → 2761.3s (+41%) | 1980.7s → 3152.7s (+59%) |
| Gap pattern | 6 distributed → 6 late-clustered | 6 distributed → 2 late |

**Key structural change:** CP-R1 gaps were **correlated across cameras** (display sleep
affects both simultaneously). CP-R10 gaps are **uncorrelated** — PPDmUg ran cleanly for
52 min while FP7oJQ had its cluster independently. This is consistent with removing the
host-level cause and exposing camera-specific Nest-side behavior.

**FP7oJQ late cluster:** The first 46 minutes were a single continuous run (2761s), then 6
gaps in the last 18 minutes. Attempts dropped from 14 to 4, but the late failures
produced longer gaps (backoff escalation on consecutive failures). This pattern suggests
Nest session timeout at ~45 min, not display sleep.

**PPDmUg:** Total gap dropped 67.6% (541→175s), attempts dropped 57% (14→6), longest run
increased 59% (1981→3153s). Clear improvement.

## Verdict: Outcome 3 — Inconclusive (mixed, single sample)

Caffeinate **materially reduced attempts** on both cameras (14→4 and 14→6) and
**eliminated the correlated gap pattern** that characterized display-sleep-induced churn.
This confirms display sleep was A cause of the CP-R1 churn.

However, FP7oJQ's total gap INCREASED (+23%) due to a late-capture cluster that was not
present in CP-R1. This camera-specific behavior — possibly a Nest session lifetime limit
at ~45 min — is a separate mechanism unmasked by removing display sleep.

A single capture cannot distinguish:
- Whether the FP7oJQ late cluster is reproducible or a one-time Nest-side event
- Whether the ~45-min session lifetime is a real limit or coincidence

**What a second run would show:** If FP7oJQ again clusters failures at ~45 min, the session
lifetime is real and needs to be addressed (proactive session refresh before timeout). If it
doesn't, the late cluster was transient.

## CP-R8 gate decision

**CP-R8 should proceed. Primary GT camera: PPDmUg.**

| Camera | Gate status | Rationale |
|---|---|---|
| **PPDmUg** | **GO — primary GT camera** | 95.9% coverage, 175s total gap, 3153s longest run. Gaps are short (65s, 97s) and occur late. Prior GT was on J_EDEw and FP7oJQ, but PPDmUg has equivalent detection evaluation data (300 annotated frames, 51 val frames in the eval manifest) and the best recording reliability. |
| **FP7oJQ** | **Proceed with known ~17% loss** | 82.8% coverage, 697s total gap. The first 46 min are clean (2761s continuous), but a late cluster at ~45 min produces 5 gaps. Usable for GT if annotations exclude gaps, but not the primary camera. A second capture would determine if the ~45-min cluster is reproducible. |
| **J_EDEw** | **Blocked — offline** | 0 segments in both captures (CP-R1 and CP-R10), 19-24 attempts. Camera is not producing data. Does not block CP-R8 (PPDmUg and FP7oJQ both have evaluation data). |

**Protocol:**
1. Always use `capture.sh` (caffeinate protection). Never raw `docker compose exec -d`.
2. Expect ~5% gap rate on PPDmUg (175s / 3900s). Gaps are flagged by the coverage metric
   and excluded from annotation.
3. FP7oJQ: plan for 1-2 session restarts in the latter half of a 65-min capture.

## Bimodal segment validation (CP-R6 — first production exercise)

**This closes the last open validation gap from CP-R6.** The bimodal detection logic and
`short_mode_*` field emission path shipped in CP-R6 code-reviewed only, with an explicit
caveat in the contract that the path had never executed from a live capture.

8 of 33 PPDmUg segments within the CP-R10 capture window emitted `is_bimodal: true`:

| Segment | short_mode_fraction | short_fps | long_dt_ms | input_frames |
|---|---|---|---|---|
| PPDmUg-20260805-154251 | 0.085 | 30.0 | 66.9 | 1860 |
| PPDmUg-20260805-154450 | 0.223 | 30.0 | 67.6 | 2010 |
| PPDmUg-20260805-154850 | 0.160 | 30.0 | 66.8 | 1950 |
| PPDmUg-20260805-155051 | 0.162 | 30.0 | 67.0 | 1950 |
| PPDmUg-20260805-161050 | 0.115 | 30.0 | 66.7 | 1920 |
| PPDmUg-20260805-161449 | 0.165 | 30.0 | 66.7 | 1980 |
| PPDmUg-20260805-161849 | 0.182 | 30.0 | 66.8 | 1980 |
| PPDmUg-20260805-162827 | 0.109 | 30.0 | 66.9 | 1889 |

**Full `_meta` line** (PPDmUg-20260805-154251, representative):

```json
{"_meta":true,"sidecar_schema":4,"timing_mode":"passthrough","source_pts":true,"pts_origin":"segment_relative","fps_method":"trimmed_mean","segment_start_epoch":1785958971,"attempt":1,"input_frame_count":1860,"output_frame_count":1860,"nominal_dt_s":0.067,"measured_fps":15.2263,"measured_fps_median":14.9254,"measured_fps_mean":15.6044,"pts_timebase":90000,"pts_tick_delta_median":6030.0,"pts_tick_delta_mean":5767.6,"pts_delta_trim_kept":1747,"pts_delta_trim_total":1859,"mismatch":false,"is_bimodal":true,"short_mode_fraction":0.085,"short_mode_fps":29.9981,"short_mode_dt_s":0.033335,"long_mode_dt_s":0.066941,"pts_wallclock_offset_s":1785958969.40764,"offset_method":"lower_envelope","drift_rate_s_per_s":0.000972112,"drift_ppm":972.112,"drift_flat":false,"n_drift_windows":12,"pts_mean_delta_ms":64.0845,"pts_stdev_delta_ms":10.2283}
```

**Key observations:**

1. **PPDmUg is the "healthy camera" — and it is bimodal.** Prior characterization treated
   PPDmUg as a steady-15fps reference. Bimodality on 8 of 33 segments (24%) during a routine
   capture confirms this is common, not a tail event.

2. **Short-mode fraction varies 8.5–22.3%** across segments. This is the minority-long-mode
   case described in contract §5: the majority is ~15fps (long mode), with a ~30fps short
   mode interspersed. `nominal_dt_s` (0.067, based on median) correctly reflects the majority.

3. **Coast-suppression guidance (contract §6.1) is validated as necessary.** The 1.5×
   threshold on these segments would classify 8.5–22.3% of real frames as gaps, inserting
   phantom coast steps. Consumers MUST check `is_bimodal` on PPDmUg segments. See
   `docs/reference/sidecar_contract.md` §6.1.

4. **`measured_fps` is inflated on bimodal segments** (15.23 vs true majority-mode 14.93).
   The trimmed mean includes some short-mode ticks, pulling the average up. `measured_fps_median`
   (14.93) and `1/nominal_dt_s` (14.93) are more accurate. This confirms the contract's
   guidance to prefer `nominal_dt_s` under bimodality.

The CP-R6 contract caveat ("End-to-end emission... has not been exercised from a live
capture") has been removed. A note has been added to the Schema History recording first
production validation.

### Drift window validation (CP-R6)

120s production segments consistently reach 12-13 drift windows (well above the >= 4 gate).
`drift_rate_s_per_s` and `drift_ppm` fields are emitted and vary between segments:

- FP7oJQ: drift_ppm range -676 to +497 (high variability — needs investigation if used)
- PPDmUg: drift_ppm range up to +972 on bimodal segments

Drift fields confirmed functional in production.

## Provenance

| Item | Value |
|---|---|
| CP-R1 baseline | 2026-08-04, start_epoch=1785871952, window=3900s, no caffeinate |
| CP-R10 test | 2026-08-05, start_epoch=1785958356, window=3900s, caffeinate active |
| Coverage metric | `services/nest_recorder/coverage_report.py` |
| Gap threshold | 2.0s (calibrated from CP-R1 inter-segment deltas: max boundary 1.07s, min gap 9.94s) |
| Timezone | Container EDT (UTC-4), `--tz-offset -4` |
