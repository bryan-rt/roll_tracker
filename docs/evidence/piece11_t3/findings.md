# Piece 11 T3: Variable-dt tracker A/B on post-fix GT

**Date:** 2026-08-30
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s, post-recorder-fix)
**GT:** 8 people, 12,468 boxes, stride-1 dense (gt-eval-fp7oJQ-132650)
**Model:** bjj-detect-all-cameras-v2 (domain-tuned yolo26n, conf 0.45, CoreML)
**Commit:** 5ec72f1 (Piece 9 + guard fix, code change before measurement)
**max_lost_seconds:** 2.0 (default, only active in treatment arm)

---

## 1. Arms

| Arm | Config | Pipeline artifacts |
|-----|--------|--------------------|
| Control | `variable_dt: false` (default) | `outputs/_t3_arms/control_variable_dt_false/` |
| Treatment | `variable_dt: true` via `--config` overlay | `outputs/_t3_arms/treatment_variable_dt_true/` |

Both arms ran on the same clip, same model, same GT, same commit. The control was measured
in GT-EVAL-1. The treatment used a fresh pipeline run with the output directory deleted first
(JSONL append contamination prevention). Control restored to the default on-disk location
afterward.

**Verification that `variable_dt` took effect:** The orchestration audit does not log the
resolved config value (known defect: `pipeline.py:685` NameError). Verification is indirect:
(1) detection count differs (8439 control vs 8437 treatment — dt_s=0.0 at frame 2 changes
BoT-SORT's IoU matching), (2) the `VariableDtBotSort` runtime assertion at `tracker.py:171`
would crash if stock BotSort was used, and the run completed successfully.

---

## 2. dt_s=0.0 guard fix

The treatment run initially crashed at frame 2 with:
```
BotSortTracker: sidecar dt_s is 0.0 at frame_index=2. Variable-dt tracker
requires strictly positive per-frame intervals.
```

**Root cause:** `tracker.py:180` had `dt_s <= 0`, rejecting 0.0. But the Kalman filter
(`variable_dt_kalman.py`) is explicitly designed to handle dt_s=0.0 as a position no-op
(CLAUDE.md:196). Empirical verification (this session): `predict()` with dt ratio 0.0
produces zero position change (cx diff = 0.000000), velocity preserved. Control with ratio 1.0
confirms velocity applies normally (cx diff = 2.000000).

The semantics are correct: two frames sharing a PTS are one capture instant, so the tracker
should not advance position between them. No-op is right.

**Fix:** `dt_s <= 0` → `dt_s < 0`. Negative dt (non-monotonic time) and None still raise.
T1 test: `test_variable_dt_guard.py` — dt_s=0.0 passes, dt_s<0 raises, dt_s=None raises.

**MUXER-PTS-1 blast radius (3 downstream consequences beyond the recorder):**
1. Piece 12: MP4 muxer rejects duplicate PTS (EINVAL), 6/18 exports failed
2. CP4.B: dt_s <= 0 guard in D0 kinematics
3. Piece 11: this tracker guard (now fixed)

---

## 3. Before/after table

**Metric basis:** FP7oJQ-20260822-132650, 1,764 val frames, parquet path,
Hungarian IoU 0.5 (CP-EVAL-1), gt-eval-fp7oJQ-132650 manifest.

### Stage A — Detection

| Metric | Control (vdt=false) | Treatment (vdt=true) | Delta |
|--------|--------------------|--------------------|-------|
| Recall@0.5 | 0.6233 | 0.6231 | -0.0002 |
| Precision@0.5 | 0.9208 | 0.9208 | 0.0000 |
| Mean IoU@0.5 | 0.7872 | 0.7873 | +0.0001 |
| Total predictions | 8,439 | 8,437 | -2 |
| Matched@0.5 | 7,771 | 7,769 | -2 |

### Stage D — Identity

| Metric | Control (vdt=false) | Treatment (vdt=true) | Delta |
|--------|--------------------|--------------------|-------|
| **correct_id (present)** | **37.2%** (4,637) | **34.3%** (4,275) | **-2.9pp** |
| present_misattributed | 17.7% (2,213) | 20.6% (2,573) | **+2.9pp** |
| stage_a_no_detection | 37.7% (4,701) | 37.7% (4,703) | +0.0pp |
| d4_unassigned | 7.4% (924) | 7.4% (924) | 0.0pp |
| Mean coverage | 0.555 | 0.555 | 0.000 |
| Mean purity | 0.536 | 0.534 | -0.002 |
| Person count (pipeline) | 17 | 17 | 0 |
| Identity precision | 0.857 | 0.667 | -0.190 |

### Switch causes

| Cause | Control | Treatment | Delta |
|-------|---------|-----------|-------|
| detection_failure | 351 (65%) | 353 (65%) | +2 |
| sloppy_box | 121 (22%) | 119 (22%) | -2 |
| true_switch | 65 (12%) | 67 (12%) | +2 |
| tracklet_dropped | 6 (1%) | 6 (1%) | 0 |
| **Total** | **543** | **545** | +2 |

---

## 4. Expectation checks

**"correct_id may not move much"** — It moved -2.9pp. This is a non-trivial negative delta.

**"present_misattributed is the metric to watch (17.7%)"** — It rose from 17.7% to 20.6%
(+2.9pp). The treatment increased misattribution, not decreased it. The 360 frames that
moved from present to present_misattributed represent identity assignments that were correct
under stock tracking but wrong under variable-dt.

**"Recall and precision should be roughly unchanged"** — Held. Recall diff is -0.0002,
precision diff is 0.0000. The variable-dt tracker did not affect detection, only tracking
associations. The 2-detection difference is at the noise level.

**"Expect dt_s == 0 handling to matter"** — It did. The initial run crashed at frame 2.
The guard fix resolved it. After the fix, the KF handled dt_s=0.0 as designed (no-op).

**"max_lost_seconds"** — 2.0 (default). This is the wall-time track lifetime replacing
frame-count `max_time_lost`. At 2.0s it matches the stock behavior (30 frames × 67ms ≈ 2.0s).

---

## 5. Interpretation notes (observations, not conclusions)

The -2.9pp delta is in the wrong direction. Possible explanations (not investigated):

1. **The variable-dt KF changes tracklet association boundaries.** Different dt ratios on
   gap frames change the predicted position, which shifts IoU assignments. Some reassignments
   may be worse for this clip's specific geometry.

2. **The effect is within noise for a single clip.** N=1 clip, no confidence interval
   computed. A 2.9pp swing on 12,475 frames is 362 frame-level changes.

3. **max_lost_seconds = 2.0 may not be optimal.** It matches stock behavior by construction
   (2.0s ≈ 30 frames × 67ms). A different value might produce a different result.

4. **The ceiling is upstream.** 37.7% of frames have no detection. The addressable population
   is the 62.3% with detections, within which misattribution rose from 17.7% to 20.6%. The
   tracker change shifted existing associations rather than recovering lost ones.

This is a single-clip measurement, not a conclusion about variable_dt's value. The feature
is architecturally correct (dt-scaled prediction is more physically accurate than constant-dt)
but may need parameter tuning or a different test corpus to show benefit.

---

## 6. Non-comparability

This A/B is internally valid (same clip, same GT, same model, same commit, same scoring).
Neither arm is comparable to the canonical 33.9% (different footage, camera, pipeline version).
