# CP4.B Results — D0 kinematics read real time (audit site #5)

**Date:** 2026-08-25
**Commit:** (this commit)
**Camera:** FP7oJQ
**Pipeline state:** `variable_dt: false`, post-CP4.B, recalibrated H (`f7d76d6`)
**Baseline:** `docs/evidence/t2_5_baseline_1/` (commit `4291f21`, corrected `4672bd5`)
**D0.5 extraction method:** output directories deleted before rerun; per-clip
`d05_split_summary` from fresh audit (one summary per run, authoritative)

---

## 1. What changed

Site #5 (`d0_bank.py:571`): `dt_s = df / fps` replaced with `dt_s = dt_ms / 1000.0`
where `dt_ms = timestamp_ms[i] - timestamp_ms[i-1]`. The `fps` parameter, its extraction
from `manifest.fps`, and the fps > 0 guard were all removed. A `timestamp_ms` column
presence guard replaced them. The `dt_ms <= 0` guard was added (parallel to the existing
`df <= 0` guard) with a dedicated `n_bad_dt_steps` counter surfaced in the audit summary.

**Precision note:** `timestamp_ms` is int milliseconds. On post-R13a footage all
`pts_time_s` values land on exact integer milliseconds (5940/90=66, 6030/90=67 —
`frame_index_join_1/findings.md` §10 precision finding), so `int(timestamp_ms)` is
lossless. The 1ms granularity produces ~1.5% quantisation on a ~67ms step, within the
precision decision's acceptance (§0.2: "+/-1ms on a 67ms interval = 1.5%"). Per-frame time
read from the parquet, not the sidecar.

---

## 2. Per-segment comparison (clip-level, never pooled)

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, clip-level. Source: parquet
artifacts from `outputs/.../FP7oJQ-20260822-{seg}/stage_D/`. D0.5 from fresh per-clip
`d05_split_summary` (output dirs deleted before rerun). `--visualize` enabled.

| Metric | 130229 baseline | 130229 CP4.B | 131129 baseline | 131129 CP4.B | 132650 baseline | 132650 CP4.B |
|--------|-----------------|--------------|-----------------|--------------|-----------------|--------------|
| Gap rate | 0.06% | — | 2.75% | — | 2.10% | — |
| **`speed_max`** | 40.20 | 40.58 | 42.09 | **31.55** | 51.74 | 53.30 |
| **`speed_p99`** | 7.10 | 7.15 | 9.51 | 9.49 | 4.08 | 4.02 |
| **`speed_p50`** | 0.51 | 0.51 | 0.48 | 0.49 | 0.28 | 0.28 |
| **`speed_impl`** | 53 | 53 | 46 | 46 | 24 | **21** |
| **D0.5 T1/T2/T3** | 0/30/14=44 | 0/30/14=44 | 0/18/18=36 | 0/17/18=**35** | 1/13/10=24 | 1/10/10=**21** |
| **D1 reconnect** | 92 | 92 | 921 | **885** | 891 | **827** |
| person_count | 106 | 106 | 9 | 9 | 17 | 17 |
| tracklets | 82 | 82 | 51 | 51 | 66 | 66 |
| det/frame | 3.86 | 3.86 | 2.22 | 2.22 | 4.78 | 4.78 |
| `n_bad_dt_steps` | n/a | 4 | n/a | 1 | n/a | 3 |
| Stage E | CRASHED | CRASHED | OK (9) | CRASHED | OK (18) | OK (19) |

---

## 3. Expectation checks

### ✅ Speed moved (not nothing)

`speed_max` moved on 131129 (42.09 → 31.55, −25%) and 132650 (51.74 → 53.30, +3%).
`speed_p99` shifted slightly on all three. The change wired through — this is not a
no-op.

### ✅ `speed_p99` did not need to track gap rate

Per baseline §12a, `speed_p99` was already flat across segments before any change
(7.10 / 9.51 / 4.08). It remains flat after (7.15 / 9.49 / 4.02). This is the expected
shape, not a wiring fault.

### ⚠ `d1_reconnect_edge_count` moved moderately

92 → 92 (0%), 921 → 885 (−3.9%), 891 → 827 (−7.2%). CP4.B is clip-scoped and does not
touch D1 gating directly. The movement is likely indirect: D0.5 splits changed (36→35,
24→21), which changes the tracklet set D1 operates on, producing fewer reconnect
candidates. The magnitude (4–7%) is proportional to the D0.5 change and confined to the
higher-dispersion segments. This is plausibly an indirect effect, not a leak.

### `speed_max` moved in both directions

130229: 40.20 → 40.58 (+1%), 131129: 42.09 → 31.55 (−25%), 132650: 51.74 → 53.30 (+3%).
No consistent direction — consistent with `speed_max` being a defect counter (baseline
§12b) rather than a physical measurement.

### ✅ Detections/frame and tracklet counts unchanged

All three segments: det/frame identical, tracklet count identical. No signal that the
timing change affected upstream Stage A behavior (expected — Stage A ran identically).

### ✅ D0.5 splits moved on higher-dispersion segments

130229 (0.06%): 44 → 44 (unchanged). 131129 (2.75%): 36 → 35 (−1). 132650 (2.10%):
24 → 21 (−3). D0.5 consumes `speed_mps_k`, which changed. The effect is present on
the higher-dispersion segments and absent on the near-gap-free segment — the expected
shape.

### ✅ `n_bad_dt_steps` fires on all segments

130229: 4, 131129: 1, 132650: 3. Each segment has exactly one duplicate-PTS frame pair
at frame index 2 (MUXER-PTS-1). The count equals the number of tracklets in the bank
frames that span both frame 1 and frame 2 — each spanning tracklet produces one zero-dt
step. Verified: 130229 has 4 tracklets spanning f1→f2 in bank_frames (4/4), 131129 has
1 (1/1), 132650 has 3 (3/3). The guard prevented zero-division on every segment.

### Stage E incidence change

| Segment | Baseline Stage E | CP4.B Stage E |
|---------|------------------|---------------|
| 130229 | CRASHED (frame_index=315) | CRASHED (frame_index=315) |
| 131129 | OK (9 sessions) | **CRASHED (frame_index=1356)** |
| 132650 | OK (18 sessions) | OK (19 sessions) |

Incidence rose from 1/3 to 2/3. The defect (`timestamp_ms lookup miss` — a frame present
in the timestamp map but absent from `person_tracks`) is unchanged; its incidence rose
because D0.5 splits changed (36→35 on 131129), producing a different person track set
that exposes the buzzer end-frame adjustment to a different untracked frame. 132650's
match session count also moved (18→19) for the same reason.

This is the defect already filed as a CP4.C input (CLAUDE.md backlog: "Stage E
`timestamp_ms lookup miss`"). Its incidence rising under a timing change strengthens the
case that CP4.C may resolve it. **Re-measure after CP4.C+CP4.D land.**

---

## 4. CP4.B validation summary

| Tier | Result |
|------|--------|
| T1 — uniform equivalence | PASS (50ms/20fps, exact at 1e-12 tolerance) |
| T1 — non-uniform hand-computed | PASS (50ms then 100ms → 10.0 then 5.0 m/s) |
| T1 — zero-dt | PASS (no raise, counter increments, no inf) |
| T2 — regression suite | 184 passed, 10 skipped, 4 pre-existing failures |
| T2.5 — per-segment comparison | Speed moved, D0.5 moved on dispersion segments, stable controls unchanged |
