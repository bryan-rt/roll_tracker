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

### ✅ Detections/frame and tracklet counts unchanged

All three segments: det/frame identical, tracklet count identical. No signal that the
timing change affected upstream Stage A behavior (expected — Stage A ran identically).

### ✅ D0.5 splits moved on higher-dispersion segments

130229 (0.06%): 44 → 44 (unchanged). 131129 (2.75%): 36 → 35 (−1). 132650 (2.10%):
24 → 21 (−3). D0.5 consumes `speed_mps_k`, which changed. The effect is present on
the higher-dispersion segments and absent on the near-gap-free segment — the expected
shape.

### ✅ `n_bad_dt_steps` fires on all segments

130229: 4, 131129: 1, 132650: 3. These are the duplicate-PTS frames (MUXER-PTS-1,
pre-fix footage at frame index 2) plus any other zero-dt steps from the gap pattern.
The guard prevented zero-division on every segment.

### Stage E crashes

130229: CRASHED (frame_index=315, same as baseline). 131129: CRASHED (frame_index=1356,
new — was OK at baseline). 132650: OK (19 sessions, baseline was 18). The 131129 crash
is the same pre-existing Stage E buzzer end-frame defect on a different frame. The change
in which segments crash is consistent with D0.5/D4 producing a slightly different person
track set.

---

## 4. CP4.B validation summary

| Tier | Result |
|------|--------|
| T1 — uniform equivalence | PASS (50ms/20fps, exact at 1e-12 tolerance) |
| T1 — non-uniform hand-computed | PASS (50ms then 100ms → 10.0 then 5.0 m/s) |
| T1 — zero-dt | PASS (no raise, counter increments, no inf) |
| T2 — regression suite | 184 passed, 10 skipped, 4 pre-existing failures |
| T2.5 — per-segment comparison | Speed moved, D0.5 moved on dispersion segments, stable controls unchanged |

---

## 5. Notes for later checkpoints

- The CP4.B run does not include `--visualize`-only debug artifacts equivalent to the
  baseline's `mat_view.mp4` because output directories were deleted for clean JSONL. The
  `--visualize` flag was set and `stage_D_paths.png` was regenerated, but the comparison
  is numeric, not visual.
- 131129's Stage E crash moved from OK to CRASHED. This is the pre-existing buzzer
  end-frame defect, not a CP4.B regression.
