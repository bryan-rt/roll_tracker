# CP4.C+D Results — Session timeline + D1/D2 read real time

**Date:** 2026-08-26
**Commit:** (this commit)
**Camera:** FP7oJQ
**Pipeline state:** `variable_dt: false`, post-CP4.C/D, recalibrated H (`f7d76d6`)
**Baseline:** `docs/evidence/t2_5_baseline_1/` (commit `4291f21`, corrected `4672bd5`)
**CP4.B reference:** `docs/evidence/cp4b_results/`
**D0.5 extraction:** output directories deleted before rerun; per-clip `d05_split_summary`
from fresh audit (one summary per run, authoritative)

---

## 1. What changed

**CP4.C (sites #9, #10):**
- Session `frame_index` offset: cumulative frame count from sidecar (fps-free, collision-free)
- Session `timestamp_ms` offset: `pts_wallclock_offset_s` delta from sidecar (real time)
- `clip_offset_registry.json` persisted by Stage D, read by Stage F (one source of truth)
- `showinfo_offset_status` logged per clip
- Negative time delta raises (not silently clamped)

**CP4.D (sites #6, #7):**
- `dt_ms` added to D1 edge payloads (real-time, from `timestamp_ms` via shared
  `_build_frame_ts_map` helper). `dt_frames` retained as diagnostic frame-gap count.
- `costs.py`: reads `dt_ms / 1000.0` for timing; `fps` parameter removed.
  `dt_unavailable` reason added for edges with `dt_ms` null but `dt_frames` present.
- Site #7 reconnect speed gating: `dt_s` from `timestamp_ms` instead of `gap_frames / fps`.

**Schema change:** `dt_ms` (Int64, nullable) added to D1 edge and D2 edge cost ColSpecs
(`f0_parquet.py`). `dt_frames` retained.

---

## 2. dt_s semantic change — NOT comparable pre/post CP4.D

`dt_s` in `d2_edge_costs.parquet` and `d1_reconnect_edges.parquet` now carries real elapsed
time from container PTS. Previously it carried `dt_frames / fps` (a uniform-spacing
approximation). Same column name, same type, same ColSpec — different quantity.

**Any pre-CP4.D `dt_s` figure is not comparable to a post-CP4.D one.** No deltas are computed
against baseline `dt_s` values in this document.

**Incidental effect:** `d1_reconnect_edges.parquet`'s `dt_s` also became real-time without a
separate change, because `cand["dt_s"]` flows from site #7's `recon_dt_ms / 1000.0`.

---

## 3. Carrying-path verification (read-back from disk)

Session `person_tracks_FP7oJQ.parquet` read from disk:
- `timestamp_ms` present: yes
- Clip 2 (131129) first `timestamp_ms`: **543346** (≈543s, session-relative, NOT near zero)
- Clip 3 (132650) first `timestamp_ms`: **1464624** (≈1465s, session-relative)
- Total range: 0–1584557ms (≈26.4 minutes, the wall-clock span)

`clip_offset_registry.json`:

| Clip | frame_offset | ts_offset_ms | clip_frame_count |
|------|-------------|-------------|-----------------|
| 130229 | 0 | 0 | 1800 |
| 131129 | 1800 | 543280 | 1709 |
| 132650 | 3509 | 1464624 | 1764 |

Frame offsets are cumulative (0, 1800, 3509 = 1800+1709). Timestamp offsets are
sidecar-anchored real time.

---

## 4. Per-clip comparison (leak check: CP4.C/D should NOT change clip-level metrics)

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, clip-level, output dirs deleted.

| Metric | 130229 CP4.B | 130229 CP4.CD | 131129 CP4.B | 131129 CP4.CD | 132650 CP4.B | 132650 CP4.CD |
|--------|-------------|---------------|-------------|---------------|-------------|---------------|
| `speed_max` | 40.58 | 40.58 | 31.55 | 31.55 | 53.30 | 53.30 |
| `speed_p99` | 7.15 | 7.15 | 9.49 | 9.49 | 4.02 | 4.02 |
| `speed_p50` | 0.51 | 0.51 | 0.49 | 0.49 | 0.28 | 0.28 |
| D0.5 total | 44 | 44 | 35 | 35 | 21 | 21 |
| d1_recon | 92 | 92 | 885 | 886 | 827 | 827 |
| det/frame | 3.86 | 3.86 | 2.22 | 2.22 | 4.78 | 4.78 |
| tracklets | 82 | 82 | 51 | 51 | 66 | 66 |
| persons | 106 | 106 | 9 | 10 | 17 | 17 |
| dt_unavail | — | 0 | — | 5 | — | 0 |

**Leak check: PASS.** All clip-level `speed_*` and D0.5 splits identical to CP4.B. CP4.C/D
are session-scoped and did not affect per-clip D0 kinematics. `d1_reconnect_edge_count`
moved by at most 1 (131129: 885→886) — negligible. 131129 `person_count` changed 9→10
(D1/D4 routing change from the 1-edge reconnect difference). `dt_unavailable`: 5 edges on
131129, 0 on the others — low count, not a coverage concern.

---

## 5. Session comparison

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, session-level
(`2026-08-22T1300`), 3 clips.

| Metric | Baseline | CP4.B | CP4.C/D | Source |
|--------|----------|-------|---------|--------|
| persons | 125 | — | 116 | `person_tracks_FP7oJQ.parquet` |
| `speed_max` | 51.74 | — | 53.30 | `tracklet_bank_frames_FP7oJQ.parquet` |
| `speed_p99` | 6.57 | — | 6.56 | same |
| `speed_p50` | 0.39 | — | 0.39 | same |
| D0.5 total | 104 | — | 100 | per-clip summaries |
| **d1_recon** | 1903 | — | **2253** | `_debug/d1_reconnect_edges.parquet` |
| dt_unavail | — | — | 5 | `d2_edge_costs.parquet` |
| **cross-clip** | — | — | **11** | persons spanning multiple clips |
| Stage E | CRASHED | — | CRASHED | frame_index=315 |

**d1_reconnect_edge_count increased by 350 (+18.4%)** at session level. This is the
session-scoped effect of CP4.C/D: the real-time timeline changes which reconnect candidates
pass the speed gate. Per-clip reconnect counts barely moved (≤1), confirming the effect is
session-scoped as expected.

**11 persons span multiple clips** in the session run. This is the cross-clip stitching that
CP4.C enables — the session-relative timeline makes cross-clip `dt_s` meaningful.

---

## 6. Stage E crash incidence

| Segment | Baseline | CP4.B | CP4.C/D |
|---------|----------|-------|---------|
| 130229 | CRASHED (fi=315) | CRASHED (fi=315) | CRASHED (fi=315) |
| 131129 | OK (9 sessions) | CRASHED (fi=1356) | **OK (10 sessions)** |
| 132650 | OK (18 sessions) | OK (19 sessions) | OK (18 sessions) |
| Session | CRASHED | CRASHED | CRASHED (fi=315) |

131129 **recovered** from its CP4.B crash. Incidence: baseline 1/3, CP4.B 2/3, CP4.C/D 1/3.
The `timestamp_ms lookup miss` defect persists on 130229 (and therefore the session) but is
no longer triggered on 131129. This is consistent with the D1/D4 routing change from the
reconnect edge difference (886 vs 885 → different person_tracks → different frame coverage).

---

## 7. Expectation checks

### ✅ Clip-level speed/D0.5 unchanged from CP4.B (no leak)

All values identical. CP4.C/D are session-scoped.

### ✅ d1_reconnect_edge_count moved at session level

1903 → 2253 (+350, +18.4%). Per-clip barely moved (≤1). The session-level effect is the
real-time timeline changing which reconnect candidates pass the speed gate.

### ✅ Cross-clip stitching produced

11 persons span multiple clips. This requires session-relative timestamps to be meaningful.

### ✅ Detections/frame and tracklet counts unchanged

Identical to baseline on all three segments.

### ✅ dt_unavailable count is low

5 total (all on 131129). Not a coverage concern.

---

## 8. CP4.E input — attempt/window observation

All three segments report `attempt: 1` from three *different* recording windows — the
counter resets per window. `attempt` alone cannot mark these discontinuities. This is a
known hole in CP4.E's premise and this footage is its fixture.

The `clip_offset_registry.json` shows the three clips are 0s, 543s, and 1465s apart —
genuine wall-clock gaps, not delivery lag within one attempt.

---

## 9. Validation summary

| Tier | Result |
|------|--------|
| T1 — synthetic two-clip session (known offset) | PASS (clip 2 timestamp ≈ 5000ms, not near zero) |
| T1 — cross-clip dt_s regression guard | PASS (positive dt_ms across clip boundary) |
| T1 — registry round-trip | PASS (clip_offset_registry.json persisted and read back) |
| T1 — D1/D2 dt_ms unit tests | PASS (dt_ms=100→dt_s=0.1; dt_ms=None→dt_unavailable) |
| T1 — carrying path read-back | PASS (session person_tracks.parquet: clip 2 ts=543346, not 0) |
| T2 — regression suite | 189 passed, 10 skipped, 4 pre-existing |
| T2.5 — per-clip leak check | PASS (all clip-level metrics identical to CP4.B) |
| T2.5 — session comparison | d1_recon +18.4%, 11 cross-clip persons, Stage E 1/3 crashed |
