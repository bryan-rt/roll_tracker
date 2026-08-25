# T2.5 Baseline 1 — Pre-CP4.B reference on verified footage

**Date:** 2026-08-24
**Camera:** FP7oJQ
**Baseline commit:** `ea987a5` (CP4.A-addendum steps 1-4)
**Homography:** recalibrated `f7d76d6` (2026-08-24, operator-verified)
**Pipeline state:** `variable_dt: false`, post-CP4.A ingest gate
**Stage F:** not run (Supabase untouched)

---

## 1. What T2.5 is and is not

T2.5 is a GT-free behavioural reference. It can **falsify** a change (impossible speeds,
absurd person counts, runaway fragmentation). It **cannot confirm** one — there is no GT,
so `correct_id` is unavailable and no identity-accuracy claim may be derived from it. A
clean-looking T2.5 run is *not* evidence the timing work helped. That remains T3, blocked
on annotation.

---

## 2. Corpus

Three long segments from `data/raw/nest/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/`:

| Segment | Frames | Duration | Gap count | Gap rate | `dt_s==0` count | `dt_s==0` frames | `attempt` | `nominal_dt_s` |
|---------|--------|----------|-----------|----------|-----------------|------------------|-----------|----------------|
| 130229 | 1800 | ~121s | 1 | 0.06% | 1 | [2] | 1 | 0.067 |
| 131129 | 1709 | ~114s | 47 | 2.75% | 1 | [2] | 1 | 0.067 |
| 132650 | 1764 | ~118s | 37 | 2.10% | 1 | [2] | 1 | 0.067 |

`a_eq_c: True` on all three. Sidecar metrics via `bjj_pipeline.contracts.f0_sidecar.parse_sidecar`.

**Excluded:** seven short segments (30–45 frames, ~17s total) and the empty `131831` (0 frames,
no sidecar). These are session-aggregation edge cases: each is the sole segment of a failed
attempt (attempts 2–10), producing 30–45 frames before dying.

**All three report `attempt: 1` from three different recording windows** — the counter resets
per window. Segment 130229 is from the first window; the gap between 130229 and 131129 is
genuine (different ffmpeg launch). See item 4 (CP4.E input).

---

## 3. Per-clip metrics

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, clip-level, source artifacts in
`outputs/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/FP7oJQ-20260822-{seg}/`.

| Metric | 130229 (0.06% gaps) | 131129 (2.75% gaps) | 132650 (2.10% gaps) | Source |
|--------|---------------------|---------------------|---------------------|--------|
| Total detections | 6,876 | 3,578 | 8,439 | `stage_A/detections.parquet` |
| Mean det/frame | 3.86 | 2.22 | 4.78 | same, `groupby(frame_index)` |
| Tracklet count | 82 | 51 | 66 | `stage_A/tracklet_summaries.parquet` |
| Short ratio (<30f) | 62.2% | 64.7% | 69.7% | same, `n_frames < 30` |
| Short ratio (<10f) | 36.6% | 41.2% | 57.6% | same, `n_frames < 10` |
| **Person count** | 106 | 9 | 17 | `stage_D/person_tracks.parquet` `person_id.nunique()` |
| **`speed_mps_k` max** | 40.20 | 42.09 | 51.74 | `stage_D/tracklet_bank_frames.parquet` |
| **`speed_mps_k` P99** | 7.10 | 9.51 | 4.08 | same |
| **`speed_mps_k` P50** | 0.51 | 0.48 | 0.28 | same |
| **`speed_is_implausible`** | 53 | 46 | 24 | same |
| **D0.5 Tier 1 (speed cap)** | 0 | 0 | 1 | `stage_D/d05_split_audit.jsonl` |
| **D0.5 Tier 2 (kinematic)** | 30 | 18 | 13 | same |
| **D0.5 Tier 3 (histogram)** | 14 | 18 | 10 | same |
| D0.5 total splits | 44 | 36 | 24 | same |
| **D1 reconnect edges** | 92 | 921 | 891 | `_debug/d1_reconnect_edges.parquet` |
| Stage E | CRASHED | OK (9 sessions) | OK (18 sessions) | `stage_E/match_sessions.jsonl` |
| `x_m` range | 46.62–57.63 | 48.40–57.99 | 47.24–55.42 | `stage_A/contact_points.parquet` |
| `y_m` range | 40.79–57.84 | 34.43–57.51 | 46.03–57.54 | same |

**Stage E crash on 130229:** `Stage E: timestamp_ms lookup miss for frame_index=315.
This frame is not in person_tracks — possible buzzer end-frame adjustment to an untracked
frame.` Stages A and D completed; Stage E artifacts not produced for this segment.

---

## 4. Session metrics

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, session-level
(`2026-08-22T1300`), 3 clips, source artifacts in
`outputs/00000000-0000-0000-0000-000000000003/sessions/2026-08-22/2026-08-22T1300/`.

| Metric | Session | Source |
|--------|---------|--------|
| **Person count** | 125 | `stage_D/person_tracks_FP7oJQ.parquet` |
| **`speed_mps_k` max** | 51.74 | `stage_D/tracklet_bank_frames_FP7oJQ.parquet` |
| **`speed_mps_k` P99** | 6.57 | same |
| **`speed_mps_k` P50** | 0.39 | same |
| **`speed_is_implausible`** | 123 | same |
| Tracklet count | 303 | `stage_D/tracklet_bank_summaries_FP7oJQ.parquet` |
| Short ratio (<30f) | 63.0% | same |
| **D0.5 splits (T1/T2/T3)** | 0 / 0 / 0 | `stage_D/d05_split_audit.jsonl` |
| **D1 reconnect edges** | 1,903 | `_debug/d1_reconnect_edges.parquet` |
| Stage E | CRASHED | same error as 130229 (frame_index=315) |
| Session fps (site #1) | 15.008 | `run_session_d` return value |

**Session D0.5 splits = 0:** D0.5 operates on the aggregated session bank where tracklet
IDs are namespaced (`{clip_id}:{tid}`). The zero count at session level vs. non-zero at
clip level reflects the difference in bank structure, not a suppression.

**Session Stage E crashed** with the same `timestamp_ms lookup miss for frame_index=315`
as clip 130229. This is a pre-existing Stage E defect, not caused by the T2.5 run.

---

## 5. CP4.B input — `dt_s <= 0` guard requirement

All three segments carry `dt_s = 0.0` at frame index 2 (MUXER-PTS-1, pre-fix footage).
CP4.B replaces `dt_s = df / fps` with a `timestamp_ms` delta. The two duplicate-PTS
frames share the **same `timestamp_ms`** — so `dt_s = 0` and the velocity computation
divides by zero. The existing guard `if df <= 0: continue` (`d0_bank.py`) operates on the
*frame* delta (always 1) and will not catch it. **CP4.B must guard `dt_s <= 0` explicitly;
these three segments are the fixture that proves it.**

---

## 6. CP4.E input — window boundaries vs attempt boundaries

All three segments report `attempt: 1` from three *different* recording windows — the
counter resets per window. **Window boundaries and attempt boundaries are not the same
thing, and `attempt` alone cannot mark this discontinuity.** The roadmap's attempt-change
requirement as written would miss it.

The attempt cascade (attempts 2–10 each yielding 30–45 frames before dying) and the 0.236
content/wall ratio (373.5s in 1580.9s) resemble repeated stream failure more than slow
delivery. That is objective 1's territory, recorded but not investigated here.

---

## 7. Expected shape of a correct CP4.B result

Effect should scale with gap rate:
- **130229 (0.06% gaps):** minimal movement — uniform fps was already nearly correct
- **131129 (2.75% gaps):** visible movement expected
- **132650 (2.10% gaps):** visible movement expected, less than 131129

A flat relationship across all three indicates a wiring fault regardless of which direction
the metric moved.

Metrics that **should be untouched** by a timing change: detections/frame, tracklet counts.
Movement there signals a different bug.

---

## 8. Excluded segments

| Segment | Frames | Attempt | Reason |
|---------|--------|---------|--------|
| 131332 | 33 | 2 | Too short for meaningful metrics |
| 131413 | 45 | 3 | Too short |
| 131451 | 43 | 4 | Too short |
| 131534 | 45 | 5 | Too short |
| 131831 | 0 | ? | Empty, no sidecar |
| 132048 | 30 | 8 | Too short |
| 132259 | 30 | 9 | Too short |
| 132508 | 30 | 10 | Too short |

These are session-aggregation edge cases. Each is the sole segment of a failed attempt.

---

## 9. Visual QA artifacts

Per-segment `stage_D_paths.png` copied to this directory:
- `stage_D_paths_130229.png`
- `stage_D_paths_131129.png`
- `stage_D_paths_132650.png`
- `stage_D_paths_session.png`

`mat_view.mp4` and `annotated.mp4` remain under `_debug/` in each clip's output directory
(not committed — video files).
