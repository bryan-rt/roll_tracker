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
| **D0.5 splits (T1/T2/T3)** | 1 / 61 / 42 = 104 | per-clip summaries (see note) |
| **D1 reconnect edges** | 1,903 | `_debug/d1_reconnect_edges.parquet` |
| Stage E | CRASHED | same error as 130229 (frame_index=315) |
| Session fps (site #1) | 15.008 | `run_session_d` return value |

**Session D0.5 splits = 104 (from per-clip summaries).** D0.5 runs per-clip only; the
session-level `d05_split_audit.jsonl` aggregates individual `d05_split_event` entries
from per-clip files but does NOT copy `d05_split_summary` events. The correct session
figure is the sum of per-clip summary tier counts: 44 + 36 + 24 = 104 (T1=1, T2=61,
T3=42). The session file's raw event count of 146 (T1=2, T2=83, T3=61) — counted from
individual `d05_split_event` entries — is contaminated: 132650's clip-level audit held
66 events from 3 `--force` reruns instead of 24 from the latest (surplus = 42, which is
exactly 146 − 104). **The 146 figure must not be cited.** See §10 for the workflow
constraint.

This is a CP4.B input: CP4.B changes the speed that feeds D0.5, and that effect appears
at clip level, not through a session-level splitter.

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

---

## 10. JSONL accumulation — CP4.B–F workflow constraint

**All JSONL files in the pipeline APPEND (`open("a")`). All parquet files TRUNCATE
(`pd.to_parquet` overwrites). `--force` does not clear files before rerunning.**

This was discovered because the session D0.5 count reported 0 (the extractor searched for
a `d05_split_summary` event in the session file, which only contains aggregated
`d05_split_event` entries), and then 146 when counting individual events (132650's
clip-level audit had accumulated 66 events from 3 pipeline runs instead of 24 from the
latest — surplus 42, exactly 146 − 104).

**Consequence for CP4.B–F comparisons:**

Parquet metrics (`speed_mps_k`, tracklet counts, `person_count`, detections, contact
points, `d1_reconnect_edges.parquet`) are safe to rerun-and-diff — the file is rewritten
wholesale.

JSONL metrics (D0.5 splits, tag observations, identity hints, export manifest, audit
events) are NOT safe to rerun-and-diff. They accumulate across every run of that clip.
Either use per-run summary events, or delete the clip output directory before rerunning.

**Asymmetry worth noting:** `d1_reconnect_edges` (the best dispersion-sensitive signal) is
parquet — safe. D0.5 splits are JSONL — not safe. CP4.B moves both.

**CP4.A note:** the `sidecar_validated` audit event goes to the appending orchestration
audit JSONL. A reread across reruns will show multiple entries per clip. Not a defect in
CP4.A — a property of the audit layer.

JSONL files affected (17 call sites across 10 source files):
`d05_split_audit.jsonl`, `orchestration_audit.jsonl`, per-stage `audit.jsonl` (A, D, E),
`export_manifest.jsonl`, `projection_debug.jsonl`, `identity_hints.jsonl`,
`tag_observations.jsonl`.

---

## 11. 130229 person count: 106 persons from off-mat spectators

130229 reports 106 person tracks against 9 (131129) and 17 (132650). This is not an
extraction artifact — `person_tracks.parquet` is parquet (truncate-on-write), with 8,513
person-frame rows and 106 distinct `person_id` values (`p0001`–`p0106`).

**Track-length distribution:**

| Metric | 130229 | 131129 | 132650 |
|--------|--------|--------|--------|
| Persons | 106 | 9 | 17 |
| Median track length (frames) | 28 | 399 | 647 |
| Persons with ≥100 frames | 23 | 7 | 12 |
| Persons with <30 frames | 54 | 0 | 3 |
| Single-tracklet persons | 98 | 9 | 16 |

**What the detector is firing on:** real people, not spurious detections. Annotated-video
frames at indices 100, 400, 1200, 1600 show 3–5 people per frame: 1 person walking on the
mat, plus 2–4 people standing or sitting near the bench/entrance area at the left edge of
the frame. These people are off-mat but within the camera's field of view. 74.9% of contact
points (5,153 / 6,876) fall outside the calibrated quad (x 51.01–57.00, y 33.96–56.02),
with `x_m` reaching 46.62 — the spectator area.

**Why 106 persons:** the off-mat people enter and exit the frame frequently, producing short
tracklets that the solver cannot stitch. 54 of 106 persons have <30 frames (<2 seconds).
The solver creates a new person_id for each fragment. This is the same fragmentation pattern
seen across the project (50%+ short-tracklet ratio) applied to spectators rather than
athletes.

**Is 130229 still usable as the low-dispersion control?** Yes, with a restriction. Its gap
rate (0.06%) is a genuine property of the footage and its `d1_reconnect_edge_count` (92) is
the signal CP4.B–F actually turn on — both are unaffected by spectator fragmentation. Its
`person_count` and short-tracklet ratios are not comparable to the other two segments. Do
not treat 106 as an anomaly to be fixed, and do not cite `person_count` across these three
segments as if it measured the same thing.

**Out-of-scope observation:** the pipeline currently tracks spectators as persons. This is a
product-level concern (ROI masking / on-mat gating) — not investigated here.

---

## 12. Observations for later checkpoints

### 12a. `speed_p99` does not track gap rate; reconnect edges do

| Segment | Gap rate | D1 reconnect edges | `speed_p99` |
|---------|----------|--------------------|-------------|
| 130229 | 0.06% | 92 | 7.10 |
| 131129 | 2.75% | 921 | 9.51 |
| 132650 | 2.10% | 891 | 4.08 |

Reconnect-edge count separates the low-dispersion segment from the other two by ~10×,
tracking gap rate closely. `speed_p99` shows no such relationship before any change has
been made.

**Consequence for CP4.B–F:** `d1_reconnect_edge_count` is the dispersion-sensitive signal;
`speed_p99` is not. The "effect should scale with gap rate" expectation (§7) applies to
reconnect edges. Do not read a flat `speed_p99` response as a wiring fault — it is already
flat at baseline.

### 12b. `speed_max` is a defect detector, not a plausibility check

`speed_max` is 40.2 / 42.1 / 51.7 m/s (90–115 mph). Athletes do not move at these speeds;
these are tracker-drift or ID-swap artifacts, consistent with the known Stage A drift
attribution.

`speed_p50` (0.28–0.51 m/s) is entirely sensible. Use `speed_p50` and `speed_p99` for
plausibility; treat `speed_max` and `speed_implausible_count` as defect counters.

### 12c. Stage E crash is not CP22 NAType

Recorded as `timestamp_ms lookup miss for frame_index=315` on 130229 and the session run
(131129 and 132650 completed normally). CP22 NAType is a null-`frame_index` crash at D2 on
PPDmUg — a different defect. This is a frame→time lookup failure in Stage E's buzzer
end-frame adjustment: a frame present in the timestamp map but absent from
`person_tracks`. **This is squarely Piece 4 territory — flagged as a CP4.C input.**
