# Timing & Cross-Camera Assumption Audit — Findings

**Date:** 2026-08-16
**Commit:** `6633025` (HEAD of `services_uploader`)
**Auditor:** Claude (read-only, three-pass protocol)
**Scope:** Every site in `src/bjj_pipeline/` and `src/pipeline_validation/` that assumes
fixed fps, uniform frame spacing, synchronized cameras, or a specific numeric frame rate.

**Reference documents:**
- `docs/reference/sidecar_contract.md` — schema v4 (CP-R6). Authoritative spec.
- `docs/evidence/frame_spacing_1/findings.md` — CP-R11, definitive frame-spacing model.

**Timing model audited against (CP-R11):**
- Single cadence (~67ms / ~15fps) with periodic single-frame gaps.
- Occasional sustained ~30fps blocks (seconds to tens of seconds).
- FP7oJQ: ~8% periodic gaps (grid mismatch, every ~12 frames). PPDmUg: ~0.45% gaps.
- Cameras differ from each other (13.85 vs 15.00 fps measured in one session).

---

## 0. Summary Table

*Digest — ordered by severity. Omits "Sidecar replacement" and "Blast radius" fields;
see per-site subsections in Section 2 for the complete six-field record.*

| # | Location | Assumption | Wrong? | Error magnitude | Priority |
|---|----------|-----------|--------|----------------|----------|
| 1 | `session_d_run.py:491` | First clip's fps applies to ALL clips and cameras | Yes | Up to 2x across cameras (13.85 vs 15.00 in same session) | P0 |
| 2 | `ffmpeg.py:121-122` | `start_sec = start_frame / fps` | Conditional | On post-fix footage: correct (probe returns 15). On pre-fix CFR: correct (probe returns 30). Wrong if probe gives wrong value for VFR edge cases. | P0 |
| 3 | `manifest.py:60-68` | `start_seconds = frame / fps` written to Supabase | Conditional | Same as #2. Values persist in `clips` table (`numeric` columns). | P0 |
| 4 | `tracker.py:63` | BoT-SORT `frame_rate` never passed; boxmot defaults to 30 | Yes | `buffer_size` 2x intended at 15fps (2.0s vs 1.0s wall-clock). | P1 |
| 5 | `d0_bank.py:571` | `dt_s = df / fps` (scalar) | Yes | Every velocity/accel in `speed_mps_k`, `accel_mps2_k` wrong by fps ratio. D0.5 inherits. | P1 |
| 6 | `costs.py:413` | `dt_s = dt_frames / fps` | Yes | Cost-layer velocity/time wrong by fps ratio. | P1 |
| 7 | `d1_graph_build.py:1408` | `dt_s = gap_frames / fps` | Yes | Reconnect speed gating wrong by fps ratio. | P1 |
| 8 | `cross_camera_evidence.py:275` | `window_frames = temporal_window_s * fps` | Yes | Window/tolerance frame counts wrong by fps ratio. Inherits session-level fps (#1). | P1 |
| 9 | `session_d_run.py:207-221` | `derive_clip_frame_offset` uses scalar fps | Yes | Frame offsets wrong by fps ratio. Compounds with #1. | P1 |
| 10 | `session_f_run.py:88` | `derive_clip_frame_offset` in export | Yes | Same as #9, applied to Stage F source registry. | P1 |
| 11 | `run.py:444-445` (Stage E) | `frame_index / fps * 1000` timestamp fallback | Dead code | `timestamp_ms` always present (verified: 0 nulls in production). | P3 |
| 12 | `session_f_run.py:397` | `fps = video_meta.fps if > 0 else 30.0` | Conditional | Hardcoded 30.0 fallback. On post-fix footage, probe returns correct value. | P2 |
| 13 | `multiplex_runner.py:406` | `fps = 30.0` fallback | Conditional | Same pattern as #12. | P2 |
| 14 | `redact.py:387` | `cv2.VideoWriter(..., float(fps), ...)` | Conditional | Receives fps from caller. Correct if caller is correct. | P2 |
| 15 | `run.py:331` (Stage F) | `fps = video_meta.fps if > 0 else manifest.fps` | Conditional | Re-probes from video. Diverges from manifest chain. | P2 |
| 16 | `run.py:119-123` (Stage F) | `_infer_last_frame = floor(duration_sec * fps) - 1` | Conditional | Correct if probe returns correct fps. | P2 |
| 17 | `pipeline.py:223` | `duration_ms = 1000 * frame_count / fps` | Conditional | Wrong if fps wrong. Not consumed by pipeline logic. | P3 |
| 18 | `d1_graph_build.py:1954,2481` | `duration_ms` in audit JSONL | No (audit-only) | Emitted to `d1_graph_built` and `d1_reconnect_audit` events. Not a computational input. | P3 |
| 19 | `visualize.py:327,351,408` | `cap_fps` from `CAP_PROP_FPS`; `timestamp_ms = fi * (1000/cap_fps)` | Yes | Eval preview timestamps wrong by fps ratio on VFR footage. Affects CP-2 A/B ruler. | P3 |
| 20 | Kalman `dt=1` (boxmot) | Unit time-step per frame | Conditional | Self-consistent under constant cadence. Wrong under variable spacing (gaps, mode switches). Separate from #4. | P2 (coast arch.) |
| 21 | `pairing.py:26` | `fps` parameter | Vestigial | Accepted but never used in function body. Dead parameter. | P4 |
| 22 | `buzzer.py:74` | `fps` parameter | Vestigial | Accepted but not used in core logic. Dead parameter. | P4 |
| 23 | `cache_detections.py:63` | `clip_fps` in sweep cache summary | Dead field | Written to JSON summary but never read by any consumer. | P4 |

---

## 1. Propagation Map

### 1.1 FPS Origin

```
pipeline.py:470  _probe_video_meta_opencv()
    └── cv2.CAP_PROP_FPS  ←─── ffmpeg container metadata
            │
            ▼
    ClipManifest.fps  (f0_manifest.py:144, single float)
            │
            ├── backfilled by ensure_manifest() (pipeline.py:427-488)
            │   if missing: _probe_video_meta_opencv() again
            │   last resort: multiplex_runner.py:406 hardcodes 30.0
            │
            ▼
    [Every downstream consumer reads manifest.fps]
```

### 1.2 Consumer Flow

```
manifest.fps ──┬── Stage A: multiplex_runner.py:393 (fps resolution chain)
               │       └── NOT passed to BotSort (frame_rate stays 30)
               │       └── FrameIterator.timestamp_ms used for physics
               │
               ├── Stage D0: d0_bank.py:1000 (fail-fast if fps<=0)
               │       └── d0_bank.py:571  dt_s = df / fps  →  speed_mps_k, accel_mps2_k
               │       └── D0.5: reads speed_mps_k (indirect)
               │
               ├── Stage D1: d1_graph_build.py:317 (extracted via _get_manifest_fields)
               │       └── d1_graph_build.py:1257 reconnect gating (fps is not None)
               │       └── d1_graph_build.py:1408  dt_s = gap_frames / fps
               │       └── d1_graph_build.py:1954,2481 audit JSONL (duration_ms, metadata)
               │
               ├── Stage D2: d2_run.py:117  fps = manifest.fps (fail-fast if <=0)
               │       └── costs.py:413  dt_s = dt_frames / fps
               │
               ├── Session D: session_d_run.py:491  FIRST clip's fps for ALL
               │       └── session_d_run.py:221  derive_clip_frame_offset(fps=fps)
               │       └── SessionManifest.fps  →  cross_camera_evidence.py:275
               │
               ├── Stage E: run.py:444  fallback timestamp (DEAD CODE)
               │       └── run.py:473  compute_pair_distances(fps=...)  (VESTIGIAL)
               │       └── run.py:507  apply_buzzer_soft_gate(fps=...)  (VESTIGIAL)
               │
               └── Stage F: run.py:331  re-probes from video_meta (divergent path!)
                       └── ffmpeg.py:121  start_sec = start_frame / fps
                       └── manifest.py:60  start_seconds → Supabase clips table
                       └── run.py:119  _infer_last_frame
                       └── run.py:335  buffer_frames = consolidate_buffer_sec * fps
                       └── session_f_run.py:397  fps = video_meta.fps else 30.0
                       └── session_f_run.py:88   derive_clip_frame_offset (session export)
                       └── redact.py:387  cv2.VideoWriter(fps=...)
```

### 1.3 Divergent Path in Stage F

Stage F does NOT read `manifest.fps` directly. It re-probes the video:

```python
# run.py:331
fps = float(video_meta.fps if video_meta.fps > 0 else getattr(manifest, "fps", 0.0))
```

`video_meta` comes from `probe_video_metadata()` (ffprobe first, OpenCV fallback).
This means Stage F can get a **different** fps than Stages D/E if the probe returns
a different value than what was stored in the manifest. In practice they will agree
because both call the same underlying probe — but the path is structurally divergent,
and a future change to one path will silently break the other.

### 1.4 Empirical Verification (Pass 3)

Tested on `PPDmUg-20260807-102005.mp4` (passthrough, source_pts=true, 15fps):

| Probe | Value |
|-------|-------|
| ffprobe `r_frame_rate` | 15/1 |
| ffprobe `avg_frame_rate` | 15/1 |
| OpenCV `CAP_PROP_FPS` | 15.0 |
| OpenCV `CAP_PROP_POS_MSEC` (frame 1) | 66.667 ms |
| Sidecar `pts_time_s` (frame 1) | 67.000 ms (0.067 s) |
| Delta (sidecar - OpenCV) | ±0.335 ms max over 6 frames |

**On passthrough VFR footage, both ffprobe and OpenCV report the correct frame rate (15fps).**
`CAP_PROP_POS_MSEC` tracks sidecar `pts_time_s` within ±0.4ms — effectively identical.

Tested on `J_EDEw-20260318-200015.mp4` (pre-fix CFR, arrival-PTS):

| Probe | Value |
|-------|-------|
| ffprobe `r_frame_rate` | 30/1 |
| OpenCV `CAP_PROP_FPS` | 30.0 |
| OpenCV `CAP_PROP_POS_MSEC` (frame 1) | 33.333 ms |

Pre-fix footage is CFR at 30fps. The probe reports what the container declares.

**Conclusion:** On post-fix passthrough footage, the manifest fps probe chain is
correct. The pipeline's fps value IS right at the source. The defect is that this
correct value is (a) never passed to BoT-SORT, and (b) used as a single scalar
applied uniformly where per-frame timing is needed.

---

## 2. Defect Sites

### 2.1 Stage A — BoT-SORT `frame_rate` (buffer_size)

**Location:** `src/bjj_pipeline/stages/detect_track/tracker.py:63-72`.
`BotSortTracker._lazy_init` builds `cfg` from `self.params` but never sets `frame_rate`.
Call sites: `multiplex_runner.py:521`, `stages/detect_track/run.py:222`,
`tools/sweep/replay_tracker.py:70`.

**Assumption:** boxmot's default `frame_rate=30` applies. boxmot==16.0.8
(`requirements.txt:50`). Formula at `.venv/.../botsort.py:94`:
`buffer_size = int(frame_rate / 30.0 * track_buffer)`.
`max_time_lost = buffer_size` at `:95`, compared against a frame counter at `:401`.

**Currently wrong?** Yes. `tracker.params = {}` in production; `frame_rate` is never
injected.

**Error magnitude:** At true 15fps with `track_buffer=30` (stock default):
- Current: `buffer_size = int(30/30 * 30) = 30 frames = 2.0s` wall-clock.
- Correct: `buffer_size = int(15/30 * 30) = 15 frames = 1.0s` wall-clock.
- The buffer is **twice the intended wall-clock duration**, not half.

Fixing `frame_rate` to 15 will shorten effective track lifespan from 2.0s to 1.0s.
The OFAT track_buffer screen (`tools/sweep/diagnostics/ofat_track_buffer_results.md`)
found stock `tb=30` optimal — but that was measured under the same wrong `frame_rate=30`,
so the effective wall-clock lifespan during the screen was already 2.0s. The
OFAT result at `tb=60` would reproduce today's 2.0s lifespan post-fix (`int(15/30*60)=30
frames = 2.0s`). **Hypothesis: the OFAT result may be rescalable** — `tb=60` post-fix
is equivalent to `tb=30` pre-fix — rather than requiring a full re-run. This hypothesis
is version-dependent: an unpinned boxmot upgrade could change the `buffer_size` formula
and silently invalidate it. Record only; do not test here.

The entire OFAT `track_buffer` corpus was measured under the same wrong `frame_rate=30`.
All sweep deltas remain internally consistent (same harness path), but their absolute
wall-clock meaning is wrong.

**Sidecar replacement:** `frame_rate = round(1.0 / nominal_dt_s)` from the sidecar
`_meta` line. Validity gate: `source_pts: true` AND `nominal_dt_s` present. Fallback
when `source_pts: false`: use `measured_fps_mean` (approximate). Per the sidecar
contract Section 6.2, do NOT use `measured_fps` (wrong by up to 2x on bimodal segments).

**Blast radius:** Changes BoT-SORT's `buffer_size` and `max_time_lost` — affects track
lifespan, fragmentation rate, and all downstream identity stitching. The OFAT sweep
results may need re-interpretation or re-run. Does not change any parquet schema.

### 2.2 Stage A — Kalman `dt=1` (variable frame spacing)

**Location:** `.venv/.../boxmot/trackers/botsort/botsort_track.py` — `STrack.multi_predict()`
uses a hardcoded unit time-step (`dt=1` per frame). This is internal to boxmot, not in
pipeline code.

**Assumption:** All frames are uniformly spaced.

**Currently wrong?** Conditional. Self-consistent under constant cadence (which is the
common case: ~92% of FP7oJQ frames, ~97% of PPDmUg frames are in the majority mode).
Wrong during:
- FP7oJQ periodic gaps (~8% of intervals): gap frame has `dt=2x nominal` but Kalman
  predicts with `dt=1`. False teleport in prediction.
- Mode-switch blocks (0.7% of FP7oJQ corpus, 2.95% of PPDmUg): minority-mode frames
  have `dt=0.5x` or `dt=2x` nominal but Kalman uses `dt=1`.

**Error magnitude:** On gap frames, Kalman prediction overshoots by 1x the per-frame
displacement. On mode-switch frames, prediction error is up to 2x. Both inject noise
into the association step. This is the coast-architecture problem
(`CLAUDE.md` Active Decisions Log, "Coast architecture" row).

**This is distinct from §2.1.** `frame_rate` feeds only `buffer_size → max_time_lost`
(frame counter comparison). It never reaches the Kalman filter. The variable-spacing
defect requires a different fix: variable-dt Kalman steps consuming per-frame `dt_s`
from the sidecar, which requires a boxmot fork or subclass (unscoped).

**Sidecar replacement:** Per-frame `dt_s` from frame rows. Validity gate: `timing_mode:
"passthrough"` AND `source_pts: true`. Consumer recipe: sidecar contract Section 6.1
(gap detection) and Section 6.2 (variable-dt Kalman).

**Blast radius:** Requires modifying boxmot internals (fork or subclass). Changes Kalman
state transition matrix per frame. Would affect all tracker-derived measurements.

### 2.3 Stage A — FrameIterator timestamps

**Location:** `src/bjj_pipeline/core/frame_iterator.py:57-61`.

**Assumption:** `CAP_PROP_POS_MSEC` returns real container PTS.

**Currently wrong?** No — **correct on passthrough VFR containers.** Empirically verified:
`CAP_PROP_POS_MSEC` tracks sidecar `pts_time_s` within ±0.4ms on
`PPDmUg-20260807-102005.mp4` (passthrough, source_pts=true, 15fps). See Section 1.4.

On pre-fix CFR containers, `CAP_PROP_POS_MSEC` returns the synthetic CFR grid (uniform
33ms), which is what the container declares. This is "correct for the container" but does
not represent real capture timing (the real timing was discarded during CFR encoding).

**Error magnitude:** ±0.4ms on passthrough footage — negligible. The `timestamp_ms`
column in `detections.parquet` and `person_tracks.parquet` is effectively correct on
post-fix footage.

**Sidecar replacement:** Not needed for passthrough footage. `CAP_PROP_POS_MSEC` already
carries the real PTS. For higher precision or gap detection, per-frame `pts_time_s` from
the sidecar could replace it, but the current path is not a defect.

**Blast radius:** None. This is a "correct already" site.

### 2.4 Stage A — quality.py velocity computation

**Location:** `src/bjj_pipeline/stages/detect_track/quality.py:310-325`.
`compute_velocity(prev_xy, prev_t_ms, xy, t_ms)` uses `t_ms` deltas from
`FrameIterator.timestamp_ms`.

**Assumption:** `t_ms` differences represent real elapsed time.

**Currently wrong?** No — on passthrough VFR containers, `FrameIterator.timestamp_ms`
is effectively correct (§2.3). Velocities computed from these timestamps are correct.

**Error magnitude:** Negligible (±0.4ms per frame, <1% velocity error).

**Sidecar replacement:** Not needed. Current path is correct on passthrough footage.

**Blast radius:** None. This is a "correct already" site.

### 2.5 Stage A — processor.py physics warnings

**Location:** `src/bjj_pipeline/stages/detect_track/processor.py:404`.
Uses `compute_velocity()` from `quality.py` with `timestamp_ms` from FrameIterator.

**Assumption:** Same as §2.4.

**Currently wrong?** No — same "correct already" reasoning as §2.4.

**Error magnitude:** Negligible.

**Sidecar replacement:** Not needed.

**Blast radius:** None.

### 2.6 Stage C — trigger engine velocity

**Location:** `src/bjj_pipeline/stages/tags/c0_triggers.py:56-57`.
`dv_thresh_mps = 2.5`, `a_thresh_mps2 = 12.0`. Motion trigger uses position history
with `timestamp_ms` from FrameIterator (`c0_triggers.py:67`).

**Assumption:** `timestamp_ms` differences represent real elapsed time.

**Currently wrong?** No — same reasoning as §2.4. The trigger computes velocity from
`timestamp_ms` deltas, which are correct on passthrough containers.

**Error magnitude:** Negligible.

**Sidecar replacement:** Not needed.

**Blast radius:** None. The thresholds are in m/s and m/s^2 (physical units), not
frame-rate-dependent.

### 2.7 Stage D — d0_bank kinematics

**Location:** `src/bjj_pipeline/stages/stitch/d0_bank.py:571`.
```python
dt_s = df / fps  # df = frame_index delta (int)
```
Produces `speed_mps_k`, `accel_mps2_k`, `speed_is_implausible`, `accel_is_implausible`
columns. Fail-fast at `:1000-1002` if `manifest.fps <= 0`.

**Assumption:** Fixed fps; uniform frame spacing; `dt_s = frame_delta / fps` is real
elapsed time.

**Currently wrong?** Yes. Uses scalar `manifest.fps` which is correct as a clip-level
average but does not account for per-frame timing variations (gaps, mode switches).

**Error magnitude:** On a 15fps passthrough clip:
- Between non-gap frames: `dt_s = 1/15 = 0.0667s` — correct.
- Across a gap (FP7oJQ, ~8% of intervals): actual dt = 0.133s, computed dt = 0.0667s.
  Velocity underestimated by 2x. Speed flag threshold (8.0 m/s) effectively becomes
  16.0 m/s across gaps.
- Frame deltas > 1 are handled correctly by `df / fps` as long as `df` correctly counts
  skipped frames (which it does — `frame_index` is a sequential counter).
- **Key subtlety:** because `frame_index` is sequential (no gaps in the counter), a
  single-frame gap appears as `df=1` with the person having moved the distance of 2
  frames. The velocity IS wrong by 2x on those frames.

D0.5 (`d05_split.py`) does not consume fps directly but reads `speed_mps_k` from
d0_bank — it inherits this error indirectly. D0.5 Tier 1 speed-cap (48 m/s) and Tier 2
kinematic spike ratio (5x) thresholds are both affected.

**Sidecar replacement:** Per-frame `dt_s` from sidecar frame rows. For the d0_bank
computation, the sidecar's per-frame `dt_s` would replace `df / fps`:
```python
# Current:  dt_s = df / fps
# Proposed: dt_s = sum of dt_s values for frames between i-1 and i
```
Validity gate: `timing_mode: "passthrough"` AND `source_pts: true`. Fallback when
`source_pts: false`: retain `df / fps` with `measured_fps_mean`.

**Blast radius:** Changes `speed_mps_k`, `accel_mps2_k`, `speed_is_implausible`,
`accel_is_implausible` columns in `tracklet_bank_frames.parquet`. D0.5 split decisions
change (reads `speed_mps_k`). D2 cost layer reads bank summaries containing these
columns. Does not change parquet schema — same column names, different values.

### 2.8 Stage D — d2 edge costs

**Location:** `src/bjj_pipeline/stages/stitch/costs.py:413`.
```python
dt_s = float(dt_frames_i) / float(fps)
```

**Assumption:** Fixed fps; `dt_frames / fps` is real elapsed time between edge endpoints.

**Currently wrong?** Yes. Same scalar-fps defect as §2.7.

**Error magnitude:** Affects velocity-based cost computation on edges spanning gaps. On
non-gap edges (majority), error is negligible. On gap-spanning edges (~8% of FP7oJQ
intervals), velocity underestimated by up to 2x, which underestimates travel cost.

**Sidecar replacement:** Sum of per-frame `dt_s` across the frame span of the edge.
Same validity gate as §2.7.

**Blast radius:** Changes edge costs in `d2_edge_costs.parquet`. May change ILP solution.
No schema change.

### 2.9 Stage D — d1 reconnect edges

**Location:** `src/bjj_pipeline/stages/stitch/d1_graph_build.py:1408`.
```python
dt_s = float(gap_frames) / float(fps)
```
Gated at `:1257` on `fps is not None`.

**Assumption:** Fixed fps.

**Currently wrong?** Yes. Same scalar-fps defect.

**Error magnitude:** Reconnect edges span larger frame gaps than adjacent-frame edges.
The velocity check `speed_mps = dist / dt_s` against `v_max_mps` (default 8.0 m/s) is
wrong by the fps ratio. At 15fps with a 30-frame gap: computed dt_s = 2.0s (correct for
uniform spacing). The error only manifests if a gap falls within the reconnect span,
which is uncommon for short gaps but possible.

**Sidecar replacement:** Same as §2.8.

**Blast radius:** Changes which reconnect edges are admitted to the graph. May change
graph topology and ILP solution. No schema change.

### 2.10 Stage D — Session fps resolution (cross-camera)

**Location:** `src/bjj_pipeline/stages/stitch/session_d_run.py:480-499`.
```python
for mp4_path, clip_cam_id in session_clips:
    ...
    if fps is None and cm.fps > 0:
        fps = cm.fps  # FIRST clip with fps>0 wins
```

**Assumption:** All clips in the session (including across cameras) share the same fps.

**Currently wrong?** Yes. CP-R11 measured 13.85fps (FP7oJQ) vs 15.00fps (PPDmUg) in
the same session. Even within a single camera, fps can vary between segments (sustained
cadence switches). The first-clip-wins strategy applies one camera's rate to all.

**Error magnitude:** Up to ~8% error (13.85 vs 15.00) across cameras. Up to 2x error
if first clip happens to be from a 30fps block while others are 15fps (rare but possible).

This fps propagates to:
- `aggregate_session_bank()` at `:514` → `derive_clip_frame_offset()` for frame offsets
- `SessionManifest.fps` → all session-level D1-D4, cross-camera evidence, Stage E, Stage F

**Sidecar replacement:** Per-clip `1.0 / nominal_dt_s` from each clip's sidecar.
Session-level consumers should accept per-clip fps, not a single scalar.
Cross-camera consumers need per-camera fps.

**Blast radius:** Fixing this requires changing `SessionManifest` to carry per-clip or
per-camera fps instead of a single scalar. All session-level consumers (D1, D2, costs,
cross_camera_evidence, Stage E, Stage F) would need to accept per-clip fps. This is
the highest-blast-radius change in the audit.

### 2.11 Stage D — cross_camera_evidence window/tolerance

**Location:** `src/bjj_pipeline/stages/stitch/cross_camera_evidence.py:275-276`.
```python
window_frames = max(1, int(temporal_window_s * fps))
tolerance_frames = max(1, int(temporal_tolerance_s * fps))
```
Docstring at `:264` calls fps "authoritative from session manifest."

**Assumption:** Session-wide scalar fps.

**Currently wrong?** Yes. Inherits session-level fps from §2.10. Additionally,
cross-camera evidence compares frames across cameras that may have different real fps.

**Error magnitude:** With default `temporal_window_s=2.5` and `temporal_tolerance_s=2.0`:
at 15fps: `window_frames=37`, `tolerance_frames=30` (correct).
At 13.85fps: `window_frames=34`, `tolerance_frames=27` (if that camera's rate were used).
Using 15fps for a 13.85fps camera: window is ~9% too wide.

**Sidecar replacement:** Per-camera fps from each camera's sidecar. Use
`1.0 / nominal_dt_s` per camera. Convert `temporal_window_s` and `tolerance_s` to
frames independently per camera. Validity gate: `source_pts: true`.

**Blast radius:** Requires per-camera fps in the cross-camera evidence builder.
No schema change. Changes `coordinate_corroborated_tags` results.

### 2.12 Stage D — derive_clip_frame_offset (session aggregation)

**Location:** `src/bjj_pipeline/stages/stitch/session_d_run.py:207-221`.
```python
def derive_clip_frame_offset(mp4_path, session_start_dt, fps):
    return round(delta_sec * fps)
```
Called at session_d_run.py aggregate step.

**Assumption:** Fixed fps; `delta_seconds * fps` converts wall-clock time to frame count.

**Currently wrong?** Yes. The function converts a wall-clock time delta (from filename
timestamps) to a frame offset using a single scalar fps. If the real fps of the clip
differs from the scalar, the offset is wrong.

**Error magnitude:** For a clip starting 300s into the session at 15fps: offset = 4500.
If fps were actually 13.85: should be 4155. Error = 345 frames (~23s of content).
On same-fps clips: error is proportional to (actual_fps - scalar_fps) * delta_sec.

**Sidecar replacement:** `output_frame_count` from the sidecar `_meta` provides the
actual number of frames in each segment. Combined with `segment_start_epoch`, cumulative
frame counts could replace fps-derived offsets entirely. However, this requires a more
fundamental change to the offset computation. Interim: per-clip fps from sidecar.

**Blast radius:** Changes frame offsets in the session bank. All session-level identity
stitching uses these offsets. Cross-camera alignment depends on them. No schema change
in parquets.

### 2.13 Stage E — timestamp fallback

**Location:** `src/bjj_pipeline/stages/matches/run.py:444-445`.
```python
return int(round((frame_index / float(manifest.fps)) * 1000.0))
```
Inside `_ts_for_frame()`, fires when `frame_index` is not in the `frame_to_ts` map.

**Assumption:** Fixed fps.

**Currently wrong?** Dead code in production. Verified: `timestamp_ms` column in
`person_tracks.parquet` has 0 nulls (76,110 rows checked). The `frame_to_ts` map is
always populated from `person_tracks`, so the fallback never fires.

**Error magnitude:** N/A (dead code).

**Sidecar replacement:** N/A.

**Blast radius:** None. Could be removed in a cleanup pass.

### 2.14 Stage E — buzzer fps (vestigial)

**Location:** `src/bjj_pipeline/stages/matches/buzzer.py:74`.
`apply_buzzer_soft_gate(... fps: float ...)`.

**Assumption:** None — `fps` is accepted as a parameter but not used in the function's
core logic. The function operates on `frame_index` values directly.

**Currently wrong?** No — vestigial parameter.

**Error magnitude:** N/A.

**Sidecar replacement:** N/A. Remove the parameter in a cleanup pass.

**Blast radius:** None (parameter removal is a signature change).

### 2.15 Stage E — pairing fps (vestigial)

**Location:** `src/bjj_pipeline/stages/matches/pairing.py:26`.
`compute_pair_distances(person_tracks_df, *, fps: float)`.

**Assumption:** None — `fps` is accepted but never referenced in the function body.
The function computes spatial distances only.

**Currently wrong?** No — vestigial parameter.

**Error magnitude:** N/A.

**Sidecar replacement:** N/A. Remove the parameter in a cleanup pass.

**Blast radius:** Callers (`run.py:473`) pass `fps=manifest.fps`. Removing the
parameter requires updating the call site. No functional change.

### 2.16 Stage F — ffmpeg export seek arithmetic

**Location:** `src/bjj_pipeline/stages/export/ffmpeg.py:121-122`.
```python
start_sec = float(start_frame) / float(fps)
duration_sec = max(1.0/fps, float(end_frame - start_frame + 1) / float(fps))
```

**Assumption:** Fixed fps; `frame / fps` converts frame index to seconds for ffmpeg
`-ss` and `-t` parameters.

**Currently wrong?** Conditional. On passthrough footage, `fps` comes from
`probe_video_metadata()` which returns the correct value (empirically verified: 15fps
on 15fps VFR containers). On pre-fix CFR footage, probe returns 30fps (correct for the
container).

The defect surfaces if the probe returns a wrong value. On current passthrough footage
this does not happen. However, the computation is structurally wrong: it converts a
frame INDEX (sequential counter from `FrameIterator`) to a seek TIME using a scalar fps,
which only works if all frames are uniformly spaced. On a VFR container with gaps or
mode switches, the seek time could be offset from the intended frame.

**Error magnitude:** On 15fps passthrough with uniform spacing: correct. With an 8%
gap rate (FP7oJQ): the maximum offset accumulates over the clip. For a frame at index
1000 on FP7oJQ (real elapsed ~72.3s due to gaps): `1000/14.93 = 66.98s` from the PTS
grid rate, or `1000/15 = 66.67s` from the nominal rate. The actual elapsed time depends
on how many gaps preceded. Maximum offset: ~0.5s per 1000 frames at 8% gap rate.
**Customer-visible**: the exported clip starts at the wrong time.

**Sidecar replacement:** `pts_time_s` from the sidecar frame row for `start_frame`
directly provides the exact seek time. No fps conversion needed. Validity gate:
`source_pts: true`.

**Blast radius:** Changes `-ss` and `-t` arguments to ffmpeg. Exported mp4 clips
start/end at different times. Supabase `clips` rows (via §2.17) change. No schema change.

### 2.17 Stage F — manifest clip seconds (Supabase)

**Location:** `src/bjj_pipeline/stages/export/manifest.py:60-68`.
```python
start_seconds = float(export_start_frame) / fps_f
end_seconds = float(export_end_frame + 1) / fps_f
duration_seconds = max(0.0, end_seconds - start_seconds)
```

**Assumption:** Fixed fps.

**Currently wrong?** Same conditional as §2.16.

**Error magnitude:** Same as §2.16. These values are written to the Supabase `clips`
table:
- `start_seconds` → `clips.start_seconds` (type: `numeric`)
- `end_seconds` → `clips.end_seconds` (type: `numeric`)
- `duration_seconds` → `clips.duration_seconds` (type: `numeric`)

Table defined at `backend/supabase/supabase/migrations/20260310160143_initial_schema.sql:37-47`.
The RPC `claimable_clips` (`20260318000003_claimable_clips_rpc.sql:14-15`) casts these
as `float` for the API response.

**Sidecar replacement:** Same as §2.16 — derive seconds from sidecar `pts_time_s` for
the start and end frames.

**Blast radius:** Changes persisted values in Supabase `clips` table. Mobile app and
web app consume these values. Flutter `app_mobile/` and React `app_web/` display
clip timing derived from these columns.

### 2.18 Stage F — `_infer_last_frame`

**Location:** `src/bjj_pipeline/stages/export/run.py:119-123`.
```python
return max(0, int(math.floor(float(duration_sec) * float(fps))) - 1)
```

**Assumption:** Fixed fps; `duration * fps` gives frame count.

**Currently wrong?** Conditional. `duration_sec` comes from `probe_video_metadata()`
which reads ffprobe `format.duration`. On the passthrough test file, ffprobe reported
`duration=16.144000` and `nb_frames=240` with `fps=15`. `floor(16.144 * 15) - 1 = 241`.
Actual frames: 240. Off by 1 — a pre-existing edge case unrelated to fps correctness.

**Error magnitude:** ±1 frame. Low severity.

**Sidecar replacement:** `output_frame_count` from sidecar `_meta` provides the exact
frame count. `last_frame = output_frame_count - 1`.

**Blast radius:** Changes the last-frame boundary for export consolidation. Minimal.

### 2.19 Stage F — session_f_run fps fallback

**Location:** `src/bjj_pipeline/stages/export/session_f_run.py:397`.
```python
fps = float(video_meta.fps) if video_meta.fps > 0 else 30.0
```

**Assumption:** Hardcoded 30.0 fallback if probe fails.

**Currently wrong?** Conditional. On passthrough footage, the probe returns the correct
value. The 30.0 fallback fires only if video probing fails entirely.

**Error magnitude:** If the fallback fires on 15fps footage: all seek times wrong by 2x.

**Sidecar replacement:** `1.0 / nominal_dt_s` from sidecar. Fallback:
`measured_fps_mean` (always present).

**Blast radius:** Same as §2.16.

### 2.20 Stage F — redact.py writer fps

**Location:** `src/bjj_pipeline/stages/export/redact.py:387`.
```python
writer = cv2.VideoWriter(str(output_video_path), fourcc, float(fps), ...)
```

**Assumption:** `fps` parameter is correct.

**Currently wrong?** Conditional. Receives fps from caller (`render_redacted_clip`
parameter). If caller provides correct fps, output is correct. The caller chain goes
through Stage F `run.py:331` which re-probes from video.

**Error magnitude:** Wrong output playback speed if fps is wrong. On passthrough footage
with correct probe: not wrong.

**Sidecar replacement:** Same fps source as caller (§2.16/§2.19).

**Blast radius:** Output video plays at wrong speed. No database impact.

### 2.21 Orchestration — manifest fps origin

**Location:** `src/bjj_pipeline/stages/orchestration/pipeline.py:207-230, 420-488`.
`_probe_video_meta_opencv()` reads `cv2.CAP_PROP_FPS`. `ensure_manifest()` stores it
in `ClipManifest.fps`.

**Assumption:** `CAP_PROP_FPS` returns the correct frame rate.

**Currently wrong?** No — on passthrough footage, OpenCV returns the correct rate
(empirically verified: 15.0 on a 15fps VFR container). On CFR footage, returns the
container's declared rate (30.0 for pre-fix CFR).

**Error magnitude:** Correct on post-fix footage. The origin is not the defect — the
defect is downstream (single scalar applied to all frames/clips/cameras).

**Sidecar replacement:** `1.0 / nominal_dt_s` from sidecar provides a more precise
clip-level rate. But the current probe is adequate as a clip-level scalar.

**Blast radius:** This is the root of the propagation chain. Any change here affects
every downstream consumer.

### 2.22 Orchestration — multiplex_runner fps fallback

**Location:** `src/bjj_pipeline/stages/orchestration/multiplex_runner.py:393-406`.
```python
fps = float(getattr(manifest, "fps", 0.0) or 0.0)
if fps <= 0.0:
    it_fps = float(it.fps or 0.0)
    fps = it_fps if it_fps > 0.0 else 0.0
    ...
if fps <= 0.0:
    fps = 30.0
```

**Assumption:** 30.0 hardcoded fallback.

**Currently wrong?** Conditional. On passthrough footage, `manifest.fps` is correct
(15.0). The 30.0 fallback fires only if both manifest and FrameIterator fail to
provide fps.

**Error magnitude:** If fallback fires: 2x error on 15fps footage.

**Sidecar replacement:** Same as §2.21.

**Blast radius:** Affects Stage A physics warnings and metadata. Low — the fallback
path is defensive and rarely fires.

### 2.23 Orchestration — duration_ms computation

**Location:** `src/bjj_pipeline/stages/orchestration/pipeline.py:223`.
```python
duration_ms = int(round(1000.0 * frame_count / fps))
```

**Assumption:** Fixed fps.

**Currently wrong?** Conditional. If fps is correct (post-fix: yes), duration_ms is
correct as a clip-level value. On VFR containers with variable spacing, the true
duration is `pts_time_s` of the last frame, not `frame_count / fps`.

**Error magnitude:** On the test file: `1000 * 240 / 15 = 16000ms`. ffprobe reports
`duration=16.144s = 16144ms`. Error: 144ms (0.9%). Small but nonzero.

`duration_ms` is stored in `ClipManifest` and summed in `session_d_run.py:496` for
`SessionManifest.duration_ms`. It is extracted in `d1_graph_build.py:317` via
`_get_manifest_fields()` but only emitted to audit JSONL at `:1954` and `:2481` — it
is NOT a computational input to D1 graph construction.

**Sidecar replacement:** `pts_time_s` of the last frame from the sidecar provides exact
clip duration.

**Blast radius:** Manifest metadata. No downstream computation depends on it.

### 2.24 Stage F — session_f_run derive_clip_frame_offset

**Location:** `src/bjj_pipeline/stages/export/session_f_run.py:88`.
```python
offset = derive_clip_frame_offset(mp4, session_start_dt, fps)
```

**Assumption:** Same scalar fps for all clips. Same function as §2.12.

**Currently wrong?** Yes. Same defect as §2.12. Applied in the Stage F source registry
to map session-level frame indices back to per-clip frame positions for export.

**Error magnitude:** Same as §2.12. Compounds with §2.10 (session fps resolution).

**Sidecar replacement:** Same as §2.12.

**Blast radius:** Wrong source clip selection and seek position for session-level exports.
Customer-visible: exported clip may come from wrong segment or wrong position within
segment.

---

## 3. Correct-Already Sites

| Site | Location | Why correct |
|------|----------|-------------|
| FrameIterator `CAP_PROP_POS_MSEC` | `frame_iterator.py:57-61` | Returns real container PTS on passthrough VFR. Empirically verified ±0.4ms vs sidecar. |
| quality.py velocity | `quality.py:310-325` | Uses FrameIterator timestamps. Correct on passthrough footage. |
| processor.py physics | `processor.py:404` | Same FrameIterator timestamps. Correct on passthrough footage. |
| c0_triggers velocity | `c0_triggers.py:56-57,67` | Same FrameIterator timestamps. Thresholds in physical units. |
| c0_scheduler `k_verify=30` | `c0_scheduler.py:82` | Pure frame count ("scan every 30th frame"). No fps semantics. |
| cli.py `fps=0.0` | `cli.py:239,282` | Placeholder for status/validate commands. Never reaches pipeline consumers. |
| d05_split.py | `d05_split.py` | No direct fps consumption. Reads pre-computed `speed_mps_k` from d0_bank (inherits error indirectly via §2.7). |
| d1_graph_build `duration_ms` | `d1_graph_build.py:1954,2481` | Emitted to audit JSONL only. Not a computational input. |

---

## 4. Sidecar Reachability

### 4.1 Nothing in `src/` currently reads `.timing.jsonl`

Confirmed via `grep -r "\.timing\.jsonl" src/` — zero matches.

### 4.2 Production path

The pipeline reads the video at `ingest_path` (the original file location under
`data/raw/nest/{gym_id}/{cam_id}/{date}/{hour}/`). `ensure_manifest()` stores this
path as `manifest.input_video_path`. The sidecar sits as a sibling
(`{name}.timing.jsonl` next to `{name}.mp4`). **Reachable** — derive sidecar path as
`Path(ingest_path).with_suffix('.timing.jsonl')`.

### 4.3 Processor service path (`run_local.sh`)

`services/processor/run_local.sh` runs Python directly with `PYTHONPATH`. No file
copying, moving, or symlinking. `processor.py` calls `run_pipeline(ingest_path=mp4_path)`
passing the discovered mp4 path as-is. **Reachable** — sidecar is co-located.

### 4.4 Sweep harness path

`tools/sweep/replay_tracker.py:44-46` opens videos at `data/raw/nest/_eval_gt/...`.
`tools/sweep/cache_detections.py` reads cached parquets; video path used only for fps
probe. `tools/sweep/run_stage_d.py:52` reads `baseline_manifest.fps`. Sidecars would
need to be co-located at the `_eval_gt` path. **Reachable if present** — but existing
`_eval_gt` footage is pre-fix CFR (no timing sidecars exist for it). New GT footage
(post CP-R8) will have sidecars.

### 4.5 Naming collision: "sidecar"

The word "sidecar" is used in two unrelated contexts:

1. **Recorder timing sidecar**: `{name}.timing.jsonl` — per-frame capture timestamps.
   Defined by `docs/reference/sidecar_contract.md`.

2. **Stage A artifact sidecars**: `outputs.py:519` ("Write keypoints sidecar" →
   `keypoints.parquet`) and `outputs.py:529` ("Write histogram sidecars" →
   `color_histograms.parquet`, `tracklet_histogram_summaries.parquet`).

These are different files in different locations with different schemas. Future code
and documentation should disambiguate: "timing sidecar" vs "Stage A sidecars" (or
"Stage A auxiliary parquets").

---

## 5. Pipeline Validation Framework

### 5.1 visualize.py — match preview rendering

**Location:** `src/pipeline_validation/stage_f/visualize.py:327,351,408`.

```python
cap_fps = cap.get(cv2.CAP_PROP_FPS)                    # :327
writer = cv2.VideoWriter(..., cap_fps, ...)              # :351
timestamp_ms = int(fi * (1000.0 / cap_fps))              # :408
```

**Assumption:** `CAP_PROP_FPS` is correct; `frame_index * (1000 / fps)` gives real
elapsed time.

**Currently wrong?** Same defect class as §2.16. On passthrough VFR footage where
`CAP_PROP_FPS` returns 15.0, the writer fps is correct. But `timestamp_ms = fi *
(1000/cap_fps)` synthesizes timestamps from a scalar fps rather than reading real PTS,
so it diverges from the sidecar on frames with gaps or mode switches.

**Error magnitude:** On uniform-spacing frames: correct. On gap frames: synthesized
timestamp is wrong by the gap duration (~67ms per gap). Cumulative: ~0.5s per 1000
frames on FP7oJQ (8% gap rate).

**Why this matters:** `visualize.py` produces the `match_preview.mp4` used for
TB-EVAL-3 evaluation. The timestamp overlay text is wrong on gap frames. More
importantly, GT2ACTUALS is the instrument for the checkpoint-2 A/B validation of
post-fix footage. If the evaluation ruler carries the same fps assumption as the
pipeline, A/B comparisons on VFR footage may not detect timing-related regressions.

**Fix priority:** P3. Evaluation tooling — not customer-facing. But should be fixed
before the CP-R8 A/B validation to ensure the ruler is clean.

**Sidecar replacement:** Read `pts_time_s` from the sidecar for the rendered frame
instead of synthesizing `fi * (1000/cap_fps)`.

### 5.2 GT2ACTUALS

`src/pipeline_validation/gt2actuals/` — confirmed: zero references to `fps`,
`frame_rate`, or `CAP_PROP_FPS`. GT2ACTUALS operates on `frame_index` joins, not
timestamps. **Not affected.**

---

## 6. ClipManifest Schema Impact

**Current:** `ClipManifest.fps: float` — single scalar per clip
(`f0_manifest.py:144`).

**What the eventual fix needs (note only — no change in this audit):**

The sidecar provides several fields that the manifest should carry for consumers that
cannot or should not read the sidecar directly:

| Field | Source | Purpose |
|-------|--------|---------|
| `nominal_dt_s` | Sidecar `_meta` | Reference interval. `1.0 / nominal_dt_s` replaces `fps`. |
| `source_pts` | Sidecar `_meta` | Validity gate for all derived timing. |
| `is_bimodal` | Sidecar `_meta` | Advisory flag for variable-dt consumers. |
| `timing_sidecar_path` | Derived | Path to `.timing.jsonl` for per-frame consumers. |

`SessionManifest` additionally needs per-clip or per-camera fps to replace the single
scalar (§2.10). This is a schema change that will need to be coordinated across all
session-level consumers.

---

## 7. Empirical Check Results

### 7.1 Post-fix passthrough VFR mp4

File: `PPDmUg-20260807-102005.mp4` (passthrough, source_pts=true, 15fps)

| Check | Result |
|-------|--------|
| ffprobe `r_frame_rate` | `15/1` — **correct** |
| ffprobe `avg_frame_rate` | `15/1` — **correct** |
| OpenCV `CAP_PROP_FPS` | `15.0` — **correct** |
| OpenCV `CAP_PROP_POS_MSEC` vs sidecar `pts_time_s` | ±0.335ms max — **effectively identical** |

### 7.2 Pre-fix CFR mp4

File: `J_EDEw-20260318-200015.mp4` (CFR, arrival-PTS, 30fps)

| Check | Result |
|-------|--------|
| ffprobe `r_frame_rate` | `30/1` — correct for container |
| OpenCV `CAP_PROP_FPS` | `30.0` — correct for container |

### 7.3 Stage E timestamp fallback

`person_tracks.parquet` from `_eval_gt_baseline_v2`: 76,110 rows, `timestamp_ms`
null count = 0. The fps fallback at `run.py:444-445` is dead code — `frame_to_ts` map
is always fully populated.

### 7.4 Stage A and Stage D time bases

**Do Stage A and Stage D currently compute different time bases for the same clip?**

**Yes, structurally, but not in practice on current footage.**

- Stage A: `FrameIterator.timestamp_ms` from `CAP_PROP_POS_MSEC`. On passthrough VFR:
  real container PTS (uniform ~67ms on 15fps footage).
- Stage D: `dt_s = frame_delta / manifest.fps`. On the same 15fps footage: `1/15 = 0.0667s`.

These agree within ±0.4ms per frame on uniform-spacing segments. They diverge on gap
frames: Stage A's `CAP_PROP_POS_MSEC` reports the real PTS (~133ms gap), while Stage D
computes `1/15 = 67ms` (wrong by 67ms). **The divergence is real but localized to gap
frames (~8% on FP7oJQ, ~0.45% on PPDmUg).**

---

## 8. What This Audit Did Not Cover

- **Recorder-side code** (`services/nest_recorder/recorder/diag_v*.sh`) — out of scope
  per brief. The sidecar contract is the interface.
- **Training pipeline** (`src/training_pipeline/`) — no fps consumption in production
  pipeline path.
- **Mobile/web apps** (`app_mobile/`, `app_web/`) — consume Supabase `clips` table
  values (§2.17 blast radius noted) but app-side logic not audited.
- **Supabase schema beyond `clips` table** — `clips` table column types confirmed for
  §2.17; wider schema not audited.
- **Actual re-measurement of error magnitudes on clean footage** — blocked on CP-R8 GT.
  All quantitative error estimates in this document are computed from the timing model
  (CP-R11) and the sidecar contract, not from measured pipeline output.
- **Whether `CAP_PROP_FPS` is correct on ALL VFR containers** — verified on one
  PPDmUg passthrough segment. Edge cases (truncated segments, very short segments,
  bimodal segments) not tested.
- **boxmot Kalman internals beyond `dt=1`** — did not audit whether other boxmot
  internal computations (ECC, CMC, appearance) have fps assumptions.

---

## 9. CATALOG Retagging

`docs/project_instructions_proposed.md` is tagged **CURRENT** in `docs/CATALOG.md` but
was generated 2026-07-29 (DOC-SYNC-3) and has not been updated since. It should be
retagged **HISTORICAL** — accurate for its date, not current state.
