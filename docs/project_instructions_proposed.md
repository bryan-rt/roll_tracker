# Roll Tracker — claude.ai Project Instructions (proposed replacement)

*Generated 2026-07-29. Paste into claude.ai Project Instructions to replace the
2026-07-26 / DOC-SYNC-2 version.*

## Project

BJJ gym SaaS pipeline. Nest cameras → YOLO+BoT-SORT tracking → AprilTag identity →
ILP stitching → per-athlete match clips → Supabase → Flutter app.
**Repo:** github.com/bryan-rt/roll_tracker | **Branch:** `services_uploader` | **Python 3.12**

## Current Status (2026-07-29)

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).

**Canonical correct_id: 33.9%** (clip-level, val-split, J_EDEw, post-D0.5-split).
Pre-split baseline was 40.5%. The -6.6pp drop is D0.5 Tier 3 fragmentation (net-negative:
35 correct / 317 false splits on vid2). Sweep baseline (replay path): 30.7%.

**GT2ACTUALS subsystem live** (`src/pipeline_validation/gt2actuals/`): permanent per-frame
GT-grounded error map with inline jump detection and D0.5 net-effect reconciliation.
Dense manifest (stride-1): J_EDEw vid1 3,001 frames, vid2 4,491 frames.

**Identity-corruption lever journey (5 inversions):** D4 emission → ILP stitch → D1 group
formation → detection under-segmentation → **Stage A tracker drift (current, 41% of vid2
jumps)**. Group handling 33%, solver misstitch 26%.

**BoT-SORT is UNTUNED:** `tracker.params = {}` — stock boxmot pedestrian defaults
(track_buffer=30, match_thresh=0.8). Never tuned for BJJ overhead fisheye grappling.
OFAT track_buffer screen complete: stock tb=30 is optimal (every deviation degrades).

**A-D contract mismatch:** Length-proportional `unexplained_tracklet_penalty` assumes
tracklet purity that Stage A doesn't deliver. Impure tracklets are disproportionately LONG.

**Purity proxies:** `max_displacement` AUC 0.82-0.85 (same-color robust). Masked appearance
+0.09 AUC. **CRITICAL CEILING: smooth-motion + same-color drifts are 58-94% of impurity
and invisible to BOTH proxies.**

**Production color defects:** Crops are unmasked rectangles (contaminated). Pose-guided torso
crop is DEAD CODE (detection-only model → keypoints NaN). Masked H+S+V validated at AUC 0.907
but NOT productionized.

**Recorder hardening (RECORDER-RELIABILITY-1/2, 2026-07-28):**
- Five production reliability fixes: RTSP socket timeout (top fix for recording gaps),
  stop_stream before regenerating (root cause of 404 cascade), access token refresh per
  attempt, failure-type-aware backoff, backgrounded sidecar extraction.
- API quota awareness: optimistic URL reuse, conditional stop_stream, 429 backoff,
  consecutive failure escalation, cross-camera quota coordination. SDM quota: 10 QPM per
  user per project shared across all cameras. Traffic cut 64%.
- ffmpeg option fragility: `-stimeout` removed in ffmpeg 7.x → use `-timeout`. Pin Debian.

**Recorder timing:**
- Per-frame `.timing.jsonl` sidecar shipped to production (RECORDER-SIDECAR-1). Under
  arrival-PTS, pts_time_s is a +/-500ms nearest-neighbor approximation (mismatch is normal).
- **RTSP stream carries TRUE capture timestamps** (source PTS uniform ~33ms/~67ms depending on
  session, 1.21ms stdev). Production recorder DISCARDS them via `-use_wallclock_as_timestamps 1`.
- **RTCP absent** across all cameras, both TCP/UDP. Absolute camera clock unavailable.
- **Absolute timing: ESTIMATED +/-14-56ms** (host-clock lower envelope + per-camera drift
  correction; FP7oJQ -603 ppm, linearly correctable).
- **Stream fps VARIES per session** (15fps and 30fps both observed). SDP unreliable.
  **Do not hardcode fps anywhere.**
- `frame_index` is a sequential `cap.read()` counter, NOT PTS-derived.
- **Sidecar schema:** `frame_index` (join key), `pts_time_s`, `host_arrival_s`, `input_n`
  (consecutive same = fabricated duplicate). Metadata: measured fps (4dp), lower-envelope
  offset, drift ppm, mismatch flag.

**CRITICAL CAVEAT: prior GT measurements made on CORRUPTED FOOTAGE.** All existing GT
footage predates recorder fixes — recorded under bursty-arrival timestamps causing
large-scale dup/drop (35% mismatch in one case). Duplicate frames → false zero-motion in
Kalman filter; dropped frames → false teleports. An unknown fraction of the 41% "Stage A
tracklet_drift" may be recorder-injected. Hold GT2ACTUALS drift attribution, purity-proxy
results, and Stage A sweep LOOSELY until re-measured on clean footage.

## Active Decisions Log

| Decision | Status | Notes |
|----------|--------|-------|
| CP-EVAL-1: Eval instrument freeze | **Active** | Hungarian IoU 0.5. v1.0 spec frozen. |
| CP-GT2ACTUALS: Dense error map | **Complete** | Per-(frame, gt_track_id) join with stage-attributed jumps. Split-family lookup fix (CP-3). Module: `src/pipeline_validation/gt2actuals/`. |
| Lever → Stage A tracker drift | **Active** | 41% of vid2 jumps. Stage A is the #1 damage source — upstream of solver, not addressable by appearance or D0.5. **Caveat:** fraction may be recorder-injected (corrupted footage). |
| D0.5 splitting: NET-NEGATIVE | **Measured** | 35 correct / 317 false (vid2). Tier 3 owns 79% damage. T3 precision 7.3%. Disable recommended. |
| CP-HSV-V: H+S+V extension | **Shipped — possibly NET-HARM** | Improved separability but amplified low-precision Tier 3. Live 33.9% vs pre-split 40.5%. |
| Appearance in solver | **DEMOTED** | No per-frame signal discriminates false from correct splits. Addresses <=26% misstitch share. |
| BoT-SORT stock defaults | **Active** | `tracker.params = {}`. OFAT tb screen complete: stock optimal. `match_thresh` sweep deferred until A-D fix. |
| A-D contract mismatch | **Diagnosed, not fixed** | Length-proportional penalty anti-correlated with reliability. Fix: discount or displacement-based D0.5 split. |
| Purity proxies | **Measured** | max_displacement AUC 0.82-0.85. Smooth+same-color blind (58-94%). |
| Color-signal defects | **Diagnosed, not fixed** | Unmasked rectangles, dead torso crop, averaging destroys modality. Masked validated, not shipped. |
| RECORDER-RELIABILITY-1/2 | **Complete** | Five reliability fixes + API quota awareness. Traffic cut 64%. Evidence: `docs/evidence/recorder_reliability_{1,2}/`. |
| SOURCE-PTS-1 | **Active — opt-in, ready for default** | `SOURCE_PTS=1` preserves true capture timestamps. Exonerated as cause of exits. Ready for default adoption. |
| RECORDER-SIDECAR-1 | **Active** | Per-segment .timing.jsonl alongside every mp4. Under arrival-PTS: +/-500ms NN approximation. COLLECTION ONLY. |
| Dup/drop: TWO mechanisms | **Diagnosed** | Mechanism 1 (arrival jitter) FIXED by source PTS. Mechanism 2 (CFR-target != capture rate) STILL PRESENT. 0.0%-vs-8% contradiction open. |
| Prior GT = corrupted footage | **Active — hold loosely** | All GT predates recorder fixes. Unknown recorder-injected fraction in drift attribution. Re-measure on clean footage. |
| RTCP absent | **Definitive** | 0 sender reports, all cameras, both TCP/UDP. Absolute camera clock unavailable from stream. |
| Stream fps varies | **Confirmed** | 15fps and 30fps both observed. SDP unreliable. Use per-clip measured fps. |
| CP-TAG-4a | **RETRACTED (SWEEP-3b)** | "+22.7pp" was a metric-basis artifact. Code in repo, effect UNKNOWN. |
| CP-REID-1: BoT-SORT ReID | **Rejected — DOMAIN GAP** | Generic osnet rejected for domain gap, NOT color-blindness. V-win doesn't reopen. |
| CP-SPLIT-VALIDATE | **Complete** | All D0.5 splits GT-validated. Systemic low precision. No threshold separates correct from spurious. |

## Overturned Conclusions

1. **D0.5 splitting: helpful → NET-NEGATIVE.** Net -282 on vid2 (35/317).
2. **CP-HSV-V: first improvement → possibly NET-HARM.** -6.6pp from low-precision T3.
3. **Appearance in solver as THE lever → DEMOTED.** Stage A drift dominates (41%).
4. **"Camera is 30fps" → CORRECTED.** Varies 15-30fps per session. Use measured fps.
5. **Sidecar "exact timing" → QUALIFIED.** +/-500ms under arrival-PTS. Exact under source-PTS (not yet adopted).
6. **CP-TAG-4a "+22.7pp" → RETRACTED.** Same-artifact frame-selection effect.
7. **RELIABILITY-1 "dup/drop resolved" → QUALIFIED.** Source PTS fixed mechanism 1 (arrival jitter) but mechanism 2 (CFR-target != capture rate) STILL PRESENT.
8. **Prior GT measurements on CORRUPTED FOOTAGE.** All GT predates recorder fixes (bursty dup/drop, 35% mismatch). Unknown recorder-injected fraction in drift/purity/sweep conclusions. Hold loosely.

## Pending / TBD (priority order)

1. **[LIVE BUG] BoT-SORT `frame_rate` = per-clip MEASURED fps.** Hardcoded/assumed 30, streams deliver 15-30. track_buffer lifespan wrong.
2. **[AUDIT] Frame-rate & cross-camera assumptions across Stages A-F.** Enumerate every fixed-fps / uniform-spacing / synchronized-camera assumption.
3. Recorder: adopt source PTS in production → exact sidecar.
4. A-D contract fix: reliability-discount penalty or displacement-based D0.5 split.
5. Per-clip measured-fps denominator for motion metrics.
6. D0.5 Tier 3: disable (recover -6.6pp) or redesign.
7. Cross-camera sync via Tier-2 with drift correction.
8. Productionize masked histograms (+0.09 AUC).
9. Stage A `match_thresh` sweep — AFTER A-D fix.
10. Resolve 0.0%-vs-8% duplicate-measurement contradiction.
11. Pin Debian version in Dockerfile + N_CAMERAS div-by-zero guard.

**Forward pipeline direction (PLANNED, not built):**
1. Dynamic fps replaces hardcoded 30 everywhere (BoT-SORT, motion metrics, Stage E windows, cross-camera timing). Source: per-clip measured fps from sidecar.
2. Padded/duplicate frames: emit normally but flag, exclude from Kalman update. Detection: consecutive same `input_n`. Keeps `frame_index` contiguous.
3. Consumer split: per-frame dt for own code; per-clip scalar fps for BoT-SORT (boxmot hardcodes unit Kalman step).
4. GT-join decision: score or exclude pipeline-flagged duplicates in GT2ACTUALS.
5. A/B validation on same new footage: old logic vs new logic behind config flag, same CVAT GT.

**Remaining recorder work (open checkpoint):**
1. Validate reuse path, token refresh past ~25-min, sustained coverage, no 429s.
2. Resolve 0.0%-vs-8% contradiction.
3. Make SOURCE_PTS=1 the default.
4. CFR-target decision: per-clip target or accept padding + pipeline-side flagging.
5. N_CAMERAS div-by-zero guard.
6. Pin Debian version in Dockerfile.

## Architecture (unchanged)

Monorepo: `src/bjj_pipeline/` (stages A-F), `src/calibration_pipeline/`,
`src/training_pipeline/`, `src/pipeline_validation/`, `services/` (nest_recorder,
processor, uploader), `backend/supabase/`, `app_mobile/`, `app_web/`.

Phase 1/2 parallelism: A+C parallel, D+E+F sequential. No cross-stage imports.
Option B undistort-on-projection. Pydantic v2, Loguru, Rich, Typer. Parquet for
tabular data. NumPy < 2 (Torch ABI).

## Working Methodology

Three-pass protocol (explore → specify → execute). Evidence-driven design.
Ambiguity protocol: surface conflicts in Pass 1, don't resolve silently.
Metric-basis discipline: no correct_id without camera set, frame range, pipeline state.
