# Roll Tracker — claude.ai Project Instructions (proposed replacement)

*Generated 2026-07-26. Paste into claude.ai Project Instructions to replace the
2026-04-05 / CP20 version.*

## Project

BJJ gym SaaS pipeline. Nest cameras → YOLO+BoT-SORT tracking → AprilTag identity →
ILP stitching → per-athlete match clips → Supabase → Flutter app.
**Repo:** github.com/bryan-rt/roll_tracker | **Branch:** `services_uploader` | **Python 3.12**

## Current Status (2026-07-26)

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).

**Canonical correct_id: 33.9%** (clip-level, val-split, J_EDEw, post-D0.5-split).
Pre-split baseline was 40.5%. The −6.6pp drop is D0.5 Tier 3 fragmentation (net-negative:
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

**A↔D contract mismatch:** Length-proportional `unexplained_tracklet_penalty` assumes
tracklet purity that Stage A doesn't deliver. Impure tracklets are disproportionately LONG.

**Purity proxies:** `max_displacement` AUC 0.82–0.85 (same-color robust). Masked appearance
+0.09 AUC. **CRITICAL CEILING: smooth-motion + same-color drifts are 58–94% of impurity
and invisible to BOTH proxies.**

**Production color defects:** Crops are unmasked rectangles (contaminated). Pose-guided torso
crop is DEAD CODE (detection-only model → keypoints NaN). Masked H+S+V validated at AUC 0.907
but NOT productionized.

**Recorder timing:**
- Per-frame `.timing.jsonl` sidecar shipped to production (RECORDER-SIDECAR-1). Under
  arrival-PTS, pts_time_s is a ±500ms nearest-neighbor approximation (mismatch is normal).
- **RTSP stream carries TRUE capture timestamps** (source PTS uniform 33ms/67ms depending on
  session). Production recorder DISCARDS them via `-use_wallclock_as_timestamps 1`.
- **RTCP absent** across all cameras, both TCP/UDP. Absolute camera clock unavailable.
- **Tier-2 cross-camera alignment: ±14–56ms** (source PTS + host-clock lower envelope).
- **Stream fps VARIES per session** (15fps and 30fps both observed). SDP unreliable.
  **Do not hardcode fps anywhere.**
- `frame_index` is a sequential `cap.read()` counter, NOT PTS-derived.

## Active Decisions Log

| Decision | Status | Notes |
|----------|--------|-------|
| CP-EVAL-1: Eval instrument freeze | **Active** | Hungarian IoU 0.5. v1.0 spec frozen. |
| CP-GT2ACTUALS: Dense error map | **Complete** | Per-(frame, gt_track_id) join with stage-attributed jumps. Split-family lookup fix (CP-3). Module: `src/pipeline_validation/gt2actuals/`. |
| Lever → Stage A tracker drift | **Active** | 41% of vid2 jumps. Stage A is the #1 damage source — upstream of solver, not addressable by appearance or D0.5. |
| D0.5 splitting: NET-NEGATIVE | **Measured** | 35 correct / 317 false (vid2). Tier 3 owns 79% damage. T3 precision 7.3%. Disable recommended. |
| CP-HSV-V: H+S+V extension | **Shipped — possibly NET-HARM** | Improved separability but amplified low-precision Tier 3. Live 33.9% vs pre-split 40.5%. |
| Appearance in solver | **DEMOTED** | No per-frame signal discriminates false from correct splits. Addresses ≤26% misstitch share. |
| BoT-SORT stock defaults | **Active** | `tracker.params = {}`. OFAT tb screen complete: stock optimal. `match_thresh` sweep deferred until A↔D fix. |
| A↔D contract mismatch | **Diagnosed, not fixed** | Length-proportional penalty anti-correlated with reliability. Fix: discount or displacement-based D0.5 split. |
| Purity proxies | **Measured** | max_displacement AUC 0.82-0.85. Smooth+same-color blind (58-94%). |
| Color-signal defects | **Diagnosed, not fixed** | Unmasked rectangles, dead torso crop, averaging destroys modality. Masked validated, not shipped. |
| RECORDER-SIDECAR-1 | **Active** | Per-segment .timing.jsonl alongside every mp4. Under arrival-PTS: ±500ms NN approximation. COLLECTION ONLY. |
| Source PTS discovery | **Diagnosed, not adopted** | RTSP carries true capture timestamps. Production still discards them. Adopting would make sidecar exact. |
| RTCP absent | **Definitive** | 0 sender reports, all cameras, both TCP/UDP. Absolute camera clock unavailable from stream. |
| Stream fps varies | **Confirmed** | 15fps and 30fps both observed. SDP unreliable. Use per-clip measured fps. |
| CP-TAG-4a | **RETRACTED (SWEEP-3b)** | "+22.7pp" was a metric-basis artifact. Code in repo, effect UNKNOWN. |
| CP-REID-1: BoT-SORT ReID | **Rejected — DOMAIN GAP** | Generic osnet rejected for domain gap, NOT color-blindness. V-win doesn't reopen. |
| CP-SPLIT-VALIDATE | **Complete** | All D0.5 splits GT-validated. Systemic low precision. No threshold separates correct from spurious. |

## Overturned Conclusions

1. **D0.5 splitting: helpful → NET-NEGATIVE.** Net −282 on vid2 (35/317).
2. **CP-HSV-V: first improvement → possibly NET-HARM.** −6.6pp from low-precision T3.
3. **Appearance in solver as THE lever → DEMOTED.** Stage A drift dominates (41%).
4. **"Camera is 30fps" → CORRECTED.** Varies 15–30fps per session. Use measured fps.
5. **Sidecar "exact timing" → QUALIFIED.** ±500ms under arrival-PTS. Exact under source-PTS (not yet adopted).
6. **CP-TAG-4a "+22.7pp" → RETRACTED.** Same-artifact frame-selection effect.

## Pending / TBD (priority order)

1. **[LIVE BUG] BoT-SORT `frame_rate` = per-clip MEASURED fps.** Hardcoded/assumed 30, streams deliver 15–30. track_buffer lifespan wrong.
2. **[AUDIT] Frame-rate & cross-camera assumptions across Stages A–F.** Enumerate every fixed-fps / uniform-spacing / synchronized-camera assumption.
3. Recorder: adopt source PTS in production → exact sidecar.
4. A↔D contract fix: reliability-discount penalty or displacement-based D0.5 split.
5. Per-clip measured-fps denominator for motion metrics.
6. D0.5 Tier 3: disable (recover −6.6pp) or redesign.
7. Cross-camera sync via Tier-2 with drift correction.
8. Productionize masked histograms (+0.09 AUC).
9. Stage A `match_thresh` sweep — AFTER A↔D fix.

## Architecture (unchanged)

Monorepo: `src/bjj_pipeline/` (stages A→F), `src/calibration_pipeline/`,
`src/training_pipeline/`, `src/pipeline_validation/`, `services/` (nest_recorder,
processor, uploader), `backend/supabase/`, `app_mobile/`, `app_web/`.

Phase 1/2 parallelism: A+C parallel, D+E+F sequential. No cross-stage imports.
Option B undistort-on-projection. Pydantic v2, Loguru, Rich, Typer. Parquet for
tabular data. NumPy < 2 (Torch ABI).

## Working Methodology

Three-pass protocol (explore → specify → execute). Evidence-driven design.
Ambiguity protocol: surface conflicts in Pass 1, don't resolve silently.
Metric-basis discipline: no correct_id without camera set, frame range, pipeline state.
