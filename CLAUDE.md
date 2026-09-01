# CLAUDE.md — Roll Tracker

## Project

BJJ gym SaaS pipeline. Nest cameras → YOLO+BoT-SORT tracking → AprilTag identity →
ILP stitching → per-athlete match clips → Supabase → Flutter app.
**Repo:** github.com/bryan-rt/roll_tracker | **Branch:** `services_uploader` | **Python 3.12**

## Working Methodology

**Three-pass protocol (mandatory for all non-trivial tasks):**
1. **Pass 1 — Explore** (Plan Mode: shift+tab ×2): Read Task Brief, explore relevant files,
   identify conflicts. ⏸ STOP — summarize and wait for approval.
2. **Pass 2 — Specify** (Plan Mode continues): Plan exact changes, verify naming/contracts
   against live code. ⏸ STOP — present plan and wait for approval.
3. **Pass 3 — Execute**: Implement, test, update CLAUDE.md if architecture changed,
   commit+push. ⏸ STOP — summarize and wait for review.

**Never skip a pause.** User approval gates each pass. Do not run Pass 2
immediately after Pass 1, and do not run Pass 3 immediately after Pass 2.

**Evidence-driven design:** Do not code from assumptions. When behavior is uncertain:
enhance logging → inspect real output → plan from evidence. Propose instrumentation
before fixes when root cause is unclear.

**Ambiguity protocol:** Surface naming conflicts, missing files, or uncovered architectural
questions in Pass 1. Do not resolve silently or guess.

## Monorepo Layout

```
src/bjj_pipeline/        # CV pipeline package (stages A→F, contracts, config, core)
src/calibration_pipeline/ # Gym setup: lens cal, H refinement, mat line detection
src/training_pipeline/    # Active learning: CVAT integration, fine-tuning, evaluation
src/pipeline_validation/  # Evaluation framework: detection, identity, match viz
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
configs/models/           # Per-model training manifests (model_id.yaml)
tools/                    # Training, evaluation, comparison, packaging scripts
docs/                     # CATALOG.md, decisions archive, checkpoints, evidence, guides, reference
.claude/rules/            # Domain-specific context (auto-loaded by path scope)
```

## Critical Constraints

- **NumPy < 2** — Torch ABI. Install ultralytics/boxmot with `--no-deps`.
- **Supabase is the exclusive integration hub** — no direct service-to-service communication.
- **Phase 1/2 parallelism boundary (NON-NEGOTIABLE)** — A+C parallel, D+E+F sequential.
- **No cross-stage imports** — stages communicate only via F0 contracts + filesystem.
- **Option B undistort-on-projection** — `project_to_world()` is the only permitted
  pixel→world path. No stage calls homography directly.

## Coding Conventions

- Stage contract: `run(config: dict, inputs: dict) -> dict`
- Pydantic v2 for data models. Loguru for logging. Rich for CLI. Typer for CLI defs.
- Parquet for tabular data. JSONL for audit/event streams. Type hints everywhere.
- Debug artifacts → `outputs/<clip_id>/_debug/`. Never pollute stage output dirs.
- Paths via `ClipOutputLayout` and env vars — no hardcoding.

## Config Resolution

`default.yaml` → `cameras/<cam_id>.yaml` → `cameras/<cam_id>/homography.json` → `--config` CLI overlay

## Current Status

*Last updated 2026-07-29.*

Pipeline A→F verified E2E. Session pipeline validated (3-camera, 35/36 clips).

**Evaluation baseline (CP-SPLIT-1 active, CP-EVAL-1 frozen instrument v1.0):**

| Camera | present | misattrib | no_det | untracked | d3_drop |
|--------|---------|-----------|--------|-----------|---------|
| FP7oJQ | 21.0% | 51.0% | 12.2% | 10.7% | 4.6% |
| J_EDEw | 12.2% | 54.0% | 11.7% | 13.5% | 7.9% |
| PPDmUg | 15.0% | 61.4% | 13.4% | 8.1% | 0.0% |

Ceiling without new models: ~35-40% present. Primary blocker: detection
under-segmentation (one box covering two grappling people). See CP7 investigation.

**CP20:** YOLOv8n-pose model, isolation gate, HSV color histograms, Tier 3 histogram
cross-camera evidence. Stage A outputs 3 new sidecars: keypoints.parquet,
color_histograms.parquet, tracklet_histogram_summaries.parquet.
- Camera geometry analysis tool complete (v6 pose decomposition, 4-phase)
- Lens calibration bounds fix applied (fixed-f candidate sweep)
- H coordinate space verified as undistorted pixel space
- Calibration wizard re-run for all 3 cameras with updated lens cal
- Cross-camera agreement verified (sub-cm, 9mm worst-case)
- ROI mask union fix: brief written, not yet applied (parked)
- **V-channel FIXED (CP-HSV-V):** Production histogram extended from H+S (144-dim)
  to H+S+V (864-dim, 18×8×6). `histogram.py` now uses channels [0,1,2] with
  `HIST_V_BINS=6`. `bhattacharyya_distance` compares flat (shape-invariant).
  Black vs white now separable (was distance ~0, now ~1.0).
  Evidence: `docs/evidence/cp_raster_plate_2/`.
- **D0.5 split precision crisis (CP-SPLIT-VALIDATE):** GT-validation of ALL D0.5
  splits revealed systemic low precision across all tiers:
  - Tier 3 (histogram): post-V new splits are 2.4% correct / 77.5% spurious.
    Pre-V T3 was also low (4-20%). Threshold sweep shows no single cutoff
    separates correct from spurious (overlapping distributions).
  - Tier 2 (kinematic): 6-22% precision — also mostly spurious.
  - Tier 1 (speed cap): too few to characterize.
  - **Spurious T3 shape:** 61% motion-correlated shadow/pose artifacts (V is
    noisy during motion — person rotates through shadow while moving), 28%
    sustained-same-person, 11% single-point blips.
  - **Design finding:** V (brightness) is unreliable during high-motion frames.
    The 2× speed kinematic corroboration gate CANNOT catch this because the
    person IS moving — the noise and the gate fire on the same condition.
  - **k-distribution:** k=2 swaps dominate (27% of tracklets), k≥3 rare (4%).
  - **Change-point feasibility:** mixed — 30% of impure tracklets show
    segmentable clean-point structure. Appearance alone is insufficient.
  - **Implication:** D0.5 Tier 3 needs a fundamental redesign, not a threshold
    bump. Motion-aware channel weighting (reduce V weight when speed is high)
    is the leading design hypothesis.
  Evidence: `docs/evidence/cp_split_validate/`.

**CP-GT2ACTUALS (completed 2026-06-10):** Dense GT-to-actuals error map with
GT-grounded jump detection + D0.5 net-effect reconciliation.
- Module: `src/pipeline_validation/gt2actuals/` (dense_join, node_gt_set, jumps)
- CLI: `python -m pipeline_validation gt2actuals --manifest-path <path>`
- Dense manifest (stride-1): J_EDEw vid1 3,001 frames, vid2 4,491 frames
- **Split-family lookup fix (CP-3):** 88% of vid2 no_id was a join artifact —
  `_resolve_tracklet_id` pointed to wrong D0.5 products because the solver
  re-stitches them. Fixed with family-aware fallback. Vid2 no_id: 58% → 6.9%.
- **Signal_trace has the SAME bug (CP-3.5):** ~~locked canonical numbers
  (40.5%/63.2%) are NOT biased~~ **RETRACTED (SWEEP-3b):** both 40.5% and 63.2%
  were computed from pre-CP-TAG-4a `person_tracks` (Jun 7 13:05); 63.2% is a
  frame-selection artifact, not a code-change effect. Freshened eval_gt baseline
  is 32.5% combined. See `tools/sweep/diagnostics/blast_radius_check.md`.
- **D0.5 net-negative on ALL cameras (CP-4+5):** vid2 (authoritative, 99.4%
  classified): 35 correct / 317 false splits (net -282). Tier 3 owns 79% of
  damage. FP7oJQ/PPDmUg thin-classification (5.8%/33.3%) — direction only.
- **Stage attribution (CP-6):** Stage A (tracklet_drift) is the #1 damage
  source at 41% of vid2 jumps. Group handling 33%, solver misstitch 26%, D0.5
  false_split jumps 0% (damage is indirect via fragmentation). HSV cannot
  discriminate false from correct splits (Bhattacharyya 0.035 vs 0.040).
  False splits are 82% isolated (color available but not discriminative).
  Disabling Tier 3 removes 79% of D0.5 damage at cost of 19 correct splits.
- Evidence: `docs/evidence/cp_gt2actuals_{1,3,3_5,4_5,5_5,6}/`

**Stage A Tuning Sweep (active):** End-to-end BoT-SORT tracker param sweep harness.
Detection cache → tracker replay → Stage D rerun → GT2ACTUALS measurement.
Tools at `tools/sweep/`. Detection cache at `outputs/_sweep/detection_cache/`.
Results append to `outputs/_sweep/results.jsonl`.
- **Freshened eval_gt baseline:** 32.5% combined correct_id (vid1 39.3%, vid2 27.5%).
  D0→D4 re-run on current code against original Stage A tracklets. Confirmed by
  both standard gt2actuals CLI and sweep harness. The former 34.7% was stale
  (mixed-provenance D2-D4 artifacts from Jun 7 pre-CP-TAG-4a code). See
  `tools/sweep/diagnostics/blast_radius_check.md`.
- **Sweep baseline:** Stock params through replay path gives 30.7% combined
  (vid1 34.1%, vid2 28.2%). The ~2pp gap vs freshened 32.5% is an environment
  artifact (likely boxmot/OpenCV version drift between Jun 9 Stage A production
  and current replay — see `tools/sweep/diagnostics/gap_explanation.md`).
  98.9% structural agreement in tracklet transitions; gap is fixed offset, not
  parameter-dependent. All sweep deltas are internally consistent.
  **Comparison rule:** sweep results must be compared against the 30.7% sweep
  baseline only (same harness path). Never compare sweep deltas against the
  32.5% freshened eval_gt baseline (different measurement path).
- Deterministic (verified: identical parquets across runs).
- ~7min/clip (replay ~20s + Stage D ~5-6min + GT2ACTUALS ~30s).
- Tag hint handling: identity_hints.jsonl tracklet_ids remapped via detection_id
  join; `tag_hint_dropped` flag surfaced in sweep summary when tag anchor lost.
- **OFAT track_buffer screen (complete):** Screened tb={5,10,15,20,30,45,60}.
  Stock default (tb30) is the best value in the grid — every deviation degrades
  correct_id. Lower values (-3 to -5pp) fragment tracklets without helping the
  solver. Higher values degrade gently (-0.4 to -1.5pp) as misstitch rises. No
  flags fired. Initial "break, don't guess" hypothesis for track_buffer was
  refuted. See `tools/sweep/diagnostics/ofat_track_buffer_results.md`.
- **Glob collision bug fix (SWEEP-3b):** `run_gt2actuals.py` symlink directory
  scoped to `_gt2a/<run_id>/` to prevent cross-run glob matches.

**Identity-corruption lever journey (5 inversions):**
D4 emission → ILP stitch → D1 group formation → detection under-segmentation →
**Stage A tracker drift (current, 41% of vid2 jumps)**. Each step was falsified by
evidence. Full history in the CP-PURITY arc summary below. The current lever is
Stage A tracking quality (BoT-SORT with stock untuned parameters).

**BoT-SORT is UNTUNED:** `tracker.params` is `{}` — the tracker runs entirely on stock
boxmot pedestrian defaults (track_buffer=30, match_thresh=0.8, new_track_thresh=0.6,
track_high/low=0.5/0.1). Never tuned for BJJ overhead fisheye grappling.

**Variable-dt tracker (Piece 11, `src/bjj_pipeline/tracking/`):**
Subclass (not fork) of boxmot's `KalmanFilterXYWH` and `BotSort`. Toggle:
`stages.stage_A.tracker.variable_dt: true` (default false). When enabled:
- KF rebuilds `_motion_mat` per step from `dt_s / nominal_dt_s` ratio (velocity stays
  in px/nominal-frame, all noise constants remain calibrated). Normal frame → 1.0,
  gap → ~2.0, 30fps block → ~0.5.
- Track lifetime uses wall-time `max_lost_seconds` (default 2.0s = today's behavior)
  instead of frame-count `max_time_lost`. Eliminates `frame_rate` entirely.
- Both KF sites replaced: `self.kalman_filter` AND `STrack.shared_kalman` (V5 trap).
- Runtime assertion proves `STrack.shared_kalman` is the subclassed filter.
- `dt_s` from schema-5 sidecar via `f0_sidecar.load_sidecar()`. No fallback.
- `dt_s=0.0` valid (same-PTS frames on bimodal segments) → ratio 0.0 → position no-op.
- Process-noise dt-scaling (sqrt(dt) for continuous-time white noise) is a recorded
  follow-up — under the ratio formulation it is genuinely second-order.
**boxmot internals dependency:** Subclass depends on `shared_kalman` (class attr),
`_update_track_states`, `_motion_mat`, `predict`/`multi_predict` signatures, `frame_count`,
`end_frame`. Verified against boxmot==16.0.8 (V1-V8). **Any boxmot version bump requires
re-verifying V1-V8** — see `docs/evidence/timing_audit_1/findings.md` sites #4, #20.
**boxmot `__version__` lags package version by one** — 16.0.8 wheel ships `__version__='16.0.7'`.
trackers/ and motion/ are byte-identical between 16.0.7 and 16.0.8.
**Single-instance-per-process is load-bearing:** `STrack.shared_kalman` and
`BaseTrack._count` are class-level/process-global. A second concurrent tracker would
silently corrupt both prediction and track IDs (stock `BaseTrack.clear_count()` in
`BotSort.__init__` is the same assumption). Multi-camera parallelization within one
process must revisit this.
Piece 8 (BoT-SORT `frame_rate` scalar) dissolved into Piece 11.

**A↔D contract mismatch:** The length-proportional `unexplained_tracklet_penalty`
(`max(base, per_frame × n_frames)`) assumes tracklet purity that Stage A does not deliver.
Impure tracklets are disproportionately LONG (they absorbed multiple people), so the
current weighting is ANTI-correlated with reliability. Two candidate fixes: (i)
reliability-DISCOUNT the penalty, (ii) use a purity proxy (`max_displacement`) as a
high-precision D0.5 SPLIT trigger. Neither built yet.

**Purity-proxy results (PURITY-PROXY-1/2):**
- `max_displacement` (path discontinuity): AUC 0.82–0.85 raw / 0.75–0.82 post-D0.5.
  Same-color robust.
- Masked appearance: +0.08–0.09 AUC over contaminated, ~0.88 on different-color but
  ~0.73 on same-color.
- **CRITICAL CEILING: smooth-motion + same-color drifts are 58% (vid2) to 94% (vid1)
  of impure tracklets and are invisible to BOTH proxies.** Multivariate combo lifts only
  the smooth+different-color subset (4–12% of impurity).
- Masked histograms raised same-color fraction dramatically (vid2 7.7%→69.2%) — unmasked
  contamination was inflating inter-person distances.
Evidence: `docs/evidence/purity_proxy_{1,2}/`.

**Production color-signal defects (histogram.py):**
- Crops are unmasked RECTANGLES (background + other athlete contaminate).
- Pose-guided torso crop is DEAD CODE (detection-only model → keypoints NaN → center-bbox
  fallback always fires).
- Tracklet summary AVERAGES histograms, destroying multi-modality that reveals impurity.
- Masked full-body H+S+V reached AUC 0.907 vs 0.815 unmasked (CP-RASTER-PLATE-2) —
  masking validated but NOT productionized.

**Recorder timing arc (WALLCLOCK-1 → CAPTURE-TIME-2):**
- WALLCLOCK-1: Container PTS is synthetic (CFR re-encode discards wall-clock).
- RECORDER-SIDECAR-1: Per-frame `.timing.jsonl` sidecar shipped to production (CFR +
  `-vf showinfo`, video byte-identical). Under arrival-PTS, input≠output frame count
  (mismatch is NORMAL), so `pts_time_s` is a nearest-neighbor approximation: mean ~80ms,
  P95 ~230ms, max ~500ms error, worst in lag windows.
- CAPTURE-TIME-1: **RTSP stream carries TRUE capture timestamps** (source PTS uniform
  33ms / 1.21ms stdev). The recorder was DISCARDING them via
  `-use_wallclock_as_timestamps 1` + `+genpts`, substituting bursty network-arrival times.
- CAPTURE-TIME-2: **RTCP definitively absent** across all cameras, both TCP and UDP
  transports. Absolute camera-clock time unavailable from stream. **Tier-2 cross-camera
  alignment achievable at ±14–56ms** using source PTS + host-clock lower envelope.
  FP7oJQ shows −603 ppm clock drift (~181ms over 5 min) — measurable, linearly
  correctable.
- **Stream fps VARIES per session AND mid-stream** (15fps and 30fps both observed from
  source PTS). SDP reports 30 when delivering 15. Cameras differ from each other (13.85
  vs 15.00 measured in one session). **CP-R11: frame-spacing characterization (supersedes
  CP-R1b)** — Each camera delivers at a single cadence (~15fps, 67ms intervals) with
  periodic single-frame gaps whose spacing is determined by a camera-internal grid mismatch.
  FP7oJQ: gap every ~12 frames (~8% rate), PPDmUg: ~0.45% gap rate. On rare occasions the
  cadence switches to ~30fps in sustained BLOCKS (not interleaved). The 15fps cadence is
  genuine (proven: PPDmUg 1,979 consecutive gap-free frames; FP7oJQ periodic gap spacing
  rules out random loss). CP-R1b's "structurally undecidable" and "bimodal interleaving"
  verdicts are superseded. Container metadata and `measured_fps` remain unreliable on
  bimodal segments (TRIM-BIMODAL defect still applies). Per-frame `dt_s` from the sidecar
  is the only reliable rate source.
  Evidence: `docs/evidence/frame_spacing_1/`, `docs/evidence/recorder_fps_adaptation_1/`.
  **Do not hardcode fps anywhere — use per-frame dt from sidecar.**
  **Consumer-facing consequence:** gaps and mode switches are distinct phenomena that both
  contribute to dt variation. Neither should be used as an analysis grouping — only their
  combination (per-frame dt dispersion) matters to a variable-dt consumer. In particular,
  do not group experiments by `is_bimodal` — a `is_bimodal: false` segment can have higher
  dt dispersion than a `is_bimodal: true` segment (see TIMING-DISPERSION-1).
- `frame_index` is a sequential `cap.read()` counter (frame_iterator.py), NOT PTS-derived.
  Works for any fps.
- Diagnostic module: `services/nest_recorder/recorder/diag_timing.sh` (parallel, does NOT
  modify production recorder). Analysis: `tools/analyze_capture_timing.py`.
  Evidence: `docs/evidence/capture_time_{1,2}/`, `docs/evidence/recorder_timing_1/`,
  `docs/evidence/wallclock_1/`.

**RECORDER-RELIABILITY-1 (2026-07-28):** Five production reliability fixes in `diag_v6.sh`:
(1) RTSP socket timeout (`-stimeout`, 10s default) — exits ffmpeg in ~10s when data stops
instead of 2+ min OS TCP timeout (top fix for recording gaps). (2) Stop stream before
regenerating — `stop_stream()` calls StopRtspStream before each retry, preventing RTSP
session orphaning at the relay (root cause of the 404 cascade: SDM concurrent-stream limit).
(3) Access token refresh per attempt — `get_access_token` before every generate/extend;
auto-refresh on 401 (token expires at ~21-25 min, fatal for 65-min windows). (4) Failure-type-
aware backoff — healthy exit (>=60s) reconnects immediately; RTSP 404 backs off moderately
(10s→30s cap); quick/unknown 3s→15s cap; backoff resets on success. (5) Sidecar extraction
backgrounded — off the critical path, PIDs waited at window end.
- Source-PTS dup/drop verdict: **camera-dependent** — PPDmUg at 15.0fps shows exact sidecar
  match; FP7oJQ at 13.85fps still produces ~8% input/output mismatch under source-PTS.
  Pixel-identical duplicates reduced 10-50x vs arrival-PTS on both cameras.
- SOURCE_PTS exonerated as cause of exits — all failures were RTSP 404 (relay lockout
  from session orphaning), session invalidation (Nest unilateral), and 401 (token expiry).
  Zero timestamp/DTS errors.
- **ffmpeg option fragility:** `-stimeout` was removed in ffmpeg 7.x; the correct option for
  current builds is `-timeout` (same microsecond units). `debian:stable-slim` silently rolled
  bookworm→trixie, changing the available ffmpeg. **Pin the Debian version in Dockerfiles.**
- Backoff bug: exponential backoff NEVER RESET on success (3→6→12→…→96s, pinned for the rest
  of the window). Now resets on healthy exit. This compounded the timeout bug — long waits
  between retries grew geometrically.
Evidence: `docs/evidence/recorder_reliability_1/`.

**RECORDER-RELIABILITY-2 (2026-07-28):** API traffic reduction + quota awareness. RELIABILITY-1
increased API calls from ~0.75/min to ~17/min, triggering SDM 429 rate limits. SDM quota:
**10 QPM per user per project** (shared across all cameras, all ExecuteDeviceCommand calls).
Fixes: (1) optimistic URL reuse (0 API calls when session still valid), (2) conditional
stop_stream (skip for dead sessions — confirmed 400="stream_token invalid"), (3) 429 backoff
(60s→300s, Retry-After support), (4) generate 404 fail-fast (3 retries then exit), (5)
consecutive failure escalation (5+ failures → slow-poll 120-300s, prevents offline cameras from
consuming shared quota), (6) cross-camera quota coordination (N_CAMERAS from v7_2, dynamic
min retry interval from 70% of quota/N, jitter on every backoff).
Evidence: `docs/evidence/recorder_reliability_2/`.

**TWO dup/drop mechanisms (DUPFIX-1/2 — duplicates resolved, drops measured):**
1. **Bursty arrival timestamps** → ffmpeg mis-inferred frame rate → dup/drop. **FIXED** by
   source PTS (pixel-identical dups reduced from 34/4530 = 0.75% to 0/segment; one exception
   PPDmUg-070422 3 frames / 0.18%).
2. **Frame drops — REAL, CHARACTERIZED, DETECTABLE (DUPFIX-2 + CP-R11).** Correction is
   pipeline-side (variable-dt Kalman step). DUPFIX-2 measured real frame drops. CP-R11
   refined the attribution:
   - **FP7oJQ (~8% gap rate):** Camera-internal grid mismatch, not network loss. The camera
     captures at ~13.85fps on a ~14.93fps PTS grid, skipping a slot every ~12 frames. Periodic
     (mode spacing = 12), not random. DUPFIX-2's "0.1-7.7%" figure was the same phenomenon
     measured before the mechanism was understood.
   - **PPDmUg (~0.45% gap rate):** Low-rate residual. 47% of segments gap-free. CFR decimation
     was eliminated by passthrough (CP-R2/R3, defaulted CP-R3).
   Both produce false teleports in the Kalman filter. Detection via `pts_time_s` gaps;
   correction direction: variable-dt Kalman step (see Active Decisions Log "Coast architecture"
   row). Evidence: `docs/evidence/recorder_dupfix_1/`, `docs/evidence/frame_spacing_1/`.

**Timing capabilities (consolidated contract):**
- **Relative per-frame timing: TRUE and camera-derived.** RTP timestamps from the sensor clock.
  Uniform ~33ms (30fps) or ~67ms (15fps) intervals, 1.21ms stdev. Intervals are real.
- **Absolute per-frame timing: ESTIMATED ±14–56ms.** Recorder-side lower-envelope offset
  against host clock. Per-camera drift measurable and linearly correctable (FP7oJQ −603 ppm
  ≈ 181ms/5min).
- **RTCP definitively ABSENT** across all cameras, both TCP and UDP. No RTP→NTP wall-clock
  mapping available. Absolute camera clock NOT accessible from stream.
- **fps varies** per session AND per camera. SDP unreliable (reports 30 when delivering 15).
  **Never hardcode fps anywhere.**
- **Sidecar contract:** `docs/reference/sidecar_contract.md` (schema v5, CP-R13b).
  Authoritative spec for `.timing.jsonl` sidecars. Key additions over schema 3:
  `source_pts` validity gate, `nominal_dt_s`, per-frame `dt_s`, `is_bimodal` + mode
  fields, drift gated at `n_drift_windows >= 4`, `input_n` deprecated. `measured_fps`
  and `measured_fps_median` omitted under `source_pts: false`. Consumer recipes for
  gap detection, BoT-SORT scalar, and cross-camera sync documented in the contract.

**CRITICAL CAVEAT: prior GT measurements made on CORRUPTED FOOTAGE.** All existing GT footage
and every measurement derived from it predates the recorder fixes and was recorded under
bursty-arrival timestamps causing large-scale dup/drop (35% input/output mismatch in one
case). Duplicate frames inject FALSE ZERO-MOTION into BoT-SORT's Kalman filter; dropped
frames inject FALSE TELEPORTS. An unknown fraction of the measured "Stage A tracklet_drift"
(41%) may be RECORDER-INJECTED rather than a tracking limitation. **Hold GT2ACTUALS drift
attribution, purity-proxy results, and Stage A sweep conclusions LOOSELY until re-measured
on clean footage.** Keep old GT clips as a regression baseline. Legacy GT is also
**structurally excluded from pipeline timing measurement** — see "Sidecar required for
pipeline timing" in the Active Decisions Log.

**CP22 (completed):** Default detection model updated to yolo26n-pose (STAL loss, better
small-object detection). ultralytics upgraded 8.3.252 → 8.4.33 (`--no-deps`).
CoreML is now the default inference path (`prefer_coreml: true`). Detector auto-loads
`.mlpackage` sibling when available (78.9 fps CoreML vs 32.5 fps MPS on M1 Air).
Batch predict unsupported with CoreML (ultralytics bug); `infer_batch()` falls back to
sequential. ANE saturated by single stream — threading hurts (2 workers = 0.54x).
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.

**CP23a (completed):** Confidence threshold test. `tools/compare_conf_thresholds.py`
showed model CAN detect grappling pairs at low conf (orange boxes appear) but also misses
some entirely. Conclusion: both confidence AND resolution/classification issues exist.

**CP23b (completed — training pipeline + model fine-tuning):**

Training pipeline infrastructure at `src/training_pipeline/` (10 modules, ~3000 lines).
Interactive CLI: `PYTHONPATH=src python -m training_pipeline`. CVAT integration via
cvat-sdk 2.62 with API compatibility fixes. Background models built for all 3 cameras.

*Training rounds completed:*

| Round | Data | Best Box mAP50 | Best Pose mAP50 | Model |
|-------|------|----------------|-----------------|-------|
| R1 | 301 frames FP7oJQ | 0.890 | 0.467 | `models/bjj-pose-r1.pt` |
| R2 | 602 frames FP7oJQ+J_EDEw | 0.891 | 0.209 | `models/bjj-pose-r2.pt` |

R2 detects significantly more people but pose quality degraded vs stock on standing people —
too little data overwrote general COCO pose knowledge.

*3-way pose comparison (completed on Kaggle):*

| Model | Dataset | Status |
|-------|---------|--------|
| bjj-pose-r2_bbox | 602 gym frames, bbox only | Trained |
| bjj-pose-vicos | 12K ViCoS BJJ frames, full keypoints | Trained |
| bjj-pose-hybrid | r2_bbox 20x upsampled + vicos_12k (24K) | Trained |

All trained from stock yolo26n-pose.pt, freeze=10, 100 epochs on T4 GPU.

*Detection-only model (active in Stage A):*

| Model | Dataset | Metrics | Status |
|-------|---------|---------|--------|
| bjj-detect-all-cameras | 902 frames, 3 cameras, bbox only | mAP@0.5=0.939, mAP@0.50-95=0.669, F1=0.89@0.537 | Superseded by v2 |

Base model: stock yolo26n.pt (detection, not pose). freeze=10, 100 epochs on T4 GPU.
Dataset: 10789 annotations across 902 frames (FP7oJQ 301 + J_EDEw 301 + PPDmUg 300).
Train/val: 749/153 (83/17%), per-camera stratified temporal split.
CoreML export: `models/bjj-detect-all-cameras.mlpackage`.
Config: `conf: 0.45`, `require_keypoints: false`, `prefer_coreml: true`.
`DetectorConfig.iou: Optional[float] = None` — NMS IoU threshold (CP7-pre-6). Default
None = production CoreML path (inert, proven by artifact-diff regression). Setting iou
to any value **bypasses CoreML → .pt** and disables end2end NMS (~32fps vs ~79fps).
See `docs/decisions-archive.md` for the end2end/CoreML double-NMS finding.

*Detection model v2 (active in Stage A since 2026-06-06):*

| Model | Dataset | Metrics | Status |
|-------|---------|---------|--------|
| bjj-detect-all-cameras-v2 | 1352 frames (902 v1 + 450 J_EDEw-200246), bbox only | agg Recall@0.5=0.882 (+0.050 vs v1) | **Active** |

Dataset at `data/training_data/detection_all_cameras_v2/`. 1199 train / 153 val (val
identical to v1). New 450 frames: J_EDEw-200246.mp4, frames 0–4490 stride 10, train only.
Source: `data/raw/nest/c8a592a4-2bca-400a-80e1-fec0e5cbea77/J_EDEw/2026-03-18/20/J_EDEw-20260318-200246.mp4`.
Manifest: `configs/models/bjj-detect-all-cameras-v2.yaml`. Raw CVAT export (with track_id):
`data/training_data/training_J_EDEw_bbox_video2.zip` (4500 labels, stride-10 subset used).
Package: `data/training_data/training_data_detection_all_cameras_v2.zip` (292 MB).

*A/B evaluation v1 vs v2 (2026-06-02, frozen 153-frame val set):*
Original Kaggle training artifact: `bjj-detect-all-cameras_1352.pt`, renamed to
`bjj-detect-all-cameras-v2.pt`. CoreML sibling exported. Symmetric overlay routing
confirmed for both models. Baselines preserved at `outputs/_eval_*_baseline_v1/` and
`outputs/_eval_*_baseline_v2/`.

| Metric | v1 (902) | v2 (1352) | Δ | Signal? |
|--------|----------|-----------|---|---------|
| **Agg Recall@0.5** | 0.832 | **0.882** | **+0.050** | **yes** |
| **Agg Precision@0.5** | 0.959 | 0.935 | -0.024 | yes |
| FP7oJQ present | 21.0% | **26.0%** | **+5.0pp** | **yes** |
| FP7oJQ misattrib | 51.0% | 52.0% | +1.0pp | noise |
| J_EDEw present | 12.2% | 11.3% | -0.9pp | noise |
| J_EDEw misattrib | 54.0% | **59.0%** | **+5.0pp** | **yes (worse)** |
| PPDmUg present | 15.0% | 15.4% | +0.4pp | noise |
| PPDmUg misattrib | 61.4% | 62.3% | +0.9pp | noise |

**Verdict:** v2 is a substantially better detector (+5pp recall), improved FP7oJQ identity
(+5pp present), but did NOT move the misattribution blocker (52-62%, flat or worse).
Confirms CP7: the blocker is detection under-segmentation, not recall. More data recovers
missed people but cannot separate already-merged pairs.

*Dataset v2 fix (2026-05-06):* FP7oJQ frame extraction was misaligned — used
`range(0, 3001, 10)` (every 10th frame across 3000) when annotations covered
frames 0–300 consecutively. Fixed to `range(0, 301, 1)`. Correct source videos:
FP7oJQ `data/cvat_tasks/round1_20260497_FP7oJQ/FP7oJQ-20260318-200014.mp4`,
J_EDEw `data/cvat_tasks/round1_20260497_J_EDEw/J_EDEw-20260318-200015.mp4`,
PPDmUg `data/raw/nest/training_samples/training_PPDmUg_3000.mp4`.
First trained model had FP7oJQ false positives from background memorization.

*Two-clip validation (2026-05-08, J_EDEw clips 200246 + 200517):*
- Stage A avg detections/frame: 9.1 and 10.0 (vs 11.9 at conf=0.25 in comparison video)
- Tracklet counts: 215 and 230 per clip
- Short tracklet ratio (<30 frames): 50.2% and 51.7% — significant fragmentation
- Very short tracklets (<10 frames): 32.6% and 37.8%
- AprilTag 1: 3 observations in clip 200246 (frames 1781–1782), 0 in clip 200517
- Tag 1 stitched to person p0003 (4 tracklets collapsed, 60s span, 1,101 detections)
- Tag 1 matched in 0 match sessions (fragmentation may disrupt proximity signal)
- Person IDs: 22 and 17 per clip; match sessions: 26 and 32 per clip
- Stage F exported all clips successfully (26 + 32 mp4s)
- Bug found and fixed: `prefer_coreml` field missing from DetectorConfig Pydantic model
  (`src/bjj_pipeline/config/models.py`) — latent since CP22d, surfaced by Pydantic
  `extra="forbid"` validation

*Key decisions:*
- Detection-only model preferred over pose model — pose supervision from domain data
  degrades bbox quality due to annotation noise on fisheye ceiling-mount footage
- ViCoS keypoints retained for future pose work but not active in current model
- FP7oJQ false positive root cause confirmed: frame extraction bug (resolved),
  plus zero empty frames across all cameras (pending)

*Key findings:*
- Bbox-only training preserves stock pose quality while improving detection
- Hybrid approach: gym bbox trains detection head, ViCoS trains pose head
- ViCoS dataset (120K frames, smartphone cameras) has domain gap from Nest overhead
- freeze=10 vs freeze=6 probe tied on 602 frames — stay frozen until more diverse data
- CVAT keypoint order differs from COCO — remapping required (see training-pipeline.md)
- MPS training has float64 issue — use CPU locally or GPU on Kaggle/Colab

*Open issues:*
- Tracklet deduplication: ~50% of tracklets <30 frames, ~35% <10 frames
- Empty frame injection: not yet implemented — next data quality step
- Bbox size tier filtering: `tools/visualize_bbox_tiers.py` built, thresholds not applied
- PPDmUg near-zero detections on held-out clip (may be empty mat or model weakness)

## Tool Inventory

| Tool | Purpose |
|---|---|
| `tools/compare_model_detections.py` | 2x2 grid comparing YOLO models visually |
| `tools/compare_conf_thresholds.py` | Side-by-side conf=0.25 vs conf=0.05 |
| `tools/merge_cvat_exports.py` | Merge OBBox bboxes + Pose keypoints with CVAT→COCO remap |
| `tools/prepare_round2_dataset.py` | Round 2 data prep (filter, merge, extract, combine) |
| `tools/prepare_vicos_dataset.py` | Convert ViCoS JSON annotations to YOLO format |
| `tools/prepare_3way_datasets.py` | Build r2_bbox, vicos_12k, hybrid datasets |
| `tools/package_vicos_for_colab.py` | Subsample and zip ViCoS data for cloud upload |
| `tools/package_for_colab.py` | Package training data + model for Colab upload |
| `tools/three_way_diff.py` | 2- or 3-panel side-by-side model comparison video |
| `tools/freeze_probe.py` | A/B freeze level comparison (20 epochs each) |
| `tools/colab_training.ipynb` | Jupyter notebook for Colab/Kaggle GPU pose training |
| `tools/colab_detection_training.ipynb` | Jupyter notebook for Colab/Kaggle GPU detection training |
| `tools/prepare_detection_dataset.py` | 3-camera detection dataset prep (track export → YOLO) |
| `tools/download_vicos.py` | Download ViCoS BJJ dataset (120K frames) |
| `tools/camera_geometry_analysis.py` | 4-phase camera diagnostic (ROI, detectability) |
| `tools/coreml_benchmark.py` | CoreML vs MPS speed comparison |
| `tools/investigate_fp7_annotations.py` | FP7oJQ false positive root cause analysis |
| `tools/visualize_bbox_tiers.py` | Color-coded bbox size tier overlays on training frames |
| `tools/compare_models.py` | Flexible 2x2 grid model comparison video tool |
| `python -m pipeline_validation evaluate` | Full model evaluation (pipeline + A/D/F eval) |
| `python -m pipeline_validation swap-diagnostic` | GT-oracle swap boundary diagnostic (CP-SWAP-1) |
| `python -m pipeline_validation swap-characterize` | Swap pattern characterization (CP-SWAP-2) |
| `python -m pipeline_validation signal-trace` | Greedy per-GT topology census (CP-TRACE-1) |
| `python -m pipeline_validation signal-trace --stage tag` | Tag signal trace (CP-TAG-1) |
| `python -m pipeline_validation gt2actuals --manifest-path <path>` | Dense GT-to-actuals error map (CP-GT2ACTUALS) |
| `tools/cp_gt2actuals_6_analysis.py` | Stage-attribution + signal-shape analysis (CP-6) |
| `tools/tag_fullscan.py` | Full-frame AprilTag scan (CP-TAG-2 ceiling experiment) |
| `tools/tag_experiment.py` | Dense GT + full-scan tag comparison (CP-TAG-2) |
| `tools/cp_tag_3_evidence.py` | CP-TAG-3 baseline evidence: tag-trace, session, carrier subcommands |
| `tools/sweep/baseline_check.py` | Reproduce baseline correct_id from gt2actuals parquets |
| `tools/sweep/cache_detections.py` | Cache Stage A detections for tracker param sweep |
| `tools/sweep/replay_tracker.py` | Replay detections through BotSort, produce remapped Stage A artifacts |
| `tools/sweep/run_stage_d.py` | Re-run Stage D (D0-D4) on sweep tracklet artifacts |
| `tools/sweep/run_gt2actuals.py` | Measure sweep run via GT2ACTUALS dense join (subprocess) |
| `tools/sweep/sweep_runner.py` | End-to-end sweep orchestrator: replay → D → GT2ACTUALS → metrics |
| `tools/cp_gt2actuals_7_dashboard.py` | Interactive 4-coloring dashboard (person_id/tracklet/HSV/velocity) |
| `tools/cp_purity_1_decomposition.py` | Through-line purity decomposition (CP-PURITY-1) |
| `tools/cp_purity_2_floor.py` | Aggregate reconciliation + addressable ceiling (CP-PURITY-2) |
| `tools/purity_proxy_1_analysis.py` | Purity proxy scores: max_displacement, kinematic features |
| `tools/purity_proxy_2_analysis.py` | Masked-appearance purity proxy analysis |
| `tools/analyze_recorder_timing.py` | Recorder per-frame timing extraction + analysis (subcommands: analyze, compare, dupfix) |
| `tools/analyze_capture_timing.py` | Multi-camera timing diagnostic session analysis |
| `tools/analyze_frame_spacing.py` | CP-R11: Frame-spacing characterization (blocked modes, gap periodicity, grid mismatch) |
| `tools/analyze_dt_dispersion.py` | TIMING-DISPERSION-1: Per-segment dt ratio dispersion, band decomposition, annotation priority |
| `services/nest_recorder/recorder/diag_timing.sh` | Multi-camera timing diagnostic (source PTS, RTCP hunt) |

## Training Data Locations

| Dataset | Location | Description |
|---|---|---|
| Round 1 | `data/training_data/round1/` | 301 frames FP7oJQ, bbox + keypoints |
| Round 2 | `data/training_data/round2/` | 301 frames J_EDEw, bbox + keypoints |
| Combined R1+R2 | `data/training_data/combined/` | 602 frames, both cameras |
| R2 bbox-only | `data/training_data/r2_bbox/` | 602 frames, keypoints zeroed |
| Hybrid | `data/training_data/hybrid/` | r2_bbox 20x upsampled + vicos_12k |
| ViCoS full | `data/vicos_bjj/` | 120K frames, YOLO labels + position labels |
| ViCoS 12K | `data/colab_package/vicos_12k.zip` | Subsampled for cloud training |
| Background models | `data/background_models/` | Per-camera .npy median frames |
| Detection all cameras | `data/training_data/detection_all_cameras/` | 902 frames, 3 cameras, detection only |
| Detection all cameras v2 | `data/training_data/detection_all_cameras_v2/` | 1352 frames (902 v1 + 450 J_EDEw-200246), detection only |
| CVAT exports | `data/training_data/training_*.zip` | Raw CVAT export zips |

## Cloud Training Setup

- **Kaggle preferred:** 30 hrs/week free T4 GPU. "Save & Run All" for background execution.
  - Dataset: "roll-tracker-training" by bryanrt
  - Input: `/kaggle/input/datasets/bryanrt/roll-tracker-training/`
- **Colab:** Works but free tier limited (~4-6 hrs before timeout).
- **Notebook:** `tools/colab_training.ipynb` (works on both with path changes)
- All models: batch=16, freeze=10 default, lr0=0.001

## Domain Context (auto-loaded by path)

| Rule file | Scope |
|-----------|-------|
| `calibration.md` | `src/calibration_pipeline/**`, `configs/cameras/**` |
| `cross-camera.md` | `src/bjj_pipeline/stages/stitch/**` |
| `pipeline-stages.md` | `src/bjj_pipeline/**` |
| `training-pipeline.md` | `src/training_pipeline/**`, training tools |
| `model-training.md` | `models/**`, dataset/training tools |
| `cvat-workflow.md` | CVAT integration, annotation workflow |
| `evaluation.md` | `src/pipeline_validation/**` |
| `signal-trace.md` | `src/pipeline_validation/signal_trace/**` |
| `tag-identity.md` | `src/bjj_pipeline/stages/tags/**`, `signal_trace/tag_trace.py` |
| `services.md` | `services/**` |
| `commands.md` | Common dev commands |
| `apps.md` | `app_mobile/**`, `app_web/**` |
| `supabase.md` | `backend/supabase/**` |

## Planned Work

**Appearance arc (decided sequence, gated on CP-RASTER-PLATE-2 GO):**
1. ~~Measure separability~~ — DONE (CP-RASTER-PLATE-2, GO verdict)
2. ~~V-channel histogram extension~~ — DONE (CP-HSV-V). `histogram.py` now
   H+S+V (864-dim, 18×8×6). All consumers verified safe (dynamic column discovery).
3. **D0.5 Tier 3: DISABLE (interim) or REDESIGN** — CP-GT2ACTUALS-6 confirmed
   Tier 3 owns 79% of D0.5 net damage (-222 of -282 on vid2). Disabling T3
   removes 241 false splits at cost of 19 correct (5.4%). CP-6 signal analysis
   found NO per-frame signal separates false from correct splits (HSV Bhattacharyya
   0.035 vs 0.040 — indistinguishable; speed overlaps at P95). Redesign needs
   temporal/structural approach, not threshold tuning. **Interim option: disable
   Tier 3 entirely to recover the -6.6pp D0.5 regression.**
4. **Mask + V-aware appearance into ILP cost layer** — CP-GT2ACTUALS-6 found
   appearance-in-solver addresses 26% of jumps (solver misstitch) and can help
   absorb D0.5 fragments. BUT color CANNOT discriminate false from correct splits
   (both have similar HSV distance at boundary). Needs structural complement.
5. **Stage A tracking/detection quality [DOMINANT LEVER]** — CP-GT2ACTUALS-6 found
   Stage A (tracklet_drift) is the #1 through-line damage source at 41% of jumps.
   Upstream of the solver — not addressable by appearance or D0.5 fixes.
6. **Tracker-level cheap-HSV-ReID** — low priority (tracker purity 0.9, low headroom).
   Stage A runs `with_reid: false` (BoT-SORT on motion+IoU only, no color).

**Evidence-economy principle (from this arc):** tags = hard constraint; clean appearance
= cost/veto, never hard; distinctiveness-weighted; only one tier may be hard.

**Checkpoint-2 timing work:** sequenced in `docs/roadmap/checkpoint2_breakdown.md`.
Pieces 0, 1, 2, 4, 6, 7, 9, 12 complete; Pieces 3, 8, 10 dissolved/resolved; Piece 11
T3 complete (result negative, held with caveats — see Coast architecture row). Only
**Piece 5** (cross-camera timing) remains, blocked on camera fleet. Six-objective execution
plan (§5 of the
roadmap) supersedes the original piece ordering:
(1) recorder coverage investigation, (2) MUXER-PTS-1 fix, (3) Pieces 4+6 (Stage D
reads time + Stage F export timing — parallel, independent of recorder), (4) player VFR
test → Stage F format decision (gates Piece 7), (5) CP22 NAType Stage E fix,
(6) annotate on clean footage. See the roadmap for piece definitions and rationale.
**CP22 NAType now blocks more than annotation (PIECE6-FIX-1 finding):** CP22 crashes
session Stage E → no session `match_sessions.jsonl` → session Stage F cannot run → the
multi-segment concat branch and `ts_offset_ms` derivation (PIECE6-FIX-1) cannot be verified
on real media → the cross-clip export path CP4.C enabled has never executed end-to-end.
Objective 5 was scoped as "needed before annotation"; it is now also blocking verification
of session export.

**Known defect (small fix):** `pipeline.py:685` references undefined variable `mode` in the
`config_resolved` audit event. The bare `except Exception: pass` at `:688` swallows the
`NameError`, so the `config_resolved` audit event has **never been written** on any run.
The event is missing entirely, not malformed. One-line fix: remove or define `mode`.

**Known defect (calibration tool):** `homography_calibrate.py:1323-1326` calls
`_load_lens_calibration(out_path)`, which reads K/D from the **existing** `homography.json`,
and undistorts the display frame. `calibrate_camera.py:5` documents Step 1 as operating on
the **raw** frame. Under `--force`, Step 1 clicks land in the old undistorted space while
Step 2 solves a new one — three steps, potentially three pixel spaces. Observed 2026-08-24
during FP7oJQ recalibration. At minimum the tool should print which space it is using;
ideally `--force` should imply a raw Step 1.

**Known defect (calibration tool):** `--force` overwrites the lens block (`camera_matrix`,
`dist_coefficients`) in `homography.json` in place with no backup and no confirmation. An
interrupted or abandoned run destroys the previous calibration. Observed 2026-08-24: a run
replaced f=950.0 with f=735.0; recovery was only possible because the original was in git.

**Known defect (audit layer):** All pipeline JSONL artifacts append (`open("a")`); `--force`
does not clear files before rerunning. `d05_split_audit.jsonl`, `orchestration_audit.jsonl`,
per-stage `audit.jsonl`, `export_manifest.jsonl`, `projection_debug.jsonl`,
`identity_hints.jsonl`, `tag_observations.jsonl` — all accumulate across reruns. Session
aggregation reads all historical events. Observed 2026-08-24: 132650 held 66 D0.5 events from
3 runs (expected 24). Parquet artifacts are unaffected (overwrite on write). **Workflow rule:**
use per-run summary events for counts, or clear the output directory before rerunning.

**Data contract change (CP4.C/D):** `dt_s` in `d2_edge_costs.parquet` and
`d1_reconnect_edges.parquet` changed meaning at CP4.D — was `dt_frames / fps` (uniform-spacing
approximation), now real elapsed time from container PTS. Same column name, same type, same
ColSpec. Any pre-CP4.D `dt_s` figure is not comparable to a post-CP4.D one. `dt_ms` (Int64,
nullable) added to D1 edge and D2 edge cost schemas (`f0_parquet.py`) as the real-time source.
`dt_frames` retained as a diagnostic frame-gap count (not used for timing computation).
`d1_reconnect_edges.parquet`'s `dt_s` became real-time incidentally (flows from site #7).
Evidence: `docs/evidence/cp4cd_results/`.

**Stage F export format (Pieces 6+7+12, complete):** Both plain and redacted paths produce
VFR H.264 with source PTS preserved. Redacted path uses PyAV with `PTS_TIMEBASE_HZ=90000`
(source's 90kHz timebase). The CFR defect (2.3% fast, ~2.7s/2min from cv2.VideoWriter) is
**FIXED** — DB/media duration gap reduced from 2.272s to 0.466s. The remaining 0.466s is 7
missing frames from `cap.read()` ending early (pre-existing, same in both paths). `fps`
parameter removed from `render_redacted_clip`. `av>=18.0` declared in `requirements.txt`
(no NumPy dependency, compatible with 1.26.4 pin). File size reduced 35% (mpeg4→h264).
Evidence: `docs/evidence/piece12_results/findings.md`.

**Known limitation (Stage F export — GOP snap):** Export seek times are derived from real
`timestamp_ms` (pipeline arithmetic error ≈ 0ms). Residual customer-visible error is ≤2.0s
from `-ss` input-seeking keyframe snap (source camera GOP = 2.0s, measured on FP7oJQ
2026-08-22). Removing the residual requires output seeking (`-ss` after `-i`, slower) or a
GOP change at the recorder.

**Known cleanup (tools):** Three tools import `derive_clip_frame_offset` / `parse_clip_timestamp`
from `session_d_run` (`cp_purity_3_oracle`, `cp_tag_3_evidence`, `analyze_recorder_timing`).
The function is deprecated in the pipeline (CP4.C replaced it with `clip_offset_registry.json`).
Migrate the tools to the registry or move the helper into `tools/`, then delete from
`session_d_run`.

**Known defect (Stage E):** `timestamp_ms lookup miss for frame_index=315` on
FP7oJQ-20260822-130229 and the session run. A frame present in the timestamp map but absent
from `person_tracks` — Stage E's buzzer end-frame adjustment (`_try_adjust_end`) references a
frame that D4 did not assign to any person. Distinct from CP22 NAType (null-`frame_index` at
D2 on PPDmUg). Flagged as a CP4.C input (frame→time lookup).

**Piece 9 (completed 2026-09-01):** Debug/eval visualization fps scalars. All three sites
resolved — `visualize.py:408` (site #19), `multiplex_runner.py:406` (site #13), and
`post_pipeline_annotator.py:217`. All now read `1.0/nominal_dt_s` from sidecar. No
hardcoded `30.0` remains. `manifest.fps` write-back retained with sidecar-derived value.
**Data contract change (Piece 9):** `manifest.fps` in `clip_manifest.json` changed meaning:
was `CAP_PROP_FPS` (container average rate including gap time, e.g. 14.708), now
`1.0/nominal_dt_s` (camera cadence from source PTS, e.g. 14.925). Same field, same type,
different quantity. Pre-Piece-9 manifests carry the container value. This is the same
pattern as `dt_s` at CP4.D. Consumers: `d2_run.py:117` (diagnostic only), visualization
writers (annotated.mp4, mat_view.mp4, annotated_post_E.mp4). Nothing validates the value;
a consumer comparing pre- and post-Piece-9 manifests would mix derivation methods.

**Planned work (checkpoint-2 remaining):**
- **Piece 5: cross-camera timing.** Site #8 (`cross_camera_evidence.py:275`). Blocked on
  camera fleet, not on work.

**Deferred (lower priority):**
- CP23b remaining: empty frame injection, bbox size tier filtering, tracklet deduplication
- CP23c: custom data flywheel (background subtraction, pseudo-labeling, active learning)
- CP22c: ROI mask geometry fix (parked)
- Inter-tracklet swap detection (cross-tracklet position-continuation + masked-appearance)
- Tracker-level cheap-HSV-ReID (low priority — tracker purity 0.9, low headroom)

**Metric-basis discipline (MANDATORY — this burned us 5 times, SWEEP-3b was the biggest):**
No correct_id number is comparable without its basis stated: camera set (single vs
3-camera), frame range (val-split vs full annotated), person_tracks level (clip vs
session), AND **pipeline state** (pre-split vs post-D0.5-split). The 58.7% figure
(CP-TRACE-2) is THREE-CAMERA aggregate. The 40.5% locked baseline was measured
PRE-split (Jun 7); current post-split state is 33.9% (CP-GT2ACTUALS-3.5). The
-6.6pp drop is D0.5 Tier 3 fragmentation, not a regression. Canonical definition:
clip-level person_tracks, val-split, greedy IoU>=0.3, with pipeline state noted.

## Overturned Conclusions (do not re-derive killed ideas)

1. **D0.5 splitting: recorded as helpful → MEASURED NET-NEGATIVE.** Per-event GT
   accounting (CP-4+5): 35 correct / 317 false splits (net -282) on vid2. Tier 3
   owns ~79% of damage; T3 precision 7.3%, T2 16.5%.
2. **CP-HSV-V (H+S → H+S+V): shipped as "first production improvement" → POSSIBLY
   NET-HARM.** Genuinely improved separability, but made Tier 3 fire 5–6× more into
   a ~7%-precision splitter. Live canonical correct_id 33.9% vs pre-split 40.5%
   baseline (−6.6pp). A real current-state regression.
3. **"Appearance in the solver" as THE lever → DEMOTED.** CP-6 showed no per-frame
   signal discriminates real vs false splits (speed P95 overlaps; HSV Bhattacharyya
   ~identical). Stage A drift dominates. Appearance-in-solver addresses at most the
   ~26% misstitch share.
4. **CAPTURE-TIME-1's "the camera is really 30fps" → CORRECTED → CONFIRMED CORRECTED
   (CP-R11).** The 15fps cadence is genuinely 15fps, not 30fps with loss. Proven by
   PPDmUg 1,979 consecutive gap-free 67ms frames and FP7oJQ periodic (not random) gap
   spacing. Source PTS rate varies per session (15fps default, 30fps observed rarely).
   FP7oJQ's ~13.85fps effective rate is 15fps with ~8% grid-mismatch gaps.
5. **Sidecar "exact per-frame timing" → QUALIFIED.** Under arrival-PTS it is a
   nearest-neighbor approximation (±500ms worst case). Under source-PTS it would be
   exact (input==output, no dup/drop) — not yet adopted in production.
6. **CP-TAG-4a "+22.7pp improvement" → RETRACTED (SWEEP-3b).** Both 40.5% and 63.2%
   read the SAME pre-commit person_tracks. Difference was full-range vs val-split
   frame selection, not a code-change effect. CP-TAG-4a's actual effect is UNKNOWN.
7. **RELIABILITY-1 "dup/drop resolved" → RE-QUALIFIED (DUPFIX-1/2), FURTHER QUALIFIED
   (MUXER-PTS-1).** Source PTS fixed mechanism 1 (arrival-timestamp jitter): zero
   pixel-identical adjacent frames **mid-stream** on 9/10 source-PTS segments (one
   exception: 3 frames / 0.18% on PPDmUg-070422). **However, the blanket "zero
   pixel-identical" claim is wrong at stream-start boundaries:** 6 of 11 attempt-first
   segments have pixel-identical adjacent frames at positions n:1/n:2 (the RTSP relay's
   duplicate IDR). The exception is narrow (1 frame per affected segment, only at attempt
   start) but the claim as previously stated was false. MUXER-PTS-1 fix drops these
   duplicates via a `select` filter. CFR-padded
   frames are re-encoded by x264 and differ at pixel level. `input_n` repetition is a
   count-mismatch artifact. **Duplicates eliminated; drops characterized and detectable —
   correction is pipeline-side.** DUPFIX-2 measured real frame drops, originally attributed
   to upstream/network loss (FP7oJQ) and CFR decimation (PPDmUg). CP-R11 refined this:
   FP7oJQ's gaps are a camera-internal grid mismatch (periodic, every ~12 frames, ~8% rate);
   CFR decimation was eliminated by passthrough (CP-R2/R3); PPDmUg's residual rate is 0.45%.
   Both produce false teleports in the Kalman filter. Detection via PTS gaps; correction
   direction: variable-dt Kalman step (see Active Decisions Log "Coast architecture" row).
   Evidence: `docs/evidence/recorder_dupfix_1/findings.md`, `docs/evidence/frame_spacing_1/`.
   RELIABILITY-1's mpdecimate "255 dups" used default thresholds (near-identical); true
   pixel-identical count on arrival-PTS control is 34 (0.75%).
8. **Prior GT measurements made on CORRUPTED FOOTAGE.** All existing GT footage and
   every measurement derived from it predates recorder fixes and was recorded under
   bursty-arrival timestamps (35% mismatch in one case). Duplicate frames → false
   zero-motion; dropped frames → false teleports. An unknown fraction of measured
   "Stage A tracklet_drift" (41%) may be recorder-injected. Hold GT2ACTUALS drift
   attribution, purity-proxy results, and Stage A sweep LOOSELY until re-measured on
   clean footage. Legacy GT is **structurally excluded from pipeline timing measurement**
   — see "Sidecar required for pipeline timing" in the Active Decisions Log. CP-R8
   (clean re-capture + CVAT annotation) is on the critical path for all checkpoint-2
   outcome validation.
9. **"Padded frames = duplicates, skip Kalman update" → DUPLICATE HALF RETIRED, DROP HALF
   OPEN (DUPFIX-1/2), BOUNDARY EXCEPTION (MUXER-PTS-1).** Framehash proves 0 pixel-identical
   adjacent frames **mid-stream** on source-PTS segments (one 0.18% exception). **Stream-start
   boundary duplicates exist:** 6/11 attempt-first segments have pixel-identical adjacent
   frames from the RTSP relay's duplicate IDR (fixed by MUXER-PTS-1 `select` filter).
   x264 re-encodes padded frames independently. `input_n`
   would drop ~7-8% of real frames — harmful. **But real frame drops exist** — originally
   measured as 0.1-7.7% on FP7oJQ, 0-3.0% on PPDmUg. CP-R11 refined: FP7oJQ's ~8% is a
   camera-internal grid mismatch (periodic, every ~12 frames); PPDmUg's residual is 0.45%
   (CFR decimation eliminated by passthrough). Drops are characterized and detectable —
   correction is pipeline-side. Detection via PTS gaps validated; correction direction:
   variable-dt Kalman step (see Active Decisions Log "Coast architecture" row).
   Evidence: `docs/evidence/recorder_dupfix_1/findings.md`, `docs/evidence/frame_spacing_1/`.

## Pipeline Validation Framework (TB-EVAL series, completed 2026-05-12)

**Module:** `src/pipeline_validation/` — three evaluation layers plus common utilities.

### Evaluating a new detection model

1. Place weights at `models/{model_id}.pt`
2. Create `configs/models/{model_id}.yaml` (see existing manifest as template)
3. Run: `PYTHONPATH=src python -m pipeline_validation evaluate --model {model_id}`
4. Review outputs at:
   - `outputs/_eval/stage_a/{model_id}/_aggregate.md` (detection quality)
   - `outputs/_eval/stage_d/{model_id}/_aggregate.md` (identity quality)
   - `outputs/_eval/stage_f/{model_id}/*/match_preview.mp4` (visualization)

The `evaluate` command runs the full pipeline rerun + Stage A/D/F evaluation.
Uses direct inference for Stage A evaluation exclusively (not parquet path).
Flags: `--skip-pipeline`, `--skip-stage-a`, `--skip-stage-d`, `--skip-stage-f`
for partial reruns; `--force` to re-run even if outputs exist; `--dry-run` to
preview the plan without executing.

Individual subcommands remain available for debugging:
`PYTHONPATH=src python -m pipeline_validation <stage-a|stage-d|stage-f|discover>`

**Manifest convention:** `configs/models/{model_id}.yaml` per model. Schema: model_id,
weights_path, pipeline_gym_id, training_data entries with annotated_range, splits, resolution.

**annotated_range is authoritative:** GT loader ONLY loads labels for frames defined by
the manifest's annotated_range x split. CVAT auto-interpolated labels outside annotated_range
are NOT trusted GT. Zip contents are advisory; annotated_range is the source of truth.

**GT evaluation surface (bjj-detect-all-cameras):**

| Camera | Annotated | Train (in-dist) | Val (held-out) |
|--------|-----------|-----------------|----------------|
| FP7oJQ | 301 (0-300 stride 1) | 250 | 51 |
| J_EDEw | 301 (0-3000 stride 10) | 250 | 51 |
| PPDmUg | 300 (0-2990 stride 10) | 249 | 51 |

Total: 902 annotated, 153 held-out. Pipeline outputs at `outputs/_eval_gt/` (gym_id `_eval_gt`,
hard links — symlinks don't work due to `Path.resolve()` following them).

### Evaluation Baselines: bjj-detect-all-cameras (val split)

**Stage A Detection (TB-EVAL-1):**

| Camera | Recall@0.5 | Precision@0.5 | Mean IoU | Recall@0.7 | Recall@0.9 |
|--------|-----------|--------------|----------|-----------|-----------|
| FP7oJQ | 0.847 | 0.989 | 0.850 | 0.756 | 0.340 |
| J_EDEw | 0.864 | 0.921 | 0.843 | 0.770 | 0.317 |
| PPDmUg | 0.750 | 0.981 | 0.834 | 0.681 | 0.208 |
| **Aggregate** | **0.832** | **0.959** | | | |

Topology (TB-EVAL-1.1): Duplicate rate 6-69%, true merge 0-28%, true split 0-16% (val).

**Stage D Identity (TB-EVAL-2):**

| Camera | ID Recall | ID Precision | Mean Coverage | Mean Purity |
|--------|-----------|-------------|--------------|-------------|
| FP7oJQ | 0.571 | 1.000 | 0.329 | 0.913 |
| J_EDEw | 0.571 | 0.833 | 0.239 | 0.832 |
| PPDmUg | 0.750 | 0.500 | 0.360 | 0.842 |

Failure mode breakdown (all cameras combined):
- detection_failure: 46% (Stage A missed person)
- tracklet_dropped: 25% (Stage D rejected tracklet)
- sloppy_box: 6% (boxes too loose)
- true_switch: 23% (Stage D mis-stitched)

**Match Preview Visualization (TB-EVAL-3):** Diagnostic mp4 per camera at
`outputs/_eval/stage_f/bjj-detect-all-cameras/{cam}/match_preview.mp4`.
Four layers: all detections (grey), person-assigned (colored), match envelopes
(orange dashed, faithful via `plan_crop_fixed_roi`), tag icons (yellow).

### GT Person Trace Layer (CP6, permanent)

`src/pipeline_validation/gt_person_trace.py` -- runs automatically as part of every
`evaluate` call. Joins five existing artifacts (per_frame_matches, detections,
d1_graph_nodes, d3_solution_ledger, person_tracks) into a per-frame per-GT-person trace.
Identity mapping is derived internally from per_frame_matches + person_tracks (CP-EVAL-1).

**Outputs** (per camera, under `outputs/_eval/stage_d/{model_id}/{camera}/`):
- `gt_person_trace.jsonl` -- one row per (camera, clip, frame, gt_person). Full chain:
  detection -> tracklet -> D1 node/carrier -> D3 status -> D4 person_id -> failure_mode.
- `gt_person_summary.json` -- per-GT-person failure-mode counts.
- `gt_camera_summary.json` / `gt_camera_summary_lite.json` (aggregate).

**Six failure modes** (full): present, stage_a_no_detection, stage_a_untracked,
d3_dropped, d4_unassigned, present_misattributed. Plus missing_canonical (GT track with
no canonical assignment). Lite mode (4) collapses the three Stage D modes into
stage_d_no_person -- used for historical baselines that lack pipeline artifacts.

**This is now the primary Stage D diagnostic.** The six-way breakdown replaces aggregate
coverage as the headline metric. After any intervention, read the per-camera breakdown to
see which mode shifted.

**Schema is a contract.** Adding columns is fine; renaming/removing requires a deliberate
migration since downstream tooling will depend on it.

**CP6 baseline results (current run, full mode):**

| Camera | present | a_miss | untracked | d3_drop | d4_unasgn | misattrib | miss_can |
|--------|---------|--------|-----------|---------|-----------|-----------|----------|
| FP7oJQ | 5.1% | 9.9% | 8.1% | 24.0% | 6.5% | 17.9% | 28.6% |
| J_EDEw | 4.7% | 11.7% | 13.5% | 49.7% | 0.7% | 19.7% | 0.0% |
| PPDmUg | 6.1% | 13.3% | 7.2% | 39.9% | 2.0% | 18.9% | 12.7% |

d3_dropped is the dominant failure mode on J_EDEw and PPDmUg. Parallel-carrier
displacement confirmed as root cause (see `docs/checkpoints/cp6_gt_trace_baseline.md` Section 4).
CP5 (parallel-carrier consolidation) verdict: **resume**.

**Baseline preservation discipline (NEW):** When preserving an eval baseline going forward,
copy BOTH `outputs/_eval/` AND the relevant `outputs/_eval_gt/{camera}/{clip}/` directories.
Pipeline artifacts are required for full-mode trace. The four historical baselines
(penalty_15 through cp4_pre) are lite-mode only because they predate this rule.

### Signal Trace (CP-TRACE-1, completed 2026-06-02)

**Module:** `src/pipeline_validation/signal_trace/` — greedy per-GT matcher + Stage A
topology census. Standalone submodule; does NOT modify the frozen instrument.

**CLI:** `PYTHONPATH=src python -m pipeline_validation signal-trace --model {model_id}`

Greedy matcher (IoU ≥ 0.3, many-to-one): each GT box independently claims its best
detection. Multiple GT people CAN match the same detection (pair-box signature).

**Topology classifications:** tight_match (1:1), pair_box (2+ GT share one detection),
split (GT matched by 2+ detections), miss (no detection at IoU ≥ 0.3).

**Baseline results (bjj-detect-all-cameras, all annotated frames):**

| Camera | tight_match | pair_box | split | miss | total |
|--------|-------------|----------|-------|------|-------|
| FP7oJQ | 2795 (66.3%) | 1010 (24.0%) | 0 (0.0%) | 409 (9.7%) | 4214 |
| J_EDEw | 2727 (64.7%) | 888 (21.1%) | 0 (0.0%) | 599 (14.2%) | 4214 |
| PPDmUg | 1658 (70.2%) | 594 (25.2%) | 0 (0.0%) | 109 (4.6%) | 2361 |
| **Aggregate** | **7180 (66.5%)** | **2492 (23.1%)** | **0** | **1117 (10.4%)** | **10789** |

Consistent with CP7-pre-3: pair_box at 21-25% of GT-person-frames is the dominant
under-segmentation signature. Split is zero (no over-segmentation at IoU ≥ 0.3).

### Signal Trace D-Stage (CP-TRACE-2, completed 2026-06-02)

Extends CP-TRACE-1 through Stage D. Joins each GT-person-frame's tracklet to
`person_tracks.parquet` (many-to-one: GROUP segments produce 2 person_ids per frame).
Also runs GROUP falsification against `d1_segments.parquet`.

**CLI:** `PYTHONPATH=src python -m pipeline_validation signal-trace --model {id} --stage d`
(also `--stage all` for both stages sequentially)

**D-trace classifications:** correct_id (dominant person_id in assigned set), wrong_id
(person_ids assigned but dominant not present), no_id (tracklet dropped by D), no_detection
(Stage A miss).

**Corrected results (bjj-detect-all-cameras, post CP-TRACE-FIX split-product resolution):**

| Camera | correct_id | wrong_id | no_id | no_detection |
|--------|-----------|---------|-------|-------------|
| **Aggregate** | **6,330 (58.7%)** | **3,039 (28.2%)** | **303 (2.8%)** | **1,117 (10.4%)** |

*(Pre-fix results showed 48.7% correct_id, 29.0% no_id — the 29% was a join-key mismatch
artifact between detections.parquet and person_tracks due to D0.5 split-product renaming.
See CP-TRIM-1.)*

**GROUP falsification (bjj-detect-all-cameras):**

| Camera | pair-box tracklets | SOLO | GROUP | not-in-graph |
|--------|-------------------|------|-------|-------------|
| FP7oJQ | 13 | 4 | 7 | 2 |
| J_EDEw | 47 | 15 | 24 | 8 |
| PPDmUg | 30 | 14 | 10 | 6 |

Verdict: GROUP engagement on pair-box tracklets is coincidental — triggered by lifecycle
events (merges/splits of other tracklets), not by the pair-box itself. Pair-boxes don't
create lifecycle events, so GROUP cannot address the under-segmentation problem.

### Signal Trace E/F + Verdict (CP-TRACE-3, completed 2026-06-02)

No-ID diagnosis, E/F signal extension, and synthesis verdict.

**CLI:** `PYTHONPATH=src python -m pipeline_validation signal-trace --model {id} --stage ef`
(or `--stage all` for full a→d→ef sequence)

**No-ID root cause (aggregate, post CP-TRIM-1 fix):** 303 no_id frames → 275 d4_frame_trim
(90.8%), 28 d3_solver_drop (9.2%). Pre-fix 29.0% was a measurement artifact: join-key
mismatch between detections.parquet (original tracklet_ids) and person_tracks (D0.5 split
products). See `_trim_report.md`.

**E/F extension:** All 36 GT people (across 3 cameras) appear in match sessions.
Stage F not available (pipeline ran --to-stage E).

**Synthesis verdict** (`outputs/_eval/signal_trace/bjj-detect-all-cameras/_verdict.md`):

Signal flow waterfall: 10,789 → 9,672 detected → 7,180 tight → 6,330 correct_id → 36/36 in match sessions.

Root cause ranking by frame impact (corrected):
1. wrong_id (28.2%) — identity misattribution (pair-box driven)
2. pair_box (23.1%) — detection under-segmentation
3. miss (10.4%) — detection recall
4. d4_frame_trim (2.5%) — graph coverage gap (genuine residual)
5. d3_solver_drop (0.3%) — negligible

**Intervention priorities:**
1. Detection pair separation (pair_box + wrong_id = 51.3%) — the dominant lever
2. Detection recall (miss) — 10.4%, diminishing returns from data alone
3. Graph coverage (d4_frame_trim) — 2.5%, low priority

**CP-TRIM-1 (completed 2026-06-02):** Investigation found 97.2% of d4_frame_trim was a
join-key mismatch artifact. Fix: split-product resolution in `stage_d_trace.py` via
`d05_split_audit.jsonl`. Pre-fix artifacts at
`outputs/_eval/signal_trace/bjj-detect-all-cameras_pre_fix/`.

### Tag Signal Trace (CP-TAG-1, completed 2026-06-03)

**Module:** `src/pipeline_validation/signal_trace/tag_trace.py` — traces tag_id through
the full pipeline (A→C→D→E) to answer: does the AprilTag signal deliver correct identity?

**CLI:** `PYTHONPATH=src python -m pipeline_validation signal-trace --model {id} --stage tag [--tag-id 1]`

**Key finding: tag detection is bbox-gated.** Stage C only scans padded detection bounding
boxes from Stage A (via `decode_apriltags_in_roi` + `bbox_pad_frac`). If Stage A misses the
person, Stage C never gets the chance to look for their tag. Detection recall directly
limits tag visibility.

**Cross-tab (Stage A × Stage D, all cameras val-split):**

| Stage A \ Stage D | correct_id | wrong_id | no_id | no_detection | Total |
|---|---|---|---|---|---|
| tight_match | 4728 (43.8%) | 2187 (20.3%) | 265 (2.5%) | 0 | 7180 |
| pair_box | 1602 (14.8%) | 852 (7.9%) | 38 (0.4%) | 0 | 2492 |
| miss | 0 | 0 | 0 | 1117 (10.4%) | 1117 |

Pair-box-driven misattribution: 852/3039 = 28.0% of all wrong_id. 72.0% of wrong_id
occurs on tight_match detections (solver/tracker errors on clean detections).

**Tag observation census:**

| Video | Tag obs | Tracklet frames | Detection rate | Chain C→D2→D4 |
|---|---|---|---|---|
| FP7oJQ-200014 | 0 | 0 | N/A | No |
| J_EDEw-200015 | 1 | 1,521 | 0.066% | Yes |
| PPDmUg-training | 0 | 0 | N/A | No |
| J_EDEw-200246* | 3 | 862 | 0.232% | Yes |

*Train-split GT, not held-out.

**Tagged person identity outcomes:**
- J_EDEw-200015 (gt_track_id=24, tracklet t366): correct_id 25.6%, wrong_id 50.5%,
  no_detection 22.9%. Tag signal chain complete but 1 observation in 1,521 frames.
- J_EDEw-200246 (gt_track_id=8, tracklets t143+t99): correct_id 16.9%, no_id 58.4%.
  Tag signal chain complete, 3 observations, but no_id dominates (no D0.5 split for this
  clip — pipeline ran under real gym_id, not _eval_gt).

**Verdict:** The tag signal mechanism (C→D2→D4) works when the tag is observed — 2/2
videos with observations have complete propagation. But tag visibility is desperately low
(0.07-0.23% of tracklet frames). The product cannot rely on AprilTags as the sole identity
mechanism. Complementary identity signals are needed.

**Outputs:** `outputs/_eval/signal_trace/{model_id}/cross_tab.{json,md}`,
`_tag_signal_verdict.md`, per-camera `tag_census.json`, `identity_hint_audit.json`,
`tagged_person_trace.parquet`, `_tagged_person_report.md`.
200246 artifacts at `J_EDEw_200246/` (separate from val-split J_EDEw).

### Tag Ceiling Experiment (CP-TAG-2, completed 2026-06-04)

**Module:** `tools/tag_fullscan.py` (standalone full-frame scan),
`tools/tag_experiment.py` (dense GT orchestrator).

**Full-frame scan:** Removed all pipeline restrictions (bbox gating, cadence) and scanned
every pixel of every frame of both J_EDEw videos. Result: **identical observation count**
to the pipeline's bbox-gated scan.

| Video | Pipeline Obs | Full-scan Obs | Full-scan Frames | Detection Rate |
|-------|-------------|--------------|-----------------|---------------|
| J_EDEw-200015 | 1 | 1 | 4530 | 0.022% |
| J_EDEw-200246 | 3 | 3 | 4500 | 0.067% |

Full-scan recovered 1 extra frame (200246 frame 1783, pipeline had 1781–1782 only).

**Verdict: Physical occlusion, not pipeline restriction.** The bbox gating and cadence
controls are not limiting tag detection. The AprilTag is below the resolution threshold
for reliable decode in >99.95% of frames at ceiling-mount fisheye distances (~3m).

**Dense GT validation:** CVAT zips contain interpolated labels at every frame (not just
stride-10 keyframes). Dense manifest at `configs/models/bjj-detect-all-cameras-dense.yaml`
loads stride=1 for J_EDEw (3,001 + 4,491 frames). Results confirm stride-10 is
representative — all percentages stable within ±0.3pp:

| Metric (Video 1) | Stride-10 | Stride-1 | Delta |
|-------------------|-----------|----------|-------|
| tight_match | 64.7% | 64.6% | -0.1pp |
| pair_box | 21.1% | 21.1% | 0.0pp |
| correct_id | 45.0% | 45.0% | 0.0pp |
| wrong_id | 37.8% | 37.5% | -0.3pp |

**Outputs:** `outputs/_experiments/tag_fullscan/` (full-scan observations),
`outputs/_eval/signal_trace/bjj-detect-all-cameras/J_EDEw/dense_gt_trace/` (dense traces),
`_tag_experiment_report.md` (full report).

### Cross-Tracklet Identity Diagnostic (completed 2026-06-05)

Deep diagnostic of tag identity propagation for tag_id=1 across both J_EDEw videos.
Verified code path (D2→D3→D4) and traced actual data.

**Architecture:** Must_link constraints are **SOFT** (2× miss_penalty, not hard ILP).
Tag identity does NOT propagate across tracklet boundaries — only must_link group
tracklets carry the tag binding. D4 assigns sequential person_ids (p0001...); tag mapping
is post-hoc via frame overlap scoring. GROUP segments cause multi-person assignment per
tracklet per frame.

**Video 1 (J_EDEw-200015, GT person 24):** 29 tracklets, **17 person_ids** for 1 GT
person. Tagged tracklet t366 has 167 person_id transitions (GROUP dilution — alternates
frame-by-frame between p0028/p0032/p0019). D4 emits 3 separate identity_assignments for
tag:1. Tag covers only last 5% of clip (frames 2759–2906).

**Video 2 (J_EDEw-200246, GT person 8):** 30 tracklets, **12 person_ids** + 11 tracklets
dropped. Tagged tracklet t99 (862 frames) **DROPPED** by solver — must_link penalty
insufficient. Tag observation captured by nested detection t143 (17 frames, bbox inside
t99), which survived on p0003 — a different person's path entirely.

**Three architectural gaps:**
1. Must_link too soft — solver can drop tagged tracklets (penalty < cost savings)
2. No path propagation — non-tagged tracklets on same path don't inherit tag anchor
3. GROUP dilution — D4 assigns multiple person_ids to tagged tracklet via GROUP nodes

### Dense GT-to-Actuals Error Map (CP-GT2ACTUALS, completed 2026-06-10)

**Module:** `src/pipeline_validation/gt2actuals/` — dense per-(frame, gt_track_id)
error map joining GT annotations against all pipeline signals.

**CLI:** `PYTHONPATH=src python -m pipeline_validation gt2actuals --manifest-path <path> [--camera <cam>]`

Uses `--manifest-path` (NOT `--model`) for explicit dense-manifest selection
(metric-basis discipline: per-row `manifest_path` + `manifest_stride` stamps).

**Schema:** One row per (frame_index, gt_track_id). Columns: identity state, match
topology, D1 node info, node_gt_set (GT-identity SET per D1 node), world coords
(x_m_eff = repaired-where-flagged), velocity, is_isolated + nullable HSV histogram
(NULL = entangled, never interpolated), tag observations, candidate detections
(n_candidate_dets, unmatched_candidate_person_ids), jump_type, jump_from_person_ids.

**State column:** no_canonical → miss → untracked → no_id → wrong_id → correct.
`is_group_ambiguous` boolean (GROUP node with node_gt_set_size >= 2).

**Jump types (GT-derived, inline):** tracklet_drift (Stage A), false_split (D0.5),
ilp_misstitch (solver), group_boundary_jump, group_membership_drift.

**Split-family lookup (CP-3 fix):** person_tracks lookup uses family-aware fallback
(resolved → raw → any split-family member) because the solver re-stitches D0.5
products under different IDs than bank_summaries predicts. The same bug exists in
`signal_trace/stage_d_trace.py` but does NOT affect locked canonical numbers
(computed pre-split, Jun 7). Fixing signal_trace is a separate gated decision.

**D0.5 net-effect (CP-4+5, per split-event):** net-negative on ALL cameras.
vid2 (authoritative, 99.4% classified): 35 correct / 317 false (net -282).
Tier 3 owns 79% of damage. FP7oJQ/PPDmUg are thin-classification (coverage artifact).

**Outputs:** `outputs/_eval/gt2actuals/{camera}/{clip}/gt2actuals_dense.parquet` +
`metadata.json`.

## Stage D Identity Investigation (CP0-CP6, completed 2026-05-19)

A seven-checkpoint investigation into why Stage D coverage was 24-36% despite Stage A
recall of 75-86%. Conclusion: the dominant failure mode is parallel-carrier displacement
in D1 graph construction, not penalty tuning.

**The arc:**
- **CP0** (`docs/reference/stage_d_audit_findings.md`): Config audit. Confirmed 7 of 8 D3 penalty
  fields are dead (never wired from config -> constraints). Only
  `unexplained_tracklet_penalty` is live (via explicit solver.py parameter, bypassing
  the broken constraints path).
- **CP1** (`docs/checkpoints/cp1_evidence.md`): Quantitative evidence. Cost inversion confirmed --
  interior BIRTH+DEATH (20.02) exceeded the flat drop penalty (15.0), so dropping
  interior tracklets was globally optimal.
- **CP2** (`docs/checkpoints/cp2_results.md`): Penalty 15->25. Partial -- helped FP7oJQ marginally,
  no effect on J_EDEw/PPDmUg. Binding constraint is not the cost floor.
- **CP2.5** (`docs/checkpoints/cp2.5_diagnostics.md`): Diagnosed flat penalty as length-agnostic.
  Recommended length-proportional.
- **CP3** (`docs/checkpoints/cp3_results.md`): Pure per-frame penalty. REGRESSION (short tracklets
  became too cheap to drop). Rolled back.
- **CP3b** (`docs/checkpoints/cp3b_results.md`): Floor-protected `max(base, per_frame*n_frames)`.
  No regression but no improvement on long tracklets. Penalty mechanism declared
  saturated.
- **CP4** (`docs/checkpoints/cp4_flow_topology.md`): Root cause found -- parallel-carrier displacement.
  When two tracklets are simultaneous carrier candidates for a merge event, D1 creates
  duplicate GROUP nodes; the solver routes one and orphans the other's entire chain.
  Penalty cannot fix this (it's structural, upstream of cost).
- **CP6** (`docs/checkpoints/cp6_gt_trace_baseline.md`): Built a permanent GT-anchored trace layer in
  pipeline_validation (see below). Confirmed CP4 at the row level AND found the picture is
  bigger than pairwise: J_EDEw has FOUR long carriers dropped (t1, t3, t5, t111), only two
  kept (t108, t2). 100% of d3_dropped frames across all cameras have a concurrent kept
  tracklet on a different GT person. Carrier competition reaches 12 simultaneous carriers
  per frame (J_EDEw, median 7).

**Current config state** (`configs/default.yaml` stages.stage_D.d3):
- `unexplained_tracklet_penalty_base: 25.0`
- `unexplained_tracklet_penalty_per_frame: 0.1`
- Formula: `max(base, per_frame * n_frames)` where n_frames = SINGLE_TRACKLET node frames
- The 7 dead penalty fields from CP0 remain present but unwired (documented, not fixed)

**Two reframings from CP6 that supersede earlier framing:**
1. The old "Stage D drops ~56% of detections / tracklet acceptance criteria suspected"
   framing is RETIRED. The mechanism is parallel-carrier displacement in D1, fully
   characterized.
2. `present_misattributed` (51-61% per camera, CP-SPLIT-1 baseline) is dominantly a
   DETECTION under-segmentation problem: one detection box covers two grappling people,
   so whichever person_id the tracklet receives, it is wrong for the other. On FP7oJQ
   (one 2.5-min clip): ~74% of misattribution is pair-box under-segmentation; of that,
   55.7% is confirmed unbracketed (detection-only-recoverable), the remainder
   indeterminate/partial pending wider-horizon and second-clip confirmation. Not a
   representation problem and not addressable by ReID/pose at the tracking layer. See
   CP7 investigation below.

**Recovery ceiling for CP5** (from CP6 trace analysis): CP5 (parallel-carrier consolidation)
recovers frames lost to d3_dropped. Conservative estimate: J_EDEw 4.7%->14.2% present,
PPDmUg 6.1%->15.7%. Ideal ceiling (every rescued drop attributed correctly where a canonical
slot is free): J_EDEw 37.5%, PPDmUg 42.1%, FP7oJQ 24.8%. All far below the >75% target.
CP5 is a necessary stepping stone, not the destination. Reaching usable coverage requires
detection-level pair separation (see CP7 investigation below).

**CP5 (completed 2026-05-21):** Parallel-carrier consolidation in D1 graph construction.
`_consolidate_parallel_triggers` helper in `d1_graph_build.py` — deterministic N-way
tiebreak (dist -> n_frames -> lexicographic carrier_id). Results (`docs/checkpoints/cp5_results.md`):
d3_dropped collapsed (J_EDEw 49.7% -> 7.9%, PPDmUg 39.9% -> 0.0%, FP7oJQ 24.0% -> 4.6%).
present rose modestly (J_EDEw 7.4%, PPDmUg 10.6%, FP7oJQ 6.4%). present_misattributed
is now the dominant failure mode (59-66% at CP5; 51-61% after CP-SPLIT-1). Solver
OPTIMAL, mergers stable. Next: see CP7 investigation.

**CP7 investigation (completed 2026-05-25, FP7oJQ only):** Eight-checkpoint read-only
investigation into the composition of `present_misattributed`. The arc inverted the
project's understanding:
- **pre-2:** 71-79% "impurity-driven" → sub-tracklet identity recommended.
- **pre-3:** Inverted: 70-78% is detection under-segmentation (one box, two people).
  Sub-tracklet identity targets 0.3-1.5%, not 71-79%.
- **pre-4/pre-6:** NMS-suppressed nested boxes investigated; NMS relaxation ruled out
  (worsened misattribution 4%→25%, fragmentation 1→4.5 tracklets/GT).
- **pre-8:** Axis-1 failure signature — apparent 84% "Branch B" (concurrent-swap node).
  SUPERSEDED by pre-9/pre-10.
- **pre-9:** The 84% was ~93% pair-box under-segmentation in disguise. True concurrent-
  swap margin: 9.9% of misattributed frames.
- **pre-10:** Pair-box spans 0% bracketed at every horizon (30f to full clip). The second
  person is never separately tracked anywhere in this clip → the lever is detection-level
  pair separation, and possibly plain recall on isolated people; the two are not yet
  separated and the separability experiment will distinguish them.

On FP7oJQ (one 2.5-min clip): ~74% of misattribution is pair-box; of that, 55.7% is
confirmed unbracketed (detection-only-recoverable), the remainder indeterminate/partial
pending wider-horizon and second-clip confirmation. 9.9% true Branch-B, 0% bracketed at
all horizons — single-clip, confirmation pending on the buzzer video. Stage D concurrent-
swap node deferred as a ~10% sidecar. Detection-level pair separation is the primary lever.

Integrity caveats:
(a) Pre-10 bracket test uses pipeline-derived GT attribution (majority-vote from
    gt_person_trace). Lean is benign (most reliable at separation points) but not
    ground-truth-verified outside 0-300.
(b) OPEN: the t10→t10_sN fragment map that moved pre-10 indeterminate 39%→13% is
    unverified — spot-check a sample of remapped carriers before treating 13% as hard.

Docs: `cp7_pre8_axis1_signature.md` (SUPERSEDED), `cp7_pre9_branchb_margin.md`,
`cp7_pre10_pairbox_bracketing.md`.

### Known Issues Surfaced by Framework

- **Stage D coverage loss is parallel-carrier displacement** (CP0-CP6, resolved diagnosis).
  See "Stage D Identity Investigation" above. The earlier "tracklet acceptance criteria"
  hypothesis was superseded -- the mechanism is in D1 graph construction. J_EDEw t201
  (tag:1) drop is partially a separate cost-bound case (mostly non-carrier fragments), not
  pure carrier displacement.
- **Stage C is a placeholder** for everything except tag observations —
  identity_hints.jsonl is empty for FP7oJQ and PPDmUg. Documented drift
  between CLAUDE.md (describes full tag pipeline) and code (placeholder run).
- **PPDmUg training sample** (`training_PPDmUg_3000.mp4`) is not pixel-identical
  to any Nest clip. Provenance unknown. Pipeline output uses clip_id
  `PPDmUg-20260318-training` via manifest's `pipeline_output_clip_id`.
- **Pipeline ingest** uses hard links not symlinks under `data/raw/nest/_eval_gt/`
  because `Path.resolve()` follows symlinks, losing the `nest` path component.

### Open Follow-ups

- **V-channel histogram extension (NEXT):** Production fix. H+S→H+S+V in
  `histogram.py`. Gated on CP-RASTER-PLATE-2 GO verdict.
- **Detection pair separation:** Primary lever for misattribution. CP7 showed ~74%
  of misattribution is pair-box under-segmentation. CP-PURITY-3 confirmed: 100% of the
  former "D1 group-formation defect" is detection under-segmentation (0 D1 logic gaps).
- Empty frame injection for training data (reduce FP rate)
- Stage C full implementation (beyond tag observations)
- PPDmUg training sample provenance

### Identity-corruption lever journey (CP-PURITY arc summary)

The lever was re-pointed by evidence FOUR times:
1. **D4 emission** — falsified (CP-PURITY-1: corruption is ILP stitch, not emission)
2. **ILP stitch** — confirmed as mechanism (p0022 follows wrong GT person)
3. **D1 group-formation** — falsified (CP-PURITY-3: GT→D oracle shows 0 D1 logic gaps;
   GROUPs are a compensation mechanism for imperfect detection, structurally unnecessary
   when detection is perfect)
4. **Detection under-segmentation** — confirmed as root cause (100% of the former
   "group-formation defect" is pair-box with no second tracklet)

Tracklet purity is healthy (0.88-0.92). The corruption enters at the solver layer
because detection under-segmentation gives it wrong input.

## Active Decisions Log

| Decision | Status | Notes |
|----------|--------|-------|
| CP-EVAL-1: Eval instrument freeze — single-path Layer 1/2 | **Active** | Frozen 2026-05-22 (cdf1037). Hungarian IoU 0.5. Identity mapping derived from `per_frame_matches.parquet` + `person_tracks.parquet` inside `gt_person_trace.py`. Spec: `docs/reference/eval_instrument_spec.md` v1.0. |
| CP-REID-1: BoT-SORT ReID experiment | **Rejected — DOMAIN GAP** | Generic `osnet_x0_25_msmt17` rejected for domain gap (street pedestrian vs overhead fisheye grappling), NOT color-blindness. V-histogram win does NOT reopen deep ReID. See updated row below. |
| CP-SWAP-1: Tracker-swap diagnostic | **Complete** | 167 GT-oracle swaps across 68/562 tracklets. Best single-feature AUC=0.663 (`bbox_aspect_change`). FP7oJQ world_accel 25.8x spike ratio, AUC=0.714. Histogram coverage 100% at swap boundaries. Module: `src/pipeline_validation/tracker_swap/`. |
| CP-SWAP-2: Swap pattern characterization | **Complete** | 47% hop_into_unoccupied, 28% cascade, 2% exchange. 41% transient (50% single-frame flickers). 45% no kinematic spike. 81% within 0.5m. Informed CP-SPLIT-1 design. |
| CP-SPLIT-1: Post-D0 tracklet splitter | **Active — PRECISION CRISIS (CP-SPLIT-VALIDATE)** | Tiered: speed cap 48 m/s + spike ratio 5x + Bhattacharyya 0.15 (corroboration 2x). Min dwell 5f. GT-validation (CP-SPLIT-VALIDATE): T3 2.4-20% precision, T2 6-22%. Spurious T3 dominated by motion-shadow (61%). Redesign needed — see CP-SPLIT-VALIDATE row below. |
| Domain-specific ReID training | **Deferred** | Superseded by CP7 finding: on FP7oJQ (one 2.5-min clip) ~74% of misattribution is detection under-segmentation, not addressable by ReID. Detection pair separation is the primary lever. |
| BoT-SORT parameter tuning | **Deferred** | iou_threshold, track_buffer experiments. After detection pair separation. |
| GROUP node assignment reform | **Deferred** | D4 boundary fix using realized_group_pairings. ~3-5pp potential. Concurrent-swap node deferred as ~10% sidecar (CP7-pre-9). |
| CP7: Misattribution decomposition | **Complete** | Eight-checkpoint investigation (pre-2→pre-10). On FP7oJQ (one 2.5-min clip): ~74% pair-box (55.7% confirmed unbracketed), 9.9% true Branch-B, 0% bracketed — single-clip, confirmation pending on buzzer video. Detection pair separation is the primary lever. See `docs/checkpoints/cp7_pre9_branchb_margin.md`, `docs/checkpoints/cp7_pre10_pairbox_bracketing.md`. |
| CP-TAG-1: Tag signal trace | **Complete** | Tag detection is bbox-gated (Stage C scans padded detection bboxes only). Tag visibility 0.07-0.23% of tracklet frames. Signal chain C→D2→D4 works (2/2 videos), but tags too rare for sole identity. Cross-tab: 28% of wrong_id from pair_box, 72% from tight_match. See `signal_trace/tag_trace.py`. |
| CP-TAG-2: Tag ceiling experiment | **Complete** | Full-frame scan (no bbox/cadence restriction) found identical tag observations to pipeline (1+3 across 9,030 frames). Bottleneck is physical occlusion at ceiling distance, not pipeline gating. Dense GT (stride-1, 10x points) confirms stride-10 is representative (±0.3pp). AprilTags cannot be sole identity mechanism. See `tools/tag_fullscan.py`, `tools/tag_experiment.py`. |
| Cross-tracklet identity diagnostic | **Complete** | Must_link is soft (2× penalty), identity doesn't propagate across tracklets, GROUP dilutes tag identity. Video 1: 17 person_ids for 1 GT person, 167 intra-tracklet transitions. Video 2: tagged tracklet (862f) dropped, tag assigned to wrong person via nested detection. Three architectural gaps: soft must_link, no path propagation, GROUP dilution. |
| CP-TAG-3: Two-clip harness + tag identity baseline | **Complete** | Two-clip J_EDEw harness under `_eval_gt`. Baseline: vid1 25.6% correct_id (t366, 1125 session transitions), vid2 22.2% correct_id (t139, 2680 session transitions). Both tagged tracklets KEPT at session level, 4 identity_assignments for tag:1 all spanning clip boundary. GROUP dilution is dominant corruption (14/12 person_ids per tagged tracklet). Prior 16.9% vid2 number was stale (pre-CP5). tag_trace.py:1181 hardcode footgun documented. Session-level trace mode gap noted — likely its own checkpoint before post-CP-TAG-4 gate. Evidence: `docs/evidence/cp_tag_3_baseline/`. Harness: `tools/cp_tag_3_evidence.py`. |
| CP-TAG-4a: Tag-anchored identity (Fix 0+A+C+D) | **Evidence RETRACTED — effect UNKNOWN** | Split-aware ping binding (Fix 0), D4 thread consumption (Fix A), hard no-drop (Fix C), carrier tests (Fix D). Code changes remain in codebase. **RETRACTED (SWEEP-3b):** The "+22.7pp improvement" claim (40.5% → 63.2%) is INVALID — both figures were computed from the same `person_tracks.parquet` (mtime Jun 7 13:05, 37 min BEFORE the CP-TAG-4a commit at 13:42). The 40.5% is full-range; 63.2% is val-split (frames 2500-3000) — a frame-selection effect, not a code-change effect. CP-TAG-4a's actual effect on correct_id is UNKNOWN. Fresh A/B measurement is a prerequisite before CP-TAG-4a's code can be trusted as net-positive; until that measurement exists, treat Fix 0+A+C+D as unvalidated, not proven harmful — do not revert based on this finding alone, and do not cite it as a win in any future planning. Downstream conclusions that treated CP-TAG-4a as a confirmed win (including "helped non-tagged identities") are unsupported. See `tools/sweep/diagnostics/blast_radius_check.md`. |
| CP-TAG-4b: Hard ping connectivity | **Deferred** | Forces tag thread through both pings. Gated on CP21 appearance costs. Soft thread visits the correct tracklet but drifts — hard connectivity without appearance costs risks misrouting. |
| GROUP dilution of tag identity | **Diagnosed** | D0.5 split products create GROUP nodes. Fix A gives exactly 1 tag→person mapping; GROUP dilution of the ENTITY path remains. **CP-PURITY-3 finding:** D1 forms GROUPs from tracklet LIFECYCLE EVENTS (merge/split), not proximity. Capacity=2 is D3 metadata, not D1-enforced. GROUPs are structurally unnecessary when detection is perfect (GT→D oracle produced 0 GROUPs with 0 logic gaps). Evidence: `docs/evidence/cp_purity_3/`. |
| CP-PURITY-1: Through-line decomposition | **Complete — numbers unverified (SWEEP-3b)** | Tagged athlete's entity (p0022) follows the WRONG person — majority GT track is 7, not tagged 8. 100% attributed to ILP stitch routing, NOT D4 emission, NOT detection-tracking. Tracklet-level purity is healthy (0.88-0.92). **Staleness exposure:** measured against pre-CP-TAG-4a `person_tracks` (mtime Jun 7 13:30). Structural finding (corruption enters at solver layer) is an architecture claim likely sound regardless of specific person_ids. Quantitative breakdown should be re-verified against freshened artifacts. Evidence: `docs/evidence/cp_purity_1/`. |
| CP-PURITY-2: Aggregate reconciliation + floor | **Complete — reconciliation RETRACTED (SWEEP-3b)** | The "+22.7pp" reconciliation that this checkpoint produced is INVALID (see CP-TAG-4a retraction above). The pair-box decomposition (72-84% mishandled) and addressable ceiling (35.9%/70.4%) were also computed from the same pre-CP-TAG-4a `person_tracks` and should be treated as unverified. Evidence: `docs/evidence/cp_purity_2/`. |
| CP-PURITY-3: GT-through-D oracle | **Complete** | Ran real D1 on perfect GT detections (one clean tracklet per person). Oracle produced 0 GROUP nodes — structurally correct (no lifecycle events with continuous tracklets). 100% of the former "group-formation defect" (29.9%/11.6%) is detection under-segmentation wearing a D1 costume. D1 has 0 genuine logic gaps. The lever is detection pair separation, not D1 graph logic. Evidence: `docs/evidence/cp_purity_3/`. |
| CP-RASTER-PLATE: Median-background masking | **Complete** | Clean plate (0% ghosts, 72% mask coverage, 0/158 absorbed). But measured in V-blind H+S space → NO_GO was INVALID (see CP-RASTER-PLATE-2). Evidence: `docs/evidence/cp_raster_plate/`. |
| CP-RASTER-PLATE-2: V-channel separability | **Complete — GO** | H+S+V AUC=0.907 vs H+S=0.815 (+9.2%). V-only AUC=0.894. Mask halves intrinsic floor (28.2%→14.6%). Same-color AUC ~0.69 (appearance strong on different-color, weak within same-color). V-extension is a production fix independent of masking. Evidence: `docs/evidence/cp_raster_plate_2/`. |
| CP-REID-1: BoT-SORT ReID experiment | **Rejected — DOMAIN GAP** | Generic `osnet_x0_25_msmt17` rejected for DOMAIN GAP (street pedestrian model vs overhead fisheye grappling), NOT for color-blindness. The V-histogram win does NOT reopen deep ReID. A cheap-HSV-tracker-embedding is a separate, lower-priority question (tracker purity 0.9, low headroom). Config: `with_reid: false`. |
| CP-SPLIT-VALIDATE: GT-validate D0.5 splits | **Complete** | GT-validated ALL D0.5 splits (pre-V and post-V, all tiers). New T3 splits: 2.4% correct, 77.5% spurious. Pre-V T3 also low (4-20%). T2 kinematic: 6-22%. Spurious shape: 61% motion-shadow/pose (V noisy during motion), 28% sustained-same, 11% blip. Threshold sweep: no single cutoff fixes (overlapping distributions). k=2 swaps dominate (27%), k≥3 rare (4%). Change-point feasibility: mixed (30% impure segmentable). Design finding: V unreliable during motion → motion-aware channel weighting needed. Evidence: `docs/evidence/cp_split_validate/`. |
| CP-SPLIT-1: Post-D0 tracklet splitter | **Active — NET-NEGATIVE, TIER 3 DISABLE RECOMMENDED** | D0.5 net-negative on ALL cameras (CP-GT2ACTUALS-4+5). vid2: 35 correct / 317 false (net -282). Tier 3 owns 79% of damage (-222). CP-GT2ACTUALS-6 signal analysis: NO per-frame signal separates false from correct splits (HSV Bhatt 0.035 vs 0.040). Disabling Tier 3 removes 241 false splits at cost of 19 correct (5.4%). **Interim recommendation: disable Tier 3.** Current config: `stage_D.d05_split`, threshold 0.15, corroboration 2×, min_dwell 5. |
| CP-GT2ACTUALS: Dense error map | **Complete** | Dense per-(frame, gt_track_id) error map with jump detection + D0.5 reconciliation. Family-aware split lookup fix (CP-3). Signal_trace has same bug but locked numbers safe (CP-3.5). Stage A (tracklet_drift) is #1 damage source at 41% of vid2 jumps (CP-6). Module: `src/pipeline_validation/gt2actuals/`. Evidence: `docs/evidence/cp_gt2actuals_*/`. |
| RECORDER-TIMING-1/2: Per-frame timing preservation | **Complete** | Both VFR (REENCODE=0) and CFR+sidecar (REENCODE=1+showinfo) capture real per-frame timing (773+ unique deltas vs 2 for CFR baseline). Video byte-identical with/without showinfo (confirmed MD5). VFR breaks GT frame-comparability; CFR+sidecar is purely additive. Nest camera now 15fps (was 30fps Mar 2026). Evidence: `docs/evidence/recorder_timing_1/`. |
| RECORDER-SIDECAR-1: Production timing sidecar | **Active — schema v5 (CP-R13b)** | Per-segment `.timing.jsonl` sidecar alongside every mp4. Schema v5 contract: `docs/reference/sidecar_contract.md`. Frame rows derived from mp4 PTS (row count = decode count by construction, CP-R13b). Key fields: `frame_index` (join key to Stage A, guaranteed 1:1), `pts_time_s`, `dt_s` (per-frame interval, null on frame 0), `nominal_dt_s` (median-based reference), `is_bimodal` + mode fields, `source_pts` validity gate. `row_source` distinguishes `"mp4"` / `"mp4_regenerated"` / `"showinfo_grid"` (legacy). `showinfo_residual` is the drop signal. `input_n` removed (was deprecated in schema 4). `host_arrival_s` may be absent on individual rows (showinfo join miss). COLLECTION ONLY — no CV pipeline stage consumes it yet. Deferred consumers: BoT-SORT frame_rate fix, dynamic-fps metrics, cross-camera sync, variable-dt Kalman step. |
| SOURCE-PTS-1: Source PTS capture | **Active — default** | `SOURCE_PTS` and `FPS_PASSTHROUGH` now default to `1` in diag_v6/v7_2/v8.sh (CP-R3). Preserves camera RTP capture timestamps (`-copyts`, no wallclock override) and VFR passthrough (no CFR resampling). Rollback: `SOURCE_PTS=0 FPS_PASSTHROUGH=0`. Sidecar includes `host_arrival_s`, lower-envelope `pts_wallclock_offset_s`, windowed drift (ppm). Per-attempt stderr files handle retry loop. Rehearsal (7 min, FP7oJQ + PPDmUg): PTS uniform 96-100%, 15fps measured, middle segments mismatch:false (exact 1:1), first/last segments boundary mismatch. Runbook: `docs/guides/runbook_cross_camera_capture.md`. **EXONERATED (RECORDER-RELIABILITY-1):** 16 segments across 3 cameras, zero timestamp/DTS errors. All failures were RTSP 404, session invalidation, or 401. Dup/drop reduced 10-50x vs arrival-PTS. |
| RECORDER-RELIABILITY-1: Production reliability | **Complete** | Five fixes in `diag_v6.sh`: (1) RTSP socket timeout 10s (top fix — dead-stream gap 2m28s→~10s), (2) stop_stream before regenerating (prevents session orphaning), (3) access token refresh per attempt, (4) failure-type-aware backoff, (5) sidecar extraction backgrounded. Source-PTS dup/drop verdict: pixel-identical dups 10-50x lower — **camera-dependent** (PPDmUg exact at 15fps; FP7oJQ ~8% mismatch at 13.85fps). Evidence: `docs/evidence/recorder_reliability_1/`. |
| RECORDER-RELIABILITY-2: API quota awareness | **Complete** | RELIABILITY-1 increased API calls to ~17/min, triggering 429 (SDM quota: 10 QPM per user per project, shared across all cameras). Fixes: (1) optimistic URL reuse (0 API calls when session valid), (2) conditional stop_stream (skip for dead sessions — 400 body confirms already terminated), (3) 429 backoff 60s→300s with Retry-After, (4) generate 404 fail-fast (3 retries), (5) consecutive failure escalation (5+ failures → slow-poll 120-300s), (6) cross-camera quota: N_CAMERAS from v7_2, dynamic min retry interval from 70%×10QPM/N, jitter on every backoff. Estimated: healthy ~1 QPM, 1-failing ~3-4 QPM, all-failing ~7 QPM (escalates to ~1-2). Evidence: `docs/evidence/recorder_reliability_2/`. |
| RECORDER-COVERAGE-1 + BACKLOG-1: Content delivery lag | **Fixed — validated at full scale** | CP-R8 (2026-08-19, FP7oJQ): 11 segments, 1,134s footage in 2,552s elapsed. The "inter-segment gaps" were delivery lag, not lost footage — content is contiguous across all segment boundaries within an attempt (verified visually). **Root cause:** ffmpeg `-t "$(( DEADLINE - now ))"` computes content duration from wall-clock remaining. Under arrival-PTS (pre-passthrough) this was accidentally correct (content time = wall time). Passthrough decoupled them, turning a coincidence into a defect. Expression unchanged since `745b1b4` (March). The March recorder was not better — tidy segment spacing was an artifact of the timestamp domain being the same as the clock domain, while CFR padded duplicates. **Fix (RECORDER-BACKLOG-1):** `TARGET_CONTENT_SECONDS` mode sets `-t` to min(content remaining, wall-clock remaining). Content accounting persists across attempts. Wall-clock safety cap (default 5× target; original 5× sizing assumed ~0.20× average from a ramp model that is now debunked — actual reconnect cost is ~10–13s of connection setup plus lost backlog, so 5× is far more conservative than intended, but harmless; do not re-derive from the debunked ramp). Legacy `WINDOW_SECONDS` preserved. **Validated (2026-08-23):** FP7oJQ `termination=content_target`, 1,798s captured against 1,800s target (2s within tolerance), 2,000s wall, 17 segments. All sidecars schema 5, `source_pts: true`. Three smoke-test-caught bugs (`f4e2f52` log() before definition, `c261502` local declarations in main loop, `01bb0cd` 10s tolerance) — bug 1 was fatal under `set -e`. **Delivery rate:** approximately real-time (1.0×) in steady state. Per-segment instantaneous rates (attempt 1, segments 1–11): 0.995–1.006×. The "0.076→0.992× ramp" from ffmpeg `speed=` was a cumulative-metric artifact (startup cost amortizing), not relay warm-up. Attempt 9 delivered 137s in 99s wall (1.376×) after two failed attempts — the sole above-real-time observation, consistent with relay draining backlog. **The coverage problem is NOT solved in general.** Wednesday's CP-R8 capture (864s in 1,634s wall) was genuinely sub-real-time for an extended period; Aug 23 proves the relay CAN deliver at 1.0× and the fix handles the condition correctly, not that the condition has disappeared. **Reconnect cost:** each reconnect loses the relay backlog and incurs ~10–13s of connection setup (not a rate re-climb — delivery resumes at ~1.0× once data flows). Evidence: `docs/evidence/recorder_coverage_1/findings.md` (HISTORICAL), `docs/evidence/recorder_coverage_2/findings.md` (CURRENT). |
| RECORDER-MUXER-PTS-1: Segment-start duplicate PTS | **Fixed (2026-08-24) — verification capture pending** | **Root cause:** NOT the segment muxer. The RTSP relay sends two H.264 frames (a B-frame and an IDR) at the same RTP timestamp when a client reconnects. The decoder outputs both, creating a duplicate PTS in the filter graph. Showinfo confirms: n:1 (B-frame) and n:2 (I-frame) share identical PTS and identical mean/stdev across all 11 affected segments. **Checksum census (11 segments):** 6/11 pixel-identical (MATCH), 5/11 differ (B-frame prediction residuals, not different captures). Both cases represent one camera capture moment — dropping one loses zero temporal coverage. **Fix:** `select` filter in `-vf` chain drops frames with duplicate PTS: `select='isnan(prev_pts)+not(eq(pts\,prev_pts))'`. No-op on clean segments. `-avoid_negative_ts make_zero` retained (compensates for negative PTS from pre-IDR B-frames at stream start). **Hypothesis (not proven):** the B-frame and IDR share one RTP timestamp because they represent the same GOP boundary moment; consistent with the data but the exact H.264/relay mechanism is not established. The 11 already-affected segments remain unusable (ingest-side recovery out of scope). **Downstream blast radius (3 consequences beyond the recorder):** (1) Piece 12: MP4 muxer rejects duplicate PTS with EINVAL, 6/18 exports failed — fixed with duplicate-PTS skip. (2) CP4.B: `dt_s <= 0` guard in D0 kinematics. (3) Piece 11: tracker guard rejected dt_s=0.0 — fixed by relaxing `<= 0` to `< 0` (the Kalman filter handles 0.0 as a position no-op by design). Pre-fix footage (pre-`select` filter) triggers all three. Evidence: `docs/evidence/muxer_pts_1/findings.md`. |
| Coast architecture: variable-dt Kalman | **Implemented (Piece 11) — T3 complete, result negative, held with caveats** | Subclass (not fork) of boxmot's `KalmanFilterXYWH` and `BotSort` at `src/bjj_pipeline/tracking/`. Uses dt ratio (`dt_s / nominal_dt_s`) to keep velocity in px/nominal-frame. Both KF sites replaced (`self.kalman_filter` + `STrack.shared_kalman`). Track lifetime via wall-time `max_lost_seconds` (default 2.0s = today's behavior) — `frame_rate` eliminated. Toggle: `variable_dt: true/false`. T1 PASS (constant-cadence = stock bit-for-bit). T2 PASS (A→F on low-dispersion 202832, A→D on high-dispersion 201606 — Stage E failure is pre-existing CP22, confirmed identical with variable_dt=false). Piece 8 dissolved. **T3 result (2026-08-30):** A/B on FP7oJQ-20260822-132650 (8-person GT, 1,764 frames). correct_id 37.2% (control) → 34.3% (treatment), -2.9pp. Misattribution rose 17.7%→20.6%. Recall/precision unchanged (detection-independent). dt_s=0.0 guard fix required (MUXER-PTS-1 duplicate PTS at frame 2). max_lost_seconds confound ruled out (30 × 0.067s = 2.010s vs 2.0s, 0.5%). **Scope limits (not a verdict):** single clip, 2.1% gap rate (near worst case for showing benefit), 37.7% of frames undetected, sixth inversion of the identity lever. Both arms preserved at `outputs/_t3_arms/`. Evidence: `docs/evidence/piece11_t3/findings.md`. Process-noise dt-scaling is a recorded follow-up (second-order under ratio formulation). |
| `is_bimodal` is diagnostic-only, never an experimental grouping | **Decided** | Under TIMING-PRINCIPLE-1 the pipeline consumes per-frame `dt_s`; no code reads `is_bimodal` and none should. Grouping analyses by it reintroduces a classification the implementation removed, and conflates gaps with mode switches — CP-R11 established these as distinct phenomena that both appear in the `dt_s` distribution. FP7oJQ's ~8% periodic gaps mean a `is_bimodal: false` segment can still have high dt dispersion (CP-R8 segment 202148: 48.3% dispersion, `is_bimodal=False` — highest in the corpus). Group by a continuous dispersion measure instead. Evidence: `docs/evidence/timing_dispersion_1/findings.md`. |
| CFR rollback path (SOURCE_PTS=0, FPS_PASSTHROUGH=0) | **Retained — last-resort escape hatch, degraded output** | CFR rollback records with arrival timestamps and resamples onto a uniform grid. Produces visibly jittery footage (user observation 2026-08-17: passthrough segments play smoothly, CFR rollback segments show lag-then-speedup from frame duplication/dropping by the encoder filling a uniform grid from bursty input). Broke silently for a full day during CP-R13a (segment muxer failure, found only by bisecting). Retained because degraded footage is recoverable (the sidecar still carries showinfo timing); no footage is not. The passthrough path depends on `-enc_time_base 1/90000` + `-fps_mode passthrough` + `-copyts` — a three-option combination whose interaction with future ffmpeg versions is not guaranteed (same class as `-stimeout` disappearing in 7.x). If passthrough fails, CFR rollback gets recording working — but footage should not be used for GT or athlete-facing clips without re-evaluation. Rollback: `SOURCE_PTS=0 FPS_PASSTHROUGH=0`. |
| Pipeline CFR refusal policy | **Decided** | If a clip's sidecar has `timing_mode: "cfr_grid"` or `source_pts: false`, the pipeline refuses the clip with a typed exception (`SidecarValidityError`). **No fallback path.** Falling back to `1/nominal_dt_s` reintroduces the scalar assumption TIMING-PRINCIPLE-1 exists to remove; falling back to the old frame-to-time conversion produces plausible-looking wrong velocities — the worse failure because nothing surfaces it. CFR rollback is already documented as producing degraded footage (see CFR rollback path row). A clip the pipeline cannot time correctly must fail visibly. Implemented in `f0_sidecar.load_sidecar()`. |
| Sidecar schema-4 policy | **Decided** | Pipeline reader (`f0_sidecar.load_sidecar()`) refuses schema <5 with `SidecarSchemaError`. `parse_sidecar()` accepts schema 4 for tooling (probe, analysis). Between versions: `input_n` removed, `mismatch` semantics inverted, drop signal relocated to `showinfo_residual`. Schema-4 footage **remains valid at the gap level and as a regression baseline** — refusing it in the pipeline reader is a timing-path policy, not a statement that the footage has no value. Schema-4 footage cannot be regenerated to schema 5 (`regenerate_sidecar.py` refuses pre-R13a containers). |
| TIMING-PRINCIPLE-1: Read time, don't convert | **Decided — prerequisite RESOLVED (CP-R13b)** | The pipeline should read time from the sidecar (`pts_time_s`, `dt_s`) rather than converting between frames and seconds via an fps scalar. Frame↔time conversion is itself the defect; with per-frame timing available, most conversions should be **deleted, not corrected**. Two exceptions: (1) boxmot `frame_rate` — a scalar by construction, sets `track_buffer` lifespan; addressed by the variable-dt fork (same principle, applied inside a dependency — see Coast architecture row). (2) `cv2.VideoWriter` / Stage F CFR re-encode — requires a scalar output fps; athletes never receive VFR clips. Fix taxonomy (DELETE-CONVERSION / FIX-SCALAR / FORK / DEAD-VESTIGIAL / AUDIT-ONLY) applied per site in `docs/evidence/timing_audit_1/findings.md` §0. **Prerequisite RESOLVED (CP-R13b):** Sidecar frame rows now derived from mp4 PTS. Row count = decode count by construction. `frame_index` maps 1:1 unconditionally — no `min(a, c)` guard needed. Showinfo retained only for `host_arrival_s` and drift, joined by PTS value. Schema 5. `a_eq_c = True` verified on all 6 fresh segments. DEL-CONV pieces 4–6 and FORK piece 11 unblocked (Pieces 3, 8, 10 dissolved/resolved). Evidence: `docs/evidence/frame_index_join_1/findings.md`, `docs/evidence/mp4_timing_precision_1/findings.md`. |
| `timestamp_ms` int-ms precision | **Closed (Piece 0b)** | All sidecar `pts_time_s` values land on exact integer milliseconds at the 1/90000 timebase (5940/90=66, 6030/90=67, 11970/90=133, 12060/90=134). `int(timestamp_ms)` rounding in `FrameIterator` is lossless on post-R13a footage. No float column needed. Piece 11 reads `dt_s` from the sidecar directly for sub-ms precision. Evidence: `docs/evidence/frame_index_join_1/findings.md` §10 precision finding. |
| Sidecar required for pipeline timing | **Implemented (CP4.A)** | The pipeline requires a valid schema-5 sidecar for all timing. No filename-anchor fallback, no legacy-footage timing path. **Rejected alternative:** anchoring session timestamps on `parse_clip_timestamp` (1-second filename resolution) — fails on the most common cross-clip stitch (person crossing a segment cut, real gap ~67ms, overestimated ~15× by a 1s anchor, rejecting valid reconnects via `dt_max_s`). `pts_wallclock_offset_s` (±14–56ms) is the right order of magnitude. **Cost:** legacy footage cannot score timing correctness. It **can** still run for T1 equivalence and T2 regression via the synthetic-sidecar generator (`f0_sidecar_testutil.py`) with constant-`dt` timing, clearly marked as synthetic. Consistent with `load_sidecar()` refusing schema <5 and with the pipeline CFR refusal policy. **Enforcement: gate at ingest — IMPLEMENTED (CP4.A).** `_validate_sidecar_ingest()` called in `run_pipeline()` after config resolution, before any stage dispatch. Covers all production paths (CLI, processor, validation). Legacy corpus runs via synthetic sidecars under `{out_root}/_synthetic_sidecars/`, gated on `stages.ingest.allow_synthetic_sidecars: true` (default false). Provenance (`"real"` or `"synthetic"` + resolved path) stamped in `sidecar_validated` audit event. `load_sidecar()` accepts optional `sidecar_path` override for the synthetic path — policy checks (schema, timing_mode, source_pts) apply identically. **`frame_iterator.py` fps fallback deleted (CP4.A):** frame 0 exempt by index (POS_MSEC returns 0.0, falsy); frames >0 fail loudly if POS_MSEC <= 0 (RuntimeError — core/ cannot import PipelineError without upward dependency). Legacy corpus verified: POS_MSEC > 0 on all frames >0 across all 4 GT clips. **Provenance is path-derived, not content-inspected.** Synthetic sidecars must never be written as mp4 siblings outside tmp test fixtures. A synthetic sidecar in a sibling location is reported as `"real"` because nothing inspects content. `tests/test_orchestration_cli.py` writes a synthetic sibling in tmp (correct precedent — tests only). **Shared resolver:** `resolve_sidecar_path()` in `contracts/f0_paths.py` is the single implementation of the `allow_synthetic` + `_synthetic_sidecars` lookup. Both the ingest gate and future Phase 2 consumers (CP4.C, CP4.E) call it. `cfg_get()` also moved to `f0_paths.py` to prevent divergent config readers. |
| Checkpoint-2 Piece 4/5 re-cut | **Decided** | Original split conflated session alignment with cross-camera work. Re-cut: **Piece 4** — Stage D reads time, clip and session, anchored on `pts_wallclock_offset_s`. Absorbs audit sites #5, #6, #7, #9, #10. Site #1 (`session_d_run.py:491`) dissolves here (DEL-CONV consequent — disappears once #9 and #8 stop requesting a session-wide scalar). **Clip-boundary discontinuity handling (CP4.E):** session aggregation classifies each clip boundary as BREAK or PERMIT using a shortfall discriminator: `shortfall = wall_gap − (prev_frames × nominal_dt_s)`. Threshold: `max(2.0s, 10 × nominal_dt_s)`. `attempt` change is an OR condition (catches retry cascades within a window). Shortfall is the discriminator; `attempt` is corroborating — on 2026-08-22 footage shortfall caught all 8 boundaries and attempt missed one (422.7s window reset, att 1→1, because `attempt` is window-scoped per `sidecar_contract.md:85`). BREAK boundaries suppress cross-clip reconnect edges via `session_segment_id`. Per-boundary decisions persisted in `clip_offset_registry.json`. **Permit-branch limitation:** no contiguous cut exists in the current corpus; only T1 synthetic validates it. Evidence: `docs/evidence/cp4e_results/`. Piece 4 must read and log `showinfo_offset_status` per clip — the sidecar anchor is better than filename timestamps, not unimpeachable, and the offset status must be available for post-hoc diagnosis if session stitching looks wrong. **Piece 5** — purely cross-camera: site #8 (`cross_camera_evidence.py:275`) and Tier 2 enabling work. **Open question (RECORDER-COVERAGE-2):** `pts_wallclock_offset_s` derives from `host_arrival_s`; under sub-real-time delivery, arrival lags capture by the accumulated delivery delay. Two cameras at different delivery rates (observed: FP7oJQ 0.94× and PPDmUg 0.25× on Aug 22) would diverge by minutes. The contract's ±14–56ms accuracy figure predates this observation. Verify before planning Piece 5. **Parked idea (Piece 5 / Tier 2):** gym buzzer as cross-camera sync anchor — a genuinely simultaneous physical event across all cameras. Not wired today (`buzzer.py` is a Stage E soft gate, downstream of D). Candidate signal if `pts_wallclock_offset_s` accuracy proves insufficient. Do not implement or schedule. Roadmap: `docs/roadmap/checkpoint2_breakdown.md`. |

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
