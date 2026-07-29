# Decisions Archive — Roll Tracker

This file is the full historical Active Decisions Log plus performance baselines and
bug fix history. It is NOT auto-loaded by Claude Code. Access it manually when needed:
`cat docs/decisions-archive.md`

---

## Active Decisions Log

| Decision | Status | Notes |
|---|---|---|
| Supabase as integration hub | ✅ Decided | No direct service coupling |
| NumPy 1.x pin | ✅ Decided | Torch ABI constraint |
| SAM masks deferred | ✅ Decided | YOLO bbox fallback for POC |
| d3_ilp2 as primary solver | ✅ Decided | d3_ilp kept for comparison. Shared helpers in d3_common. |
| AprilTag family: 36h11 (~587 IDs) | ✅ Decided | Cell size optimized for Nest cameras at gym distances. No family migration planned. |
| Check-in mechanism: WiFi SSID+BSSID | ✅ Decided | GPS rejected (indoor unreliable). 3hr TTL auto-expiry. |
| profiles.tag_id not globally unique | ✅ Decided | Unique within (tag_id + gym_id + active time window). Stage F uses check-in to disambiguate. |
| Athlete tag assignment: DB-assigned at signup | ✅ Decided | `tag_id_seq` cycling sequence (0–586). Physical merchandise ships with athlete's tag. |
| Gym membership: single gym per athlete | ✅ Decided | `profiles.home_gym_id` FK. Can relax later. |
| Subscription history: gym_subscriptions table | ✅ Decided | Separate table from day one. |
| Clip identity: denormalized profile IDs on clips | ✅ Decided | Stage F writes tag IDs; uploader resolves tag→profile. Null = unresolved, backfillable. |
| Camera auto-registration: discovery-derived cam_id | ✅ Decided | cam_id = last 6 chars of SDM path. REST upsert on discovery. |
| Recording file path: gym-scoped production path | ✅ Decided | GYM_ID presence is the mode switch. |
| Pipeline ingest path: gym-scoped, backward compatible | ✅ Decided | Both new and legacy paths accepted. gym_id inferred from path structure. |
| Pipeline output path: gym-scoped | ✅ Decided | Legacy fallback: `outputs/legacy/`. |
| Collision detection: uploader tag dedup | ✅ Decided | Signal A (same tag both fighters) + Signal B (>1 check-in). collision_flagged status. claim_clip() RPC. |
| YOLO masks disabled in Stage A | ✅ Decided | use_seg: false. Detection-only YOLOv8n. Mask code preserved for Stage F redaction. |
| MPS auto-detection | ✅ Decided | device: "auto" → MPS > CUDA > CPU. Validated on M1 Air. |
| Phase 1/2 parallelism boundary | ✅ Decided (NON-NEGOTIABLE) | A+C parallel (MAX_WORKERS=2). D+E+F sequential. |
| Native processor execution | ✅ Decided | run_local.sh for Mac. Docker for Linux. |
| Uploader sentinel pattern | ✅ Decided | .uploaded file instead of deleting manifest. |
| Session pooler URL | ✅ Decided | Supavisor port 5432. |
| Processor Phase 1 worker count | ✅ Decided | MAX_WORKERS=2, MPS. QoS P-core pinning. ~1.9 min/clip. |
| caffeinate -is for Mac runs | ✅ Decided | Prevents idle/display sleep. |
| Stale worker cleanup | ✅ Decided | run_local.sh kills orphaned workers at startup and on trap. |
| Session-level Stage D aggregation (CP14c) | ✅ Decided | {clip_id}:{tracklet_id} namespacing. Wall-clock frame offset. |
| Session-level stitching: schedule-based clip grouping (CP14a) | ✅ Decided | SCHEDULE_JSON. SessionOutputLayout. Sentinels. |
| Session-level Stage F export (CP14e) | ✅ Decided | Multi-source extraction. Per-camera manifests + merge. source_video_ids text[]. |
| Stage E two-layer engagement (CP14d) | ✅ Decided | cap2 GROUP seeds + proximity hysteresis. Both optional. Zero matches valid. |
| Cross-camera identity merge (CP14f) | ✅ Decided | Union-find on shared tags. Presence-based. gp_ global IDs. Fallback for CP17. |
| Option B undistort-on-projection | ✅ Decided | cv2.undistortPoints before H. Strict enforcement: project_to_world() only. |
| Calibration pipeline as separate module | ✅ Decided | src/calibration_pipeline/ alongside bjj_pipeline. |
| Inter-camera homography sync | ✅ Decided | Mat walk + least-squares affine. Three correction layers. |
| Multipass mode removed (CP16-cleanup) | ✅ Decided | multiplex_AC is the only execution path. No --mode flag. |
| Stage C tag detection sensitivity tuning | ✅ Decided | k_verify 30→10, n_ramp 60→90, blur.min_var 60→50, motion.dv_thresh 2.5→2.0. |
| Session export manifest overwrite bug | ✅ Fixed | Per-camera manifests + explicit merge step. |
| Session match_sessions overwrite bug | ✅ Fixed | Per-camera scoping + merge. Same class as export manifest overwrite. |
| CP16a: F0 projection utility | ✅ Completed | project_to_world() in f0_projection.py. Debug artifact projection_debug.jsonl. |
| CP16b: Calibration pipeline skeleton | ✅ Completed | Functional lens calibration + 3 stubs. Two-step chain. |
| CP17 Tier 1: Two-pass cross-camera ILP | ✅ Implemented | Tag corroboration. corroboration_miss_multiplier 10x. Must-link bug fixed. |
| Gym setup calibration tool | ✅ Implemented | lens_calibration functional. mat_walk + mat_line_detection implemented. drift_detection stub. CP19 unified calibration wizard (3-step: initial H → lens cal → H refinement). |
| CP17 Tier 2: Coordinate evidence | ✅ Implemented | `build_cross_camera_coordinate_evidence()` compares D4 person tracks across cameras via rolling-window spatial proximity. Merges into `corroborated_tags` for same 10x ILP boost. Conflicts logged as Signal C (audit-only). Config: `cross_camera.coordinate_evidence` (disabled by default until validated on real sessions). |
| CP18: Calibration pipeline | ✅ Completed | Layer 1 (footpath + mat line) + Layer 2 (fingerprint). Affine correction approach abandoned due to J_EDEw regression. Superseded by CP19 direct H refinement. |
| H on disk is mat→img | ✅ Decided | multiplex_runner auto-detects and inverts to img→mat. projected_polylines use mat→img (the on-disk direction). |
| Footpath primary over edge touches | ✅ Decided | Mat line detection guarded — falls back to footpath-only when combined signal conflicts. |
| Projected polylines saved at calibration time | ✅ Decided | Dense-sampled mat edge points in homography.json. Used by mat_line_detection for line matching. |
| CP19: Unified calibration pipeline | ✅ Implemented | Replaces CP18 affine correction. Phase A (polyline lens cal) + Phase B (mat-line H refinement via RANSAC). Integrated into save handlers + batch recalibration script. Empty-frame selection via temporal median. Results: 1.0-1.3px reproj, 61-82% inliers across 3 cameras. |
| Cross-camera calibration verification | ✅ Complete | `calibration_verify.py` pairwise world-coordinate agreement. 9mm worst-case deviation across 3 cameras. |
| Undistortion pipeline audit | ✅ Complete | All 9 code paths verified correct (2026-04-02). Convention: u_px/v_px = raw pixel, x_m/y_m = world via project_to_world(). See `docs/undistortion_audit.md`. |
| Pose decomposition (v6) as canonical height model | ✅ Decided | Replaces polynomial/affine fitting (v1–v5). Uses K⁻¹@H decomposition → SVD-orthogonalized 3×4 P matrix. Zero training data. |
| H in undistorted pixel space | ✅ Decided | Verified by tracing wizard Step 3 code path. Comments saying "raw" are stale. |
| Lens calibration fixed-f candidate sweep | ✅ Decided | Replaces loose-bounds single optimizer. `_get_f_candidates` from `homography_calibrate.py`. k bounds ±1.0. |
| Camera geometry analysis tool (4-phase) | ✅ Implemented | height surface → ROI mask → detectability → coverage optimization. `tools/camera_geometry_analysis.py`. |
| Stage D coverage root cause (CP4/CP6) | ✅ Diagnosed | Parallel-carrier displacement in D1 graph construction. Not penalty-tunable. 100% of d3_dropped frames have a concurrent kept tracklet on a different GT person. |
| unexplained_tracklet_penalty floor-protected length-proportional (CP3b) | ✅ Implemented | max(base=25.0, per_frame=0.1 × n_frames). Protects short tracklets, adds length pressure. Saturated — can't overcome flow topology. |
| CP3 pure per-frame penalty | ❌ Rejected | Regression — short tracklets became too cheap to drop. Rolled back. |
| GT Person Trace layer (CP6) | ✅ Implemented | Permanent layer in pipeline_validation. Per-frame per-GT-person trace through all stages. Six-mode failure breakdown is now the primary Stage D metric. |
| present_misattributed is a representation ceiling (CP6) | ✅ Understood | Tracklets cover multiple GT persons (33–53 tracklets per GT person in J_EDEw). One person_id per tracklet → inherent misattribution. Needs ReID/pose identity, not routing fixes. |
| Eval baseline preservation includes pipeline artifacts | ✅ Decided | Copy both _eval/ and _eval_gt/{cam}/{clip}/ for full-mode trace. Historical baselines (pre-CP6) are lite-mode only. |
| CP5 parallel-carrier consolidation in D1 | ✅ Implemented | d3_dropped collapsed: J_EDEw 49.7%→7.9%, PPDmUg 39.9%→0%, FP7oJQ 24.0%→4.6%. present_misattributed now dominant (59–66%). Solver OPTIMAL, mergers stable. |
| DetectorConfig.iou: tunable NMS threshold (CP7-pre-6) | ✅ Ratified | Optional[float]=None. Default-inert (proven by artifact-diff regression: detections `0ceee2a1…`, person_tracks `8e6383d2…` identical pre/post). Setting iou bypasses CoreML → .pt + disables end2end NMS. Runtime WARNING emitted. See entry below. |
| CP-EVAL-1: Frozen eval instrument v1.0 | ✅ Decided | Single-path Layer 1/2 (cdf1037). Hungarian IoU 0.5. Identity mapping: per_frame_matches + person_tracks. Spec: `docs/eval_instrument_spec.md`. |
| CP-REID-1: BoT-SORT ReID experiment | ❌ Rejected | Generic osnet_x0_25_msmt17 — negligible improvement, 2-3x runtime overhead. Domain gap too large for overhead fisheye. (84157bb) |
| CP-SWAP-1: Tracker-swap diagnostic | ✅ Complete | 167 GT-oracle swaps, best AUC 0.663 (bbox_aspect_change). Marginal single-feature separability. Module: `pipeline_validation/tracker_swap/`. (b989832) |
| CP-SWAP-2: Swap pattern characterization | ✅ Complete | 47% hop_into_unoccupied, 28% cascade, 2% exchange. 41% transient. 45% no kinematic spike. Informed splitter design. (3afee17) |
| CP-SPLIT-1: Post-D0 tracklet splitter | ✅ Implemented | Tiered detection + dwell filter at D0.5. present +14.6/+4.8/+4.4pp, misattr -8/-7/-5pp vs CP5 baseline. Config-driven thresholds. (fce5758, validator fix af258b7) |
| ROI mask union fix | 🔲 Pending | Replace band polygon with `foot_poly.union(head_poly)` in `run_phase2`. |
| Processor service dockerization | 📋 MVP task | Pipeline runs natively now. Docker for Linux deployment. |
| Notification channel for drift alerts | 📋 TBD | Supabase Realtime likely. |
| Gym owner web app stack | 📋 TBD | Blueprint + homography calibration UI. |
| Flutter app state | 📋 Draft | Tested on Pixel 7 Pro. Not production-ready. |
| Pricing/subscription tier model | 📋 TBD | Gym-level, usage-based likely. |

---

## Performance Baseline

Current representative baseline (M1 Air, MPS 2-worker QoS, 36 clips):
- Phase 1 (A+C): ~1.9 min/clip → ~69 min representative
- Phase 2 (D+E+F): ~68 min sequential
- Total: ~120 min representative (173 min actual including stale worker contamination)

## Bug Fix History

- **Run 1 (2026-03-20):** 30/36 failed — degenerate bbox bug. Fixed ab526b7.
- **Run 2 (2026-03-21a):** 7 Phase 2 errors — Stage D/F bugs. Fixed 4e825a4.
- **Run 3 (2026-03-21b):** 34/36 manifests. 2 remaining D edge cases.
- **Run 4 (2026-03-22):** 35/36 manifests. 1 remaining: PPDmUg-202751 (NAType in frame_index).

## Known Open Issue

PPDmUg-20260318-202751 fails at D2 — `int(bank_df["frame_index"].min())` returns NAType.
Degenerate clip with extremely sparse tracklets. Needs null-safe integer handling in
D2 `compute_edge_costs()`.

## CP7-pre-6: NMS IoU Tunable — Ratification Record (2026-05-22)

**What:** `DetectorConfig.iou: Optional[float] = None` added to `models.py`, wired through
`detector.py`, `run.py`, and `multiplex_runner.py`. Committed cf823be, ratified retroactively.

**Default-inert proof:** Artifact-diff regression test ran full A→E pipeline on FP7oJQ
(4530 frames) before and after the plumbing change with iou unset. Both runs loaded the
production CoreML backend (`models/bjj-detect-all-cameras.mlpackage`). Results:
- `detections.parquet`: `0ceee2a176a7164ec1e7a3d481772c3f` (identical)
- `person_tracks.parquet`: `8e6383d25d5e954e36632043ffe5ba2b` (identical)

Both regression arms confirmed loading CoreML (the production inference path), so the
proof exercises the actual production backend, not a weaker .pt-only test.

**End2end/CoreML double-NMS finding:** YOLOv26n models have NMS baked into the model
graph (`model.end2end = True`). The ultralytics `model.predict(iou=X)` kwarg is ignored
when `end2end` is active — Python-side NMS is bypassed. CoreML exports additionally bake
NMS into the compiled model. The `iou` kwarg is therefore doubly inert on the production
CoreML path. Making NMS tunable required: (1) skip CoreML, load .pt directly;
(2) set `model.end2end = False` and `model.model[-1].end2end = False` on the Detect head.

**Coupling when iou is set:** Setting `iou` to any non-None value triggers:
1. CoreML bypass (falls back to .pt weights on MPS)
2. End2end NMS disabled on the model graph
3. Python-side NMS activated with the specified IoU threshold
4. Runtime WARNING logged with perf impact (~32 fps .pt/MPS vs ~79 fps CoreML/ANE)

This means setting `iou` is NOT just a threshold tweak — it substitutes the inference
backend. Detection output will differ from production CoreML even at `iou=0.7` (matching
the ultralytics default) because the NMS implementation differs.

**Sweep caveat:** All CP7-pre-6 sweep numbers (docs/cp7_pre6_nms_sweep.md) were produced
on .pt with Python-side NMS, NOT production CoreML/ANE. The cross-arm trend (relaxing NMS
monotonically worsens misattribution) is valid; the absolute numbers are not
production-comparable.

**Sweep conclusion (settled):** NMS relaxation ruled out as standalone fix. Every
relaxation step worsened misattribution (4.0% → 25.1%), fragmentation (1.0 → 4.5
tracklets/GT), and solo-context regression. See docs/cp7_pre6_nms_sweep.md.

## Applied Migrations (23 total)

Phase A: 000001–000007 (gyms, gym_members→dropped, subscriptions, checkins, homography, columns, correction)
Phase E: 000001–000008 (RLS+trigger, profiles fixes ×4, checkin source+tag seq, storage policies)
Cameras+recorder: 000001–000005 (cameras, log_events app_version, checkin upsert unique,
clips collision status, claimable clips RPC, device_tokens, log_events insert policy)
CP14e+f: 000001–000002 (clips source_video_ids, clips global_person_ids)

## Identity Quality Investigation Arc (CP-EVAL-1 → CP-SPLIT-1, 2026-05-22/23)

**Problem:** After CP5 (parallel-carrier consolidation), `present_misattributed` dominated
at 59-66% across all cameras. Stage F match preview videos showed visible identity jumping
during grappling — BoT-SORT swaps which detection it assigns to which tracklet when people
overlap.

**Investigation sequence:**

1. **CP-EVAL-1** (cdf1037): Froze the evaluation instrument to ensure all subsequent
   experiments are measured on the same yardstick. Single-path Layer 1/2, Hungarian IoU 0.5.

2. **CP-REID-1** (84157bb): Tested BoT-SORT's built-in ReID (`osnet_x0_25_msmt17`).
   Result: negligible improvement. The pedestrian ReID model trained on MSMT17 has too
   large a domain gap from overhead fisheye BJJ footage. 2-3x runtime overhead not justified.

3. **CP-SWAP-1** (b989832): Built GT-oracle swap diagnostic. Identified 167 swap events
   across 68 tracklets. Measured GT-free signal separability: best single-feature AUC=0.663
   (`bbox_aspect_change`). Marginal — multi-feature detector needed.

4. **CP-SWAP-2** (3afee17): Characterized swap patterns. Key findings:
   - Only 2% are clean two-body exchanges; 47% are one-sided hops, 28% cascades
   - 41% are transient flickers that self-correct (50% last exactly 1 frame)
   - 45% show no kinematic spike (gradual drift, not teleportation)
   - 81% occur within 0.5m (grappling proximity confirmed)

5. **CP-SPLIT-1** (fce5758): Built tiered tracklet splitter at D0.5 (post-D0, pre-D1):
   - Tier 1: Hard speed cap (48 m/s — teleportation)
   - Tier 2: Kinematic spike (5x median, min 5 m/s, 3x isolation ratio, ≤2 frame duration)
   - Tier 3: Histogram Bhattacharyya (0.15 threshold, 2x kinematic corroboration)
   - Min dwell filter: 5 frames (avoids splitting on transient flickers)

**Results vs CP5 baseline:**

| Camera | present | misattributed |
|--------|---------|---------------|
| FP7oJQ | 6.4% → 21.0% (+14.6pp) | 59.0% → 51.0% (-8.0pp) |
| J_EDEw | 7.4% → 12.2% (+4.8pp) | 61.0% → 54.0% (-7.0pp) |
| PPDmUg | 10.6% → 15.0% (+4.4pp) | 66.0% → 61.4% (-4.6pp) |

d3_dropped unchanged across all cameras. Splitter thresholds are initial calibration.

**Current ceiling:** ~35-40% present without new models. Misattributed remains dominant
(51-61%). The fundamental blocker is detection under-segmentation — see CP7 investigation
below.

## CP7 Misattribution Decomposition (pre-8 → pre-10, 2026-05-25)

**Problem:** After CP-SPLIT-1, `present_misattributed` remained dominant (51-61%).
CP7-pre-3 had established ~70% was detection under-segmentation, but the pre-8
Axis-1/Axis-2 investigation attempted to measure the recoverable share for a
Stage D concurrent-swap node vs detection separation.

**Investigation sequence:**

1. **CP7-pre-8** (Axis-1 signature characterization): Classified misattributed frames
   as Branch A (GROUP routing failure) vs Branch B (concurrent-alive tracklet swap).
   Result: apparent 84.3% Branch B, 6.9% Branch A, 33.9% ambiguous. Recommended
   concurrent-swap node class. **SUPERSEDED** — the 84.3% was not measured against
   detection geometry.

2. **CP7-pre-9** (Branch-B margin disambiguation): Applied CP7-pre-3's containment test
   to the two suspect buckets (ambiguous_a_b + branch_b_persistent, 1,811 frames).
   Result: 92.8% of those frames are pair-box (one detection covering two GT people).
   True concurrent-swap margin: 9.9% (223/2,259). Zero concurrent_role (genuine A/B
   co-causation). The "concurrent tracklet holding canonical identity" was a consequence
   of pair-box under-segmentation, not an independent swap failure.

   Tau sweep stability: pair_box share 77.9% (tau=0.9) to 94.7% (tau=0.3). Robust.

3. **CP7-pre-10** (pair-box bracketing): Tested whether pair-box spans ever resolve into
   two separately-tracked boxes elsewhere in the clip (which would enable offline identity
   propagation through the merged span). Horizon sweep: 30/90/300/full-clip frames.
   Result: **0% bracketed at every horizon.** The second person is never separately
   tracked anywhere in this clip → the lever is detection-level pair separation, and
   possibly plain recall on isolated people; the two are not yet separated and the
   separability experiment will distinguish them.

   Fragment-map fix: original run showed 39% indeterminate due to D0.5 split tracklet ID
   mismatch (gt_person_trace uses pre-split IDs, bank_frames uses post-split). Fragment
   resolution (t10→t10_sN) reduced indeterminate to 13%. OPEN: spot-check remapped
   carriers before treating 13% as hard.

**Corrected misattribution hierarchy (FP7oJQ, one 2.5-min clip):**

| Cause | Frames | % of 2,259 | Fix path |
|-------|--------|------------|----------|
| Pair-box, unbracketed | 1,259 | 55.7% | Detection separation |
| Pair-box, indeterminate | 262 | 11.6% | Likely detection |
| Pair-box, half-bracket/short | 160 | 7.1% | Mixed |
| True Branch B (Axis-1) | 223 | 9.9% | Concurrent-swap node (~10% sidecar) |
| Pure Branch A | 157 | 6.9% | GROUP routing |
| Other | 198 | 8.8% | Unknown |

**Conclusion:** Detection-level pair separation is the primary lever (~74% of
misattribution, of which 55.7% is confirmed unbracketed). Stage D concurrent-swap node
deferred as ~10% sidecar. Single-clip finding; confirmation pending on buzzer video
(which forces separations).

Pipeline-attribution caveat: bracket test uses majority-vote GT attribution from
gt_person_trace, most reliable at separation points (benign lean, not ground-truth-
verified outside 0-300).

---

## CP-TRACE Series (CP-TRACE-1 through CP-TRACE-FIX, completed 2026-06-02)

Signal trace investigation: greedy per-GT matcher + topology census + D-stage preservation
trace + E/F extension. Standalone submodule at `src/pipeline_validation/signal_trace/`.

**Corrected root cause ranking (post CP-TRACE-FIX, aggregate all cameras):**

| Root cause | Frame impact | Notes |
|-----------|-------------|-------|
| wrong_id | 28.2% | Identity misattribution. 72% from tight_match frames, 28% from pair_box. |
| pair_box | 23.1% | Detection under-segmentation (one box, two people) |
| miss | 10.4% | Detection recall failure |
| d4_frame_trim | 2.5% | Genuine graph coverage gap (residual after fix) |
| d3_solver_drop | 0.3% | Negligible |

**Key correction:** Original 29% no_id (CP-TRACE-2) was a measurement artifact. The
join-key mismatch between detections.parquet (original tracklet_ids) and person_tracks
(D0.5 split products) caused false negatives. CP-TRIM-1 diagnosed this; CP-TRACE-FIX
implemented split-product resolution via d05_split_audit.jsonl in stage_d_trace.py.
Pre-fix artifacts preserved at `outputs/_eval/signal_trace/bjj-detect-all-cameras_pre_fix/`.

**Cross-tab (Stage A × Stage D):** 72% of wrong_id occurs on tight_match frames (solver/
tracker errors on clean detections). 28% from pair_box frames (detection under-segmentation
directly causing misattribution). Causality note: a tracklet may be tight_match at frame N
but influenced by pair_box at earlier frames in its lifecycle.

**GROUP falsification:** Confirmed irrelevant to pair-boxes. GROUP engagement on pair-box
tracklets is coincidental — triggered by lifecycle events (merges/splits of OTHER tracklets),
not by the pair-box itself. Pair-boxes don't create lifecycle events.

**E/F extension:** All 36 GT people (across 3 cameras) appear in match sessions. Signal
reaches Stage E despite identity fragmentation.

---

## CP-TAG Series (CP-TAG-1, CP-TAG-2, completed 2026-06-04)

Tag signal investigation: does AprilTag identity reach the correct person?

**CP-TAG-1 findings:**
- Tag detection is bbox-gated (Stage C scans padded detection bboxes only via c0_gating.py)
- Tag detection is cadence-gated (C0Scheduler 3-state: SEEKING every frame, VERIFIED every 10th)
- Tag visibility: 4 observations across ~7,500 frames on 2 J_EDEw videos (0.05%)
- Signal chain C→D2→D4 works when tags are detected (2/2 videos with observations)
- Cross-tab: 28% of wrong_id from pair_box, 72% from tight_match

**CP-TAG-2 findings (full-frame scan experiment):**
- Full-frame every-frame scan (no restrictions) found IDENTICAL observations to pipeline
- Physical occlusion is the bottleneck, not pipeline gating
- AprilTag 36h11 at rashguard size is below decode threshold (~25px vs ~40px minimum)
  at ceiling-mount Nest camera distance (~3m)
- Dense GT (stride-1, 10x evaluation points) confirms stride-10 is representative (±0.3pp)

---

## Cross-Tracklet Identity Diagnostic (completed 2026-06-05)

Deep diagnostic of tag identity propagation for the tag_id=1 person across both J_EDEw
videos. Explored: D2 constraints, D3 solver behavior, D4 emit logic, D0.5 split routing.

**Architecture findings:**
- Must_link constraints are SOFT (penalty = 2× miss_penalty), not hard ILP constraints
- Tag identity does NOT propagate across tracklet boundaries — only tracklets explicitly
  in the must_link group carry the tag binding
- D4 assigns sequential person_ids (p0001...), NOT tag-derived. Tag mapping is post-hoc.
- GROUP segments cause multiple person_ids per tracklet per frame → tag identity dilutes

**Data findings:**

| Video | GT | Tracklets | Person_ids | Tag outcome |
|-------|-----|-----------|-----------|-------------|
| J_EDEw-200015 | gt_24 | 29 | 17 | 3 tag person_ids (GROUP dilution); tag covers last 5% of clip |
| J_EDEw-200246 | gt_8 | 30 | 12 (+11 dropped) | Tagged tracklet DROPPED; tag assigned to wrong person via nested detection |

**Three architectural gaps identified:**
1. **Must_link too soft:** Tagged tracklet t99 (862 frames) dropped — penalty insufficient
2. **No path propagation:** Non-tagged tracklets on same path don't inherit tag anchor
3. **GROUP dilution:** Tagged tracklet has 167 person_id transitions (3 separate identity_assignments emitted)

**Planned fixes (updated post-CP-TAG-4a):**
- ~~Hard must_link~~ — FIXED by CP-TAG-4a Fix C (hard no-drop for ping-carrying products)
- Path-based tag propagation — still open
- GROUP dilution mitigation — partially addressed by Fix A (thread consumption)

---

## CP-TAG-4a: Tag-Anchored Identity (COMPLETED 2026-06-07, CORRECTED 2026-06-08)

Fix 0: split-aware ping binding. Fix A: D4 thread consumption. Fix C: hard no-drop.
Fix D: carrier selection tests.

**CORRECTED:** The initial "correct_id decreased (vid1 25.6%→17.6%, vid2 22.2%→19.1%)"
verdict was a METRIC-BASIS ARTIFACT. It compared session-full-range coverage against a
clip-level val-split baseline. CP-PURITY-2 Measurement 1 reconciled: J_EDEw val-split
clip-level correct_id went 40.5% (baseline) → 63.2% (post-CP-TAG-4a) = **+22.7pp
improvement**. CP-TAG-4a helped non-tagged identities.
Evidence: `docs/evidence/cp_purity_2/`, `docs/evidence/cp_tag_4_post/`.

---

## CP-PURITY-1: Through-line Decomposition (2026-06-07)

Tagged athlete's entity (p0022) follows the WRONG person — majority GT track is 7, not
the tagged 8. 100% attributed to ILP stitch routing, NOT D4 emission, NOT detection-
tracking. Tracklet-level purity is healthy (0.88-0.92). The corruption enters at the
solver layer. Evidence: `docs/evidence/cp_purity_1/`.

---

## CP-PURITY-2: Aggregate Reconciliation + Floor (2026-06-08)

Reconciled CP-TAG-4a as +22.7pp improvement (see above). Pair-box 72-84% mishandled
(no GROUP node despite proximity). Addressable ceiling: tight_match 35.9%/70.4%.
Evidence: `docs/evidence/cp_purity_2/`.

---

## CP-PURITY-3: GT-through-Stage-D Oracle (2026-06-09)

Ran real D1 code on perfect GT detections (one clean tracklet per GT person).
**Oracle produced 0 GROUP nodes** — structurally correct: D1 forms GROUPs from tracklet
LIFECYCLE EVENTS (merge/split), not proximity. With continuous GT tracklets, no
lifecycle events occur, so no GROUPs form.

**Key findings:**
- 100% of the former "group-formation defect" (CP-PURITY-2: 29.9%/11.6%) is detection
  under-segmentation wearing a D1 costume. D1 has 0 genuine logic gaps.
- GROUPs are structurally unnecessary when detection is perfect.
- Capacity=2 is D3 metadata (d1_graph_build.py:1179), NOT enforced in D1.
- D0.5 produced 0 splits on clean GT (no false positives).
Evidence: `docs/evidence/cp_purity_3/`.

**Identity-corruption lever journey (re-pointed 4 times):**
1. D4 emission — falsified (CP-PURITY-1)
2. ILP stitch — confirmed as mechanism
3. D1 group-formation — falsified (CP-PURITY-3: 0 logic gaps)
4. Detection under-segmentation — confirmed as root cause

---

## CP-RASTER-PLATE: Median-background Masking (2026-06-09)

Clean plate mechanics: 0% ghosts, 72% mask coverage, 0/158 self-absorption.
But measured in V-blind H+S feature space → NO_GO verdict was INVALID.
Evidence: `docs/evidence/cp_raster_plate/`.

---

## CP-RASTER-PLATE-2: V-Channel Separability (2026-06-09) — **GO**

Corrected the crippled H+S feature space with H+S+V. Results:

| Feature Space | AUC (all) | Same-color AUC | Floor |
|---|---|---|---|
| Baseline H+S (production) | 0.815 | 0.690 | 28.2% |
| Masked H+S+V | **0.907** | 0.693 | **14.6%** |
| V-only | 0.894 | **0.719** | 16.9% |

**V-channel is a production fix independent of masking** (+9.2% AUC). V alone carries
nearly all the missing signal (AUC 0.894). Mask halves intrinsic floor (28.2%→14.6%).
Skin-inclusive beats torso-only (+2.1%). Same-color AUC ~0.69 (appearance strong on
different-color, weak within same-color clusters — value scales with gym color diversity).

**Honest ceiling:** appearance is NOT a blanket solution. It helps when people wear
different colors; it is weak within same-color clusters.
Evidence: `docs/evidence/cp_raster_plate_2/`.

---

## Metric-Basis Discipline (recorded 2026-06-09)

This burned us 3 times during the CP-TAG → CP-PURITY arc. No correct_id number is
comparable without its basis stated:
- Camera set: single vs 3-camera (58.7% is 3-camera, ~40.5% is single J_EDEw)
- Frame range: val-split vs full annotated
- person_tracks level: clip vs session
Canonical definition: clip-level person_tracks, val-split, greedy IoU>=0.3.

---

## CP-REID-1 Rejection Clarified (2026-06-09)

Rejected generic osnet_x0_25_msmt17 for DOMAIN GAP (street pedestrian model vs overhead
fisheye grappling), NOT for color-blindness. The V-histogram win does NOT reopen deep
ReID — the domain gap remains. A future cheap-HSV-tracker-embedding is a SEPARATE,
lower-priority question (tracker purity 0.9, low headroom).

---

## Forward Plan: Appearance Arc (decided 2026-06-09, updated 2026-06-10)

1. ~~Measure separability~~ — DONE (CP-RASTER-PLATE-2, GO verdict)
2. ~~V-channel histogram extension~~ — DONE (CP-HSV-V, commit 753061b)
3. **D0.5 Tier 3 redesign [NEXT]** — CP-SPLIT-VALIDATE showed current mechanism is
   2.4% precision. Memoryless single-frame comparison + flat threshold is broken.
   Motion-aware channel weighting is the leading hypothesis (V noisy during motion).
   Tier 2 also low-precision (6-22%) — may need parallel rethink.
4. **Mask + V-aware appearance into D0.5 + ILP cost layer** — CP21 scope.
5. **Cheap-HSV-ReID** — low priority (tracker purity 0.9).

Evidence-economy principle: tags = hard constraint; appearance = cost/veto, never hard;
distinctiveness-weighted; only one tier may be hard.

---

## CP-HSV-V: V-Channel Histogram Extension (COMPLETED 2026-06-09)

Production histogram extended from H+S (144-dim) to H+S+V (864-dim, 18×8×6).
V_bins=6 validated by AUC sweep (same-color AUC drops at V=4, V=8 adds nothing).
All 58 consumers verified safe (dynamic column discovery). Non-histogram artifacts
unchanged (detection/tracking/graph hashes identical). Commit 753061b.

---

## CP-SPLIT-VALIDATE: GT-Validate D0.5 Splits (COMPLETED 2026-06-10)

GT-validated ALL D0.5 splits (pre-V 144-dim and post-V 864-dim, both tiers, both clips).

**Headline:** New post-V Tier-3 splits are 2.4% correct / 77.5% spurious. But pre-V T3
was ALSO low (4-20%). Tier 2 kinematic is 6-22%. The precision crisis is systemic.

**Spurious T3 shape (the key design finding):**
- 61% motion-correlated shadow/pose artifacts — V (brightness) is noisy during motion.
  Person rotates through shadow while moving. The 2× speed kinematic corroboration gate
  CANNOT catch this because the person IS moving — noise and gate fire on the same
  condition. **This is a strong argument that V should be DOWN-WEIGHTED during high-motion
  frames, not trusted equally.**
- 28% sustained-same-person — genuine persistent appearance shift on one GT person
  (different lighting zones on mat). Change-point detection would also fire here →
  the redesign needs lighting invariance, not just windowing.
- 11% single-point blips — a larger min_dwell alone would suppress these, but they're
  only 11% of the problem.

**Threshold sweep:** {0.15: 3.9%, 0.18: 4.6%, 0.22: 6.7%, 0.25: 8.9%} precision.
Correct and spurious distributions overlap — no single threshold separates them.
Rules out simple recalibration for 864-dim.

**k-distribution:** k=2 swaps dominate (27%), k≥3 rare (4%). The segmenter must handle
k≥2 but k>3 is uncommon.

**Change-point feasibility:** Mixed. 30% of impure tracklets show segmentable clean-point
structure. Appearance alone is insufficient for reliable segmentation.

**Redesign options (no fix chosen):**
(a) min_dwell alone: insufficient (only 11% blips)
(b) Full change-point: premature (30% feasibility)
(c) Threshold recalibration: insufficient (overlapping distributions)
(d) Motion-aware channel weighting: leading hypothesis (61% motion-shadow)
(e) Tier 2 also needs attention (6-22% precision)

Evidence: `docs/evidence/cp_split_validate/`.

---

## CP-GT2ACTUALS: Dense GT-to-Actuals Error Map (2026-06-10)

Permanent per-(frame, gt_track_id) error map joining GT annotations against all pipeline
signals. Module: `src/pipeline_validation/gt2actuals/` (dense_join, node_gt_set, jumps).
CLI: `python -m pipeline_validation gt2actuals --manifest-path <path>`.

Key findings:
- **Split-family lookup fix (CP-3):** 88% of vid2 no_id was a join artifact. Fixed with
  family-aware fallback. Vid2 no_id: 58% → 6.9%.
- **D0.5 net-negative ALL cameras (CP-4+5):** vid2: 35 correct / 317 false (net -282).
  Tier 3 owns 79% of damage.
- **Stage attribution (CP-6):** Stage A (tracklet_drift) is #1 damage source at 41% of
  vid2 jumps. Group handling 33%, solver misstitch 26%.
- **No signal discriminates false from correct splits:** HSV Bhattacharyya 0.035 vs 0.040.
Evidence: `docs/evidence/cp_gt2actuals_{1,3,3_5,4_5,5_5,6}/`.

## Stage A Tuning Sweep (SWEEP-1 through SWEEP-4, 2026-06-30 to 2026-07-03)

End-to-end BoT-SORT tracker param sweep: detection cache → tracker replay → Stage D rerun
→ GT2ACTUALS measurement. Tools at `tools/sweep/`.

- **Freshened eval_gt baseline (SWEEP-3b):** 32.5% combined correct_id. Former 34.7% stale.
- **Sweep baseline:** 30.7% combined (replay path). ~2pp gap is environment artifact (boxmot
  version drift). Sweep deltas internally consistent.
- **CP-TAG-4a evidence RETRACTED (SWEEP-3b):** "+22.7pp" was same-artifact frame-selection
  effect. CP-TAG-4a's actual effect UNKNOWN.
- **OFAT track_buffer (SWEEP-4):** tb={5,10,15,20,30,45,60}. Stock tb=30 is best. Every
  deviation degrades. "Break, don't guess" hypothesis REFUTED. Lower values fragment
  tracklets; solver's length-proportional penalty punishes the fragments.
Evidence: `tools/sweep/diagnostics/`.

## Purity Proxy Analysis (PURITY-PROXY-1/2, 2026-07)

- `max_displacement` (path discontinuity): AUC 0.82–0.85 raw / 0.75–0.82 post-D0.5.
  Same-color robust. Best single proxy.
- Masked appearance: +0.08–0.09 AUC over contaminated. ~0.88 different-color, ~0.73
  same-color.
- **CRITICAL CEILING: smooth-motion + same-color drifts = 58–94% of impurity, invisible to
  BOTH proxies.** Measuring same-color with MASKED histograms raised fraction dramatically
  (vid2 7.7%→69.2%).
Evidence: `docs/evidence/purity_proxy_{1,2}/`.

## Recorder Timing Arc (WALLCLOCK-1 → CAPTURE-TIME-2, 2026-07)

- **WALLCLOCK-1:** Container PTS is synthetic — CFR re-encode discards wall-clock.
- **RECORDER-SIDECAR-1:** Per-frame `.timing.jsonl` sidecar shipped to production. Under
  arrival-PTS, pts_time_s is ±500ms NN approximation (mismatch is normal).
- **CAPTURE-TIME-1:** RTSP stream carries TRUE capture timestamps (source PTS uniform 33ms,
  1.21ms stdev). Recorder discards them via `-use_wallclock_as_timestamps 1`.
- **CAPTURE-TIME-2:** RTCP definitively absent (0 sender reports, all cameras, TCP+UDP).
  Tier-2 alignment ±14–56ms. FP7oJQ −603 ppm clock drift. **Stream fps varies per session
  (15 and 30 both observed). SDP unreliable. Do not hardcode fps.**
- Diagnostic module: `services/nest_recorder/recorder/diag_timing.sh`.
- **ffmpeg option fragility:** `-stimeout` removed in ffmpeg 7.x; correct option is `-timeout`
  (same microsecond units). `debian:stable-slim` silently rolled bookworm→trixie. Pin Debian.
Evidence: `docs/evidence/{wallclock_1,recorder_timing_1,capture_time_{1,2}}/`.

---

## Dup/Drop Qualification (2026-07-29)

The RECORDER-RELIABILITY-1 "dup/drop resolved" verdict is QUALIFIED. There are TWO independent
mechanisms and source PTS only fixed one:

1. **Bursty arrival timestamps** → ffmpeg mis-inferred the frame rate → dup/drop. **FIXED** by
   source PTS (pixel-identical dups reduced 10-50x).
2. **CFR encode target ≠ actual capture rate** → encoder pads/drops to fill the grid. **STILL
   PRESENT.** Evidence: FP7oJQ captures at 13.85fps against a 15fps CFR target →
   `15.0 / 13.85 = 1.083` → sidecar shows input 554 → output 600 (8.3% fabricated). PPDmUg
   is exact (640/640) only because its rate happens to equal the target (15.00 = 15.00).
   Since stream fps VARIES per session (15 and 30 both observed) and BETWEEN cameras in the
   same session (13.85 vs 15.00), any fixed CFR target will mismatch some camera some of the
   time.

**OPEN CONTRADICTION:** RELIABILITY-1 reported FP7oJQ at "0.0% typical" pixel-identical
duplicates, but the sidecar reports 8% fabricated frames. CFR padding produces bit-identical
frames, so `mpdecimate` should detect them. One measurement is wrong. Logged as an open item
to resolve before trusting either number.

---

## Corrupted Footage Caveat (2026-07-29)

**CRITICAL:** All existing GT footage and every measurement derived from it predates the
recorder fixes (RECORDER-RELIABILITY-1/2) and was recorded under the OLD logic:
bursty-arrival timestamps causing large-scale dup/drop (35% input/output mismatch measured
in one case) and a user-visible pause/speed-up jitter artifact.

**Impact on prior conclusions:**
- Duplicate frames inject FALSE ZERO-MOTION into BoT-SORT's Kalman filter.
- Dropped frames inject FALSE TELEPORTS.
- An unknown fraction of the measured "Stage A tracklet_drift" (recorded as the dominant
  damage source at ~41% of vid2 jumps) may be RECORDER-INJECTED rather than a tracking
  limitation.

**This qualifies — without invalidating — the following:**
- GT2ACTUALS drift attribution (CP-6 stage-attribution analysis)
- Purity-proxy results (PURITY-PROXY-1/2)
- Stage A tuning sweep (SWEEP-1 through SWEEP-4)

**Guidance:** Hold these conclusions LOOSELY until re-measured on clean (post-fix) footage.
Keep old GT clips as a regression baseline. Do not discard prior measurements — the
structural findings (lever hierarchy, architectural gaps) are likely sound; the quantitative
breakdowns (41% drift, proxy AUC values, sweep deltas) are the uncertain part.

---

## Remaining Recorder Work (open checkpoint, 2026-07-29)

1. Validate reuse path firing, token refresh past the ~25-min barrier, sustained coverage,
   no 429s in a real capture session.
2. Resolve the 0.0%-vs-8% duplicate-measurement contradiction (mpdecimate vs sidecar input_n).
3. Make `SOURCE_PTS=1` the DEFAULT (currently opt-in) — the "make it canon" step.
4. Decide CFR-target-vs-actual-rate: per-clip CFR target, or accept padding and let the
   pipeline flag duplicates via `input_n` (preferred — no recorder change needed).
5. `N_CAMERAS` div-by-zero guard: `(10*7/10)/N` integer-divides to 0 at N≥8, then `60/0`
   crashes at startup. Add a floor of 1.
6. Pin the Debian version in the Dockerfile (`debian:stable-slim` silently rolled and
   removed `-stimeout`).
