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
(51-61%). The fundamental blocker is IoU-only tracking — BoT-SORT cannot distinguish
overlapping bodies. Next investment: domain-specific ReID model fine-tuned on existing
CVAT track_id annotations (zero additional annotation work).
