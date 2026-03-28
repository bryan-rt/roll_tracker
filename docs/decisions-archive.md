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
| Gym setup calibration tool | 🔧 In Progress | lens_calibration functional. mat_walk/drift/inter_camera = stubs → CP18. |
| CP17 Tier 2/3: Coordinate evidence | 📋 Planned | Stubbed. Activated by CP18 inter-camera alignment. |
| CP18: Calibration pipeline fleshed out | 📋 Planned | Mat walk, drift detection, coordinate evidence activation. |
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

## Applied Migrations (23 total)

Phase A: 000001–000007 (gyms, gym_members→dropped, subscriptions, checkins, homography, columns, correction)
Phase E: 000001–000008 (RLS+trigger, profiles fixes ×4, checkin source+tag seq, storage policies)
Cameras+recorder: 000001–000005 (cameras, log_events app_version, checkin upsert unique,
clips collision status, claimable clips RPC, device_tokens, log_events insert policy)
CP14e+f: 000001–000002 (clips source_video_ids, clips global_person_ids)
