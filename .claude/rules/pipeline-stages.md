---
paths:
  - "src/bjj_pipeline/**"
---

# CV Pipeline Stages

## Phase 1 — Online (parallel, per-clip via multiplex_AC)
- **Stage A** `detect_track`: YOLO detection + BoT-SORT. Loads
  `calibration_correction.json` when present (CP18 affine correction applied after
  `project_to_world()`; config: `stages.stage_A.calibration_correction.enabled`,
  default True). Outputs: detections, tracklet_frames, tracklet_summaries,
  contact_points (all .parquet), audit.jsonl.
- **Stage B** `masks`: SAM — deferred for POC. Falls back to YOLO bbox.
- **Stage C** `tags`: AprilTag identity. C0 scheduling/cadence, C1 ROI scan, C2 voting.
  Outputs: tag_observations.jsonl, identity_hints.jsonl.

## Phase 2 — Offline (sequential, never parallelize)
- **Stage D** `stitch`: ILP stitching via OR-Tools. D0 bank tables → D1 graph → D2 constraints
  → D3 ILP solve (d3_ilp2 MCF solver, shared helpers in d3_common) → D4 person_tracks.
- **Stage E** `matches`: Two-layer engagement. E0 input validation → E1 cap2 GROUP seeds →
  E2 proximity hysteresis → E3 union+buffer → E4 buzzer gate (optional) → E5 min duration →
  E6 identity enrichment. Zero matches is valid (no exception).
- **Stage F** `export`: ffmpeg clip cutting, Supabase DB write, manifest.

## F0 Contract Layer
- `f0_manifest.py` — ClipManifest (includes gym_id). Init/load/write per stage.
- `f0_paths.py` — ClipOutputLayout, SessionOutputLayout, StageLetter. Canonical path resolution.
- `f0_parquet.py` — Read/write helpers.
- `f0_projection.py` — `project_to_world()` with optional cv2.undistortPoints before H.
  CameraProjection NamedTuple. **Only permitted projection path.**
- `f0_validate.py` — Post-stage validators.
- Ingest: `validate_ingest_path()` → IngestPathInfo. `compute_output_root()` for gym-scoped output.
  Accepts both `{gym_id}/{cam_id}/{date}/{hour}/` and legacy `{cam_id}/{date}/{hour}/`.

## AprilTag: 36h11 (~587 IDs)
- Do NOT change family. Larger cells = better detection at gym distances.
- Scale beyond 587 via WiFi check-in disambiguation, not family migration.
- `tag_id` unique within `(tag_id + gym_id + active session)`.

## Session Aggregation
- Per-clip D0 banks combined with `{clip_id}:{tracklet_id}` namespacing.
- Frame indices offset by wall-clock time relative to session start.
- Identity hint frame_index IS offset (D3 tag ping binding requires same frame space as D1).
- D1→D4 run unchanged via SessionStageLayoutAdapter + SessionManifest.
