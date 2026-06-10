---
paths:
  - "src/bjj_pipeline/**"
---

# CV Pipeline Stages

## Phase 1 — Online (parallel, per-clip via multiplex_AC)
- **Stage A** `detect_track`: `bjj-detect-all-cameras-v2.pt` detection-only model (CP23b,
  domain-tuned yolo26n, 1352 frames 3 cameras) + BoT-SORT on raw frames. Detection-only:
  no pose head, `require_keypoints: false`, `conf: 0.45`. CoreML `.mlpackage` is the
  active inference path (`prefer_coreml: true`). Keypoints sidecar writes NaN columns
  when model has no pose head — this is expected behavior.
  Projects contact points to world coordinates via `project_to_world()` (undistortion
  applied internally when K+dist present). Optionally loads `calibration_correction.json`
  (CP18 affine, superseded by CP19). CP20 additions: keypoints extraction (17 COCO
  keypoints when pose model used), isolation gate (per-detection is_isolated flag,
  `require_keypoints` config controls whether H4 torso keypoint check is applied — set
  false for detect-only models), HSV color histogram extraction (torso-crop with
  center-bbox fallback, falls back to center-bbox when no keypoints available).
  **V-channel (CP-HSV-V):** histogram is H+S+V (18×8×6=864-dim, `histogram.py`
  channels [0,1,2]). V added 2026-06-09; prior was H+S only (144-dim).
  `bhattacharyya_distance` compares flat (shape-invariant). `HIST_V_BINS=6`.
  Outputs: detections, tracklet_frames, tracklet_summaries, contact_points (all .parquet),
  keypoints.parquet, color_histograms.parquet, tracklet_histogram_summaries.parquet,
  audit.jsonl.
- **Stage B** `masks`: SAM — deferred for POC. Falls back to YOLO bbox.
- **Stage C** `tags`: AprilTag identity. C0 scheduling/cadence, C1 ROI scan, C2 voting.
  Outputs: tag_observations.jsonl, identity_hints.jsonl.

## Phase 2 — Offline (sequential, never parallelize)
- **Stage D** `stitch`: ILP stitching via OR-Tools. D0 bank tables → D0.5 tracklet split
  → D1 graph → D2 constraints → D3 ILP solve (d3_ilp2 MCF solver exclusively, d3_ilp
  kept for comparison only, shared helpers in d3_common) → D4 person_tracks.
  - **D0.5** `d05_split` (CP-SPLIT-1): Post-D0 tracklet splitter. Tiered swap boundary
    detection (speed cap, kinematic spike+isolation, histogram Bhattacharyya) with min-dwell
    filter. Modifies `tracklet_bank_frames.parquet` and `tracklet_bank_summaries.parquet`
    in-place. Writes `d05_split_audit.jsonl`. Does NOT modify `stage_A/detections.parquet`.
    Config: `stage_D.d05_split` (Optional[dict], defaults enabled). Runs inside D1+ guard
    in `run.py`. D4 join safety: bank_frames↔detections join on (clip_id, camera_id,
    frame_index, detection_id), not tracklet_id. Split IDs use `{tid}_s{N}` suffix.
    **NET-NEGATIVE (CP-GT2ACTUALS-4+5):** D0.5 is net-negative on ALL cameras.
    vid2 (authoritative, 99.4% classified): 35 correct / 317 false (net -282).
    Tier 3 owns 79% of damage (-222 of -282). CP-GT2ACTUALS-6 signal analysis:
    NO per-frame signal separates false from correct splits (HSV Bhattacharyya
    0.035 vs 0.040 — indistinguishable). False splits are 82% isolated (color
    available but not discriminative). Disabling Tier 3 removes 241 false splits
    at cost of 19 correct (5.4%). **Interim recommendation: disable Tier 3.**
    FP7oJQ/PPDmUg thin-classification (coverage artifact, 5.8%/33.3%).
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
