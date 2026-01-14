# D0 — Offline Cleanup, Tracklet Repair, and Bank Curation (Pre-D)

**Owner:** D0 worker  
**Status:** READY (new)  
**Execution class:** OFFLINE (may use full past+future over a clip)  
**Where it runs:** after A/B/C artifacts exist; before D1 MCF graph building

## Purpose

D0 performs **offline, hindsight-allowed** cleanup and preparation over the *online* outputs from Stage A (and optionally Stage B refinements), to improve downstream global association (Stage D) and ReID stability **without changing any online stage behavior**.

This worker exists because A/B/C run in **multiplex_ABC** and must not use future frames. D0 is explicitly allowed to use full-clip context.

## Inputs (authoritative sources)

Primary:
- `stage_A/detections.parquet`
- `stage_A/tracklet_frames.parquet` (includes x/y/vx/vy/on_mat per v0.3.0)
- `stage_A/masks/*.npz` (canonical per-detection masks)
- `stage_A/tracklet_summaries.parquet`
- `stage_A/audit.jsonl`

Optional refinements:
- Stage B refined masks + sparse geometry overrides (if B ran)
- Stage C tag observations / identity hints (for anchor-aware cleanup, optional)

## Responsibilities

### 1) Tracklet repair (offline smoothing / deglitch)
- Detect and smooth **single-frame spikes** in (x,y) and velocities
- Fill short gaps if allowed (but do **not** invent long occlusion bridges; leave that to MCF)
- Produce “repaired” trajectories suitable for cost modeling (D2/D5)

### 2) Junk removal / pruning
- Identify micro-tracklets that are clearly spurious (very short, tiny area, off-mat, low confidence)
- Decide whether to:
  - mark them as “ignore” for downstream, or
  - keep them but with low-quality flags
- Must be deterministic and auditable.

### 3) ReID bank curation prep (inputs for D4)
- Curate a per-tracklet “best frames” set:
  - stable pose, non-entangled, good mask quality
  - diverse viewpoints over time
- Provide a deterministic, limited-size list of crop/mask references for embedding extraction.

### 4) Stage B trigger hindsight (optional)
- Using full-clip context, suggest additional segments that *should* have been refined by B (for later policy tuning), but do not require re-running B in this worker.

## Outputs (proposal; keep additive until explicitly made canonical)

D0 should **not mutate** Stage A artifacts. Prefer additive artifacts under `stage_D0/`:

- `stage_D0/tracklet_frames_clean.parquet`  
  Same key columns as `tracklet_frames.parquet`, plus:
  - `x_m_clean`, `y_m_clean`, `vx_mps_clean`, `vy_mps_clean`
  - `cleaning_reason_codes_json`
  - `is_pruned_candidate` (bool)

- `stage_D0/reid_bank_candidates.parquet`  
  Rows keyed by `tracklet_id` + `frame_index` + `detection_id`, with:
  - `rank`, `reason_code`, `mask_quality_score`, `crop_hint` (optional)

- `stage_D0/audit.jsonl`  
  Counts, thresholds, and summary stats.

> NOTE: If we decide these should be canonical, we will bump F0. Until then, these are D0-scoped artifacts with their own validation tests.

## Determinism & audit requirements

- All heuristics must be parameterized in config (F2) and recorded in audit.
- Output ordering must be stable.
- No randomness unless seeded and logged.

## Acceptance Criteria

- Runs end-to-end on a real clip after A1R4 outputs exist.
- Produces additive artifacts and an audit log.
- Demonstrates at least one repaired trajectory example (debug plot or stats).
- Includes a CPU-only pytest smoke test using small fixture data.

## Non-responsibilities

- Global identity stitching (Stage D1–D6)
- AprilTag decoding (Stage C)
- Homography creation (D7)
- Online per-frame detection/tracking (A1)

## Handoff Notes

Downstream (Stage D) should preferentially read cleaned fields when present:
- `x_m_clean/y_m_clean` if available, else fall back to Stage A `x_m/y_m`.

This keeps D0 strictly optional and non-breaking.
