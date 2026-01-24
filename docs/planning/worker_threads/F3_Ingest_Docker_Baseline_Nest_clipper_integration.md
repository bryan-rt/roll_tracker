---
layout: page
---

# F3 — Ingest & Docker Baseline (Nest clipper integration)

## Update: F0 + F3 are complete (locked constraints)

### F3 (Stage 0 ingest) — locked input contract
- Ingest writes clips to: `data/raw/nest/<camera_id>/YYYY-MM-DD/HH/<camera_id>-YYYYMMDD-HHMMSS.mp4`
- Processing reads only from `data/raw/**`
- Outputs written only to `outputs/<clip_id>/...` (see F0 layout)
- Ingest service lives under `services/nest_recorder/` (Docker-based)

### F0 (contracts) — authoritative source of truth
All stage I/O **must** follow the contracts in:
- `src/bjj_pipeline/contracts/f0_models.py`
- `src/bjj_pipeline/contracts/f0_parquet.py`
- `src/bjj_pipeline/contracts/f0_paths.py`
- `src/bjj_pipeline/contracts/f0_manifest.py`
- `src/bjj_pipeline/contracts/f0_validate.py`

Do **not** invent schemas locally. If you need a change, propose it as “Manager Decision Needed” with a schema version bump.

### Locked artifact families (by stage)
Stage A — Detection & Tracklets (must write):
- `stage_A/detections.parquet`
- `stage_A/tracklet_frames.parquet`
- `stage_A/tracklet_summaries.parquet`
- `stage_A/contact_points.parquet` (baseline geometry; full coverage)
- `stage_A/audit.jsonl`

Stage B — Masks & Geometry (optional / deferred for current POC):
- `stage_B/contact_points_refined.parquet` (subset overrides only, when B runs)
- `stage_B/masks/*.npz` (canonical mask storage; referenced by relative path)
- `stage_B/audit.jsonl`

Stage C — Identity Anchoring (AprilTags):
- `stage_C/tag_observations.jsonl`
- `stage_C/identity_hints.jsonl` (must_link / cannot_link keyed to tracklet_id; anchor_key like `tag:<tag_id>`)
- `stage_C/audit.jsonl`

Stage D — Global Stitching (MCF):
- `stage_D/person_tracks.parquet`
- `stage_D/identity_assignments.jsonl` (final identities keyed to person_id)
- `stage_D/audit.jsonl`

Stage E — Match Sessions:
- `stage_E/match_sessions.jsonl`
- `stage_E/audit.jsonl`

Stage F — Export & Persistence:
- `stage_F/export_manifest.jsonl`
- exported `.mp4` clips
- `stage_F/audit.jsonl`

### Locked output layout + manifest backbone
Each run is anchored by:
- `outputs/<clip_id>/clip_manifest.json`

Stages must:
1) Read inputs from the manifest (or well-defined raw input path for Stage A)
2) Write artifacts to their stage folder
3) Register artifact paths in the manifest
4) Validate outputs via `f0_validate.py` before claiming completion

---

## Module-specific context (F3 — ingest + Docker baseline)

### What exists today
- A working ingest service under `services/nest_recorder/` that writes 2.5-min clips to `data/raw/nest/...` following the locked contract.
- OAuth refresh is currently failing (`invalid_grant`) but is isolated to credentials and **does not block downstream processing** using simulated clips.

### Clarification: ingest vs processing phases
- Ingest is strictly **Stage 0** and is **not part of multiplex execution**.
- Ingest outputs are immutable inputs to processing; no processing artifacts are written during ingest.
- Processing pipelines (multiplex_AC + offline D/E/X) assume ingest has already completed successfully.

### Your job
- Treat ingest as **Stage 0** and define the integration boundary into the processing pipeline:
  - What metadata must be produced (camera_id, timestamps, duration, fps if known)
  - Whether to produce a sidecar JSON next to each mp4 (recommended) and/or register in a manifest
- Define repo-wide Docker conventions based on what’s already in `services/nest_recorder/`:
  - naming, compose patterns, mounts, logging, secrets handling
  - how future `services/processor/` container should be shaped

### POC alignment notes
- Ingest does **not** need to know about:
  - multiplex vs multipass execution
  - Stage B enablement / deferral
  - downstream artifact schemas beyond clip identity
- Ingest must, however, preserve:
  - deterministic filenames
  - correct `camera_id` derivation
  - stable timestamps (used later for audits and exports)

### Deliverables required back to Manager
1) **Design Spec**: integration boundary and Docker baseline rules
2) **Interface Contract**: ingest outputs (clip path + sidecar) and invariants
3) **Copilot Prompt Pack**: minimal edits needed to align with F0 manifest backbone (if any)
4) **Acceptance Criteria**: prove ingest outputs are valid and deterministic

### Explicit non-responsibilities (anti-drift)
- Ingest must not:
  - write to `outputs/`
  - create or modify `clip_manifest.json`
  - perform any video decoding beyond basic validation

---

## Kickoff (what to do first in this worker thread)
Please begin by producing:
1) A **proposed plan** (bullets) for how ingest integrates with the F0 manifest + downstream stages.
2) A list of **questions / assumptions** you need confirmed.
3) A draft **Interface Contract** (ingest outputs).

End your first response by asking me to review/approve the plan before you go deeper.

Also include a bullet explicitly confirming alignment with the locked F0/F3 contracts (artifacts, paths, manifest).

## Checkpoint discipline

**All workers must keep the pipeline runnable end-to-end at every checkpoint.**  
This is a non-negotiable POC constraint to prevent drift and ensure incremental validation.

Minimum requirements for *any* stage/worker deliverable:

- Provide a `run()` implementation that is callable by orchestration: `def run(config: dict, inputs: dict) -> dict`.
- Ensure artifacts written match **F0** contracts and pass `roll-tracker validate`.
- Add at least one *realistic* smoke test (pytest) that:
  - runs the stage on a small fixture (or mocked inputs), and
  - asserts the expected artifacts exist and validate.
- Ensure the manager can run:
  - `roll-tracker run --clip <path> --camera <camera_id> --to-stage <your_stage>`
  - and receive deterministic outputs under `outputs/<clip_id>/...`.

If performance becomes an issue, prefer reducing resolution / selecting fewer candidates over skipping frames at POC time.
