## Addendum — 2026-01-10 (Stage A baseline geometry + Stage B refinement)

### Contract update (requires schema_version bump)
Manager decision: baseline contact point + homography projection + on_mat lives in Stage A.

New / updated artifacts:

Stage A — Detection & Tracklets (+ Geometry)
- detections.parquet
- tracklet_frames.parquet
- tracklet_summaries.parquet
- contact_points.parquet   <-- NEW (canonical baseline geometry; full coverage)
- audit.jsonl

Stage B — Masks & Geometry Refinement (optional)
- contact_points_refined.parquet  <-- NEW (subset overrides when SAM refinement is run)
- stage_B/masks/*.npz (canonical mask storage; paths only)
- audit.jsonl

### Downstream rule
Downstream stages should:
1) load Stage A contact_points as baseline
2) apply Stage B refined overrides when present
This preserves the ability to skip Stage B while keeping geometry available.
# F0 — Core Contracts & Artifact Schemas


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
- `stage_B/contact_points_refined.parquet` (subset overrides; only when B runs)
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

## Standard context: full pipeline (POC, offline, BJJ practice)

### What we are building
An **offline** (batch) video processing pipeline for BJJ practice footage. Input is a saved ~2.5 minute clip. Output is:
- **Stitched per-athlete trajectories** (stable person identities across time)
- **“Who vs who” match sessions** (start/end via hysteresis on ground-plane distance)
- **Exported match clips** (cropped, optionally privacy-redacted)
- **Database rows + audit logs** (opt-in compliance and debugging)

### Non‑negotiables (manager-locked constraints)
- **Min-Cost Flow (MCF) stitching is mandatory** (no Hungarian tracklet linking as the final association method).
- **Offline-first design**: we may use an “online tracker” (BoT-SORT) only as a *tracklet generator*.
- **AprilTags are hard identity anchors**: must-link / cannot-link constraints must be enforced in stitching.
- **Homography is used to compute ground-plane (x,y) in meters** from an on-mat contact point.
- **Modularity + contract-first**: every stage reads/writes versioned artifacts defined in `F0`.
- **Deterministic + debuggable**: every stage must produce debug/audit artifacts explaining key decisions.

### Pipeline stages (high level)
1) **Stage A — Detect + Tracklets (local association)**
   - Tooling target: detector (YOLO or similar) + tracker (BoT-SORT via BoxMOT).
   - Output: frame-level detections + short, high-precision **tracklets** (intentionally allowed to break).

2) **Stage B — Mask refinement + geometry overrides (optional / deferred for current POC)**
   - Tooling target: SAM/SAM2 selective refinement + OpenCV.
   - Output: refined masks and **subset** contact point overrides (not full coverage).

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag decode in expanded bbox ROI (mask is a soft hint; Stage B optional) + deterministic registry.
   - Output: tag observations (frame-level) + identity constraints/hints (must_link / cannot_link).

4) **Stage D — Global stitching (Min-Cost Flow)**
   - Tooling target: MCF solver (start with OR-Tools or NetworkX; optimize later).
   - Inputs: tracklets + (x,y) + (optional) ReID similarity + AprilTag constraints.
   - Output: stitched person tracks across entire clip.

5) **Stage E — Match session detection (hysteresis)**
   - Tooling target: deterministic state machine on pairwise distances.
   - Output: match sessions (who vs who, start/end, confidence, evidence).

6) **Stage F — Export + privacy + database**
   - Tooling target: ffmpeg (clip/crop), optional mask-based redaction, SQLite/Postgres persistence.
   - Output: mp4 clips + metadata rows + audit trails.

### Canonical tool choices (POC defaults)
These are defaults; workers may propose alternatives but must align with constraints.
- **Tracking**: BoxMOT **BoT-SORT** (as tracklet generator)
- **Masks**: YOLO-seg online where possible; **SAM/SAM2 offline** where higher fidelity needed
- **AprilTags**: OpenCV ArUco (`cv2.aruco`) decoder (current implementation)
- **ReID (optional early, likely later)**: OSNet / torchreid or FastReID; ideally on masked crops
- **Stitching**: **Min-Cost Flow** (OR-Tools min-cost flow or NetworkX as baseline)
- **Video I/O**: OpenCV for reading frames when needed; ffmpeg for export
- **Data**: JSONL for event logs; Parquet for high-volume tables (decide in F0/F2)

### Contracts & artifacts (must be defined centrally in F0)
Workers should assume the following artifact families will exist, with exact schemas defined in F0:
- `detections` (per-frame detections)
- `tracklets` (tracklet spans + summaries)
- `masks` (mask references, RLE/paths) and `contact_points` (u,v and x,y)
- `tag_observations` (frame-level tag detections) and `identity_hints` (must_link / cannot_link constraints)
- `person_tracks` (stitched per-person timeline)
- `match_sessions`
- `export_manifest` / DB rows / audit logs

### Definition of “done” for a worker thread
A worker thread is “done” only when it returns to the Manager:
- **Design Spec** (assumptions, algorithm, edge cases, failure modes)
- **Interface Contract** (inputs/outputs + invariants, including artifact schema deltas if any)
- **Copilot Prompt Pack** (file-by-file prompts) and/or **Acceptance Criteria** (tests + checks)


---


## Module-specific context (F0 — contracts)
You are defining the **canonical schemas** that every other stage must follow. This is the most important anti-drift lever.

### Must include
- Artifact schema versioning strategy (e.g., `schema_version: "0.1"` at top-level)
- Pydantic (or dataclasses) models for:
  - Detection, TrackletFrame, TrackletSummary, MaskRef, ContactPoint, TagObservation, IdentityAssignment,
    PersonTrackPoint, MatchSession, ExportManifest, AuditEvent
- Serialization rules:
  - JSONL: one object per line for events/observations/sessions
  - Parquet: columnar tables for detections/person_tracks/contact_points
- Stable identifiers:
  - `clip_id`, `camera_id`, `frame_index`, `timestamp_ms`
  - `tracklet_id` (Stage A), `person_id` (post-stitch), `tag_id` (AprilTag), `match_id`
- Minimal required fields vs optional fields (use `Optional[...]` explicitly)

### Invariants you must enforce
- `frame_index` monotonic within a clip
- Tracklet frames reference valid detections
- Person tracks must map back to source tracklets (traceability)
- Tag observations must include confidence + ROI method used

### Downstream override rule (Manager-locked)
Downstream consumers (D/E/X) must:
1) load `stage_A/contact_points.parquet` as the baseline
2) if present, apply overrides from `stage_B/contact_points_refined.parquet` by `(frame_index, detection_id)` (or an equivalent stable key)

This preserves the ability to skip Stage B while keeping geometry available and deterministic.

### What you should propose
- Where we store artifact files (e.g., `outputs/<clip_id>/<stage>/...`)
- Naming conventions and a “manifest” file linking all outputs


---


## Worker responsibilities in this thread

Produce deliverables that are directly usable by the Manager and by GitHub Copilot. Keep scope strictly within this module.

### Required outputs back to Manager
- **Design Spec**: approach, assumptions, edge cases, failure modes, performance notes (POC-level)
- **Interface Contract**: exact input/output artifacts and invariants (propose schema changes to F0 explicitly)
- **Copilot Prompt Pack**: prompts per file (or per function), with acceptance tests
- **Acceptance Criteria**: checklist + unit tests (and synthetic fixtures where possible)

### Guardrails (anti-drift)
- Do not invent new stages.
- Do not change global constraints (MCF mandatory, AprilTags as anchors, offline-first).
- If you need a cross-module change, propose it explicitly as a “Manager Decision Needed”.


---


## Kickoff (what to do first in this worker thread)
Please begin by producing:
1) A **proposed plan** for this module (bullets are fine), including key decisions and tradeoffs.
2) A list of **questions / assumptions** you need confirmed.
3) A draft **Interface Contract** for this module (even if some fields are TBD pending F0).

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
