## Addendum — 2026-01-08 (Ownership Clarification)

### ReID Ownership
Stage D4 is the sole owner of ReID embedding computation.

Inputs:
- curated crop candidates from Stage A / B
- optional refined masks

Responsibilities:
- mask erosion
- quality filtering
- feature bank construction
- embedding distance computation

Stage A, B, and C must not compute embeddings.

### Feature Bank Policy
Embeddings are computed on-demand and stored in a bank
(best-K or clustered exemplars).
Rolling averages are explicitly discouraged.
# D4 — ReID Embeddings (masked crops) (optional)

> **POC posture:** D4 is **OPTIONAL** and must not block Stage D.  
> If disabled, Stage D must still run end-to-end using geometry + MCF costs + C2 constraints only.


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
- `stage_A/contact_points.parquet` *(canonical baseline geometry; join-friendly)*
- `stage_A/audit.jsonl`

Stage B — Masks & Geometry:
- **DEFERRED for POC.**
- (Future, optional) `stage_B/contact_points_refined.parquet` *(sparse overrides only)*
- (Future, optional) `stage_B/masks/*.npz` *(refined mask hints; not required)*
- (Future) `stage_B/audit.jsonl`

Stage C — Identity Anchoring (AprilTags):
- `stage_C/tag_observations.jsonl`
- `stage_C/identity_hints.jsonl`
   - must_link: `tracklet_id -> anchor_key="tag:<tag_id>"`
   - cannot_link: `tracklet_id -> anchor_key="tracklet:<other_tracklet_id>"` *(symmetric pairs emitted)*
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

2) **Stage B — Masks + contact point + homography (offline refinement)**
   - Tooling target: SAM/SAM2 offline refinement (or fallback masks) + OpenCV.
   - Output: mask references + stable “ground contact point” per frame + projected ground-plane coordinates.

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag detection applied inside mask ROI + voting registry.
   - Output: tag observations (frame-level) + stable identity assignments + conflicts.

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
- **AprilTags**: Python apriltag detector (library choice can be decided in C1)
- **ReID (optional early, likely later)**: OSNet / torchreid or FastReID; ideally on masked crops
- **Stitching**: **Min-Cost Flow** (OR-Tools min-cost flow or NetworkX as baseline)
- **Video I/O**: OpenCV for reading frames when needed; ffmpeg for export
- **Data**: JSONL for event logs; Parquet for high-volume tables (decide in F0/F2)

### Contracts & artifacts (must be defined centrally in F0)
Workers should assume the following artifact families will exist, with exact schemas defined in F0:
- `detections` (per-frame detections)
- `tracklets` (tracklet spans + summaries)
- `masks` (mask references, RLE/paths) and `contact_points` (u,v and x,y)
- `tag_observations` (frame-level tag detections) and `identity_assignments`
- `person_tracks` (stitched per-person timeline)
- `match_sessions`
- `export_manifest` / DB rows / audit logs

### Definition of “done” for a worker thread
A worker thread is “done” only when it returns to the Manager:
- **Design Spec** (assumptions, algorithm, edge cases, failure modes)
- **Interface Contract** (inputs/outputs + invariants, including artifact schema deltas if any)
- **Copilot Prompt Pack** (file-by-file prompts) and/or **Acceptance Criteria** (tests + checks)


---


## Module-specific context (D4 — ReID embeddings)

This worker explores adding **optional** appearance embeddings to help stitch tracklets that are ambiguous under geometry alone.

### Hard constraints (do not violate)
- **No new canonical artifacts** without an explicit F0 bump.
- Must be fully deterministic (same inputs + config → identical embeddings and similarity scores).
- Must not require Stage B (deferred).
- Must not read video in a way that breaks the offline artifact model (D runs offline, but can open raw video frames if needed *only* for the specific crops it chooses; avoid full decode).

### POC recommendation
- Treat D4 as **dev-only** unless/until it demonstrates measurable benefit on real clips.
- Default: `enabled: false`.

### Inputs
- Authoritative:
   - `stage_A/detections.parquet` (bbox + mask_ref if present)
   - `stage_A/tracklet_frames.parquet` (tracklet_id keyed timeline)
   - `stage_A/tracklet_summaries.parquet` (spans)
- Optional:
   - Stage A masks referenced by `mask_ref` (soft guidance for crop masking)
   - Raw video frames **only for selected frames** (no full-pass decode)

### Outputs
- **No new F0 artifacts by default.**
- If you produce anything, it must be **dev-only** under:
   - `outputs/<clip_id>/_debug/` or `outputs/<clip_id>/_cache/`
- D4 may optionally return an **in-memory** function that D2/D3 can call to obtain a similarity score for a candidate edge:
   - `sim(tracklet_a, tracklet_b) -> float`

### Determinism requirements
- Fixed model weights and preprocessing (explicit paths / versions).
- Stable sampling strategy:
   - deterministic selection of frames within each tracklet (e.g., first/middle/last valid bbox frames)
   - stable crop extraction rules (bbox pad fraction, clamp to bounds)
- Explicit random seeds set (if model uses any nondeterminism).

---

## Borrow from roll_it_back (appearance ideas) — adaptation notes
If roll_it_back contains appearance/crop logic, we may reuse concepts, but must adapt:
- roll_tracker IDs: `tracklet_id`, `detection_id`, `frame_index` (no legacy IDs)
- crop sources: bbox-expanded ROI; masks are hints only (do not hard-clip unless explicitly configured)
- outputs must remain dev-only unless F0 is bumped

---

## Acceptance criteria (D4)
1) **Optional & non-blocking**
    - With D4 disabled, Stage D produces identical outputs as before.
2) **Deterministic**
    - Same clip + config produces byte-identical cached embeddings/scores.
3) **Useful interface**
    - A single function or table can be consumed by D2 cost function as an optional term:
       - `cost += reid_weight * (1 - similarity)`
4) **Pytest**
    - Synthetic crops → stable embeddings (hash) and stable similarity results.


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
