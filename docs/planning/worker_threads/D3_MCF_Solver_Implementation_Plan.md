---
layout: page
---

# D3 — MCF Solver Implementation Plan


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
   - Tooling target: AprilTag detection in expanded-bbox ROIs + deterministic registry/voting.
   - Output:
     - `tag_observations.jsonl`
     - `identity_hints.jsonl` (must_link / cannot_link constraints for D)

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


## Module-specific context (D3 — solver wiring)
This worker implements the actual solver wiring for Stage D using:
- **Graph model** (nodes/edges) defined in **D1**
- **Costs + constraints** defined in **D2**
- **Birth/death + mat-zone gating** defined in **D5**
- Optional post-pass ILP refinement in **D6** (deferred unless explicitly enabled)

### POC execution model (locked)
- Stage D runs **offline** (artifact-driven) after multiplex_AC completes.
- Stage D must not depend on Stage B (deferred).

### Inputs (authoritative)
From Stage A:
- `stage_A/tracklet_frames.parquet` (frame-level geometry incl. `x_m,y_m,vx_m,vy_m,on_mat,contact_conf`)
- `stage_A/tracklet_summaries.parquet` (track spans; start/end frames)
- `stage_A/contact_points.parquet` (join-friendly baseline geometry; not required for solver wiring if frames has endpoints)

From Stage C:
- `stage_C/identity_hints.jsonl` (must_link + cannot_link constraints)

### Outputs (F0-locked)
Stage D must write:
- `stage_D/person_tracks.parquet`
- `stage_D/identity_assignments.jsonl`
- `stage_D/audit.jsonl`

### Borrowed from roll_it_back (solver pattern) — adaptation rules
We may reuse “MCF as primary stitcher” implementation ideas, but must adapt to roll_tracker contracts:
- roll_tracker’s unit of stitching is **tracklet_id** from Stage A.
- roll_tracker’s world coords are `x_m,y_m` (meters) and may be null for some frames.
- identity constraints are provided via C2 as **must_link(tag)** and **cannot_link(tracklet↔tracklet)**.
We must not carry forward any assumptions from roll_it_back about:
- different schema/IDs
- implicit video reads
- non-deterministic sampling

### Primary design goal for D3
Build a solver that is:
- deterministic,
- auditable,
- configurable,
- and can be “upgraded” (D6 ILP) without changing artifacts.

### Must include
- A concrete solver backend choice for POC, with a clean abstraction so we can swap later:
   - **POC recommendation:** OR-Tools min-cost flow if feasible with our constraint set; otherwise a deterministic two-phase approach:
      1) MCF for association within feasible edges (no identity constraints yet)
      2) apply must_link/cannot_link as hard merges/splits over the MCF result
   - If identity constraints cannot be represented directly in the chosen MCF formulation, this worker must:
      - document that limitation explicitly, and
      - implement an evidence-based workaround that remains deterministic and auditable.

- Exact mapping from D1 objects to solver inputs:
   - node indexing strategy (stable, deterministic)
   - edge indexing strategy (stable, deterministic)
   - cost vector construction using D2

- Exact handling of constraints from C2:
   - must_link groups (tracklets anchored to the same tag must map to same person_id)
   - cannot_link pairs (tracklets that must not share person_id)
   - constraint consistency checks (audit-only; never “fix” silently)

- Output emission rules:
   - deterministic person_id assignment (stable ordering)
   - traceability back to source tracklets (required for audit/debug)

### Invariants
- Deterministic results for same inputs + config (byte-identical outputs).
- Every person_track must map back to source tracklets (traceability required).
- Identity constraints are enforced as hard constraints in the final output:
   - if a constraint cannot be satisfied, Stage D must **fail fast** with a clear audit error (do not guess).
- Stage D reads upstream artifacts only via F0 contracts/manifest, never stage internals.

---

## D3 — Proposed implementation plan (POC)

### Step 1: Build “tracklet endpoint features” table (deterministic)
From `tracklet_frames.parquet`, compute per-tracklet:
- start_frame, end_frame
- start (x,y,vx,vy,on_mat,contact_conf)
- end (x,y,vx,vy,on_mat,contact_conf)

Rules:
- endpoint rows chosen deterministically:
   - start = first frame with non-null x/y (or first frame overall if missing policy is penalize)
   - end   = last  frame with non-null x/y
- if missing_geom_policy="disallow", exclude that tracklet from stitching graph and emit audit counts (do not silently include).

### Step 2: Candidate edge generation (delegated model from D1; filtered by D5)
Generate candidate links (A -> B) only when:
- B starts after A ends (dt >= 1 frame)
- dt <= dt_max_s (config)
- D5 gating allows the transition (mat zones / birth-death)

### Step 3: Edge cost computation (D2)
For each candidate edge:
- compute edge features (dist_m, dt_s, vjump, reliability flags)
- compute total edge cost + breakdown

Write audit quantiles for:
- dt_s, dist_m, vjump, total_cost

### Step 4: Solve association (MCF core)
POC target: produce a set of disjoint paths over tracklets.

Implementation must define:
- supply/demand conventions
- path cost = sum(edge costs) + birth/death costs (D5)
- disallow illegal edges by omitting them (not by huge cost, unless explicitly configured)

### Step 5: Apply identity constraints (C2) to paths
Regardless of whether the solver integrates constraints directly, the *final* result must satisfy:
- must_link groups: all tracklets anchored to the same tag share person_id
- cannot_link pairs: tracklets in the pair do not share person_id

POC-safe enforcement approach (deterministic, audit-first):
1) Build connected components of must_link (tag groups).
2) Validate that no cannot_link exists within a must_link component:
   - if violated, fail fast with audit event and list of offending tracklets/tags.
3) When assigning person_ids:
   - start from solved paths
   - merge paths that touch the same must_link component
   - if a merge would violate cannot_link, split deterministically and fail fast if unsplittable.

### Step 6: Emit artifacts
`identity_assignments.jsonl`:
- one record per `person_id` with:
   - list of source tracklet_ids
   - anchored_tag_id (if must_link exists, else null)
   - evidence summary (counts of must_links/cannot_links applied)

`person_tracks.parquet`:
- per-frame (or sampled) trajectory for each person_id
- must include traceability columns back to source tracklet_id and original frame_index.

### Step 7: Validation hooks (must be implementable)
Stage D validation must check:
- all referenced tracklet_ids exist in Stage A artifacts
- must_link/cannot_link constraints satisfied in final assignments
- person_tracks spans are consistent with source tracklet spans

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
1) A **proposed solver wiring plan** (bullets), including:
   - backend choice (OR-Tools vs NetworkX vs hybrid) with constraints implications
   - deterministic indexing / ordering strategy
   - constraint enforcement strategy (must_link/cannot_link) and fail-fast conditions
2) A list of **questions / assumptions** you need confirmed (only if blocking).
3) A draft **Interface Contract** for D3 code:
   - functions/classes for:
     - building endpoint summaries
     - generating candidate edges
     - running solve()
     - applying constraints
     - writing artifacts
   - acceptance tests (pytest) for must_link + cannot_link enforcement and determinism

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
