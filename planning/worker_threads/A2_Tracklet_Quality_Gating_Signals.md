## Addendum — 2026-01-14 (POC update: A2 deferred; C0 owns online tag scheduling; B deferred)

⚠️ **A2 is DEFERRED for the current POC.**

We are prioritizing an end-to-end A + C → D proof-of-concept:
- **Phase 1:** `multiplex_AC` (Stage A + Stage C online in a single decode pass)
- **Phase 2:** offline `D → E → X` (artifact-driven)

Online “when to attempt tag decode” scheduling now lives in **Stage C family** as **C0** (Tag Decode Scheduling & Cadence). Stage B (SAM refinement) is also **deferred** for the POC.

This document is retained as a future spec for offline/advanced quality analytics once the POC is complete.

## Addendum — 2026-01-08 (Post-D7 / Z3 Alignment)

### Role Clarification
Stage A2 does not detect or track. It annotates Stage A tracklets
with quality and gating signals used downstream.

Examples:
- occlusion likelihood
- mask instability
- excessive acceleration in mat-space
- off-mat persistence

### Multiplex Compatibility
A2 logic must be frame-local and multiplex-compatible.
It must not assume ownership of video IO.

### Downstream Contract
A2 outputs quality flags only.
No geometry, crops, or embeddings are produced here.
# A2 — Tracklet Quality & Gating Signals


---

## Addendum — Split A2 (Online) vs D0 (Offline)

This addendum updates A2 based on **A1R4 completion** and the now-locked **multiplex_ABC** “online” execution model.

### A2 Scope (Online, multiplex-safe)
A2 is **online-only**: it may use *past* context (short rolling windows) but must not use future frames.

A2’s (future) purpose is to compute **quality / interaction signals** and annotate tracklets/detections with lightweight health metrics.
For the current POC, online tag decode cadence is handled by **C0/C1** (Stage C family), and Stage B is deferred.

Concretely, A2 should focus on:
- **Entanglement / proximity scoring**: detect when two athletes are likely engaged (close distance in meters, overlapping boxes/masks, sustained contact over a rolling window).
- **Merge/split suspicion signals**: sudden mask area changes, aspect ratio spikes, mask topology anomalies, IoU instability, duplicate track IDs near-colliding.
- **Mask quality flags** (beyond A1’s per-frame gating): sustained low-quality segments, flicker frequency, “stringy” masks, perimeter/area heuristics.
- **(Future) Trigger policy outputs**: candidate windows for analysis/refinement. Out of scope for current POC.

### What moved OUT of A2 (Offline) → new worker D0
Any logic that uses **future context** or requires full-tracklet hindsight belongs in **D0 (offline cleanup & bank curation)**, e.g.:
- repairing location jumps using forward/backward smoothing
- removing tiny/junk tracklets with global thresholds
- curating crops/mask examples for ReID banks
- consolidating per-tracklet statistics with look-ahead

### Outputs / contracts (planning-level; keep additive)
A2 should prefer **additive, optional** artifacts so we don’t mutate A1 canonical outputs:
- `stage_A/quality_signals.jsonl` (or `stage_A/quality_signals.parquet`) keyed by `detection_id`/`tracklet_id` + `frame_index`
- a compact `stage_A/stage_B_triggers.jsonl` list for orchestration (deterministic ordering) *(future; Stage B deferred)*

If we decide to make these canonical later, we will bump F0 explicitly; until then, they are treated as stage-scoped artifacts validated by A2’s own tests.

### Acceptance criteria (A2)
- Works in multiplex mode (single-pass) with bounded memory (rolling windows)
- Deterministic: identical inputs/config produce identical triggers
- Includes audit counts: #flags, #triggers, top reason codes
- Provides at least one pytest smoke test (can run CPU-only)

### Relationship to Stage B
Stage B is **DEFERRED** for the current POC. This relationship is retained for future reactivation.
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
- `stage_A/audit.jsonl`

Stage B — Masks & Geometry:
- `stage_B/contact_points.parquet`
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

2) **Stage B — Masks + contact point + homography (offline refinement)**
   - **DEFERRED for POC.**
   - Tooling target (future): SAM/SAM2 refinement (or fallback masks) + OpenCV.
   - Output (future): refined masks + sparse overrides where needed.

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag detection applied inside **expanded bbox ROI** (mask may be used as a soft hint) + voting registry.
   - Output: tag observations (frame-level) + identity hints/constraints for Stage D.

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
- **Masks**: YOLO-seg online where possible; **SAM/SAM2 deferred for POC**
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


## Module-specific context (A2 — quality/gating signals)
Downstream stitching needs signals beyond (x,y). Define quality fields that help:
- ignore junk edges
- pick “best frames” for AprilTag scanning
- detect when SAM is worth running

### Must include
- Per-frame metrics: bbox area, aspect ratio change, velocity jump, detection conf
- Per-tracklet metrics: stability score, occlusion likelihood, “merged blob” flags
- Recommend thresholds but keep configurable


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

---

## Integration with Z3 Multiplexer Runner

If/when **Z3 (Single-pass Multiplexer Runner)** is implemented, this stage may run in a mode where orchestration provides a shared `FrameIterator` and calls per-frame processors. **Do not change F0 outputs** (Parquet schemas, mask storage, audits) to “bundle” B/C features into Stage A artifacts.

Stage A should remain responsible for:
- detector outputs (bboxes + scores + class)
- tracker association → tracklets

Anything like:
- mask refinement (Stage B),
- AprilTag observation/hints (Stage C),
- ReID embedding banks (D4),
should remain in their respective stages unless a manager-approved F0 schema bump is made.

Debug visualization outputs (if enabled) are **non-canonical** and must not become required artifacts for stage completion.

## Update after Z3 completion (2026-01-07)
Z3 introduced an **optional single-pass multiplex mode** (`multiplex_ABC`) that runs **Stages A→B→C within a shared frame loop** (video decoded once), while preserving:
- **F0 artifact contracts + paths** (each stage still writes its own canonical artifacts)
- **F1 stage contract** (`run(config, inputs) -> dict`) and skip/resume semantics
- **F2 config hashing + orchestration audit discipline**

### What this means for this worker
- Your stage code **must support both**:
  - **Multipass** execution (stage reads inputs from disk artifacts / manifest)
  - **Multiplex** execution (stage receives needed per-frame / per-clip state via the orchestration-provided state provider)
- Do **not** write debug videos from within stages. Dev visualization output is **owned by orchestration** and lives under:
  - `outputs/<clip_id>/_debug/` (non-canonical, dev-only)
- Prefer **pure functions / explicit state**:
  - per-frame update entrypoints should be deterministic and side-effect controlled
  - any expensive optional computations should be **gated** and recorded in stage audit (what ran, why)

### Interface expectations (keep flexible, but follow intent)
- If you introduce a per-frame API (recommended for A/B/C), keep it behind the stage module so orchestration can call it in multiplex mode.
- Ensure stage outputs can still be produced in multipass mode by reading upstream artifacts from disk (parity requirement until multipass is retired).
