## Addendum — 2026-01-14 (POC update: Stage B deferred; A owns baseline geometry; C online)

⚠️ **Stage B is DEFERRED for the current POC.**

We are prioritizing an end-to-end A + C → D proof-of-concept:
- **Phase 1:** `multiplex_AC` (Stage A + Stage C online in a single decode pass)
- **Phase 2:** offline `D → E → X` (artifact-driven)

Baseline contact point + homography projection + `on_mat` are already owned by **Stage A** (as implemented in A1R4 via `tracklet_frames.parquet`). This document is retained as a spec for a future Stage B reactivation where we add **sparse overrides** (refined contact points / refined masks) only when it yields measurable lift.

## Addendum — 2026-01-10 (A/B scope lock: baseline in A, refinement in B)

### Updated ownership model (Manager decision)
Baseline contact point + homography projection + on_mat is owned by Stage A.
Stage B no longer owns "compute x/y for every frame"; instead, Stage B performs *selective refinement*:
- only on low-quality YOLO masks and/or entanglement candidates
- uses SAM-derived masks where invoked
- overwrites/improves contact point estimation and/or mat-space projection on those rows

### F0 contract note (schema bump alignment)
After the F0 bump:
- Stage A canonical artifact: stage_A/contact_points.parquet (full coverage)
- Stage B refinement artifact: stage_B/contact_points_refined.parquet (subset / overrides)

Downstream consumers (D/E/F) should default to Stage A contact_points and apply Stage B refined overrides when present.

> **Repo reality note (Jan 2026):** This split is now implemented: Stage A writes `stage_A/contact_points.parquet` (baseline/full coverage) and Stage B writes `stage_B/contact_points_refined.parquet` (subset overrides when B runs).

### Multiplex implication (Z3)
Stage B should be multiplex-friendly:
- able to receive per-frame inputs/state (detections + optional YOLO masks)
- able to fire refinement conditionally and sparsely
Stage B must not require full-frame SAM on every frame.
## Addendum — 2026-01-08 (Post-D7 Alignment)

### Homography Guarantee
Stage B2 assumes homography.json exists and is valid.
No fallback or calibration logic is permitted here.

### Contact Point Semantics
Stage B2 may emit refined contact points derived from
SAM masks or improved geometry.

Refined points must be additive artifacts and never overwrite
Stage A contact points.
# B2 — Contact Point Extraction + Homography


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
- `stage_B/contact_points_refined.parquet`
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


## Module-specific context (B2 — contact point + homography)
We need a stable (u,v) contact point on the mat plane:
- preferred: average of bottom 5% of mask pixels (largest component)
- fallback: bottom-center of bbox if mask unreliable

### Must include
- Exact algorithm for contact point extraction
- Smoothing over time (optional) without lagging too much
- Homography projection: (u,v,1) -> H -> (x,y) normalized
- Units: ensure x,y in meters (camera calibration assumption stated)

### Acceptance tests
- Synthetic masks with noise; confirm stable contact points
- Known homography test vectors


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


## 🔒 Clarification (F1 + B3 Alignment)

This stage **must not assume homography creation**.

Assumptions now guaranteed by orchestration:
- A valid homography exists at:
  `configs/cameras/<camera_id>/homography.json`
- Homography validation happens **before this stage runs**

This worker is responsible only for:
- consuming H
- projecting contact points
- surfacing reprojection anomalies as diagnostics (not failures)

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

## Alignment with F1/F2 homography behavior

- **Canonical homography file path:** `configs/cameras/<camera_id>/homography.json` (camera-scoped).
- **Config integration:** if the file exists, it is auto-merged by F2 loader and reflected in `config_hash` + `config_sources`.
- **Enforcement:** orchestration preflight (F1) should fail-fast or launch the calibrator if homography is required for the run window and the file is missing.

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
