---
layout: page
---

# D2 — MCF Cost Function + Constraints


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
- `stage_A/contact_points.parquet`  *(canonical baseline geometry; join-friendly)*
- `stage_A/audit.jsonl`

Stage B — Masks & Geometry:
- **DEFERRED for POC.**
- (Future, optional) `stage_B/contact_points_refined.parquet`  *(sparse overrides only)*
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
   - **Canonical baseline geometry** now lives in:
     - `stage_A/contact_points.parquet` (join-ready)

2) **Stage B — Masks + contact point + homography (offline refinement)**
   - **DEFERRED for POC.**
   - (Future) Tooling target: SAM/SAM2 offline refinement + OpenCV.
   - (Future) Output: refined masks + sparse geometry overrides (`contact_points_refined.parquet`).

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag detection in expanded-bbox ROIs + registry/voting.
   - Output:
     - `tag_observations.jsonl` (frame-level tag detections)
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


## Module-specific context (D2 — costs + constraints)
This is the core of correctness for Stage D. D2 defines **exactly** how we turn our artifacts into:
- numeric costs for candidate associations (MCF edges), and
- hard constraints (AprilTag anchors + cannot-links) that the solver must respect.

This worker produces a spec + implementation plan that D3 (solver wiring) can consume without ambiguity.

### Ground truth inputs available in roll_tracker (POC)
Authoritative, required for D:
- Stage A:
   - `stage_A/tracklet_frames.parquet` (per-frame tracklet geometry incl. `x_m,y_m,vx_m,vy_m,on_mat,contact_conf`)
   - `stage_A/tracklet_summaries.parquet` (start/end frames; span features)
   - `stage_A/contact_points.parquet` (join-friendly per-detection geometry; canonical baseline)
- Stage C:
   - `stage_C/identity_hints.jsonl` (must_link + cannot_link constraints)

Optional (not required for POC correctness):
- D4 embeddings (if later enabled) may contribute a soft appearance cost term.
- Stage B refined overrides are deferred and must be ignored by default.

### Join strategy (must be explicit and stable)
We must be able to compute costs without “mystery joins”.

- For *edge features* between tracklets, compute endpoints using:
   - tracklet “tail” summary from `tracklet_frames` (last valid frame with coords)
   - tracklet “head” summary from `tracklet_frames` (first valid frame with coords)
- When we need per-detection geometry, join via:
   - `(clip_id, camera_id, frame_index, detection_id)` into `stage_A/contact_points.parquet`
- For constraints:
   - must_link uses `tracklet_id` directly
   - cannot_link uses `(tracklet_id, anchor_key="tracklet:<other>")` and is symmetric

### Must include
#### A) Cost function components (POC baseline: geometry-first)
Costs are defined over candidate “links” between tracklets (tail -> head). D1 defines the graph; D2 defines how each edge gets a cost.

**Borrowed from `roll_it_back` (downstream/costs.py) — adapt to roll_tracker:**
- A distance term normalized by feasible motion in time:
   - `dist_norm = ||p2 - p1|| / max(eps, v_max_mps * dt_s)`
- A velocity-change (“vjump”) term:
   - compare end velocity of tracklet A vs implied velocity needed to reach start of B
   - penalize large discontinuities
- A small “base_env” bias added to each candidate edge:
   - prevents pathological “free” links when all other terms are near-zero
   - makes the solver prefer “cleaner” explanations when combined with birth/death costs (D5)

**Adaptation notes for roll_tracker:**
- Use `x_m,y_m,vx_m,vy_m` from `tracklet_frames.parquet`.
- If `x_m/y_m` are null for an endpoint, mark the edge as “geometry_missing”:
   - either disallow (strict mode) or allow with a large fixed penalty (POC-safe default should be explicit in config).
- Use `contact_conf` and `on_mat` to scale reliability:
   - low `contact_conf` should *increase* uncertainty; do not over-trust footpoint geometry.

**New ideas for roll_tracker (not in roll_it_back):**
- **Contact-quality weighting:** scale distance/vjump penalties by a reliability factor derived from `contact_conf` at both endpoints.
- **Mat-zone gating hooks:** integrate cleanly with D5 (birth/death + mat zones) by ensuring D2 exposes:
   - edge features and flags needed for gating (e.g., endpoints on_mat, dt, dist_m).

#### B) Optional soft terms (POC off by default)
- **Appearance/ReID** (D4): add `w_reid * (1 - cosine_sim)` only if embeddings are present.
- **Temporal gap prior:** mild penalty increasing with `dt_s` to discourage long jumps (unless constrained by must_link).

#### C) Constraint encoding (hard constraints)
We do **not** “paper over” identity constraints with big negative costs unless explicitly decided. Constraints must be representable as solver restrictions.

Inputs from C2 (`stage_C/identity_hints.jsonl`):
- `must_link: tracklet_id -> tag:<tag_id>`
- `cannot_link: tracklet_id -> tracklet:<other_tracklet_id>` (symmetric)

**Must-link handling (hard anchor):**
- D2 defines how D3 must translate must_link constraints into solver restrictions:
   - all tracklets with the same `tag:<id>` must be assigned the same final `person_id` (identity anchor group)
   - (POC) treat as *hard*; do not allow violations with penalty

**Cannot-link handling (hard separation):**
- For each cannot_link pair `(tA, tB)`:
   - forbid solutions where both tracklets map to the same `person_id`
   - (POC) treat as hard; do not allow violations with penalty

#### D) “needs_split” (explicitly not canonical in POC)
If any prior docs mention `needs_split`, treat it as **non-canonical** for now.
In POC, the only identity constraints are those emitted by C2.

### Acceptance tests
1) **Cost sanity / monotonicity**
   - increasing spatial gap increases edge cost (holding dt constant)
   - increasing dt increases allowed travel; `dist_norm` decreases accordingly
2) **Must-link dominance**
   - two tracklets with the same `tag:<id>` are forced into the same identity group even if geometry is weak
3) **Cannot-link correctness**
   - a cannot_link pair is never assigned the same identity even if geometry cost is minimal
4) **Determinism**
   - cost computation uses stable endpoint selection + stable sorting so repeated runs are byte-identical

### Config surface (F2-compatible; names are suggestions)
Recommend placing under `config["stage_D"]["d2_costs"]`:
- `enabled: bool` (default true)
- `v_max_mps: float` *(roll_it_back analog: max_speed)*
- `w_dist: float`
- `w_vjump: float`
- `base_env_cost: float`
- `dt_max_s: float` *(disallow edges beyond this gap unless must_link requires it)*
- `missing_geom_policy: "disallow" | "penalize"`
- `missing_geom_cost: float` *(if penalize)*
- `contact_conf_floor: float` *(below this, treat geometry as unreliable)*
- `use_reid: bool` *(default false; D4-owned)*
- `w_reid: float` *(only if use_reid)*

All resolved values must be recorded in `stage_D/audit.jsonl` as a `d2_config_resolved` event.

### Required audit outputs (D2-owned; written by Stage D)
Define the minimal audit signals D3 must emit on D2’s behalf:
- counts:
  - n_tracklets, n_candidate_edges, n_edges_disallowed_missing_geom, n_edges_disallowed_dt
- summary stats:
  - dist_m quantiles, dt_s quantiles, cost quantiles (for each term and total)
- constraint stats:
  - n_must_link_groups, n_must_link_tracklets
  - n_cannot_link_pairs (unique, after symmetry collapse)


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
1) A **proposed cost+constraint plan** (bullets), including:
   - the exact edge feature set needed from Stage A tables
   - the exact cost terms + normalization
   - the exact constraint semantics for must_link / cannot_link
2) A list of **questions / assumptions** you need confirmed (only if truly blocking).
3) A draft **Interface Contract** for D2 outputs into D3:
   - function signatures (pure Python) for computing:
     - edge_features(tracklet_id_a, tracklet_id_b) -> dict
     - edge_cost(edge_features, cfg) -> float + term breakdown
     - constraint sets from `identity_hints.jsonl`

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
