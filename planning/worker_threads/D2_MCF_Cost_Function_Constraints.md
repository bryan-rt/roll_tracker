# D2 — MCF Cost Function + Constraints

## PM Addendum (locked for POC)

- **D2 is solver-agnostic**: it produces per-edge scalar costs + term breakdowns and a normalized constraint specification.
- **D3 enforces feasibility** (MCF/ILP). D2 does **not** enforce constraints inside a solver.
- D2 must consume **canonical D1 graph artifacts**:
   - `stage_D/d1_graph_nodes.parquet`
   - `stage_D/d1_graph_edges.parquet`
- D2 must assign costs for **all D1 edge types** (`BIRTH`, `DEATH`, `MERGE`, `SPLIT`, `CONTINUE`).
- Continuation costs (only for edges with `dt_frames`) must reflect **temporal ordering + kinematic plausibility** (`v_req = dist/dt`), not distance-only gating.
- Implement **merge/split coherence**: edges that enter/exit GROUP segments along the labeled participants should be strongly preferred via bonuses/penalties.
- D0 CP3 kinematics/flags are **soft evidence only**; do not clamp/smooth geometry in D2.



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
This is the core of correctness for Stage D. D2 defines **exactly** how we turn artifacts into:
- numeric costs for candidate associations (graph edges), and
- a **solver-agnostic constraint specification** (must_link / cannot_link semantics + normalized sets).

**Important separation of concerns (locked):**
- **D2**: defines costs + constraint semantics/spec (no enforcement)
- **D3**: enforces feasibility + runs the chosen solver (MCF or ILP)

### Ground truth inputs available in roll_tracker (POC)
Authoritative, required for D2:
- Stage D (D1 canonical graph):
  - `stage_D/d1_graph_nodes.parquet` (typed node/segment table; endpoints + segment metadata)
  - `stage_D/d1_graph_edges.parquet` (typed edge table; dt_frames + edge type + merge/split metadata)
- Stage C:
  - `stage_C/identity_hints.jsonl` (must_link + cannot_link constraints; D2 consumes and normalizes into spec)

Optional (not required for POC correctness):
- `stage_D/tracklet_bank_frames.parquet` (D0 bank; CP3 kinematics/flags if needed for scoring)
- Stage A tables (`contact_points`, `tracklet_frames`) may be used for **feature enrichment only** if D1/D0 does not provide a field, but D2 must not re-derive graph structure/endpoints from Stage A.
- D4 embeddings (if later enabled) may contribute a soft appearance cost term.
- Stage B refined overrides are deferred and must be ignored by default.

### Join strategy (must be explicit and stable)
### Edge types (D1 canonical; D2 must price all)
D1 emits the following `edge_type` values in `d1_graph_edges.parquet`:
- `EdgeType.BIRTH`  (`SOURCE -> node`, capacity matches node capacity)
- `EdgeType.DEATH`  (`node -> SINK`, capacity matches node capacity)
- `EdgeType.MERGE`  (`SOLO(disappearing) -> GROUP(carrier)`, capacity = 1, `merge_end` populated)
- `EdgeType.SPLIT`  (`GROUP(carrier) -> SOLO(new)` or `GROUP -> GROUP`, capacity = 1, `split_start` populated)
- `EdgeType.CONTINUE` (temporal continuity between segments, capacity = 1, `dt_frames` populated)

D2 must compute costs using D1’s canonical tables as the primary join surface.

- Node features come from `d1_graph_nodes.parquet` (including endpoints and segment metadata).
- Edge features come from `d1_graph_edges.parquet` (including `dt_frames`, edge type, and merge/split participation fields).
- If enrichment is required, joins must be explicit and stable (e.g., keyed by `base_tracklet_id` + frame indices).

### Must include
#### A) Cost function components (POC baseline: geometry-first)
Costs are defined over D1 graph edges (`d1_graph_edges.parquet`). D1 defines the graph; D2 defines how each edge gets a cost.

**Borrowed from `roll_it_back` (downstream/costs.py) — adapt to roll_tracker:**
- A distance term normalized by feasible motion in time:
   - `dist_norm = ||p2 - p1|| / max(eps, v_max_mps * dt_s)`
- A velocity-change (“vjump”) term:
   - compare end velocity of tracklet A vs implied velocity needed to reach start of B
   - penalize large discontinuities
- A small “base_env” bias added to each candidate edge:
   - prevents pathological “free” links when all other terms are near-zero
   - makes the solver prefer “cleaner” explanations when combined with birth/death costs (D5)

**POC baseline equation (locked intent): kinematics-aware, not distance-only gating**


**Applicability rule (important):**
- This kinematic continuation equation applies **only** when `edge_type == CONTINUE` (or any edge with non-null `dt_frames`).
- For `MERGE/SPLIT/BIRTH/DEATH`, `dt_frames` is expected to be null and kinematic continuation is **not applicable**.

Recommended penalty shape:
- `hinge2(z) = max(0, z - 1)^2`

Total cost:
- `cost = base_env_cost`
- `+ w_time * log1p(dt_s)`
- `+ w_vreq * hinge2(v_req / v_ref_mps)`
- `+ w_geom_missing * I(geom_missing)`

Optional (POC safe, default ON):
- `+ w_flags * I(endpoint_speed_or_accel_flagged)` (soft evidence only)

Optional (POC safe, default OFF):
- accel term: `a_req` penalty
- ReID term: `w_reid * (1 - cosine_sim)`

**Adaptation notes for roll_tracker:**
- Prefer endpoints and segment metadata from `d1_graph_nodes.parquet`.
- Use D0 CP3 kinematics/flags (if available) as soft penalties or reliability scalars.
- If geometry is missing at either endpoint, follow `missing_geom_policy` (penalize or disallow) but do not invent geometry.

#### Merge/Split coherence (POC ON by default)
Because overlap/merge/split handling is the primary POC hypothesis, D2 must strongly prefer coherence with D1’s merge/split semantics:

- Entering a GROUP segment from a labeled merge participant should receive a **bonus** (negative cost):
  - example: `SOLO(disappearing_tracklet_id) -> GROUP(carrier)` during a merge span
- Exiting a GROUP segment to a labeled split participant should receive a **bonus**:
  - example: `GROUP(carrier) -> SOLO(new_tracklet_id)` at/after split start
- Unlabeled edges that touch GROUP segments should receive a **penalty** unless kinematics strongly supports them.


#### Edge-type priors (POC ON by default)
Because several D1 edge types do not carry `dt_frames`, D2 must include simple priors by `edge_type`
to prevent degenerate solutions and to encode a baseline preference structure:

- `BIRTH` and `DEATH`: costs come from **D5** policy/constants (or temporary stub values until D5 lands).
- `MERGE` and `SPLIT`: small costs near zero by default (structural events), optionally:
   - apply a mild penalty to discourage gratuitous MERGE/SPLIT churn if needed.
- `CONTINUE`: feature-based kinematic cost as defined above.

Notes:
- Avoid making MERGE/SPLIT “free” *and* BIRTH/DEATH “free” simultaneously; that can yield many equivalent optima.
- All priors must be emitted in audit term breakdowns for explainability.

**New ideas for roll_tracker (not in roll_it_back):**
- **Contact-quality weighting:** scale distance/vjump penalties by a reliability factor derived from `contact_conf` at both endpoints.
- **Mat-zone gating hooks:** integrate cleanly with D5 (birth/death + mat zones) by ensuring D2 exposes:
   - edge features and flags needed for gating (e.g., endpoints on_mat, dt, dist_m).

#### B) Optional soft terms (POC off by default)
- **Appearance/ReID** (D4): add `w_reid * (1 - cosine_sim)` only if embeddings are present.
- **Temporal gap prior:** mild penalty increasing with `dt_s` to discourage long jumps (unless constrained by must_link).

#### C) Constraint encoding (hard constraints)
We do **not** “paper over” identity constraints with big negative costs unless explicitly decided.

**D2 output is a normalized constraint spec; D3 enforces feasibility.**

Inputs from C2 (`stage_C/identity_hints.jsonl`):
- `must_link: tracklet_id -> tag:<tag_id>`
- `cannot_link: tracklet_id -> tracklet:<other_tracklet_id>` (symmetric)

**Must-link handling (hard anchor):**
- D2 defines the semantics and produces normalized groups for D3 to enforce:
   - all tracklets with the same `tag:<id>` must be assigned the same final `person_id` (identity anchor group)
   - (POC) treat as *hard*; do not allow violations with penalty

**Cannot-link handling (hard separation):**
- For each cannot_link pair `(tA, tB)` (collapse symmetry; treat as undirected unique pairs):
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

### D0 CP3 evidence integration (reminder)
- D0 CP3 provides dt-aware speed/accel estimates and implausibility flags.
- D2 may use them as soft penalties or reliability scalars.
- D2 must never clamp/smooth geometry.


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
    - the exact D1 edge types and which cost formula applies to each
    - the exact edge feature set needed (prefer D1/D0; Stage A only for enrichment)
    - the exact cost terms + normalization (including edge-type priors)
   - the exact constraint semantics for must_link / cannot_link
2) A list of **questions / assumptions** you need confirmed (only if truly blocking).
3) A draft **Interface Contract** for D2 outputs into D3:
    - function signatures (pure Python) for computing:
       - edge_features(edge_row, nodes_df, cfg) -> dict
       - edge_cost(edge_row, edge_features, cfg) -> float + term breakdown
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

---

# ✅ D2 Completion Report (POC)

## Status
Stage D2 — Cost Function + Constraint Layer — is **complete for POC scope** and ready to support solver implementation in Stage D3.

D2 now produces deterministic, solver-agnostic per-edge costs and normalized constraint sets while preserving strict adherence to F0 contracts.

## Key Outcomes

- All D1 edge types are priced.
- Continuation costs are kinematics-aware using a quadratic hinge.
- Merge/split coherence strongly rewards identity continuity through GROUP carriers.
- D0 kinematic flags are treated as soft evidence only.
- Disallowed edges are explicitly logged with canonical reasons.
- Outputs are deterministic and contract-validated.

## Artifact Readiness

D2 writes:

- `stage_D/d2_edge_costs.parquet`
- `stage_D/d2_constraints.json`

Both artifacts are manifest-registered and validated.

## Empirical Validation

Stress testing on an 8-athlete grappling scene confirmed:

- Stable SOLO/GROUP segmentation
- Repeated merge/split behavior correctly priced
- Minimal geometry disallows
- Healthy cost hierarchy:

```
CONTINUE << MERGE/SPLIT << BIRTH/DEATH
```

This creates a well-conditioned optimization problem for the solver.

## Determinism

D2 guarantees byte-stable outputs via:

- stable edge ordering  
- canonical JSON encoding  
- normalized constraint sets  

Repeated runs produce identical artifacts given identical inputs.

## Intentional Deferrals

- Velocity-change (“vjump”) term — deferred until post-solver tuning.
- Birth/death policy — owned by D5.

## Risk Assessment

Systemic risk is LOW.  
No cost collapse, physics violations, or merge incoherence observed.

## Manager Verdict

👉 **D2 is complete and solver-ready.**

No schema revisions required.  
No architectural changes required.

## Next Step

Proceed directly to:

# ➜ Stage D3 — Feasibility + Min-Cost Flow Solver

Key D3 responsibilities:

- enforce must-link / cannot-link constraints  
- remove disallowed edges  
- model GROUP capacity correctly  
- detect infeasible constraint sets  

Solver behavior should guide any future cost tuning.

---
