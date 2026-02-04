# D3 — Solver Wiring + Feasibility Enforcement (MCF/ILP)


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

### Critical update (post-D1 CP2 + D2 completion)
- D1 graph artifacts are now **canonical and F0-locked**:
   - `stage_D/d1_graph_nodes.parquet`
   - `stage_D/d1_graph_edges.parquet`
- D2 produces **canonical solver inputs** (D3 must consume; must not recompute):
   - `stage_D/d2_edge_costs.parquet`
   - `stage_D/d2_constraints.json`

### POC execution model (locked)
- Stage D runs **offline** (artifact-driven) after multiplex_AC completes.
- Stage D must not depend on Stage B (deferred).

### Inputs (authoritative, POC)
Stage D3 must read upstream inputs **via manifest** and treat the following as the solver’s authoritative surfaces:

From Stage D (D1 canonical graph):
- `stage_D/d1_graph_nodes.parquet` (typed nodes/segments incl. SOURCE/SINK, SOLO/GROUP, capacities, segment metadata)
- `stage_D/d1_graph_edges.parquet` (typed edges incl. `edge_type`, capacities, and `dt_frames` for CONTINUE only)

From Stage D (D2 canonical costs + normalized constraints):
- `stage_D/d2_edge_costs.parquet` (per-edge scalar cost + term breakdown + allowed/disallowed + reason codes)
- `stage_D/d2_constraints.json` (normalized `must_link_groups` + `cannot_link_pairs`, deterministic ordering)

From Stage D (optional enrichment only; not required for graph/cost definition):
- `stage_D/tracklet_bank_frames.parquet` (D0 bank; traceability/join helpers if needed for person_tracks emission)

### Outputs (F0-locked)
Stage D must write:
- `stage_D/person_tracks.parquet`
- `stage_D/identity_assignments.jsonl`
- `stage_D/audit.jsonl`

### Borrowed from roll_it_back (solver pattern) — adaptation rules
We may reuse “MCF as primary stitcher” implementation ideas, but must adapt to roll_tracker contracts:
- roll_tracker’s unit of stitching (for the solver) is **D1 node_id / segment nodes** (SOLO/GROUP), not raw Stage-A tracklets.
- roll_tracker’s world coords are `x_m,y_m` (meters) and may be null for some segments/endpoints; missing-geom policy is handled upstream in D2 and represented in `d2_edge_costs.parquet` as allow/disallow and/or penalties.
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
- A concrete solver backend choice for POC, with a clean abstraction so we can swap later.
  - **POC baseline checkpoints may start with MCF-like flow without hard identity constraints** (to validate wiring).
  - **However, the POC definition requires hard must_link/cannot_link enforcement in the final output.**
  - Therefore D3 must deliver one of:
    1) **Single-stage ILP** (recommended for POC correctness): encode flow + hard constraints directly.
    2) **Two-stage approach** (allowed only if it is exact and auditable): a first solve to propose structure, followed by a second exact feasibility solve that enforces must_link/cannot_link (e.g., ILP repair/re-solve). If the second stage cannot satisfy constraints, fail-fast (no heuristic “fixes”).

- **Non-negotiable:** Do not “enforce” hard constraints by huge negative costs. Constraints must be enforced as feasibility restrictions, or we fail-fast if infeasible.

- Exact mapping from D1 objects to solver inputs:
   - node indexing strategy (stable, deterministic)
   - edge indexing strategy (stable, deterministic)
   - cost vector construction using D2 (join by `edge_id`; assert 1:1 coverage)

- Exact handling of constraints from D2 normalized spec:
   - must_link groups (entities anchored to the same tag must map to the same final identity)
   - cannot_link pairs (forbid co-assignment to the same identity)
   - unsatisfiable detection + diagnostics (fail-fast; never “fix” silently)

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

## D3 — Proposed implementation plan (POC, checkpointed)

### Step 0 (D3-POC-1): “Graph compiles” (no solving)
Load canonical solver inputs and validate joins:
- Read `d1_graph_nodes.parquet` and `d1_graph_edges.parquet`
- Read `d2_edge_costs.parquet` and assert **1:1** join on `edge_id`
- Read `d2_constraints.json` and validate canonical ordering/dedup invariants

Prune edges deterministically:
- Drop edges marked disallowed by D2 (do not reinterpret)
- Emit audit counts:
   - counts by `edge_type`
   - counts by disallow reason (if present)
   - n_nodes (by node_type), n_edges_kept

**Exit condition:** audit-only run succeeds deterministically; no solver yet.

### Step 1 (D3-POC-2): “Solver runs (no hard constraints yet)”
Goal: verify optimization wiring + capacity handling (especially GROUP nodes).
- Use D1 nodes/edges + D2 costs after pruning disallowed edges
- Solve a baseline min-cost flow style objective **without** hard must_link/cannot_link (temporarily)
- Emit:
   - `stage_D/audit.jsonl` with solver objective + flow stats
   - placeholder but deterministic `identity_assignments.jsonl` and `person_tracks.parquet` (even if constraints not yet applied)

**Exit condition:** deterministic solve produces stable outputs; GROUP capacity behavior is validated (2 units can traverse GROUP spans).

### Step 2 (D3-POC-3): Integrate birth/death policy (D5-min)
Goal: integrate birth/death constants without blocking the solver wiring.
- If D5 is not implemented yet, use the D2 placeholder priors and record this explicitly in audit.
- If D5-min exists, consume its constants/policy and apply them to BIRTH/DEATH edges deterministically.

**Exit condition:** solver still runs deterministically; birth/death totals appear sane in audit.

### Step 3 (D3-POC-4): Enforce hard constraints (must_link / cannot_link) with fail-fast UNSAT
Goal: final POC correctness requirement.
- Enforce **must_link** groups and **cannot_link** pairs as feasibility constraints.
- If infeasible, fail-fast with audit diagnostics:
   - which group/pair
   - minimal implicated node_ids / base_tracklet_ids
   - counts of constraints

Implementation options (choose one; document clearly):
1) **Single-stage ILP (recommended):** encode flow + constraints directly.
2) **Two-stage exact approach:** baseline propose + exact re-solve enforcing constraints (no heuristics).

**Exit condition:** produced outputs satisfy all constraints; deterministic; or fail-fast UNSAT with clear diagnostics.

### Step 4: Emit artifacts (F0-locked)
`identity_assignments.jsonl`:
- one record per `person_id` with:
   - list of member `node_id`s and underlying `base_tracklet_id`s
   - anchored tag key (if any)
   - constraint application summary (counts of must_link/cannot_link satisfied)

`person_tracks.parquet`:
- must include traceability back to:
   - `person_id`
   - source `base_tracklet_id`
   - frame_index
- D3 may use `tracklet_bank_frames.parquet` (D0) for per-frame geometry emission, but must not invent geometry.

### Step 5: Validation hooks (must be implementable)
Stage D validation must check:
- all referenced `base_tracklet_id`s exist upstream (via manifest-registered artifacts)
- all must_link/cannot_link constraints are satisfied in final assignments (or UNSAT fail-fast)
- person_tracks spans are consistent with source spans (no out-of-range frames)

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
   - backend choice (ILP recommended for hard constraints; MCF-only allowed only as an intermediate checkpoint) with clear constraints implications
   - deterministic indexing / ordering strategy
   - constraint enforcement strategy using `d2_constraints.json` (must_link/cannot_link) and fail-fast UNSAT conditions
2) A list of **questions / assumptions** you need confirmed (only if blocking).
3) A draft **Interface Contract** for D3 code:
    - functions/classes for:
       - loading canonical D1/D2 artifacts (manifest-driven)
       - pruning disallowed edges (deterministic)
       - running solve() for checkpoints (POC-2/3/4)
       - enforcing hard constraints (exact; no heuristic penalties)
       - writing F0-locked artifacts
   - acceptance tests (pytest) for must_link + cannot_link enforcement and determinism

End your first response by asking me to review/approve the plan before you go deeper.

Also include a bullet explicitly confirming alignment with the locked F0/F3 contracts (artifacts, paths, manifest).

### POC checkpoints (must keep pipeline runnable)
- D3-POC-1: graph compiles + joins + pruning (audit-only)
- D3-POC-2: solver runs without hard constraints (wiring + GROUP capacity correctness)
- D3-POC-3: birth/death costs integrated (D5-min or explicit placeholder)
- D3-POC-4: hard constraints enforced (must_link/cannot_link) with UNSAT fail-fast

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
