---
layout: page
---

# roll_tracker planning pack (expanded)

Each file in `planning/worker_threads/` is meant to be pasted as the **first message** in a dedicated worker chat.

**Workflow**
1) Create a new worker chat whose title starts with the worker ID (e.g., `D2`).
2) Paste the entire contents of the corresponding worker markdown file.
3) Ask the worker to deliver the required outputs back to the Manager thread.
4) Paste the worker’s summary back into the Manager thread to “lock” decisions.

**Important**
- Min-Cost Flow stitching is mandatory.
- All stages communicate only through versioned artifacts defined in F0.

## Hybrid pipeline (current POC execution model)
- **Phase 1 (online decode pass):** run **Stage A + Stage C** together in `multiplex_AC` (single video decode loop).
- **Phase 2 (offline artifact pass):** run **D → E → X** sequentially (artifact-driven).
- **Stage B (SAM refinement work)** is **deferred** until after the A+C→D POC.

### Stage C note (POC ROI + scheduling)
- Stage C uses **expanded bbox ROIs** as the primary decode region (do not hard-clip to YOLO masks).
- Decode cadence / backoff / ramp-up is owned by **C0** (Tag Decode Scheduling & Cadence).

## Current locks
- **F3** ingest contract: clips under `data/raw/nest/...`
- **F0** contracts: stage artifacts + manifest anchored at `outputs/<clip_id>/clip_manifest.json`

Worker docs include an update section summarizing these locks. Always align to them.

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

### POC clarification (Jan 2026)
For the current POC, the active target is `multiplex_AC` (A + C). `multiplex_ABC` exists as an architectural capability, but **Stage B is deferred** and must not be required for a valid run.

---

## Stage D POC map: bare-bones offline stitcher (MCF) (Jan 2026 lock)

This section defines the **minimum coherent** end-to-end deliverable for Stage D:
an **offline, artifact-driven Min-Cost Flow (MCF) stitcher** that converts Stage A tracklets into
entity-level trajectories, and demonstrates improved behavior versus the legacy Hungarian approach,
especially under **overlap / merge / split** tracklet behavior and long gaps between AprilTag pings.

### Core hypothesis (POC focus)
When two people overlap and the tracker produces fragmented tracklets (merge/split), a properly modeled
MCF global optimization will propagate identity **more robustly** than a local Hungarian matcher, enabling
better stitching across:
- overlap-induced fragmentation,
- partial occlusions,
- long temporal gaps between AprilTag anchor pings.

### What “success” means for the POC
The POC is successful if we can:
1) Run **A + C** (multiplex) and then **D** (offline) end-to-end deterministically.
2) Produce canonical Stage D artifacts (`person_tracks`, `identity_assignments`, audit).
3) Produce stitch outputs suitable for **qualitative manual review** against prior Hungarian runs,
   especially across overlap/occlusion segments and sparse tag pings.

### POC scope (what must exist)
Stage D is implemented as D0–D5 workers, but the POC does **not** require full polish in every worker.

**Required for POC**
- **D0**: minimum hygiene + endpoint evidence (no identity decisions).
- **D1**: deterministic graph construction (nodes/edges + source/sink).
- **D2**: geometry-first costs + constraint semantics (must_link / cannot_link).
- **D3**: solver wiring + constraint enforcement + canonical output emission.
- **D5**: minimal birth/death + missing-geometry policy (zones optional).

**Explicitly optional / deferred**
- **D4**: ReID embeddings (off by default for the POC).
- Advanced D0 scoring: junk detection, jump detection, sophisticated smoothing.
- Complex mat-zone modeling: polygon zones, nuanced penalties beyond baseline constants.

### Artifact contract (Stage D outputs)
Stage D writes:
- `stage_D/person_tracks.parquet` — entity-level tracks over time.
- `stage_D/identity_assignments.jsonl` — mapping/evidence for person_id ↔ tracklets ↔ tag anchors.
- `stage_D/audit.jsonl` — deterministic audit trail of graph stats, constraints, solver summary.

D0 may additionally write internal “bank” artifacts used by later D stages, but **must not** invent new
public-facing schemas without an F0 bump.

### Minimal D0 requirements (POC)
D0 is the last geometry-modifying step. For the POC, D0 must guarantee:
- Stable per-tracklet endpoints where geometry exists.
- “No poison” behavior: suppress catastrophic velocity spikes that would explode costs.
- Emit basic per-tracklet evidence needed downstream:
  - start_frame / end_frame
  - start/end (x_m, y_m) when available
  - on_mat prevalence or endpoint on_mat
  - contact_conf summary (or equivalent reliability scalar)
  - missing-geometry indicators

D0 must **never**:
- merge/split tracklets,
- assign identity,
- discard tracklets (score/flag only).

### Minimal D5 requirements (POC)
D5 provides policy + constants (no new artifacts):
- `base_birth_cost`, `base_death_cost`
- `missing_geom_policy`: `penalize` (default) or `disallow`
- optional toggles:
  - `require_on_mat_for_birth`
  - `require_on_mat_for_death`

Zones are deferred; the POC can treat “mat gating” as a simple boolean policy using `on_mat`.

### Minimal D1 graph model (POC)
Build a tracklet-level time-respecting graph.

**Nodes**
- one node per tracklet (from Stage A summaries), carrying endpoint evidence.

**Arcs**
- `SOURCE -> tracklet` (birth)
- `tracklet -> SINK` (death)
- `tracklet_i -> tracklet_j` continuation candidates where:
  - `end_frame_i < start_frame_j`
  - dt <= max_dt_frames
  - **kinematic feasibility** holds (human-plausible ability to bridge the gap):
    - required average speed `v_req = dist_m / dt_s` is within a configured bound, and
    - (optional) required acceleration / velocity-jump is within configured bounds
  - missing-geometry handling follows D5 `missing_geom_policy` (penalize vs disallow)

**Candidate generation**
- Must be deterministic (sorting, tie-breaking).
- Prefer conservative gating for POC (fewer candidates, but correct).
- Never allow edges that violate hard constraints (e.g., cannot_link-known incompatibilities).

### Minimal D2 costs + constraints (POC)
**Edge cost (geometry-first baseline)**
For each candidate continuation edge i→j:
- `dist_norm = dist_m / max(1e-6, dt_s * v_ref_mps)`
- `cost = w_dist * dist_norm + w_bias`
- Optional: `w_vjump * vjump_penalty` where vjump is computed from repaired endpoints.

**Birth/death costs**
- Use D5 constants, optionally conditioned on on_mat policy.

**Constraints**
- `must_link(tracklet -> tag)` is treated as a hard feasibility requirement on final paths.
- `cannot_link(tracklet_a <-> tracklet_b)` must be enforced (hard).
- If constraints are unsatisfiable: fail-fast with a clear audit report.

Note: For the POC, we accept that some constraints may need implementation tricks (e.g., grouping or
multi-commodity approximations), but the output must satisfy them deterministically. Do not “cheat” with
large negative costs unless explicitly locked by the Manager.

### Minimal D3 solver wiring + output emission (POC)
D3 must:
- Solve the graph with an MCF-capable method (implementation details per worker plan).
- Enforce constraints deterministically and fail-fast when unsatisfiable.
- Emit canonical artifacts:
  - `person_tracks.parquet`
  - `identity_assignments.jsonl`
  - `audit.jsonl` (include graph stats, cost stats, constraint counts, solver objective).

### Iterative checkpoints (keep runnable at every step)
We will build Stage D POC in a sequence that keeps end-to-end runs valid:

**Checkpoint D-POC-1: “Graph compiles”**
- D1 builds nodes + candidate edges; D3 can run in “dry run” mode; audit includes counts.

**Checkpoint D-POC-2: “Solver runs”**
- D3 solves with baseline costs (no constraints beyond basic gating).
- Emits `person_tracks` with deterministic person_id assignment.

**Checkpoint D-POC-3: “Birth/death modeled”**
- Add SOURCE/SINK costs from D5; validate that tracks can start/end reasonably.

**Checkpoint D-POC-4: “Identity hints enforced”**
- Must_link + cannot_link are enforced; unsat is fail-fast with audit diagnostics.

**Checkpoint D-POC-5: “Manual qualitative review readiness”**
- Ensure Stage D outputs are stable and interpretable for **manual** comparison against prior Hungarian runs:
  - consistent person_id assignment rules,
  - audit contains enough graph/solver stats to debug surprising results,
  - identity assignments are traceable to tag anchors (where present).

### Deferred feature backlog (earmarked)
- D0 Checkpoint 4: degenerate/junk tracklet scoring.
- D0 Checkpoint 6: identity-swap / “tid jump” evidence detection.
- Sophisticated velocity models (accel jerk bounds, contact-aware physics).
- D4 embeddings and appearance gating.
- Rich D5 mat-zone polygons + zone-aware birth/death penalties.
- Future evaluation harness (metrics + golden clips + regression tracking) — explicitly deferred.

---

## Roadmap note
We will push iteratively forward to complete the bare-bones offline stitcher first and gather evidence
on whether MCF improves identity propagation under overlap/merge/split behavior compared to Hungarian.
Only after we have POC evidence will we decide which deferred elements are worth implementing next.
