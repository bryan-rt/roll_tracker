# D6 — Global ILP Optimizer (Post-MCF)

> **POC posture:** D6 is **OPTIONAL / DEFERRED** unless MCF alone fails on real clips.
> Treat D6 as a “Phase 2 refinement” that can be bolted on without changing F0 artifacts.

## Update: Locked constraints to honor (F0 + F3)

- **F3 ingest contract**: clips live at `data/raw/nest/<camera_id>/YYYY-MM-DD/HH/<camera_id>-YYYYMMDD-HHMMSS.mp4`
- **Processing reads only** from `data/raw/**` and **writes only** to `outputs/<clip_id>/...`
- **F0 contracts are authoritative**: do not invent schemas; use `src/bjj_pipeline/contracts/*`
- **Run anchor**: `outputs/<clip_id>/clip_manifest.json`
- **Stage artifacts are locked** (Parquet/JSONL + masks `.npz`), and paths must be **relative** to `outputs/<clip_id>/`

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

---

## Module-specific context (D6 — ILP refinement)

This worker explores improving stitching beyond min-cost flow using an ILP/MIP to enforce global constraints.
The core idea: run MCF first, then optionally solve a refinement problem over a reduced candidate set.

### Hard constraints (do not violate)
- **No new canonical artifacts** without an explicit F0 bump.
- Must be **deterministic** (same inputs + config → same solution).
- Must not require Stage B.
- Must be able to run as a post-pass over existing Stage D candidate structures.

### Inputs (authoritative)
- Stage A: `tracklet_frames.parquet`, `tracklet_summaries.parquet`, `contact_points.parquet` (geometry / spans)
- Stage C: `identity_hints.jsonl` (must_link/cannot_link constraints)
- Stage D (from MCF): candidate graph / intermediate solution (in-memory or dev-only cache)

### Outputs
- By default, D6 should output:
  - **an updated in-memory assignment** (same shape as MCF output), and/or
  - **dev-only diagnostics** under `outputs/<clip_id>/_debug/` / `_cache/`
- Final canonical outputs remain Stage D’s:
  - `stage_D/person_tracks.parquet`
  - `stage_D/identity_assignments.jsonl`
  - `stage_D/audit.jsonl`

---

## Borrow from roll_it_back — adaptation notes
If roll_it_back used global optimization (ILP) or constraint propagation:
- Reuse concepts (reduced candidate sets, hard constraints, penalty terms),
- Adapt identifiers + artifacts:
  - variables reference `tracklet_id` and D-edge candidates
  - hard constraints consume C2 hints (must_link/cannot_link)
  - geometry terms come from Stage A (x_m,y_m,on_mat)
  - keep all outputs deterministic and auditable

---

## D6 problem framing (POC-safe baseline)

### When to run D6
Default: **off**.
Enable only if one of these is true (config-controlled):
- MCF solution violates known-hard constraints (should be rare if D2 enforced them),
- MCF produces clearly fragmented identities around tag anchors,
- or we want to test a refinement path on a small clip.

### Variable design (typical)
Assuming D1/D2/D3 already define candidate edges A->B and births/deaths:
- Binary variables for selecting edges (or re-selecting edges) within a constrained neighborhood.
- Optional variables for “identity anchor assignment” per connected component (if needed).

### Hard constraints (must enforce)
- Flow constraints (each tracklet has at most one successor and one predecessor in a person path).
- C2 **cannot_link** between tracklets:
  - disallow selecting paths that place both in same person identity.
- C2 **must_link** anchors:
  - enforce that all tracklets assigned to same tag anchor are consistent (exact encoding depends on D2 contract).

### Objective (typical)
Minimize:
- sum(edge_costs) + birth/death costs (from D2/D5)
- plus optional penalties:
  - anchor inconsistency (should be hard if feasible)
  - excessive identity switches
  - implausible speed jumps (based on geometry deltas)

All terms must be documented and deterministic.

---

## Solver guidance (implementation choices)
Prefer OR-Tools MIP/CP-SAT only if:
- model size is constrained (small neighborhood)
- we can guarantee deterministic solve via fixed params

Otherwise:
- keep D6 as a research sandbox and do not merge into the POC critical path.

---

## Acceptance criteria (D6)
1) **Non-blocking**
   - With D6 disabled, pipeline outputs unchanged.
2) **Deterministic**
   - identical inputs/config → identical refined solution (including tie-breakers / solver params).
3) **Constraint compliance**
   - never violates C2 cannot_link / must_link (as encoded).
4) **Audit evidence**
   - logs why refinement ran, model size, solve status, and delta vs MCF.
5) **Pytest**
   - tiny synthetic graph where MCF makes a known suboptimal choice; ILP corrects it deterministically.

---

## Required deliverables back to Manager
1) A minimal ILP refinement formulation:
   - variables (what decisions are being changed vs MCF)
   - objective terms (costs + any penalties)
   - hard constraints (flow + must_link/cannot_link + uniqueness, if applicable)
2) Library pick + determinism plan:
   - how ties are broken
   - solver parameters pinned for reproducibility
3) Integration plan into Stage D (post-MCF):
   - where the candidate neighborhood comes from
   - what gets updated (in-memory vs dev-only debug)
   - what goes into `stage_D/audit.jsonl`
4) Test plan:
   - at least 3 tiny deterministic synthetic graphs
   - includes must_link and cannot_link cases

---

## Kickoff
Please begin by writing:
1) The minimal viable ILP refinement we implement first (small neighborhood only).
2) Exact artifact touch-points (what you read, what you write).
3) A deterministic test plan with at least 3 synthetic cases.

End by asking me to approve the plan before implementation.

Also include an explicit bullet confirming alignment with the locked F0/F3 constraints (paths, manifest, artifacts).

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

If performance becomes an issue, prefer reducing neighborhood size / selecting fewer candidates over skipping frames at POC time.
