---
layout: page
---

# D5 — MCF Birth/Death + Mat-Zone Gating

> **Why this matters:** Birth/death decisions are **foundational** to MCF quality.
> D5 owns the **policy** and **costs** for:
> - when a tracklet is allowed to start a new person (birth)
> - when a tracklet is allowed to end a person (death)
> - when links are disallowed/penalized due to mat-zone / off-mat state

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

## Module-specific context (D5 — births/deaths + gating)

This worker defines:
1) **Birth costs / eligibility** (source edges)
2) **Death costs / eligibility** (sink edges)
3) **Mat-zone / on_mat gating** used to disallow or penalize links

### Inputs (authoritative)
From Stage A:
- `tracklet_frames.parquet` (frame-level `x_m,y_m,on_mat,contact_conf`)
- `tracklet_summaries.parquet` (start/end frame spans)
- `contact_points.parquet` (join-friendly geometry table; optional if frames already has endpoints)

### Outputs
- No new artifacts. D5 feeds parameters + helper functions into D2/D3.
- D5 must define config keys under `stage_D.d5_birth_death` (or similar) and document defaults.

### Borrow from roll_it_back — adaptation notes
We may reuse the **concepts**:
- “edge eligibility by zone”
- “penalize births/deaths away from boundaries”
- “prefer continuity on-mat”
But adapt to roll_tracker signals:
- use `on_mat` + `contact_conf` + (x,y) from Stage A
- do not rely on Stage B refined masks (deferred)
- do not assume a specific mat coordinate frame beyond homography outputs

---

## D5 policy (POC baseline)

### 1) Birth eligibility + cost
Birth edges represent starting a new person track at a tracklet’s start.

POC-safe baseline:
- Allow birth if:
  - tracklet has at least one valid geometry sample (x_m,y_m non-null) **OR** missing_geom_policy allows it.
  - (optional) start endpoint is on_mat == true (if configured as hard gate).
- Birth cost:
  - base_birth_cost (constant)
  - plus boundary penalty if start is far from mat boundary / entry zones (if zones configured)

### 2) Death eligibility + cost
Death edges represent ending a person track at a tracklet’s end.

POC-safe baseline:
- Allow death if:
  - end endpoint is valid (or missing_geom_policy allows it)
  - (optional) end endpoint is on_mat == true (if configured)
- Death cost:
  - base_death_cost (constant)
  - plus boundary penalty if end is far from mat boundary / exit zones (if zones configured)

### 3) Mat-zone gating for links
For candidate link A->B (A ends before B starts):
- If either endpoint lacks geometry:
  - follow missing_geom_policy: "penalize" or "disallow" (configurable, deterministic)
- If both endpoints have geometry:
  - hard-disallow if endpoints violate zone rules (e.g., A ends off-mat but B starts deep on-mat with no plausible transition)
  - otherwise apply a zone transition penalty (added to D2 edge cost)

### 4) Missing geometry policy (must be explicit)
Because homography/contacts can be null, D5 must define:
- `missing_geom_policy: "penalize" | "disallow"`
- `missing_geom_penalty: float` (used when penalize)

All decisions must be auditable (counts of disallowed vs penalized edges).

---

## Configuration (F2-compatible; suggested keys)
Under `config["stage_D"]["d5_birth_death"]`:
- `base_birth_cost: float`
- `base_death_cost: float`
- `require_on_mat_for_birth: bool`
- `require_on_mat_for_death: bool`
- `missing_geom_policy: str`  # "penalize" | "disallow"
- `missing_geom_penalty: float`
- (optional) `mat_zones: ...`  # named polygons/rects in (x,y) meters for entry/exit preference

All resolved values must be logged by Stage D audit.

---

## Acceptance criteria (D5)
1) **Edge gating is deterministic**
   - same inputs/config → same allowed/disallowed sets and costs.
2) **Birth/death are explicitly modeled**
   - D3 solver uses D5 birth/death costs (not implicit defaults).
3) **Audit evidence**
   - counts of births/deaths chosen
   - counts of candidate links removed due to gating
   - counts of endpoints missing geometry and how handled
4) **Pytest**
   - synthetic endpoints verify:
     - birth/death eligibility rules
     - missing geometry policy behavior
     - mat-zone penalty application

---

## Required deliverables back to Manager
1) Formal definitions for:
   - on-mat scoring
   - entry/exit zones
   - birth/death costs
2) How these modify D1 graph construction + D2 cost function
3) Tests to add (synthetic tracklets):
   - person walks onto mat boundary → allowed birth
   - person disappears mid-mat → high death, solver prefers link to later reappearance
   - off-mat spectators → excluded/low priority
4) Copilot prompt pack to implement in `src/bjj_pipeline/stages/stitch/*`

---

## Kickoff
Please begin by proposing:
1) Exact formulas (with constants) for mat gating and birth/death costs.
2) Where in code these belong (graph vs costs modules).
3) A minimal test plan with fixtures.

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

If performance becomes an issue, prefer reducing resolution / selecting fewer candidates over skipping frames at POC time.
