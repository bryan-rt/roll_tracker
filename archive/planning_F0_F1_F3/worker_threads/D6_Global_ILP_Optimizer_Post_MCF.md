# D6 — Global ILP Optimizer (Post-MCF)

## Update: Locked constraints to honor (F0 + F3)

- **F3 ingest contract**: clips live at `data/raw/nest/<camera_id>/YYYY-MM-DD/HH/<camera_id>-YYYYMMDD-HHMMSS.mp4`
- **Processing reads only** from `data/raw/**` and **writes only** to `outputs/<clip_id>/...`
- **F0 contracts are authoritative**: do not invent schemas; use `src/bjj_pipeline/contracts/*`
- **Run anchor**: `outputs/<clip_id>/clip_manifest.json`
- **Stage artifacts are locked** (Parquet/JSONL + masks `.npz`), and paths must be **relative** to `outputs/<clip_id>/`

---

## Why this worker exists
We are **not deferring ILP**. The solver stack is:

1) **MCF (Stage D core)**: produces locally consistent `person_tracks` under hard constraints.
2) **ILP (global)**: resolves remaining global inconsistencies and enforces high-level constraints that MCF may not capture cleanly, especially:
- long-range identity consistency across difficult occlusions
- uniqueness constraints tied to AprilTags
- “one person can’t be in two places at once”
- (optional) cross-clip/session consistency

The ILP does not replace MCF; it **post-processes** MCF outputs to produce globally consistent assignments and conflict resolution signals.

---

## Scope
### In-scope
- Define ILP variables, objective, and constraints using:
  - Stage D `person_tracks.parquet`
  - Stage C `identity_hints.jsonl`
  - ReID banks / similarity signals (if available)
  - homography-space feasibility gates
- Decide the ILP “granularity”:
  - link-person-chains across tracklet fragments, OR
  - resolve conflicts where multiple person_tracks claim same tag
- Implementation library recommendation:
  - OR-Tools CP-SAT OR PuLP OR Pyomo (pick one and justify)
- Output integration:
  - must remain compatible with F0:
    - primary outputs stay `stage_D/person_tracks.parquet` and `stage_D/identity_assignments.jsonl`
  - ILP may produce:
    - updated `identity_assignments.jsonl` (preferred)
    - optional debug artifact registered in manifest

### Out-of-scope (for now)
- Cross-day, multi-session global optimization (future)
- Full multi-hypothesis tracking explosion

---

## Key design decisions to make
### 1) What ILP is optimizing
We need a precise statement:
- either it assigns **track chains** to **identities (tag anchors)**,
- or it selects **merge/split decisions** among candidate links produced by MCF,
- or it resolves **conflicting identity assignments** with a global objective.

### 2) Objective function (candidate)
Minimize:
- (1 - ReID similarity) for selected links
- spatial infeasibility penalties (teleport)
- number of births/deaths (encourage continuity) — consistent with D5
Subject to:
- hard constraints from AprilTags (must_link/cannot_link)
- uniqueness (one tag per time)

### 3) Constraints (must have)
- **Tag uniqueness over time**: a given `tag:<id>` cannot be assigned to two simultaneous chains.
- **Mutual exclusion**: a person_id cannot map to two tags; if undecided, must remain unknown.
- **Reachability**: if chain A ends at (x1,y1,t1) and chain B begins at (x2,y2,t2), require feasible speed or penalize heavily.
- **Evidence consistency**: must_link implies same identity; cannot_link forbids.

---

## Inputs & Outputs
### Inputs
- `stage_D/person_tracks.parquet` (from MCF)
- `stage_C/identity_hints.jsonl`
- optional: feature-bank similarities (D4) and mat-zone gates (D5)

### Outputs
- Update or produce `stage_D/identity_assignments.jsonl` with:
  - final assignment decisions
  - evidence summary and conflicts resolved
- If ILP changes person_track grouping, document how it maps back to `person_id` stability (may require a controlled re-index).

---

## Implementation expectations
- Start with a **minimal ILP** that only resolves:
  1) duplicate tag assignments
  2) impossible simultaneity
  3) best assignment of untagged chains to tagged anchors when evidence is strong

Then expand to richer linking.

---

## Deliverables back to Manager
1) ILP formulation: variables, objective, constraints (math + narrative)
2) Library pick + rationale
3) Integration plan into Stage D:
   - where it runs (post MCF)
   - how it reads/writes artifacts and respects F0
4) Test plan:
   - synthetic conflicts (two chains claim same tag)
   - long occlusion gaps with later tag reappearance
   - cannot_link enforcement

---

## Kickoff
Please begin by writing:
1) The minimal viable ILP formulation we can implement first.
2) The exact artifact touch-points (what you read, what you write).
3) A test plan with at least 3 deterministic synthetic cases.

End by asking me to approve the plan before implementation.

Also include an explicit bullet confirming alignment with the locked F0/F3 constraints (paths, manifest, artifacts).
