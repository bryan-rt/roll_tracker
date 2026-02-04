# 2026-01-27 Update: Implementation Reality Check

## Canonical D1 Outputs (F0-locked, solver-agnostic)
D1 now produces three canonical artifacts, defined in F0 and used by both MCF and ILP solvers:
- `stage_D/d1_graph_nodes.parquet`
- `stage_D/d1_graph_edges.parquet`
- `stage_D/d1_segments.parquet`

Schemas are versioned and centrally defined in F0 (`f0_parquet.py`). All fields required for downstream solvers and pricing are present.

## Identity/AprilTag Integration
- D1 node and edge schemas include join keys and identity-hint fields (`base_tracklet_id`, `segment_type`, `must_link_anchor_key`, etc.).
- These enable direct joining with Stage C AprilTag/identity payloads for pricing and constraint enforcement.

## Extensibility for ReID/RGB
- The `payload_json` field in D1 artifacts is a lossless, forwards-compatible container for future additions (e.g., RGB crops, ReID embeddings).

## Debug and Validation
- D1 writes audit/debug artifacts for traceability.
- All outputs are validated against F0 contracts before completion.

*The rest of this document remains accurate and reflects the current implementation. See F0 for authoritative schemas and update this doc if any contract changes are made in code.*
# D1 — MCF Graph Model (Nodes/Edges)

## Addendum — 2026-01-18 (POC reality check: A+C online; Stage B deferred; C2 constraints live)

### Pipeline execution model (locked for POC)
- **Phase 1 (online decode pass):** `multiplex_AC` runs **Stage A + Stage C** in a single shared frame loop (decode once).
- **Phase 2 (offline artifact pass):** **Stage D** runs offline (artifact-driven) and owns the Min-Cost Flow stitcher.

### Stage B status (locked)
- **Stage B is DEFERRED for the POC.** Do not depend on any Stage B artifacts for D1/D2/D3.

### Canonical geometry source (F-bump implemented)
- **Stage A** now owns canonical, join-friendly geometry via:
   - `stage_A/contact_points.parquet` (baseline per-frame/per-detection contact + x/y + on_mat)

### Identity constraints (C2 complete)
- Stage C now emits:
   - `stage_C/identity_hints.jsonl` containing:
      - `must_link(tracklet_id -> tag:<id>)` anchors
      - `cannot_link(tracklet_id <-> tracklet_id)` conflicts (symmetric)

**Clarification (POC-locked):**

D1 treats identity hints as **annotations and logical infeasibility guards**, not as full
identity enforcement.

Specifically:
- `must_link` constraints are **carried forward as annotations** for downstream enforcement
   (D3) and are *not* used to prune or force edges in D1.
- `cannot_link` constraints are used **only** to prune *logically impossible*
   **SOLO → SOLO continuation edges**, where a continuation edge explicitly represents
   a same-identity hypothesis.

No other edge types (GROUP, MERGE, SPLIT, BIRTH, DEATH) are pruned based on identity hints.
Final identity feasibility is enforced exclusively in D3.


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
- `stage_A/contact_points.parquet`
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


# D1 Completion Addendum — 2026-01-25

This addendum records the **final design decisions, invariants, and domain-specific reasoning**
that were validated during D1 implementation and artifact review.

It is intended to:
- prevent accidental regression in future refactors
- give D2/D3 workers correct mental models
- document *why* certain logic exists (not just *what* exists)

---

## D1 Purpose (clarified and locked)

D1 exists to **enumerate all temporally and spatially plausible identity hypotheses**
given raw tracklets and hard identity constraints.

D1 **does not**:
- score identity hypotheses
- choose between competing hypotheses
- enforce final identity decisions

D1 **does**:
- model ambiguity explicitly
- preserve alternative explanations
- encode temporal capacity constraints for downstream solvers

This separation is non-negotiable for correctness.

---

## Lifespan segmentation (MANDATORY)

**Decision:** `enable_lifespan_segmentation = true` is mandatory for POC and beyond.

### Rationale
In grappling domains (BJJ / wrestling):
- meaningful identity events occur *inside* long-lived tracklets
- merges and splits are not aligned with tracklet start/end
- long occlusions (seconds to minutes) are common and expected

Therefore, treating a tracklet as a single atomic node is incorrect.

### Implementation invariant
Each carrier tracklet is deterministically segmented into:
- `SOLO` segments (capacity = 1)
- `GROUP` segments (capacity = 2)

Segments are:
- contiguous
- non-overlapping
- ordered by time

Downstream workers must treat segments — not raw tracklets — as the true graph nodes.

---

## Semantics of GROUP segments (critical)

A `GROUP` segment represents **identity ambiguity**, not tracker failure.

Specifically:
- capacity > 1 means multiple people may be represented by a single carrier
- no identity assignment is implied or decided
- ambiguity is intentionally preserved for MCF / ILP resolution

GROUP does **not** mean:
- poor detection
- incorrect tracking
- persons are merged permanently

This distinction is essential for D2 cost modeling and D3 solving.

---

## Event classes D1 is designed to capture (validated)

D1 explicitly supports all of the following:

1) **Start-group → later split**
   - video begins with two people in one bbox
   - split may occur long after start

2) **Merge → occlusion → delayed split**
   - merge and split separated by long temporal gaps

3) **Never-merge individuals**
   - remain SOLO for entire lifespan

4) **Merge without split before video end**
   - modeled as `merge_open_end`
   - identity intentionally unresolved

All four cases were observed in real artifacts and verified.

---

## Group span clamping (bug fix, now locked)

### Correct invariant
Group spans must be clamped to the *carrier tracklet lifespan*:

```
group_start = max(group_start_raw, carrier.start_frame)
group_end   = min(group_end_raw,   carrier.end_frame)
```

### Reason
Carriers may persist across:
- multiple merges
- multiple splits
- partial occlusions

Group spans must never exceed the carrier’s actual temporal support.

This fix is considered **correct and final**.

---

## Split search horizon (domain-specific decision)

`split_search_horizon_frames` defines:
> how long after a merge we are willing to associate a reappearance as a split

### Grappling-specific reasoning
- partners may be fully occluded for tens of seconds
- spatial continuity matters more than short-term temporal proximity

Therefore:
- the horizon must be **large but finite**
- not infinite (avoid pathological matches)
- not small (would miss valid splits)

Tuning this is a **domain modeling choice**, not a bug fix.

---

## Other D1 gates (how to think about them)

### `merge_trigger_max_age_frames`
Acts as a **noise guard**, not an identity decision.

Purpose:
- prevents stale disappearances from creating false merges
- limits influence of detector dropouts

It should be tuned conservatively and treated as a sanity filter.

---

### `max_continue_gap_frames`
Purely a **graph sanity constraint**.

It:
- prevents absurd temporal jumps
- does *not* encode identity likelihood

Identity plausibility belongs in **D2 costs**, not D1 gating.

---

## Identity constraints (C2 integration — clarified)

Stage C constraints are treated as **hard feasibility constraints**:

- `must_link(tracklet → tag:<id>)`
   - enforces shared identity across segments

- `cannot_link(tracklet ↔ tracklet)`
   - forbids shared identity in any valid flow

**Clarification of responsibility split:**

- D1 **does not enforce identity feasibility globally**
- D1 **does not collapse ambiguity**
- D1 **does not force assignments**

D1 only applies `cannot_link` when a graph edge *semantically asserts identity sameness*
(i.e., SOLO → SOLO continuation).

All other uses of identity hints — including must-link enforcement, multi-edge feasibility,
and final identity resolution — are the responsibility of **D3**.

---

## Explicit non-responsibilities of D1 (locked)

D1 must **not**:
- assign costs (D2)
- decide identity (D3)
- collapse ambiguity early
- globally enforce identity feasibility
- infer births/deaths beyond structural arcs

If future work appears to “need” this, the fix belongs downstream.

---

## D1 status

✅ Graph structure complete  
✅ Domain cases covered  
✅ Ambiguity preserved correctly  
✅ Ready for D2 cost modeling and D3 solving  

D1 is considered **POC-complete**.

Future work should build *on top of* this structure, not revise it.


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
   - Tooling target (future): SAM/SAM2 offline refinement + sparse overrides.

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag detection in **expanded bbox ROI** + deterministic voting (C2).
   - Output: `tag_observations.jsonl` + `identity_hints.jsonl` (constraints, not final IDs).

4) **Stage D — Global stitching (Min-Cost Flow)**
   - Tooling target: MCF solver (start with OR-Tools or NetworkX; optimize later).
   - Inputs: tracklets + canonical (x,y) + optional ReID + **C2 constraints**.
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
- **AprilTags**: OpenCV ArUco (`cv2.aruco`) (current implementation)
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


## Module-specific context (D1 — MCF graph model)
Define the stitcher graph at the **tracklet level**, with explicit modeling for:
- **birth / death** (source/sink arcs)
- **identity anchors** (tag must_link)
- **identity conflicts** (tracklet↔tracklet cannot_link)

This worker is the **formal definition** of graph objects that D2/D3 implement.

### Inputs available to D1 (POC)
Authoritative (must work with just these):
- `stage_A/tracklet_summaries.parquet` (tracklet spans)
- `stage_A/contact_points.parquet` (canonical geometry; join-ready)
- `stage_C/identity_hints.jsonl` (C2 constraints)

Helpful but optional:
- `stage_A/tracklet_frames.parquet` (more detailed per-frame trajectories if needed for edge features)
- `stage_A/detections.parquet` (bbox stats / size gates)
- `stage_A/audit.jsonl` (debug counters / sanity checks)

**Hard rule:** D1 must not depend on Stage B.

### Must include
- Node definition: **tracklet node** with:
   - `tracklet_id`, `(start_frame, end_frame)`, `n_frames`
   - endpoint features derived deterministically from `contact_points`:
      - `(x_start,y_start)` and `(x_end,y_end)` using the first/last valid `on_mat` rows when possible
      - fallback behavior when x/y missing (do not guess; mark as unknown and gate edges accordingly)
- Explicit **SOURCE** and **SINK** nodes and arc templates:
   - `SOURCE -> tracklet` (birth)
   - `tracklet -> SINK` (death)
   - (optional) direct `SOURCE -> SINK` for “unused flow” if your solver formulation needs it
- Edge gating:
  - time gap max
  - spatial max (ground-plane)
  - optional appearance gate
- Identity constraints integration:
   - must_link(tracklet → tag:<id>) becomes a hard requirement on feasible assignments
   - cannot_link(tracklet ↔ tracklet) forbids same-identity grouping in the stitch solution
- Debug outputs (dev-only) to explain:
   - why an edge exists / is pruned (reason codes)
   - which constraints eliminated which pairings

### Invariants
- Graph construction deterministic given artifacts + config
- Graph is **join-friendly**: every node/edge references upstream identifiers (`tracklet_id`, frame ranges) and never invents hidden IDs.

### Non-responsibilities (D1)
- Do not implement costs (belongs to D2).
- Do not implement the solver (belongs to D3).
- Do not decide births/deaths thresholds (belongs to D5; D1 just models the arcs).


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

### POC-specific kickoff requirements (add these)
- Confirm exactly which fields from `stage_A/contact_points.parquet` you will treat as authoritative for:
   - endpoints
   - edge gating
   - “unknown geometry” handling
- Specify how you will interpret **C2 hints** in the graph model:
   - must_link as an anchor constraint
   - symmetric cannot_link as a hard exclusion
- Provide explicit reason codes for edge pruning (these will be mirrored in D2/D3 audits).

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
