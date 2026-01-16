# C2 — Identity Registry (Voting + Conflicts)

## Addendum — 2026-01-XX (Post-F0G: canonical join keys; Stage B deferred; canonical outputs only)

### POC lock: execution + Stage B posture
- Stage C runs online in **`multiplex_AC`** (A + C). Stage B is **DEFERRED** and must not be required.

### Canonical join keys / data expectations
- C2 should treat `(frame_index, detection_id)` as the primary join key for C1 evidence, with `tracklet_id` as the identity-voting key.
- If geometry is needed for gating/weighting (optional), in multipass it must come from **Stage A `stage_A/contact_points.parquet`**.

### Output contract lock (F0-aligned)
- Canonical outputs remain: `stage_C/tag_observations.jsonl`, `stage_C/identity_hints.jsonl`, `stage_C/audit.jsonl`
- Any additional registry/conflict/evidence files are **dev-only** unless F0 is explicitly bumped.

> Note: In POC, primary ROI is `roi_source="bbox_expanded"`. Treat mask-hint sources as secondary evidence quality signals only.

## Addendum — 2026-01-14 (POC update: C online in multiplex_AC; bbox-expanded ROI; Stage B deferred; outputs must match F0)

### Hybrid pipeline execution model (locked for POC)
We are using a **single pipeline** with two phases:
- **Phase 1 (online decode pass):** `multiplex_AC` runs **Stage A + Stage C** in a single shared frame loop (decode once).
- **Phase 2 (offline artifact pass):** `D → E → X` runs sequentially (artifact-driven; no multiplex).

### Stage B status (locked)
Stage B (refined masks / SAM) is **DEFERRED for the POC**. C2 must remain fully functional using C1 observations derived from Stage A-only ROIs.

### ROI metadata policy (updated)
C1’s ROI source policy is now:
- primary: `roi_source="bbox_expanded"`
- optional hints: `stage_A_mask_hint` and (future) `stage_B_mask_hint`

C2 must treat ROI source and ROI quality stats as first-class evidence metadata for weighting / tie-breaking.

### Output contract correction (must align with F0)
F0 lists canonical C-stage outputs as:
- `stage_C/tag_observations.jsonl`
- `stage_C/identity_hints.jsonl` (must-link / cannot-link constraints)
- `stage_C/audit.jsonl`

Any additional files (e.g., `identity_registry.jsonl`, `conflicts.jsonl`, `evidence.jsonl`) must be treated as **non-canonical dev artifacts** unless/until F0 is explicitly bumped.


## Addendum — Evidence ROI & Hints Derived From A/B Masks (post-A1R4)

This addendum aligns C2 with the new “mask-first” pipeline reality:
- Stage A provides canonical masks and per-frame geometry for every detection/tracklet.
- Stage B may provide refined masks (sparse), improving tag decode confidence during entanglements.

### C2 evidence policy (locked)
When C2 considers evidence from C1 (tag observations), it should treat the following as first-class metadata:
- `roi_source` (B mask vs A mask vs bbox fallback)
- any ROI quality stats (area, blur score, etc. if logged by C1)

This improves conflict resolution by allowing C2 to down-weight lower-trust evidence (e.g., bbox fallback) without changing decode logic.

### Hint generation remains unchanged (but add clarity)
C2 still outputs registry decisions / voting outcomes as:
- `stage_C/identity_hints.jsonl` (must-link / cannot-link constraints)

But now we explicitly recommend:
- If competing hints exist for the same `tracklet_id` window, prefer hints derived from:
   - `roi_source="bbox_expanded"` (POC default; strongest because it is the primary decode ROI)
   - then `roi_source="stage_A_mask_hint"`
   - then (future) `roi_source="stage_B_mask_hint"` *(Stage B deferred for POC)*
   - and prefer all over `roi_source="bbox_only"` / `bbox_fallback`, **all else equal**.

> Note: if/when Stage B returns, its masks are still treated as **hints** for ROI quality, not as hard crop boundaries; evidence weighting should reflect improved quality metrics, not merely the presence of a refined mask.

### Non-responsibility reaffirmed
C2 must not:
- request or trigger Stage B execution
- alter Stage A tracklets
- rewrite masks

C2 only consumes the artifacts and emits identity constraints/hints.

### Acceptance criteria update
C2 is considered successful when:
- it deterministically produces hints from C1 observations, and
- it records reason codes that include ROI-source sensitivity (e.g., "prefer_refined_mask_evidence"),
- and it remains compatible whether Stage B ran or not.

**POC update:** Success must be demonstrable with Stage A-only observations (`bbox_expanded` ROIs) and without any Stage B artifacts.

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
   - Tooling target (future): SAM/SAM2 offline refinement (or fallback masks) + OpenCV.
   - Output (future): refined masks + sparse overrides where needed.

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
   - Tooling target: AprilTag detection applied inside **expanded bbox ROI** + voting registry.
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

### Tooling defaults
Stage tooling choices are defined in their own workers (see F0 for constraints). C2 focuses on identity voting and conflict resolution.

### Contracts & artifacts
See F0 for authoritative schemas. C2 writes `identity_registry.jsonl`, `conflicts.jsonl`, and `evidence.jsonl`; `hints.jsonl` remain non-binding suggestions to A/B.

> **Correction:** C2’s canonical outputs must remain limited to the F0-locked artifacts (`tag_observations.jsonl`, `identity_hints.jsonl`, `audit.jsonl`). Any additional “registry/conflict/evidence” files are dev-only unless F0 is bumped.

### Definition of done
Follow the standard manager checklist; add deterministic tests for vote aggregation and conflict detection; outputs must validate via F0.


---


## Module-specific context (C2 — identity registry)
Registry converts flickery tag observations into stable identities.

### Must include
- Voting window / decay strategy
- Consensus threshold rules
- Conflict detection rules:
  - if strong votes for two tags, emit `needs_split` for stitcher
- Output format consumed by MCF constraints

### Acceptance tests
- Flicker scenario stabilizes correctly
- Conflicting tags triggers conflict flag


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
Follow F1’s checkpoint rules; ensure identity votes are reproducible, auditable, and artifacts validate.

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

> **POC update:** current target multiplex mode is `multiplex_AC` (A + C). This does not change C2’s artifact-driven logic; it should remain able to run as part of multiplex (in-memory event ingestion) or as a multipass stage (reading `tag_observations.jsonl` from disk).
