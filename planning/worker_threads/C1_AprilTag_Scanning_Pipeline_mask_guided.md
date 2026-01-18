## Addendum — 2026-01-XX (Post-F0G + C0 boundary)

### POC mode & Stage B posture (lock)
- Current target is **`multiplex_AC`** (A + C). Stage B is **DEFERRED** and must not be required.

### Canonical geometry source (multipass clarity)
- In multipass mode, any geometry gating should use **Stage A `stage_A/contact_points.parquet`** (join-ready by `(frame_index, detection_id)`).
- In multiplex mode, geometry may be supplied in-memory; C1 must still be able to run bbox-only.

### Audit ownership (avoid duplication with C0)
- **C0 logs** attempt-vs-skip decisions + scheduler state + skip reasons.
- **C1 logs** per-attempt ROI + decode attempt metadata + outcomes/failure codes.

## Addendum — 2026-01-14 (POC update: C runs online in multiplex_AC; C0 cadence; bbox-expanded ROI; Stage B deferred)

### Hybrid pipeline execution model (locked for POC)
We are using a **single pipeline** with two phases:
- **Phase 1 (online decode pass):** `multiplex_AC` runs **Stage A + Stage C** in a single shared frame loop (decode once).
- **Phase 2 (offline artifact pass):** `D → E → X` runs sequentially (artifact-driven; no multiplex).

### Stage B status (locked)
Stage B (SAM refinement / refined masks) is **DEFERRED for the POC**. C1 must function end-to-end using Stage A outputs only.

### Decode scheduling (new C0 responsibility)
When and how often to attempt decoding is owned by **C0 (Tag Decode Scheduling & Cadence)**:
- aggressive until first successful decode per `tracklet_id`
- backoff after success
- ramp-up on occlusion/ambiguity events (bbox overlap, track instability proxies)

C1 owns the **decoder + evidence emission** (observations + identity hints) and logs per-attempt ROI + decode metadata only when a decode is attempted.

### ROI policy (important correction)
**Primary decode ROI is NOT a tight mask crop.** The primary ROI is an **expanded bbox** derived from Stage A detections:
- Start from Stage A bbox
- Apply configurable padding (and clip to image bounds)
- Optionally use masks as a **soft hint** (e.g., to prioritize pixels or compute “likely tag fits” checks)
- **Never hard-clip the decode region to the YOLO mask**, because YOLO masks frequently under-cover torso/chest/back regions where tags live.

## Addendum — 2026-01-08 (Crop Candidate Contract)

### Input Sources
Stage C runs **online** in multiplex and consumes the already-decoded frame provided by orchestration.
It may optionally consume **curated crop candidates** (if present) for offline reprocessing, but C1 must not depend on crops for the POC.
When running outside multiplex (future multipass), C1 may re-open raw video frames **only for the frames/ROIs it chooses to scan**.

### Hard Evidence Policy
AprilTag detections are treated as hard constraints.
Stage C outputs identity_hints.jsonl only.

Stage C must not perform appearance embeddings or soft matching.
# C1 — AprilTag Scanning Pipeline (mask-guided)

## Addendum — Mask Source Priority (post-A1R4, with optional B overrides)

This addendum locks how C1 chooses the ROI for AprilTag scanning now that:
- **Stage A** emits **canonical masks for every detection** (`stage_A/masks/*.npz`)
- **Stage B** may emit **refined masks** for a sparse set of detections/frames (`stage_B/masks/*.npz`)

### ROI source priority (authoritative)
For each `(frame_index, detection_id)` being scanned:

1) **Expanded bbox ROI (primary; POC default)**  
   - Use bbox from `stage_A/detections.parquet` and apply configurable padding.
   - Rationale: tags live on torso/chest/back; YOLO masks often under-cover these regions.

2) **(Future) Stage B refined mask hint** if present  
   - Use: `stage_B/masks/*.npz` *(Stage B deferred for POC)*
   - Use as a **soft hint** only (e.g., pixel weighting, “tag fits” checks), not as a hard crop boundary.

3) **Stage A canonical mask hint** otherwise  
   - Use: `stage_A/masks/*.npz`
   - Use as a **soft hint** only; do not hard-clip decode ROI to the mask.

4) **Fallback behavior**  
   - If mask loading fails unexpectedly, proceed with bbox-only ROI and log an audit warning with counts.

### Geometry usage (important clarification)
C1 may use Stage A’s geometry for:
- deciding *whether* to scan (e.g., on_mat, proximity heuristics, confidence thresholds)
- optional gating (skip when athlete is off-mat, too small, etc.)

But **C1 must not mutate** Stage A geometry. C1 only emits tag observations and identity hints.

### Determinism & audit requirements (C1)
When scanning, C1 must record per-attempt metadata in `stage_C/audit.jsonl`, including:
- `roi_source`: "bbox_expanded" | "stage_B_mask_hint" | "stage_A_mask_hint" | "bbox_only"
- `roi_px_area`, `roi_bbox` (optional but helpful)
- decode results + failure reason code (e.g., "no_candidates", "low_contrast", "motion_blur")

Additionally, each C1 attempt event should include enough keys to join downstream:
- `clip_id`, `camera_id`
- `frame_index`, `timestamp_ms`
- `detection_id`, `tracklet_id`

> C0 will separately log skip/attempt scheduling decisions; C1 should not duplicate skip logging.

### Acceptance criteria update
C1 is considered successful when:
- it can scan an entire clip deterministically in `multiplex_AC` using **expanded bbox ROIs**, and
- audit logs show ROI-source counts and decode attempt stats (including skip reasons from C0 cadence decisions), and
- (future) when Stage B is reactivated, C1 can incorporate refined masks as optional hints with no manual intervention.

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

### Canonical tool choices (POC defaults)
These are defaults; workers may propose alternatives but must align with constraints.
- **Tracking**: BoxMOT **BoT-SORT** (as tracklet generator)
- **Masks**: YOLO-seg online where possible; **SAM/SAM2 deferred for POC**
- **AprilTags**: OpenCV ArUco (`cv2.aruco`) decoder (current implementation)
- **ReID (optional early, likely later)**: OSNet / torchreid or FastReID; ideally on masked crops
- **Stitching**: **Min-Cost Flow** (OR-Tools min-cost flow or NetworkX as baseline)
- **Video I/O**: OpenCV for reading frames when needed; ffmpeg for export
- **Data**: JSONL for event logs; Parquet for high-volume tables (decide in F0/F2)

### Contracts & artifacts
Authoritative schemas live in F0; C1 consumes `stage_A`/`stage_B` masks and emits `stage_C/tag_observations.jsonl`.

### Definition of done
Follow the standard manager checklist; add unit tests validating ROI selection and decode determinism.


---


## Module-specific context (C1 — AprilTag scanning)
AprilTag scanning should happen throughout the clip (not only during matches).
Use masks/ROIs to reduce compute and false positives.

### Must include
- ROI extraction from **expanded bbox** (mask may be used as a soft hint; never hard-clip decode ROI to mask)
- Preprocessing steps (contrast, grayscale, thresholding) proposed and configurable
- Confidence scoring + false positive handling
- Strategy for picking “best frames” and cadence (from **C0** state machine; A2/B1 are deferred for POC)

### Invariants
- Every tag observation includes: frame_index, tracklet_id/person_id (if known), tag_id, confidence, roi bounds, method


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

## Integration with Z3 Multiplexer Runner

If/when **Z3 (Single-pass Multiplexer Runner)** is implemented, this stage may run in a mode where orchestration provides a shared `FrameIterator` and calls per-frame processors. **Do not change F0 outputs** (Parquet schemas, mask storage, audits) to “bundle” B/C features into Stage A artifacts.

Stage A should remain responsible for:
- detector outputs (bboxes + scores + class)
- tracker association → tracklets

Anything like:
- mask refinement (Stage B),
- AprilTag observation/hints (Stage C),
- ReID embedding banks (D4),
should remain in their respective stages unless a manager-approved F0 schema bump is made.

Debug visualization outputs (if enabled) are **non-canonical** and must not become required artifacts for stage completion.

## Update after Z3 completion (2026-01-07)
Z3 introduced an **optional single-pass multiplex mode** (`multiplex_AC`) that runs **Stages A + C within a shared frame loop** (video decoded once), while preserving:
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
