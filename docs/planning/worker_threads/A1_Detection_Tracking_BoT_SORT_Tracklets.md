---
layout: page
---

## Addendum — 2026-01-14 (Hybrid POC alignment: multiplex_AC; Stage B deferred; C online)

### Hybrid pipeline execution model (locked for POC)
We are using a **single pipeline** with two phases:
- **Phase 1 (online decode pass):** `multiplex_AC` runs **Stage A + Stage C** in a single shared frame loop (decode once).
- **Phase 2 (offline artifact pass):** `D → E → X` runs sequentially (artifact-driven; no multiplex).

### Stage B status (locked)
Stage B (SAM refinement / sparse overrides) is **DEFERRED for the POC**. Do not plan A1 deliverables or invariants around Stage B being present.

### Stage C status (locked)
Stage C runs **online** in the multiplex pass with Stage A and should use **expanded bbox ROIs** as the primary decode region (masks are optional hints only; never hard-clip decode ROIs).

## Addendum — 2026-01-10 (Post-Z3 + D7 + A/B Scope Lock)

### Locked scope clarification (Manager decision)
Stage A is the canonical owner of baseline geometry for every detection / tracklet-frame:
- contact point estimation in pixel space (u,v) using bbox or YOLO mask
- homography projection to mat-space (x_m, y_m) using configs/cameras/<camera_id>/homography.json (D7-produced)
- on_mat classification using mat_blueprint.json

Stage A must NOT implement calibration or validation logic.
Homography presence/validity is a hard precondition enforced by orchestration preflight (D7/F1).

### F0 contract note (schema bump required)
This scope implies an F0 contract update:
- NEW canonical artifact: stage_A/contact_points.parquet (baseline geometry; full coverage)

Stage A must continue to emit the locked Stage A artifacts:
- stage_A/detections.parquet
- stage_A/tracklet_frames.parquet
- stage_A/tracklet_summaries.parquet
- stage_A/audit.jsonl
...and in addition emit stage_A/contact_points.parquet after the F0 bump lands.

> **Repo reality note (Jan 2026):** Baseline contact point + homography fields are carried in `stage_A/tracklet_frames.parquet`, and Stage A also writes `stage_A/contact_points.parquet` (baseline/full coverage). Stage B emits `stage_B/contact_points_refined.parquet` only when refinement runs.

### Multiplex parity requirement (Z3)
Stage A logic must be callable in both:
- multipass mode (stage reads raw clip, writes artifacts)
- multiplex mode (stage participates in a shared frame iterator; **POC target is multiplex_AC**)

Canonical outputs MUST remain identical across both modes.

### Quality posture (unchanged)
Tracklets should be high precision and allowed to break rather than guess through entanglement.
## Addendum — 2026-01-08 (Post-D7 / Z3 Alignment)

### Locked Preconditions
Stage A assumes a valid homography exists for the camera.
Pipeline orchestration (F1 + D7) guarantees homography preflight
before Stage A executes. Stage A must not implement calibration,
fallback, or validation logic.

### Scope Clarification
Stage A owns:
- YOLO person detection (boxes + confidence)
- MOT association (BoT-SORT)
- Optional lightweight YOLO segmentation masks
- Per-frame foot / ground-contact estimation
- Homography projection of contact point → mat-space (X, Y)
- on_mat classification using mat blueprint

Stage A must not:
- Run SAM or any expensive mask refinement
- Decode AprilTags
- Compute ReID embeddings

### Crop Candidate Emission
Stage A may emit curated crop candidates for downstream consumers
(C: AprilTag, D: ReID). Crops must be selective (best-K / keyframes),
not per-frame exhaustive.

If emitted, Stage A writes:
- image files under stage_A/crops/
- crop_candidates.parquet index with metadata only

Stage A outputs are immutable and must never be overwritten by later stages.
# A1 — Detection & Tracking (BoT-SORT Tracklets)

---

## Completion Summary — A1R4

**Status:** COMPLETE (A1R4 accepted)

This worker is now production-ready and unblocks A2/B/C/D work.

### What A1 Now Owns (as-implemented)
Stage A is the **canonical owner** of per-frame *online* signals for every detection / tracklet, including:

- Detection + tracking (YOLO + BoT-SORT) with deterministic `detection_id` and stable `tracklet_id`
- **Canonical masks** for every detection (YOLO-seg when passing gates; bbox fallback otherwise)
- **Contact point extraction** (mask-first with bbox fallback) with confidence
- **Homography projection** px→meters and per-frame velocities
- **Mat inclusion** (`on_mat`) using `mat_blueprint.json` polygon

Stage A runs in **multiplex_AC** (single-pass frame loop with Stage C) and writes canonical artifacts at end-of-run, preserving determinism and F0 validation.
**POC lock:** current target is `multiplex_AC` (A + C). Stage B is deferred.

### Canonical Outputs (authoritative)
- `stage_A/detections.parquet`
- `stage_A/tracklet_frames.parquet` (now includes geometry/velocity/on_mat fields per the v0.3.0 schema)
- `stage_A/tracklet_summaries.parquet` (intentionally thin)
- `stage_A/masks/*.npz` (canonical mask outputs)
- `stage_A/audit.jsonl`

Dev visualization remains orchestration-owned under `outputs/<clip_id>/_debug/`.

### Stage B Relationship (updated)
Stage B is **DEFERRED for the POC**. The system should achieve an end-to-end A + C → D proof-of-concept without relying on SAM refinement. If/when B is reactivated, it should remain selective and emit sparse overrides; it must not duplicate Stage A’s full-frame pass.

### Accepted Non-Blocking Deferrals
- Tracklet summaries remain minimal by design
- Zero velocities can be real (stationary athletes)
- Global identity stitching remains Stage D

### PM Sign-off
Stage A is complete and production-ready; artifacts validate and dev videos confirm behavior. Proceed to A2/B/C/D.

## Completion Report — A1R4 (Stage A Final Wrap-Up)

**Status:** ✅ COMPLETE / healthy (per A1R4 validation runs)

### Executive summary
Stage A is functionally complete and behaving as intended across its core responsibilities:

- Detection + segmentation (YOLO)  
- Mask gating + deterministic bbox fallback  
- Local tracking / tracklet generation (BoT-SORT)  
- Contact point extraction  
- Homography projection to metric space  
- Mat inclusion testing (`on_mat`)  
- Canonical artifact emission (NPZ + parquet + audit)  
- Debug visualization outputs (annotated + mat-view)

No blocking issues remain. Remaining gaps are explicitly accepted deferrals or non-core enhancements.

### 1) Detection & segmentation
**Result:** ✅ Complete  
**What’s implemented**
- YOLO detection + segmentation are operational when `use_seg=true`.
- Segmentation is attempted whenever enabled; masks are resized to full-frame resolution before leaving the detector.
- Deterministic `detection_id` assignment per frame.

**Evidence to reference**
- `outputs/<clip_id>/stage_A/audit.jsonl`: `detector_use_seg=true`, `detector_seg_model_loaded=true`, `mask_shapes_sample=[[H,W], …]`
- `outputs/<clip_id>/_debug/annotated.mp4`: visible segmentation overlays aligned to people

### 2) Mask gating & fallback behavior
**Result:** ✅ Complete (matches intended semantics)  
**What’s implemented**
- Mask quality gating uses:
  - area fraction thresholds  
  - detection confidence  
- If a mask is rejected, a deterministic bbox-fallback mask is substituted.
- Downstream stages always receive a valid mask (no “missing-mask” holes).

**Evidence to reference**
- `stage_A/audit.jsonl`: `mask_source` (expected `"yolo_seg"` in normal conditions), `mask_quality≈0.93–0.96` in recent runs  
- Parquet outputs + annotated video show no unexpected fallback usage

### 3) Tracking (BoT-SORT tracklets)
**Result:** ✅ Complete  
**What’s implemented**
- Tracklet IDs (`tid`) remain stable across frames.
- Identity continuity is preserved across typical motion/occlusion patterns.
- Tracker receives the required per-frame inputs (no `NoneType` wiring failures).

**Evidence to reference**
- `outputs/<clip_id>/_debug/annotated.mp4`: consistent `tid` labels per person
- `stage_A/tracklet_frames.parquet`: consistent `tracklet_id` reuse across frames

### 4) Contact point extraction
**Result:** ✅ Complete  
**What’s implemented**
- Contact points computed from segmentation mask when available; bbox fallback otherwise.
- Contact confidence is recorded; pixel-space `(u_px, v_px)` is stable.

**Evidence to reference**
- `stage_A/tracklet_frames.parquet`: `contact_method="yolo_mask"` when segmentation is available; non-zero smoothly varying `(u_px, v_px)`

### 5) Homography projection
**Result:** ✅ Complete  
**What’s implemented**
- Homography is correctly loaded, validated, and applied.
- Pixel coordinates project to sane metric-space `(x_m, y_m)` without scale explosions or drift.
- Direction of homography had to be inversed for proper coordinate space conversion.

**Evidence to reference**
- `stage_A/audit.jsonl`: `homography_converted=true`
- `stage_A/tracklet_frames.parquet`: stable plausible `(x_m, y_m)` values; velocity magnitudes now reflect realistic motion

### 6) Mat inclusion (`on_mat`)
**Result:** ✅ Complete  
**What’s implemented**
- `configs/mat_blueprint.json` is loaded correctly (including list-format blueprints).
- `point_in_mat()` correctly evaluates inclusion.
- `on_mat` aligns with visual position in debug video.

**Evidence to reference**
- `stage_A/audit.jsonl`: blueprint path and blueprint type (expected `list`)
- `stage_A/tracklet_frames.parquet`: `on_mat=true/false` consistent with position

### 7) Canonical artifact outputs
**Result:** ✅ Complete  
**What’s emitted**
- **Masks:** `outputs/<clip_id>/stage_A/masks/*.npz`  
  - If `mask_source="yolo_seg"`, NPZ contains YOLO segmentation mask  
  - If fallback occurs, NPZ contains deterministic bbox mask
- **Tables:**  
  - `detections.parquet`: boxes + confidences (+ mask refs / geometry as defined by F0)  
  - `tracklet_frames.parquet`: geometry, velocities, `on_mat`, contact point fields  
  - `tracklet_summaries.parquet`: minimal but consistent and validator-safe
- **Audit:** `stage_A/audit.jsonl` emits key configuration + runtime events.

### 8) Visualization (non-blocking, now complete)
**Result:** ✅ Complete  
- `outputs/<clip_id>/_debug/annotated.mp4`: bbox, mask, `tid`, confidence overlays  
- `outputs/<clip_id>/_debug/mat_view.mp4`: plots `(x_m, y_m)` points and trails

### Accepted deferrals (explicitly non-blocking)
- **Tracklet summaries are “thin”**: only spans + counts; richer quality scoring/aggregates deferred intentionally.  
- **Zero velocities at times**: expected due to stationary athletes; confirmed as plausible.

### Final verdict / PM-ready sign-off
Stage A (A1R4) is **DONE**. All functional goals are met; outputs are deterministic, interpretable, and downstream-safe. Proceed to Stage B / Stage D without revisiting Stage A fundamentals, unless adding optional enhancements (summary richness, additional quality scoring).


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
  - Tooling target (future): SAM/SAM2 refinement (or fallback masks).
  - Output (future): refined masks + sparse overrides where needed.

3) **Stage C — Identity anchoring (AprilTag scanning + registry)**
  - Tooling target: AprilTag detection applied inside **expanded bbox ROI** (mask may be used as a soft hint).
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
- **AprilTags**: OpenCV ArUco (`cv2.aruco`) decoder inside expanded bbox ROI (current implementation)
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


## Module-specific context (A1 — detection/tracking tracklets)
BoT-SORT is used as a **tracklet generator**. We want high precision and we allow breaks under ambiguity.

### Must include
- Detector choice assumptions (we can start with boxes; seg optional)
- Tracker parameters tuned for:
  - shorter max_age/track_buffer
  - stricter association threshold
  - export *every frame* record of tracklet_id + bbox + conf
- Output artifacts:
  - detections table
  - tracklets summary (start/end frames, stats)

### Invariants
- Tracklet IDs are stable within the clip run
- Every tracklet frame has a detection row


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
