---
layout: page
---

# D0 — Offline Cleanup: Tracklet Repair, Geometry Smoothing & Birth/Death Prep

Status: MANAGER-LOCKED FOR POC  
Pipeline Phase: Offline (post A + C, pre MCF graph construction)

**Owner:** D0 worker  
**Status:** READY (new)  
**Execution class:** OFFLINE (may use full past+future over a clip)  
**Where it runs:** after A/B/C artifacts exist; before D1 MCF graph building

## Purpose (Authoritative)

D0 prepares **clean, physically plausible tracklets** for global identity stitching.
It is the *only* stage allowed to modify per-tracklet geometry prior to MCF.

This stage exists to:

- Remove physically impossible motion artifacts
- Repair short occlusion-induced geometry failures
- Normalize contact point trajectories
- Produce **birth/death–ready tracklet metadata** for MCF

D0 does **not** assign identities and does **not** merge tracklets.

## Inputs (Authoritative)

### Required
- `stage_A/tracklet_frames.parquet`
- `stage_A/tracklet_summaries.parquet`
- `stage_A/contact_points.parquet`  *(canonical geometry source)*

### Optional (Non-blocking)
- `stage_C/identity_hints.jsonl` *(read-only; never alters geometry)*

D0 must run correctly **without** Stage B or ReID.

## Core Responsibilities (POC Scope)

### 1. Geometry Repair (Borrowed & Adapted from *roll it back*)

Borrow the *concept*, not the implementation:

From *roll it back*:
- Detection of sudden world-space jumps caused by lower-body occlusion
- Use of short temporal windows to infer implausible motion

Adaptations for **Roll Tracker**:
- Use **Stage A canonical contact points** (`x_m`, `y_m`)
- Never reproject from bbox bottoms
- Operate **per-tracklet**, never across tracklets
- Repairs must be:
  - local in time
  - reversible
  - explicitly audited

Allowed repairs:
- Short linear interpolation over ≤ N frames
- Velocity clamping using physically plausible limits
- Contact point snap-back when occlusion ends

Forbidden:
- Long-range interpolation
- Identity-aware smoothing
- Cross-tracklet borrowing

All repairs must emit audit events with:
- original vs repaired displacement
- method used
- frame window affected

### 2. Contact Stability & Motion Smoothing (New)

Introduce light, deterministic smoothing to improve MCF cost stability:

- Median filtering on `(x_m, y_m)` within short windows
- Velocity outlier suppression (vx, vy)

Rules:
- Smoothing must never shift start/end frames
- Must preserve monotonic time
- Must not invent new contact points

### 3. Tracklet Validity Scoring (New)

Each tracklet receives metadata used downstream:

- fraction_on_mat
- mean_velocity
- max_velocity
- num_repaired_frames

These are **features**, not decisions.
D0 must not drop tracklets.

### 4. Birth / Death Preparation (Critical for MCF)

D0 is responsible for computing *candidate birth and death zones*.

For each tracklet:
- start_frame
- end_frame
- start_position (x_m, y_m)
- end_position (x_m, y_m)

These feed directly into:
- D1 node construction
- D2 birth/death cost functions

D0 must not decide births/deaths — only prepare evidence.

## Outputs (Canonical for Stage D)

D0 produces **cleaned tracklet banks** consumed by D1–D6:

- `stage_D/tracklet_frames_cleaned.parquet`
- `stage_D/tracklet_summaries_cleaned.parquet`
- `stage_D/audit.jsonl`

No other artifacts are written.

## Explicit Non-Responsibilities

D0 must NOT:

- Merge tracklets
- Assign identities
- Apply AprilTag logic
- Use ReID embeddings
- Read or write Stage A artifacts
- Look at future frames beyond local windows

All global reasoning belongs to D1+.

## Determinism & Auditability (Non-negotiable)

- All operations are deterministic
- No randomness, no learning
- Same inputs → byte-identical outputs

Audit events include:
- repair_applied
- repair_skipped_reason
- smoothing_applied
- smoothing_skipped_reason

## Relationship to Other D Stages

- **D1** consumes cleaned tracklets as immutable nodes
- **D2/D5** rely on accurate birth/death metadata
- **D3** assumes geometry is physically plausible
- **D4** (ReID) may optionally consume cleaned crops later
- **D6** assumes D0 never violates physical constraints

D0 is the *last chance* to fix geometry before optimization.

## Definition of Done (POC)

D0 is complete when:

- Cleaned tracklets pass validation
- Birth/death metadata is present for all tracklets
- Repairs are sparse, auditable, and conservative
- MCF can run without geometry explosions

Any future expansion (long occlusions, multi-contact models)
requires a new manager-approved milestone.
