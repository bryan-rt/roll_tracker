# C0 — Tag Decode Scheduling & Cadence (Online, multiplex-safe)

---

## Addendum — 2026-01-XX (Post-F0G: Stage A canonical contact points; audit ownership; mux mode name)

### Execution mode (POC lock)
- POC online mode is **`multiplex_AC` only** (Stage A + Stage C in a shared frame loop).
- Stage B is **DEFERRED** and **must not be required** for Stage C to run or validate.

### Canonical geometry source (POC lock)
Stage A is the canonical owner of baseline geometry via:
- `stage_A/contact_points.parquet` (join-ready by `(frame_index, detection_id)`; includes `tracklet_id`, `on_mat`, `u_px/v_px`, `x_m/y_m`, etc.)

Implications:
- **Multiplex mode:** C0 may receive some geometry fields in-memory, but must still function with bbox-only inputs.
- **Multipass mode:** Stage C (scheduler/gating) should read **Stage A `contact_points.parquet`** (not Stage B, not tracklet_frames) for optional gating.

### Audit ownership boundary (avoid duplication)
- **C0 audit:** attempt-vs-skip decisions + scheduler state + skip reason codes + summary counters.
- **C1 audit:** ROI bbox/source + decode attempt metadata + decode outcome/failure codes (only when a decode is attempted).

## Status — 2026-01-14 (Manager-locked for POC)

### Hybrid pipeline execution model (POC)
- **Phase 1 (online decode pass):** `multiplex_AC` runs **Stage A + Stage C** in a single shared frame loop (decode once).
- **Phase 2 (offline artifact pass):** `D → E → X` runs sequentially (artifact-driven; no multiplex).

### Scope posture
- **Stage B (SAM/refined masks) is DEFERRED** for the POC.
- Stage C must work end-to-end using Stage A detections/tracklets (+ optional masks as hints).

---

## Why this worker exists

AprilTag decode is expensive and its success is **opportunistic** (angle, motion blur, distance). If we scan everything all the time, we waste compute and increase false positives.

**C0** defines the deterministic, multiplex-safe policy for:
- **When** to attempt decoding
- **How often** to retry per tracklet
- **When to back off** after a successful decode
- **When to ramp up again** after ambiguity/occlusion events

This lets **C1** focus purely on the decoder + evidence emission.

---

## Role clarification (division of responsibility)

### C0 owns
- The **state machine** for per-`tracklet_id` decode cadence
- Candidate selection per frame (which detections/tracklets to pass to C1)
- Deterministic backoff / ramp-up logic
- Audit counters + reason codes for *attempted* vs *skipped* scans

### C0 does NOT own
- AprilTag detection/decoding implementation (C1)
- Identity voting / conflicts / must-link + cannot-link generation (C2)
- Any video IO beyond what multiplex provides (C0 must not open video files)
- Mask refinement (Stage B is deferred)

---

## Inputs (from Stage A + orchestration)

In multiplex mode, C0 receives per-frame inputs from the shared frame loop:
- `frame_index`
- `frame_bgr` (or equivalent) provided by orchestration (shared decode)
- Stage A per-frame detection/track info:
  - `detection_id`, `tracklet_id`
  - bbox (x1,y1,x2,y2) + detection confidence
  - (optional) mask path or in-memory mask reference (hint only)
   - (optional) geometry fields (e.g., on_mat, x_m, y_m, vx, vy) if provided in-memory by orchestration

**Important:** C0 must be able to operate with *only* bbox + tracklet_id + conf if optional fields are missing.

---

## Outputs (F0-aligned)

### Canonical stage outputs remain in Stage C
This worker must not introduce new F0 artifacts by default. Stage C canonical outputs stay:
- `stage_C/tag_observations.jsonl` (C1)
- `stage_C/identity_hints.jsonl` (C2)
- `stage_C/audit.jsonl` (C0+C1+C2 may contribute stage_C audit events)

### Audit events (required)
C0 must write audit lines to `stage_C/audit.jsonl` capturing:
- attempt/skip per `(frame_index, detection_id)` with a compact reason code
- scheduler state per tracklet (e.g., SEEKING / VERIFIED / RAMP_UP)
- per-run counters summary:
  - total_candidates_seen
  - total_decode_attempts
  - total_skips (by reason)
  - attempts_before_first_success (distribution or summary)
  - backoff_state_counts (e.g., SEEKING vs VERIFIED)

> Note: per-attempt ROI/decode metadata lives in C1 audit events to avoid duplication.

---

## Deterministic cadence policy (baseline spec)

### Per-tracklet decode state machine
Maintain a small state per `tracklet_id`:

1) **SEEKING** (no decode yet)
   - Attempt decode aggressively when gating allows.
   - Example cadence: every `k_seek` frames (configurable; default small).

2) **VERIFIED** (decoded at least once)
   - Back off dramatically to periodic re-verify.
   - Example cadence: every `k_verify` frames (configurable; default larger).

3) **RAMP_UP** (after ambiguity/occlusion event)
   - Temporarily increase attempts for `n_ramp` frames or until re-verified.

State transitions:
- SEEKING → VERIFIED: first successful decode for that `tracklet_id`.
- VERIFIED → RAMP_UP: occlusion/ambiguity trigger fires.
- RAMP_UP → VERIFIED: successful decode during ramp OR ramp window expires.

### Gating signals (what C0 may use)
All gating must be **frame-local** (may use short rolling history, no future).

Allowed (POC):
- bbox area threshold (skip tiny bboxes)
- detection confidence threshold
- optional `on_mat` gating (skip off-mat if reliable)
- optional “tag-fit” heuristic (cheap):
  - estimate whether an AprilTag-sized square could fit inside the **expanded bbox ROI**

### Join contract (downstream-friendly; POC lock)
- **Primary unit of work / record:** `(clip_id, frame_index, detection_id)`
- `tracklet_id` is attached for scheduling state and aggregation, but is not the primary join key.
- Any scheduler decisions and audit counters should be attributable to `(frame_index, detection_id)` first.

Occlusion/ambiguity triggers (POC proxies):
- sustained bbox overlap between two tracklets (IoU over threshold for M frames)
- sudden bbox area jumps
- tracklet discontinuity / reset (new tid appears near old position)

**Note:** these are intentionally lightweight; anything requiring hindsight belongs in offline D0 (if/when created).

---

## ROI policy (interface to C1)

For each scheduled attempt, C0 must hand C1:
- `frame_index`, `detection_id`, `tracklet_id`
- **expanded bbox ROI** parameters (padding config)
- optional mask reference as a *hint* only

**Hard rule:** never hard-clip the decode region to the YOLO mask. The decode ROI is bbox-expanded.

---

## Configuration (F2-compatible keys; names are suggestions)

Recommend placing under `config["stage_C"]["c0_scheduler"]`:
- `enabled: bool` (default true in multiplex_AC)
- `k_seek: int` (e.g., 1–3)
- `k_verify: int` (e.g., 15–30)
- `n_ramp: int` (e.g., 20–60)
- `bbox_pad_px: int` or `bbox_pad_frac: float`
- `min_bbox_area_px: int`
- `min_det_conf: float`
- `iou_overlap_thresh: float`
- `iou_overlap_window: int`

All defaults must be deterministic and recorded in Stage C audit/config snapshot.

---

## Acceptance criteria (C0)

1) **Multiplex-safe**
   - Runs in the shared frame loop; does not open video files.
2) **Deterministic**
   - Same inputs + config produce identical attempt/skip decisions and audit counts.
3) **Backoff behavior works**
   - After first decode, attempts reduce markedly; ramp-up re-engages after overlap events.
4) **Audit evidence**
   - `stage_C/audit.jsonl` includes attempt/skip reason codes and summary counters.
5) **Pytest smoke test**
   - Synthetic sequence of bbox tracks triggers SEEKING→VERIFIED and VERIFIED→RAMP_UP transitions deterministically.

---

## Kickoff (first response expected from this worker)

Please begin by producing:
1) A precise proposed cadence + gating policy (with reason codes).
2) A minimal state machine spec and data structures.
3) A test plan + fixtures (synthetic tracklet timeline).

End by asking the Manager to approve the policy before implementation.
