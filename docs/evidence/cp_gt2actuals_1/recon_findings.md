# CP-GT2ACTUALS-1: Reconnaissance Findings

**Date:** 2026-06-10
**Purpose:** Gate the CP-2 schema design by confirming per-frame signal availability,
GT cadence, join extractability, and reuse surface. No code, no schema.

---

## 1. GT Cadence — Dense or Stride-10?

### Answer: Both available. Dense manifest exists and is loadable for both J_EDEw clips.

**Standard manifest** (`configs/models/bjj-detect-all-cameras.yaml`):
- FP7oJQ: stride=1, frames 0-300 (301 frames) — already dense
- J_EDEw vid1: stride=10, frames 0-3000 (301 frames)
- PPDmUg: stride=10, frames 0-2990 (300 frames)

**Dense manifest** (`configs/models/bjj-detect-all-cameras-dense.yaml`):
- FP7oJQ: stride=1, frames 0-300 (301 frames) — unchanged
- J_EDEw vid1: **stride=1**, frames 0-3000 (**3,001 frames**)
- J_EDEw vid2 (200246): **stride=1**, frames 0-4490 (**4,491 frames**, train-only, no val)
- PPDmUg: stride=10, frames 0-2990 (300 frames) — unchanged

The GT loader (`common/gt_loader.py:8-16`) enforces the correctness contract: it ONLY
loads labels for frames in `annotated_range × split`. The dense manifest's stride=1 causes
`enumerate_annotated_frames` (`manifest.py:34-37`) to emit every integer in
`range(start, stop+1, 1)`, loading all CVAT-interpolated labels.

**Which cadence the layer defaults to:** There is no "default" — the manifest is selected
by `--model` CLI argument. The standard `bjj-detect-all-cameras` gives stride-10 for
J_EDEw; the dense manifest (`bjj-detect-all-cameras-dense.yaml`) must be explicitly
selected. However, the dense manifest shares the same `model_id: bjj-detect-all-cameras`
and `weights_path`, so it needs a distinct file path or model_id override to avoid
ambiguity.

**CP-GT2ACTUALS design implication:** The layer should accept a manifest path explicitly
(not just model_id) so it can use the dense manifest. Alternatively, it can hardcode or
default to the dense manifest path for J_EDEw clips.

**Dense-adequacy assumption confirmed:** CVAT interpolated boxes track the correct person
across interpolated frames (they're linear interpolations between hand-labeled keyframes
within the same track_id). This is adequate for identity labeling, coordinate/color trends,
and jump localization. Sub-pixel position accuracy is not required — we read trends, not
exact positions.

**Residual gap:** PPDmUg has NO dense manifest (stride=10 in both). This limits PPDmUg
to 300 frames. If dense PPDmUg GT is needed, a new dense manifest entry would be required
(assuming the CVAT zip contains interpolated labels, which it likely does).

---

## 2. Per-Frame Signal Availability

### 2a. GT box + matched detection — PER ANNOTATED FRAME

**Grain:** One GT box per (frame_index, gt_track_id) from the CVAT zip.
**Matching:** Greedy matcher (`signal_trace/greedy_matcher.py:26-52`) is many-to-one
(IoU >= 0.3). Each GT box independently claims its best detection. Multiple GT can match
the same detection (pair_box signature).
**Coverage:** Every annotated frame has GT boxes loaded; matching produces results for all.
**Join key:** (frame_index) for GT→detection matching.

### 2b. Pipeline tracklet_id, person_id, node_type, GROUP roles — PER DETECTION PER FRAME

**Source files:**
- `detections.parquet` (Stage A): one row per detection per frame. Columns include
  `detection_id`, `tracklet_id`, `frame_index`, `clip_id`, `camera_id`.
  (`outputs.py:196-197`)
- `person_tracks.parquet` (Stage D): maps `detection_id` → `person_id`.
  (`gt_person_trace.py:87,231`)
- `d1_graph_nodes.parquet` (Stage D): D1 node info with `node_type`, `start_frame`,
  `end_frame`, `base_tracklet_id`, `carrier_tracklet_id`, `disappearing_tracklet_id`,
  `new_tracklet_id`. (`gt_person_trace.py:163-198`)

**Grain:** Per detection per frame. Every detection in every frame has a tracklet_id.
Person_id coverage depends on whether the tracklet survives D3 (dropped tracklets have
no person_id). D1 node info is per (tracklet_id, frame_index) range.

### 2c. World coordinates (x_m, y_m) + is_repaired — PER DETECTION PER FRAME

**Source:** `tracklet_bank_frames.parquet` (Stage D, D0 output).
Created from `tracklet_frames.parquet` (Stage A) by `d0_bank.py:622-666`.

**Exact columns:**
- `x_m`, `y_m` — original world coords from Stage A projection
- `x_m_repaired`, `y_m_repaired` — CP2 occlusion-repaired coords (added at `d0_bank.py:738-739`)
- `is_repaired` — boolean flag (`d0_bank.py:742-743`, `d0_bank.py:874`)
- `repair_method` — string describing repair method (`d0_bank.py:744`)

**Repaired-resolution rule** (from `d0_bank.py:535-541`):
```python
x_eff = x_m_repaired where is_repaired == True, else x_m
y_eff = y_m_repaired where is_repaired == True, else y_m
```
CP-2 should compute an `x_m_eff` / `y_m_eff` using this same rule, or simply read both
columns and let analysis code resolve per-row.

**Grain:** One row per (tracklet_id, frame_index, detection_id) — same as
`tracklet_frames.parquet`. Per-detection per-frame.

### 2d. Velocity — PER DETECTION PER FRAME (with NaN at tracklet boundaries)

**Exact columns** (added by `_apply_cp3_kinematics`, `d0_bank.py:479-605`):
- `vx_mps_k` — x velocity (m/s)
- `vy_mps_k` — y velocity (m/s)
- `speed_mps_k` — speed magnitude (m/s)
- `accel_mps2_k` — acceleration (m/s^2)
- `speed_is_implausible` — boolean flag (speed > v_max)
- `accel_is_implausible` — boolean flag (accel > a_max)

**Grain:** Same as bank_frames — per (tracklet_id, frame_index). Velocity is NaN for the
first frame of each tracklet (no prior frame to diff against). Also NaN where either
endpoint has non-finite world coords.

**Source:** `tracklet_bank_frames.parquet`, same file as world coords.

### 2e. Histogram + is_isolated — SPARSE (histograms only on isolated detections)

**THIS IS THE SPARSE SIGNAL. Characterized below.**

**Source:** `color_histograms.parquet` (Stage A, `outputs.py:530-538`).

**Row coverage:** One row per detection per frame — `append_histogram_row` is called for
EVERY detection (`processor.py:506-512`), regardless of isolation status.

**Column coverage:**
- `frame_index`, `track_id` (= tracklet_id), `is_isolated`, `crop_method` — present
  for every row.
- `hist_0` through `hist_863` (864 columns) — **NaN for non-isolated detections**.

**is_isolated determination** (`isolation.py:1-7,39`): Per-detection, per-frame flag.
Four heuristics must all pass (bbox non-overlap, etc.). This is computed for every
detection in every frame and stored in both `keypoints.parquet` (`outputs.py:275`) and
`color_histograms.parquet` (`outputs.py:303`).

**Sparsity characterization:**
- `is_isolated` is present per-frame for EVERY detection — no holes.
- Histogram VALUES are NaN for non-isolated frames (`histogram.py:120-121`: `if not
  is_isolated: return None, "not_isolated"`).
- From CP-SPLIT-VALIDATE evidence: histogram coverage is typically 40-60% of tracklet
  frames (varies by camera/clip). Grappling pairs are non-isolated by definition, so
  pair-box frames (23.1% of GT-person-frames) always have NaN histograms.

**Tracklet-level summary:** `tracklet_histogram_summaries.parquet` (`outputs.py:554-579`)
is per-TRACKLET (not per-frame). One row per tracklet_id with averaged histogram across
isolated frames. Written only for tracklets with at least one isolated frame with a valid
histogram.

**CP-2 schema implication:** The `is_isolated` column is per-frame (no holes). The
histogram columns have per-frame PRESENCE but NaN VALUES for non-isolated frames. The
schema should document this as a per-column resolution limit, not a row-level gap. The
error map will have `is_isolated` for every row, and histogram columns will be NaN where
`is_isolated == False`.

### 2f. Stage C tags — SPARSE EVENTS (0.02-0.23% of frames)

**Source:** `tag_observations.jsonl` (Stage C, `f0_validate.py:588-596`).

**Schema fields:** `frame_index`, `timestamp_ms`, `detection_id`, `tag_id`, `tag_family`,
`confidence`, `roi_method`, plus base metadata (`schema_version`, `artifact_type`,
`clip_id`, `camera_id`, `pipeline_version`, `created_at_ms`).

**Grain:** One record per tag observation event. Extremely sparse — 0-3 observations
per clip across ~4,500 frames (CP-TAG-2: 0.022-0.067% detection rate).

**Join key:** `(frame_index, detection_id)` — joinable to detections.parquet. The
detection_id links to a specific tracklet_id.

**identity_hints.jsonl** (Stage C → D2): One record per identity hint after C2 voting.
Contains `tracklet_id`, `constraint` (must_link/cannot_link), `anchor_key`, `confidence`,
`evidence`. Grain: per-tracklet sparse event.

**CP-2 schema implication:** Tag signals are sparse events. The error map should have a
`has_tag_observation` boolean column (per-frame), and optionally a `tag_id` column (NULL
for most frames). These are event-level joins, not uniform per-frame signals.

---

## 3. Carried-GT-Identity SET per Node per Frame — Extractable?

### Answer: Partially exists. `_build_d1_lookup` provides the node→tracklet mapping, but NOT the GT identity set directly. CP-2 must derive the GT identity set itself.

**What exists** (`gt_person_trace.py:163-198`):
- `_build_d1_lookup` returns `dict[(tracklet_id, frame_idx), list[node_info]]`
- Each `node_info` contains: `node_id`, `node_type`, `segment_type`,
  `carrier_tracklet_id`
- GROUP nodes are indexed by ALL four roles: base, carrier, disappearing, new tracklet_ids
  (lines 180-188)
- This means: given a (tracklet_id, frame_idx), you can find which D1 nodes it
  participates in and what role it plays

**What CP-2 must derive:**
The GT identity set per D1 node per frame requires composing two joins:
1. **GT → detection:** greedy matcher (GT box → detection → tracklet_id)
2. **tracklet_id → D1 node:** `_build_d1_lookup`

The composition gives: for each D1 node at each frame, which GT people have tracklets
participating in this node? This is the set needed for `group_ambiguous` /
`group_membership_drift` states.

`gt_person_trace.py` already does a version of this in `_compute_full_trace`
(lines 398-481): it walks per_frame_matches rows, looks up d1_info for each
(tracklet_id, frame_idx), and checks person_ids. But it does this per-GT-person
(one row per (frame, gt_track_id)), not per-D1-node.

**CP-2 approach:** Invert the existing trace. Instead of "for each GT person, which node
are they in?", compute "for each node at each frame, which GT people's tracklets
participate?". This is a GROUP BY on the existing trace keyed by (node_id, frame_idx).
The raw data is all present; CP-2 just needs to pivot.

---

## 4. Reusable Join Primitives — Import Surface for CP-2

### From `src/pipeline_validation/common/`:

| Module | Function/Class | What it provides | CP-2 reuse? |
|--------|---------------|-----------------|-------------|
| `manifest.py` | `load_manifest(path)` | Load + validate YAML manifest → `ModelManifest` | YES — load dense manifest |
| `manifest.py` | `enumerate_split_frames(export, split)` | Frame list for train/val split | YES — enumerate GT frames |
| `manifest.py` | `enumerate_annotated_frames(export)` | All annotated frames (train+val) | YES — dense frame enumeration |
| `gt_loader.py` | `extract_zip(zip_path)` | Context manager: CVAT zip → temp dir | YES |
| `gt_loader.py` | `parse_label_file(path, resolution)` | YOLO 6-field → `list[GTBox]` | YES |
| `gt_loader.py` | `frame_index_from_filename(path)` | Extract frame index from label filename | YES |
| `matching.py` | `iou_matrix(gt, pred)` | Vectorized IoU matrix | YES (if Hungarian needed) |
| `matching.py` | `hungarian_match(cost_matrix, threshold)` | 1:1 Hungarian matcher (frozen instrument, IoU 0.5) | Maybe — frozen instrument uses this |
| `schemas.py` | `GTBox`, `ExportEntry`, `ModelManifest` | Pydantic schemas | YES |

### From `src/pipeline_validation/signal_trace/`:

| Module | Function | What it provides | CP-2 reuse? |
|--------|---------|-----------------|-------------|
| `greedy_matcher.py` | `greedy_match(gt_boxes, det_boxes, iou_threshold)` | Many-to-one greedy matching (IoU >= 0.3) | YES — primary matcher for error map |

### From `src/pipeline_validation/gt_person_trace.py`:

| Function | What it provides | CP-2 reuse? |
|---------|-----------------|-------------|
| `_derive_identity_mapping(pfm, pt)` | gt_track_id → canonical_person_id | YES — identity ground truth |
| `_build_d1_lookup(nodes_df)` | (tracklet_id, frame_idx) → [node_info] | YES — D1 node resolution |
| `_build_d3_status(ledger_path)` | tracklet_id → 'explained'/'dropped' | YES — D3 status |
| `_build_person_ids_for_detection(pt_df)` | detection_id → set[person_id] | YES — person_id lookup |

### What CP-2 must BUILD (not importable):

1. **Dense GT frame iteration + detection loading** — load GT at every frame in the dense
   manifest, load detections.parquet, run greedy_match per frame.
2. **Bank-frame join** — join matched detections to `tracklet_bank_frames.parquet` by
   (tracklet_id, frame_index) for world coords + velocity.
3. **Histogram join** — join to `color_histograms.parquet` by (frame_index, track_id) for
   histogram values + is_isolated.
4. **Tag event join** — load `tag_observations.jsonl`, join by (frame_index, detection_id).
5. **D1 node GT-identity-set computation** — invert the trace to get per-node per-frame
   GT identity sets (GROUP BY on greedy match results keyed by node_id).
6. **State classification** — the new state column logic (correct, wrong_id,
   group_ambiguous, group_membership_drift, etc.).

---

## Summary Table: Per-Signal Resolution

| Signal | Grain | Per-frame? | Sparse? | Join key |
|--------|-------|-----------|---------|----------|
| GT box | per (frame, gt_track_id) | YES (at annotated frames) | No | frame_index |
| Detection box | per (frame, detection_id) | YES | No | frame_index, detection_id |
| tracklet_id | per detection | YES | No | detection_id |
| person_id | per detection | YES (if survived D3) | Partial (dropped = NULL) | detection_id |
| D1 node_type + roles | per (tracklet_id, frame range) | YES (within node span) | No | (tracklet_id, frame_index) |
| x_m, y_m | per (tracklet_id, frame) | YES | No | (tracklet_id, frame_index) |
| x_m_repaired, is_repaired | per (tracklet_id, frame) | YES | No | (tracklet_id, frame_index) |
| vx/vy/speed_mps_k | per (tracklet_id, frame) | YES (NaN at boundaries) | NaN at tracklet start | (tracklet_id, frame_index) |
| **is_isolated** | per detection | **YES** | **No** | (frame_index, track_id) |
| **histogram values** | per detection | **ROW exists, VALUES NaN when !isolated** | **~40-60% have values** | (frame_index, track_id) |
| tag_observation | per event | **SPARSE** (~0.02-0.07%) | **YES** | (frame_index, detection_id) |
| identity_hint | per event | **SPARSE** | **YES** | tracklet_id |

**Schema-gating conclusion:** All signals are joinable. Two signals have documented
sparsity that the schema must respect:
1. **Histogram values** — present as columns but NaN for non-isolated frames. The
   `is_isolated` flag is always available to indicate validity.
2. **Tag observations** — extremely sparse events (<0.1% of frames). Should be modeled
   as a boolean flag + optional tag_id, not as a dense column.

No signal has a resolution that blocks the dense error map design. The schema can proceed.
