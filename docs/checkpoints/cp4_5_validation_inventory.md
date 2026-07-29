# CP4.5 — GT-Anchored Per-Person Tracking Diagnostic: Inventory

**Date:** 2026-05-14
**Branch:** `services_uploader`
**Scope:** Read-only investigation of `src/pipeline_validation/` and eval output artifacts.

---

## 1. Code Inventory of `src/pipeline_validation/`

### `common/schemas.py`
Pydantic models. `GTBox` (class_id, cx/cy/w/h, track_id, x1/y1/x2/y2). `GTTrackSequence`
(gt_track_id, camera_id, split, frames: list of {frame_index, person_id, match_status, iou}).
`ExportEntry` (camera, resolution, annotated_range, splits). `ModelManifest` (model config).

### `common/gt_loader.py`
Extracts CVAT GT annotation zips to temp dirs. Parses 6-field YOLO labels
(class cx cy w h track_id) into `GTBox` objects. Strictly filtered by `annotated_range`
from model manifest. Returns `dict[frame_index, list[GTBox]]`.

### `common/matching.py`
Vectorized IoU matrix computation + Hungarian one-to-one matching via
`scipy.optimize.linear_sum_assignment`. Threshold at IoU >= 0.5. Returns list of
`(gt_idx, pred_idx)` pairs. **This is the only GT-to-prediction matching code in the
entire validation framework.**

### `common/manifest.py`
Loads model manifest YAML. Enumerates split frames (train/val) from annotated_range.

### `common/frame_index.py`
Re-exports `frame_index_from_filename` from gt_loader.

### `stage_a/evaluate.py`
**Per-frame GT-to-detection matching.** For each annotated frame: loads GT boxes, loads
pipeline detections from `detections.parquet`, runs Hungarian IOU matching at thresholds
0.5/0.7/0.9. **Persists results to `per_frame_matches.parquet`** with columns:
`model_id, camera_id, split, frame_index, gt_track_id, gt_x1/y1/x2/y2,
pred_detection_id, pred_x1/y1/x2/y2, iou, match_status`.

Key detail: `pred_detection_id` is the Stage A detection_id (e.g., `d000000_7`), and
`gt_track_id` is the CVAT annotation track. But **no `tracklet_id` column** — the join
from detection_id to tracklet_id is not performed here.

Also writes `report.json` (recall, precision, mean_iou per split) and `report.md`.

### `stage_d/evaluate.py`
**Self-contained per-frame GT-to-pipeline matching + identity evaluation.** Does NOT
depend on Stage A eval outputs — re-runs its own Hungarian matching from scratch.

Key functions:
- `_match_gt_to_pipeline()` (line 141): per-frame Hungarian match GT boxes against
  `detections.parquet`. Returns per-GT-box: {gt_track_id, person_id, iou, match_status,
  detection_id}. `person_id` comes from `det_to_person` dict (detection_id → person_id
  from `person_tracks.parquet`).
- `_build_gt_track_sequences()` (line 189): assembles per-GT-track temporal sequences
  from per-frame results. Each frame entry has {frame_index, person_id, match_status, iou}.
  **detection_id is discarded here** — not carried into the sequence.
- `_compute_identity_mapping()` (line 221): most-frequent-vote per GT track → canonical
  person_id, coverage, purity.
- `_classify_switches()` (line 338): walks sequences to find gaps/switches. Classifies
  as detection_failure / tracklet_dropped / sloppy_box / true_switch.

**Persists:**
- `gt_track_sequences.jsonl` — per-GT-track frame-by-frame {frame_index, person_id,
  match_status, iou}. **No detection_id, no tracklet_id.**
- `identity_mapping.json` — canonical_person_id, purity, coverage per GT track.
- `id_switches.jsonl` — classified switch/gap events per GT track.
- `report.json` / `report.md` — aggregate metrics.

### `stage_f/visualize.py`
Match preview video renderer. Four visual layers: all detections (grey), person-assigned
(colored), match envelopes, tag icons. Uses `detections.parquet` + `person_tracks.parquet`
+ `match_sessions.parquet` + `tag_observations.jsonl`. Writes mp4 per camera.

### `cli.py` / `__main__.py`
CLI entry point. Subcommands: `discover`, `stage-a`, `stage-d`, `stage-f`, `evaluate`
(runs all three in sequence). The `evaluate` command optionally runs the full pipeline
first (`--force`), then Stage A eval, Stage D eval, Stage F viz.

### `reports/markdown.py` / `reports/gallery.py`
Report formatting helpers. Not relevant to the per-person trace question.

---

## 2. Eval Output Artifacts (J_EDEw post-CP3b)

### Stage A outputs (`outputs/_eval/stage_a/bjj-detect-all-cameras/J_EDEw/`)

| File | Schema | Rows | Key columns |
|------|--------|------|-------------|
| `per_frame_matches.parquet` | Per-frame GT-to-detection matches | 4,395 | frame_index, gt_track_id, pred_detection_id, iou, match_status, split |
| `report.json` | Aggregate metrics | 1 | recall@0.5, precision@0.5, mean_iou per split |
| `report.md` | Human-readable report | — | — |
| `failures/` | Frame PNGs of failure cases | — | — |

### Stage D outputs (`outputs/_eval/stage_d/bjj-detect-all-cameras/J_EDEw/`)

| File | Schema | Key columns |
|------|--------|-------------|
| `gt_track_sequences.jsonl` | Per-GT-track frame-by-frame | gt_track_id, frames: [{frame_index, person_id, match_status, iou}] |
| `identity_mapping.json` | Per-GT-track canonical mapping | canonical_person_id, purity, coverage per gt_track |
| `id_switches.jsonl` | Classified switch/gap events | gt_track_id, cause, gap_length, person_id_before/after |
| `report.json` / `report.md` | Aggregate identity metrics | — |
| `lowest_purity_strip.jpg` | Visual diagnostic of worst track | — |

### Stage F outputs (`outputs/_eval/stage_f/bjj-detect-all-cameras/`)

| File | Key content |
|------|-------------|
| `{camera}/match_preview.mp4` | 4-layer diagnostic video per camera |

---

## 3. Inventory Questions

### (a) Per-frame table mapping GT person → Stage A tracklet IDs covering them?

**Partial.** `per_frame_matches.parquet` maps GT person (gt_track_id) → Stage A
detection_id (pred_detection_id) per frame via IOU matching. But **detection_id →
tracklet_id is not joined.** The detections.parquet has both `detection_id` and
`tracklet_id` columns, so the join is trivial (one LEFT JOIN), but it's not currently
performed or persisted.

### (b) Per-frame table mapping GT person → D1 segmentation state of their tracklets?

**Absent.** No validation artifact links GT persons to D1 node information (SOLO vs
GROUP, which carrier, segment boundaries). The Stage D evaluator (`stage_d/evaluate.py`)
loads `detections.parquet` and `person_tracks.parquet` but never loads `d1_graph_nodes.parquet`,
`d1_segments.parquet`, or any D1 artifacts.

### (c) Per-frame table mapping GT person → D3 routing outcome of their tracklets?

**Absent.** The Stage D evaluator does not load `d3_solution_ledger.json` or any D3
artifacts. The `tracklet_dropped` classification in `_classify_switches()` is inferred
indirectly: if a GT frame has `match_status == "matched"` (detection exists) but
`person_id is None` (not in person_tracks), it's classified as `tracklet_dropped`.
This inference is correct but doesn't distinguish:
- Tracklet dropped by D3 solver (in `dropped_tracklets` list)
- Tracklet kept by D3 solver but not emitted by D4 (span filtering)
- Tracklet not present in D1 at all (hypothetical pre-D1 filtering)

### (d) Per-frame table mapping GT person → D4 final global_person_id?

**Exists fully.** `gt_track_sequences.jsonl` contains per-GT-track per-frame `person_id`
(the D4 global person ID). This is the final assignment. The Stage D evaluator chains
detection_id → person_id via `person_tracks.parquet` and reports it per frame.

### (e) Closest existing artifact to "for GT person P, what happened to every tracklet across D0-D4"?

**`gt_track_sequences.jsonl`** is the closest, but it only provides the **end state**
(person_id or None per frame). It does not carry the intermediate chain:
detection_id → tracklet_id → D1 node → D3 explained/dropped → D4 person_id.

The intermediate data EXISTS in the pipeline outputs (`detections.parquet` has tracklet_id,
`d1_graph_nodes.parquet` has segmentation, `d3_solution_ledger.json` has dropped/explained),
but the validation framework never joins them together into a single per-GT-person trace.

---

## 4. State Classification

**State B — computed but not fully persisted.**

The critical GT-to-detection matching (step a) IS computed per frame in both Stage A and
Stage D evaluators. The Stage A evaluator even persists it to `per_frame_matches.parquet`
with `pred_detection_id`. But the chain from detection_id → tracklet_id → D1/D3/D4 state
is never joined.

Specifically:
- **(a) GT → tracklet_id**: **one join away.** `per_frame_matches.parquet` has
  `pred_detection_id`; `detections.parquet` has `detection_id + tracklet_id`.
  A single LEFT JOIN on `pred_detection_id = detection_id` adds `tracklet_id`.
- **(b) GT → D1 segmentation**: **two joins away.** After getting tracklet_id, join
  against `d1_graph_nodes.parquet` on `base_tracklet_id` with frame-range overlap.
- **(c) GT → D3 routing**: **one lookup.** After getting tracklet_id, check membership
  in `d3_solution_ledger.json` `explained_tracklets` vs `dropped_tracklets`.
- **(d) GT → D4 person_id**: **already exists** in `gt_track_sequences.jsonl`.

No schema design is needed. The data is all present in existing artifacts. The missing
piece is a query that joins them.

---

## 5. Recommendation

**State B — add a query script. ~100-150 lines, no infrastructure changes.**

Write a standalone script (e.g., `tools/gt_person_trace.py` or a new subcommand in
pipeline_validation) that:

1. Loads `per_frame_matches.parquet` (Stage A eval output) for the GT→detection_id link
2. Joins against `detections.parquet` to get tracklet_id per GT frame
3. Loads `d3_solution_ledger.json` to classify each tracklet as explained/dropped
4. Loads `d1_graph_nodes.parquet` to identify each tracklet's D1 segmentation
   (SOLO/GROUP, carrier, node frame ranges)
5. Loads `person_tracks.parquet` to get the D4 person_id
6. Produces a per-GT-person per-frame trace table with columns:
   `gt_track_id, frame_index, detection_id, tracklet_id, d1_node_type, d1_carrier,
   d3_status (explained/dropped), d4_person_id`
7. Writes the trace to `outputs/_eval/stage_d/{model_id}/{camera}/gt_person_trace.parquet`

Estimated scope:
- 1 new file (~100-150 lines)
- 0 modifications to existing pipeline_validation code
- Can be invoked standalone or integrated as a post-step after `stage-d` eval
- Runtime: seconds (joins on existing parquets, no model inference)

This script would immediately answer the question CP4.5 is asking: for each GT person,
which frames were Stage A misses vs D3 drops vs D4 misattributions? And for D3 drops,
which specific tracklets were dropped and why (via cross-reference to the solver ledger)?

The alternative (State C — a full new evaluation layer) is unnecessary. The data already
exists; we just need to connect it.
