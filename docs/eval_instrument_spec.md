# Evaluation Instrument Specification

**Version:** 1.0
**Frozen:** 2026-05-22 (CP-EVAL-1)
**Module:** `src/pipeline_validation/`

## Overview

The evaluation framework measures pipeline quality through two layers. Layer 1 measures
detection geometry (does the pipeline find each person?). Layer 2 measures identity
survival (does the pipeline assign each person a stable identity?). Layer 2 consumes
Layer 1's output — there is exactly one matcher invocation in the authoritative path.

## Layer 1 — Detection Geometry

**Matcher algorithm:** Hungarian assignment via `scipy.optimize.linear_sum_assignment`
(IoU-maximizing). Implementation: `src/pipeline_validation/common/matching.py`.

**IoU threshold:** 0.5. Predicted-GT pairs with IoU below this threshold are not matched.

**Coordinate space:** Raw distorted pixels at native camera resolution. GT bounding boxes
are denormalized from CVAT normalized coordinates using the manifest's declared resolution.
Pipeline detections (`detections.parquet` columns `x1`, `y1`, `x2`, `y2`) are already in
raw distorted pixel space. No undistortion is applied anywhere in Layer 1.

**Invocation:** `src/pipeline_validation/stage_a/evaluate.py` runs the matcher per frame,
per camera, per split.

**Output artifact:** `per_frame_matches.parquet` at
`outputs/_eval/stage_a/{model_id}/{camera_id}/`.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| model_id | str | Model identifier |
| camera_id | str | Camera identifier |
| split | str | "train" or "val" |
| frame_index | int | Frame number in source video |
| gt_track_id | int | GT person track ID (from CVAT annotation) |
| gt_x1, gt_y1, gt_x2, gt_y2 | float | GT bounding box in raw distorted pixels |
| pred_detection_id | str or null | Matched pipeline detection ID, null if unmatched GT |
| pred_x1, pred_y1, pred_x2, pred_y2 | float or null | Predicted bbox, null if unmatched GT |
| iou | float | IoU of the match (0.0 for unmatched rows) |
| match_status | str | "matched", "unmatched_gt", or "unmatched_pred" |

## Layer 2 — Identity Survival

**Input:** Layer 1's `per_frame_matches.parquet` plus pipeline artifact
`stage_D/person_tracks.parquet`.

**Identity mapping derivation:** Performed inside `gt_person_trace.py`, not by any
external evaluation module. The algorithm:

1. Filter `per_frame_matches.parquet` to matched rows (`pred_detection_id` is not null).
2. Join `pred_detection_id` to `person_tracks.parquet` to obtain `person_id`.
3. Group by `gt_track_id`. For each group, count occurrences of each `person_id`.
4. Select the canonical `person_id` by most-frequent vote. Tiebreak: earliest frame
   index wins (i.e., `min(key=lambda pid: (-count[pid], earliest_frame[pid]))`).
5. GT tracks with zero matched frames or no `person_id` mappings receive
   `canonical_person_id = None`.

This derivation replaces any prior dependency on `identity_mapping.json` from
`stage_d/evaluate.py`.

**Failure mode resolution order (frozen, first match wins):**

| Priority | Mode | Condition |
|----------|------|-----------|
| 1 | missing_canonical | GT track has no canonical person_id (None) |
| 2 | stage_a_no_detection | Frame has no matched prediction (pred_detection_id is null) |
| 3 | stage_a_untracked | Matched detection has no tracklet_id in detections.parquet |
| 4 | d3_dropped | Tracklet was dropped by D3 solver |
| 5 | d4_unassigned | Detection not present in person_tracks (no person_id assigned) |
| 6 | present_misattributed | Detection has person_id(s) but canonical is not among them |
| 7 | present | Detection has canonical person_id — correctly attributed |

The resolution order is a contract. Reordering changes metric values and invalidates
baselines.

**Lite mode (4 failure modes):** When full pipeline artifacts are unavailable, Layer 2
collapses to four modes: `present`, `stage_a_no_detection`, `stage_d_no_person`
(aggregating `stage_a_untracked` + `d3_dropped` + `d4_unassigned`),
`present_misattributed`, and `missing_canonical`. Lite mode derives its identity mapping
from `gt_track_sequences.jsonl` using the same most-frequent-vote algorithm.

**Output artifact:** `gt_person_trace.parquet` at
`outputs/_eval/stage_d/{model_id}/{camera_id}/`.

**Smoke test invariant:** For every GT person, the sum of failure mode counts equals the
total number of frames for that person. This is verified on every trace computation.

## Relationship to `stage_d/evaluate.py`

`stage_d/evaluate.py` is a secondary diagnostic tool. It runs its own Hungarian matcher
and produces `identity_mapping.json`, `gt_track_sequences.jsonl`, and aggregate metrics.
These artifacts are useful for switch classification, purity analysis, and historical
comparison, but they are NOT consumed by the authoritative Layer 2 trace.

If `stage_d/evaluate.py`'s matcher diverges from Layer 1's matcher (different threshold,
coordinate space, or algorithm), its outputs are advisory only. The authoritative identity
mapping is always derived from Layer 1's `per_frame_matches.parquet`.

## Versioning Rule

Changes to any of the following require:
1. Explicit justification documented in this file
2. Version bump (e.g., 1.0 → 1.1 for additive, 2.0 for breaking)
3. Re-baseline of all metrics against all cameras

**Frozen elements:**
- Matcher algorithm (Hungarian, IoU-maximizing)
- IoU threshold (0.5)
- Coordinate space (raw distorted pixels)
- Failure mode definitions (7 full, 5 lite including missing_canonical)
- Failure mode resolution order (priority 1–7 as tabled above)
- Identity mapping derivation algorithm (most-frequent vote, earliest-frame tiebreak)
- Smoke test invariant

**Not frozen (may evolve without version bump):**
- Adding new columns to output artifacts (additive schema changes)
- Adding new summary/rollup artifacts
- Logging and diagnostic output format
