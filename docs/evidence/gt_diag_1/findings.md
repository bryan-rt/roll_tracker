# GT-DIAG-1: GT-to-pipeline sequence diagnostic

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**GT:** 8 tracks, 12,468 boxes, stride-1 dense
**Pipeline state:** `variable_dt: true` (VDT-DEFAULT-1), post-Piece-9
**Source:** Extends `gt_person_trace.parquet` (CP6, Hungarian IoU 0.5, CP-EVAL-1 frozen)
**Module:** `src/pipeline_validation/gt_sequence_diag/`

---

## 1. Per-GT-track summary

| GT | On-mat | In-quad % | Matched | Clip% | Presence% | Low-conf | Median box area | Segments | Tracklets | Persons | Group segs |
|----|--------|-----------|---------|-------|-----------|----------|-----------------|----------|-----------|---------|------------|
| 0  | ON MAT | 97.1% | 1,705 | 96.7% | 96.6% | | 60,959 | 91 | 10 | 6 | 31 |
| 1  | ON MAT | 3.8% | 1,624 | 92.1% | 92.0% | | 39,452 | 84 | 7 | 5 | 7 |
| 2  | ON MAT | 100.0% | 985 | 55.8% | 55.8% | | 14,270 | 196 | 15 | 7 | 39 |
| 3  | ON MAT | 80.7% | 1,178 | 66.8% | 66.7% | | 13,561 | 159 | 18 | 9 | 27 |
| 4  | ON MAT | 99.8% | 819 | 46.4% | 46.4% | * | 43,401 | 173 | 10 | 5 | 16 |
| 5  | ON MAT | 100.0% | 1,321 | 74.9% | 74.8% | | 5,145 | 143 | 8 | 8 | 6 |
| 6  | ON MAT | 100.0% | 68 | 3.9% | 3.9% | * | 5,060 | 47 | 4 | 4 | 6 |
| 7  | ON MAT | 0.0% | 72 | 4.1% | 60.0% | | 30,457 | 8 | 1 | 2 | 0 |

**Mat classification (GT-VERIFY-1):** All 8 tracks are on the mat blueprint (x 42–58,
y 34–58). Criterion: >= 50% of projected contact points (`contact_point_from_bbox` +
`project_to_world()` with production H/K/D) within blueprint bounds. In-quad % shows the
fraction inside the calibrated quad (x 51–57, y 34–56) — positions outside the quad are
homography extrapolations (less reliable but not off-mat). GT 1 (3.8%) and GT 7 (0.0%)
are outside the quad; their world positions are extrapolated. All detections used
`bbox_fallback` (zero masks); GT and tracklets share identical contact-point code.

Coverage floor: 50% of presence frames. Flagged (*): GT 4 (46.4%), GT 6 (3.9%).
GT 7 is 60.0% of its 120-frame presence but only 4.1% of the clip — both figures shown.

**d1_node_ids cardinality:** 0 rows have >1 node. Every matched frame maps to exactly one D1
node or none. The JSON list in gt_person_trace is always length 0 or 1 on this clip. Single
d1_node_id is used throughout; no multi-node data was discarded.

---

## 2. Median box area — far-field hypothesis

| GT | Median box area (px^2) | Recall | Notes |
|----|----------------------|--------|-------|
| 0 | 60,959 | 96.7% | Largest — close to camera, high recall |
| 4 | 43,401 | 46.4% | Large but under-detected |
| 1 | 39,452 | 92.1% | Near quad edge (extrapolated), high recall |
| 7 | 30,457 | 60.0% | Brief, outside quad (extrapolated) |
| 2 | 14,270 | 55.8% | Moderate |
| 3 | 13,561 | 66.8% | Moderate |
| **5** | **5,145** | **74.9%** | Small — but reasonable recall |
| **6** | **5,060** | **3.9%** | Small — nearly invisible |

GT 6 and GT 5 have nearly identical median box area (5,060 vs 5,145 px^2), yet recall is 3.9%
vs 74.9%. **Box area alone does not explain GT 6's invisibility.** GT 6 is not simply far from
the camera — it is a similar size to GT 5 but the detector almost never fires. This points to
occlusion, pose, or background confusion rather than pure distance.

GT 4 is an anomaly: large boxes (43,401 px^2, 3rd largest) with low recall (46.4%). Large
and under-detected — likely grappling-induced under-segmentation (pair-box problem).

---

## 3. Tracklet purity

| GT | Segments w/purity | Mean purity | Pure (>=0.9) | Impure (<0.5) |
|----|-------------------|-------------|--------------|---------------|
| 0 | 62 | 0.748 | 17 (27%) | 4 (6%) |
| 1 | 49 | 1.000 | 49 (100%) | 0 |
| 2 | 124 | 0.689 | 58 (47%) | 41 (33%) |
| 3 | 97 | 0.620 | 38 (39%) | 41 (42%) |
| 4 | 89 | 0.723 | 47 (53%) | 32 (36%) |
| 5 | 77 | 0.773 | 34 (44%) | 1 (1%) |
| 6 | 23 | 0.108 | 1 (4%) | 21 (91%) |
| 7 | 4 | 1.000 | 4 (100%) | 0 |

GT 1 (off-mat) and GT 7 (off-mat, brief) are perfectly pure — they occupy isolated tracklets.
GT 6 is the opposite: 91% impure, meaning the few times it is detected, the detection belongs
to a tracklet that is mostly tracking someone else. **No identity conclusion can be drawn for
GT 6; its value here is as detection evidence.**

GT 2 and GT 3 (on-mat, moderate area) have 33-42% impure segments — the grappling core where
tracker drift and pair-box contamination are worst. GT 5 is an outlier: small boxes but only 1%
impure, suggesting spatial isolation on the mat.

---

## 4. Group spans

132 group segments across all GT tracks. 130 of 132 have only one GT person matched to the
group node (the other person was undetected during the span). Only 2 segments have two GT
people matched, both with mean GT-box IoU = 0.282 (below 0.3 threshold — false group).

**In-group vs non-group misattribution:**

| Location | Misattributed segments |
|----------|----------------------|
| Inside group span | 54 (19.6%) |
| Outside group span | 221 (80.4%) |
| **Total** | **275** |

80.4% of misattributed segments occur outside group spans. The dominant identity problem is
not group ambiguity — it is solver routing on SOLO nodes and tracker drift.

The 2 false-group segments (IoU <= 0.3) have well-separated GT boxes grouped together. These
are real errors, but at 2 of 132 (1.5%) they are not a significant contributor.

---

## 5. Edge-cost analysis — four populations

100 node-sequence boundaries analysed (points where the D1 node changes between consecutive
segments of a GT track).

| Population | Count | Description |
|------------|-------|-------------|
| **no_edge_exists** | **63 (63%)** | No edge between the two nodes — graph-construction gap |
| chosen_wrong | 21 (21%) | Solver selected an edge GT disagrees with |
| chosen_correct | 15 (15%) | Solver selected an edge GT agrees with |
| available_not_chosen | 1 (1%) | Correct edge existed but lost on cost |

**Population sizes are small.** 100 boundaries with 8 GT tracks on one clip. Patterns below
should be read as directional, not statistical.

### Population 1: no_edge_exists (63) — graph construction

The dominant failure mode. At 63 of 100 boundaries, no edge existed between the GT-correct
source and destination nodes. This is a **D1 graph-construction** problem, not a cost-weight
problem — the solver never had the right option.

By GT track: GT 2 accounts for 40 (63%), GT 0 for 14, GT 3 for 7, GT 4 for 1, GT 1 for 1.
GT 2 and GT 3 (moderate-area on-mat tracks) dominate.

### Population 2: chosen_wrong (21) — cost failures

All 21 are cost failures (`is_allowed=True`); zero gate failures.

| Cost term | Mean | Median | Min | Max |
|-----------|------|--------|-----|-----|
| total_cost | 0.11 | 0.02 | 0.02 | 0.61 |
| term_env | 0.01 | 0.01 | 0.01 | 0.01 |
| term_time | 0.00 | 0.01 | 0.00 | 0.01 |
| term_vreq | 0.03 | 0.00 | 0.00 | 0.45 |
| term_flags | 0.01 | 0.00 | 0.00 | 0.25 |
| term_group_coherence | 0.02 | 0.00 | 0.00 | 0.05 |

The costs are low — median total_cost is 0.02 for both chosen_correct and chosen_wrong. The
solver is not being driven to wrong answers by high costs; the wrong edges simply have similar
costs to the right ones, and the graph structure (missing edges) constrains the choice.

### Population 3: available_not_chosen (1) — the answer was there

One boundary: the correct edge existed with total_cost 0.017, is_allowed=True, but was not
selected. With N=1, this is not a pattern.

### Population 4: no_edge_exists (63) — most actionable

This is the most actionable finding. 63% of node boundaries have no candidate edge connecting
the GT-correct pair of nodes. The solver cannot pick the right answer if it does not exist.
This points at D1 candidate generation (`d1_graph_build.py`) as the primary target for
identity improvement on this clip, not D2 cost weights or D3 solver logic.

---

## 6. GT-VERIFY-1: Mat classification and GT handling verification

### On-mat classification — verified, corrected (all 8 on-mat)

**Method:** Projected GT contact points via `contact_point_from_bbox` + `project_to_world()`
with production H/K/D. Checked against mat blueprint bounds (x 42–58, y 34–58). Criterion:
on-mat if >= 50% of projected points within blueprint.

| GT | x range (m) | y range (m) | Blueprint % | Quad % | Classification |
|----|-------------|-------------|-------------|--------|---------------|
| 0 | 52.78–54.46 | 54.69–56.16 | 100.0% | 97.1% | ON MAT |
| 1 | 50.07–51.75 | 53.34–57.56 | 100.0% | 3.8% | ON MAT (extrapolated) |
| 2 | 52.75–54.69 | 49.29–51.76 | 100.0% | 100.0% | ON MAT |
| 3 | 50.73–55.25 | 46.92–51.72 | 100.0% | 80.7% | ON MAT |
| 4 | 53.21–54.79 | 54.54–56.05 | 100.0% | 99.8% | ON MAT |
| 5 | 53.30–55.34 | 45.59–47.11 | 100.0% | 100.0% | ON MAT |
| 6 | 53.48–54.81 | 45.67–46.76 | 100.0% | 100.0% | ON MAT |
| 7 | 47.28–49.36 | 51.41–55.85 | 100.0% | 0.0% | ON MAT (extrapolated) |

GT 1 and GT 7 fall outside the calibrated quad but inside the mat blueprint. Their world
positions are homography extrapolations — less reliable than in-quad positions, but not
off-mat. The previous "off-mat" classification was an undocumented eyeball judgement.

The previous 32.9% on-mat / 58.9% off-mat split and the conclusion that "off-mat people
are easier to detect" (GT-EVAL-1) are void — there is no off-mat population in this clip.

### Mask-vs-bbox contact path

All 8,437 detections used `mask_source = "bbox_fallback"` — zero masks. GT and tracklets
go through identical contact-point code (`contact_point_from_bbox`). No discrepancy possible.

### GT 1 / GT 5 merger (p0013) — solver failure on spatially unrelated people

p0013 is the canonical person_id for both GT 1 and GT 5. These two tracks are never spatially
close: minimum pixel distance 431px, zero IoU overlap across all 1,191 co-occurring frames,
metres apart in world space (GT 1: x 50.1–51.8; GT 5: x 53.3–55.3). The merger is a genuine
solver failure — the ILP stitched two unrelated people into one identity with no spatial basis.

### T2.5 baseline spectator language

`docs/evidence/t2_5_baseline_1/findings.md` §11 attributes 130229's 106 person tracks to
"off-mat spectators." This is a different clip (130229), a different analysis (uses pipeline
`x_m` contact points, not eyeball), and refers to people at x_m 46.62 — genuinely outside the
mat. It does not need the same correction.

### Timeline legend

The tracklet band now uses purity-semantic colours instead of per-tracklet-id colours:
- Green: pure (purity >= 0.9)
- Orange: impure (0.5–0.9)
- Red: heavily impure (< 0.5)

The legend is complete: 8 entries covering all rendered colour elements. The per-id colour map
used in GT-DIAG-1's original timeline was unreadable (50+ unique tracklets against 5 legend
entries). The purity-semantic scheme makes the diagnostic signal (tracklet contamination)
visible at a glance.

---

## 7. Artifacts

| File | Description |
|------|-------------|
| `gt_sequence_table.parquet` | 901 rows, one per contiguous segment per GT track |
| `gt_sequence_table.csv` | Same, human-readable |
| `edge_cost_analysis.csv` | 100 node boundaries with population classification |
| `timeline.png` | Three-layer timeline: tracklet purity (top), D1 node (mid), person_id (bot) |
| `compact_view.md` | Chronological text: correct vs actual paths, mm:ss, inline costs |
| `annotated_gt.mp4` | Pipeline boxes + GT boxes + IoU intersection regions |
| `mat_view_gt.mp4` | Mat canvas with pipeline points + GT hollow circles |

Module: `src/pipeline_validation/gt_sequence_diag/`
CLI: `PYTHONPATH=src python -m pipeline_validation.gt_sequence_diag.run --trace <path> --pipeline-dir <path> --pfm <path> --video <path> --camera <id> --output <path>`
