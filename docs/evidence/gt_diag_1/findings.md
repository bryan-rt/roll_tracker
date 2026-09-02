# GT-DIAG-1: GT-to-pipeline sequence diagnostic

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**GT:** 8 tracks, 12,468 boxes, stride-1 dense
**Pipeline state:** `variable_dt: true` (VDT-DEFAULT-1), post-Piece-9
**Source:** Extends `gt_person_trace.parquet` (CP6, Hungarian IoU 0.5, CP-EVAL-1 frozen)
**Module:** `src/pipeline_validation/gt_sequence_diag/`

---

## 1. Per-GT-track summary

| GT | On-mat | Matched | Clip% | Presence% | Low-conf | Median box area | Segments | Tracklets | Persons | Group segs |
|----|--------|---------|-------|-----------|----------|-----------------|----------|-----------|---------|------------|
| 0  | ON MAT | 1,705 | 96.7% | 96.6% | | 60,959 | 91 | 10 | 6 | 31 |
| 1  | OFF MAT | 1,624 | 92.1% | 92.0% | | 39,452 | 84 | 7 | 5 | 7 |
| 2  | ON MAT | 985 | 55.8% | 55.8% | | 14,270 | 196 | 15 | 7 | 39 |
| 3  | ON MAT | 1,178 | 66.8% | 66.7% | | 13,561 | 159 | 18 | 9 | 27 |
| 4  | ON MAT | 819 | 46.4% | 46.4% | * | 43,401 | 173 | 10 | 5 | 16 |
| 5  | ON MAT | 1,321 | 74.9% | 74.8% | | 5,145 | 143 | 8 | 8 | 6 |
| 6  | ON MAT | 68 | 3.9% | 3.9% | * | 5,060 | 47 | 4 | 4 | 6 |
| 7  | OFF MAT | 72 | 4.1% | 60.0% | | 30,457 | 8 | 1 | 2 | 0 |

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
| 1 | 39,452 | 92.1% | Off-mat spectator — near camera edge |
| 7 | 30,457 | 60.0% | Off-mat, brief |
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

## 6. Artifacts

| File | Description |
|------|-------------|
| `gt_sequence_table.parquet` | 901 rows, one per contiguous segment per GT track |
| `gt_sequence_table.csv` | Same, human-readable |
| `edge_cost_analysis.csv` | 100 node boundaries with population classification |
| `timeline.png` | Three-layer timeline (tracklet / D1 node / person_id) |

Module: `src/pipeline_validation/gt_sequence_diag/`
CLI: `PYTHONPATH=src python -m pipeline_validation.gt_sequence_diag.run --trace <path> --pipeline-dir <path> --pfm <path> --output <path>`
