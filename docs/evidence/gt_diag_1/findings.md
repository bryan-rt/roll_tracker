# GT-DIAG-1: GT-to-pipeline sequence diagnostic

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**GT:** 8 tracks, 12,468 boxes, stride-1 dense
**Pipeline state:** `variable_dt: true` (VDT-DEFAULT-1), post-Piece-9
**Source:** Extends `gt_person_trace.parquet` (CP6, Hungarian IoU 0.5, CP-EVAL-1 frozen)
**Module:** `src/pipeline_validation/gt_sequence_diag/`

---

## 1. Per-GT-track summary

| GT | On-mat | In-quad % | Blueprint % | Matched | Clip% | Presence% | Low-conf | Median box area | Segments | Tracklets | Persons | Group segs |
|----|--------|-----------|-------------|---------|-------|-----------|----------|-----------------|----------|-----------|---------|------------|
| 0  | ON MAT | 97.1% | 100.0% | 1,705 | 96.7% | 96.6% | | 60,959 | 91 | 10 | 6 | 31 |
| 1  | ON MAT | 3.8% | 100.0% | 1,624 | 92.1% | 92.0% | | 39,452 | 84 | 7 | 5 | 7 |
| 2  | ON MAT | 100.0% | 100.0% | 985 | 55.8% | 55.8% | | 14,270 | 196 | 15 | 7 | 39 |
| 3  | ON MAT | 80.7% | 100.0% | 1,178 | 66.8% | 66.7% | | 13,561 | 159 | 18 | 9 | 27 |
| 4  | ON MAT | 99.8% | 100.0% | 819 | 46.4% | 46.4% | * | 43,401 | 173 | 10 | 5 | 16 |
| 5  | ON MAT | 100.0% | 100.0% | 1,321 | 74.9% | 74.8% | | 5,145 | 143 | 8 | 8 | 6 |
| 6  | ON MAT | 100.0% | 100.0% | 68 | 3.9% | 3.9% | * | 5,060 | 47 | 4 | 4 | 6 |
| 7  | **OFF MAT** | 0.0% | 100.0% | 72 | 4.1% | 60.0% | | 30,457 | 8 | 1 | 2 | 0 |

**Mat classification (GT-VERIFY-2):** 7 of 8 tracks are on the mat. Criterion:
point-in-any-mat-rectangle from `configs/mat_blueprint.json` — individual mat surfaces,
not the blueprint bounding box. GT 7 at x 47.3–49.4, y 51.4–55.9 falls in the gap between
the left mats (y 34–41 at x<50) and the right mats (x≥50); no mat rectangle covers that
region. Three columns: `on_mat` (mat rectangle, the classification), `in_blueprint_pct`
(canvas bounding box x 42–58, y 34–58), `in_quad_pct` (calibrated quad x 51–57, y 34–56).
Projection via `contact_point_from_bbox` + `project_to_world()` with production H/K/D.
All detections used `bbox_fallback` (zero masks).

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

GT 1 (on-mat but outside quad, isolated) and GT 7 (off-mat, brief) are perfectly pure — they occupy isolated tracklets.
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

## 6. GT-VERIFY-2: Mat classification and GT handling verification

### On-mat classification — verified, corrected (7 on-mat, GT 7 off-mat)

**Method:** Point-in-any-mat-rectangle from `configs/mat_blueprint.json` (9 individual mat
surfaces). Projected GT contact points via `contact_point_from_bbox` + `project_to_world()`
with production H/K/D. On-mat if >= 50% of projected points fall inside any mat rectangle.

| GT | x range (m) | y range (m) | Mat rect % | Blueprint % | Quad % | Classification |
|----|-------------|-------------|-----------|-------------|--------|---------------|
| 0 | 52.78–54.46 | 54.69–56.16 | 100.0% | 100.0% | 97.1% | ON MAT |
| 1 | 50.07–51.75 | 53.34–57.56 | 100.0% | 100.0% | 3.8% | ON MAT (extrapolated) |
| 2 | 52.75–54.69 | 49.29–51.76 | 100.0% | 100.0% | 100.0% | ON MAT |
| 3 | 50.73–55.25 | 46.92–51.72 | 100.0% | 100.0% | 80.7% | ON MAT |
| 4 | 53.21–54.79 | 54.54–56.05 | 100.0% | 100.0% | 99.8% | ON MAT |
| 5 | 53.30–55.34 | 45.59–47.11 | 100.0% | 100.0% | 100.0% | ON MAT |
| 6 | 53.48–54.81 | 45.67–46.76 | 100.0% | 100.0% | 100.0% | ON MAT |
| 7 | 47.28–49.36 | 51.41–55.85 | **0.0%** | 100.0% | 0.0% | **OFF MAT** |

GT 7 walks past the mats into another room. At x 47.3–49.4, y 51.4–55.9, it falls in the
gap between the left mats (y 34–41 at x<50) and the right mats (x≥50). It is inside the
blueprint bounding box (x 42–58, y 34–58) but no individual mat rectangle covers that area.
Operator-confirmed from visual inspection of `mat_view_gt.mp4` frame 219.

GT 1 falls outside the calibrated quad but inside right-side mat rectangles. Its world
positions are homography extrapolations — less reliable than in-quad positions, but on-mat.

Three columns: `on_mat` (mat rectangle — the classification), `in_blueprint_pct` (canvas
bounding box), `in_quad_pct` (calibrated quad — extrapolation reliability indicator).

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

## 7. Recall-gated engagement: the causal chain to product loss (GT-VERIFY-2)

Engagement detection requires BOTH athletes simultaneously detected for `engage_min_frames`
(15) consecutive frames within `engage_dist_m` (0.75m). Pair co-detection rate ≈ product of
individual recalls.

| GT pair | Individual recalls | Co-detection | Max consecutive <0.75m | Engagement |
|---------|-------------------|-------------|----------------------|------------|
| gt0↔gt4 | 0.966 × 0.464 | 43.8% (772f) | 58 frames | SURVIVED (truncated f692-1735) |
| gt2↔gt3 | 0.558 × 0.668 | 24.9% (440f) | 21 frames | SURVIVED (f814-1763) |
| gt5↔gt6 | 0.749 × 0.039 | 0.2% (4f) | **2 frames** | **KILLED** (need 15) |

GT 5 and GT 6 wrestle for the entire clip. GT 6 has 3.9% recall. They are co-detected in
4 frames out of 1,764 — never more than 2 consecutive. No match session is created. GT 6
receives no clip. The binding constraint is consecutive frames (`engage_min_frames`), not
aggregate co-detection rate.

---

## 8. GT engagement via Stage E's own logic (GT-VERIFY-2)

**Production thresholds:** `engage_dist_m=0.75`, `disengage_dist_m=2.0`,
`engage_min_frames=15`, `hysteresis_frames=450` (~30s@15fps),
`min_clip_duration_frames=150` (~10s@15fps).

GT tracks fed through `compute_pair_distances` + `run_proximity_hysteresis` unchanged:

| GT pair | Frames | Duration | Partial | Operator confirms |
|---------|--------|----------|---------|-------------------|
| gt0↔gt4 | 0–1763 | full clip | start+end | GT0↔GT4 |
| gt2↔gt3 | 812–1763 | 951f | end | GT2↔GT3 |
| gt5↔gt6 | 0–1763 | full clip | start+end | GT5↔GT6 |

GT1 alone (no engagement), GT7 alone (off-mat, brief). Exact match to operator observation.

**Threshold flapping (tested, not the cause here):**
- gt0↔gt4: 0 frames above 2.0m — no flapping possible
- gt5↔gt6: 0 frames above 2.0m — no flapping possible
- gt2↔gt3: 689 frames above 2.0m at clip start (f0-688), BEFORE engagement begins at f812.
  Once engaged, zero crossings above 2.0m. The 689-frame run exceeds `hysteresis_frames`
  (450) but occurs before engagement — it is a correct non-engagement, not flapping.

**Frame-based threshold note:** `hysteresis_frames=450` and `min_clip_duration_frames=150`
are frame-count-based. At ~15fps these are ~30s and ~10s; at 30fps they would be ~15s and
~5s. These should arguably be time-based (analogous to the variable-dt Kalman work) but
are not changed here — one clip is not a basis.

---

## 9. Stage E evaluation against GT target of 3 matches (GT-VERIFY-2)

Stage E produced 23 match sessions. GT target: 3. Three-bucket classification based on
whether the two person_ids' GT compositions contain a genuinely-engaged GT pair:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| CORRECT_ENGAGED | 6 | 26% | Dominant GTs are a real pair (right pair, wrong granularity) |
| CONTAMINATED | 13 | 57% | A real pair present but not dominant (identity corruption) |
| PHANTOM | 4 | 17% | No real pair present (same-GT-person or non-engaged) |

GT pairs covered: 2 of 3 (GT0↔GT4: 3 sessions, GT2↔GT3: 2 sessions + 1 via non-dominant).
**GT5↔GT6: zero sessions** (recall-gated — see §7).

All 23 sessions involve multi-GT person_ids (every person_id maps to multiple GT tracks).
The 13 contaminated sessions would resolve if D4 stopped fragmenting identities — they
contain a real engagement inside corrupted person_ids. The 4 phantoms are between genuinely
non-engaged GT people or fragments of the same person.

Stage F would produce up to 23 clips. GT target: 3 clips. Ratio: ≤7.7× over-production.

---

## 10. Partner-tolerant scoring (GT-VERIFY-2)

Two identity metrics:

| Metric | Value | Basis |
|--------|-------|-------|
| **Strict correct_id** | **34.3%** (4,275 / 12,475) | Current definition, comparable to history |
| **Partner-tolerant** | **37.4%** (4,665 / 12,475) | Swap between engaged GT pair at that frame counts as correct |

Delta: **+3.1pp**. Tolerance applies only while the pair is actively engaged, only between
the two engaged tracks, never across pair boundaries.

**GT1↔GT5 merger (p0013) confirmed still an error:** GT1 is never engaged with anyone.
GT5 is engaged with GT6. The merger crosses a pair boundary — still wrong under tolerance.

Rationale: the product delivers a clip containing both athletes; AprilTag pings after the
match resolve which is which. A swap between rolling partners mid-match is not a product
failure provided both are in the clip.

---

## 11. Occlusion partition (GT-VERIFY-2)

4,703 `stage_a_no_detection` frames partitioned by GT-box overlap:

| Partition | Count | % | Description |
|-----------|-------|---|-------------|
| Overlapping (IoU>0 with another GT) | 4,217 | 89.7% | Occlusion-driven — expected |
| No overlap | 486 | 10.3% | Interesting cases |

**Per GT track:**

| GT | No-det | Overlapping | No-overlap | Notes |
|----|--------|-------------|------------|-------|
| 0 | 60 | 60 (100%) | 0 | All occlusion-driven |
| 1 | 141 | 2 (1%) | 139 (99%) | Isolated, outside quad — not occlusion |
| 2 | 780 | 770 (99%) | 10 | Grappling core — mostly occlusion |
| 3 | 587 | 284 (48%) | 303 (52%) | Mixed — large no-overlap population |
| 4 | 946 | 946 (100%) | 0 | All occlusion-driven |
| 5 | 444 | 444 (100%) | 0 | All occlusion-driven |
| **6** | **1,697** | **1,697 (100%)** | **0** | **All overlapping — GT 6 is always under GT 5** |
| 7 | 48 | 14 (29%) | 34 (71%) | Off-mat, brief |

**GT 6 (extreme case):** 1,697 no-detection frames, ALL overlapping with another GT box.
GT 6's box area (5,060 px²) is comparable to GT 5's (5,145 px²), yet recall is 3.9% vs
74.9%. The occlusion partition explains this: GT 6 is always underneath GT 5 (same mat
position, continuous grappling). The detector sees one person-sized box and detects one
person. This is not a model weakness or distance problem — it is physical occlusion from
the ceiling angle.

**No-overlap characterization (486 frames):**
- Flicker (detected in adjacent frame): 91 (18.7%) — transient misses
- Sustained (not in adjacent): 395 (81.3%) — persistent blind spots
- Area: median 6,814 px², mean 28,100 px² — mix of small and large boxes
- GT 3 dominates (303 frames) — moderate-area on-mat track with unexplained sustained loss
- GT 1 contributes 139 frames — isolated, outside quad, not occlusion

---

## 12. Artifacts

| File | Description |
|------|-------------|
| `gt_sequence_table.parquet` | 901 rows, one per contiguous segment per GT track |
| `gt_sequence_table.csv` | Same, human-readable |
| `edge_cost_analysis.csv` | 100 node boundaries with population classification |
| `timeline.png` | Three-layer timeline: tracklet purity (top), D1 node (mid), person_id (bot) |
| `compact_view.md` | Chronological text: correct vs actual paths, mm:ss, inline costs |
| `annotated_gt.mp4` | Pipeline boxes + GT boxes + IoU intersection regions |
| `mat_view_gt.mp4` | Mat canvas: pipeline points + GT circles + engagement overlays (green=GT, orange dashed=Stage E) |
| `gt_verify_2_analysis.json` | Full GT-VERIFY-2 analysis: engagement, recall-gating, Stage E eval, tolerance, occlusion |

Module: `src/pipeline_validation/gt_sequence_diag/`
CLI: `PYTHONPATH=src python -m pipeline_validation.gt_sequence_diag.run --trace <path> --pipeline-dir <path> --pfm <path> --video <path> --camera <id> --output <path>`
