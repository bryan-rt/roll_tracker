# NOEDGE-1: Characterisation of the 63 no-edge boundaries

**Date:** 2026-09-02
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**Source:** GT-DIAG-1 `edge_cost_analysis.csv` (100 node-sequence boundaries)
**Module:** Read-only analysis of existing artifacts. No pipeline changes.

---

## 1. Context

GT-DIAG-1 found 63 of 100 node-sequence boundaries where no edge exists in
`d2_edge_costs.parquet` — the correct connection was never a candidate. Against 21
chosen-wrong, 15 chosen-correct, and 1 available-but-not-chosen. The initial reading
pointed at D1 graph construction.

Inspection of the data prompted a closer look: boundary frames 379 and 380 for GT 0 show
a single-frame excursion into tracklet t49 and straight back — counted as two no-edge
boundaries but representing one identity flicker event.

---

## 2. Dwell-time partition

For each no-edge boundary, the dwell time in the destination segment (consecutive frames
the GT track stays before moving again):

| Dwell | Count | % | Interpretation |
|-------|-------|---|----------------|
| 1–2 frames (flicker) | 33 | 52% | GT assignment briefly jumps to a different tracklet |
| 3–15 frames (brief) | 23 | 37% | Short excursion into a neighbouring tracklet |
| >15 frames (genuine) | 7 | 11% | A real segment the solver had to bridge |

**52% of the 63 are flicker — the GT assignment wobbles for 1–2 frames then returns.**

### Deduplicated event count

An out-and-back pair (A→B, B→A within the dwell window) is one event, not two boundaries.

| Metric | Count |
|--------|-------|
| Raw boundaries | 63 |
| Paired (out-and-back) | 25 pairs = 50 boundaries |
| Unpaired (one-way) | 13 |
| **Deduplicated events** | **38** |

After deduplication, the event dwell distribution:

| Dwell | Events |
|-------|--------|
| Flicker (1–2f) | 25 |
| Brief (3–15f) | 12 |
| Genuine (>15f) | 1 |

**The 63 collapses to 38 events. Only 1 is a genuine long-dwell transition.**

---

## 3. Detection gap vs graph-construction gap

For each boundary: frame gap between source node's end_frame and destination node's
start_frame, and whether GT was detected during the gap.

| Case | Count | % | Meaning |
|------|-------|---|---------|
| Adjacent or overlapping | 63 | 100% | Nodes overlap in time (frame_gap ≤ 0) |
| No detections in gap | 0 | 0% | — |
| Detected in gap | 0 | 0% | — |

**All 63 are temporally overlapping nodes.** Frame gap statistics: mean -209 frames,
min -533, max 0. No detection voids, no genuine temporal gaps.

This means: the source and destination D1 nodes exist at the same time. The GT track is
being tracked by tracklet A in one node and tracklet B in the other, and the GT-to-detection
assignment (per_frame_matches, Hungarian IoU 0.5) briefly jumps from A to B. D2 has no edge
between the two nodes because they are not in a lifecycle relationship (no merge/split event
connects them) — and it should not need one, because the GT person did not physically move
between them.

**The "no-edge" is not a missing connection across a detection void. It is a measurement
artifact from the GT matcher jumping between concurrent tracklets.**

---

## 4. D1 candidate-generation parameters

Parameters governing D1 edge creation and their basis:

| Parameter | Value | Basis | Notes |
|-----------|-------|-------|-------|
| `reconnect_max_gap_frames` | 250 | **frame-based** | Default; not in config. Irrelevant here (all boundaries are overlapping, not gapped) |
| `reconnect_enabled` | true | — | |
| `reconnect_boundary_slack_frames` | 2 | **frame-based** | |
| `split_search_horizon_frames` | 2700 | **frame-based** | Config overrides default 120 |
| `min_group_duration_frames` | 10 | **frame-based** | |
| `merge_dist_m` | 1.5 | physical (metres) | |
| `split_dist_m` | 2.0 | physical (metres) | |
| `v_max_mps` | 8.0 | physical (m/s) | Used for reconnect speed gate |

D2 cost parameters:
| Parameter | Value | Basis |
|-----------|-------|-------|
| `reconnect_v_max_mps` | 8.0 (from d0) | physical (m/s) |
| `endpoint_search_window_frames` | from d1.carrier_coord_window_frames | **frame-based** |

**Four of the frame-based parameters (`reconnect_max_gap_frames`, `split_search_horizon_frames`,
`min_group_duration_frames`, `endpoint_search_window_frames`) are of the same class as the
timing constants removed elsewhere by the variable-dt work.** At ~15fps vs 30fps, frame-based
values represent different time durations. None are changed here.

**These parameters are not the cause of the 63.** The boundaries are all on overlapping nodes
where reconnect parameters do not apply. D1 creates lifecycle edges (CONTINUE, MERGE, SPLIT)
between nodes with temporal adjacency and spatial proximity. Two nodes that overlap by ~209
frames are not in a lifecycle relationship — they are concurrent tracklets serving different
people, and the GT matcher's assignment briefly switches between them.

---

## 5. Clustering

### By GT track

| GT | Boundaries | % | Recall | No-det overlap | No-det no-overlap |
|----|-----------|---|--------|----------------|-------------------|
| 2 | 40 | 63% | 0.558 | 770 | 10 |
| 0 | 14 | 22% | 0.966 | 60 | 0 |
| 3 | 7 | 11% | 0.668 | 284 | 303 |
| 1 | 1 | 2% | 0.920 | 2 | 139 |
| 4 | 1 | 2% | 0.464 | 946 | 0 |

GT 2 alone accounts for **63%** of all no-edge boundaries. GT 2 is a moderate-area on-mat
track (14,270 px², recall 0.558) in the grappling core — the region with the worst
pair-box under-segmentation and tracker drift. Its 40 boundaries are flicker from the GT
matcher jumping between the many concurrent tracklets covering the same physical area.

GT 0 at 14 boundaries (22%) is surprising given its high recall (0.966). These cluster in
the middle of the clip where GT 0 transitions between rolling partners.

### By frame region

| Region | Count | % |
|--------|-------|---|
| 0–440 (first third) | 10 | 16% |
| 441–880 (middle third) | 37 | 59% |
| 881–1763 (last third) | 16 | 25% |

Concentrated in the middle third — the active grappling period with the most concurrent
overlapping tracklets.

### Occlusion at boundary frames

| Condition | Count | % |
|-----------|-------|---|
| Occluded (GT box IoU>0 with another GT) | 26 | 41% |
| Not occluded | 37 | 59% |

Only 41% are occluded at the boundary frame. The majority are identity flicker on
non-occluded frames, consistent with tracker drift rather than physical occlusion causing
the assignment jump.

---

## 6. Revised interpretation

**The 63 no-edge boundaries do NOT represent 63 distinct D1 graph-construction failures.**

They represent **38 deduplicated events**, of which 25 are 1–2 frame flicker, 12 are
brief 3–15 frame excursions, and 1 is a genuine long transition. All are on temporally
overlapping nodes (mean overlap 209 frames) with zero detection gaps. GT 2 accounts for
63% of the count.

The mechanism is **GT matcher assignment flicker**: the Hungarian matcher (IoU 0.5) briefly
assigns the GT box to a different concurrent tracklet for 1–2 frames, creating two
"boundaries" that look like graph gaps but are actually measurement noise from the
evaluation instrument tracking a person across concurrent, overlapping tracklets.

**Where the fix belongs:**
- The flicker events (25/38 = 66% of deduplicated) need no D1 fix — they are evaluation
  artifacts. The GT person did not physically transition; the matcher briefly preferred a
  different tracklet. A temporal smoothing filter on GT assignment would collapse them.
- The brief events (12/38 = 32%) are ambiguous — some may be genuine transitions across
  concurrent tracklets that the solver should route, but many are likely extended flicker.
- The 1 genuine event is too small a sample to characterize.

**The prior conclusion — "D1 graph construction is the primary target for identity improvement
on this clip" — is not supported at the level stated.** The 63% no-edge count was inflated by
double-counting and flicker. Of the 100 boundaries, the revised breakdown considering
dwell time is:

| Population | Count | Interpretation |
|------------|-------|----------------|
| no_edge_exists (flicker, dwell ≤2f) | 33 | Evaluation artifact — GT matcher noise |
| no_edge_exists (brief, 3–15f) | 23 | Ambiguous — may be real transitions or extended flicker |
| no_edge_exists (genuine, >15f) | 7 | Real segments the solver had to bridge (but no detection gap) |
| chosen_wrong | 21 | Cost failures — D2 problem |
| chosen_correct | 15 | Working correctly |
| available_not_chosen | 1 | Correct edge existed, lost on cost — D2/D3 |

The actionable failures are the 21 chosen-wrong (D2 cost weights) and the 7 genuine
no-edge boundaries. Together they are 28 — less than half the original 63+21=84 "problem"
count, and split between D1 (7) and D2 (21+1=22). **D2 cost weights have more actionable
failures than D1 candidate generation on this clip.**

---

## 7. Artifacts

Analysis consumed existing artifacts only. No new parquets generated.

Source data:
- `docs/evidence/gt_diag_1/edge_cost_analysis.csv` (100 boundaries)
- `docs/evidence/gt_diag_1/gt_sequence_table.parquet` (901 segments)
- `outputs/_eval/stage_d/gt-eval-fp7oJQ-132650/FP7oJQ/gt_person_trace.parquet`
- `outputs/.../stage_D/d1_graph_nodes.parquet`
- `outputs/.../stage_D/d2_edge_costs.parquet`
