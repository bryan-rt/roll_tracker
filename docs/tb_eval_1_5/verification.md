# TB-EVAL-1.5 Pipeline Rerun Verification Report

Generated: 2026-05-09

Pipeline: bjj-detect-all-cameras.pt (conf=0.45, prefer_coreml=true)
Gym ID: _eval_gt | Stages: A through D (E, F skipped)
Cross-camera Stage D: skipped (Stage C placeholder, Tier 2 disabled)

**Note:** Stage C is currently a placeholder — writes empty JSONL files.
Stage D operates on Tier 3 (HSV histogram) evidence only. No AprilTag
identity hints are available for these clips.

---

## FP7oJQ (FP7oJQ-20260318-200014)

- **stage_A/detections.parquet**: 44381 rows, 4530 unique frames (0-4529)
  - GT sample frames present: [0, 100, 200, 300] (4/4)
  - tracklet_id non-null: 44381/44381 (100.0%)
- **stage_A/tracklet_summaries.parquet**: 251 tracklets
- **stage_A/keypoints.parquet**: 44381 rows
- **stage_A/color_histograms.parquet**: 44381 rows
- **stage_A/tracklet_histogram_summaries.parquet**: 134 rows
- **stage_C/identity_hints.jsonl**: empty (expected — Stage C placeholder)
- **stage_C/tag_observations.jsonl**: empty (expected — Stage C placeholder)
- **stage_D/person_tracks.parquet**: 25144 rows, 14 distinct person_ids
- **stage_D/identity_assignments.jsonl**: 0 entries

## J_EDEw (J_EDEw-20260318-200015)

- **stage_A/detections.parquet**: 49160 rows, 4530 unique frames (0-4529)
  - GT sample frames present: [0, 100, 200, 300] (4/4)
  - tracklet_id non-null: 49160/49160 (100.0%)
- **stage_A/tracklet_summaries.parquet**: 236 tracklets
- **stage_A/keypoints.parquet**: 49160 rows
- **stage_A/color_histograms.parquet**: 49160 rows
- **stage_A/tracklet_histogram_summaries.parquet**: 119 rows
- **stage_C/identity_hints.jsonl**: 1 entries
- **stage_C/tag_observations.jsonl**: 1 entries
- **stage_D/person_tracks.parquet**: 22830 rows, 15 distinct person_ids
- **stage_D/identity_assignments.jsonl**: 0 entries

## PPDmUg (PPDmUg-20260318-training)

- **stage_A/detections.parquet**: 19243 rows, 2998 unique frames (0-2997)
  - GT sample frames present: [0, 100, 200, 300] (4/4)
  - tracklet_id non-null: 19243/19243 (100.0%)
- **stage_A/tracklet_summaries.parquet**: 73 tracklets
- **stage_A/keypoints.parquet**: 19243 rows
- **stage_A/color_histograms.parquet**: 19243 rows
- **stage_A/tracklet_histogram_summaries.parquet**: 34 rows
- **stage_C/identity_hints.jsonl**: empty (expected — Stage C placeholder)
- **stage_C/tag_observations.jsonl**: empty (expected — Stage C placeholder)
- **stage_D/person_tracks.parquet**: 11948 rows, 7 distinct person_ids
- **stage_D/identity_assignments.jsonl**: 0 entries
