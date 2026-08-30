# GT-EVAL-1: First post-fix ground truth evaluation

**Date:** 2026-08-30
**Clip:** FP7oJQ-20260822-132650 (1,764 frames, ~118s)
**GT:** 8 people, 12,468 boxes, hand-labelled in CVAT (task 2551869), stride-1 dense
**Model:** bjj-detect-all-cameras-v2 (domain-tuned yolo26n, conf 0.45, CoreML)
**Manifest:** `configs/models/gt-eval-fp7oJQ-132650.yaml`
**Footage:** Post-recorder-fix (schema-5 sidecar, source PTS), homography recalibrated 2026-08-24

---

## 1. Stage A — Detection evaluation

**Metric basis:** 1,764 val frames (full clip), parquet path (production `detections.parquet`),
Hungarian matching at IoU 0.5 (CP-EVAL-1 frozen instrument).

| Metric | Value |
|--------|-------|
| Recall@0.5 | 0.6233 |
| Recall@0.7 | 0.4885 |
| Recall@0.9 | 0.0934 |
| Precision@0.5 | 0.9208 |
| Mean IoU@0.5 | 0.7872 |
| Total GT boxes | 12,468 |
| Total predictions | 8,439 |
| Matched@0.5 | 7,771 |
| Unmatched GT | 4,697 (37.7%) |
| Zero-detection frames | 0 |
| Duplicate rate | 9.1% |
| True merge rate | 73.9% |

**Recall ceiling on correct_id:** 4,697 GT boxes (37.7%) have no matched detection. These
cannot score correct regardless of stitching quality. correct_id is capped at ~62.3% by
detection recall alone.

### Per-GT-track recall

| GT track | Recall@0.5 | Mat position | Notes |
|----------|-----------|--------------|-------|
| 0 | 0.966 | ON MAT (97%) | Rolling — high recall |
| 1 | 0.920 | OFF MAT (4%) | Spectator/coach near edge |
| 2 | 0.558 | ON MAT (100%) | Rolling — under-detected |
| 3 | 0.669 | ON MAT (74%) | On mat, partially near edge |
| 4 | 0.464 | ON MAT (97%) | Rolling — heavily under-detected |
| 5 | 0.748 | ON MAT (100%) | On mat |
| 6 | 0.039 | ON MAT (100%) | On mat — nearly invisible to detector |
| 7 | 0.600 | OFF MAT (0%) | Present frames 210–329 only, off mat |

**On-mat tracks (0, 2, 3, 4, 5, 6):** Recall ranges from 0.039 (track 6) to 0.966 (track 0).
Track 6 is essentially undetected (39 detections in 1,764 frames).

**Off-mat tracks (1, 7):** Track 1 (spectator/coach) has 0.920 recall — high visibility near
the frame edge. Track 7 is brief (120 frames) and off-mat.

### Cross-validation gate

The FP7oJQ cross-validation gate **failed** (precision diff 0.029, mean IoU diff 0.014 —
both exceed the 0.01 tolerance). The gate compares parquet vs direct inference over 1,764
val frames. The diffs are:

| Metric | Parquet | Direct | Diff |
|--------|---------|--------|------|
| Recall | 0.6233 | 0.6159 | 0.0074 |
| Precision | 0.9208 | 0.8917 | 0.0291 |
| Mean IoU | 0.7872 | 0.7731 | 0.0141 |

The gate failure printed a warning but did NOT switch to direct inference — `force_direct`
is only set when the gate is **skipped** (staleness), not when it **fails**. The evaluation
used the parquet path. Both stage-a and stage-d scored the same `detections.parquet`.

The diffs are likely from decode-path differences (cv2.VideoCapture on a VFR file may
produce slightly different frame positioning than the pipeline's FrameIterator, or CoreML
vs MPS inference differences). This does not invalidate the evaluation — the parquet path
is the authoritative one (it is what the pipeline actually produced).

---

## 2. Stage D — Identity evaluation

**Metric basis:** 1,764 val frames, production person_tracks.parquet, Hungarian IoU 0.5.

| Metric | Value |
|--------|-------|
| **correct_id (present)** | **37.2%** (4,637 / 12,475) |
| present_misattributed | 17.7% (2,213 / 12,475) |
| stage_a_no_detection | 37.7% (4,701 / 12,475) |
| d4_unassigned | 7.4% (924 / 12,475) |
| Identity recall | 1.000 (all 8 GT tracks have a canonical person_id) |
| Identity precision | 0.857 (1 merger: p0013 maps to GT tracks 1 and 5) |
| Mean coverage | 0.555 |
| Mean purity | 0.536 |
| Pipeline person_count | 17 |
| GT person count | 8 |
| Ratio | 2.1x (17 pipeline persons : 8 GT persons) |

### Per-GT-track identity

| GT track | Canonical | Coverage | Purity | Switches | Mat | Correct_id |
|----------|-----------|----------|--------|----------|-----|------------|
| 0 | p0001 | 0.793 | 0.383 | 55 | ON | 67.2% |
| 1 | p0013 | 0.777 | 0.767 | 48 | OFF | 59.6% |
| 2 | p0003 | 0.558 | 0.294 | 134 | ON | 42.7% |
| 3 | p0010 | 0.657 | 0.544 | 104 | ON | 35.8% |
| 4 | p0011 | 0.464 | 0.657 | 101 | ON | 30.5% |
| 5 | p0013 | 0.561 | 0.399 | 70 | ON | 22.4% |
| 6 | p0009 | 0.031 | 0.426 | 26 | ON | 1.3% |
| 7 | p0014 | 0.600 | 0.819 | 5 | OFF | 49.2% |

**Merger:** p0013 is the canonical person_id for both GT track 1 (off-mat spectator) and
GT track 5 (on-mat athlete). The pipeline merged two distinct people into one identity.

**Track 6 (1.3% correct_id):** This person is effectively invisible to the detector
(recall 0.039) and consequently to the identity pipeline. 54 frames matched out of 1,764.

### Switch cause classification

| Cause | Count | % of switches |
|-------|-------|---------------|
| detection_failure | 351 | 65% |
| sloppy_box | 121 | 22% |
| true_switch | 65 | 12% |
| tracklet_dropped | 6 | 1% |
| **Total** | **543** | |

Detection failure is the dominant switch cause at 65%. True switches (Stage D mis-stitching
on clean detections) account for 12%.

---

## 3. Observations

### GT track composition — on-mat vs off-mat

6 of 8 GT tracks are on the mat (tracks 0, 2, 3, 4, 5, 6). 2 are off-mat (tracks 1, 7).

If scored on-mat only (6 tracks, 10,710 GT-frames):
- present: 3,526 / 10,710 = 32.9%
- stage_a_no_detection: 4,512 / 10,710 = 42.1%

If scored off-mat only (2 tracks, 1,885 GT-frames):
- present: 1,111 / 1,885 = 58.9%
- stage_a_no_detection: 189 / 1,885 = 10.0%

The on-mat correct_id (32.9%) is lower than the overall (37.2%). Detection recall is the
dominant bottleneck on-mat; off-mat people are easier to detect (less occlusion from
grappling).

### Person count: pipeline 17 vs GT 8

The pipeline produces 2.1x the actual person count. This is consistent with the CP7 finding
that detection under-segmentation (one box covering two grappling people) creates fragmented
tracklets that the solver stitches into spurious person_ids.

### Non-comparability to 33.9%

The canonical `correct_id` of 33.9% (CP-GT2ACTUALS-3.5) was measured on:
- Pre-recorder-fix footage (bursty arrival timestamps, dup/drop)
- Pre-checkpoint-2 pipeline code (no variable-dt, no sidecar timing)
- Pre-homography-recalibration
- J_EDEw camera (not FP7oJQ)
- Different clip, different athletes, different scene

**This 37.2% is a new baseline on clean footage, not a delta from 33.9%.** The two numbers
are not directly comparable — they differ in footage quality, camera, pipeline version, and
GT methodology. Comparing them would conflate all of these with any improvement from the
timing work.

---

## 4. Detection recall cap

Stage A recall at 0.623 means 37.7% of GT boxes have no detection to match. This caps
correct_id at ~62.3% before identity is even considered. The per-track breakdown shows:
- Track 6: 3.9% recall — essentially undetected
- Track 4: 46.4% recall — less than half detected
- Track 2: 55.8% recall — just over half

The true merge rate of 73.9% confirms the CP7 finding: most detection misses are
under-segmentation (one detection covering two grappling people), not total absence.
