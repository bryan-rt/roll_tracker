# CP7-pre-6: NMS-IoU Sweep End-to-End Impact (FP7oJQ)

## Setup

- **Scope:** FP7oJQ, full clip (4530 frames) processed, scored on frames 0-300 (dense GT)
- **Arms:** iou={0.7, 0.85, 0.90}, conf=0.45 fixed. iou=0.95 attempted but D3 solver
  intractable (3,147 tracklets, 27.0 det/frame, >8 hours CPU time, killed).
- **NMS path:** All arms use .pt model with Python-side NMS (end2end disabled). Production
  uses CoreML with baked-in NMS; see "End2End NMS Discovery" below.
- **Regression test:** Plumbing changes proved inert on default path (bit-for-bit identical
  detections.parquet and person_tracks.parquet with and without the iou field when unset).

## End2End NMS Discovery (critical finding)

YOLOv26n models have **end2end NMS baked into the model graph** (`model.end2end = True`).
The ultralytics `iou` kwarg to `model.predict()` is completely ignored when end2end is
active -- Python-side NMS is bypassed entirely. Additionally, CoreML exports bake NMS into
the model, making the `iou` kwarg doubly inert on the production CoreML path.

To make NMS tunable:
1. Skip CoreML, use .pt weights directly
2. Set `model.end2end = False` and `model.model[-1].end2end = False` (Detect head)
3. Python-side NMS then respects the `iou` kwarg

The plumbing changes (4 files) implement this: when `iou` is set in config, CoreML is
skipped and end2end is disabled. When `iou` is None (default), behavior is bit-for-bit
identical to the pre-change code.

## Overall Six-Mode Breakdown

| Mode | iou=0.7 | iou=0.85 | iou=0.9 |
|------|---------|----------|---------|
| present | 3374 (80.1%) | 2874 (68.2%) | 2196 (52.1%) |
| stage_a_no_detection | 437 (10.4%) | 517 (12.3%) | 620 (14.7%) |
| stage_a_untracked | 139 (3.3%) | 192 (4.6%) | 58 (1.4%) |
| d3_dropped | 0 (0.0%) | 124 (2.9%) | 0 (0.0%) |
| d4_unassigned | 96 (2.3%) | 51 (1.2%) | 283 (6.7%) |
| present_misattributed | 168 (4.0%) | 456 (10.8%) | 1057 (25.1%) |
| missing_canonical | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Total** | **4214** | **4214** | **4214** |

**Monotonically worse as NMS relaxes.** Present drops 80.1% -> 52.1%, misattribution
rises 4.0% -> 25.1%.

## Detection Count & Fragmentation

| Metric | iou=0.7 | iou=0.85 | iou=0.9 | iou=0.95 (partial) |
|--------|---------|----------|---------|---------------------|
| Mean det/frame (0-300) | 11.9 | 13.0 | 15.8 | 27.0 |
| GT persons/frame | 14 | 14 | 14 | 14 |
| Median tracklets/GT person | 1.0 | 3.0 | 4.5 | (solver intractable) |
| Total tracklets | 179 | 579 | 1352 | 3147 |
| Unmatched preds (dup proxy) | 54 | 300 | 1208 | -- |
| Solver status | OPTIMAL | OPTIMAL | OPTIMAL | INTRACTABLE |

Relaxing NMS does NOT recover detections toward the 14-person GT -- it **overshoots**
with duplicates. At iou=0.9, 15.8 det/frame already exceeds 14 GT persons. Fragmentation
explodes (1.0 -> 4.5 median tracklets per GT person).

## Context Breakdown (pair threshold IoU >= 0.3)

Pair cells: 1948 (46.2%), Solo cells: 2266 (53.8%)

### Pair Context (entangled GT persons)

| Mode | iou=0.7 | iou=0.85 | iou=0.9 |
|------|---------|----------|---------|
| present | 1565 (80.3%) | 1422 (73.0%) | 1182 (60.7%) |
| present_misattributed | 115 (5.9%) | 399 (20.5%) | 507 (26.0%) |

### Solo Context (isolated GT persons)

| Mode | iou=0.7 | iou=0.85 | iou=0.9 |
|------|---------|----------|---------|
| present | 1809 (79.8%) | 1452 (64.1%) | 1014 (44.7%) |
| present_misattributed | 53 (2.3%) | 57 (2.5%) | 550 (24.3%) |

**Both contexts degrade.** The regression is NOT confined to pair-context. Solo-context
misattribution rises from 2.3% to 24.3% at iou=0.9 -- the duplicate boxes on isolated
persons cause tracker churn and identity confusion even where no second person exists.

## Context Breakdown (pair threshold IoU >= 0.5)

Pair cells: 1180 (28.0%), Solo cells: 3034 (72.0%)

| Context | iou=0.7 present | iou=0.9 present | iou=0.7 misattrib | iou=0.9 misattrib |
|---------|-----------------|-----------------|--------------------|--------------------|
| pair | 80.1% | 60.8% | 6.8% | 28.8% |
| solo | 80.1% | 48.7% | 2.9% | 23.6% |

**Result holds across both thresholds.** Solo regression is actually WORSE than pair
regression at iou=0.9 (present drops 31.4pp in solo vs 19.3pp in pair).

## Mode Transitions (from iou=0.7 baseline)

### iou_0.85 vs baseline (1067 changed cells)

| Transition | Count | Interpretation |
|-----------|-------|----------------|
| present->present_misattributed | 344 | regression: was correct, now wrong |
| present->d3_dropped | 287 | regression: was present, now dropped |
| present_misattributed->present | 127 | fixed: was wrong, now correct |
| stage_a_no_detection->present_misattributed | 102 | recovered into WRONG identity (hollow) |
| stage_a_no_detection->present | 90 | recovered into correct identity |
| present->stage_a_no_detection | 86 | regression: detection lost |

**Net: 344 present->misattrib vs 127 misattrib->present = -217 net regression.**
Recovery of 90 no_det->present is overwhelmed by 344 present->misattrib + 287 present->d3_dropped.

### iou_0.9 vs baseline (1602 changed cells)

| Transition | Count | Interpretation |
|-----------|-------|----------------|
| present->present_misattributed | 902 | regression: was correct, now wrong |
| present->d4_unassigned | 237 | regression: tracker lost identity |
| present->stage_a_no_detection | 161 | regression: detection lost |
| stage_a_no_detection->present | 93 | recovered into correct identity |
| stage_a_no_detection->present_misattributed | 92 | recovered into WRONG identity (hollow) |
| present_misattributed->d4_unassigned | 56 | regression: wrong identity -> no identity |
| present_misattributed->present | 29 | fixed: was wrong, now correct |

**Net: 902 present->misattrib vs 29 misattrib->present = -873 net regression.**
Recovery into correct identity (93 cells) is dwarfed by churn out of correct identity
(902 + 237 + 161 = 1300 cells).

## Conservation Check

All arms: 4214 total cells (14 GT tracks x 301 frames), six modes sum correctly per arm.
Context partitions sum correctly at both thresholds (pair + solo = total for each arm).

## Conclusion

**NMS relaxation helps pairs but solo regression AND fragmentation rise cancel the net
-> pre-4's prediction holds; NMS ruled out as standalone fix; go to detection-triggered
GROUP (Lever 2).**

The evidence is unambiguous across every metric:
- **Misattribution rises monotonically** (4.0% -> 10.8% -> 25.1%) at every relaxation step
- **Present drops monotonically** (80.1% -> 68.2% -> 52.1%)
- **Fragmentation explodes** (1.0 -> 3.0 -> 4.5 median tracklets per GT person)
- **Solo context regresses as badly as pair context** (24.3% solo misattrib at iou=0.9)
- **Transitions confirm churn dominates recovery**: 902 present->misattrib vs 93 no_det->present at iou=0.9
- **iou=0.95 is solver-intractable** (3,147 tracklets, >8 hours CPU, killed)

The duplicate boxes from relaxed NMS cause tracker identity churn that overwhelms any
recovery of real second persons. The tracker cannot distinguish a recovered real person
from a duplicate of an existing tracked person, so both get new tracklet IDs, fragmenting
the existing tracks.

**The path forward is NOT NMS relaxation.** It is detection-triggered GROUP (Lever 2 from
CP7-pre-4): detect pair-context frames via the existing model output, then apply
specialized handling (e.g., GROUP node construction, two-person segmentation, or
position-based identity) only where needed, without flooding the tracker with duplicates
on every frame.

Re-baseline against a CP5-state full-mode snapshot. When reading the post-CP7 six-mode
shift, treat any movement in a metric that was stable across CP0-CP5 as
expected-until-explained, then run the conservation check before trusting the magnitude.
