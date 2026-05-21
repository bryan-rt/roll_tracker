# CP7-pre-4: Detection Under-Segmentation Mechanism + GROUP-Path Reachability

*Generated 2026-05-21. Read-only investigation. No code changes.*

CP7-pre-3 established that 70-78% of misattribution is detection under-segmentation
(one detection spanning a grappling pair). This document determines the mechanism
(NMS suppression vs under-detection) and confirms whether Stage D's GROUP path
can reach the affected frames.

---

## Frame-Space Verification

The eval trace `frame_idx` and `detections.parquet` `frame_index` are both
**clip-relative, 0-based** — they directly index the source video frames. The eval
clips are single-clip (not session-aggregated), so no session offset applies. The
Part B re-inference uses `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, frame_idx)` on
the same source videos that produced the original detections.

---

## Part A: GROUP-Path Reachability

### A.1 Lifecycle Adjacency

For each under-seg misattributed frame, is any tracklet's start or end frame
within W frames?

| Camera | Under-seg frames | W=5 | W=15 | W=30 |
|--------|-----------------|-----|------|------|
| FP7oJQ | 301 | 48.8% | 84.7% | 100% |
| J_EDEw | 301 | 58.8% | 87.7% | 97.7% |
| PPDmUg | 295 | 30.8% | 61.4% | 83.4% |

At W=5 (tight window), only 31-59% of under-seg frames have a nearby lifecycle event.
At W=30, coverage rises to 83-100% — but that's because with 73-251 tracklets per
clip, lifecycle events are dense. **Lifecycle adjacency is incidental, not causal.**

### A.2 GROUP Node Span Coverage

| Camera | Under-seg frames | Inside any GROUP span |
|--------|------------------|-----------------------|
| FP7oJQ | 301 | 100.0% |
| J_EDEw | 301 | 99.3% |
| PPDmUg | 295 | 100.0% |

99-100% of under-seg frames fall inside some GROUP node's [start_frame, end_frame].
This sounds like GROUP coverage is excellent — but it's **misleading**. GROUP nodes
have wide spans (top tracklets span 1000-4500 frames), and with 62-168 GROUP nodes
per camera their spans tile the entire clip.

### A.3 Matched Tracklet GROUP Participation

The real question: does the *specific tracklet assigned to the pair-box* participate
in a GROUP node?

| Camera | Unique under-seg tracklets | In GROUP | Long carriers (>1000 fr) |
|--------|---------------------------|----------|--------------------------|
| FP7oJQ | 19 | 84% | 11% |
| J_EDEw | 90 | 74% | 12% |
| PPDmUg | 33 | 94% | 18% |

74-94% of the affected tracklets DO participate in GROUP nodes. But this still doesn't
mean the GROUP mechanism can address the pair-box problem, because:

### A.4 Verdict: GROUP is structurally irrelevant

The GROUP mechanism operates on **tracklet lifecycle events** (a tracklet ending =
merge trigger; a new tracklet starting = split trigger). It routes flow through
capacity-2 nodes when two tracklets share a world-coordinate neighborhood at a
boundary frame.

The under-seg pair-box problem is fundamentally different:
- **Both tracklets are alive** simultaneously (CP7-pre-2 Q3: 95-99% of fragmentation
  events are ID switches between concurrent tracklets)
- **The pair-box is a single detection** assigned to one tracklet — there's no
  lifecycle event at that frame
- **No CONTINUE edge** connects concurrent tracklets (CONTINUE requires
  successor.start_frame > predecessor.end_frame)

Even though GROUP spans technically cover the frame and the tracklet technically
participates in GROUP nodes elsewhere in its life, **the pair-box frame itself
generates no lifecycle event and is invisible to D1's identity-routing topology.**
The detection input is wrong; no amount of downstream routing can split a single
detection into two people.

---

## Part B: NMS-Suppressed vs Under-Detection

### B.1 Where NMS Lives

**NMS is runtime-tunable in Python**, not baked into the CoreML export.

| Component | Detail |
|-----------|--------|
| CoreML model format | `mlProgram` (NOT a pipeline with NMS stage) |
| NMS application | Ultralytics Python postprocessing (`utils/nms.py`) |
| Runtime `conf` | **0.45** (from `configs/default.yaml`) |
| Runtime `iou` (NMS-IoU) | **0.7** (ultralytics default, not overridden in pipeline config) |
| Agnostic NMS | False |
| Max detections | 300 |

The CoreML `.mlpackage` contains only the neural network; NMS happens after
inference in `ultralytics.utils.nms.non_max_suppression()`. **Both `conf` and
`iou` thresholds are simple keyword arguments to `model.predict()`** — changing
them requires zero code changes and zero model re-export.

### B.2 Deterministic Sample

N=10 per camera, selected as the first 10 unique under-seg frames sorted by
frame_idx. Re-inference with:
- **Production:** conf=0.45, iou=0.7 (current pipeline settings)
- **Relaxed NMS:** conf=0.45, iou=0.95 (near-disabled NMS)
- **Low conf + relaxed:** conf=0.1, iou=0.95 (maximum sensitivity)

### B.3 Results: All 30 Frames are NMS-Suppressed

| Camera | Frames | NMS-suppressed (conf=0.45) | NMS-suppressed (conf=0.1) | Under-detection |
|--------|--------|----------------------------|---------------------------|-----------------|
| FP7oJQ | 10 | **10** | 0 | **0** |
| J_EDEw | 10 | **10** | 0 | **0** |
| PPDmUg | 10 | **9** | 1 | **0** |

**30/30 frames: NMS-suppressed. 0/30 under-detection.**

The second person's box IS proposed by the detector at conf >= 0.45 in 29/30 cases.
In 1 case (PPDmUg frame 90) the second box needs conf lowered to 0.1 to appear.
In NO case does the detector fail to propose a second box entirely.

### B.4 All Recovered Boxes Are Nested

Within the NMS-suppressed frames:

| Camera | Nested (IoU >= 0.5 with pair-box) | Cleanly separable |
|--------|----------------------------------|-------------------|
| FP7oJQ | 10 | 0 |
| J_EDEw | 10 | 0 |
| PPDmUg | 10 | 0 |

**30/30 nested, 0/30 separable.** The recovered second box has high IoU with the
pair-box (median 0.98 for FP7oJQ/J_EDEw, 0.98 for PPDmUg) — the two people's
boxes are stacked/overlapping, not side-by-side. This is the expected geometry for
overhead cameras viewing grappling pairs: two bodies are physically on top of each
other, producing heavily overlapping bounding boxes.

### B.5 Detection Count Gains from Relaxed NMS

| Camera | Production (mean) | Relaxed iou=0.95 (mean) | Low conf+relaxed (mean) | GT persons/frame |
|--------|-------------------|-------------------------|-------------------------|------------------|
| FP7oJQ | 10.3 | 12.4 (+2.1) | 17.0 (+6.7) | 14 |
| J_EDEw | 12.3 | 13.6 (+1.3) | 18.3 (+6.0) | 14 |
| PPDmUg | 4.2 | 5.1 (+0.9) | 8.2 (+4.0) | 6-8 |

Relaxing NMS to iou=0.95 recovers ~1-2 boxes per frame (closing part of the
detection deficit). Lowering conf to 0.1 recovers ~4-7 more (but most of these
are likely false positives or duplicate boxes rather than new people).

### B.6 Per-Frame Detail (Appendix)

**FP7oJQ** (frames 0-9, stride 1):

| Frame | GT | 2nd GT | N_gt | N_prod | N_rel | N_low | IoU_2nd_rel | Class | Sub | IoU_rec_pair |
|-------|----|----|------|--------|-------|-------|-------------|-------|-----|-------------|
| 0 | 15 | 17 | 14 | 11 | 13 | 18 | 0.878 | NMS_SUP | nest | 0.993 |
| 1 | 15 | 14 | 14 | 11 | 14 | 17 | 0.984 | NMS_SUP | nest | 0.986 |
| 2 | 15 | 14 | 14 | 11 | 14 | 17 | 0.985 | NMS_SUP | nest | 0.989 |
| 3 | 15 | 14 | 14 | 10 | 13 | 17 | 0.987 | NMS_SUP | nest | 0.985 |
| 4 | 15 | 14 | 14 | 10 | 12 | 17 | 0.980 | NMS_SUP | nest | 0.987 |
| 5 | 15 | 14 | 14 | 10 | 12 | 17 | 0.962 | NMS_SUP | nest | 0.991 |
| 6 | 15 | 19 | 14 | 10 | 11 | 16 | 0.927 | NMS_SUP | nest | 0.965 |
| 7 | 14 | 19 | 14 | 9 | 11 | 17 | 0.964 | NMS_SUP | nest | 0.964 |
| 8 | 15 | 19 | 14 | 11 | 12 | 17 | 0.955 | NMS_SUP | nest | 0.975 |
| 9 | 15 | 19 | 14 | 10 | 12 | 17 | 0.926 | NMS_SUP | nest | 0.966 |

**J_EDEw** (frames 0-90, stride 10):

| Frame | GT | 2nd GT | N_gt | N_prod | N_rel | N_low | IoU_2nd_rel | Class | Sub | IoU_rec_pair |
|-------|----|----|------|--------|-------|-------|-------------|-------|-----|-------------|
| 0 | 15 | 20 | 14 | 12 | 14 | 18 | 0.901 | NMS_SUP | nest | 0.977 |
| 10 | 14 | 16 | 14 | 13 | 13 | 18 | 0.894 | NMS_SUP | nest | 0.969 |
| 20 | 14 | 24 | 14 | 13 | 13 | 21 | 0.956 | NMS_SUP | nest | 0.993 |
| 30 | 14 | 16 | 14 | 13 | 14 | 18 | 0.903 | NMS_SUP | nest | 0.960 |
| 40 | 15 | 25 | 14 | 13 | 14 | 20 | 0.951 | NMS_SUP | nest | 0.984 |
| 50 | 22 | 24 | 14 | 12 | 15 | 22 | 0.927 | NMS_SUP | nest | 0.979 |
| 60 | 16 | 18 | 14 | 12 | 14 | 19 | 0.675 | NMS_SUP | nest | 0.840 |
| 70 | 14 | 18 | 14 | 12 | 13 | 15 | 0.714 | NMS_SUP | nest | 0.974 |
| 80 | 14 | 19 | 14 | 12 | 13 | 15 | 0.961 | NMS_SUP | nest | 0.984 |
| 90 | 14 | 19 | 14 | 11 | 13 | 17 | 0.974 | NMS_SUP | nest | 0.988 |

**PPDmUg** (frames 0-90, stride 10):

| Frame | GT | 2nd GT | N_gt | N_prod | N_rel | N_low | IoU_2nd_rel | Class | Sub | IoU_rec_pair |
|-------|----|----|------|--------|-------|-------|-------------|-------|-----|-------------|
| 0 | 4 | 5 | 6 | 5 | 5 | 9 | 0.802 | NMS_SUP | nest | 0.564 |
| 10 | 2 | 4 | 6 | 5 | 5 | 8 | 0.974 | NMS_SUP | nest | 0.975 |
| 20 | 7 | 2 | 6 | 4 | 5 | 10 | 0.941 | NMS_SUP | nest | 0.994 |
| 30 | 4 | 5 | 6 | 4 | 5 | 9 | 0.761 | NMS_SUP | nest | 0.987 |
| 40 | 7 | 6 | 6 | 4 | 5 | 7 | 0.966 | NMS_SUP | nest | 0.527 |
| 50 | 7 | 2 | 6 | 4 | 5 | 9 | 0.958 | NMS_SUP | nest | 0.984 |
| 60 | 2 | 5 | 6 | 4 | 5 | 8 | 0.732 | NMS_SUP | nest | 0.982 |
| 70 | 4 | 7 | 6 | 4 | 5 | 7 | 0.971 | NMS_SUP | nest | 0.992 |
| 80 | 5 | 7 | 6 | 4 | 6 | 8 | 0.975 | NMS_SUP | nest | 0.988 |
| 90 | 4 | 5 | 6 | 4 | 5 | 7 | 0.455 | NMS_LOW | nest | 0.609 |

---

## Fork-Closer

**NMS-suppression dominant, runtime-tunable, but ALL nested.**

The detector proposes the second person's box. NMS kills it because the pair-box
and the individual box overlap heavily (IoU > 0.5, typically > 0.9). The fix is
not a simple `iou` threshold change, because:

1. **Raising NMS-IoU globally** (e.g. 0.7 -> 0.95) would recover the suppressed
   boxes for grappling pairs, but would also **retain duplicate boxes for
   non-grappling people**, significantly increasing false positives. The
   production run already shows +2.1 boxes/frame at iou=0.95 on FP7oJQ (14 GT,
   10.3 prod, 12.4 relaxed) — some of those extra boxes are duplicates.

2. **The recovered boxes are nested**, not separable. Two overlapping boxes for
   the same spatial region create severe downstream problems: the tracker will
   either merge them into one tracklet (same as pair-box) or oscillate between
   them (worse than pair-box). Simply having two boxes doesn't help if they
   can't be cleanly separated.

3. **The real fix is at the detection architecture level**: the model needs to
   learn to produce tight per-person boxes for grappling pairs, not one
   encompassing box plus a suppressed duplicate. This is a detector training
   problem (pair-separation supervision), not a post-processing tweak.

### Recommended CP7 direction

| Priority | Intervention | Expected impact | Effort |
|----------|-------------|-----------------|--------|
| 1 | **Training data with per-person annotations in grappling pairs** | Teaches detector to propose tight individual boxes instead of pair-boxes. Addresses 70-78% of misattribution at source. | Medium: annotate grappling frames with per-person boxes, retrain |
| 2 | **NMS-IoU raise as stopgap** (e.g. 0.7 -> 0.85) with a grappling-aware post-filter | Recovers some suppressed boxes; post-filter removes duplicates for solo people. Partial fix. | Low: config change + ~50 lines post-filter |
| 3 | **Instance segmentation / SAM** for pair separation | Mask-level separation of overlapping people. Stage B (SAM) was deferred for POC but is the natural home. | High: requires mask model integration |
| 4 | **Stitch/canonical improvement** (D3 identity) | Fixes the 21-29% stitch-driven misattribution. Orthogonal to pair-separation. | Medium: ReID embeddings in D3 cost |

The nested-box finding confirms that the future home for a "detection-triggered
GROUP" mechanism (detecting that a single box covers two people and splitting the
identity downstream) is the right architecture for the grappling case specifically.
But the primary lever is upstream: get the detector to produce individual boxes.
