# CP7-pre: ReID Inventory

*Generated 2026-05-21. Read-only investigation. No code changes.*

CP5 collapsed d3_dropped; present_misattributed (59-66%) is now the dominant Stage D
failure mode. This document inventories ReID-relevant machinery that already exists
and evaluates whether it can separate same-clip athletes.

---

## 1. HSV Histogram Signal -- Extraction

### 1.1 Production Location and Schema

**Per-detection histograms:** `stage_A/color_histograms.parquet`
- Produced in `src/bjj_pipeline/stages/detect_track/outputs.py` (lines 288-312, 529-541)
- Schema: `frame_index`, `track_id`, `is_isolated`, `crop_method`, `hist_0`..`hist_143`
- Color space: HSV (H x S, Value discarded). 18 H-bins x 8 S-bins = 144 bins
- Normalization: sum = 1.0 per histogram
- Computed in `src/bjj_pipeline/stages/detect_track/histogram.py`

**Per-tracklet summaries:** `stage_A/tracklet_histogram_summaries.parquet`
- Produced in `outputs.py` (lines 554-598)
- Schema: `tracklet_id`, `camera_id`, `clip_id`, `n_isolated_frames`,
  `crop_method_distribution_json`, `hist_0`..`hist_143`
- Values: mean of per-frame histograms from isolated frames only, re-normalized

### 1.2 Torso-Crop vs Center-Bbox Fallback

From `histogram.py` lines 105-132:

```python
def extract_histogram(frame_bgr, bbox, keypoints, is_isolated, min_kp_conf=0.3):
    if not is_isolated:
        return None, "not_isolated"
    # Primary: pose-guided torso crop
    if keypoints is not None:
        crop = _torso_crop_from_keypoints(frame_bgr, keypoints, min_kp_conf=min_kp_conf)
        if crop is not None and crop.size > 0:
            return compute_hsv_histogram(crop), "torso_pose"
    # Fallback: center-cropped bbox (60% center)
    crop = _center_crop_from_bbox(frame_bgr, bbox)
    if crop is not None and crop.size > 0:
        return compute_hsv_histogram(crop), "center_bbox"
    return None, "crop_failed"
```

**On the detection-only model** (`bjj-detect-all-cameras`): keypoints is always None.
The torso-crop path is never reached. Every isolated detection uses the center-bbox
fallback. Confirmed in artifacts: 0% `torso_pose`, 100% `center_bbox` across all cameras.

### 1.3 Histogram Distinguishability -- Make-or-Break Test

**Test setup:** 8 key tracklets from J_EDEw (t1, t2, t3, t5, t9, t94, t108, t111),
including carriers dropped pre-CP5 (t3, t5, t111) and kept carriers (t1, t2, t108).
All have abundant isolated frames (294-3902).

**Pairwise Bhattacharyya distances:**

|        |    t1 |  t108 |  t111 |    t2 |    t3 |    t5 |    t9 |   t94 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| t1     | 0.000 | 0.138 | 0.191 | 0.052 | 0.160 | 0.173 | 0.088 | 0.142 |
| t108   | 0.138 | 0.000 | 0.257 | 0.115 | 0.196 | 0.134 | 0.127 | 0.087 |
| t111   | 0.191 | 0.257 | 0.000 | 0.244 | 0.437 | 0.052 | 0.275 | 0.124 |
| t2     | 0.052 | 0.115 | 0.244 | 0.000 | 0.081 | 0.183 | 0.087 | 0.119 |
| t3     | 0.160 | 0.196 | 0.437 | 0.081 | 0.000 | 0.371 | 0.119 | 0.218 |
| t5     | 0.173 | 0.134 | 0.052 | 0.183 | 0.371 | 0.000 | 0.192 | 0.042 |

**GT person coverage per tracklet** (from CP5 trace, J_EDEw): every key tracklet covers
10-14 GT persons. Dominant GT person accounts for only 18-35% of frames. Tracklets
physically drift across multiple people due to tracking fragmentation + grappling.

**Verdict: HSV histograms cannot separate same-clip athletes.**

Distances are compressed into the 0.04-0.44 range. The closest pairs (t5/t94: 0.042,
t1/t2: 0.052, t111/t5: 0.052) are near-identical despite covering different GT persons.
The maximum distance in this set (t111/t3: 0.437) is modest.

**Broader distribution (all J_EDEw tracklet pairs with >=10 isolated frames):**

| Camera | Tracklets | Pairs | Median dist | <0.1 | <0.2 | >0.5 |
|--------|-----------|-------|-------------|------|------|------|
| FP7oJQ | 94 | 4,371 | 0.198 | 15.9% | 50.6% | 5.8% |
| J_EDEw | 82 | 3,321 | 0.292 | 7.8% | 29.8% | 21.8% |
| PPDmUg | 23 | 253 | 0.166 | 22.1% | 65.2% | 1.6% |

FP7oJQ and PPDmUg are worse: >50% and >65% of pairs have distance <0.2. The center-bbox
crop on overhead fisheye footage captures mostly mat + gi fabric, which is dominated by
similar hues across all athletes. HSV histograms from center-bbox crops are a weak signal
for intra-camera identity.

**Root cause:** (a) overhead camera angle means crops are mostly torso/back, not face;
(b) center-bbox fallback includes mat and background pixels; (c) gi/rashguard colors
are not diverse enough at 18-bin H resolution to separate 14 people on one mat.

---

## 2. HSV Histogram Signal -- Consumption

### 2.1 Cross-Camera Evidence Builder

`src/bjj_pipeline/stages/stitch/cross_camera_evidence.py`,
`build_cross_camera_histogram_evidence()` (lines 545-703):

1. Loads `tracklet_histogram_summaries.parquet` per clip
2. Maps tracklet_id -> person_id via D4 `person_tracks.parquet`
3. Computes weighted-average per-person histograms (by n_isolated_frames)
4. Pairwise cross-camera Bhattacharyya distance for all person pairs
5. Emits `cost_modifiers.cross_camera_pairs` array and `tag_propagations` dict

### 2.2 Is cost_modifiers consumed intra-camera?

**No. cost_modifiers is computed but not consumed by the D3 solver.**

The data flow:
1. `cross_camera_evidence.py` builds `cost_modifiers` dict
2. `services/processor/processor.py` (lines 732-733) places it into `constraints_overlay`
3. `session_d_run.py` (lines 536-559) merges overlay into constraints before D3
4. `d3_ilp2.py` (lines 1483-1488) reads and **logs** it:
   ```python
   cost_modifiers = (constraints or {}).get("cost_modifiers", {})
   if cost_modifiers:
       n_pairs = len(cost_modifiers.get("cross_camera_pairs", []))
       logger.info("CP20: cost_modifiers present with {} cross-camera pairs", n_pairs)
   ```
5. **No further reference.** The variable is never used in objective construction.

**Histograms are used only for cross-camera tag propagation** (similarity >= threshold
+ one side tagged -> propagate). They do NOT feed intra-camera D3 cost in any way.

Searched: `d1_graph_build.py`, `d2_constraints.py`, `costs.py`, `solver.py`,
`d3_common.py`, `d3_ilp2.py` -- zero histogram references in any intra-camera path.

---

## 3. Isolation Gate

### 3.1 Is is_isolated populated on the detection-only model?

**Yes.** The isolation gate (`src/bjj_pipeline/stages/detect_track/isolation.py`) has
4 heuristics (H1-H4). H4 (torso keypoint plausibility) requires keypoints but is
**gated by `require_keypoints` config**, which is `false` for the detection-only model.
With H4 skipped, H1 (aspect ratio >= 0.8), H2 (pairwise IoU < 0.3), and H3 (bbox area
bounds) still run. Detections passing all three are flagged `is_isolated = True`.

### 3.2 Isolation Distribution Per Camera

| Camera | Total detections | Isolated | % Isolated |
|--------|-----------------|----------|------------|
| FP7oJQ | 44,381 | 23,148 | 52.2% |
| J_EDEw | 49,160 | 24,858 | 50.6% |
| PPDmUg | 19,243 | 7,269 | 37.8% |

**Crop method:** 100% `center_bbox` (no `torso_pose`, no `crop_failed`) on all cameras.

### 3.3 Per-Tracklet Isolated Frame Distribution

| Camera | Tracklets | w/ >=10 iso frames | Median iso frames | Max iso frames |
|--------|-----------|-------------------|------------------|----------------|
| FP7oJQ | 134 | 94 (70.1%) | 30 | 3,537 |
| J_EDEw | 119 | 82 (68.9%) | 49 | 3,902 |
| PPDmUg | 34 | 23 (67.6%) | 24 | 1,269 |

**All tracklets have >=1 isolated frame** (min=1 across all cameras). ~70% of tracklets
have >=10 isolated frames, which is the `min_isolated_frames` threshold for histogram
evidence. Long tracklets (the carriers that matter for identity) have hundreds to thousands
of isolated frames.

**Conclusion:** The isolation gate provides a usable pool of clean appearance samples.
~50% of frames are isolated (FP7oJQ/J_EDEw), dropping to 38% on PPDmUg (more crowded
mat). The bottleneck is not sample availability but signal quality (Section 1.3).

---

## 4. D3 Injection Hook

### 4.1 Current Penalty Infrastructure

The D3 solver (`d3_ilp2.py`) has one active cost mechanism:
- **Explain-or-penalize** (lines 1908-2044): per-SINGLE_TRACKLET node, `max(base_floor,
  per_frame * n_frames)` penalty for dropping a tracklet. Wired from config via
  `solver.py` lines 64-84.

### 4.2 Constraints Overlay (CP17)

`session_d_run.py` lines 536-560: before D3, a `constraints_overlay` dict is merged
into the D2 constraints JSON. This is the existing injection point for cross-camera
evidence. Currently carries:
- `cross_camera_evidence` (tag must-links)
- `corroboration_miss_multiplier`
- `cost_modifiers` (histogram, logged but unconsumed)

### 4.3 Where an Appearance Cost Would Plug In

**Smallest wiring change for an intra-camera appearance distance:**

The D3 objective is built in `d3_ilp2.py` lines 2031-2078. Edge costs are read from
`scaled_cost` dict (keyed by edge_id). To add appearance distance:

1. **Compute appearance distances** per edge during D2 (in `costs.py`): for each MERGE
   edge connecting two tracklets, look up their histogram summaries and compute
   Bhattacharyya distance. Store as `appearance_cost_modifier[edge_id] = delta`.

2. **Inject into constraints** via the existing overlay mechanism or directly in
   `d2_constraints.py`.

3. **Apply in D3 objective** (lines 2033-2037): add the modifier to edge cost:
   ```python
   ci = int(scaled_cost.get(str(eid), 0))
   appearance_mod = appearance_cost_modifiers.get(str(eid), 0)
   ci_adjusted = ci + appearance_mod
   ```

**Estimated change:** ~5 lines in `d3_ilp2.py` + ~15 lines in `costs.py` to compute
and emit the modifiers. The overlay mechanism already exists; no new plumbing needed.

**However:** Given Section 1.3's finding that HSV histograms cannot distinguish same-clip
athletes (median distance 0.17-0.29, 30-65% of pairs below 0.2), wiring histogram
distances into D3 would add noise, not signal. A stronger appearance descriptor is
needed before this hook is useful.

---

## 5. Learned-Embedding Cost (Scoping Only)

### 5.1 BoT-SORT ReID: Already Wired, Just Disabled

`src/bjj_pipeline/stages/detect_track/tracker.py` lines 39-84:

```python
class BotSortTracker:
    def __init__(self, *, with_reid: bool, params=None):
        self.with_reid = bool(with_reid)
        ...
        cfg["with_reid"] = self.with_reid
        cfg["reid_weights"] = str(cfg.get("reid_weights") or "")
        self._tracker = BotSort(**cfg)
```

Config: `stages.stage_A.tracker.with_reid` (default `false` in `models.py` line 146).
Setting `with_reid: true` in config and providing `reid_weights` path is sufficient --
no code changes required. BoxMOT handles embedding extraction internally during
`tracker.update()`.

### 5.2 ReID Placeholder

`src/bjj_pipeline/stages/stitch/reid.py` is a single-line placeholder:
```python
"""Role: appearance descriptors and light ReID hooks (placeholder)."""
```

No functionality. Purpose slot exists for future CP7 work.

### 5.3 Latency Impact

**Current budget:** CoreML on ANE runs the detection model at 78.9 fps. ANE is
saturated by a single stream (2 workers = 0.54x throughput, per CLAUDE.md CP22).

**ReID model pass:** BoxMOT's default ReID is OSNet (~2M params). On M1 Air:
- **MPS:** ~15-25 fps for a batch of crops (estimated, model-dependent)
- **CPU:** ~5-10 fps
- **ANE:** Not available (OSNet has no CoreML export in BoxMOT)

**Impact on Phase 1:** Currently detection runs at ~79 fps. Adding a ReID forward pass
on every detection's crop would add ~40-100ms per frame (at 8-10 detections/frame on
MPS, more on CPU). This would reduce effective throughput from ~79 fps to ~10-20 fps
on MPS, or ~5-8 fps on CPU. A 4-8x slowdown.

**Mitigation strategies** (scope only, no implementation):
1. Run ReID only on isolated frames (~50% of frames) -> 2-4x slowdown instead
2. Run ReID only on tracklet-boundary frames (first/last N frames per tracklet)
3. Run ReID as a post-hoc pass on tracklet crops (decouple from Phase 1 real-time)
4. Export OSNet to CoreML and share ANE -- but CP22 showed ANE saturated by one stream

### 5.4 Architecture Decision: Where ReID Belongs

Two options:

**Option A -- Phase 1 (BoT-SORT with_reid=true):** Embeddings improve tracker
association in real-time. Reduces tracklet fragmentation at source. But 4-8x slower,
and BoT-SORT's internal embedding usage is opaque (hard to extract for downstream D3).

**Option B -- Post-hoc (Stage A.5 or D-time):** Run ReID on isolated-frame crops
after tracking. Extract embeddings per tracklet, store as sidecar parquet. Feed into
D3 as appearance distance. Full control over how embeddings are used. Decoupled from
Phase 1 latency.

Option B is more aligned with the pipeline architecture (stages communicate via
contracts, not internal state). It also avoids the ANE saturation problem since the
ReID pass can run asynchronously.

---

## Summary and Recommendations for CP7

### What exists and works:
- Isolation gate: 50% of frames are clean samples (adequate pool)
- HSV histogram extraction: fully operational, 144-bin H x S descriptors per tracklet
- Cross-camera histogram evidence: Bhattacharyya distance, tag propagation
- D3 injection hook: constraints overlay mechanism + cost_modifiers placeholder
- BoT-SORT ReID: fully wired, one config flag to enable

### What doesn't work:
- **HSV histograms cannot separate same-clip athletes.** Median Bhattacharyya distance
  is 0.17-0.29 across cameras; 30-65% of tracklet pairs have distance <0.2. The
  center-bbox crop from overhead fisheye captures too much mat/background and too little
  discriminative clothing detail. This is the critical finding.

### What CP7 needs:
1. **A stronger appearance descriptor.** HSV histograms are insufficient. Options:
   - Learned ReID embeddings (OSNet, CLIP, or domain-fine-tuned)
   - Torso-crop quality improvement (pose model would help, but adds latency)
   - Color histogram with background subtraction (remove mat pixels)
2. **Post-hoc architecture** (Option B above) to avoid Phase 1 latency regression
3. **Intra-camera D3 wiring** of appearance distances (the hook exists, needs signal)

The existing HSV histogram infrastructure is useful scaffolding (extraction pipeline,
isolation gate, parquet contracts, cross-camera evidence builder) but the signal itself
is too weak for intra-camera identity separation. CP7 should focus on replacing or
augmenting the descriptor, not on wiring the existing one deeper.
