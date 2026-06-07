# CP-TAG-3 Carrier Evidence: Tagged Tracklet Geometry

## Context

In vid2 (J_EDEw-20260318-200246), tag_id=1 is observed at frame(s) [1781].
Tagged tracklet(s): ['t139'].

## Comparison with Prior Diagnostic

The prior cross-tracklet diagnostic (pre-CP5, real gym_id) found:
- t99 (862 frames) and t143 (17 frames, nested bbox inside t99)
- Tag obs captured by BOTH t99 and t143 at frames 1781-1782
- t99 DROPPED by solver; tag identity routed to wrong person via t143

**Current state (v2 model, _eval_gt, post-CP5/CP-SPLIT-1):**
- Tagged tracklet(s): ['t139']
- Nesting detected at observation frames: False
- The old t99/t143 nested detection no longer occurs
- Carrier-selection rule question may need reframing

## Tagged Tracklet Details

### t139
- Length: 2747 frames [1571, 4458]
- GROUP segments: 0, SOLO: 1
- Person IDs: ['p0001', 'p0004', 'p0005', 'p0008', 'p0010', 'p0012', 'p0013', 'p0015', 'p0016', 'p0018', 'p0019', 'p0020', 'p0021', 'p0022', 'p0033']
- Person ID transitions: 2277

## Observation Frame Geometry

### Frame 1781
- Detections: 11
  - t103: [1.0, 282.0, 74.0, 425.0] area=10439
  - t126: [2.0, 265.0, 68.0, 358.0] area=6138
  - t134: [985.0, 158.0, 1040.0, 235.0] area=4235
  - t107: [22.0, 255.0, 86.0, 376.0] area=7744
  - t88: [354.0, 236.0, 410.0, 312.0] area=4256
  - t85: [409.0, 225.0, 484.0, 297.0] area=5400
  - t117: [467.0, 109.0, 520.0, 228.0] area=6307
  - t139: [617.0, 124.0, 678.0, 295.0] area=10431 **[TAGGED]**
  - t156: [679.0, 282.0, 720.0, 605.0] area=13243
  - t5: [703.0, 99.0, 760.0, 243.0] area=8208
  - t148: [941.0, 276.0, 1094.0, 600.0] area=49572
- Tag center: (647.5, 209.5) on t139
- No overlapping detections with tagged tracklet

