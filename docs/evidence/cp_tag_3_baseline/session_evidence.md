# CP-TAG-3 Session-Level Evidence

## Session Scope

Two-clip, single-camera (J_EDEw only). No Tier 3 cross-camera evidence.
This is a controlled experiment for the clip-boundary question.
**Post-CP-TAG-4 re-measure gate MUST reuse this exact session scope**
for apples-to-apples comparison.

- Clip 1: J_EDEw-20260318-200015 (offset 0 frames)
- Clip 2: J_EDEw-20260318-200246 (offset 4530 frames)
- FPS: 30.0
- Session output: `outputs/_eval_gt/sessions/2026-03-18/cp_tag_3_baseline`

## Results

### (a) Tagged tracklet drop status
- Session-level tagged tracklets: ['J_EDEw-20260318-200015:t366', 'J_EDEw-20260318-200246:t139']
- `J_EDEw-20260318-200015:t366`: KEPT (2221 rows)
- `J_EDEw-20260318-200246:t139`: KEPT (4428 rows)

### (b) tag:1 person_ids: 4 assignments
- p0015: frames [22, 8758], spans_boundary=True
- p0024: frames [724, 9029], spans_boundary=True
- p0031: frames [2638, 9029], spans_boundary=True
- p0032: frames [2692, 9029], spans_boundary=True

### (c) tag:1 assignment count: 4

### (d) Tagged tracklet person_id transitions
- `J_EDEw-20260318-200015:t366`: 1125 transitions, person_ids=['p0003', 'p0005', 'p0008', 'p0011', 'p0015', 'p0018', 'p0023', 'p0024', 'p0025', 'p0026', 'p0028', 'p0029', 'p0031', 'p0032']
  - p0003: 584 frames
  - p0015: 453 frames
  - p0029: 216 frames
  - p0028: 170 frames
  - p0032: 147 frames
  - p0018: 133 frames
  - p0031: 126 frames
  - p0026: 116 frames
  - p0005: 116 frames
  - p0024: 57 frames
  - p0008: 44 frames
  - p0023: 38 frames
  - p0025: 15 frames
  - p0011: 6 frames
- `J_EDEw-20260318-200246:t139`: 2680 transitions, person_ids=['p0003', 'p0005', 'p0013', 'p0014', 'p0017', 'p0023', 'p0024', 'p0025', 'p0034', 'p0037', 'p0039', 'p0045']
  - p0024: 857 frames
  - p0037: 801 frames
  - p0025: 598 frames
  - p0045: 509 frames
  - p0014: 478 frames
  - p0013: 331 frames
  - p0023: 308 frames
  - p0003: 211 frames
  - p0039: 176 frames
  - p0017: 127 frames
  - p0034: 21 frames
  - p0005: 11 frames
