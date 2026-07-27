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

### (b) tag:1 person_ids: 1 assignments
- p0022: frames [632, 9029], spans_boundary=True

### (c) tag:1 assignment count: 1

### (d) Tagged tracklet person_id transitions
- `J_EDEw-20260318-200015:t366`: 1114 transitions, person_ids=['p0001', 'p0007', 'p0008', 'p0013', 'p0017', 'p0019', 'p0020', 'p0021', 'p0025', 'p0027', 'p0028', 'p0030', 'p0031', 'p0032']
  - p0001: 584 frames
  - p0028: 314 frames
  - p0025: 275 frames
  - p0017: 273 frames
  - p0013: 170 frames
  - p0032: 147 frames
  - p0008: 141 frames
  - p0027: 119 frames
  - p0019: 67 frames
  - p0031: 65 frames
  - p0030: 33 frames
  - p0007: 19 frames
  - p0021: 8 frames
  - p0020: 6 frames
- `J_EDEw-20260318-200246:t139`: 2676 transitions, person_ids=['p0002', 'p0005', 'p0007', 'p0008', 'p0013', 'p0014', 'p0017', 'p0019', 'p0022', 'p0028', 'p0029', 'p0031', 'p0034', 'p0035', 'p0039']
  - p0014: 676 frames
  - p0034: 543 frames
  - p0008: 516 frames
  - p0039: 480 frames
  - p0005: 425 frames
  - p0019: 321 frames
  - p0035: 303 frames
  - p0031: 251 frames
  - p0029: 194 frames
  - p0028: 176 frames
  - p0007: 173 frames
  - p0017: 149 frames
  - p0002: 148 frames
  - p0022: 62 frames
  - p0013: 11 frames
