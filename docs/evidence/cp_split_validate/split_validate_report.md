# CP-SPLIT-VALIDATE: GT-Validate D0.5 Splits

## Phase 0: Reconstructed Split Counts

| Clip | Pre-V T1 | Pre-V T2 | Pre-V T3 | Post-V T1 | Post-V T2 | Post-V T3 |
|------|----------|----------|----------|-----------|-----------|-----------|
| J_EDEw-20260318-200015 | 1 | 165 | 25 | 1 | 158 | 149 |
| J_EDEw-20260318-200246 | 2 | 104 | 46 | 2 | 92 | 260 |

## Phase 1: GT Validation

### pre_v
| Clip | Tier | Correct | Spurious | Undecidable | Precision |
|------|------|---------|----------|-------------|-----------|
| J_EDEw-20260318-200015 | tier1_speed_cap | 0 | 0 | 1 | 0 |
| J_EDEw-20260318-200015 | tier2_kinematic_spike | 19 | 74 | 72 | 0.204 |
| J_EDEw-20260318-200015 | tier3_histogram | 3 | 12 | 10 | 0.2 |
| J_EDEw-20260318-200246 | tier1_speed_cap | 1 | 0 | 1 | 1.0 |
| J_EDEw-20260318-200246 | tier2_kinematic_spike | 6 | 95 | 3 | 0.059 |
| J_EDEw-20260318-200246 | tier3_histogram | 2 | 43 | 1 | 0.044 |

### post_v
| Clip | Tier | Correct | Spurious | Undecidable | Precision |
|------|------|---------|----------|-------------|-----------|
| J_EDEw-20260318-200015 | tier1_speed_cap | 0 | 0 | 1 | 0 |
| J_EDEw-20260318-200015 | tier2_kinematic_spike | 19 | 68 | 71 | 0.218 |
| J_EDEw-20260318-200015 | tier3_histogram | 4 | 71 | 74 | 0.053 |
| J_EDEw-20260318-200246 | tier1_speed_cap | 1 | 0 | 1 | 1.0 |
| J_EDEw-20260318-200246 | tier2_kinematic_spike | 6 | 83 | 3 | 0.067 |
| J_EDEw-20260318-200246 | tier3_histogram | 9 | 246 | 5 | 0.035 |

**NEW T3 headline:** 8 correct, 262 spurious, 68 undecidable. Correct fraction: 0.024, Spurious fraction: 0.775.

## Phase 2: Spurious T3 Characterization

**J_EDEw-20260318-200015:** 71 spurious → {'motion_shadow_pose': 36, 'single_point_blip': 5, 'sustained_same_person': 30}
**J_EDEw-20260318-200246:** 246 spurious → {'motion_shadow_pose': 157, 'single_point_blip': 31, 'sustained_same_person': 58}

## Phase 3: Threshold Sweep

| Threshold | Surviving | Correct | Spurious | Precision |
|-----------|-----------|---------|----------|-----------|
| 0.15 | 409 | 13 | 317 | 0.039 |
| 0.18 | 220 | 8 | 166 | 0.046 |
| 0.22 | 94 | 5 | 70 | 0.067 |
| 0.25 | 55 | 4 | 41 | 0.089 |

## Phase 4: k-Distribution

**J_EDEw-20260318-200015:** k_dist={'1': 120, '2': 60, '3': 5}, impure=65/185
**J_EDEw-20260318-200246:** k_dist={'1': 116, '2': 32, '3': 6, '4': 1, '5': 2}, impure=41/157

## Phase 5: Change-Point Feasibility

**J_EDEw-20260318-200015:** Mixed — some impure tracklets not segmentable from clean points alone
  pure_clean=7/13, impure_seg=10/41
**J_EDEw-20260318-200246:** Mixed — some impure tracklets not segmentable from clean points alone
  pure_clean=23/45, impure_seg=13/36
