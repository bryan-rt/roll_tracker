# CP-RASTER-PLATE: Median-Background Masking + Appearance Separability

## Phase A: Median Background Plate

- Clips sampled: 12
- Frames sampled: 396
- Low-occupancy clips (empirical): ['J_EDEw-20260318-202016.mp4', 'J_EDEw-20260318-200517.mp4', 'J_EDEw-20260318-201246.mp4', 'J_EDEw-20260318-201016.mp4']
- Ghost pixels before fallback: 0.3%
- Ghost pixels after fallback: 0.0%

**Note:** Plate built from same footage as test clips. Production path needs held-out/rolling background.

## Phase B: Masked Histogram Extraction

### J_EDEw-20260318-200015
- Detections sampled: 1952
- Tracklets: 76
- Coverage: mean=70.7%, median=72.4%, p10=45.9%, p90=90.5%
- Degenerate masks: 35 (1.8%)

**Degenerate masks by gi color:**

| Gi Color | Total | Degenerate | Frac |
|----------|-------|------------|------|
| blue | 45 | 0 | 0.0% |
| orange_yellow | 173 | 0 | 0.0% |
| red | 102 | 24 | 23.5% |
| white_gray | 1632 | 11 | 0.7% |

### J_EDEw-20260318-200246
- Detections sampled: 2289
- Tracklets: 82
- Coverage: mean=68.4%, median=71.8%, p10=43.9%, p90=87.3%
- Degenerate masks: 21 (0.9%)

**Degenerate masks by gi color:**

| Gi Color | Total | Degenerate | Frac |
|----------|-------|------------|------|
| blue | 30 | 0 | 0.0% |
| orange_yellow | 298 | 17 | 5.7% |
| red | 60 | 0 | 0.0% |
| white_gray | 1901 | 4 | 0.2% |

## Phase C: Separability

**Aggregate AUC:** baseline=0.7576, masked=0.7697, delta=0.0121

**Distinct-color pairs (PRIMARY):** baseline AUC=None, masked AUC=None, delta=None
  - Color classifier: Color distinctiveness from GT-grounded gi-color labels (method-independent: assigned from union of both methods, majority-voted per GT person). Distinct = different gi color.

**Same-color pairs:** baseline AUC=0.746, masked AUC=0.7615, delta=0.0154

**Intrinsic-color floor:** 35.4% of different-person pairs inseparable under best mask

## Verdict

**NO_GO**

Mask quality gate: PASS (degenerate fraction: 1.3%)

Mask does not meaningfully improve separability: aggregate AUC delta=+0.012, same-color AUC delta=+0.015. Intrinsic-color floor: 35.4% inseparable. WARNING: This session has only 2 distinct gi color(s) (heavily white-gi). Distinct-color AUC is non-evaluable (0 same-person pairs in the distinct group). Results reflect worst-case color diversity. A session with more color variety may show different results. On this white-gi-heavy session, the bottleneck is intrinsic color similarity, not the extraction ROI.
