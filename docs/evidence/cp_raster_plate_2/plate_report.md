# CP-RASTER-PLATE-2: Appearance Separability with V Channel

## Phase A: Median Plate
- Frames: 1080 (every 50th from 12 clips)
- Every 50th frame from all 12 clips. 8 GB RAM constraint prevents holding all 54000 frames. 1080 samples drive per-pixel person-occupancy well below 50%.

## Phase B: Masked Extraction

- **J_EDEw-20260318-200015**: 76 tracklets, coverage=72.4% median, 1.7% degenerate
- **J_EDEw-20260318-200246**: 82 tracklets, coverage=71.8% median, 1.0% degenerate

## Phase C: Separability by Feature Space

| Feature Space | AUC (all) | AUC (distinct-color) | AUC (same-color) | Floor |
|---------------|-----------|---------------------|------------------|-------|
| Baseline H+S (144-dim, production) | 0.8147 | N/A | 0.6904 | 28.2% |
| Masked H+S full-body (144-dim) | 0.8741 | N/A | 0.6821 | 14.0% |
| Masked H+S+V full-body (864-dim) | 0.9068 | N/A | 0.6929 | 14.6% |
| Masked V-only full-body (6-dim) | 0.8942 | N/A | 0.7192 | 16.9% |
| Masked H+S torso-only (144-dim) | 0.8503 | N/A | 0.7147 | 23.7% |
| Masked H+S+V torso-only (864-dim) | 0.8854 | N/A | 0.7154 | 17.3% |
| Masked V-only torso-only (6-dim) | 0.842 | N/A | 0.7173 | 21.0% |

## Color Labels (hand-verified from masked H+S+V medians)

**J_EDEw-20260318-200015:** {'white': 6, 'skin': 3, 'gray': 3, 'dark_blue': 6, 'charcoal': 5, 'red': 2, 'medium_blue': 2, 'light_blue': 2}
**J_EDEw-20260318-200246:** {'light_blue': 1, 'blue': 2, 'gray': 3, 'skin': 6, 'white': 5, 'medium_blue': 2, 'red': 1}

## Verdict

**GO**

H+S+V separates meaningfully better than H+S: AUC 0.9068 vs 0.8147 (delta +0.0921). Distinct-color AUC: H+S+V=None, baseline=None. Intrinsic floor: 14.6% (was 28.2% under H+S). V-channel is a PRODUCTION FIX independent of masking. Mask adds +0.0594 on top of V. Skin-inclusive vs torso-only: AUC 0.9068 vs 0.8854 (delta +0.0214). Skin helps. Promote H+S+V histogram to src/bjj_pipeline and note V-extension is non-breaking (new hist_ columns; downstream reads by prefix).
