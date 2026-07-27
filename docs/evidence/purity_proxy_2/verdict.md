# PURITY-PROXY-2 Verdict

## Question
Does MASKED appearance improve purity separation over PROXY-1's contaminated signal,
and does path + masked-appearance beat path-alone?

## vid2 (authoritative)

### Extraction
- Detections: 47402, Degenerate: 287 (0.6%)
- Median coverage: 0.685

### Drift-frame survival guard
- Total drift frames: 41
- Survived: 40 (97.6%)
- Gated out: 1 (2.4%)

### Masked appearance AUC (Level A) vs PROXY-1 contaminated

| Proxy | Masked AUC | PROXY-1 AUC (contaminated) | Delta |
|-------|-----------|---------------------------|-------|
| masked_max_pairwise_bhatt | 0.776 | (see PROXY-1) | - |
| masked_max_deviation | 0.739 | (see PROXY-1) | - |
| masked_n_modes | 0.706 | (see PROXY-1) | - |

### Masked appearance AUC (Level B — fix-relevant)

| Proxy | AUC | N |
|-------|-----|---|
| masked_max_pairwise_bhatt | 0.737 | 383 |
| masked_max_deviation | 0.790 | 383 |
| masked_n_modes | 0.568 | 383 |

### Same-color caveat (MASKED)
- Same-color: 18 (69.2%)
- Diff-color: 8
- Mean masked inter-GT Bhatt: 0.1278

- diff_color AUC: 0.883 (n_impure=8)
- same_color AUC: 0.728 (n_impure=18)

### Multivariate: path + masked-appearance vs path-alone
- Full set: path=0.799, app=0.773, combo=0.808, lift=+0.009
- Smooth+diff-color: path=0.857, app=0.916, combo=0.920, lift=+0.063

### Impure tracklet partition
- Teleport (path catches): 8 (30.8%)
- Smooth + diff-color (appearance helps): 3 (11.5%)
- Smooth + same-color (BLIND): 15 (57.7%)

## vid1 (corroboration)

### Extraction
- Detections: 51195, Degenerate: 1491 (2.9%)
- Median coverage: 0.708

### Drift-frame survival guard
- Total drift frames: 100
- Survived: 98 (98.0%)
- Gated out: 2 (2.0%)

### Masked appearance AUC (Level A) vs PROXY-1 contaminated

| Proxy | Masked AUC | PROXY-1 AUC (contaminated) | Delta |
|-------|-----------|---------------------------|-------|
| masked_max_pairwise_bhatt | 0.746 | (see PROXY-1) | - |
| masked_max_deviation | 0.755 | (see PROXY-1) | - |
| masked_n_modes | 0.574 | (see PROXY-1) | - |

### Masked appearance AUC (Level B — fix-relevant)

| Proxy | AUC | N |
|-------|-----|---|
| masked_max_pairwise_bhatt | 0.659 | 221 |
| masked_max_deviation | 0.689 | 221 |
| masked_n_modes | 0.563 | 221 |

### Same-color caveat (MASKED)
- Same-color: 45 (95.7%)
- Diff-color: 2
- Mean masked inter-GT Bhatt: 0.0268

- diff_color AUC: 0.803 (n_impure=2)
- same_color AUC: 0.744 (n_impure=45)

### Multivariate: path + masked-appearance vs path-alone
- Full set: path=0.763, app=0.732, combo=0.770, lift=+0.008
- Smooth+diff-color: path=0.459, app=0.795, combo=0.844, lift=+0.385

### Impure tracklet partition
- Teleport (path catches): 1 (2.1%)
- Smooth + diff-color (appearance helps): 2 (4.3%)
- Smooth + same-color (BLIND): 44 (93.6%)
