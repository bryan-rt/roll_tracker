# PURITY-PROXY-1 Verdict

## Question
Does any TRACKLET-AGGREGATE signal separate GT-pure from GT-impure tracklets?

## Entity confirmation
d3_ilp2's unexplained_tracklet_penalty charges against post-D0.5 products
(base_tracklet_id from SINGLE_TRACKLET nodes). Level B is fix-relevant.

## vid2 (authoritative)

### Level A: Raw Stage A tracklets
- Total: 157, Pure: 102 (65.0%), Temporal impure: 26 (16.6%), Spatial-only: 29, Both: 20
- k-distribution: {2: 24, 3: 1, 5: 1}

### Level B: Post-D0.5 products (FIX-RELEVANT)
- Total: 507, Pure: 392 (77.3%), Temporal impure: 22 (4.3%), Spatial-only: 93, Both: 15
- k-distribution: {2: 22}

### D0.5 effectiveness
- Impure raw tracklets: 26
- Fully fixed: 8, Partially: 13, Still impure: 5

### Proxy scores (Level A — raw tracklets)

| Proxy | AUC | Pure mean | Impure mean | N |
|-------|-----|-----------|-------------|---|
| max_deviation_from_mean | 0.653 | 0.3821 | 0.4799 | 78 |
| max_pairwise_bhatt | 0.691 | 1.0678 | 2.0595 | 78 |
| n_appearance_modes | 0.665 | 6.9057 | 8.1600 | 78 |
| max_displacement_m | 0.850 | 0.1884 | 0.8247 | 142 |
| mean_displacement_m | 0.663 | 0.0266 | 0.0303 | 142 |
| n_teleports | 0.650 | 0.0076 | 0.5000 | 157 |

Appearance coverage: 78 scoreable, 79 N/A (50.3%)

### Proxy scores (Level B — post-D0.5, fix-relevant)

| Proxy | AUC | Pure mean | Impure mean | N |
|-------|-----|-----------|-------------|---|
| max_deviation_from_mean | 0.705 | 0.1751 | 0.2632 | 356 |
| max_displacement_m | 0.821 | 0.1736 | 0.4280 | 488 |
| max_pairwise_bhatt | 0.726 | 0.2365 | 0.4807 | 356 |
| n_appearance_modes | 0.673 | 2.0737 | 3.0588 | 356 |
| n_teleports | 0.542 | 0.0103 | 0.0909 | 507 |

### Proxy 3: Tag contradiction
- Tracklets with tags: 1
- Contradictions: 0
- Precision: None, Recall: 0.0

### Proxy 4: Tracker-internal confidence
- UNAVAILABLE: local_track_conf is 100% NULL in BoT-SORT output.
- Would require instrumenting the tracker association step.

### Same-color caveat
- Impure tracklets: 26
- Same-color (undetectable by appearance): 2 (7.7%)
- Different-color (detectable): 24 (92.3%)
- Mean inter-GT Bhattacharyya: 0.2919

## vid1 (corroboration)

### Level A: Raw Stage A tracklets
- Total: 185, Pure: 110 (59.5%), Temporal impure: 47 (25.4%), Spatial-only: 28, Both: 39
- k-distribution: {2: 46, 3: 1}

### Level B: Post-D0.5 products (FIX-RELEVANT)
- Total: 350, Pure: 215 (61.4%), Temporal impure: 50 (14.3%), Spatial-only: 85, Both: 38
- k-distribution: {2: 50}

### D0.5 effectiveness
- Impure raw tracklets: 47
- Fully fixed: 6, Partially: 23, Still impure: 18

### Proxy scores (Level A — raw tracklets)

| Proxy | AUC | Pure mean | Impure mean | N |
|-------|-----|-----------|-------------|---|
| max_deviation_from_mean | 0.717 | 0.1884 | 0.3052 | 54 |
| max_pairwise_bhatt | 0.710 | 0.2692 | 0.5213 | 54 |
| n_appearance_modes | 0.618 | 2.7895 | 5.6286 | 54 |
| max_displacement_m | 0.827 | 0.1647 | 0.4260 | 154 |
| mean_displacement_m | 0.622 | 0.0277 | 0.0289 | 154 |
| n_teleports | 0.507 | 0.0145 | 0.0213 | 185 |

Appearance coverage: 54 scoreable, 131 N/A (70.8%)

### Proxy scores (Level B — post-D0.5, fix-relevant)

| Proxy | AUC | Pure mean | Impure mean | N |
|-------|-----|-----------|-------------|---|
| max_deviation_from_mean | 0.709 | 0.1379 | 0.2354 | 154 |
| max_displacement_m | 0.750 | 0.1541 | 0.3203 | 319 |
| max_pairwise_bhatt | 0.691 | 0.1779 | 0.3797 | 154 |
| n_appearance_modes | 0.615 | 1.7317 | 2.5484 | 154 |
| n_teleports | 0.508 | 0.0067 | 0.0200 | 350 |

### Proxy 3: Tag contradiction
- Tracklets with tags: 0
- Contradictions: 0
- Precision: None, Recall: 0.0

### Proxy 4: Tracker-internal confidence
- UNAVAILABLE: local_track_conf is 100% NULL in BoT-SORT output.
- Would require instrumenting the tracker association step.

### Same-color caveat
- Impure tracklets: 47
- Same-color (undetectable by appearance): 38 (80.9%)
- Different-color (detectable): 9 (19.1%)
- Mean inter-GT Bhattacharyya: 0.0948
