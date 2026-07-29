# CP6: GT Person Trace Baseline

*Generated 2026-05-18. Model: bjj-detect-all-cameras. Full 6-mode trace on current run;
lite 4-mode on 4 historical baselines.*

---

## Section 1 -- Cross-Snapshot Failure-Mode Breakdown

All values are percentages of total GT-frame pairs per camera. Current run rolled up
to 4-mode (lite) for apples-to-apples comparison with historical baselines.

### FP7oJQ (4,214 GT-frame pairs, 14 GT tracks)

| Snapshot | present | stage_a_miss | stage_d_no_person | misattributed | missing_canonical |
|----------|---------|-------------|-------------------|---------------|-------------------|
| penalty_15 | 32.3 | 17.3 | 18.6 | 3.3 | 28.6 |
| cp2_penalty_25 | 32.4 | 18.6 | 15.0 | 5.4 | 28.6 |
| cp3b_pre | 26.7 | 18.8 | 24.9 | 1.0 | 28.6 |
| cp4_pre | 32.3 | 18.6 | 15.0 | 5.6 | 28.6 |
| **current** | **5.1** | **9.9** | **38.6** | **17.9** | **28.6** |

### J_EDEw (4,214 GT-frame pairs, 14 GT tracks)

| Snapshot | present | stage_a_miss | stage_d_no_person | misattributed | missing_canonical |
|----------|---------|-------------|-------------------|---------------|-------------------|
| penalty_15 | 13.4 | 23.2 | 53.3 | 10.2 | 0.0 |
| cp2_penalty_25 | 14.8 | 23.2 | 51.4 | 10.6 | 0.0 |
| cp3b_pre | 13.7 | 23.2 | 53.6 | 9.5 | 0.0 |
| cp4_pre | 14.9 | 23.2 | 51.4 | 10.5 | 0.0 |
| **current** | **4.7** | **11.7** | **63.9** | **19.7** | **0.0** |

### PPDmUg (2,361 GT-frame pairs, 8 GT tracks)

| Snapshot | present | stage_a_miss | stage_d_no_person | misattributed | missing_canonical |
|----------|---------|-------------|-------------------|---------------|-------------------|
| penalty_15 | 19.6 | 21.7 | 36.9 | 9.1 | 12.7 |
| cp2_penalty_25 | 20.3 | 21.7 | 36.6 | 8.6 | 12.7 |
| cp3b_pre | 20.8 | 21.7 | 37.9 | 6.9 | 12.7 |
| cp4_pre | 18.8 | 21.7 | 36.6 | 10.1 | 12.7 |
| **current** | **6.1** | **13.3** | **49.0** | **18.9** | **12.7** |

**Key observation:** The current run shows a dramatic regression in `present` (correct
attribution) across all cameras despite a significant improvement in `stage_a_miss`
(better detection). Stage A miss dropped 7-12pp, but that gain was more than consumed by
increases in `stage_d_no_person` (+13-25pp) and `misattributed` (+9-15pp). Detection
improved; stitching degraded.

The historical baselines are lite-mode snapshots from earlier pipeline runs. Differences
in stage_a_miss between baselines and current may reflect pipeline code changes (Stage A
pre/post-processing) rather than model changes, since all runs use the same detection model.

---

## Section 2 -- Current Run, Per-GT-Person Breakdown (Full 6-Mode)

### FP7oJQ

| gt_track | canonical | total | present | a_miss | untracked | d3_drop | d4_unasgn | misattrib | miss_can |
|----------|-----------|-------|---------|--------|-----------|---------|-----------|-----------|----------|
| 14 | p0006 | 301 | 24 | 26 | 10 | 126 | 79 | 36 | 0 |
| 15 | p0006 | 301 | 23 | 5 | 8 | 179 | 39 | 47 | 0 |
| 16 | p0012 | 301 | 55 | 0 | 38 | 68 | 25 | 115 | 0 |
| 17 | None | 301 | 0 | 0 | 0 | 0 | 0 | 0 | 301 |
| 18 | p0007 | 301 | 0 | 79 | 40 | 123 | 27 | 32 | 0 |
| 19 | p0008 | 301 | 10 | 18 | 36 | 100 | 19 | 118 | 0 |
| 20 | None | 301 | 0 | 0 | 0 | 0 | 0 | 0 | 301 |
| 21 | None | 301 | 0 | 0 | 0 | 0 | 0 | 0 | 301 |
| 22 | p0010 | 301 | 1 | 6 | 37 | 81 | 24 | 152 | 0 |
| 23 | p0004 | 301 | 23 | 23 | 33 | 123 | 3 | 96 | 0 |
| 24 | p0002 | 301 | 11 | 110 | 35 | 50 | 18 | 77 | 0 |
| 25 | p0012 | 301 | 35 | 112 | 46 | 51 | 2 | 55 | 0 |
| 26 | p0009 | 301 | 34 | 37 | 58 | 110 | 37 | 25 | 0 |
| 27 | None | 301 | 0 | 0 | 0 | 0 | 0 | 0 | 301 |

4 of 14 GT tracks have no canonical assignment (missing_canonical = 100%). These are
GT tracks with zero matched frames in person_tracks -- likely short-lived or occluded
people that Stage D never stitched. Identity mapping min purity: 0.00, mean: 0.61.

### J_EDEw

| gt_track | canonical | total | present | a_miss | untracked | d3_drop | d4_unasgn | misattrib | miss_can |
|----------|-----------|-------|---------|--------|-----------|---------|-----------|-----------|----------|
| 14 | p0008 | 301 | 8 | 33 | 47 | 153 | 3 | 57 | 0 |
| 15 | p0005 | 301 | 2 | 21 | 27 | 171 | 0 | 80 | 0 |
| 16 | p0009 | 301 | 6 | 19 | 65 | 172 | 1 | 38 | 0 |
| 17 | p0011 | 301 | 47 | 8 | 56 | 136 | 3 | 51 | 0 |
| 18 | p0006 | 301 | 9 | 30 | 38 | 143 | 3 | 78 | 0 |
| 19 | p0012 | 301 | 5 | 3 | 44 | 173 | 0 | 76 | 0 |
| 20 | p0006 | 301 | 34 | 3 | 31 | 171 | 2 | 60 | 0 |
| 21 | p0006 | 301 | 9 | 20 | 47 | 173 | 2 | 50 | 0 |
| 22 | p0002 | 301 | 29 | 73 | 37 | 108 | 7 | 47 | 0 |
| 23 | p0013 | 301 | 3 | 74 | 48 | 136 | 0 | 40 | 0 |
| 24 | p0001 | 301 | 5 | 9 | 22 | 182 | 2 | 81 | 0 |
| 25 | p0001 | 301 | 3 | 16 | 43 | 178 | 1 | 60 | 0 |
| 26 | p0013 | 301 | 6 | 128 | 21 | 93 | 2 | 51 | 0 |
| 27 | p0002 | 301 | 32 | 54 | 44 | 105 | 4 | 62 | 0 |

All 14 GT tracks have canonical assignments. d3_dropped is the dominant failure mode for
13 of 14 GT tracks (gt_track_26 dominated by stage_a_miss at 43%). Identity mapping min
purity: 0.27, mean: 0.62.

### PPDmUg

| gt_track | canonical | total | present | a_miss | untracked | d3_drop | d4_unasgn | misattrib | miss_can |
|----------|-----------|-------|---------|--------|-----------|---------|-----------|-----------|----------|
| 0 | p0007 | 283 | 25 | 6 | 18 | 154 | 3 | 77 | 0 |
| 1 | p0007 | 278 | 22 | 26 | 15 | 137 | 5 | 73 | 0 |
| 2 | None | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 300 |
| 3 | p0002 | 300 | 56 | 11 | 27 | 135 | 2 | 69 | 0 |
| 4 | p0005 | 300 | 17 | 56 | 31 | 153 | 22 | 21 | 0 |
| 5 | p0005 | 300 | 10 | 165 | 27 | 77 | 7 | 14 | 0 |
| 6 | p0008 | 300 | 0 | 2 | 17 | 152 | 6 | 123 | 0 |
| 7 | p0003 | 300 | 13 | 47 | 34 | 133 | 3 | 70 | 0 |

1 GT track (gt_track_2) has no canonical assignment. d3_dropped is dominant for all
GT tracks with canonical assignments except gt_track_5 (dominated by stage_a_miss at
55%). Identity mapping min purity: 0.00, mean: 0.61.

---

## Section 3 -- Spot-Check Report

### J_EDEw t1

- **293 matched GT frames**, range 0-3000, **D3: dropped**
- Covers 10 GT persons: gt_track 14, 15, 16, 17, 19, 20, 21, 24, 25, 26
- All 293 frames classified as `d3_dropped`
- D1 carrier status: carrier (272 frames), non_carrier (21 frames) -- predominantly carrier
- Final person_id: none (all dropped)
- This is the prototypical parallel-carrier victim: a long-running carrier tracklet
  spanning the entire clip, dropped by D3 because it couldn't be incorporated into
  the flow solution alongside its parallel carrier (t2).

### J_EDEw t2

- **78 matched GT frames**, range 0-850, **D3: explained**
- Covers 10 GT persons: gt_track 14, 15, 16, 19, 20, 21, 24, 25, 26, 27
- 76 frames `present_misattributed`, 2 frames `present`
- D1 carrier status: carrier (62), non_carrier (16) -- also primarily carrier
- Final person_ids: p0004 (76 frames), p0009 (2 frames)
- The surviving parallel carrier. Assigned to p0004 but mostly misattributed -- only 2
  of its 78 GT-matched frames have the correct canonical person.

### J_EDEw t201

- **57 matched GT frames**, range 2240-2830, **D3: dropped**
- Covers 9 GT persons: gt_track 14, 16, 18, 19, 21, 22, 23, 24, 27
- All 57 frames classified as `d3_dropped`
- D1 carrier status: non_carrier (49), carrier (8) -- mostly subordinate
- This is the tag-anchored tracklet noted in prior CPs. Tag 1 observed at frame 2770
  (from identity_hints). Despite carrying a Tier 1 identity anchor, D3 dropped it.

### PPDmUg t1

- **293 matched GT frames**, range 0-2990, **D3: dropped**
- Covers 7 GT persons: gt_track 0, 1, 2, 3, 4, 6, 7
- 203 frames d3_dropped, 90 frames missing_canonical (gt_track_2, no canonical assignment)
- D1 carrier status: carrier (178), non_carrier (115) -- majority carrier
- Final person_id: none (all dropped)
- Mirror of J_EDEw t1. Full-clip-spanning carrier dropped by D3.

### PPDmUg t2

- **293 matched GT frames**, range 0-2990, **D3: explained**
- Covers 6 GT persons: gt_track 0, 1, 2, 3, 6, 7
- 241 frames present_misattributed, 52 frames present (gt_track_3 = p0002)
- D1 carrier status: carrier (203), non_carrier (90) -- majority carrier
- Final person_ids: p0001 (182 frames), p0002 (52 frames)
- The surviving parallel carrier. 82% misattributed. Only gt_track_3 frames
  reach correct attribution (p0002) ~18% of the time.

### FP7oJQ t62

- **0 matched GT frames** in the trace.
- t62 exists in detections.parquet (3,702 detections, frames 741-4529) and is
  D3-explained, but none of its detections matched any GT box at annotated frames.
- This is not a failure -- it's a tracklet active primarily outside the annotated
  frame range (301 frames from 0-300). The annotated range does not cover frames 741+.

---

## Section 4 -- CP5 Falsification Verdict

### J_EDEw pair: t1 (dropped) vs t2 (explained)

**Q1: Different GT persons?** Overlap: 9 shared GT persons out of 10 (t1) and 10 (t2).
Only gt_track_17 unique to t1, gt_track_27 unique to t2. The tracklets cover
essentially the same population of people.

**Q2: Temporal overlap?** 75 frames overlap (out of 293 t1, 78 t2). At ALL 75
overlapping frames, they cover DIFFERENT GT persons at the same frame. This is the
parallel-carrier signature: two carrier tracklets active simultaneously, each matched
to a different physical person.

**Q3: Dominant failure mode?** d3_dropped dominates for 9 of 10 GT persons covered
by the dropped tracklet (t1). Percentages range from 45-60% d3_dropped per GT person.
Only gt_track_26 has stage_a_miss as dominant (43%), with d3_dropped second at 31%.

**Verdict: `CP5-supports`.** The parallel-carrier displacement hypothesis is confirmed.
t1 and t2 are concurrent carrier tracklets covering the same population. D3 dropped t1
(the longer carrier with 293 frames) and kept t2 (78 frames), destroying coverage for
10 GT persons. This is a structural D3 limitation, not a detection or tracking issue.

### PPDmUg pair: t1 (dropped) vs t2 (explained)

**Q1: Different GT persons?** Overlap: 6 shared GT persons out of 7 (t1) and 6 (t2).
Only gt_track_4 unique to t1. Near-complete overlap.

**Q2: Temporal overlap?** 286 frames overlap (out of 293 t1, 293 t2 -- near-total).
At ALL 286 overlapping frames, they cover DIFFERENT GT persons. This is an even cleaner
parallel-carrier signature than J_EDEw: two carrier tracklets active across the entire
clip, each tracking a different person frame-by-frame.

**Q3: Dominant failure mode?** d3_dropped dominates for 6 of 7 GT persons (excluding
gt_track_2 which is missing_canonical). Percentages range from 44-54% d3_dropped per
GT person.

**Verdict: `CP5-supports`.** Same pattern as J_EDEw but more extreme: near-total
temporal overlap (97.6% of frames), near-total GT person overlap.

### Overall Verdict

**CP5-supports (both pairs).** The parallel-carrier displacement pattern is confirmed
on both cameras with evaluation data. Key structural findings:

1. **d3_dropped is the #1 failure mode** for almost every GT person across J_EDEw
   (13/14) and PPDmUg (6/7 with canonical assignments). It accounts for 50% of J_EDEw
   frames and 40% of PPDmUg frames.

2. **Parallel carriers are the mechanism.** Both t1/t2 pairs are concurrent carrier
   tracklets covering the same physical people. D3 can only incorporate one into the
   flow solution; the other is dropped wholesale, taking all its GT coverage with it.

3. **The surviving carrier is mostly misattributed.** J_EDEw t2 has 97.4%
   misattribution; PPDmUg t2 has 82.3%. Even the "explained" carrier rarely produces
   correct person assignment. This means CP5 alone won't fully fix attribution --
   but it will recover the ~50% of frames lost to d3_dropped.

4. **Stage A is NOT the bottleneck.** Stage A miss improved significantly in the
   current run (9.9-13.3% vs 17-23% in baselines). Despite this, overall attribution
   worsened because Stage D couldn't utilize the additional detections.

**Recommendation: Resume CP5 (parallel-carrier consolidation).** The evidence is
unambiguous. d3_dropped is the dominant failure mode, parallel carriers are the
mechanism, and the fix is structural (consolidate parallel carriers before D3 solving).
The high misattribution rate on surviving carriers is a secondary concern for a
follow-up checkpoint.

---

## Section 5 -- Longitudinal Observations

### Historical intervention impact (from cross-snapshot comparison)

**CP2 (penalty_15 -> cp2_penalty_25):** Increasing the D3 penalty from 0.15 to 0.25
had minimal impact. J_EDEw `stage_d_no_person` improved by ~2pp (53.3 -> 51.4%), but
`present` improved only ~1pp. FP7oJQ `stage_d_no_person` improved ~4pp (18.6 -> 15.0%),
`present` held steady. The penalty parameter is a minor lever.

**CP3b (cp3b_pre):** The floor-protected length-proportional penalty showed a
regression on FP7oJQ (`present` dropped from 32.4 to 26.7%, `stage_d_no_person` rose
from 15.0 to 24.9%). J_EDEw was essentially flat. This confirms that CP3b was correctly
identified as a regression and rolled back.

**CP4 (cp4_pre):** Matches cp2_penalty_25 almost exactly across all cameras.
This is expected -- CP4 was a diagnostic/investigation checkpoint, not a config change.
The cp4_pre baseline was captured with the same D config as cp2_penalty_25.

### Current run anomaly

The current run shows dramatically worse results than all baselines despite using the
same model (bjj-detect-all-cameras). Two likely explanations:

1. **Pipeline code drift.** The current pipeline (`--force` rerun) uses the latest
   Stage A-E code, which may have different behavior (e.g., different tracklet
   merging in BoT-SORT, different Stage D graph construction) than the code used
   when baselines were captured.

2. **Stage A improvement is real but double-edged.** Lower stage_a_miss means more
   detections enter Stage D. More detections produce more tracklets, more graph nodes,
   and more pressure on the D3 solver. Without parallel-carrier consolidation (CP5),
   the additional tracklets make D3's job harder, not easier.

Both explanations point to the same conclusion: Stage D is the bottleneck, and
parallel-carrier consolidation is the highest-leverage fix.

### missing_canonical stability

missing_canonical was stable across all five CP6 snapshots (FP7oJQ: 28.6%, J_EDEw: 0.0%,
PPDmUg: 12.7%). This was initially interpreted as a GT annotation property, but **CP5
disproved this**: after parallel-carrier consolidation, missing_canonical dropped to 0%
on all cameras. The mechanism: missing_canonical GT tracks had matched detections (Stage A
found them), but those detections lived in dropped tracklets with no person_id. When CP5
reduced dropped tracklets from 37 to 4 (FP7oJQ), 37 to 2 (J_EDEw), 14 to 0 (PPDmUg),
those detections gained person_ids, and canonical assignments emerged. **missing_canonical
is a pipeline-dependent variable, not a GT property.** It was stable across CP6 snapshots
only because those snapshots had similar Stage D routing (all pre-consolidation).

### Baseline preservation lesson

Historical baselines are lite-mode only -- they captured gt_track_sequences.jsonl and
identity_mapping.json but not the pipeline artifacts needed for full 6-mode trace.
Future baselines should copy both `outputs/_eval/` AND the relevant
`outputs/_eval_gt/{camera}/{clip}/` directories to enable full-mode retrospective
analysis.
