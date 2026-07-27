# Tagged Person Report: J_EDEw / J_EDEw-20260318-200246

> **Caveat:** This video uses train-split GT annotations (not held-out).
> Results are indicative but not evaluation-grade.

## Tagged person identification

- **tag_id:** 1
- **gt_track_id:** 8
- **tracklet_ids:** ['t139']
- **tag observations:** 1
- **vote detail:** {8: 1}

## Tag visibility

- **Total observations:** 1
- **Observation frames:** [1781]
- **Tracklet lifetime frames:** 2747
- **Tag detection rate:** 0.0364%
- **Tag consistency:** {'all_tag_ids_observed': {'1': 1}, 'misreads': 0}

### Detection context breakdown

| Context | Total frames | Frames with tag | Tag rate |
|---|---|---|---|
| tight_match | 317 | 0 | 0.0000% |
| pair_box | 62 | 0 | 0.0000% |
| split | 0 | 0 | N/A |
| miss | 71 | 0 | 0.0000% |

### Bbox-gated diagnostic

- **Mechanism:** bbox_gated
- **Note:** Tag detection is bbox-gated: Stage C scans padded detection bboxes only, never the full frame. If Stage A misses the person, Stage C never gets the chance to look for their tag. Improved detection recall may increase tag observation rate.
- **Window:** +/- 30 frames around each observation
- **Frames checked:** 61
- **Frames with any detection:** 61
- **Frames with covering detection:** 61
- **Detection coverage rate:** 100.0%

## Identity hint propagation (C -> D2 -> D4)

- **Stage C hints emitted:** True
- **D2 constraints created:** True
- **D4 person assigned:** True
- **Chain complete:** True

### Must-link hints
- tracklet=t139, anchor=tag:1, conf=1.0, evidence=tracklet_consensus_tag

### D2 must-link groups
- anchor=tag:1, tracklets=['t139']

### D2 tag pings
- tracklet=t139, frame=1781, conf=1.0

### Person assignments for tagged tracklets
- t139: person_ids=['p0022'], n_frames=149, range=[1571, 1728]

## Per-frame trace summary

- **Total GT frames:** 450
- **Stage A:** {'tight_match': 317, 'miss': 71, 'pair_box': 62}
- **Stage D:** {'wrong_id': 243, 'correct_id': 86, 'no_detection': 71, 'no_id': 50}
- **Frames with tag observed:** 0
- **Person IDs assigned:** {'p0004': 450}
- **Frames where person_id in match session:** 0

### Key events
- Pair-box frames: 62 (first=0, last=3930)

## Failure analysis

- correct_id: 86 (19.1%)
- wrong_id: 243 (54.0%)
- no_id: 50 (11.1%)
- no_detection: 71 (15.8%)

### wrong_id frames detail
- Person IDs assigned (wrong): {'p0004': 243}
- Stage A context: {'tight_match': 182, 'pair_box': 61}
