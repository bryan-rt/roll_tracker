# Tagged Person Report: J_EDEw / J_EDEw-20260318-200015

## Tagged person identification

- **tag_id:** 1
- **gt_track_id:** 24
- **tracklet_ids:** ['t366']
- **tag observations:** 1
- **vote detail:** {24: 1}

## Tag visibility

- **Total observations:** 1
- **Observation frames:** [2770]
- **Tracklet lifetime frames:** 1521
- **Tag detection rate:** 0.0657%
- **Tag consistency:** {'all_tag_ids_observed': {'1': 1}, 'misreads': 0}

### Detection context breakdown

| Context | Total frames | Frames with tag | Tag rate |
|---|---|---|---|
| tight_match | 108 | 1 | 0.9259% |
| pair_box | 124 | 0 | 0.0000% |
| split | 0 | 0 | N/A |
| miss | 69 | 0 | 0.0000% |

### Bbox-gated diagnostic

- **Mechanism:** bbox_gated
- **Note:** Tag detection is bbox-gated: Stage C scans padded detection bboxes only, never the full frame. If Stage A misses the person, Stage C never gets the chance to look for their tag. Improved detection recall may increase tag observation rate.
- **Window:** +/- 30 frames around each observation
- **Frames checked:** 61
- **Frames with any detection:** 61
- **Frames with covering detection:** 38
- **Detection coverage rate:** 62.3%

## Identity hint propagation (C -> D2 -> D4)

- **Stage C hints emitted:** True
- **D2 constraints created:** True
- **D4 person assigned:** True
- **Chain complete:** True

### Must-link hints
- tracklet=t366, anchor=tag:1, conf=1.0, evidence=tracklet_consensus_tag

### D2 must-link groups
- anchor=tag:1, tracklets=['t366']

### D2 tag pings
- tracklet=t366, frame=2770, conf=1.0

### Person assignments for tagged tracklets
- t366: person_ids=['p0019', 'p0028', 'p0032'], n_frames=242, range=[2759, 2897]

## Per-frame trace summary

- **Total GT frames:** 301
- **Stage A:** {'pair_box': 124, 'tight_match': 108, 'miss': 69}
- **Stage D:** {'wrong_id': 152, 'correct_id': 77, 'no_detection': 69, 'no_id': 3}
- **Frames with tag observed:** 1
- **Person IDs assigned:** {'p0010': 301}
- **Frames where person_id in match session:** 0

### Key events
- First tag detection: frame 2770
- Last tag detection: frame 2770
- Pair-box frames: 124 (first=0, last=2910)

## Failure analysis

- correct_id: 77 (25.6%)
- wrong_id: 152 (50.5%)
- no_id: 3 (1.0%)
- no_detection: 69 (22.9%)

### wrong_id frames detail
- Person IDs assigned (wrong): {'p0010': 152}
- Stage A context: {'tight_match': 88, 'pair_box': 64}
