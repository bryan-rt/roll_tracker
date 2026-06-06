# Signal Trace Submodule

Applies to: `src/pipeline_validation/signal_trace/**`

## Location

`src/pipeline_validation/signal_trace/` — standalone read-only analysis submodule.
Does NOT modify the frozen evaluation instrument (CP-EVAL-1).

## Files

| File | Purpose |
|------|---------|
| `greedy_matcher.py` | Many-to-one IoU matcher (IoU >= 0.3, different from frozen instrument's Hungarian IoU 0.5) |
| `stage_a_census.py` | Stage A topology census — classifies GT-person-frames into tight_match/pair_box/split/miss |
| `stage_d_trace.py` | D-stage signal preservation trace. CP-TRACE-FIX: split-product resolution via d05_split_audit.jsonl |
| `group_falsification.py` | Tests whether GROUP segments address pair-box under-segmentation (answer: no) |
| `no_id_diagnosis.py` | Diagnoses no_id into d4_frame_trim vs d3_solver_drop |
| `stage_ef_trace.py` | Extends trace through Stage E/F (match session presence) |
| `verdict.py` | Synthesis verdict with root cause ranking |
| `tag_trace.py` | Tag signal trace — traces tag_id through A→C→D→E, builds cross-tab |

## CLI

```bash
PYTHONPATH=src python -m pipeline_validation signal-trace --model {id} --stage {a|d|ef|tag|all}
```

Flags: `--camera`, `--gym-id`, `--iou-threshold`, `--tag-id`

## Key Design Points

- Greedy matcher is many-to-one (multiple GT can match same detection = pair_box)
- Uses IoU >= 0.3 (NOT the frozen instrument's 0.5 — captures weaker overlaps)
- D-trace join uses split-product resolution (CP-TRACE-FIX): if tracklet_id from
  detections.parquet is a pre-split ID, resolves to post-split product via
  d05_split_audit.jsonl. This was a critical fix — without it, 29% appeared as
  no_id when in reality it was a join-key mismatch.
- Tag trace dispatches separately from a/d/ef (self-contained, line 1523 in cli.py)

## Output Location

```
outputs/_eval/signal_trace/{model_id}/
  {camera_id}/
    gt_signal_trace_stage_a.parquet
    gt_signal_trace_d.parquet
    topology_summary.json
    stage_d_summary.json
    tag_census.json
    identity_hint_audit.json
    tagged_person_trace.parquet
  {camera_id}_200246/        # v2 manifest train-split (separate dir)
  cross_tab.json
  cross_tab.md
  _verdict.md
  _tag_signal_verdict.md
  _tag_experiment_report.md
```

Pre-fix artifacts preserved at `outputs/_eval/signal_trace/bjj-detect-all-cameras_pre_fix/`.

## Corrected Aggregate Results (post CP-TRACE-FIX)

**Stage A:** 66.5% tight_match, 23.1% pair_box, 0% split, 10.4% miss
**Stage D:** 58.7% correct_id, 28.2% wrong_id, 2.8% no_id, 10.4% no_detection

Root cause ranking: wrong_id (28.2%) > pair_box (23.1%) > miss (10.4%) > d4_frame_trim (2.5%) > d3_solver_drop (0.3%)
