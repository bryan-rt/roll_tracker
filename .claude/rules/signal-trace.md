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
PYTHONPATH=src python -m pipeline_validation signal-trace --model {id} --stage {a|d|ef|trim|tag|all}
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

**NOTE:** The 58.7% is a THREE-CAMERA aggregate. Single-camera J_EDEw baseline is ~40.5%
(measured PRE-split, Jun 7). Current post-split state: 33.9% (CP-GT2ACTUALS-3.5).
These numbers are NOT comparable without stating the basis (camera set, frame range,
person_tracks level, pipeline state). Canonical: clip-level, val-split, greedy IoU>=0.3.

**KNOWN BUG (CP-GT2ACTUALS-3.5):** `stage_d_trace.py`'s `_compute_dominant_person_ids`
and `run_d_trace` use single-resolution `_resolve_tracklet_id` without the family-aware
fallback. If D0.5 splits exist, this produces inflated no_id (vid2: 58% → 6.9% with
fix). The locked canonical numbers (40.5%/63.2%) are NOT biased because they were
computed before D0.5 splits existed (Jun 7 vs Jun 9). But re-running signal_trace
today would produce the bug. The fix is in `gt2actuals/dense_join.py`
(`_lookup_person_ids_family`); applying it to signal_trace is a separate gated decision.

**CP-PURITY-3 finding (2026-06-09):** GT→D oracle (perfect detections through real D1)
produced 0 GROUP nodes with 0 logic gaps. 100% of the "group-formation defect"
(CP-PURITY-2: 29.9%/11.6%) is detection under-segmentation, not D1 logic.
D1 forms GROUPs from lifecycle events, not proximity — structurally unnecessary when
detection is perfect. Evidence: `docs/evidence/cp_purity_3/`.
