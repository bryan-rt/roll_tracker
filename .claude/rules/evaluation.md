---
paths:
  - "src/pipeline_validation/**"
---

# Evaluation Framework Rules

## Frozen Instrument (CP-EVAL-1, v1.0)

Spec: `docs/eval_instrument_spec.md`. Do NOT modify without a deliberate migration brief.

- **Single matcher:** `common/matching.py` — Hungarian, IoU 0.5, scipy. No additional matchers.
- **Identity mapping:** Derived in `gt_person_trace.py` from `per_frame_matches.parquet` +
  `person_tracks.parquet`. Does NOT read `identity_mapping.json` (that's `stage_d/evaluate.py`'s
  secondary output).
- **Failure mode order (frozen):** missing_canonical → stage_a_no_detection →
  stage_a_untracked → d3_dropped → d4_unassigned → present_misattributed → present.
  First match wins. Do not reorder.

## Swap Diagnostic (CP-SWAP-1/2)

- `tracker_swap/diagnostic.py` uses IoU >= 0.3 for GT assignment (intentionally lower
  than the frozen 0.5 — captures weaker overlaps at swap boundaries).
- `tracker_swap/characterize.py` reads `swap_events.jsonl` from CP-SWAP-1 output.
  Run `swap-diagnostic` before `swap-characterize`.

## Output Structure

```
outputs/_eval/
  stage_a/{model_id}/     # TB-EVAL-1 detection quality
  stage_d/{model_id}/     # TB-EVAL-2 identity + GT person trace
  stage_f/{model_id}/     # TB-EVAL-3 match preview mp4s
  tracker_swap/{model_id}/ # CP-SWAP-1/2 swap diagnostics
  gt2actuals/{camera}/{clip}/  # CP-GT2ACTUALS dense error map
  experiments/             # Named experiment results (e.g., cp_reid_1/)
```

## CLI Commands

```bash
python -m pipeline_validation evaluate --model {id}           # Full eval (A+D+F)
python -m pipeline_validation swap-diagnostic --model {id}    # CP-SWAP-1
python -m pipeline_validation swap-characterize --model {id}  # CP-SWAP-2
python -m pipeline_validation gt2actuals --manifest-path <path>  # Dense error map
```

## Dense Error Map (CP-GT2ACTUALS)

Module: `src/pipeline_validation/gt2actuals/` (dense_join, node_gt_set, jumps).
Uses `--manifest-path` (explicit, not model_id) for metric-basis discipline.
Dense manifest: `configs/models/bjj-detect-all-cameras-dense.yaml`.

**Split-family lookup:** person_tracks join uses family-aware fallback
(resolved → raw → any split-family member) because D4 re-stitches D0.5
products. The same bug exists in `signal_trace/stage_d_trace.py` but does
NOT affect locked canonical numbers (computed pre-split). Fixing signal_trace
is a separate gated decision.

**Jump types:** tracklet_drift (Stage A), false_split (D0.5), ilp_misstitch
(solver), group_boundary_jump, group_membership_drift. GT-derived only.
