# CP-TAG-3 Baseline Provenance

## Why the prior 25.6%/16.9% numbers are not directly comparable

The prior tag trace numbers reported in CLAUDE.md came from **two different pipeline
vintages and different gym_ids**:

### Vid1 (25.6% correct_id)
- **Source:** `outputs/_eval_gt/J_EDEw/.../J_EDEw-20260318-200015/stage_D/`
- **Pipeline state:** Post-CP5, post-CP-SPLIT-1, v2 detection model (June 2, 2026)
- **Gym ID:** `_eval_gt`
- **GT split:** Val-split (51 held-out frames, 301 total annotated)
- **Provenance:** Clean -- current code produced these outputs

### Vid2 (16.9% correct_id)
- **Source:** `outputs/c8a592a4.../J_EDEw/.../J_EDEw-20260318-200246/stage_D/`
- **Pipeline state:** Pre-CP5, pre-CP-SPLIT-1, unknown detection model (~May 6, 2026)
- **Gym ID:** `c8a592a4-2bca-400a-80e1-fec0e5cbea77` (real gym, NOT `_eval_gt`)
- **GT split:** Train-split only (450 frames)
- **Provenance:** Stale -- produced 31 days before CP5, missing D0.5 splitter entirely

### Implications
- The 16.9% was dominated by 58.4% no_id -- mostly the artifact of missing CP5 (parallel
  carriers being dropped, not a tag identity failure per se)
- The clean vid2 baseline under `_eval_gt` with current code shows 22.2% correct_id /
  50.9% wrong_id / 11.1% no_id -- the no_id collapse into wrong_id confirms CP5 rescued
  the tracklets but they were misattributed (pair-box under-segmentation)
- Comparing 25.6% to 16.9% across pipeline versions was misleading -- the gap was mostly
  CP5/CP-SPLIT-1, not vid1 vs vid2 structural differences

## tag_trace.py:1181 hardcode footgun

`src/pipeline_validation/signal_trace/tag_trace.py` line 1181 hardcodes:
```python
real_gym_id = "c8a592a4-2bca-400a-80e1-fec0e5cbea77"
```

This means the official `signal-trace --stage tag` command will **silently read the stale
real-gym-id outputs** for vid2 (J_EDEw-200246), even though fresh `_eval_gt` outputs now
exist. The official trace was designed for the v2 manifest's 200246 entry, which predates
the `_eval_gt` convention for that clip.

**Impact:** After CP-TAG-4, running `signal-trace --stage tag` will report vid2 numbers
from the OLD pipeline state, not the post-fix state. Any re-measurement using the official
trace will silently produce wrong numbers for vid2.

**No fix now** (read-only constraint on `src/pipeline_validation/signal_trace/`). This is
registered for the future session-level-trace checkpoint, which should:
1. Either parameterize the gym_id for vid2 in the tag trace
2. Or build the session-level trace mode that reads `_eval_gt` outputs directly

**Workaround:** Use `tools/cp_tag_3_evidence.py tag-trace` instead of the official
signal-trace for vid2 measurements. The faithfulness self-check (vid1 numbers match
official) validates this code path.

## Session artifact freshness

The existing session run at `outputs/c8a592a4.../sessions/2026-03-18/2026-03-18T2000/`
was produced **March 27, 2026** (commit 9d04ce5), 71 days before the current HEAD. It
predates CP5, CP-SPLIT-1, CP-EVAL-1, and the detection model v2. It is NOT usable for
current-state session-level evidence.

The CP-TAG-3 baseline uses a fresh session run at
`outputs/_eval_gt/sessions/2026-03-18/cp_tag_3_baseline/` produced with current HEAD code.

## Session-level trace mode gap

The signal trace infrastructure (`src/pipeline_validation/signal_trace/`) operates per-clip
only -- it reads per-clip outputs, not session-level outputs. A session-level trace mode
does not exist. This means:
- Per-clip tag trace baselines can be compared before/after CP-TAG-4
- Session-level metrics (cross-clip boundary spanning, session identity_assignments) require
  the ad-hoc query script (`tools/cp_tag_3_evidence.py session`)
- Building a session-level trace mode is likely its own small checkpoint before the
  post-CP-TAG-4 re-measure gate
