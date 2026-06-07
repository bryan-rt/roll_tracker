# CP-TAG-3 Baseline Evidence Archive

## Purpose

Before-state baseline for the tag identity recovery arc (CP-TAG-3 -> CP-TAG-4 ->
re-measure gate -> CP21 Tier 2 appearance). Instrumentation and evidence collection
ONLY -- no solver or pipeline behavior changes.

## Reproduction Commands

```bash
# 1. Per-clip pipeline (vid2; vid1 already has current outputs)
PYTHONPATH=src python -m bjj_pipeline.stages.orchestration.cli run \
  --clip data/raw/nest/_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200246.mp4 \
  --camera J_EDEw --to-stage E

# 2. Official signal trace for vid1 (val-split)
PYTHONPATH=src python -m pipeline_validation signal-trace \
  --model bjj-detect-all-cameras --stage tag --camera J_EDEw

# 3. All evidence extraction (tag-trace + session + carrier)
PYTHONPATH=src python tools/cp_tag_3_evidence.py all
```

## Session Scope (IMPORTANT for re-measure gate)

The cp_tag_3_baseline session is **two-clip, single-camera** (J_EDEw only):
- J_EDEw-20260318-200015 (4530 frames, offset 0)
- J_EDEw-20260318-200246 (4500 frames, offset 4530)

This is NOT the production 3-camera session shape. No Tier 3 cross-camera
histogram evidence is in play. This is the controlled experiment for the
clip-boundary tag identity question.

**Post-CP-TAG-4 re-measure gate MUST reuse this exact session scope** for
apples-to-apples comparison.

## Archive Contents

| File | Description |
|------|-------------|
| `tag_trace_summary.json` | Vid1 faithfulness check + vid2 tag trace baseline |
| `vid2_tag_trace/` | Tagged person trace report + parquet for vid2 |
| `session_evidence.json` | Session-level query results (drop status, transitions) |
| `session_evidence.md` | Human-readable session evidence report |
| `carrier_evidence.json` | Tagged tracklet geometry at observation frames |
| `carrier_evidence.md` | Human-readable carrier geometry report |
| `provenance.md` | Baseline provenance analysis + footgun documentation |
| `vid2_a_trace.parquet` | Stage A census trace for vid2 (intermediate artifact) |
| `vid2_d_trace.parquet` | Stage D trace for vid2 (intermediate artifact) |

## Key Baseline Numbers

### Vid1 (J_EDEw-200015, val-split, 301 GT frames, gt_track_id=24)
- correct_id: 77 (25.6%)
- wrong_id: 152 (50.5%)
- no_id: 3 (1.0%)
- no_detection: 69 (22.9%)
- Tagged tracklet: t366, 1 tag obs at frame 2770

### Vid2 (J_EDEw-200246, train-split, 450 GT frames, gt_track_id=8)
- correct_id: 100 (22.2%)
- wrong_id: 229 (50.9%)
- no_id: 50 (11.1%)
- no_detection: 71 (15.8%)
- Tagged tracklet: t139, 1 tag obs at frame 1781

### Session-level (two-clip J_EDEw)
- Both tagged tracklets KEPT (no solver drops)
- 4 identity_assignments for tag:1 (all span clip boundary)
- t366: 1,125 person_id transitions, 14 unique person_ids
- t139: 2,680 person_id transitions, 12 unique person_ids
- GROUP dilution is the dominant identity corruption mechanism

## Pipeline State

- Detection model: bjj-detect-all-cameras-v2.pt
- CP5 (parallel-carrier consolidation): active
- CP-SPLIT-1 (D0.5 tracklet splitter): active
- CP17 must-link bug fix: active (format parsing fix)
- Must-link: soft (2x miss_penalty), NOT hard
- Branch: services_uploader, HEAD at time of run
