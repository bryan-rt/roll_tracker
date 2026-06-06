# Tag Identity Architecture

Applies to: `src/bjj_pipeline/stages/tags/**`, `src/bjj_pipeline/stages/stitch/d2*`,
`src/bjj_pipeline/stages/stitch/d4*`, `src/pipeline_validation/signal_trace/tag_trace.py`

## Signal Chain

```
Stage C (observations) → identity_hints.jsonl → D2 (must_link_groups)
  → D3 (soft penalty) → D4 (post-hoc tag-to-person mapping)
```

## D2: Must-Link Groups

- Constructed from `identity_hints.jsonl` (Stage C output)
- Contains ONLY tracklets that directly observed the tag
- Does NOT include adjacent/stitched tracklets
- D0.5 routes hints to split products by `first_seen_frame`

## D3: Soft Constraint (NOT Hard)

- Must_link penalty = 2x miss_penalty (soft BoolVar, not hard ILP constraint)
- Cross-camera corroboration boosts penalty 10x
- Solver CAN violate must_link by paying the penalty
- Tagged tracklets CAN be dropped (observed in J_EDEw-200246: t99 dropped)

## D4: Person ID Assignment

- Person_ids are sequential (p0001, p0002, ...) — NOT tag-derived
- One person_id per ILP entity path (SOURCE→SINK)
- All tracklets on same path get same person_id
- Tag-to-person mapping is POST-HOC via frame overlap scoring:
  - For each person_id, count frames on tagged tracklets
  - Dominant tag → identity_assignment in `identity_assignments.jsonl`
- GROUP segments cause multiple person_ids per tracklet per frame
  → tag identity dilutes across all person_ids sharing the tagged tracklet

## Tag Detection Gating

- **Bbox-gated:** `c0_gating.py` only scans padded detection bboxes (`bbox_pad_frac: 0.15`)
- **Cadence-gated:** C0Scheduler 3-state machine (SEEKING k=1, VERIFIED k=10, RAMP_UP k=1)
- **Physical visibility is the primary bottleneck** — full-frame scan found identical
  observations to pipeline (CP-TAG-2). Tag is ~25px at ceiling distance vs ~40px decode minimum.

## Known Issues (from cross-tracklet diagnostic)

1. **Must_link too soft:** Tagged tracklet t99 (862 frames) dropped in J_EDEw-200246.
   Penalty insufficient to force solver to keep it.
2. **No path propagation:** Non-tagged tracklets on same solver path do NOT inherit tag.
   Tag confined to must_link group tracklets only.
3. **GROUP dilution:** Video 1 tagged tracklet t366 has 167 person_id transitions
   (alternates frame-by-frame between 3 person_ids due to GROUP segments).
   D4 emits 3 separate identity_assignments for same tag.
4. **Nested detection capture:** At frame 1781, both t99 and t143 (nested bbox inside
   t99) captured the tag. When t99 was dropped, tag flowed to wrong person via t143.

## Planned Fixes

- Hard must_link for tag-observed tracklets (prevent solver from dropping them)
- Path-based tag propagation (non-tagged tracklets on same path inherit tag anchor)
- GROUP dilution mitigation (unclear mechanism — may resolve with above two fixes)

## Tools

| Tool | Purpose |
|------|---------|
| `tools/tag_fullscan.py` | Standalone full-frame AprilTag scan (no pipeline dependency) |
| `tools/tag_experiment.py` | Dense GT + three-way tag comparison orchestrator |

## Dense GT

- Manifest: `configs/models/bjj-detect-all-cameras-dense.yaml` (stride=1 for J_EDEw)
- CVAT zips contain interpolated labels at every frame
- Dense evaluation confirms stride-10 is representative (±0.3pp)
