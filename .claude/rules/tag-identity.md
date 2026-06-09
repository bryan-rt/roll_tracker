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

## Known Issues (from cross-tracklet diagnostic + CP-TAG-4a + CP-PURITY arc)

1. **Must_link too soft:** FIXED by CP-TAG-4a Fix C (hard no-drop for ping-carrying
   products via explained==1 + fallback ladder).
2. **No path propagation:** Non-tagged tracklets on same solver path do NOT inherit tag.
   Tag confined to must_link group tracklets only.
3. **GROUP dilution:** D0.5 split products create GROUP nodes; tagged tracklet gets
   multiple person_ids. CP-TAG-4a Fix A (thread consumption) gives exactly 1 tag→person
   mapping, but GROUP dilution of the entity path remains.
4. **Nested detection capture:** FIXED by CP-TAG-4a Fix 0 (split-aware ping binding via
   d05_split_audit expansion).

**CP-TAG-4a is a confirmed +22.7pp improvement** (CP-PURITY-2 reconciliation). The
initial "correct_id decreased" verdict was a metric-basis artifact. Evidence:
`docs/evidence/cp_purity_2/`.

## D1 GROUP Formation (architectural understanding from CP-PURITY-3)

- D1 forms GROUPs from tracklet **LIFECYCLE EVENTS** (merge/split), NOT proximity.
  A GROUP needs a second tracklet born/dying near a carrier, persisting >=
  min_group_duration_frames (10). Sustained proximity alone does NOT create a group.
- **Capacity=2 is D3 metadata**, NOT enforced in D1 (d1_graph_build.py:1179). D1 stamps
  it on nodes; the solver uses it. Any capacity reasoning belongs to D3.
- **GROUPs are structurally unnecessary when detection is perfect.** GT→D oracle with
  one clean tracklet per person produced 0 GROUP nodes with 0 logic gaps (CP-PURITY-3).
  GROUPs compensate for detection under-segmentation.

## Remaining Fixes

- Path-based tag propagation (non-tagged tracklets on same path inherit tag anchor)
- CP-TAG-4b: Hard ping connectivity (gated on CP21 appearance costs)

## Tools

| Tool | Purpose |
|------|---------|
| `tools/tag_fullscan.py` | Standalone full-frame AprilTag scan (no pipeline dependency) |
| `tools/tag_experiment.py` | Dense GT + three-way tag comparison orchestrator |

## Dense GT

- Manifest: `configs/models/bjj-detect-all-cameras-dense.yaml` (stride=1 for J_EDEw)
- CVAT zips contain interpolated labels at every frame
- Dense evaluation confirms stride-10 is representative (±0.3pp)
