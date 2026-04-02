---
paths:
  - "src/bjj_pipeline/stages/stitch/**"
---

# Cross-Camera Identity (CP17 + CP14f)

## Two-Pass Architecture (CP17)
- **Pass 1:** Each camera solves its ILP independently (existing D3).
- **Pass 2:** Each camera re-solves with cross-camera evidence injected between D2 and D3
  via `constraints_overlay` parameter to `session_d_run.py`.

## Evidence Channels

### Tag Channel (Tier 1 — active)
Hard must_link constraints for high-confidence tag co-observations (same tag_id on 2+
cameras). Deterministic, forces assignment. `corroboration_miss_multiplier` default 10x
boosts MCF-2a miss penalty and MCF-3a must-link penalty.

### Coordinate Channel (Tier 2 — implemented, disabled by default)
`build_cross_camera_coordinate_evidence()` in `cross_camera_evidence.py`. Compares D4
person track world coordinates across camera pairs via rolling-window spatial proximity.

**Algorithm:**
1. Load `person_tracks_{cam_id}.parquet` (D4 output) per camera — uses final
   x_m/y_m (repaired coordinates already folded in by D4).
2. For each cross-camera person pair, compute frame-level spatial proximity using
   `pd.merge_asof` with configurable temporal tolerance.
3. Rolling window analysis: fraction of windows where median distance < threshold.
4. Binary gate: spatial_agreement_ratio >= threshold → coordinate-corroborated.
5. Tag linkage: if either person has a tag, that tag gets added to `corroborated_tags`
   (same dict the solver reads, same 10x boost — no solver changes needed).
6. Conflict detection (Signal C): if both persons have different tags but are spatially
   corroborated, log as `coordinate_conflict` to session audit JSONL. Audit-only for
   now — no user-facing behavior until validated on real sessions.

**Config** (`cross_camera.coordinate_evidence`):
```yaml
enabled: false                # disabled until validated
temporal_window_s: 2.5        # rolling window size
temporal_tolerance_s: 2.0     # max clock offset for frame matching
proximity_threshold_m: 0.5    # max distance to count as proximate
agreement_ratio_threshold: 0.6  # min fraction of proximate windows
```

**fps:** Passed as explicit parameter from `SessionManifest.fps` (authoritative source).

## Evidence Builder
- `cross_camera_evidence.py` builds both tag and coordinate evidence from Pass 1 outputs.
- Tag evidence: `build_cross_camera_tag_evidence()` — identity_assignments JSONL.
- Coordinate evidence: `build_cross_camera_coordinate_evidence()` — person_tracks parquet.
- Processor calls tag evidence first, then coordinate evidence (if enabled), merges
  coordinate-corroborated tags into the tag evidence dict before injecting into overlay.

## CP14f Post-Hoc Merge (fallback)
- `cross_camera_merge.py` — union-find on shared tags after per-camera ILP.
- Presence-based: same tag_id on 2+ cameras = same athlete. Filters by min_tag_observations
  (>=2) and min_assignment_confidence (>=0.5).
- Intra-camera dedup: at most one (cam_id, person_id) per (cam_id, tag_id).
- Deterministic global IDs: `gp_` prefix + sha256 of sorted member keys.
- Remains as baseline. CP17 is the primary path.

## Graceful Degradation
Tag-only → tag+coordinate evidence (CP17 Tier 1+2).
Each tier adds signal; lower tiers never break.

## Temporal Sync Tiers (planned)
- **Tier A (current):** Filename-derived clock alignment. Session boundary is the
  temporal constraint.
- **Tier B (planned):** Buzzer audio detection for sub-second cross-camera sync.
- **Tier C (planned):** Spatial ICP registration for sub-frame alignment.

## Undistortion Audit (2026-04-02)
All 9 pipeline code paths verified correct. Convention: u_px/v_px = raw pixel space,
x_m/y_m = world via `project_to_world()`. See `docs/undistortion_audit.md`.

## Processor Flow
Loop 1 (D+E per camera) → cross-camera merge → CP17 Pass 2 (tag evidence + coordinate
evidence → merged overlay → re-solve per camera) → re-merge → Loop 2 (F per camera).
Merge failure logs error and passes empty map — never blocks Stage F.
