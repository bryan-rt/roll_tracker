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
- **Tag channel (active):** Hard must_link constraints for high-confidence tag co-observations
  (same tag_id on 2+ cameras). Deterministic, forces assignment. `corroboration_miss_multiplier`
  default 10x boosts MCF-2a miss penalty and MCF-3a must-link penalty.
- **Coordinate channel (stubbed in ILP):** Soft cost modifications — modifies edge costs,
  solver prefers but can override. CP18 corrections now flow through Stage A (corrected
  x_m/y_m), but CP17 Tier 2 coordinate evidence injection into the ILP constraints_overlay
  is still stubbed. Requires validated inter-camera alignment before activation.

## Evidence Builder
- `cross_camera_evidence.py` builds evidence from Pass 1 identity assignments.
- Single round sufficient for tags. Architecture supports multi-round for coordinates (capped).

## CP14f Post-Hoc Merge (fallback)
- `cross_camera_merge.py` — union-find on shared tags after per-camera ILP.
- Presence-based: same tag_id on 2+ cameras = same athlete. Filters by min_tag_observations
  (>=2) and min_assignment_confidence (>=0.5).
- Intra-camera dedup: at most one (cam_id, person_id) per (cam_id, tag_id).
- Deterministic global IDs: `gp_` prefix + sha256 of sorted member keys.
- Remains as baseline. CP17 is the primary path.

## Graceful Degradation
Tag-only → tag+rough coords (CP16+CP18) → tag+precise coords (mat walk).
Each tier adds signal; lower tiers never break.

## Processor Flow
Loop 1 (D+E per camera) → cross-camera merge → CP17 Pass 2 → re-merge → Loop 2 (F per camera).
Merge failure logs error and passes empty map — never blocks Stage F.
