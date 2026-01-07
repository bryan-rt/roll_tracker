# D5 — MCF Birth/Death + Mat-Zone Gating

## Update: Locked constraints to honor (F0 + F3)

- **F3 ingest contract**: clips live at `data/raw/nest/<camera_id>/YYYY-MM-DD/HH/<camera_id>-YYYYMMDD-HHMMSS.mp4`
- **Processing reads only** from `data/raw/**` and **writes only** to `outputs/<clip_id>/...`
- **F0 contracts are authoritative**: do not invent schemas; use `src/bjj_pipeline/contracts/*`
- **Run anchor**: `outputs/<clip_id>/clip_manifest.json`
- **Stage artifacts are locked** (Parquet/JSONL + masks `.npz`), and paths must be **relative** to `outputs/<clip_id>/`

---

## Why this worker exists
A Min-Cost Flow (MCF) stitcher needs explicit **birth** and **death** edges so identities can enter/exit the scene naturally. In a gym, people also exist **off-mat**, and we do not want the solver to waste capacity trying to track them or incorrectly stitch on/off-mat fragments.

This worker defines:
- how to model **entry/exit** in the MCF graph
- how to use the **mat polygon** to gate tracklets (“on mat” vs “off mat”)
- how costs change depending on boundary proximity and context

---

## Scope
### In-scope
- Mat polygon + zones usage (from `mat_blueprint.json` in meters)
- Definitions:
  - on-mat “active zone”
  - boundary “entry/exit zone” (buffer ring)
- Birth/death edges:
  - which nodes get source/sink edges
  - cost formulas (low at boundaries, high mid-mat)
- Constraints interacting with AprilTags:
  - cannot “birth” a second instance of the same `tag:<id>` in overlapping time
  - must respect `must_link` / `cannot_link` hints
- How this integrates into existing D1/D2/D3 code modules

### Out-of-scope (for now)
- Multi-camera cross-view re-identification (handled by future work)
- Fully online, real-time gating (we’re offline/batch)

---

## Inputs & Outputs
### Inputs (from locked artifacts)
- Stage A tracklets (frames/summaries)
- Stage B contact points (ground-plane x,y)
- Stage C identity hints (must_link/cannot_link keyed by tracklet_id)
- Stage D graph construction inputs (this worker influences D1/D2)

### Outputs
No new artifact family. This worker changes **how Stage D’s MCF graph is built** and how costs/edges are computed.
- If you add debug outputs, they must be registered in the manifest and remain optional.

---

## Design: Mat-zone gating
Define `is_on_mat(point_xy)` using the mat boundary polygon.
- Tracklet “on-mat score” = fraction of its contact points inside polygon
- Gate rules (POC):
  - if score < threshold (e.g., 0.2), treat as off-mat and exclude from match logic
  - still allow birth/death near boundary (walk-ons)

Define boundary buffers:
- entry/exit zone = within `d_buffer_m` meters of boundary (e.g., 0.5m)

---

## Design: Birth & Death costs
For each tracklet i, compute:
- start point `p_start`, end point `p_end`
- distance to boundary `db_start`, `db_end`

Birth edge: `SOURCE -> i`
- cost low if `db_start` small (near boundary)
- cost high if `db_start` large (starts mid-mat)

Death edge: `i -> SINK`
- cost low if `db_end` small
- cost high if `db_end` large

Rationale:
- prevents “teleport” by encouraging continuation when tracklets end mid-mat.

---

## Interactions with identity anchors (AprilTags)
If tracklet has a strong anchor to `tag:<id>`:
- enforce uniqueness across time:
  - two chains cannot both contain the same tag in overlapping frames
- birth/death behavior:
  - allow “re-entry” across gaps if tag evidence exists later
  - discourage “new person with same tag” by infinite/very large cost

---

## Required deliverables back to Manager
1) Formal definitions for:
   - on-mat scoring
   - entry/exit zones
   - birth/death costs
2) How these modify D1 graph construction + D2 cost function
3) Tests to add (synthetic tracklets):
   - person walks onto mat boundary → allowed birth
   - person disappears mid-mat → high death, solver prefers link to later reappearance
   - off-mat spectators → excluded/low priority
4) Copilot prompt pack to implement in `src/bjj_pipeline/stages/stitch/*`

---

## Kickoff
Please begin by proposing:
1) Exact formulas (with constants) for mat gating and birth/death costs.
2) Where in code these belong (graph vs costs modules).
3) A minimal test plan with fixtures.

End by asking me to approve the plan before implementation.

Also include an explicit bullet confirming alignment with the locked F0/F3 constraints (paths, manifest, artifacts).
