# CP4.F Results — Retire the session fps scalar (site #1)

**Date:** 2026-08-26
**Commit:** (this commit)
**Pipeline state:** `variable_dt: false`, post-CP4.F, recalibrated H (`f7d76d6`)

---

## 1. What changed

Behaviour-neutral cleanup. No timing arithmetic changed.

1. **`d1_graph_build.py:1291`** — stale `fps is not None` guard removed from reconnect
   block. Nothing inside computes with fps (CP4.D removed all uses). The guard would have
   silently disabled reconnect edge generation if fps became unavailable.

2. **`derive_clip_frame_offset`** — deprecated, not deleted. Three tools/ callers remain
   (`cp_purity_3_oracle`, `cp_tag_3_evidence`, `analyze_recorder_timing`). Backlog item filed.

3. **`d2_run.py`** — fps precondition (`raise ValueError`) dropped. fps is now nullable in
   audit output (None when unavailable, not 0.0). No audit consumer does arithmetic on the
   field. `:17` comment reconciled from "fps source of truth" to "fps: diagnostic only."

4. **`SessionManifest.fps`** — marked as Piece 5 residual, consumed only by
   `cross_camera_evidence.py` (#8).

5. **`aggregate_session_bank`** — fps parameter removed. Was vestigial (CP4.C made frame
   offsets cumulative and timestamp offsets sidecar-anchored).

---

## 2. fps classification inventory

| Location | Use | Classification |
|----------|-----|---------------|
| `frame_iterator.py:35` | Reads `CAP_PROP_FPS` from video | Infrastructure (manifest population) |
| `config/models.py:58` | Config field | Schema |
| `SessionManifest.fps` | Threaded to `cross_camera_evidence` | **Piece 5 residual** (#8) |
| `derive_clip_frame_offset` | Dead in pipeline, live in tools/ | **DEPRECATED** |
| `d2_run.py:117` | Nullable in audit output | Audit-only |
| `d1_graph_build.py:317` | Extracted, used only in audit at `:2529` | Audit-only |
| `cross_camera_evidence.py:275` | `window_frames = temporal_window_s * fps` | **Computation** — site #8, Piece 5 |
| `multiplex_runner.py:391-406` | fps backfill + manifest write | Infrastructure |

Site #1 (`session_d_run.py:491`) is reduced to one consumer: `cross_camera_evidence` (#8).
It cannot be fully eliminated until Piece 5 removes #8.

---

## 3. Null test: all metrics identical to CP4.E

**Basis:** camera FP7oJQ, `variable_dt: false`, H=`f7d76d6`, output dirs deleted, D0.5 from
fresh summaries.

| Metric | CP4.E | CP4.F | Identical? |
|--------|-------|-------|-----------|
| 130229 speed/d05/recon/persons | 40.58/44/92/106 | 40.58/44/92/106 | ✅ |
| 131129 speed/d05/recon/persons | 31.55/35/886/10 | 31.55/35/886/10 | ✅ |
| 132650 speed/d05/recon/persons | 53.3/21/827/17 | 53.3/21/827/17 | ✅ |
| Session recon | 1805 | 1805 | ✅ |
| Cross-clip decomposition | 0 | 0 | ✅ |
| Session persons | 126 | 126 | ✅ |
| Stage E | 1/3 crashed | 1/3 crashed | ✅ |

**No metric moved.** The change is behaviour-neutral as designed.

**fps-nullable audit verification:** No downstream code does arithmetic on the fps field
in audit dicts. `None` serialises to JSON `null`. No None guard needed.
