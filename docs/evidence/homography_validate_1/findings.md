# HOMOG-VALIDATE-1: FP7oJQ Homography Regression and Recalibration

**Date:** 2026-08-24
**Camera:** FP7oJQ
**Footage:** `data/raw/nest/00000000-0000-0000-0000-000000000003/FP7oJQ/2026-08-22/13/`
**Recalibration commit:** `f7d76d6`

---

## 4a. What was wrong

`configs/cameras/FP7oJQ/homography.json` switched calibration mode on 2026-04-02 (`220f9dd`)
from `overlay_rect` to a 4-point `interactive_clicks` fit. The result recorded `converged:
False`, `inlier_ratio: 0.653`, with all four correspondences clustered in one 6×8m quad
(x 51.0–57.0, y 42.0–49.9). The lens block also changed in the same commit (f 1281.24 →
950.0, k1 −0.380 → −0.219).

Athletes in the Saturday (2026-08-22) footage projected to x 49.2–55.2, y 49.5–57.4 under
this H — so 86.8% of 8,439 contact points fell outside the region H was fitted to
(segment FP7oJQ-20260822-132650, contact_points.parquet, `correction_matrix = None`).

Note: the lens change from f=1281→950 may have been a DELIBERATE correction (CLAUDE.md
records "Lens calibration bounds fix applied (fixed-f candidate sweep)" as part of CP19/CP20
work). The round f=950.0 is consistent with a fixed-f candidate sweep, while 1281.24 looks
like free-parameter optimisation. The H change, not the lens change, is what was wrong.

---

## 4b. How it was found, and two false starts

Both false starts are reusable traps and are recorded here for that reason.

**False start 1:** Drew the **cached** `projected_polylines[].pixel_points` array onto a
raw (distorted) frame. Those points are in undistorted pixel space (H stored in
`homography.json` is in undistorted pixel space — `.claude/rules/calibration.md:101`). The
comparison was meaningless: straight overlays on a curved image is the signature of
undistorted coordinates on a distorted frame, not a camera shift.

**False start 2:** Correctly undistorted the frame, then drew the same cached array. The
polylines tracked mat features where they existed, but 28 of 32 polylines lay outside
the 6×8m calibrated quad — so this tested homography *extrapolation* rather than
calibration validity. A 4-point fit extrapolating 20m past its correspondences is not a
test of whether the fit is correct.

**Root cause of both:** the verification reimplemented projection instead of calling the
pipeline's own code. **Rule going forward: validate by running production code with
temporary instrumentation attached; never reimplement pipeline logic in a script.** A
reimplementation tests the reimplementation — it can pass while production is broken, or
fail while production is fine, and the output cannot distinguish the two.

**Valid method:** `tools/homography_overlay_check.py` — forward-only `project_to_world()`
over a subsampled pixel grid (step=6, 57,600 samples), contours drawn at integer-metre
crossings on the **raw** frame. No inversion, no frame undistortion, no
`projected_polylines`. If this overlay is wrong, the production path is wrong — which is
the question.

---

## 4c. The fix

Recalibrated 2026-08-24 against `FP7oJQ-20260822-132650.mp4`. H-only (`--skip-lens
--force`); lens block unchanged from `2edce38` (f=950.0, k1=-0.2188, k2=0.0251).

| Metric | Value | Guide criterion | Pass? |
|---|---|---|---|
| `mean_reproj_error_px` | 1.28 | < 3px | PASS |
| `max_reproj_error_px` | 4.09 | — | — |
| `n_matched_lines` | 9 | ≥ 5 | PASS |
| `n_distinct_edges_matched` | 8 | ≥ 3 | PASS |
| `converged` | **True** | True | PASS |
| `inlier_ratio` | 0.547 | — | — |
| `fit.num_points` | 4 | — | — |

Calibrated quad moved from x 51.0–57.0, y 42.0–49.9 (8m y span) to x 51.01–57.00,
y 33.96–56.02 (22m y span, full visible mat length).

**Known property:** the base fit used 4 points (the guide recommends 6–8), so there is
zero click-point redundancy. The quality comes from Step 3's mat-line RANSAC refinement
(47/86 line inliers). A future recalibration should use 6–8 spread points.

**`inlier_ratio` fell from 0.653 to 0.547** while every other metric improved. These are
**not comparable across calibrations** — different candidate line sets over different
spatial extents. Do not record this as a regression.

---

## 4d. Verification

Operator-confirmed via `_debug/mat_view.mp4` on `FP7oJQ-20260822-132650`: athlete
placement matches direct observation of the footage.

---

## 4e. Blast radius (recorded, NOT scoped — do not investigate)

If FP7oJQ's H was wrong from 2026-04-02 to 2026-08-24, every world-coordinate figure
computed on that camera in that window is affected:

- `speed_mps_k` — feeds D0.5 Tier 1 speed cap (48 m/s) and Tier 2 kinematic corroboration
- `max_displacement` — purity proxies (AUC 0.82–0.85 raw, 0.75–0.82 post-D0.5)
- D1 reconnect speed gating (`d1_graph_build.py:1408`)
- CP-SWAP `world_accel` (best single-feature AUC=0.714 on FP7oJQ)
- `correct_id` is frame-indexed and fps-free, but D0.5 splits feed it and D0.5 consumes
  speed, so it is indirectly exposed

FP7oJQ is one of three eval cameras — partial contamination, not total.

---

## 4f. Other cameras (deferred)

`J_EDEw` and `PPDmUg` were last calibrated in a single commit (`fef502d`) and did **not**
go through the April `overlay_rect` → `interactive_clicks` switch, so they are probably
unaffected — but this is unverified. Decision: **defer.** They will be recalibrated when
fresh footage from those cameras is available.

Piece 4 is FP7oJQ-only (Saturday footage, T2.5 baseline, CP4.B–F all single-camera), so
nothing current depends on them.

**Constraint to carry:** the legacy T2 corpus is J_EDEw/PPDmUg, and T2 may only be cited
for "nothing broke structurally." No J_EDEw or PPDmUg world-coordinate number may be cited
as evidence that a timing change *helped* until those cameras are recalibrated.
Frame-space signals are unaffected.
