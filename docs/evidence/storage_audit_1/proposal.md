# STORAGE-AUDIT-1: Disk Space Inventory & Deletion Proposal

**Date:** 2026-06-29
**Disk:** 228 GB total, **6.6 GB free (97% full)** — critical
**Repo footprint measured:** ~33.7 GB

---

## 1. Top-Level Footprint Summary

| Area | Size | Notes |
|------|------|-------|
| `outputs/` | 16.0 GB | Pipeline runs, eval, baselines, debug |
| `data/` | 13.0 GB | Training data, raw footage, colab packages |
| `outputs_cp20_baseline/` | 4.1 GB | Named frozen baseline |
| `.git/` | 347 MB | Git history |
| `models/` | 234 MB | Model weights + CoreML exports |
| `runs/` | 113 MB | YOLO training run artifacts |
| `calibration_results/` | 1.3 MB | Calibration outputs |
| `configs/cameras/*/` | 14.7 MB | Includes regenerable roi_mask.png |

---

## 2. outputs/ Breakdown (16.0 GB)

### 2a. Main gym outputs (gym c8a592a4, 8.3 GB)

| Camera | Size | Clips | Stage F mp4s | _debug |
|--------|------|-------|-------------|--------|
| J_EDEw | 5.6 GB | 12 clips | ~1.3 GB | ~1.0 GB |
| FP7oJQ | 1.6 GB | 12 clips | ~400 MB | ~1.1 GB |
| PPDmUg | 759 MB | 12 clips | ~780 MB | ~660 MB |
| sessions | 350 MB | 1 session | — | 1 MB |

**Largest single clip:** J_EDEw-200015 = 3.2 GB (stage_F alone = 3.0 GB)

**_debug totals across main gym:** 3.5 GB (annotated.mp4 videos dominate)

**stage_F export mp4 totals:** ~2.5 GB (match clip exports across all cameras)

### 2b. Test gym outputs (gym 00000000, 1.3 GB)

Two-clip validation from CP23b (2026-05-23). 4 clips across FP7oJQ + J_EDEw + PPDmUg.
Not referenced by any evidence doc. Not a named baseline.

| Clip | Size |
|------|------|
| FP7oJQ-143651 | 454 MB |
| FP7oJQ-144154 | 236 MB |
| J_EDEw-143649 | 275 MB |
| J_EDEw-144150 | 210 MB |
| PPDmUg/ | 153 MB |

### 2c. NMS sweep (1.1 GB)

CP7 NMS experiment (iou 0.7, 0.85, 0.9, 0.95). Results summarized in
`docs/cp7_pre4_nms_experiment.md`. Experiment complete — findings documented.
Not referenced by any evidence provenance file.

### 2d. Evaluation directories

| Directory | Size | Notes |
|-----------|------|-------|
| `_eval/` | 1.1 GB | Current eval (stage_f mp4s = 922 MB) |
| `_eval_baseline_v2/` | 1.1 GB | v2 model baseline |
| `_eval_baseline_v1/` | 581 MB | v1 model baseline |
| `_eval_baseline_penalty_15/` | 351 MB | CP1 baseline — referenced by docs |
| `_eval_baseline_cp2_penalty_25/` | 335 MB | CP2 baseline — referenced by docs |
| `_eval_baseline_cp5_pre/` | 332 MB | CP5 pre-fix baseline |
| `_eval_baseline_cp4_pre/` | 332 MB | CP4 pre-fix baseline |
| `_eval_baseline_cp3b_pre/` | 315 MB | CP3b pre-fix baseline |
| `_eval_gt/` | 295 MB | GT pipeline outputs (hard links) |
| `_eval_gt_baseline_v2/` | 198 MB | GT baseline v2 |
| `_eval_gt_baseline_v1/` | 181 MB | GT baseline v1 |
| `_eval_gt_baseline_cp5_pre/` | 159 MB | GT baseline cp5-pre |
| `_eval_gt_oracle/` | 38 MB | CP-PURITY-3 oracle experiment |

**Total eval baselines:** ~2.2 GB (`_eval_baseline_*`)
**Total eval_gt baselines:** ~538 MB (`_eval_gt_baseline_*`)

### 2e. Other

| Directory | Size | Notes |
|-----------|------|-------|
| `_benchmarks/` | 101 MB | Model comparison videos/images |
| `_debug/` | 78 MB | Top-level debug (bbox tiers, FP7 investigation) |
| `_analysis/` | 10 MB | Analysis outputs |
| `_gt_ceiling/` | 1.3 MB | Small |
| `_experiments/` | 28 KB | Tiny |
| `background/` | 344 KB | Tiny |

---

## 3. data/ Breakdown (13.0 GB)

| Area | Size | Notes |
|------|------|-------|
| `training_data/` | 6.0 GB | Datasets + zips (hybrid=4.0GB, det=515MB, etc.) |
| `colab_package/` | 5.1 GB | Upload zips for Kaggle/Colab |
| `raw/nest/` | 1.4 GB | Source footage (IRREPLACEABLE) |
| `cvat_tasks/` | 710 MB | CVAT source videos (annotation workflow) |
| `background_models/` | 3.7 MB | Per-camera .npy |

### Training data detail

| Path | Size | Status |
|------|------|--------|
| `hybrid/` | 4.0 GB | r2_bbox 20x upsampled + vicos_12k |
| `detection_all_cameras_v2/` | 301 MB | **Active** — current model dataset |
| `training_data_detection_all_cameras_v2.zip` | 292 MB | Zip of above |
| `detection_all_cameras/` | 214 MB | v1 dataset (superseded by v2) |
| `training_data_detection_all_cameras.zip` | 208 MB | Zip of v1 |
| `r2_bbox/` | 155 MB | Round 2 bbox-only |
| `combined/` | 155 MB | R1+R2 combined |
| `unpacked_cvat_video/` | 127 MB | Intermediate extraction |
| `training_FP7oJQ_clip1_0-300.zip` | 127 MB | Raw CVAT export zip |
| `round1/` | 96 MB | Round 1 dataset |
| `round2/` | 59 MB | Round 2 dataset |
| `round2_unpacked/` | 53 MB | Intermediate |
| `unpacked_yolo_pose/` | 36 MB | Intermediate |
| `unpacked_coco/` | 32 MB | Intermediate |
| Various CVAT zips | ~85 MB | Raw CVAT exports |
| `unpacked_yolo_obbox/` | 18 MB | Intermediate |
| `filtered/` | 18 MB | Intermediate |

### Colab package detail

| File | Size | Notes |
|------|------|-------|
| `training_data_hybrid.zip` | 3.8 GB | Hybrid dataset for cloud upload |
| `vicos_12k.zip` | 961 MB | ViCoS subsample |
| `training_data.zip` | 149 MB | R1+R2 combined |
| `training_data_r2bbox.zip` | 148 MB | R2 bbox-only |
| `bjj-pose-r1.pt` | 18 MB | Model copy for upload |
| `yolo26n-pose.pt` | 7.5 MB | Model copy for upload |

---

## 4. Disposition Classifications

### PRESERVE — Source of Truth

| Path | Size | Rationale |
|------|------|-----------|
| `outputs/_eval/gt2actuals/` | 48 MB | Current GT2ACTUALS error map — tuning phase source of truth |
| `outputs/_eval_gt/` | 295 MB | GT pipeline outputs — hard links, evidence-referenced |
| `outputs/_eval/signal_trace/` | 11 MB | Signal trace results (referenced by evidence) |
| `outputs/_eval/stage_a/` | 77 MB | Current eval stage A results |
| `outputs/_eval/stage_d/` | 14 MB | Current eval stage D results |
| `outputs/_eval/_debug/` | 1.8 MB | CP7 pre-8/9/10 evidence (referenced by docs) |
| `outputs/_eval_gt_oracle/` | 38 MB | CP-PURITY-3 oracle results (evidence-referenced) |
| `outputs/c8a592a4.../sessions/` | 350 MB | Session pipeline outputs (evidence-referenced) |
| `models/bjj-detect-all-cameras-v2.pt` | 5.1 MB | **Active** detection model |
| `models/bjj-detect-all-cameras-v2.mlpackage` | 4.8 MB | Active CoreML export |
| `models/bjj-detect-all-cameras.pt` | 5.1 MB | v1 model (baseline reference) |
| `models/bjj-detect-all-cameras.mlpackage` | 4.8 MB | v1 CoreML |
| `models/yolo26n-pose.pt` | 7.5 MB | Stock model (never modify) |
| `models/yolo26n.pt` | 5.3 MB | Stock detection base |
| `data/raw/nest/` | 1.4 GB | **IRREPLACEABLE** source footage |
| `data/training_data/detection_all_cameras_v2/` | 301 MB | Active model's training dataset |
| `data/training_data/round1/` | 96 MB | Source annotation data |
| `data/training_data/round2/` | 59 MB | Source annotation data |
| `data/training_data/training_J_EDEw_bbox_video2.zip` | 2.9 MB | Raw CVAT export (v2 source) |
| `data/cvat_tasks/` | 710 MB | CVAT source videos (annotation workflow) |
| `data/background_models/` | 3.7 MB | Per-camera background models |
| `configs/cameras/*/` | 14.7 MB | Camera configs + calibration |

**Subtotal PRESERVE-SOT:** ~3.3 GB

### PRESERVE — Baseline

| Path | Size | Rationale |
|------|------|-----------|
| `outputs_cp20_baseline/` | 4.1 GB | Named frozen CP20 baseline |
| `outputs/_eval_baseline_v2/` | 1.1 GB | v2 model eval baseline |
| `outputs/_eval_baseline_v1/` | 581 MB | v1 model eval baseline |
| `outputs/_eval_gt_baseline_v2/` | 198 MB | GT baseline v2 |
| `outputs/_eval_gt_baseline_v1/` | 181 MB | GT baseline v1 |

**Subtotal PRESERVE-BASELINE:** ~6.2 GB

### CANDIDATE DELETE — Superseded Baselines

These are intermediate checkpoints from the CP0-CP6 penalty-tuning arc.
All findings are documented in `docs/cp{1,2,2.5,3,3b,4,5}_*.md`. The arc
concluded with CP5 (parallel-carrier consolidation). The penalty_15 baseline
is referenced by `docs/` but only for the report format, not for the data itself.
These baselines predate the current v2 model and CP5 fix — they represent
pipeline states that no longer exist.

| Path | Size | Superseded by | Evidence ref? |
|------|------|--------------|---------------|
| `outputs/_eval_baseline_penalty_15/` | 351 MB | _eval_baseline_v1 (later, same model) | Generic `docs/` reference to format, not data |
| `outputs/_eval_baseline_cp2_penalty_25/` | 335 MB | _eval_baseline_v1 | Generic `docs/` reference to format |
| `outputs/_eval_baseline_cp5_pre/` | 332 MB | _eval_baseline_v1 (post-CP5 is current) | No evidence ref |
| `outputs/_eval_baseline_cp4_pre/` | 332 MB | _eval_baseline_cp5_pre → v1 | No evidence ref |
| `outputs/_eval_baseline_cp3b_pre/` | 315 MB | _eval_baseline_cp4_pre → v1 | No evidence ref |
| `outputs/_eval_gt_baseline_cp5_pre/` | 159 MB | _eval_gt_baseline_v1 | No evidence ref |

**Subtotal CANDIDATE DELETE — Superseded baselines:** ~1.8 GB

**Safety check:** None of these are under `gt2actuals/` or referenced by
`docs/evidence/` provenance files. The penalty_15 and cp2 references in
`docs/` are to the *findings* (documented in markdown), not to the output
artifacts themselves. The CLAUDE.md notes these are "lite-mode only" (lack
pipeline artifacts for full-mode trace) — they cannot be used for new analysis.

### CANDIDATE DELETE — Debug/Scratch

| Path | Size | Rationale |
|------|------|-----------|
| Main gym `_debug/` (all clips) | 3.5 GB | Annotated debug mp4s. Regenerable by re-running pipeline. Largest single cost center. |
| `outputs/_debug/` (top-level) | 78 MB | bbox_tiers, fp7 investigation — regenerable |
| `outputs/00000000.../` (test gym) | 1.3 GB | CP23b two-clip validation. Not evidence-referenced, not a baseline. Superseded by current eval. |
| `outputs/_nms_sweep/` | 1.1 GB | CP7 NMS experiment. Findings in docs. Not evidence-referenced. |
| `data/training_data/unpacked_cvat_video/` | 127 MB | Intermediate extraction — regenerable from zips |
| `data/training_data/round2_unpacked/` | 53 MB | Intermediate — regenerable from zips |
| `data/training_data/unpacked_yolo_pose/` | 36 MB | Intermediate — regenerable from zips |
| `data/training_data/unpacked_coco/` | 32 MB | Intermediate — regenerable from zips |
| `data/training_data/unpacked_yolo_obbox/` | 18 MB | Intermediate — regenerable from zips |
| `data/training_data/filtered/` | 18 MB | Intermediate — regenerable |
| `runs/` | 113 MB | Training run artifacts (weights in models/, results in docs) |
| `outputs/_benchmarks/` | 101 MB | Model comparison videos — regenerable via tools/compare_models.py |

**Subtotal CANDIDATE DELETE — Debug/scratch:** ~6.5 GB

### CANDIDATE DELETE — Regenerable Large Media

| Path | Size | Rationale |
|------|------|-----------|
| Main gym `stage_F/` mp4 exports | ~2.5 GB | Match clip mp4s. Regenerable by re-running stage F. Not needed for tuning analysis (tuning compares stage A/D metrics, not mp4s). |
| `outputs/_eval/stage_f/` | 922 MB | match_preview.mp4s — regenerable via `pipeline_validation evaluate` |
| `outputs/_eval/tracker_swap/` | 66 MB | CP-SWAP diagnostic — findings complete, data regenerable |

**Subtotal CANDIDATE DELETE — Regenerable media:** ~3.5 GB

### CANDIDATE DELETE — Superseded/Duplicate Training Data

| Path | Size | Rationale |
|------|------|-----------|
| `data/training_data/hybrid/` | 4.0 GB | Superseded — hybrid model not active, data regenerable from r2_bbox + vicos |
| `data/colab_package/training_data_hybrid.zip` | 3.8 GB | Zip of above — regenerable |
| `data/training_data/detection_all_cameras/` | 214 MB | v1 dataset — superseded by v2 |
| `data/training_data/training_data_detection_all_cameras.zip` | 208 MB | Zip of v1 — superseded |
| `data/training_data/r2_bbox/` | 155 MB | Intermediate for hybrid — not active |
| `data/training_data/combined/` | 155 MB | R1+R2 combined — not active |
| `data/colab_package/vicos_12k.zip` | 961 MB | ViCoS subsample — not active, downloadable |
| `data/colab_package/training_data.zip` | 149 MB | Old combined dataset zip |
| `data/colab_package/training_data_r2bbox.zip` | 148 MB | Old r2_bbox zip |
| `data/colab_package/bjj-pose-r1.pt` | 18 MB | Duplicate of models/bjj-pose-r1.pt |
| `data/colab_package/yolo26n-pose.pt` | 7.5 MB | Duplicate of models/yolo26n-pose.pt |

**Subtotal CANDIDATE DELETE — Superseded training data:** ~9.8 GB

### UNCERTAIN — Needs Human Decision

| Path | Size | Question |
|------|------|----------|
| `outputs_cp20_baseline/` | 4.1 GB | Named baseline but very large. Is the pre-CP20 pipeline state still needed for comparison, or have CP20+ changes been validated enough to discard? Default: PRESERVE. |
| `data/raw/nest/00000000.../` | 614 MB | Test gym raw footage. Irreplaceable if deleted — but is this gym still needed? |
| `data/raw/nest/calibration_test/` | 26 MB | Calibration test footage — still needed for re-calibration? |
| `data/training_data/training_FP7oJQ_clip1_0-300.zip` | 127 MB | Raw CVAT export — redundant with round1/? Or needed as provenance? |
| `data/training_data/training_data_detection_all_cameras_v2.zip` | 292 MB | Zip of active v2 dataset. Redundant with unpacked dir, but useful as Kaggle upload artifact. |
| `models/bjj-pose-r1.pt` | 18 MB | Round 1 pose model — historical, not active |
| `models/bjj-pose-hybrid.pt` | 17 MB | Hybrid pose model — not active |
| `models/bjj-pose-vicos.pt` | 7.5 MB | ViCoS model — not active |
| `models/bjj-pose-r2.pt` | 7.4 MB | R2 pose model — not active |
| `models/bjj-pose-r2_bbox.pt` | 7.4 MB | R2 bbox model — not active |
| `models/yolov8s-*.pt` (3 files) | 67 MB | Stock YOLOv8s models — why are they here? |
| `models/yolo11*-pose.pt` (2 files) | 25 MB | Stock YOLO11 models — still needed? |
| `models/yolov8n-pose.*` (pt+mlpackage) | 13 MB | Old stock pose model |
| `models/yolov8n.pt` | 6.3 MB | Old stock detection model |
| `models/osnet_x0_25_msmt17.pt` | 2.9 MB | ReID model — rejected (CP-REID-1) |
| `models/bjj-pose-r1.mlpackage` | 5.9 MB | CoreML of non-active model |
| `models/yolo26n-pose.mlpackage` | 5.9 MB | CoreML of stock pose — not used in prod |
| `models/yolov8n-pose.mlpackage` | 6.5 MB | CoreML of old stock — not used |
| `data/cvat_tasks/test_import/` | 19 MB | Test import directory — still needed? |

**Subtotal UNCERTAIN:** ~5.3 GB (dominated by cp20_baseline at 4.1 GB)

---

## 5. Reclaimable Space Summary

| Category | Estimated Size |
|----------|---------------|
| CANDIDATE DELETE — Debug/scratch | **6.5 GB** |
| CANDIDATE DELETE — Regenerable media | **3.5 GB** |
| CANDIDATE DELETE — Superseded training data | **9.8 GB** |
| CANDIDATE DELETE — Superseded baselines | **1.8 GB** |
| **Total CANDIDATE DELETE** | **~21.6 GB** |
| UNCERTAIN (if approved) | ~5.3 GB |
| **Grand total if all approved** | **~26.9 GB** |

**Immediate relief from high-confidence deletes alone: ~21.6 GB** — would bring
free space from 6.6 GB to ~28 GB, enough headroom for the Stage A tuning phase.

---

## 6. Priority Order for Deletion (by size, high-confidence first)

1. **data/training_data/hybrid/** + **data/colab_package/training_data_hybrid.zip** = 7.8 GB
   - Not active, regenerable. Largest single win.
2. **Main gym _debug/ dirs** = 3.5 GB
   - All annotated.mp4 debug videos. Regenerable by pipeline re-run.
3. **outputs/_nms_sweep/** = 1.1 GB
   - CP7 NMS experiment complete. Findings documented.
4. **outputs/00000000.../** (test gym) = 1.3 GB
   - CP23b validation complete. Not evidence-referenced.
5. **data/colab_package/vicos_12k.zip** = 961 MB
   - ViCoS subsample. Downloadable from source. Not active.
6. **outputs/_eval/stage_f/** = 922 MB
   - match_preview mp4s. Regenerable.
7. **Main gym stage_F/ mp4 exports** = ~2.5 GB
   - Match clips. Regenerable.
8. **Superseded eval baselines** (penalty_15 through cp3b_pre) = 1.8 GB
   - Pre-v2, pre-CP5. Findings documented. Cannot be used for new analysis.
9. **Superseded training data** (v1 detection, r2_bbox, combined, intermediates) = ~1.0 GB
   - Superseded by v2 or regenerable from zips.
10. **data/colab_package/** remaining non-hybrid = ~322 MB
    - Old upload packages.
11. **runs/** = 113 MB
    - Training artifacts. Final weights already in models/.
12. **outputs/_benchmarks/** = 101 MB
    - Comparison videos. Regenerable.

---

## 7. Evidence Cross-Reference Verification

Paths referenced in `docs/evidence/`:
- `outputs/_eval_gt_oracle/J_EDEw/...` — **PRESERVED** (38 MB)
- `outputs/_eval_gt/sessions/.../cp_tag_3_baseline` — **PRESERVED** (in _eval_gt)
- `outputs/_eval_gt/J_EDEw/...` — **PRESERVED** (in _eval_gt)
- `outputs/c8a592a4.../sessions/...` — **PRESERVED** (350 MB)
- `outputs/c8a592a4.../J_EDEw/.../stage_D/` — **PRESERVED** (stage_D data not proposed for delete)
- `outputs/_eval/_debug/cp7_pre*` — **PRESERVED** (in _eval/_debug, 1.8 MB)
- `outputs/_eval/signal_trace/bjj-detect-all-cameras_pre_fix/` — **PRESERVED** (in signal_trace)

Paths referenced in `docs/*.md`:
- `outputs/_eval_baseline_penalty_15/` — referenced for format/structure, not data. **CANDIDATE DELETE** — findings fully documented in `docs/cp1_evidence.md` et al.
- `outputs/_eval_baseline_cp2_penalty_25/` — same situation. **CANDIDATE DELETE**.
- `outputs/_eval_gt/{cam}/.../stage_*` — **PRESERVED** (in _eval_gt).

**Confirmed: NO delete candidate overlaps with gt2actuals, evidence-referenced paths, or current _eval_gt.**

---

## 8. What This Proposal Does NOT Delete

- `outputs/_eval/gt2actuals/` — source of truth for tuning phase
- `outputs/_eval_gt/` — GT pipeline artifacts (hard links, evidence-referenced)
- `outputs/_eval_gt_oracle/` — CP-PURITY-3 evidence
- `outputs/_eval/signal_trace/` — signal trace results
- `outputs/_eval/_debug/` — CP7 evidence artifacts
- `data/raw/nest/` — irreplaceable source footage
- `data/cvat_tasks/` — annotation source videos
- `data/training_data/detection_all_cameras_v2/` — active model's dataset
- `data/training_data/round1/`, `round2/` — source annotation data
- Any `docs/evidence/` files
- Any `src/` files
- Current model weights (`bjj-detect-all-cameras-v2.*`, `yolo26n-pose.pt`, `yolo26n.pt`)
- Main gym `stage_A/`, `stage_B/`, `stage_C/`, `stage_D/`, `stage_E/` data (only _debug + stage_F mp4s proposed)
- Session pipeline outputs (evidence-referenced)

---

## 9. Execution Record (2026-06-29)

**Starting state:** 228 GB disk, 5.7 GB free (97.5% full)

### TIER 1 — Debug/scratch/intermediates (~13.3 GB reclaimed)

| Path | Size | Action | Result |
|------|------|--------|--------|
| Main gym `_debug/` (36 clip dirs) | ~2.6 GB | DELETED | Annotated mp4s, regenerable by pipeline re-run |
| Main gym `stage_F/` (36 clip dirs) | ~2.5 GB | DELETED | Match clip mp4 exports, regenerable |
| `outputs/_eval/stage_f/` | 922 MB | DELETED | match_preview mp4s, regenerable |
| `outputs/_eval/tracker_swap/` | 66 MB | DELETED | CP-SWAP diagnostic complete |
| `outputs/_nms_sweep/` | 1.1 GB | DELETED | CP7 NMS experiment, findings documented |
| `outputs/00000000.../` (test gym) | 1.3 GB | DELETED | No GT/annotations, not evidence-referenced |
| `outputs/_debug/` (top-level) | 78 MB | DELETED | bbox_tiers, fp7 investigation, regenerable |
| `outputs/_benchmarks/` | 101 MB | DELETED | Model comparison videos, regenerable |
| Training intermediates (6 dirs) | 284 MB | DELETED | unpacked_*, filtered, round2_unpacked |
| v1 detection dataset + zip | 422 MB | DELETED | Superseded by v2 |
| Old colab packages (5 files) | ~1.28 GB | DELETED | vicos_12k.zip, old zips, duplicate model copies |
| `runs/` | 113 MB | DELETED | Training artifacts, weights already in models/ |

**Post-TIER-1:** 19 GB free

### TIER 2 — Dataset (hybrid, ~8 GB reclaimed)

| Path | Size | Action | Result |
|------|------|--------|--------|
| `data/colab_package/training_data_hybrid.zip` | 3.8 GB | DELETED | Redundant zip of unzipped dataset |
| `data/training_data/hybrid/` | 4.0 GB | DELETED | ViCoS source not on disk but downloadable via `tools/download_vicos.py`; assembly script `tools/prepare_3way_datasets.py` + `data/training_data/r2_bbox/` exist; model weights preserved at `models/bjj-pose-hybrid.pt` |

**Post-TIER-2:** 27 GB free

### TIER 3 — Baselines (~6 GB reclaimed)

Per-path evidence re-check performed. Each verified against `docs/evidence/` and
`docs/*.md`. Two baselines (penalty_15, cp2) are referenced in `docs/cp2_results.md`
and `docs/cp3_results.md` for the findings (fully documented in markdown), not for
the data artifacts themselves. All are lite-mode only (pre-v2/pre-CP5, cannot be
used for full-mode trace).

| Path | Size | Action | Evidence ref? |
|------|------|--------|--------------|
| `outputs_cp20_baseline/` | 4.1 GB | DELETED | None |
| `outputs/_eval_baseline_penalty_15/` | 351 MB | DELETED | docs/cp2_results.md (findings only) |
| `outputs/_eval_baseline_cp2_penalty_25/` | 335 MB | DELETED | docs/cp3_results.md (findings only) |
| `outputs/_eval_baseline_cp5_pre/` | 332 MB | DELETED | None |
| `outputs/_eval_baseline_cp4_pre/` | 332 MB | DELETED | None |
| `outputs/_eval_baseline_cp3b_pre/` | 315 MB | DELETED | None |
| `outputs/_eval_gt_baseline_cp5_pre/` | 159 MB | DELETED | None |

**Post-TIER-3:** 33 GB free

### Summary

| Metric | Before | After |
|--------|--------|-------|
| Free disk | 5.7 GB | 33 GB |
| Reclaimed | — | **~27.3 GB** |
| Disk utilization | 97.5% | 85.5% |

### Preserved (confirmed intact)

- `outputs/_eval/gt2actuals/` (48 MB) — tuning phase source of truth
- `outputs/_eval_gt/` (295 MB) — GT pipeline hard links, evidence-referenced
- `outputs/_eval_gt_oracle/` (38 MB) — CP-PURITY-3 oracle evidence
- `outputs/_eval/signal_trace/` (11 MB) — signal trace results
- `outputs/_eval/_debug/` (1.8 MB) — CP7 evidence artifacts
- `outputs/_eval/stage_a/` (77 MB) — current eval detection results
- `outputs/_eval/stage_d/` (14 MB) — current eval identity results
- `outputs/_eval_baseline_v2/` (1.1 GB) — v2 model baseline
- `outputs/_eval_baseline_v1/` (581 MB) — v1 model baseline
- `outputs/_eval_gt_baseline_v2/` (198 MB) — GT baseline v2
- `outputs/_eval_gt_baseline_v1/` (181 MB) — GT baseline v1
- `data/raw/nest/` (1.4 GB) — irreplaceable source footage
- `data/training_data/detection_all_cameras_v2/` (301 MB) — active model dataset
- `data/cvat_tasks/` (710 MB) — annotation source videos
- Main gym `stage_A/`-`stage_E/` pipeline data — all intact
- All `docs/evidence/` files — untouched
- All `src/` files — untouched
- All current model weights — untouched
