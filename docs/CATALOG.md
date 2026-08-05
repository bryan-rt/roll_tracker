# docs/ Catalog

*Status-tagged index of every documentation file. Updated 2026-07-30.*

**Status tags:**
- **CURRENT** — reflects present state, safe to act on
- **HISTORICAL** — point-in-time checkpoint record; accurate for its date, NOT current state
- **FROZEN** — superseded design-intent record; retained for architectural rationale
- **CORRECTED** — contains a conclusion later overturned; correction noted inline

---

## Canonical Context

| Path | Purpose | Status |
|------|---------|--------|
| `CLAUDE.md` | Primary CLI context: architecture, status, decisions, constraints | **CURRENT** |
| `docs/CATALOG.md` | This file — status-tagged index of all docs | **CURRENT** |
| `docs/decisions-archive.md` | Full historical decisions log + performance baselines + bug fixes | **CURRENT** |
| `docs/project_instructions_proposed.md` | claude.ai web-session Project Instructions (paste target) | **CURRENT** |

## Guides & Runbooks

| Path | Purpose | Status |
|------|---------|--------|
| `docs/guides/calibration_guide.md` | Camera calibration workflow (lens cal + H refinement) | **CURRENT** |
| `docs/guides/runbook_cross_camera_capture.md` | Multi-camera source-PTS capture runbook | **CURRENT** |

## Roadmaps

| Path | Purpose | Status |
|------|---------|--------|
| `docs/roadmap/recorder_productionization.md` | Live plan for recorder productionization (CP-R1 through CP-R10) | **CURRENT** |

## Reference & Audits

| Path | Purpose | Status |
|------|---------|--------|
| `docs/reference/sidecar_contract.md` | Sidecar timing contract -- schema v4 (CP-R6). Authoritative spec for `.timing.jsonl`. | **CURRENT** |
| `docs/reference/eval_instrument_spec.md` | CP-EVAL-1 frozen instrument v1.0 spec | **CURRENT** |
| `docs/reference/undistortion_audit.md` | All 9 undistortion code paths verified (2026-04-02) | **CURRENT** |
| `docs/reference/stage_d_audit_findings.md` | CP0: Stage D config audit (7/8 penalty fields dead) | **HISTORICAL** |
| `docs/reference/pipeline_validation_discovery.md` | Pipeline validation framework discovery notes | **HISTORICAL** |

## Evidence — Recorder Timing Arc

| Path | Purpose | Status |
|------|---------|--------|
| `docs/evidence/capture_time_1/findings.md` | Source PTS discovery on J_EDEw (30fps session) | **CORRECTED** — "30fps" generalized; corrected by CAPTURE-TIME-2 (fps varies) |
| `docs/evidence/capture_time_2/findings.md` | RTCP absence, cross-camera drift, fps varies | **CURRENT** |
| `docs/evidence/recorder_timing_1/findings.md` | CFR+sidecar timing preservation analysis | **CURRENT** |
| `docs/evidence/recorder_reliability_1/findings.md` | Five production recorder fixes + dup/drop verdict | **CORRECTED** — mpdecimate "255 dups" was near-identical (default thresholds), not pixel-identical. Pixel-identical on arrival-PTS is 34 (0.75%). See DUPFIX-1. |
| `docs/evidence/recorder_reliability_2/findings.md` | API quota awareness + traffic reduction | **CURRENT** |
| `docs/evidence/recorder_dupfix_1/findings.md` | DUPFIX-1/2: Duplicate contradiction resolved (0 pixel-identical dups); frame drops measured (FP7oJQ 0.1-7.7% PTS gaps, PPDmUg 0-3.0% CFR decimation); conservation test, boundary attribution, PTS gap analysis | **CURRENT** |
| `docs/evidence/recorder_fps_adaptation_1/findings.md` | CP-R1b: Bimodal frame-rate oscillation (interleaved 33/67ms inter-frame intervals). TRIM-BIMODAL defect: trimmed mean discards minority mode (36% discard on transition segment). Genuine rate change vs relay frame loss is structurally undecidable from PTS. Per-frame dt is the only reliable rate source. | **CURRENT** |
| `docs/evidence/recorder_boundary_fix_1/findings.md` | CP-R5: PTS-based segment boundary split replaces line-position split. PPDmUg seg1 residual -135 -> -109 (correct), +30 -> +0 (exact). FP7oJQ leading edge 47 frames recovered (3.1s of dropped data). Schema bumped to 3. | **CURRENT** |
| `docs/evidence/wallclock_1/findings.md` | Container PTS is synthetic (CFR discards wall-clock) | **HISTORICAL** |

## Evidence — Identity Investigation Arc

| Path | Purpose | Status |
|------|---------|--------|
| `docs/evidence/cp_gt2actuals_1/recon_findings.md` | Dense GT reconciliation (CP-GT2ACTUALS-1) | **HISTORICAL** |
| `docs/evidence/cp_gt2actuals_3/validation_findings.md` | Split-family lookup fix (CP-3) | **HISTORICAL** |
| `docs/evidence/cp_gt2actuals_3_5/findings.md` | Signal_trace same-bug analysis (CP-3.5) | **HISTORICAL** |
| `docs/evidence/cp_gt2actuals_4_5/findings.md` | D0.5 net-effect per-event (CP-4+5) | **HISTORICAL** |
| `docs/evidence/cp_gt2actuals_5_5/findings.md` | D0.5 cross-camera thin-classification (CP-5.5) | **HISTORICAL** |
| `docs/evidence/cp_gt2actuals_6/findings.md` | Stage attribution: A drift 41%, group 33%, solver 26% (CP-6) | **HISTORICAL** |

## Evidence — Purity & Appearance Arc

| Path | Purpose | Status |
|------|---------|--------|
| `docs/evidence/cp_purity_1/decomposition_report.md` | Through-line purity decomposition | **HISTORICAL** |
| `docs/evidence/cp_purity_2/decomposition_report.md` | Aggregate reconciliation + ceiling | **HISTORICAL** |
| `docs/evidence/cp_purity_3/oracle_report.md` | GT-through-D oracle (0 GROUP nodes, 0 logic gaps) | **HISTORICAL** |
| `docs/evidence/cp_raster_plate/plate_report.md` | Median-background masking (V-blind, NO_GO invalid) | **HISTORICAL** |
| `docs/evidence/cp_raster_plate_2/plate_report.md` | V-channel separability (H+S+V AUC 0.907, GO) | **CURRENT** |
| `docs/evidence/cp_split_validate/split_validate_report.md` | GT-validate all D0.5 splits (precision crisis) | **HISTORICAL** |
| `docs/evidence/purity_proxy_1/verdict.md` | max_displacement + kinematic purity proxies | **HISTORICAL** |
| `docs/evidence/purity_proxy_2/verdict.md` | Masked-appearance purity proxy analysis | **HISTORICAL** |

## Evidence — Tag Identity Arc

| Path | Purpose | Status |
|------|---------|--------|
| `docs/evidence/cp_tag_3_baseline/README.md` | CP-TAG-3 overview: two-clip harness + baseline | **HISTORICAL** |
| `docs/evidence/cp_tag_3_baseline/carrier_evidence.md` | Carrier tracklet analysis for tagged person | **HISTORICAL** |
| `docs/evidence/cp_tag_3_baseline/session_evidence.md` | Session-level tag identity trace | **HISTORICAL** |
| `docs/evidence/cp_tag_3_baseline/provenance.md` | Data provenance for CP-TAG-3 | **HISTORICAL** |
| `docs/evidence/cp_tag_3_baseline/vid1_tag_trace.md` | Vid1 tag signal trace detail | **HISTORICAL** |
| `docs/evidence/cp_tag_3_baseline/vid2_tag_trace/_tagged_person_report.md` | Vid2 tagged person trace | **HISTORICAL** |
| `docs/evidence/cp_tag_4_post/README.md` | CP-TAG-4a post-fix verification | **HISTORICAL** |
| `docs/evidence/cp_tag_4a_verify/findings.md` | CP-TAG-4a code verification | **HISTORICAL** |

## Evidence — Other

| Path | Purpose | Status |
|------|---------|--------|
| `docs/evidence/storage_audit_1/proposal.md` | Storage cleanup proposal (eval baselines) | **HISTORICAL** |

## Checkpoint Records

All checkpoint records are **HISTORICAL** — accurate for their date, not current state.

| Path | Purpose |
|------|---------|
| `docs/checkpoints/cp1_evidence.md` | CP1: Cost inversion evidence (penalty 15 < BIRTH+DEATH 20) |
| `docs/checkpoints/cp2_results.md` | CP2: Penalty 15→25 results |
| `docs/checkpoints/cp2.5_diagnostics.md` | CP2.5: Length-agnostic penalty diagnosis |
| `docs/checkpoints/cp3_results.md` | CP3: Pure per-frame penalty (REGRESSION, rolled back) |
| `docs/checkpoints/cp3b_results.md` | CP3b: Floor-protected length-proportional penalty |
| `docs/checkpoints/cp4_flow_topology.md` | CP4: Parallel-carrier displacement root cause |
| `docs/checkpoints/cp4_5_validation_inventory.md` | CP4.5: Validation artifact inventory |
| `docs/checkpoints/cp5_results.md` | CP5: Parallel-carrier consolidation results |
| `docs/checkpoints/cp6_gt_trace_baseline.md` | CP6: GT person trace baseline + carrier competition |
| `docs/checkpoints/cp7_pre_reid_inventory.md` | CP7: Pre-ReID feature inventory |
| `docs/checkpoints/cp7_pre2_misattribution_cause.md` | CP7-pre-2: Misattribution cause analysis |
| `docs/checkpoints/cp7_pre3_impurity_decomposition.md` | CP7-pre-3: Impurity decomposition (70-78% under-segmentation) |
| `docs/checkpoints/cp7_pre4_detection_mechanism.md` | CP7-pre-4: Detection mechanism investigation |
| `docs/checkpoints/cp7_pre5_gt_ceiling_run.md` | CP7-pre-5: GT ceiling run |
| `docs/checkpoints/cp7_pre5_gt_overlap_baseline.md` | CP7-pre-5: GT overlap baseline |
| `docs/checkpoints/cp7_pre6_nms_sweep.md` | CP7-pre-6: NMS IoU sweep (relaxation worsens misattrib) |
| `docs/checkpoints/cp7_pre7_failure_topology.md` | CP7-pre-7: Failure topology census |
| `docs/checkpoints/cp7_pre8_axis1_signature.md` | CP7-pre-8: Axis-1 signature (SUPERSEDED by pre-9/10) |
| `docs/checkpoints/cp7_pre9_branchb_margin.md` | CP7-pre-9: True Branch-B margin = 9.9% |
| `docs/checkpoints/cp7_pre10_pairbox_bracketing.md` | CP7-pre-10: Pair-box 0% bracketed at all horizons |
| `docs/checkpoints/tb_eval_1_5/verification.md` | TB-EVAL-1.5: Pipeline verification logs |

## Tooling Diagnostics

| Path | Purpose | Status |
|------|---------|--------|
| `tools/sweep/diagnostics/blast_radius_check.md` | SWEEP-3b: CP-TAG-4a retraction evidence | **CURRENT** |
| `tools/sweep/diagnostics/gap_explanation.md` | Sweep vs eval_gt baseline gap (2pp environment artifact) | **CURRENT** |
| `tools/sweep/diagnostics/ofat_track_buffer_results.md` | SWEEP-4: OFAT track_buffer screen (stock optimal) | **CURRENT** |
| `tools/sweep/diagnostics/step1_results.md` | SWEEP-1: Initial sweep step results | **HISTORICAL** |

## Frozen Design Intent

All files in `docs/archive/planning/` are **FROZEN** — January 2026 worker-thread
specifications. Superseded as SPECIFICATION by CLAUDE.md and `.claude/rules/*`. Retained
as a DESIGN-INTENT RECORD (e.g. A1/A2 establish the "short, high-precision tracklets,
intentionally allowed to break" intent).

| Path | Purpose |
|------|---------|
| `docs/archive/planning/README.md` | Planning pack overview + Stage D POC map |
| `docs/archive/planning/index.md` | Planning section index |
| `docs/archive/planning/WORKER_THREAD_INDEX.md` | Worker thread master index by stage |
| `docs/archive/planning/worker_threads/index.md` | Worker threads directory index |
| `docs/archive/planning/worker_threads/A1_*.md` | Stage A: Detection + BoT-SORT tracking |
| `docs/archive/planning/worker_threads/A2_*.md` | Stage A: Tracklet quality gating signals |
| `docs/archive/planning/worker_threads/B1_*.md` | Stage B: Mask refinement (SAM, deferred) |
| `docs/archive/planning/worker_threads/B2_*.md` | Stage B: Contact point extraction |
| `docs/archive/planning/worker_threads/B3_*.md` | Stage B: Camera calibration + drift monitor |
| `docs/archive/planning/worker_threads/C0_*.md` | Stage C: Tag decode scheduling + cadence |
| `docs/archive/planning/worker_threads/C1_*.md` | Stage C: AprilTag scanning pipeline |
| `docs/archive/planning/worker_threads/C2_*.md` | Stage C: Identity registry + voting |
| `docs/archive/planning/worker_threads/D0_*.md` | Stage D: Offline cleanup + bank curation |
| `docs/archive/planning/worker_threads/D1_*.md` | Stage D: MCF graph model |
| `docs/archive/planning/worker_threads/D2_*.md` | Stage D: MCF cost function + constraints |
| `docs/archive/planning/worker_threads/D3_*.md` | Stage D: MCF solver implementation |
| `docs/archive/planning/worker_threads/D4_*.md` | Stage D: ReID embeddings (optional) |
| `docs/archive/planning/worker_threads/D5_*.md` | Stage D: Birth/death + mat zone gating |
| `docs/archive/planning/worker_threads/D6_*.md` | Stage D: Global ILP optimizer |
| `docs/archive/planning/worker_threads/D7_*.md` | Stage D: Homography calibrator tool |
| `docs/archive/planning/worker_threads/E1_*.md` | Stage E: Match session state machine |
| `docs/archive/planning/worker_threads/E2_*.md` | Stage E: Multi-mat spatial partitioning |
| `docs/archive/planning/worker_threads/F0_*.md` | Stage F: Core contracts + artifact schemas |
| `docs/archive/planning/worker_threads/F1_*.md` | Stage F: Pipeline orchestration CLI |
| `docs/archive/planning/worker_threads/F2_*.md` | Stage F: Config system + environment |
| `docs/archive/planning/worker_threads/F3_*.md` | Stage F: Ingest + Docker + Nest integration |
| `docs/archive/planning/worker_threads/X1_*.md` | Clip export (ffmpeg crop + smoothing) |
| `docs/archive/planning/worker_threads/X2_*.md` | Opt-in privacy redaction |
| `docs/archive/planning/worker_threads/X3_*.md` | Database schema + persistence |
| `docs/archive/planning/worker_threads/Z1_*.md` | Observability + debug artifacts |
| `docs/archive/planning/worker_threads/Z2_*.md` | End-to-end POC test harness |
| `docs/archive/planning/worker_threads/Z3_*.md` | Single-pass multiplexer runner |

## Historical Analysis Tools

These Python tools wrote checkpoint reports to `docs/checkpoints/`. All are **HISTORICAL**.

| Path | Output |
|------|--------|
| `tools/cp7_pre7_failure_topology.py` | `docs/checkpoints/cp7_pre7_failure_topology.md` |
| `tools/cp7_pre8_axis1_diagnostic.py` | `docs/checkpoints/cp7_pre8_axis1_signature.md` |
| `tools/cp7_pre9_branchb_margin.py` | `docs/checkpoints/cp7_pre9_branchb_margin.md` |
| `tools/cp7_pre10_pairbox_bracketing.py` | `docs/checkpoints/cp7_pre10_pairbox_bracketing.md` |
