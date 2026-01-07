# Worker Thread Index
Use these files as the starter message for each worker chat.

## Threads
- **F0** — Core Contracts & Artifact Schemas → `planning/worker_threads/F0_Core_Contracts_Artifact_Schemas.md`
- **F1** — Pipeline Orchestration & CLI → `planning/worker_threads/F1_Pipeline_Orchestration_CLI.md`
- **F2** — Config System & Environment → `planning/worker_threads/F2_Config_System_Environment.md`
- **F3** — Ingest & Docker Baseline (Nest clipper integration) → `planning/worker_threads/F3_Ingest_Docker_Baseline_Nest_clipper_integration.md`
- **A1** — Detection & Tracking (BoT-SORT Tracklets) → `planning/worker_threads/A1_Detection_Tracking_BoT_SORT_Tracklets.md`
- **A2** — Tracklet Quality & Gating Signals → `planning/worker_threads/A2_Tracklet_Quality_Gating_Signals.md`
- **B1** — Mask Refinement Strategy (SAM vs fallback) → `planning/worker_threads/B1_Mask_Refinement_Strategy_SAM_vs_fallback.md`
- **B2** — Contact Point Extraction + Homography → `planning/worker_threads/B2_Contact_Point_Extraction_Homography.md`
- **B3 — Camera Calibration: Homography Preflight + Drift Monitor** → `planning/worker_threads/B3_Camera_Calibration_Homography_Preflight_Drift_Monitor.md`
- **C1** — AprilTag Scanning Pipeline (mask-guided) → `planning/worker_threads/C1_AprilTag_Scanning_Pipeline_mask_guided.md`
- **C2** — Identity Registry (Voting + Conflicts) → `planning/worker_threads/C2_Identity_Registry_Voting_Conflicts.md`
- **D1** — MCF Graph Model (Nodes/Edges) → `planning/worker_threads/D1_MCF_Graph_Model_Nodes_Edges.md`
- **D2** — MCF Cost Function + Constraints → `planning/worker_threads/D2_MCF_Cost_Function_Constraints.md`
- **D3** — MCF Solver Implementation Plan → `planning/worker_threads/D3_MCF_Solver_Implementation_Plan.md`
- **D4** — ReID Embeddings (masked crops) (optional) → `planning/worker_threads/D4_ReID_Embeddings_masked_crops_optional.md`
- **D5 — MCF Birth/Death + Mat-Zone Gating** → `planning/worker_threads/D5_MCF_Birth_Death_MatZone_Gating.md`
- **D6 — Global ILP Optimizer (Post-MCF)** → `planning/worker_threads/D6_Global_ILP_Optimizer_Post_MCF.md`
- **E1** — Match Session State Machine → `planning/worker_threads/E1_Match_Session_State_Machine.md`
- **E2** — Gym Multi-Mat / Spatial Partitioning (optional) → `planning/worker_threads/E2_Gym_Multi_Mat_Spatial_Partitioning_optional.md`
- **X1** — Clip Export (ffmpeg) + Crop Smoothing → `planning/worker_threads/X1_Clip_Export_ffmpeg_Crop_Smoothing.md`
- **X2** — Opt-In Privacy Redaction → `planning/worker_threads/X2_Opt_In_Privacy_Redaction.md`
- **X3** — Database Schema + Persistence → `planning/worker_threads/X3_Database_Schema_Persistence.md`
- **Z1** — Observability & Debug Artifacts → `planning/worker_threads/Z1_Observability_Debug_Artifacts.md`
- **Z2** — End-to-End POC Test Harness → `planning/worker_threads/Z2_End_to_End_POC_Test_Harness.md`

## Checkpoint discipline

**All workers must keep the pipeline runnable end-to-end at every checkpoint.**  
This is a non-negotiable POC constraint to prevent drift and ensure incremental validation.

Minimum requirements for *any* stage/worker deliverable:

- Provide a `run()` implementation that is callable by orchestration: `def run(config: dict, inputs: dict) -> dict`.
- Ensure artifacts written match **F0** contracts and pass `roll-tracker validate`.
- Add at least one *realistic* smoke test (pytest) that:
  - runs the stage on a small fixture (or mocked inputs), and
  - asserts the expected artifacts exist and validate.
- Ensure the manager can run:
  - `roll-tracker run --clip <path> --camera <camera_id> --to-stage <your_stage>`
  - and receive deterministic outputs under `outputs/<clip_id>/...`.

If performance becomes an issue, prefer reducing resolution / selecting fewer candidates over skipping frames at POC time.

## Completion status updates

- F2 (Config System & Environment): ✅ COMPLETE (merge semantics locked; resolved config recorded via orchestration audit; homography JSON auto-merged when present)
- **Z3** — Single-pass Multiplexer Runner + Dev Visualization → `planning/worker_threads/Z3_Single_pass_Multiplexer_Runner_and_Dev_Visualization.md`

## Status snapshot (2026-01-07)

Completed and signed-off:
- **F0** — Core Contracts & Artifact Schemas ✅
- **F1** — Pipeline Orchestration & CLI ✅ (includes homography preflight requirement)
- **F2** — Config System & Environment ✅
- **F3** — Ingest integration & raw data contract ✅
- **Z3** — Single-pass multiplexer runner & dev visualization ✅

Cross-cutting notes:
- **Multiplex mode (`multiplex_ABC`)** is now available for online stages **A/B/C** (single video decode; shared frame loop).
- **Artifacts remain stage-scoped and contract-locked** (F0). Dev visualizations live only under `outputs/<clip_id>/_debug/` and are non-canonical.
