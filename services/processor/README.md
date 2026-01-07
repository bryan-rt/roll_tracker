# Processor Service (scaffold)

Purpose: Offline processing service that will orchestrate the `bjj_pipeline` over recorded sessions. This scaffold documents inputs/outputs and environment-driven paths. No implementation code is present yet.

Non-negotiables:
- Do not alter Nest ingestion behavior.
- Do not add cloud/DB logic here.
- Do not redesign the Python pipeline.
- Only mounts/paths via environment variables (see repository `.env.example`).

Inputs (expected):
- Session directory produced by `services/nest_recorder` (TBD finalized schema).
- Configs from `configs/` and optional camera configs from `configs/cameras/`.
- Environment variables for base directories:
  - `ROLL_TRACKER_CONFIGS_DIR`, `ROLL_TRACKER_CAMERA_CONFIGS_DIR`, `ROLL_TRACKER_DATA_DIR`, `ROLL_TRACKER_OUTPUTS_DIR`.

Outputs (expected):
- Derived artifacts written to `outputs/<session-id>/` (e.g., detections, masks, tracklets, clips).
- Logs/telemetry stored under `outputs/<session-id>/logs/` (naming TBD).

Dev/Run notes:
- No Dockerfile or compose included yet for this service.
- In dev, a parent compose may run containers with `sleep infinity` and use `docker exec` to invoke the CLI.

Status: Scaffold only. This folder contains documentation and contracts; implementation will be added later without changing the overall pipeline design.
