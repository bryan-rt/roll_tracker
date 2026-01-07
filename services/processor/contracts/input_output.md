# Processor Service Contracts — Input/Output (draft)

Version: 0.2-draft
Status: Scaffold only

## Filesystem Contracts
- Input clips:
  - `data/raw/nest/...` — recorded MP4 clips organized by camera and time.
  - `data/samples/...` — sample MP4s for local testing.
  - Consumers must supply explicit file paths (single clip or lists).
- Output run directory:
  - `outputs/runs/<run_id>/`
    - `run_manifest.json` — summarizes inputs, parameters, and produced artifacts.
    - Stage artifacts (detections, masks, tracklets, stitched graphs, exports) under subfolders.

## Inputs (context)
- `configs_dir`: base configs (`configs/`), optional `camera_configs_dir` (`configs/cameras/`).
- Environment variables (for path resolution only):
  - `ROLL_TRACKER_CONFIGS_DIR`, `ROLL_TRACKER_CAMERA_CONFIGS_DIR`, `ROLL_TRACKER_DATA_DIR`, `ROLL_TRACKER_OUTPUTS_DIR`.

## Notes
- No implementation included; this is documentation-only.
- No pipeline redesign; contracts mirror current offline pipeline expectations.
- No cloud/DB logic here; upload/bundling handled by separate service.
