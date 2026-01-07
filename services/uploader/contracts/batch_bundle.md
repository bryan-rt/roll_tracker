# Uploader Service Contracts — Batch Bundle (draft)

Version: 0.2-draft
Status: Scaffold only

## Filesystem Contracts
- Input runs:
  - `outputs/runs/<run_id>/` directories produced by the Processor service.
  - Must contain `run_manifest.json` and expected stage artifacts.
- Output (persistence):
  - Remote persistence (outside this repo) — service is responsible for preparing data for upload.
  - Maintain local upload state/idempotency markers (e.g., `.uploaded`, `.retry`, checksum indices) adjacent to `outputs/runs/<run_id>/` or under a designated state folder.

## Bundle Guidance (optional)
- `manifest.json` fields:
  - `bundle_id` (UUID), `created_at` (ISO-8601), `run_id`, `items` (relative paths + checksums), `source` (`outputs/runs/<run_id>/`).
- `payload/` directory contains selected artifacts per policy (clips, detections, stitched results, logs).

## Notes
- Packaging/upload logic is out of scope for this scaffold.
- No cloud/DB code in this repo; external systems perform transfer.
- Idempotency is critical; ensure re-runs do not duplicate uploads.
