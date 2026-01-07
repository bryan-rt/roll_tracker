# Uploader Service (scaffold)

Purpose: Prepares local batch bundles for an external upload step. This scaffold defines the bundle shape and local packaging only; it does not implement cloud or database logic.

Non-negotiables:
- Do not add cloud/DB logic.
- Do not alter Nest ingestion.
- Only folder structure and documentation are provided.

Inputs (expected):
- Processed artifacts from `outputs/<session-id>/` (e.g., manifests, clips, detections).
- Optional redaction maps and anonymization metadata.

Outputs (expected):
- A local `batch_bundle/` directory containing an immutable bundle (manifest + payload) ready for transfer by external tooling.

Dev/Run notes:
- No Dockerfile or compose included yet for this service.
- Bundles are strictly local artifacts; any upload mechanics live outside this repo.

Status: Scaffold only. No implementation code has been added.
