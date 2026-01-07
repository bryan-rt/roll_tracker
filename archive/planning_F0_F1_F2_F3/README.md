# roll_tracker planning pack (expanded)

Each file in `planning/worker_threads/` is meant to be pasted as the **first message** in a dedicated worker chat.

**Workflow**
1) Create a new worker chat whose title starts with the worker ID (e.g., `D2`).
2) Paste the entire contents of the corresponding worker markdown file.
3) Ask the worker to deliver the required outputs back to the Manager thread.
4) Paste the worker’s summary back into the Manager thread to “lock” decisions.

**Important**
- Min-Cost Flow stitching is mandatory.
- All stages communicate only through versioned artifacts defined in F0.

## Current locks
- **F3** ingest contract: clips under `data/raw/nest/...`
- **F0** contracts: stage artifacts + manifest anchored at `outputs/<clip_id>/clip_manifest.json`

Worker docs include an update section summarizing these locks. Always align to them.
