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

## Update after Z3 completion (2026-01-07)
Z3 introduced an **optional single-pass multiplex mode** (`multiplex_ABC`) that runs **Stages A→B→C within a shared frame loop** (video decoded once), while preserving:
- **F0 artifact contracts + paths** (each stage still writes its own canonical artifacts)
- **F1 stage contract** (`run(config, inputs) -> dict`) and skip/resume semantics
- **F2 config hashing + orchestration audit discipline**

### What this means for this worker
- Your stage code **must support both**:
  - **Multipass** execution (stage reads inputs from disk artifacts / manifest)
  - **Multiplex** execution (stage receives needed per-frame / per-clip state via the orchestration-provided state provider)
- Do **not** write debug videos from within stages. Dev visualization output is **owned by orchestration** and lives under:
  - `outputs/<clip_id>/_debug/` (non-canonical, dev-only)
- Prefer **pure functions / explicit state**:
  - per-frame update entrypoints should be deterministic and side-effect controlled
  - any expensive optional computations should be **gated** and recorded in stage audit (what ran, why)

### Interface expectations (keep flexible, but follow intent)
- If you introduce a per-frame API (recommended for A/B/C), keep it behind the stage module so orchestration can call it in multiplex mode.
- Ensure stage outputs can still be produced in multipass mode by reading upstream artifacts from disk (parity requirement until multipass is retired).
