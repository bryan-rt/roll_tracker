# Z3 — Single-pass Multiplexer Runner + Dev Visualization

### Purpose
Refactor the runtime architecture so **Stages A→B→C** can be executed in a **single pass over frames** (shared `FrameIterator`) while still:
- Writing **separate, F0-locked artifacts** per stage (`stage_A/*`, `stage_B/*`, `stage_C/*`)
- Keeping the **F1 stage `run(config, inputs)->dict` contract**
- Preserving **resume/skip semantics** (F1) and **config hashing + audits** (F2)

Add a **developer visualization switch** that outputs:
1) an annotated video (bbox + masks + track_id + conf)  
2) a “mat view” video (mat blueprint plotted + per-track XY points + labels)

This worker owns the architecture + implementation plan (and optionally the code if time allows), and must not change F0 schemas without a manager-approved version bump.

### Why we want this
Running A then re-reading the video for B then re-reading for C is correct but expensive and slows iteration. A single-pass “multiplexer”:
- Speeds up iteration (especially for mask generation + tag scanning)
- Allows smarter cadence decisions (e.g., run SAM only when needed)
- Keeps artifacts modular and stage-owned (still write stage outputs exactly as F0 defines)

### Non-goals / constraints (hard)
- **Do not** collapse A/B/C outputs into a single mega-artifact.
- **Do not** add new columns to F0 artifacts ad-hoc (schema drift).
- **Do not** move AprilTag decoding, ReID, or homography responsibilities across stages unless the worker updates the relevant stage docs + keeps F0 outputs intact.
- **Do not** require a GPU-only path; multiplexer should degrade gracefully.

### Current contracts to respect (must align)
- **F0:** artifact families & manifest anchoring under `outputs/<clip_id>/...`
- **F1:** orchestration invokes stage `run(config, inputs)->dict` and registers artifacts in `clip_manifest.json`
- **F2:** config merge + canonical camera_id + caches under `outputs/<clip_id>/_cache`
- **B3/D6:** homography preflight and calibration path: `configs/cameras/<camera_id>/homography.json`

---

## Proposed design

### 1) Multiplexer as an orchestration-internal execution mode
Add an optional orchestration mode, e.g.
- `roll-tracker run ... --mode multipass` (default / current)
- `roll-tracker run ... --mode multiplex_ABC`

The multiplexer runs inside orchestration, not as a standalone stage, to avoid forcing F0 schema changes.

**Key idea:** the orchestrator still “runs stages” A/B/C, but the “run” implementations for these stages can optionally accept a `FrameIterator` handle in `inputs` (an object, not serialized) when orchestrator is in multiplexer mode.

- In normal mode: A/B/C stages open the video themselves.
- In multiplexer mode: orchestration opens the video once, constructs `FrameIterator`, passes it to a “multiplex runner” which delegates per-frame work to A/B/C processors.

### 2) Separate “processor” vs “stage writer” roles
To keep artifacts clean and avoid tight coupling, each stage gets two layers:
- **Processor**: per-frame logic, stateless-ish, returns records/events.
- **Writer**: buffered aggregation that emits the stage artifacts exactly as F0 defines at end-of-clip.

This can be implemented as:
- `StageAProcessor.on_frame(frame)` → yields detections/tracklets rows
- `StageBProcessor.on_frame(frame, stageA_state)` → yields masks/contact_points
- `StageCProcessor.on_frame(frame, stageA_state, stageB_state)` → yields tag observations / identity hints

Writers manage:
- Parquet row buffers / chunked writes
- mask `.npz` files
- audit.jsonl per stage

### 3) Cadence controls (optional but planned)
The multiplexer enables safe cadence without breaking tracking:
- detection/tracking can still run every frame
- heavy mask refinement (SAM) can run conditionally (see B1)
- ReID embedding extraction can be event-driven (e.g., stable track windows)
- AprilTag scanning can be gated by “tag ROI quality” events

Cadence must be explicit in config and audited (F2/F1), so reruns are comparable.

---

## Developer visualization (“--visualize”) requirements
When enabled (dev-only), produce:
1) `outputs/<clip_id>/stage_A/_debug/annotated.mp4`  
   - overlays: bbox, track_id, conf, and whichever masks are available (YOLO mask always; SAM mask only if computed)
2) `outputs/<clip_id>/stage_B/_debug/mat_view.mp4` (or `stage_C/_debug` if tags drive it)
   - plots: mat blueprint in world coords + per-tid (x,y) points + labels

Implementation notes:
- Use OpenCV for drawing, and either OpenCV VideoWriter or ffmpeg wrapper (prefer OpenCV for rapid iteration).
- Debug outputs must be **clearly labeled as non-canonical** and excluded from “required artifacts” for stage completion.

---

## Deliverables
1) Architecture decision record in this doc: what runs where and why.
2) Concrete code touchpoints:
   - orchestration changes (mode flag, iterator creation, multiplex runner)
   - stage changes (processor/writer split, optional iterator support)
   - debug viz module (reusable helper)
3) Tests:
   - multiplexer vs multipass parity on a small fixture (schemas and counts match)
   - visualize flag produces debug outputs without affecting canonical artifacts
4) Worker-to-manager kickoff: start by proposing an API sketch + minimal integration plan, then ask manager to approve.

---

## Kickoff prompt for the worker thread
Start by replying with:
1) A proposed folder/module layout for the multiplexer runner (files + functions).
2) A minimal incremental plan (PR1: iterator + StageA integration; PR2: add B; PR3: add C; PR4: viz).
3) A list of risks and how you’ll ensure F0 artifacts don’t drift.
Then ask me to confirm the proposed boundaries before you write code.

### Status
- COMPLETE

### Implemented Goals (Slice 2)
- **Single-pass multiplexer execution (A/B/C):** Orchestration supports `--mode multiplex_ABC`; video is decoded once via a shared `FrameIterator`; stages A/B/C run within the multiplexer window and emit separate, F0-locked artifacts under `stage_A/`, `stage_B/`, `stage_C/`. Multipass remains the default and is preserved for offline stages and parity validation.
- **Strict separation of orchestration vs stage logic:** Z3 owns wiring (iterator lifetime, run-plan decisions, state access, artifact registration). No detection/masking/tag-decoding logic was changed; A/B/C workers retain ownership of compute semantics.
- **State provider abstraction:** Supports windows like “A skipped, B runs” and “A+B skipped, C runs”. State may come from existing on-disk artifacts (multipass parity) or in-memory results produced earlier in the same multiplex run. This layer is architectural only (no stage-specific logic).
- **Developer visualization:** Enabled via `--visualize` (dev-only, non-canonical). Writes `outputs/<clip_id>/_debug/annotated.mp4` and `outputs/<clip_id>/_debug/mat_view.mp4`. Visualization runs per frame after A/B/C, consuming available outputs. `mat_view.mp4` renders `configs/mat_blueprint.json` and is confirmed working. Visualization code imports only when `--visualize` is enabled.
- **cv2 dependency behavior:** If `--visualize` is false, multiplex runs without importing OpenCV. If `--visualize` is true and `cv2` is missing, the pipeline fails fast with a clear error. No best-effort/silent degradation paths.
- **Force-stage semantics:** `--force-stage A --force-stage B --force-stage C` limits execution strictly to the A..C window. Multiplex no longer spills into Stage D/E/F during POC runs.
- **CLI + docs alignment:** CLI help reflects actual debug output locations. Z3 design decisions and constraints are locked into this document.

### Locked Decisions (Must Respect)
- **Multiplex vs Multipass:** Multiplex is the execution model for online stages (A/B/C). Multipass remains supported and is expected to continue for offline stages (D/E/F) until explicitly migrated.
- **Stage ownership boundaries:** Z3 must not implement/refactor A/B/C compute logic. A/B/C workers own processors and writers; Z3 only wires them together.
- **Artifact contracts:** F0 schemas remain unchanged. Each stage continues to emit its own canonical artifacts. Debug videos are explicitly non-canonical.
- **Debug visualization location:** All dev visualization output lives under `outputs/<clip_id>/_debug/`. No stage-specific `_debug/` folders are used for mux output.
- **Blueprint handling:** `configs/mat_blueprint.json` is the single source of truth. Blueprint rendering is visualization-only and does not affect canonical outputs.

### File & Function Touchpoints (Z3)
- **Orchestration:**
   - `stages/orchestration/pipeline.py`: mode selection (multipass vs `multiplex_ABC`), force-stage window handling
   - `stages/orchestration/multiplex_runner.py`: single-pass frame loop, state provider integration, conditional visualization wiring
- **Core:**
   - `core/frame_iterator.py`: shared frame decode for multiplex
   - `core/state_provider.py`: disk-backed + in-memory state access abstraction
- **Visualization (dev-only):**
   - `viz/mux_visualizer.py`
   - `viz/mat_view.py`
   - `viz/video_writer.py`

### Explicit Non-Goals
- Implement detection, tracking, masking, ReID, or AprilTag logic
- Modify F0 schemas or artifact layouts
- Remove multipass execution
- Add GPU-only or performance-specific optimizations

### Handoff Notes for Project Manager
- Z3 Slice 2 is complete and production-safe.
- Multiplex execution is available for A/B/C and validated via real outputs.
- Future work:
   - Migrate Stage A/B/C compute into processor/writer form
   - Extend multiplex to offline stages only if justified
   - Parity testing between multipass and multiplex outputs

### Kickoff Prompt for Next Worker (Post-Z3)
- **Assumptions:**
   - A single-pass frame iterator exists and is stable.
   - Frames are received via the multiplex runner when enabled.
   - Your stage must emit F0 artifacts exactly as before.
- **Focus:**
   - Concentrate on stage logic — orchestration is handled.