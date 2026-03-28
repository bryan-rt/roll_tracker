# CLAUDE.md — Roll Tracker

## Project

BJJ gym SaaS pipeline. Nest cameras → YOLO+BoT-SORT tracking → AprilTag identity →
ILP stitching → per-athlete match clips → Supabase → Flutter app.
**Repo:** github.com/bryan-rt/roll_tracker | **Branch:** `services_uploader` | **Python 3.12**

## Working Methodology

**Three-pass protocol (mandatory for all non-trivial tasks):**
1. **Pass 1 — Explore** (Plan Mode: shift+tab ×2): Read Task Brief, explore relevant files,
   identify conflicts. ⏸ STOP — summarize and wait for approval.
2. **Pass 2 — Specify** (Plan Mode continues): Plan exact changes, verify naming/contracts
   against live code. ⏸ STOP — present plan and wait for approval.
3. **Pass 3 — Execute**: Implement, test, update CLAUDE.md if architecture changed,
   commit+push. ⏸ STOP — summarize and wait for review.

**Evidence-driven design:** Do not code from assumptions. When behavior is uncertain:
enhance logging → inspect real output → plan from evidence. Propose instrumentation
before fixes when root cause is unclear.

**Ambiguity protocol:** Surface naming conflicts, missing files, or uncovered architectural
questions in Pass 1. Do not resolve silently or guess.

## Monorepo Layout

```
src/bjj_pipeline/        # CV pipeline package (stages A→F, contracts, config, core)
src/calibration_pipeline/ # Gym setup tools (lens cal, mat walk stubs)
services/                 # Docker: nest_recorder, processor, uploader
backend/supabase/         # Migrations, config.toml
app_mobile/               # Flutter athlete app
app_web/                  # Vite+React gym owner app
configs/                  # default.yaml, per-camera overrides, homography.json
```

## Critical Constraints

- **NumPy < 2** — Torch ABI. Install ultralytics/boxmot with `--no-deps`.
- **Supabase is the exclusive integration hub** — no direct service-to-service communication.
- **Phase 1/2 parallelism boundary (NON-NEGOTIABLE)** — A+C parallel, D+E+F sequential.
- **No cross-stage imports** — stages communicate only via F0 contracts + filesystem.
- **Option B undistort-on-projection** — `project_to_world()` is the only permitted
  pixel→world path. No stage calls homography directly.

## Coding Conventions

- Stage contract: `run(config: dict, inputs: dict) -> dict`
- Pydantic v2 for data models. Loguru for logging. Rich for CLI. Typer for CLI defs.
- Parquet for tabular data. JSONL for audit/event streams. Type hints everywhere.
- Debug artifacts → `outputs/<clip_id>/_debug/`. Never pollute stage output dirs.
- Paths via `ClipOutputLayout` and env vars — no hardcoding.

## Config Resolution

`default.yaml` → `cameras/<cam_id>.yaml` → `cameras/<cam_id>/homography.json` → `--config` CLI overlay

## Current Status

- **Head:** `0f5edd5` | Pipeline A→F verified E2E. Session pipeline validated (3-camera).
- **CP17 Tier 1 implemented:** Two-pass cross-camera ILP with tag corroboration.
- **Open issue:** PPDmUg-202751 — NAType in frame_index at D2. Needs null-safe fix.
- **Next:** CP17 Tier 2/3 (coordinate evidence) stubbed for CP18.
- **Apps:** Flutter tested on Pixel 7 Pro. Web app has mat editor + admin pricing.
- **Supabase:** 23 migrations applied locally and remotely.

## Domain Context (auto-loaded by path)

See `.claude/rules/` — each file has `paths:` frontmatter scoping it to relevant directories.
Full historical decisions archive: `docs/decisions-archive.md` (not auto-loaded).

## Never Touch

- `data/` `outputs/` `services/nest_recorder/secrets/` `.env` files
- Applied migration SQL files in `backend/supabase/supabase/migrations/`
