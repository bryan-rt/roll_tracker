# Stage D Audit Findings (CP0)

**Date:** 2026-05-13
**Branch:** `services_uploader`
**Scope:** Read-only investigation of five claims about Stage D config/runtime behavior.
**Method:** Code trace + audit JSONL inspection on `_eval_gt` pipeline outputs.

---

## Summary of Falsified vs Confirmed Claims

All five claims survived investigation. The diagnosis is solid and CP1 can proceed.

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Config layering is standard: `default.yaml` base → camera → CLI overlay | **Confirmed** |
| 2 | Pipeline can dump effective merged config at runtime | **Confirmed with caveat** — code exists but silently fails (see Sidebar A) |
| 3 | `solo_ping_miss_penalty_mult` (and 6 other D3 penalty fields) are dead paths | **Confirmed** — all 7 fields are dead; wiring from config→constraints was never built |
| 4 | `birth_non_entrance_add_cost` and `death_non_exit_add_cost` use code default 8.0 | **Confirmed** — not set anywhere in YAML or camera configs |
| 5 | `unexplained_tracklet_penalty: 15.0` is applied at runtime | **Confirmed** — verified in audit JSONL for all 3 eval clips |

No contradictions with CLAUDE.md were found. No CLAUDE.md update required.

---

## Question 1 — Config Layering Direction

**Answer: Standard layered config. `default.yaml` is always the base.**

### Load order (traced from `src/bjj_pipeline/config/loader.py:137-230`)

```
1. configs/default.yaml                        (required base)
2. configs/cameras/<camera_id>.yaml            (optional; merged via deep_merge)
3. CLI --config overlay (.yaml or .json)       (optional; merged via deep_merge)
4. configs/cameras/<camera_id>/homography.json (optional; merged as camera.homography)
```

Each layer is merged onto the previous via `deep_merge()` (loader.py:13-41): dicts merge
recursively, scalars and lists are replaced by the overlay.

### Concrete trace: `stages.stage_D.d3.unexplained_tracklet_penalty`

| Source | Value | Applies? |
|--------|-------|----------|
| `configs/default.yaml:286` | `15.0` | Yes — loaded as base |
| `configs/cameras/FP7oJQ.yaml` | n/a | No file exists |
| `configs/cameras/J_EDEw.yaml` | n/a | No file exists |
| `configs/cameras/PPDmUg.yaml` | n/a | No file exists |
| CLI overlay | n/a | Not used in eval runs |

**Result:** Effective value = 15.0 from `default.yaml`.

### Back-compat shim: `to_runtime_config()` (loader.py:68-101)

After loading, `to_runtime_config()` copies each `stages.stage_X` block to a top-level
`stage_X` key. This is load-bearing back-compat: many stage implementations use
`config["stage_D"]` rather than `config["stages"]["stage_D"]`. It explains the dual
`_cfg_get` lookup pattern seen across the codebase (e.g., `solver.py:41-43`,
`solver.py:56-59`, `solver.py:64-67` — each tries `stages.stage_D.*` first, falls back
to `stage_D.*`). This is not a bug; it's intentional compatibility infrastructure. Do
not attempt to "clean up" the dual lookups without removing the shim first.

---

## Question 2 — Effective Config Dump

**Answer: Code exists but is silently broken. Partial dump available via D2 audit.**

### Intended mechanism

`pipeline.py:670-683` writes a `config_resolved` event to `orchestration_audit.jsonl`
containing both `resolved_config` (raw merged dict) and `runtime_config` (with back-compat
shims). This would be the ideal single source of truth for "what config did this run use?"

**However, this code silently fails.** See Sidebar A below.

### Working partial dump

`d2_run.py:166-182` writes a `d2_config_resolved` event to `stage_D/audit.jsonl`
containing the `d2_costs` config subset, resolved speed parameters, and fps. This is
the only working config dump in the eval outputs today.

### Cheapest fix location

Fix the `NameError` in `pipeline.py:681` (replace `mode` with a valid value or remove it).
The `config_resolved` event will then start appearing in `orchestration_audit.jsonl`.
See Sidebar A for details.

---

## Question 3 — Dead D3 Config Fields (Exhaustive)

**Answer: 7 of the 8 penalty-related D3 config fields are dead paths.** Only
`unexplained_tracklet_penalty` is live (covered in Question 5).

### The wiring gap

The D3 config fields live under `stages.stage_D.d3.*` in `default.yaml`. The ILP solver
(`d3_ilp2.py`) reads penalty values from the `constraints` dict, which comes from
`d2_constraints.json` (written by `d2_constraints.py`). But `d2_constraints.py` only
processes `identity_hints.jsonl` — it knows nothing about D3 penalty config. Nobody
bridges the gap.

The `solver.py:run_d3()` function reads the pipeline config and passes only two values
to `solve_structure_ilp2()` as explicit parameters:
- `unexplained_tracklet_penalty` (solver.py:64-68) — **LIVE**
- `group_boundary_window_frames` (solver.py:56-61) — **LIVE**

All other D3 penalty fields would need to flow through `constraints`, but no code
copies them there.

### Per-field trace

| Field | YAML location | YAML value | Consumer in d3_ilp2.py | How it's read | Dead? | Effective runtime value | Fix location |
|-------|--------------|------------|----------------------|---------------|-------|------------------------|-------------|
| `solo_ping_miss_penalty_mult` | `stages.stage_D.d3` (L296) | `50.0` | `_infer_ping_miss_penalty_unscaled()` (L694) | `constraints.get("solo_ping_miss_penalty_mult")` | **DEAD** | Fallback: hardcoded `50.0` (L701) | Wire through `solver.py:run_d3()` → `solve_structure_ilp2()` or inject into constraints dict |
| `group_ping_miss_penalty_mult` | `stages.stage_D.d3` (L297) | `60.0` | **None** — no code reads this field | n/a | **DEAD** | Never used; solo penalty (50.0) applies to all pings | Add group-specific miss penalty logic to `_solve_identity_ilp2_identity_only()` |
| `solo_ping_miss_penalty_abs` | `stages.stage_D.d3` (commented out, L299) | `null` | `_infer_ping_miss_penalty_unscaled()` (L688) | `constraints.get("solo_ping_miss_penalty_abs")` | **DEAD** | Never set in constraints → skipped | Same as `solo_ping_miss_penalty_mult` |
| `tag_fragment_start_penalty` | `stages.stage_D.d3` (L288) | `2500.0` | **None** — zero references in `src/bjj_pipeline/stages/stitch/` | n/a | **DEAD** | Never used | Build fragment-start penalty logic in ILP solver |
| `tag_fragment_start_penalty_mult` | `stages.stage_D.d3` (L303) | `20.0` | **None** — zero references in `src/bjj_pipeline/stages/stitch/` | n/a | **DEAD** | Never used | Same as above |
| `unexplained_group_ping_penalty` | `stages.stage_D.d3` (L307) | `5000.0` | **None** — zero references in `src/bjj_pipeline/stages/stitch/` | n/a | **DEAD** | Never used | Build GROUP-ping explain-or-penalize logic in ILP solver |
| `penalty_ref_edge_cost_quantile` | `stages.stage_D.d3` (L292) | `0.95` | **None** — zero references in stitch code | n/a | **DEAD** | Never computed; `penalty_ref_edge_cost` is never populated in constraints | Compute from allowed edge cost distribution in `solver.py` or `d3_compile.py`, inject into constraints |
| `penalty_ref_edge_cost_min` | `stages.stage_D.d3` (L293) | `0.01` | **None** — zero references in stitch code | n/a | **DEAD** | Never used | Same as above |

### Why the dead path exists

`_infer_ping_miss_penalty_unscaled()` (d3_ilp2.py:676-701) was written to support a
future config-driven penalty system with this preference order:

1. `solo_ping_miss_penalty_abs` (absolute override) — from `constraints`
2. `solo_ping_miss_penalty_mult * penalty_ref_edge_cost` (scaled) — from `constraints`
3. Hardcoded fallback: `50.0`

The function's docstring says: *"This keeps ILP2 runnable even when compile doesn't yet
forward all config-derived fields."* The "yet" is the clue — the forwarding was planned
but never built. Both `solo_ping_miss_penalty_abs` and `penalty_ref_edge_cost` would
need to be injected into the constraints dict by `solver.py` or `d3_compile.py`.

### Coincidental correctness

For `solo_ping_miss_penalty_mult`: the YAML value is 50.0 and the hardcoded fallback is
also 50.0. So the runtime behavior happens to match the intended config, but only by
coincidence. If someone changed the YAML value, nothing would change at runtime.

### CP3 work list (derived)

To make all D3 penalty fields live, the minimal wiring changes are:

1. **Compute `penalty_ref_edge_cost`** from allowed-edge cost distribution (using
   `penalty_ref_edge_cost_quantile` and `penalty_ref_edge_cost_min`) — natural location:
   `d3_compile.py` after pruning, or `solver.py:run_d3()` after compilation.
2. **Forward config fields into constraints dict** (or as explicit parameters to
   `solve_structure_ilp2()`): `solo_ping_miss_penalty_mult`, `solo_ping_miss_penalty_abs`,
   `group_ping_miss_penalty_mult`, `group_ping_miss_penalty_abs`,
   `penalty_ref_edge_cost` (computed), `tag_fragment_start_penalty`,
   `tag_fragment_start_penalty_mult`, `unexplained_group_ping_penalty`.
3. **Add consumer logic** in `_solve_identity_ilp2_identity_only()` for fields that have
   no consumer at all: `group_ping_miss_penalty_mult`, `tag_fragment_start_penalty(_mult)`,
   `unexplained_group_ping_penalty`.

---

## Question 4 — `birth_non_entrance_add_cost` and `death_non_exit_add_cost`

**Answer: Not overridden anywhere. Runtime value = code default 8.0.**

### Trace

| Location | Value |
|----------|-------|
| `configs/default.yaml` | Not present (grep confirms zero hits in all `.yaml` files) |
| `configs/cameras/*.yaml` | No camera YAML files exist |
| `src/bjj_pipeline/config/models.py:476-477` | Pydantic defaults: `8.0` |
| `src/bjj_pipeline/stages/stitch/costs.py:333-334` | `cfg.get("birth_non_entrance_add_cost", 8.0)` / `cfg.get("death_non_exit_add_cost", 8.0)` |
| `stage_D/audit.jsonl` → `d2_config_resolved` → `d2_costs` | Keys absent from audited config (confirmed for FP7oJQ) |

The D2 runner (`d2_run.py:82`) reads only `stage_D.d2_costs` from config. Since
`birth_non_entrance_add_cost` is not under `d2_costs` in `default.yaml`, it never enters
the `cfg` dict passed to `compute_edge_costs()`. The `cfg.get(..., 8.0)` fallback in
`costs.py` provides the effective value.

**These are live code paths** — the penalties are applied to BIRTH/DEATH edges at runtime.
They just can't be tuned via config today. To make them configurable, add them under
`stages.stage_D.d2_costs` in `default.yaml`.

---

## Question 5 — `unexplained_tracklet_penalty: 15.0` Verification

**Answer: Confirmed live and applied at 15.0 across all three eval clips.**

### Config → runtime trace

```
configs/default.yaml:286
  stages.stage_D.d3.unexplained_tracklet_penalty: 15.0
    ↓ deep_merge (no camera/CLI override)
    ↓ to_runtime_config() copies to config["stage_D"]["d3"]["unexplained_tracklet_penalty"]
solver.py:64-68
  penalty = _cfg_get(config, "stages.stage_D.d3.unexplained_tracklet_penalty",
            _cfg_get(config, "stage_D.d3.unexplained_tracklet_penalty", None))
  → penalty = 15.0
solver.py:75
  solve_structure_ilp2(..., unexplained_tracklet_penalty=15.0)
d3_ilp2.py:2298
  → passed as explicit parameter (not via constraints dict)
d3_ilp2.py:2325
  → forwarded to solve_structure_ilp2_core()
d3_ilp2.py:1411
  → forwarded to _solve_identity_ilp2_identity_only()
  → applied as ILP penalty term for unexplained base tracklet IDs
```

This is the **one D3 penalty field with correct wiring** — it bypasses the broken
constraints path entirely by flowing as an explicit function parameter.

### Audit JSONL verification

Extracted from `stage_D/audit.jsonl` → `d3_ilp_summary` → `explain_or_penalize`:

| Camera | `unexplained_tracklet_penalty` | `n_tracklets_total` | `n_tracklets_explained` | `n_tracklets_unexplained` |
|--------|-------------------------------|--------------------|-----------------------|--------------------------|
| FP7oJQ | **15.0** | 251 | 212 | 39 |
| J_EDEw | **15.0** | 236 | 198 | 38 |
| PPDmUg | **15.0** | 73 | 58 | 15 |

All three clips show `unexplained_tracklet_penalty: 15.0`, matching the config value.

Unexplained tracklet rates: FP7oJQ 15.5%, J_EDEw 16.1%, PPDmUg 20.5%. These represent
tracklets the ILP solver chose to leave unassigned (paying 15.0 penalty each) rather than
route through the graph. Whether 15.0 is the right value is a CP2 question.

---

## Sidebar A — Broken Config Dump in `pipeline.py`

`src/bjj_pipeline/stages/orchestration/pipeline.py:670-683` writes a `config_resolved`
event to `orchestration_audit.jsonl` that would contain both `resolved_config` and
`runtime_config` — the full effective merged config for every pipeline run. This is
exactly the diagnostic tool needed for CP2 verification of penalty changes.

**The event is silently dropped.** Line 681 references a variable `mode` that is not
defined in `run_pipeline()`'s scope (it's not a parameter, not a local, not imported).
This raises `NameError`, caught by the bare `except Exception` on line 684. The
`config_resolved` event never makes it to disk. The `run_started` event on line 687
writes successfully, masking the failure.

This is a one-line fix (remove the `"mode": mode` line, or pass `mode` as a parameter).
It would be a good tiny Task Brief on its own — the fix is trivial but the diagnostic
value for CP2+ is high.

---

## Sidebar B — No Per-Camera YAML Overrides Exist

The config loader supports `configs/cameras/<camera_id>.yaml` as a camera-specific
overlay (loader.py:160-165). However, no such files exist today:

```
configs/cameras/FP7oJQ/   → homography.json, roi_mask.json, height_surface.json, PNGs
configs/cameras/J_EDEw/   → homography.json, roi_mask.json, height_surface.json, PNGs
configs/cameras/PPDmUg/   → homography.json, roi_mask.json, height_surface.json, PNGs
```

All three cameras use `default.yaml` as their sole config source (plus per-camera
`homography.json`). If camera-specific D3 penalty tuning is needed (e.g., different
`unexplained_tracklet_penalty` for PPDmUg's higher unexplained rate), camera YAML
files would need to be created.
