# Blast-Radius Check: Mixed-Provenance Baseline Artifacts

## Step 1: Freshened Baseline Confirmation

Stale D2-D4 artifacts backed up to `outputs/_eval_gt_stale_backup_20260630/`.
Stage D re-run (D0->D4) on current code/artifacts via `freshen_eval_gt.py`.
Standard gt2actuals CLI re-run against freshened `outputs/_eval_gt/`.

| Clip | Stale (34.7%) | Freshened | Delta |
|------|--------------|-----------|-------|
| vid1 | 40.3% | **39.3%** | -1.0pp |
| vid2 | 30.6% | **27.5%** | -3.1pp |
| **Combined** | **34.7%** | **32.5%** | **-2.2pp** |

The freshened baseline is **32.5%** — confirmed by both the standard gt2actuals CLI
and the sweep harness (after fixing a glob collision bug in `run_gt2actuals.py`).

### Glob collision bug (discovered during verification)

`run_gt2actuals.py` created symlinks at `outputs/_sweep/_gt2a_gym/<cam>/<run_id>/`
for glob resolution. When multiple sweep runs existed, the glob
`_gt2a_gym/<cam>/**/<clip_id>` matched ALL runs, and sorted order returned the
wrong one. Fix: scoped gym_id to `_gt2a/<run_id>/` so each run's glob is unique.

### Three baselines explained

| Baseline | Value | Source |
|----------|-------|--------|
| Stale pre-existing | 34.7% | Mixed-provenance D2-D4 from Jun 7, pre-CP-TAG-4a code |
| Freshened eval_gt | 32.5% | D0->D4 re-run on current code, original Stage A tracklets |
| Sweep replay | 30.7% | Different tracklet assignments from BotSort replay |

The 2.2pp gap (34.7% -> 32.5%) is the staleness artifact.
The 1.8pp gap (32.5% -> 30.7%) is from the replay producing different tracklet
assignments than the original pipeline (expected — see SWEEP-3 commit message).

## Step 2: Provenance Table for Locked CLAUDE.md Figures

### CP-TAG-4a "+22.7pp improvement" — EVIDENCE RETRACTED

**Finding:** Both the 40.5% "baseline" and the 63.2% "post-fix" figures were computed
from the SAME `person_tracks.parquet` (mtime Jun 7 13:05:03, 37 minutes BEFORE the
CP-TAG-4a commit at 13:42). The +22.7pp difference is a frame-selection effect
(full-range 4,214 GT-person-frames vs val-split 714 frames), not a code-change effect.

**Evidence chain:**
1. `signal_preservation_summary.json` (mtime Jun 7, 14:27): 40.5% computed over 4,214
   GT person-frames (all annotated, stride-10, 301 frames)
2. `docs/evidence/cp_purity_2/m1_reconciliation.json`: 63.2% computed over 714
   GT person-frames (val-split only, frames 2500-3000)
3. `tools/cp_purity_2_floor.py` line 330: `_load_clip_person_tracks(VID1_DIR)` reads
   from `outputs/_eval_gt/.../stage_D/person_tracks.parquet` — the same file
4. No intermediate or cached person_tracks exists — `_load_clip_person_tracks` is a
   direct `pd.read_parquet()` call

**Implication:** CP-TAG-4a's actual effect on correct_id is UNKNOWN. The code changes
(Fix 0+A+C+D) remain in the codebase but their net effect has never been measured via
a proper before/after comparison.

### CP-PURITY-1: "100% attributed to ILP stitch routing"

- Source: `tools/cp_purity_1_decomposition.py`, reads session-level
  `person_tracks_J_EDEw.parquet` (mtime Jun 7 13:30, pre-CP-TAG-4a commit)
- **At risk: YES** — the specific numbers (which person_id follows which GT person)
  were measured against pre-CP-TAG-4a person_tracks. The structural finding (corruption
  enters at solver layer due to detection under-segmentation) is an architecture claim
  that doesn't depend on specific person_ids, but the quantitative breakdown should be
  treated as unverified.

### Full provenance table

| Locked figure | Source artifact | Artifact mtime | CP-TAG-4a commit | At risk? |
|---|---|---|---|---|
| 40.5% pre-split baseline | signal-trace, full annotated | Jun 7, 14:27 (reads Jun 7, 13:05 person_tracks) | Jun 7, 13:42 | **Stale** — pre-commit person_tracks |
| 63.2% post-CP-TAG-4a | CP-PURITY-2 val-split | Jun 8-9 (reads Jun 7, 13:05 person_tracks) | Jun 7, 13:42 | **INVALID** — same artifact as 40.5%, frame-selection effect |
| +22.7pp improvement | Delta of above | N/A | N/A | **INVALID** — not a code-change measurement |
| 33.9% post-split combined | CP-GT2ACTUALS-3.5 | Jun 10 (reads Jun 7, 13:05 person_tracks) | Jun 7, 13:42 | **Stale** — pre-commit person_tracks |
| 34.7% gt2actuals combined | baseline.json | Jun 10 (reads Jun 7, 13:05 person_tracks) | Jun 7, 13:42 | **Stale** — now freshened to 32.5% |
| CP-PURITY-1 100% ILP stitch | cp_purity_1_decomposition.py | Jun 8 (reads Jun 7, 13:30 session person_tracks) | Jun 7, 13:42 | **Stale** — structural claim likely sound, numbers unverified |
| CP-PURITY-3 GT-through-D oracle | cp_purity_3 evidence | Jun 9 | N/A | **Clean** — uses GT detections directly, no person_tracks dependency |

## Step 3: Process Gap

`ClipManifest` has `created_at_ms` and `pipeline_version` but no per-stage timestamps
or git commit SHAs. No mechanism exists to detect D0/D1 vs D2-D4 staleness
automatically. A future hardening item: a pre-flight check that warns if Stage D
artifacts are older than Stage A/C artifacts they were built from, or older than the
current git HEAD's relevant source files.
