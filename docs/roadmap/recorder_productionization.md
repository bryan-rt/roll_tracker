# Nest Recorder — Full Productionization Plan

> **Ownership:** Scope is owned by the web session — adding, removing, or resequencing
> checkpoints happens there. The CLI owns STATUS ONLY: ticking boxes, recording outcomes,
> and linking evidence. Do not add, remove, or resequence checkpoints from a CLI session.
> If a checkpoint appears wrong or incomplete, flag it rather than editing it.

> **Resequencing (2026-07-31):** CP-R2 now precedes CP-R1. CP-R2 builds passthrough behind
> a new opt-in flag with the default unchanged, so it does not alter production recording
> behaviour and does not require a rebuild. Building it first lets the CP-R1 capture run
> with passthrough enabled, satisfying CP-R1's reliability goals and CP-R2's measurement
> goals in a single capture rather than two.

## Execution Order

Checkpoint IDs are labels, not sequence. Work proceeds in this order:

| # | Checkpoint | Why here |
|---|---|---|
| 1 | CP-R9 — Smoke harness + caffeinate wiring | Small; de-risks every checkpoint after it. Caffeinate is a CP-R8 prerequisite. |
| 2 | CP-R4 — Pin image + startup assertion | Requires a rebuild; surface any ffmpeg-version surprise now, not before GT capture. |
| 3 | CP-R5 — Sidecar boundary fix | Prerequisite for CP-R6. Validated by the existing DUPFIX instrument. |
| 4 | CP-R6 — Sidecar contract v2 (incl. TRIM-BIMODAL) | The payoff — where recorder work becomes usable downstream. |
| 5 | CP-R10 — Session churn investigation | 1-3 min gaps are unacceptable during a GT session. Must resolve before CP-R8. |
| 6 | CP-R7 — Hardening remainder | Low priority, no dependents. |
| 8 | CP-R13a — Encoder timebase fix | Prerequisite for CP-R13b. Preserves real PTS through x264. |
| 9 | CP-R13b — Mp4-derived sidecars | The sidecar join guarantee. Schema 5. |
| 7 | CP-R8 — Clean-footage GT capture | Depends on all of the above. |

**Hard dependencies:**
- CP-R5 → CP-R6 — the boundary bug corrupts per-segment counts *and* `base_pts` (the
  sidecar's time origin). Fixing it later would shift the origin under consumers.
- TRIM-BIMODAL → CP-R6 — `measured_fps` cannot be contracted as authoritative while it
  misreports bimodal segments.
- CP-R9 (caffeinate) → CP-R8 — the GT capture is long and unattended.
- CP-R4 → CP-R8 — do not record footage you care about on an unpinned image.

CP-R4 is otherwise independent. Most CP-R7 items are standalone.

---

**Design principle:** Upstream network loss cannot be prevented. Recorder-injected
corruption can be eliminated entirely. Where loss is unavoidable, **preserve the evidence**
rather than smoothing it away — a visible gap is information; a normalized gap is a silent
false teleport. (CP-R11: measured gaps are predominantly camera-internal grid mismatch, not
transport loss — but genuine upstream loss remains unpreventable where it occurs.)

**Architectural basis:** True timestamps can always be resampled downstream. A resampled
grid can never be un-resampled. Collect at maximum fidelity; decide consumption later.

---

## The two switches (they are NOT one thing)

| Switch | Side | Controls | Today's default | Target |
|---|---|---|---|---|
| `SOURCE_PTS` | input | Keep camera capture timestamps vs substitute network-arrival times | `0` (arrival) | `1` |
| `fps_mode` | output | Resample onto a uniform grid vs pass frames through | unset → CFR | `passthrough` |

**`SOURCE_PTS=1` alone still resamples.** That combination is what runs today and it is what
produced the 8% padding on FP7oJQ. Both switches must move.

Good news: `FPS_MODE_OPTS=(-fps_mode passthrough)` **already exists** in `diag_v6.sh`, gated
to `REENCODE=0` (stream copy). The change is to apply it in the re-encode branch too —
a proven option, already in the codebase.

Target production config: `SOURCE_PTS=1` + `REENCODE=1` + `-fps_mode passthrough`
= true timestamps, no resampling, showinfo sidecar retained.

---

## CP-R1 — Validation capture

**Status:** ✅ Complete
**Evidence:** 65-min capture 2026-08-04 19:32-20:37 UTC. All four validation targets
confirmed: optimistic URL reuse fired, token refresh carried past the ~21-25 min barrier
(17 extends/camera, zero 401s), no 429s. Coverage PARTIAL — session churn produced 1-3 min
gaps (see CP-R10). J_EDEw produced zero segments. Uncovered a pre-existing crash
(`local` outside function, fixed in `6112dc0`).

Gates every production change below. Last unvalidated reliability behaviours.

- 30-min capture via `diag_v7_2.sh`, detached, `SOURCE_PTS=1`, per
  `docs/guides/runbook_cross_camera_capture.md`.
- Reconcile the runbook's `WINDOW_SECONDS=3900` against the 1800 in the checkpoint brief
  before launching.
- Confirm: optimistic URL reuse fires ("0 API calls"), token refresh carries past the old
  ~21–25 min barrier, sustained coverage with no multi-minute gaps, no 429s.
- Run the DUPFIX instrument over the output — first conservation + PTS-gap measurement on a
  long window rather than 45s smoke tests.

**Done when:** all four behaviours confirmed; per-attempt conservation and gap figures
recorded for the full window.

**Watch for:** if the measured gap rate materially exceeds the 0.1–7.7% seen so far, that
changes the checkpoint-2 calculus on `track_buffer` and coast-step injection.

---

## CP-R2 — Passthrough: build and measure (flag stays opt-in)

**Status:** ✅ Complete
**Evidence:** commit `b1f3b08` + smoke tests 2026-08-04

Build it, prove it, do **not** default it yet.

**Evidence basis:** `docs/evidence/recorder_dupfix_1/findings.md`

**The change:** add `FPS_MODE_OPTS=(-fps_mode passthrough)` to the `REENCODE=1|2` branch of
`diag_v6.sh`, behind a new opt-in env var (e.g. `FPS_PASSTHROUGH`, default `0`). Script is
live-mounted; no rebuild.

**Measure against a paired CFR capture on the same cameras, same window:**

- [ ] Attempt-level conservation closes (Σ showinfo == Σ decoded output). Under passthrough
      there is no resampling, so any residual is boundary-attribution or real loss only.
- [ ] `framehash_adjacent_dups` = 0 across all segments.
- [ ] PPDmUg's CFR decimation is **gone** — stated as *attempt-level conservation closes
      on PPDmUg attempt_1* (previously −105 frames). Do NOT anchor this criterion to the
      15.86-vs-15 rate gap: that 15.86 comes from `070219`, a first segment with a
      documented startup transient (10ms PTS stdev vs 0.47ms steady-state), so it is likely
      a transient artifact rather than the camera's true rate. Conservation is measured
      directly and does not depend on that attribution.
- [ ] Upstream gaps are **preserved and visible** in the PTS stream, not padded over.
- [ ] `check_stage_a_compat()` passes: `cv2_iterated_count` matches container metadata and
      `CAP_PROP_FPS` returns a sane rate on a VFR file.

**Risk to verify explicitly — segment durations.** `-copyts` plus `-f segment
-segment_time 120` plus `-reset_timestamps 1` is a combination whose interaction is not
obvious. Confirm segments still land at ~120s and that `segment_time` is evaluated against
the expected timebase. This is the single most likely way passthrough breaks something.

**Secondary risk:** VFR output plus `-g 30` means keyframe *timing* varies with rate.
Confirm segmentation still cuts cleanly.

**Done when:** all boxes checked on real footage from all three cameras, with a CFR control
capture for comparison.

---

## CP-R3 — Flip the defaults (script-only, instantly revertible)

**Status:** ✅ Complete
**Evidence:** defaults flipped in diag_v6/v7_2/v8.sh; rollback via `SOURCE_PTS=0 FPS_PASSTHROUGH=0`

Only after CP-R2 passes on real footage.

- `SOURCE_PTS` default `0` → `1`.
- `fps_mode` passthrough becomes default for `REENCODE=1|2`.
- **Keep both switches.** `SOURCE_PTS=0` and `FPS_PASSTHROUGH=0` remain reachable as
  rollback paths — exactly as you intended.
- Log the active timing configuration at recorder startup so any clip's provenance is
  recoverable from its log.
- 60s three-camera smoke test.

Script-only and live-mounted, so rollback is editing one line and restarting — no rebuild.

**Done when:** default capture produces source-PTS, non-resampled footage; smoke test
passes; rollback verified to work.

---

## CP-R4 — Pin the base image + startup assertion (requires rebuild)

**Status:** ✅ Complete
**Evidence:** Rebuilt 2026-08-05. `debian:trixie-slim` (Debian 13.6), ffmpeg 7.1.5-0+deb13u1
(unchanged from prior image). `check_ffmpeg_opts.sh` asserts `-timeout`, `-fps_mode`,
`-copyts` — called from both `entrypoint.sh` (production) and `smoke_test.sh` (dev).
Assertion failure demonstrated (exit 1 on missing option). `COPY *.sh` prevents future
scripts from being absent in production. No apt pin on ffmpeg (assertion is the protection;
trixie will not cross 7.x boundary). `smoke_test.sh` both modes passing post-rebuild.

`debian:stable-slim` silently rolled bookworm→trixie and invalidated `-stimeout`, which
would have caused total recording failure. Now that correct behaviour depends on `-timeout`
**and** `-fps_mode`, an unpinned base is a larger liability than before.

- Pin the Debian version and ffmpeg major in `recorder/Dockerfile`.
- Add a **startup assertion**: verify required ffmpeg options exist before recording
  (`-timeout`, `-fps_mode`, `-copyts`). Fail loudly at container start rather than silently
  at capture time.
- Rebuild + 60s smoke test.

**Done when:** pinned; assertion demonstrably fires on a deliberately broken option; smoke
test passes.

---

## CP-R5 — Fix the sidecar boundary split (live correctness bug)

**Status:** ✅ Complete
**Evidence:** `docs/evidence/recorder_boundary_fix_1/findings.md`. PTS-based split replaces
line-position split. PPDmUg seg1 residual +30 -> +0 (exact). FP7oJQ recovered 47 leading-
edge frames (3.1s). Schema bumped to 3. Muxer lag mechanism unexplained (0.27s PPDmUg vs
3.1s FP7oJQ — ruled out x264 lookahead, not ruled out muxing queue / stderr interleave).

**Evidence basis:** `docs/evidence/recorder_dupfix_1/findings.md`

DUPFIX-2 confirmed boundary misattribution, and the magnitude — roughly nine seconds of
stderr offset at attempt start — is **not explained by encoder buffering**. Every production
sidecar currently has an unreliable per-segment `input_count`, and the mechanism is unknown.

- Replace the stderr log-line-position split with a **PTS-range-derived** boundary. Frames
  belong to a segment by timestamp, not by where a log line landed.
- Re-run the DUPFIX instrument against the old split to quantify what changed.
- Resolve or formally park the ~9s offset mechanism.
- **Note:** `pts_time_s` in the sidecar is segment-relative (base-subtracted from the first
  PTS in the segment's attributed stderr range). CP-R5's boundary fix **has shifted** the
  time origin for affected segments — documented in
  `docs/evidence/recorder_boundary_fix_1/findings.md`.
- **Corroboration (CP-R2 smoke test, 2026-08-04):** mismatch pattern on PPDmUg passthrough
  reproduces the boundary hypothesis exactly: seg1 ni=357/output=330 (+27), seg2 300/300
  (0), seg3 228/255 (−27). The +27/−27 cancel, middle segment is clean. Strongest
  confirmation yet of the start-of-attempt stderr offset.

Do this **before** CP-R6, so the new contract isn't built on an unexplained parsing bug.

**Done when:** per-segment counts reconcile without position-based attribution; conservation
closes on attempts with no real loss.

---

## CP-R6 — Sidecar contract v4 (the handoff — where the value is realized)

**Status:** ✅ Complete
**Evidence:** `docs/reference/sidecar_contract.md` (authoritative spec)

**Evidence basis:** `docs/evidence/recorder_dupfix_1/findings.md`

The sidecar is collection-only today. This is what makes it consumable. **Without this
step, passthrough produces correct footage that nothing reads.**

Replace **constructed** fields with **observed** ones:

| Field | Source | Purpose |
|---|---|---|
| `dt_s` | consecutive source PTS delta | True per-frame interval. Kills the fps bug at source. |
| `measured_fps` | trimmed mean of tick deltas | Segment-level summary. See "Bimodal rate representation" below for limitations. |
| `gap_flag` | `dt_s > 1.5x` nominal | False-teleport signal, preserved not smoothed. |
| `implied_missing_frames` | `round(dt/nominal) - 1` | Coast-step count for Stage A injection. |
| `is_duplicate` | framehash equality | Observed, not inferred from a count mismatch. Should be permanently 0 post-CP-R3 — a regression canary. |
| `pts_wallclock_offset_s` | host-clock lower envelope | Absolute time anchor (±14–56ms estimated). Already emitted in `_meta`; needs contracting, not building. |
| `drift_ppm` | per-camera measured drift | Linear clock correction (e.g. FP7oJQ −603 ppm) |
| `drift_flat` | existing `_meta` flag | Whether linear correction is valid for this clip |
| `schema_version` | — | Lets the pipeline assert compatibility. |

RTCP is absent on all cameras (CAPTURE-TIME-2), so an absolute camera clock is unavailable
from the stream. The `pts_wallclock_offset_s`, `drift_ppm`, and `drift_flat` fields are the best
available estimate, not ground truth.

**Retire `input_n` as a duplicate signal.** DUPFIX proved it is an arithmetic construction,
not an observation.

**Resolved anomaly: `pts_time` precision (CP-R2b).** ffmpeg's showinfo emits `pts_time` with
only 3 decimal places. Fixed by extracting raw `pts` ticks (integer) and dividing by the
timebase (parsed from the showinfo config line, fallback 90000). True camera rate is
~15.000fps (tick deltas alternate 6030/5940 at 1/90000 timebase, mean ~6000 = exactly
15.000fps), not the 14.9254 the quantized median computed. `measured_fps` now uses a
trimmed mean of tick deltas (gap-robust AND alternation-correct). `sidecar_schema: 2`
marks the new format.

**Caveat: `pts_stdev_delta_ms` measures tick alternation, not camera jitter.** v1 and v2
sidecars agree (~0.47ms on clean PPDmUg segments). The camera alternates 6030/5940 ticks
(±30 ticks = ±0.333ms) to distribute a non-integer mean of ~6000 ticks — this alternation
dominates the stdev. The field should not be used as a jitter proxy; it measures the
encoder's tick-distribution pattern. RELIABILITY-1's elevated-stdev startup-transient
finding (~5ms on first segments) likely reflects real startup behaviour overlaid on
alternation, but the two are not separable from stdev alone.

**measured_fps is only valid under source-PTS.** Under `SOURCE_PTS=0` (arrival-PTS rollback),
inter-frame deltas are burst-distributed and the trimmed mean returns nonsense (~14000 on
PPDmUg, measured 2026-08-04 CP-R3 smoke test). The span-based `measured_fps_mean` still reads
~15 in the same segments — the disagreement between the two fields is the diagnostic that
catches this. CP-R6's contract must scope `measured_fps` to `timing_mode: "passthrough"` /
source-PTS only, or emit an explicit validity flag. Consumers must not read `measured_fps`
from an arrival-PTS sidecar.

**Bimodal rate representation (TRIM-BIMODAL + per-clip-scalar — CP-R1b).**

CP-R1b proved that frames arrive at two discrete rates — 33ms (~30fps) and 67ms (~15fps) —
interleaved within a single segment, with the short-mode proportion shifting mid-stream
within one continuous ffmpeg invocation. Container metadata (`r_frame_rate`, `CAP_PROP_FPS`)
records the rate at container creation and is stale when the stream changes underneath.
PPDmUg shows the same mechanism at lower magnitude (interleaved 33ms frames producing
15.2-17.6 `measured_fps`). Evidence: `docs/evidence/recorder_fps_adaptation_1/findings.md`.

A per-clip scalar fps is therefore **provably insufficient** — no single number describes
a segment containing a proportion transition. `dt_s` per frame is the only reliable rate
source.

The current trimmed mean (lo = median x 0.5, hi = median x 1.5) additionally **misreports
bimodal segments** (TRIM-BIMODAL defect): the majority mode captures the median, and the
bounds discard the minority mode as "outliers." Three failure modes observed:

| Segment | Short-mode % | Discard % | measured_fps | Correct? |
|---------|-------------|-----------|-------------|----------|
| FP7oJQ-163102 | 65.9% | 36.0% | 30.0019 | Wrong — reports majority mode only |
| PPDmUg-163240 | 16.1% | 10.8% | 15.4530 | Inflated — trims 2970-tick, keeps 3060-tick |
| PPDmUg-163041 | 29.9% | 0.0% | 17.6351 | Correct (by luck: lo = 2970, exactly the tick value) |

Does NOT affect stable-rate segments (all controls show correct `measured_fps`).

CP-R6 must decide how the contract represents a bimodal segment. Leading direction: detect
bimodality (two peaks in the tick-delta histogram separated by ~2x) and report both modes
plus their proportions (`mode_1_fps`, `mode_1_proportion`, `mode_2_fps`,
`mode_2_proportion`, `is_bimodal` flag), rather than forcing a single scalar.

**Free drop metric from trimmed mean.** `pts_delta_trim_total - pts_delta_trim_kept` is
the number of PTS gaps (dropped frames) per segment. FP7oJQ smoke test: 8–9% discarded on
all three segments, consistent with DUPFIX's 0.1–7.7% per-attempt range. These are real
false teleports reaching the tracker; coast-step injection (checkpoint 2) addresses them.

**Open anomaly: drift instability on short segments.** CP-R2 smoke test measured
`drift_ppm: 2449` on a 20s PPDmUg segment with only 2 drift windows, vs RELIABILITY-1's
−603 ppm on FP7oJQ over 5 minutes. A drift figure computed from ≤2 windows is unstable and
should not be contracted as authoritative without a minimum-window guard (e.g. require
n_drift_windows ≥ 4 for `drift_flat: false`).

Write the contract into `.claude/rules/` or `docs/reference/` so checkpoint 2 codes against
a spec rather than reverse-engineering a JSONL.

**Done when:** schema documented, emitted in production, sample validates against it.
Bimodal-segment representation decided and implemented (the TRIM-BIMODAL fix is part of
this checkpoint, not a separate item).

---

## CP-R7 — Hardening and operability

**Status:** ⚠ Partial (three-camera smoke test pending — J_EDEw offline)
**Evidence:** Open anomaly register at `docs/evidence/recorder_dupfix_1/findings.md` § Open Anomalies.

- ~~`N_CAMERAS=0` edge~~ **Done.** Floor `N_CAMERAS` at 1 before the division (line 28).
- ~~Per-camera coverage/uptime metric~~ **Done (CP-R10).** `services/nest_recorder/coverage_report.py`.
- ~~Park open anomalies~~ **Done.** Two entries in `docs/evidence/recorder_dupfix_1/findings.md`
  § Open Anomalies: (A) 1867 mpdecimate count — PARKED UNEXPLAINED, `Frames - Dups == nb_frames`
  pattern across 4 segments suggests column-semantics issue; (B) 0.18% PPDmUg dups — PARKED,
  explained as encoder quantization on static scene.
- ~~J_EDEw intermittent offline~~ **Resolved — outside recorder scope.** The camera has been
  unavailable since 2026-05-31. Across both 65-min captures (CP-R1 2026-08-04, CP-R10
  2026-08-05) and all smoke tests since, J_EDEw produced zero segments with 19-24 retry
  attempts per session. Observed failure signature: `generate_stream` succeeds (API returns
  RTSP URL), ffmpeg receives ~10-11s of data, stream terminates, session classified as dead.
  Retry and backoff escalation (to 300s slow-poll after 5 consecutive failures) work correctly.
  Availability is a hardware/network question outside the recorder's scope. **Forward test:**
  if J_EDEw returns and works normally, the signature was what an offline camera looks like
  through the SDM API. If it returns and STILL shows the ~11s-then-terminate pattern, that is
  a camera-specific failure (not availability) and becomes worth investigating.
- **Three-camera smoke test:** PENDING. Requires all three cameras online. Every smoke test in
  this series has had at least one SKIP (J_EDEw). Zero-SKIP confirmation is the remaining
  deliverable for CP-R7 completion.

---

## CP-R8 — Clean-footage GT capture

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

Terminal recorder deliverable and the bridge to checkpoint 2.

All existing GT predates these fixes and carries both duplicates and drops. Every CV
conclusion resting on it — the 41% Stage A drift attribution, the purity proxies, the
Stage A sweeps — is held loosely for that reason.

- Capture a GT-suitable session on the fully fixed recorder.
- Send to CVAT for annotation.
- Keep old GT as a regression baseline; do not discard it.

**Done when:** clean footage with GT exists, and drift attribution can be re-measured.

**Mid-roll cadence-switch caveat (CP-R11/R12).** CP-R11 established that 30fps blocks last
seconds to tens of seconds. A cadence switch landing mid-roll during GT footage is exactly
the case coast-step injection handles worst, and produces a GT clip whose cadence changes
partway through. Mitigations:

- Check captured GT footage for mode switches **before** sending to CVAT. The `is_bimodal`
  flag in the segment's `.timing.jsonl` sidecar makes this checkable without decoding.
- If a switch lands mid-roll, note it -- that clip becomes a useful test case for the variable-
  dt implementation, but a poor baseline for measuring drift attribution.
- PPDmUg (primary GT camera, 95.9% coverage) has 12.5% of segments with mode switches --
  this is not a tail event.
- Capture more footage than the GT set requires, and select mode-switch-free clips for the
  drift-attribution baseline. Both cameras carry exposure (FP7oJQ 0.7% of segments, PPDmUg
  12.5%), so this is a selection problem, not an avoidance problem. Checking `is_bimodal`
  across the captured set BEFORE annotation is cheap; re-annotating is not.

---

## CP-R9 — Smoke harness + caffeinate wiring

**Status:** ✅ Complete
**Evidence:** `smoke_test.sh` default + rollback both passing (37 PASS / 0 FAIL / 1 SKIP
per mode, 2026-08-05). `capture.sh` with `caffeinate -dim -t`. `run_process.sh` `-s` dropped.

Two operational gaps that de-risk every checkpoint after them.

**Regression smoke harness.** A single scripted command running the standard check:
`diag_v7_2.sh` (never `diag_v6.sh` standalone — that fails Generate 404 because `.env`
`DEVICE_*` values are human-readable names, not SDM device paths; `diag_v7_2.sh` resolves
them), short `SEG_SECONDS`, then assert on startup log values, `timing_mode`,
`sidecar_schema`, `measured_fps` sanity, and segment durations. **Rationale: the ad-hoc
smoke test has failed twice — once run against `diag_v6.sh` standalone and misdiagnosed
(CP-R3), once skipped entirely (CP-R3 Pass 1).** Scripting it makes every later checkpoint
cheaper and harder to get wrong.

**Caffeinate wiring gap.** `services/nest_recorder/run_process.sh` has
`caffeinate -dims -s -w $`, and `docs/decisions-archive.md` records it as a decision — but
the runbook instructs `docker compose exec -d` **directly**, bypassing the wrapper entirely.
So caffeinate never runs on the documented capture path. Two further issues: `-s` only holds
while on AC power, and `-w $` dies when the wrapper exits, so a detached `exec -d` would
outlive its own caffeinate. Fix the wiring so long captures are protected regardless of
invocation path.

**Done when:** smoke harness passes on a healthy container; caffeinate protects the
documented capture path; runbook updated.

---

## CP-R10 — Session churn investigation

**Status:** ✅ Complete (Outcome 3 — inconclusive, single sample)
**Evidence:** `docs/evidence/session_churn_1/findings.md`

**Result:** Caffeinate materially reduced attempts on both cameras (14→4 FP7oJQ, 14→6 PPDmUg)
and eliminated the correlated gap pattern that characterized display-sleep-induced churn.
Display sleep confirmed as A cause. However, FP7oJQ's total gap increased (+23%) due to a
late-capture cluster at ~45 min (possibly a Nest session lifetime limit), masking the
improvement. PPDmUg improved clearly: total gap -67.6%, longest run +59%.

**CP-R8 gate:** Proceed. Use `capture.sh` (caffeinate). PPDmUg is the primary GT camera
(95.9% coverage). FP7oJQ may need proactive session refresh for captures >45 min.

**Coverage metric:** `services/nest_recorder/coverage_report.py` (pulled forward from CP-R7).
Session-scoped (`--start-epoch` / `--window`), per-camera, named gaps.

**Bimodal validation (CP-R6, opportunistic):** 8 PPDmUg segments emitted `is_bimodal: true`
with valid `short_mode_*` fields. Contract caveat removed.

**Drift windows (CP-R6, opportunistic):** 120s segments reach 12-13 drift windows.
`drift_rate_s_per_s` and `drift_ppm` confirmed emitted in production.

---

## Definition of "production-ready"

The recorder is done when all of these hold:

1. Recorder-injected corruption is **zero**: no duplicates, no CFR decimation.
2. Unavoidable upstream loss is **detected and flagged**, not concealed.
3. True per-frame timing reaches the **container** (CP-R13a) and the **sidecar** (CP-R6/R13b).
   The sidecar's `frame_index` join to Stage A is guaranteed by construction (row count =
   decode count, schema 5). Per-frame `dt_s` is the authoritative timing source, not a
   per-clip fps scalar.
4. A rebuild **cannot silently** invalidate required ffmpeg options.
5. Correct behaviour is the **default**, with rollback switches retained.
6. A **single command** regression-tests the whole path.
7. Clean GT footage exists for downstream re-measurement.
8. Sustained coverage without systematic multi-minute gaps.

CP-R1→R4 deliver 1, 2, 4, 5. CP-R5→R6 + CP-R13a/b deliver 3. CP-R9 delivers 6 + enables
unattended capture. CP-R7 delivers remaining hardening. CP-R10 delivers 8. CP-R8 delivers 7.
CP-R11 delivers frame-spacing characterization (analysis only, no recorder changes).

---

## CP-R11 -- Definitive frame-spacing characterization

**Status: COMPLETE (2026-08-07).**

Analysis-only checkpoint. 283 source-PTS segments (247K intervals) across 3 days.

**Key findings:**
- Modes come in BLOCKS, not interleaved (supersedes CP-R1b Section 4).
- 15fps cadence is genuine — PPDmUg 1,979 consecutive gap-free frames (supersedes CP-R1b
  Section 5 "structurally undecidable").
- FP7oJQ gaps are periodic (mode=12 spacing) from camera-internal grid mismatch, not
  network loss. Gap rate ~8%, exactly predicted by grid-rate/effective-rate ratio.
- PPDmUg: 0.45% gap rate, 47% of segments gap-free. Not gap-free across all segments.
- Coast-step injection handles gaps (1 step per gap) but cannot represent mode switches.
  Variable dt is the direction (see CP-R12).
- V4 contract's `is_bimodal`, `nominal_dt_s`, and coast recipe work correctly as designed.
  Only the explanatory text about "interleaving" needs updating.

**Evidence:** `docs/evidence/frame_spacing_1/findings.md`
**Tool:** `tools/analyze_frame_spacing.py`

---

## CP-R12 -- Contract prose correction + coast architecture decision

**Status: COMPLETE (2026-08-07).**

Documentation-only checkpoint. No code changes.

- Sidecar contract Section 5 corrected: single cadence + sustained blocks + periodic grid-mismatch
  gaps. "Structurally undecidable" retired with pair-sum identity preserved as historical note.
- Section 6.1 coast guidance revised for the blocked model with measured exposure table.
- Section 10 `gap_flag` rejection rationale updated (local-cadence argument replaces undecidability).
- Coast architecture decision recorded in CLAUDE.md Active Decisions Log: variable dt is the
  direction, subclass-vs-fork scoping is open.
- CP-R8 mid-roll cadence-switch caveat added with practical mitigation (over-capture + select).
- Seven stale `blocked on CP-R5/R6` markers cleared across CLAUDE.md (both complete).

**Evidence:** No new evidence -- consumes CP-R11's findings.

---

## CP-R13a — Encoder timebase fix

**Status:** ✅ Complete
**Evidence:** `docs/evidence/mp4_timing_precision_1/findings.md`

Added `-enc_time_base 1/90000` to the passthrough libx264 encode path. Without it, x264
requantized all PTS onto a uniform 1/15360 grid, destroying the 5940/6030 tick alternation
and producing zero-tick pairs on 30fps blocks. After it, the mp4 carries real capture timing
at the RTP 90000 timebase. 90000 is the RTP timebase for H.264 (RFC 6184, §6.2).

Verified: 5940/6030 alternation present in mp4, 0 disagreements across 299 frame-for-frame
comparisons against the sidecar.

**Scoped to passthrough only.** Under CFR, `-enc_time_base 1/90000` breaks the segment
muxer's cut-point calculation, producing a single unsegmented file (152,933 frames in one
case) instead of `SEG_SECONDS` segments. Captures ran hours past their `WINDOW_SECONDS`
deadline. Found by bisecting the rollback test failure — the CP-R13a smoke test confirmed
"segments produced" but did not check segment count or duration, which would have caught it.
Fixed in `34a9a72` by scoping to passthrough only. CFR resamples to a uniform grid and does
not need PTS precision preserved.

**Lesson:** "segments produced" is not a sufficient rollback assertion. Segment count and
duration are the checks that would have caught this.

---

## CP-R13b — Mp4-derived sidecars (schema 5)

**Status:** ✅ Complete
**Evidence:** `docs/evidence/mp4_timing_precision_1/findings.md`, `docs/evidence/frame_index_join_1/findings.md`

Sidecar frame rows and tick statistics derived from the mp4's PTS rather than reconstructed
from ffmpeg stderr. Row count equals decode count by construction — the boundary attribution
defect (Piece 0: 33% of production-length segments) is eliminated.

Showinfo retained only for `host_arrival_s` and drift, joined by PTS value with delta-pattern
offset detection (k=-10..+10, bidirectional, margin-checked).

Schema 5 changes: `input_n` removed. `row_source` added (`"mp4"` / `"mp4_regenerated"` /
`"showinfo_grid"`). `showinfo_frame_count` and `showinfo_residual` preserve the drop signal.
`mismatch` structurally false. Assertion: row count = decode count, fail loudly.

Regeneration tool: `tools/regenerate_sidecar.py`. Refuses pre-CP-R13a footage (container
timebase check — 1/15360 would produce degraded timing).

**Regression found and fixed (`34a9a72`):** `$mismatch` variable was passed to the CFR awk
but never defined. Under `set -u` (nounset) this silently killed sidecar extraction — same
class as the `local`-outside-function bug (`6112dc0`) that cost the CP-R1 capture.

Verified: `a_eq_c` true on all 6 fresh segments. Smoke test both modes passing.

---

## Explicitly not recorder work (checkpoint 2)

Flagged so it isn't lost. Reframed under TIMING-PRINCIPLE-1: most of these are now
DELETE-CONVERSION sites (read time from sidecar, don't convert). See
`docs/evidence/timing_audit_1/findings.md` §0 for the full taxonomy.

1. **Timing consumption across Stages A–F.** 23 sites enumerated and classified
   (DELETE-CONVERSION / FIX-SCALAR / FORK / DEAD-VESTIGIAL / AUDIT-ONLY) in
   `docs/evidence/timing_audit_1/findings.md` §0. Most conversions are deleted rather than
   corrected — see TIMING-PRINCIPLE-1 in the Active Decisions Log.
2. **Variable-dt Kalman step (FORK)** — the decided direction (Active Decisions Log, "Coast
   architecture" row). Coast-step injection was evaluated (CP-R11) and found insufficient for
   mode switches (cannot inject negative time). Only subclass-vs-fork scoping remains open.
3. **Re-measure drift attribution on clean footage** — after CP-R8.
