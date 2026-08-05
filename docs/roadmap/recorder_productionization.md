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
false teleport.

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

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

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

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

**Evidence basis:** `docs/evidence/recorder_dupfix_1/findings.md`

DUPFIX-2 confirmed boundary misattribution, and the magnitude — roughly nine seconds of
stderr offset at attempt start — is **not explained by encoder buffering**. Every production
sidecar currently has an unreliable per-segment `input_count`, and the mechanism is unknown.

- Replace the stderr log-line-position split with a **PTS-range-derived** boundary. Frames
  belong to a segment by timestamp, not by where a log line landed.
- Re-run the DUPFIX instrument against the old split to quantify what changed.
- Resolve or formally park the ~9s offset mechanism.
- **Note:** `pts_time_s` in the sidecar is segment-relative (base-subtracted from the first
  PTS in the segment's attributed stderr range). CP-R5's boundary fix will shift the time
  origin for affected segments — this is expected, not an anomaly.
- **Corroboration (CP-R2 smoke test, 2026-08-04):** mismatch pattern on PPDmUg passthrough
  reproduces the boundary hypothesis exactly: seg1 ni=357/output=330 (+27), seg2 300/300
  (0), seg3 228/255 (−27). The +27/−27 cancel, middle segment is clean. Strongest
  confirmation yet of the start-of-attempt stderr offset.

Do this **before** CP-R6, so the new contract isn't built on an unexplained parsing bug.

**Done when:** per-segment counts reconcile without position-based attribution; conservation
closes on attempts with no real loss.

---

## CP-R6 — Sidecar contract v2 (the handoff — where the value is realized)

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

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

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

- `N_CAMERAS=0` edge: the `-lt 1` floor at `diag_v6.sh:31` protects the retry interval, but
  the division at line 30 still crashes if `N_CAMERAS=0` is set explicitly. One line.
- Per-camera coverage/uptime metric logged per session — currently assessed by reading
  stderr by hand.
- J_EDEw intermittent offline: characterise, then decide actionable vs accepted.
- Park open anomalies with named status: the 1867 mpdecimate count; the 0.18% PPDmUg
  duplicates on empty-FOV footage.

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

---

## CP-R9 — Smoke harness + caffeinate wiring

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

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

**Status:** 🔲 Not started
**Evidence:** *(link to docs/evidence/... once it exists)*

Depends on CP-R9 (caffeinate fix required for the cheapest test).

**Observation (CP-R1, 2026-08-04):** on healthy cameras, Nest invalidated RTSP sessions
every 1-8 minutes. FP7oJQ and PPDmUg each needed 14 ffmpeg attempts to produce 32 segments,
with 1-3 minutes of retry/backoff between healthy runs. The recorder recovered correctly
every time — this is not a recorder bug — but each cycle is **1-3 minutes of footage not
captured**. For a BJJ gym that is rolls not recorded. During a GT capture it is holes in
footage you are paying to annotate.

**Leading hypothesis:** the host's display slept repeatedly during the capture (user
observation). Display sleep can trigger network-interface power management on macOS. This
fits better than a Nest-side concurrent-session limit — the churn occurred even with
`stop_stream` calls, and the sessions were freshly generated.

**Cheapest test:** run a capture with display sleep disabled (see CP-R9 caffeinate work) and
compare churn rate against the 2026-08-04 baseline of ~14 attempts / 32 segments per camera
over 65 minutes.

**Alternatives if that fails:** SDM concurrent-stream limits; orphaned sessions from earlier
smoke tests; relay-side timeout; network-interface power management independent of display.

**Done when:** churn root cause identified; either resolved or accepted with a stated
coverage budget.

---

## Definition of "production-ready"

The recorder is done when all of these hold:

1. Recorder-injected corruption is **zero**: no duplicates, no CFR decimation.
2. Unavoidable upstream loss is **detected and flagged**, not concealed.
3. True per-frame timing and an authoritative per-clip fps are **emitted in a documented
   contract**.
4. A rebuild **cannot silently** invalidate required ffmpeg options.
5. Correct behaviour is the **default**, with rollback switches retained.
6. A **single command** regression-tests the whole path.
7. Clean GT footage exists for downstream re-measurement.
8. Sustained coverage without systematic multi-minute gaps.

CP-R1→R4 deliver 1, 2, 4, 5. CP-R5→R6 deliver 3. CP-R9 delivers 6 + enables unattended
capture. CP-R7 delivers remaining hardening. CP-R10 delivers 8. CP-R8 delivers 7.

---

## Explicitly not recorder work (checkpoint 2)

Flagged so it isn't lost. Items 3 and 4 are **not currently on the pending list**.

1. **Dynamic fps replaces hardcoded 30** — BoT-SORT `frame_rate`, `speed_mps_k`, Stage E
   windows. Largest single error in the system today; independent of all recorder work.
2. **Consume `gap_flag` in Stage A** — detection is recorder-side, response is pipeline-side.
3. **Coast-step injection** — feed the tracker N detection-free frames across a flagged gap
   so predictions match real elapsed time. Requires no boxmot change. Try this before any
   fork.
4. **boxmot variable-dt (fork or subclass)** — contingent. Open only if measurement shows
   coasting insufficient. Check whether boxmot permits Kalman injection/subclassing first;
   the code change is a few lines, the cost is maintaining divergence from upstream.
5. **Re-measure drift attribution on clean footage** — after CP-R8.
