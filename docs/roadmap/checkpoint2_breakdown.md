# Checkpoint 2 — Work Breakdown

*Created 2026-08-19 (DOC-SYNC-7). Authoritative sequence for timing work.*

**Basis:** commit `bfd5349` (TIMING-PRINCIPLE-1), `docs/evidence/timing_audit_1/findings.md`
§0 + §0.5, `docs/reference/sidecar_contract.md` §6, CLAUDE.md Active Decisions Log
(TIMING-PRINCIPLE-1 + Coast architecture rows).

**This is a sequence, not a set of Task Briefs.** Approve the sequence first; each piece is
briefed individually when it comes up.

Site numbers (#1–#23) refer to the audit's §0 summary table. Fix classes are the §0.5
taxonomy. No piece mixes fix classes.

---

## 0. Three findings that shaped this breakdown

Recorded here because each one changes the sequence, and each should be reflected back into
CLAUDE.md when the relevant piece lands.

### 0.1 The open prerequisite's evidence measures the wrong pair

The Decision Log states the prerequisite as *"`frame_index` must map 1:1 between sidecar rows
and decoded mp4 frames"* and cites FP7oJQ-20260807-102006 at **233 sidecar rows vs
`output_frame_count` 238**. These are different comparisons. Four counts exist:

| | Count | Source | Meaning |
|---|---|---|---|
| (a) | sidecar frame rows | recorder mux loop | rows the recorder emitted |
| (b) | `output_frame_count` | ffprobe `nb_frames` (contract line 72) | container's claimed frame count |
| (c) | decoded frames | `cap.read()` in `FrameIterator` | what Stage A actually indexes |
| (d) | `input_frame_count` | showinfo lines (contract line 71) | recorder input frames |

`mismatch` is defined (contract line 79) as `(d) != (b)`. The join that must hold is
**(a) <-> (c)**. The cited evidence is **(a) vs (b)**.

Consequences:

- ffprobe `nb_frames` is often a container-header value rather than a decode count, so
  (a)!=(b) may be a metadata artifact while (a)=(c) holds — **false alarm**.
- (a)=(b) would not establish (a)=(c) — **false all-clear**.
- Scoping the investigation to `mismatch: true` segments is unsafe, because `mismatch`
  compares (d) to (b) and is not the flag that predicts a join break.

**Piece 0 must decode and count (c) directly.** Comparing metadata to metadata cannot settle it.

### 0.2 `timestamp_ms` is already a threaded contract column — it is the delivery vehicle

`f0_parquet.py:165,192,266,290` declare `timestamp_ms` (int) as a required column on four
artifacts. `f0_validate.py:342,590` enforce it as a join key. `d4_*.py:434` carries it into
`person_tracks`. Stage A is the only stage that decodes video.

Therefore DEL-CONV does **not** mean "every stage reads the sidecar." It means:

> Stage A populates `timestamp_ms` from the sidecar's `pts_time_s`. Downstream stages delete
> their `frame/fps` conversions and read the column that is already present.

This shrinks the work substantially and narrows the partial-application window to a single
hop (A->D) rather than five independent adoptions.

**Precision decision this forces.** `timestamp_ms` is int milliseconds; `pts_time_s` at a
90kHz timebase resolves to ~11us. Rounding to ms is fine for Stage F seek arithmetic and
Stage D velocities (+/-1ms on a 67ms interval = 1.5%). It is marginal for reconstructing
`dt_s` by differencing for a Kalman step. **Recommended:** do not add a float column; have
the fork (Piece 10/11) read `dt_s` from the sidecar directly. Keeps the parquet schema
untouched. Decide in Piece 3.

### 0.3 The planned A/B cannot run on existing GT — CP-R8 is promoted

All GT footage predates the recorder fix and therefore has **no `.timing.jsonl`**. The new
code path requires one. "Old logic vs new logic on the same footage, behind a config flag"
is not executable on the existing GT corpus.

CP-R8 moves from *"complete, gates nothing"* to **gating all outcome validation for
checkpoint 2**. Every piece below can be shipped on Tier 1/Tier 2 evidence; none of the
timing improvement can be scored until clean GT with sidecars exists.

**Validation tiers used throughout:**

| Tier | Method | Needs CP-R8? | What it proves |
|---|---|---|---|
| **T1** | Unit / contract tests; **synthetic-sidecar equivalence** | No | The refactor is behavior-preserving under uniform timing |
| **T2** | Regression on existing GT: runs clean, schema stable, no crash | No | Nothing broke structurally |
| **T3** | `correct_id` A/B on clean GT with real sidecars | **Yes** | The timing change actually helps |

**Synthetic-sidecar equivalence (T1) is the load-bearing pre-CP-R8 technique.** Generate a
constant-`dt` sidecar for an existing GT clip, run the new path, require output to match the
old path within tolerance. This separates "the refactor broke something" from "the timing
change moved the metric" — which is exactly what makes the eventual T3 result interpretable.
Build it once in Piece 2; reuse in every subsequent piece.

### 0.4 The A/B ruler is in better shape than the constraint assumed

`grep` for `fps` / `timestamp_ms` across `gt2actuals/`, `stage_d/`, `common/` returns
nothing. The `correct_id` metric is frame-indexed and time-free. Site #19's defect is
confined to `stage_f/visualize.py` (preview overlay timestamps, `VideoWriter` rate).

**#19 does not gate the numeric A/B.** It gates *qualitative* review — a human watching a
preview to judge a timing change would read wrong timestamps. Sequenced accordingly
(Piece 9), but it must precede the first human review of A/B output.

---

## 1. The sequence

Legend — **Ships:** `alone` = independently shippable; `with N` = must land in the same
commit/PR as piece N.

### Piece 0 — Resolve the `frame_index` join prerequisite
**Class:** MEASUREMENT (not a fix) | **Ships:** alone | **Blocks:** 3,4,5,6,10,11
**Status:** COMPLETE

**Scope.** Determine empirically whether sidecar row `frame_index` maps 1:1 to
`FrameIterator`'s decoded frame index. Decode with the production `FrameIterator`, count
frames (c), compare against sidecar row count (a), `output_frame_count` (b), and
`input_frame_count` (d). Report all four per segment.

Sample: >=20 segments spanning both cameras; deliberately include FP7oJQ-20260807-102006 (the
233/238 case), several `mismatch: false` segments, at least one `is_bimodal: true` segment,
and segments with high gap density. Do not scope to `mismatch: true` — see section 0.1.

Also settles the audit's open C2 question: whether `CAP_PROP_POS_MSEC` deltas quantized at
exactly one frame interval are an index-alignment artifact rather than an OpenCV timestamp
defect. Compare `POS_MSEC` per decoded frame against sidecar `pts_time_s` for the same
`frame_index`.

**Done.** An evidence document stating, per segment, all four counts; a verdict on whether
(a)<->(c) is 1:1; if not, the failure pattern and a proposed alternative join key; a verdict
on C2; a recommendation on whether `output_frame_count` is trustworthy for #16.

**Validation.** T1 — self-validating measurement. No GT needed.

Evidence: `docs/evidence/frame_index_join_1/findings.md`.

---

### Piece 1 — Remove dead and vestigial fps carriers
**Class:** DEAD-VESTIGIAL | **Sites:** #11, #21, #22, #23 | **Ships:** alone | **Blocks:** none
**Status:** COMPLETE

**Scope.** Delete the Stage E `frame_index / fps * 1000` timestamp fallback (#11, verified
dead — 0 nulls in production); remove the unused `fps` parameter from `compute_pair_distances`
(#21) and `buzzer` (#22); remove the never-read `clip_fps` field from the sweep cache summary
(#23). Update call sites.

**Done.** Sites removed; no caller passes a now-absent parameter; existing tests pass.

**Validation.** T1 + T2. No GT needed, no behavior change expected.

**Why here.** Zero-risk, independent of the join question, and it shrinks the surface every
later piece has to reason about.

---

### Piece 2 — Sidecar reader module + synthetic-sidecar test harness
**Class:** foundation (no behavior change) | **Ships:** alone | **Blocks:** 3,6,7,8
**Status:** COMPLETE

**Scope.** A contracts-layer module that locates a segment's `.timing.jsonl`, parses `_meta`
and frame rows, enforces the validity model (**omission means invalid**; gate on
`source_pts`; `dt_s` requires `passthrough` AND `source_pts: true`), and exposes per-frame
`pts_time_s` / `dt_s` plus the derived scalar `1.0 / nominal_dt_s`. Must refuse to serve
timing fields when their gate is absent rather than substituting a default.

Also: confirm reachability on all three paths (production, `run_local.sh` processor, sweep) —
the audit called these reachable; verify on the real machine.

Also: build the **synthetic-sidecar generator** (section 0.3) — given a clip and a constant dt,
emit a schema-v5-conformant sidecar. This is the T1 harness for every later piece.

**Done.** Module + unit tests covering: valid passthrough, `source_pts: false`, missing
sidecar, `is_bimodal: true`, drift fields absent below `n_drift_windows >= 4`. Generator
produces a sidecar the reader accepts. **No production consumer wired.** No pipeline
behavior changes.

**Validation.** T1. No GT needed.

**Note.** Do not use `measured_fps` for the scalar — contract section 6.2 and the Decision Log both
specify `1.0 / nominal_dt_s`. Do not read `input_n` (deprecated).

---

### Piece 3 — Stage A becomes the timing source of truth
**Class:** DEL-CONV | **Ships:** **with Piece 4** | **Depends:** 0, 2 | **Blocks:** 4,5,6,10,11
**Status:** NOT STARTED

**Scope.** Populate `timestamp_ms` in Stage A outputs from sidecar `pts_time_s` instead of
`CAP_PROP_POS_MSEC` / fps fallback. Behind a config flag with the old path retained. Resolve
the int-ms precision decision from section 0.2 and record it.

**Done.** `timestamp_ms` provably sourced from the sidecar when one is present and valid;
old path preserved under the flag; flag documented in CLAUDE.md; **the value verified to
arrive in the written parquet**, not merely at the call site (the checkpoint-1 failure mode —
correct in the edited file, dropped by the carrying path).

**Validation.** T1 (synthetic-sidecar equivalence: constant-dt sidecar must reproduce old
`timestamp_ms` within tolerance) + T2. T3 blocked on CP-R8.

**Why it ships with Piece 4.** Stage A emitting real time while Stage D still computes
`df / fps` puts two disagreeing time bases in one pipeline — a *new* inconsistency, not a
partial improvement. The hop must close in one landing.

---

### Piece 4 — Stage D per-clip kinematics read time
**Class:** DEL-CONV | **Sites:** #5, #6, #7 | **Ships:** **with Piece 3** | **Depends:** 0, 2, 3
**Status:** NOT STARTED

**Scope.** Delete the `dt_s = frames / fps` conversions in `d0_bank.py:571`,
`costs.py:413`, `d1_graph_build.py:1408`. Read `timestamp_ms` from the parquet instead.
D0.5 inherits automatically (indirect dependency, no separate change).

Also absorbs session-level alignment: replace `derive_clip_frame_offset`'s
`round(delta_sec * fps)` (#9 `session_d_run.py:207-221`, #10 `session_f_run.py:88`) with
alignment on `pts_time_s` + `pts_wallclock_offset_s`. Site #1 (`session_d_run.py:491`)
dissolves once #9 and #8 stop requesting a session-wide scalar.

**Piece 4 must read and log `showinfo_offset_status` per clip.** The sidecar anchor
(`pts_wallclock_offset_s`) derives from `host_arrival_s`, which comes from the showinfo
join — the layer whose boundary attribution CP-R13b routed around rather than fixed. The
contract's own worked example (line 345) shows
`showinfo_offset_status: "ambiguous_fallback_k0"`. Logging the status per clip makes it
possible to tell whether the anchor was confident or a fallback if session stitching later
looks wrong. Cheap now, expensive to reconstruct later.

**Done.** No `/ fps` remains in the three sites; velocities derived from real per-frame time;
D0.5 speed/accel inputs verified to change consistently. Session alignment uses
`pts_wallclock_offset_s`. `showinfo_offset_status` logged per clip.

**Validation.** T1 (equivalence under synthetic constant-dt sidecar) + T2. T3 blocked.

---

### Piece 5 — Cross-camera timing
**Class:** DEL-CONV | **Site:** #8 | **Ships:** alone (after 3+4) | **Depends:** 0, 3, 4
**Status:** NOT STARTED

**Scope.** Replace `temporal_window_s * fps` -> frame-count comparison (#8
`cross_camera_evidence.py:275`) with direct time comparison. Confirm session aggregation no
longer applies one scalar across cameras that measure differently (13.85 vs 15.00 in the
same session).

Cross-camera sync accuracy is +/-14-56ms per the contract; record the residual.

**Done.** #8 reads time directly; cross-camera sync verified on the 3-camera session corpus.

**Validation.** T1 + T2 on the 3-camera session corpus (35/36 clips). T3 blocked.

**Parked idea (Tier 2).** A gym buzzer is a genuinely simultaneous physical event across all
cameras, making it a natural cross-camera sync anchor. Not wired that way today —
`buzzer.py` is a Stage E soft gate that snaps engagement end frames within a single clip,
downstream of D and unrelated to cross-camera identity. Candidate signal if
`pts_wallclock_offset_s` accuracy proves insufficient. Do not implement or schedule.

---

### Piece 6 — Stage F export timing (customer-visible)
**Class:** DEL-CONV | **Sites:** #2, #3, #16 | **Ships:** alone | **Depends:** 0, 2, 3
**Status:** NOT STARTED

**Scope.** Replace `start_sec = start_frame / fps` (#2 `ffmpeg.py:121-122`) — `pts_time_s`
for that frame *is* the seek time. Same for `manifest.py:60-68` (#3), whose values persist to
the Supabase `clips` table (`numeric` columns). #16 (`_infer_last_frame`) only if Piece 0
cleared `output_frame_count`; otherwise deferred and explicitly noted as still PROVISIONAL.

**Done.** Exported clip boundaries verified against a known match interval on FP7oJQ (the 8%
gap camera, where the error accumulates to ~5.7s by frame 1000 and ~10.3s by frame 1800);
persisted `clips` values consistent with the exported media.

**Validation.** T1 + T2 + **direct media inspection** — this is the one piece whose
correctness is checkable without GT: seek to the computed time in the exported file and
confirm the match is there. Do this on FP7oJQ specifically.

**Priority note.** Highest customer-visible severity in the audit. Independent of the Stage D
work — it can ship before Piece 5 if that is preferred.

---

### Piece 7 — Stage F CFR output scalar
**Class:** FIX-SCALAR | **Sites:** #12, #14, #15 | **Ships:** alone | **Depends:** 2
**Status:** NOT STARTED

**Scope.** The documented exception: `cv2.VideoWriter` and the ffmpeg CFR re-encode require a
scalar; athletes never receive VFR. Supply `1.0 / nominal_dt_s` from the sidecar. Remove the
hardcoded `30.0` fallbacks (#12 `session_f_run.py:397`, #13 `multiplex_runner.py:406`) and
reconcile #15 (`run.py:331`) — Stage F currently **re-probes fps from `video_meta` rather
than reading the manifest**, an independent fps source that would survive a manifest-only fix.

**Done.** One resolution path for the output scalar; no hardcoded 30.0 remains; #15's
divergence closed or documented as intentional.

**Validation.** T1 + T2 + media inspection (playback rate correct, duration correct).

---

### Piece 8 — ~~BoT-SORT `frame_rate` scalar~~ DISSOLVED INTO PIECE 11
**Status:** DISSOLVED

Piece 8's scope (fix `frame_rate` scalar) is fully absorbed by Piece 11 (variable-dt
tracker). Doing Piece 8 alone would shorten effective track lifespan from 2.0s to 1.0s,
which the OFAT screen associates with more fragmentation — a metric movement in the wrong
direction that Piece 11 would then partly undo. One change, one measurement.

---

### Piece 9 — Fix the A/B instrument
**Class:** DEL-CONV (timestamp) + FIX-SCALAR (writer) | **Site:** #19 | **Ships:** alone | **Depends:** 2
**Status:** NOT STARTED

**Scope.** `visualize.py:408` — `timestamp_ms = fi * (1000/cap_fps)` -> read time directly.
`:327,351` — `VideoWriter` scalar from the sidecar.

**Done.** Preview overlay timestamps match sidecar `pts_time_s`; preview playback rate correct.

**Validation.** T1 + visual inspection.

**Placement rationale (section 0.4).** The numeric `correct_id` metric is frame-indexed and
fps-free — confirmed, `gt2actuals/`, `stage_d/`, `common/` contain no fps or timestamp
references. So this does **not** gate the numeric A/B. It gates *human* review of preview
video. Must land before anyone eyeballs A/B output to judge a timing change; need not land
before the numbers are computed.

---

### Piece 10 — boxmot subclass-vs-fork scoping
**Class:** FORK (scoping only) | **Site:** #20 | **Ships:** alone | **Depends:** 0
**Status:** NOT STARTED

**Scope.** Determine whether `boxmot==16.0.8` permits subclassing `BotSort` or injecting the
`KalmanFilterXYWH` instance, versus requiring a hard fork. Read the installed source. Produce
a recommendation with the maintenance cost of each option.

**Done.** A written recommendation: subclass / inject / fork, with the specific extension
point identified and the divergence-maintenance burden estimated.

**Validation.** T1 — a scoping document, no code.

**Note.** Explicitly open in the Decision Log. Cheap, and it de-risks Piece 11. Can run in
parallel with Pieces 5-9.

---

### Piece 11 — Variable-dt Kalman step (absorbs Piece 8)
**Class:** FORK (implementation) | **Sites:** #4, #20 | **Ships:** alone | **Depends:** 0, 2
**Status:** COMPLETE

**Scope.** Subclass (not fork) boxmot's `KalmanFilterXYWH` and `BotSort`. The KF subclass
rebuilds `_motion_mat` per step from a dt *ratio* (`dt_s / nominal_dt_s`), keeping velocity
in pixels-per-nominal-frame (all noise constants remain calibrated). The BotSort subclass
replaces both KF sites (`self.kalman_filter` and `STrack.shared_kalman`) and overrides
`_update_track_states` with wall-time `max_lost_seconds` (default 2.0 = today's behavior),
eliminating `frame_rate` entirely. Piece 8 dissolved into this piece.

Module: `src/bjj_pipeline/tracking/` (subclasses, not vendored copy).
Toggle: `stages.stage_A.tracker.variable_dt: true/false` (default false).
Config: `stages.stage_A.tracker.max_lost_seconds: 2.0`.

**Done.** T1 PASS (constant-cadence segment matches stock bit-for-bit, 500 comparisons, 0
mismatches). Bimodal segment (200 frames, dt ratios 0.0–2.0) runs without error. Runtime
assertion proves `STrack.shared_kalman` is the subclassed filter. `dt_s=0.0` (same-PTS
frames on bimodal segments) handled as ratio 0.0 → Kalman position no-op.

**Note.** CP-R8 bimodal exposure (3/11 segments) is higher than CP-R11's 1/139 — the pre-fix
requantization was erasing the signal. Variable dt is more urgent than the original exposure
figures suggested.

---

### CP-R8 — Clean GT capture and annotation
**Class:** manual | **Ships:** alone | **Blocks:** all T3 validation
**Status:** NOT STARTED — promoted to critical path

**Scope.** Manual capture on the fixed recorder + CVAT annotation. Unchanged in content from
the existing backlog item — **changed in priority**.

**Done.** Clean GT clips with valid sidecars, annotated, registered in the eval corpus.

**Why promoted (section 0.3).** No existing GT clip has a sidecar, so the new code path cannot
run on any of them. Until this lands, every piece above is validated for *correctness of
refactor* and none for *benefit*. It should start as early as scheduling allows — ideally in
parallel with Pieces 1-2, since it is manual work that blocks nothing upstream.

---

## 2. Landing groups

| Group | Pieces | Constraint |
|---|---|---|
| **Parallel-safe, start now** | 0, 1, 2, CP-R8 | Independent of each other and of the join verdict |
| **Gated on join verdict** | 3+4 (together), 5, 6, 11 | Re-plan required if (a)<->(c) is not 1:1 |
| **Atomic pair** | 3 + 4 | Must land in one commit — see Piece 3 rationale |
| **Independent scalars** | 7, ~~8~~, 9 | Depend only on Piece 2 (8 dissolved into 11) |
| **Scoping** | 10 | Parallel with 5-9 |

**Ordering recommendation:** 0 || 1 || 2 || CP-R8 -> (3+4) -> 6 -> 5 -> 7 -> 9 -> 10 -> 11.
(Piece 8 dissolved into 11.)

Piece 6 before Piece 5 because it is the customer-visible defect and is independent of the
session work. Piece 8 late because it is the only piece expected to move `correct_id` in a
confusing direction.

---

## 3. Open items to reflect back into CLAUDE.md

*Each of these should be recorded when the relevant piece lands, not before.*

1. **Restate the TIMING-PRINCIPLE-1 prerequisite** in terms of (a)<->(c), and drop the
   `mismatch: true` scoping. The current wording cites evidence that measures a different
   pair (section 0.1).
2. **Record CP-R8's promotion** from "gates nothing" to "gates all checkpoint-2 outcome
   validation" (section 0.3).
3. **Record that `timestamp_ms` is the DEL-CONV delivery vehicle** (section 0.2) — downstream
   stages read the parquet column, not the sidecar.
4. **Record the int-ms precision decision** once Piece 3 settles it.
5. **Record that the numeric eval metric is fps-free** (section 0.4), so #19's placement is
   not revisited later on a false assumption.

---

## 4. Piece 4/5 re-cut rationale (DOC-SYNC-7)

*Added 2026-08-19. Records the decision that reshaped Pieces 4 and 5.*

The original Piece 4 was "Stage D per-clip kinematics read time" (sites #5, #6, #7). The
original Piece 5 was "session alignment and cross-camera timing" (sites #1, #8, #9, #10).
This conflated two problems: putting clips on one timeline (session alignment) and
reconciling two cameras (cross-camera sync).

Under the sidecar-required decision (see CLAUDE.md Active Decisions Log), session alignment
anchors on `pts_wallclock_offset_s`, which requires a valid schema-5 sidecar. The filename-
anchor fallback was rejected: `parse_clip_timestamp` has 1-second resolution, and the most
common cross-clip stitch — a person crossing a segment cut — has a real gap of ~67ms. A 1s
anchor overestimates that by ~15x, which would reject valid reconnects via `dt_max_s`.

Re-cut:

- **Piece 4** absorbs session alignment (sites #5, #6, #7, #9, #10). Site #1 dissolves here
  as a DEL-CONV consequent. "Put clips on one timeline" is one problem.
- **Piece 5** becomes purely cross-camera: site #8 and whatever Tier 2 enabling requires.
  "Reconcile two cameras" is a different problem.

The `showinfo_offset_status` logging requirement belongs to Piece 4 because that is where
the anchor is consumed. The buzzer cross-camera sync idea is parked against Piece 5 / Tier 2.
