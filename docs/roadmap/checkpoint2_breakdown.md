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

Therefore DEL-CONV does **not** mean "every stage reads the sidecar." The delivery
architecture is:

> **Read the parquet for per-frame time; read the sidecar for segment-level timeline
> facts.** Stage A is the only stage that decodes video, so `timestamp_ms` (via
> `POS_MSEC`, proven equivalent to sidecar `pts_time_s` by Piece 0b) carries per-frame
> capture time to every downstream consumer. Segment-scoped facts with no parquet
> carrier — `attempt`, `pts_wallclock_offset_s`, `showinfo_offset_status`,
> `nominal_dt_s`, per-frame `dt_s` — are read from the sidecar directly. Piece 4 does
> both.

This shrinks the work substantially: downstream stages delete their `frame/fps`
conversions and read the `timestamp_ms` column already present in the parquet.

**Precision decision this forces.** `timestamp_ms` is int milliseconds; `pts_time_s` at a
90kHz timebase resolves to ~11us. Rounding to ms is fine for Stage F seek arithmetic and
Stage D velocities (+/-1ms on a 67ms interval = 1.5%). It is marginal for reconstructing
`dt_s` by differencing for a Kalman step. **Recommended:** do not add a float column; have
the fork (Piece 11) reads `dt_s` from the sidecar directly. Keeps the parquet schema
untouched. **Decision closed (Piece 0b, §10 precision finding):** all `pts_time_s` values
land on exact integer milliseconds (5940/90=66, 6030/90=67), so `int(timestamp_ms)`
rounding is lossless on post-R13a footage. No float column needed; Piece 11 reads `dt_s`
from the sidecar as recommended.

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
**Class:** MEASUREMENT (not a fix) | **Ships:** alone | **Blocks:** 4,5,6,10,11
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
**Class:** foundation (no behavior change) | **Ships:** alone | **Blocks:** 6,7,8
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

### Piece 3 — ~~Stage A becomes the timing source of truth~~ DISSOLVED (Piece 0b)
**Status:** DISSOLVED

Piece 3's scope was "populate `timestamp_ms` from sidecar `pts_time_s` instead of
`CAP_PROP_POS_MSEC`." Piece 0b (§10 A2) proved this unnecessary: `CAP_PROP_POS_MSEC`
matches sidecar `pts_time_s` with **0.000ms** max deviation on post-R13a footage
(FP7oJQ 133817, PPDmUg 133829), with the 5940/6030 tick alternation fully visible as
distinct 66/67ms deltas. `FrameIterator.timestamp_ms` already provides exact capture
PTS — no sidecar read needed for per-frame timestamps.

The int-ms precision decision this piece was to settle is also closed (§10 precision
finding): all `pts_time_s` values land on exact integer milliseconds, so
`int(timestamp_ms)` rounding is lossless. No float column needed.

**Verified in live code:** `frame_iterator.py:57` reads `POS_MSEC`; `load_sidecar` is
called only from `multiplex_runner.py:530` (under `variable_dt`) and
`tools/sweep/replay_tracker.py:75`. Piece 3 was never implemented and does not need
to be.

Evidence: `docs/evidence/frame_index_join_1/findings.md` §10 A2 and precision finding.

---

### Piece 4 — Stage D reads real time (clip + session)
**Class:** DEL-CONV | **Sites:** #5, #6, #7, #9, #10 | **Ships:** alone | **Depends:** 0, 2
**Status:** COMPLETE — CP4.A–F all done. Site #1 reduced to one consumer (#8, Piece 5).

**Scope.** Delete the `dt_s = frames / fps` conversions in `d0_bank.py:571`,
`costs.py:413`, `d1_graph_build.py:1408` (#5, #6, #7). Read `timestamp_ms` from the
parquet instead. D0.5 inherits automatically (indirect dependency, no separate change).

Also absorbs session-level alignment (DOC-SYNC-7 re-cut, see §4): replace
`derive_clip_frame_offset`'s `round(delta_sec * fps)` (#9 `session_d_run.py:207-221`,
#10 `session_f_run.py:88`) with alignment on `pts_time_s` + `pts_wallclock_offset_s`.
Site #1 (`session_d_run.py:491`) dissolves once #9 and #8 stop requesting a session-wide
scalar.

**Piece 4 must read and log `showinfo_offset_status` per clip.** The sidecar anchor
(`pts_wallclock_offset_s`) derives from `host_arrival_s`, which comes from the showinfo
join — the layer whose boundary attribution CP-R13b routed around rather than fixed. The
contract's own worked example (line 345) shows
`showinfo_offset_status: "ambiguous_fallback_k0"`. Logging the status per clip makes it
possible to tell whether the anchor was confident or a fallback if session stitching later
looks wrong. Cheap now, expensive to reconstruct later.

**Piece 4 must treat `attempt` changes as hard breaks (RECORDER-COVERAGE-2).** Session
aggregation must treat an `attempt` change between consecutive clips as a discontinuity of
unknown duration: no reconnect edges across it, no cross-clip tracklet joins. Without this
the pipeline stitches tracklets across a genuine teleport in space and time.
`f0_sidecar.py:231` exposes `attempt`; nothing in `src/bjj_pipeline/stages/` reads it yet.
PPDmUg Aug 23 (5 segments across 5 different attempts, 183s content in 7,243s wall)
demonstrates that wall-clock spacing alone cannot distinguish delivery lag (within an
attempt) from genuine discontinuity (between attempts).

**Done.** No `/ fps` remains in the three sites; velocities derived from real per-frame time;
D0.5 speed/accel inputs verified to change consistently. Session alignment uses
`pts_wallclock_offset_s`. `showinfo_offset_status` logged per clip.

**Validation.** T1 (equivalence under synthetic constant-dt sidecar) + T2. T3 blocked.

### Approved checkpoint sequence (2026-08-24)

| CP | Scope | Depends | Ships |
|---|---|---|---|
| CP4.A | Sidecar ingest gate + per-clip timing record; frame_iterator fallback fixed | — | alone | **DONE** |
| CP4.B | D0 kinematics read real time (site #5) | A | alone | **DONE** |
| CP4.C | Session timeline: anchored offsets + session-relative `timestamp_ms` (#9, #10); log `showinfo_offset_status` | A | **with D** | **DONE** |
| CP4.D | D1/D2 read real time (#6, #7) | C | **with C** | **DONE** |
| CP4.E | Clip-boundary discontinuity handling (shortfall + attempt OR) | C, D | alone | **DONE** |
| CP4.F | Retire the session fps scalar (#1) to one documented consumer | C, D, E | alone | **DONE** |

**Why C and D are atomic.** `aggregate_session_bank` (`session_d_run.py:302-330`)
offsets `frame_index`, `start_frame`, and `end_frame` but **not** `timestamp_ms`. D
without C reads clip-relative `timestamp_ms` against globally-offset `frame_index`
and produces a small or negative cross-clip `dt_s` with both endpoints present and
nothing raised — the silently-plausible-wrong-number failure class.

**Validation ceiling for CP4.C and CP4.E.** Both anchor on `pts_wallclock_offset_s`,
for which legacy footage has no real value — the synthetic generator fabricates it.
T2 on the 3-camera session corpus is a **smoke test** (the mechanism runs, output is
plausible); it does **not** validate that the session timeline is correct. **T1 on a
synthetic two-clip session with a known offset is the sole correctness evidence for
session alignment until CP-R8 footage exists.** This is a deliberate choice: the
alternative — deriving fixture offsets from filename timestamps — was considered and
rejected to keep the rejected production anchor out of the test path entirely.

---

### Piece 5 — Cross-camera timing
**Class:** DEL-CONV | **Site:** #8 | **Ships:** alone (after 4) | **Depends:** 0, 4
**Status:** NOT STARTED

**Scope.** Replace `temporal_window_s * fps` -> frame-count comparison (#8
`cross_camera_evidence.py:275`) with direct time comparison. Confirm session aggregation no
longer applies one scalar across cameras that measure differently (13.85 vs 15.00 in the
same session).

Cross-camera sync accuracy is +/-14-56ms per the contract; record the residual.

**Open question (RECORDER-COVERAGE-2):** `pts_wallclock_offset_s` derives from
`host_arrival_s`. Under sub-real-time delivery, arrival lags capture by the accumulated
delivery delay, and that delay grows through a run. Two cameras at different delivery rates
(observed: FP7oJQ 0.94× and PPDmUg 0.25× on Aug 22) would diverge by minutes. The
contract's ±14–56ms accuracy figure (CAPTURE-TIME-2) predates this observation. **Verify
before planning Piece 5.** This may partly explain historically weak cross-camera evidence.

**Done.** #8 reads time directly; cross-camera sync verified on the 3-camera session corpus.

**Validation.** T1 + T2 on the 3-camera session corpus (35/36 clips). T3 blocked.

**Parked idea (Tier 2).** A gym buzzer is a genuinely simultaneous physical event across all
cameras, making it a natural cross-camera sync anchor. Not wired that way today —
`buzzer.py` is a Stage E soft gate that snaps engagement end frames within a single clip,
downstream of D and unrelated to cross-camera identity. Candidate signal if
`pts_wallclock_offset_s` accuracy proves insufficient. Do not implement or schedule.

---

### Piece 6 — Stage F export timing (customer-visible)
**Class:** DEL-CONV | **Sites:** #2, #3, #16 | **Ships:** alone | **Depends:** 0, 2
**Status:** COMPLETE

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

### Piece 7 — Stage F output format (Shape 3 hybrid)
**Class:** FIX-SCALAR + VFR | **Sites:** #12, #14, #15 | **Ships:** alone | **Depends:** 2
**Status:** COMPLETE

**Shape 3 (hybrid).** Plain path: VFR-preserving re-encode via `-fps_mode passthrough`
and `-enc_time_base -1` (verified on ffmpeg 7.1.1 with crop). Redacted path: CFR at
`1.0 / nominal_dt_s` from the sidecar (cv2.VideoWriter hard constraint — no per-frame
timestamp API). Two paths produce different timing characteristics (VFR h264 vs CFR mpeg4),
explicitly documented.

**Sites closed:**
- **#12** (`session_f_run.py`): DELETED — consumer chain was dead (`SourceClipInfo.fps`
  never read). `SourceClipInfo.fps` field also deleted.
- **#14** (`redact.py` VideoWriter rate): fixed — receives `nominal_fps` from sidecar.
- **#15** (`run.py` independent probe-derived fps): closed — `probe_video_metadata` fps
  extraction deleted, sidecar `nominal_fps` is sole source. Probe retained for width/height.
- **#13** (`multiplex_runner.py` 30.0 fallback): DEFERRED to Piece 9 — consumer is debug
  viz (`MuxVisualizer`) and `manifest.fps` backfill. Same class as `visualize.py:408` and
  `post_pipeline_annotator.py:217`.

**buffer_frames:** `consolidate_buffer_sec * nominal_fps` (was `* probed fps`).

**CFR divergence (redacted path, quantified):** source is VFR at avg ~14.582fps; redacted
output is CFR at nominal 14.925fps. Over 60s of content, redacted clip is 1.37s shorter
(~2.3%). Scales linearly (~2.74s over 120s). Known, deliberate — forced by cv2.VideoWriter.
Piece 12 removes it by replacing VideoWriter with ffmpeg piped output.

**GOP snap:** unchanged. ≤2.0s residual from source camera GOP. Backlog item.

**Validation:** T2 suite green (196 passed). Media inspection: plain path VFR confirmed
(`r != avg`, h264); redacted path CFR confirmed (`r == avg` at nominal_fps, mpeg4).
Session path NOT TESTABLE (CP22 blocks session Stage E).

Evidence: `docs/evidence/piece7_results/findings.md`.

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

### Piece 10 — ~~boxmot subclass-vs-fork scoping~~ RESOLVED BY PIECE 11
**Status:** RESOLVED

Piece 10's question — subclass, inject, or fork? — was answered by Piece 11 doing it.
The decision is **subclass, not fork**: `VariableDtBotSort(BotSort)` and
`VariableDtKalmanFilterXYWH(KalmanFilterXYWH)` at `src/bjj_pipeline/tracking/`.
Extension points (V1–V8) are documented in the class docstrings of both subclasses,
which is where someone doing a boxmot version bump will be standing.

The formal maintenance-cost analysis called for in the original deliverable was
deliberately NOT written. The V1–V8 dependency list and the "re-verify on version
bump" caveat live in the code that depends on them; a standalone document would
duplicate that in a place nobody reads at the moment of risk. The subclass is shipped
and working (T1+T2 PASS).

**Known trap for version bumps:** boxmot `__version__` lags the package version by one —
the 16.0.8 wheel reports `__version__='16.0.7'`. trackers/ and motion/ are
byte-identical between 16.0.7 and 16.0.8. Anyone verifying a bump against the V1–V8
list must check the wheel version, not `__version__`.

---

### Piece 11 — Variable-dt Kalman step (absorbs Piece 8)
**Class:** FORK (implementation) | **Sites:** #4, #20 | **Ships:** alone | **Depends:** 0, 2
**Status:** IMPLEMENTED — T1+T2 PASS, blocked on muxer PTS fix for full corpus

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
mismatches). T2 PASS: A→F on low-dispersion 202832 (1710 frames, 4.3% dispersion, 14
exports); A→D on high-dispersion 201606 (1950 frames, 29.8% dispersion, 9 persons); Stage E
failure on 201606 is pre-existing CP22 NAType (confirmed: fails identically with
variable_dt=false).

**Blocked:** Duplicate-PTS muxer artifact at frame index 2 on attempt-first segments
(RECORDER-MUXER-PTS-1). `dt_s=0.0` → raises under variable_dt=true. Second reproduction
(Aug 23): FP7oJQ 3/17 segments affected (attempt-first only, attempts >1 deterministic);
PPDmUg 5/5 (every segment is attempt-first). **Decision: fix the muxer and re-capture
affected segments.** Do not annotate affected segments until fixed.

**Note.** CP-R8 bimodal exposure (3/11 segments) is higher than CP-R11's 1/139 — the pre-fix
requantization was erasing the signal. Variable dt is more urgent than the original exposure
figures suggested.

**T3 sanity check:** effect size should increase with per-segment dt dispersion (fraction
of frames with `|dt_s/nominal_dt_s - 1| > 0.25`). A flat relationship across dispersion
indicates a wiring fault, regardless of which direction the metric moved. **Do not group by
`is_bimodal`** — the flag is advisory, cannot fire when the majority mode is the short one,
and does not track dispersion (the most dispersed CP-R8 segment, 202148 at 48.3%, is flagged
`is_bimodal=False`). See `docs/evidence/timing_dispersion_1/findings.md`.

---

### CP-R8 — Clean GT capture and annotation
**Class:** manual | **Ships:** alone | **Blocks:** all T3 validation
**Status:** CAPTURED — CP-R8 (11 segments, 9 clean, 2 blocked on MUXER-PTS-1) + Aug 23 (17 segments, 14 clean, 3 blocked on MUXER-PTS-1)

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
| **Complete** | 0, 1, 2, 4, 6, 7, 11 | 0-2, 4, 6 implemented. 7: Shape 3 hybrid (VFR plain, CFR redacted). 11: T1+T2 PASS, T3 pending annotation. |
| **Not started** | 5, 9 | 5: cross-camera (#8). 9: debug/eval viz fps scalars (#13, #19, post_pipeline_annotator). |
| **Dissolved** | 3, 8, 10 | 3: dissolved by Piece 0b (POS_MSEC = sidecar PTS). 8: absorbed into Piece 11. 10: resolved by Piece 11 (scoping answered by implementation). |

**Ordering:** superseded by the six-objective execution plan in §5. The original sequence
(`0 -> 1 -> 2 -> (3+4) -> 6 -> 5 -> 7 -> 9 -> 10 -> 11`) was written before the recorder
coverage gap was observed and before Piece 11 was implemented. §5 is the current plan.

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
4. **Record the int-ms precision decision** — closed (Piece 0b, §10 precision finding).
   All `pts_time_s` values land on exact integer milliseconds; `int(timestamp_ms)` rounding
   is lossless. Piece 11 reads `dt_s` from the sidecar directly. No float column added.
5. **Record that the numeric eval metric is fps-free** (section 0.4), so #19's placement is
   not revisited later on a false assumption.

---

## 4. Piece 4/5 re-cut rationale (DOC-SYNC-7)

*Added 2026-08-19. Decision record — operative boundaries are in the piece bodies above.*

The original split conflated two problems: putting clips on one timeline (session alignment)
and reconciling two cameras (cross-camera sync). The re-cut separated them:

- **Piece 4** absorbed session alignment (sites #5, #6, #7, #9, #10) — "put clips on one
  timeline." The original 3+4 atomic-pair constraint is moot — Piece 3 dissolved (Piece 0b
  proved `POS_MSEC` delivers exact capture PTS, making sidecar-sourced `timestamp_ms`
  unnecessary). Piece 4 depends on Pieces 0 and 2 only.
- **Piece 5** became purely cross-camera: site #8 and Tier 2 enabling.

The filename-anchor fallback for session alignment was rejected: `parse_clip_timestamp` has
1-second resolution vs ~67ms real gap at segment cuts (15x overestimate, rejects valid
reconnects). Session alignment anchors on `pts_wallclock_offset_s` instead.

---

## 5. Six-objective execution plan (DOC-SYNC-8)

*Added 2026-08-22. Supersedes the §2 ordering recommendation. Pieces 0, 1, 2, 11 are
complete (implementation); this plan covers the remaining work.*

| # | Objective | Type | Pieces | Blocks / blocked by |
|---|-----------|------|--------|---------------------|
| 1 | **Recorder coverage investigation** — delivery rate model corrected, fix validated (RECORDER-COVERAGE-2). Delivery ~1.0× steady state; "inter-segment gaps" were delivery lag (within attempt) and genuine discontinuities (between attempts). BACKLOG-1 validated at 1,800s scale. Sustained sub-real-time delivery (Wednesday 0.53×) cause still open. | Investigation | — | Model corrected, fix validated. Remaining recorder work: MUXER-PTS-1 fix (objective 2). Camera fleet health blocks multi-camera GT independently. |
| 2 | **RECORDER-MUXER-PTS-1 fix** — duplicate PTS at frame index 2, first segment of each attempt | Small fix | — | Fold into objective 1's recorder work; must land before next capture |
| 3 | **Pieces 4 + 6** — Stage D reads real time (clip + session) + Stage F export timing | Build | 4, 6 | **Independent of all recorder work; can run in parallel with objectives 1-2** |
| 4 | **Player VFR test → Stage F format decision** | Test, then decide | 7 (gated) | Test first: if Flutter player handles VFR, Stage F never converts and Piece 7 dissolves |
| 5 | **CP22 NAType Stage E fix** | Small fix | — | Needed before annotation — blocks Stage E on 201606 (high-dispersion clip) |
| 6 | **Annotate** | Manual | CP-R8 | After the recorder is fixed and a clean session is captured |

### Ordering rationale

**Annotation is last** because MUXER-PTS-1 must be fixed before the next capture, and the
camera fleet health issue (J_EDEw offline, PPDmUg flickering) blocks multi-camera GT. The
RECORDER-COVERAGE-1 "44% coverage" symptom is handled by the BACKLOG-1 fix (it keeps
pulling until content arrives), but the cause of sustained sub-real-time delivery (Wednesday
0.53×) is still unexplained (RECORDER-COVERAGE-2). A session can still take far longer than
the target content duration; the fix ensures it captures the content regardless.

**Objective 3 is parallel** because Pieces 4 and 6 are pure DEL-CONV — they touch Stage D
and Stage F timing conversions, not recorder code. Both are independent of the recorder
investigation and of the player test. Piece 6 is the customer-visible defect (~10s seek
offset by frame 1800 on FP7oJQ, where gaps accumulate).

**Objective 4 is test-then-decide** rather than build. Flutter's `video_player` wraps
AVPlayer (iOS) and ExoPlayer (Android) — both support VFR natively, but the Flutter wrapper
may not pass through VFR timestamps. If the player handles VFR, Stage F should stop
re-encoding and Piece 7 dissolves. If it does not, Piece 7 supplies `1.0 / nominal_dt_s`
as the CFR target. The test determines which path.

**Objective 5 (CP22 NAType)** is a small fix to `Stage E` that causes a crash on 201606
(one of the high-dispersion clips from TIMING-DISPERSION-1). It must land before annotation
produces results that run through Stage E.

### Relationship to piece definitions

The piece definitions in §1 describe scope of work. This plan describes execution order
and grouping. Pieces not listed in the plan (3, 5, 9, 10) are not cancelled — they are
sequenced after the six objectives, folded into them, or resolved:

- **Piece 3** dissolved (Piece 0b). Not a prerequisite for Piece 4.
- **Piece 5** (cross-camera) is downstream of objective 3 and not on the critical path.
- **Piece 9** (A/B instrument fix) gates human review of preview video, not numeric
  measurement. Sequenced after objective 3.
- **Piece 10** resolved by Piece 11 (see tombstone in §1).
