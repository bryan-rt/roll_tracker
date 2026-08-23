# RECORDER-COVERAGE-2: Aug 23 full-scale validation

**Cameras:** FP7oJQ, PPDmUg (J_EDEw excluded — offline)
**Session:** 2026-08-23, started ~12:36 EDT
**Recorder:** `diag_v7_2.sh` with SOURCE_PTS=1, FPS_PASSTHROUGH=1, TARGET_CONTENT_SECONDS=1800, MAX_WALLCLOCK_SECONDS=7200
**Generated:** 2026-08-23.

---

## 1. RECORDER-BACKLOG-1 validated at full scale

FP7oJQ terminated with `termination=content_target`: 1,798s captured against an 1,800s
target (2s within tolerance), 2,000s wall, 17 segments, 9 attempts.

All 17 segments: `sidecar_schema: 5`, `row_source: "mp4"`, `timing_mode: "passthrough"`,
`source_pts: true`. Zero `cfr_grid`. Zero orphans.

The content-target path works at 1,800s — a different regime from the 120s smoke test.
Three smoke-test-caught bugs (`f4e2f52`, `c261502`, `01bb0cd`) justify never skipping a
smoke test: bug 1 (`log()` called before definition) was fatal under `set -e` and would
have killed the session on all cameras.

PPDmUg terminated with `termination=wallclock_cap`: 183s of 1,800s target (1,617s shortfall),
7,243s wall, 35 attempts. Explicit shortfall line emitted. The camera was genuinely offline
for the vast majority of the window — the recorder behaved correctly by retrying and
eventually hitting the wallclock cap.

---

## 2. Delivery rate: real-time in steady state (Q2 answer)

### The measurement

Instantaneous delivery rate is defined as the content duration of segment N divided by the
wall-clock interval between `segment_start_epoch` of segment N and segment N+1. **This rate
is only meaningful between two consecutive segments of the same attempt.** The wall-clock
interval between the last segment of one attempt and the first segment of the next spans dead
time plus reconnect, and dividing content by that interval yields a number shaped like a rate
that isn't one.

FP7oJQ attempt 1 (13 segments) provides 11 valid rate measurements (segments 1–12; segment
13 is the last in the attempt, so no wall-clock interval to the next same-attempt segment):

| # | Segment | Frames | PTS (s) | Wall dt (s) | Rate |
|---|---------|--------|---------|-------------|------|
| 1 | 123642 | 1,770 | 120.4 | 120 | 1.003 |
| 2 | 123842 | 1,770 | 120.9 | 121 | 0.999 |
| 3 | 124043 | 1,830 | 118.6 | 118 | 1.005 |
| 4 | 124241 | 1,770 | 120.9 | 121 | 0.999 |
| 5 | 124442 | 1,680 | 120.4 | 121 | 0.995 |
| 6 | 124643 | 1,650 | 118.7 | 119 | 0.997 |
| 7 | 124842 | 1,680 | 120.7 | 120 | 1.006 |
| 8 | 125042 | 1,680 | 120.9 | 121 | 0.999 |
| 9 | 125243 | 1,740 | 118.3 | 118 | 1.003 |
| 10 | 125441 | 1,680 | 120.5 | 121 | 0.996 |
| 11 | 125642 | 1,650 | 119.5 | 120 | 0.996 |
| 12 | 125842 | 1,650 | 119.5 | — | — |
| 13 | 130046 | 686 | 49.5 | — | — |

Segments 1–11: all between **0.995× and 1.006×**. Flat near 1.0× across 24 minutes.

Segment 12 has no valid rate (it is followed by segment 13, its last same-attempt peer,
with no further segment to provide a wall-clock endpoint). Segment 13 is a partial
(49.5s) — the session died mid-segment (run.log shows session death and backoff). Attempt 1
delivered 1,489s against an 1,800s target (311s short); attempts 5, 7, and 9 collected the
remainder. This is why there are three attempt boundaries and three MUXER-PTS-1 segments in
this capture: steady 1.0× delivery for 24 minutes, then a session drop, then three
reconnects to collect the remaining ~310s.

### The cumulative `speed=` artifact

ffmpeg's `speed=` field is **cumulative** — total content ÷ total wall time since process
launch. It necessarily starts near zero and climbs, because the first ~10–13 seconds of
each attempt are connection setup with no content flowing. A rising `speed=` curve is
exactly what a constant-rate relay would produce with a fixed startup cost.

The "0.076× → 0.992× ramp" reported from ffmpeg `speed=` is this artifact, not evidence
of relay warm-up. The per-segment instantaneous rates are flat at 1.0× from the first
measurement.

**Rule: never use ffmpeg `speed=` as an instantaneous delivery rate.** It measures a
cumulative average that amortizes startup cost over an ever-growing denominator. Use
per-segment wall-clock deltas within the same attempt instead.

### By attempt

| Attempt | Segments | Content (s) | Wall (s) | Overall rate | Notes |
|---------|----------|-------------|----------|--------------|-------|
| 1 | 13 | 1,489 | 1,494 | **0.997×** | 11 valid per-segment rates: 0.995–1.006× |
| 5 | 1 | 97 | — | — | Single segment, no valid instantaneous rate |
| 7 | 1 | 83 | — | — | Single segment, no valid instantaneous rate |
| 9 | 2 | 137 | 99 | **1.376×** | See below |

Attempt 9 is the sole genuine above-real-time observation: 137s of content in 99s of wall
time, following two failed attempts (6 and 8). This is the one piece of evidence that the
relay can drain accumulated backlog — content that accumulated on the relay during the dead
period arrives faster than real-time when the connection resumes.

### The corrected model

**Delivery is approximately real-time (1.0×) in steady state, and can burst above 1.0× to
drain backlog after a gap.** FP7oJQ attempt 1 delivered 1,489s of content in 1,494s wall
(0.997×). Attempt 9 delivered 137s in 99s wall (1.376×), following two failed attempts —
consistent with draining accumulated backlog.

**This does NOT mean the coverage problem is solved in general.** Wednesday's CP-R8 capture
(attempt 3: 864s content in 1,634s wall) was genuinely sub-real-time for an extended period.
Aug 23 proves the relay CAN deliver at 1.0× and the `-t` fix handles the condition correctly
when it occurs. Wednesday's sustained sub-real-time delivery remains unexplained. The
difference between the two sessions is unknown — it may be relay load, network path, time of
day, or another factor. What today proves is a validated fix, not a disappeared problem.

---

## 3. Attempt boundaries are hard breaks

`_meta.attempt` (sidecar contract line 85) distinguishes two cases that look identical in
wall-clock terms:

| Boundary | Content | Signal |
|----------|---------|--------|
| Same `attempt` | Contiguous, regardless of how large the wall-clock gap looks | — |
| `attempt` changes | **Real discontinuity** — stream died, relay backlog lost | `attempt` differs |

### FP7oJQ attempt→segment mapping

| Segments | Attempt | Boundary type |
|----------|---------|---------------|
| 123642 – 130046 (13 segments) | 1 | — |
| 130325 (1 segment) | 5 | **Hard break** (att 1→5) |
| 130549 (1 segment) | 7 | **Hard break** (att 5→7) |
| 130817 – 130941 (2 segments) | 9 | **Hard break** (att 7→9) |

Three content discontinuities. Attempts 2, 3, 4, 6, 8 produced no segments (failed
attempts — confirmed from `attempt_log.jsonl` and ffmpeg stderr sizes <3KB).

### PPDmUg: every boundary is an attempt boundary

PPDmUg's flickering camera produces the clearest demonstration that wall-clock spacing
alone cannot distinguish delivery lag from genuine discontinuity:

| # | Segment | Att | Epoch | Frames | PTS (s) | Att change |
|---|---------|-----|-------|--------|---------|------------|
| 1 | 123830 | 4 | 1787503110 | 694 | 46.5 | — |
| 2 | 124855 | 9 | 1787503735 | 490 | 34.3 | **Hard break** (4→9) |
| 3 | 134200 | 19 | 1787506920 | 356 | 23.7 | **Hard break** (9→19) |
| 4 | 135811 | 22 | 1787507891 | 341 | 22.7 | **Hard break** (19→22) |
| 5 | 140901 | 24 | 1787508541 | 969 | 64.5 | **Hard break** (22→24) |

Total: 5 segments, 5 different attempts, 191.7s content in 7,243s wall. Every segment
boundary is a genuine discontinuity where the stream died and content was lost.

Without the `attempt` field, the wall-clock gap between PPDmUg segments 1→2 (625s) looks
the same as FP7oJQ's within-attempt gap between segments 12→13 (124s → 159s including
delivery lag). Only `attempt` disambiguates. **This is the load-bearing evidence for the
Piece 4 requirement** (see §3.1).

### 3.1 Piece 4 requirement: attempt-aware session aggregation

`f0_sidecar.py:231` exposes `attempt`. Nothing in `src/bjj_pipeline/stages/` reads it.
`session_d_run` currently aggregates all clips into one timeline and builds reconnect
edges across every boundary, including genuine discontinuities.

**Requirement:** Session aggregation must treat an `attempt` change between consecutive
clips as a **hard break**: no reconnect edges across it, no cross-clip tracklet joins, and
the session timeline records a discontinuity of unknown duration rather than treating the
clips as adjacent. Without this the pipeline will stitch tracklets across a genuine teleport
in space and time, and has no other signal that would let it detect one.

---

## 4. `pts_wallclock_offset_s` under delivery lag — open question (Piece 5)

`offset_method: "lower_envelope"` computes the offset from `host_arrival_s`. Under
sub-real-time delivery, arrival is later than capture by the accumulated lag, and that lag
grows through a run. So the offset maps PTS onto **arrival** time, not **capture** time.

Two cameras delivering at different rates in the same session — observed: FP7oJQ 0.94× and
PPDmUg 0.25× on Aug 22 — would be lagged by different and growing amounts, so their
"wallclock-aligned" timelines could diverge by minutes.

Piece 5's cross-camera sync assumes `pts_wallclock_offset_s` yields comparable capture times
across cameras. Under sub-real-time delivery it may instead reflect arrival time, differing
per camera. The contract's ±14–56ms accuracy figure (CAPTURE-TIME-2) predates this
observation. **Verify before planning Piece 5.** This may partly explain historically weak
cross-camera evidence.

---

## 5. Camera fleet health (2026-08-23, operational finding)

- **FP7oJQ:** Healthy. 17 segments, 1,798s content, content-target termination.
- **PPDmUg:** Flickering. 5 segments across 35 attempts, 183s content, wallclock-cap
  termination. Camera generates valid RTSP URLs via the SDM API then serves 10–65s of data
  before going silent, triggering consecutive-failure escalation (slow-poll 120–300s).
- **J_EDEw:** Excluded from this capture (offline on Aug 22 with 0s content across 9
  attempts). Confirmed at source: the Nest app shows only 1 of 3 cameras up.

**This is a camera problem, not a recorder defect.** The recorder behaved correctly:
retried a genuinely offline camera until the wallclock cap, emitted explicit shortfall, and
terminated cleanly.

**Consequence:** Multi-camera GT is currently unobtainable, which blocks Piece 5 and all
cross-camera work independently of the timing questions. Any near-term GT capture will be
single-camera (FP7oJQ).

---

## 6. MUXER-PTS-1: second reproduction

The segment muxer duplicate-PTS defect (MUXER-PTS-1) reproduces on this capture.

**FP7oJQ:** `dt_s=0.0` at frame_index=2 on exactly the **3 attempt-first segments**
(130325/att5, 130549/att7, 130817/att9). Attempt 1's first segment (123642) is clean.
Non-first segments within an attempt (130941/att9) are clean. 14 of 17 segments are clean.

**PPDmUg:** `dt_s=0.0` at frame_index=2 on **all 5 segments** — but every PPDmUg segment is
attempt-first by definition (each attempt produced at most 1 segment).

**Pattern confirmed across two captures (CP-R8 + this):** attempt-first segments only, not
segment-first in general. Attempt 1's first segment is exempt (123642 here, 200827 was
affected in CP-R8 — not perfectly deterministic on attempt 1, but deterministic on attempts
>1).

Combined across CP-R8 + Aug 23: **9 of 9** attempt-first segments for attempts >1 are
affected (CP-R8 202356; Aug 23 FP7oJQ 130325/130549/130817; Aug 23 PPDmUg all 5). 1 of 2
attempt-1 first segments are affected (CP-R8 200827 yes, this capture 123642 no).
**Deterministic on attempts >1; the muxer fix is immediately verifiable** — a single capture
with multiple attempts will confirm or deny.

---

## 7. `measured_fps` consumer audit

No pipeline stage, tracker, or evaluation tool reads `measured_fps` as a rate. Consumers:

| Consumer | Usage |
|----------|-------|
| `f0_sidecar.py` | Parses into `SidecarMeta`, never computes from it |
| `smoke_test.sh` | Band-check assertion (13.5–16.5 / 27–33 / 6.75–8.25) |
| `diag_v6.sh` / `diag_timing.sh` | Emits to sidecar |
| Analysis tools | Display only |

`smoke_test.sh`'s band-check is itself questionable — it would fail on a legitimate session
whose median interval falls outside the expected bands (e.g., a hypothetical 20fps session).

**Candidate for removal at the next schema change.** The `818988d` commit already demoted
it to advisory in the contract. Not removing now to avoid a schema bump for a cosmetic
change.

---

## 8. Cadence observation

PPDmUg's median inter-frame interval was ~33ms in one Aug 22 session and ~67ms in others;
FP7oJQ ~67ms throughout Aug 23. Under variable-dt this requires no handling.

---

## 9. Derivation methods and basis

- **Instantaneous delivery rate:** content PTS span of segment N ÷ (`segment_start_epoch`
  of segment N+1 − `segment_start_epoch` of segment N). Only valid between consecutive
  segments of the **same attempt**. Camera: FP7oJQ, attempt 1, Aug 23.
- **Segment content duration:** `pts_time_s` of the last frame row in the sidecar.
- **Frame count:** sidecar row count minus 1 (header row).
- **Attempt mapping:** `attempt` field in sidecar header + `attempt_log.jsonl`.
- **MUXER-PTS-1 check:** `dt_s` of frame_index=2 (sidecar row index 3).
- **`measured_fps` audit:** `grep -r measured_fps src/ services/ tools/`.
