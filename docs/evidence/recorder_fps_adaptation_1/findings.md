# CP-R1b: Frame-Rate Bimodality and TRIM-BIMODAL Defect

> **PARTIALLY SUPERSEDED 2026-08-07 (CP-R11).** Sections 4 and 5 corrected below.
> The "bimodal interleaving" model is replaced by "blocked modes with periodic gaps."
> The "structurally undecidable" verdict is qualified — sustained regularity over
> 1,979 consecutive PPDmUg frames and periodic (not random) FP7oJQ gap spacing
> resolve the ambiguity. See `docs/evidence/frame_spacing_1/findings.md` for the
> full CP-R11 analysis (283 segments, 247K intervals vs 8 segments here).
> Sections 1-3, 6-11 remain valid as written.

**Date:** 2026-08-04
**Code state:** `6112dc0` (CP-R3 + local-outside-function fix)
**Footage:** CP-R1 capture (19:32-20:37 UTC) + smoke tests (hours 09-14)
**Tool:** `tools/analyze_fps_adaptation.py`
**Runtime:** 335 seconds (sequential decode, no seeking)

## 1. Headline: measured_fps misreports bimodal segments

The CP-R1 capture's apparent "jump from 15fps to 30fps" on FP7oJQ is partially a
**TRIM-BIMODAL artifact**. The transition segment (FP7oJQ-163102) contains a 66/34 mix of
33ms and 67ms inter-frame intervals, but `measured_fps` reports 30.0019 because the
trimmed mean discards the entire 67ms mode as outliers (36% discard rate). The segment is
a bimodal mix, not 30fps.

The three subsequent segments (163300, 163459, 163659) are genuinely near-pure 30fps
(short-mode 99-100%, discard 0-1%). A real proportion shift did occur — but the transition
was misreported by the broken metric.

PPDmUg's "17.6fps outlier" is correctly reported (0% discard, boundary survives), but its
"15.45fps ramp" is partially inflated (10.8% discard on 163240 — trims 2970-tick frames
but keeps 3060-tick frames).

**The honest metric is short-mode proportion** (fraction of inter-frame intervals at
~33ms vs ~67ms), not `measured_fps`. All bimodal characterizations in this document use
short-mode proportion.

## 2. Data Inventory

### Cameras

| Camera | Resolution | Segments (passthrough) | Segments (cfr_grid, excluded) | Hours |
|--------|-----------|----------------------|------------------------------|-------|
| FP7oJQ | 1920x1080 | 44 | 9 | 09, 10, 11, 14, 15 |
| PPDmUg | 1280x720 | 46 | 12 | 09, 10, 11, 14, 15 |
| J_EDEw | — | 0 | 0 | offline |

### Run partitioning (hour 15)

| Run label | Segments per camera | Time range (UTC) | Notes |
|-----------|-------------------|------------------|-------|
| `aborted` | 1 | 19:08-19:10 | First capture, killed by `local`-outside-function crash |
| `capture` | 31 | 19:32-20:37 | Full 65-min window, successful |

### Exclusions

- All `timing_mode: "cfr_grid"` segments excluded (arrival-PTS bursty deltas produce
  garbage `measured_fps`; documented in CP-R6 roadmap caveat).
- 20-second smoke test segments have weak resolving power for mid-segment rate changes.
  Their segment-level `measured_fps` is valid for stable segments; absence of bimodality
  within them is weak evidence.

### Sampling

- Phase 2 transition segments: stride 15 (one sample per 15 frames for brightness +
  detection; full per-frame dt from sidecar).
- Short segments (<30s): stride 30. All other segments: stride 150.
- Sequential decode with `cap.read()` (no seeking — C1 correction).
- Detection thresholds: counts reported at both conf >= 0.25 and conf >= 0.45.

## 3. TRIM-BIMODAL: Named Defect in Production Sidecar

### The defect

The CP-R2b trimmed mean (lo = median x 0.5, hi = median x 1.5) assumes a unimodal dt
distribution with outlier gaps. Under bimodal oscillation (two discrete rates), the
majority mode captures the median, and the lo/hi bounds discard the minority mode as
"outliers."

### Three failure modes observed

| Segment | Short-mode % | Discard % | Failure mode | measured_fps | Correct? |
|---------|-------------|-----------|-------------|-------------|----------|
| FP7oJQ-163102 | 65.9% | 36.0% | Discards entire 67ms mode (hi=4590 < 5940) | 30.0019 | **Wrong** — reports majority mode only |
| PPDmUg-163240 | 16.1% | 10.8% | Partial: trims 2970 but keeps 3060 (lo=3015) | 15.4530 | **Inflated** |
| PPDmUg-163041 | 29.9% | 0.0% | Boundary survives (lo=2970 = tick value exactly) | 17.6351 | **Correct** (by luck) |

### Blast radius

Does NOT affect stable-rate segments. All controls show correct `measured_fps` with
legitimate discard rates (FP7oJQ ~8.4% from real frame gaps, PPDmUg ~0%). Only segments
containing bimodal oscillation are affected.

### Fix direction (not implemented here — CP-R1b is analysis only)

Detect bimodality (e.g. two peaks in the tick-delta histogram separated by ~2x) and
report both modes plus their proportions, rather than forcing a single scalar. A
bimodal-aware sidecar would emit: `mode_1_fps`, `mode_1_proportion`, `mode_2_fps`,
`mode_2_proportion`, `is_bimodal` flag. Blocks CP-R6's contracting of `measured_fps`
as authoritative until resolved.

## 4. Short-Mode Proportion Shift (the real observation)

> **SUPERSEDED 2026-08-07 (CP-R11).** The "oscillation" pattern reported below from
> 100-frame sliding windows is a windowing artifact on blocked data. Per-interval
> RLE analysis (CP-R11) shows clean blocks of fast (F194, F205, F370) and slow
> (S10-S24) modes with sharp boundaries. A 100-frame window straddling a block
> boundary reports intermediate proportions that appear as "oscillation" but are
> actually window-averaged block transitions. The data below is not wrong at its
> measurement resolution — the sliding window genuinely shows varying proportions —
> but the underlying mechanism is blocked, not interleaved.

### FP7oJQ

| Segment | Short-mode % | measured_fps | Discard % | Note |
|---------|-------------|-------------|-----------|------|
| 150842-162900 (28 segments) | 0.0% | 15.0 | 8-9% | Stable, all long-mode |
| 163102 (transition) | 65.9% | 30.0019 | 36.0% | Bimodal MIX, misreported |
| 163300 | 99.3% | 30.0004 | 0.7% | Near-pure short-mode |
| 163459 | 99.0% | 30.0010 | 1.0% | Near-pure short-mode |
| 163659 | 100.0% | 29.9994 | 0.0% | Pure short-mode |

The proportion shifted from 0% to 66% to 99-100% across 4 segments (~8 minutes). The
last three segments are genuinely near-pure 30fps (not artifacts). A real change occurred.

### PPDmUg

| Segment | Short-mode % | measured_fps | Discard % | Note |
|---------|-------------|-------------|-----------|------|
| 150849-162841 (28 segments) | 0.0% | ~15.0 | 0-0.5% | Stable, all long-mode |
| 154828 (isolated outlier) | 8.4% | 15.2216 | 5.8% | Brief excursion |
| 163041 | 29.9% | 17.6351 | 0.0% | Peak short-mode |
| 163240 | 16.1% | 15.4530 | 10.8% | Declining |
| 163439 | 8.3% | 15.2215 | 5.5% | Declining |
| 163640 | 0.2% | 15.0088 | 0.1% | Recovered to long-mode |

PPDmUg's short-mode proportion rose to 30% then declined to 0% — a transient excursion,
not a sustained shift. Opposite direction from FP7oJQ (which rose and stayed high).

### The shift occurs mid-stream, not at attempt boundaries

Both FP7oJQ segments 162900 (0% short-mode) and 163102 (65.9% short-mode) are in
**attempt 14** — one continuous 33-minute ffmpeg invocation (20:04:56-20:37:32 UTC).
Stream reconnection is ruled out as the mechanism.

### Within-segment oscillation pattern

FP7oJQ-163102 does not transition once. Sliding-window analysis (100-frame windows) shows
the short-mode proportion oscillating: 0% -> 96% -> 11% -> 100% -> 43% -> 94% -> 67% ->
98% -> 100% -> 57% -> 0%. The stream switches between the two modes multiple times,
sometimes sustaining one mode for hundreds of frames (run of 542 consecutive long-mode
frames = 36 seconds), sometimes flipping within frames.

Both modes show **instantaneous frame-level switching** — each individual dt is either
~33ms (2970/3060 ticks) or ~67ms (5940/6030 ticks), never an intermediate value. The
camera has two discrete rates and switches between them.

## 5. Is This Frame Loss or Genuine Rate Change? (Structurally Undecidable)

> **QUALIFIED 2026-08-07 (CP-R11).** The pair-sum identity below is mathematically
> valid for any single interval. However, the conclusion that 15fps vs 30fps+loss
> is undecidable does not survive 283 segments of data:
> (1) PPDmUg delivered 1,979 consecutive 15fps intervals with zero exceptions
>     (probability 2^-1979 under alternate-frame loss).
> (2) FP7oJQ's gaps have periodic spacing (mode=12 frames, doublet harmonic at
>     7+17=24). Random loss produces geometric inter-gap spacings, not periodic.
> (3) The gap count exactly matches the rate deficit predicted by the camera's
>     grid-rate/effective-rate ratio (14.93/13.85 -> 1 gap per 12.4 frames).
> The 15fps cadence is genuine, and FP7oJQ's gaps are a camera-internal grid
> mismatch, not frame loss. See CP-R11 Sections 4.2, 4.3.

### The identity

The 15fps tick pattern [6030, 6030, 5940] repeating (mean 6000) is exactly the pair-sum
sequence of the 30fps pattern [2970, 2970, 3060] repeating (mean 3000):

- 2970 + 2970 = 5940
- 2970 + 3060 = 6030
- 3060 + 2970 = 6030

The 30fps source shows 2:1 ratio of 2970:3060 with 33.6% consecutive-same pairs
(measured from the stable 30fps control segment FP7oJQ-163300). Under every-other-frame
loss, the expected long-delta ratios are: 5940 at 33.6%, 6030 at 66.2%, 6120 at 0.2%.

Observed 15fps ratio (PPDmUg-162841 control): 5940 at 33.3%, 6030 at 66.7%, 6120 at 0.0%.

**These match to within 0.3%.** This is not a coincidence — it is an algebraic identity.
The 15fps tick sequence IS the pair-sum of the 30fps tick sequence. Therefore:

**No PTS-based analysis can ever discriminate genuine 15fps encoding from systematic
every-other-frame loss of a 30fps source.** Not with more segments, not with finer
measurement. This is a structural limitation.

### Does it matter? (depends on the consumer)

**Kinematics — does not matter.** The frames received carry correct timestamps. Velocity
computed over a 67ms interval is right whether the intervening frame never existed or was
lost. `dt_s` is correct either way.

**Tracking quality — matters.** Half the temporal resolution means larger inter-frame
displacement during fast motion, more Kalman gate misses, more ID switches. This is a
real cost regardless of cause.

**Fixability — matters most.** If it is relay throttling, it may be addressable (stream
profile, transport settings, bandwidth allocation). If it is genuine 15fps encoding, it
is not fixable from the consumer side. This is the only reason to continue investigating.

### What would resolve it (future capture, not this checkpoint)

**RTP sequence numbers.** Missing sequence numbers prove frame loss directly — each RTP
packet carries a monotonically increasing sequence number, and gaps indicate loss. This
requires capturing ffmpeg output at higher loglevel (`-loglevel debug` or
`-rtsp_flags +print_rtp_info`), which is not available from existing footage.

**SDP declared frame rate at connect.** If SDP declares 30fps and the stream delivers
15fps, that is evidence (not proof) of relay throttling. CLAUDE.md already records SDP as
unreliable ("reports 30 when delivering 15"), but the unreliability itself could be the
signal.

**Recommend:** on a future diagnostic capture, add `-loglevel debug` to one camera's
ffmpeg invocation to surface RTP sequence numbers and SDP negotiation. Do NOT run a
capture now.

## 6. Brightness and People-Count Analysis

### Discriminator results (H1 vs H2 from original brief)

The original brief hypothesized H1 (activity-driven fps) vs H2 (light-driven fps). Both
are rendered moot by the TRIM-BIMODAL finding — the apparent "rate change" is a proportion
shift in a bimodal distribution, not a clean rate switch that either hypothesis predicted.

The brightness and detection data are reported for completeness and because they constrain
the proportion-shift mechanism:

**PPDmUg (strongest divergent case):** Zero people detected across all 46 passthrough
segments (both thresholds). Brightness stable at 103-113. Short-mode proportion varies
0-30% with no change in either brightness or people. **Neither H1 nor H2 explains the
short-mode variation on an empty, stably-lit scene.**

**FP7oJQ (frame-level alignment, stride 15 on transition segment 163102):**
- Brightness flat at 85.7-86.9 during the oscillation region (frames 0-750), while
  short-mode proportion swings 0-100%.
- Brightness gradually rises from 85 to 88 over the segment — this occurs at different
  times and in different directions from the short-mode oscillation.
- People count varies (0-4) independently of both brightness and short-mode proportion.
  Frames 435-750 show 100% short-mode with 0 people; frames 2000-2500 show 0% short-mode
  with 1 person.

**Neither H1 (activity) nor H2 (light) is supported at the frame level.**

### Opposite-direction simultaneity

FP7oJQ's short-mode proportion rose (0% -> 66% -> 99%) while PPDmUg's fell (30% -> 0%)
in the same ~6-minute window. Any shared-cause hypothesis must explain opposite
responses from two cameras in the same gym at the same time.

Camera FOV cannot be established from the repo (calibration data contains homography
matrices and lens parameters but no camera position, angle, or coverage descriptions).
The two cameras have different resolutions (1920x1080 vs 1280x720). Without FOV
documentation, the opposite-direction finding weakens but does not rule out
environment-driven explanations (cameras viewing different areas under different
conditions).

### Camera coverage limitation

J_EDEw produced zero segments (offline). Every conclusion rests on two cameras. PPDmUg
carries a known empty-FOV confound (zero people in all footage). FP7oJQ consistently
detects 1 person even in empty-gym smoke tests — likely a static false positive
(background feature); the meaningful signal is the jump from 1 to 2-4 in the last segments.

## 7. Secondary Questions

### PPDmUg 15.2/17.6 outliers — what are they?

Short-mode proportion excursions: 8.4% and 29.9% respectively. The "17.6fps" is 70% of
frames at 67ms and 30% at 33ms, producing a blended average of ~57ms. The "ramp" from
17.6 -> 15.0 is a short-mode proportion decline (30% -> 16% -> 8% -> 0%), not a
continuous rate change. PPDmUg-154828 (15.2216) is an isolated earlier excursion at 8.4%
short-mode, unrelated to the late-session cluster.

### Instantaneous vs gradual

**Instantaneous at the frame level.** Each individual dt is either ~33ms or ~67ms — never
intermediate. The camera has two discrete modes. The proportion shifts over time, but
each frame commits to one mode.

### Does the rate return?

**Yes.** FP7oJQ oscillates between modes multiple times within segment 163102 (run lengths
from 1 to 542 frames), then the last three segments settle near 100% short-mode. PPDmUg's
excursion returns to 0% short-mode by the final segment.

### Attempt-boundary alignment

**No.** The proportion shift occurs mid-stream within attempt 14 for both cameras. Stream
reconnection is ruled out.

## 8. Container Metadata Unreliability

`r_frame_rate` (ffprobe) reads `15/1` on FP7oJQ-163102 even though the segment is 66%
short-mode. The container was created at the rate in force when segmentation started; the
stream changed underneath. Resolution unchanged (1920x1080 throughout). Bitrate roughly
doubles at the transition (555 kbps -> 1002 kbps -> 1700 kbps at constant resolution),
confirming the camera chose to spend more bandwidth rather than being constrained by it.

## 9. Consequences for Checkpoint 2

### Per-clip scalar fps is provably insufficient

The short-mode proportion changes mid-segment within a single ffmpeg invocation. No
single scalar describes a clip containing a proportion transition. `measured_fps` is
additionally broken on these segments (TRIM-BIMODAL).

### Per-frame dt_s is the only reliable source

This holds regardless of the genuine-rate-change vs frame-loss question (Section 5).
`dt_s` is correct under both interpretations. CP-R6's sidecar contract v2 must surface
`dt_s` per frame.

### Container metadata cannot be trusted

`r_frame_rate`, `CAP_PROP_FPS`, and `avg_frame_rate` record container-creation metadata,
not delivered rate. They are stale when the proportion shifts mid-segment.

### Prior camera characterizations are confounded

Every prior "camera fps" figure is a delivery-rate measurement at a specific moment. If
the frame-loss interpretation is correct:

- FP7oJQ "13.85fps" (DUPFIX) = 30fps with ~54% loss
- PPDmUg "15.00fps" (DUPFIX) = 30fps with ~50% loss
- CAPTURE-TIME-1's original "30fps from all cameras" (later marked CORRECTED) may have
  been right, and the "correction" was observing loss

If the genuine-rate-change interpretation is correct, these are real camera rates. **We
cannot distinguish the two from existing data** (Section 5).

### Impact on coast-step injection

If the 67ms intervals are frame loss, the drop channel is far larger than the 0.1-7.7%
measured from trim-discard (those gaps are the ~8% discard rate on stable 15fps segments,
measuring ADDITIONAL loss beyond the baseline ~50%). Coast-step injection sized against a
nominal 67ms interval that is itself a loss artifact would inject steps between every
received frame — functionally equivalent to running the tracker at the true source rate,
which may be the right answer.

## 10. Reinterpretation of Prior Findings

| Prior finding | If genuine rate change | If frame loss |
|--------------|----------------------|--------------|
| "Camera is 15fps" (CAPTURE-TIME-2) | Correct, camera encodes at 15fps | Wrong — camera is 30fps, relay delivers 15fps |
| "SDP reports 30 when delivering 15" (CLAUDE.md) | SDP is wrong | SDP is correct, delivery is throttled |
| FP7oJQ 13.85fps (DUPFIX) | Real camera rate | 30fps with ~54% loss |
| "fps varies per session" (CAPTURE-TIME-1) | Camera changes rate | Loss rate varies |
| "Corrupted footage caveat" (CLAUDE.md) | Applies to arrival-PTS era only | Applies broadly — loss is ongoing |
| 8.4% FP7oJQ trim-discard on stable 15fps | Real frame drops beyond 15fps base | Additional loss beyond ~50% baseline |

## 11. Short-Segment Caveat

Hours 09-14 contain 20-second segments (smoke tests). All show 0% short-mode proportion
and stable 15fps. A 20-second window could contain a proportion shift but the short
duration limits resolving power. The PPDmUg-154828 outlier (8.4% short-mode, from
mid-capture at 15:48) shows that excursions do occur on 120-second segments — suggesting
the 20-second segments simply did not happen to capture one, not that the phenomenon is
absent in earlier hours.
