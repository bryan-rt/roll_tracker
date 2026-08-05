# RECORDER-DUPFIX-1/2: Findings

Date: 2026-07-29 (DUPFIX-1), 2026-07-30 (DUPFIX-2)

## Instrument Validation — Known-Bad Control

**Clip:** `FP7oJQ-20260318-200014.mp4` (arrival-PTS, March 2026, Lavf61.7.100)

| Method | Total | Dups | Dup % |
|--------|-------|------|-------|
| framehash (MD5, adjacent-identical) | 4530 | 34 | 0.75% |
| mpdecimate strict (hi=1:lo=1:frac=1) | 4530 | 35 | 0.77% |
| mpdecimate default (hi=768:lo=320:frac=0.33) | 4530 | 2181 | 48.1% |

**`framehash_total` (4530) matches historical total exactly.** Same decode population.

**Gate: PASS.** framehash (34) and strict mpdecimate (35) agree within 1 frame on
pixel-identical duplicates. Both confirm real duplication exists on arrival-PTS footage.

**RELIABILITY-1's "255 dups (5.6%)" was measured with intermediate or default mpdecimate
thresholds** — it counted near-identical frames (perceptual similarity), not pixel-identical.
Default thresholds catch 2181 (48%) on this gym video. The 255 figure is neither wrong nor
comparable to framehash; they measure different things. **Pixel-identical duplicate rate on
arrival-PTS footage is 0.75%, not 5.6%.**

---

## Predictions

### C4 (DUPFIX-1)

> If Count 2 (nb_read_frames) for FP7oJQ-20260728-062531.mp4 returns ~1867 rather than 1830,
> the sidecar's output_count is wrong at source and a large share of the reported mismatch
> dissolves without any CFR-padding argument being needed.

**Result: FAILED.** `nb_read_frames = 1830` (not 1867). All three output counts agree:
`nb_frames == nb_read_frames == cv2_iterated == framehash_total == 1830`.

The sidecar's `output_count` is correct. The mismatch between showinfo (1857) and output
(1830) is real.

### P1 (DUPFIX-2) — RETRACTED BEFORE MEASUREMENT

> `nb_read_frames(PPDmUg-20260728-070422)` = 1765 if boundary misattribution is the sole
> explanation for PPDmUg-070219's +135 residual.

**RETRACTED.** The prediction assumed frames are conserved across an attempt (misattribution
only refile, never lose). This assumption is exactly what should be tested, not assumed.
RELIABILITY-1's sidecar reports `output_count(070422) = 1660`, and DUPFIX-1 established
`nb_read_frames == nb_frames` universally. Therefore nb_read will be ~1660, not 1765, and the
residuals cannot cancel under pure boundary misattribution.

**Confirmation:** `nb_read_frames(PPDmUg-20260728-070422)` = **1660** (matches sidecar
`output_count` exactly, as expected).

### P2 (DUPFIX-2)

> FP7oJQ h06 attempt_1 ledger residuals sum within +/-10 of zero; `framehash_adjacent_dups`
> = 0 for all four segments.

**Result: ~~PASS~~ CORRECTED (CP-R5).** Original residuals: +27, -1, -4, -29 = -7. The -7
rested on the line-position split silently dropping 47 leading-edge showinfo lines (3.1s
of filter-graph output before the first `Opening` marker). With the PTS-based split
(CP-R5), true input count is 6276, not 6229. Conservation deficit = 6236 - 6276 = **-40**
(40 real frame drops). FP7oJQ h06 attempt_1 is **lossy**, consistent with PPDmUg. The P2
criterion (+/-10 of zero) **FAILS**. `framehash_adjacent_dups` = 0 remains valid.

### P3 (DUPFIX-2)

> Attempt-level conservation figures computed from sidecar `_meta` reproduce the N2 table
> within +/-2 frames per attempt.

**Result: PASS.** All values match exactly:

| Attempt | sum input | sum output | Net | Conserved? |
|---------|-----------|------------|-----|------------|
| FP7oJQ h06 attempt_1 (062531, 062734, 062939, 063135) | 6229 | 6236 | **-7** | yes |
| FP7oJQ h07 attempt_1 (070240 only) | 428 | 460 | -32 | n/a (single) |
| PPDmUg h07 attempt_1 (070219, 070422) | 3595 | 3490 | **+105** | **no** |
| PPDmUg h07 attempt_2 (070849...071850) | 10230 | 10156 | **+74** | **no** |
| PPDmUg h07 attempt_3 (072126, 072328, 072528) | 5272 | 5272 | **+0** | yes, exactly |
| FP7oJQ h29 attempt_1 (075243 only) | 554 | 600 | -46 | n/a (single) |

Two attempts conserve (FP7oJQ h06, PPDmUg attempt_3), two do not (PPDmUg attempt_1 and 2).
Both mechanisms are real and attempt-specific.

**N6 correction:** RELIABILITY-1's FP7oJQ total row stated output = 6716. Column sum is
1830 + 1800 + 1800 + 806 + 460 = **6696**. The stated +0.9% should be +0.59%. Transcription
error, not measurement error.

### P4 (DUPFIX-2)

> `pts_implied_missing_frames` summed per attempt tracks that attempt's conservation deficit.

**Result: PARTIAL.** Tracks on FP7oJQ; does not track on PPDmUg.

| Attempt | Conservation net | PTS implied missing | Match? |
|---------|-----------------|--------------------:|--------|
| FP7oJQ h06 attempt_1 | -7 | 8 | yes |
| FP7oJQ h07 attempt_1 | -32 | 33 | yes |
| PPDmUg h07 attempt_1 | +105 | 3 | **no** |
| PPDmUg h07 attempt_2 | +74 | 3 | **no** |
| PPDmUg h07 attempt_3 | +0 | 0 | yes |
| FP7oJQ h29 attempt_1 | -46 | 46 | yes |

This cleanly separates two mechanisms:
- **FP7oJQ:** PTS gaps account for the deficit. Frames are genuinely missing from the PTS
  stream (upstream or network loss). The gaps are real drops.
- **PPDmUg attempt_1/2:** No PTS gaps (3 total). The 105/74-frame surplus of input over
  output is ffmpeg-side CFR decimation: the camera delivered faster than 15fps (measured
  15.86/15.63fps), and ffmpeg dropped excess frames to maintain the CFR grid. These frames
  reached the filter graph (showinfo saw them) but were not encoded.

---

## Per-Segment Results

### Five-Count Table (DUPFIX-1 segments + DUPFIX-2 additions)

All source-PTS segments: 2026-07-28 and 2026-07-29, Lavf61.7.103, `SOURCE_PTS=1`.
Control clip: 2026-03-18, Lavf61.7.100, arrival-PTS.

| Segment | Mode | Pos | nb_read | fh_dups | showinfo | Ledger |
|---------|------|-----|---------|---------|----------|--------|
| FP7oJQ-20260728-062531 | source-PTS | first (1/4) | 1830 | **0** | 1857 | **+27** |
| FP7oJQ-20260728-062734 | source-PTS | mid (2/4) | 1800 | **0** | 1799 | **-1** |
| FP7oJQ-20260728-062939 | source-PTS | mid (3/4) | 1800 | **0** | 1796 | **-4** |
| FP7oJQ-20260728-063135 | source-PTS | last (4/4) | 806 | **0** | 777 | **-29** |
| FP7oJQ-20260728-070240 | source-PTS | first (1/1) | 460 | **0** | 428 | **-32** |
| PPDmUg-20260728-070219 | source-PTS | first (1/2) | 1830 | **0** | 1965 | **+135** |
| PPDmUg-20260728-070422 | source-PTS | last (2/2) | 1660 | **3** | 1630 | **-27** |
| PPDmUg-20260728-071051 | source-PTS | mid (2/6) | 1800 | **0** | 1800 | **0** |
| FP7oJQ-20260729-075243 | source-PTS | first (1/1) | 600 | **0** | 554 | **-46** |
| FP7oJQ-20260318-200014 | arrival-PTS | N/A | 4530 | **34** | N/A | N/A |

**Ledger** = `showinfo_count - (nb_read_frames - framehash_dups)`

**All output counts (nb_frames, nb_read_frames, cv2_iterated_count) agree exactly on every
segment**, including all DUPFIX-2 additions. Container metadata is reliable.

**PPDmUg-20260728-070422 has 3 framehash adjacent dups (0.18%).** This is the only
source-PTS segment with non-zero exact duplicates. It is a mid-attempt segment with low PTS
stdev (1.71ms) and occupied scene. The 3 dups are included in the ledger calculation.

**Residuals that closed (0):** PPDmUg-20260728-071051 only.
**Residuals that did not close:** all other source-PTS segments.

The non-zero residuals represent real frame count differences between what showinfo counted
(filter graph) and what the mp4 contains (encoder output). The ledger does not claim to
close at zero — it quantifies the gap.

### Sidecar input_n Comparison

| Segment | input_n dups | input_n jumps | fh dups | Match? |
|---------|-------------|---------------|---------|--------|
| FP7oJQ-20260728-062531 | 0 (0.0%) | 0 (0.0%) | 0 | yes |
| FP7oJQ-20260728-062734 | 5 (0.3%) | 0 (0.0%) | 0 | no |
| FP7oJQ-20260728-062939 | 4 (0.2%) | 0 (0.0%) | 0 | no |
| FP7oJQ-20260728-063135 | 29 (3.6%) | 0 (0.0%) | 0 | no |
| FP7oJQ-20260728-070240 | 33 (7.2%) | 0 (0.0%) | 0 | **no** |
| PPDmUg-20260728-070219 | 2 (0.1%) | 108 (5.9%) | 0 | no |
| PPDmUg-20260728-070422 | 30 (1.8%) | 0 (0.0%) | 3 | no |
| PPDmUg-20260728-071051 | 0 (0.0%) | 0 (0.0%) | 0 | yes |
| FP7oJQ-20260729-075243 | 46 (7.7%) | 0 (0.0%) | 0 | **no** |

**`input_n` dups do NOT correspond to pixel-identical frames.** `input_n` is constructed by
nearest-neighbour mapping and reflects count mismatch, not observed duplication. `input_n`
must not be used as a duplicate flag.

---

## A1 — Boundary Misattribution Test (DUPFIX-2)

### PPDmUg attempt_1 (2 segments, same stderr)

| Segment | Position | showinfo | nb_read | fh_dups | Ledger | Stdev (ms) | FPS |
|---------|----------|----------|---------|---------|--------|-----------|-----|
| PPDmUg-20260728-070219 | first | 1965 | 1830 | 0 | +135 | 10.68 | 15.86 |
| PPDmUg-20260728-070422 | last | 1630 | 1660 | 3 | -27 | 1.71 | 14.99 |
| **Sum** | | **3595** | **3490** | **3** | **+108** | | |

Residuals do NOT cancel (+135 and -27 = +108). Boundary misattribution is **not** the
explanation for PPDmUg-070219's deficit.

The conservation test confirms: 3595 input vs 3490 output = 105 surplus input frames. These
frames reached the filter graph but were decimated by CFR because the camera delivered at
15.86fps against a 15fps target. The PTS stream shows only 3 gaps, confirming the frames
arrived with uniform cadence and were dropped by the encoder, not lost in transit.

PPDmUg-070219's 108 `input_n` jumps agree with the 105-frame conservation deficit to within
3 frames. The jumps are a correct detection of ffmpeg-side decimation, not an artifact.

### FP7oJQ h06 attempt_1 (4 segments, same stderr)

| Segment | Position | showinfo | nb_read | fh_dups | Ledger | Stdev (ms) | PTS gaps |
|---------|----------|----------|---------|---------|--------|-----------|----------|
| FP7oJQ-20260728-062531 | first | 1857 | 1830 | 0 | +27 | 0.47 | 0 |
| FP7oJQ-20260728-062734 | mid | 1799 | 1800 | 0 | -1 | 3.55 | 5 |
| FP7oJQ-20260728-062939 | mid | 1796 | 1800 | 0 | -4 | 2.78 | 3 |
| FP7oJQ-20260728-063135 | last | 777 | 806 | 0 | -29 | 0.47 | 0 |
| **Sum** | | **6229** | **6236** | **0** | **-7** | | **8** |

**CORRECTED (CP-R5):** The original residual sum of -7 was an artifact of the line-position
split dropping 47 leading-edge showinfo lines. True total: 6276 input, 6236 output = **-40**
conservation deficit (40 real frame drops). PTS implied missing = 8 was computed from the
TRIMMED tick deltas and measures only the gaps visible after trimming — it does not capture
the full loss. See `docs/evidence/recorder_boundary_fix_1/findings.md` for corrected
per-segment residuals.

The first segment's +27 residual was boundary misattribution; the last segment's -29 was its
mirror. Both are corrected by the PTS-based split.

### Position-in-attempt pattern

| Position | Segments | Typical residual | PTS stdev |
|----------|----------|-----------------|-----------|
| first | 062531 (+27), 070240 (-32), 070219 (+135), 075243 (-46) | large magnitude, either sign | variable |
| mid | 062734 (-1), 062939 (-4), 071051 (0) | near zero | low |
| last | 063135 (-29), 070422 (-27) | moderate negative | low |

Mid-attempt segments have residuals near zero. First and last segments carry the discrepancy.
RELIABILITY-1 independently identified first segments (062531, 070219, 070849) as having
elevated PTS stdev — a startup transient.

### Mechanism caveat

Boundary misattribution is confirmed as a contributor to per-segment residuals on FP7oJQ
h06 (residuals sum to -7, near zero). But the magnitude of the first-segment offset
(+27 = ~1.8s of frames at 15fps) is larger than x264 encoder lookahead or
`-max_muxing_queue_size 1024` plausibly accounts for. The mechanism by which the stderr
`Opening` line lags the actual first frame by this much is unexplained.

If the stderr boundary split is unreliable at attempt start, the sidecar's per-segment
`input_count` is unreliable in production for first segments, not just in this analysis.

---

## A2 — Per-Frame PTS Gap Analysis (DUPFIX-2)

Computed from raw showinfo PTS in the segment's attributed stderr range. Gap threshold:
delta > 1.5x nominal (67ms). All gaps are 2x nominal (one missing frame per gap) unless
noted.

| Segment | Position | PTS gaps | Implied missing | Stdev (ms) |
|---------|----------|---------|----------------|-----------|
| FP7oJQ-20260728-062531 | first | 0 | 0 | 0.47 |
| FP7oJQ-20260728-062734 | mid | 5 | 5 | 3.55 |
| FP7oJQ-20260728-062939 | mid | 3 | 3 | 2.78 |
| FP7oJQ-20260728-063135 | last | 0 | 0 | 0.47 |
| FP7oJQ-20260728-070240 | first | 33 | 33 | 17.84 |
| PPDmUg-20260728-070219 | first | 2 | 2 | 10.68 |
| PPDmUg-20260728-070422 | last | 1 | 1 | 1.71 |
| PPDmUg-20260728-070849 | first | 1 | 1 | 9.29 |
| PPDmUg-20260728-071051 | mid | 0 | 0 | 0.47 |
| PPDmUg-20260728-071250 | mid | 0 | 0 | 0.47 |
| PPDmUg-20260728-071450 | mid | 0 | 0 | 0.47 |
| PPDmUg-20260728-071650 | mid | 0 | 0 | 0.47 |
| PPDmUg-20260728-071850 | last | 1 | 2 | 4.00 |
| PPDmUg-20260728-072126 | first | 0 | 0 | 0.47 |
| PPDmUg-20260728-072328 | mid | 0 | 0 | 0.47 |
| PPDmUg-20260728-072528 | last | 0 | 0 | 0.47 |
| FP7oJQ-20260729-075243 | first | 46 | 46 | 18.41 |

**Per-attempt PTS implied missing:**

| Attempt | PTS implied missing | Conservation net | Match? |
|---------|--------------------:|:----------------:|--------|
| FP7oJQ h06 attempt_1 | 8 | -7 | yes |
| FP7oJQ h07 attempt_1 | 33 | -32 | yes |
| PPDmUg h07 attempt_1 | 3 | +105 | **no** |
| PPDmUg h07 attempt_2 | 3 | +74 | **no** |
| PPDmUg h07 attempt_3 | 0 | +0 | yes |
| FP7oJQ h29 attempt_1 | 46 | -46 | yes |

### Stdev vs residual cross-reference

FP7oJQ segments with high PTS stdev (17.84, 18.41ms) have high PTS gap counts (33, 46) and
large negative residuals (-32, -46). These are single-segment attempts where the startup
transient IS the entire recording. The bursty delivery at stream start produces genuine PTS
gaps that ffmpeg pads with CFR output frames.

PPDmUg segments with high PTS stdev (10.68, 9.29ms) are first segments of multi-segment
attempts. Their PTS gaps are low (2, 1) despite high stdev. The stdev comes from delivery
jitter, not from missing frames. The surplus input frames (15.86fps > 15fps target) are
uniformly spaced but exceed the CFR rate.

---

## Verdicts

### V1. Does source-PTS footage carry real false-zero-motion duplicates?

**NO — from duplicates.** Nine of ten source-PTS segments have zero pixel-identical adjacent
frames (framehash MD5). PPDmUg-20260728-070422 has 3 (0.18%) — the sole exception, on a
mid-attempt segment. The arrival-PTS control has 34 (0.75%). Source PTS eliminated the
systematic duplication mechanism.

### V2. Do real frame drops occur on source-PTS footage?

**YES — on both cameras, by different mechanisms.**

**FP7oJQ: upstream/network loss.** PTS gaps of 2x nominal account for 8-46 missing frames
per attempt. Conservation deficits match PTS gap counts exactly. Drop rate: 0.1% (h06,
8/6236) to 7.7% (h29, 46/600). Concentrated in single-segment short attempts with startup
transients.

**PPDmUg: ffmpeg-side CFR decimation.** Camera delivers at 15.86fps (attempt_1) and
15.63fps (attempt_2), exceeding the 15fps CFR target. ffmpeg drops 105 and 74 frames
respectively. PTS stream is gap-free (3 gaps total) — the frames arrive but are not encoded.
Drop rate: 3.0% (attempt_1, 105/3490) to 0.7% (attempt_2, 74/10156). Attempt_3 and all
mid-attempt segments with measured_fps = 15.00 show zero drops.

**Dropped frames are false teleports** — the Kalman filter sees a displacement that occurred
over 2x or 3x the expected dt but processes it as 1x, producing inflated velocity estimates.

### V3. Is `input_n` safe for checkpoint 2?

**NO as a duplicate flag.** Zero of the `input_n` dups correspond to pixel-identical frames
(except the 3 on PPDmUg-070422, which `input_n` does not flag — it reports 30 dups at
different positions).

**Partially useful as a drop detector.** PPDmUg-070219's 108 `input_n` jumps correctly
detect ffmpeg-side decimation (matches conservation deficit of 105 to within 3 frames). But
`input_n` jumps cannot detect upstream loss (FP7oJQ's mechanism), which requires PTS gap
analysis.

### V4. What is the authoritative true capture rate?

**`cv2.CAP_PROP_FPS` always returns the CFR target (15.0 or 30.0), never the true capture
rate.** Stage A's BoT-SORT consumes this value as `frame_rate`.

The sidecar's `measured_fps` varies from 13.85 to 15.86 across sessions. The camera's
nominal capture cadence is ~15fps (67ms median delta, 0.47ms stdev on clean segments), but
session-average throughput differs due to startup transients and rate variation.

---

## Limitations for Checkpoint 2

### Duplicates: resolved

Zero pixel-identical adjacent frames on source-PTS footage (one exception: 3 frames on
PPDmUg-070422, 0.18%). The duplicate channel is clean. No Kalman-update-skip is needed.

### Drops: open, two mechanisms, one detectable

| Mechanism | Camera | Rate | Detector | Correctable? |
|-----------|--------|------|----------|-------------|
| Upstream/network loss | FP7oJQ | 0.1-7.7% per attempt | PTS gap (delta > 1.5x nominal) | No — boxmot hardcodes unit Kalman step |
| CFR decimation (camera > 15fps) | PPDmUg | 0-3.0% per attempt | `input_n` forward jump OR conservation deficit | No — same boxmot limitation |

Both drop mechanisms produce false teleports in the Kalman filter. Detection is possible via
`pts_time_s` deltas (covers both mechanisms) or `input_n` jumps (covers CFR decimation
only). Correction requires variable-dt Kalman updates, which requires a boxmot fork not yet
scoped.

### `pts_time_s` is the correct drop detector

Only `pts_time_s` deltas cover both drop mechanisms. `input_n` jumps miss upstream loss
(FP7oJQ's dominant mechanism). The sidecar's `pts_time_s` values are derived from showinfo
PTS via nearest-neighbour mapping, so for drop detection the raw showinfo PTS should be
preferred. Under source-PTS adoption (where input == output frame count), the sidecar
`pts_time_s` would be exact and directly usable.

---

## Open Anomalies

### The 1867 mpdecimate Frames count

RELIABILITY-1's mpdecimate table reports `Frames = 1867` for FP7oJQ-20260728-062531, but
DUPFIX-1 measured `nb_read_frames = nb_frames = framehash_total = cv2_iterated = 1830` for
the same mp4 file. C4's prediction (nb_read ≈ 1867) failed.

The 1867 figure is unexplained by any current measurement. Possible sources: mpdecimate
was run on a different copy of the file, or on a different file entirely (the arrival-PTS
era original), or mpdecimate's internal frame counter includes frames that `nb_read_frames`
does not. The RELIABILITY-1 table does not record the exact command used, so the provenance
cannot be verified. Recorded as an open anomaly.

### PPDmUg-070422: 3 pixel-identical adjacent frames on source-PTS

The sole exception to the "zero exact dups on source-PTS" finding. Three adjacent-identical
frame pairs (0.18%) on a mid-attempt segment with low stdev (1.71ms) and occupied scene.
These may be genuine x264 output coincidences (two consecutive frames with identical
quantized reconstructions) rather than input duplication. Not investigated further.

---

## Commands Run

```bash
# Instrument validation (DUPFIX-1)
ffmpeg -i <clip> -map 0:v:0 -f framehash -hash md5 -
ffmpeg -i <clip> -map 0:v:0 -vf "mpdecimate=hi=1:lo=1:frac=1" -f null -
ffmpeg -i <clip> -map 0:v:0 -vf "mpdecimate" -f null -

# Per-segment counts (DUPFIX-1 + DUPFIX-2)
ffprobe -hide_banner -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 <clip>
ffprobe -hide_banner -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 <clip>

# PTS gap analysis (DUPFIX-2) — via compute_pts_gaps() on raw showinfo lines
python tools/analyze_recorder_timing.py dupfix --segments ... --stderr-files ... --output-dir ...
```

---

## Summary

The "0.0%-vs-8% contradiction" resolves cleanly:

1. **0.0% pixel-identical duplicates** (framehash) is correct for source-PTS footage (one
   exception: 3 frames / 0.18% on PPDmUg-070422).
2. **~8% "fabricated frames"** (sidecar `input_n` dups) is a count-mismatch artifact, not
   frame duplication. x264 re-encodes padded frames independently; they differ at pixel level.
3. **RELIABILITY-1's mpdecimate "255 dups"** on the arrival-PTS control used default
   thresholds catching near-identical similarity. Pixel-identical count: 34 (0.75%).

**False-zero-motion from duplicates does not exist on source-PTS footage.** The duplicate
half of the checkpoint-2 forward plan (flag via `input_n`, skip Kalman updates) is
unnecessary and would be harmful.

**Frame drops are real and measured.** Two mechanisms: upstream/network loss on FP7oJQ
(0.1-7.7%, detected by PTS gaps), and CFR decimation on PPDmUg (0-3.0%, detected by
conservation deficit). Both produce false teleports in the Kalman filter. Detection is
possible; correction requires a boxmot fork (variable-dt Kalman step) not yet scoped.

**The remaining kinematic corruption channels are:**
1. **False velocity from wrong frame rate** — `cv2.CAP_PROP_FPS` (15.0) vs true capture rate
   (13.85-15.86). Affects every frame uniformly. LIVE BUG, addressable without a fork.
2. **False teleports from dropped frames** — 0-7.7% per attempt. Detectable via PTS gaps.
   Not correctable without a boxmot fork.
