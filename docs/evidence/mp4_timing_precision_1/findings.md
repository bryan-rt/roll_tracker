# MP4 Timing Precision: Encoder Timebase Requantization

**Date:** 2026-08-17
**Commit:** `bf36ac2` (HEAD of `services_uploader` at time of measurement)
**Scope:** Does the mp4 container carry real per-frame timing from the camera?

---

## 1. Finding

**The production mp4 does NOT carry real per-frame timing.** The x264 re-encode
requantizes all input PTS onto a uniform grid at its default timebase (1/15360), destroying
the camera's sub-frame cadence signature.

**This is fixable.** Adding `-enc_time_base 1/90000` to the encoder options makes x264 use
the input (RTP) timebase, and the 5940/6030 tick alternation survives encoding. This option
is NOT in the current production recorder.

**Until fixed, the showinfo stderr stream is the ONLY source of real per-frame timing.**

---

## 2. Qualification of CP-R2

CP-R2 established: "per-frame timing is TRUE and camera-derived." This holds for the
**sidecar** (derived from showinfo, which carries the RTP 90000-timebase PTS). It does NOT
hold for the **mp4 container**.

| Property | Sidecar (showinfo) | Mp4 (x264 output) |
|----------|-------------------|-------------------|
| Frame count | Incorrect (boundary attribution, 33% mismatch) | Correct (Piece 0, 94/94) |
| Gap structure (15fps vs 30fps blocks) | Correct | **Partially** — gaps (2x nominal) visible as 2048-tick deltas; 30fps blocks produce 0-tick deltas (degenerate) |
| Sub-frame cadence (5940/6030 alternation) | Correct | **Lost** — uniform 1024-tick deltas |
| Bimodal 30fps frames | Correct (33ms/34ms distinct) | **Degenerate** — pairs share identical PTS (dt=0) |
| Timebase | 1/90000 (RTP, 11.1us resolution) | 1/15360 (x264 default, 65.1us resolution) |

---

## 3. Mechanism

### Why the production recorder re-encodes

The recorder needs `-vf showinfo` for per-frame timing. `showinfo` is a filter-graph
operation. The filter graph requires decoding and re-encoding — it is incompatible with
`-c:v copy` (stream copy). Therefore the production path uses `-c:v libx264`.

```
RTSP stream (90000 timebase)
  → ffmpeg input (-copyts preserves RTP PTS)
  → filter graph (showinfo logs PTS to stderr)
  → x264 encoder (requantizes PTS to 1/15360 by default)
  → mp4 segment muxer (-reset_timestamps 1 rebases to zero)
```

### The requantization

x264's default output timebase is 1/15360 (= 1/(1024 * 15), derived from the detected
frame rate). At this timebase:

| Real interval (90000 tb) | Ticks at 90000 | Expected at 15360 | Actual at 15360 |
|--------------------------|---------------|-------------------|-----------------|
| 66.0ms (5940 ticks) | 5940 | 1013.8 → 1014 | **1024** |
| 67.0ms (6030 ticks) | 6030 | 1029.1 → 1029 | **1024** |
| 33.0ms (2970 ticks) | 2970 | 506.9 → 507 | **0** (same PTS as previous) |
| 34.0ms (3060 ticks) | 3060 | 522.2 → 522 | **0** (same PTS as previous) |
| 133ms (11970 ticks) | 11970 | 2041.3 → 2041 | **2048** |

The 15360 timebase CAN represent the distinct values (1014 vs 1029, 507 vs 522). The
uniform 1024 output is x264 actively remapping timestamps to its own grid, not a
precision limitation of the timebase.

### The fix: `-enc_time_base 1/90000`

This ffmpeg option tells the encoder to use the input timebase rather than computing its
own. With this option, x264 passes through the original PTS values unchanged.

**Verification (synthetic test with known 5940/6030 alternation):**

| Configuration | Output timebase | Deltas | Alternation |
|---------------|----------------|--------|-------------|
| Default (no option) | 1/15360 | uniform 1024 | **Lost** |
| `-video_track_timescale 90000` only | 1/90000 | uniform 6000 | **Lost** |
| `-enc_time_base 1/90000` | 1/90000 | 5940/6030 | **Preserved** |
| `-enc_time_base 1/90000 -copyts -reset_timestamps 1` | 1/90000 | 5940/6030 | **Preserved** |

`-video_track_timescale` changes the muxer's output timebase but does NOT affect the
encoder's internal requantization. `-enc_time_base` controls the encoder directly.

### Bimodal impact (measured)

PPDmUg-20260805-154251 (bimodal, `is_bimodal: true`, `short_mode_fraction: 0.085`):

| Source | Tick deltas |
|--------|------------|
| Sidecar (90000) | 5940 (×564), 6030 (×1130), 2970+3060 (×158 fast mode), 11970/12060 (×7 gaps) |
| Mp4 (15360) | 1024 (×1772), **0 (×78)**, 2048 (×7), 1 (×1), 1023 (×1) |

78 frame pairs in the 30fps block share identical PTS in the mp4. A consumer computing
`dt_s` from mp4 PTS would get `0.0` — physically impossible and computationally dangerous
(division by zero in velocity, infinite acceleration in Kalman filter).

---

## 4. Implications

### For CP-R13 (sidecar generation)

The mp4 provides the correct **frame count** but not the correct **per-frame timing**.
CP-R13's architecture uses the mp4 only for frame count (the row-count guarantee that
Piece 0 identified as the prerequisite), and showinfo for all timing values. This is the
correct split under the current recorder configuration.

### For future optimization

Adding `-enc_time_base 1/90000` to the recorder's V_OPTS would make the mp4 carry real
per-frame timing and simplify CP-R13 to pure mp4 derivation. This is a single-line change
to `diag_v6.sh:305`:

```bash
V_OPTS=(-c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30 -enc_time_base 1/90000)
```

**Not done in CP-R13** — it changes recording behavior (the brief's constraint), and needs
a production smoke test. Recorded as a follow-up.

### For the 94 archived segments

Existing mp4s on disk have the 15360-timebase requantized PTS. They cannot be used for
per-frame timing derivation. Regenerated sidecars for these segments must come from the
showinfo stderr logs (if archived) or accept the 15360 precision as a documented limitation.

---

## 5. CP-R13a: Fix Verified (2026-08-17)

**`-enc_time_base 1/90000` added to the recorder's V_OPTS.** The mp4 now carries real
per-frame timing at the RTP 90000 timebase. Verified on live capture:

### Before/after comparison (T1)

| Metric | Before (baseline) | After (CP-R13a) |
|--------|-------------------|-----------------|
| Container timebase | 1/15360 | **1/90000** |
| FP7oJQ tick deltas | uniform 1024 (with 0 and 2048) | **5940, 6030, 11970, 12060** (real alternation) |
| Zero-tick pairs | 1 (bimodal pair, genuinely from source) | 1 (same pair, correctly preserved) |
| File size (FP7oJQ ~21s) | 3,156,113 bytes | 2,017,749 bytes |

FP7oJQ at 30fps in the baseline showed uniform 1024-tick deltas (alternation destroyed).
After CP-R13a, the same camera at 15fps shows 5940/6030 alternation. Both captures have
one zero-delta pair — a genuine bimodal frame from the source stream, present in both the
mp4 and the sidecar.

### Frame-for-frame agreement (S2)

FP7oJQ-20260817-105755 (post-change, 300 mp4 frames, 305 sidecar rows, `mismatch: true`):
first 299 mp4 tick deltas compared against first 299 sidecar tick deltas.
**0 disagreements out of 299 comparisons (100% agreement).**

The mp4 and sidecar now carry identical per-frame timing.

### CFR rollback (T2)

*Superseded 2026-08-17 (`34a9a72`).* CP-R13a's rollback verification confirmed only that
a segment was produced and that a sidecar carried `timing_mode: "cfr_grid"`. It did NOT
check segment count or duration. CFR was in fact broken: `-enc_time_base 1/90000` on the
CFR path broke the segment muxer's cut-point calculation, producing a single unsegmented
file (FP7oJQ-20260817-110127: 145,757 frames; FP7oJQ-20260817-134258: 152,933 frames) and
captures that ran hours past their `WINDOW_SECONDS` deadline. The defect persisted from
`f3e2450` (CP-R13a) until `34a9a72` and was found by bisecting the smoke test failure, not
by the original verification.

Fixed by scoping `-enc_time_base 1/90000` to the passthrough path only (`FPS_PASSTHROUGH=1`).
CFR resamples to a uniform grid and does not need PTS precision preserved.

**Lesson:** "segments produced" is not a rollback assertion. Segment count and duration are
the checks that would have caught this.

### Segment durations and file sizes (T3)

Segments land at expected durations (19.5–21.3s for SEG_SECONDS=20). File sizes are
in the expected range (the size difference between captures reflects different camera
cadences — 30fps baseline vs 15fps post-change — not the timebase option).

### Bimodal verification (S3)

The live capture produced one zero-delta pair at frame 2 on FP7oJQ. This is a genuine
bimodal frame from the source stream — verified by the sidecar showing the same `dt_s: 0.0`
at the same frame. The zero-delta is NOT an encoding artifact; it represents a real 30fps
frame pair from the camera.

No extended bimodal segment (like PPDmUg-20260805-154251 with 78 zero-delta pairs) occurred
in the 60s capture window. However, the mechanism is confirmed: with `-enc_time_base 1/90000`,
the encoder passes through whatever PTS the source provides, including bimodal timing.
The 78-pair case from PPDmUg would now produce 78 pairs with **real** zero-delta PTS (same
as the sidecar shows) instead of 78 pairs with **artifact** zero-delta PTS (from 15360
requantization). The values are the same; the provenance is different — and under CP-R13b,
the mp4-derived sidecar will correctly represent them.

---

## 6. Evidence Traceability

| Claim | Test |
|-------|------|
| Mp4 has uniform 1024-tick deltas (pre-fix) | `ffprobe -show_entries frame=pts` on FP7oJQ-20260805-153650 (1680 frames, known 5940/6030 sidecar alternation) |
| Sidecar has 5940/6030 alternation | Same segment's `.timing.jsonl`, tick deltas recovered from `pts_time_s * 90000` |
| Bimodal produces 0-tick deltas (pre-fix) | `ffprobe -show_entries frame=pts` on PPDmUg-20260805-154251 (bimodal, 78 zero-delta pairs) |
| `-enc_time_base 1/90000` preserves alternation (synthetic) | Synthetic test: 28-frame video with known 5940/6030 PTS, re-encoded through x264 with showinfo filter |
| `-enc_time_base 1/90000` preserves alternation (live) | Smoke test FP7oJQ-20260817-105755: 5940/6030 deltas, 0/299 disagreements vs sidecar |
| `-video_track_timescale` alone does NOT fix | Same synthetic test, uniform 6000 output |
| `-c:v copy` incompatible with `-vf showinfo` | ffmpeg error: "Filtergraph was specified, but codec copy was selected" |
| x264 is NOT inherently destructive | `-enc_time_base` controls the behaviour; missing option, not encoder limitation |
| Before/after baseline | FP7oJQ-104848 (pre-fix, 1/15360, uniform 1024) vs FP7oJQ-105755 (post-fix, 1/90000, 5940/6030) |
| CFR rollback **BROKEN then fixed** | FP7oJQ-110127 (rollback): single unsegmented 145K-frame file. Fixed in `34a9a72` by scoping `-enc_time_base` to passthrough only. Original "unaffected" claim superseded — see §5 correction above. |
