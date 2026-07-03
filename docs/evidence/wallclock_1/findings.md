# WALLCLOCK-1: Per-Frame Timing Signal Feasibility

**Date:** 2026-07-03
**Verdict: TIMESTAMPS ARE SYNTHETIC — wallclock approach NOT viable.**

## Death-Check Result: FAIL (SYNTHETIC)

### Container PTS (primary source)

Target clip: `FP7oJQ-20260318-200014.mp4` (4530 frames, 151s, 30fps)

Inter-frame delta distribution across ALL 4530 frames:

| Metric | Value |
|--------|-------|
| Min delta | 0.033333000 s |
| Max delta | 0.033334000 s |
| Mean delta | 0.033333333 s |
| Stdev delta | 0.000000471 s |
| Unique deltas (6dp) | {0.033333, 0.033334} |

The two "unique" deltas differ by 1 microsecond — floating-point representation of
exactly 1/30s. **Zero jitter. Zero anomaly. Dead flat.**

### Observed-lag windows — NO timing anomaly detected

| Window | Frames | Unique deltas |
|--------|--------|---------------|
| ~2s (1.9–2.2) | 9 | {0.033333, 0.033334} |
| ~4s (3.8–4.2) | 12 | {0.033333, 0.033334} |
| ~6s (5.8–6.2) | 12 | {0.033333, 0.033334} |
| ~7s (6.8–7.2) | 12 | {0.033333, 0.033334} |
| ~9s (8.8–9.2) | 12 | {0.033333, 0.033334} |
| ~10-11s (9.8–11.2) | 42 | {0.033333, 0.033334} |

Every observed-lag location has identical timing to the rest of the clip.
The timestamps carry NO information about the lag events.

### Raw mp4 stts/ctts atoms

**stts (sample-to-time):** Single entry — `count=4530, delta=512`. ALL 4530 frames
have identical sample duration. At timescale 15360 (confirmed in ffmpeg stderr:
`15360 tbn`), delta=512/15360 = 0.033333... s = exactly 1/30s. A single-entry stts
is the canonical marker of constant-framerate encoding.

**ctts (composition time offsets):** 4374 entries with offsets {0, 512, 1024, 2560}.
These are B-frame reordering offsets (IBB pattern), NOT capture timing variation.
All values are multiples of the sample duration (512).

### Nest SDM metadata

Directory contains 11 JSON files (`generate_1.json`, `extend_*.json`). Contents:
RTSP stream URLs, stream extension tokens, expiration timestamps. **Session-level
stream management metadata only — no per-frame timing information.**

No other sidecar files with per-frame data exist in the clip directory.

### Source of the problem: recorder re-encodes to CFR

The recorder (`services/nest_recorder/recorder/diag_v6.sh`) captures via:

```
ffmpeg ... \
  -use_wallclock_as_timestamps 1 -fflags +genpts+igndts \
  -i <RTSP_URL> \
  -c:v libx264 -preset veryfast -crf 23 -g 30 -keyint_min 30 \
  -f segment -segment_time 150 -reset_timestamps 1 \
  ...
```

Key flags and their effect on timing:

| Flag | Effect |
|------|--------|
| `-use_wallclock_as_timestamps 1` | Replaces source PTS with local wall-clock at packet reception (INPUT side only) |
| `-fflags +genpts+igndts` | Generates PTS from scratch, ignores source DTS |
| `-c:v libx264` | Re-encodes video → output written at constant 30fps |
| `-reset_timestamps 1` | Resets timestamps to 0 at each segment boundary |

The wall-clock IS available at capture time (`-use_wallclock_as_timestamps 1`),
but it is consumed by ffmpeg internally for input synchronization and then
**discarded** when libx264 writes the output at constant framerate. The output stts
contains no trace of the original timing.

### How lag manifests in the output

Since the output is constant-framerate, camera lag manifests as:
- **Pause:** Duplicate/near-identical frames (ffmpeg receives no new video packets
  during the pause but continues emitting frames at 30fps, repeating the last decoded
  frame or interpolating)
- **Speed-up:** Dropped frames (ffmpeg receives a burst of backed-up frames and
  drops excess to maintain 30fps output rate)

In both cases, the CONTENT changes but the TIMESTAMPS remain perfectly uniform.
The lag signal is in the pixel domain (motion shape), not the timing domain.

### Audio DTS anomalies (secondary observation)

The ffmpeg stderr shows non-monotonic DTS warnings on the audio stream:
```
Non-monotonic DTS; previous: 4599, current: 4597; changing to 4600
Non-monotonic DTS; previous: 12989, current: 12987; changing to 12990
```

These ARE evidence of timing irregularities in the RTSP source, but:
1. They're on the AUDIO stream, not video
2. They were "corrected" by ffmpeg (forced monotonic)
3. The corrected values are what's written to the output
4. The warning messages themselves are in stderr (runtime log), not in the mp4

This is a breadcrumb, not a usable signal.

## Cross-Camera Consistency

All three cameras use the same recorder (v6) and produce identical timing structure:

| Camera | Clip | stts entries | delta | nb_frames | fps |
|--------|------|-------------|-------|-----------|-----|
| FP7oJQ | 200014 | 1 | 512 | 4530 | 30/1 |
| J_EDEw | 200015 | 1 | 512 | 4530 | 30/1 |
| PPDmUg | 200019 | 1 | 512 | 4530 | 30/1 |

All clips: single stts entry, uniform delta, constant 30fps.
The synthetic-timestamp behavior is consistent across all capture paths.

## Timing Sources Summary

| Source | Exists? | Real timing? | Notes |
|--------|---------|-------------|-------|
| Container PTS/DTS | Yes | **No** — synthetic | Dead-flat 33.333ms, zero jitter |
| stts atom | Yes | **No** — single uniform entry | count=4530, delta=512 |
| ctts atom | Yes | **No** — B-frame reordering | IBB offsets, not capture timing |
| Nest SDM JSON | Yes | **No** — session-level only | Stream tokens, no per-frame data |
| ffmpeg stderr | Yes | **Partial** — audio DTS warnings | Timing irregularity breadcrumb, not usable |
| RTSP RTP timing | Existed at capture | **Discarded** by re-encode | `-use_wallclock_as_timestamps` consumes it |

**All sources agree or are silent. No source diverges with real timing.**

## Correction Factor Estimate

N/A — cannot compute true/nominal delta ratio because no true timing signal exists.

## Verdict

**The wallclock approach is NOT viable for existing footage.**

Per-frame timestamps are synthetic (idealized 1/30s uniform). They carry zero
information about camera lag events. The lag that the user observes is real, but it
manifests as frame content changes (duplicate frames during pause, dropped frames
during speed-up), not as timing anomalies.

**Recommendation: fall back to motion-shape discrimination.** Specifically:
- **Duplicate-frame detection** (pixel-level or perceptual hash similarity between
  adjacent frames) can identify pause events
- **Optical flow magnitude** can identify speed-up events (anomalously high apparent
  motion from dropped intermediate frames)
- These are per-frame, content-based signals — they work regardless of timestamp quality

**Future recorder improvement (separate work):** The recorder COULD preserve real
timing by using `-c:v copy` (stream copy, no re-encode) or `-vsync vfr` (variable
framerate output). This would write non-uniform stts entries reflecting actual capture
intervals. But this is a recorder change that would only affect FUTURE footage, not
the existing corpus.
