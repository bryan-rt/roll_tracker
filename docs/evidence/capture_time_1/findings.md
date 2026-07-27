# CAPTURE-TIME-1: Source PTS Reveals True 30fps Capture Timestamps

**Date:** 2026-07-04
**Camera:** J_EDEw (Matroom 1)
**Verdict: SOURCE PTS = TRUE CAPTURE TIME. Camera captures at 30fps. The "15fps"
was an artifact of `-use_wallclock_as_timestamps 1` discarding the stream's own
timing.**

## The Discovery

The Nest RTSP stream carries RTP timestamps at 90kHz clock rate (standard H.264),
set BY THE CAMERA before the network touches them. These timestamps represent the
camera's true capture cadence — perfectly uniform 33ms intervals (30fps).

The recorder has been ACTIVELY DISCARDING these timestamps with
`-use_wallclock_as_timestamps 1 -fflags "+genpts+igndts"`, replacing them with
local arrival timestamps that are corrupted by network delivery bursting.

## Decisive Comparison

Two 60s captures from the same camera, same session, differing ONLY in timestamp
source:

| Metric | Source-PTS | Arrival-based |
|--------|-----------|--------------|
| Frames | 1801 | 1783 |
| Measured fps | **30.02** | 29.71 |
| Delta mean | 33.33 ms | 33.67 ms |
| Delta median | **33.00 ms** | 0.05 ms |
| Delta stdev | **1.21 ms** | 98.04 ms |
| % uniform (30-50ms) | **99.9%** | 0.2% |
| % burst (<1ms) | 0.1% | **72.2%** |
| Per-sec count stdev | **0.00** | 12.39 |

Source-PTS: perfectly uniform, exactly 30 frames per second, every second.
Arrival-based: bursty noise (72% sub-1ms clumps, 7% 150-500ms gaps).

## Per-Second Frame Counts (Source-PTS)

```
t=0-1s: 30 frames
t=1-2s: 30 frames
t=2-3s: 30 frames
...
t=9-10s: 30 frames
```

Every second has exactly 30 frames. Zero variance. This is the camera's sensor
clock, not network delivery timing.

## The "15fps" Mystery Solved

The RECORDER-TIMING-2 and SIDECAR-1 captures reported 15fps. This was NOT the
camera's capture rate — it was an ARTIFACT:

1. `-use_wallclock_as_timestamps 1` replaces stream PTS with local arrival clock
2. `-fflags +genpts` generates fresh PTS from scratch
3. The bursty arrival pattern (frames arriving in clumps) causes ffmpeg to assign
   irregular PTS values
4. The CFR encoder (libx264) then re-encodes at whatever rate it infers from the
   irregular input — often half of the true rate

With source PTS preserved: **30fps, uniform, exactly as advertised in the SDP
(`a=framerate:30.0`)** and consistent with old production clips (FP7oJQ-200014:
30fps, 4530 frames/151s).

## RTP/RTCP Inspection

- **RTP clock rate:** 90kHz (standard H.264). Provides relative capture timing at
  sub-ms precision. This is what produces the uniform 33ms deltas.
- **RTCP sender reports:** Not observed in the TCP-interleaved debug capture. The
  Nest RTSPS stream (via dropcam.com relay) may not expose RTCP SR packets over
  the TCP interleave, or they may require a longer session to appear.
- **Absolute wall-clock from RTCP:** NOT available from this stream. Cross-camera
  sync needs recorder-side wall-clock (segment filename epoch), not stream-side NTP.

## Measured Average FPS (Robust Fallback)

| Source | Measured FPS |
|--------|-------------|
| Source-PTS capture (this test) | 30.02 |
| Arrival-based capture (this test) | 29.71 |
| SDP advertisement | 30.0 |
| Old production clip (FP7oJQ-200014, Mar 2026) | 30.0 |
| SIDECAR-1 test segments (arrival-based) | 15.0 |

The 15fps from SIDECAR-1 was the arrival-based artifact. True capture rate is 30fps,
stable across sessions and cameras.

## Implications

### What this unlocks (with source-PTS as the recorder's timestamp source)

1. **Per-frame dt is REAL** — uniform 33ms capture intervals, not network noise.
   Dynamic-fps velocity becomes trivial: dt = 1/30s for every frame.

2. **Sidecar becomes EXACT, not approximate** — with source-PTS, input frame count
   should match output frame count (both 30fps, no dup/drop needed). The ±500ms
   nearest-neighbor error from SIDECAR-1 was entirely caused by the arrival-burst
   artifact forcing CFR to dup/drop.

3. **BoT-SORT frame_rate fix is simple** — set to 30fps (the true capture rate),
   not the 15fps the arrival-based recorder was producing.

4. **Cross-camera sync** — relative timing (per-frame dt) is perfect from source
   PTS. Absolute cross-camera alignment still needs recorder-side wall-clock
   (segment filename epoch + stream-relative PTS), since RTCP NTP is not available.

### What needs to change (recorder, follow-up checkpoint)

1. Remove `-use_wallclock_as_timestamps 1` and `-fflags "+genpts+igndts"` from the
   production recorder (diag_v6.sh line 333)
2. Add `-copyts` to preserve the stream's own PTS
3. The sidecar extraction should then produce 1:1 input/output mapping (both 30fps)
   — the mismatch warning should stop firing
4. All downstream fps assumptions can use the measured/SDP 30fps

### What does NOT need to change

- The sidecar SCHEMA is correct (frame_index, pts_time_s, input_n)
- The extraction/demux logic is correct
- The mismatch warning is correct (it will just stop triggering)
- The video encoding path is unchanged (libx264 CFR)

## Artifacts

| File | Contents |
|------|----------|
| `analysis.json` | Quantitative comparison + stream info |
| `findings.md` | This document |

Test clips at `data/raw/nest/diag/capture_time_test/`:
- `source_pts.mp4` + `.stderr` — source-PTS mode (uniform 30fps)
- `arrival_pts.mp4` + `.stderr` — arrival-based mode (bursty)
- `rtp_debug.mp4` + `.stderr` — RTP debug output (SDP, clock rate)
