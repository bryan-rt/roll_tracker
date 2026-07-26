# CAPTURE-TIME-2: Multi-Camera Timing Diagnostic

**Date:** 2026-07-26
**Cameras:** J_EDEw (failed — relay 404), FP7oJQ, PPDmUg
**Verdict: Source PTS carries true capture timestamps. Current stream fps is 15fps
(not 30fps as in July 3 J_EDEw test). RTCP is absent across all transports. Tier-2
alignment (source PTS + host-clock lower envelope) achievable at ~14-56ms per-camera
uncertainty.**

## Diagnostic Module Built

`services/nest_recorder/recorder/diag_timing.sh` — parallel module, does NOT modify
the production recorder. Capabilities:

1. Concurrent multi-camera capture with source PTS (no wallclock override)
2. Per-frame (source_PTS, host_arrival_time) pairs via $EPOCHREALTIME timestamped stderr
3. Lower-envelope PTS→wallclock offset estimation with windowed drift check
4. RTCP hunt across TCP and UDP transports
5. Full SDM response capture + session manifest

## Smoke Test Results (60s, all 3 cameras)

### Source PTS Verification

| Camera | Frames | Duration | Measured FPS | Delta Median | % Uniform | Per-sec stdev |
|--------|--------|----------|-------------|-------------|-----------|---------------|
| FP7oJQ | 867 | 57.73s | 13.85 | 67.00 ms | 84.5% | 0.00 |
| PPDmUg | 880 | 58.60s | 15.00 | 67.00 ms | 99.8% | 0.00 |
| J_EDEw | — | — | — | — | — | — (relay 404) |

**PPDmUg: Source PTS is true capture time at 15fps.** 99.8% of deltas in 60-70ms,
exactly 15 frames per second, every second. Zero per-second variance.

**FP7oJQ: Source PTS is true capture time at ~15fps** with some jitter. 84.5% in
60-70ms, 7.7% at 100-150ms (network-induced PTS stretching). Per-second counts are
perfectly 15 — the jitter is within-second frame spacing, not frame dropping.

### FPS Discovery: 15fps, not 30fps

CAPTURE-TIME-1 (Jul 3) showed J_EDEw at 30fps with source PTS. Today (Jul 26),
FP7oJQ and PPDmUg both deliver at **15fps** with source PTS. The SDP still advertises
`a=framerate:30.0`, but the actual stream delivers at 15fps.

Possible explanations:
- Camera-specific: different Nest cameras may have different quality settings
- Time-varying: Nest may have changed the stream quality between Jul 3 and Jul 26
- Session-specific: quality may vary by network conditions or relay load

**The measured fps per clip is the reliable number.** Do not hardcode 30 or 15 — read
it from the source PTS or ffprobe.

### Lower-Envelope Offset + Drift

| Camera | Offset (s) | Drift Rate (ppm) | Drift Flat? | Windowed Spread |
|--------|-----------|------------------|-------------|-----------------|
| FP7oJQ | 1785106319.903 | -603.0 | NO ⚠ | 56.19 ms |
| PPDmUg | 1785106323.668 | 95.5 | YES ✓ | 13.64 ms |

**PPDmUg: Flat drift (95 ppm), 14ms windowed spread.** Lower envelope is stable.
Alignment holds to ~14ms over 60s.

**FP7oJQ: Trending drift (-603 ppm), 56ms windowed spread.** The host clock and
camera clock are drifting at 0.6ms/s. Over 5 minutes, this accumulates to ~181ms of
alignment degradation. The linear fit captures this — consumers should apply the drift
correction or re-anchor periodically.

**Cross-camera offset spread: 3765ms.** This is dominated by the staggered start
(cameras started 2-4s apart). The offsets correctly reflect when each camera's stream
began in real time — applying each camera's own offset to its source PTS produces
absolute times that can be compared across cameras.

### RTCP Verdict

| Camera | TCP (trace) | UDP (trace) |
|--------|------------|-------------|
| J_EDEw | 0 mentions | Connection refused |
| FP7oJQ | 0 mentions | Connected, 0 mentions |
| PPDmUg | 0 mentions | Connected, 0 mentions |

**RTCP sender reports NOT available under any transport.**

- TCP (interleaved): no RTCP mentions in ffmpeg trace output (3 cameras tested)
- UDP: J_EDEw relay refuses UDP connection. FP7oJQ and PPDmUg connected but no
  RTCP packets observed.

**Absolute camera clock is NOT available from the stream.** Cross-camera sync
must rely on Tier-2: source PTS + host-clock lower-envelope anchors.

### Tier-2 Alignment Uncertainty

Per-camera alignment precision (bounded by lower-envelope windowed spread):
- PPDmUg: ±14ms (flat drift, stable)
- FP7oJQ: ±56ms (trending drift, correctable to ~14ms with linear fit)

Cross-camera alignment residual = differential relay latency (unknown from timing
alone). The world-coordinate validation (user's live capture) will measure this
directly by comparing projected positions of a person visible in two cameras.

## User Instructions for Live Session

```bash
# Start container
cd services/nest_recorder
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d recorder

# Run 5-min diagnostic during a session with a person visible in 2+ cameras
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'WINDOW_SECONDS=300 /app/diag_timing.sh'

# Clips land at data/raw/nest/diag/timing_<TS>/
# Analyze timing
cd /path/to/roll_tracker
source .venv/bin/activate
python tools/analyze_capture_timing.py analyze-session \
  data/raw/nest/diag/timing_<TS>/

# Feed clips through pipeline for world-coordinate alignment analysis
# (each camera's subdir has pipeline-compatible CFR mp4 + timing sidecar)
```

## Artifacts

| File | Contents |
|------|----------|
| `services/nest_recorder/recorder/diag_timing.sh` | Diagnostic module (parallel, no prod change) |
| `tools/analyze_capture_timing.py` | Session analysis tool |
| `docs/evidence/capture_time_2/findings.md` | This document |

Smoke test session: `data/raw/nest/diag/timing_20260726-185152/`
