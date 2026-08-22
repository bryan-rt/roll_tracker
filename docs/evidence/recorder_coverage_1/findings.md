# RECORDER-COVERAGE-1: Content delivery lag, not inter-segment gaps

**Camera:** FP7oJQ
**Session:** 2026-08-19 (CP-R8 capture)
**Recorder:** `diag_v8.sh` with SOURCE_PTS=1, FPS_PASSTHROUGH=1 (post-RELIABILITY-1/2)
**Generated:** 2026-08-22. Corrected 2026-08-22 (RECORDER-BACKLOG-1).

## Summary

The recorder captures every frame it receives. The Nest cloud relay delivers at
0.10–0.53× real-time (ffmpeg `speed=` field), so a wall-clock-bounded window ends
with the relay still behind: a 30-minute window yielded ~19 minutes of continuous
footage from the session's start. **Content is contiguous across all segment boundaries
within an attempt** (verified visually: last frame of segment N and first frame of
segment N+1 show the same scene with motion continuing). The single discontinuity is
at the attempt-1 → attempt-3 boundary, where the stream was genuinely down and the
relay backlog was lost on reconnect. What was lost is the session's *tail*, never
pulled — not footage missing between files.

## The root cause: `-t` computed from wall-clock remaining

The ffmpeg `-t` flag controls **output content duration**, not wall-clock duration.
Under arrival-PTS (pre-passthrough), content time equalled wall-clock time, so
`-t "$(( DEADLINE - now ))"` was accidentally correct. Passthrough decoupled them:
content arrives at <1× real-time, so the `-t` value was the right *kind* of signal
(content seconds) but the wrong *amount* (shrunk by wall time already consumed by
previous attempts and backoff, not by content time captured).

The expression is unchanged since `745b1b4` (March 2026). The defect was latent under
arrival-PTS and became active when passthrough was enabled.

## The "gap" metric is lag, not loss

The wall-clock interval between consecutive segment filenames (e.g., 459 seconds
between `200827` and `201606`) is the time it took to accumulate 120 seconds of
content at sub-real-time delivery. It is not a gap in the footage. The PTS span
within each segment is continuous, and the content PTS across segment boundaries is
continuous. Naming segments by `-strftime 1` (wall-clock at file open) makes the lag
visible in filenames — it is not a recording defect.

## The March recorder was not better

Under arrival-PTS, segments cut on arrival time and filenames marched at tidy
intervals because content time = wall-clock time (by the same accidental equivalence
that made `-t` correct). The tidy spacing was the artifact — the same underlying
content arrived at the same sub-real-time rate, padded with duplicate frames by the
CFR encoder filling a uniform grid from bursty input. The "gaps" were always there;
they were hidden by the timestamp domain being the same as the clock domain.

## Reconnect loses the backlog

On reconnect, the Nest cloud relay does not resume where it stopped — it jumps
forward to the live edge. The one observed discontinuity (attempt 1 → attempt 3) is
at a reconnect boundary. Content before the reconnect is retained; content during the
dead period is genuinely lost and unrecoverable. This is a load-bearing constraint on
the recorder design: reconnects are expensive in content, not just in time. Each
reconnect also resets the delivery rate to ~0.10× (TCP congestion control / relay
buffer ramp), requiring a fresh climb.

## Delivery rate observations (ffmpeg `speed=` field)

| Attempt | Wall time | Content | Start speed | End speed | Notes |
|---------|-----------|---------|-------------|-----------|-------|
| 1 | 911s | 269s | 0.15× | 0.30× | Still climbing at termination |
| 2 | 11s | 0s | — | — | Reuse of expired session |
| 3 | 1634s | 864s | 0.10× | 0.53× | Still climbing at termination |

Both attempts show a monotonically rising trajectory that had not plateaued when they
ended. 0.53× is where attempt 3 was stopped, not a measured ceiling. The Q2 long-pull
test (RECORDER-BACKLOG-1 §Q2) may show materially better yield.

**Retracted figure:** An earlier analysis reported "0.26–0.78×" delivery rates. The
0.78× was derived from per-segment content-over-wall-clock (PTS span / inter-segment
wall-clock interval) rather than ffmpeg's `speed=` field. That metric conflates lag
with rate and should not be used. All delivery rate figures should use the ffmpeg
`speed=` field (cumulative content / cumulative wall time).

## Per-segment table

| # | Segment | Att | Frames | PTS span (s) | Wall gap to next (s) |
|---|---------|-----|--------|-------------|----------------------|
| 1 | 200827 | 1 | 1,980 | 120.7 | 338 |
| 2 | 201606 | 1 | 1,950 | 120.4 | 222 |
| 3 | 202148 | 1 | 548 | 28.2 | 100 |
| 4 | 202356 | 3 | 1,740 | 121.2 | 155 |
| 5 | 202832 | 3 | 1,710 | 118.9 | 113 |
| 6 | 203224 | 3 | 1,680 | 120.5 | 108 |
| 7 | 203612 | 3 | 1,680 | 120.7 | 141 |
| 8 | 204034 | 3 | 1,890 | 118.8 | 149 |
| 9 | 204502 | 3 | 1,620 | 119.9 | 42 |
| 10 | 204744 | 3 | 1,680 | 121.3 | 51 |
| 11 | 205036 | 3 | 321 | 23.1 | — |

Wall gaps are **lag** (time to accumulate the next 120s of content at sub-real-time
delivery), not loss. The decreasing gap size in attempt 3 (155 → 42s) reflects the
delivery rate climbing from 0.10× toward 0.53×.

## Derivation method

- **Segment duration** = `pts_time_s` of the last frame (camera-clock time from sidecar).
- **Wall gap** = next segment's `segment_start_epoch` − (this segment's
  `segment_start_epoch` + this segment's PTS span). Measures delivery lag.
- **Delivery speed** = ffmpeg's `speed=` field in stderr progress lines (cumulative
  content / cumulative wall time).

## SDM quota impact for longer runs

Extend calls occur every ~180s (1 per 3 min). Even at 3 cameras × 150 min content
at 0.25× delivery (600 min wall time), total API rate is ~1 QPM — well under the
10 QPM quota.

## Fix applied (RECORDER-BACKLOG-1)

`TARGET_CONTENT_SECONDS` mode: ffmpeg `-t` is set to the lesser of content remaining
and wall-clock remaining. Content accounting persists across attempts. Wall-clock
safety cap (default 5× target) prevents unbounded runs. Legacy `WINDOW_SECONDS` mode
preserved for `smoke_test.sh` and backward compatibility.
