# RECORDER-COVERAGE-1: Inter-segment gap observation

**Camera:** FP7oJQ
**Session:** 2026-08-19 (CP-R8 capture)
**Recorder:** `diag_v8.sh` with SOURCE_PTS=1, FPS_PASSTHROUGH=1 (post-RELIABILITY-1/2)
**Generated:** 2026-08-22

## Derivation method

All figures use the **PTS-span method:**
- **Segment duration** = `pts_time_s` of the last frame (time from frame 0 to the final
  frame, read from the sidecar). This is camera-clock time, not frame-count × nominal_dt.
- **Elapsed time** = `segment_start_epoch` of the last segment + its PTS span −
  `segment_start_epoch` of the first segment. Runs from the first segment's start to
  the last segment's end.
- **Gap** = next segment's `segment_start_epoch` − (this segment's `segment_start_epoch` +
  this segment's PTS span).
- **Coverage** = sum of segment durations / elapsed time.

The `frames × nominal_dt_s` method gives a different total (1,125s vs 1,134s) because
`nominal_dt_s` is a median-based reference, not the per-frame sum. PTS span is the
authoritative clock. All figures below use PTS span.

## Headline

- **Elapsed:** 2,552s (42m 32s)
- **Total footage:** 1,134s (18m 54s)
- **Coverage: 44.4%**
- **Inter-segment gaps:** 42–338s within healthy attempts

## Per-segment table

| # | Segment | Att | Frames | PTS span (s) | Gap to next (s) |
|---|---------|-----|--------|-------------|-----------------|
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
| 11 | 205036 | 3 | 321 | 23.1 | -- |

## Attempt structure

| Attempt | Segments | Footage (s) | Elapsed (s) | Coverage | Notes |
|---------|----------|------------|------------|----------|-------|
| 1 | 3 | 269 | 829 | 32.5% | Healthy; died after 202148 (548-frame tail) |
| 2 | 0 | 0 | ~100 | 0% | Dead (~100s gap between att1 end and att3 start) |
| 3 | 8 | 864 | 1,623 | 53.3% | Healthy; ran to end of window |

Attempt 3 ran 1,623s healthy and produced 864s of footage. The remaining 759s is
distributed across 7 inter-segment gaps (mean 108s, range 42–155s). These gaps occur
**within a single healthy attempt** — they are not reconnection events.

## The observation

Segments are approximately 2 minutes of footage followed by a 1–5 minute gap, repeating.
This pattern is consistent across both healthy attempts. The gaps are:

- Not confined to attempt boundaries (7 of 10 gaps are within attempt 3)
- Not correlated with recording failures (attempt 3 was healthy throughout)
- Regular enough to suggest a systematic cause (not random network drops)

## Hypothesis: passthrough recorder introduced the gaps (UNTESTED)

The pre-passthrough recorder (arrival-PTS, CFR re-encode, roughly two months prior) is
reported not to have produced inter-segment gaps of this magnitude. It collected frames
poorly in other respects (bursty dup/drop from arrival timestamps, 35% input/output
mismatch in one case), but segment-to-segment continuity was better.

**This is a report from memory, not a measurement.** The comparison has not been run.
Validating it requires:

1. Running the pre-passthrough recorder (SOURCE_PTS=0 FPS_PASSTHROUGH=0) on the same
   camera for a comparable window
2. Computing the same coverage metric on the resulting segments
3. Comparing gap patterns

If confirmed, the leading candidates for the mechanism are:
- The passthrough ffmpeg option combination (`-copyts`, `-fps_mode passthrough`,
  `-enc_time_base 1/90000`) interacting with the segment muxer differently than CFR mode
- The segment duration target interacting with VFR timing boundaries
- An ffmpeg version or configuration change concurrent with the passthrough switch

**Do not assume the hypothesis is correct.** The pre-passthrough recorder had its own
problems (RECORDER-RELIABILITY-1/2 fixed five bugs, several of which affected segment
production). The comparison must be apples-to-apples on the current recorder codebase
with only the PTS mode toggled.

## Consequence for annotation

Every capture taken before this is understood loses more than half the session. Annotating
42%-coverage footage from a recorder about to change risks doing expensive CVAT work twice.
This is why annotation (objective 6) is sequenced after the recorder investigation
(objective 1) in the checkpoint-2 execution plan.
