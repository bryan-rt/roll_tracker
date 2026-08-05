# CP-R5: Sidecar Segment Boundary Fix

**Date:** 2026-08-05
**Code state:** post-`193d011` (CP-R4)

## The defect

`extract_timing_sidecars()` split ffmpeg stderr into per-segment ranges by **log-line
position**: `Opening...mp4...for writing` markers from the **muxer**, sliced against
`Parsed_showinfo` lines from the **filter graph**. The muxer lags, so frames belonging to
segment N+1 were attributed to segment N.

## The fix: PTS-based slicing via cumulative duration boundaries

1. `ffprobe -show_entries format=duration` for each segment mp4, cumulative-sum to get
   elapsed-time boundaries within the attempt.
2. Anchor at the first showinfo PTS tick in the attempt.
3. Single-pass awk assigns each showinfo line to the segment whose cumulative-duration
   window contains its elapsed PTS.
4. Final segment absorbs everything from its start boundary onward.

Works identically for passthrough and CFR: under `SOURCE_PTS=0`, showinfo PTS are arrival-
wallclock and segment durations are wallclock, so elapsed-time bucketing aligns regardless
of the input/output count ratio. Burstiness affects the value of `measured_fps`, not the
monotonicity of elapsed time.

**Opening markers** are still used for segment DISCOVERY (which mp4 files were created) —
this is reliable. Only their LINE NUMBERS are no longer used as split points.

## Anchor pre-check

Per attempt, the PTS span (last - first showinfo PTS) is compared against the sum of
segment durations. These should agree within 0.2s (about 2 frames at 15fps):

| Attempt | PTS span (s) | Duration sum (s) | Difference |
|---------|-------------|------------------|-----------|
| PPDmUg h07 | 232.866 | 232.816 | +0.050s |
| FP7oJQ h06 | 419.000 | 419.020 | -0.020s |

Both within tolerance — the anchor is valid.

## Validation: archived DUPFIX segments

Convention: residual = output - input (DUPFIX-2's convention; negative = real drops).

All four multi-segment attempts re-measured. Single-segment attempts (FP7oJQ h07, FP7oJQ
h29) not re-measured — the boundary split only affects multi-segment attempts.

### PPDmUg h07 attempt_1

| Segment | nb_frames | old showinfo | old residual | new showinfo | new residual |
|---------|-----------|-------------|-------------|-------------|-------------|
| PPDmUg-20260728-070219 | 1830 | 1965 | **-135** | 1939 | **-109** |
| PPDmUg-20260728-070422 | 1660 | 1630 | **+30** | 1660 | **+0** |
| **TOTAL** | **3490** | **3595** | **-105** | **3599** | **-109** |

- seg1 now has **exactly** nb_frames showinfo lines (residual 0) — the +30 bias is eliminated.
- seg0 residual changes from -135 to -109: the 109 missing frames are real drops that occurred
  during seg0's recording period, now correctly attributed.
- **Leading edge recovered:** 4 lines (3599 - 3595). These were showinfo lines before the
  first `Opening` marker that the old method dropped.
- **Attempt total:** old -105 -> new -109. The 4 recovered leading-edge lines mean 4 more
  input frames counted, producing a 4-frame larger deficit. This is a correction, not a
  regression: the old -105 undercounted the real drop total by 4.

### FP7oJQ h06 attempt_1

| Segment | nb_frames | old showinfo | old residual | new showinfo | new residual |
|---------|-----------|-------------|-------------|-------------|-------------|
| FP7oJQ-20260728-062531 | 1830 | 1857 | **-27** | 1866 | **-36** |
| FP7oJQ-20260728-062734 | 1800 | 1799 | **+1** | 1795 | **+5** |
| FP7oJQ-20260728-062939 | 1800 | 1796 | **+4** | 1798 | **+2** |
| FP7oJQ-20260728-063135 | 806 | 777 | **+29** | 817 | **-11** |
| **TOTAL** | **6236** | **6229** | **+7** | **6276** | **-40** |

- Per-segment residuals shifted, some worsening individually. The arithmetic confirms
  redistribution with no frames lost or invented:
  - Sum of per-segment residuals = -40 = attempt total. **Check.**
  - Sum of per-segment input shifts (new - old) = +9 -4 +2 +40 = **+47** = leading edge. **Check.**
  - The 47 recovered leading-edge lines redistributed across segments: seg0 gained +9 input
    (residual -27 -> -36), seg1 lost -4 (residual +1 -> +5), seg2 gained +2 (residual +4 -> +2),
    seg3 gained +40 (residual +29 -> -11, sign flip — the old +29 was an artifact of the
    line-position split stealing 40 frames from seg3 to preceding segments).
- seg0's residual worsening (-27 -> -36) reflects 9 additional frames correctly assigned to
  it (from the leading edge), making its drop count more accurate, not less.
- seg3's sign flip (+29 -> -11) is the largest single correction: 40 input frames that
  belonged to seg3 were misattributed to earlier segments by the line-position method.
- **Leading edge recovered:** 47 lines (6276 - 6229). Significantly more than PPDmUg's 4.
  See "Mechanism investigation" for why the asymmetry — these are the same measurement as
  the muxer lag (the gap between filter-graph output and muxer file creation).
- **Attempt total:** old +7 -> new -40. The old +7 was misleading — it appeared that the
  attempt gained 7 frames, but this was an artifact of dropping 47 leading-edge lines. The
  true deficit is -40 (40 real frame drops). **DUPFIX-2's P2 ("redistributive, not lossy")
  is corrected: FP7oJQ h06 attempt_1 IS lossy.** See correction in
  `docs/evidence/recorder_dupfix_1/findings.md`.

### PPDmUg h07 attempt_2

| Segment | nb_frames | old showinfo | old residual | new showinfo | new residual |
|---------|-----------|-------------|-------------|-------------|-------------|
| PPDmUg-20260728-070849 | 1800 | 1905 | **-105** | 1879 | **-79** |
| PPDmUg-20260728-071051 | 1800 | 1800 | **+0** | 1800 | **+0** |
| PPDmUg-20260728-071250 | 1800 | 1798 | **+2** | 1800 | **+0** |
| PPDmUg-20260728-071450 | 1800 | 1803 | **-3** | 1800 | **+0** |
| PPDmUg-20260728-071650 | 1800 | 1797 | **+3** | 1800 | **+0** |
| PPDmUg-20260728-071850 | 1156 | 1127 | **+29** | 1155 | **+1** |
| **TOTAL** | **10156** | **10230** | **-74** | **10234** | **-78** |

- Mid-segments (seg1-seg4): all four go to **exact zero** residual. The PTS-based split
  eliminates the ±2/±3 jitter that the line-position method produced.
- Leading edge: 4 lines (10234 - 10230).
- Attempt total: old -74 -> new -78.

### PPDmUg h07 attempt_3

| Segment | nb_frames | old showinfo | old residual | new showinfo | new residual |
|---------|-----------|-------------|-------------|-------------|-------------|
| PPDmUg-20260728-072126 | 1830 | 1856 | **-26** | 1838 | **-8** |
| PPDmUg-20260728-072328 | 1800 | 1800 | **+0** | 1800 | **+0** |
| PPDmUg-20260728-072528 | 1642 | 1616 | **+26** | 1643 | **-1** |
| **TOTAL** | **5272** | **5272** | **+0** | **5281** | **-9** |

- The "exactly 0" conservation was an artifact: 9 leading-edge lines were dropped by the
  old method. True deficit: **-9** (9 real frame drops).
- Mid-segment (seg1): unchanged at zero.

### Summary: loss is universal, magnitude varies

**DUPFIX-2's finding that "two attempts conserve" is corrected.** Every multi-segment attempt
shows real frame loss under the PTS-based split:

| Attempt | Old deficit | New deficit | Leading edge |
|---------|-----------|-----------|-------------|
| FP7oJQ h06 att_1 | +7 | **-40** | 47 |
| PPDmUg h07 att_1 | -105 | **-109** | 4 |
| PPDmUg h07 att_2 | -74 | **-78** | 4 |
| PPDmUg h07 att_3 | +0 | **-9** | 9 |

Loss is present in every attempt. The "attempt-specific conservation" framing was an
artifact of the old split dropping leading-edge lines, masking small deficits as zero.

## Mechanism investigation: leading edge = muxer lag (one measurement, not two)

The "leading edge" (showinfo lines before the first Opening marker) and the "muxer lag"
(delay between filter-graph output and muxer file creation) are the **same gap** viewed
from two perspectives. Both measure the interval between the filter graph starting to emit
showinfo and the segment muxer opening the first file.

| Camera | Leading-edge lines | Wall-clock lag | Mechanism |
|--------|-------------------|---------------|-----------|
| PPDmUg h07 | 4 | 0.27s | Consistent with encoder startup buffering |
| FP7oJQ h06 | 47 | 3.1s | NOT explained by encoder buffering alone |

The order-of-magnitude asymmetry (4 vs 47) suggests a per-stream or per-session variable,
not a fixed pipeline delay. The 3.1s on FP7oJQ cannot be x264 lookahead (`-preset veryfast`
= ~10-40 frames = 0.7-2.7s at 15fps). Candidates: `-max_muxing_queue_size 1024` filling
(1024 packets at 15fps = ~68s — far larger than needed, but queue fill rate depends on
how fast the encoder produces packets), frame-threading startup latency, or stderr write
interleaving between the filter graph and muxer output streams.

**Verdict:** The fix removes the dependency on `Opening` marker line position entirely.
The lag mechanism remains unexplained, the magnitude varies per camera (0.27s vs 3.1s),
and the asymmetry is not accounted for. Ruled out: x264 lookahead alone. Not ruled out:
muxing queue depth, frame-threading startup, stderr buffering interleave.

## Schema bump

`sidecar_schema` bumped from 2 to 3. The fix changes `input_frame_count` and `pts_time_s`
origin for affected segments. A consumer reading schema-2 and schema-3 sidecars for the
same footage will get different numbers. CP-R6 will define the contract formally; the
bump prevents silent semantic change under a stable version number.

## pts_time_s origin shift

The `base_pts` (first PTS tick assigned to each segment) shifts because different showinfo
lines are now assigned to each segment. The shift is proportional to the number of frames
that moved across boundaries:

| Segment | Frames shifted | Origin shift direction |
|---------|---------------|----------------------|
| PPDmUg seg0 | -26 (fewer frames from seg1 leaked in) | Earlier (correct) |
| PPDmUg seg1 | +30 (gained its own frames back) | Earlier |
| FP7oJQ seg0 | +9 (gained leading edge, lost seg1 leakage) | Earlier |
| FP7oJQ seg3 | +40 (gained its own frames back) | Earlier |

## Smoke test

Both modes passing post-fix:
- Default: 18 PASS, 0 FAIL, 0 WARN, 2 SKIP (FP7oJQ session churn, J_EDEw offline)
- Rollback: 18 PASS, 0 FAIL, 0 WARN, 2 SKIP

`sidecar_schema: 3` confirmed in sidecar `_meta`.
