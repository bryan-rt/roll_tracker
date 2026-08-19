# Sidecar Timing Contract -- Schema v4

*Authoritative specification for `.timing.jsonl` sidecars produced by the Nest recorder.
Pipeline consumers code against this document, not the recorder implementation.*

*Established: 2026-08-05 (CP-R6). Schema version: 4.*

---

## 1. File Structure

Each video segment `{name}.mp4` has a sibling `{name}.timing.jsonl`. The file contains:

1. **One `_meta` line** (always line 1) -- segment-level metadata and summary statistics.
2. **N frame rows** (lines 2..N+1) -- one per output frame, keyed on `frame_index`.

All lines are independent JSON objects (JSONL format). No array wrapper.

---

## 2. Validity Model

**Single gate: `source_pts`.** This boolean in `_meta` indicates whether the sidecar was
produced from camera-derived RTP capture timestamps (`true`) or from bursty network-arrival
timestamps (`false`). It is the sole validity gate for all derived fields.

**Omission means invalid.** Fields that depend on a specific condition are **absent** from
the JSON when that condition is not met. A consumer that reads a field that exists can trust
it within its stated precision. A consumer that looks for a field and finds it absent knows
the sidecar was produced under conditions where that field has no meaning.

**Three validity axes** (schema 5):

**1. `source_pts` gate** — fields that require camera-derived RTP timestamps:
- `_meta`: `nominal_dt_s`, `measured_fps`, `measured_fps_median`, `is_bimodal` (and its
  sub-fields), `pts_wallclock_offset_s`, `offset_method`, `drift_rate_s_per_s`, `drift_flat`,
  `drift_ppm`, `n_drift_windows`
- Frame rows: `dt_s`

**2. Showinfo availability gate** — fields that require the showinfo stderr log. Absent when
`row_source: "mp4_regenerated"` (offline regeneration without showinfo):
- `_meta`: `showinfo_frame_count`, `showinfo_residual`, `showinfo_pts_offset`,
  `showinfo_matched_count`, `showinfo_unmatched_mp4_count`, `showinfo_surplus_count`,
  `showinfo_offset_status`
- Frame rows: `host_arrival_s` (wholesale absent when showinfo unavailable; **also absent
  per-row** on individual frames where the showinfo-to-mp4 PTS join found no match, even
  when showinfo is available for the segment)

Note: `host_arrival_s` has three absence conditions: (a) `source_pts: false` → absent on
all rows, (b) `row_source: "mp4_regenerated"` → absent on all rows, (c) showinfo join
miss → absent on individual rows while present on others in the same segment.

**3. Always present** regardless of `source_pts` or `row_source`:
- `_meta`: `_meta`, `sidecar_schema`, `timing_mode`, `source_pts`, `pts_origin`,
  `fps_method`, `row_source`, `segment_start_epoch`, `attempt`, `input_frame_count`,
  `output_frame_count`, `measured_fps_mean`, `pts_timebase`, `pts_tick_delta_median`,
  `pts_tick_delta_mean`, `pts_delta_trim_kept`, `pts_delta_trim_total`, `mismatch`,
  `pts_mean_delta_ms`, `pts_stdev_delta_ms`
- Frame rows: `frame_index`, `pts_time_s`

**Additional gating rules:**
- `drift_rate_s_per_s` and `drift_ppm` are **omitted** when `n_drift_windows < 4` (unstable
  estimate from insufficient data). `drift_flat` is emitted as `true` and `n_drift_windows`
  is always present so consumers know why drift fields are absent.
- `short_mode_fraction`, `short_mode_fps`, `short_mode_dt_s`, `long_mode_dt_s` are present
  **only when** `is_bimodal: true`.
- `output_fps` is present **only in** `timing_mode: "cfr_grid"`.
- `dt_s` is present **only in** `timing_mode: "passthrough"` AND `source_pts: true`.

---

## 3. `_meta` Field Reference

### Always present

| Field | Type | Description |
|-------|------|-------------|
| `_meta` | `true` | Literal marker identifying this as the metadata line. |
| `sidecar_schema` | int | Schema version. This document specifies version **5**. |
| `timing_mode` | string | `"passthrough"` (1:1 input-to-output, no resampling) or `"cfr_grid"` (uniform output grid, nearest-neighbor mapped from input). |
| `source_pts` | bool | `true` if produced from camera RTP capture timestamps; `false` if from network-arrival timestamps. **The validity gate.** |
| `pts_origin` | string | `"segment_relative"` -- all PTS values are zero-based from the first frame of this segment. |
| `fps_method` | string | `"trimmed_mean"` -- the algorithm used for `measured_fps`. |
| `segment_start_epoch` | int | Unix epoch (seconds) of segment start, parsed from the segment filename. |
| `attempt` | int | Retry attempt counter within the recording window (1-indexed). |
| `input_frame_count` | int | Under schema 5: equals `output_frame_count` (mp4-driven). Under schema ≤4: count of showinfo lines attributed to this segment. |
| `output_frame_count` | int | Count of output frames in the segment mp4 (from ffprobe `nb_frames`). Under schema 5: also the sidecar row count (by construction). |
| `row_source` | string | **Schema 5+.** Provenance of frame rows: `"mp4"` (live capture, mp4-derived rows + showinfo join for host_arrival), `"mp4_regenerated"` (offline regeneration, mp4 only, no showinfo fields), `"showinfo_grid"` (legacy schema ≤4). |
| `measured_fps_mean` | float | Span-based fps: `(output_frame_count - 1) / pts_span_seconds`. Approximately correct even under arrival-PTS; useful as a sanity check. |
| `pts_timebase` | int | PTS tick rate (typically 90000 for RTSP). Ticks-to-seconds: `ticks / pts_timebase`. |
| `pts_tick_delta_median` | float | Median of inter-frame tick deltas (sorted). |
| `pts_tick_delta_mean` | float | Arithmetic mean of inter-frame tick deltas. |
| `pts_delta_trim_kept` | int | Count of tick deltas within the trimmed-mean window `[0.5x, 1.5x]` of the median. |
| `pts_delta_trim_total` | int | Total count of tick deltas (`input_frame_count - 1`). |
| `mismatch` | bool | Under schema 5: structurally `false` (`input_frame_count == output_frame_count` by construction). Under schema ≤4: `true` when showinfo count ≠ output count. Consumers should read `showinfo_residual` (schema 5) for the drop signal. |
| `pts_mean_delta_ms` | float | Mean of tick deltas converted to milliseconds. |
| `pts_stdev_delta_ms` | float | Standard deviation of tick deltas in milliseconds. **Caveat:** measures the camera's tick-distribution pattern (alternation of e.g. 5940/6030 ticks), not jitter. Do not use as a jitter proxy. |

### Present when `source_pts: true`

| Field | Type | Description |
|-------|------|-------------|
| `nominal_dt_s` | float | Median-based expected inter-frame interval in seconds (`pts_tick_delta_median / pts_timebase`). **The reference value for gap detection.** See Section 6 for the consumer recipe. |
| `measured_fps` | float | Trimmed mean of input capture cadence in fps (`pts_timebase / trimmed_mean_tick`). **Under bimodality, reports the majority mode only** -- see Section 5. |
| `measured_fps_median` | float | `pts_timebase / pts_tick_delta_median`. Subject to tick-alternation quantization (e.g. 14.9254 instead of 15.0000 when ticks alternate 5940/6030). |
| `is_bimodal` | bool | Whether the segment contains frame intervals at two discrete rates (~2x apart) in sustained blocks. See Section 5. |
| `pts_wallclock_offset_s` | float | Lower-envelope offset: `min(host_arrival - pts_time)` across all frames. Anchors segment-relative PTS to host wall-clock. **Estimated accuracy: +/-14-56ms** (CAPTURE-TIME-2). |
| `offset_method` | string | `"lower_envelope"` -- the algorithm used. |
| `drift_flat` | bool | `true` if drift is negligible (< 0.0001 s/s) or unmeasurable (< 4 windows). When `true` and `n_drift_windows >= 4`, drift is confirmed flat; when `true` and `n_drift_windows < 4`, drift is unknown (insufficient data). |
| `n_drift_windows` | int | Count of 10-second windows used for drift estimation. Below 4, drift fields are unreliable and `drift_rate_s_per_s` / `drift_ppm` are omitted. |
| `drift_rate_s_per_s` | float | **Present only when `n_drift_windows >= 4`.** OLS slope of windowed lower-envelope offset vs elapsed time. Units: seconds of clock offset per second of elapsed time. Example: -0.000000603 = camera clock loses 603ns per second relative to host. |
| `drift_ppm` | float | **Present only when `n_drift_windows >= 4`.** `drift_rate_s_per_s * 1e6`. Example: -0.603 = -603 parts per million. |

### Present when `is_bimodal: true` (implies `source_pts: true`)

| Field | Type | Description |
|-------|------|-------------|
| `short_mode_fraction` | float | Fraction of inter-frame tick deltas in the short-mode cluster (< 0.75x median). Range 0-1. |
| `short_mode_fps` | float | `pts_timebase / mean_short_mode_tick`. Typically ~30 fps. |
| `short_mode_dt_s` | float | Mean short-mode interval in seconds. Typically ~0.033s. |
| `long_mode_dt_s` | float | Mean long-mode interval in seconds. Typically ~0.067s. |

### Present only in `timing_mode: "cfr_grid"`

| Field | Type | Description |
|-------|------|-------------|
| `output_fps` | float | Output grid rate from ffprobe `r_frame_rate`. The uniform spacing of output frames. |

### Present when `row_source: "mp4"` (schema 5+, showinfo available)

| Field | Type | Description |
|-------|------|-------------|
| `showinfo_frame_count` | int | Count of showinfo lines attributed to this segment by the boundary split. This is the recorder's own count of frames it saw pass through the filter graph — distinct from the mp4's decoded frame count. |
| `showinfo_residual` | int | `output_frame_count - showinfo_frame_count`. **The drop signal.** Positive = mp4 has frames showinfo missed (attributed to a neighbouring segment). Negative = showinfo has surplus lines (frames attributed here but present in a neighbouring mp4). Replaces `mismatch` as the diagnostic for boundary attribution and real frame loss. |
| `showinfo_pts_offset` | int | PTS tick offset computed during the showinfo-to-mp4 join (`mp4_pts[0] - showinfo_pts[best_k]`). Diagnostic — consumers do not need this value. |
| `showinfo_matched_count` | int | Count of mp4 frames matched to a showinfo line during the PTS join. Frames with a match carry `host_arrival_s`; unmatched frames do not. |
| `showinfo_unmatched_mp4_count` | int | Count of mp4 frames with no showinfo match. These rows lack `host_arrival_s`. |
| `showinfo_surplus_count` | int | Count of showinfo lines with no corresponding mp4 frame (discarded during the join). |
| `showinfo_offset_status` | string | Join quality indicator: `"determined"` (unique best shift with margin), `"best_effort"` (best shift has nonzero disagreements), `"ambiguous_fallback_k0"` (uniform deltas, no shift distinguishable — assumed k=0), `"uniform_assumed_k0"` (zero variation in deltas), `"undetermined"` (insufficient data). |

---

## 4. Frame Row Field Reference

### Always present

| Field | Type | Description |
|-------|------|-------------|
| `frame_index` | int | 0-indexed sequential counter. **Join key to Stage A** (`FrameIterator`'s `cap.read()` counter). Under schema 5, row count = mp4 decoded frame count by construction. |
| `pts_time_s` | float | Segment-relative PTS in seconds. Under passthrough: the frame's actual capture PTS (base-subtracted). Under CFR: the nearest-neighbor input PTS mapped to this output grid point (approximate). |

### Removed in schema 5

| Field | Type | Description |
|-------|------|-------------|
| `input_n` | int | **Removed.** Present in schema ≤4 only. Under passthrough: always equalled `frame_index` (identity). Under CFR: nearest-neighbor input frame index. Deprecated since schema 4. |

### Present when `timing_mode: "passthrough"` AND `source_pts: true`

| Field | Type | Description |
|-------|------|-------------|
| `dt_s` | float or null | Inter-frame interval in seconds: `pts_time_s[i] - pts_time_s[i-1]`. **`null` on frame 0** (no predecessor). This is the ground truth for per-frame timing. Consumers MUST handle `null` on the first frame. |

### Present when `source_pts: true`

| Field | Type | Description |
|-------|------|-------------|
| `host_arrival_s` | float | Host-side `$EPOCHREALTIME` when this frame's showinfo line was written to stderr. Used with `pts_time_s` for the lower-envelope offset calculation. Not useful to pipeline consumers directly. **Under schema 5:** may be **absent on individual rows** when the showinfo-to-mp4 PTS join found no matching showinfo line for that frame. This is distinct from being absent wholesale under `source_pts: false`. A missing `host_arrival_s` on a single row within an otherwise-present segment means the recorder's boundary attribution assigned that showinfo line to a different segment. |

---

## 5. Bimodal Rate Representation

### The phenomenon

Nest cameras deliver frames at a single cadence (~67ms / ~15fps) with two additional
phenomena:

1. **Periodic single-frame gaps.** FP7oJQ produces a doubled interval (~133ms) every ~12
   frames, caused by a camera-internal grid mismatch between its real capture rate (~13.85fps)
   and its PTS timestamp grid (~14.93fps). PPDmUg has a much lower gap rate (~0.45%).

2. **Sustained cadence switches.** On rare occasions, the cadence switches to ~33ms (~30fps)
   in sustained blocks lasting seconds to tens of seconds (CP-R11 measured blocks of 194, 205,
   and 370 frames). The two cadences are always at a 2:1 ratio.

The 15fps cadence is genuine camera-side encoding, not 30fps with frame loss. PPDmUg
delivered 1,979 consecutive gap-free 67ms frames (131.9s) -- no physical loss mechanism
produces zero-jitter alternation over that span (CP-R11). FP7oJQ's gaps are periodic (mode
spacing = 12 frames), not random, and their count matches the grid-rate/effective-rate deficit
exactly.

**Historical note (pair-sum identity, CP-R1b).** For any single interval, the 15fps tick
pattern [5940, 6030] is the algebraic pair-sum of the 30fps pattern [2970, 3060]:
2970+2970=5940, 2970+3060=6030. This identity made the question appear structurally
undecidable from a single interval. CP-R11 resolved it by examining 283 segments (247K
intervals): sustained regularity and periodic gap structure are incompatible with frame loss.

### Detection

The `is_bimodal` flag uses a **structural** test on the trimmed-mean discards, not a
magnitude threshold:

1. The trimmed mean discards tick deltas outside `[0.5x, 1.5x]` of the median.
2. Discards are partitioned into below-cutoff and above-cutoff.
3. **Bimodal (majority long-mode):** a material fraction (>30%) of discarded deltas fall
   **below** the low cutoff. This indicates a short-mode cluster at ~0.5x the median, which
   is the structural signature of bimodality. Gap-induced discards scatter **above** the high
   cutoff (at 2x, 3x, 4x the median).
4. Requires at least 3 total discarded deltas to avoid noise on very short segments.

**Known limitation:** When the majority mode is short (~30fps, median ~3000 ticks), the
minority long-mode deltas (~6000 ticks) are discarded **above** the high cutoff, in the same
direction as gap-induced discards. The below-median test does not fire in this case.
`is_bimodal` may be `false` on a majority-30fps bimodal segment. The flag is advisory, not
authoritative. Consumers handling critical bimodal cases should also inspect the raw `dt_s`
distribution.

### What consumers get

When `is_bimodal: true`, four additional fields appear in `_meta`:

- `short_mode_fraction`: what proportion of frames are at the fast rate.
- `short_mode_fps` / `short_mode_dt_s`: the fast-rate interval.
- `long_mode_dt_s`: the slow-rate interval.

`measured_fps` continues to report the **majority mode** (via trimmed mean, which discards
the minority). Under bimodality it is wrong by up to 2x. Consumers needing a per-clip scalar
rate should use `1 / nominal_dt_s` and be aware it reflects whichever mode captured the
median.

### `nominal_dt_s` discontinuity at the 50% crossover

The median flips between ~3000 and ~6000 ticks at the crossover point where neither mode is
a clear majority. This means `nominal_dt_s` can **halve between adjacent segments** of the
same stream (e.g. segment N at 0.067s, segment N+1 at 0.033s). CP-R1b observed exactly this
progression (0% -> 66% -> 99% short-mode across consecutive FP7oJQ segments).

Consumers comparing gap counts or coast behaviour across segments MUST account for this.
When `is_bimodal: true`, `short_mode_dt_s` and `long_mode_dt_s` provide the two stable
reference intervals independent of which won the median.

---

## 6. Consumer Recipes

**Principle (TIMING-PRINCIPLE-1):** Consumers should prefer reading `pts_time_s` and `dt_s`
directly over deriving a rate and converting between frames and seconds. Frame-to-time
conversion via a scalar fps is itself the defect the sidecar exists to eliminate. Most
pipeline sites that currently compute `frame_count / fps` or `frame_delta / fps` should
**delete the conversion** and read the sidecar's per-frame timing instead. Sections 6.1–6.3
below give recipes for the remaining cases where a scalar or derived value is needed.

### 6.1 Gap Detection and Coast-Step Injection (Stage A)

**The sidecar does not classify gaps.** Whether an interval exceeds the expected cadence
depends on the local block's baseline, which the segment-level `nominal_dt_s` cannot express
when a mode switch has occurred. The sidecar provides the raw timing; the consumer decides.

**Recommended recipe for unimodal segments (`is_bimodal: false`):**

```
For each frame i where dt_s is not null:
  if dt_s > 1.5 * nominal_dt_s:
    coast_steps = round(dt_s / nominal_dt_s) - 1
    # Insert coast_steps predict-without-update cycles before this frame
```

This handles FP7oJQ's periodic grid-mismatch gaps correctly: at 15fps nominal (0.067s), the
threshold is 0.1005s, and gaps at 0.133s produce exactly `coast_steps = 1`. Gaps are always
single missed grid slots.

**Bimodal segments (`is_bimodal: true`) -- sustained-block exposure:**

Under the blocked model (CP-R11), a mode switch means an entire sustained block runs at a
cadence the segment's `nominal_dt_s` does not describe. Coast injection has no mechanism for
frames arriving EARLY (dt < nominal) -- it cannot inject negative time. Every frame in a
minority-mode block is affected, not a scattered few.

Measured exposure (CP-R11, 283 segments):

| Camera | Minority-mode frames | % of corpus | Segments with switches | % |
|--------|---------------------|-------------|----------------------|---|
| FP7oJQ | 833 | 0.70% | 1 / 139 | 0.7% |
| PPDmUg | 3,812 | 2.95% | 18 / 144 | 12.5% |

Consumers using coast-step injection SHOULD check `is_bimodal` and be aware that on bimodal
segments, the `1.5x` threshold will classify real minority-mode frames as gaps -- inserting
**phantom time** for an entire block's duration. The athlete moved 2x the usual distance in
2x the usual time; velocity is correct, but the Kalman filter receives phantom predict-
without-update cycles.

The correct solution for bimodal segments is **variable-dt Kalman steps** consuming per-frame
`dt_s` directly. This handles both gaps and mode switches with one mechanism. See the coast
architecture decision in `CLAUDE.md` Active Decisions Log.

If variable dt is not available, suppress coast injection on `is_bimodal: true` segments and
accept the `nominal_dt_s` mismatch on minority-mode frames as a documented limitation.

### 6.2 BoT-SORT Frame Rate Scalar (Exception to TIMING-PRINCIPLE-1)

BoT-SORT requires one `frame_rate` scalar per clip because boxmot hardcodes a unit Kalman
time step. This is a **documented exception** to the read-time-don't-convert principle —
boxmot's API requires a scalar by construction. The recommended value:

```
frame_rate = 1.0 / nominal_dt_s
```

This reports the majority mode. The documented cost:

- On unimodal segments: correct.
- On bimodal segments: wrong for minority-mode frames. The tracker's Kalman prediction will
  under- or over-shoot on ~`min(short_mode_fraction, 1-short_mode_fraction)` of frames.
- Under `source_pts: false` (arrival-PTS rollback): `nominal_dt_s` is absent. Fall back to
  `measured_fps_mean` as the scalar, understanding it is approximate.

A variable-dt Kalman step consuming per-frame `dt_s` directly would eliminate this lie
entirely but requires a boxmot fork (unscoped).

### 6.3 Cross-Camera Synchronization

To align two cameras' timelines:

1. Read `pts_wallclock_offset_s` from each camera's `_meta`.
2. For frame `i` of camera A: `wall_time = pts_time_s + pts_wallclock_offset_s`.
3. Similarly for camera B.
4. If `drift_flat: false` on either camera, apply linear correction:
   `corrected_offset = pts_wallclock_offset_s + drift_rate_s_per_s * pts_time_s`

**Accuracy: +/-14-56ms** (CAPTURE-TIME-2). RTCP is absent on all cameras; absolute camera
clock is unavailable from the stream.

---

## 7. Worked Examples (schema 5)

*All examples below are from real production sidecars captured 2026-08-17.*

### `_meta` line (passthrough, source_pts=true, unimodal 15fps)

Source: PPDmUg-20260817-133829 (300 frames, 15fps, perfect showinfo match).

```json
{"_meta":true,"sidecar_schema":5,"timing_mode":"passthrough","source_pts":true,"pts_origin":"segment_relative","fps_method":"trimmed_mean","row_source":"mp4","segment_start_epoch":1786988309,"attempt":1,"input_frame_count":300,"output_frame_count":300,"showinfo_frame_count":300,"showinfo_residual":0,"showinfo_pts_offset":-1805490,"showinfo_matched_count":300,"showinfo_unmatched_mp4_count":0,"showinfo_surplus_count":0,"showinfo_offset_status":"ambiguous_fallback_k0","nominal_dt_s":0.067000,"measured_fps":15.0003,"measured_fps_median":14.9254,"measured_fps_mean":15.0003,"pts_timebase":90000,"pts_tick_delta_median":6030.0,"pts_tick_delta_mean":5999.9,"pts_delta_trim_kept":299,"pts_delta_trim_total":299,"mismatch":false,"is_bimodal":false,"pts_wallclock_offset_s":1786988308.163707,"offset_method":"lower_envelope","drift_flat":false,"n_drift_windows":2,"pts_mean_delta_ms":66.6656,"pts_stdev_delta_ms":0.4718}
```

Note: `drift_rate_s_per_s` and `drift_ppm` are absent because `n_drift_windows < 4`.
`showinfo_offset_status: "ambiguous_fallback_k0"` — PPDmUg has uniform deltas (all 6030),
so no shift is distinguishable from the delta pattern; k=0 assumed.

### Frame rows (first 3 frames)

```json
{"frame_index":0,"pts_time_s":0.000000,"dt_s":null,"host_arrival_s":1786988309.096267}
{"frame_index":1,"pts_time_s":0.067000,"dt_s":0.067000,"host_arrival_s":1786988309.105819}
{"frame_index":2,"pts_time_s":0.133000,"dt_s":0.066000,"host_arrival_s":1786988309.107861}
```

Note: `input_n` removed in schema 5. `host_arrival_s` present on all rows (perfect match).

### `_meta` line (cfr_grid, source_pts=false — rollback)

Source: FP7oJQ-20260817-193321 (630 frames, 30fps CFR, arrival-PTS).

```json
{"_meta":true,"sidecar_schema":5,"timing_mode":"cfr_grid","source_pts":false,"pts_origin":"segment_relative","fps_method":"trimmed_mean","row_source":"mp4","segment_start_epoch":1787009601,"attempt":1,"input_frame_count":630,"output_frame_count":630,"showinfo_frame_count":154,"showinfo_residual":476,"output_fps":30.0000,"measured_fps_mean":7.5614,"pts_timebase":90000,"pts_tick_delta_median":6.0,"pts_tick_delta_mean":11902.5,"pts_delta_trim_kept":33,"pts_delta_trim_total":153,"mismatch":false,"pts_mean_delta_ms":132.2500,"pts_stdev_delta_ms":555.9970}
```

Note: `nominal_dt_s`, `measured_fps`, `measured_fps_median`, `is_bimodal`, all drift fields,
and `host_arrival_s` are absent (arrival-PTS). `showinfo_residual: 476` — encoder produced
630 output frames from 154 showinfo input frames (arrival-PTS duplication). CFR statistics
describe the INPUT capture cadence (bursty arrival), not the output 30fps grid.

### `_meta` line (regenerated, no showinfo)

Source: regenerated from FP7oJQ-20260817-133817.mp4 via `tools/regenerate_sidecar.py`.

```json
{"_meta":true,"sidecar_schema":5,"timing_mode":"passthrough","source_pts":true,"pts_origin":"segment_relative","fps_method":"trimmed_mean","row_source":"mp4_regenerated","segment_start_epoch":1786988297,"attempt":0,"input_frame_count":300,"output_frame_count":300,"nominal_dt_s":0.067,"measured_fps":14.9994,"measured_fps_median":14.9254,"measured_fps_mean":13.8426,"pts_timebase":90000,"pts_tick_delta_median":6030,"pts_tick_delta_mean":6501.7,"pts_delta_trim_kept":272,"pts_delta_trim_total":299,"mismatch":false,"is_bimodal":false,"pts_mean_delta_ms":72.2408,"pts_stdev_delta_ms":19.2403}
```

Note: `showinfo_frame_count`, `showinfo_residual`, all showinfo join fields, `host_arrival_s`,
drift fields, and `pts_wallclock_offset_s` are absent — showinfo was not available during
regeneration. `row_source: "mp4_regenerated"` identifies this provenance.

---

## 8. `timing_mode` Values

| Value | Meaning | `dt_s` in frame rows | `output_fps` in `_meta` |
|-------|---------|----------------------|-------------------------|
| `"passthrough"` | Each output frame IS the input frame. 1:1 mapping. Under schema 5: frame rows derived from mp4 PTS; `input_n` removed. | Yes (when `source_pts: true`) | No |
| `"cfr_grid"` | Output frames on a uniform grid. Each maps to nearest-neighbor input via two-pointer. `pts_time_s` is approximate. | No | Yes |

---

## 9. Schema History

| Version | Date | Changes |
|---------|------|---------|
| 1 | 2026-07 | Initial. `pts_time` (3 decimal, quantized). |
| 2 | 2026-08-02 | Integer tick precision via `pts` field. `measured_fps` from trimmed mean of tick deltas. `pts_stdev_delta_ms` added. |
| 3 | 2026-08-05 | PTS-based segment boundary split (CP-R5). `pts_origin: "segment_relative"`. `input_frame_count` corrected. |
| 4 | 2026-08-05 | Contract established (CP-R6). `source_pts` validity gate. `nominal_dt_s`, `dt_s`, `is_bimodal` + mode fields added. `measured_fps`/`measured_fps_median` omitted under `source_pts: false`. Drift fields gated at `n_drift_windows >= 4`. `input_n` deprecated. First production validation of bimodal emission (2026-08-05, CP-R10): 8 of 33 PPDmUg segments emitted `is_bimodal: true` with valid `short_mode_*` fields. |
| 4 (prose) | 2026-08-07 | Sections 5 and 6.1 explanatory text corrected for blocked-mode model (CP-R11, CP-R12). No emission change -- `is_bimodal`, `nominal_dt_s`, and the detection logic are validated correct. "Structurally undecidable" retired. Section 10 `gap_flag` rationale updated. |
| 5 | 2026-08-17 | CP-R13b: frame rows and tick statistics derived from mp4 PTS (row count = decode count by construction). Showinfo retained only for `host_arrival_s` and drift, joined by PTS value. `input_n` removed. New fields: `row_source` (`"mp4"` / `"mp4_regenerated"` / `"showinfo_grid"`), `showinfo_frame_count`, `showinfo_residual` (drop signal), `showinfo_pts_offset`, `showinfo_matched_count`, `showinfo_unmatched_mp4_count`, `showinfo_surplus_count`, `showinfo_offset_status`. `mismatch` structurally false. `input_frame_count` = `output_frame_count` (mp4-driven). `host_arrival_s` may be absent on individual rows when the showinfo join has no match. Prerequisite: CP-R13a (`-enc_time_base 1/90000`). Pre-R13a footage cannot be regenerated (container timebase check). CFR path: row count from mp4, statistics from showinfo (describes input capture cadence, not output grid). |

---

## 10. Rejected Fields

| Field | Reason |
|-------|--------|
| `gap_flag` | Under the blocked model, whether an interval is a gap depends on the LOCAL cadence of the current block. The sidecar computes a segment-level `nominal_dt_s`, which cannot express a per-block baseline. A segment containing a mode switch has two valid baselines, so a single `gap_flag` would be wrong for one of the blocks. Consumers with block context can make the call; the sidecar cannot. Consumer recipe documented instead (Section 6.1). |
| `implied_missing_frames` | Same rationale as `gap_flag` -- bakes a local-cadence-dependent judgment into a segment-level data format. Consumer computes `round(dt_s / nominal_dt_s) - 1` when needed. |
| `is_duplicate` | Under passthrough + source-PTS, pixel-identical adjacent duplicates do not occur (DUPFIX-1: zero on 9/10 segments, 3 frames / 0.18% on one exception). A duplicate signal should be observation-based (framehash) if ever needed, not sidecar-derived. |
| `dt_s` under CFR | Under `timing_mode: "cfr_grid"`, `pts_time_s` is a nearest-neighbour construction (each output grid point mapped to the closest input frame). Differencing adjacent `pts_time_s` values does not yield a real inter-frame interval -- it yields the spacing of the nearest-neighbour mapping, which can be zero (two grid points map to the same input) or jump by 2x (a grid point skipped). Per-frame timing under CFR is `1 / output_fps` (uniform grid), not a sidecar field. |

---

## 11. Evidence Traceability

| Claim | Source |
|-------|--------|
| Source PTS = true capture timestamps | `docs/evidence/capture_time_1/findings.md`, `docs/evidence/capture_time_2/findings.md` |
| RTCP absent, cross-camera offset +/-14-56ms | `docs/evidence/capture_time_2/findings.md` |
| FP7oJQ drift -603 ppm | `docs/evidence/capture_time_2/findings.md` |
| Frame-spacing characterization: blocked modes, periodic gaps, grid mismatch | `docs/evidence/frame_spacing_1/findings.md` (CP-R11, supersedes CP-R1b) |
| ~~Bimodal frame-rate oscillation~~ | ~~`docs/evidence/recorder_fps_adaptation_1/findings.md`~~ (CP-R1b, partially superseded by CP-R11 -- Sections 4, 5 corrected in place) |
| TRIM-BIMODAL defect (36% discard) | `docs/evidence/recorder_fps_adaptation_1/findings.md` |
| Zero pixel-identical duplicates under source-PTS | `docs/evidence/recorder_dupfix_1/findings.md` |
| Frame drops 0.1-7.7% (FP7oJQ), 0-3.0% (PPDmUg) | `docs/evidence/recorder_dupfix_1/findings.md` |
| PTS-based boundary fix (CP-R5) | `docs/evidence/recorder_boundary_fix_1/findings.md` |
| `pts_stdev_delta_ms` measures tick alternation | `docs/roadmap/recorder_productionization.md` (CP-R6 section) |
| `measured_fps` ~14000 under arrival-PTS | CP-R3 smoke test (2026-08-04) |
| Drift instability on short segments (2449 ppm, 2 windows) | CP-R2 smoke test (2026-08-04) |
