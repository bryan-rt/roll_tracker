# Piece 0: `frame_index` Join Prerequisite — Findings

**Date:** 2026-08-16
**Commit:** `bfd5349` (HEAD of `services_uploader`)
**Tool:** `tools/probe_frame_index_join.py`
**Sample:** 94 passthrough+source_pts segments (47 FP7oJQ, 47 PPDmUg), all from
`00000000-0000-0000-0000-000000000003`, captured 2026-08-05 and 2026-08-07.

---

## 1. Verdict

**(a)↔(c) is 1:1 under an identifiable condition: `mismatch: false`.**

When `mismatch: false` (input_frame_count == output_frame_count), sidecar frame-row
count equals decoded frame count on **every** segment tested (45/45). The `frame_index`
join is sound.

When `mismatch: true`, sidecar row count disagrees with decoded frame count on **every**
segment tested (49/49). The deficit is **always at the END** (sub-ms timestamp alignment
from frame 0 through the overlap region, divergence only at the tail). This is a
**boundary attribution defect** — the same class of bug CP-R5 fixed for the line-position
split, now manifesting as a residual in the PTS-based split.

**The join is recoverable without a different join key.** For `mismatch: false` segments,
the join works as-is. For `mismatch: true` segments, the sidecar's first `min(a, c)` rows
align 1:1 with the mp4's first `min(a, c)` decoded frames. The deficit rows at the tail
(positive residual: mp4 has unmatched frames; negative residual: sidecar has unmatched
rows) can be handled by truncation or padding. However, **the recommended fix is to
eliminate the mismatch at the source** (see §8).

**DEL-CONV pieces are conditionally unblocked.** On `mismatch: false` segments (45/94
in this sample, including all 8 bimodal segments and all mid-attempt PPDmUg segments),
the join is sound and pieces 3–6, 10, 11 can proceed. On `mismatch: true` segments, the
join requires a guard: use only the first `min(a, c)` rows.

---

## 2. Per-Segment Table

### Key relationships (94 segments)

| Relationship | Always holds? | Count |
|---|---|---|
| **(a) = (d)** (sidecar rows = input_frame_count) | **Yes** | 94/94 |
| **(b) = (c)** (output_frame_count = decoded frames) | **Yes** | 94/94 |
| **(b) = (e)** (output_frame_count = nb_read_frames) | **Yes** | 94/94 |
| **(a) = (c)** (sidecar rows = decoded frames) | **No** | 45/94 |
| contiguity (`frame_index` = {0..N-1}) | **Yes** | 94/94 |
| pts_time_s strictly increasing | **No** | 76/94 |
| pts deltas are integer multiples of nominal_dt_s | **No** | 80/94 |

**Identities established:**
- `(a) = (d)` universally: the sidecar emits exactly one row per input (showinfo) frame.
- `(b) = (c) = (e)` universally: `output_frame_count` is truthful — it equals both the
  decode count from FrameIterator and ffprobe's `-count_frames`. This is NOT a metadata
  lie; the container genuinely has that many frames.
- The join predicate `(a) = (c)` fails when `(d) ≠ (b)`, i.e., when `mismatch: true`.

### Residual sign breakdown (by camera, never pooled)

Convention: `residual = (b) - (d) = output - input` (CP-R5 sign convention).

| Camera | Total | Positive (+) | Negative (-) | Zero (0) |
|--------|-------|-------------|-------------|----------|
| FP7oJQ | 47 | 20 | 13 | 14 |
| PPDmUg | 47 | 8 | 8 | 31 |
| **Total** | **94** | **28** | **21** | **45** |

- **Positive residual** (mp4 has more frames than sidecar rows): the sidecar is SHORT.
  The mp4's tail frames have no corresponding sidecar row — their showinfo lines were
  attributed to the previous segment. Under `min(a, c)` truncation: the sidecar covers
  `a` frames correctly; the remaining `c - a` decoded frames at the tail **fall back to
  `nominal_dt_s`** for timing (no `dt_s` available). Pieces 3–6 should use `nominal_dt_s`
  for these frames, not drop them — the frames are real and Stage A indexes them.
- **Negative residual** (sidecar has more rows than mp4 frames): the sidecar is LONG.
  The sidecar's tail rows describe frames attributed to this segment but present in the
  NEXT mp4. Under `min(a, c)` truncation: the sidecar is truncated to `c` rows; the
  surplus `a - c` rows at the tail are **discarded** (they correspond to frames in a
  different file). The PTS monotonicity violations (§5) are concentrated in these surplus
  rows.
- **Zero residual** (`mismatch: false`): perfect 1:1 correspondence.

FP7oJQ has far more mismatches (33/47 = 70%) than PPDmUg (16/47 = 34%). PPDmUg's
mid-attempt segments are overwhelmingly zero-residual.

### Breakdown by segment length

The 52% mismatch rate (49/94) overstates production impact. The sample is dominated by
short smoke-test segments (SEG_SECONDS=20). Production runs SEG_SECONDS=120 (~1800 frames).

| Segment length | Total | Break | Rate | Notes |
|----------------|-------|-------|------|-------|
| Short (<600 frames) | 35 | 29 | **83%** | Smoke-test segments (20s) |
| Long (>=600 frames) | 59 | 20 | **34%** | Production-length segments (120s) |
| Production (>=1500 frames) | 55 | 18 | **33%** | Most representative of deployment |

By sign within production-length segments: +11 positive, -7 negative, 37 zero.

The structural fragility is real regardless of rate — Option A (§8) is recommended
either way — but the urgency is sized by the 33% production rate, not the 52% sample
rate.

### Full per-segment data

Full results in `docs/evidence/frame_index_join_1/probe_results.json` (94 segments,
all six quantities, `_meta` context, alignment data, POS_MSEC pattern analysis).

---

## 3. Attempt-Level Conservation

CP-R5 established: under attribution (not insertion), per-segment residuals redistribute
within an attempt while the attempt total holds. Under insertion, the total grows.

**12 multi-segment attempts tested** (single-segment attempts excluded — cannot exhibit
redistribution):

| Attempt | Segments | Residuals | Total | Pattern |
|---------|----------|-----------|-------|---------|
| FP7oJQ/2026-08-05/11/att1 | 3 | [-3, 0, +2] | **-1** | Redistribution |
| FP7oJQ/2026-08-05/12/att1 | 3 | [-3, -2, +4] | **-1** | Redistribution |
| FP7oJQ/2026-08-05/13/att1 | 3 | [-6, +1, +4] | **-1** | Redistribution |
| FP7oJQ/2026-08-05/15/att1 | 27 | [−8,+1,+6,−13,+1,0,+1,0,+1,+1,0,0,+1,+1,−1,+2,0,0,+1,−1,0,+1,0,0,+1,0,+4] | **-1** | Redistribution |
| FP7oJQ/2026-08-05/15/att3 | 3 | [-40, 0, +40] | **0** | Redistribution |
| FP7oJQ/2026-08-07/10/att1 | 6 | [-6, 0, +5, -9, -2, +10] | **-2** | Redistribution |
| PPDmUg/2026-08-05/12/att1 | 3 | [-18, 0, +17] | **-1** | Redistribution |
| PPDmUg/2026-08-05/13/att1 | 3 | [-5, 0, +4] | **-1** | Redistribution |
| PPDmUg/2026-08-05/15/att1 | 30 | [−8,0,+7,−8,26×0,+8] | **-1** | Redistribution |
| PPDmUg/2026-08-05/15/att3 | 4 | [-1, +1, 0, 0] | **0** | Redistribution |
| PPDmUg/2026-08-05/15/att6 | 2 | [-10, +9] | **-1** | Redistribution |
| PPDmUg/2026-08-07/10/att1 | 5 | [-7, 0, +6, -7, +7] | **-1** | Redistribution |

**Result: redistribution on every attempt.** Positive and negative residuals within
each attempt offset each other within 0–2 frames. Attempt totals are −1 (10 attempts),
0 (2 attempts), or −2 (1 attempt). The −1/−2 are real frame drops (consistent with
CP-R11's measured gap rates). No attempt total is positive — **no frames are inserted.**

This is **attribution, not insertion.** The PTS-based boundary split misassigns a
small number of frames to the wrong segment at each boundary, but the attempt-level
total is conserved. This is the same defect class CP-R5 fixed for the line-position
split — a residual of that fix.

---

## 4. Positional Alignment — Deficit Is at the END

### Cross-correlation anchor (A2)

For each disagreeing segment, the sidecar's `pts_time_s` sequence was aligned against
the decoded `CAP_PROP_POS_MSEC` sequence at offsets k=0..15. The best-k result:

| Category | Segments | best_k | Interpretation |
|----------|----------|--------|---------------|
| **Negative residual** (sidecar long) | 21 | k=0 on all 21 | Deficit at END |
| **Positive residual** (sidecar short) | 28 | k=0 on 25, k=1 on 3 | Deficit at END |

**best_k = 0 on 46/49 disagreeing segments.** The deficit is at the end, not the start.

For negative-residual segments: k=0 MAE < 0.4ms on all 21 segments — **perfect
alignment from frame 0 through all sidecar rows.** The extra sidecar rows at the tail
describe frames attributed to this segment but present in the next mp4.

For positive-residual segments on FP7oJQ: k=0 MAE ranges from 1–63ms because the
comparison includes FP7oJQ's ~8% periodic gaps where `POS_MSEC` and sidecar `pts_time_s`
can differ by up to 67ms (see §6). But best_k is still 0, confirming alignment starts
at frame 0.

**There is no silent shift at the start.** The worst case (every `frame_index` lookup
wrong) does not apply.

### Per-frame divergence pattern (positive-residual example)

FP7oJQ-20260807-102006 (deficit = +5, sidecar 233 rows, mp4 238 frames):

| sidecar_i | decoded_i | sidecar_pts_ms | decoded_pos_ms | delta_ms |
|-----------|-----------|---------------|----------------|----------|
| 0 | 0 | 0.0 | 0.0 | 0.0 |
| 10 | 10 | 666.0 | 666.7 | -0.7 |
| 92 | 92 | 6600.0 | 6600.0 | 0.0 |
| 138 | 138 | 9933.0 | 9933.3 | -0.3 |
| 161 | 161 | 11600.0 | 11533.3 | **+66.7** |
| 232 | 232 | 16733.0 | 16666.7 | **+66.3** |

Alignment is sub-ms through index 138, then jumps to +66.7ms at index 161. The jump
is exactly one frame interval — the mp4 has one frame at this point that the sidecar
doesn't describe (attributed to the previous segment). The sidecar's `pts_time_s`
thereafter runs ~67ms ahead of the decoded `POS_MSEC`, consistently to the end.

The +5 deficit means the mp4 has 5 frames the sidecar doesn't describe. These 5 frames
are distributed across the segment (one near index 161 visible above, others at
intermediate positions), each causing a cumulative +67ms shift.

---

## 5. PTS Monotonicity

18/94 segments have non-strictly-increasing `pts_time_s`. All 18 correlate with
negative-residual segments. The violation pattern: the sidecar's tail rows contain
`pts_time_s` values from the NEXT segment's PTS range (which resets to near-zero due
to `-reset_timestamps 1`). Example:

```
FP7oJQ-20260805-112725 at i=1: pts[1]=0.067 >= pts[2]=0.067 (duplicate)
```

This is a direct consequence of the boundary attribution defect: rows belonging to the
next segment are spliced onto the end of this one, and those rows carry PTS values from
the next segment's zero-based range.

The 8 bimodal segments that show "discontinuities" (deltas at ~0.5× nominal) are a
separate, expected phenomenon: these are the 30fps mode-switch blocks within a 15fps
majority segment. The deltas at 0.033s are real fast-mode intervals, not errors.

---

## 6. C2 Verdict — `CAP_PROP_POS_MSEC` Is Real Container PTS

**`CAP_PROP_POS_MSEC` tracks real container PTS, including gaps.** It does NOT synthesize
a uniform grid.

Evidence from FP7oJQ-20260807-102006 (8.0% gap density):

| Metric | Value |
|--------|-------|
| Unique POS_MSEC delta values | `[66.7, 133.3]` — exactly two |
| Gap fraction (deltas > 1.5× median) | 8.02% |
| Uniform fraction (within 5% of median) | 91.98% |

OpenCV reports exactly two delta values: 66.7ms (normal frames) and 133.3ms (gap
frames). This is the real gap structure, not a synthesized grid.

**The earlier timing audit's FP7oJQ comparison (§7.1b) was confounded by the
`CAP_PROP_FPS` value.** The audit computed "expected" POS_MSEC as `frame_index / 13.89
× 1000`, which produced a uniform grid at the effective rate. The actual POS_MSEC shows
gaps. The 67ms divergence observed in the audit was not a POS_MSEC defect — it was the
difference between a uniform-rate computation and the real non-uniform PTS.

**Corrected C2 answer:** `CAP_PROP_POS_MSEC` is correct on both cameras. On PPDmUg
(0.45% gaps): sub-ms agreement with sidecar. On FP7oJQ (8% gaps): POS_MSEC shows real
gaps at 133ms, matching the sidecar's `dt_s > 0.1` intervals.

**`CAP_PROP_FPS` is still the wrong value for FP7oJQ** (13.89 = effective rate, not
the nominal 15.0), but `POS_MSEC` — which is what `FrameIterator.timestamp_ms` actually
reads — is correct. The timing audit's §2.3 and §2.4 should be re-read with this
correction: FrameIterator timestamps are correct on BOTH cameras.

---

## 7. `output_frame_count` Trustworthiness

**(b) = (c) = (e) universally.** `output_frame_count` exactly matches the decode count
(FrameIterator) and ffprobe `-count_frames` on all 94 segments.

**Site #16 (`_infer_last_frame`) verdict:** `output_frame_count` from the sidecar `_meta`
is a reliable source for the decoded frame count — it is the same value FrameIterator
would produce. However, the PROVISIONAL classification stands for two reasons:

1. **Per-segment, not per-session.** `_infer_last_frame` operates on a video that may
   span multiple segments. A single sidecar cannot describe a concatenated video. The
   fix would need to sum `output_frame_count` across the relevant segments.

2. **Boundary residual.** On `mismatch: true` segments, `output_frame_count` is correct
   for the mp4 but differs from `input_frame_count` (= sidecar row count). If site #16
   uses `output_frame_count` while other sites use sidecar row count, the two will
   disagree by the residual.

---

## 8. Recommendation

### The defect

The PTS-based boundary split (CP-R5) assigns showinfo lines to segments using cumulative
ffprobe-duration windows. When the duration windows don't exactly align with the frames
the segment muxer actually wrote, a small number of frames are misattributed:

- **First segment** of an attempt: gains extra rows at the tail (negative residual, rows
  belong to the next segment).
- **Last segment** (and some mid-segments on FP7oJQ): is short by the same count
  (positive residual, frames are in the mp4 but their showinfo lines were assigned to the
  previous segment).
- Mid-attempt PPDmUg segments: nearly always zero residual (26/30 in the largest attempt).

### Option A — Derive the sidecar from the mp4 itself (RECOMMENDED)

After each segment closes, run `ffprobe -show_frames` on the finished mp4 and read the
PTS of every frame actually in it. Row count equals frame count **by construction** —
there is nothing to attribute. `dt_s` is unaffected: it is still consecutive-PTS
differencing, just read from the artifact the pipeline decodes rather than reconstructed
from an ffmpeg stderr log.

**What Option A fixes:**
- Row count = decode count (the join prerequisite, by construction).
- PTS monotonicity (read from the mp4's actual frame order).
- The boundary attribution defect (eliminated — no cross-segment log parsing).

**What Option A does NOT fix:**
- **Frames genuinely dropped by the recorder.** The mp4 is what it is; dropped frames
  are gone. `dt_s` will show real gaps where drops occurred, and consumers will handle
  them via coast/variable-dt as designed. This is not a regression — the current sidecar
  also cannot recover dropped frames.
- **The 94 segments already on disk.** Existing sidecars retain their boundary
  misattribution. They would need to be regenerated from their mp4s (feasible — the mp4s
  are the authoritative source). Alternatively, consumers can apply the `min(a, c)`
  truncation guard as an interim measure.

**`-reset_timestamps 1` note:** The recorder uses `-reset_timestamps 1` (when the
segment muxer supports it), which rebases each segment's stored PTS toward zero. This
changes the ORIGIN, not the intervals. `dt_s` (consecutive-PTS differencing) is
unaffected. The sidecar contract declares `pts_origin: "segment_relative"` — under
Option A, this claim should be re-confirmed against the mp4-derived values, since the
mp4's stored PTS may have a small offset from the showinfo-derived PTS (the rebasing
happens in the muxer, after the filter graph). However, the intervals (`dt_s`) will be
identical regardless of the origin.

### Option B — Fix the boundary assignment

Cheaper if the cause is a simple off-by-one. But this would be the **third attempt** at
the same class of fix (line-position split → CP-R5 PTS-based split → this), and the
first two both looked correct when they shipped. The root cause is that the sidecar is
derived from a stderr log (a side channel) rather than from the artifact itself. Each
fix addresses one failure mode of the log-to-segment mapping while leaving the
structural fragility intact.

### Assertion (recommended regardless of option)

Add a post-extraction assertion in the sidecar generator: **emitted row count must equal
the segment's `nb_frames` (from ffprobe)**. Fail loudly if they differ. This would have
caught the defect at CP-R5 and would catch any future regression.

---

## 9. Implications for DEL-CONV Pieces

### Pieces 3–6, 10, 11: conditionally unblocked

The `frame_index → dt_s` lookup is sound **when `mismatch: false`**, which is:
- 45/94 segments in this sample (48%)
- All 8 bimodal segments (100%)
- All PPDmUg mid-attempt segments (26/30 = 87% in the largest attempt)
- 14/47 FP7oJQ segments (30%)

For `mismatch: true` segments, the interim guard depends on residual sign:

**Positive residual** (sidecar short, mp4 has extra frames at tail):
1. All `a` sidecar rows are valid — look up `dt_s` for `frame_index` 0..a-1.
2. The `c - a` decoded frames at the tail (typically 1–13, outlier 40) have no sidecar
   row. **Fall back to `nominal_dt_s` for these frames** — do not drop them, they are
   real frames that Stage A indexes.

**Negative residual** (sidecar long, surplus rows at tail):
1. Truncate the sidecar to `c` rows (= `output_frame_count`). The surplus `a - c` rows
   describe frames in a different mp4 and must be **discarded**.
2. All `c` retained rows are valid.

**Guard implementation:**
1. Read `mismatch` and `output_frame_count` from sidecar `_meta`.
2. If `mismatch: true`, use only the first `min(a, output_frame_count)` sidecar rows.
3. For decoded `frame_index` values beyond the sidecar's range, use `nominal_dt_s`.

This is a viable interim guard while the recorder fix (Option A or B) is implemented.
The fallback to `nominal_dt_s` on the tail frames introduces at most one frame interval
of error on FP7oJQ gap frames — acceptable for the sizes observed (1–13 frames on most
production-length segments, 40 frames on one outlier).

Production-relevant rate: 33% of production-length segments (>=1500 frames) have
`mismatch: true` (18/55). The guard fires on these; the other 67% use the sidecar as-is.

### Timing audit corrections

The C2 answer changes the timing audit's FrameIterator verdict:

- **§2.3 (FrameIterator timestamps):** Reclassify as **correct on BOTH cameras.** The
  earlier "wrong on FP7oJQ" verdict was based on a comparison against a uniform grid at
  `CAP_PROP_FPS=13.89`, not against the real PTS. `POS_MSEC` tracks real container PTS
  including gaps.
- **§2.4–§2.6 (quality.py, processor.py, c0_triggers):** Similarly reclassify. Velocities
  from FrameIterator timestamps are correct.

The timing audit's **§7.1b** FP7oJQ comparison should carry a correction note: the
67ms divergence was between sidecar `pts_time_s` and a uniform-grid computation, not
between sidecar and `POS_MSEC`. `POS_MSEC` itself shows the real gap structure.
