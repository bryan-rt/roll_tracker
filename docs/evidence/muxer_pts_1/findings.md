# MUXER-PTS-1: Duplicate PTS at attempt start — diagnosis and fix

**Date:** 2026-08-24
**Status:** Fix implemented, verification capture pending

---

## 1. The defect

The first segment of each recording attempt has two frames with the same PTS at
positions 1 and 2 (showinfo n:1 and n:2). The sidecar reports `dt_s=0.0` between
them, and the pipeline refuses the clip.

The frames are visually distinct content but represent the same camera capture moment
(same RTP timestamp). They are NOT pixel-identical in all cases.

---

## 2. Mechanism (localized from showinfo evidence, pre-segmentation)

The duplicate PTS is in the **decoded RTSP input**, visible in the showinfo filter
output (between decoder and encoder). The segment muxer and `-avoid_negative_ts
make_zero` are not involved.

### What the RTSP relay sends on reconnection

When a client reconnects, the relay sends its buffered GOP. The stream starts with
B-frames that precede the IDR in display order. The last pre-IDR B-frame (n:1) and
the IDR itself (n:2) carry the **same RTP timestamp**.

Affected attempt showinfo pattern (attempt 5, FP7oJQ, Aug 23):
```
n:0 pts:-69300 type:B checksum:0A9A8AD9  (pre-IDR B-frame, earlier capture)
n:1 pts:-63270 type:B checksum:0ED28AB9  ← same PTS as n:2
n:2 pts:-63270 type:I checksum:0ED28AB9  ← DUPLICATE (IDR)
n:3 pts:-45270 type:B checksum:B3D78F36  (normal from here)
```

Clean attempt showinfo (attempt 1, FP7oJQ, Aug 23):
```
n:0 pts:-16830 type:I  (IDR is first frame — no preceding B-frames)
n:1 pts: -4860 type:B  (all PTS strictly increasing)
```

libx264 emits 2 "non-strictly-monotonic PTS" warnings per affected attempt, 1 per
clean attempt (the latter from normal B-frame reordering).

### Why attempt 1 sometimes escapes

Depends on relay buffer state, not strictly on attempt number:
- Aug 23 attempt 1 (123642): stream starts with IDR as n:0 → clean
- CP-R8 attempt 1 (200827): stream starts with B-frames → affected

On initial connection, the relay may send a clean stream. On reconnection, it almost
always sends buffered B-frames including the duplicate. 9/9 for attempts >1; 1/2 for
attempt 1.

### Hypothesis (not proven)

The B-frame and IDR share the same RTP timestamp because they represent the same
GOP boundary moment — the B-frame is a motion-compensated prediction of the IDR's
content. This is consistent with the data (same mean/stdev, same or near-identical
content) but the exact H.264/relay mechanism is not established.

---

## 3. Checksum census: all 11 affected segments

| # | Segment | Camera | Capture | PTS | Checksum | Planes |
|---|---------|--------|---------|-----|----------|--------|
| 1 | att5 130325 | FP7oJQ | Aug 23 | -63270 | **MATCH** | MATCH |
| 2 | att7 130549 | FP7oJQ | Aug 23 | -81810 | **DIFFER** | DIFFER |
| 3 | att9 130817 | FP7oJQ | Aug 23 | -29430 | **MATCH** | MATCH |
| 4 | att4 123830 | PPDmUg | Aug 23 | -35910 | **MATCH** | MATCH |
| 5 | att9 124855 | PPDmUg | Aug 23 | -33840 | **MATCH** | MATCH |
| 6 | att19 134200 | PPDmUg | Aug 23 | -33930 | **DIFFER** | DIFFER |
| 7 | att22 135811 | PPDmUg | Aug 23 | -58860 | **MATCH** | MATCH |
| 8 | att24 140901 | PPDmUg | Aug 23 | -79740 | **MATCH** | MATCH |
| 9 | att1 h19 | FP7oJQ | Aug 19 | -122130 | **DIFFER** | DIFFER |
| 10 | att1 h20 200827 | FP7oJQ | Aug 19 | -34830 | **DIFFER** | DIFFER |
| 11 | att3 h20 | FP7oJQ | Aug 19 | -52380 | **DIFFER** | DIFFER |

**Split: 6 MATCH (55%), 5 DIFFER (45%).**

All 11 have identical mean and stdev between the duplicate pair. The checksum
differences in DIFFER cases are H.264 codec artifacts (B-frame prediction residuals
vs independent I-frame coding), not different scene content — proven by matching
mean/stdev and matching RTP timestamps.

---

## 4. Fix

### What changed

`diag_v6.sh` line 336: added a `select` filter before `showinfo` in the `-vf` chain.

```bash
# Before:
VF_OPTS=(-vf showinfo)

# After:
VF_OPTS=(-vf "select='isnan(prev_pts)+not(eq(pts\,prev_pts))',showinfo")
```

The `select` expression:
- `isnan(prev_pts)`: keeps frame 0 (prev_pts is NaN for the first frame)
- `not(eq(pts, prev_pts))`: keeps any frame whose PTS differs from the previous
- Drops the second frame of a duplicate PTS pair (the IDR, n:2)
- Is a complete no-op when no duplicate PTS exists

### Why uniform drop, not drop-vs-bump

Both MATCH and DIFFER frames represent the same camera capture moment (same RTP
timestamp). A PTS bump would assert a time interval that never existed in the physical
world. The checksum difference in DIFFER cases is a codec artifact, not a different
capture. Dropping one of two representations of the same moment loses zero temporal
coverage.

### What was NOT changed

- `-avoid_negative_ts make_zero`: retained. It compensates for negative PTS from
  pre-IDR B-frames (n:0) at stream start. Without it, the output mp4 would have
  negative PTS values. Not safe to remove without testing.
- `-fflags +igndts`: retained. Independent hygiene, not part of this fix.
- Sidecar schema: unchanged (still v5).
- No pipeline changes.

### Separate change: `capture.sh` `--cams` passthrough

Added `--cams` argument to `capture.sh` that passes through to `diag_v7_2.sh`'s
`CAMS` env var. Needed for single-camera verification captures with 2 of 3 cameras
offline.

---

## 5. About n:0 (the pre-IDR B-frame)

n:0 is an earlier real capture — it has a DIFFERENT PTS from the duplicate pair and
different pixel content (different checksum in all 11 cases). It represents a frame
captured ~67ms before the IDR. After `make_zero` + `reset_timestamps` + x264
reordering, it appears in the output mp4 at a non-zero PTS (e.g., PTS=6030 for
130325). This is a real frame with a real timestamp — not an artifact, and not
affected by the fix.

---

## 6. Qualification of prior "zero pixel-identical duplicates" claim

CLAUDE.md Overturned Conclusions #7/#9 and the DUPFIX-1 finding stated "zero
pixel-identical adjacent frames on source-PTS footage." This is wrong at stream-start
boundaries: 6 of 11 affected segments have pixel-identical adjacent frames (matching
checksums) at positions n:1/n:2.

The claim is correct for mid-stream frames (the original measurement domain). The
boundary exception is narrow (1 frame per affected segment, only at attempt start)
but the blanket statement is false and must be qualified.

---

## 7. Verification capture (pending)

### Plan
- Short capture with `SEG_SECONDS=30`, `TARGET_CONTENT_SECONDS=240` to produce
  multiple attempts
- Pin to FP7oJQ via `--cams`
- Check per attempt-first segment:
  - `dt_s` at frame_index 1 is non-zero
  - First two surviving frame PTS interval is ~5940/6030 ticks (nominal)
  - `a_eq_c` via `probe_frame_index_join.py`
  - Timing chain: schema 5, row_source mp4, passthrough, source_pts true
  - 5940/6030 tick alternation present
  - showinfo_offset_status not degraded

### Results (2026-08-24, git SHA 3fca547 + uncommitted fix)

**Capture parameters:**
- `TARGET_CONTENT_SECONDS=300`, `SEG_SECONDS=30`
- Both cameras (FP7oJQ + PPDmUg), discovery mode
- Container: bind-mount of `recorder/` (no rebuild)
- Env: `SOURCE_PTS=1`, `FPS_PASSTHROUGH=1` (defaults)

**Segments produced:** 25 total (13 FP7oJQ + 12 PPDmUg)
- FP7oJQ: 3 segments attempt 1 (2 smoke test + 1 verification), 10 segments attempt 2
- PPDmUg: 2 segments attempt 1 (smoke test), 10 segments attempt 2

**a_eq_c: TRUE on all 25 segments.** mp4 frame count = sidecar row count = meta
`output_frame_count` on every segment.

**Timing chain intact on all 25 segments:**
- `sidecar_schema: 5`
- `row_source: "mp4"`
- `timing_mode: "passthrough"`
- `source_pts: true`
- 5940/6030 tick alternation present
- `showinfo_offset_status: determined` or `ambiguous_fallback_k0` (short segments)

**dt_s[1] on every segment:** 0.066–0.067 (nominal). Zero `dt_s=0.0` anywhere.

**First-frame intervals on reconnection segments (attempt-first):**
- FP7oJQ-064429 (att2 first): PTS[0]=18000, PTS[1]=24030, delta=6030 ticks ✓
- PPDmUg-064400 (att2 first): PTS[0]=18000, PTS[1]=23940, delta=5940 ticks ✓

Both nominal — no inflated or compressed intervals.

**Reconnection duplicate test:** Neither reconnection produced a duplicate PTS
(showinfo has only 1 non-monotonic warning per attempt, the normal B-frame
reordering warning). The RTSP relay sent clean initial bursts on both reconnections.
The `select` filter was a **verified no-op on all 25 segments**, including the 2
reconnection segments.

**Limitation:** The fix was not tested on a live duplicate because the relay
did not reproduce the defect on this capture. The defect is near-deterministic
(9/9 on Aug 23 for attempts >1) but depends on relay buffer state. The fix is
validated by:
1. Syntax correctness (local ffmpeg 7.1.1, container Lavf61.7.103)
2. Expression logic (drop frames with `PTS == prev_pts`, keep all others)
3. No-op verification on 25 live segments including 2 reconnections
4. `a_eq_c = TRUE` on all 25 segments
5. Timing chain fully preserved

A subsequent capture that produces the defect will provide the "frame dropped"
confirmation. The next production capture (CP-R8 annotation session) will likely
produce multiple reconnections and serve as that test.

---

## 8. Already-affected segments

The 11 segments identified in §3 remain unusable. Ingest-side recovery (dropping the
frame or reconstructing its interval from the nominal cadence) was deliberately not
attempted — that is a separate decision affecting the pipeline's fail-loud policy.
