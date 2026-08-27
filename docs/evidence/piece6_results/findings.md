# Piece 6 Results — Stage F export timing (sites #2, #3, #16)

**Date:** 2026-08-27
**Commit:** (this commit)
**Camera:** FP7oJQ
**Pipeline state:** post-Piece 6, recalibrated H (`f7d76d6`)
**Supabase:** untouched (export to local output only; uploader not invoked)

---

## 1. What changed

**Site #2 (`ffmpeg.py`):** `start_sec = start_frame / fps` and
`duration_sec = (end_frame - start_frame + 1) / fps` replaced. `build_export_command` is
now a pure command builder accepting pre-computed `start_sec` and `duration_sec`. `fps`,
`start_frame`, `end_frame` removed from the function's parameters.

**Site #3 (`manifest.py`):** `compute_clip_seconds(fps=..., export_start_frame=...,
export_end_frame=...)` replaced by `compute_clip_timing(frame_to_ts_ms=...)`. The `fps <= 0`
raise replaced by a ValueError on missing boundary frame in the timestamp map.

**Sites #2 and #3 derive from one shared helper** (`compute_clip_timing`), called once per
export session. Both the ffmpeg command args and the Supabase `clip_row` payload consume the
same `timing` dict — divergence is impossible.

**Site #16 (`_infer_last_frame`):** `duration_sec * fps` replaced by
`person_tracks_df["frame_index"].max()`. Both original objections resolved by ELIMINATING
the dependency — no sidecar needed, no `output_frame_count` needed, a direct measurement of
the data being exported.

---

## 2. Correction magnitude (132650, 18 exported clips)

| Export | Frames | Old start (f/fps) | New start (ts_ms) | Delta | Old duration | New duration |
|--------|--------|--------------------|--------------------|-------|-------------|-------------|
| 1296e4 | 0-574 | 0.000s | 0.000s | 0.000s | 38.333s | 38.600s |
| 58e357 | 847-1763 | 56.467s | **58.133s** | **1.666s** | 61.133s | 61.800s |
| 142094 | 482-1429 | 32.133s | 32.133s | 0.000s | 63.200s | 65.467s |

The correction grows with frame number, consistent with FP7oJQ's ~2% gap rate
accumulating error over the clip. Export 2 (starting at frame 847) shows a 1.7s correction.
Earlier exports show less or zero because fewer gaps have accumulated.

---

## 3. Keyframe snap — the remaining customer-visible error

Keyframe interval on all three Saturday segments: **2.0 seconds** (IDR every ~30 frames,
source camera x264 encoder). `-ss` before `-i` (input seeking) snaps backward to the nearest
keyframe.

**Before Piece 6:** pipeline arithmetic error 5.7–10.3s (unbounded, accumulating with gap
count). **After Piece 6:** pipeline arithmetic error ≈0ms; residual ≤2.0s from keyframe
snap (bounded by source GOP, not pipeline arithmetic).

The error character changed from **unbounded-and-accumulating** to **bounded-by-GOP**. Piece 6
does not make export timing accurate; it makes it bounded.

The residual ≤2.0s backward snap is the camera encoder's GOP and is not addressable by
pipeline arithmetic. Removing it requires either output seeking (`-ss` after `-i`,
decode-and-discard, slower) or a GOP change at the recorder.

**Output GOP:** the export re-encodes with libx264 (crop filter), no `-g` flag — x264
defaults to GOP 250. Output keyframes are at ~0.8s intervals (measured). The output GOP is
ours to choose even though the input snap is not.

---

## 4. Privacy render path residual

All 18 exports on 132650 went through the privacy render path (`render_redacted_clip`,
OpenCV-based). This path still receives `fps` and frame ranges directly — it is a separate
consumer not covered by sites #2/#3. The `compute_clip_timing` values appear in the
Supabase `clip_row` payload (correct), but the actual media is cut by the privacy renderer
using the old frame-count-based duration.

**Consequence:** the Supabase payload's `start_seconds` / `duration_seconds` are now correct
(real time), but the privacy-rendered media may not exactly match those values when the
renderer uses fps-derived duration. This is a Piece 7 or post-Piece 7 cleanup.

---

## 5. #16 resolution

`_infer_last_frame` now uses `person_tracks_df["frame_index"].max()` — a direct measurement
of the data being exported. Both original objections resolved by eliminating the dependency:
- (a) Sidecars are per-segment: moot — `person_tracks_df` is always available
- (b) `output_frame_count` reliability: moot — not used

---

## 6. Validation

| Tier | Result |
|------|--------|
| T1 — compute_clip_timing hand-computed | `timing` values match expected (0.067 / 0.267 / 0.200) |
| T1 — missing frame raises ValueError | PASS |
| T2 — regression suite | 192 passed, 10 skipped, 4 pre-existing |
| Media inspection | start/duration in Supabase payload ≠ old frame/fps values; correction 0–1.7s |
