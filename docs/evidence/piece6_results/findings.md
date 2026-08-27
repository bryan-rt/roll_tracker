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

**PIECE6-FIX-1 correction (2026-08-27):** The session export path
(`session_f_run.py:_extract_session_clip`) was left half-migrated in this commit. Piece 6
added `frame_to_ts_ms` and `compute_clip_timing` to the session function but did not update
the `export_clip` call sites in either the single-segment or multi-segment branches — they
still passed the removed `fps`, `start_frame`, `end_frame` kwargs. Both branches would raise
`TypeError` if exercised. Discovered during Piece 7 Pass 1; fixed in PIECE6-FIX-1. Piece 6's
media inspection covered the clip-level path only; the session path has never run on
production footage.

---

## 2. Correction magnitude (132650, 18 exported clips)

| Export | Start frame | Old start (f/15fps) | New start (ts_ms) | Delta |
|--------|------------|--------------------|--------------------|-------|
| 1296e4 | 0 | 0.000s | 0.000s | 0.000s |
| e8d127 | 205 | 13.667s | 13.667s | 0.000s |
| 142094 | 482 | 32.133s | 32.133s | 0.000s |
| 85cf8f | 631 | 42.067s | 42.667s | 0.600s |
| e9a0fa | 699 | 46.600s | 47.600s | 1.000s |
| 58e357 | 847 | 56.467s | **58.133s** | **1.666s** |

The correction grows with frame number: 0s at frame 0, 0.6s at frame 631, 1.7s at frame
847. The largest `export_start_frame` in these 18 exports is 847 (of 1764 total frames).

**Reconciliation with the audit's 5.7-10.3s projection:** The audit attached the 5.7s
(frame 1000) and 10.3s (frame 1800) figures to **FP7oJQ-20260807-102006** at its **8%
gap rate** (`timing_audit_1/findings.md:289`). Segment 132650 has a **2.1%** gap rate,
so accumulated error is proportionally smaller. At 2.1%, extrapolating to frame 1800
yields ~2.5s, not 10.3s. The audit figures remain valid for 8%-gap footage; this
measurement validates the mechanism at a lower rate. To demonstrate the full 5.7-10.3s
correction, an 8%-gap segment would need to be exported with matches starting at frame
1000+.

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

## 4. Privacy render path

All 18 exports on 132650 went through the privacy render path (`render_redacted_clip`,
OpenCV-based). This renderer decodes frame-by-frame via `CAP_PROP_POS_FRAMES` (exact
frame selection for mask overlay) — it genuinely needs frame indices, not seek times.
`fps` is the `cv2.VideoWriter` output rate scalar (Piece 7 #12, FIX-SCALAR class).

Both paths now receive `timing["start_seconds"]` and `timing["duration_seconds"]` from
the shared `compute_clip_timing` helper. The privacy path receives them for consistency
(passed to `render_redacted_clip` as `start_sec`/`duration_sec`). The equivalence between
`CAP_PROP_POS_FRAMES` seeking and `timestamp_ms`-derived timing is guaranteed on post-R13a
footage (Piece 0b §10 A2: POS_MSEC = timestamp_ms at zero deviation).

The Supabase payload carries the correct `start_seconds` / `duration_seconds` for both
paths. The output rate scalar (`fps` in VideoWriter) remains a FIX-SCALAR site for Piece 7.
Note: redacted clips are written at a FIXED output rate — this is precisely the CFR/VFR
question Piece 7 decides.

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
