# Piece 7 Results — Stage F output format (sites #12, #14, #15)

**Date:** 2026-08-27
**Commit:** (this commit)
**Camera:** FP7oJQ
**Pipeline state:** post-PIECE6-FIX-1, Shape 3 (hybrid VFR/CFR)
**ffmpeg:** 7.1.1 (Homebrew, Apple clang 16.0.0)

---

## 1. Shape decision

**Shape 3 (hybrid):**
- **Plain path** (ffmpeg re-encode with crop): VFR-preserving via `-fps_mode passthrough`
  and `-enc_time_base -1`. Source PTS intervals preserved through the re-encode.
- **Redacted path** (cv2.VideoWriter): CFR at `1.0 / nominal_dt_s` from the sidecar.
  cv2.VideoWriter takes a scalar fps; there is no per-frame timestamp API. This is a
  hard constraint, not a choice.

The two paths produce different timing characteristics. This is explicit and deliberate:

| | Plain path | Redacted path |
|---|---|---|
| Codec | h264 (libx264) | mpeg4 (cv2 mp4v) |
| Timing | VFR (source PTS preserved) | CFR at nominal_fps |
| Frame count | Source frame count for the time range | Source frame count (1:1 decode) |
| Duration | Matches source content duration | Shorter than source by the VFR/CFR divergence |

---

## 2. Sites closed

**Site #12 (`session_f_run.py:451`, `fps = ... else 30.0`):** DELETED, not fixed. The
consumer chain was dead: `fps` was stored in `SourceClipInfo.fps` which was never read.
PIECE6-FIX-1 removed the last live consumer (`_extract_session_clip`'s `export_clip` call).
`SourceClipInfo.fps` field also deleted. `probe_video_metadata` retained for width/height.

**Site #14 (`redact.py:392`, `cv2.VideoWriter` rate):** Fixed. `render_redacted_clip`
now receives `nominal_fps` (= `sidecar_data.nominal_fps` = `1.0 / nominal_dt_s`) instead
of the probed `video_meta.fps`. Single source: the sidecar.

**Site #15 (`run.py:339-340`, independent fps source):** Closed. The independent
`probe_video_metadata` → `video_meta.fps` extraction is deleted. `nominal_fps` from the
sidecar is the sole fps source in Stage F. `probe_video_metadata` retained for width/height
only. The two-source divergence class that produced the CP4.C frame-offset bug is eliminated.

**Site #13 (`multiplex_runner.py:406`, `fps = 30.0`):** DEFERRED to Piece 9. Live consumer
is `MuxVisualizer` (debug visualization VideoWriter) and the `manifest.fps` backfill (which
feeds `post_pipeline_annotator.py:217`). All three visualization fps scalars (#13,
`visualize.py:408`, `post_pipeline_annotator.py:217`) are the same class — debug/eval
VideoWriter rate. Grouping them in Piece 9 is cleaner than splitting across pieces. After
this commit, `run.py`'s `getattr(manifest, "fps", 0.0)` fallback is deleted, so the
`manifest.fps` backfill no longer feeds the export path.

---

## 3. VFR flag verification

**Flags:** `-fps_mode passthrough -enc_time_base -1` (ffmpeg 7.1.1).
`-fps_mode` confirmed present (`ffmpeg -h full`; replaces deprecated `-vsync`).

**Throwaway test:** 10s crop export from FP7oJQ-132650 at ~50s.

| | VFR (with flags) | CFR (default, no flags) |
|---|---|---|
| r_frame_rate | 15/1 | 15/1 |
| avg_frame_rate | 213000/15001 (14.199) | 71/5 (14.2) |
| nb_frames | 142 | 142 |
| PTS intervals | 66-67ms + 133-134ms gaps | uniform 66.667ms |

Frame count identical (142 vs 142). No duplication or drops from the flags. PTS interval
distribution proves the distinction: VFR preserves the source's real 66/67ms alternation
plus 133/134ms periodic gaps (FP7oJQ grid mismatch from CP-R11); CFR quantizes to a
uniform 1/15s grid.

---

## 4. Media inspection — clip-level path

### Plain path (privacy disabled)

Export: `mengage_6458b385129ce9b2.mp4` from FP7oJQ-20260822-132650.

| Field | Value |
|-------|-------|
| codec_name | h264 |
| r_frame_rate | 15/1 |
| avg_frame_rate | 78570000/5388001 (14.582) |
| nb_frames | 873 |
| duration | 60.000000s |

`r_frame_rate != avg_frame_rate` — **VFR confirmed**. Duration matches `compute_clip_timing`
(60.0s). Codec is h264 (libx264 re-encode with crop).

### Redacted path (privacy enabled)

Same export ID re-exported with `privacy_mode: blur_non_focus_bbox`.

| Field | Value |
|-------|-------|
| codec_name | mpeg4 |
| r_frame_rate | 597/40 (14.925) |
| avg_frame_rate | 597/40 (14.925) |
| nb_frames | 875 |
| duration | 58.626466s |

`r_frame_rate == avg_frame_rate` — **CFR confirmed** at `nominal_fps` = 1/0.067 = 14.925.

### CFR divergence (redacted path) — quantified

The source content spans 60.0s (from `compute_clip_timing`). The redacted output is
58.63s — **1.37s shorter**. This is a known, quantified divergence forced by
cv2.VideoWriter:

- Source average rate: ~14.582 fps (VFR, with ~8% periodic gaps)
- VideoWriter output rate: 14.925 fps (CFR at nominal)
- The writer stamps 875 frames at 14.925fps = 58.63s
- The content's real duration is 60.0s (875 frames at the real average rate)
- Divergence: ~2.3% faster, scaling linearly (~2.74s over 120s)

This is the same magnitude as the failure VFR-PLAYER-TEST-1 was built to detect (~2.4s
over 120s). VFR-PLAYER-TEST-1 confirmed the player does not introduce that drift — but the
export pipeline introduces ~2.7s of its own on every redacted clip.

**FIXED by Piece 12.** cv2.VideoWriter replaced by PyAV with `PTS_TIMEBASE_HZ=90000`.
Redacted output is now VFR H.264 with source PTS preserved. DB/media duration gap reduced
from 2.272s (2.4%) to 0.466s (0.5%). The remaining 0.466s is 7 missing frames from
`cap.read()` ending early — pre-existing, same in both paths. File size reduced 35%
(mpeg4→h264). See `docs/evidence/piece12_results/findings.md`.

### Session path

NOT TESTABLE on real media. CP22 crashes session Stage E → no session `match_sessions.jsonl`.
PIECE6-FIX-1's value tests carry the `ts_offset_ms` derivation correctness.

---

## 5. GOP snap — unchanged

The VFR flags do not change `-ss` input seeking behavior. Keyframe snap is still bounded by
the source camera's GOP (2.0s on FP7oJQ, IDR every ~30 frames). The error character remains
**bounded-by-GOP** (Piece 6 finding). The output GOP is libx264 default (~250 frames).

Removing the residual requires output seeking (`-ss` after `-i`) or a GOP change at the
recorder. Backlog item, unchanged from Piece 6.

---

## 6. buffer_frames

`buffer_frames = consolidate_buffer_sec * nominal_fps` (was `* fps` from probe).

On 132650: `5.0 * 14.925 = 74.6 → 75 frames` (was `5.0 * 15.0 = 75.0 → 75`). Negligible
change for this segment. On a segment with a different `nominal_dt_s` the values would
diverge; the sidecar value is correct by construction.

---

## 7. Validation

| Tier | Result |
|------|--------|
| T1 — session_f_extract tests | 4 passed (PIECE6-FIX-1, updated for fps removal) |
| T2 — full suite | 196 passed, 10 skipped, 4 pre-existing |
| Media — plain path | VFR confirmed: r != avg, h264, duration matches compute_clip_timing |
| Media — redacted path | CFR confirmed: r == avg at nominal_fps, mpeg4, divergence quantified |
| Media — session path | NOT TESTABLE (CP22 blocks session Stage E) |
