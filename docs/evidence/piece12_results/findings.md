# Piece 12 Results — VFR redaction rendering (replace cv2.VideoWriter)

**Date:** 2026-08-27
**Commit:** (this commit)
**Camera:** FP7oJQ-20260822-132650
**Mechanism:** PyAV with rate=90000 (source 90kHz timebase)
**ffmpeg:** 7.1.1 (Homebrew) | **av:** 18.1.0 | **numpy:** 1.26.4

---

## 1. The defect fixed

Redacted clips were CFR at `1/nominal_dt_s` (14.925fps) via cv2.VideoWriter. Source content
is VFR at avg ~14.582fps. A 60s match exported as 58.63s — **2.3% fast, ~2.7s short over
two minutes.** The `clips` table said 60.0s; the file played 58.6s.

Privacy mode is the production default. Zero plain-path exports exist. Every athlete-facing
clip had this defect.

---

## 2. Mechanism

**PyAV with `rate=PTS_TIMEBASE_HZ` (90000).** Single process, no intermediate files, no
system binary.

Architecture: `cv2.VideoCapture.read()` + `CAP_PROP_POS_MSEC` → numpy crop+blur →
`av.VideoFrame.from_ndarray(format='bgr24')` with `pts = round((pts_ms - base_pts_ms) * 90)`
→ `stream.encode()` → `container.mux()` → MP4 with `+faststart`.

**Why 90000:** The source uses a 90kHz PTS timebase. At this rate, the codec context has
1-tick = 1/90000s resolution, preserving the source's exact 5940-tick (66ms) and 6030-tick
(67ms) intervals. At `rate=15`, the codec context quantizes PTS to a 6000-tick (66.67ms) grid,
collapsing the 5940/6030 distinction — silently reintroducing the CFR defect.

**Integer-ms precision is lossless** (Piece 0b §10). `CAP_PROP_POS_MSEC` returns exact integer
ms on post-R13a footage (5940/90=66, 6030/90=67). `round(ms * 90)` produces exact 90kHz ticks.
No quantization introduced.

**Output PTS are clip-relative** (first frame = 0). This matches the plain ffmpeg path, which
produces clip-relative PTS via `-ss`.

**POS_MSEC after cap.read()** reports the PTS of the frame just decoded, matching FrameIterator
ordering (frame_iterator.py:52-59). Verified empirically: after seeking to frame 100 and
reading, POS_MSEC = 6733.0 (frame 100's PTS, not frame 101's).

---

## 3. Guard: PTS_TIMEBASE_HZ

Three layers protecting the 90000 constant:

1. **Named constant** (`PTS_TIMEBASE_HZ = 90_000`) with inline reasoning and the 66×90=5940 /
   67×90=6030 verification.
2. **Regression test** (`test_redact_vfr.py::TestPtsHistogramGuard`): encodes a fixture with
   known non-uniform intervals and asserts the output PTS tick histogram retains distinct
   5940 and 6030 entries. **Demonstrated failing at rate=15** (tick intervals become
   {6174720, 6082560, 12349440} — different scale, 5940 absent) **and passing at rate=90000**
   (exact {6030: 8, 5940: 5, 12060: 1} match).
3. **Docstring invariant**: "output PTS ticks must equal input PTS ticks."

---

## 4. H.264 codec change

### Even-dimension constraint

H.264 YUV420P requires even width and height. Some crop plans produce odd dimensions (e.g.
545px height). The renderer rounds down (`& ~1`), trimming one pixel — not visible. The old
mpeg4 codec accepted odd dimensions.

### Duplicate-PTS condition (MUXER-PTS-1 in the export path)

Piece 12's H.264 codec change surfaced a latent duplicate-PTS condition that mpeg4 tolerated
by ignoring timestamps. The RTSP relay sends two H.264 frames at the same RTP timestamp on
reconnect (MUXER-PTS-1). The MP4 muxer rejects duplicate PTS with EINVAL (returned 22). 6 of
18 exports failed — all starting at or near frame 0, where the duplicate lives (frame index 2
in the sidecar). The old mpeg4 writer was unaffected because it ignored timestamps entirely.

**Fix (same task):** skip duplicate-PTS frames with `n_dup_pts_skipped` counted. The first
frame of a duplicate pair is kept; the second is skipped. Both carry the same capture instant
(MUXER-PTS-1). The two frames are pixel-identical in 6 of 11 measured segments and differ only
in B-frame prediction residuals in the other 5 — the choice is genuinely arbitrary for decoded
output. First is chosen because it is the frame the decoder emits at that PTS and requires no
lookahead.

**Result:** 18 of 18 exports succeeded after the fix.

**n_dup_pts_skipped per export:** max 1, only on the 6 exports whose range includes frame 0
(where the MUXER-PTS-1 duplicate lives at frame index 2). The 12 exports starting mid-file
have 0 duplicates. This is exactly consistent with the sidecar data: one `dt_s==0` per segment,
at the stream-start boundary.

### Codec change

mpeg4 → h264. File size decreased 35% (8326 KB → 5413 KB) at the same visual quality
(libx264 crf=23). Visual spot-check: blur correctly applied to non-focus people, no color
swap (format='bgr24' handles OpenCV BGR directly), no artifacts.

---

## 5. Media inspection

### Redacted export (Piece 12, VFR h264)

Export: `mengage_e94adc0987a79e9f.mp4` (redacted, privacy=blur_non_focus_bbox).

| Field | Piece 7 (old, CFR mpeg4) | Piece 12 (new, VFR h264) |
|-------|--------------------------|--------------------------|
| codec | mpeg4 | **h264** |
| r_frame_rate | 597/40 (14.925) | 15/1 |
| avg_frame_rate | 597/40 (14.925) | 123750000/8460091 (14.628) |
| VFR? | **No** (r = avg) | **Yes** (r ≠ avg) |
| nb_frames | 1376 | 1375 |
| duration | 92.194s | **94.000s** |
| compute_clip_timing | 94.466s | 94.466s |
| **DB/media gap** | **2.272s (2.4%)** | **0.466s (0.5%)** |
| file_size | 8326 KB | **5413 KB (−35%)** |

**PTS tick intervals:** {6030: 894, 5940: 444, 12060: 10, 11970: 26}. Source's real 66/67ms
alternation plus 133/134ms periodic gaps — preserved exactly.

**First PTS tick:** 0 (clip-relative). Matches plain path.

### Duration gap analysis (18 exports)

The old 2.272s CFR rate divergence is **eliminated** across all 18 exports. A smaller residual
remains in three categories:

| Category | Count | Magnitude | Cause |
|----------|-------|-----------|-------|
| Effectively zero | 12 | ≤0.001s | No residual. |
| B-frame reorder artifact | 5 | ~67ms (1 frame) | Container `format_dur` computed from max(DTS), not max(PTS). H.264 B-frame reordering puts max(DTS) below max(PTS). All frames present with correct PTS; playback uses PTS and is correct. |
| cap.read() shortfall | 1 | 0.466s (7 frames) | Pre-existing: `cap.read()` returns False before reaching `export_end_frame`. Same loop, same break condition as the old path. |

**B-frame reorder detail:** The 5 gap clips have a PTS-DTS depth of 200ms (3 frame intervals)
at the clip end, while the 12 no-gap clips have 133ms (2 frame intervals). The ~67ms gap is
exactly one frame interval — the difference in B-frame reorder depth at the final GOP. This is
a container metadata artifact; all frames are present, PTS are correct, playback duration is
correct. The gap clips and the 6 dup-skip clips are completely disjoint sets (zero overlap).

**Dup-skip vs gap overlap:** Explicitly verified — zero overlap. The 6 dup-skip clips all start
at frame 0 (MUXER-PTS-1 duplicate at frame 2). The 5 gap clips all start mid-file with 0
duplicates. Different mechanisms, different clips.

### Session path

NOT TESTABLE on real media. CP22 blocks session Stage E.

---

## 6. Remaining plain-vs-redacted differences

After Piece 12, both paths produce:
- H.264 codec via libx264 (crf=23, preset=veryfast)
- VFR with source PTS preserved
- `+faststart` (moov at front)

The only remaining difference is the render method: the plain path uses ffmpeg's crop filter
in a single process; the redacted path uses Python crop+blur + PyAV encode. This is inherent
to what each path does, not a timing or format divergence.

---

## 7. Dependencies

| Package | Version | NumPy requirement |
|---------|---------|-------------------|
| av (PyAV) | 18.1.0 | None declared |
| numpy | 1.26.4 | Pinned (Torch ABI) |

No ABI conflict. `av` is a pip dependency (declared in `requirements.txt`), not a system
binary.

---

## 8. Validation

| Tier | Result |
|------|--------|
| T1 — PTS histogram guard (rate=90000) | PASS: {6030:8, 5940:5, 12060:1} exact match |
| T1 — PTS histogram guard (rate=15 fail) | Demonstrated: 5940 absent, intervals completely different |
| T1 — exact tick match | PASS: output ticks == input ticks |
| T1 — first PTS = 0 | PASS |
| T1 — last PTS = expected | PASS |
| T1 — duplicate-PTS skip | PASS: 2 dup-PTS frames in, 1 out; output ticks == input minus dup, elementwise; counter == 1 |
| T2 — full suite | 201 passed (+6 new), 10 skipped, 4 pre-existing |
| Media — redacted VFR | VFR confirmed, PTS intervals match source |
| Media — duration gap | 2.272s CFR gap eliminated; residual: 12×≤0.001s, 5×~67ms (B-frame metadata), 1×0.466s (cap.read shortfall) |
| Media — dup-PTS skip | 6 exports skipped 1 dup each (frame 0 start only); 12 exports: 0 dups |
| Media — session path | NOT TESTABLE (CP22 blocks session Stage E) |
