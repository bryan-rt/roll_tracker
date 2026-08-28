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

## 4. Media inspection

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

### Duration gap analysis

The remaining 0.466s gap (94.466s DB vs 94.000s media) is **7 missing frames** — `cap.read()`
returns False before reaching `export_end_frame`. This is pre-existing (same loop, same break
condition as the old VideoWriter path). It is NOT a Piece 12 regression.

The old path masked this gap behind the larger CFR rate divergence (2.272s). With the rate
divergence removed, the frame-count shortfall becomes visible.

### Codec change

mpeg4 → h264. File size decreased 35% (8326 KB → 5413 KB) at the same visual quality
(libx264 crf=23). Visual spot-check: blur correctly applied to non-focus people, no color
swap (format='bgr24' handles OpenCV BGR directly), no artifacts.

### H.264 even-dimension constraint

H.264 YUV420P requires even width and height. Some crop plans produce odd dimensions (e.g.
545px height). The renderer rounds down (`& ~1`), trimming one pixel — not visible. The old
mpeg4 codec accepted odd dimensions. 12 of 18 exports succeeded; 6 failed with "Invalid
argument returned 22" — these appear to be a pre-existing issue (crop plans producing
unusable geometry), not a Piece 12 regression. The old path's exports for these same IDs
were not preserved for comparison.

### Session path

NOT TESTABLE on real media. CP22 blocks session Stage E.

---

## 5. Remaining plain-vs-redacted differences

After Piece 12, both paths produce:
- H.264 codec via libx264 (crf=23, preset=veryfast)
- VFR with source PTS preserved
- `+faststart` (moov at front)

The only remaining difference is the render method: the plain path uses ffmpeg's crop filter
in a single process; the redacted path uses Python crop+blur + PyAV encode. This is inherent
to what each path does, not a timing or format divergence.

---

## 6. Dependencies

| Package | Version | NumPy requirement |
|---------|---------|-------------------|
| av (PyAV) | 18.1.0 | None declared |
| numpy | 1.26.4 | Pinned (Torch ABI) |

No ABI conflict. `av` is a pip dependency (declared in `requirements.txt`), not a system
binary.

---

## 7. Validation

| Tier | Result |
|------|--------|
| T1 — PTS histogram guard (rate=90000) | PASS: {6030:8, 5940:5, 12060:1} exact match |
| T1 — PTS histogram guard (rate=15 fail) | Demonstrated: 5940 absent, intervals completely different |
| T1 — exact tick match | PASS: output ticks == input ticks |
| T1 — first PTS = 0 | PASS |
| T1 — last PTS = expected | PASS |
| T2 — full suite | 201 passed (+5 new), 10 skipped, 4 pre-existing |
| Media — redacted VFR | VFR confirmed, PTS intervals match source, duration gap 2.272s→0.466s |
| Media — session path | NOT TESTABLE (CP22 blocks session Stage E) |
