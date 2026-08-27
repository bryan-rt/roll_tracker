# VFR Player Test 1 — Does the Android app play a variable-frame-rate clip correctly?

**Date:** 2026-08-27
**Device:** Pixel 7 Pro (`2A191FDH300C9Z`), Android, `c2.exynos.h264.decoder`
**App:** `com.example.mobile_app`, ExoPlayer `AndroidXMedia3/1.4.1`
**Clip:** `FP7oJQ-20260822-132650.mp4` (raw Nest segment, no CFR re-encode)
**Supabase:** hosted instance `zwwdduccwrkmkvawwjpc`

---

## 1. ffprobe ground truth

| Field | Value |
|-------|-------|
| `r_frame_rate` | 15/1 (container nominal) |
| `avg_frame_rate` | 14.708 (5292000/359801) |
| video frames | 1764 |
| **true duration** | **120.021333 s** |
| audio | AAC, 5626 frames = 120.02 s (fixed-rate reference clock) |

`r_frame_rate ≠ avg_frame_rate` is the VFR signature.

---

## 2. Prediction

If the player renders at the nominal 15fps, video exhausts at 1764/15 = **117.6s** while
audio runs to 120.02s → **~2.4s of A/V drift** accumulating through the clip (~1.2s at the
halfway point).

- Player reports **~120.0s** → it read the container duration. VFR likely handled correctly.
- Player reports **~117.6s** → it is computing from `r_frame_rate`. **That is the failure.**

---

## 3. Result: VFR handled correctly for linear playback on this device

### Two independent, non-circular confirmations

**1. Wall-clock playback = 120s (stopwatch).** Operator re-ran the test with a stopwatch: the
video plays for exactly 2 minutes. 120s wall-clock against a 120.021s container duration
rules out the 117.6s `r_frame_rate` failure case by a 2.4s margin — far outside stopwatch
error.

**2. A/V sync held through the full run.** Audio runs on a fixed-rate clock (AAC at
48kHz). Had video rendered at nominal 15fps, it would have exhausted 2.4s early and drift
would have been visible and audible near the end. The operator reported: "audio seems to be
in line with the clip and it looks smooth."

### Displayed-duration circularity

The app displays `clip.durationSeconds` (`main.dart:317`), which is `json['duration_seconds']`
from the clips row (`supabase_service.dart:47`) — the value WE inserted (120.021333). The
player screen (`ClipPlayerScreen`) shows the video with no UI controls and does not render
`controller.value.duration`. So the displayed "120 seconds" is our own value echoed back,
not a player-derived measurement. **The displayed duration is uninformative.** The stopwatch
and A/V sync carry the entire result.

---

## 4. Operator observations (verbatim)

> "I opened the app, found the video, it has a 120 second duration listed, I played the
> video until the end and the audio seems to be in line with the clip and it looks smooth.
> My app doesn't currently have any type of scrubbing mechanism built in though, so I can't
> pause/rewind/fastforward."

Stopwatch re-run (reported separately): video plays for exactly 2 minutes.

---

## 5. Log evidence

### Player initialization
```
09:36:07.786 I/ExoPlayerImpl(27103): Init 1c28806 [AndroidXMedia3/1.4.1] [cheetah, Pixel 7 Pro, Google, 37]
09:36:08.951 I/DMCodecAdapterFactory(27103): Creating an asynchronous MediaCodec adapter for track type video
09:36:09.019 D/CCodec  (27103): allocate(c2.exynos.h264.decoder)
09:36:09.022 I/CCodec  (27103): Created component [c2.exynos.h264.decoder] for [c2.exynos.h264.decoder]
09:36:09.031 I/MediaCodec(27103): MediaCodec will operate in async mode
09:36:09.059 I/DMCodecAdapterFactory(27103): Creating an asynchronous MediaCodec adapter for track type audio
09:36:09.078 I/MediaCodec(27103): MediaCodec will operate in async mode
```

### Player release (clean)
```
09:40:30.969 I/ExoPlayerImpl(27103): Release 1c28806 [AndroidXMedia3/1.4.1] [cheetah, Pixel 7 Pro, Google, 37] [media3.common, media3.exoplayer, media3.decoder, media3.datasource, media3.extractor]
09:40:30.971 I/MediaFocusControl( 1474): abandonAudioFocus() from uid/pid 10317/27103
```

### Errors / exceptions
None. CCodec `BAD_INDEX` warnings are standard Pixel initialization noise, not playback
errors. No `IllegalStateException`, no renderer errors.

### HTTP / signed-URL errors
None. No 403, no expiry, no connection failures.

### Buffering / rebuffering
No buffering, rebuffering, or underrun events logged from the app process. Over a hosted
stream with no rebuffering, "smooth" is a meaningful signal rather than a network-masked one.

### Frame rate / PTS / timestamps
ExoPlayer does not log duration or frame rate at INFO level by default. No explicit PTS or
frame-rate lines from the player. The duration is not extractable from logs at this
verbosity.

---

## 6. Limitations — record explicitly

**1. SEEK UNTESTED.** The app has no scrubbing mechanism. A player can honour PTS during
linear playback and still land wrong on a seek — that is a common divergence point for VFR.
This matters for Piece 6, which makes Stage F seek to real timestamps. Residual for Piece 7.

**2. ONE DEVICE.** Pixel 7 Pro, `c2.exynos.h264.decoder` (Exynos hardware decoder).
Android hardware decoders vary across chipsets; this does not establish behaviour on Samsung
Exynos (different generation), MediaTek, Qualcomm, or older devices.

---

## 7. Conclusion

ExoPlayer via `video_player` (AndroidXMedia3/1.4.1) handles a raw Nest VFR segment correctly
**for linear playback on this device.** The container duration (120.021s) is honoured, not
the `r_frame_rate`-derived 117.6s. A/V sync holds through the full clip. No errors, no
buffering, no rebuffering.

This is NOT "VFR works" unqualified. Seek behaviour and cross-device compatibility are
untested.

---

## 8. Cleanup (prepared, NOT executed)

Synthetic clips row and storage object remain in the hosted instance for potential Piece 7
re-use. Execute when testing is complete:

```sql
-- Remove the synthetic clips row
DELETE FROM public.clips WHERE id = 'a5541b7d-3a74-4e5b-8cff-c31d2026e50e';
```

```bash
# Remove the storage object
cd backend/supabase/supabase
npx supabase storage rm \
  "ss:///match-clips/gym/c8a592a4-2bca-400a-80e1-fec0e5cbea77/camera/FP7oJQ/date/2026-08-22/video/FP7oJQ-20260822-132650/clips/vfrtest_132650.mp4" \
  --linked --experimental
```
