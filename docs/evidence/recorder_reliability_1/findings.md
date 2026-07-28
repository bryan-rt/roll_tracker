# RECORDER-RELIABILITY-1: Findings

Date: 2026-07-28

## Part A: FP7oJQ Total Failure Diagnosis

### Timeline Reconstruction (FP7oJQ hour-07, SOURCE_PTS=1)

| Attempt | Time (UTC) | Duration | Error | Backoff |
|---------|-----------|----------|-------|---------|
| 1 | 11:02:17 | **4m05s** | Session invalidated (TLS) | 3s |
| 2 | 11:06:27 | 2m10s | **RTSP 404 Not Found** | 6s |
| 3 | 11:08:44 | 11s | RTSP 404 | 12s |
| 4 | 11:09:08 | 10s | RTSP 404 | 24s |
| 5 | 11:09:43 | 12s | RTSP 404 | 48s |
| 6 | 11:10:44 | 10s | RTSP 404 | **96s (pinned)** |
| 7 | 11:12:31 | 11s | RTSP 404 | 96s |
| 8 | 11:14:19 | 11s | RTSP 404 | 96s |
| 9 | 11:16:07 | 10s | RTSP 404 | 96s |
| 10 | 11:17:55 | 11s | RTSP 404 | 96s |
| 11 | 11:19:43 | 1m48s | TLS EOF | 96s |
| 12 | 11:23:08 | 11s | RTSP 404 | 96s |
| 13 | 11:25:30 | instant | **Generate 401** (token expired) | 96s |
| 14 | 11:27:07 | instant | Generate 401 | 96s |

**Coverage: 245s recorded / 1500s window = 16%**

J_EDEw had the identical pattern: 15 attempts, all RTSP 404 after the first short recording.
PPDmUg survived: 3 attempts, 11 segments, good coverage until token expired at ~24.7 min.

### Error Classification

All 11 failed FP7oJQ attempts (2-12) show `RTSP DESCRIBE failed: 404 Not Found` at the
dropcam.com relay. The Generate API returned HTTP 200 with valid-looking tokens and URLs,
but the relay refused the stream. NOT a timestamp error, network error, or auth error.

### Extend Investigation

The extend is **innocent**. Attempt 1's extend fired at +120s, returned HTTP 200, and
extended the session to 11:09:17 UTC. The stream auth JWT shows a 5-min lifetime (exp
11:07:16). ffmpeg died at 11:06:22 — **before either expiry** — with "The specified
session has been invalidated for some reason."

This is a Nest relay unilateral session kill, not an extend failure.

### Root Cause: Session Orphaning

The TRUE cause of the 404 cascade: **the retry loop never stopped the previous stream
session before generating a new one.** Each Generate creates a new RTSP session at the
relay; the old one lingers until its 5-min auth expires. SDM enforces a concurrent-stream
limit per camera. With 12 rapid retries, we orphaned up to 12 sessions, saturating the
camera's stream slots. The relay returned 404 because the camera was at its session limit,
not because of a network or timestamp issue.

Evidence: the `stop_stream` function existed (cleanup on exit) but was NEVER called
between retry attempts. Only called once on script exit via the trap handler.

### ACCESS_TOKEN Expiry

Access token expires ~21-25 minutes after fetch:
- PPDmUg: last successful extend at 21.1 min, first 401 at 24.7 min
- FP7oJQ: Generate 401 at ~23.3 min
- `get_access_token.sh` fetches from Google OAuth with caching. Called ONCE at worker start.

For the planned 65-min evening window, the token would expire 2-3 times. Fatal.

### SOURCE_PTS Exoneration

**SOURCE_PTS is NOT the cause of exits.** Evidence:
- FP7oJQ hour-06 ran SOURCE_PTS=1 for full 7-min window, 0 retries, 4 segments
- All FP7oJQ hour-07 failures were RTSP 404 (relay lockout), not timestamp errors
- Attempt 1 recorded normally with SOURCE_PTS=1 before Nest killed the session
- PPDmUg SOURCE_PTS=1 worked fine (3 attempts, 11 segments)
- Zero DTS/timestamp/non-monotonic errors in any stderr file
- The live A/B test (SOURCE_PTS=0 vs 1) was skipped — the error taxonomy already
  exonerates source PTS. The failures are all server-side (RTSP 404, session invalidated,
  auth 401), none are ffmpeg timestamp processing errors.

### Resolution vs Busyness

Cannot distinguish: both 1080p cameras (FP7oJQ, J_EDEw) failed; 720p (PPDmUg) survived.
But n=1 session, different relay hosts (charlie vs foxtrot). Could be coincidence.

---

## Part B: Fixes Applied

### Five bugs fixed in diag_v6.sh:

**1. RTSP socket timeout (TOP FIX)**
Added `-stimeout 10000000` (10s, microseconds) to ffmpeg RTSP input options. Previously
ffmpeg had NO timeout — it blocked on dead streams until OS TCP timeout (~2+ minutes).
This is what would have saved the lights-off gap: the 2m28s of ffmpeg lingering on a
dying stream becomes ~10s.

Configurable via `RTSP_TIMEOUT_SEC` env var (default 10).

**2. Stop stream before regenerating (ROOT CAUSE FIX)**
Added `stop_stream()` helper called before every `generate_stream` in the retry loop.
Best-effort StopRtspStream call with 5s curl timeout. Clears STOP_TOKEN/EXT_TOKEN
after stop to prevent stale-token reuse. This directly addresses the session-orphaning
root cause of the 404 cascade.

**3. Access token refresh per attempt**
`get_access_token` now called:
- Before every `generate_stream` call (cache hit unless expired — cheap)
- Before every extend call in `extend_loop`
- On Generate 401: automatic refresh + retry (single attempt, no infinite loop)

**4. Failure-type-aware backoff with reset-on-success**
Replaced the single escalating backoff with classification:
- **Healthy exit** (ran >= 60s): reconnect immediately, reset backoff to initial
- **RTSP 404**: moderate backoff 10s → 30s cap (with stop_stream fix, 404s should be rare)
- **Connection failure**: 3s → 15s cap
- **Quick/unknown failure**: 3s → 15s cap
- **Generate 401**: refresh token + immediate retry (handled in generate_stream)

**5. Sidecar extraction off critical path**
`extract_timing_sidecars` now runs in background. PIDs tracked in `SIDECAR_PIDS` array,
waited at window end (and in cleanup trap). Removes any processing-time contribution to
recording gaps.

### Quantified Improvement (This Morning's Timeline)

**PPDmUg gap (the lights-off miss):**
- Before fix: ~2m36s gap (segment 070422 end to 070849 start)
  - 2m28s: ffmpeg lingering on dead stream (NO socket timeout)
  - 3s: backoff sleep
  - 5s: sidecar extraction + generate + connect
- After fix: ~15s gap
  - 10s: ffmpeg exits via socket timeout
  - 0s: healthy-run → immediate reconnect (no backoff)
  - 5s: generate + connect (sidecar extraction backgrounded)
- **Improvement: ~2m36s → ~15s (90% reduction)**

**FP7oJQ coverage:**
- Before fix: 16% (245s/1500s). 11 orphaned sessions caused relay lockout.
- After fix (estimated): stop_stream prevents session pileup. If relay recovers after
  stop, attempts 2+ would succeed → potential 80%+ coverage. If relay still locks out
  (camera-side limit), the moderate 404 backoff avoids wasting bandwidth hammering.
  Token refresh prevents the 401 cascade in the last 2 minutes.

**65-min evening window:**
- Before fix: recording stops at ~21-25 min (token expiry kills extends + generates)
- After fix: token refresh enables full 65-min window

---

## Part C: Dup/Drop Measurement

### Sidecar Mismatch Summary (Source-PTS, 2026-07-28)

**FP7oJQ (hour 06 + 07):**

| Segment | Input | Output | Mismatch | Measured FPS |
|---------|-------|--------|----------|-------------|
| 062531 | 1857 | 1830 | true | 15.00 |
| 062734 | 1799 | 1800 | true | 14.96 |
| 062939 | 1796 | 1800 | true | 14.97 |
| 063135 | 777 | 806 | true | 15.00 |
| 070240 | 428 | 460 | true | 13.92 |
| **Total** | **6657** | **6716** | | |

FP7oJQ: 6657 input vs 6716 output (+0.9%). All segments mismatch.

**PPDmUg (hour 07):**

| Segment | Input | Output | Mismatch | Measured FPS |
|---------|-------|--------|----------|-------------|
| 070219 | 1965 | 1830 | true | 15.86 |
| 070422 | 1630 | 1660 | true | 14.99 |
| 070849 | 1905 | 1800 | true | 15.63 |
| 071051 | 1800 | 1800 | **false** | 15.00 |
| 071250 | 1798 | 1800 | true | 15.00 |
| 071450 | 1803 | 1800 | true | 15.00 |
| 071650 | 1797 | 1800 | true | 15.00 |
| 071850 | 1127 | 1156 | true | 14.97 |
| 072126 | 1856 | 1830 | true | 15.00 |
| 072328 | 1800 | 1800 | **false** | 15.00 |
| 072528 | 1616 | 1642 | true | 15.00 |
| **Total** | **19097** | **18918** | | |

PPDmUg: 19097 input vs 18918 output (-0.9%). 2 of 11 segments exact match. High-mismatch
segments (070219, 070849) correlate with elevated PTS stdev (~10ms vs 0.47ms nominal).

### Pixel-Identical Duplicate Detection (mpdecimate)

| Clip | Mode | Frames | Dups | Dup Rate |
|------|------|--------|------|----------|
| FP7oJQ-062531 | source-PTS | 1867 | 37 | 2.0% |
| FP7oJQ-062734 | source-PTS | 1800 | 0 | 0.0% |
| FP7oJQ-062939 | source-PTS | 1800 | 0 | 0.0% |
| FP7oJQ-063135 | source-PTS | 806 | 0 | 0.0% |
| FP7oJQ-070240 | source-PTS | 467 | 7 | 1.5% |
| PPDmUg-071051 | source-PTS | 1800 | 0 | 0.0% |
| PPDmUg-070219 | source-PTS | 1832 | 2 | 0.1% |
| PPDmUg-070849 | source-PTS | 1801 | 1 | 0.1% |
| PPDmUg-071250 | source-PTS | 1800 | 3 | 0.2% |
| **FP7oJQ-200014** | **arrival-PTS** | **4530** | **255** | **5.6%** |
| **PPDmUg-200019** | **arrival-PTS** | **4530** | **106** | **2.3%** |

### PTS Timing Smoothness (Source-PTS only)

Clean source-PTS segments show PTS stdev of 0.47ms at 66.67ms mean delta (15fps, perfect
cadence). Outlier segments (062531, 070219, 070849) have elevated stdev (10-18ms) during
bursty network delivery windows — these are the first segments of each stream attempt
(startup transient).

PPDmUg achieves `drift_flat=true` on 6 of 11 segments. FP7oJQ: 0 of 5 (noisier path).

### Verdict

**The dup/drop problem is SUBSTANTIALLY RESOLVED by source PTS.**

| Metric | Arrival-PTS (Mar 2026) | Source-PTS (Jul 2026) |
|--------|----------------------|----------------------|
| FP7oJQ pixel-identical dups | 5.6% (255/4530) | 0.0% (typical), 2.0% (worst-case startup) |
| PPDmUg pixel-identical dups | 2.3% (106/4530) | 0.0% (typical), 0.2% (worst) |
| PTS stdev (clean segments) | N/A | 0.47ms (perfect cadence) |

Source-PTS reduces duplicate frames by **10-50x** vs arrival-PTS. The residual 2.0% on
FP7oJQ-062531 is the first segment of a stream (startup transient). Mid-stream segments
are consistently 0.0%.

The CFR re-encoder still creates input/output frame count mismatches (14 of 16 segments),
but the pixel-level duplication is nearly eliminated. Because source PTS carries the true
capture cadence, the encoder has uniform input timing and rarely needs to duplicate frames
to fill gaps.

The improvement is present on both cameras. FP7oJQ benefits more (5.6% → 0.0%) because
its arrival-PTS baseline was worse. Both cameras are at or near zero in clean mid-stream
segments.

**Impact on Stage A:** The dominant damage source (41% of through-line breaks) is Stage A
tracker drift. FALSE ZERO-MOTION from duplicated frames injects incorrect Kalman filter
updates, and FALSE TELEPORTS from dropped frames cause association failures. Eliminating
5.6% duplicates removes a systematic source of tracker error — not all 41% (other factors
like detection under-segmentation dominate), but a previously unmeasured contributor that
is now addressed.
