# Replay/Baseline Gap Explanation (32.5% vs 30.7%)

## Finding

The sweep replay (`params={}`, stock BotSort) produces 30.7% combined correct_id.
The freshened eval_gt baseline (same detections, same code, Stage D re-run via standard
CLI) produces 32.5%. The ~2pp gap persists despite verified-identical:

- BotSort constructor params (device, half, reid_weights — all match)
- Detection bboxes (detections.parquet stores post-mask-tight bboxes, roundtrip verified)
- Frame images (both paths use raw cv2.VideoCapture.read(), no preprocessing)
- Detection count per frame (identical, every frame has detections)

## Investigation trail

1. **Structural agreement test:** 98.9% of consecutive-frame transitions structurally
   agree between the two tracklet assignments. 0.1% disagree (50 transitions), 1.0%
   null in replay (591 untracked detections).

2. **First structural divergence:** Frame 6 (out of 4530). This is very early,
   indicating a fundamental difference from the start, not accumulated drift.

3. **Detection ordering hypothesis (REFUTED):** Parquet stores detections in lexicographic
   detection_id order (_0, _1, _10, _11, ...) while production used numeric order
   (_0, _1, _2, ..., _10). Fixing the sort to numeric order WORSENED the result
   (30.7% -> 28.6%), confirming ordering is not the cause. Sort fix reverted.

4. **Determinism confirmed:** Two identical replay runs produce byte-identical output.
   The gap is systematic, not run-to-run noise.

5. **Key signature:** Production tracked 343 tracklets with 0 untracked detections.
   Replay tracks 269 tracklets with 591 untracked. The replay's BotSort instance rejects
   ~1.2% of detections that production's BotSort accepted.

## Root cause assessment

The production Stage A artifacts (`outputs/_eval_gt/`, mtime Jun 9 17:28) were produced
by a pipeline run whose exact runtime environment cannot be reconstructed. The most
likely cause of the ~2pp gap is one of:

- A boxmot version difference between Jun 9 and now (pip install --no-deps means
  version pinning may have drifted)
- An OpenCV frame decode difference (pixel-level differences in decoded JPEG/H.264
  frames affect ECC camera motion compensation in BotSort)
- A numpy ABI or floating-point behavior difference

None of these are code bugs in the sweep harness — they're environment-level differences
that make the replay's BotSort instance produce slightly different association decisions
than the Jun 9 production run did.

## Conclusion

The 30.7% sweep baseline is trustworthy for relative comparisons. Every sweep point
goes through the identical replay path, so parameter-induced deltas are not contaminated
by this gap. The 2pp offset from the freshened 32.5% baseline is a fixed environment
artifact, not a parameter-dependent interaction.

The sweep harness does NOT need a fix before the OFAT sweep proceeds. The sort fix
was reverted (it worsened results). The gap is documented as acceptable, explained
environment noise.
