from __future__ import annotations

from bjj_pipeline.stages.tags.c0_scheduler import C0Scheduler, Candidate


def test_seeking_attempts_every_k_seek() -> None:
    s = C0Scheduler(enabled=True, k_seek=2, k_verify=10, n_ramp=10)
    tid = "t1"
    det = "d1"

    attempts = []
    for fi in range(0, 6):
        c = Candidate(frame_index=fi, timestamp_ms=fi * 33, detection_id=det, tracklet_id=tid)
        d = s.step(frame_index=fi, timestamp_ms=fi * 33, candidates=[c])[0]
        if d.decision == "attempt":
            attempts.append(fi)

    assert attempts == [0, 2, 4]


def test_success_transitions_to_verified_and_uses_k_verify() -> None:
    s = C0Scheduler(enabled=True, k_seek=1, k_verify=3, n_ramp=10)
    tid = "t1"
    det = "d1"

    # initial attempt at frame 0
    c0 = Candidate(frame_index=0, timestamp_ms=0, detection_id=det, tracklet_id=tid)
    d0 = s.step(frame_index=0, timestamp_ms=0, candidates=[c0])[0]
    assert d0.decision == "attempt"

    # simulate successful decode -> VERIFIED
    s.on_decode_success(tracklet_id=tid, frame_index=0)

    attempts = []
    for fi in range(1, 9):
        c = Candidate(frame_index=fi, timestamp_ms=fi * 33, detection_id=det, tracklet_id=tid)
        d = s.step(frame_index=fi, timestamp_ms=fi * 33, candidates=[c])[0]
        if d.decision == "attempt":
            attempts.append(fi)

    # verified cadence every 3 frames after last attempt at 0 -> 3,6
    assert attempts == [3, 6]
