from __future__ import annotations

from bjj_pipeline.stages.tags.c0_scheduler import C0Scheduler, Candidate
from bjj_pipeline.stages.tags.c0_triggers import C0TriggerEngine


def test_overlap_trigger_enters_ramp_up_from_verified() -> None:
    trig = C0TriggerEngine(
        {
            "enabled": True,
            "overlap": {"iou_thresh": 0.2, "window_frames": 2},
            "motion": {"enabled": False},
        }
    )

    sched = C0Scheduler(enabled=True, k_seek=1, k_verify=10, n_ramp=5, trigger_cooldown_frames=0)

    sched.on_decode_success(tracklet_id="t1", frame_index=0)

    for fi in [1, 2]:
        c1 = Candidate(
            frame_index=fi,
            timestamp_ms=fi * 33,
            detection_id=f"d{fi}_1",
            tracklet_id="t1",
            x1=0,
            y1=0,
            x2=100,
            y2=100,
        )
        c2 = Candidate(
            frame_index=fi,
            timestamp_ms=fi * 33,
            detection_id=f"d{fi}_2",
            tracklet_id="t2",
            x1=10,
            y1=10,
            x2=110,
            y2=110,
        )
        events = trig.update(frame_index=fi, timestamp_ms=fi * 33, candidates=[c1, c2])
        sched.apply_triggers(
            frame_index=fi,
            timestamp_ms=fi * 33,
            trigger_events=[
                {"tracklet_id": e.tracklet_id, "trigger": e.trigger, "details": e.details}
                for e in events
            ],
        )

    d = sched.step(
        frame_index=2,
        timestamp_ms=66,
        candidates=[Candidate(frame_index=2, timestamp_ms=66, detection_id="d2", tracklet_id="t1")],
    )[0]
    assert d.state_after == "RAMP_UP"


def test_motion_trigger_vel_jump_fires() -> None:
    trig = C0TriggerEngine(
        {
            "enabled": True,
            "overlap": {"window_frames": 0},
            "motion": {
                "enabled": True,
                "prefer_metric": True,
                "dv_thresh_mps": 1.0,
                "a_thresh_mps2": 9999.0,
            },
        }
    )

    c0 = Candidate(frame_index=0, timestamp_ms=0, detection_id="d0", tracklet_id="t1", x_m=0.0, y_m=0.0)
    assert trig.update(frame_index=0, timestamp_ms=0, candidates=[c0]) == []

    c1 = Candidate(frame_index=1, timestamp_ms=100, detection_id="d1", tracklet_id="t1", x_m=0.0, y_m=0.0)
    trig.update(frame_index=1, timestamp_ms=100, candidates=[c1])

    c2 = Candidate(frame_index=2, timestamp_ms=200, detection_id="d2", tracklet_id="t1", x_m=1.0, y_m=0.0)
    events = trig.update(frame_index=2, timestamp_ms=200, candidates=[c2])
    assert any(e.trigger == "vel_jump" for e in events)
