from __future__ import annotations

import pandas as pd

from bjj_pipeline.contracts.f0_manifest import init_manifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.detect_track.outputs import StageAWriter
from bjj_pipeline.stages.orchestration.pipeline import _validate_stage_outputs


def test_stage_a_writes_contact_points_parquet_and_validates(tmp_path):
    clip_id = "cam03-20260103-124000"
    camera_id = "cam03"
    layout = ClipOutputLayout(clip_id=clip_id, root=tmp_path)

    writer = StageAWriter(layout=layout, clip_id=clip_id, camera_id=camera_id)

    # One detection, one tracklet frame
    writer.append_detection_row(
        frame_index=0,
        timestamp_ms=0,
        detection_id="det0",
        class_name="person",
        confidence=0.9,
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        tracklet_id="trkA",
    )

    writer.append_tracklet_frame_row(
        frame_index=0,
        timestamp_ms=0,
        tracklet_id="trkA",
        detection_id="det0",
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        u_px=50.0,
        v_px=200.0,
        x_m=0.1,
        y_m=0.2,
        on_mat=True,
        contact_conf=0.75,
        contact_method="bbox_bottom_center",
    )

    writer.write_all()

    cp_path = layout.stage_A_contact_points_parquet()
    assert cp_path.exists()

    df = pd.read_parquet(cp_path)
    required_cols = [
        "clip_id",
        "camera_id",
        "frame_index",
        "timestamp_ms",
        "detection_id",
        "tracklet_id",
        "u_px",
        "v_px",
        "x_m",
        "y_m",
        "on_mat",
        "contact_conf",
        "contact_method",
    ]
    for c in required_cols:
        assert c in df.columns

    # End-to-end stage A validation (uses f0_validate + ordering rules)
    m = init_manifest(
        clip_id=clip_id,
        camera_id=camera_id,
        input_video_path="/dev/null",
        fps=30.0,
        frame_count=1,
        duration_ms=33,
        pipeline_version="test",
        created_at_ms=0,
    )
    _validate_stage_outputs(m, layout, "A", resolved_config={})
