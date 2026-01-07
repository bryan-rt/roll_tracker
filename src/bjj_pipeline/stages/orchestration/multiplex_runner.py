from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.core.frame_iterator import FrameIterator, FramePacket
from bjj_pipeline.viz.mux_visualizer import MuxVisualizer, load_mat_blueprint


def _write_placeholder_stage_A(layout: ClipOutputLayout, manifest: ClipManifest, *, camera_id: str, pkt0: FramePacket | None) -> None:
    layout.ensure_dirs_for_stage("A")
    ts_ms = int(pkt0.timestamp_ms) if pkt0 is not None else 0

    det = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "frame_index": 0,
        "timestamp_ms": ts_ms,
        "detection_id": "d1",
        "class_name": "person",
        "confidence": 0.9,
        "x1": 10.0,
        "y1": 10.0,
        "x2": 20.0,
        "y2": 20.0,
        "tracklet_id": "t1",
    }])
    tf = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "frame_index": 0,
        "timestamp_ms": ts_ms,
        "detection_id": "d1",
        "tracklet_id": "t1",
        "local_track_conf": 0.9,
    }])
    ts = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "tracklet_id": "t1",
        "start_frame": 0,
        "end_frame": 0,
        "n_frames": 1,
        "quality_score": 0.5,
        "reason_codes_json": "[]",
    }])

    det.to_parquet(layout.detections_parquet())
    tf.to_parquet(layout.tracklet_frames_parquet())
    ts.to_parquet(layout.tracklet_summaries_parquet())
    Path(layout.audit_jsonl("A")).write_text(json.dumps({
        "event": "placeholder_stage_output",
        "stage": "A",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
    }) + "\n", encoding="utf-8")


def _write_placeholder_stage_B(layout: ClipOutputLayout, manifest: ClipManifest, *, camera_id: str, pkt0: FramePacket | None) -> None:
    layout.ensure_dirs_for_stage("B")
    layout.ensure_mask_dirs()
    ts_ms = int(pkt0.timestamp_ms) if pkt0 is not None else 0

    cp = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "frame_index": 0,
        "timestamp_ms": ts_ms,
        "detection_id": "d1",
        "u_px": 10.0,
        "v_px": 20.0,
        "x_m": 0.1,
        "y_m": 0.2,
        # F0 contract requires these exact fields (and is strict about unexpected columns)
        "method": "placeholder",
        "confidence": 0.5,
    }])
    cp.to_parquet(layout.contact_points_parquet())

    # required by required_outputs_for_stage(B) via glob: stage_B/masks/*.npz
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    np.savez_compressed(layout.mask_npz_path(frame_index=0, detection_id="d1"), mask=dummy_mask)

    Path(layout.audit_jsonl("B")).write_text(json.dumps({
        "event": "placeholder_stage_output",
        "stage": "B",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
    }) + "\n", encoding="utf-8")


def _write_placeholder_stage_C(layout: ClipOutputLayout, manifest: ClipManifest, *, camera_id: str) -> None:
    layout.ensure_dirs_for_stage("C")

    hints = [{
        "schema_version": "0",
        "artifact_type": "identity_hint",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "pipeline_version": manifest.pipeline_version,
        "created_at_ms": 0,
        "tracklet_id": "t1",
        "anchor_key": "tag:123",
        "constraint": "must_link",
        "confidence": 0.9,
        "evidence": "placeholder",
    }]
    Path(layout.identity_hints_jsonl()).write_text("\n".join(json.dumps(r) for r in hints) + "\n", encoding="utf-8")
    Path(layout.tag_observations_jsonl()).write_text(json.dumps({
        "schema_version": "0",
        "artifact_type": "tag_observation",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "pipeline_version": manifest.pipeline_version,
        "created_at_ms": 0,
        "tag_id": "123",
        "frame_index": 0,
        "evidence": "placeholder",
    }) + "\n", encoding="utf-8")
    Path(layout.audit_jsonl("C")).write_text(json.dumps({
        "event": "placeholder_stage_output",
        "stage": "C",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
    }) + "\n", encoding="utf-8")


def run_multiplex_ABC(*,
    ingest_path: Path,
    layout: ClipOutputLayout,
    manifest: ClipManifest,
    camera_id: str,
    runtime_config: Dict[str, Any],
    resolved_config: Dict[str, Any],
    cfg_hash: str,
    run_plan: Dict[str, Dict[str, Any]],
    visualize: bool = False,
) -> None:
    """Run a single-pass loop over frames for the A/B/C window.

    Slice 1 responsibilities:
    - Prove single-pass orchestration by opening the video once and iterating frames once.
    - Optionally write dev-only debug videos (annotated frames + empty mat canvas).
    - If a stage is scheduled to run, emit placeholder *valid* canonical artifacts so resume/validation works.
      (Real stage implementations will replace these in Slice 2+.)
    """

    letters_to_run = [l for l, spec in run_plan.items() if spec.get("should_run") and l in {"A","B","C"}]
    # Only open the video if needed for visualize or for placeholder timebase/frame size
    need_frames = visualize or bool(letters_to_run)

    pkt0: FramePacket | None = None

    if need_frames:
        it = FrameIterator(ingest_path)
        fps = it.fps or 30.0

        viz: MuxVisualizer | None = None
        ann_writer = mat_writer = None
        if visualize:
            debug_dir = layout.clip_root / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = debug_dir / "annotated.mp4"
            mat_view_path = debug_dir / "mat_view.mp4"
            blueprint = load_mat_blueprint(Path("configs") / "mat_blueprint.json")
            viz = None

        for pkt in it:
            if pkt0 is None:
                pkt0 = pkt
                if visualize:
                    assert pkt0 is not None
                    h, w = pkt0.image_bgr.shape[:2]
                    viz = MuxVisualizer(
                        annotated_path=annotated_path,
                        mat_view_path=mat_view_path,
                        fps=fps,
                        frame_size=(w, h),
                        mat_blueprint=blueprint,
                    )
                    ann_writer, mat_writer = viz.open()
            if visualize and viz is not None and ann_writer is not None and mat_writer is not None:
                ann, mat = viz.render_frame(pkt.image_bgr, pkt.frame_index)
                ann_writer.write(ann)
                mat_writer.write(mat)

        if ann_writer is not None:
            ann_writer.close()
        if mat_writer is not None:
            mat_writer.close()

    # Emit placeholder canonical artifacts for any stage that is scheduled to run.
    # NOTE: These are intentionally minimal but must pass F0 validators.
    if "A" in letters_to_run:
        _write_placeholder_stage_A(layout, manifest, camera_id=camera_id, pkt0=pkt0)
    if "B" in letters_to_run:
        _write_placeholder_stage_B(layout, manifest, camera_id=camera_id, pkt0=pkt0)
    if "C" in letters_to_run:
        _write_placeholder_stage_C(layout, manifest, camera_id=camera_id)
