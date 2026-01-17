from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.core.frame_iterator import FrameIterator, FramePacket
from bjj_pipeline.viz.mux_visualizer import MuxVisualizer, load_mat_blueprint


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Get nested config value from dict-like or object-like config."""
    cur: Any = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        return default
    return default if cur is None else cur


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _homography_to_img_to_mat(h: np.ndarray, payload: Dict[str, Any] | None = None) -> np.ndarray:
    """Return an image->mat homography.

    Repo homography.json currently stores H that maps mat/world -> image pixels
    (validated by its correspondences). Stage A needs image pixels -> mat/world.
    We choose direction using correspondences when available, otherwise invert.
    """
    H = np.asarray(h, dtype=np.float64).reshape((3, 3))

    # If correspondences are present, select direction with evidence (lower reprojection error).
    if payload and isinstance(payload, dict):
        corr = payload.get("correspondences") or {}
        ip = corr.get("image_points_px")
        mp = corr.get("mat_points")
        if isinstance(ip, list) and isinstance(mp, list) and ip and mp:
            try:
                u, v = float(ip[0][0]), float(ip[0][1])
                x, y = float(mp[0][0]), float(mp[0][1])

                p_img = np.array([u, v, 1.0], dtype=np.float64)
                p_mat = np.array([x, y, 1.0], dtype=np.float64)

                def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
                    q = M @ p
                    return q[:2] / q[2]

                # If H is mat->img, H@mat should match image point.
                pred_img = _apply(H, p_mat)
                err_mat_to_img = float(np.linalg.norm(pred_img - p_img[:2]))

                # If H is img->mat, H@img should match mat point.
                pred_mat = _apply(H, p_img)
                err_img_to_mat = float(np.linalg.norm(pred_mat - p_mat[:2]))

                # If H behaves like mat->img (small err in that direction), invert for Stage A.
                if err_mat_to_img <= err_img_to_mat:
                    return np.linalg.inv(H)
                # Otherwise H already looks like img->mat.
                return H
            except Exception:
                pass

    # Conservative default: invert (consistent with current cam03 payloads).
    return np.linalg.inv(H)


def _load_homography_matrix(cfg: Any, camera_id: str) -> np.ndarray:
    """Load 3x3 homography matrix for camera.

    D7 preflight guarantees this exists and is valid. For Stage A geometry, we require
    an image(px)->mat/world transform. Repo homography.json currently stores mat->image,
    so we convert to image->mat here (with a correspondence-based direction check).
    """
    p = _cfg_get(cfg, "homography_path", None)
    if p:
        pp = Path(str(p))
        if pp.exists():
            j = _load_json(pp)
            H_raw = np.asarray(j.get("H", j.get("homography", j.get("matrix"))), dtype=np.float64)
            return _homography_to_img_to_mat(H_raw, j)

    cam_dir = Path("configs") / "cameras" / camera_id
    candidates = [
        cam_dir / "homography.json",
        cam_dir / "homography_pipeline.json",
        cam_dir / "homography_from_npy.json",
    ]
    for pp in candidates:
        if pp.exists():
            j = _load_json(pp)
            H_raw = np.asarray(j.get("H", j.get("homography", j.get("matrix"))), dtype=np.float64)
            return _homography_to_img_to_mat(H_raw, j)

    raise FileNotFoundError(f"Homography not found for camera_id={camera_id}. Tried: {candidates}")


def _botsort_params_with_defaults(resolved_config: Dict[str, Any], params: Dict[str, Any], *, with_reid: bool) -> Dict[str, Any]:
    """BoxMOT v16+ requires reid_weights/device/half even when with_reid=False."""
    out = dict(params or {})
    compute_cfg = dict((_cfg_get(resolved_config, "compute", {}) or {}))
    device = compute_cfg.get("device", None) or "cpu"
    half = bool(compute_cfg.get("half", False))
    out.setdefault("device", device)
    out.setdefault("half", half)
    # BoxMOT still requires the key; empty string is acceptable when ReID is off.
    reid_weights = _cfg_get(
        resolved_config,
        "stages.stage_A.tracker.reid_weights",
        _cfg_get(resolved_config, "tracker.reid_weights", ""),
    )
    out.setdefault("reid_weights", str(reid_weights or ("" if not with_reid else "")))
    return out


def _write_placeholder_stage_A(layout: ClipOutputLayout, manifest: ClipManifest, *, camera_id: str, pkt0: FramePacket | None) -> None:
    layout.ensure_dirs_for_stage("A")
    ts_ms = int(pkt0.timestamp_ms) if pkt0 is not None else 0

    # Keep placeholders validator-safe: bbox must have x2>=x1 and y2>=y1.
    x1, y1, x2, y2 = 10.0, 10.0, 20.0, 20.0
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

    det = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "frame_index": 0,
        "timestamp_ms": ts_ms,
        "detection_id": "d1",
        "class_name": "person",
        "confidence": 0.9,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
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

    cp = pd.DataFrame([{
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "frame_index": 0,
        "timestamp_ms": ts_ms,
        "detection_id": "d1",
        "tracklet_id": "t1",
        "u_px": 15.0,
        "v_px": 20.0,
        "x_m": 0.1,
        "y_m": 0.2,
        "on_mat": True,
        "contact_conf": 0.5,
        "contact_method": "placeholder",
    }])

    det.to_parquet(layout.detections_parquet(), index=False)
    tf.to_parquet(layout.tracklet_frames_parquet(), index=False)
    ts.to_parquet(layout.tracklet_summaries_parquet(), index=False)
    cp.to_parquet(layout.stage_A_contact_points_parquet(), index=False)
    p = Path(layout.audit_jsonl("A"))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "placeholder_stage_output",
            "stage": "A",
            "clip_id": manifest.clip_id,
            "camera_id": camera_id,
        }) + "\n")


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

    p = Path(layout.audit_jsonl("B"))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "placeholder_stage_output",
            "stage": "B",
            "clip_id": manifest.clip_id,
            "camera_id": camera_id,
        }) + "\n")


def _write_placeholder_stage_C(
    layout: ClipOutputLayout,
    manifest: ClipManifest,
    *,
    camera_id: str,
    mode: str,
    resolved_config: Dict[str, Any],
) -> None:
    """Write *contract-valid* Stage C outputs without emitting fake observations.

    Milestone 1 behavior:
    - tag_observations.jsonl is empty (0 records)
    - identity_hints.jsonl is empty (0 records)
    - audit.jsonl is non-empty with a run header + summary (for evidence/review)
    """

    layout.ensure_dirs_for_stage("C")

    # Touch empty JSONL files
    Path(layout.identity_hints_jsonl()).write_text("", encoding="utf-8")
    Path(layout.tag_observations_jsonl()).write_text("", encoding="utf-8")

    # Audit: header + summary (always present)
    tag_family_default = _cfg_get(resolved_config, "stages.stage_C.tag_family", "36h11")
    header = {
        "event": "stage_C_run_header",
        "stage": "C",
        "schema_version": "0",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "pipeline_version": manifest.pipeline_version,
        "created_at_ms": int(getattr(manifest, "created_at_ms", 0) or 0),
        "mode": mode,
        "tag_family_default": str(tag_family_default) if tag_family_default is not None else "36h11",
        "note": "stage_C_enabled_no_decode_yet",
    }
    summary = {
        "event": "stage_C_run_summary",
        "stage": "C",
        "clip_id": manifest.clip_id,
        "camera_id": camera_id,
        "counters": {
            "total_candidates_seen": 0,
            "total_decode_attempts": 0,
            "total_tag_observations_emitted": 0,
            "total_identity_hints_emitted": 0,
        },
    }
    Path(layout.audit_jsonl("C")).write_text(
        json.dumps(header) + "\n" + json.dumps(summary) + "\n",
        encoding="utf-8",
    )


def run_multiplex_AC(*,
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
    """Run a single-pass loop over frames for the A/C window.

    Slice 1 responsibilities:
    - Prove single-pass orchestration by opening the video once and iterating frames once.
    - Optionally write dev-only debug videos (annotated frames + empty mat canvas).
    - If a stage is scheduled to run, emit placeholder *valid* canonical artifacts so resume/validation works.
      (Real stage implementations will replace these in Slice 2+.)
    """

    letters_to_run = [l for l, spec in run_plan.items() if spec.get("should_run") and l in {"A","C"}]
    stage_c_enabled = "C" in letters_to_run
    # Only open the video if needed for visualize or for placeholder timebase/frame size
    need_frames = visualize or bool(letters_to_run)

    pkt0: FramePacket | None = None

    # ------------------------------
    # Optional REAL Stage A wiring
    # ------------------------------
    stage_a_enabled = "A" in letters_to_run
    stage_a_processor = None
    stage_a_writer = None
    stage_a_placeholder_written = False
    stage_a_allow_placeholder = bool(
        _cfg_get(resolved_config, "stages.stage_A.allow_placeholder", False)
    )

    # ------------------------------
    # Stage C (C0) scheduler wiring
    # ------------------------------
    stage_c_audit_f = None
    stage_c_counts = {
        "total_candidates_seen": 0,
        "total_scheduled_attempts": 0,
        "total_skipped": 0,
    }
    stage_c_skips_by_reason = defaultdict(int)
    stage_c_tag_family_default = _cfg_get(resolved_config, "stages.stage_C.tag_family", "36h11")
    stage_c_sched_cfg = _cfg_get(resolved_config, "stages.stage_C.c0_scheduler", {}) or {}
    stage_c_sched_enabled = bool(stage_c_sched_cfg.get("enabled", True))
    stage_c_k_seek = int(stage_c_sched_cfg.get("k_seek", 1))
    stage_c_k_verify = int(stage_c_sched_cfg.get("k_verify", 30))
    stage_c_n_ramp = int(stage_c_sched_cfg.get("n_ramp", 60))
    stage_c_scheduler = None

    if need_frames:
        it = FrameIterator(ingest_path)
        fps = it.fps or 30.0

        if stage_a_enabled:
            # Lazy imports so unit tests do not require ultralytics/boxmot.
            try:
                from bjj_pipeline.stages.detect_track.processor import StageAProcessor
                from bjj_pipeline.stages.detect_track.detector import UltralyticsYoloDetector
                from bjj_pipeline.stages.detect_track.tracker import BotSortTracker
                from bjj_pipeline.stages.detect_track.outputs import StageAWriter
            except Exception as e:
                layout.ensure_dirs_for_stage("A")
                p = Path(layout.audit_jsonl("A"))
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "event": "stage_a_real_import_failed",
                        "stage": "A",
                        "clip_id": manifest.clip_id,
                        "camera_id": camera_id,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }) + "\n")
                if stage_a_allow_placeholder:
                    _write_placeholder_stage_A(layout, manifest, camera_id=camera_id, pkt0=pkt0)
                    stage_a_enabled = False
                    stage_a_placeholder_written = True
                else:
                    raise RuntimeError(
                        "Stage A is scheduled to run in multiplex_ABC but real dependencies could not be imported. "
                        "Install boxmot/ultralytics (and any required backends), or set stages.stage_A.allow_placeholder=true."
                    ) from e
            else:
                layout.ensure_dirs_for_stage("A")
                H = _load_homography_matrix(resolved_config, camera_id)
                blueprint = load_mat_blueprint(Path("configs") / "mat_blueprint.json")

                stage_a_writer = StageAWriter(layout=layout, clip_id=manifest.clip_id, camera_id=camera_id)

                stage_a_writer.audit(
                    "stage_a_setup",
                    {
                        "homography_converted": True,
                        "mat_blueprint_path": str(Path("configs") / "mat_blueprint.json"),
                        "mat_blueprint_type": str(type(blueprint)),
                        "mat_blueprint_len": (len(blueprint) if hasattr(blueprint, "__len__") else None),
                    },
                )

                try:
                    model_path = str(
                        _cfg_get(
                            resolved_config,
                            "stages.stage_A.detector.model_path",
                            _cfg_get(resolved_config, "detector.model_path", "models/yolov8n.pt"),
                        )
                    )
                    seg_model_path = _cfg_get(
                        resolved_config,
                        "stages.stage_A.detector.seg_model_path",
                        _cfg_get(resolved_config, "detector.seg_model_path", None),
                    )
                    use_seg_cfg = _cfg_get(
                        resolved_config,
                        "stages.stage_A.detector.use_seg",
                        _cfg_get(resolved_config, "detector.use_seg", None),
                    )
                    # Checkpoint policy: always attempt YOLO segmentation when possible.
                    use_seg = True if use_seg_cfg is None else bool(use_seg_cfg)
                    if use_seg_cfg is False:
                        stage_a_writer.audit("stage_a_use_seg_overridden_true", {"configured_use_seg": False})
                    conf = float(_cfg_get(resolved_config, "stages.stage_A.detector.conf", _cfg_get(resolved_config, "detector.conf", 0.25)))
                    imgsz = _cfg_get(resolved_config, "stages.stage_A.detector.imgsz", _cfg_get(resolved_config, "detector.imgsz", None))
                    device = _cfg_get(resolved_config, "stages.stage_A.detector.device", _cfg_get(resolved_config, "detector.device", None))

                    detector = UltralyticsYoloDetector(
                        model_path=model_path,
                        seg_model_path=str(seg_model_path) if seg_model_path is not None else None,
                        use_seg=use_seg,
                        conf=conf,
                        imgsz=int(imgsz) if imgsz is not None else None,
                        device=str(device) if device is not None else None,
                    )

                    with_reid = bool(
                        _cfg_get(
                            resolved_config,
                            "stages.stage_A.tracker.with_reid",
                            _cfg_get(resolved_config, "tracker.with_reid", False),
                        )
                    )
                    params = _cfg_get(
                        resolved_config,
                        "stages.stage_A.tracker.params",
                        _cfg_get(resolved_config, "tracker.params", {}),
                    )
                    if not isinstance(params, dict):
                        params = dict(params)
                    params = _botsort_params_with_defaults(resolved_config, params, with_reid=with_reid)
                    tracker = BotSortTracker(with_reid=with_reid, params=params)

                    stage_a_processor = StageAProcessor(
                        config=resolved_config,
                        homography=H,
                        mat_blueprint=blueprint,
                        writer=stage_a_writer,
                        detector=detector,
                        tracker=tracker,
                    )
                except Exception as e:
                    layout.ensure_dirs_for_stage("A")
                    p = Path(layout.audit_jsonl("A"))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "event": "stage_a_real_setup_failed",
                            "stage": "A",
                            "clip_id": manifest.clip_id,
                            "camera_id": camera_id,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "tracker_with_reid": bool(with_reid),
                            "tracker_params_keys": sorted(list(params.keys())) if isinstance(params, dict) else None,
                            "compute_device": _cfg_get(resolved_config, "compute.device", None),
                            "compute_half": _cfg_get(resolved_config, "compute.half", None),
                        }) + "\n")
                    if stage_a_allow_placeholder:
                        # If any real Stage A setup fails, optionally fall back to placeholder outputs.
                        _write_placeholder_stage_A(layout, manifest, camera_id=camera_id, pkt0=pkt0)
                        stage_a_enabled = False
                        stage_a_placeholder_written = True
                    else:
                        raise RuntimeError(
                            "Stage A is scheduled to run in multiplex_ABC but real Stage A setup failed. "
                            "Fix the error (see stage_A/audit.jsonl) or set stages.stage_A.allow_placeholder=true."
                        ) from e

        viz: MuxVisualizer | None = None
        ann_writer = mat_writer = None
        if visualize:
            debug_dir = layout.clip_root / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = debug_dir / "annotated.mp4"
            mat_view_path = debug_dir / "mat_view.mp4"
            blueprint = load_mat_blueprint(Path("configs") / "mat_blueprint.json")
            viz = None

        # Stage C output files (always empty JSONLs in M2; audit contains decisions)
        if stage_c_enabled:
            from bjj_pipeline.stages.tags.c0_gating import evaluate_scannability
            from bjj_pipeline.stages.tags.c0_scannability_map import load_scannability_map
            from bjj_pipeline.stages.tags.c0_scheduler import C0Scheduler, Candidate

            gating_cfg = _cfg_get(resolved_config, "stages.stage_C.c0_scheduler.gating", {}) or {}
            prior_map = None
            if bool(gating_cfg.get("use_scannability_prior", False)):
                prior_path = gating_cfg.get("prior_path")
                if prior_path is None:
                    prior_path = Path("configs") / "cameras" / camera_id / "scannability_map.json"
                prior_map = load_scannability_map(Path(prior_path))

            layout.ensure_dirs_for_stage("C")
            Path(layout.identity_hints_jsonl()).write_text("", encoding="utf-8")
            Path(layout.tag_observations_jsonl()).write_text("", encoding="utf-8")

            # Open audit file for streaming append (header now, decisions per frame, summary at end).
            audit_path = Path(layout.audit_jsonl("C"))
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            stage_c_audit_f = audit_path.open("w", encoding="utf-8")

            stage_c_scheduler = C0Scheduler(
                enabled=stage_c_sched_enabled,
                k_seek=stage_c_k_seek,
                k_verify=stage_c_k_verify,
                n_ramp=stage_c_n_ramp,
            )

            header = {
                "event": "stage_C_run_header",
                "stage": "C",
                "schema_version": "0",
                "clip_id": manifest.clip_id,
                "camera_id": camera_id,
                "pipeline_version": manifest.pipeline_version,
                "created_at_ms": int(getattr(manifest, "created_at_ms", 0) or 0),
                "mode": "multiplex_AC",
                "tag_family_default": str(stage_c_tag_family_default) if stage_c_tag_family_default is not None else "36h11",
                "scheduler": {
                    "enabled": bool(stage_c_sched_enabled),
                    "k_seek": int(stage_c_k_seek),
                    "k_verify": int(stage_c_k_verify),
                    "n_ramp": int(stage_c_n_ramp),
                },
            }
            stage_c_audit_f.write(json.dumps(header) + "\n")

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

            if stage_a_processor is not None and stage_a_writer is not None:
                try:
                    overlays = stage_a_processor.process_frame(
                        frame_bgr=pkt.image_bgr,
                        frame_index=int(pkt.frame_index),
                        timestamp_ms=int(pkt.timestamp_ms),
                    )
                    if visualize and viz is not None and ann_writer is not None and mat_writer is not None:
                        ann, mat = viz.render_frame(pkt.image_bgr, pkt.frame_index, overlays=overlays)
                        ann_writer.write(ann)
                        mat_writer.write(mat)

                    # Stage C (C0) cadence decisions (gating + cadence; no decode in M2)
                    if stage_c_enabled and stage_c_audit_f is not None and stage_c_scheduler is not None:
                        # Candidate selection is join-first: (frame_index, timestamp_ms, detection_id) + tracklet_id.
                        candidates = []
                        for o in overlays or []:
                            # OverlayItem guarantees these are strings in Stage A code path.
                            if getattr(o, "detection_id", None) and getattr(o, "tracklet_id", None):
                                x1 = int(round(getattr(o, "x1", 0)))
                                y1 = int(round(getattr(o, "y1", 0)))
                                x2 = int(round(getattr(o, "x2", 0)))
                                y2 = int(round(getattr(o, "y2", 0)))
                                det_conf = float(getattr(o, "confidence", 1.0))
                                on_mat = getattr(o, "on_mat", None)

                                gate = evaluate_scannability(
                                    frame_bgr=pkt.image_bgr,
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                    det_conf=det_conf,
                                    on_mat=on_mat,
                                    gating_cfg=gating_cfg,
                                    prior_map=prior_map,
                                )

                                candidates.append(
                                    Candidate(
                                        frame_index=int(pkt.frame_index),
                                        timestamp_ms=int(pkt.timestamp_ms),
                                        detection_id=str(o.detection_id),
                                        tracklet_id=str(o.tracklet_id),
                                        x1=x1,
                                        y1=y1,
                                        x2=x2,
                                        y2=y2,
                                        det_conf=det_conf,
                                        on_mat=on_mat,
                                        scannable=bool(gate.scannable),
                                        gate_reason=str(gate.reason),
                                        gate_meta={
                                            "roi_xyxy": list(gate.roi_xyxy),
                                            "roi_area": int(gate.roi_area),
                                            "roi_w": int(gate.roi_w),
                                            "roi_h": int(gate.roi_h),
                                            "blur_var": gate.blur_var,
                                            "contrast_std": gate.contrast_std,
                                            "prior": gate.prior,
                                            "on_mat": gate.on_mat,
                                        },
                                    )
                                )

                        stage_c_counts["total_candidates_seen"] += int(len(candidates))

                        decisions = stage_c_scheduler.step(
                            frame_index=int(pkt.frame_index),
                            timestamp_ms=int(pkt.timestamp_ms),
                            candidates=candidates,
                        )

                        for d in decisions:
                            if d.decision == "attempt":
                                stage_c_counts["total_scheduled_attempts"] += 1
                            else:
                                stage_c_counts["total_skipped"] += 1
                                stage_c_skips_by_reason[str(d.reason)] += 1

                            ev = {
                                "event": "c0_decision",
                                "stage": "C",
                                "clip_id": manifest.clip_id,
                                "camera_id": camera_id,
                                "frame_index": int(d.candidate.frame_index),
                                "timestamp_ms": int(d.candidate.timestamp_ms),
                                "detection_id": str(d.candidate.detection_id),
                                "tracklet_id": str(d.candidate.tracklet_id),
                                "x1": int(getattr(d.candidate, "x1", 0)),
                                "y1": int(getattr(d.candidate, "y1", 0)),
                                "x2": int(getattr(d.candidate, "x2", 0)),
                                "y2": int(getattr(d.candidate, "y2", 0)),
                                "det_conf": float(getattr(d.candidate, "det_conf", 0.0)),
                                "on_mat": getattr(d.candidate, "on_mat", None),
                                "scannable": bool(getattr(d.candidate, "scannable", True)),
                                "decision": d.decision,
                                "reason": d.reason,
                                "gate_meta": (d.gate_meta if d.gate_meta is not None else dict(getattr(d.candidate, "gate_meta", {}) or {})),
                                "state_before": d.state_before,
                                "state_after": d.state_after,
                            }
                            stage_c_audit_f.write(json.dumps(ev) + "\n")
                except Exception as e:
                    # Capture full traceback so we can pinpoint the exact file:line of failure
                    tb = traceback.format_exc()

                    # Best-effort: prove which detector implementation module is actually loaded at runtime
                    detector_mod = None
                    detector_file = None
                    try:
                        detector_mod = getattr(getattr(stage_a_processor, "detector", None), "__class__", None).__module__
                        import importlib
                        m = importlib.import_module(detector_mod) if detector_mod else None
                        detector_file = getattr(m, "__file__", None) if m is not None else None
                    except Exception:
                        detector_mod = detector_mod or None
                        detector_file = detector_file or None

                    # Make failures actionable: include frame index and input types/shapes.
                    event = {
                        "frame_index": int(pkt.frame_index),
                        "timestamp_ms": int(pkt.timestamp_ms),
                        "frame_bgr_type": str(type(pkt.image_bgr)),
                        "frame_bgr_shape": getattr(pkt.image_bgr, "shape", None),
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "error_repr": repr(e),
                        "traceback": tb,
                        "detector_module": detector_mod,
                        "detector_module_file": detector_file,
                    }
                    stage_a_writer.audit("stage_a_process_frame_failed", event)
                    # Persist immediately so it's visible even if we crash before write_all()
                    try:
                        stage_a_writer.flush_audit_now()
                    except Exception:
                        pass
                    raise

        if ann_writer is not None:
            ann_writer.close()
        if mat_writer is not None:
            mat_writer.close()

        if stage_c_enabled and stage_c_audit_f is not None:
            # Emit summary at end of run (M2: cadence-only).
            summary = {
                "event": "stage_C_run_summary",
                "stage": "C",
                "clip_id": manifest.clip_id,
                "camera_id": camera_id,
                "counters": {
                    "total_candidates_seen": int(stage_c_counts["total_candidates_seen"]),
                    "total_scheduled_attempts": int(stage_c_counts["total_scheduled_attempts"]),
                    "total_skipped": int(stage_c_counts["total_skipped"]),
                    "skips_by_reason": dict(stage_c_skips_by_reason),
                    "total_decode_attempts": 0,
                    "total_tag_observations_emitted": 0,
                    "total_identity_hints_emitted": 0,
                },
            }
            stage_c_audit_f.write(json.dumps(summary) + "\n")
            try:
                stage_c_audit_f.close()
            except Exception:
                pass

        if stage_a_processor is not None and stage_a_writer is not None:
            stage_a_writer.audit("stage_a_completed", {"n_frames": int(it.n_frames) if hasattr(it, "n_frames") else None})
            stage_a_writer.write_all()

            # Validate canonical outputs immediately (keeps resume/skip logic honest).
            try:
                from bjj_pipeline.contracts.f0_validate import validate_detections_df, validate_tracklet_tables, validate_stage_A_contact_points_df
                validate_detections_df(pd.read_parquet(layout.detections_parquet()))
                validate_tracklet_tables(
                    pd.read_parquet(layout.tracklet_frames_parquet()),
                    pd.read_parquet(layout.tracklet_summaries_parquet()),
                )
                validate_stage_A_contact_points_df(pd.read_parquet(layout.stage_A_contact_points_parquet()))
            except Exception:
                # Orchestration will surface validation errors later; avoid hard fail here.
                pass

    # Emit placeholder canonical artifacts for any stage that is scheduled to run.
    # NOTE: These are intentionally minimal but must pass F0 validators.
    if "A" in letters_to_run:
        if stage_a_processor is None and not stage_a_placeholder_written:
            _write_placeholder_stage_A(layout, manifest, camera_id=camera_id, pkt0=pkt0)
    if "B" in letters_to_run:
        _write_placeholder_stage_B(layout, manifest, camera_id=camera_id, pkt0=pkt0)


def run_multiplex_ABC(**kwargs):
    """Backwards-compatible alias (old mode name was multiplex_ABC)."""
    return run_multiplex_AC(**kwargs)
