from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from bjj_pipeline.stages.detect_track.detector import DetectorBackend
from bjj_pipeline.stages.detect_track.tracker import TrackerBackend
from bjj_pipeline.stages.detect_track.types import Detection, TrackState, OverlayItem
from bjj_pipeline.stages.detect_track.quality import (
    bbox_from_mask,
    bbox_fallback_mask,
    compute_mask_quality,
    mask_passes_gate,
    contact_point_from_mask,
    contact_point_from_bbox,
    project_uv_to_xy,
    point_in_mat,
    compute_velocity,
    is_physics_warning,
)
from bjj_pipeline.stages.detect_track.outputs import StageAWriter


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Get nested config value from dict-like or object-like config.

    path uses dot-notation, e.g. "stages.stage_A.tracker.with_reid".
    """
    cur: Any = cfg
    for key in path.split("."):
        if cur is None:
            return default
        # dict-like
        if isinstance(cur, dict):
            cur = cur.get(key, None)
            continue
        # pydantic-like or plain object
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        return default
    return default if cur is None else cur


class StageAProcessor:
    """
    Deterministic per-frame engine for Stage A.

    This class contains *all* A1 logic:
      YOLO → mask gate → tight bbox → BoT-SORT → contact → homography → physics → writer

    It is safe for:
      - multipass
      - multiplex (Z3)
      - deterministic replay
    """

    def __init__(
        self,
        *,
        config: Any,
        homography: np.ndarray,
        mat_blueprint: Any,
        writer: StageAWriter,
        detector: DetectorBackend,
        tracker: TrackerBackend,
    ):
        self.cfg = config
        self.H = homography
        self.mat_blueprint = mat_blueprint
        self.writer = writer
        self.detector = detector
        self.tracker = tracker

        # Tracklet state for velocity / physics
        self._track_state: Dict[str, TrackState] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int, timestamp_ms: int) -> list[OverlayItem]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise TypeError(
                f"StageAProcessor.process_frame expected frame_bgr np.ndarray, got {type(frame_bgr)}"
            )
        frame_for_detector = frame_bgr  # keep detector input immutable for evidence/debugging
        # 1) Detect
        try:
            # Evidence-grade breadcrumb: proves the exact input at the detector boundary (only persisted on crash via flush)
            self.writer.audit(
                "stage_a_detector_call",
                {
                    "frame_index": int(frame_index),
                    "timestamp_ms": int(timestamp_ms),
                    "detector_input_type": str(type(frame_for_detector)),
                    "detector_input_shape": getattr(frame_for_detector, "shape", None),
                    "detector_input_id": int(id(frame_for_detector)),
                },
            )
            dets = self.detector.infer(
                clip_id=self.writer.clip_id,
                camera_id=self.writer.camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame_bgr=frame_for_detector,
            )

            # Persist “seg reality” in audit.jsonl (no reliance on Python logging configuration).
            # This makes it explicit whether we're actually getting masks back from the detector.
            try:
                det_class = type(self.detector).__name__
                det_mod = getattr(type(self.detector), "__module__", None)
                # Introspect common UltralyticsYoloDetector attributes (safe for other backends).
                use_seg = getattr(self.detector, "use_seg", None)
                model_path = getattr(self.detector, "model_path", None)
                seg_model_path = getattr(self.detector, "seg_model_path", None)
                seg_model_loaded = None
                if hasattr(self.detector, "_seg_model"):
                    seg_model_loaded = bool(getattr(self.detector, "_seg_model") is not None)

                n_dets = int(len(dets)) if dets is not None else 0
                n_masks_present = 0
                mask_shapes_sample: list[list[int]] = []
                mask_source_counts: Dict[str, int] = {}
                for d in (dets or []):
                    ms = getattr(d, "mask_source", None)
                    if ms is not None:
                        mask_source_counts[str(ms)] = int(mask_source_counts.get(str(ms), 0) + 1)
                    m = getattr(d, "mask", None)
                    if m is not None:
                        n_masks_present += 1
                        if len(mask_shapes_sample) < 3:
                            try:
                                mask_shapes_sample.append(list(getattr(m, "shape", ())))
                            except Exception:
                                pass

                self.writer.audit(
                    "stage_a_detector_result",
                    {
                        "frame_index": int(frame_index),
                        "timestamp_ms": int(timestamp_ms),
                        "detector_class": det_class,
                        "detector_module": det_mod,
                        "detector_use_seg": use_seg,
                        "detector_model_path": str(model_path) if model_path is not None else None,
                        "detector_seg_model_path": str(seg_model_path) if seg_model_path is not None else None,
                        "detector_seg_model_loaded": seg_model_loaded,
                        "n_dets": n_dets,
                        "n_masks_present": int(n_masks_present),
                        "mask_shapes_sample": mask_shapes_sample,
                        "mask_source_counts": mask_source_counts,
                    },
                )
            except Exception:
                # Never fail the pipeline due to debug/audit bookkeeping
                pass
        except Exception as e:
            # Evidence-grade failure context: proves what we passed into detector.infer()
            self.writer.audit(
                "stage_a_detector_failed",
                {
                    "frame_index": int(frame_index),
                    "timestamp_ms": int(timestamp_ms),
                    "detector_input_type": str(type(frame_for_detector)),
                    "detector_input_shape": getattr(frame_for_detector, "shape", None),
                    "detector_input_id": int(id(frame_for_detector)),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "error_repr": repr(e),
                },
            )
            # Persist immediately so it's visible even if we crash before write_all()
            try:
                self.writer.flush_audit_now()
            except Exception:
                pass
            raise

        # 2) Normalize / gate masks (deterministic)
        gate_obj = _cfg_get(self.cfg, "stages.stage_A.masks.gate", _cfg_get(self.cfg, "masks.gate", {}))
        gate_cfg = gate_obj.model_dump() if hasattr(gate_obj, "model_dump") else dict(gate_obj)
        frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])

        gated: list[Detection] = []
        for d in dets:
            bbox = (d.x1, d.y1, d.x2, d.y2)
            if d.mask is not None:
                q = compute_mask_quality(
                    d.mask,
                    bbox,
                    min_area_frac=float(gate_cfg.get("min_area_frac", 0.10)),
                    max_area_frac=float(gate_cfg.get("max_area_frac", 1.10)),
                )
                if mask_passes_gate(det_conf=float(d.confidence), mask_quality=float(q), gate_cfg=gate_cfg):
                    d.mask_quality = float(q)
                    d.mask_source = "yolo_seg"
                else:
                    d.mask = None
                    d.mask_quality = None
                    d.mask_source = "bbox_fallback"
            else:
                d.mask = None
                d.mask_quality = None
                d.mask_source = "bbox_fallback"
            # If no usable mask, we still want deterministic mask available for downstream (bbox fallback)
            if d.mask is None:
                d.mask = bbox_fallback_mask((frame_h, frame_w), bbox)
            gated.append(d)

        # Post-gate visibility: are we *actually* using seg masks vs fallback masks?
        try:
            src_counts: Dict[str, int] = {}
            n_has_mask = 0
            n_fallback = 0
            n_yolo = 0
            for d in gated:
                ms = getattr(d, "mask_source", None)
                if ms is not None:
                    src_counts[str(ms)] = int(src_counts.get(str(ms), 0) + 1)
                if getattr(d, "mask", None) is not None:
                    n_has_mask += 1
                if getattr(d, "mask_source", None) == "bbox_fallback":
                    n_fallback += 1
                if getattr(d, "mask_source", None) == "yolo_seg":
                    n_yolo += 1

            self.writer.audit(
                "stage_a_mask_gating_summary",
                {
                    "frame_index": int(frame_index),
                    "timestamp_ms": int(timestamp_ms),
                    "n_dets_in": int(len(dets)),
                    "n_dets_out": int(len(gated)),
                    "n_masks_present_out": int(n_has_mask),
                    "n_mask_source_yolo_seg": int(n_yolo),
                    "n_mask_source_bbox_fallback": int(n_fallback),
                    "mask_source_counts": src_counts,
                    "gate_min_area_frac": float(gate_cfg.get("min_area_frac", 0.10)),
                    "gate_max_area_frac": float(gate_cfg.get("max_area_frac", 1.10)),
                },
            )
        except Exception:
            pass

        # 3) Tight bbox from mask (if enabled)
        use_mask_bbox = bool(
            _cfg_get(
                self.cfg,
                "stages.stage_A.tracker.use_mask_bbox",
                _cfg_get(self.cfg, "tracker.use_mask_bbox", False),
            )
        )
        if use_mask_bbox:
            for d in gated:
                if d.mask is not None:
                    mb = bbox_from_mask(d.mask)
                    if mb is not None:
                        x1, y1, x2, y2 = mb
                        if x2 > x1 and y2 > y1:
                            d.x1, d.y1, d.x2, d.y2 = float(x1), float(y1), float(x2), float(y2)

        # 4) Tracker update
        with_reid = bool(
            _cfg_get(self.cfg, "stages.stage_A.tracker.with_reid", _cfg_get(self.cfg, "tracker.with_reid", False))
        )
        # BoxMOT/BoT-SORT requires img (np.ndarray) even when ReID is disabled
        frame_for_tracker = frame_for_detector
        tracked = self.tracker.update(
            frame_index=frame_index,
            detections=gated,
            frame_bgr=frame_for_tracker,
        )

        # Index detections by id for fast lookup
        det_by_id: Dict[str, Detection] = {d.detection_id: d for d in gated}

        overlays: list[OverlayItem] = []

        # 5) For each tracked detection → geometry + write
        for td in tracked:
            det = det_by_id.get(td.detection_id)
            if det is None:
                # Should not happen; tracker should only return known det ids
                self.writer.audit("tracker_unknown_detection_id", {"detection_id": td.detection_id, "frame_index": frame_index})
                continue

            bbox = (det.x1, det.y1, det.x2, det.y2)
            if det.mask is not None:
                u, v, method, contact_conf = contact_point_from_mask(
                    det.mask, bbox, det_conf=float(det.confidence), mask_quality=float(det.mask_quality or 0.0)
                )
            else:
                u, v, method, contact_conf = contact_point_from_bbox(bbox, det_conf=float(det.confidence))

            x_m = y_m = None
            on_mat = None
            if u is not None and v is not None:
                x_m, y_m = project_uv_to_xy(self.H, float(u), float(v))
                on_mat = point_in_mat(float(x_m), float(y_m), self.mat_blueprint)

            vx = vy = None
            # physics audit only
            prev = self._track_state.get(td.tracklet_id)
            if prev is not None and x_m is not None and y_m is not None:
                dt_s = max(0.0, (float(timestamp_ms) - float(prev.last_timestamp_ms)) / 1000.0)
                # compute_velocity supports timestamp-based; here we compute dt_s explicitly.
                vx0, vy0, speed = compute_velocity((prev.last_x_m, prev.last_y_m), prev.last_timestamp_ms, (float(x_m), float(y_m)), int(timestamp_ms))
                vx, vy = vx0, vy0
                max_speed_mps = float(
                    _cfg_get(
                        self.cfg,
                        "stages.stage_A.tracker.physics.max_speed_mps",
                        _cfg_get(self.cfg, "tracker.physics.max_speed_mps", 8.0),
                    )
                )
                if is_physics_warning(float(speed), float(max_speed_mps)):
                    self.writer.audit(
                        "physics_warning",
                        {
                            "tracklet_id": td.tracklet_id,
                            "frame_index": frame_index,
                            "speed_mps": float(speed),
                            "max_speed_mps": float(max_speed_mps),
                        },
                    )

            if x_m is not None and y_m is not None:
                self._track_state[td.tracklet_id] = TrackState(
                    tracklet_id=td.tracklet_id,
                    last_frame=int(frame_index),
                    last_x_m=float(x_m),
                    last_y_m=float(y_m),
                    last_timestamp_ms=int(timestamp_ms),
                )

            # Always write a canonical mask (yolo_seg if gated; otherwise bbox_fallback)
            mask_ref = self.writer.write_mask_npz(frame_index=frame_index, detection_id=det.detection_id, mask=det.mask)

            self.writer.append_detection_row(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detection_id=det.detection_id,
                tracklet_id=td.tracklet_id,
                class_name=det.class_name,
                confidence=float(det.confidence),
                x1=float(det.x1),
                y1=float(det.y1),
                x2=float(det.x2),
                y2=float(det.y2),
                mask_ref=mask_ref,
                mask_source=det.mask_source,
                mask_quality=det.mask_quality,
            )

            self.writer.append_tracklet_frame_row(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                tracklet_id=td.tracklet_id,
                detection_id=det.detection_id,
                local_track_conf=td.local_track_conf,
                x1=float(td.x1),
                y1=float(td.y1),
                x2=float(td.x2),
                y2=float(td.y2),
                u_px=float(u) if u is not None else None,
                v_px=float(v) if v is not None else None,
                x_m=float(x_m) if x_m is not None else None,
                y_m=float(y_m) if y_m is not None else None,
                vx_m=float(vx) if vx is not None else None,
                vy_m=float(vy) if vy is not None else None,
                on_mat=bool(on_mat) if on_mat is not None else None,
                contact_conf=float(contact_conf) if contact_conf is not None else None,
                contact_method=str(method) if method is not None else None,
            )

            overlays.append(
                OverlayItem(
                    tracklet_id=td.tracklet_id,
                    detection_id=det.detection_id,
                    confidence=float(det.confidence),
                    x1=float(det.x1),
                    y1=float(det.y1),
                    x2=float(det.x2),
                    y2=float(det.y2),
                    mask=det.mask,
                    mask_source=det.mask_source,
                    u_px=float(u) if u is not None else None,
                    v_px=float(v) if v is not None else None,
                    x_m=float(x_m) if x_m is not None else None,
                    y_m=float(y_m) if y_m is not None else None,
                    on_mat=bool(on_mat) if on_mat is not None else None,
                )
            )

        return overlays

    def finalize(self):
        return self.writer.write_all()
