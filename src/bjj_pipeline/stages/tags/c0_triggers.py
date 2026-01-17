from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math

from bjj_pipeline.stages.tags.c0_scheduler import Candidate


@dataclass(frozen=True)
class TriggerEvent:
    tracklet_id: str
    trigger: str  # "overlap" | "vel_jump" | "accel_jump"
    details: Dict[str, Any]


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = float(a_area + b_area - inter)
    return float(inter / denom) if denom > 0 else 0.0


class C0TriggerEngine:
    """Frame-local (rolling-history allowed) triggers for cadence ramp-up.

    Deterministic and lightweight:
      - overlap trigger: sustained IoU above threshold for window_frames
      - motion trigger: velocity / acceleration jumps (prefer metric if available)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))

        overlap_cfg = dict(self.cfg.get("overlap", {}) or {})
        self.overlap_iou_thresh = float(overlap_cfg.get("iou_thresh", 0.35))
        self.overlap_window_frames = int(overlap_cfg.get("window_frames", 6))

        motion_cfg = dict(self.cfg.get("motion", {}) or {})
        self.motion_enabled = bool(motion_cfg.get("enabled", True))
        self.motion_prefer_metric = bool(motion_cfg.get("prefer_metric", True))

        self.dv_thresh_mps = float(motion_cfg.get("dv_thresh_mps", 2.5))
        self.a_thresh_mps2 = float(motion_cfg.get("a_thresh_mps2", 12.0))

        self.dv_thresh_pxps = float(motion_cfg.get("dv_thresh_pxps", 250.0))
        self.a_thresh_pxps2 = float(motion_cfg.get("a_thresh_pxps2", 1200.0))

        # Rolling state
        self._overlap_counts: Dict[Tuple[str, str], int] = {}
        self._pos_hist: Dict[str, List[Tuple[int, int, float, float, bool]]] = {}
        # (frame_index, timestamp_ms, x, y, metric)

    def update(self, *, frame_index: int, timestamp_ms: int, candidates: List[Candidate]) -> List[TriggerEvent]:
        if not self.enabled:
            return []

        events: List[TriggerEvent] = []
        if candidates:
            events.extend(self._update_overlap(frame_index=frame_index, candidates=candidates))
            events.extend(self._update_motion(frame_index=frame_index, timestamp_ms=timestamp_ms, candidates=candidates))
        return events

    def _update_overlap(self, *, frame_index: int, candidates: List[Candidate]) -> List[TriggerEvent]:
        if self.overlap_window_frames <= 0:
            return []

        current_pairs: Dict[Tuple[str, str], int] = {}
        events: List[TriggerEvent] = []

        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                if a.tracklet_id == b.tracklet_id:
                    continue

                iou = _iou_xyxy((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2))
                if iou < self.overlap_iou_thresh:
                    continue

                tid1, tid2 = sorted([a.tracklet_id, b.tracklet_id])
                key = (tid1, tid2)
                prev = self._overlap_counts.get(key, 0)
                cur = prev + 1
                current_pairs[key] = cur

                if cur == self.overlap_window_frames:
                    details = {
                        "pair": [tid1, tid2],
                        "iou": float(iou),
                        "window_frames": int(self.overlap_window_frames),
                    }
                    events.append(TriggerEvent(tracklet_id=tid1, trigger="overlap", details=details))
                    events.append(TriggerEvent(tracklet_id=tid2, trigger="overlap", details=details))

        self._overlap_counts = current_pairs
        return events

    def _candidate_position(self, c: Candidate) -> Tuple[float, float, bool]:
        if self.motion_prefer_metric and c.x_m is not None and c.y_m is not None:
            return float(c.x_m), float(c.y_m), True
        return float(c.x1 + c.x2) / 2.0, float(c.y1 + c.y2) / 2.0, False

    def _update_motion(self, *, frame_index: int, timestamp_ms: int, candidates: List[Candidate]) -> List[TriggerEvent]:
        if not self.motion_enabled:
            return []

        events: List[TriggerEvent] = []

        for c in candidates:
            x, y, metric = self._candidate_position(c)
            hist = self._pos_hist.setdefault(c.tracklet_id, [])
            hist.append((frame_index, timestamp_ms, x, y, metric))
            if len(hist) > 3:
                hist.pop(0)

        for tid, hist in list(self._pos_hist.items()):
            if len(hist) < 3:
                continue

            (f0, t0, x0, y0, m0), (f1, t1, x1, y1, m1), (f2, t2, x2, y2, m2) = hist[-3:]
            if not (m0 == m1 == m2):
                continue

            dt01 = (t1 - t0) / 1000.0
            dt12 = (t2 - t1) / 1000.0
            if dt01 <= 0 or dt12 <= 0:
                continue

            vx01 = (x1 - x0) / dt01
            vy01 = (y1 - y0) / dt01
            vx12 = (x2 - x1) / dt12
            vy12 = (y2 - y1) / dt12

            dv = math.hypot(vx12 - vx01, vy12 - vy01)
            a = dv / dt12

            if m2:
                dv_thresh = self.dv_thresh_mps
                a_thresh = self.a_thresh_mps2
                units = "metric"
            else:
                dv_thresh = self.dv_thresh_pxps
                a_thresh = self.a_thresh_pxps2
                units = "pixel"

            if dv >= dv_thresh:
                events.append(
                    TriggerEvent(
                        tracklet_id=tid,
                        trigger="vel_jump",
                        details={"dv": float(dv), "threshold": float(dv_thresh), "units": units},
                    )
                )

            if a >= a_thresh:
                events.append(
                    TriggerEvent(
                        tracklet_id=tid,
                        trigger="accel_jump",
                        details={"a": float(a), "threshold": float(a_thresh), "units": units},
                    )
                )

        return events
