from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import cv2
import numpy as np
import pandas as pd

from bjj_pipeline.viz.mux_visualizer import load_mat_blueprint
from bjj_pipeline.viz.mat_view import _iter_rects, render_mat_canvas


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_audit_event(audit_path: Path, event: Dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _compute_to_px(*, blueprint: Any, width: int, height: int, margin_px: int) -> Optional[Any]:
    """Recompute the same blueprint->pixel transform used by render_mat_canvas.

    We intentionally duplicate the transform logic here so we can draw polylines without
    changing the viz helper's API (keeps this checkpoint additive + low risk).
    """
    rects = list(_iter_rects(blueprint))
    if not rects:
        return None

    xs = [x for x, _, w, _, _ in rects] + [x + w for x, _, w, _, _ in rects]
    ys = [y for _, y, _, h, _ in rects] + [y + h for _, y, _, h, _ in rects]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    usable_w = max(width - 2 * margin_px, 1)
    usable_h = max(height - 2 * margin_px, 1)
    scale = min(usable_w / span_x, usable_h / span_y)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int(margin_px + (x - min_x) * scale)
        py = int(margin_px + (y - min_y) * scale)
        return px, py

    return to_px


def _draw_solid_polyline(img: np.ndarray, pts: List[Tuple[int, int]], *, color: Tuple[int, int, int], thickness: int) -> None:
    if len(pts) < 2:
        return
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_dotted_polyline(
    img: np.ndarray,
    pts: List[Tuple[int, int]],
    *,
    color: Tuple[int, int, int],
    thickness: int,
    dash_px: int = 6,
    gap_px: int = 6,
) -> None:
    if len(pts) < 2:
        return

    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        seg_len = float((dx * dx + dy * dy) ** 0.5)
        if seg_len <= 1e-6:
            continue

        # Parameterize along the segment.
        step = dash_px + gap_px
        t = 0.0
        while t < seg_len:
            t0 = t
            t1 = min(t + dash_px, seg_len)

            a0 = t0 / seg_len
            a1 = t1 / seg_len

            p0 = (int(round(x1 + dx * a0)), int(round(y1 + dy * a0)))
            p1 = (int(round(x1 + dx * a1)), int(round(y1 + dy * a1)))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

            t += step


def _choose_source(layout: Any, *, group_by: str) -> Tuple[Literal["person_tracks", "tracklet_bank_frames"], str]:
    """Return (source, group_col)."""
    if group_by == "person":
        return "person_tracks", "person_id"
    if group_by == "tracklet":
        return "tracklet_bank_frames", "tracklet_id"

    # auto
    try:
        pt = Path(layout.person_tracks_parquet())
        if pt.exists():
            return "person_tracks", "person_id"
    except Exception:
        pass
    return "tracklet_bank_frames", "tracklet_id"


def _load_df(layout: Any, source: str) -> pd.DataFrame:
    if source == "person_tracks":
        path = Path(layout.person_tracks_parquet())
        return pd.read_parquet(path)
    if source == "tracklet_bank_frames":
        path = Path(layout.tracklet_bank_frames_parquet())
        return pd.read_parquet(path)
    raise ValueError(f"Unknown source: {source}")


def render_stage_d_paths_png(*, config: Dict[str, Any], inputs: Dict[str, Any]) -> Optional[Path]:
    """Render a debug PNG of footpaths on the mat blueprint.

    - If person_tracks exists, plot grouped by person_id.
    - Else plot grouped by tracklet_id from tracklet_bank_frames.
    - If repaired columns exist, overlay repaired segments as dotted lines.
    """
    layout = inputs["layout"]
    manifest = inputs["manifest"]

    # Config
    qa_cfg = (
        config.get("stages", {})
        .get("stage_D", {})
        .get("qa", None)
        or config.get("stage_D", {}).get("qa", None)
        or {}
    )

    enabled = bool(qa_cfg.get("enabled", False))
    if not enabled:
        return None

    mat_blueprint_path = str(qa_cfg.get("mat_blueprint_path", "configs/mat_blueprint.json"))
    output_name = str(qa_cfg.get("output_name", "stage_D_paths.png"))
    canvas_size_px = int(qa_cfg.get("canvas_size_px", 640))
    margin_px = int(qa_cfg.get("margin_px", 24))
    group_by = str(qa_cfg.get("group_by", "auto"))
    prefer_repaired = bool(qa_cfg.get("prefer_repaired", True))

    audit_path = Path(layout.audit_jsonl("D"))

    bp_path = Path(mat_blueprint_path)
    blueprint = load_mat_blueprint(bp_path)
    if blueprint is None:
        _write_audit_event(
            audit_path,
            {
                "event": "stage_D_visual_qa_skipped",
                "event_type": "stage_D_visual_qa_skipped",
                "timestamp": _now_ms(),
                "clip_id": getattr(manifest, "clip_id", None),
                "camera_id": getattr(manifest, "camera_id", None),
                "reason": "mat_blueprint_unreadable",
                "mat_blueprint_path": mat_blueprint_path,
            },
        )
        return None

    to_px = _compute_to_px(blueprint=blueprint, width=canvas_size_px, height=canvas_size_px, margin_px=margin_px)
    if to_px is None:
        _write_audit_event(
            audit_path,
            {
                "event": "stage_D_visual_qa_skipped",
                "event_type": "stage_D_visual_qa_skipped",
                "timestamp": _now_ms(),
                "clip_id": getattr(manifest, "clip_id", None),
                "camera_id": getattr(manifest, "camera_id", None),
                "reason": "mat_blueprint_empty",
                "mat_blueprint_path": mat_blueprint_path,
            },
        )
        return None

    source, group_col = _choose_source(layout, group_by=group_by)
    try:
        df = _load_df(layout, source)
    except FileNotFoundError:
        _write_audit_event(
            audit_path,
            {
                "event": "stage_D_visual_qa_skipped",
                "event_type": "stage_D_visual_qa_skipped",
                "timestamp": _now_ms(),
                "clip_id": getattr(manifest, "clip_id", None),
                "camera_id": getattr(manifest, "camera_id", None),
                "reason": "source_parquet_missing",
                "source": source,
            },
        )
        return None

    # Required cols
    need = {group_col, "frame_index", "x_m", "y_m"}
    if not need.issubset(set(df.columns)):
        _write_audit_event(
            audit_path,
            {
                "event": "stage_D_visual_qa_skipped",
                "event_type": "stage_D_visual_qa_skipped",
                "timestamp": _now_ms(),
                "clip_id": getattr(manifest, "clip_id", None),
                "camera_id": getattr(manifest, "camera_id", None),
                "reason": "missing_required_columns",
                "source": source,
                "missing": sorted(list(need - set(df.columns))),
            },
        )
        return None

    # Sort deterministically within group
    df = df.sort_values([group_col, "frame_index"], kind="mergesort").reset_index(drop=True)

    # Base canvas
    title = f"Stage D Paths ({group_col})"
    img = render_mat_canvas(
        blueprint=blueprint,
        width=canvas_size_px,
        height=canvas_size_px,
        margin_px=margin_px,
        title=title,
    )

    # Decide whether repaired columns exist
    has_repaired = (
        prefer_repaired
        and "x_m_repaired" in df.columns
        and "y_m_repaired" in df.columns
        and "is_repaired" in df.columns
    )

    # Draw each path
    groups = df.groupby(group_col, sort=False)
    for _, g in groups:
        # base polyline
        base_pts: List[Tuple[int, int]] = []
        for x, y in zip(g["x_m"].tolist(), g["y_m"].tolist()):
            if x is None or y is None:
                continue
            try:
                base_pts.append(to_px(float(x), float(y)))
            except Exception:
                continue
        _draw_solid_polyline(img, base_pts, color=(0, 0, 0), thickness=2)

        if has_repaired:
            # draw repaired segments only (dotted)
            gr = g[g["is_repaired"] == True]  # noqa: E712
            if not gr.empty and "repair_span_id" in gr.columns:
                for _, span in gr.groupby("repair_span_id", sort=False):
                    span = span.sort_values(["frame_index"], kind="mergesort")
                    rep_pts: List[Tuple[int, int]] = []
                    for x, y in zip(span["x_m_repaired"].tolist(), span["y_m_repaired"].tolist()):
                        if x is None or y is None:
                            continue
                        try:
                            rep_pts.append(to_px(float(x), float(y)))
                        except Exception:
                            continue
                    _draw_dotted_polyline(img, rep_pts, color=(128, 128, 128), thickness=2)
            else:
                rep_pts: List[Tuple[int, int]] = []
                for x, y in zip(g["x_m_repaired"].tolist(), g["y_m_repaired"].tolist()):
                    if x is None or y is None:
                        continue
                    try:
                        rep_pts.append(to_px(float(x), float(y)))
                    except Exception:
                        continue
                _draw_dotted_polyline(img, rep_pts, color=(128, 128, 128), thickness=2)

    out_dir = Path(layout.clip_root) / "_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    cv2.imwrite(str(out_path), img)

    _write_audit_event(
        audit_path,
        {
            "event": "stage_D_visual_qa_written",
            "event_type": "stage_D_visual_qa_written",
            "timestamp": _now_ms(),
            "clip_id": getattr(manifest, "clip_id", None),
            "camera_id": getattr(manifest, "camera_id", None),
            "source": source,
            "group_col": group_col,
            "output_png": str(Path("_debug") / output_name),
        },
    )
    return out_path
