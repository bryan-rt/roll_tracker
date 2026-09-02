"""Render annotated_gt.mp4 and mat_view_gt.mp4 for GT-VERIFY-1.

annotated_gt.mp4: pipeline boxes + GT boxes + IoU intersection regions
mat_view_gt.mp4: mat canvas with pipeline points + GT hollow circles

Both consume existing matcher output (per_frame_matches / gt_person_trace).
No reimplemented matching or geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_projection import project_to_world, load_calibration_from_payload, CameraProjection
from bjj_pipeline.contracts.f0_sidecar import load_sidecar
from bjj_pipeline.stages.detect_track.quality import contact_point_from_bbox
from bjj_pipeline.stages.orchestration.multiplex_runner import _homography_to_img_to_mat, _load_json
from bjj_pipeline.viz.mat_view import render_mat_canvas
from bjj_pipeline.viz.mux_visualizer import load_mat_blueprint
from bjj_pipeline.viz.video_writer import VideoWriter


# Distinct colours for GT tracks (8 tracks)
GT_COLORS = [
    (0, 200, 200),    # GT 0 - yellow-cyan
    (200, 100, 0),    # GT 1 - dark orange
    (0, 200, 0),      # GT 2 - green
    (200, 0, 200),    # GT 3 - magenta
    (100, 200, 255),  # GT 4 - light blue
    (0, 100, 200),    # GT 5 - dark blue
    (200, 200, 0),    # GT 6 - cyan-yellow
    (150, 150, 0),    # GT 7 - olive
]

PERSON_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 200, 100), (200, 100, 255), (100, 255, 200),
    (255, 150, 200), (200, 255, 100), (100, 200, 150),
    (150, 100, 255), (255, 100, 200), (100, 150, 255),
    (200, 200, 100), (100, 200, 200), (200, 100, 100),
    (150, 255, 150), (255, 200, 200),
]


def _load_projection(camera_id: str) -> CameraProjection:
    cam_dir = Path("configs") / "cameras" / camera_id
    hj_path = cam_dir / "homography.json"
    j = _load_json(hj_path)
    H_raw = np.asarray(j.get("H", j.get("homography")), dtype=np.float64)
    cm, dc = load_calibration_from_payload(j)
    H = _homography_to_img_to_mat(H_raw, j)
    return CameraProjection(H=H, camera_matrix=cm, dist_coefficients=dc)


def _person_color(pid: str) -> tuple:
    if pid is None:
        return (180, 180, 180)
    idx = hash(pid) % len(PERSON_COLORS)
    return PERSON_COLORS[idx]


def render_annotated_gt(
    video_path: Path,
    pfm_path: Path,
    person_tracks_path: Path,
    output_path: Path,
    sidecar_path: Path | None = None,
) -> None:
    """Render annotated_gt.mp4 with pipeline boxes, GT boxes, and IoU intersection."""
    sidecar = load_sidecar(sidecar_path or video_path)
    fps = 1.0 / sidecar.nominal_dt_s

    pfm = pd.read_parquet(pfm_path)
    pt = pd.read_parquet(person_tracks_path)

    # Build per-frame lookup for GT matches
    gt_by_frame: dict[int, pd.DataFrame] = {}
    for fi, g in pfm.groupby("frame_index"):
        gt_by_frame[int(fi)] = g

    # Build per-frame lookup for person tracks
    pt_by_frame: dict[int, pd.DataFrame] = {}
    for fi, g in pt.groupby("frame_index"):
        pt_by_frame[int(fi)] = g

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = VideoWriter(output_path, fps=fps, frame_size=(w, h))

    fi = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw person tracks (pipeline boxes)
            pt_frame = pt_by_frame.get(fi)
            if pt_frame is not None:
                for _, r in pt_frame.iterrows():
                    x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
                    pid = r["person_id"]
                    col = _person_color(pid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    label = str(pid) if pid else "?"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

            # Draw GT boxes and IoU intersection
            gt_frame = gt_by_frame.get(fi)
            if gt_frame is not None:
                for _, r in gt_frame.iterrows():
                    gt_id = r["gt_track_id"]
                    status = r["match_status"]

                    if pd.isna(gt_id):
                        continue  # unmatched pred row

                    gt_id_int = int(gt_id)
                    gt_col = GT_COLORS[gt_id_int % len(GT_COLORS)]

                    gx1 = int(r["gt_x1"]) if pd.notna(r.get("gt_x1")) else None
                    gy1 = int(r["gt_y1"]) if pd.notna(r.get("gt_y1")) else None
                    gx2 = int(r["gt_x2"]) if pd.notna(r.get("gt_x2")) else None
                    gy2 = int(r["gt_y2"]) if pd.notna(r.get("gt_y2")) else None

                    if gx1 is None:
                        continue

                    if status == "unmatched_gt":
                        # Red dashed GT box — no matched detection
                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                        cv2.putText(frame, f"GT{gt_id_int}!", (gx1, gy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                    elif status == "matched":
                        # Yellow GT box
                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), gt_col, 2)
                        cv2.putText(frame, f"GT{gt_id_int}", (gx1, gy2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, gt_col, 1, cv2.LINE_AA)

                        # IoU intersection region — green semi-transparent
                        px1 = r.get("pred_x1")
                        py1 = r.get("pred_y1")
                        px2 = r.get("pred_x2")
                        py2 = r.get("pred_y2")
                        if pd.notna(px1):
                            ix1 = int(max(gx1, px1))
                            iy1 = int(max(gy1, py1))
                            ix2 = int(min(gx2, px2))
                            iy2 = int(min(gy2, py2))
                            if ix2 > ix1 and iy2 > iy1:
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2),
                                              (0, 200, 0), -1)
                                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                                # Hatch lines
                                for dy in range(0, iy2 - iy1, 6):
                                    y_line = iy1 + dy
                                    cv2.line(frame, (ix1, y_line), (ix2, y_line),
                                             (0, 180, 0), 1)

            cv2.putText(frame, f"frame={fi}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            vw.write(frame)
            fi += 1
    finally:
        cap.release()
        vw.close()


def render_mat_view_gt(
    video_path: Path,
    pfm_path: Path,
    person_tracks_path: Path,
    output_path: Path,
    camera_id: str = "FP7oJQ",
    mat_blueprint_path: Path = Path("configs/mat_blueprint.json"),
    sidecar_path: Path | None = None,
    gt_engagements: list | None = None,
    stage_e_sessions: list | None = None,
) -> None:
    """Render mat_view_gt.mp4 with pipeline points + GT hollow circles + engagement overlays."""
    sidecar = load_sidecar(sidecar_path or video_path)
    fps = 1.0 / sidecar.nominal_dt_s
    proj = _load_projection(camera_id)

    pfm = pd.read_parquet(pfm_path)
    pt = pd.read_parquet(person_tracks_path)
    blueprint = load_mat_blueprint(mat_blueprint_path)

    # Build per-frame lookups
    gt_by_frame: dict[int, pd.DataFrame] = {}
    for fi, g in pfm.groupby("frame_index"):
        gt_by_frame[int(fi)] = g

    pt_by_frame: dict[int, pd.DataFrame] = {}
    for fi, g in pt.groupby("frame_index"):
        pt_by_frame[int(fi)] = g

    mat_size = (640, 640)
    vw = VideoWriter(output_path, fps=fps, frame_size=mat_size)

    cap = cv2.VideoCapture(str(video_path))
    fi = 0
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break

            # Pipeline points
            points = []
            pt_frame = pt_by_frame.get(fi)
            if pt_frame is not None:
                for _, r in pt_frame.iterrows():
                    u, v, _, _ = contact_point_from_bbox((r["x1"], r["y1"], r["x2"], r["y2"]))
                    x_m, y_m = project_to_world(
                        (u, v), proj.H, proj.camera_matrix, proj.dist_coefficients
                    )
                    if not (np.isnan(x_m) or np.isnan(y_m)):
                        points.append((x_m, y_m, str(r["person_id"]), True))

            mat_img = render_mat_canvas(
                blueprint=blueprint,
                width=mat_size[0], height=mat_size[1],
                points=points,
                frame_index=fi,
            )

            # Now add GT circles on top
            gt_frame = gt_by_frame.get(fi)
            if gt_frame is not None and blueprint:
                # Reconstruct to_px from render_mat_canvas's logic
                from bjj_pipeline.viz.mat_view import _iter_rects
                rects = list(_iter_rects(blueprint))
                if rects:
                    margin_px = 24
                    xs_r = [x for x, _, w, _, _ in rects] + [x + w for x, _, w, _, _ in rects]
                    ys_r = [y for _, y, _, h, _ in rects] + [y + h for _, y, _, h, _ in rects]
                    min_x, max_x = min(xs_r), max(xs_r)
                    min_y, max_y = min(ys_r), max(ys_r)
                    span_x = max(max_x - min_x, 1e-6)
                    span_y = max(max_y - min_y, 1e-6)
                    usable_w = max(mat_size[0] - 2 * margin_px, 1)
                    usable_h = max(mat_size[1] - 2 * margin_px, 1)
                    scale = min(usable_w / span_x, usable_h / span_y)

                    def to_px(x: float, y: float):
                        px = int(margin_px + (x - min_x) * scale)
                        py = int(margin_px + (y - min_y) * scale)
                        return px, py

                    for _, r in gt_frame.iterrows():
                        gt_id = r["gt_track_id"]
                        if pd.isna(gt_id):
                            continue
                        gx1 = r.get("gt_x1")
                        if pd.isna(gx1):
                            continue

                        gt_id_int = int(gt_id)
                        u, v, _, _ = contact_point_from_bbox(
                            (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])
                        )
                        x_m, y_m = project_to_world(
                            (u, v), proj.H, proj.camera_matrix, proj.dist_coefficients
                        )
                        if np.isnan(x_m) or np.isnan(y_m):
                            continue

                        px, py = to_px(x_m, y_m)
                        gt_col = GT_COLORS[gt_id_int % len(GT_COLORS)]
                        # Hollow circle, larger than pipeline dots
                        cv2.circle(mat_img, (px, py), 7, gt_col, 2)
                        cv2.putText(mat_img, f"GT{gt_id_int}",
                                    (px + 9, py - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, gt_col, 1, cv2.LINE_AA)

            # Engagement overlays
            if gt_frame is not None and blueprint and rects:
                # Collect GT world positions for this frame
                gt_world: dict[str, tuple[float, float]] = {}
                for _, r in gt_frame.iterrows():
                    gt_id = r["gt_track_id"]
                    if pd.isna(gt_id) or pd.isna(r.get("gt_x1")):
                        continue
                    u, v, _, _ = contact_point_from_bbox(
                        (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])
                    )
                    x_m, y_m = project_to_world(
                        (u, v), proj.H, proj.camera_matrix, proj.dist_coefficients
                    )
                    if not (np.isnan(x_m) or np.isnan(y_m)):
                        gt_world[f"gt{int(gt_id)}"] = (x_m, y_m)

                # Draw GT engagement rectangles (green)
                if gt_engagements:
                    for iv in gt_engagements:
                        if iv.start_frame <= fi <= iv.end_frame:
                            pa, pb = iv.person_id_a, iv.person_id_b
                            if pa in gt_world and pb in gt_world:
                                xa, ya = gt_world[pa]
                                xb, yb = gt_world[pb]
                                pad = 0.3
                                rx1 = min(xa, xb) - pad
                                ry1 = min(ya, yb) - pad
                                rx2 = max(xa, xb) + pad
                                ry2 = max(ya, yb) + pad
                                px1, py1 = to_px(rx1, ry1)
                                px2, py2 = to_px(rx2, ry2)
                                overlay = mat_img.copy()
                                cv2.rectangle(overlay, (px1, py1), (px2, py2),
                                              (0, 200, 0), -1)
                                cv2.addWeighted(overlay, 0.15, mat_img, 0.85, 0, mat_img)
                                cv2.rectangle(mat_img, (px1, py1), (px2, py2),
                                              (0, 200, 0), 2)
                                label = f"{pa}<->{pb}"
                                cv2.putText(mat_img, label,
                                            (px1 + 2, py1 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (0, 200, 0), 1, cv2.LINE_AA)

                # Draw Stage E engagement rectangles (orange dashed)
                if stage_e_sessions:
                    # Map pipeline person_id -> GT world positions (approximate via trace)
                    for sess in stage_e_sessions:
                        sf, ef = sess["start_frame"], sess["end_frame"]
                        if sf <= fi <= ef:
                            dom_a = sess.get("dominant_gt_a")
                            dom_b = sess.get("dominant_gt_b")
                            if dom_a is not None and dom_b is not None:
                                ka = f"gt{dom_a}"
                                kb = f"gt{dom_b}"
                                if ka in gt_world and kb in gt_world:
                                    xa, ya = gt_world[ka]
                                    xb, yb = gt_world[kb]
                                    pad = 0.5
                                    rx1 = min(xa, xb) - pad
                                    ry1 = min(ya, yb) - pad
                                    rx2 = max(xa, xb) + pad
                                    ry2 = max(ya, yb) + pad
                                    px1, py1 = to_px(rx1, ry1)
                                    px2, py2 = to_px(rx2, ry2)
                                    # Dashed effect via short segments
                                    color = (0, 140, 255)  # orange
                                    dash_len = 8
                                    for edge in [
                                        ((px1, py1), (px2, py1)),
                                        ((px2, py1), (px2, py2)),
                                        ((px2, py2), (px1, py2)),
                                        ((px1, py2), (px1, py1)),
                                    ]:
                                        p1, p2 = edge
                                        dx = p2[0] - p1[0]
                                        dy = p2[1] - p1[1]
                                        length = max(int(np.sqrt(dx ** 2 + dy ** 2)), 1)
                                        for d in range(0, length, dash_len * 2):
                                            s1 = d / length
                                            s2 = min((d + dash_len) / length, 1.0)
                                            sp1 = (int(p1[0] + dx * s1), int(p1[1] + dy * s1))
                                            sp2 = (int(p1[0] + dx * s2), int(p1[1] + dy * s2))
                                            cv2.line(mat_img, sp1, sp2, color, 1)

            vw.write(mat_img)
            fi += 1
    finally:
        cap.release()
        vw.close()
