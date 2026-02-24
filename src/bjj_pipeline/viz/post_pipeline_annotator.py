from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json

import cv2  # type: ignore
import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.viz.video_writer import VideoWriter


@dataclass
class _MatchInterval:
    match_id: str
    person_a: str
    person_b: str
    start_frame: int
    end_frame: int


def _iter_frames(video_path: Path) -> Tuple[Iterable[Tuple[int, np.ndarray]], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for annotation: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def _gen():
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        cap.release()

    return _gen(), fps, (w, h)


def _load_person_tracks(layout: ClipOutputLayout) -> Optional[pd.DataFrame]:
    path = Path(layout.person_tracks_parquet())
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Require minimal columns
    needed = {"frame_index", "person_id", "x1", "y1", "x2", "y2"}
    if not needed.issubset(df.columns):
        return None
    # Normalize person_id to string for consistent dict keys
    df = df.copy()
    df["person_id"] = df["person_id"].astype(str)
    return df


def _load_match_intervals(layout: ClipOutputLayout, *, clip_id: str) -> Dict[int, List[_MatchInterval]]:
    """Load match_sessions.jsonl and build a per-frame map.

    Returns: dict frame_index -> list of intervals active on that frame.
    """
    out: Dict[int, List[_MatchInterval]] = {}
    path = Path(layout.match_sessions_jsonl())
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("artifact_type") != "match_session":
            continue
        core = rec.get("core") or {}
        if core.get("clip_id") != clip_id:
            continue
        pid_a = str(core.get("person_id_a"))
        pid_b = str(core.get("person_id_b"))
        try:
            s = int(core.get("start_frame"))
            e = int(core.get("end_frame"))
        except Exception:
            continue
        mid = str(core.get("match_id"))
        interval = _MatchInterval(match_id=mid, person_a=pid_a, person_b=pid_b, start_frame=s, end_frame=e)
        for fi in range(s, e + 1):
            out.setdefault(fi, []).append(interval)
    return out


def _pick_color_for_person(person_id: str) -> Tuple[int, int, int]:
    # deterministic pseudo-random but stable across runs
    h = hash(person_id) & 0xFFFFFF
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    # avoid too-dark colors
    if r + g + b < 96:
        r = min(r + 64, 255)
        g = min(g + 64, 255)
        b = min(b + 64, 255)
    return int(b), int(g), int(r)  # BGR


def _draw_box(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: Tuple[int, int, int],
    thick: int = 2,
    dashed: bool = False,
) -> None:
    if dashed:
        # simple dashed rectangle by skipping segments
        pts = [
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (int(round(x1)), int(round(y2))),
            (int(round(x1)), int(round(y1))),
        ]
        dash_px = 6
        gap_px = 6
        for (x_start, y_start), (x_end, y_end) in zip(pts[:-1], pts[1:]):
            dx = float(x_end - x_start)
            dy = float(y_end - y_start)
            seg_len = float((dx * dx + dy * dy) ** 0.5)
            if seg_len <= 1e-6:
                continue
            t = 0.0
            while t < seg_len:
                t0 = t
                t1 = min(t + dash_px, seg_len)
                a0 = t0 / seg_len
                a1 = t1 / seg_len
                p0 = (int(round(x_start + dx * a0)), int(round(y_start + dy * a0)))
                p1 = (int(round(x_start + dx * a1)), int(round(y_start + dy * a1)))
                cv2.line(frame, p0, p1, color, thick, cv2.LINE_AA)
                t += dash_px + gap_px
    else:
        cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thick)


def _annotate_frame(
    frame_bgr: np.ndarray,
    tracks_for_frame: pd.DataFrame,
    *,
    match_intervals: Dict[int, List[_MatchInterval]],
    frame_index: int,
) -> None:
    height, width = frame_bgr.shape[:2]
    for _, row in tracks_for_frame.iterrows():
        try:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
        except Exception:
            continue
        if not (0 <= x1 < width and 0 <= x2 <= width and 0 <= y1 < height and 0 <= y2 <= height):
            # keep it simple; skip clearly invalid boxes
            pass
        person_id = str(row["person_id"])
        color = _pick_color_for_person(person_id)
        intervals = match_intervals.get(frame_index, [])
        in_match = any((iv.person_a == person_id or iv.person_b == person_id) for iv in intervals)
        _draw_box(frame_bgr, x1, y1, x2, y2, color=color, thick=3 if in_match else 2, dashed=in_match)

        label = f"pid={person_id}"
        cv2.putText(frame_bgr, label, (int(round(x1)), max(0, int(round(y1)) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_post_pipeline_annotation(
    manifest: ClipManifest,
    layout: ClipOutputLayout,
    *,
    output_name: str = "annotated_post_E.mp4",
) -> Optional[Path]:
    """Render a post-pipeline annotated video using Stage D/E outputs.

    This is a dev-only helper; failures should be non-fatal to the pipeline.
    """
    video_path = Path(manifest.input_video_path)
    if not video_path.exists():
        return None

    tracks = _load_person_tracks(layout)
    if tracks is None or tracks.empty:
        return None

    # index tracks by frame for fast lookup
    tracks_by_frame: Dict[int, pd.DataFrame] = {}
    for fi, g in tracks.groupby("frame_index"):
        try:
            idx = int(fi)
        except Exception:
            continue
        tracks_by_frame[idx] = g

    match_intervals = _load_match_intervals(layout, clip_id=manifest.clip_id)

    frame_iter, fps_src, frame_size = _iter_frames(video_path)

    fps = float(manifest.fps or 0.0)
    if fps <= 0.0:
        fps = fps_src if fps_src > 0.0 else 30.0

    if frame_size[0] <= 0 or frame_size[1] <= 0:
        # fallback to first frame size
        frame_iter, fps_src2, frame_size = _iter_frames(video_path)
        fps_src = fps_src2

    writer_path = layout.clip_root / "_debug" / output_name
    vw = VideoWriter(writer_path, fps=fps, frame_size=(int(frame_size[0]), int(frame_size[1])))

    try:
        for fi, frame in frame_iter:
            tf = tracks_by_frame.get(fi)
            if tf is not None and not tf.empty:
                _annotate_frame(frame, tf, match_intervals=match_intervals, frame_index=fi)
            cv2.putText(frame, f"frame={fi}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            vw.write(frame)
    finally:
        vw.close()

    return writer_path
