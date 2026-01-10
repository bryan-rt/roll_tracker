# src/bjj_pipeline/tools/homography_calibrate.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from math import isfinite


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_3x3(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3,3), got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("Homography contains non-finite values.")
    return H


def _normalize_h(H: np.ndarray) -> np.ndarray:
    """Normalize so H[2,2] == 1 when possible (common convention)."""
    H = _ensure_3x3(H)
    denom = H[2, 2]
    if abs(denom) > 1e-12:
        H = H / denom
    return H


def _default_homography_json_path(configs_root: Path, camera_id: str) -> Path:
    return configs_root / "cameras" / camera_id / "homography.json"


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _load_existing_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _write_homography_json(
    out_path: Path,
    camera_id: str,
    H: np.ndarray,
    source: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "H": _normalize_h(H).astype(float).tolist(),  # matches your reference artifact
        "camera_id": camera_id,
        "source": source,
        "created_at": _iso_utc_now(),
    }
    if extra:
        # Additive only — never break existing keys
        for k, v in extra.items():
            if k in payload:
                continue
            payload[k] = v

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"✔ Wrote homography → {out_path}")


# -------------------------
# Optional: interactive mode
# -------------------------
@dataclass
class ClickPairs:
    image_points_px: List[Tuple[float, float]]
    mat_points: List[Tuple[float, float]]


def _try_load_mat_blueprint(mat_blueprint_path: Path) -> Optional[Dict[str, Any]]:
    if not mat_blueprint_path.exists():
        return None
    try:
        with mat_blueprint_path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_rects_from_blueprint(blueprint: Any) -> List[Tuple[float, float, float, float, str]]:
    """
    Evidence-based parser for configs/mat_blueprint.json:
      the file is a JSON array of rectangle specs:
        [{"label": str, "x": number, "y": number, "width": number, "height": number}, ...]
    Returns: [(x, y, w, h, label), ...] with numeric validity checks applied.
    """
    if not isinstance(blueprint, list):
        return []
    rects: List[Tuple[float, float, float, float, str]] = []
    for r in blueprint:
        if not isinstance(r, dict):
            continue
        x = r.get("x"); y = r.get("y"); w = r.get("width"); h = r.get("height")
        if not all(isinstance(v, (int, float)) and isfinite(float(v)) for v in (x, y, w, h)):
            continue
        x = float(x); y = float(y); w = float(w); h = float(h)
        if w <= 0 or h <= 0:
            continue
        label = r.get("label") if isinstance(r.get("label"), str) else ""
        rects.append((x, y, w, h, label))
    return rects

def _render_mat_blueprint_rects(ax, blueprint: Any) -> None:
    """
    Evidence-based renderer for configs/mat_blueprint.json:
    the file is a JSON array of rectangle specs:
      [{"label": str, "x": number, "y": number, "width": number, "height": number}, ...]
    """
    try:
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError:
        # If matplotlib isn't available, caller will already error in interactive mode.
        return

    if not isinstance(blueprint, list):
        print("[D7] mat_blueprint: expected a JSON list of rectangles; got", type(blueprint).__name__)
        return []
    rects = _parse_rects_from_blueprint(blueprint)

    if not rects:
        print("[D7] mat_blueprint: no valid rectangles found to render.")
        return []

    # Draw rectangles
    for (x, y, w, h, label) in rects:
        ax.add_patch(Rectangle((x, y), w, h, fill=False))
        if label.strip():
            ax.text(x + 0.5 * w, y + 0.5 * h, label, ha="center", va="center")

    # Fit view to rectangles with padding
    xs = [x for (x, _, w, _, _) in rects] + [x + w for (x, _, w, _, _) in rects]
    ys = [y for (_, y, _, h, _) in rects] + [y + h for (_, y, _, h, _) in rects]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = max(1.0, 0.05 * (xmax - xmin))
    pad_y = max(1.0, 0.05 * (ymax - ymin))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    return rects


def _rect_union_bounds(rects: List[Tuple[float, float, float, float, str]]) -> Tuple[float, float, float, float]:
    xs = [x for (x, _, w, _, _) in rects] + [x + w for (x, _, w, _, _) in rects]
    ys = [y for (_, y, _, h, _) in rects] + [y + h for (_, y, _, h, _) in rects]
    return (min(xs), min(ys), max(xs), max(ys))


def _make_union_mask(
    rects: List[Tuple[float, float, float, float, str]],
    *,
    step_m: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize the UNION of rectangles into a boolean mask (no shapely dependency).
    Returns: mask, xs, ys where:
      mask shape = (len(ys), len(xs)) and mask[j,i] indicates point (xs[i], ys[j]) is inside union.
    """
    xs = np.arange(xmin, xmax + 1e-9, step_m, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, step_m, dtype=float)
    mask = np.zeros((ys.size, xs.size), dtype=bool)
    for (x, y, w, h, _) in rects:
        x2 = x + w
        y2 = y + h
        xi0 = int(np.searchsorted(xs, x, side="left"))
        xi1 = int(np.searchsorted(xs, x2, side="right"))
        yi0 = int(np.searchsorted(ys, y, side="left"))
        yi1 = int(np.searchsorted(ys, y2, side="right"))
        mask[yi0:yi1, xi0:xi1] = True
    return mask, xs, ys


def _iter_masked_polylines_constant_x(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x_idx: int,
) -> List[np.ndarray]:
    """
    For a fixed x column index, return a list of polyline point arrays (mat coords)
    representing contiguous segments where mask==True.
    """
    col = mask[:, x_idx]
    segments: List[np.ndarray] = []
    start = None
    for j, inside in enumerate(col):
        if inside and start is None:
            start = j
        if (not inside) and start is not None:
            js = np.arange(start, j, dtype=int)
            pts = np.column_stack([np.full(js.shape, xs[x_idx]), ys[js]])
            segments.append(pts)
            start = None
    if start is not None:
        js = np.arange(start, col.size, dtype=int)
        pts = np.column_stack([np.full(js.shape, xs[x_idx]), ys[js]])
        segments.append(pts)
    return segments


def _iter_masked_polylines_constant_y(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    y_idx: int,
) -> List[np.ndarray]:
    row = mask[y_idx, :]
    segments: List[np.ndarray] = []
    start = None
    for i, inside in enumerate(row):
        if inside and start is None:
            start = i
        if (not inside) and start is not None:
            is_ = np.arange(start, i, dtype=int)
            pts = np.column_stack([xs[is_], np.full(is_.shape, ys[y_idx])])
            segments.append(pts)
            start = None
    if start is not None:
        is_ = np.arange(start, row.size, dtype=int)
        pts = np.column_stack([xs[is_], np.full(is_.shape, ys[y_idx])])
        segments.append(pts)
    return segments


def _project_polyline_mat_to_img(H: np.ndarray, pts_mat: np.ndarray) -> np.ndarray:
    """
    pts_mat: (N,2) in mat coords. Returns (N,2) image coords.
    """
    import cv2  # local optional dep
    pts = np.asarray(pts_mat, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return proj.reshape(-1, 2)


def _qa_overlay_dialog(
    *,
    camera_id: str,
    frame_rgb: np.ndarray,
    H_mat_to_img: np.ndarray,
    rects: List[Tuple[float, float, float, float, str]],
    grid_spacing_m: float = 0.5,
    sample_step_m: float = 0.05,
) -> bool:
    """
    Show a QA overlay window: mat-union grid (0.5m spacing) projected onto the frame via H.
    Returns True if accepted, False if redo requested.
    """
    import matplotlib.pyplot as plt  # noqa

    if not rects:
        print("[D7][QA] No blueprint rects available; cannot render grid QA.")
        return True

    xmin, ymin, xmax, ymax = _rect_union_bounds(rects)
    mask, xs, ys = _make_union_mask(rects, step_m=sample_step_m, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    # Determine which raster columns/rows correspond to the requested grid spacing.
    x0 = np.ceil(xmin / grid_spacing_m) * grid_spacing_m
    y0 = np.ceil(ymin / grid_spacing_m) * grid_spacing_m
    grid_xs = np.arange(x0, xmax + 1e-9, grid_spacing_m)
    grid_ys = np.arange(y0, ymax + 1e-9, grid_spacing_m)

    # Map grid positions to nearest raster indices.
    x_idxs = [int(np.argmin(np.abs(xs - gx))) for gx in grid_xs]
    y_idxs = [int(np.argmin(np.abs(ys - gy))) for gy in grid_ys]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame_rgb)
    ax.set_axis_off()
    ax.set_title(f"QA Overlay — {camera_id} (grid {grid_spacing_m}m). Keys: [a]=accept  [r]=redo")
    # IMPORTANT: lock axes to image pixel coordinates to prevent autoscale from hiding the image.
    h, w = frame_rgb.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # y-down to match image convention
    ax.set_autoscale_on(False)

    def _plot_polyline_img(pts_img: np.ndarray) -> None:
        pts_img = np.asarray(pts_img, dtype=float)
        if pts_img.ndim != 2 or pts_img.shape[1] != 2:
            return
        ok = np.isfinite(pts_img).all(axis=1)
        pts_img = pts_img[ok]
        if pts_img.shape[0] < 2:
            return
        ax.plot(pts_img[:, 0], pts_img[:, 1], linewidth=1.0, clip_on=True)

    # Vertical grid lines (constant x)
    for xi in x_idxs:
        segs = _iter_masked_polylines_constant_x(mask, xs, ys, x_idx=xi)
        for pts_mat in segs:
            if pts_mat.shape[0] < 2:
                continue
            pts_img = _project_polyline_mat_to_img(H_mat_to_img, pts_mat)
            _plot_polyline_img(pts_img)

    # Horizontal grid lines (constant y)
    for yi in y_idxs:
        segs = _iter_masked_polylines_constant_y(mask, xs, ys, y_idx=yi)
        for pts_mat in segs:
            if pts_mat.shape[0] < 2:
                continue
            pts_img = _project_polyline_mat_to_img(H_mat_to_img, pts_mat)
            _plot_polyline_img(pts_img)

    decision = {"accept": False, "redo": False}

    def on_key(event):
        k = (event.key or "").lower()
        if k == "a":
            decision["accept"] = True
            plt.close(fig)
        elif k == "r":
            decision["redo"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return bool(decision["accept"]) and not bool(decision["redo"])


def _interactive_calibrate(
    camera_id: str,
    out_path: Path,
    video_path: Path,
    mat_blueprint_path: Path,
) -> None:
    """
    Minimal interactive calibrator skeleton:
      - shows first frame + mat blueprint plot
      - collects alternating clicks (frame -> mat) until >= 4 pairs
      - solves H and saves canonical JSON

    Notes:
      - Keeps dependencies optional: only imports cv2/matplotlib when invoked.
      - You can later make this match the full D7 UI spec (undo/clear/save keys, overlays, etc).
    """
    import cv2  # noqa
    import matplotlib.pyplot as plt  # noqa

    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    blueprint = _try_load_mat_blueprint(mat_blueprint_path)
    if blueprint is None:
        print(f"[D7] mat_blueprint not found or unreadable: {mat_blueprint_path}")
    else:
        print(f"[D7] loaded mat_blueprint: {mat_blueprint_path}")

    pairs = ClickPairs(image_points_px=[], mat_points=[])

    fig, (ax_img, ax_mat) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Homography Calibrator — {camera_id}")

    ax_img.imshow(frame_rgb)
    ax_img.set_title("Image (px) — click points here")
    ax_img.set_axis_off()

    ax_mat.set_title("Mat blueprint — click corresponding points here")
    ax_mat.set_xlabel("mat_x")
    ax_mat.set_ylabel("mat_y")
    ax_mat.grid(True)

    # Evidence-based rendering for your rectangle-list blueprint schema
    rects: List[Tuple[float, float, float, float, str]] = []
    if blueprint is not None:
        rects = _render_mat_blueprint_rects(ax_mat, blueprint)

    state = {"expect": "img"}  # img then mat alternating
    # --- Persistent point/label artists (avoid ax.lines.clear(), which breaks on newer mpl) ---
    img_point_artists: List[Any] = []
    img_text_artists: List[Any] = []
    mat_point_artists: List[Any] = []
    mat_text_artists: List[Any] = []

    def _add_point(ax, x: float, y: float, *, color: str, label: str):
        # plot() returns a Line2D; keep references so we can remove on undo/clear
        pt = ax.plot([x], [y], marker="o", linestyle="none", color=color)[0]
        txt = ax.text(x, y, label, color=color, fontsize=10, ha="left", va="bottom")
        return pt, txt

    def _undo_last():
        # If we have an unmatched image point (img > mat), remove that image point.
        if len(pairs.image_points_px) > len(pairs.mat_points):
            pairs.image_points_px.pop()
            if img_point_artists:
                img_point_artists.pop().remove()
            if img_text_artists:
                img_text_artists.pop().remove()
            state["expect"] = "img"
            return
        # Otherwise remove the last complete pair.
        if pairs.image_points_px and pairs.mat_points:
            pairs.image_points_px.pop()
            pairs.mat_points.pop()
            if img_point_artists:
                img_point_artists.pop().remove()
            if img_text_artists:
                img_text_artists.pop().remove()
            if mat_point_artists:
                mat_point_artists.pop().remove()
            if mat_text_artists:
                mat_text_artists.pop().remove()
            state["expect"] = "img"

    def _clear_all():
        pairs.image_points_px.clear()
        pairs.mat_points.clear()
        for a in img_point_artists:
            a.remove()
        for a in img_text_artists:
            a.remove()
        for a in mat_point_artists:
            a.remove()
        for a in mat_text_artists:
            a.remove()
        img_point_artists.clear()
        img_text_artists.clear()
        mat_point_artists.clear()
        mat_text_artists.clear()
        state["expect"] = "img"

    def on_click(event):
        if event.inaxes not in (ax_img, ax_mat):
            return
        # Ignore clicks with no data coords (can happen on some backends)
        if event.xdata is None or event.ydata is None:
            return

        if state["expect"] == "img":
            if event.inaxes is not ax_img:
                print("Click on IMAGE (left) first.")
                return
            x = float(event.xdata); y = float(event.ydata)
            pairs.image_points_px.append((x, y))
            n = len(pairs.image_points_px)
            label = f"r{n}"
            pt, txt = _add_point(ax_img, x, y, color="red", label=label)
            img_point_artists.append(pt)
            img_text_artists.append(txt)
            state["expect"] = "mat"
            print(f"Image point #{n} = ({x:.1f}, {y:.1f}) [{label}]")
        else:
            if event.inaxes is not ax_mat:
                print("Click on MAT (right) next.")
                return
            x = float(event.xdata); y = float(event.ydata)
            pairs.mat_points.append((x, y))
            n = len(pairs.mat_points)
            label = f"b{n}"
            pt, txt = _add_point(ax_mat, x, y, color="blue", label=label)
            mat_point_artists.append(pt)
            mat_text_artists.append(txt)
            state["expect"] = "img"
            print(f"Mat point   #{n} = ({x:.3f}, {y:.3f}) [{label}]")

        fig.canvas.draw_idle()

    def on_key(event):
        k = (event.key or "").lower()

        if k == "u":
            _undo_last()
            print("Undo.")
            fig.canvas.draw_idle()

        elif k == "c":
            _clear_all()
            print("Cleared.")
            fig.canvas.draw_idle()

        elif k == "s":
            if len(pairs.image_points_px) != len(pairs.mat_points) or len(pairs.image_points_px) < 4:
                print("Need >= 4 complete point pairs before solving.")
                return

            img_pts = np.array(pairs.image_points_px, dtype=float)
            mat_pts = np.array(pairs.mat_points, dtype=float)

            # Compute H mapping mat -> image or image -> mat?
            # Convention: we'll compute H such that [x_img,y_img,1]^T ~ H * [x_mat,y_mat,1]^T
            # This matches the typical "project mat coords into image" overlay use.
            import cv2  # noqa

            H, mask = cv2.findHomography(mat_pts, img_pts, method=cv2.RANSAC)
            H = _ensure_3x3(H)

            inliers = int(mask.sum()) if mask is not None else None
            # QA loop: overlay a 0.5m grid (union of blueprint rectangles) projected onto the frame.
            # Accept -> write homography.json and exit. Redo -> clear points and continue selecting.
            print("[D7] Computed homography. Launching QA overlay... (accept=a, redo=r)")
            accepted = _qa_overlay_dialog(
                camera_id=camera_id,
                frame_rgb=frame_rgb,
                H_mat_to_img=H,
                rects=rects,
                grid_spacing_m=0.5,
                sample_step_m=0.05,
            )
            if not accepted:
                print("[D7] QA requested redo. Clearing points; please re-select correspondences.")
                _clear_all()
                fig.canvas.draw_idle()
                return

            _write_homography_json(
                out_path=out_path,
                camera_id=camera_id,
                H=H,
                source={"type": "interactive_clicks", "video": str(video_path)},
                extra={
                    "correspondences": {
                        "image_points_px": pairs.image_points_px,
                        "mat_points": pairs.mat_points,
                    },
                    "fit": {
                        "method": "cv2.findHomography_ransac",
                        "num_points": len(pairs.image_points_px),
                        "inliers": inliers,
                    },
                    "qa": {
                        "grid_spacing_m": 0.5,
                        "sample_step_m": 0.05,
                        "accepted": True,
                    },
                },
            )
            print("[D7] Saved homography (accepted). Closing calibrator and returning to pipeline...")
            plt.close(fig)

        elif k == "q":
            print("Quit without saving.")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: click IMAGE then MAT (repeat). Keys: [u]=undo [c]=clear [s]=solve+save [q]=quit")
    plt.show()


def main():
    p = argparse.ArgumentParser(
        prog="python -m bjj_pipeline.tools.homography_calibrate",
        description="Homography calibration/import tool (D7). Writes configs/cameras/<camera>/homography.json",
    )
    p.add_argument("--camera", required=True, help="Camera id (e.g. cam03)")
    p.add_argument(
        "--configs-root",
        default="configs",
        help="Repo configs root (default: ./configs)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Override output JSON path (default: configs/cameras/<camera>/homography.json)",
    )

    sub = p.add_subparsers(dest="mode", required=True)

    p_import = sub.add_parser("import", help="Import an existing matrix (.npy or .json) and write canonical homography.json")
    p_import.add_argument("--npy", default=None, help="Path to homography_matrix.npy")
    p_import.add_argument("--json", default=None, help="Path to existing homography.json containing top-level 'H'")

    p_interactive = sub.add_parser("interactive", help="Interactive click-based calibration from a video")
    p_interactive.add_argument("--video", required=True, help="Path to an mp4 to grab the first frame from")
    p_interactive.add_argument(
        "--mat-blueprint",
        default="configs/mat_blueprint.json",
        help="Path to mat blueprint json (default: configs/mat_blueprint.json)",
    )

    p_placeholder = sub.add_parser(
        "placeholder",
        help="Write a placeholder identity homography.json marked as needing calibration (useful for tests/onboarding)",
    )

    args = p.parse_args()

    configs_root = Path(args.configs_root)
    out_path = Path(args.out) if args.out else _default_homography_json_path(configs_root, args.camera)

    if args.mode == "import":
        if bool(args.npy) == bool(args.json):
            raise SystemExit("Provide exactly one of --npy or --json")

        if args.npy:
            npy_path = Path(args.npy)
            H = _load_npy(npy_path)
            _write_homography_json(
                out_path=out_path,
                camera_id=args.camera,
                H=H,
                source={"type": "imported_npy", "path": str(npy_path)},
            )
        else:
            json_path = Path(args.json)
            payload = _load_existing_json(json_path)
            if "H" not in payload:
                raise ValueError(f"Input JSON missing top-level 'H': {json_path}")
            H = np.array(payload["H"], dtype=float)
            _write_homography_json(
                out_path=out_path,
                camera_id=args.camera,
                H=H,
                source={"type": "imported_json", "path": str(json_path)},
                extra={k: v for k, v in payload.items() if k not in {"H", "camera_id", "source", "created_at"}},
            )

    elif args.mode == "interactive":
        _interactive_calibrate(
            camera_id=args.camera,
            out_path=out_path,
            video_path=Path(args.video),
            mat_blueprint_path=Path(args.mat_blueprint),
        )
    elif args.mode == "placeholder":
        H = np.eye(3, dtype=float)
        _write_homography_json(
            out_path=out_path,
            camera_id=args.camera,
            H=H,
            source={"type": "placeholder_identity"},
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
