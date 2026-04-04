"""Camera geometry analysis — height surfaces, ROI masks, detectability, coverage optimization.

Standalone diagnostic tool for the gym setup wizard flow.  Analyzes camera
geometry using existing pipeline outputs to produce detectability heatmaps
and optimized per-camera configurations.

Usage:
    python tools/camera_geometry_analysis.py all --outputs outputs \
        --gym-id c8a592a4-2bca-400a-80e1-fec0e5cbea77
    python tools/camera_geometry_analysis.py phase1 --outputs outputs --gym-id <uuid>

Affine offset model (v3):
    H_mat2img maps the ground plane to distorted pixels (foot positions, trusted).
    An affine offset (2x3, 6 DOF) maps world (X,Y) to the pixel displacement
    from foot to head: offset(X,Y) = [a1*X + a2*Y + a3, b1*X + b2*Y + b3].
    Pixel height at any mat position is ||offset(X,Y)||.
    Anchored to H_foot so it cannot go degenerate (unlike a free H_head
    homography which collapses when training data is spatially concentrated).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from sklearn.linear_model import LinearRegression, RANSACRegressor
import typer

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_PERSON_HEIGHT_M = 1.83
DEFAULT_DETECTION_FLOOR_PX = 45.0
DEFAULT_OVERLAP_BUFFER_M = 1.0
KNEELING_FRACTION = 0.45  # kneeling height as fraction of standing
ROI_HEAD_BUFFER_FRAC = 0.10  # 10% buffer above head for ROI mask
PERIMETER_SAMPLE_SPACING_M = 0.1
MAT_GRID_SPACING_M = 0.25
MIN_DATA_POINTS_WARN = 50
MIN_DATA_POINTS_H_HEAD = 20  # cv2.findHomography needs ≥4, we want margin

DETECTABILITY_CONFIGS: List[Dict[str, Any]] = [
    {"name": "640_full", "imgsz": 640, "roi_crop": False, "sahi": None},
    {"name": "800_full", "imgsz": 800, "roi_crop": False, "sahi": None},
    {"name": "960_full", "imgsz": 960, "roi_crop": False, "sahi": None},
    {"name": "1280_full", "imgsz": 1280, "roi_crop": False, "sahi": None},
    {"name": "640_roi", "imgsz": 640, "roi_crop": True, "sahi": None},
    {"name": "800_roi", "imgsz": 800, "roi_crop": True, "sahi": None},
    {"name": "960_roi", "imgsz": 960, "roi_crop": True, "sahi": None},
    {
        "name": "640_sahi2x2",
        "imgsz": 640,
        "roi_crop": True,
        "sahi": {"grid": [2, 2], "overlap": 0.2},
    },
    {
        "name": "640_sahi3x2",
        "imgsz": 640,
        "roi_crop": True,
        "sahi": {"grid": [3, 2], "overlap": 0.2},
    },
    {
        "name": "640_sahi2x1",
        "imgsz": 640,
        "roi_crop": True,
        "sahi": {"grid": [2, 1], "overlap": 0.2},
    },
]


# ── Data Loading ───────────────────────────────────────────────────────────────


def discover_cameras(outputs_root: Path, gym_id: str) -> List[str]:
    """Discover camera IDs from output directory structure."""
    gym_dir = outputs_root / gym_id
    if not gym_dir.is_dir():
        console.print(f"[red]Gym directory not found: {gym_dir}[/red]")
        raise typer.Exit(1)
    skip = {"sessions", "_analysis", "_benchmarks", "_debug"}
    cameras = sorted(
        d.name
        for d in gym_dir.iterdir()
        if d.is_dir() and d.name not in skip
    )
    if not cameras:
        console.print(f"[red]No camera directories found in {gym_dir}[/red]")
        raise typer.Exit(1)
    return cameras


def load_camera_geometry(
    cam_id: str, configs_root: Path = Path("configs")
) -> Dict[str, Any]:
    """Load camera geometry from homography.json.

    Returns dict with H_mat2img (mat->distorted pixel), K, D, image dimensions.
    """
    h_path = configs_root / "cameras" / cam_id / "homography.json"
    if not h_path.exists():
        raise FileNotFoundError(f"Homography not found: {h_path}")
    payload = json.loads(h_path.read_text())

    H_mat2img = np.asarray(payload["H"], dtype=np.float64).reshape(3, 3)

    K = None
    if payload.get("camera_matrix") is not None:
        K = np.asarray(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)

    D = None
    if payload.get("dist_coefficients") is not None:
        D = np.asarray(payload["dist_coefficients"], dtype=np.float64).ravel()

    # Image size from lens_calibration or projected_polylines
    img_size = None
    if "lens_calibration" in payload and payload["lens_calibration"]:
        img_size = payload["lens_calibration"].get("image_size")
    if img_size is None and "projected_polylines" in payload:
        img_size = payload["projected_polylines"].get("image_wh")

    return {
        "cam_id": cam_id,
        "H_mat2img": H_mat2img,
        "K": K,
        "D": D,
        "img_w": img_size[0] if img_size else 1280,
        "img_h": img_size[1] if img_size else 720,
        "payload": payload,
    }


def load_mat_blueprint(configs_root: Path = Path("configs")) -> Polygon:
    """Load mat blueprint and return union polygon in world coords."""
    bp_path = configs_root / "mat_blueprint.json"
    mats = json.loads(bp_path.read_text())
    polys = [box(m["x"], m["y"], m["x"] + m["width"], m["y"] + m["height"]) for m in mats]
    union = unary_union(polys)
    if isinstance(union, MultiPolygon):
        union = max(union.geoms, key=lambda g: g.area)
    return union


def load_mat_rectangles(configs_root: Path = Path("configs")) -> List[Dict]:
    """Load raw mat rectangles for plotting."""
    bp_path = configs_root / "mat_blueprint.json"
    return json.loads(bp_path.read_text())


def load_tracklet_data(
    outputs_root: Path, gym_id: str, cam_id: str
) -> pd.DataFrame:
    """Load and merge tracklet_frames + detections for all clips of a camera."""
    cam_dir = outputs_root / gym_id / cam_id
    tf_files = sorted(cam_dir.rglob("stage_A/tracklet_frames.parquet"))
    det_files = sorted(cam_dir.rglob("stage_A/detections.parquet"))

    if not tf_files:
        console.print(f"[yellow]No tracklet_frames found for {cam_id}[/yellow]")
        return pd.DataFrame()

    tf = pd.concat([pd.read_parquet(f) for f in tf_files], ignore_index=True)
    det = pd.concat([pd.read_parquet(f) for f in det_files], ignore_index=True)

    bbox_cols = ["detection_id", "x1", "y1", "x2", "y2", "confidence"]
    return tf.merge(det[bbox_cols], on="detection_id", how="left")


def find_camera_frame(
    outputs_root: Path,
    gym_id: str,
    cam_id: str,
    data_root: Path = Path("data"),
) -> Optional[np.ndarray]:
    """Try to grab a single video frame for visualization background."""
    search_dirs = [
        data_root / "raw" / "nest" / gym_id / cam_id,
        data_root / "raw" / "nest" / cam_id,
        data_root / "raw" / cam_id,
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for video in sorted(d.rglob("*.mp4"))[:1]:
            cap = cv2.VideoCapture(str(video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
    return None


# ── Projection Helpers ─────────────────────────────────────────────────────────


def project_world_to_pixel(
    H_mat2img: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Project Nx2 world coords to distorted pixel coords via H.

    H on disk is mat->distorted pixel (verified empirically: mean error
    2.7-5.8 px across all cameras using stored correspondences).
    """
    n = points.shape[0]
    pts_h = np.hstack([points, np.ones((n, 1))])  # Nx3
    proj = (H_mat2img @ pts_h.T).T  # Nx3
    w = proj[:, 2:3].copy()
    w[w == 0] = 1e-10
    return proj[:, :2] / w


def verify_h_projection(geom: Dict[str, Any]) -> Dict[str, Any]:
    """Verify H projection accuracy against stored correspondences."""
    payload = geom["payload"]
    corr = payload.get("correspondences", {})
    ip = corr.get("image_points_px", [])
    mp = corr.get("mat_points", [])
    if not ip or not mp:
        return {"verified": False, "reason": "no_correspondences"}

    mat_pts = np.array(mp, dtype=np.float64)
    img_pts = np.array(ip, dtype=np.float64)
    proj = project_world_to_pixel(geom["H_mat2img"], mat_pts)
    errors = np.sqrt(np.sum((proj - img_pts) ** 2, axis=1))

    return {
        "verified": True,
        "mean_error_px": float(errors.mean()),
        "max_error_px": float(errors.max()),
        "per_point_errors": errors.tolist(),
    }


# ── Quadratic Offset Height Model ──────────────────────────────────────────────


def _offset_features(xy: np.ndarray) -> np.ndarray:
    """Build degree-2 feature matrix: [x, y, x², xy, y², 1].

    The 1/distance perspective falloff is nonlinear — quadratic terms
    capture the curvature that a linear affine misses.  Returns Nx6.
    """
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([x, y, x ** 2, x * y, y ** 2, np.ones(len(x))])


_MIN_PIXEL_HEIGHT = 20.0  # physical floor — no person bbox is shorter than this


def fit_offset_affine(
    H_foot: np.ndarray,
    foot_world_xy: np.ndarray,
    head_pixel_xy: np.ndarray,
    mat_poly: Polygon,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Fit quadratic offset model: head_px = foot_px(H_foot) + offset(world_xy).

    Instead of fitting a free 8-DOF homography for the head plane (which
    degenerates when training data is spatially concentrated), we fit the
    OFFSET between the trusted H_foot projection and the observed head
    pixel position as a degree-2 polynomial (2x6, 12 DOF).

    Features: [x, y, x², xy, y², 1] — captures 1/distance perspective.

    Returns (offset_coefs_2x6, fit_metrics).  None if fitting fails.
    """
    n = len(foot_world_xy)
    if n < MIN_DATA_POINTS_H_HEAD:
        return None, {"error": "insufficient_points", "n_data_points": n}

    # Diagnostic: spatial coverage of training data
    mb = mat_poly.bounds  # (xmin, ymin, xmax, ymax)
    x_m, y_m = foot_world_xy[:, 0], foot_world_xy[:, 1]
    mat_span_x = mb[2] - mb[0]
    mat_span_y = mb[3] - mb[1]
    x_cov = (x_m.max() - x_m.min()) / mat_span_x if mat_span_x > 0 else 0
    y_cov = (y_m.max() - y_m.min()) / mat_span_y if mat_span_y > 0 else 0
    console.print(f"  Spatial coverage: x={x_cov:.0%}, y={y_cov:.0%}")

    # Compute offsets: observed head position minus H_foot-projected foot position
    foot_px_from_H = project_world_to_pixel(H_foot, foot_world_xy)
    offset_observed = head_pixel_xy - foot_px_from_H  # Nx2 (dx, dy)

    # Feature matrix: [x, y, x², xy, y², 1]
    A_feat = _offset_features(foot_world_xy)

    # Fit dx and dy with RANSAC
    ransac_dx = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        residual_threshold=None, random_state=42,
    )
    ransac_dy = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        residual_threshold=None, random_state=42,
    )
    ransac_dx.fit(A_feat, offset_observed[:, 0])
    ransac_dy.fit(A_feat, offset_observed[:, 1])

    offset_affine = np.vstack([
        ransac_dx.estimator_.coef_,  # [a1, a2, a3] for dx
        ransac_dy.estimator_.coef_,  # [b1, b2, b3] for dy
    ])

    # Residual stats per component
    dx_inliers = ransac_dx.inlier_mask_
    dy_inliers = ransac_dy.inlier_mask_
    dx_pred = ransac_dx.predict(A_feat)
    dy_pred = ransac_dy.predict(A_feat)
    dx_res = np.abs(offset_observed[dx_inliers, 0] - dx_pred[dx_inliers])
    dy_res = np.abs(offset_observed[dy_inliers, 1] - dy_pred[dy_inliers])

    console.print(
        f"  Affine dx: inliers={dx_inliers.sum()}/{n}, "
        f"residual mean={dx_res.mean():.1f}px, p95={np.percentile(dx_res, 95):.1f}px"
    )
    console.print(
        f"  Affine dy: inliers={dy_inliers.sum()}/{n}, "
        f"residual mean={dy_res.mean():.1f}px, p95={np.percentile(dy_res, 95):.1f}px"
    )
    if np.percentile(dx_res, 95) > 30 or np.percentile(dy_res, 95) > 30:
        console.print("  [yellow]Warning: p95 residual > 30px — affine may be too simple[/yellow]")

    # Diagnostic: compare H_head (degenerate) vs affine at mat corners
    mat_corners = np.array([
        [mb[0], mb[1]], [mb[2], mb[1]], [mb[2], mb[3]], [mb[0], mb[3]],
        [(mb[0] + mb[2]) / 2, (mb[1] + mb[3]) / 2],
    ])
    corner_labels = ["BL", "BR", "TR", "TL", "Center"]

    # H_head for comparison (fit but don't use)
    src = foot_world_xy.reshape(-1, 1, 2).astype(np.float64)
    dst = head_pixel_xy.reshape(-1, 1, 2).astype(np.float64)
    H_head_cmp, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    foot_corner_px = project_world_to_pixel(H_foot, mat_corners)
    corner_feat = _offset_features(mat_corners)
    affine_offset = corner_feat @ offset_affine.T
    affine_head_px = foot_corner_px + affine_offset
    affine_heights = np.sqrt(np.sum(affine_offset ** 2, axis=1))

    console.print("  Corner/center diagnostics (affine vs H_head):")
    if H_head_cmp is not None:
        h_head_cond = np.linalg.cond(H_head_cmp)
        console.print(f"    H_head cond={h_head_cond:.1e}")
        hhead_corner_px = project_world_to_pixel(H_head_cmp, mat_corners)
        hhead_heights = np.sqrt(np.sum((foot_corner_px - hhead_corner_px) ** 2, axis=1))
    else:
        hhead_heights = np.full(5, np.nan)

    for i, lb in enumerate(corner_labels):
        ah = affine_heights[i]
        hh = hhead_heights[i]
        hh_flag = " DEGEN" if (not np.isnan(hh) and (hh > 500 or hh < 20)) else ""
        console.print(f"    {lb}: affine={ah:.0f}px, H_head={hh:.0f}px{hh_flag}")

    # Combined reprojection error (Euclidean on training data)
    pred_offset = A_feat @ offset_affine.T
    pred_head = foot_px_from_H + pred_offset
    combined_inliers = dx_inliers & dy_inliers
    errs = np.sqrt(np.sum((pred_head[combined_inliers] - head_pixel_xy[combined_inliers]) ** 2, axis=1))

    metrics: Dict[str, Any] = {
        "n_data_points": n,
        "n_inliers_dx": int(dx_inliers.sum()),
        "n_inliers_dy": int(dy_inliers.sum()),
        "dx_residual_mean_px": round(float(dx_res.mean()), 2),
        "dx_residual_p95_px": round(float(np.percentile(dx_res, 95)), 2),
        "dy_residual_mean_px": round(float(dy_res.mean()), 2),
        "dy_residual_p95_px": round(float(np.percentile(dy_res, 95)), 2),
        "reproj_error_mean_px": round(float(errs.mean()), 2) if len(errs) > 0 else 0.0,
        "reproj_error_p95_px": round(float(np.percentile(errs, 95)), 2) if len(errs) > 0 else 0.0,
        "spatial_coverage_x": round(x_cov, 3),
        "spatial_coverage_y": round(y_cov, 3),
    }
    return offset_affine, metrics


def compute_pixel_height(
    H_foot: np.ndarray,
    offset_coefs: np.ndarray,
    world_xy: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pixel height of a standing person at each mat position.

    Uses H_foot (trusted ground-plane homography) + quadratic offset model.
    pixel_height(X,Y) = ||offset_coefs @ features(X,Y)||

    Returns:
        (pixel_heights, in_frame, foot_px, head_px).
        in_frame checks foot projection only — head can be above frame.
    """
    foot_px = project_world_to_pixel(H_foot, world_xy)
    feat = _offset_features(world_xy)
    offset = feat @ offset_coefs.T  # Nx2 (dx, dy)
    head_px = foot_px + offset

    pixel_heights = np.clip(
        np.sqrt(np.sum(offset ** 2, axis=1)),
        _MIN_PIXEL_HEIGHT, None,
    )

    # in_frame: foot must be in frame (head can extend above — still detectable)
    in_frame = (
        (foot_px[:, 0] >= 0) & (foot_px[:, 0] < img_w)
        & (foot_px[:, 1] >= 0) & (foot_px[:, 1] < img_h)
    )

    return pixel_heights, in_frame, foot_px, head_px


# ── Mat Perimeter ──────────────────────────────────────────────────────────────


def sample_perimeter(polygon: Polygon, spacing_m: float = 0.1) -> np.ndarray:
    """Sample points along polygon perimeter at regular intervals."""
    boundary = polygon.exterior
    n_samples = max(int(boundary.length / spacing_m), 4)
    points = []
    for i in range(n_samples):
        pt = boundary.interpolate(i / n_samples, normalized=True)
        points.append([pt.x, pt.y])
    return np.array(points)


# ── Mat Grid Helpers ───────────────────────────────────────────────────────────


def _mat_grid(mat_poly: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a dense grid inside the mat polygon.

    Returns (grid_pts Nx2, on_mat boolean mask for the full grid).
    """
    bounds = mat_poly.bounds
    gx = np.arange(bounds[0], bounds[2] + 0.01, MAT_GRID_SPACING_M)
    gy = np.arange(bounds[1], bounds[3] + 0.01, MAT_GRID_SPACING_M)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    import shapely

    pts = shapely.points(grid_pts[:, 0], grid_pts[:, 1])
    on_mat = shapely.contains(mat_poly, pts)
    return grid_pts[on_mat], on_mat


def _draw_mat_rects(ax: plt.Axes, configs_root: Path, alpha: float = 0.3) -> None:
    """Draw mat blueprint rectangles on an axes."""
    for m in load_mat_rectangles(configs_root):
        rect = mpatches.Rectangle(
            (m["x"], m["y"]),
            m["width"],
            m["height"],
            linewidth=1,
            edgecolor="gray",
            facecolor="#f0f0f0",
            alpha=alpha,
        )
        ax.add_patch(rect)


def _load_offset_affine(cam_id: str, configs_root: Path) -> Optional[np.ndarray]:
    """Load offset_affine (2x3) from Phase 1 output (height_surface.json)."""
    surface_path = configs_root / "cameras" / cam_id / "height_surface.json"
    if not surface_path.exists():
        return None
    data = json.loads(surface_path.read_text())
    raw = data.get("offset_affine")
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float64).reshape(2, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Head Homography Fitting
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase1(
    outputs_root: Path,
    gym_id: str,
    cameras: List[str],
    reference_height_m: Optional[float],
    configs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Phase 1: Fit affine offset model per camera from standing detections."""
    console.print("\n[bold cyan]═══ Phase 1: Affine Offset Model Fitting ═══[/bold cyan]")

    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")

        df = load_tracklet_data(outputs_root, gym_id, cam_id)
        if df.empty:
            console.print("  [yellow]No data, skipping[/yellow]")
            continue

        geom = load_camera_geometry(cam_id, configs_root)

        # Filter to high-quality upright detections
        df = df.dropna(subset=["x1", "y1", "x2", "y2", "x_m", "y_m", "confidence"])
        df["bbox_w"] = df["x2"] - df["x1"]
        df["bbox_h"] = df["y2"] - df["y1"]

        mask = (
            (df["bbox_h"] / df["bbox_w"].clip(lower=1) > 1.2)
            & (df["confidence"] > 0.5)
            & (df["on_mat"] == True)  # noqa: E712
            & (df["bbox_w"] > 0)
            & (df["bbox_h"] > 0)
        )
        filtered = df[mask].copy()

        # Proximity filter: reject detections with nearby neighbour in same frame
        if len(filtered) > 0:
            filtered["cx"] = (filtered["x1"] + filtered["x2"]) / 2
            filtered["cy"] = (filtered["y1"] + filtered["y2"]) / 2

            keep: List[int] = []
            for _key, group in filtered.groupby(["clip_id", "frame_index"]):
                if len(group) == 1:
                    keep.append(group.index[0])
                    continue
                centers = group[["cx", "cy"]].to_numpy(dtype=np.float64, na_value=np.nan)
                widths = group["bbox_w"].to_numpy(dtype=np.float64, na_value=np.nan)
                for i, idx in enumerate(group.index):
                    dists = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
                    dists[i] = np.inf
                    if dists.min() > 1.5 * widths[i]:
                        keep.append(idx)

            filtered = filtered.loc[keep]

        n_data = len(filtered)
        console.print(f"  Qualifying detections: {n_data}")

        if n_data < MIN_DATA_POINTS_H_HEAD:
            console.print(f"  [red]Insufficient data (<{MIN_DATA_POINTS_H_HEAD}), skipping[/red]")
            continue
        if n_data < MIN_DATA_POINTS_WARN:
            console.print(
                f"  [yellow]Warning: <{MIN_DATA_POINTS_WARN} points, model may be low-confidence[/yellow]"
            )

        # Extract world foot positions and head pixel positions
        foot_world = np.column_stack([
            filtered["x_m"].values.astype(np.float64),
            filtered["y_m"].values.astype(np.float64),
        ])
        head_pixel = np.column_stack([
            ((filtered["x1"] + filtered["x2"]) / 2).values.astype(np.float64),
            filtered["y1"].values.astype(np.float64),
        ])
        heights = filtered["bbox_h"].values.astype(np.float64)

        mat_poly = load_mat_blueprint(configs_root)

        # Fit affine offset model
        offset_affine, metrics = fit_offset_affine(
            geom["H_mat2img"], foot_world, head_pixel, mat_poly,
        )

        if offset_affine is None:
            console.print(f"  [red]Affine fitting failed: {metrics.get('error')}[/red]")
            continue

        console.print(
            f"  Affine reproj: mean={metrics['reproj_error_mean_px']:.1f}px, "
            f"p95={metrics['reproj_error_p95_px']:.1f}px"
        )

        # Optional reference height scaling
        ref_cal = None
        if reference_height_m is not None:
            px_heights, in_frame, _, _ = compute_pixel_height(
                geom["H_mat2img"], offset_affine, foot_world,
                geom["img_w"], geom["img_h"],
            )
            valid = in_frame & (px_heights > 0)
            if valid.any():
                pred_median = float(np.median(px_heights[valid]))
                ppm = pred_median / reference_height_m
                ref_cal = {
                    "reference_height_m": reference_height_m,
                    "pixels_per_meter_at_median": round(ppm, 2),
                }
                console.print(f"  Reference calibration: {ppm:.1f} px/m at median position")

        # Scatter data (subsample for JSON size)
        max_scatter = 2000
        if n_data > max_scatter:
            idx = np.random.default_rng(42).choice(n_data, max_scatter, replace=False)
        else:
            idx = np.arange(n_data)
        scatter = [
            {
                "x_m": round(float(foot_world[i, 0]), 3),
                "y_m": round(float(foot_world[i, 1]), 3),
                "bbox_height_px": round(float(heights[i]), 1),
            }
            for i in idx
        ]

        surface_data: Dict[str, Any] = {
            "camera_id": cam_id,
            "offset_affine": offset_affine.tolist(),
            "fit_metrics": metrics,
            "scatter_data": scatter,
            "reference_calibration": ref_cal,
        }

        cam_config_dir = configs_root / "cameras" / cam_id
        cam_config_dir.mkdir(parents=True, exist_ok=True)
        surface_path = cam_config_dir / "height_surface.json"
        surface_path.write_text(json.dumps(surface_data, indent=2))
        console.print(f"  Saved: {surface_path}")

        _plot_phase1(
            cam_id, geom, offset_affine, foot_world, heights,
            metrics, configs_root, analysis_dir,
        )
        results[cam_id] = surface_data

    return results


def _plot_phase1(
    cam_id: str,
    geom: Dict,
    offset_affine: np.ndarray,
    foot_world: np.ndarray,
    heights: np.ndarray,
    metrics: Dict,
    configs_root: Path,
    analysis_dir: Path,
) -> None:
    mat_poly = load_mat_blueprint(configs_root)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _draw_mat_rects(ax, configs_root, alpha=0.5)

    vmin, vmax = np.percentile(heights, 5), np.percentile(heights, 95)
    sc = ax.scatter(
        foot_world[:, 0], foot_world[:, 1],
        c=heights, cmap="viridis", s=3, alpha=0.5, vmin=vmin, vmax=vmax,
    )
    plt.colorbar(sc, ax=ax, label="BBox Height (px)")

    # Contour from affine offset model
    bounds = mat_poly.bounds
    gx = np.linspace(bounds[0] - 1, bounds[2] + 1, 100)
    gy = np.linspace(bounds[1] - 1, bounds[3] + 1, 100)
    GX, GY = np.meshgrid(gx, gy)
    grid_flat = np.column_stack([GX.ravel(), GY.ravel()])

    GZ, _, _, _ = compute_pixel_height(
        geom["H_mat2img"], offset_affine, grid_flat,
        geom["img_w"], geom["img_h"],
    )
    GZ = GZ.reshape(GX.shape)

    import shapely

    mat_mask = shapely.contains(mat_poly, shapely.points(GX.ravel(), GY.ravel())).reshape(GX.shape)
    GZ_masked = np.where(mat_mask, GZ, np.nan)

    cs = ax.contour(GX, GY, GZ_masked, levels=10, colors="black", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    reproj = metrics["reproj_error_mean_px"]
    n_pts = metrics["n_data_points"]
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"{cam_id} — Affine Offset Height Surface (reproj={reproj:.1f}px, n={n_pts})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = analysis_dir / f"{cam_id}_height_surface.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: ROI Mask Construction
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase2(
    outputs_root: Path,
    gym_id: str,
    cameras: List[str],
    reference_height_m: Optional[float],
    configs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Phase 2: Build perspective-aware ROI mask per camera using two homographies."""
    console.print("\n[bold cyan]═══ Phase 2: ROI Mask Construction ═══[/bold cyan]")

    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mat_poly = load_mat_blueprint(configs_root)
    perimeter_pts = sample_perimeter(mat_poly, PERIMETER_SAMPLE_SPACING_M)

    results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")

        geom = load_camera_geometry(cam_id, configs_root)
        H = geom["H_mat2img"]
        img_w, img_h = geom["img_w"], geom["img_h"]

        # Verify H projection accuracy
        verif = verify_h_projection(geom)
        if verif.get("verified"):
            console.print(
                f"  H projection error: mean={verif['mean_error_px']:.1f}px, "
                f"max={verif['max_error_px']:.1f}px"
            )

        # Load affine offset model from Phase 1
        offset_affine = _load_offset_affine(cam_id, configs_root)
        if offset_affine is None:
            console.print("  [red]No offset model — run Phase 1 first[/red]")
            continue

        # Project feet via H, heads via affine offset
        foot_px = project_world_to_pixel(H, perimeter_pts)
        feat = np.column_stack([perimeter_pts, np.ones(len(perimeter_pts))])
        head_px = foot_px + feat @ offset_affine.T

        # Filter to points where the foot projects inside (or near) the frame
        in_frame = (
            (foot_px[:, 0] >= -50) & (foot_px[:, 0] < img_w + 50)
            & (foot_px[:, 1] >= -50) & (foot_px[:, 1] < img_h + 50)
        )
        foot_px = foot_px[in_frame]
        head_px = head_px[in_frame]
        peri_pts_vis = perimeter_pts[in_frame]
        console.print(f"  Perimeter points in frame: {in_frame.sum()}/{len(in_frame)}")

        if len(foot_px) < 4:
            console.print("  [red]Too few in-frame perimeter points, skipping[/red]")
            continue

        # Safety margin: extend 10% beyond head in foot->head direction
        direction = head_px - foot_px
        head_margin_px = head_px + direction * ROI_HEAD_BUFFER_FRAC

        # Clip to frame bounds
        head_margin_clipped = head_margin_px.copy()
        head_margin_clipped[:, 0] = np.clip(head_margin_clipped[:, 0], 0, img_w - 1)
        head_margin_clipped[:, 1] = np.clip(head_margin_clipped[:, 1], 0, img_h - 1)
        foot_clipped = foot_px.copy()
        foot_clipped[:, 0] = np.clip(foot_clipped[:, 0], 0, img_w - 1)
        foot_clipped[:, 1] = np.clip(foot_clipped[:, 1], 0, img_h - 1)

        # Build ROI polygon: band from foot perimeter forward to head+margin, reversed to close
        roi_boundary = np.vstack([foot_clipped, head_margin_clipped[::-1]])
        roi_poly = Polygon(roi_boundary.tolist())
        if not roi_poly.is_valid:
            roi_poly = roi_poly.buffer(0)
        frame_bounds = box(0, 0, img_w, img_h)
        roi_poly = roi_poly.intersection(frame_bounds)

        if roi_poly.is_empty:
            console.print("  [red]ROI polygon is empty, skipping[/red]")
            continue

        if isinstance(roi_poly, MultiPolygon):
            roi_poly = max(roi_poly.geoms, key=lambda g: g.area)

        outer_coords = np.array(roi_poly.exterior.coords)

        # Rasterize binary mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [outer_coords.astype(np.int32)], 255)

        # Bounding rect
        x_min = max(0, int(outer_coords[:, 0].min()))
        y_min = max(0, int(outer_coords[:, 1].min()))
        x_max = min(img_w, int(outer_coords[:, 0].max()))
        y_max = min(img_h, int(outer_coords[:, 1].max()))

        # Save outputs
        cam_config_dir = configs_root / "cameras" / cam_id
        cam_config_dir.mkdir(parents=True, exist_ok=True)

        mask_path = cam_config_dir / "roi_mask.png"
        cv2.imwrite(str(mask_path), mask)

        roi_data: Dict[str, Any] = {
            "camera_id": cam_id,
            "image_size": [img_w, img_h],
            "outer_boundary_px": outer_coords.tolist(),
            "inner_boundary_px": foot_clipped.tolist(),
            "bounding_rect": [x_min, y_min, x_max - x_min, y_max - y_min],
            "head_buffer_frac": ROI_HEAD_BUFFER_FRAC,
            "method": "two_homography",
            "h_projection_error": verif,
        }

        roi_json_path = cam_config_dir / "roi_mask.json"
        roi_json_path.write_text(json.dumps(roi_data, indent=2))
        console.print(f"  Saved: {mask_path}")
        console.print(f"  Saved: {roi_json_path}")
        console.print(f"  ROI bounding rect: {roi_data['bounding_rect']}")

        _plot_phase2(
            cam_id, geom, foot_px, head_px, head_margin_clipped,
            outer_coords, foot_clipped, mask,
            outputs_root, gym_id, analysis_dir,
        )
        results[cam_id] = roi_data

    return results


def _plot_phase2(
    cam_id: str,
    geom: Dict,
    foot_px: np.ndarray,
    head_px: np.ndarray,
    head_margin_px: np.ndarray,
    outer_coords: np.ndarray,
    foot_clipped: np.ndarray,
    mask: np.ndarray,
    outputs_root: Path,
    gym_id: str,
    analysis_dir: Path,
) -> None:
    img_w, img_h = geom["img_w"], geom["img_h"]

    frame = find_camera_frame(outputs_root, gym_id, cam_id)
    title_suffix = ""
    if frame is not None and geom["K"] is not None and geom["D"] is not None:
        frame = cv2.undistort(frame, geom["K"], geom["D"])
        title_suffix = ", undistorted"
    elif frame is None:
        frame = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), alpha=0.7)

    # ROI mask overlay (green)
    mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    mask_overlay[mask > 0] = [0, 1, 0, 0.2]
    ax.imshow(mask_overlay)

    # Mat edge: foot_clipped traces the L-shape perimeter (ordered, no convex hull)
    ax.plot(foot_clipped[:, 0], foot_clipped[:, 1], "b-", linewidth=2, label="Mat edge (foot)")
    # Head boundary with margin
    ax.plot(outer_coords[:, 0], outer_coords[:, 1], "r-", linewidth=2, label="Head boundary + margin")

    # Height vectors (yellow arrows) foot -> head
    step = max(1, len(foot_px) // 40)
    for i in range(0, len(foot_px), step):
        fx, fy = foot_px[i]
        hx, hy = head_px[i]
        if 0 <= fx < img_w and 0 <= fy < img_h:
            ax.annotate(
                "",
                xy=(hx, hy),
                xytext=(fx, fy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
            )

    # Verify correspondences (cyan = actual, magenta = H projection)
    payload = geom["payload"]
    corr = payload.get("correspondences", {})
    ip = corr.get("image_points_px", [])
    mp = corr.get("mat_points", [])
    if ip and mp:
        ip_arr = np.array(ip)
        proj = project_world_to_pixel(geom["H_mat2img"], np.array(mp))
        ax.scatter(ip_arr[:, 0], ip_arr[:, 1], c="cyan", s=80, marker="x", linewidths=2, label="Corr. (actual)", zorder=5)
        ax.scatter(proj[:, 0], proj[:, 1], c="magenta", s=80, marker="+", linewidths=2, label="Corr. (H proj.)", zorder=5)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title(f"{cam_id} — ROI Mask ({img_w}x{img_h}{title_suffix})")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = analysis_dir / f"{cam_id}_roi_mask.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Detectability Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_sahi_tile_scale(
    eff_w: int, eff_h: int, grid: List[int], overlap: float, imgsz: int
) -> Tuple[float, int]:
    """Compute effective scale for a SAHI tile configuration."""
    cols, rows = grid
    tile_w = eff_w / (1 + (cols - 1) * (1 - overlap)) if cols > 1 else eff_w
    tile_h = eff_h / (1 + (rows - 1) * (1 - overlap)) if rows > 1 else eff_h
    scale = imgsz / max(tile_w, tile_h)
    return scale, cols * rows


def run_phase3(
    cameras: List[str],
    detection_floor_px: float,
    configs_root: Path,
    outputs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Phase 3: Detectability analysis using two-homography model + kneeling scalar."""
    console.print("\n[bold cyan]═══ Phase 3: Detectability Analysis ═══[/bold cyan]")

    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mat_poly = load_mat_blueprint(configs_root)
    grid_pts_mat, _ = _mat_grid(mat_poly)
    gx_mat, gy_mat = grid_pts_mat[:, 0], grid_pts_mat[:, 1]
    console.print(f"  Mat grid: {len(grid_pts_mat)} points")
    console.print(f"  Kneeling fraction: {KNEELING_FRACTION} (scores use kneeling height)")

    all_results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")

        geom = load_camera_geometry(cam_id, configs_root)
        img_w, img_h = geom["img_w"], geom["img_h"]

        offset_affine = _load_offset_affine(cam_id, configs_root)
        if offset_affine is None:
            console.print("  [red]No offset model — run Phase 1 first[/red]")
            continue

        roi_path = configs_root / "cameras" / cam_id / "roi_mask.json"
        roi_data = json.loads(roi_path.read_text()) if roi_path.exists() else None

        # Compute standing pixel heights and in-frame mask
        standing_heights, in_frame, _, _ = compute_pixel_height(
            geom["H_mat2img"], offset_affine, grid_pts_mat, img_w, img_h,
        )
        kneeling_heights = standing_heights * KNEELING_FRACTION
        n_in_frame = int(in_frame.sum())
        console.print(f"  In-frame grid points: {n_in_frame}/{len(grid_pts_mat)}")

        config_results: List[Dict[str, Any]] = []

        for cfg in DETECTABILITY_CONFIGS:
            name = cfg["name"]
            imgsz = cfg["imgsz"]
            roi_crop = cfg["roi_crop"]
            sahi = cfg["sahi"]

            if roi_crop and roi_data:
                br = roi_data["bounding_rect"]
                eff_w, eff_h = br[2], br[3]
            else:
                eff_w, eff_h = img_w, img_h

            if sahi is not None:
                scale, n_tiles = _compute_sahi_tile_scale(
                    eff_w, eff_h, sahi["grid"], sahi["overlap"], imgsz
                )
                speed_mult = n_tiles * (imgsz / 640) ** 2
            else:
                scale = imgsz / max(eff_w, eff_h)
                speed_mult = (imgsz / 640) ** 2

            # Score based on kneeling height (conservative: if kneeling is detectable,
            # standing is definitely detectable)
            effective_kneeling = kneeling_heights * scale
            scores = effective_kneeling / detection_floor_px
            # Zero out off-screen points
            scores[~in_frame] = 0.0

            aspect = eff_w / eff_h if eff_h > 0 else 1.0
            padding_waste = 1.0 - min(aspect, 1.0 / aspect) if aspect > 0 else 0.0

            # visible_coverage: fraction of in-frame points with score > 1.0
            vis_cov = (
                round(float(np.mean(scores[in_frame] > 1.0)) * 100, 1)
                if n_in_frame > 0
                else 0.0
            )
            vis_scores = scores[in_frame] if n_in_frame > 0 else scores
            # min/mean over standing effective heights for in-frame points
            eff_standing = standing_heights * scale
            vis_standing = eff_standing[in_frame] if n_in_frame > 0 else eff_standing

            result: Dict[str, Any] = {
                "name": name,
                "imgsz": imgsz,
                "roi_crop": roi_crop,
                "sahi": sahi,
                "mat_coverage_pct": round(float(np.mean(scores > 1.0)) * 100, 1),
                "visible_coverage_pct": vis_cov,
                "n_in_frame": n_in_frame,
                "kneeling_fraction": KNEELING_FRACTION,
                "min_effective_px": round(float(vis_standing.min()), 1) if len(vis_standing) > 0 else 0,
                "mean_effective_px": round(float(vis_standing.mean()), 1) if len(vis_standing) > 0 else 0,
                "padding_waste_pct": round(padding_waste * 100, 1),
                "speed_relative": round(speed_mult, 2),
                "scores": scores.tolist(),
            }
            config_results.append(result)

        # Print table
        table = Table(title=f"Camera: {cam_id} ({img_w}x{img_h}, {n_in_frame}/{len(gx_mat)} in-frame)")
        table.add_column("Configuration", style="cyan")
        table.add_column("Vis cov%", justify="right")
        table.add_column("Min px_h", justify="right")
        table.add_column("Mean px_h", justify="right")
        table.add_column("Padding%", justify="right")
        table.add_column("Speed", justify="right")

        for r in config_results:
            table.add_row(
                r["name"],
                f"{r['visible_coverage_pct']:.0f}%",
                f"{r['min_effective_px']:.0f}px",
                f"{r['mean_effective_px']:.0f}px",
                f"{r['padding_waste_pct']:.0f}%",
                f"{r['speed_relative']:.1f}x",
            )
        console.print(table)

        _plot_phase3(cam_id, gx_mat, gy_mat, config_results, in_frame, configs_root, analysis_dir)

        report: Dict[str, Any] = {
            "camera_id": cam_id,
            "image_size": [img_w, img_h],
            "detection_floor_px": detection_floor_px,
            "kneeling_fraction": KNEELING_FRACTION,
            "grid_spacing_m": MAT_GRID_SPACING_M,
            "n_grid_points": len(gx_mat),
            "n_in_frame": n_in_frame,
            "configurations": [
                {k: v for k, v in r.items() if k != "scores"}
                for r in config_results
            ],
            "per_config_scores": {r["name"]: r["scores"] for r in config_results},
        }
        report_path = analysis_dir / f"{cam_id}_detectability_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        console.print(f"  Report: {report_path}")

        all_results[cam_id] = {
            "config_results": config_results,
            "grid_x": gx_mat.tolist(),
            "grid_y": gy_mat.tolist(),
        }

    return all_results


def _plot_phase3(
    cam_id: str,
    gx: np.ndarray,
    gy: np.ndarray,
    config_results: List[Dict],
    in_frame: np.ndarray,
    configs_root: Path,
    analysis_dir: Path,
) -> None:
    n_configs = len(config_results)
    ncols = 4
    nrows = max(1, (n_configs + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    cmap = plt.cm.RdYlGn
    boundaries = [0, 0.5, 0.7, 1.0, 1.5, 3.0]
    norm = BoundaryNorm(boundaries, cmap.N)

    out_of_frame = ~in_frame

    for idx, result in enumerate(config_results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        scores = np.array(result["scores"])
        _draw_mat_rects(ax, configs_root, alpha=0.2)
        # In-frame points: full opacity
        ax.scatter(gx[in_frame], gy[in_frame], c=scores[in_frame], cmap=cmap, norm=norm, s=8, marker="s")
        # Out-of-frame points: reduced opacity (camera cannot see these)
        if out_of_frame.any():
            ax.scatter(gx[out_of_frame], gy[out_of_frame], c="lightgray", s=4, marker="s", alpha=0.3)

        cov = result["visible_coverage_pct"]
        spd = result["speed_relative"]
        ax.set_title(f"{result['name']}\nvis_cov={cov:.0f}% spd={spd:.1f}x", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for idx in range(n_configs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Detectability Score (kneeling)")

    fig.suptitle(f"{cam_id} — Detectability by Configuration", fontsize=14, y=1.01)
    out_path = analysis_dir / f"{cam_id}_detectability_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Coverage-Aware ROI Optimization
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase4(
    cameras: List[str],
    detection_floor_px: float,
    overlap_buffer_m: float,
    min_coverage_override: Optional[int],
    configs_root: Path,
    outputs_root: Path,
    reference_height_m: Optional[float],
    gym_id: str = "",
) -> Dict[str, Any]:
    """Phase 4: Multi-camera coverage optimization."""
    console.print("\n[bold cyan]═══ Phase 4: Coverage-Aware ROI Optimization ═══[/bold cyan]")

    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mat_poly = load_mat_blueprint(configs_root)
    mat_rects = load_mat_rectangles(configs_root)
    grid_pts_mat, _ = _mat_grid(mat_poly)
    n_pts = len(grid_pts_mat)

    console.print(f"  Mat grid: {n_pts} points, {len(cameras)} cameras")

    if len(cameras) < 2:
        console.print("  [yellow]Single camera — no zone restriction possible[/yellow]")
        report_path = analysis_dir / f"{cameras[0]}_detectability_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            for cfg in report.get("configurations", []):
                if cfg.get("visible_coverage_pct", 0) >= 95:
                    console.print(f"  Recommended: {cfg['name']} ({cfg['visible_coverage_pct']:.0f}% coverage)")
                    break
        return {"single_camera": True, "cameras": cameras}

    # Load Phase 3 baseline (640_full) scores per camera
    coverage_matrix: Dict[str, np.ndarray] = {}
    cam_geoms: Dict[str, Dict] = {}

    for cam_id in cameras:
        report_path = analysis_dir / f"{cam_id}_detectability_report.json"
        if not report_path.exists():
            console.print(f"  [red]No Phase 3 report for {cam_id} — run Phase 3 first[/red]")
            continue
        report = json.loads(report_path.read_text())
        scores = report.get("per_config_scores", {}).get("640_full", [])
        if scores:
            coverage_matrix[cam_id] = np.array(scores)
        cam_geoms[cam_id] = load_camera_geometry(cam_id, configs_root)

    if len(coverage_matrix) < 2:
        console.print("  [red]Need Phase 3 data for at least 2 cameras[/red]")
        return {}

    cam_list = list(coverage_matrix.keys())
    det_threshold = 0.7

    coverage_count = np.zeros(n_pts)
    for scores in coverage_matrix.values():
        coverage_count += (scores > det_threshold).astype(float)

    actual_min = int(coverage_count.min())
    min_coverage = min_coverage_override if min_coverage_override is not None else min(2, actual_min)

    console.print(f"  Min coverage at any point: {actual_min} cameras")
    console.print(f"  Coverage constraint: {min_coverage}")

    single_cov = (coverage_count == 1).sum()
    if single_cov > 0:
        console.print(f"  [yellow]Warning: {single_cov} points have only 1 camera[/yellow]")
    zero_cov = (coverage_count == 0).sum()
    if zero_cov > 0:
        console.print(f"  [red]Warning: {zero_cov} points have 0 coverage![/red]")

    score_matrix = np.column_stack([coverage_matrix[c] for c in cam_list])
    primary_cam_idx = np.argmax(score_matrix, axis=1)

    restricted_zones: Dict[str, np.ndarray] = {}

    for c_idx, cam_id in enumerate(cam_list):
        cam_scores = coverage_matrix[cam_id]
        include = (primary_cam_idx == c_idx).copy()

        for i in range(n_pts):
            if include[i] or cam_scores[i] <= det_threshold:
                continue
            other_covering = sum(
                1 for oc in cam_list if oc != cam_id and coverage_matrix[oc][i] > det_threshold
            )
            if other_covering < min_coverage:
                include[i] = True

        included_pts = grid_pts_mat[include]
        if len(included_pts) > 0:
            for i in range(n_pts):
                if include[i] or cam_scores[i] <= det_threshold:
                    continue
                dists = np.sqrt(
                    (grid_pts_mat[i, 0] - included_pts[:, 0]) ** 2
                    + (grid_pts_mat[i, 1] - included_pts[:, 1]) ** 2
                )
                if dists.min() <= overlap_buffer_m:
                    include[i] = True

        restricted_zones[cam_id] = include
        n_included = int(include.sum())
        console.print(f"  {cam_id}: {n_included}/{n_pts} points in restricted zone")
        if n_included == 0:
            console.print(
                f"  [yellow]{cam_id} is fully redundant at min_coverage={min_coverage}. "
                f"Consider --min-coverage {min_coverage + 1} to require redundancy.[/yellow]"
            )

    # Validate coverage constraint
    coverage_check = np.zeros(n_pts)
    for cam_id in cam_list:
        coverage_check += (restricted_zones[cam_id] & (coverage_matrix[cam_id] > det_threshold)).astype(float)

    violations = (coverage_check < min_coverage).sum()
    if violations > 0:
        console.print(f"  [yellow]Fixing {violations} coverage violations...[/yellow]")
        for i in range(n_pts):
            if coverage_check[i] >= min_coverage:
                continue
            for cam_id in cam_list:
                if not restricted_zones[cam_id][i] and coverage_matrix[cam_id][i] > det_threshold:
                    restricted_zones[cam_id][i] = True
                    coverage_check[i] += 1
                    if coverage_check[i] >= min_coverage:
                        break
        remaining = (coverage_check < min_coverage).sum()
        console.print(f"  After fix: {remaining} remaining violations")

    # Compute optimal imgsz per camera
    per_camera_results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cam_list:
        geom = cam_geoms[cam_id]
        H = geom["H_mat2img"]
        img_w, img_h = geom["img_w"], geom["img_h"]

        zone = restricted_zones[cam_id]
        zone_pts = grid_pts_mat[zone]
        if len(zone_pts) == 0:
            continue

        offset_affine = _load_offset_affine(cam_id, configs_root)

        # Project zone to pixel space for bounding rect
        zone_foot_px = project_world_to_pixel(H, zone_pts)
        if offset_affine is not None:
            zone_feat = np.column_stack([zone_pts, np.ones(len(zone_pts))])
            zone_offset = zone_feat @ offset_affine.T
            zone_head_px = zone_foot_px + zone_offset
            direction = zone_head_px - zone_foot_px
            zone_head_margin = zone_head_px + direction * ROI_HEAD_BUFFER_FRAC
        else:
            zone_head_margin = zone_foot_px.copy()
            zone_head_margin[:, 1] -= 100

        all_pts = np.vstack([zone_foot_px, zone_head_margin])
        all_pts[:, 0] = np.clip(all_pts[:, 0], 0, img_w - 1)
        all_pts[:, 1] = np.clip(all_pts[:, 1], 0, img_h - 1)

        x_min = max(0, int(all_pts[:, 0].min()) - 10)
        y_min = max(0, int(all_pts[:, 1].min()) - 10)
        x_max = min(img_w, int(all_pts[:, 0].max()) + 10)
        y_max = min(img_h, int(all_pts[:, 1].max()) + 10)
        restricted_w = x_max - x_min
        restricted_h = y_max - y_min

        # Compute heights for zone points via affine offset model
        if offset_affine is not None:
            zone_standing, zone_in_frame, _, _ = compute_pixel_height(
                H, offset_affine, zone_pts, img_w, img_h,
            )
        else:
            zone_standing = np.full(len(zone_pts), 50.0)
            zone_in_frame = np.ones(len(zone_pts), dtype=bool)

        zone_kneeling = zone_standing * KNEELING_FRACTION

        # Optimal imgsz: 95% of in-frame zone points have kneeling score > 1.0
        optimal_imgsz = 1536
        optimal_coverage = 0.0
        for test_imgsz in [640, 800, 960, 1280, 1536]:
            scale = test_imgsz / max(restricted_w, restricted_h)
            eff_k = zone_kneeling * scale
            if zone_in_frame.any():
                cov = float(np.mean(eff_k[zone_in_frame] / detection_floor_px > 1.0)) * 100
            else:
                cov = float(np.mean(eff_k / detection_floor_px > 1.0)) * 100
            if cov >= 95 and optimal_imgsz == 1536:
                optimal_imgsz = test_imgsz
                optimal_coverage = cov
        if optimal_coverage == 0:
            scale = optimal_imgsz / max(restricted_w, restricted_h)
            eff_k = zone_kneeling * scale
            if zone_in_frame.any():
                optimal_coverage = float(np.mean(eff_k[zone_in_frame] / detection_floor_px > 1.0)) * 100
            else:
                optimal_coverage = float(np.mean(eff_k / detection_floor_px > 1.0)) * 100

        # Full-frame reference imgsz
        full_roi_imgsz = 1536
        for test_imgsz in [640, 800, 960, 1280, 1536]:
            scale = test_imgsz / max(img_w, img_h)
            eff_k = zone_kneeling * scale
            if zone_in_frame.any():
                cov = float(np.mean(eff_k[zone_in_frame] / detection_floor_px > 1.0)) * 100
            else:
                cov = float(np.mean(eff_k / detection_floor_px > 1.0)) * 100
            if cov >= 95:
                full_roi_imgsz = test_imgsz
                break

        speed_restricted = (optimal_imgsz / 640) ** 2
        speed_full = (full_roi_imgsz / 640) ** 2

        aspect = restricted_w / restricted_h if restricted_h > 0 else 1.0
        padding_waste = 1.0 - min(aspect, 1.0 / aspect) if aspect > 0 else 0.0

        vis_standing = zone_standing[zone_in_frame] if zone_in_frame.any() else zone_standing
        min_eff_px = float(vis_standing.min() * (optimal_imgsz / max(restricted_w, restricted_h))) if len(vis_standing) > 0 else 0

        # Save restricted ROI mask
        cam_config_dir = configs_root / "cameras" / cam_id
        restricted_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if len(all_pts) >= 3:
            hull = MultiPoint(all_pts.tolist()).convex_hull
            hull = hull.intersection(box(0, 0, img_w, img_h))
            if not hull.is_empty:
                if isinstance(hull, MultiPolygon):
                    hull = max(hull.geoms, key=lambda g: g.area)
                if hasattr(hull, "exterior"):
                    hull_coords = np.array(hull.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(restricted_mask, [hull_coords], 255)
                    cv2.imwrite(str(cam_config_dir / "roi_mask_restricted.png"), restricted_mask)

        det_config: Dict[str, Any] = {
            "camera_id": cam_id,
            "roi_mask": "roi_mask_restricted.png",
            "roi_bounding_rect": [x_min, y_min, restricted_w, restricted_h],
            "imgsz": optimal_imgsz,
            "sahi": None,
            "coverage_zone_description": f"restricted_zone_{cam_id}",
            "effective_min_person_px": round(min_eff_px, 1),
            "effective_coverage_pct": round(optimal_coverage, 1),
            "aspect_ratio": round(aspect, 2),
            "padding_waste_pct": round(padding_waste * 100, 1),
            "generated_at": pd.Timestamp.now().isoformat(),
            "coverage_constraint": f"min_{min_coverage}_cameras",
            "phase4_version": "2.0",
        }
        (cam_config_dir / "detection_config.json").write_text(json.dumps(det_config, indent=2))

        per_camera_results[cam_id] = {
            "full_roi_imgsz": full_roi_imgsz,
            "restricted_shape": f"{restricted_w}x{restricted_h}",
            "optimal_imgsz": optimal_imgsz,
            "coverage_pct": round(optimal_coverage, 1),
            "speed_full": round(speed_full, 1),
            "speed_restricted": round(speed_restricted, 1),
            "restricted_mask": restricted_mask,
        }

    # ── Plots ──

    _plot_phase4_coverage(
        grid_pts_mat, coverage_count, restricted_zones, cam_list,
        mat_rects, analysis_dir, configs_root,
    )
    for cam_id in cam_list:
        if cam_id in per_camera_results:
            _plot_phase4_camera(
                cam_id, cam_geoms[cam_id], per_camera_results[cam_id],
                outputs_root, gym_id, configs_root, analysis_dir,
            )

    # ── Summary table ──

    table = Table(title="System-Level Coverage Analysis")
    table.add_column("Camera")
    table.add_column("Full ROI imgsz@95%", justify="right")
    table.add_column("Restricted shape", justify="right")
    table.add_column("Optimal imgsz", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Speed vs full@640", justify="right")

    total_full = 0.0
    total_restricted = 0.0
    for cam_id in cam_list:
        if cam_id not in per_camera_results:
            continue
        r = per_camera_results[cam_id]
        total_full += r["speed_full"]
        total_restricted += r["speed_restricted"]
        table.add_row(
            cam_id,
            f"{r['full_roi_imgsz']} ({r['speed_full']:.1f}x)",
            r["restricted_shape"],
            str(r["optimal_imgsz"]),
            f"{r['coverage_pct']:.0f}%",
            f"{r['speed_restricted']:.1f}x",
        )

    table.add_section()
    speedup = total_full / total_restricted if total_restricted > 0 else 1
    table.add_row("TOTAL", f"{total_full:.1f}x", "", "", "", f"{total_restricted:.1f}x")
    console.print(table)
    console.print(f"  System speedup from zone restriction: {speedup:.1f}x faster")

    report_out: Dict[str, Any] = {
        "cameras": cam_list,
        "n_grid_points": n_pts,
        "min_coverage_constraint": min_coverage,
        "actual_min_coverage": actual_min,
        "total_speed_full": round(total_full, 2),
        "total_speed_restricted": round(total_restricted, 2),
        "speedup": round(speedup, 2),
        "per_camera": {
            cam_id: {k: v for k, v in r.items() if k != "restricted_mask"}
            for cam_id, r in per_camera_results.items()
        },
    }
    report_path = analysis_dir / "coverage_optimization_report.json"
    report_path.write_text(json.dumps(report_out, indent=2, default=str))
    console.print(f"  Report: {report_path}")

    return report_out


def _plot_phase4_coverage(
    grid_pts: np.ndarray,
    coverage_count: np.ndarray,
    restricted_zones: Dict[str, np.ndarray],
    cam_list: List[str],
    mat_rects: List[Dict],
    analysis_dir: Path,
    configs_root: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _draw_mat_rects(ax, configs_root, alpha=0.3)

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=max(3, coverage_count.max()))
    sc = ax.scatter(
        grid_pts[:, 0], grid_pts[:, 1],
        c=coverage_count, cmap=cmap, norm=norm, s=10, marker="s",
    )
    plt.colorbar(sc, ax=ax, label="Camera Coverage Count")

    colors = ["blue", "red", "green", "orange", "purple"]
    for c_idx, cam_id in enumerate(cam_list):
        zone = restricted_zones[cam_id]
        zone_pts = grid_pts[zone]
        color = colors[c_idx % len(colors)]
        n_zone = int(zone.sum())
        if n_zone < 3:
            ax.plot(
                [], [], color=color, linewidth=2, linestyle="--",
                label=f"{cam_id} (0 pts — redundant)",
            )
            continue
        hull = MultiPoint(zone_pts.tolist()).convex_hull
        if hasattr(hull, "exterior"):
            coords = np.array(hull.exterior.coords)
            ax.plot(
                coords[:, 0], coords[:, 1],
                color=color,
                linewidth=2, linestyle="--", label=f"{cam_id} ({n_zone} pts)",
            )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Multi-Camera Coverage Map")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = analysis_dir / "coverage_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {out_path}")


def _plot_phase4_camera(
    cam_id: str,
    geom: Dict,
    results: Dict,
    outputs_root: Path,
    gym_id: str,
    configs_root: Path,
    analysis_dir: Path,
) -> None:
    img_w, img_h = geom["img_w"], geom["img_h"]

    frame = find_camera_frame(outputs_root, gym_id, cam_id)
    title_suffix = ""
    if frame is not None and geom["K"] is not None and geom["D"] is not None:
        frame = cv2.undistort(frame, geom["K"], geom["D"])
        title_suffix = ", undistorted"
    elif frame is None:
        frame = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), alpha=0.7)

    roi_path = configs_root / "cameras" / cam_id / "roi_mask.json"
    if roi_path.exists():
        roi_data = json.loads(roi_path.read_text())
        outer = np.array(roi_data["outer_boundary_px"])
        ax.plot(outer[:, 0], outer[:, 1], "w--", linewidth=1.5, alpha=0.7, label="Full ROI")

    restricted_mask = results.get("restricted_mask")
    if restricted_mask is not None:
        overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
        overlay[restricted_mask > 0] = [0.2, 0.6, 1.0, 0.3]
        ax.imshow(overlay)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title(
        f"{cam_id} — Restricted ROI "
        f"(imgsz={results['optimal_imgsz']}, cov={results['coverage_pct']:.0f}%{title_suffix})"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = analysis_dir / f"{cam_id}_restricted_roi.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

app = typer.Typer(
    help="Camera geometry analysis — height surfaces, ROI masks, detectability, coverage.",
    add_completion=False,
)


def _common(outputs: Path, gym_id: str, configs_root: Path) -> Tuple[Path, str, List[str], Path]:
    outputs = Path(outputs)
    configs_root = Path(configs_root)
    cameras = discover_cameras(outputs, gym_id)
    console.print(f"Cameras: {', '.join(cameras)}")
    return outputs, gym_id, cameras, configs_root


@app.command("phase1")
def cmd_phase1(
    outputs: Path = typer.Option(..., help="Outputs root directory"),
    gym_id: str = typer.Option(..., "--gym-id", help="Gym ID (UUID)"),
    reference_height_m: Optional[float] = typer.Option(None, help="Known person height for calibration"),
    configs_root: Path = typer.Option(Path("configs"), help="Configs root"),
) -> None:
    """Phase 1: Fit head-plane homography per camera from standing detections."""
    outputs, gym_id, cameras, configs_root = _common(outputs, gym_id, configs_root)
    run_phase1(outputs, gym_id, cameras, reference_height_m, configs_root)


@app.command("phase2")
def cmd_phase2(
    outputs: Path = typer.Option(..., help="Outputs root directory"),
    gym_id: str = typer.Option(..., "--gym-id", help="Gym ID"),
    reference_height_m: Optional[float] = typer.Option(None, help="Person height in meters"),
    configs_root: Path = typer.Option(Path("configs"), help="Configs root"),
) -> None:
    """Phase 2: Build perspective-aware ROI mask per camera."""
    outputs, gym_id, cameras, configs_root = _common(outputs, gym_id, configs_root)
    run_phase2(outputs, gym_id, cameras, reference_height_m, configs_root)


@app.command("phase3")
def cmd_phase3(
    outputs: Path = typer.Option(..., help="Outputs root directory"),
    gym_id: str = typer.Option(..., "--gym-id", help="Gym ID"),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX, help="Detection floor in pixels"),
    configs_root: Path = typer.Option(Path("configs"), help="Configs root"),
) -> None:
    """Phase 3: Detectability analysis for candidate YOLO configurations."""
    outputs, gym_id, cameras, configs_root = _common(outputs, gym_id, configs_root)
    run_phase3(cameras, detection_floor_px, configs_root, outputs)


@app.command("phase4")
def cmd_phase4(
    outputs: Path = typer.Option(..., help="Outputs root directory"),
    gym_id: str = typer.Option(..., "--gym-id", help="Gym ID"),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX, help="Detection floor in pixels"),
    overlap_buffer_m: float = typer.Option(DEFAULT_OVERLAP_BUFFER_M, help="Overlap buffer in meters"),
    min_coverage: Optional[int] = typer.Option(None, help="Override min coverage constraint"),
    reference_height_m: Optional[float] = typer.Option(None, help="Person height in meters"),
    configs_root: Path = typer.Option(Path("configs"), help="Configs root"),
) -> None:
    """Phase 4: Multi-camera coverage-aware ROI optimization."""
    outputs, gym_id, cameras, configs_root = _common(outputs, gym_id, configs_root)
    run_phase4(
        cameras, detection_floor_px, overlap_buffer_m,
        min_coverage, configs_root, outputs, reference_height_m, gym_id,
    )


@app.command("all")
def cmd_all(
    outputs: Path = typer.Option(..., help="Outputs root directory"),
    gym_id: str = typer.Option(..., "--gym-id", help="Gym ID"),
    reference_height_m: Optional[float] = typer.Option(None, help="Person height for calibration"),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX, help="Detection floor in pixels"),
    overlap_buffer_m: float = typer.Option(DEFAULT_OVERLAP_BUFFER_M, help="Overlap buffer in meters"),
    min_coverage: Optional[int] = typer.Option(None, help="Override min coverage constraint"),
    configs_root: Path = typer.Option(Path("configs"), help="Configs root"),
) -> None:
    """Run all four phases sequentially."""
    outputs, gym_id, cameras, configs_root = _common(outputs, gym_id, configs_root)

    run_phase1(outputs, gym_id, cameras, reference_height_m, configs_root)
    run_phase2(outputs, gym_id, cameras, reference_height_m, configs_root)
    run_phase3(cameras, detection_floor_px, configs_root, outputs)
    run_phase4(
        cameras, detection_floor_px, overlap_buffer_m,
        min_coverage, configs_root, outputs, reference_height_m, gym_id,
    )

    console.print("\n[bold green]All phases complete.[/bold green]")


if __name__ == "__main__":
    app()
