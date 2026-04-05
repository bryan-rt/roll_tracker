"""Camera geometry analysis — height surfaces, ROI masks, detectability, coverage optimization.

Standalone diagnostic tool for the gym setup wizard flow.  Analyzes camera
geometry using existing pipeline outputs to produce detectability heatmaps
and optimized per-camera configurations.

Usage:
    python tools/camera_geometry_analysis.py all --outputs outputs \
        --gym-id c8a592a4-2bca-400a-80e1-fec0e5cbea77
    python tools/camera_geometry_analysis.py phase1 --outputs outputs --gym-id <uuid>

Pose decomposition model (v6):
    H maps mat world coords to undistorted pixels.  K is the intrinsic matrix.
    M = K^-1 @ H = [r1 | r2 | t] (up to scale).  r3 = cross(r1, r2), SVD-
    orthogonalized.  P = K @ [r1 | r2 | r3 | t] is the 3x4 projection matrix.
    pixel_height at (X,Y) = ||P @ [X,Y,0,1] - P @ [X,Y,Z_person,1]||.
    No fitting, no RANSAC, no extrapolation — exact camera geometry.
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
import typer

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_PERSON_HEIGHT_M = 1.75  # average standing height
DEFAULT_DETECTION_FLOOR_PX = 45.0
DEFAULT_OVERLAP_BUFFER_M = 1.0
KNEELING_HEIGHT_M = 0.79  # kneeling ≈ 0.45 × 1.75
ROI_HEAD_BUFFER_FRAC = 0.10  # 10% buffer above head for ROI mask
PERIMETER_SAMPLE_SPACING_M = 0.1
MAT_GRID_SPACING_M = 0.25

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
    """Load camera geometry from homography.json."""
    h_path = configs_root / "cameras" / cam_id / "homography.json"
    if not h_path.exists():
        raise FileNotFoundError(f"Homography not found: {h_path}")
    payload = json.loads(h_path.read_text())

    H = np.asarray(payload["H"], dtype=np.float64).reshape(3, 3)

    K = None
    if payload.get("camera_matrix") is not None:
        K = np.asarray(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)

    D = None
    if payload.get("dist_coefficients") is not None:
        D = np.asarray(payload["dist_coefficients"], dtype=np.float64).ravel()

    img_size = None
    if "lens_calibration" in payload and payload["lens_calibration"]:
        img_size = payload["lens_calibration"].get("image_size")
    if img_size is None and "projected_polylines" in payload:
        img_size = payload["projected_polylines"].get("image_wh")

    return {
        "cam_id": cam_id,
        "H": H,
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
    return json.loads((configs_root / "mat_blueprint.json").read_text())


def load_tracklet_data(outputs_root: Path, gym_id: str, cam_id: str) -> pd.DataFrame:
    """Load and merge tracklet_frames + detections for all clips of a camera."""
    cam_dir = outputs_root / gym_id / cam_id
    tf_files = sorted(cam_dir.rglob("stage_A/tracklet_frames.parquet"))
    det_files = sorted(cam_dir.rglob("stage_A/detections.parquet"))
    if not tf_files:
        console.print(f"[yellow]No tracklet_frames found for {cam_id}[/yellow]")
        return pd.DataFrame()
    tf = pd.concat([pd.read_parquet(f) for f in tf_files], ignore_index=True)
    det = pd.concat([pd.read_parquet(f) for f in det_files], ignore_index=True)
    return tf.merge(det[["detection_id", "x1", "y1", "x2", "y2", "confidence"]], on="detection_id", how="left")


def find_camera_frame(
    outputs_root: Path, gym_id: str, cam_id: str, data_root: Path = Path("data"),
) -> Optional[np.ndarray]:
    """Try to grab a single video frame for visualization background."""
    for d in [data_root / "raw" / "nest" / gym_id / cam_id,
              data_root / "raw" / "nest" / cam_id,
              data_root / "raw" / cam_id]:
        if not d.is_dir():
            continue
        for video in sorted(d.rglob("*.mp4"))[:1]:
            cap = cv2.VideoCapture(str(video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
    return None


# ── Pose Decomposition ────────────────────────────────────────────────────────


def decompose_camera_pose(
    H: np.ndarray,
    K: np.ndarray,
    validation_world_pts: Optional[np.ndarray] = None,
    validation_img_pts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decompose world-to-image homography H into 3x4 projection matrix P.

    H maps mat world (X,Y) to undistorted pixel coords.
    M = K^-1 @ H = [r1 | r2 | t] up to scale.
    r3 = cross(r1, r2), SVD-orthogonalized for proper rotation.
    P = K @ [r1_raw | r2_raw | r3_SVD | t_raw].

    Uses raw r1/r2/t (exact for Z=0) and SVD-corrected r3 (for Z!=0).

    Returns (P_3x4, diagnostics_dict).
    """
    K_inv = np.linalg.inv(K)
    M = K_inv @ H

    scale = np.linalg.norm(M[:, 0])
    r1 = M[:, 0] / scale
    r2 = M[:, 1] / scale
    t = M[:, 2] / scale

    # Ensure points are in front of camera (positive z-depth)
    if validation_world_pts is not None and len(validation_world_pts) > 0:
        pt = np.array([validation_world_pts[0, 0], validation_world_pts[0, 1], 0.0])
        z_check = (r1 * pt[0] + r2 * pt[1] + t)[2]
        if z_check < 0:
            scale = -scale
            r1 = M[:, 0] / scale
            r2 = M[:, 1] / scale
            t = M[:, 2] / scale

    # r3 from cross product, SVD-orthogonalized
    r3_raw = np.cross(r1, r2)
    R_approx = np.column_stack([r1, r2, r3_raw])
    U, _, Vt = np.linalg.svd(R_approx)
    R_svd = U @ Vt
    if np.linalg.det(R_svd) < 0:
        U[:, -1] *= -1
        R_svd = U @ Vt
    r3 = R_svd[:, 2]

    # Build P: raw r1/r2/t for exact Z=0, SVD r3 for Z!=0
    P = K @ np.column_stack([r1, r2, r3, t])

    # Diagnostics
    diag: Dict[str, Any] = {
        "scale": round(float(scale), 6),
        "r1_norm": round(float(np.linalg.norm(r1)), 4),
        "r2_norm": round(float(np.linalg.norm(r2)), 4),
        "r1_dot_r2": round(float(np.dot(r1, r2)), 4),
    }

    if validation_world_pts is not None and validation_img_pts is not None:
        errs = []
        for i in range(len(validation_world_pts)):
            proj = P @ np.array([validation_world_pts[i, 0], validation_world_pts[i, 1], 0.0, 1.0])
            px = proj[:2] / proj[2]
            errs.append(float(np.linalg.norm(px - validation_img_pts[i])))
        diag["reproj_error_mean_px"] = round(np.mean(errs), 4)
        diag["reproj_error_max_px"] = round(np.max(errs), 4)

    return P, diag


def project_3d(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """Project Nx3 world points to pixel coords via P (3x4)."""
    n = points_3d.shape[0]
    pts_h = np.hstack([points_3d, np.ones((n, 1))])  # Nx4
    proj = (P @ pts_h.T).T  # Nx3
    w = proj[:, 2:3].copy()
    w[w == 0] = 1e-10
    return proj[:, :2] / w


def compute_pixel_height(
    P: np.ndarray,
    world_xy: np.ndarray,
    person_height_m: float,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pixel height of a person at each mat position.

    Returns (pixel_heights, in_frame, foot_px, head_px).
    in_frame: True if foot projects inside the camera frame.
    """
    foot_3d = np.column_stack([world_xy, np.zeros(len(world_xy))])
    head_3d = np.column_stack([world_xy, np.full(len(world_xy), person_height_m)])

    foot_px = project_3d(P, foot_3d)
    head_px = project_3d(P, head_3d)

    pixel_heights = np.linalg.norm(foot_px - head_px, axis=1)

    in_frame = (
        (foot_px[:, 0] >= 0) & (foot_px[:, 0] < img_w)
        & (foot_px[:, 1] >= 0) & (foot_px[:, 1] < img_h)
    )

    return pixel_heights, in_frame, foot_px, head_px


# ── Mat Helpers ────────────────────────────────────────────────────────────────


def sample_perimeter(polygon: Polygon, spacing_m: float = 0.1) -> np.ndarray:
    """Sample points along polygon perimeter at regular intervals."""
    boundary = polygon.exterior
    n_samples = max(int(boundary.length / spacing_m), 4)
    return np.array([
        [boundary.interpolate(i / n_samples, normalized=True).x,
         boundary.interpolate(i / n_samples, normalized=True).y]
        for i in range(n_samples)
    ])


def _mat_grid(mat_poly: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a dense grid inside the mat polygon."""
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
        ax.add_patch(mpatches.Rectangle(
            (m["x"], m["y"]), m["width"], m["height"],
            linewidth=1, edgecolor="gray", facecolor="#f0f0f0", alpha=alpha,
        ))


def _load_camera_pose(cam_id: str, configs_root: Path) -> Optional[np.ndarray]:
    """Load P (3x4) from Phase 1 output (height_surface.json)."""
    path = configs_root / "cameras" / cam_id / "height_surface.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    raw = data.get("P")
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float64).reshape(3, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Camera Pose Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase1(
    outputs_root: Path,
    gym_id: str,
    cameras: List[str],
    reference_height_m: Optional[float],
    configs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Phase 1: Decompose camera pose from H and K. Validate against tracklet data."""
    console.print("\n[bold cyan]═══ Phase 1: Camera Pose Decomposition ═══[/bold cyan]")

    person_h = reference_height_m or DEFAULT_PERSON_HEIGHT_M
    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")

        geom = load_camera_geometry(cam_id, configs_root)
        H, K = geom["H"], geom["K"]
        if K is None:
            console.print("  [red]No camera_matrix (K) — cannot decompose. Skipping.[/red]")
            continue

        # Correspondences for validation
        payload = geom["payload"]
        corr = payload.get("correspondences", {})
        ip = np.array(corr.get("image_points_px", []), dtype=np.float64)
        mp = np.array(corr.get("mat_points", []), dtype=np.float64)

        P, diag = decompose_camera_pose(
            H, K,
            validation_world_pts=mp if len(mp) > 0 else None,
            validation_img_pts=ip if len(ip) > 0 else None,
        )

        console.print(f"  ||r1||={diag['r1_norm']}, ||r2||={diag['r2_norm']}, r1·r2={diag['r1_dot_r2']}")
        if "reproj_error_mean_px" in diag:
            console.print(f"  Z=0 reproj: mean={diag['reproj_error_mean_px']:.4f}px")

        # Validate heights at correspondence positions
        if len(mp) > 0:
            heights, _, foot_px, head_px = compute_pixel_height(
                P, mp, person_h, geom["img_w"], geom["img_h"],
            )
            console.print(f"  Height predictions (Z={person_h}m) at correspondences:")
            for i in range(len(mp)):
                console.print(
                    f"    ({mp[i,0]:.0f},{mp[i,1]:.0f}): "
                    f"foot=({foot_px[i,0]:.0f},{foot_px[i,1]:.0f}) "
                    f"head=({head_px[i,0]:.0f},{head_px[i,1]:.0f}) "
                    f"height={heights[i]:.0f}px"
                )

        # Validate against observed bbox heights (if tracklet data exists)
        df = load_tracklet_data(outputs_root, gym_id, cam_id)
        scatter_data = []
        if not df.empty:
            df = df.dropna(subset=["x1", "y1", "x2", "y2", "x_m", "y_m", "confidence"])
            df["bbox_h"] = df["y2"] - df["y1"]
            df["bbox_w"] = df["x2"] - df["x1"]
            upright = df[
                (df["bbox_h"] / df["bbox_w"].clip(lower=1) > 1.2)
                & (df["confidence"] > 0.5)
                & (df["on_mat"] == True)  # noqa: E712
            ]
            if len(upright) > 0:
                obs_xy = np.column_stack([
                    upright["x_m"].values.astype(np.float64),
                    upright["y_m"].values.astype(np.float64),
                ])
                obs_h = upright["bbox_h"].values.astype(np.float64)
                pred_h, _, _, _ = compute_pixel_height(P, obs_xy, person_h, geom["img_w"], geom["img_h"])
                ratio = pred_h / obs_h.clip(min=1)
                console.print(
                    f"  Predicted/observed bbox_h ratio: "
                    f"median={np.median(ratio):.2f}, p25={np.percentile(ratio,25):.2f}, p75={np.percentile(ratio,75):.2f}"
                )
                # Subsample for scatter plot
                n_max = 2000
                idx = np.random.default_rng(42).choice(len(obs_xy), min(n_max, len(obs_xy)), replace=False)
                scatter_data = [
                    {"x_m": round(float(obs_xy[i, 0]), 3),
                     "y_m": round(float(obs_xy[i, 1]), 3),
                     "bbox_height_px": round(float(obs_h[i]), 1)}
                    for i in idx
                ]

        # Save
        surface_data: Dict[str, Any] = {
            "camera_id": cam_id,
            "P": P.tolist(),
            "decomposition_diagnostics": diag,
            "person_height_m": person_h,
            "scatter_data": scatter_data,
        }
        cam_dir = configs_root / "cameras" / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / "height_surface.json").write_text(json.dumps(surface_data, indent=2))
        console.print(f"  Saved: {cam_dir / 'height_surface.json'}")

        _plot_phase1(cam_id, geom, P, person_h, scatter_data, diag, configs_root, analysis_dir)
        results[cam_id] = surface_data

    return results


def _plot_phase1(
    cam_id: str, geom: Dict, P: np.ndarray, person_h: float,
    scatter_data: List[Dict], diag: Dict, configs_root: Path, analysis_dir: Path,
) -> None:
    mat_poly = load_mat_blueprint(configs_root)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _draw_mat_rects(ax, configs_root, alpha=0.5)

    # Scatter observed bbox heights
    if scatter_data:
        xs = [d["x_m"] for d in scatter_data]
        ys = [d["y_m"] for d in scatter_data]
        hs = [d["bbox_height_px"] for d in scatter_data]
        vmin, vmax = np.percentile(hs, 5), np.percentile(hs, 95)
        sc = ax.scatter(xs, ys, c=hs, cmap="viridis", s=3, alpha=0.5, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Observed BBox Height (px)")

    # Contour from pose model
    bounds = mat_poly.bounds
    gx = np.linspace(bounds[0] - 1, bounds[2] + 1, 100)
    gy = np.linspace(bounds[1] - 1, bounds[3] + 1, 100)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
    GZ, _, _, _ = compute_pixel_height(P, grid_xy, person_h, geom["img_w"], geom["img_h"])
    GZ = GZ.reshape(GX.shape)

    import shapely
    mat_mask = shapely.contains(mat_poly, shapely.points(GX.ravel(), GY.ravel())).reshape(GX.shape)
    GZ_masked = np.where(mat_mask, GZ, np.nan)

    cs = ax.contour(GX, GY, GZ_masked, levels=10, colors="black", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    reproj = diag.get("reproj_error_mean_px", 0)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"{cam_id} — Pose Decomposition Height Surface (reproj={reproj:.4f}px)")
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
    """Phase 2: Build perspective-aware ROI mask per camera using 3D projection."""
    console.print("\n[bold cyan]═══ Phase 2: ROI Mask Construction ═══[/bold cyan]")

    person_h = reference_height_m or DEFAULT_PERSON_HEIGHT_M
    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mat_poly = load_mat_blueprint(configs_root)
    perimeter_pts = sample_perimeter(mat_poly, PERIMETER_SAMPLE_SPACING_M)

    results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")
        geom = load_camera_geometry(cam_id, configs_root)
        img_w, img_h = geom["img_w"], geom["img_h"]

        P = _load_camera_pose(cam_id, configs_root)
        if P is None:
            console.print("  [red]No P matrix — run Phase 1 first[/red]")
            continue

        # Project foot and head for each perimeter point
        _, _, foot_px, head_px = compute_pixel_height(
            P, perimeter_pts, person_h, img_w, img_h,
        )

        # Filter to in-frame perimeter points (foot inside or near frame)
        in_frame = (
            (foot_px[:, 0] >= -50) & (foot_px[:, 0] < img_w + 50)
            & (foot_px[:, 1] >= -50) & (foot_px[:, 1] < img_h + 50)
        )
        foot_px = foot_px[in_frame]
        head_px = head_px[in_frame]
        console.print(f"  Perimeter points in frame: {in_frame.sum()}/{len(in_frame)}")

        if len(foot_px) < 4:
            console.print("  [red]Too few in-frame perimeter points, skipping[/red]")
            continue

        # Safety margin: extend 10% beyond head in foot->head direction
        direction = head_px - foot_px
        head_margin_px = head_px + direction * ROI_HEAD_BUFFER_FRAC

        # Clip and build ROI band polygon
        head_margin_clipped = np.clip(head_margin_px, [0, 0], [img_w - 1, img_h - 1])
        foot_clipped = np.clip(foot_px, [0, 0], [img_w - 1, img_h - 1])

        roi_boundary = np.vstack([foot_clipped, head_margin_clipped[::-1]])
        roi_poly = Polygon(roi_boundary.tolist())
        if not roi_poly.is_valid:
            roi_poly = roi_poly.buffer(0)
        roi_poly = roi_poly.intersection(box(0, 0, img_w, img_h))
        if roi_poly.is_empty:
            console.print("  [red]ROI polygon is empty, skipping[/red]")
            continue
        if isinstance(roi_poly, MultiPolygon):
            roi_poly = max(roi_poly.geoms, key=lambda g: g.area)

        outer_coords = np.array(roi_poly.exterior.coords)

        # Rasterize
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [outer_coords.astype(np.int32)], 255)

        x_min = max(0, int(outer_coords[:, 0].min()))
        y_min = max(0, int(outer_coords[:, 1].min()))
        x_max = min(img_w, int(outer_coords[:, 0].max()))
        y_max = min(img_h, int(outer_coords[:, 1].max()))

        cam_dir = configs_root / "cameras" / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        mask_path = cam_dir / "roi_mask.png"
        cv2.imwrite(str(mask_path), mask)

        roi_data: Dict[str, Any] = {
            "camera_id": cam_id,
            "image_size": [img_w, img_h],
            "outer_boundary_px": outer_coords.tolist(),
            "inner_boundary_px": foot_clipped.tolist(),
            "bounding_rect": [x_min, y_min, x_max - x_min, y_max - y_min],
            "head_buffer_frac": ROI_HEAD_BUFFER_FRAC,
            "method": "pose_decomposition",
        }
        (cam_dir / "roi_mask.json").write_text(json.dumps(roi_data, indent=2))
        console.print(f"  Saved: {mask_path}")
        console.print(f"  ROI bounding rect: {roi_data['bounding_rect']}")

        _plot_phase2(cam_id, geom, foot_px, head_px, head_margin_clipped,
                     outer_coords, foot_clipped, mask, outputs_root, gym_id, analysis_dir)
        results[cam_id] = roi_data

    return results


def _plot_phase2(
    cam_id: str, geom: Dict, foot_px: np.ndarray, head_px: np.ndarray,
    head_margin_px: np.ndarray, outer_coords: np.ndarray, foot_clipped: np.ndarray,
    mask: np.ndarray, outputs_root: Path, gym_id: str, analysis_dir: Path,
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

    mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    mask_overlay[mask > 0] = [0, 1, 0, 0.2]
    ax.imshow(mask_overlay)

    ax.plot(foot_clipped[:, 0], foot_clipped[:, 1], "b-", linewidth=2, label="Mat edge (foot)")
    ax.plot(outer_coords[:, 0], outer_coords[:, 1], "r-", linewidth=2, label="Head boundary + margin")

    step = max(1, len(foot_px) // 40)
    for i in range(0, len(foot_px), step):
        fx, fy = foot_px[i]
        hx, hy = head_px[i]
        if 0 <= fx < img_w and 0 <= fy < img_h:
            ax.annotate("", xy=(hx, hy), xytext=(fx, fy),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title(f"{cam_id} — ROI Mask ({img_w}x{img_h}{title_suffix})")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(analysis_dir / f"{cam_id}_roi_mask.png", dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {analysis_dir / f'{cam_id}_roi_mask.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Detectability Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_sahi_tile_scale(
    eff_w: int, eff_h: int, grid: List[int], overlap: float, imgsz: int
) -> Tuple[float, int]:
    cols, rows = grid
    tile_w = eff_w / (1 + (cols - 1) * (1 - overlap)) if cols > 1 else eff_w
    tile_h = eff_h / (1 + (rows - 1) * (1 - overlap)) if rows > 1 else eff_h
    return imgsz / max(tile_w, tile_h), cols * rows


def run_phase3(
    cameras: List[str],
    detection_floor_px: float,
    configs_root: Path,
    outputs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Phase 3: Detectability analysis using pose decomposition + kneeling height."""
    console.print("\n[bold cyan]═══ Phase 3: Detectability Analysis ═══[/bold cyan]")

    analysis_dir = outputs_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mat_poly = load_mat_blueprint(configs_root)
    grid_pts_mat, _ = _mat_grid(mat_poly)
    gx_mat, gy_mat = grid_pts_mat[:, 0], grid_pts_mat[:, 1]
    console.print(f"  Mat grid: {len(grid_pts_mat)} points")
    console.print(f"  Kneeling height: {KNEELING_HEIGHT_M}m (scores use kneeling projection)")

    all_results: Dict[str, Dict[str, Any]] = {}

    for cam_id in cameras:
        console.print(f"\n[bold]Camera: {cam_id}[/bold]")
        geom = load_camera_geometry(cam_id, configs_root)
        img_w, img_h = geom["img_w"], geom["img_h"]

        P = _load_camera_pose(cam_id, configs_root)
        if P is None:
            console.print("  [red]No P matrix — run Phase 1 first[/red]")
            continue

        roi_path = configs_root / "cameras" / cam_id / "roi_mask.json"
        roi_data = json.loads(roi_path.read_text()) if roi_path.exists() else None

        # Standing heights for min/mean display
        standing_h, in_frame, _, _ = compute_pixel_height(
            P, grid_pts_mat, DEFAULT_PERSON_HEIGHT_M, img_w, img_h,
        )
        # Kneeling heights for scoring (project at Z=KNEELING_HEIGHT_M)
        kneeling_h, _, _, _ = compute_pixel_height(
            P, grid_pts_mat, KNEELING_HEIGHT_M, img_w, img_h,
        )
        n_in_frame = int(in_frame.sum())
        console.print(f"  In-frame grid points: {n_in_frame}/{len(grid_pts_mat)}")

        config_results: List[Dict[str, Any]] = []

        for cfg in DETECTABILITY_CONFIGS:
            name, imgsz = cfg["name"], cfg["imgsz"]
            roi_crop, sahi = cfg["roi_crop"], cfg["sahi"]

            if roi_crop and roi_data:
                br = roi_data["bounding_rect"]
                eff_w, eff_h = br[2], br[3]
            else:
                eff_w, eff_h = img_w, img_h

            if sahi is not None:
                scale, n_tiles = _compute_sahi_tile_scale(eff_w, eff_h, sahi["grid"], sahi["overlap"], imgsz)
                speed_mult = n_tiles * (imgsz / 640) ** 2
            else:
                scale = imgsz / max(eff_w, eff_h)
                speed_mult = (imgsz / 640) ** 2

            eff_kneeling = kneeling_h * scale
            scores = eff_kneeling / detection_floor_px
            scores[~in_frame] = 0.0

            aspect = eff_w / eff_h if eff_h > 0 else 1.0
            padding_waste = 1.0 - min(aspect, 1.0 / aspect) if aspect > 0 else 0.0

            vis_cov = round(float(np.mean(scores[in_frame] > 1.0)) * 100, 1) if n_in_frame > 0 else 0.0
            vis_standing = standing_h[in_frame] * scale if n_in_frame > 0 else standing_h * scale

            config_results.append({
                "name": name, "imgsz": imgsz, "roi_crop": roi_crop, "sahi": sahi,
                "mat_coverage_pct": round(float(np.mean(scores > 1.0)) * 100, 1),
                "visible_coverage_pct": vis_cov,
                "n_in_frame": n_in_frame,
                "min_effective_px": round(float(vis_standing.min()), 1) if len(vis_standing) > 0 else 0,
                "mean_effective_px": round(float(vis_standing.mean()), 1) if len(vis_standing) > 0 else 0,
                "padding_waste_pct": round(padding_waste * 100, 1),
                "speed_relative": round(speed_mult, 2),
                "scores": scores.tolist(),
            })

        # Print table
        table = Table(title=f"Camera: {cam_id} ({img_w}x{img_h}, {n_in_frame}/{len(gx_mat)} in-frame)")
        table.add_column("Configuration", style="cyan")
        table.add_column("Vis cov%", justify="right")
        table.add_column("Min px_h", justify="right")
        table.add_column("Mean px_h", justify="right")
        table.add_column("Padding%", justify="right")
        table.add_column("Speed", justify="right")
        for r in config_results:
            table.add_row(r["name"], f"{r['visible_coverage_pct']:.0f}%",
                          f"{r['min_effective_px']:.0f}px", f"{r['mean_effective_px']:.0f}px",
                          f"{r['padding_waste_pct']:.0f}%", f"{r['speed_relative']:.1f}x")
        console.print(table)

        _plot_phase3(cam_id, gx_mat, gy_mat, config_results, in_frame, configs_root, analysis_dir)

        report: Dict[str, Any] = {
            "camera_id": cam_id, "image_size": [img_w, img_h],
            "detection_floor_px": detection_floor_px,
            "kneeling_height_m": KNEELING_HEIGHT_M,
            "n_grid_points": len(gx_mat), "n_in_frame": n_in_frame,
            "configurations": [{k: v for k, v in r.items() if k != "scores"} for r in config_results],
            "per_config_scores": {r["name"]: r["scores"] for r in config_results},
        }
        (analysis_dir / f"{cam_id}_detectability_report.json").write_text(json.dumps(report, indent=2))
        console.print(f"  Report: {analysis_dir / f'{cam_id}_detectability_report.json'}")

        all_results[cam_id] = {"config_results": config_results, "grid_x": gx_mat.tolist(), "grid_y": gy_mat.tolist()}

    return all_results


def _plot_phase3(
    cam_id: str, gx: np.ndarray, gy: np.ndarray, config_results: List[Dict],
    in_frame: np.ndarray, configs_root: Path, analysis_dir: Path,
) -> None:
    n_configs = len(config_results)
    ncols = 4
    nrows = max(1, (n_configs + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    cmap = plt.cm.RdYlGn
    norm = BoundaryNorm([0, 0.5, 0.7, 1.0, 1.5, 3.0], cmap.N)
    out_of_frame = ~in_frame

    for idx, result in enumerate(config_results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        scores = np.array(result["scores"])
        _draw_mat_rects(ax, configs_root, alpha=0.2)
        ax.scatter(gx[in_frame], gy[in_frame], c=scores[in_frame], cmap=cmap, norm=norm, s=8, marker="s")
        if out_of_frame.any():
            ax.scatter(gx[out_of_frame], gy[out_of_frame], c="lightgray", s=4, marker="s", alpha=0.3)
        ax.set_title(f"{result['name']}\nvis_cov={result['visible_coverage_pct']:.0f}% spd={result['speed_relative']:.1f}x", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for idx in range(n_configs, nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, label="Detectability (kneeling)")
    fig.suptitle(f"{cam_id} — Detectability by Configuration", fontsize=14, y=1.01)
    fig.savefig(analysis_dir / f"{cam_id}_detectability_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Plot: {analysis_dir / f'{cam_id}_detectability_grid.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Coverage-Aware ROI Optimization
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase4(
    cameras: List[str], detection_floor_px: float, overlap_buffer_m: float,
    min_coverage_override: Optional[int], configs_root: Path, outputs_root: Path,
    reference_height_m: Optional[float], gym_id: str = "",
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
        return {"single_camera": True, "cameras": cameras}

    coverage_matrix: Dict[str, np.ndarray] = {}
    cam_geoms: Dict[str, Dict] = {}
    for cam_id in cameras:
        rp = analysis_dir / f"{cam_id}_detectability_report.json"
        if not rp.exists():
            console.print(f"  [red]No Phase 3 report for {cam_id}[/red]")
            continue
        report = json.loads(rp.read_text())
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
    for s in coverage_matrix.values():
        coverage_count += (s > det_threshold).astype(float)

    actual_min = int(coverage_count.min())
    min_coverage = min_coverage_override if min_coverage_override is not None else min(2, actual_min)
    console.print(f"  Min coverage: {actual_min} cameras, constraint: {min_coverage}")

    if (coverage_count == 0).sum() > 0:
        console.print(f"  [red]{(coverage_count == 0).sum()} points have 0 coverage[/red]")

    score_matrix = np.column_stack([coverage_matrix[c] for c in cam_list])
    primary = np.argmax(score_matrix, axis=1)

    restricted_zones: Dict[str, np.ndarray] = {}
    for c_idx, cam_id in enumerate(cam_list):
        cs = coverage_matrix[cam_id]
        include = (primary == c_idx).copy()
        for i in range(n_pts):
            if include[i] or cs[i] <= det_threshold:
                continue
            if sum(1 for oc in cam_list if oc != cam_id and coverage_matrix[oc][i] > det_threshold) < min_coverage:
                include[i] = True
        inc_pts = grid_pts_mat[include]
        if len(inc_pts) > 0:
            for i in range(n_pts):
                if include[i] or cs[i] <= det_threshold:
                    continue
                if np.sqrt(((grid_pts_mat[i] - inc_pts) ** 2).sum(axis=1)).min() <= overlap_buffer_m:
                    include[i] = True
        restricted_zones[cam_id] = include
        n_inc = int(include.sum())
        console.print(f"  {cam_id}: {n_inc}/{n_pts} points")
        if n_inc == 0:
            console.print(f"  [yellow]{cam_id} redundant at min_coverage={min_coverage}[/yellow]")

    # Fix violations
    cov_check = np.zeros(n_pts)
    for c in cam_list:
        cov_check += (restricted_zones[c] & (coverage_matrix[c] > det_threshold)).astype(float)
    violations = (cov_check < min_coverage).sum()
    if violations > 0:
        console.print(f"  [yellow]Fixing {violations} violations...[/yellow]")
        for i in range(n_pts):
            if cov_check[i] >= min_coverage:
                continue
            for c in cam_list:
                if not restricted_zones[c][i] and coverage_matrix[c][i] > det_threshold:
                    restricted_zones[c][i] = True
                    cov_check[i] += 1
                    if cov_check[i] >= min_coverage:
                        break
        console.print(f"  After fix: {(cov_check < min_coverage).sum()} remaining")

    # Per-camera optimal imgsz
    per_cam: Dict[str, Dict[str, Any]] = {}
    person_h = reference_height_m or DEFAULT_PERSON_HEIGHT_M

    for cam_id in cam_list:
        geom = cam_geoms[cam_id]
        img_w, img_h_px = geom["img_w"], geom["img_h"]
        zone = restricted_zones[cam_id]
        zone_pts = grid_pts_mat[zone]
        if len(zone_pts) == 0:
            continue

        P = _load_camera_pose(cam_id, configs_root)
        if P is None:
            continue

        # Zone bounding rect (foot + head)
        _, _, z_foot, z_head = compute_pixel_height(P, zone_pts, person_h, img_w, img_h_px)
        z_margin = z_head + (z_head - z_foot) * ROI_HEAD_BUFFER_FRAC
        all_pts = np.vstack([z_foot, z_margin])
        all_pts = np.clip(all_pts, [0, 0], [img_w - 1, img_h_px - 1])
        x_min = max(0, int(all_pts[:, 0].min()) - 10)
        y_min = max(0, int(all_pts[:, 1].min()) - 10)
        x_max = min(img_w, int(all_pts[:, 0].max()) + 10)
        y_max = min(img_h_px, int(all_pts[:, 1].max()) + 10)
        rw, rh = x_max - x_min, y_max - y_min

        # Kneeling heights for coverage scoring
        kneel_h, z_in_frame, _, _ = compute_pixel_height(P, zone_pts, KNEELING_HEIGHT_M, img_w, img_h_px)

        optimal_imgsz, optimal_cov = 1536, 0.0
        for test in [640, 800, 960, 1280, 1536]:
            sc = test / max(rw, rh)
            eff = kneel_h * sc
            vis = z_in_frame
            cov = float(np.mean(eff[vis] / detection_floor_px > 1.0)) * 100 if vis.any() else 0
            if cov >= 95 and optimal_imgsz == 1536:
                optimal_imgsz, optimal_cov = test, cov
        if optimal_cov == 0:
            sc = optimal_imgsz / max(rw, rh)
            optimal_cov = float(np.mean(kneel_h[z_in_frame] * sc / detection_floor_px > 1.0)) * 100 if z_in_frame.any() else 0

        full_imgsz = 1536
        for test in [640, 800, 960, 1280, 1536]:
            sc = test / max(img_w, img_h_px)
            eff = kneel_h * sc
            if z_in_frame.any() and float(np.mean(eff[z_in_frame] / detection_floor_px > 1.0)) * 100 >= 95:
                full_imgsz = test
                break

        # Save restricted mask + config
        cam_dir = configs_root / "cameras" / cam_id
        r_mask = np.zeros((img_h_px, img_w), dtype=np.uint8)
        if len(all_pts) >= 3:
            hull = MultiPoint(all_pts.tolist()).convex_hull
            hull = hull.intersection(box(0, 0, img_w, img_h_px))
            if not hull.is_empty:
                if isinstance(hull, MultiPolygon):
                    hull = max(hull.geoms, key=lambda g: g.area)
                if hasattr(hull, "exterior"):
                    cv2.fillPoly(r_mask, [np.array(hull.exterior.coords, dtype=np.int32)], 255)
                    cv2.imwrite(str(cam_dir / "roi_mask_restricted.png"), r_mask)

        det_config = {
            "camera_id": cam_id, "roi_mask": "roi_mask_restricted.png",
            "roi_bounding_rect": [x_min, y_min, rw, rh],
            "imgsz": optimal_imgsz, "sahi": None,
            "effective_coverage_pct": round(optimal_cov, 1),
            "aspect_ratio": round(rw / rh if rh > 0 else 1, 2),
            "generated_at": pd.Timestamp.now().isoformat(),
            "coverage_constraint": f"min_{min_coverage}_cameras",
            "phase4_version": "3.0",
        }
        (cam_dir / "detection_config.json").write_text(json.dumps(det_config, indent=2))

        per_cam[cam_id] = {
            "full_roi_imgsz": full_imgsz, "restricted_shape": f"{rw}x{rh}",
            "optimal_imgsz": optimal_imgsz, "coverage_pct": round(optimal_cov, 1),
            "speed_full": round((full_imgsz / 640) ** 2, 1),
            "speed_restricted": round((optimal_imgsz / 640) ** 2, 1),
            "restricted_mask": r_mask,
        }

    # Plots
    _plot_phase4_coverage(grid_pts_mat, coverage_count, restricted_zones, cam_list, analysis_dir, configs_root)
    for c in cam_list:
        if c in per_cam:
            _plot_phase4_camera(c, cam_geoms[c], per_cam[c], outputs_root, gym_id, configs_root, analysis_dir)

    # Summary table
    table = Table(title="System-Level Coverage Analysis")
    table.add_column("Camera")
    table.add_column("Full imgsz@95%", justify="right")
    table.add_column("Restricted", justify="right")
    table.add_column("Optimal", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Speed", justify="right")
    total_f, total_r = 0.0, 0.0
    for c in cam_list:
        if c not in per_cam:
            continue
        r = per_cam[c]
        total_f += r["speed_full"]
        total_r += r["speed_restricted"]
        table.add_row(c, f"{r['full_roi_imgsz']} ({r['speed_full']:.1f}x)", r["restricted_shape"],
                      str(r["optimal_imgsz"]), f"{r['coverage_pct']:.0f}%", f"{r['speed_restricted']:.1f}x")
    table.add_section()
    table.add_row("TOTAL", f"{total_f:.1f}x", "", "", "", f"{total_r:.1f}x")
    console.print(table)
    speedup = total_f / total_r if total_r > 0 else 1
    console.print(f"  Speedup: {speedup:.1f}x")

    report_out = {
        "cameras": cam_list, "n_grid_points": n_pts,
        "min_coverage_constraint": min_coverage, "actual_min_coverage": actual_min,
        "speedup": round(speedup, 2),
        "per_camera": {c: {k: v for k, v in r.items() if k != "restricted_mask"} for c, r in per_cam.items()},
    }
    (analysis_dir / "coverage_optimization_report.json").write_text(json.dumps(report_out, indent=2, default=str))
    return report_out


def _plot_phase4_coverage(
    grid_pts: np.ndarray, coverage_count: np.ndarray, restricted_zones: Dict[str, np.ndarray],
    cam_list: List[str], analysis_dir: Path, configs_root: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _draw_mat_rects(ax, configs_root, alpha=0.3)
    norm = Normalize(vmin=0, vmax=max(3, coverage_count.max()))
    sc = ax.scatter(grid_pts[:, 0], grid_pts[:, 1], c=coverage_count, cmap=plt.cm.RdYlGn, norm=norm, s=10, marker="s")
    plt.colorbar(sc, ax=ax, label="Coverage Count")
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, c in enumerate(cam_list):
        zone = restricted_zones[c]
        n_z = int(zone.sum())
        color = colors[i % len(colors)]
        if n_z < 3:
            ax.plot([], [], color=color, lw=2, ls="--", label=f"{c} (0 — redundant)")
            continue
        hull = MultiPoint(grid_pts[zone].tolist()).convex_hull
        if hasattr(hull, "exterior"):
            coords = np.array(hull.exterior.coords)
            ax.plot(coords[:, 0], coords[:, 1], color=color, lw=2, ls="--", label=f"{c} ({n_z})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Multi-Camera Coverage Map")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(analysis_dir / "coverage_map.png", dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {analysis_dir / 'coverage_map.png'}")


def _plot_phase4_camera(
    cam_id: str, geom: Dict, results: Dict, outputs_root: Path,
    gym_id: str, configs_root: Path, analysis_dir: Path,
) -> None:
    img_w, img_h = geom["img_w"], geom["img_h"]
    frame = find_camera_frame(outputs_root, gym_id, cam_id)
    title_sfx = ""
    if frame is not None and geom["K"] is not None and geom["D"] is not None:
        frame = cv2.undistort(frame, geom["K"], geom["D"])
        title_sfx = ", undistorted"
    elif frame is None:
        frame = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), alpha=0.7)

    roi_path = configs_root / "cameras" / cam_id / "roi_mask.json"
    if roi_path.exists():
        rd = json.loads(roi_path.read_text())
        outer = np.array(rd["outer_boundary_px"])
        ax.plot(outer[:, 0], outer[:, 1], "w--", lw=1.5, alpha=0.7, label="Full ROI")

    rm = results.get("restricted_mask")
    if rm is not None:
        overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
        overlay[rm > 0] = [0.2, 0.6, 1.0, 0.3]
        ax.imshow(overlay)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title(f"{cam_id} — Restricted ROI (imgsz={results['optimal_imgsz']}, cov={results['coverage_pct']:.0f}%{title_sfx})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(analysis_dir / f"{cam_id}_restricted_roi.png", dpi=150)
    plt.close(fig)
    console.print(f"  Plot: {analysis_dir / f'{cam_id}_restricted_roi.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

app = typer.Typer(help="Camera geometry analysis — pose decomposition model.", add_completion=False)


def _common(outputs: Path, gym_id: str, configs_root: Path) -> Tuple[Path, str, List[str], Path]:
    outputs, configs_root = Path(outputs), Path(configs_root)
    cameras = discover_cameras(outputs, gym_id)
    console.print(f"Cameras: {', '.join(cameras)}")
    return outputs, gym_id, cameras, configs_root


@app.command("phase1")
def cmd_phase1(
    outputs: Path = typer.Option(...), gym_id: str = typer.Option(..., "--gym-id"),
    reference_height_m: Optional[float] = typer.Option(None),
    configs_root: Path = typer.Option(Path("configs")),
) -> None:
    """Phase 1: Decompose camera pose from H and K."""
    o, g, c, cr = _common(outputs, gym_id, configs_root)
    run_phase1(o, g, c, reference_height_m, cr)


@app.command("phase2")
def cmd_phase2(
    outputs: Path = typer.Option(...), gym_id: str = typer.Option(..., "--gym-id"),
    reference_height_m: Optional[float] = typer.Option(None),
    configs_root: Path = typer.Option(Path("configs")),
) -> None:
    """Phase 2: Build perspective-aware ROI mask."""
    o, g, c, cr = _common(outputs, gym_id, configs_root)
    run_phase2(o, g, c, reference_height_m, cr)


@app.command("phase3")
def cmd_phase3(
    outputs: Path = typer.Option(...), gym_id: str = typer.Option(..., "--gym-id"),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX),
    configs_root: Path = typer.Option(Path("configs")),
) -> None:
    """Phase 3: Detectability analysis."""
    o, g, c, cr = _common(outputs, gym_id, configs_root)
    run_phase3(c, detection_floor_px, cr, o)


@app.command("phase4")
def cmd_phase4(
    outputs: Path = typer.Option(...), gym_id: str = typer.Option(..., "--gym-id"),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX),
    overlap_buffer_m: float = typer.Option(DEFAULT_OVERLAP_BUFFER_M),
    min_coverage: Optional[int] = typer.Option(None),
    reference_height_m: Optional[float] = typer.Option(None),
    configs_root: Path = typer.Option(Path("configs")),
) -> None:
    """Phase 4: Multi-camera coverage optimization."""
    o, g, c, cr = _common(outputs, gym_id, configs_root)
    run_phase4(c, detection_floor_px, overlap_buffer_m, min_coverage, cr, o, reference_height_m, g)


@app.command("all")
def cmd_all(
    outputs: Path = typer.Option(...), gym_id: str = typer.Option(..., "--gym-id"),
    reference_height_m: Optional[float] = typer.Option(None),
    detection_floor_px: float = typer.Option(DEFAULT_DETECTION_FLOOR_PX),
    overlap_buffer_m: float = typer.Option(DEFAULT_OVERLAP_BUFFER_M),
    min_coverage: Optional[int] = typer.Option(None),
    configs_root: Path = typer.Option(Path("configs")),
) -> None:
    """Run all four phases."""
    o, g, c, cr = _common(outputs, gym_id, configs_root)
    run_phase1(o, g, c, reference_height_m, cr)
    run_phase2(o, g, c, reference_height_m, cr)
    run_phase3(c, detection_floor_px, cr, o)
    run_phase4(c, detection_floor_px, overlap_buffer_m, min_coverage, cr, o, reference_height_m, g)
    console.print("\n[bold green]All phases complete.[/bold green]")


if __name__ == "__main__":
    app()
