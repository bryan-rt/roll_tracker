#!/usr/bin/env python3
"""
Convert a homography .npy file into the canonical homography.json.

Two ways to use:

1) Explicit paths
   python -m bjj_pipeline.tools.npy_to_homography_json \
       --npy configs/cameras/cam03/homography_matrix.npy \
       --camera cam03 \
       --out configs/cameras/cam03/homography.json

2) Auto-discover for a camera (preferred)
   python -m bjj_pipeline.tools.npy_to_homography_json \
       --camera cam03

   This will search configs/cameras/<camera>/ for homography_*.npy and write
   configs/cameras/<camera>/homography.json

Output JSON schema (loader expects key 'H' as a 3x3 list):
   { "H": [[...],[...],[...]] }
"""

import argparse
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def _discover_npy(camera: str) -> Path:
    cam_dir = Path("configs") / "cameras" / camera
    # Prefer common names first
    preferred = [
        cam_dir / "homography_matrix.npy",
        cam_dir / "homography.npy",
    ]
    for p in preferred:
        if p.exists():
            return p
    # Fallback: any homography_*.npy in folder
    candidates = sorted(
        (Path(p) for p in glob.glob(str(cam_dir / "homography_*.npy"))),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No homography .npy found in {cam_dir} (looked for homography_*.npy)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", help="Path to homography .npy (if omitted, auto-discovers under configs/cameras/<camera>/)")
    parser.add_argument("--camera", required=True, help="Camera ID (e.g. cam03)")
    parser.add_argument("--out", help="Output homography.json path (default: configs/cameras/<camera>/homography.json)")
    args = parser.parse_args()

    npy_path = Path(args.npy) if args.npy else _discover_npy(args.camera)
    out_path = Path(args.out) if args.out else (Path("configs") / "cameras" / args.camera / "homography.json")

    H = np.load(npy_path)
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3,3), got {H.shape}")

    # Minimal schema the loader expects: top-level 'H' as 3x3 list
    payload = {
        "H": H.astype(float).tolist(),
        # Optional metadata fields (ignored by loader but useful to humans)
        "camera_id": args.camera,
        "source": {"type": "imported_npy", "path": str(npy_path)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✔ Wrote homography.json → {out_path}")


if __name__ == "__main__":
    main()
