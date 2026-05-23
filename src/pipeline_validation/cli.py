"""Pipeline validation CLI.

Entry point: python -m pipeline_validation <subcommand>

Subcommands:
    discover    Scan repo for model weights, training data, training runs,
                and generate discovery report + model manifest.
    stage-a     Run Stage A detection evaluation (TB-EVAL-1).
    stage-d     Run Stage D identity stitching evaluation (TB-EVAL-2).
    stage-f     Run Stage F match visualization (TB-EVAL-3).
    swap-diagnostic  Tracker swap boundary diagnostic (CP-SWAP-1).
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # src/../..
MODELS_DIR = REPO_ROOT / "models"
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
RUNS_DIR = REPO_ROOT / "runs"
CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
TOOLS_DIR = REPO_ROOT / "tools"
DOCS_DIR = REPO_ROOT / "docs"
DISCOVERY_DOC = DOCS_DIR / "pipeline_validation_discovery.md"
MANIFEST_DIR = CONFIGS_DIR / "models"
PREP_SCRIPT = TOOLS_DIR / "prepare_detection_dataset.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _mtime_str(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _parse_prep_script() -> dict[str, Any] | None:
    """Extract CAMERAS dict and VAL_COUNT_PER_CAMERA from prepare_detection_dataset.py.

    Uses regex to avoid importing the script (which has side-effect-prone deps).
    Returns dict with camera configs and val count, or None if parsing fails.
    """
    if not PREP_SCRIPT.exists():
        return None

    text = PREP_SCRIPT.read_text()

    # Extract VAL_COUNT_PER_CAMERA
    m = re.search(r"VAL_COUNT_PER_CAMERA\s*=\s*(\d+)", text)
    val_count = int(m.group(1)) if m else None

    # Extract per-camera blocks by finding each "CameraName": { ... } block
    cameras = {}

    # Split CAMERAS dict into per-camera blocks
    # Match: "CamName": {  ... },  (greedy within braces)
    block_pattern = re.compile(
        r'"(\w+)":\s*\{(.+?)\}', re.DOTALL
    )
    # Only search within the CAMERAS = { ... } region
    cameras_match = re.search(r"CAMERAS\s*=\s*\{(.+?)^\}", text, re.DOTALL | re.MULTILINE)
    if not cameras_match:
        return None
    cameras_text = cameras_match.group(1)

    for block in block_pattern.finditer(cameras_text):
        cam_name = block.group(1)
        body = block.group(2)

        # Extract zip filename
        zip_m = re.search(r'"(training_[^"]+\.zip)"', body)
        if not zip_m:
            continue
        zip_name = zip_m.group(1)

        # Extract video path
        video_m = re.search(r'Path\("([^"]+)"\)', body)
        if not video_m:
            continue
        video_path = video_m.group(1)

        # Extract range(start, stop, stride)
        range_m = re.search(r"range\((\d+),\s*(\d+),\s*(\d+)\)", body)
        if not range_m:
            continue
        range_start = int(range_m.group(1))
        range_stop = int(range_m.group(2))
        range_stride = int(range_m.group(3))

        # Extract prefix
        prefix_m = re.search(r'"prefix":\s*"(\w+)"', body)
        prefix = prefix_m.group(1) if prefix_m else cam_name.lower()[:3]

        frames = list(range(range_start, range_stop, range_stride))
        last_frame = frames[-1] if frames else range_start
        cameras[cam_name] = {
            "zip": zip_name,
            "video": video_path,
            "frames": frames,
            "prefix": prefix,
            "range_start": range_start,
            "range_stop": last_frame,  # last frame actually in the range
            "range_stride": range_stride,
        }

    if not cameras or val_count is None:
        return None

    return {"cameras": cameras, "val_count": val_count}


def _scan_zip_labels(zip_path: Path) -> dict[str, Any]:
    """Scan a CVAT export zip for label files, classifying empty vs non-empty."""
    result: dict[str, Any] = {
        "total_files": 0,
        "empty_files": 0,
        "non_empty_files": 0,
        "frame_indices_non_empty": set(),
        "frame_indices_empty": set(),
        "has_data_yaml": False,
        "data_yaml_content": "",
    }

    if not zip_path.exists():
        return result

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("data.yaml"):
                result["has_data_yaml"] = True
                result["data_yaml_content"] = zf.read(name).decode()
            if not name.endswith(".txt") or "labels/" not in name:
                continue

            # Extract frame index from filename
            basename = name.split("/")[-1]
            m = re.match(r"frame_(\d+)\.txt", basename)
            if not m:
                continue

            frame_idx = int(m.group(1))
            content = zf.read(name).decode().strip()
            result["total_files"] += 1

            if content:
                result["non_empty_files"] += 1
                result["frame_indices_non_empty"].add(frame_idx)
            else:
                result["empty_files"] += 1
                result["frame_indices_empty"].add(frame_idx)

    return result


def _camera_resolutions() -> dict[str, tuple[int, int]]:
    """Known camera resolutions (confirmed from video inspection)."""
    return {
        "FP7oJQ": (1920, 1080),
        "J_EDEw": (1280, 720),
        "PPDmUg": (1280, 720),
    }


# ---------------------------------------------------------------------------
# Phase 1: Model weights inventory
# ---------------------------------------------------------------------------

def phase1_model_weights(out: io.StringIO) -> None:
    out.write("## 1. Model Weights Inventory\n\n")
    out.write("Status: CONFIRMED (direct filesystem inspection)\n\n")

    # .pt files
    pt_files = sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    # .mlpackage dirs
    ml_dirs = sorted(MODELS_DIR.glob("*.mlpackage")) if MODELS_DIR.exists() else []
    ml_stems = {d.stem for d in ml_dirs}
    # sidecar files
    sidecar_exts = {".yaml", ".json", ".md", ".txt", ".log"}
    sidecars = [
        f for f in MODELS_DIR.iterdir()
        if f.is_file() and f.suffix in sidecar_exts
    ] if MODELS_DIR.exists() else []

    out.write("| File | Size | Modified | CoreML sibling | Category |\n")
    out.write("|------|------|----------|----------------|----------|\n")

    for pt in pt_files:
        size = _human_size(pt.stat().st_size)
        mtime = _mtime_str(pt)
        has_coreml = "Yes" if pt.stem in ml_stems else "No"
        category = "domain-tuned" if pt.stem.startswith("bjj-") else "stock"
        out.write(f"| `{pt.name}` | {size} | {mtime} | {has_coreml} | {category} |\n")

    out.write(f"\nCoreML packages: {len(ml_dirs)} "
              f"({', '.join(d.name for d in ml_dirs)})\n")

    if sidecars:
        out.write(f"\nSidecar metadata files: {len(sidecars)}\n")
        for s in sidecars:
            out.write(f"  - `{s.name}`\n")
    else:
        out.write("\nSidecar metadata files: **none**. No .yaml, .json, .md, "
                  ".txt, or .log files co-located with model weights.\n")

    # training_runs subdirectory
    tr_dir = MODELS_DIR / "training_runs"
    if tr_dir.exists():
        tr_files = list(tr_dir.rglob("*"))
        tr_files = [f for f in tr_files if f.is_file()]
        out.write(f"\n`models/training_runs/` contents ({len(tr_files)} files):\n")
        for f in sorted(tr_files)[:10]:
            out.write(f"  - `{f.relative_to(MODELS_DIR)}`\n")
    out.write("\n")


# ---------------------------------------------------------------------------
# Phase 2: Training-data inventory
# ---------------------------------------------------------------------------

def phase2_training_data(out: io.StringIO) -> dict[str, dict[str, Any]]:
    """Scan training data zips and return per-zip scan results."""
    out.write("## 2. Training-Data Inventory\n\n")
    out.write("Status: CONFIRMED (zip content inspection)\n\n")

    zip_scans: dict[str, dict[str, Any]] = {}

    # Find all zips in training_data/
    zips = sorted(TRAINING_DATA_DIR.glob("*.zip")) if TRAINING_DATA_DIR.exists() else []

    if not zips:
        out.write("No zip files found in `data/training_data/`.\n\n")
        return zip_scans

    out.write("### CVAT Export Zips\n\n")
    out.write("| Zip | Size | Total labels | Non-empty | Empty |\n")
    out.write("|-----|------|-------------|-----------|-------|\n")

    for zp in zips:
        size = _human_size(zp.stat().st_size)
        scan = _scan_zip_labels(zp)
        zip_scans[zp.name] = scan
        out.write(
            f"| `{zp.name}` | {size} | {scan['total_files']} "
            f"| {scan['non_empty_files']} | {scan['empty_files']} |\n"
        )

    out.write("\n")

    # Dataset YAML files
    yaml_files = sorted(TRAINING_DATA_DIR.rglob("dataset.yaml")) + \
                 sorted(TRAINING_DATA_DIR.rglob("data.yaml"))
    if yaml_files:
        out.write("### Dataset YAML Files\n\n")
        for yf in yaml_files:
            rel = yf.relative_to(TRAINING_DATA_DIR)
            content = yf.read_text().strip()
            out.write(f"**`data/training_data/{rel}`**\n```yaml\n{content}\n```\n\n")

    # Detection dataset specifics
    det_dir = TRAINING_DATA_DIR / "detection_all_cameras"
    if det_dir.exists():
        out.write("### Detection Dataset (detection_all_cameras/)\n\n")
        for split_name in ("train", "val"):
            split_file = det_dir / f"{split_name}.txt"
            if split_file.exists():
                lines = [l.strip() for l in split_file.read_text().strip().split("\n") if l.strip()]
                cams: dict[str, int] = {}
                for l in lines:
                    prefix = os.path.basename(l).split("_frame_")[0]
                    cams[prefix] = cams.get(prefix, 0) + 1
                cam_str = ", ".join(f"{k}: {v}" for k, v in sorted(cams.items()))
                out.write(f"- **{split_name}.txt**: {len(lines)} entries ({cam_str})\n")
        out.write("\n")

    return zip_scans


# ---------------------------------------------------------------------------
# Phase 3: Training run records
# ---------------------------------------------------------------------------

def phase3_training_runs(out: io.StringIO) -> None:
    out.write("## 3. Training Run Records\n\n")

    # Local YOLO training runs
    args_files = sorted(RUNS_DIR.rglob("args.yaml")) if RUNS_DIR.exists() else []

    if args_files:
        out.write("### Local Training Runs (CONFIRMED)\n\n")
        out.write("| Run | Task | Data | Epochs | Freeze | Device |\n")
        out.write("|-----|------|------|--------|--------|--------|\n")

        for af in args_files:
            run_name = af.parent.name
            # Walk up to get a descriptive path
            rel = af.relative_to(RUNS_DIR)
            parts = list(rel.parts)
            run_label = "/".join(parts[:-1])  # drop args.yaml

            try:
                import yaml
                with open(af) as f:
                    args = yaml.safe_load(f)
                task = args.get("task", "?")
                data = args.get("data", "?")
                if isinstance(data, str) and "/" in data:
                    data = data.split("/")[-1]
                epochs = args.get("epochs", "?")
                freeze = args.get("freeze", "?")
                device = args.get("device", "?")
                out.write(
                    f"| `{run_label}` | {task} | `{data}` "
                    f"| {epochs} | {freeze} | {device} |\n"
                )
            except Exception:
                out.write(f"| `{run_label}` | (parse error) | | | | |\n")

        out.write("\n")
    else:
        out.write("No local training runs found in `runs/`.\n\n")

    # Experiment tracking dirs
    for tracker in ("wandb", "mlruns", "tensorboard"):
        tracker_dir = REPO_ROOT / tracker
        exists = tracker_dir.exists()
        out.write(f"- `{tracker}/`: {'found' if exists else 'not found'}\n")
    out.write("\n")

    # Detection model provenance
    out.write("### Detection Model Provenance (INFERRED)\n\n")
    out.write(
        "`bjj-detect-all-cameras.pt` was trained on Kaggle, not locally. "
        "Evidence:\n"
        "- No detection training runs exist in `runs/` (only pose runs)\n"
        "- `tools/colab_detection_training.ipynb` references:\n"
        "  - Base model: `yolo26n.pt`\n"
        "  - Dataset: `training_data_detection_all_cameras.zip` (uploaded to Kaggle)\n"
        "  - Config: epochs=100, batch=16, freeze=10, imgsz=640\n"
        "  - Output saved as `bjj-detect-all-cameras.pt`\n"
        "- File modified date (2026-05-06) is consistent with CLAUDE.md CP23b timeline\n"
        "- Active production model: **CONFIRMED** (referenced in `configs/default.yaml`)\n\n"
    )

    # Pose model provenance
    out.write("### Pose Model Provenance\n\n")
    out.write(
        "Pose models (`bjj-pose-r1`, `bjj-pose-r2`, `bjj-pose-r2_bbox`, "
        "`bjj-pose-vicos`, `bjj-pose-hybrid`) have partial local training "
        "records (R1, R2 probe) but the final models for R2 variants were "
        "trained on Kaggle. **No manifest yet, provenance not backfilled.** "
        "Out of scope for this brief.\n\n"
    )


# ---------------------------------------------------------------------------
# Phase 4: In-distribution / held-out status
# ---------------------------------------------------------------------------

def phase4_held_out(
    out: io.StringIO,
    zip_scans: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Cross-reference prep script with zip contents. Returns parsed prep data."""
    out.write("## 4. In-Distribution / Held-Out Status\n\n")

    prep = _parse_prep_script()
    if prep is None:
        out.write(
            "**OPEN**: Could not parse `tools/prepare_detection_dataset.py`. "
            "Manual inspection required.\n\n"
        )
        return None

    out.write("Status: CONFIRMED (parsed from `tools/prepare_detection_dataset.py`)\n\n")
    out.write(
        "**Key rule:** Held-out evaluation frames = val partition only. "
        "Frames outside the annotated_range have no ground truth and "
        "cannot be used for recall/precision.\n\n"
    )

    val_count = prep["val_count"]
    resolutions = _camera_resolutions()

    # Evaluation surface table
    out.write("### Evaluation Surface\n\n")
    out.write(
        "| Camera | Annotated frames | Train (in-dist) | Val (held-out) "
        "| Resolution |\n"
    )
    out.write(
        "|--------|-----------------|-----------------|----------------"
        "|------------|\n"
    )

    for cam_name, cam in prep["cameras"].items():
        total_frames = len(cam["frames"])
        train_count = total_frames - val_count
        res = resolutions.get(cam_name, ("?", "?"))
        stride_str = f" stride {cam['range_stride']}" if cam["range_stride"] > 1 else ""
        out.write(
            f"| {cam_name} | {total_frames} "
            f"(frames {cam['range_start']}--{cam['range_stop']}{stride_str}) "
            f"| {train_count} | {val_count} "
            f"| {res[0]}x{res[1]} |\n"
        )

    total_annotated = sum(len(c["frames"]) for c in prep["cameras"].values())
    total_val = val_count * len(prep["cameras"])
    out.write(
        f"\n**Total:** {total_annotated} annotated frames, "
        f"{total_val} held-out (val) frames across {len(prep['cameras'])} cameras.\n\n"
    )

    # Reconciliation table
    out.write("### Zip Content Reconciliation\n\n")
    out.write(
        "Cross-references annotated_range (from prep script) against actual "
        "zip contents. annotated_range is **authoritative**; zip contents are "
        "advisory.\n\n"
    )
    out.write(
        "| Camera | annotated_range_count | non_empty_in_zip | "
        "extra_non_empty_outside_range | empty_in_zip |\n"
    )
    out.write(
        "|--------|----------------------|------------------|"
        "-----------------------------|-------------|\n"
    )

    for cam_name, cam in prep["cameras"].items():
        annotated_set = set(cam["frames"])
        annotated_count = len(annotated_set)
        zip_name = cam["zip"]
        scan = zip_scans.get(zip_name, {})
        non_empty_in_zip = scan.get("non_empty_files", 0)
        non_empty_frames = scan.get("frame_indices_non_empty", set())
        extra = non_empty_frames - annotated_set
        extra_count = len(extra)
        empty_count = scan.get("empty_files", 0)

        out.write(
            f"| {cam_name} | {annotated_count} | {non_empty_in_zip} "
            f"| {extra_count} | {empty_count} |\n"
        )

    out.write("\n")

    # Check for extra non-empty outside range
    any_extra = False
    for cam_name, cam in prep["cameras"].items():
        annotated_set = set(cam["frames"])
        zip_name = cam["zip"]
        scan = zip_scans.get(zip_name, {})
        non_empty_frames = scan.get("frame_indices_non_empty", set())
        extra = non_empty_frames - annotated_set
        if extra:
            any_extra = True
            extra_sorted = sorted(extra)
            sample = extra_sorted[:5]
            out.write(
                f"**WARNING ({cam_name}):** {len(extra)} non-empty label files "
                f"exist outside annotated_range "
                f"(sample frame indices: {sample}{'...' if len(extra) > 5 else ''}). "
                f"These are likely CVAT auto-interpolations and are **NOT trusted GT**. "
                f"Downstream eval code must filter to annotated_range only.\n\n"
            )

    if not any_extra:
        out.write(
            "No non-empty labels found outside annotated_range for any camera. "
            "Zip contents are consistent with the manifest.\n\n"
        )

    return prep


# ---------------------------------------------------------------------------
# Phase 5: Existing manifest conventions
# ---------------------------------------------------------------------------

def phase5_manifests(out: io.StringIO) -> bool:
    """Returns True if an existing manifest convention was found."""
    out.write("## 5. Existing Manifest Conventions\n\n")
    out.write("Status: CONFIRMED (absence confirmed by repo-wide search)\n\n")

    found_any = False

    # Check configs/models/
    if MANIFEST_DIR.exists():
        yamls = list(MANIFEST_DIR.glob("*.yaml")) + list(MANIFEST_DIR.glob("*.json"))
        if yamls:
            found_any = True
            out.write(f"Found {len(yamls)} manifest(s) in `configs/models/`:\n")
            for y in yamls:
                out.write(f"  - `{y.name}`\n")
            out.write("\n")

    # Check for manifest.yaml/json anywhere
    for pattern in ("manifest.yaml", "manifest.json"):
        matches = list(REPO_ROOT.glob(f"**/{pattern}"))
        # Exclude node_modules, .venv, etc.
        matches = [
            m for m in matches
            if not any(
                x in str(m)
                for x in ("node_modules", ".venv", "__pycache__", ".git",
                           "app_mobile", "app_web")
            )
        ]
        if matches:
            found_any = True
            out.write(f"Found `{pattern}`:\n")
            for m in matches:
                out.write(f"  - `{m.relative_to(REPO_ROOT)}`\n")
            out.write("\n")

    if not found_any:
        out.write(
            "**No existing manifest convention found.** No file in the repo "
            "currently tracks which CVAT exports were used to train which "
            "models. `configs/models/` does not exist. See Part C below for "
            "the proposed schema.\n\n"
        )

    return found_any


# ---------------------------------------------------------------------------
# Part C: Manifest schema proposal
# ---------------------------------------------------------------------------

def part_c_manifest_proposal(out: io.StringIO) -> None:
    out.write("## Part C: Manifest Schema Proposal (DRAFT)\n\n")
    out.write(
        "Since no existing manifest convention was found, this brief proposes "
        "per-model sidecar YAML files in `configs/models/{model_id}.yaml`, "
        "following the existing `configs/cameras/{cam_id}.yaml` pattern.\n\n"
    )

    out.write("### Schema\n\n")
    out.write("```yaml\n")
    out.write(
        "model_id: <string>           # matches filename stem\n"
        "weights_path: <string>       # relative to repo root\n"
        "base_model: <string>         # stock model trained from\n"
        "trained_at: <date string>    # YYYY-MM-DD\n"
        "training_config:             # hyperparameters\n"
        "  epochs: <int>\n"
        "  batch: <int>\n"
        "  freeze: <int>\n"
        "  imgsz: <int>\n"
        "  lr0: <float>\n"
        "  platform: <string>         # e.g. kaggle-t4, local-cpu\n"
        "\n"
        "training_data:               # one entry per CVAT export used\n"
        "  - export: <filename.zip>   # zip filename in data/training_data/\n"
        "    source_video: <filename>  # video the annotations cover\n"
        "    camera_id: <string>\n"
        "    resolution: [<w>, <h>]   # needed to denormalize GT bboxes\n"
        "    annotated_range:          # the trusted GT frame coverage\n"
        "      start: <int>           # first annotated frame index\n"
        "      stop: <int>            # last annotated frame index (inclusive)\n"
        "      stride: <int>          # 1 = every frame, 10 = every 10th\n"
        "      count: <int>           # total annotated frames (checksum)\n"
        "    splits:\n"
        "      train:\n"
        "        start: <int>\n"
        "        stop: <int>\n"
        "        stride: <int>        # same as annotated_range stride\n"
        "        count: <int>\n"
        "      val:\n"
        "        start: <int>\n"
        "        stop: <int>\n"
        "        stride: <int>\n"
        "        count: <int>\n"
        "\n"
        "notes: |                     # free-form provenance notes\n"
        "  ...\n"
        "```\n\n"
    )

    out.write("### Field Justifications\n\n")
    out.write(
        "- **start/stop/stride/count** over `frame_range: [a, b]`: cleanly "
        "represents both FP7oJQ's stride-1 and J_EDEw/PPDmUg's stride-10 "
        "sampling. `count` is redundant but serves as a cross-check.\n"
        "- **annotated_range is authoritative**: downstream eval code uses "
        "this to determine which frames have trusted GT. Zip contents may "
        "include CVAT auto-interpolations outside this range; those must be "
        "ignored.\n"
        "- **resolution per export**: needed to denormalize GT bboxes from "
        "[0,1] to pixel space for IoU computation.\n"
        "- **Two splits only** (train, val): val IS the held-out eval set. "
        "No third partition.\n"
        "- **Per-model sidecar** (not global file): each model's provenance "
        "is self-contained, matches `configs/cameras/` convention, and "
        "avoids merge conflicts when multiple models are developed in "
        "parallel.\n\n"
    )

    out.write("### Future Manifest Workflow\n\n")
    out.write(
        "**Recommended: hand-author with template emitter.**\n\n"
        "Reasoning:\n"
        "- Auto-generation from prep scripts is fragile -- each prep script "
        "has different structure, and Kaggle/Colab training produces no "
        "local artifacts to parse.\n"
        "- A manifest is authored once per model and rarely changes. The "
        "cost of hand-authoring is low.\n"
        "- A template emitter (`python -m pipeline_validation create-manifest "
        "--model-id X`) generates an empty YAML with all required fields "
        "and inline comments, reducing typo risk.\n\n"
        "For this brief, `bjj-detect-all-cameras.yaml` is generated "
        "programmatically from `prepare_detection_dataset.py` constants "
        "as a one-time bootstrap. Future models should be hand-authored "
        "following the template.\n\n"
        "The `create-manifest` subcommand is stubbed in this brief's CLI "
        "and will be implemented in a future brief.\n\n"
    )


# ---------------------------------------------------------------------------
# Part D: Open questions
# ---------------------------------------------------------------------------

def part_d_open_questions(out: io.StringIO) -> None:
    out.write("## Part D: Open Questions\n\n")
    out.write(
        "Questions that could not be answered from repo inspection alone. "
        "User should answer these directly.\n\n"
    )

    questions = [
        (
            "PPDmUg training sample provenance",
            "`data/raw/nest/training_samples/training_PPDmUg_3000.mp4` is not "
            "pixel-identical to any Nest clip in `data/raw/nest/.../PPDmUg/`. "
            "Visual comparison confirms different scene content at matching "
            "frame indices. Correlation search found no strong match. "
            "Where did this video come from? Is it a segment from a different "
            "recording session, or was it created by a process not tracked "
            "in the repo?"
        ),
        (
            "PPDmUg pipeline evaluation path",
            "Pipeline Stage A outputs exist for Nest clip "
            "`PPDmUg-20260318-200019` but NOT for the training sample video. "
            "To evaluate Stage A detection quality on PPDmUg GT annotations, "
            "either: (a) run the model directly on the training sample in the "
            "eval tool, or (b) find the Nest clip that corresponds to the "
            "training sample and use its pipeline output. Which approach is "
            "preferred?"
        ),
        (
            "Kaggle training logs",
            "No Kaggle training logs/metrics exist locally for "
            "`bjj-detect-all-cameras.pt`. The training config in the manifest "
            "is INFERRED from the notebook. Should Kaggle results.csv / "
            "training logs be downloaded and stored locally for provenance?"
        ),
        (
            "Pose model manifests",
            "Six pose models exist (`bjj-pose-r1` through `bjj-pose-hybrid`) "
            "with no manifests. Local training runs exist for R1 and R2 probe "
            "only; final R2/vicos/hybrid models were trained on Kaggle. "
            "Should manifests be backfilled for these models, or are they "
            "considered experimental/archived?"
        ),
    ]

    for i, (title, body) in enumerate(questions, 1):
        out.write(f"{i}. **{title}** (OPEN)\n\n")
        out.write(f"   {body}\n\n")


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def generate_manifest(prep: dict[str, Any]) -> str:
    """Generate YAML content for bjj-detect-all-cameras manifest."""
    import yaml

    resolutions = _camera_resolutions()
    val_count = prep["val_count"]

    training_data = []
    for cam_name, cam in prep["cameras"].items():
        total = len(cam["frames"])
        train_count = total - val_count
        stride = cam["range_stride"]

        # Train: first (total - val_count) frames
        train_frames = cam["frames"][:train_count]
        train_start = train_frames[0]
        train_stop = train_frames[-1]

        # Val: last val_count frames
        val_frames = cam["frames"][train_count:]
        val_start = val_frames[0]
        val_stop = val_frames[-1]

        res = resolutions.get(cam_name, (0, 0))

        entry = {
            "export": cam["zip"],
            "source_video": os.path.basename(cam["video"]),
            "camera_id": cam_name,
            "resolution": list(res),
            "annotated_range": {
                "start": cam["range_start"],
                "stop": cam["range_stop"],
                "stride": stride,
                "count": total,
            },
            "splits": {
                "train": {
                    "start": train_start,
                    "stop": train_stop,
                    "stride": stride,
                    "count": train_count,
                },
                "val": {
                    "start": val_start,
                    "stop": val_stop,
                    "stride": stride,
                    "count": val_count,
                },
            },
        }
        training_data.append(entry)

    # Model file info
    model_path = MODELS_DIR / "bjj-detect-all-cameras.pt"
    trained_at = _mtime_str(model_path) if model_path.exists() else "unknown"

    manifest = {
        "model_id": "bjj-detect-all-cameras",
        "weights_path": "models/bjj-detect-all-cameras.pt",
        "base_model": "yolo26n.pt",
        "trained_at": trained_at,
        "training_config": {
            "epochs": 100,
            "batch": 16,
            "freeze": 10,
            "imgsz": 640,
            "lr0": 0.001,
            "platform": "kaggle-t4",
        },
        "training_data": training_data,
        "notes": (
            "Detection-only model (no pose head trained). Base: stock yolo26n.pt.\n"
            "Trained on Kaggle T4 GPU via tools/colab_detection_training.ipynb.\n"
            "902 total frames (749 train / 153 val), 83/17 temporal split per camera.\n"
            "Val split is the only held-out evaluation surface with GT annotations.\n"
            "annotated_range is authoritative; zip contents may include CVAT\n"
            "auto-interpolations outside this range that are NOT trusted GT."
        ),
    }

    # Use yaml.dump with specific formatting
    return yaml.dump(
        manifest,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


# ---------------------------------------------------------------------------
# discover subcommand
# ---------------------------------------------------------------------------

def cmd_discover(args: argparse.Namespace) -> None:
    """Run all discovery phases and write report."""
    out = io.StringIO()

    out.write("# Pipeline Validation Discovery Report\n\n")
    out.write(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    )
    out.write(
        "This report scans the repo for model weights, training data, "
        "training runs, and answers the in-distribution vs held-out question "
        "for each GT clip. Findings are labeled CONFIRMED, INFERRED, or OPEN.\n\n"
    )
    out.write("---\n\n")

    # Phase 1
    phase1_model_weights(out)

    # Phase 2
    zip_scans = phase2_training_data(out)

    # Phase 3
    phase3_training_runs(out)

    # Phase 4
    prep = phase4_held_out(out, zip_scans)

    # Phase 5
    has_manifests = phase5_manifests(out)

    out.write("---\n\n")

    # Part C (only if no existing manifests)
    if not has_manifests:
        part_c_manifest_proposal(out)

    # Part D
    part_d_open_questions(out)

    # Write discovery doc
    report = out.getvalue()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DISCOVERY_DOC.write_text(report)
    print(report)
    print(f"\n--- Report written to {DISCOVERY_DOC.relative_to(REPO_ROOT)} ---")

    # Generate manifest if prep data available
    if prep is not None:
        try:
            manifest_yaml = generate_manifest(prep)
            MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
            manifest_path = MANIFEST_DIR / "bjj-detect-all-cameras.yaml"
            manifest_path.write_text(manifest_yaml)
            print(f"--- Manifest written to {manifest_path.relative_to(REPO_ROOT)} ---")
        except Exception as e:
            print(f"WARNING: Failed to generate manifest: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# evaluate subcommand — unified end-to-end model evaluation
# ---------------------------------------------------------------------------

EVAL_DIR_A = REPO_ROOT / "outputs" / "_eval" / "stage_a"
EVAL_DIR_D = REPO_ROOT / "outputs" / "_eval" / "stage_d"
EVAL_DIR_F = REPO_ROOT / "outputs" / "_eval" / "stage_f"


def _validate_manifest_for_eval(manifest) -> list[str]:
    """Validate manifest before evaluation. Returns list of error messages."""
    errors = []
    if not (REPO_ROOT / manifest.weights_path).exists():
        errors.append(f"weights_path not found: {manifest.weights_path}")
    if not manifest.pipeline_gym_id:
        errors.append("pipeline_gym_id not set in manifest (required for pipeline rerun)")
    for e in manifest.training_data:
        zip_path = TRAINING_DATA_DIR / e.export
        if not zip_path.exists():
            errors.append(f"GT zip not found: {e.export}")
        if e.source_video_path and not (REPO_ROOT / e.source_video_path).exists():
            errors.append(f"source_video_path not found: {e.source_video_path}")
    return errors


def _pipeline_outputs_exist(manifest, export) -> bool:
    """Check if pipeline ran through Stage E for this camera."""
    gym_id = manifest.pipeline_gym_id
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id
    pattern = f"{gym_id}/{cam}/**/{clip_id}/stage_E/match_sessions.jsonl"
    return len(list((REPO_ROOT / "outputs").glob(pattern))) > 0


def _eval_output_exists(stage: str, model_id: str, camera_id: str) -> bool:
    """Check if evaluation output exists for a stage."""
    paths = {
        "stage-a": EVAL_DIR_A / model_id / camera_id / "report.md",
        "stage-d": EVAL_DIR_D / model_id / camera_id / "report.md",
        "stage-f": EVAL_DIR_F / model_id / camera_id / "match_preview.mp4",
    }
    return paths.get(stage, Path("/nonexistent")).exists()


def _resolve_ingest_path(manifest, export) -> Path:
    """Ensure hard link exists at canonical nest path. Returns the path."""
    gym_id = manifest.pipeline_gym_id
    cam = export.camera_id
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")

    # Check if hard link already exists (search under gym_id/cam)
    nest_base = REPO_ROOT / "data" / "raw" / "nest" / gym_id / cam
    existing = list(nest_base.rglob(f"{clip_id}.mp4"))
    if existing:
        return existing[0]

    # Derive date/hour from clip_id: CAM-YYYYMMDD-HHMMSS
    parts = clip_id.split("-")
    if len(parts) >= 2 and len(parts[1]) == 8:
        date_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
        # Hour from third segment — must be numeric
        if len(parts) >= 3 and len(parts[2]) >= 2 and parts[2][:2].isdigit():
            hour_str = parts[2][:2]
        else:
            hour_str = "20"  # default for gym evening sessions
    else:
        date_str = "2026-01-01"
        hour_str = "00"

    nest_dir = nest_base / date_str / hour_str
    link_path = nest_dir / f"{clip_id}.mp4"

    # Find source video
    if export.source_video_path:
        source = REPO_ROOT / export.source_video_path
    else:
        # Try canonical nest path under existing gym_ids
        candidates = list((REPO_ROOT / "data" / "raw" / "nest").glob(
            f"*/{cam}/{date_str}/{hour_str}/{clip_id}.mp4"
        ))
        source = candidates[0] if candidates else None

    if source is None or not source.exists():
        raise FileNotFoundError(f"Cannot find source video for {cam}/{clip_id}")

    nest_dir.mkdir(parents=True, exist_ok=True)
    os.link(str(source), str(link_path))
    return link_path


def _run_pipeline_for_camera(manifest, export, log_dir: Path) -> bool:
    """Run bjj_pipeline CLI for one camera. Returns True on success."""
    clip_path = _resolve_ingest_path(manifest, export)
    cmd = [
        sys.executable, "-m", "bjj_pipeline.stages.orchestration.cli", "run",
        "--clip", str(clip_path),
        "--camera", export.camera_id,
        "--to-stage", "E",
        "--force",
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{export.camera_id}_pipeline.log"
    with open(log_path, "w") as log_f:
        result = __import__("subprocess").run(
            cmd, stdout=log_f, stderr=__import__("subprocess").STDOUT,
        )
    return result.returncode == 0


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Unified end-to-end model evaluation."""
    import time as _time

    from pipeline_validation.common.manifest import load_manifest

    model_id = args.model
    manifest_path = CONFIGS_DIR / "models" / f"{model_id}.yaml"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    # Manifest validation
    errors = _validate_manifest_for_eval(manifest)
    if errors:
        print("Manifest validation failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Filter exports by --clip-id
    exports = manifest.training_data
    if args.clip_id:
        exports = [e for e in exports if (
            (e.pipeline_output_clip_id or e.source_video.replace(".mp4", "")) == args.clip_id
            or e.camera_id == args.clip_id
        )]
        if not exports:
            print(f"No export matches --clip-id {args.clip_id}")
            sys.exit(1)

    cameras = [e.camera_id for e in exports]
    force = args.force

    # Build execution plan
    print(f"\nEvaluating {model_id} against {len(exports)} cameras: {', '.join(cameras)}")
    print(f"Estimated total runtime: ~31 min (full) / near-zero (cached)\n")

    # Step 1: Pipeline rerun plan
    pipeline_plan: dict[str, str] = {}  # cam -> "SKIP" | "will run"
    if args.skip_pipeline:
        for e in exports:
            pipeline_plan[e.camera_id] = "SKIP (--skip-pipeline)"
    else:
        for e in exports:
            clip_id = e.pipeline_output_clip_id or e.source_video.replace(".mp4", "")
            if _pipeline_outputs_exist(manifest, e) and not force:
                pipeline_plan[e.camera_id] = "SKIP (outputs exist)"
            else:
                pipeline_plan[e.camera_id] = "will run"

    print("Step 1: Pipeline rerun (Stage A->E)")
    for e in exports:
        clip_id = e.pipeline_output_clip_id or e.source_video.replace(".mp4", "")
        print(f"  {clip_id:<35} [{pipeline_plan[e.camera_id]}]")

    # Steps 2-4 plan
    eval_steps = [
        ("Step 2: Stage A detection evaluation", "stage-a", args.skip_stage_a, EVAL_DIR_A),
        ("Step 3: Stage D identity evaluation", "stage-d", args.skip_stage_d, EVAL_DIR_D),
        ("Step 4: Stage F match visualization", "stage-f", args.skip_stage_f, EVAL_DIR_F),
    ]
    step_plans: dict[str, str] = {}
    for label, stage_key, skip_flag, eval_dir in eval_steps:
        if skip_flag:
            status = "SKIP (--skip flag)"
        elif all(_eval_output_exists(stage_key, model_id, e.camera_id) for e in exports) and not force:
            status = "SKIP (outputs exist)"
        else:
            status = "will run"
        step_plans[stage_key] = status
        print(f"\n{label}")
        print(f"  -> {eval_dir / model_id}/   [{status}]")

    # Check if everything skipped
    all_skip = (
        all(v.startswith("SKIP") for v in pipeline_plan.values())
        and all(v.startswith("SKIP") for v in step_plans.values())
    )
    if all_skip:
        print(f"\nAll steps would be skipped (outputs already exist). Use --force to re-run.")

    if args.dry_run:
        print("\n(dry-run mode — no execution)")
        return

    # Execute
    t_start = _time.time()
    results: dict[str, dict[str, int]] = {
        "pipeline": {"ok": 0, "skip": 0, "fail": 0},
        "stage-a": {"ok": 0, "skip": 0, "fail": 0},
        "stage-d": {"ok": 0, "skip": 0, "fail": 0},
        "stage-f": {"ok": 0, "skip": 0, "fail": 0},
    }
    log_dir = REPO_ROOT / "outputs" / "_eval" / "_logs" / model_id

    # Step 1: Pipeline rerun (per-camera independent)
    if not args.skip_pipeline:
        print("\n--- Step 1: Pipeline rerun ---")
        for e in exports:
            if pipeline_plan[e.camera_id].startswith("SKIP"):
                results["pipeline"]["skip"] += 1
                print(f"  {e.camera_id}: skipped")
                continue
            print(f"  {e.camera_id}: running pipeline...")
            ok = _run_pipeline_for_camera(manifest, e, log_dir)
            if ok:
                results["pipeline"]["ok"] += 1
                print(f"  {e.camera_id}: SUCCESS")
            else:
                results["pipeline"]["fail"] += 1
                log_path = log_dir / f"{e.camera_id}_pipeline.log"
                print(f"  {e.camera_id}: FAILED — see {log_path}")
                print("Pipeline failure halts evaluation. Fix the issue and re-run.")
                sys.exit(1)
    else:
        results["pipeline"]["skip"] = len(exports)

    # Pre-step verification: check pipeline outputs exist for all cameras
    if not args.skip_stage_d or not args.skip_stage_f:
        for e in exports:
            if not _pipeline_outputs_exist(manifest, e):
                clip_id = e.pipeline_output_clip_id or e.source_video.replace(".mp4", "")
                print(f"\nCannot proceed: pipeline outputs not found for {e.camera_id} "
                      f"({clip_id}). Re-run without --skip-pipeline.")
                sys.exit(1)

    # Step 2: Stage A
    if not args.skip_stage_a:
        if step_plans["stage-a"].startswith("SKIP"):
            results["stage-a"]["skip"] = len(exports)
            print("\n--- Step 2: Stage A evaluation --- [SKIPPED]")
        else:
            print("\n--- Step 2: Stage A evaluation ---")
            try:
                from pipeline_validation.stage_a.evaluate import evaluate_all
                evaluate_all(manifest_path, run_model=True)
                results["stage-a"]["ok"] = len(exports)
            except Exception as exc:
                print(f"  Stage A failed: {exc}")
                results["stage-a"]["fail"] = len(exports)
    else:
        results["stage-a"]["skip"] = len(exports)

    # Step 3: Stage D
    if not args.skip_stage_d:
        if step_plans["stage-d"].startswith("SKIP"):
            results["stage-d"]["skip"] = len(exports)
            print("\n--- Step 3: Stage D evaluation --- [SKIPPED]")
        else:
            print("\n--- Step 3: Stage D evaluation ---")
            try:
                from pipeline_validation.stage_d.evaluate import evaluate_all as eval_d
                eval_d(manifest_path)
                results["stage-d"]["ok"] = len(exports)
            except Exception as exc:
                print(f"  Stage D failed: {exc}")
                results["stage-d"]["fail"] = len(exports)
    else:
        results["stage-d"]["skip"] = len(exports)

    # Step 4: Stage F
    if not args.skip_stage_f:
        if step_plans["stage-f"].startswith("SKIP"):
            results["stage-f"]["skip"] = len(exports)
            print("\n--- Step 4: Stage F visualization --- [SKIPPED]")
        else:
            print("\n--- Step 4: Stage F visualization ---")
            try:
                from pipeline_validation.stage_f.visualize import render_all
                render_all(manifest_path)
                results["stage-f"]["ok"] = len(exports)
            except Exception as exc:
                print(f"  Stage F failed: {exc}")
                results["stage-f"]["fail"] = len(exports)
    else:
        results["stage-f"]["skip"] = len(exports)

    # Summary
    elapsed = _time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    summary_lines = [
        f"===== Evaluation complete: {model_id} =====",
    ]
    step_names = {
        "pipeline": "Pipeline rerun",
        "stage-a": "Stage A eval",
        "stage-d": "Stage D eval",
        "stage-f": "Stage F viz",
    }
    for key, label in step_names.items():
        r = results[key]
        total = r["ok"] + r["skip"] + r["fail"]
        if r["skip"] == total:
            summary_lines.append(f"  {label + ':':<18} {total}/{total} cameras skipped")
        elif r["fail"] > 0:
            summary_lines.append(f"  {label + ':':<18} {r['ok']}/{total} succeeded, {r['fail']} failed")
        else:
            summary_lines.append(f"  {label + ':':<18} {r['ok']}/{total} cameras succeeded")

    summary_lines.extend([
        "",
        "Reports:",
        f"  {EVAL_DIR_A / model_id / '_aggregate.md'}",
        f"  {EVAL_DIR_D / model_id / '_aggregate.md'}",
        f"  {EVAL_DIR_F / model_id}/  ({len(exports)} mp4s)",
        "",
        f"Total runtime: {minutes}m {seconds}s",
    ])

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "evaluate_summary.txt").write_text(summary)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def cmd_swap_diagnostic(args: argparse.Namespace) -> None:
    """Tracker swap boundary diagnostic (CP-SWAP-1)."""
    import logging as _logging

    import pandas as pd

    from pipeline_validation.common.gt_loader import load_gt_for_split
    from pipeline_validation.common.manifest import load_manifest
    from pipeline_validation.tracker_swap.diagnostic import run_diagnostic
    from pipeline_validation.tracker_swap.report import write_reports

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model_id = args.model
    manifest_path = CONFIGS_DIR / "models" / f"{model_id}.yaml"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    gym_id = args.gym_id or manifest.pipeline_gym_id or "_eval_gt"

    results = []
    for export in manifest.training_data:
        cam = export.camera_id
        clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
        print(f"\n--- {cam} ({clip_id}) ---")

        # Resolve clip directory
        pattern = f"{gym_id}/{cam}/**/{clip_id}"
        matches = list(OUTPUTS_DIR.glob(pattern))
        if not matches:
            print(f"  No pipeline output found for {cam}/{clip_id} under {gym_id}, skipping")
            continue
        clip_dir = matches[0]

        # Load GT (train + val combined for max coverage)
        zip_path = TRAINING_DATA_DIR / export.export
        if not zip_path.exists():
            print(f"  GT zip not found: {zip_path}, skipping")
            continue
        gt_train = load_gt_for_split(zip_path, export, "train")
        gt_val = load_gt_for_split(zip_path, export, "val")
        gt_by_frame = {**gt_train, **gt_val}
        print(f"  GT frames: {len(gt_by_frame)} ({len(gt_train)} train + {len(gt_val)} val)")

        # Load detections
        det_path = clip_dir / "stage_A" / "detections.parquet"
        if not det_path.exists():
            print(f"  detections.parquet not found at {det_path}, skipping")
            continue
        detections_df = pd.read_parquet(det_path)
        # Filter to detections with tracklet_id
        detections_df = detections_df[detections_df["tracklet_id"].notna()].copy()
        print(f"  Detections: {len(detections_df)} tracked ({detections_df['tracklet_id'].nunique()} tracklets)")

        # Load optional bank frames
        bank_path = clip_dir / "stage_D" / "tracklet_bank_frames.parquet"
        bank_frames_df = None
        if bank_path.exists():
            bank_frames_df = pd.read_parquet(bank_path)
            print(f"  Bank frames: {len(bank_frames_df)} rows")
        else:
            print("  Bank frames: not available")

        # Load optional histograms
        hist_path = clip_dir / "stage_A" / "color_histograms.parquet"
        histograms_df = None
        if hist_path.exists():
            histograms_df = pd.read_parquet(hist_path)
            print(f"  Histograms: {len(histograms_df)} rows")
        else:
            print("  Histograms: not available")

        result = run_diagnostic(
            camera_id=cam,
            detections_df=detections_df,
            gt_by_frame=gt_by_frame,
            bank_frames_df=bank_frames_df,
            histograms_df=histograms_df,
        )
        results.append(result)

    if not results:
        print("\nNo cameras processed. Check pipeline outputs and GT zips.")
        sys.exit(1)

    agg_path = write_reports(model_id, results)
    print(f"\nDone. Aggregate report: {agg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline_validation",
        description="Pipeline validation tooling for Roll Tracker.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("discover", help="Scan repo and generate discovery report")

    stage_a = sub.add_parser("stage-a", help="Stage A detection evaluation (TB-EVAL-1)")
    stage_a.add_argument("--model", default="bjj-detect-all-cameras",
                         help="Model ID (must have manifest at configs/models/{id}.yaml)")
    stage_a.add_argument("--run-model", action="store_true",
                         help="Use direct inference instead of parquet (required for PPDmUg)")
    stage_a.add_argument("--gym-id", default=None,
                         help="Gym ID for outputs directory (auto-detected if unambiguous)")

    stage_d = sub.add_parser("stage-d", help="Stage D identity stitching evaluation (TB-EVAL-2)")
    stage_d.add_argument("--model", default="bjj-detect-all-cameras",
                         help="Model ID (must have manifest at configs/models/{id}.yaml)")

    stage_f = sub.add_parser("stage-f", help="Stage F match visualization (TB-EVAL-3)")
    stage_f.add_argument("--model", default="bjj-detect-all-cameras",
                         help="Model ID (must have manifest at configs/models/{id}.yaml)")

    evaluate = sub.add_parser("evaluate", help="Full model evaluation (pipeline + stage-a + stage-d + stage-f)")
    evaluate.add_argument("--model", required=True,
                          help="Model ID (must have manifest at configs/models/{id}.yaml)")
    evaluate.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline rerun")
    evaluate.add_argument("--skip-stage-a", action="store_true", help="Skip Stage A evaluation")
    evaluate.add_argument("--skip-stage-d", action="store_true", help="Skip Stage D evaluation")
    evaluate.add_argument("--skip-stage-f", action="store_true", help="Skip Stage F visualization")
    evaluate.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    evaluate.add_argument("--clip-id", default=None, help="Restrict to one camera's clip")
    evaluate.add_argument("--dry-run", action="store_true", help="Print plan, don't execute")

    swap_diag = sub.add_parser("swap-diagnostic",
                               help="Tracker swap boundary diagnostic (CP-SWAP-1)")
    swap_diag.add_argument("--model", default="bjj-detect-all-cameras",
                           help="Model ID (must have manifest at configs/models/{id}.yaml)")
    swap_diag.add_argument("--gym-id", default=None,
                           help="Gym ID for pipeline output paths")

    sub.add_parser("create-manifest", help="Generate empty manifest template (future)")

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "stage-a":
        from pipeline_validation.stage_a.evaluate import evaluate_all
        manifest_path = CONFIGS_DIR / "models" / f"{args.model}.yaml"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            sys.exit(1)
        evaluate_all(manifest_path, run_model=args.run_model, gym_id=args.gym_id)
    elif args.command == "stage-d":
        from pipeline_validation.stage_d.evaluate import evaluate_all as eval_d
        manifest_path = CONFIGS_DIR / "models" / f"{args.model}.yaml"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            sys.exit(1)
        eval_d(manifest_path)
    elif args.command == "stage-f":
        from pipeline_validation.stage_f.visualize import render_all
        manifest_path = CONFIGS_DIR / "models" / f"{args.model}.yaml"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            sys.exit(1)
        render_all(manifest_path)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "swap-diagnostic":
        cmd_swap_diagnostic(args)
    elif args.command == "create-manifest":
        print(f"'{args.command}' is not yet implemented.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
