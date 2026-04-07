"""Dataset manager — accumulate, validate, and split training data."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from loguru import logger

from training_pipeline.pseudo_labels import COCO_KEYPOINT_NAMES

_DEFAULT_DATA_DIR = Path("data/training_data")


@dataclass
class IngestStats:
    """Statistics from ingesting a batch of annotations."""

    frames_added: int = 0
    annotations_added: int = 0
    skipped_existing: int = 0
    session_id: str = ""
    cam_id: str = ""


@dataclass
class DatasetStats:
    """Overall dataset statistics."""

    total_frames: int = 0
    total_annotations: int = 0
    frames_per_camera: Dict[str, int] = field(default_factory=dict)
    frames_per_round: Dict[int, int] = field(default_factory=dict)
    annotation_sources: Dict[str, int] = field(default_factory=dict)
    sessions: List[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """A single validation problem in an annotation."""

    image_id: int
    annotation_id: int
    issue_type: str  # skeleton_crossing | keypoint_outside_bbox | missing_keypoints
    severity: str = "error"  # error | warning
    details: str = ""


@dataclass
class ValidationReport:
    """Result of validating a COCO annotation file."""

    total_annotations: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


def _load_manifest(data_dir: Path) -> Dict[str, Any]:
    """Load or create the dataset manifest."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {
        "sessions": {},
        "total_frames": 0,
        "total_annotations": 0,
        "rounds": {},
    }


def _save_manifest(data_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _coco_to_yolo_keypoints(
    annotation: Dict,
    img_width: int,
    img_height: int,
) -> str:
    """Convert a COCO annotation to YOLO keypoint format.

    YOLO format: class x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
    All coordinates normalized to [0, 1].
    """
    bbox = annotation["bbox"]  # [x, y, w, h] in pixels
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    parts = [f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"]

    kps = annotation.get("keypoints", [0.0] * 51)
    for i in range(17):
        kv = int(kps[i * 3 + 2])
        kx = kps[i * 3] / img_width if kv > 0 else 0.0
        ky = kps[i * 3 + 1] / img_height if kv > 0 else 0.0
        parts.append(f"{kx:.6f} {ky:.6f} {kv}")

    return " ".join(parts)


def ingest_annotations(
    corrected_coco_json: str | Path,
    session_id: str,
    cam_id: str,
    images_dir: str | Path,
    data_dir: Optional[Path] = None,
    round_num: int = 0,
) -> IngestStats:
    """Ingest corrected COCO annotations into the accumulated training dataset.

    Converts COCO annotations to YOLO format, copies images, updates manifest.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    coco_path = Path(corrected_coco_json)
    images_src = Path(images_dir)

    # Ensure directories exist
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)
    (data_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Load COCO JSON
    coco_data = json.loads(coco_path.read_text())
    images = {img["id"]: img for img in coco_data.get("images", [])}

    # Save source COCO JSON to annotations archive
    archive_name = f"{session_id}_{cam_id}_corrected.json"
    shutil.copy2(coco_path, data_dir / "annotations" / archive_name)

    manifest = _load_manifest(data_dir)
    stats = IngestStats(session_id=session_id, cam_id=cam_id)
    session_key = f"{session_id}_{cam_id}"

    # Track which images have already been ingested
    existing_images: Set[str] = set()
    if session_key in manifest.get("sessions", {}):
        existing_images = set(manifest["sessions"][session_key].get("images", []))

    # Process annotations grouped by image
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in images.items():
        stem = f"{session_id}_{cam_id}_{img_id:06d}"

        if stem in existing_images:
            stats.skipped_existing += 1
            continue

        # Copy image
        src_img = images_src / img_info["file_name"]
        if src_img.exists():
            dst_img = data_dir / "images" / f"{stem}.jpg"
            shutil.copy2(src_img, dst_img)
        else:
            logger.warning(f"Image not found: {src_img}")
            continue

        # Convert annotations to YOLO format
        anns = anns_by_image.get(img_id, [])
        label_lines = []
        for ann in anns:
            line = _coco_to_yolo_keypoints(ann, img_info["width"], img_info["height"])
            label_lines.append(line)
            stats.annotations_added += 1

        label_path = data_dir / "labels" / f"{stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n" if label_lines else "")

        existing_images.add(stem)
        stats.frames_added += 1

    # Update manifest
    sessions = manifest.setdefault("sessions", {})
    sessions[session_key] = {
        "session_id": session_id,
        "cam_id": cam_id,
        "images": sorted(existing_images),
        "frame_count": len(existing_images),
        "round": round_num,
    }
    manifest["total_frames"] = sum(
        s.get("frame_count", 0) for s in sessions.values()
    )
    manifest["total_annotations"] = manifest.get("total_annotations", 0) + stats.annotations_added
    rounds = manifest.setdefault("rounds", {})
    rounds[str(round_num)] = rounds.get(str(round_num), 0) + stats.frames_added

    _save_manifest(data_dir, manifest)
    logger.info(
        f"Ingested {stats.frames_added} frames, {stats.annotations_added} annotations "
        f"from {session_key}"
    )
    return stats


def get_stats(data_dir: Optional[Path] = None) -> DatasetStats:
    """Compute overall dataset statistics from the manifest."""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    manifest = _load_manifest(data_dir)

    stats = DatasetStats()
    stats.total_frames = manifest.get("total_frames", 0)
    stats.total_annotations = manifest.get("total_annotations", 0)

    for key, session in manifest.get("sessions", {}).items():
        cam_id = session.get("cam_id", "unknown")
        stats.frames_per_camera[cam_id] = (
            stats.frames_per_camera.get(cam_id, 0) + session.get("frame_count", 0)
        )
        stats.sessions.append(key)

    for round_str, count in manifest.get("rounds", {}).items():
        stats.frames_per_round[int(round_str)] = count

    return stats


def generate_dataset_yaml(
    val_session_ids: List[str],
    data_dir: Optional[Path] = None,
) -> Path:
    """Generate ultralytics-compatible dataset.yaml with session-based train/val split.

    Parameters
    ----------
    val_session_ids : Session IDs to use for validation (never mix with training).
    data_dir : Training data directory.

    Returns
    -------
    Path to the generated dataset.yaml.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    manifest = _load_manifest(data_dir)

    train_images: List[str] = []
    val_images: List[str] = []

    for key, session in manifest.get("sessions", {}).items():
        session_id = session.get("session_id", "")
        imgs = session.get("images", [])

        if session_id in val_session_ids:
            val_images.extend(imgs)
        else:
            train_images.extend(imgs)

    # Write train.txt and val.txt
    images_dir = data_dir / "images"

    train_txt = data_dir / "train.txt"
    train_txt.write_text(
        "\n".join(str(images_dir / f"{stem}.jpg") for stem in train_images) + "\n"
    )

    val_txt = data_dir / "val.txt"
    val_txt.write_text(
        "\n".join(str(images_dir / f"{stem}.jpg") for stem in val_images) + "\n"
    )

    # Build dataset.yaml
    dataset_config = {
        "path": str(data_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "person"},
        "kpt_shape": [17, 3],
    }

    yaml_path = data_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_config, default_flow_style=False))

    logger.info(
        f"Dataset YAML: {yaml_path} "
        f"(train={len(train_images)}, val={len(val_images)})"
    )
    return yaml_path


def validate_annotations(coco_json: str | Path) -> ValidationReport:
    """Validate a COCO annotation file for common issues.

    Checks for:
    - Skeleton crossing (left/right swap)
    - Keypoints outside bbox
    - Missing required keypoints (at least some torso points)
    """
    coco_data = json.loads(Path(coco_json).read_text())
    images = {img["id"]: img for img in coco_data.get("images", [])}
    report = ValidationReport()

    # Paired left/right keypoint indices for crossing check
    lr_pairs = [
        (1, 2),   # eyes
        (3, 4),   # ears
        (5, 6),   # shoulders
        (7, 8),   # elbows
        (9, 10),  # wrists
        (11, 12), # hips
        (13, 14), # knees
        (15, 16), # ankles
    ]

    for ann in coco_data.get("annotations", []):
        report.total_annotations += 1
        ann_id = ann["id"]
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        kps = ann.get("keypoints", [])

        bx, by, bw, bh = bbox
        img_info = images.get(img_id, {})

        # Check keypoints outside bbox (with 20% margin)
        margin_x = bw * 0.2
        margin_y = bh * 0.2
        for i in range(17):
            if i * 3 + 2 >= len(kps):
                break
            kx, ky, kv = kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]
            if kv == 0:
                continue
            if (kx < bx - margin_x or kx > bx + bw + margin_x or
                    ky < by - margin_y or ky > by + bh + margin_y):
                report.issues.append(ValidationIssue(
                    image_id=img_id,
                    annotation_id=ann_id,
                    issue_type="keypoint_outside_bbox",
                    details=f"Keypoint {COCO_KEYPOINT_NAMES[i]} at ({kx:.0f}, {ky:.0f}) "
                            f"is outside bbox [{bx:.0f}, {by:.0f}, {bw:.0f}, {bh:.0f}]",
                ))

        # Check skeleton crossing (left should be on viewer's right = higher x)
        for li, ri in lr_pairs:
            if li * 3 + 2 >= len(kps) or ri * 3 + 2 >= len(kps):
                continue
            lx, lv = kps[li * 3], kps[li * 3 + 2]
            rx, rv = kps[ri * 3], kps[ri * 3 + 2]
            if lv > 0 and rv > 0 and lx > rx:
                # Left keypoint is to the right of the right keypoint — possible swap.
                # Downgraded to warning: grappling means people face all directions,
                # so left/right crossings are common and not necessarily errors.
                report.issues.append(ValidationIssue(
                    image_id=img_id,
                    annotation_id=ann_id,
                    issue_type="skeleton_crossing",
                    severity="warning",
                    details=f"{COCO_KEYPOINT_NAMES[li]} (x={lx:.0f}) is right of "
                            f"{COCO_KEYPOINT_NAMES[ri]} (x={rx:.0f})",
                ))

        # Check for missing required keypoints (need at least 2 torso points)
        torso_indices = [5, 6, 11, 12]  # shoulders + hips
        visible_torso = sum(
            1 for i in torso_indices
            if i * 3 + 2 < len(kps) and kps[i * 3 + 2] > 0
        )
        if visible_torso < 2 and ann.get("num_keypoints", 0) > 0:
            report.issues.append(ValidationIssue(
                image_id=img_id,
                annotation_id=ann_id,
                issue_type="missing_keypoints",
                details=f"Only {visible_torso}/4 torso keypoints visible",
            ))

    return report
