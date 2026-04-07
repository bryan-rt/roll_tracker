"""CVAT SDK integration — task management for annotation workflows."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

# COCO keypoint labels for project setup
_PERSON_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class TaskSummary:
    """Summary of a CVAT annotation task."""

    task_id: int
    name: str
    status: str
    assignee: str = ""
    updated_date: str = ""


@dataclass
class TaskStatus:
    """Detailed status of a CVAT annotation task."""

    task_id: int
    status: str
    assignee: str = ""
    updated_date: str = ""
    frame_count: int = 0


def connect(url: str, username: str, password: str):
    """Authenticate to a CVAT instance.

    Returns a cvat_sdk Client object. Raises ImportError if cvat-sdk is not installed.
    """
    try:
        from cvat_sdk import make_client
    except ImportError:
        raise ImportError(
            "cvat-sdk is required for CVAT integration. "
            "Install with: pip install cvat-sdk cvat-cli"
        )

    client = make_client(host=url, credentials=(username, password))
    logger.info(f"Connected to CVAT at {url}")
    return client


def create_project_if_needed(client, name: str = "BJJ Training Data") -> int:
    """Create a CVAT project with COCO person keypoint skeleton, or return existing ID."""
    from cvat_sdk.api_client.models import (
        LabelRequest,
        ProjectWriteRequest,
        SublabelRequest,
    )

    # Check for existing project
    projects = client.projects.list()
    for p in projects.results:
        if p.name == name:
            logger.info(f"Using existing CVAT project: {name} (id={p.id})")
            return p.id

    # Build person label with keypoint sublabels
    sublabels = [
        SublabelRequest(name=kp, type="points")
        for kp in _PERSON_KEYPOINTS
    ]
    person_label = LabelRequest(
        name="person",
        type="skeleton",
        sublabels=sublabels,
    )

    project = client.projects.create(
        ProjectWriteRequest(name=name, labels=[person_label])
    )
    logger.info(f"Created CVAT project: {name} (id={project.id})")
    return project.id


def upload_task(
    client,
    task_name: str,
    data_zip: str | Path,
    project_id: Optional[int] = None,
) -> int:
    """Create a CVAT task with pre-annotations from a zip file.

    The zip should contain images/ and annotations.json in COCO Keypoints format.

    Returns the task ID.
    """
    from cvat_sdk.api_client.models import TaskWriteRequest

    data_zip = Path(data_zip)
    if not data_zip.exists():
        raise FileNotFoundError(f"Data zip not found: {data_zip}")

    # Create task
    task_spec = TaskWriteRequest(name=task_name)
    if project_id is not None:
        task_spec.project_id = project_id

    task = client.tasks.create(task_spec)
    task_id = task.id
    logger.info(f"Created CVAT task: {task_name} (id={task_id})")

    # Upload data (images from zip)
    # Extract images to a temp dir for upload, then upload annotations separately
    import zipfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(data_zip) as zf:
            zf.extractall(tmpdir)

        # Upload images
        images_dir = tmpdir / "images"
        if images_dir.exists():
            image_files = sorted(images_dir.glob("*.jpg"))
            if image_files:
                task.upload_data(
                    resources=[str(f) for f in image_files],
                    image_quality=95,
                )
                logger.info(f"Uploaded {len(image_files)} images to task {task_id}")

        # Upload annotations
        ann_file = tmpdir / "annotations.json"
        if ann_file.exists():
            task.import_annotations(
                format_name="COCO Keypoints 1.0",
                filename=str(ann_file),
            )
            logger.info(f"Uploaded pre-annotations to task {task_id}")

    return task_id


def check_task_status(client, task_id: int) -> TaskStatus:
    """Query the status of a CVAT annotation task."""
    task = client.tasks.retrieve(task_id)
    assignee = ""
    if task.assignee:
        assignee = getattr(task.assignee, "username", str(task.assignee))

    return TaskStatus(
        task_id=task_id,
        status=task.status,
        assignee=assignee,
        updated_date=str(task.updated_date) if task.updated_date else "",
        frame_count=task.size or 0,
    )


def list_open_tasks(
    client,
    project_prefix: str = "BJJ Training Data",
) -> List[TaskSummary]:
    """List all tasks in projects matching the prefix."""
    tasks_list = client.tasks.list()
    results = []

    for t in tasks_list.results:
        # Filter by project name prefix if project is set
        project_name = ""
        if t.project_id:
            try:
                proj = client.projects.retrieve(t.project_id)
                project_name = proj.name
            except Exception:
                project_name = f"project_{t.project_id}"

        if project_prefix and project_name and not project_name.startswith(project_prefix):
            continue

        assignee = ""
        if t.assignee:
            assignee = getattr(t.assignee, "username", str(t.assignee))

        results.append(TaskSummary(
            task_id=t.id,
            name=t.name,
            status=t.status,
            assignee=assignee,
            updated_date=str(t.updated_date) if t.updated_date else "",
        ))

    return results


def download_annotations(
    client,
    task_id: int,
    output_path: str | Path,
) -> Path:
    """Export corrected COCO JSON annotations from a completed task.

    Returns path to the downloaded annotations file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task = client.tasks.retrieve(task_id)

    # Export annotations in COCO Keypoints format.
    # export_dataset() writes the zip to the filename path and returns it.
    with tempfile.TemporaryDirectory() as tmpdir:
        import zipfile

        export_zip = Path(tmpdir) / "export.zip"
        task.export_dataset(
            format_name="COCO Keypoints 1.0",
            filename=str(export_zip),
        )

        # The export is a zip containing annotations/instances_default.json
        with zipfile.ZipFile(export_zip) as zf:
            # Find the annotations JSON inside
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                raise ValueError(f"No JSON found in CVAT export for task {task_id}")

            # Extract the first JSON file
            json_name = json_files[0]
            zf.extract(json_name, tmpdir)
            extracted = Path(tmpdir) / json_name

            # Copy to output path
            import shutil
            shutil.copy2(extracted, output_path)

    logger.info(f"Downloaded annotations for task {task_id} -> {output_path}")
    return output_path
