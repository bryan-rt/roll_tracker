"""Upload a video to CVAT as a bare task with no pre-populated annotations.

Why this exists: the training pipeline's _run_stage_a_inference uses per-frame
detection indices as tracklet_id, which group unrelated people across frames.
When hand-labelling, these degenerate pre-populated tracks are worse than
nothing — correcting wrong associations is slower than drawing fresh. This
script uploads the video directly, skipping Stage A inference entirely.

Usage:
    PYTHONPATH=src python tools/cvat_upload_bare.py \
        --clip data/raw/nest/.../FP7oJQ-20260822-132650.mp4 \
        --task-name gt_FP7oJQ_20260822_132650
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from training_pipeline.config import load_config
from training_pipeline.cvat_integration import connect, create_project_if_needed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a video to CVAT with no pre-populated annotations."
    )
    parser.add_argument("--clip", required=True, type=Path, help="Path to the video file.")
    parser.add_argument("--task-name", required=True, help="CVAT task name.")
    args = parser.parse_args()

    clip_path = args.clip
    task_name = args.task_name

    if not clip_path.exists():
        logger.error(f"Clip not found: {clip_path}")
        sys.exit(1)

    # Load CVAT credentials from training pipeline config
    cfg = load_config()
    if not cfg.cvat_username or not cfg.cvat_password:
        logger.error("CVAT credentials not configured. Run training pipeline option 8.")
        sys.exit(1)

    logger.info(f"Connecting to CVAT at {cfg.cvat_url}")
    client = connect(cfg.cvat_url, cfg.cvat_username, cfg.cvat_password)

    # Set organization context — the project lives in the "Roll Tracker" org.
    # Without this, task creation fails with "task and project should be in
    # the same organization".
    orgs = client.organizations.list()
    org_list = getattr(orgs, "results", orgs)
    for org in org_list:
        if hasattr(org, "slug"):
            client.organization_slug = org.slug
            logger.info(f"Organization context: {org.slug}")
            break

    project_id = create_project_if_needed(client, cfg.cvat_project_name)
    logger.info(f"Target project: {cfg.cvat_project_name} (id={project_id})")

    # Create task and upload video — no annotations
    from cvat_sdk.api_client.models import TaskWriteRequest

    task_spec = TaskWriteRequest(name=task_name)
    task_spec.project_id = project_id

    task = client.tasks.create(task_spec)
    task_id = task.id
    logger.info(f"Created task: {task_name} (id={task_id})")

    task.upload_data(
        resources=[str(clip_path)],
        params={"image_quality": 95},
    )
    logger.info(f"Uploaded video: {clip_path.name}")

    # Report
    task_url = f"{cfg.cvat_url}/tasks/{task_id}"
    task_obj = client.tasks.retrieve(task_id)
    frame_count = getattr(task_obj, "size", "unknown")

    print()
    print(f"Task ID:     {task_id}")
    print(f"Task URL:    {task_url}")
    print(f"Frame count: {frame_count}")
    print(f"Task name:   {task_name}")
    print(f"Project:     {cfg.cvat_project_name}")


if __name__ == "__main__":
    main()
