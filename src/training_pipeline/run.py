"""Interactive CLI orchestrator for the training pipeline active learning loop.

Entry point: python -m src.training_pipeline.run
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from rich.console import Console
from rich.table import Table

from training_pipeline.config import TrainingPipelineConfig, load_config, save_config
from training_pipeline.state import (
    CvatTaskInfo,
    PipelineState,
    RoundInfo,
    load_state,
    record_round,
    record_session,
    save_state,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_model(state: PipelineState, cfg: TrainingPipelineConfig) -> str:
    """Return the path to the current best model."""
    if state.rounds:
        return state.rounds[-1].model
    return cfg.base_model


def _prompt_choice(prompt: str, valid: set[str]) -> str:
    """Prompt user for a menu choice."""
    while True:
        choice = console.input(f"[bold cyan]{prompt}[/] ").strip()
        if choice in valid:
            return choice
        console.print(f"[red]Invalid choice. Options: {', '.join(sorted(valid))}[/]")


def _prompt_confirm(prompt: str) -> bool:
    """Prompt for yes/no confirmation."""
    resp = console.input(f"[bold cyan]{prompt} (y/n)[/] ").strip().lower()
    return resp in ("y", "yes")


def _prompt_text(prompt: str, default: str = "") -> str:
    """Prompt for text input with optional default."""
    suffix = f" [{default}]" if default else ""
    resp = console.input(f"[bold cyan]{prompt}{suffix}:[/] ").strip()
    return resp or default


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def _action_process_footage(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 1: Process new footage and generate CVAT tasks."""
    from training_pipeline.background import (
        build_background_model,
        detect_foreground,
        load_background_model,
    )
    from training_pipeline.export_to_cvat import export_video_task
    from training_pipeline.pseudo_labels import generate_pseudo_labels

    # Prompt for input
    clip_dir = _prompt_text("Path to clip(s) or session directory")
    if not clip_dir:
        console.print("[red]No path provided.[/]")
        return

    clip_dir = Path(clip_dir)
    session_id = _prompt_text("Session ID", default=clip_dir.stem)

    # Find clips
    if clip_dir.is_file():
        clips = [clip_dir]
    else:
        clips = sorted(clip_dir.glob("*.mp4"))
        if not clips:
            # Recursive search for nested camera directories
            # (e.g., data/raw/nest/{gym}/{cam}/{date}/{hour}/*.mp4).
            # Exclude pipeline output artifacts (_debug, exports, stage_*).
            _exclude = {"_debug", "exports", "stage_A", "stage_B", "stage_C",
                        "stage_D", "stage_E", "stage_F"}
            clips = sorted(
                p for p in clip_dir.rglob("*.mp4")
                if not _exclude.intersection(p.parts)
            )

    if not clips:
        console.print(f"[red]No MP4 clips found in {clip_dir}[/]")
        return

    console.print(f"Found {len(clips)} clip(s)")

    # Determine camera IDs from clip filenames or parent dirs
    cam_clips: dict[str, list[Path]] = {}
    for clip in clips:
        # Try to extract camera ID from filename (e.g., FP7oJQ-20260318-200014.mp4)
        cam_id = clip.stem.split("-")[0] if "-" in clip.stem else clip.parent.name
        cam_clips.setdefault(cam_id, []).append(clip)

    console.print(f"Cameras: {', '.join(cam_clips.keys())}")

    # Step 1: Run Stage A inference with current model
    current_model = _current_model(state, cfg)
    console.print(f"\n[bold]Step 1:[/] Running Stage A inference with {current_model}")

    from ultralytics import YOLO
    model = YOLO(current_model)

    stage_a_outputs: dict[str, Path] = {}
    for cam_id, cam_clip_list in cam_clips.items():
        for clip in cam_clip_list:
            out_dir = cfg.cvat_tasks_dir / f"{session_id}_{cam_id}" / "stage_a"
            out_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"  Inference on {clip.name} ({cam_id})...")
            _run_stage_a_inference(model, clip, cam_id, out_dir, cfg)
            stage_a_outputs[cam_id] = out_dir

    # Step 2: Background subtraction (if models exist)
    console.print("\n[bold]Step 2:[/] Background subtraction")
    bg_detections: dict[str, dict] = {}
    for cam_id in cam_clips:
        try:
            bg_model = load_background_model(cam_id, cfg.background_models_dir)
            console.print(f"  {cam_id}: background model loaded")
            # We'd process frames here — simplified for initial build
            bg_detections[cam_id] = {}
        except FileNotFoundError:
            console.print(f"  {cam_id}: no background model (skipping)")

    # Step 3: Cross-camera pseudo-labeling
    pseudo_labels = {}
    if len(stage_a_outputs) > 1:
        console.print("\n[bold]Step 3:[/] Cross-camera pseudo-labeling")
        try:
            pseudo_labels = generate_pseudo_labels(
                stage_a_outputs,
                iou_threshold=cfg.cross_camera_iou_threshold,
            )
            for cam_id, labels in pseudo_labels.items():
                console.print(f"  {cam_id}: {len(labels)} pseudo-labels")
        except Exception as e:
            console.print(f"  [yellow]Pseudo-labeling failed: {e}[/]")
    else:
        console.print("\n[bold]Step 3:[/] Single camera — skipping pseudo-labeling")

    # Step 4: Export CVAT tasks (video mode with track interpolation)
    console.print(f"\n[bold]Step 4:[/] Exporting CVAT video tasks (keyframe interval={cfg.keyframe_interval})")
    task_count = 0
    exported_tasks: dict[str, Tuple[Path, Path, Path]] = {}  # name -> (video, xml, clip)
    for cam_id, cam_clip_list in cam_clips.items():
        for clip in cam_clip_list:
            stage_a_path = stage_a_outputs.get(cam_id)
            if stage_a_path is None:
                continue

            pl = pseudo_labels.get(cam_id, [])
            bg = bg_detections.get(cam_id, {})

            video_path, xml_path = export_video_task(
                clip_path=clip,
                cam_id=cam_id,
                session_id=session_id,
                stage_a_path=stage_a_path,
                pseudo_labels=pl if pl else None,
                bg_detections=bg if bg else None,
                keyframe_interval=cfg.keyframe_interval,
            )
            task_name = f"{session_id}_{cam_id}"
            exported_tasks[task_name] = (video_path, xml_path, clip)
            console.print(f"  Created: {xml_path.parent.name}/")
            task_count += 1

    # Step 5: Upload to CVAT (if configured)
    if cfg.cvat_username and cfg.cvat_password:
        if _prompt_confirm("Upload tasks to CVAT?"):
            _upload_video_to_cvat(cfg, state, session_id, exported_tasks)
    else:
        console.print(
            "\n[yellow]CVAT credentials not configured. "
            "Run option 8 to configure, then upload manually.[/]"
        )

    console.print(
        f"\n[green]Created {task_count} tasks. "
        f"Start annotating camera-by-camera in CVAT.[/]"
    )


def _run_stage_a_inference(
    model, clip: Path, cam_id: str, out_dir: Path, cfg: TrainingPipelineConfig
) -> None:
    """Run YOLO inference and save detections/keypoints as parquet."""
    import cv2
    import numpy as np
    import pandas as pd

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        logger.warning(f"Cannot open {clip}")
        return

    det_rows = []
    kp_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preds = model.predict(source=frame, verbose=False, conf=0.25, device=cfg.device)
        r0 = preds[0] if preds else None
        boxes_obj = getattr(r0, "boxes", None) if r0 is not None else None

        if boxes_obj is not None and len(boxes_obj) > 0:
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            clses = boxes_obj.cls.cpu().numpy()

            kps_obj = getattr(r0, "keypoints", None)
            kps_data = None
            if kps_obj is not None and hasattr(kps_obj, "data") and kps_obj.data is not None:
                kps_data = kps_obj.data.cpu().numpy()

            for k in range(len(xyxy)):
                if int(clses[k]) != 0:
                    continue
                # tracklet_id here is a per-frame detection index, NOT a real
                # tracker-assigned tracklet ID. This is consistent within the
                # training pipeline's parquet files (detections + keypoints use
                # the same per-frame index) but does NOT match Stage A pipeline output.
                det_rows.append({
                    "frame_index": frame_idx,
                    "tracklet_id": k,
                    "x1": float(xyxy[k, 0]),
                    "y1": float(xyxy[k, 1]),
                    "x2": float(xyxy[k, 2]),
                    "y2": float(xyxy[k, 3]),
                    "confidence": float(confs[k]),
                })

                if kps_data is not None and k < kps_data.shape[0]:
                    kp_names = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle",
                    ]
                    kp_row = {"frame_index": frame_idx, "track_id": k}
                    for j, name in enumerate(kp_names):
                        kp_row[f"kp_{name}_x"] = float(kps_data[k, j, 0])
                        kp_row[f"kp_{name}_y"] = float(kps_data[k, j, 1])
                        kp_row[f"kp_{name}_conf"] = float(kps_data[k, j, 2])
                    kp_rows.append(kp_row)

        frame_idx += 1

    cap.release()

    if det_rows:
        pd.DataFrame(det_rows).to_parquet(out_dir / "detections.parquet", index=False)
    if kp_rows:
        pd.DataFrame(kp_rows).to_parquet(out_dir / "keypoints.parquet", index=False)

    logger.info(f"Stage A: {frame_idx} frames, {len(det_rows)} detections -> {out_dir}")


def _upload_to_cvat(
    cfg: TrainingPipelineConfig,
    state: PipelineState,
    session_id: str,
    cam_clips: dict[str, list[Path]],
) -> None:
    """Upload exported CVAT tasks."""
    from training_pipeline.cvat_integration import connect, create_project_if_needed, upload_task

    try:
        client = connect(cfg.cvat_url, cfg.cvat_username, cfg.cvat_password)
    except Exception as e:
        console.print(f"[red]CVAT connection failed: {e}[/]")
        return

    project_id = create_project_if_needed(client, cfg.cvat_project_name)

    for cam_id in cam_clips:
        task_dir = cfg.cvat_tasks_dir / f"{session_id}_{cam_id}"
        zip_path = task_dir / f"{session_id}_{cam_id}.zip"
        if not zip_path.exists():
            continue

        task_name = f"{session_id}_{cam_id}"
        try:
            task_id = upload_task(client, task_name, zip_path, project_id)
            state.cvat_tasks[task_name] = CvatTaskInfo(
                task_id=task_id, status="created",
                session_id=session_id, cam_id=cam_id,
            )
            console.print(f"  Uploaded: {task_name} (task_id={task_id})")
        except Exception as e:
            console.print(f"  [red]Upload failed for {task_name}: {e}[/]")

    save_state(state)


def _upload_video_to_cvat(
    cfg: TrainingPipelineConfig,
    state: PipelineState,
    session_id: str,
    exported_tasks: dict[str, Tuple[Path, Path, Path]],
) -> None:
    """Upload video tasks with track annotations to CVAT."""
    from training_pipeline.cvat_integration import (
        connect,
        create_project_if_needed,
        upload_video_task,
    )

    try:
        client = connect(cfg.cvat_url, cfg.cvat_username, cfg.cvat_password)
    except Exception as e:
        console.print(f"[red]CVAT connection failed: {e}[/]")
        return

    project_id = create_project_if_needed(client, cfg.cvat_project_name)

    for task_name, (video_path, xml_path, clip_path) in exported_tasks.items():
        # Parse cam_id from task name
        parts = task_name.rsplit("_", 1)
        cam_id = parts[1] if len(parts) > 1 else ""

        try:
            task_id = upload_video_task(client, task_name, video_path, xml_path, project_id)
            state.cvat_tasks[task_name] = CvatTaskInfo(
                task_id=task_id, status="created",
                session_id=session_id, cam_id=cam_id,
                video_path=str(clip_path),
            )
            console.print(f"  Uploaded: {task_name} (task_id={task_id})")
        except Exception as e:
            console.print(f"  [red]Upload failed for {task_name}: {e}[/]")

    save_state(state)


def _action_check_progress(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 2: Check CVAT annotation progress."""
    if not state.cvat_tasks:
        console.print("[yellow]No CVAT tasks tracked.[/]")
        return

    if not cfg.cvat_username:
        # Show local state only
        table = Table(title="CVAT Tasks (local state)")
        table.add_column("Task")
        table.add_column("Task ID")
        table.add_column("Status")
        for name, info in state.cvat_tasks.items():
            table.add_row(name, str(info.task_id), info.status)
        console.print(table)
        return

    from training_pipeline.cvat_integration import check_task_status, connect

    try:
        client = connect(cfg.cvat_url, cfg.cvat_username, cfg.cvat_password)
    except Exception as e:
        console.print(f"[red]CVAT connection failed: {e}[/]")
        return

    table = Table(title="CVAT Annotation Progress")
    table.add_column("Task")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Assignee")
    table.add_column("Frames")
    table.add_column("Updated")

    for name, info in state.cvat_tasks.items():
        try:
            status = check_task_status(client, info.task_id)
            info.status = status.status
            table.add_row(
                name, str(info.task_id), status.status,
                status.assignee, str(status.frame_count), status.updated_date,
            )
        except Exception as e:
            table.add_row(name, str(info.task_id), f"[red]error: {e}[/]", "", "", "")

    console.print(table)
    save_state(state)


def _action_pull_annotations(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 3: Pull completed annotations from CVAT.

    Handles both video tasks (CVAT XML + frame extraction) and legacy
    image tasks (COCO JSON).
    """
    from training_pipeline.cvat_integration import (
        connect,
        download_annotations,
        download_video_annotations,
    )
    from training_pipeline.dataset import (
        cvat_video_xml_to_coco,
        extract_frames_from_video,
        ingest_annotations,
        validate_annotations,
    )

    if not cfg.cvat_username:
        console.print("[red]CVAT credentials not configured. Run option 8.[/]")
        return

    # Find completed tasks
    completed = {
        name: info for name, info in state.cvat_tasks.items()
        if info.status == "completed"
    }

    if not completed:
        console.print("[yellow]No completed CVAT tasks found. Check progress first (option 2).[/]")
        return

    try:
        client = connect(cfg.cvat_url, cfg.cvat_username, cfg.cvat_password)
    except Exception as e:
        console.print(f"[red]CVAT connection failed: {e}[/]")
        return

    for name, info in completed.items():
        console.print(f"\nPulling annotations for: {name}")

        # Use stored session_id/cam_id; fall back to name parsing for legacy state
        session_id = info.session_id or name.rsplit("_", 1)[0]
        cam_id = info.cam_id or (name.rsplit("_", 1)[1] if "_" in name else "unknown")

        is_video_task = bool(info.video_path)

        if is_video_task:
            # Video task: download CVAT XML, convert to COCO, extract frames
            output_xml = cfg.training_data_dir / "annotations" / f"{name}_corrected.xml"
            try:
                download_video_annotations(client, info.task_id, output_xml)
            except Exception as e:
                console.print(f"  [red]Download failed: {e}[/]")
                continue

            video_path = Path(info.video_path)
            if not video_path.exists():
                console.print(f"  [red]Source video not found: {video_path}[/]")
                continue

            # Convert CVAT XML to COCO JSON
            console.print("  Converting CVAT XML to COCO format...")
            coco_data = cvat_video_xml_to_coco(output_xml, video_path)
            output_json = cfg.training_data_dir / "annotations" / f"{name}_corrected.json"
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(coco_data, indent=2))
            console.print(
                f"  Converted: {len(coco_data['images'])} frames, "
                f"{len(coco_data['annotations'])} annotations"
            )

            # Extract frames from video
            frame_indices = [img["id"] for img in coco_data["images"]]
            images_dir = cfg.cvat_tasks_dir / name / "images"
            console.print(f"  Extracting {len(frame_indices)} frames from video...")
            extract_frames_from_video(video_path, frame_indices, images_dir)
        else:
            # Legacy image task: download COCO JSON directly
            output_json = cfg.training_data_dir / "annotations" / f"{name}_corrected.json"
            try:
                download_annotations(client, info.task_id, output_json)
            except Exception as e:
                console.print(f"  [red]Download failed: {e}[/]")
                continue
            images_dir = cfg.cvat_tasks_dir / name / "images"

        # Validate
        report = validate_annotations(output_json)
        errors = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        if warnings:
            console.print(f"  [dim]Validation: {len(warnings)} warnings (non-blocking)[/]")
        if errors:
            console.print(f"  [yellow]Validation: {len(errors)} errors found[/]")
            for issue in errors[:5]:
                console.print(f"    - {issue.issue_type}: {issue.details}")
            if len(errors) > 5:
                console.print(f"    ... and {len(errors) - 5} more")
            if not _prompt_confirm("Proceed with ingestion despite errors?"):
                continue
        else:
            console.print("  [green]Validation: clean[/]")

        # Ingest into dataset
        stats = ingest_annotations(
            output_json, session_id, cam_id, images_dir,
            data_dir=cfg.training_data_dir,
            round_num=state.current_round + 1,
        )
        record_session(state, name, frame_count=stats.frames_added)
        console.print(
            f"  Ingested: {stats.frames_added} frames, "
            f"{stats.annotations_added} annotations"
        )

    save_state(state)


def _action_view_stats(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 4: View dataset statistics."""
    from training_pipeline.dataset import get_stats

    stats = get_stats(cfg.training_data_dir)

    console.print(f"\n[bold]Dataset Statistics[/]")
    console.print(f"  Total frames: {stats.total_frames}")
    console.print(f"  Total annotations: {stats.total_annotations}")
    console.print(f"  Sessions: {len(stats.sessions)}")

    if stats.frames_per_camera:
        table = Table(title="Frames per Camera")
        table.add_column("Camera")
        table.add_column("Frames", justify="right")
        for cam, count in sorted(stats.frames_per_camera.items()):
            table.add_row(cam, str(count))
        console.print(table)

    if stats.frames_per_round:
        table = Table(title="Frames per Round")
        table.add_column("Round")
        table.add_column("Frames", justify="right")
        for r, count in sorted(stats.frames_per_round.items()):
            table.add_row(str(r), str(count))
        console.print(table)


def _action_train(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 5: Run training."""
    from training_pipeline.dataset import generate_dataset_yaml, get_stats
    from training_pipeline.train import suggest_hyperparameters, train_model

    stats = get_stats(cfg.training_data_dir)
    if stats.total_frames == 0:
        console.print("[red]No training data available. Process and annotate footage first.[/]")
        return

    round_num = state.current_round + 1
    total_frames = stats.total_frames
    base_model = _current_model(state, cfg)

    # Show suggested hyperparameters
    hp = suggest_hyperparameters(round_num, total_frames)
    console.print(f"\n[bold]Training Round {round_num}[/]")
    console.print(f"  Base model: {base_model}")
    console.print(f"  Total frames: {total_frames}")
    console.print(f"  {hp.reason}")
    console.print(f"  freeze={hp.freeze}, epochs={hp.epochs}, lr0={hp.lr0}")

    # Ask for overrides
    overrides = None
    if _prompt_confirm("Override hyperparameters?"):
        freeze = _prompt_text("  freeze", str(hp.freeze))
        epochs = _prompt_text("  epochs", str(hp.epochs))
        lr0 = _prompt_text("  lr0", str(hp.lr0))
        overrides = {
            "freeze": int(freeze),
            "epochs": int(epochs),
            "lr0": float(lr0),
        }

    # Validation split
    val_sessions = []
    if state.sessions_validation:
        console.print(f"  Validation sessions: {state.sessions_validation}")
    else:
        val_input = _prompt_text("  Validation session ID(s) (comma-separated)")
        if val_input:
            val_sessions = [s.strip() for s in val_input.split(",")]
            state.sessions_validation = val_sessions

    if not _prompt_confirm("Start training?"):
        return

    # Generate dataset
    dataset_yaml = generate_dataset_yaml(
        state.sessions_validation,
        cfg.training_data_dir,
    )

    # Train
    console.print("\n[bold]Training started...[/]")
    result = train_model(
        dataset_yaml=dataset_yaml,
        base_model=base_model,
        round_num=round_num,
        total_frames=total_frames,
        output_dir=cfg.training_runs_dir / f"round_{round_num}",
        overrides=overrides,
        device=cfg.device,
        imgsz=cfg.imgsz,
        batch_size=cfg.batch_size,
    )

    # Record round
    info = RoundInfo(
        round=round_num,
        model=result.model_path,
        base_model=base_model,
        total_frames=total_frames,
        training_date=str(__import__("datetime").date.today()),
        val_session=",".join(state.sessions_validation),
        metrics={"mAP50": result.mAP50, "mAP50_95": result.mAP50_95},
        freeze=overrides.get("freeze", hp.freeze) if overrides else hp.freeze,
        epochs=overrides.get("epochs", hp.epochs) if overrides else hp.epochs,
        notes=f"Round {round_num} — {hp.reason}",
    )
    record_round(state, info)
    save_state(state)

    console.print(f"\n[green]Training complete![/]")
    console.print(f"  Model: {result.model_path}")
    console.print(f"  mAP50: {result.mAP50:.4f}")
    console.print(f"  mAP50-95: {result.mAP50_95:.4f}")
    console.print(f"  Time: {result.training_time_s:.0f}s")

    # Compare to previous round
    if len(state.rounds) >= 2:
        prev = state.rounds[-2]
        delta = result.mAP50 - prev.metrics.get("mAP50", 0)
        sign = "+" if delta >= 0 else ""
        console.print(f"  vs Round {prev.round}: {sign}{delta:.4f} mAP50")


def _action_evaluate(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 6: Evaluate model with diff video and metrics."""
    from training_pipeline.evaluate import compare_rounds, generate_diff_video

    if not state.rounds:
        console.print("[red]No trained models yet. Run training first (option 5).[/]")
        return

    # Determine old vs new model
    current = state.rounds[-1]
    if len(state.rounds) >= 2:
        old_model = state.rounds[-2].model
        old_round = state.rounds[-2].round
    else:
        old_model = cfg.base_model
        old_round = 0

    new_model = current.model
    new_round = current.round

    console.print(f"\n[bold]Evaluation[/]")
    console.print(f"  Old: Round {old_round} ({old_model})")
    console.print(f"  New: Round {new_round} ({new_model})")

    # Diff video
    clip_path = _prompt_text("Clip path for diff video")
    if clip_path and Path(clip_path).exists():
        output = Path(f"outputs/_benchmarks/diff_r{old_round}_vs_r{new_round}.mp4")
        console.print("Generating diff video...")
        generate_diff_video(
            old_model, new_model, clip_path, output,
            device=cfg.device,
        )
        console.print(f"  [green]Diff video: {output}[/]")

    # Metrics comparison (if validation dataset exists)
    dataset_yaml = cfg.training_data_dir / "dataset.yaml"
    if dataset_yaml.exists():
        console.print("Computing metrics comparison...")
        report = compare_rounds(
            old_model, new_model, dataset_yaml,
            round_a_num=old_round, round_b_num=new_round,
            device=cfg.device,
        )
        table = Table(title="Metrics Comparison")
        table.add_column("Metric")
        table.add_column(f"Round {old_round}", justify="right")
        table.add_column(f"Round {new_round}", justify="right")
        table.add_column("Delta", justify="right")
        table.add_row(
            "mAP50",
            f"{report.metrics_a.mAP50:.4f}",
            f"{report.metrics_b.mAP50:.4f}",
            f"{report.mAP50_delta:+.4f}",
        )
        table.add_row(
            "mAP50-95",
            f"{report.metrics_a.mAP50_95:.4f}",
            f"{report.metrics_b.mAP50_95:.4f}",
            f"{report.mAP50_95_delta:+.4f}",
        )
        console.print(table)

        if report.improved:
            console.print(
                f"[green]Round {new_round} shows improvement. "
                f"Consider promoting (option 7).[/]"
            )
        else:
            console.print(f"[yellow]Round {new_round} shows regression. Review carefully.[/]")


def _action_promote(cfg: TrainingPipelineConfig, state: PipelineState) -> None:
    """Option 7: Promote model to pipeline default."""
    if not state.rounds:
        console.print("[red]No trained models to promote.[/]")
        return

    current = state.rounds[-1]
    model_path = Path(current.model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/]")
        return

    round_num = current.round
    target_name = f"bjj-pose-r{round_num}.pt"
    target_path = Path("models") / target_name

    console.print(f"\n[bold]Promote Model[/]")
    console.print(f"  Source: {model_path}")
    console.print(f"  Target: {target_path}")

    if not _prompt_confirm("Proceed?"):
        return

    # Copy model
    shutil.copy2(model_path, target_path)
    console.print(f"  Copied to {target_path}")

    # Update symlink
    link_path = Path("models/active_model")
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_name)
    console.print(f"  Symlink: models/active_model -> {target_name}")

    # Optionally update default.yaml
    if _prompt_confirm("Update configs/default.yaml to use new model?"):
        import yaml

        default_yaml = Path("configs/default.yaml")
        if default_yaml.exists():
            config_data = yaml.safe_load(default_yaml.read_text())
            stages = config_data.setdefault("stages", {})
            stage_a = stages.setdefault("stage_A", {})
            detector = stage_a.setdefault("detector", {})
            detector["model_path"] = f"models/{target_name}"
            default_yaml.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))
            console.print(f"  Updated configs/default.yaml")

    console.print(f"\n[green]Model promoted. Pipeline will now use {target_name} for inference.[/]")


def _action_configure(cfg: TrainingPipelineConfig) -> TrainingPipelineConfig:
    """Option 8: Configure settings."""
    console.print("\n[bold]Current Settings:[/]")
    for key, value in cfg.model_dump().items():
        console.print(f"  {key}: {value}")

    console.print("\nEditable settings:")
    console.print("  1. CVAT URL")
    console.print("  2. CVAT username")
    console.print("  3. CVAT password")
    console.print("  4. Device")
    console.print("  5. Base model")
    console.print("  0. Back")

    choice = _prompt_choice("Setting to change", {"0", "1", "2", "3", "4", "5"})

    if choice == "1":
        cfg.cvat_url = _prompt_text("CVAT URL", cfg.cvat_url)
    elif choice == "2":
        cfg.cvat_username = _prompt_text("CVAT username", cfg.cvat_username)
    elif choice == "3":
        cfg.cvat_password = _prompt_text("CVAT password")
    elif choice == "4":
        cfg.device = _prompt_text("Device (mps/cuda/cpu)", cfg.device)
    elif choice == "5":
        cfg.base_model = _prompt_text("Base model path", cfg.base_model)

    if choice != "0":
        save_config(cfg)
        console.print("[green]Settings saved.[/]")

    return cfg


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive training pipeline orchestrator."""
    cfg = load_config()
    state = load_state(cfg.training_data_dir / "pipeline_state.json")

    while True:
        current_model = _current_model(state, cfg)
        last_map = 0.0
        if state.rounds:
            last_map = state.rounds[-1].metrics.get("mAP50", 0.0)

        open_tasks = sum(1 for t in state.cvat_tasks.values() if t.status != "completed")
        completed_tasks = sum(1 for t in state.cvat_tasks.values() if t.status == "completed")

        console.print(f"\n[bold]Training Pipeline — Round {state.current_round + 1}[/]")
        console.print("=" * 50)
        console.print(f"Current model: {current_model}")
        if state.rounds:
            console.print(f"Last training: Round {state.current_round} (mAP50={last_map:.4f})")
        console.print(f"Open CVAT tasks: {open_tasks} ({completed_tasks} completed)")
        console.print(f"Total accumulated frames: {state.total_annotated_frames}")
        console.print()
        console.print("  1. Process new footage -> generate CVAT tasks")
        console.print("  2. Check CVAT annotation progress")
        console.print("  3. Pull completed annotations from CVAT")
        console.print("  4. View dataset statistics")
        console.print("  5. Run training")
        console.print("  6. Evaluate model — diff video + metrics")
        console.print("  7. Promote model to pipeline default")
        console.print("  8. Configure settings")
        console.print("  0. Exit")

        choice = _prompt_choice("\nChoice", {"0", "1", "2", "3", "4", "5", "6", "7", "8"})

        if choice == "0":
            console.print("[dim]Goodbye.[/]")
            break
        elif choice == "1":
            _action_process_footage(cfg, state)
        elif choice == "2":
            _action_check_progress(cfg, state)
        elif choice == "3":
            _action_pull_annotations(cfg, state)
        elif choice == "4":
            _action_view_stats(cfg, state)
        elif choice == "5":
            _action_train(cfg, state)
        elif choice == "6":
            _action_evaluate(cfg, state)
        elif choice == "7":
            _action_promote(cfg, state)
        elif choice == "8":
            cfg = _action_configure(cfg)


if __name__ == "__main__":
    main()
