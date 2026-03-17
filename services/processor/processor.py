"""Processor service: polls for new recordings and invokes bjj_pipeline."""

import json
import sys
import time
import traceback
from pathlib import Path

from config import ProcessorSettings

# bjj_pipeline is installed as a package in the container
from bjj_pipeline.stages.orchestration.pipeline import (
    IngestPathInfo,
    run_pipeline,
    validate_ingest_path,
    PipelineError,
)
from bjj_pipeline.stages.orchestration.cli import _load_config
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


def _log(event: str, **kwargs) -> None:
    """Emit a structured JSON log line to stdout."""
    record = {"event": event, "ts": time.time(), **kwargs}
    print(json.dumps(record), flush=True)


def _derive_cam_id(mp4_path: Path, settings: ProcessorSettings) -> str | None:
    """Extract cam_id from the path by trying validate_ingest_path with the
    directory name at the expected cam_id position."""
    parts = mp4_path.resolve().parts
    try:
        idx = parts.index("nest")
    except ValueError:
        return None
    remaining = parts[idx + 1:]

    def _is_date(s: str) -> bool:
        return len(s) == 10 and s[4] == "-" and s[7] == "-"

    if len(remaining) >= 4 and _is_date(remaining[1]):
        return remaining[0]  # OLD: nest/<cam>/<date>/<hour>/<file>
    elif len(remaining) >= 5 and _is_date(remaining[2]):
        return remaining[1]  # NEW: nest/<gym_id>/<cam>/<date>/<hour>/<file>
    return None


def _is_already_processed(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> bool:
    """Check if the clip's output directory already has a Stage F export manifest."""
    try:
        info = validate_ingest_path(mp4_path, cam_id)
    except PipelineError:
        return False

    base_root = settings.OUTPUT_ROOT
    if info.gym_id is not None:
        scoped_root = base_root / info.gym_id / info.cam_id / info.date_str / info.hour_str
    else:
        scoped_root = base_root / "legacy" / info.cam_id / info.date_str / info.hour_str

    clip_id = mp4_path.stem
    layout = ClipOutputLayout(clip_id=clip_id, root=scoped_root)
    return layout.export_manifest_jsonl().exists()


def _discover_clips(settings: ProcessorSettings) -> list[Path]:
    """Find all MP4 files under SCAN_ROOT, optionally filtered by GYM_ID."""
    scan = settings.SCAN_ROOT
    if settings.GYM_ID:
        scan = scan / settings.GYM_ID

    clips = sorted(scan.glob("**/*.mp4"))
    # Exclude diag directory
    clips = [c for c in clips if "/diag/" not in str(c) and "\\diag\\" not in str(c)]
    return clips


def _process_clip(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> None:
    """Invoke the pipeline on a single clip."""
    cfg, cfg_hash, cfg_sources = _load_config(cam_id, None)

    # Config overlay: pass gym_id to Stage F as backup
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        if info.gym_id:
            stages_blk = cfg.setdefault("stages", {})
            stage_f_blk = stages_blk.setdefault("stage_F", {})
            stage_f_blk.setdefault("gym_id", info.gym_id)
    except PipelineError:
        pass

    run_pipeline(
        ingest_path=mp4_path,
        camera_id=cam_id,
        config=cfg,
        out_root=settings.OUTPUT_ROOT,
        config_sources=cfg_sources,
        config_hash_override=cfg_hash,
        to_stage="F",
    )


def main() -> None:
    settings = ProcessorSettings()
    _log("processor_started",
         scan_root=str(settings.SCAN_ROOT),
         output_root=str(settings.OUTPUT_ROOT),
         poll_interval=settings.POLL_INTERVAL_SECONDS,
         run_until=settings.RUN_UNTIL,
         gym_id=settings.GYM_ID)

    while True:
        try:
            clips = _discover_clips(settings)
            for mp4 in clips:
                cam_id = _derive_cam_id(mp4, settings)
                if cam_id is None:
                    continue

                if _is_already_processed(mp4, cam_id, settings):
                    continue

                _log("clip_started", clip=str(mp4), cam_id=cam_id)
                try:
                    _process_clip(mp4, cam_id, settings)
                    _log("clip_completed", clip=str(mp4), cam_id=cam_id)
                except Exception as e:
                    _log("clip_error", clip=str(mp4), cam_id=cam_id,
                         error=str(e), traceback=traceback.format_exc())

        except Exception as e:
            _log("poll_error", error=str(e), traceback=traceback.format_exc())

        time.sleep(settings.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
