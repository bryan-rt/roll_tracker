"""Processor service: polls for new recordings and invokes bjj_pipeline.

Two-phase architecture:
  Phase 1 (parallel): Stages A+C via multiplex_AC, one worker per camera
  Phase 2 (sequential): Stages D+E+F, one clip at a time

Phase 1/Phase 2 boundary is load-bearing for future cross-clip global
stitching — do not parallelize Stage D+E+F under any circumstances.
"""

import ctypes
import ctypes.util
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def _pin_to_performance_cores() -> None:
    """Request P-core scheduling via macOS QoS. No-op on non-Apple platforms."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        QOS_CLASS_USER_INITIATED = 0x19
        libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)
    except Exception:
        pass  # Non-macOS or unavailable — silently skip

from config import ProcessorSettings

# bjj_pipeline is installed as a package in the container
from bjj_pipeline.stages.orchestration.pipeline import (
    IngestPathInfo,
    compute_output_root,
    run_pipeline,
    validate_ingest_path,
    PipelineError,
)
from bjj_pipeline.stages.orchestration.cli import _load_config
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout

# Keywords in PipelineError messages that indicate an empty/uninteresting clip
# rather than a real pipeline failure. Case-insensitive matching.
_SKIP_KEYWORDS = (
    "no detections",
    "empty",
    "no person",
    "no match",
    "tracker output shape: (0",
    "no valid frames",
    "no tracklets",
    "positive definite",
)


def _log(event: str, **kwargs) -> None:
    """Emit a structured JSON log line to stdout."""
    record = {"event": event, "ts": time.time(), **kwargs}
    print(json.dumps(record), flush=True)


def _derive_cam_id(mp4_path: Path, settings: ProcessorSettings) -> str | None:
    """Extract cam_id from the path directory structure."""
    parts = mp4_path.resolve().parts
    try:
        idx = parts.index("nest")
    except ValueError:
        return None
    remaining = parts[idx + 1:]

    def _is_date(s: str) -> bool:
        return len(s) == 10 and s[4] == "-" and s[7] == "-"

    if len(remaining) >= 4 and _is_date(remaining[1]):
        return remaining[0]
    elif len(remaining) >= 5 and _is_date(remaining[2]):
        return remaining[1]
    return None


def _get_clip_output_root(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> Path | None:
    """Get the scoped output root for a clip."""
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        return compute_output_root(info, base_root=settings.OUTPUT_ROOT)
    except PipelineError:
        return None


def _get_layout(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> ClipOutputLayout | None:
    """Get the ClipOutputLayout for a clip."""
    scoped_root = _get_clip_output_root(mp4_path, cam_id, settings)
    if scoped_root is None:
        return None
    return ClipOutputLayout(clip_id=mp4_path.stem, root=scoped_root)


# --- Sentinel helpers ---

def _mark_processing(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> None:
    """Write .processing sentinel in the clip's output directory."""
    scoped_root = _get_clip_output_root(mp4_path, cam_id, settings)
    if scoped_root is None:
        return
    clip_dir = scoped_root / mp4_path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / ".processing").touch()


def _is_processing(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> bool:
    """Check if .processing sentinel exists."""
    scoped_root = _get_clip_output_root(mp4_path, cam_id, settings)
    if scoped_root is None:
        return False
    return (scoped_root / mp4_path.stem / ".processing").exists()


def _clear_processing(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> None:
    """Remove .processing sentinel."""
    scoped_root = _get_clip_output_root(mp4_path, cam_id, settings)
    if scoped_root is None:
        return
    sentinel = scoped_root / mp4_path.stem / ".processing"
    sentinel.unlink(missing_ok=True)


def _is_phase1_complete(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> bool:
    """Check if Phase 1 (A+C) is complete by looking for Stage C output artifacts."""
    layout = _get_layout(mp4_path, cam_id, settings)
    if layout is None:
        return False
    return layout.tag_observations_jsonl().exists()


def _is_already_processed(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> bool:
    """Check if the clip is fully processed (Stage F done or uploaded)."""
    layout = _get_layout(mp4_path, cam_id, settings)
    if layout is None:
        return False
    stage_f_dir = layout.export_manifest_jsonl().parent
    if layout.export_manifest_jsonl().exists():
        return True
    if (stage_f_dir / ".uploaded").exists():
        return True
    return False


def _is_recent_clip(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> bool:
    """Return True if the clip is recent enough or has a .processing sentinel."""
    # Never age out an in-progress clip
    if _is_processing(mp4_path, cam_id, settings):
        return True
    if settings.MAX_CLIP_AGE_HOURS <= 0:
        return True
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        folder_dt = datetime.strptime(
            f"{info.date_str} {info.hour_str}", "%Y-%m-%d %H"
        ).replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(tz=timezone.utc) - folder_dt).total_seconds() / 3600.0
        return age_hours <= settings.MAX_CLIP_AGE_HOURS
    except Exception:
        return True


def _discover_clips(settings: ProcessorSettings) -> list[Path]:
    """Find all MP4 files under SCAN_ROOT, optionally filtered by GYM_ID."""
    scan = settings.SCAN_ROOT
    if settings.GYM_ID:
        scan = scan / settings.GYM_ID
    clips = sorted(scan.glob("**/*.mp4"))
    return [c for c in clips if "/diag/" not in str(c) and "\\diag\\" not in str(c)]


def _build_config(cam_id: str, mp4_path: Path, device_override: str | None = None):
    """Load config and apply gym_id + device overlays."""
    cfg, cfg_hash, cfg_sources = _load_config(cam_id, None)
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        if info.gym_id:
            stages_blk = cfg.setdefault("stages", {})
            stage_f_blk = stages_blk.setdefault("stage_F", {})
            stage_f_blk.setdefault("gym_id", info.gym_id)
    except PipelineError:
        pass
    # Override device if specified
    if device_override and device_override != "auto":
        stages_blk = cfg.setdefault("stages", {})
        stage_a_blk = stages_blk.setdefault("stage_A", {})
        det_blk = stage_a_blk.setdefault("detector", {})
        det_blk["device"] = device_override
    return cfg, cfg_hash, cfg_sources


# --- Phase 1: Stages A+C (parallel) ---

def _process_clip_phase1(mp4_path_str: str, cam_id: str, settings_dict: dict) -> dict:
    """Run Stages A+C on a single clip. Runs in a subprocess via ProcessPoolExecutor."""
    mp4_path = Path(mp4_path_str)
    settings = ProcessorSettings(**settings_dict)
    cfg, cfg_hash, cfg_sources = _build_config(cam_id, mp4_path, settings.PARALLEL_DEVICE)
    try:
        run_pipeline(
            ingest_path=mp4_path,
            camera_id=cam_id,
            config=cfg,
            out_root=settings.OUTPUT_ROOT,
            config_sources=cfg_sources,
            config_hash_override=cfg_hash,
            to_stage="C",
            mode="multiplex_AC",
            visualize=settings.VISUALIZE,
        )
        return {"status": "completed", "clip": mp4_path_str, "cam_id": cam_id}
    except PipelineError as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in _SKIP_KEYWORDS):
            return {"status": "skipped", "clip": mp4_path_str, "cam_id": cam_id, "reason": str(e)}
        raise
    except Exception:
        raise


# --- Phase 2: Stages D+E+F (sequential) ---

def _process_clip_phase2(mp4_path: Path, cam_id: str, settings: ProcessorSettings) -> None:
    """Run Stages D+E+F sequentially on a single clip."""
    cfg, cfg_hash, cfg_sources = _build_config(cam_id, mp4_path, settings.SEQUENTIAL_DEVICE)
    run_pipeline(
        ingest_path=mp4_path,
        camera_id=cam_id,
        config=cfg,
        out_root=settings.OUTPUT_ROOT,
        config_sources=cfg_sources,
        config_hash_override=cfg_hash,
        from_stage="D",
        to_stage="F",
        mode="multiplex_AC",
        visualize=False,
    )


def main() -> None:
    settings = ProcessorSettings()
    _log("processor_started",
         scan_root=str(settings.SCAN_ROOT),
         output_root=str(settings.OUTPUT_ROOT),
         poll_interval=settings.POLL_INTERVAL_SECONDS,
         gym_id=settings.GYM_ID,
         max_clip_age_hours=settings.MAX_CLIP_AGE_HOURS,
         max_workers=settings.MAX_WORKERS,
         parallel_device=settings.PARALLEL_DEVICE,
         sequential_device=settings.SEQUENTIAL_DEVICE,
         visualize=settings.VISUALIZE)

    while True:
        try:
            clips = _discover_clips(settings)
            # Build work list: (mp4_path, cam_id) for unprocessed, recent clips
            work = []
            for mp4 in clips:
                cam_id = _derive_cam_id(mp4, settings)
                if cam_id is None:
                    continue
                if not _is_recent_clip(mp4, cam_id, settings):
                    continue
                if _is_already_processed(mp4, cam_id, settings):
                    continue
                work.append((mp4, cam_id))

            if not work:
                time.sleep(settings.POLL_INTERVAL_SECONDS)
                continue

            # --- Phase 1: parallel A+C ---
            phase1_needed = [(mp4, cam_id) for mp4, cam_id in work
                            if not _is_phase1_complete(mp4, cam_id, settings)]

            if phase1_needed:
                _log("phase1_started", clip_count=len(phase1_needed))
                # Mark all as processing before starting
                for mp4, cam_id in phase1_needed:
                    _mark_processing(mp4, cam_id, settings)

                # Serialize settings for subprocess
                settings_dict = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in settings.__dict__.items()
                }

                with ProcessPoolExecutor(max_workers=settings.MAX_WORKERS,
                                       initializer=_pin_to_performance_cores) as pool:
                    futures = {
                        pool.submit(_process_clip_phase1, str(mp4), cam_id, settings_dict): (mp4, cam_id)
                        for mp4, cam_id in phase1_needed
                    }
                    for future in as_completed(futures):
                        mp4, cam_id = futures[future]
                        try:
                            result = future.result()
                            if result["status"] == "completed":
                                _log("phase1_completed", clip=str(mp4), cam_id=cam_id)
                            elif result["status"] == "skipped":
                                _log("clip_skipped", clip=str(mp4), cam_id=cam_id,
                                     reason=result.get("reason", ""))
                                _clear_processing(mp4, cam_id, settings)
                        except Exception as e:
                            _log("phase1_error", clip=str(mp4), cam_id=cam_id,
                                 error=str(e), traceback=traceback.format_exc())

                _log("phase1_barrier", message="All Phase 1 workers finished")

            # --- Phase 2: sequential D+E+F ---
            phase2_needed = [(mp4, cam_id) for mp4, cam_id in work
                            if _is_phase1_complete(mp4, cam_id, settings)
                            and not _is_already_processed(mp4, cam_id, settings)]

            for mp4, cam_id in phase2_needed:
                _log("phase2_started", clip=str(mp4), cam_id=cam_id)
                try:
                    _process_clip_phase2(mp4, cam_id, settings)
                    _log("clip_completed", clip=str(mp4), cam_id=cam_id)
                    _clear_processing(mp4, cam_id, settings)
                except PipelineError as e:
                    err_str = str(e).lower()
                    if any(kw in err_str for kw in _SKIP_KEYWORDS):
                        _log("clip_skipped", clip=str(mp4), cam_id=cam_id, reason=str(e))
                        _clear_processing(mp4, cam_id, settings)
                    else:
                        _log("clip_error", clip=str(mp4), cam_id=cam_id,
                             error=str(e), traceback=traceback.format_exc())
                except Exception as e:
                    _log("clip_error", clip=str(mp4), cam_id=cam_id,
                         error=str(e), traceback=traceback.format_exc())

        except Exception as e:
            _log("poll_error", error=str(e), traceback=traceback.format_exc())

        time.sleep(settings.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
