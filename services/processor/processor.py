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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout, SessionOutputLayout
from bjj_pipeline.contracts.f0_models import SCHEMA_VERSION_DEFAULT
from bjj_pipeline.stages.stitch.session_d_run import (
    run_session_d,
    SessionStageLayoutAdapter,
)
from bjj_pipeline.stages.stitch.cross_camera_merge import run_cross_camera_merge

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


def _build_config(cam_id: str, mp4_path: Path, device_override: str | None = None,
                  config_overlay: str | None = None):
    """Load config and apply gym_id + device overlays."""
    overlay_path = Path(config_overlay) if config_overlay else None
    cfg, cfg_hash, cfg_sources = _load_config(cam_id, overlay_path)
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
    cfg, cfg_hash, cfg_sources = _build_config(cam_id, mp4_path, settings.PARALLEL_DEVICE, settings.CONFIG_OVERLAY)
    try:
        run_pipeline(
            ingest_path=mp4_path,
            camera_id=cam_id,
            config=cfg,
            out_root=settings.OUTPUT_ROOT,
            config_sources=cfg_sources,
            config_hash_override=cfg_hash,
            to_stage="C",
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
    cfg, cfg_hash, cfg_sources = _build_config(cam_id, mp4_path, settings.SEQUENTIAL_DEVICE, settings.CONFIG_OVERLAY)
    run_pipeline(
        ingest_path=mp4_path,
        camera_id=cam_id,
        config=cfg,
        out_root=settings.OUTPUT_ROOT,
        config_sources=cfg_sources,
        config_hash_override=cfg_hash,
        from_stage="D",
        to_stage="F",
        visualize=False,
    )


# --- Session state machine (CP14a) ---

_SESSION_PRE_BUFFER_MINUTES = 30  # clips arriving this many minutes before window start still match


def _parse_schedule_json(schedule_json_str: str) -> dict | None:
    """Parse SCHEDULE_JSON string. Returns parsed dict or None on failure."""
    try:
        d = json.loads(schedule_json_str)
        if not isinstance(d, dict):
            _log("schedule_parse_error", error="SCHEDULE_JSON is not a JSON object")
            return None
        if "timezone" not in d or "schedules" not in d:
            _log("schedule_parse_error", error="SCHEDULE_JSON missing 'timezone' or 'schedules'")
            return None
        return d
    except Exception as e:
        _log("schedule_parse_error", error=str(e))
        return None


def _clip_wall_clock_dt(mp4_path: Path, cam_id: str) -> datetime | None:
    """Derive UTC datetime from ingest path date+hour folder.

    Treats folder date/hour as UTC (conservative; avoids tz dependency in the
    ingest path itself).
    """
    try:
        info = validate_ingest_path(mp4_path, cam_id)
        return datetime.strptime(
            f"{info.date_str} {info.hour_str}", "%Y-%m-%d %H"
        ).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _find_session_for_clip(
    clip_dt: datetime,
    schedule: dict,
) -> tuple[str, str, str] | None:
    """Given a clip's wall-clock time (UTC), find the matching schedule window.

    Returns (date_str, session_id, session_end_iso) or None if no match.

    date_str: YYYY-MM-DD in gym's local timezone
    session_id: f"{date_str}T{start_hhmm_no_colon}" e.g. "2026-03-18T2000"
    session_end_iso: ISO string of session end in UTC (tz-aware, round-trips
                     through datetime.fromisoformat())
    """
    try:
        tz = ZoneInfo(schedule["timezone"])
    except Exception:
        return None

    local_dt = clip_dt.astimezone(tz)
    local_day_abbr = local_dt.strftime("%a")  # e.g. "Mon", "Tue"
    local_date_str = local_dt.strftime("%Y-%m-%d")

    for entry in schedule.get("schedules", []):
        days = entry.get("days", [])
        if local_day_abbr not in days:
            continue

        start_str = entry.get("start", "")  # "20:00"
        end_str = entry.get("end", "")      # "22:30"
        try:
            sh, sm = int(start_str.split(":")[0]), int(start_str.split(":")[1])
            eh, em = int(end_str.split(":")[0]), int(end_str.split(":")[1])
        except Exception:
            continue

        window_start = local_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
        window_end = local_dt.replace(hour=eh, minute=em, second=0, microsecond=0)

        # Handle windows that cross midnight
        if window_end <= window_start:
            window_end += timedelta(days=1)

        # Match if clip falls within [start - pre_buffer, end]
        pre_buffer = timedelta(minutes=_SESSION_PRE_BUFFER_MINUTES)
        if (window_start - pre_buffer) <= local_dt <= window_end:
            start_hhmm = f"{sh:02d}{sm:02d}"
            session_id = f"{local_date_str}T{start_hhmm}"
            session_end_utc = window_end.astimezone(timezone.utc)
            return local_date_str, session_id, session_end_utc.isoformat()

    return None


def _get_session_layout(
    session_id: str, date_str: str, gym_id: str, settings: ProcessorSettings
) -> SessionOutputLayout:
    """Construct SessionOutputLayout from session identifiers."""
    return SessionOutputLayout(
        gym_id=gym_id,
        date=date_str,
        session_id=session_id,
        root=settings.OUTPUT_ROOT,
    )


def _mark_camera_phase1_complete(
    session_layout: SessionOutputLayout, cam_id: str
) -> None:
    """Write .phase1_complete_{cam_id} sentinel."""
    session_layout.ensure_session_root()
    session_layout.phase1_complete_sentinel(cam_id).touch()


def _is_camera_phase1_complete(
    session_layout: SessionOutputLayout, cam_id: str
) -> bool:
    return session_layout.phase1_complete_sentinel(cam_id).exists()


def _is_session_ready(session_layout: SessionOutputLayout) -> bool:
    return session_layout.session_ready_sentinel().exists()


def _is_session_tag_required(session_layout: SessionOutputLayout) -> bool:
    return session_layout.tag_required_sentinel().exists()


def _is_session_uploaded(session_layout: SessionOutputLayout) -> bool:
    return session_layout.uploaded_sentinel().exists()


def _evaluate_session_readiness(
    session_id: str,
    date_str: str,
    session_end_utc: datetime,
    cam_ids: set[str],
    session_clips: list[tuple[Path, str]],
    settings: ProcessorSettings,
    schedule: dict | None,
) -> None:
    """Evaluate whether a session is ready for Phase 2.

    Called at the end of each poll cycle for each active session. Writes
    sentinels (.phase1_complete_{cam_id}, .session_ready, .tag_required)
    as gates are satisfied. Never raises.
    """
    try:
        # Resolve gym_id: prefer from ingest path, fall back to settings.GYM_ID
        gym_id = settings.GYM_ID
        for mp4, cam_id in session_clips:
            try:
                info = validate_ingest_path(mp4, cam_id)
                if info.gym_id:
                    gym_id = info.gym_id
                    break
            except Exception:
                continue
        if not gym_id:
            return

        session_layout = _get_session_layout(session_id, date_str, gym_id, settings)

        # Already decided or completed — skip
        if (_is_session_ready(session_layout)
                or _is_session_tag_required(session_layout)
                or session_layout.session_completed_sentinel().exists()):
            return

        # Step 1: Check per-camera Phase 1 completion
        cameras_complete: set[str] = set()
        cameras_incomplete: set[str] = set()
        for cam in cam_ids:
            cam_clips = [(mp4, cid) for mp4, cid in session_clips if cid == cam]
            all_done = all(
                _is_phase1_complete(mp4, cid, settings) for mp4, cid in cam_clips
            )
            if all_done and cam_clips:
                if not _is_camera_phase1_complete(session_layout, cam):
                    _mark_camera_phase1_complete(session_layout, cam)
                    _log("session_phase1_camera_complete",
                         session_id=session_id, cam_id=cam,
                         clip_count=len(cam_clips))
                cameras_complete.add(cam)
            else:
                cameras_incomplete.add(cam)

        all_cameras_done = len(cameras_incomplete) == 0 and len(cameras_complete) > 0
        now_utc = datetime.now(tz=timezone.utc)
        buffer = timedelta(minutes=settings.SESSION_END_BUFFER_MINUTES)
        wall_clock_passed = now_utc > (session_end_utc + buffer)

        # Step 2: Check if ready
        proceed = False
        if all_cameras_done and wall_clock_passed:
            proceed = True
        elif wall_clock_passed and len(cameras_complete) > 0:
            # Timeout: some cameras incomplete but wall clock passed
            for cam in cameras_incomplete:
                _log("session_camera_timeout",
                     session_id=session_id, cam_id=cam,
                     message="Wall-clock gate passed but camera Phase 1 incomplete")
            proceed = True

        if proceed:
            # Check tag observations across all session clips
            has_any_tags = False
            for mp4, cam_id in session_clips:
                clip_layout = _get_layout(mp4, cam_id, settings)
                if clip_layout and clip_layout.tag_observations_jsonl().exists():
                    tag_path = clip_layout.tag_observations_jsonl()
                    if tag_path.stat().st_size > 0:
                        has_any_tags = True
                        break

            if not has_any_tags:
                session_layout.ensure_session_root()
                session_layout.tag_required_sentinel().touch()
                _log("session_tag_required",
                     session_id=session_id,
                     cameras_complete=sorted(cameras_complete),
                     cameras_incomplete=sorted(cameras_incomplete))
            else:
                session_layout.ensure_session_root()
                session_layout.session_ready_sentinel().touch()
                _log("session_ready",
                     session_id=session_id,
                     cameras_complete=sorted(cameras_complete),
                     cameras_incomplete=sorted(cameras_incomplete))
        else:
            # Log status for observability
            remaining_s = max(0.0, ((session_end_utc + buffer) - now_utc).total_seconds())
            _log("session_readiness_status",
                 session_id=session_id,
                 cameras_complete=sorted(cameras_complete),
                 cameras_incomplete=sorted(cameras_incomplete),
                 wall_clock_remaining_s=round(remaining_s, 1),
                 all_cameras_done=all_cameras_done,
                 wall_clock_passed=wall_clock_passed)

    except Exception as e:
        _log("session_evaluation_error",
             session_id=session_id, error=str(e),
             traceback=traceback.format_exc())


def _run_session_phase2(
    session_id: str,
    date_str: str,
    gym_id: str,
    session_clips: list[tuple[Path, str]],
    settings: ProcessorSettings,
) -> None:
    """Run session-level Stages D → E → F for each camera that completed Phase 1.

    Invoked when .session_ready sentinel exists. Writes .processing sentinel
    at start, clears on completion or error. Never raises.
    """
    session_layout = _get_session_layout(session_id, date_str, gym_id, settings)

    try:
        # Write processing sentinel
        session_layout.processing_sentinel().parent.mkdir(parents=True, exist_ok=True)
        session_layout.processing_sentinel().touch()
        _log("session_phase2_start", session_id=session_id, gym_id=gym_id)

        # Build config from the first clip's camera
        cam_ids = sorted({cam_id for _, cam_id in session_clips})
        first_cam = cam_ids[0] if cam_ids else "unknown"
        first_mp4 = next((mp4 for mp4, cid in session_clips if cid == first_cam), None)
        if first_mp4:
            cfg, _, _ = _build_config(first_cam, first_mp4, settings.SEQUENTIAL_DEVICE, settings.CONFIG_OVERLAY)
        else:
            overlay_path = Path(settings.CONFIG_OVERLAY) if settings.CONFIG_OVERLAY else None
            cfg, _, _ = _load_config(first_cam, overlay_path)

        # --- Loop 1: all cameras → D + E ---
        adapters: dict[str, SessionStageLayoutAdapter] = {}
        manifests: dict[str, object] = {}

        for cam_id in cam_ids:
            if not _is_camera_phase1_complete(session_layout, cam_id):
                _log("session_d_skipped", session_id=session_id, cam_id=cam_id,
                     reason="phase1_not_complete")
                continue

            session_manifest = None

            # --- Stage D ---
            try:
                _log("session_d_start", session_id=session_id, cam_id=cam_id)
                session_manifest = run_session_d(
                    config=cfg,
                    session_layout=session_layout,
                    session_clips=session_clips,
                    cam_id=cam_id,
                    output_root=settings.OUTPUT_ROOT,
                )
                _log("session_d_completed", session_id=session_id, cam_id=cam_id)
            except PipelineError as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in _SKIP_KEYWORDS):
                    _log("session_d_skipped", session_id=session_id, cam_id=cam_id,
                         reason=str(e))
                else:
                    _log("session_d_error", session_id=session_id, cam_id=cam_id,
                         error=str(e), traceback=traceback.format_exc())
            except Exception as e:
                _log("session_d_error", session_id=session_id, cam_id=cam_id,
                     error=str(e), traceback=traceback.format_exc())

            adapter = SessionStageLayoutAdapter(session_layout, cam_id)
            adapters[cam_id] = adapter
            if session_manifest is not None:
                manifests[cam_id] = session_manifest

            # --- Stage E (session-level) ---
            if session_manifest is not None:
                try:
                    _log("session_e_start", session_id=session_id, cam_id=cam_id)
                    from bjj_pipeline.stages.matches.run import run as run_stage_e
                    run_stage_e(cfg, {"layout": adapter, "manifest": session_manifest})
                    _log("session_e_completed", session_id=session_id, cam_id=cam_id)
                except PipelineError as e:
                    err_str = str(e).lower()
                    if any(kw in err_str for kw in _SKIP_KEYWORDS):
                        _log("session_e_skipped", session_id=session_id, cam_id=cam_id,
                             reason=str(e))
                    else:
                        _log("session_e_error", session_id=session_id, cam_id=cam_id,
                             error=str(e), traceback=traceback.format_exc())
                except Exception as e:
                    _log("session_e_error", session_id=session_id, cam_id=cam_id,
                         error=str(e), traceback=traceback.format_exc())

        # --- Merge per-camera match_sessions into shared file ---
        stage_e_dir = session_layout.stage_dir("E")
        merged_matches_path = session_layout.session_match_sessions_jsonl()
        all_match_records: list[dict] = []
        for cam_id in cam_ids:
            cam_matches = stage_e_dir / f"match_sessions_{cam_id}.jsonl"
            if not cam_matches.exists():
                continue
            with cam_matches.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_match_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        merged_matches_path.parent.mkdir(parents=True, exist_ok=True)
        with merged_matches_path.open("w", encoding="utf-8") as f:
            for rec in all_match_records:
                f.write(json.dumps(rec, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False) + "\n")
        _log("session_match_merge_completed", session_id=session_id,
             n_cameras=len(cam_ids), total_matches=len(all_match_records))

        # --- Cross-camera identity merge (CP14f) ---
        global_id_map: dict[str, str] = {}
        if len(adapters) > 0:
            try:
                _log("cross_camera_merge_start", session_id=session_id,
                     cam_count=len(adapters))
                global_id_map = run_cross_camera_merge(
                    config=cfg,
                    session_layout=session_layout,
                    cam_ids=list(adapters.keys()),
                    adapter_map=adapters,
                )
                _log("cross_camera_merge_completed", session_id=session_id,
                     n_global_ids=len(global_id_map))
            except Exception as e:
                _log("cross_camera_merge_error", session_id=session_id,
                     error=str(e), traceback=traceback.format_exc())
                global_id_map = {}

        # --- CP17 Pass 2: re-solve with cross-camera tag corroboration ---
        cc_cfg = cfg.get("cross_camera", {})
        pass2_enabled = bool(cc_cfg.get("pass2_enabled", False))
        if pass2_enabled and len(adapters) >= 2 and len(manifests) >= 2:
            try:
                from bjj_pipeline.stages.stitch.cross_camera_evidence import (
                    build_cross_camera_tag_evidence,
                    build_cross_camera_coordinate_evidence,
                    build_cross_camera_histogram_evidence,
                )
                cc_evidence = build_cross_camera_tag_evidence(
                    cam_ids=list(adapters.keys()),
                    adapter_map=adapters,
                )
                n_corr = cc_evidence.get("n_corroborated_tags", 0)
                _log("cp17_pass2_evidence", session_id=session_id,
                     n_corroborated_tags=n_corr,
                     n_total_tags=cc_evidence.get("n_total_tags_observed", 0))

                # CP17 Tier 2: coordinate evidence from world-space proximity
                coord_cfg = cc_cfg.get("coordinate_evidence", {})
                if coord_cfg.get("enabled", False):
                    # Extract authoritative fps from first available session manifest
                    _session_fps = next(
                        (m.fps for m in manifests.values() if hasattr(m, "fps") and m.fps > 0),
                        30.0,
                    )
                    coord_evidence = build_cross_camera_coordinate_evidence(
                        cam_ids=list(adapters.keys()),
                        adapter_map=adapters,
                        config=cfg,
                        fps=_session_fps,
                    )
                    # Merge coordinate-corroborated tags into tag evidence
                    for tag_key, coord_info in coord_evidence.get(
                        "coordinate_corroborated_tags", {}
                    ).items():
                        if tag_key not in cc_evidence["corroborated_tags"]:
                            cc_evidence["corroborated_tags"][tag_key] = coord_info
                        else:
                            cc_evidence["corroborated_tags"][tag_key][
                                "coordinate_evidence"
                            ] = coord_info.get("coordinate_evidence")
                    cc_evidence["n_corroborated_tags"] = len(
                        cc_evidence["corroborated_tags"]
                    )
                    n_corr = cc_evidence["n_corroborated_tags"]
                    # Log conflicts to session audit
                    for conflict in coord_evidence.get("coordinate_conflicts", []):
                        _log("coordinate_conflict",
                             session_id=session_id, **conflict)
                    _log("cp17_coordinate_evidence",
                         session_id=session_id,
                         fps=_session_fps,
                         n_coordinate_corroborated=coord_evidence.get(
                             "n_coordinate_corroborated_tags", 0),
                         n_coordinate_conflicts=coord_evidence.get(
                             "n_coordinate_conflicts", 0),
                         n_corroborated_tags_after_merge=n_corr)

                # CP20 Tier 3: histogram appearance evidence
                hist_cfg = cc_cfg.get("histogram_evidence", {})
                hist_cost_modifiers = {}
                if hist_cfg.get("enabled", True):
                    try:
                        hist_evidence = build_cross_camera_histogram_evidence(
                            cam_ids=list(adapters.keys()),
                            adapter_map=adapters,
                            config=cfg,
                            session_clips=session_clips,
                            output_root=settings.OUTPUT_ROOT,
                        )
                        # Propagate tag discoveries into corroborated_tags
                        for tag_key, prop in hist_evidence.get("tag_propagations", {}).items():
                            if tag_key not in cc_evidence["corroborated_tags"]:
                                cc_evidence["corroborated_tags"][tag_key] = {
                                    "histogram_evidence": prop,
                                }
                            else:
                                cc_evidence["corroborated_tags"][tag_key][
                                    "histogram_evidence"
                                ] = prop
                        cc_evidence["n_corroborated_tags"] = len(
                            cc_evidence["corroborated_tags"]
                        )
                        n_corr = cc_evidence["n_corroborated_tags"]
                        hist_cost_modifiers = hist_evidence.get("cost_modifiers", {})
                        hist_stats = hist_evidence.get("stats", {})
                        _log("cp20_histogram_evidence",
                             session_id=session_id,
                             n_pairs=hist_stats.get("n_pairs_compared", 0),
                             n_high_sim=hist_stats.get("n_high_similarity", 0),
                             n_tag_propagations=hist_stats.get("n_tag_propagations", 0),
                             mean_similarity=hist_stats.get("mean_similarity", 0),
                             n_corroborated_tags_after_merge=n_corr)
                    except Exception as e:
                        _log("cp20_histogram_evidence_error",
                             session_id=session_id,
                             error=str(e), traceback=traceback.format_exc())

                if n_corr > 0:
                    corr_mult = float(cc_cfg.get("corroboration_miss_multiplier", 10.0))
                    overlay = {
                        "cross_camera_evidence": cc_evidence,
                        "corroboration_miss_multiplier": corr_mult,
                    }
                    if hist_cost_modifiers:
                        overlay["cost_modifiers"] = hist_cost_modifiers
                    for cam_id in cam_ids:
                        if cam_id not in manifests:
                            continue
                        try:
                            _log("cp17_pass2_start", session_id=session_id, cam_id=cam_id)
                            p2_manifest = run_session_d(
                                config=cfg,
                                session_layout=session_layout,
                                session_clips=session_clips,
                                cam_id=cam_id,
                                output_root=settings.OUTPUT_ROOT,
                                constraints_overlay=overlay,
                            )
                            if p2_manifest is not None:
                                manifests[cam_id] = p2_manifest
                            _log("cp17_pass2_completed", session_id=session_id, cam_id=cam_id)
                        except Exception as e:
                            _log("cp17_pass2_error", session_id=session_id, cam_id=cam_id,
                                 error=str(e), traceback=traceback.format_exc())

                    # Re-run cross-camera merge on Pass 2 identity assignments
                    try:
                        _log("cp17_pass2_merge_start", session_id=session_id)
                        global_id_map = run_cross_camera_merge(
                            config=cfg,
                            session_layout=session_layout,
                            cam_ids=list(adapters.keys()),
                            adapter_map=adapters,
                        )
                        _log("cp17_pass2_merge_completed", session_id=session_id,
                             n_global_ids=len(global_id_map))
                    except Exception as e:
                        _log("cp17_pass2_merge_error", session_id=session_id,
                             error=str(e), traceback=traceback.format_exc())
                else:
                    _log("cp17_pass2_skipped", session_id=session_id,
                         reason="no_corroborated_tags")
            except Exception as e:
                _log("cp17_pass2_error", session_id=session_id,
                     error=str(e), traceback=traceback.format_exc())

        # --- Loop 2: all cameras → F ---
        for cam_id in cam_ids:
            try:
                _log("session_f_start", session_id=session_id, cam_id=cam_id)
                from bjj_pipeline.stages.export.session_f_run import run_session_f
                run_session_f(
                    config=cfg,
                    session_layout=session_layout,
                    session_clips=session_clips,
                    cam_id=cam_id,
                    output_root=settings.OUTPUT_ROOT,
                    global_id_map=global_id_map,
                )
                _log("session_f_completed", session_id=session_id, cam_id=cam_id)
            except PipelineError as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in _SKIP_KEYWORDS):
                    _log("session_f_skipped", session_id=session_id, cam_id=cam_id,
                         reason=str(e))
                else:
                    _log("session_f_error", session_id=session_id, cam_id=cam_id,
                         error=str(e), traceback=traceback.format_exc())
            except Exception as e:
                _log("session_f_error", session_id=session_id, cam_id=cam_id,
                     error=str(e), traceback=traceback.format_exc())

        # --- Merge per-camera export manifests into single uploader-facing file ---
        stage_f_dir = session_layout.stage_dir("F")
        merged_path = session_layout.session_export_manifest_jsonl()
        all_records: list[dict] = []
        cam_files_found = 0

        for cam_id in cam_ids:
            cam_manifest = stage_f_dir / f"export_manifest_{cam_id}.jsonl"
            if not cam_manifest.exists():
                _log("session_merge_missing", session_id=session_id, cam_id=cam_id,
                     message="Per-camera manifest not found, skipping")
                continue
            cam_files_found += 1
            with cam_manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("status") == "no_matches":
                            continue  # Filter out per-camera no_matches headers
                        all_records.append(rec)
                    except json.JSONDecodeError:
                        continue

        merged_path.parent.mkdir(parents=True, exist_ok=True)
        if all_records:
            with merged_path.open("w", encoding="utf-8") as f:
                for rec in all_records:
                    f.write(json.dumps(rec, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False) + "\n")
            _log("session_merge_completed", session_id=session_id,
                 cam_files=cam_files_found, total_records=len(all_records))
        else:
            with merged_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "schema_version": SCHEMA_VERSION_DEFAULT,
                    "artifact_type": "export_manifest",
                    "status": "no_matches",
                    "clip_id": session_id,
                    "pipeline_version": "session",
                    "created_at_ms": int(time.time() * 1000),
                }, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            _log("session_merge_no_matches", session_id=session_id,
                 cam_files=cam_files_found)

        session_layout.session_completed_sentinel().touch()
        _log("session_phase2_completed", session_id=session_id)

    except Exception as e:
        _log("session_phase2_error", session_id=session_id,
             error=str(e), traceback=traceback.format_exc())
    finally:
        # Clear processing sentinel
        try:
            session_layout.processing_sentinel().unlink(missing_ok=True)
        except Exception:
            pass


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
         visualize=settings.VISUALIZE,
         schedule_json_set=bool(settings.SCHEDULE_JSON),
         session_end_buffer_minutes=settings.SESSION_END_BUFFER_MINUTES)

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

            # --- Session readiness evaluation (CP14a) ---
            if settings.SCHEDULE_JSON:
                schedule = _parse_schedule_json(settings.SCHEDULE_JSON)
                if schedule is not None:
                    sessions: dict[tuple[str, str, str], dict] = {}
                    for mp4 in clips:
                        cam_id = _derive_cam_id(mp4, settings)
                        if cam_id is None:
                            continue
                        clip_dt = _clip_wall_clock_dt(mp4, cam_id)
                        if clip_dt is None:
                            continue
                        result = _find_session_for_clip(clip_dt, schedule)
                        if result is None:
                            continue
                        date_str, session_id, session_end_iso = result
                        key = (session_id, date_str, session_end_iso)
                        if key not in sessions:
                            sessions[key] = {"cam_ids": set(), "clips": []}
                        sessions[key]["cam_ids"].add(cam_id)
                        sessions[key]["clips"].append((mp4, cam_id))

                    for (session_id, date_str, session_end_iso), info in sessions.items():
                        session_end_utc = datetime.fromisoformat(session_end_iso)
                        _evaluate_session_readiness(
                            session_id=session_id,
                            date_str=date_str,
                            session_end_utc=session_end_utc,
                            cam_ids=info["cam_ids"],
                            session_clips=info["clips"],
                            settings=settings,
                            schedule=schedule,
                        )

                    # --- Session Phase 2 trigger (CP14c) ---
                    for (session_id, date_str, session_end_iso), info in sessions.items():
                        # Resolve gym_id from first clip's ingest path
                        gym_id = settings.GYM_ID
                        for mp4, cam_id in info["clips"]:
                            try:
                                i = validate_ingest_path(mp4, cam_id)
                                if i.gym_id:
                                    gym_id = i.gym_id
                                    break
                            except Exception:
                                continue
                        if not gym_id:
                            continue

                        session_layout = _get_session_layout(
                            session_id, date_str, gym_id, settings
                        )
                        if (
                            _is_session_ready(session_layout)
                            and not session_layout.processing_sentinel().exists()
                            and not _is_session_uploaded(session_layout)
                            and not session_layout.session_completed_sentinel().exists()
                        ):
                            _log("session_phase2_trigger",
                                 session_id=session_id,
                                 cam_count=len(info["cam_ids"]))
                            _run_session_phase2(
                                session_id=session_id,
                                date_str=date_str,
                                gym_id=gym_id,
                                session_clips=info["clips"],
                                settings=settings,
                            )

        except Exception as e:
            _log("poll_error", error=str(e), traceback=traceback.format_exc())

        time.sleep(settings.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
